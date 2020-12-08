import torch
import mpu
import random
import numpy as np
from scipy.stats import poisson


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst) - 1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i


class ConstructBlockStrategy:
    def __init__(self, args, tokenizer, bert_prob=0.5, gpt_prob=0.5, min_gpt_ratio=0.2, block_ratio=0.2,
                 average_block_length=3,
                 max_block_length=20):
        self.args = args
        self.tokenizer = tokenizer
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        prob_normalizer = bert_prob + gpt_prob
        self.bert_prob = bert_prob / prob_normalizer
        self.gpt_prob = gpt_prob / prob_normalizer
        self.min_generation_length = int(min_gpt_ratio * args.seq_length)
        self.block_ratio = block_ratio
        self.total_mask = int(block_ratio * args.seq_length)
        self.length_distribution = [poisson.pmf(i, average_block_length) for i in range(1, max_block_length)]

    @staticmethod
    def sample_spans(span_lengths, total_length, rng, offset=0):
        blank_length = total_length - sum(span_lengths)
        m = blank_length - len(span_lengths) + 1
        places = [rng.randrange(m + 1) for _ in range(len(span_lengths))]
        places.sort()
        spans = []
        for place, span_length in zip(places, span_lengths):
            start = offset + place
            end = offset + place + span_length
            spans.append((start, end))
            offset += span_length + 1
        return spans

    def sample_span_in_document(self, tokens, masked_lengths, rng):
        rng.shuffle(masked_lengths)
        mask_spans = []
        mask_index = 0
        indices = [-1] + [i for i, item in enumerate(tokens) if item == self.args.eod_token]
        last_index = len(tokens)
        for index in reversed(indices):
            length = last_index - index - 1
            if index == -1:
                spans = self.sample_spans(masked_lengths[mask_index:], length, rng, offset=index + 1)
                mask_spans += spans
            else:
                current_masked_total = int(length * self.block_ratio)
                current_masked_length, current_count = 0, 0
                while masked_lengths[mask_index + current_count] + current_masked_length <= current_masked_total:
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(masked_lengths[mask_index:mask_index + current_count], length,
                                              rng, offset=index + 1)
                    mask_spans += spans
                    mask_index += current_count
            last_index = index
        return mask_spans

    def construct_blocks(self, samples):
        rng = random.Random(self.args.iteration * self.world_size + self.rank)
        token_batch, target_batch, loss_mask_batch, position_id_batch, block_position_id_batch = [], [], [], [], []
        if rng.random() < self.bert_prob:
            masked_lengths, masked_count = [], 0
            while masked_count < self.total_mask:
                block_length = \
                    rng.choices(range(1, len(self.length_distribution) + 1), weights=self.length_distribution)[0]
                masked_lengths.append(block_length)
                masked_count += block_length
            for sample in samples:
                tokens, loss_masks = sample['text', 'loss_masks']
                block_spans = self.sample_span_in_document(tokens, masked_lengths, rng)
                position_ids = [1] * len(tokens)
                for start, end in block_spans:
                    position_ids[start: end] = 0
                    position_ids[start] = 1
                position_ids = np.cumsum(position_ids) - 1
                target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
                rng.shuffle(block_spans)
                for start, end in block_spans:
                    target_tokens += [self.tokenizer.get_command('sop').Id] + tokens[start: end]
                    targets += tokens[start: end] + [self.tokenizer.get_command('eop').Id]
                    target_position_id = position_ids[start: end]
                    target_position_ids += target_position_id + [target_position_id[0]]
                    target_block_position_ids += list(range(1, end - start + 1))
                block_spans.sort(key=lambda x: x[0])
                source_tokens, source_position_ids = [], []
                last = 0
                for start, end in block_spans:
                    source_tokens += tokens[last: start]
                    source_tokens.append(self.tokenizer.get_command('MASK').Id)
                    source_position_ids += position_ids[last: start]
                    source_position_ids.append(position_ids[start])
                    last = end + 1
                if last < len(tokens):
                    source_tokens += tokens[last:]
                    source_position_ids += position_ids[last:]
                tokens = source_tokens + target_tokens
                targets = source_tokens + targets
                position_ids = source_position_ids + target_position_ids
                block_position_ids = [0] * len(source_tokens) + target_block_position_ids
                token_batch.append(tokens)
                target_batch.append(targets)
                position_id_batch.append(position_ids)
                block_position_id_batch.append(block_position_ids)
                loss_mask_batch.append([0] * len(source_tokens) + [1] * len(target_tokens))
        else:
            start_indices = [sample['loss_masks'].index(1) for sample in samples]
            end_indices = [rindex(sample['loss_mask'], 1) for sample in samples]
            start_index, end_index = max(start_indices), min(end_indices) - self.min_generation_length
            if end_index < start_index + 1:
                end_index = start_index + 1
            division = rng.randrange(start_index, end_index)
            for sample in samples:
                tokens, loss_masks = sample['text', 'loss_masks']
                source_tokens, target_tokens = tokens[:division], tokens[division:]
                target_masks = loss_masks[division:]
                tokens = source_tokens + [self.tokenizer.get_command('MASK').Id,
                                          self.tokenizer.get_command('sop').Id] + target_tokens[:-1]
                targets = source_tokens + [self.tokenizer.get_command('MASK').Id] + target_tokens
                loss_masks = [0] * (len(source_tokens) + 1) + target_masks
                position_ids = list(range(len(source_tokens) + 1)) + [len(source_tokens)] * len(target_masks)
                block_position_ids = [0] * len(source_tokens) + list(range(len(target_tokens) + 1))
                token_batch.append(tokens)
                target_batch.append(targets)
                loss_mask_batch.append(loss_masks)
                position_id_batch.append(position_ids)
                block_position_id_batch.append(block_position_ids)
            return {'text': torch.tensor(token_batch, dtype=torch.long),
                    'target': torch.tensor(target_batch, dtype=torch.long),
                    'loss_mask': torch.tensor(loss_mask_batch, dtype=torch.long),
                    'position_id': torch.tensor(position_id_batch, dtype=torch.long),
                    'block_position_id': torch.tensor(block_position_id_batch, dtype=torch.long),
                    'attention_mask': torch.tensor(division + 1, dtype=torch.long)}


def get_masks_and_position_ids_blocklm(
        data,
        eod_token,
        reset_position_ids,
        reset_attention_mask,
        loss_mask=None,
        attention_mask=None,
        transformer_xl=False,
        mem_length=None,
        construct_block_strategy=None,
):
    assert not transformer_xl and mem_length is None and attention_mask is None, 'blocklm does not support transformer-xl!'

    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if not transformer_xl:
        loss_mask[data == eod_token] = 0.0
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

    return attention_mask, loss_mask, position_ids
