import torch
import torch.utils.data
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


def index_in_list(lst, val, start=None):
    if start is None:
        start = 0
    for i in range(start, len(lst)):
        if lst[i] == val:
            return i


class ConstructBlockStrategy:
    def __init__(self, args, tokenizer, bert_prob=0.5, gpt_prob=0.5, min_gpt_ratio=0.5, block_ratio=0.2,
                 average_block_length=3, max_block_length=40, block_position_encoding=True):
        self.args = args
        self.tokenizer = tokenizer
        self.count = 0
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        prob_normalizer = bert_prob + gpt_prob
        self.bert_prob = bert_prob / prob_normalizer
        self.gpt_prob = gpt_prob / prob_normalizer
        self.min_generation_length = int(min_gpt_ratio * args.seq_length)
        self.block_ratio = block_ratio
        self.total_mask = int(block_ratio * args.seq_length)
        self.block_length_distribution = [poisson.pmf(i, average_block_length) for i in range(1, max_block_length)]
        self.block_position_encoding = block_position_encoding
        self.gap_length_distribution = [poisson.pmf(i, average_block_length) for i in range(0, max_block_length)]

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
        indices = [-1] + np.where(tokens == self.args.eod_token)[0].tolist()
        last_index = len(tokens)
        documents = []
        for index in reversed(indices):
            documents.append((index + 1, last_index - index - 1))
            last_index = index
        documents.sort(key=lambda x: x[1])
        for i, (offset, length) in enumerate(documents):
            if i == len(documents) - 1:
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(masked_lengths) and masked_lengths[
                    mask_index + current_count] + current_masked_length + current_count <= length:
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(masked_lengths[mask_index: mask_index + current_count], length, rng,
                                              offset=offset)
                    mask_spans += spans
                if mask_index + current_count < len(masked_lengths) - 1:
                    print(length, masked_lengths[mask_index:], masked_lengths[:mask_index], indices)
            else:
                current_masked_total = int(length * self.block_ratio)
                current_masked_length, current_count = 0, 0
                while mask_index + current_count < len(masked_lengths) and masked_lengths[
                    mask_index + current_count] + current_masked_length <= current_masked_total:
                    current_masked_length += masked_lengths[mask_index + current_count]
                    current_count += 1
                if current_count > 0:
                    spans = self.sample_spans(masked_lengths[mask_index:mask_index + current_count], length,
                                              rng, offset=offset)
                    mask_spans += spans
                    mask_index += current_count
        return mask_spans

    def construct_blocks(self, samples):
        worker_info = torch.utils.data.get_worker_info()
        worker_id, num_workers = worker_info.id, worker_info.num_workers
        rng = random.Random((self.count * num_workers + worker_id) * self.world_size + self.rank)
        self.count += 1
        token_batch, target_batch, loss_mask_batch, position_id_batch = [], [], [], []
        if rng.random() < self.bert_prob:
            masked_lengths, masked_count = [], 0
            while masked_count < self.total_mask:
                block_length = \
                    rng.choices(range(1, len(self.block_length_distribution) + 1),
                                weights=self.block_length_distribution)[0]
                masked_lengths.append(block_length)
                masked_count += block_length
            attention_mask = self.args.seq_length - masked_count + len(masked_lengths)
            for sample in samples:
                rng.shuffle(masked_lengths)
                tokens, loss_masks = sample['text'], sample['loss_mask']
                block_spans = self.sample_span_in_document(tokens, masked_lengths, rng)
                if len(block_spans) < len(masked_lengths):
                    continue
                position_ids = np.ones(len(tokens), dtype=np.long)
                for start, end in block_spans:
                    if self.block_position_encoding:
                        position_ids[start + 1: end] = 0
                    else:
                        position_ids[start] = 2
                        if end < len(position_ids):
                            position_ids[end] = rng.choices(range(0, len(self.gap_length_distribution)),
                                                            weights=self.gap_length_distribution)[0]
                position_ids = np.cumsum(position_ids) - 1
                rng.shuffle(block_spans)
                target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
                for start, end in block_spans:
                    target_tokens.append([self.tokenizer.get_command('sop').Id])
                    target_tokens.append(tokens[start: end])
                    targets.append(tokens[start: end])
                    targets.append([self.tokenizer.get_command('eop').Id])
                    target_position_id = position_ids[start: end]
                    if self.block_position_encoding:
                        target_position_ids.append(target_position_id)
                        target_position_ids.append([target_position_id[0]])
                        target_block_position_ids.append(np.arange(1, end - start + 2, dtype=np.long))
                    else:
                        target_position_ids.append([target_position_id[0] - 1])
                        target_position_ids.append(target_position_id)
                block_spans.sort(key=lambda x: x[0])
                source_tokens, source_position_ids = [], []
                last = 0
                for start, end in block_spans:
                    source_tokens.append(tokens[last: start])
                    source_tokens.append([self.tokenizer.get_command('MASK').Id])
                    if self.block_position_encoding:
                        source_position_ids.append(position_ids[last: start])
                        source_position_ids.append([position_ids[start]])
                    else:
                        source_position_ids.append(position_ids[last: start])
                        source_position_ids.append([position_ids[start] - 1])
                    last = end
                if last < len(tokens):
                    source_tokens.append(tokens[last:])
                    source_position_ids.append(position_ids[last:])
                source_length = sum(map(len, source_tokens))
                assert source_length == attention_mask
                assert self.args.eod_token not in np.concatenate(target_tokens).tolist()
                tokens = np.concatenate(source_tokens + target_tokens)
                targets = np.concatenate(source_tokens + targets)
                loss_masks = np.ones(len(tokens), dtype=np.long)
                loss_masks[:source_length] = 0
                token_batch.append(tokens)
                target_batch.append(targets)
                loss_mask_batch.append(loss_masks)
                position_ids = np.concatenate(source_position_ids + target_position_ids)
                if self.block_position_encoding:
                    block_position_ids = np.concatenate(
                        [np.zeros(source_length, dtype=np.long)] + target_block_position_ids)
                    position_id_batch.append([position_ids, block_position_ids])
                else:
                    position_id_batch.append(position_ids)
        else:
            start_indices = [index_in_list(sample['loss_mask'], 1) for sample in samples]
            end_indices = [rindex(sample['loss_mask'], 1) for sample in samples]
            start_index, end_index = max(start_indices), min(end_indices) - self.min_generation_length
            if end_index < start_index + 1:
                end_index = start_index + 1
            division = rng.randrange(start_index, end_index)
            for sample in samples:
                tokens, loss_masks = sample['text'], sample['loss_mask']
                source_tokens, target_tokens = tokens[:division], tokens[division:]
                target_masks = loss_masks[division:]
                tokens = np.concatenate((source_tokens, [self.tokenizer.get_command('MASK').Id,
                                                         self.tokenizer.get_command('sop').Id], target_tokens[:-1]))
                targets = np.concatenate((source_tokens, [self.tokenizer.get_command('MASK').Id], target_tokens))
                loss_masks = np.concatenate((np.zeros(len(source_tokens) + 1, dtype=np.long), target_masks))
                token_batch.append(tokens)
                target_batch.append(targets)
                loss_mask_batch.append(loss_masks)
                if self.block_position_encoding:
                    position_ids = np.arange(len(source_tokens) + 1 + len(target_tokens), dtype=np.long)
                    position_ids[len(source_tokens) + 1:] = len(source_tokens)
                    block_position_ids = np.concatenate(
                        (np.zeros(len(source_tokens), dtype=np.long), np.arange(len(target_tokens) + 1, dtype=np.long)))
                    position_id_batch.append([position_ids, block_position_ids])
                else:
                    position_ids = np.arange(len(source_tokens) + 1 + len(target_tokens), dtype=np.long)
                    position_ids[len(source_tokens) + 1:] -= 1
                    position_id_batch.append(position_ids)
            attention_mask = division + 1
        return {'text': torch.tensor(token_batch, dtype=torch.long),
                'target': torch.tensor(target_batch, dtype=torch.long),
                'loss_mask': torch.tensor(loss_mask_batch, dtype=torch.long),
                'position_id': torch.tensor(position_id_batch, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long)}


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
