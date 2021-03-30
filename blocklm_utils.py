import torch
import torch.utils.data
import mpu
import random
import copy
import numpy as np
from utils import print_rank_0
from scipy.stats import poisson


def rindex(lst, val, start=None):
    if start is None:
        start = len(lst) - 1
    for i in range(start, -1, -1):
        if lst[i] == val:
            return i
    return -1


def index_in_list(lst, val, start=None):
    if start is None:
        start = 0
    for i in range(start, len(lst)):
        if lst[i] == val:
            return i
    return -1


class ConstructBlockStrategy:
    def __init__(self, args, tokenizer, max_seq_length, bert_prob=1.0, gap_sentence_prob=0.0, gpt_infill_prob=0.5,
                 gpt_min_ratio=0.5, bert_ratio=0.15, gap_sentence_ratio=0.15, average_block_length=3,
                 max_block_length=40, block_mask_prob=0.0, context_mask_ratio=0.0, context_mask_range=3,
                 block_position_encoding=True, encoder_decoder=False, shuffle_blocks=True, sentinel_token=False,
                 task_mask=False, random_position=False, masked_lm=False):
        self.args = args
        self.tokenizer = tokenizer
        self.count = 0
        self.max_seq_length = max_seq_length
        self.rank = mpu.get_data_parallel_rank()
        self.world_size = mpu.get_data_parallel_world_size()
        # self.rank = 0
        # self.world_size = 1
        assert 0.0 <= bert_prob <= 1.0
        self.bert_prob = bert_prob
        self.gap_sentence_prob = gap_sentence_prob
        self.gpt_prob = 1 - bert_prob - gap_sentence_prob
        assert self.gpt_prob >= 0.0
        self.infill_prob = gpt_infill_prob
        self.min_generation_length = int(gpt_min_ratio * args.seq_length)
        self.bert_ratio = bert_ratio
        self.bert_total_mask = int(self.bert_ratio * args.seq_length)
        self.gap_sentence_ratio = gap_sentence_ratio
        self.gap_sent_total_mask = int(self.gap_sentence_ratio * args.seq_length)
        self.block_length_distribution = [poisson.pmf(i, average_block_length) for i in range(1, max_block_length)]
        self.block_mask_prob = block_mask_prob
        self.context_mask_ratio = context_mask_ratio
        self.context_total_mask = int(self.context_mask_ratio * args.seq_length)
        self.context_mask_range = context_mask_range
        self.block_position_encoding = block_position_encoding
        self.encoder_decoder = encoder_decoder
        self.shuffle_blocks = shuffle_blocks
        self.sentinel_token = sentinel_token
        self.generation_mask = 'gMASK' if task_mask else 'MASK'
        self.generation_mask = self.tokenizer.get_command(self.generation_mask).Id
        self.gap_sentence_mask = 'sMASK' if task_mask else 'MASK'
        self.gap_sentence_mask = self.tokenizer.get_command(self.gap_sentence_mask).Id
        self.random_position = random_position
        self.masked_lm = masked_lm
        print_rank_0(f"BERT prob {self.bert_prob}, GPT prob {self.gpt_prob}, infill prob {self.infill_prob}")
        print_rank_0(f"min generation length {self.min_generation_length}, block ratio {self.bert_ratio}")
        print_rank_0(f"block length distribution {self.block_length_distribution}")
        print_rank_0(f"block mask prob {self.block_mask_prob}, context total mask {self.context_total_mask}")

    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        if ';' in tok:
            return True
        if ':' in tok:
            return True
        return False

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
            start_index = index
            if start_index + 1 < len(tokens) and tokens[start_index + 1] == self.tokenizer.get_command('ENC').Id:
                start_index += 1
            length = last_index - start_index - 1
            if last_index == len(tokens) and length > 0:
                length -= 1
            documents.append((start_index + 1, length))
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
                current_masked_total = int(length * self.bert_ratio)
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

    def make_masked_data(self, tokens, loss_masks, attention_mask, block_spans, rng, task='bert'):
        position_ids = np.arange(len(tokens), dtype=np.long)
        targets = copy.deepcopy(tokens)
        mask_id = self.tokenizer.get_command('MASK').Id
        mlm_masks = np.zeros(len(tokens), dtype=np.long)
        for start, end in block_spans:
            for idx in range(start, end):
                tokens[idx] = mask_id
            mlm_masks[start: end] = 1
        loss_masks = loss_masks * mlm_masks
        return tokens, targets, loss_masks, position_ids

    def make_block_data(self, tokens, loss_masks, attention_mask, block_spans, rng, task='bert'):
        position_ids = np.ones(len(tokens), dtype=np.long)
        for start, end in block_spans:
            position_ids[start + 1: end] = 0
        position_ids = np.cumsum(position_ids) - 1
        if self.random_position and position_ids[-1] < self.max_seq_length - 1:
            position_bias = self.max_seq_length - position_ids[-1]
            position_bias = rng.randrange(0, position_bias)
            position_ids = position_ids + position_bias
        if self.encoder_decoder or not self.shuffle_blocks:
            block_spans.sort(key=lambda x: x[0])
        else:
            rng.shuffle(block_spans)
        if self.sentinel_token:
            block_spans = [(start, end, idx) for idx, (start, end) in enumerate(block_spans)]
        else:
            block_spans = [(start, end, 0) for start, end in block_spans]
        target_tokens, target_position_ids, target_block_position_ids, targets = [], [], [], []
        for start, end, idx in block_spans:
            sop_token = 'sop' if idx == 0 else f"sop{idx}"
            target_tokens.append([self.tokenizer.get_command(sop_token).Id])
            span_tokens = copy.deepcopy(tokens[start: end])
            if self.block_mask_prob > 0.0 and task == 'bert':
                for sub_idx in range(len(span_tokens)):
                    if random.random() < self.block_mask_prob:
                        span_tokens[sub_idx] = self.tokenizer.get_command('dBLOCK').Id
            target_tokens.append(span_tokens)
            targets.append(tokens[start: end])
            targets.append([self.tokenizer.get_command('eop').Id])
            if not self.sentinel_token:
                target_position_id = position_ids[start: end]
                target_position_ids.append(target_position_id)
                target_position_ids.append([target_position_id[0]])
            else:
                target_position_ids.append([self.max_seq_length] * (end - start + 1))
            if self.block_position_encoding:
                target_block_position_ids.append(np.arange(1, end - start + 2, dtype=np.long))
            else:
                target_block_position_ids.append([1] * (end - start + 1))
        block_spans.sort(key=lambda x: x[0])
        source_tokens, source_position_ids, local_spans = [], [], []
        last, current_length = 0, 0
        for start, end, idx in block_spans:
            if task == 'generation':
                mask_id = self.generation_mask
            elif task == 'gap_sentence':
                mask_id = self.gap_sentence_mask
            else:
                mask_token = 'MASK' if idx == 0 else f'MASK{idx}'
                mask_id = self.tokenizer.get_command(mask_token).Id
            local_spans.append((current_length, current_length + start - last))
            source_tokens.append(tokens[last: start])
            source_tokens.append([mask_id])
            source_position_ids.append(position_ids[last: start])
            source_position_ids.append([position_ids[start]])
            current_length += start - last + 1
            last = end
        if last < len(tokens):
            local_spans.append((current_length, current_length + len(tokens) - last))
            source_tokens.append(tokens[last:])
            source_position_ids.append(position_ids[last:])
        source_length = sum(map(len, source_tokens))
        if attention_mask is not None:
            assert source_length == attention_mask
        if target_tokens and self.args.eod_token in np.concatenate(target_tokens).tolist():
            print("Found EOS in target", self.tokenizer.DecodeIds(tokens))
            raise RuntimeError
        if self.encoder_decoder:
            target_tokens = target_tokens + [self.tokenizer.get_command('eop').Id]
            loss_masks = np.ones(len(target_tokens), dtype=np.long)
            return source_tokens, target_tokens, loss_masks
        else:
            tokens = np.concatenate(source_tokens + target_tokens)
            if task == 'bert' and self.context_total_mask > 0:
                mask_candidates = set()
                for start, end in local_spans:
                    if start != 0:
                        local_end = min(end, start + self.context_mask_range)
                        mask_candidates.update(range(start, local_end))
                    if end != 0:
                        local_start = max(start, end - self.context_mask_range)
                        mask_candidates.update(range(local_start, end))
                mask_pos = rng.sample(mask_candidates, self.context_total_mask)
                for pos in mask_pos:
                    tokens[pos] = self.tokenizer.get_command('dBLOCK').Id
            targets = np.concatenate(source_tokens + targets)
            loss_masks = np.ones(len(tokens), dtype=np.long)
            loss_masks[:source_length] = 0
            position_ids = np.concatenate(source_position_ids + target_position_ids)
            block_position_ids = np.concatenate(
                [np.zeros(source_length, dtype=np.long)] + target_block_position_ids)
            position_ids = np.stack([position_ids, block_position_ids], axis=0)
            if attention_mask is not None:
                return tokens, targets, loss_masks, position_ids
            else:
                return tokens, targets, loss_masks, position_ids, source_length

    def generate_blank_data(self, sample, masked_lengths, attention_mask, rng, task='bert'):
        rng.shuffle(masked_lengths)
        tokens, loss_masks = sample['text'], sample['loss_mask']
        assert tokens[0] == self.tokenizer.get_command('ENC').Id
        block_spans = self.sample_span_in_document(tokens, masked_lengths, rng)
        if len(block_spans) < len(masked_lengths):
            return None
        if self.masked_lm:
            data = self.make_masked_data(tokens, loss_masks, attention_mask, block_spans, rng)
        else:
            data = self.make_block_data(tokens, loss_masks, attention_mask, block_spans, rng, task=task)
        return data

    def construct_blocks(self, samples):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        else:
            worker_id, num_workers = 0, 1
        rng = random.Random((self.count * num_workers + worker_id) * self.world_size + self.rank)
        self.count += 1
        token_batch, target_batch, loss_mask_batch, position_id_batch = [], [], [], []
        source_batch, target_batch = [], []
        rand = rng.random()
        if rand < self.bert_prob:
            mode = 'bert'
            masked_lengths, masked_count = [], 0
            while masked_count < self.bert_total_mask:
                block_length = \
                    rng.choices(range(1, len(self.block_length_distribution) + 1),
                                weights=self.block_length_distribution)[0]
                masked_lengths.append(block_length)
                masked_count += block_length
            if self.masked_lm:
                attention_mask = self.args.seq_length
            else:
                attention_mask = self.args.seq_length - masked_count + len(masked_lengths)
            for sample in samples:
                data = self.generate_blank_data(sample, masked_lengths, attention_mask, rng, task='bert')
                if data is not None:
                    if self.encoder_decoder:
                        source_tokens, target_tokens, loss_masks = data
                        source_batch.append(source_tokens)
                        target_batch.append(target_tokens)
                        loss_mask_batch.append(loss_masks)
                    else:
                        tokens, targets, loss_masks, position_ids = data
                        token_batch.append(tokens)
                        target_batch.append(targets)
                        loss_mask_batch.append(loss_masks)
                        position_id_batch.append(position_ids)
        elif rand < self.bert_prob + self.gap_sentence_prob:
            mode = 'sentence'
            attention_mask = []
            for sample in samples:
                tokens, loss_masks = sample['text'], sample['loss_mask']
                sentence_spans = []
                last_index = 1 if tokens[0] == self.tokenizer.get_command('ENC').Id else 0
                for i in range(len(tokens)):
                    if self.contains_sentence_end(tokens[i]):
                        if last_index < i + 1:
                            sentence_spans.append((last_index, i + 1))
                        last_index = i + 1
                    elif tokens[i] == self.tokenizer.get_command('eos').Id:
                        last_index = i + 1
                if last_index < len(tokens):
                    sentence_spans.append((last_index, len(tokens)))
                if not sentence_spans:
                    print(self.tokenizer.DecodeIds(tokens[1:]).encode('utf-8'))
                rng.shuffle(sentence_spans)
                block_spans, block_length = [], 0
                for start, end in sentence_spans:
                    block_spans.append((start, end))
                    block_length += end - start
                    if block_length >= self.gap_sent_total_mask:
                        break
                data = self.make_block_data(tokens, loss_masks, None, block_spans, rng, task='gap_sentence')
                tokens, targets, loss_masks, position_ids, sep = data
                token_batch.append(tokens)
                target_batch.append(targets)
                loss_mask_batch.append(loss_masks)
                position_id_batch.append(position_ids)
                attention_mask.append(sep)
            seq_lengths = list(map(len, token_batch))
            if seq_lengths.count(seq_lengths[0]) != len(seq_lengths):
                max_length = max(seq_lengths)
                token_batch = [np.concatenate((tokens, np.zeros(max_length - len(tokens), dtype=np.long))) for tokens in
                               token_batch]
                target_batch = [np.concatenate((targets, np.zeros(max_length - len(targets), dtype=np.long))) for
                                targets in
                                target_batch]
                loss_mask_batch = [np.concatenate((loss_masks, np.zeros(max_length - len(loss_masks), dtype=np.long)))
                                   for loss_masks in loss_mask_batch]
                position_id_batch = [
                    np.concatenate((position_ids, np.zeros((2, max_length - position_ids.shape[1]), dtype=np.long)),
                                   axis=1) for position_ids in position_id_batch]
        else:
            # start_indices = [index_in_list(sample['loss_mask'], 1) for sample in samples]
            # end_indices = [rindex(sample['loss_mask'], 1) for sample in samples]
            # start_index, end_index = max(start_indices), min(end_indices) - self.min_generation_length
            # if end_index < start_index + 1:
            #     end_index = start_index + 1
            # division = rng.randrange(start_index, end_index)
            mode = 'gpt'
            generation_length = rng.randint(self.min_generation_length, len(samples[0]['text']) - 2)
            attention_mask = self.args.seq_length - generation_length + 1
            for sample in samples:
                multiple_doc = index_in_list(sample['text'], self.tokenizer.get_command('eos').Id) not in [-1, len(
                    sample['text']) - 1]
                if multiple_doc or rng.random() < self.infill_prob:
                    division = len(samples[0]['text']) - generation_length
                    tokens, loss_masks = sample['text'], sample['loss_mask']
                    source_tokens, target_tokens = tokens[:division], tokens[division:]
                    target_masks = loss_masks[division:]
                    tokens = np.concatenate((
                        source_tokens, [self.generation_mask, self.tokenizer.get_command('sop').Id],
                        target_tokens[:-1], [self.tokenizer.get_command('pad').Id]))
                    targets = np.concatenate(
                        (source_tokens, [self.generation_mask], target_tokens, [self.tokenizer.get_command('pad').Id]))
                    loss_masks = np.concatenate((np.zeros(len(source_tokens) + 1, dtype=np.long), target_masks, [0]))
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    position_ids = np.arange(len(source_tokens) + len(target_tokens) + 2, dtype=np.long)
                    position_ids[len(source_tokens) + 1:] = len(source_tokens)
                    if self.block_position_encoding:
                        block_position_ids = np.concatenate(
                            (np.zeros(len(source_tokens), dtype=np.long),
                             np.arange(len(target_tokens) + 2, dtype=np.long)))
                    else:
                        block_position_ids = np.concatenate((np.zeros(len(source_tokens) + 1, dtype=np.long),
                                                             np.ones(len(target_tokens) + 1, dtype=np.long)))
                    position_id_batch.append([position_ids, block_position_ids])
                else:
                    tokens, targets, loss_masks, position_ids = self.generate_blank_data(sample, [generation_length],
                                                                                         attention_mask, rng,
                                                                                         task='generation')
                    token_batch.append(tokens)
                    target_batch.append(targets)
                    loss_mask_batch.append(loss_masks)
                    position_id_batch.append(position_ids)
                    if tokens is None:
                        print(sample, generation_length, multiple_doc)
        if self.encoder_decoder:
            return {
                'text': torch.tensor(source_batch, dtype=torch.long),
                'target': torch.tensor(target_batch, dtype=torch.long),
                'loss_mask': torch.tensor(loss_mask_batch, dtype=torch.long)}
        else:
            return {'text': torch.tensor(token_batch, dtype=torch.long),
                    'target': torch.tensor(target_batch, dtype=torch.long),
                    'loss_mask': torch.tensor(loss_mask_batch, dtype=torch.long),
                    'position_id': torch.tensor(position_id_batch, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'mode': mode}
