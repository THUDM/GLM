# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""dataset objects for jsons, csvs, and BERT datasets"""

import math
from bisect import bisect_right
from itertools import accumulate
import random
import torch

from torch.utils import data
import numpy as np

from .lazy_loader import LazyLoader
from utils import print_rank_0
import SwissArmyTransformer.data_utils.configure_data


class LengthSamplingDataset(data.Dataset):
    def __init__(self, ds):
        self.ds = ds
        if ds.is_lazy:
            lens = map(lambda idx: self.ds.get_text_len(idx) // 10 + 1, range(len(self.ds)))
        else:
            lens = map(lambda d: len(d['tokens']) // 10 + 1 if isinstance(d, dict) else len(d), self.ds)
        self.weighting = list(accumulate(lens))
        self.total_len = self.weighting[-1] if self.weighting else 0
        print_rank_0(
            f"Dataset {ds.name} document count {len(self.ds)}, token count {self.total_len}")

    @property
    def name(self):
        return self.ds.name

    @property
    def is_lazy(self):
        return self.ds.is_lazy

    @property
    def tokenizer(self):
        return self.ds.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self.ds._tokenizer = tokenizer

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data_idx = bisect_right(self.weighting, idx)
        return self.ds[data_idx]


class ConcatDataset(data.Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, **kwargs):
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'
        self.is_lazy = sum([isinstance(ds, LazyLoader) or (hasattr(ds, 'is_lazy') and ds.is_lazy) for ds in
                            self.datasets]) == len(self.datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)
        self._lens = None

    @property
    def tokenizer(self):
        if self.datasets:
            return self.datasets[0].tokenizer
        else:
            return None

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        for ds in self.datasets:
            ds._tokenizer = tokenizer

    def get_text_len(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].get_text_len(sample_idx)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class ScaleDataset(data.Dataset):
    def __init__(self, ds, ratio, block_size=10000):
        self.ds = ds
        self.ds_len = len(self.ds)
        self.ratio = ratio
        reserved_ratios = ratio - math.floor(ratio)
        self.reserved_len = int(math.floor(ratio) * self.ds_len)
        if block_size is not None:
            block_size = min(block_size, self.ds_len)
        self.block_size = block_size
        if reserved_ratios > 0:
            indices_len = self.ds_len if block_size is None else block_size
            indices = random.sample(range(indices_len), int(reserved_ratios * indices_len))
            if block_size is None:
                self.sub_dataset = SplitDataset(ds, indices)
            else:
                self.sub_dataset = BlockedRandomSplitDataset(ds, indices, block_size=block_size)
            self.total_len = int(math.floor(ratio) * self.ds_len) + len(self.sub_dataset)
        else:
            self.sub_dataset = None
            self.total_len = int(math.floor(ratio) * self.ds_len)

    @property
    def name(self):
        return f"Scale {self.ratio} of " + self.ds.name

    @property
    def is_lazy(self):
        return isinstance(self.ds, LazyLoader) or (
                hasattr(self.ds, 'is_lazy') and self.ds.is_lazy)

    def get_text_len(self, index):
        if index < self.reserved_len:
            return self.ds.get_text_len(index % self.ds_len)
        else:
            return self.sub_dataset.get_text_len(index % self.ds_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        if index < self.reserved_len:
            return self.ds[index % self.ds_len]
        else:
            return self.sub_dataset[index % self.ds_len]


class MultiSamplingDataset(data.Dataset):
    def __init__(self, datasets, reweight=True, temperature=1.0, max_limit=None):
        self.datasets = list(datasets)
        self.reweight = reweight
        self.temperature = temperature
        self.lens = [len(dataset) for dataset in datasets]
        if max_limit is None:
            max_limit = float('inf')
        self.weights = np.array([min(l, max_limit) ** temperature for l in self.lens])
        names = [ds.name for ds in datasets]
        print(list(zip(names, self.weights)))
        self.total_len = sum(self.lens)
        self.cumulative_lens = list(accumulate(self.lens))
        if self.reweight:
            print_rank_0(list(zip(self.lens, self.weights)))
        else:
            print_rank_0(list(zip(self.lens)))
        self.weights /= self.weights.sum()

    def __len__(self):
        return self.total_len * 1000

    @property
    def tokenizer(self):
        if self.datasets:
            return self.datasets[0].tokenizer
        else:
            return None

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        for ds in self.datasets:
            ds.tokenizer = tokenizer

    def __getitem__(self, idx):
        if self.reweight:
            rng = random.Random(idx)
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            dataset_idx = rng.choice(np.arange(len(self.datasets)), p=self.weights)
            dataset = self.datasets[dataset_idx]
            sample_idx = rng.choice(np.arange(len(dataset)))
            item = self.datasets[dataset_idx][sample_idx]
        else:
            dataset_idx = bisect_right(self.cumulative_lens, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_lens[dataset_idx - 1]
            item = self.datasets[dataset_idx][sample_idx]
        return item


class BlockedRandomSplitDataset(SwissArmyTransformer.data_utils.configure_data.BlockedRandomSplitDataset):
    @property
    def is_lazy(self):
        return isinstance(self.wrapped_data, LazyLoader) or (
                hasattr(self.wrapped_data, 'is_lazy') and self.wrapped_data.is_lazy)

    @property
    def name(self):
        return "Blocked split of " + self.wrapped_data.name

    @property
    def tokenizer(self):
        return self.wrapped_data.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self.wrapped_data._tokenizer = tokenizer

    def get_text_len(self, index):
        return self.wrapped_data.get_text_len(
            (index // len(self.indices)) * self.block_size + self.indices[index % len(self.indices)])


class SplitDataset(data.Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_inds (1D array-like): List of indices part of subset
    """

    def __init__(self, ds, split_inds, **kwargs):
        self.split_inds = list(split_inds)
        self.ds = ds

    def __len__(self):
        return len(self.split_inds)

    def get_text_len(self, idx):
        return self.ds.get_text_len(self.split_inds[idx])

    def __getitem__(self, index):
        return self.ds[self.split_inds[index]]

    @property
    def is_lazy(self):
        return isinstance(self.ds, LazyLoader) or (hasattr(self.ds, 'is_lazy') and self.ds.is_lazy)

    @property
    def name(self):
        return "Split of " + self.ds.name

    @property
    def tokenizer(self):
        return self.ds.tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self.ds._tokenizer = tokenizer

    def __iter__(self):
        for idx in self.split_inds:
            yield self.ds[idx]


def split_ds(ds, split=None, shuffle=True, save_splits=None, load_splits=None, block_size=10000):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
        save_splits: save split indices to file
        load_splits: load split indices from file
        block_size: block size used to split dataset
    """
    if split is None:
        split = [.8, .2, .0]
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum
    if block_size is not None:
        block_size = min(block_size, len(ds))
    indices_len = len(ds) if block_size is None else block_size
    if load_splits is not None:
        indices = np.load(load_splits)
        assert len(indices) == indices_len
        print_rank_0(f"Load split indices from {load_splits}")
    else:
        indices = np.arange(indices_len)
        if shuffle:
            rng = np.random.RandomState(1234)
            rng.shuffle(indices)
        if save_splits is not None:
            if torch.distributed.get_rank() == 0:
                np.save(save_splits, indices)
                print(f"Save split indices to {save_splits}")
    start_idx = 0
    residual_idx = 0
    rtn_ds = [None] * len(split)
    for i, f in enumerate(split):
        if f != 0:
            proportion = indices_len * split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            split_inds = indices[start_idx:start_idx + max(split_, 1)]
            if block_size is None:
                rtn_ds[i] = SplitDataset(ds, split_inds)
            else:
                rtn_ds[i] = BlockedRandomSplitDataset(ds, split_inds, block_size)
            start_idx += split_
            residual_idx %= 1
    return rtn_ds


class XLDataset(data.Dataset):
    def __init__(self, ds, tokenizer, max_seq_len=1024, mem_len=None, sample_across_doc=True, **kwargs):
        self.ds = ds
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        if mem_len is None:
            mem_len = max_seq_len
        self.mem_len = mem_len
        self.sample_across_doc = sample_across_doc
        self.indices, self.num_samples = None, None
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_indices()

    def init_indices(self):
        if self.is_lazy:
            lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
        else:
            lens = np.array([len(d['prompt']) + len(d['text']) if isinstance(d, dict) else len(d) for d in self.ds])
        self.indices = list(accumulate(lens))
        print_rank_0(f"Dataset document count {len(lens)}, token count {self.indices[-1]}")
        self.num_samples = self.indices[-1] // self.max_seq_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens, targets, loss_mask, attention_mask = self.getidx(idx)
        tokens = self.pad_seq(tokens)
        targets = self.pad_seq(targets)
        loss_mask = self.pad_seq(loss_mask, pad_id=0)
        return {'text': np.array(tokens), "target": np.array(targets), "loss_mask": np.array(loss_mask),
                "attention_mask": np.array(attention_mask)}

    def getidx(self, idx):
        tokens, targets, loss_masks = [], [], []
        attention_mask = np.concatenate((np.zeros((self.max_seq_len, self.mem_len), dtype=np.long),
                                         np.ones((self.max_seq_len, self.max_seq_len), dtype=np.long)), axis=1)
        sample_idx = bisect_right(self.indices, idx * self.max_seq_len)
        last_end = 0 if sample_idx == 0 else self.indices[sample_idx - 1]
        token_offset = idx * self.max_seq_len - last_end
        if token_offset != 0:
            history = min(self.mem_len, token_offset)
            attention_mask[:, -self.max_seq_len - history:-self.max_seq_len] = 1
        count = 0
        while len(tokens) < self.max_seq_len and sample_idx < len(self.ds):
            item = self.ds[sample_idx]
            text, masks = item['tokens'], item['loss_masks']
            text = text + [self.tokenizer.get_command('eos').Id]
            end = min(len(text) - 1, token_offset + self.max_seq_len - len(tokens))
            masks = masks + [1]
            if count > 0:
                current = len(tokens)
                attention_mask[current:, :current + self.mem_len] = 0
            tokens += text[token_offset: end]
            targets += text[token_offset + 1: end + 1]
            loss_masks += masks[token_offset + 1: end + 1]
            count += 1
            sample_idx += 1
            token_offset = 0
        return tokens, targets, loss_masks, attention_mask

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id if pad_id is None else pad_id] * (num_pad_tokens)
        return seq


class BlockDataset(data.Dataset):
    def __init__(self, ds, tokenizer,
                 max_seq_len=1024,
                 sample_across_doc=True,
                 non_sentence_start=0.0, filter_english=False, **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.sample_across_doc = sample_across_doc
        self.non_sentence_start = non_sentence_start
        self.filter_english = filter_english
        self.is_lazy = False
        if self.filter_english:
            import fasttext
            self.model = fasttext.load_model('/dataset/fd5061f6/english_data/lid.176.bin')
            print_rank_0("Load language detection model")
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True

    def get_weighted_samples(self, np_rng):
        while True:
            idx = np_rng.randint(self.ds_len)
            tokens, loss_mask = self.getidx(idx)
            if self.filter_english:
                text = self.tokenizer.DecodeIds(tokens[:1024])
                lang = self.model.predict(text.replace('\n', ''))[0][0]
                if lang == '__label__en':
                    break
            else:
                break
        return tokens, loss_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        tokens, loss_mask = self.get_weighted_samples(rng)
        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len + 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            move_count = 0
            strip_left_tokens = rng.randint(tokens_to_strip)
            if rng.random() > self.non_sentence_start:
                if rng.random() < 0.5:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens > 0 and not self.contains_sentence_end(
                            tokens[strip_left_tokens - 1]):
                        strip_left_tokens -= 1
                        move_count += 1
                else:
                    while move_count < self.max_seq_len // 2 and strip_left_tokens < len(
                            tokens) and not self.contains_sentence_end(tokens[strip_left_tokens - 1]):
                        strip_left_tokens += 1
                        move_count += 1
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens[strip_left_tokens:]
            loss_mask = [0] + loss_mask[strip_left_tokens:]
            if len(tokens) == 2 and tokens[1] == self.tokenizer.get_command('eos').Id:
                tokens, loss_mask = [], []
            tokens, loss_mask = self.right_strip_seq(tokens, loss_mask, self.max_seq_len)
        else:
            tokens = [self.tokenizer.get_command('ENC').Id] + tokens
            loss_mask = [0] + loss_mask
            # Sample multiple documents
            if self.sample_across_doc:
                while len(tokens) < self.max_seq_len:
                    new_tokens, new_loss_mask = self.get_weighted_samples(rng)
                    new_tokens = [self.tokenizer.get_command('ENC').Id] + new_tokens
                    new_loss_mask = [0] + new_loss_mask
                    is_last = len(new_tokens) >= self.max_seq_len - len(tokens)
                    new_tokens, new_loss_mask = self.right_strip_seq(new_tokens, new_loss_mask,
                                                                     self.max_seq_len - len(tokens))
                    tokens += new_tokens
                    loss_mask += new_loss_mask
                    if is_last:
                        break
        return {'text': np.array(tokens), "loss_mask": np.array(loss_mask)}

    def right_strip_seq(self, tokens, loss_mask, seq_length):
        strip_right_tokens = len(tokens) - seq_length
        if strip_right_tokens > 0:
            while strip_right_tokens < len(tokens) - 1 and not self.contains_sentence_end(
                    tokens[-strip_right_tokens - 1]):
                strip_right_tokens += 1
            if len(tokens) - strip_right_tokens < seq_length // 2:
                strip_right_tokens = len(tokens) - seq_length
            tokens = tokens[:-strip_right_tokens]
            loss_mask = loss_mask[:-strip_right_tokens]
        return tokens, loss_mask

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command('eos').Id]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id if pad_id is None else pad_id] * (num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
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
        if '\n' in tok:
            return True
        return False


class GPT2Dataset(data.Dataset):

    def __init__(self, ds, tokenizer,
                 max_seq_len=1024,
                 num_samples=None,
                 weighted=True,
                 sample_across_doc=True,
                 random_across_doc_sampling=True,
                 sentence_start=False, **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = num_samples
        if num_samples is None:
            self.num_samples = 1000 * self.ds_len
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.weighted = weighted
        self.sample_across_doc = sample_across_doc
        self.random_across_doc_sampling = random_across_doc_sampling
        self.sentence_start = sentence_start
        self.is_lazy = False
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True

    def get_weighted_samples(self, np_rng):
        return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get possibly weighted random index from dataset
        data_idx = self.get_weighted_samples(rng)
        #        data_idx = rng.choice(self.ds_len, p=self.weighting)
        tokens, loss_mask = self.getidx(data_idx)

        # truncate or pad tokens
        num_tokens = len(tokens)
        tokens_to_strip = num_tokens - self.max_seq_len - 1

        # randomly choose a position for start
        if tokens_to_strip > 0:
            strip_left_tokens = rng.randint(tokens_to_strip + 1)
            tokens = tokens[strip_left_tokens:]
            loss_mask = loss_mask[strip_left_tokens:]
            # if self.sentence_start:
            #     token_copy = list(tokens)
            #     not_done = True
            #     while (len(token_copy) > 0) and not_done:
            #         tok = token_copy.pop(0)
            #         if self.contains_sentence_end(tok):
            #             tokens = token_copy
            #             not_done = False
            strip_right_rokens = len(tokens) - self.max_seq_len - 1
            if strip_right_rokens > 0:
                tokens = tokens[:-strip_right_rokens]
                loss_mask = loss_mask[:-strip_right_rokens]
        # Sample multiple documents
        if self.sample_across_doc:
            while (len(tokens) < (self.max_seq_len + 1)):
                if self.random_across_doc_sampling:
                    data_idx = self.get_weighted_samples(rng)
                else:
                    data_idx = (data_idx + 1) % self.ds_len
                new_tokens, new_loss_mask = self.getidx(data_idx)
                tokens += new_tokens
                loss_mask += new_loss_mask
            tokens = tokens[:(self.max_seq_len + 1)]
            loss_mask = loss_mask[:(self.max_seq_len + 1)]

        tokens = self.pad_seq(tokens)
        loss_mask = self.pad_seq(loss_mask, pad_id=0)
        return {'text': np.array(tokens), "loss_mask": np.array(loss_mask)}

    def getidx(self, data_idx):
        data = self.ds[data_idx]
        tokens, loss_masks = data['tokens'], data['loss_masks']
        tokens = tokens + [self.tokenizer.get_command('eos').Id]
        loss_masks = loss_masks + [1]
        return tokens, loss_masks

    def pad_seq(self, seq, pad_id=None):
        total_tokens = self.max_seq_len + 1
        num_pad_tokens = max(0, total_tokens - len(seq))
        seq += [self.tokenizer.get_command('pad').Id if pad_id is None else pad_id] * (num_pad_tokens)
        return seq

    # TODO: rewrite this function for chinese
    def contains_sentence_end(self, tok):
        tok = self.tokenizer.IdToToken(tok)
        if '.' in tok:
            return True
        if '?' in tok:
            return True
        if '!' in tok:
            return True
        return False
