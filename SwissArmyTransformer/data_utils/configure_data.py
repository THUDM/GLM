# -*- encoding: utf-8 -*-
'''
@File    :   configure_data.py
@Time    :   2021/01/11 23:28:38
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import copy

import numpy as np
import torch
from bisect import bisect_right
from functools import partial

from torch.utils import data
from .samplers import DistributedBatchSampler

from SwissArmyTransformer import mpu


def make_data_loader(dataset, batch_size, num_iters, args):
    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    distributed = world_size > 1

    sampler = torch.utils.data.SequentialSampler(dataset)
    drop_last = distributed
    # the GPUs in the same model parallel group receive the same data
    if distributed: # TODO reformat this, but it is not urgent
        gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)
        batch_sampler = DistributedBatchSampler(sampler,
            batch_size,
            drop_last,
            rank,
            world_size,
            gradient_accumulation_steps=gradient_accumulation_steps)
    else:
        batch_sampler = torch.utils.data.BatchSampler(sampler,
                                                        batch_size,
                                                        drop_last)
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_sampler=batch_sampler,
                                            num_workers=args.num_workers,
                                            pin_memory=True)
    return data_loader


def make_dataset_full(path, split, args, create_dataset_function, **kwargs):
    """function to create datasets+tokenizers for common options"""
    print('make dataset ...', path)
    if split is None:
        split = [1.]

    assert isinstance(path, list)
    
    ds = []
    for p in path:
        d = create_dataset_function(p, args)
        ds.append(d)
    ds = ConcatDataset(ds)
    if should_split(split):
        ds = split_ds(ds, split, block_size=args.block_size)
    else:
        ds = RandomMappingDataset(ds)

    # if should_split(split):
    #     ds = split_ds(ds, split) # Large dataset, cannot shuffle, randomly mapping
    # FIXME this will merge valid set and train set.
    return ds

def make_loaders(args, create_dataset_function):
    """makes training/val/test
    Args:
        args.train_data, args.valid_data, args.test_data: str. Paths to the dataset.
        args.split: str. format: "8,1,1". how to split train_data.
        args.dataset_type: use to create the right datasets. 
    """
    make_dataset = partial(make_dataset_full, 
                        create_dataset_function=create_dataset_function)

    world_size = torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())
    batch_size = args.batch_size * world_size
    eval_batch_size = batch_size
    if args.eval_batch_size is not None:
        eval_batch_size = args.eval_batch_size * world_size
    
    split = get_split(args)

    data_set_args = {
        'path': args.train_data,
        'split': split,
    }

    eval_set_args = copy.copy(data_set_args)
    eval_set_args['split'] = [1.]
    
    # make datasets splits and tokenizer
    train = None
    valid = None
    test = None

    if args.train_data is not None:
        train = make_dataset(**data_set_args, args=args)
        if should_split(split):
            train, valid, test = train

    # make training and val dataset if necessary
    if valid is None and args.valid_data is not None:
        eval_set_args['path'] = args.valid_data
        valid = make_dataset(**eval_set_args, args=args)
    if test is None and args.test_data is not None:
        eval_set_args['path'] = args.test_data
        test = make_dataset(**eval_set_args, args=args)

    # wrap datasets with data loader
    if train is not None and args.batch_size > 0:
        train = make_data_loader(train, batch_size, args.train_iters, args)
        args.do_train = True
    else:
        args.do_train = False
    eval_batch_size = eval_batch_size if eval_batch_size != 0 else batch_size
    if valid is not None:
        valid = make_data_loader(valid, eval_batch_size, args.train_iters, args)
        args.do_valid = True
    else:
        args.do_valid = False
    if test is not None:
        test = make_data_loader(test, eval_batch_size, len(test) // eval_batch_size + 1, args)
        args.do_test = True
    else:
        args.do_test = False

    return train, valid, test



def get_split(args):
    """
    Get dataset splits from comma separated string list
    """
    splits = []
    if args.split.find(',') != -1:
        splits = [float(s) for s in args.split.split(',')]
    elif args.split.find('/') != -1:
        splits = [float(s) for s in args.split.split('/')]
    else:
        splits = [float(args.split)]
    split_total = sum(splits)
    if split_total < 1.:
        splits.append(1-split_total)
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    if args.valid_data is not None:
        splits[1] = 0.
    if args.test_data is not None:
        splits[2] = 0.
    final_sum = sum(splits)
    return [s/final_sum for s in splits]

def should_split(split):
    """
    given split proportions checks if should split
    Examples:
    >>> should_split([10,0,0]) 
    False
    >>> should_split([1,.1,.2])
    True
    """
    return max(split) / sum(split) != 1.

def split_ds(ds, split=[.8,.2,.0], block_size = 10000):
    """
    Split a dataset into subsets given proportions of how
    much to allocate per split. If a split is 0% returns None for that split.
    Purpose: Useful for creating train/val/test splits
    Arguments:
        ds (Dataset or array-like): Data to be split.
        split (1D array-like): proportions to split `ds`. `sum(splits) != 0`
        shuffle (boolean): Randomly split dataset. Default: True
    """
    split_sum = sum(split)
    if split_sum == 0:
        raise Exception('Split cannot sum to 0.')
    split = np.array(split)
    split /= split_sum

    assert block_size <= len(ds)

    start_idx = 0
    residual_idx = 0
    rtn_ds = [None]*len(split)
    indices = np.random.permutation(np.array(range(block_size)))
    for i, f in enumerate(split):
        if f != 0:
            proportion = block_size*split[i]
            residual_idx += proportion % 1
            split_ = int(int(proportion) + residual_idx)
            rtn_ds[i] = BlockedRandomSplitDataset(ds, indices[range(start_idx, start_idx+max(split_, 1))], block_size)
            rtn_ds[i] = RandomMappingDataset(rtn_ds[i])
            start_idx += split_
            residual_idx %= 1
    return rtn_ds

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
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class SplitDataset(data.Dataset):
    """
    Dataset wrapper to access a subset of another dataset.
    Purpose: useful to index into existing datasets, possibly
    large-scale datasets as the subindexing operation is done in an
    on-the-fly manner.
    Arguments:
        ds (Dataset or array-like): List of datasets to be subindexed
        split_range (Tuple): (Left, Right)
    """
    def __init__(self, ds, split_range, **kwargs):
        self.split_range = split_range
        self.wrapped_data = ds

    def __len__(self):
        return self.split_range[1] - self.split_range[0]

    def __getitem__(self, index):
        index += self.split_range[0]
        assert index < self.split_range[1]
        return self.wrapped_data[index]

    def __iter__(self):
        for idx in range(*self.split_range):
            yield self.wrapped_data[idx]

class RandomMappingDataset(data.Dataset):
    '''
    Dataset wrapper to randomly mapping indices to original order.
    Will also enlarge the length
    '''
    def __init__(self, ds, **kwargs):
        self.wrapped_data = ds

    def __len__(self):
        return len(self.wrapped_data) * 200

    def __getitem__(self, index):
        rng = random.Random(index)
        rng = np.random.RandomState(seed=[rng.randint(0, 2**32-1) for _ in range(16)])
        index = rng.randint(len(self.wrapped_data))
        return self.wrapped_data[index]

class BlockedRandomSplitDataset(data.Dataset):
    '''
    Dataset wrapper to access a subset of another dataset.
    Use block algorithm to reduce memory
    '''
    def __init__(self, ds, indices, block_size,**kwargs):
        if type(indices) is not np.ndarray:
            indices = np.array(indices)
        indices = np.sort(indices)
        self.block_size = block_size
        self.wrapped_data = ds
        self.wrapped_data_len = len(ds)
        self.indices = indices
        self.len = len(indices) * (len(ds) // block_size) + np.sum(indices < (len(ds) % block_size))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.wrapped_data[(index // len(self.indices)) * self.block_size + self.indices[index % len(self.indices)]]


class EnlargedDataset(data.Dataset):
    '''
    Dataset wrapper to enlarge the dataset
    '''
    def __init__(self, ds, scale=200, **kwargs):
        self.wrapped_data = ds
        self.wrapped_data_len = len(ds)
        self.scale = scale

    def __len__(self):
        return self.wrapped_data_len * self.scale

    def __getitem__(self, index):
        return self.wrapped_data[index%self.wrapped_data_len]


