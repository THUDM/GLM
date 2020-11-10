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
"""utils for loading text from disk"""
import os
import mmap
import pickle as pkl
import time
import numpy as np
from itertools import accumulate

import torch
from torch.multiprocessing import Lock

def get_lazy_path(path):
    """
    Gets directory path where lazy files are stored.
    """
    return os.path.splitext(path)[0]+'.lazy'

def exists_lazy(path, data_type='data'):
    """
    Check if we've already made a lazy version of this file for the `data_type` field.
    """
    if not os.path.exists(get_lazy_path(path)):
        return False
    contents = os.listdir(get_lazy_path(path))
    if data_type not in contents:
        return False
    if data_type+'.len.pkl' not in contents:
        return False
    return True

lazy_data_type = np.int32

def make_lazy(path, dataset, data_type='data', is_array=False):
    """
    Make lazy version of `data_type` field of the file. Byte offsets
    corresponding to data indices are stored in a `.len.pkl` data file.
    """
    lazypath = get_lazy_path(path)
    if not os.path.exists(lazypath):
        os.makedirs(lazypath)
    datapath = os.path.join(lazypath, data_type)
    lenpath = os.path.join(lazypath, data_type+'.len.pkl')
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with open(datapath, 'wb') as f:
            lengths = []
            for s in dataset:
                if isinstance(s, dict):
                    s = s['text']
                if is_array:
                    encoded = np.array(s, dtype=lazy_data_type).tobytes(order='C')
                    f.write(encoded)
                    lengths.append(len(s))
                else:
                    encoded = s.encode('utf-8')
                    f.write(encoded)
                    lengths.append(len(encoded))
        with open(lenpath, 'wb') as f:
            pkl.dump(lengths, f)
    else:
        while not os.path.exists(lenpath):
            time.sleep(1)


def split_strings(strings, start, chr_lens):
    """
    Split strings based on string lengths and given start.
    """
    return [strings[i-start:j-start] for i, j in zip([start]+chr_lens[:-1], chr_lens)]

class ProcessorTokenizer:
    """
    callable class that runs a preprocessing, as well as tokenization step,
    on input text.
    """
    def __init__(self, tokenizer, process_fn=None):
        self.tokenizer = tokenizer
        self.process_fn = process_fn

    def __call__(self, string):
        if self.tokenizer is not None:
            string =  self.tokenizer(string, process_fn=self.process_fn)
        elif self.process_fn is not None:
            string =  self.process_fn(string)
        return string

class lazy_array_loader(object):
    """
    Arguments:
        path: path to directory where array entries are concatenated into one big string file
            and the .len file are located
        data_type (str): Some datsets have multiple fields that are stored in different paths.
            `data_type` specifies which of these fields to load in this class
        mem_map  (boolean): Specifies whether to memory map file `path`
        map_fn (callable): Fetched strings are passed through map_fn before being returned.

    Example of lazy loader directory structure:
    file.json
    file.lazy/
        data_type1
        data_type1.len.pkl
        data_type2
        data_type2.len.pkl
    """
    def __init__(self, path, data_type='data', mem_map=False, map_fn=None, is_array=False):
        lazypath = get_lazy_path(path)
        datapath = os.path.join(lazypath, data_type)
        #get file where array entries are concatenated into one big string
        self._file = open(datapath, 'rb')
        self.file = self._file
        self.is_array = is_array
        #memory map file if necessary
        lenpath = os.path.join(lazypath, data_type+'.len.pkl')
        self.lens = pkl.load(open(lenpath, 'rb'))
        self.ends = list(accumulate(self.lens))
        self.dumb_ends = list(self.ends)
        self.mem_map = mem_map
        if self.mem_map:
            if is_array:
                if self.ends[-1] == 0:
                    self.file = np.array([], dtype=lazy_data_type)
                else:
                    self.file = np.memmap(self.file, dtype=lazy_data_type, mode='r', order='C')
            else:
                if self.ends[-1] == 0:
                    self.file = []
                else:
                    self.file = mmap.mmap(self.file.fileno(), 0, prot=mmap.PROT_READ)
        self.read_lock = Lock()
        self.process_fn = map_fn
        self.map_fn = map_fn
        self._tokenizer = None
        self.is_lazy = True

    def SetTokenizer(self, tokenizer):
        """
        logic to set and remove (set to None) tokenizer.
        combines preprocessing/tokenization into one callable.
        """
        if tokenizer is None:
            if not hasattr(self, '_tokenizer'):
                self._tokenizer = tokenizer
        else:
            self._tokenizer = tokenizer
        self.map_fn = ProcessorTokenizer(tokenizer, self.process_fn)

    def GetTokenizer(self):
        return self._tokenizer

    def __getitem__(self, index):
        """
        read file and splice strings based on string ending array `self.ends`
        """
        if not isinstance(index, slice):
            if index == 0:
                start = 0
            else:
                start = self.ends[index-1]
            end = self.ends[index]
            rtn = self.file_read(start, end)
            if self.map_fn is not None:
                return self.map_fn(rtn)
        else:
            # if slice, fetch strings with 1 diskread and then splice in memory
            chr_lens = self.ends[index]
            if index.start == 0 or index.start is None:
                start = 0
            else:
                start = self.ends[index.start-1]
            stop = chr_lens[-1]
            strings = self.file_read(start, stop)
            rtn = split_strings(strings, start, chr_lens)
            if self.map_fn is not None:
                return self.map_fn([s for s in rtn])
        return rtn

    def __len__(self):
        return len(self.ends)

    def file_read(self, start=0, end=None):
        """read specified portion of file"""
        data_type_size = np.dtype(lazy_data_type).itemsize
        # atomic reads to avoid race conditions with multiprocess dataloader
        self.read_lock.acquire()
        if not self.mem_map:
            # seek to start of file read
            if self.is_array:
                start = start * data_type_size
                end = end * data_type_size if end is not None else None
            self.file.seek(start)
            # read to end of file if no end point provided
            if end is None:
                rtn = self.file.read()
            #else read amount needed to reach end point
            else:
                rtn = self.file.read(end-start)
            if self.is_array:
                rtn = np.ndarray(shape=(len(rtn) / data_type_size,), dtype=lazy_data_type, buffer=rtn, order='C')
            else:
                rtn = rtn.decode('utf-8', 'ignore')
        else:
            rtn = self.file[start:end]
            if self.is_array:
                rtn = rtn.copy()
        self.read_lock.release()
        #TODO: @raulp figure out mem map byte string bug
        #if mem map'd need to decode byte string to string
        # # rtn = str(rtn)
        # if self.mem_map:
        #     rtn = rtn.decode('unicode_escape')
        return rtn

