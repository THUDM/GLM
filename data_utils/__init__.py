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
"""utils for creating datasets"""
import os
import math
import time
import random
import torch

from .samplers import DistributedBatchSampler
from .datasets import split_ds, ConcatDataset, SplitDataset, BertSentencepairDataset, \
    GPT2Dataset, ShuffleDataset, XLDataset, BlockDataset
from .lazy_loader import exists_lazy, LazyWriter, MultiLazyWriter, ScatterLazyWriter, LazyLoader, exists_scatter, \
    get_scatter_path
from .tokenization import Tokenization, CommandToken, Tokenizer, CharacterLevelTokenizer, BertWordPieceTokenizer, \
    GPT2BPETokenizer, make_tokenizer
from . import corpora

TRAIN_DATA = 0
VAL_DATA = 1
TEST_DATA = 2


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


def get_ext(path):
    """gets path extension"""
    return os.path.splitext(path)[1]


def get_language_datasets(tokenizer, pre_tokenize, data_parallel_rank, loader_scatter=None, no_lazy_loader=False):
    from .corpora import MultilingualReader
    languages = []
    with open("./languages.txt") as file:
        for line in file:
            languages.append(line.strip())
    target_path = "/dataset/fd5061f6/english_data/xiaoice.lazy"
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for lang in languages:
        path = os.path.join(target_path, lang)
        print(f"Creating lazy loader for language {lang}")
        if loader_scatter is None:
            writer = MultiLazyWriter(path, data_types=['prompt', 'text'], is_array=pre_tokenize)
        else:
            writer = ScatterLazyWriter(path, data_type=['prompt', 'text'], is_array=pre_tokenize,
                                            loader_scatter=loader_scatter)
        reader = MultilingualReader(language=lang, writer=writer, tokenizer=tokenizer, tokenize=pre_tokenize)
        reader.process()
        writer.close()


def get_dataset(name, tokenizer, pre_tokenize, data_parallel_rank, loader_scatter=None, no_lazy_loader=False,
                half_lazy_loader=False):
    """gets dataset object based on keyword args and file at `path`"""
    global_rank = torch.distributed.get_rank()
    if not supported_corpus(name):
        raise NotImplementedError('dataset %s is not supported' % name)
    dataset = corpora.NAMED_CORPORA[name]
    path = dataset.PATH
    if not (exists_lazy(path, data_type='text')) and not (
            loader_scatter is not None and exists_scatter(path, data_type='text', scatter_num=loader_scatter)):
        # create cached version of dataset for lazy loading if it doesn't exist
        if global_rank == 0:
            print(f"Creating lazy loader for dataset {name}")
            writer = MultiLazyWriter(path, data_types=['prompt', 'text'], is_array=pre_tokenize)
            reader = dataset(writer=writer, tokenizer=tokenizer, tokenize=pre_tokenize)
            reader.process()
            writer.close()
        else:
            while not os.path.exists(LazyWriter.get_len_path(path, data_type='prompt')) and not os.path.exists(
                    LazyWriter.get_len_path(path, data_type='text')):
                time.sleep(1)
    map_fn = (lambda x: x.tolist()) if pre_tokenize else None
    if loader_scatter is not None:
        if not (exists_scatter(path, data_type='prompt', scatter_num=loader_scatter) and exists_scatter(path,
                                                                                                        data_type='text',
                                                                                                        scatter_num=loader_scatter)):
            if global_rank == 0:
                print(f"Creating scatter loader for dataset {name}")
                prompts = LazyLoader(path, data_type='prompt', map_fn=map_fn, mem_map=True,
                                     is_array=pre_tokenize)
                texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True,
                                   is_array=pre_tokenize)
                writer = ScatterLazyWriter(path, ['prompt', 'text'], is_array=pre_tokenize,
                                           loader_scatter=loader_scatter)

                for idx in range(len(texts)):
                    writer.write({"prompt": prompts[idx], "text": texts[idx]})
                writer.close()
            else:
                while not (
                        exists_scatter(path, data_type='prompt', scatter_num=loader_scatter) and exists_scatter(
                    path, data_type='text', scatter_num=loader_scatter)):
                    time.sleep(1)
        scatter_path = get_scatter_path(path, scatter_rank=data_parallel_rank % loader_scatter)
        print(f"Rank {global_rank} is using scatter from {scatter_path}")
        prompts = LazyLoader(scatter_path, data_type='prompt', map_fn=map_fn, mem_map=True,
                             is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
        texts = LazyLoader(scatter_path, data_type='text', map_fn=map_fn, mem_map=True,
                           is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
    else:
        prompts = LazyLoader(path, data_type='prompt', map_fn=map_fn, mem_map=True,
                             is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
        texts = LazyLoader(path, data_type='text', map_fn=map_fn, mem_map=True,
                           is_array=pre_tokenize, load_memory=no_lazy_loader, half_load=half_lazy_loader)
    text = corpora.PromptDataset(prompt_loader=prompts, text_loader=texts, tokenizer=tokenizer,
                                 to_tokenize=not pre_tokenize)
    if loader_scatter is None:
        loader_scatter = 1
    for scatter_id in range(loader_scatter):
        if data_parallel_rank % loader_scatter == scatter_id and data_parallel_rank // loader_scatter == 0:
            print(f"Create dataset {name} at scatter {scatter_id} with {len(text)} documents")
            for i in range(10):
                sample_tokens = text[i]['tokens'][:1024]
                print(sample_tokens)
                print(tokenizer.DecodeIds(sample_tokens).encode('utf-8'))
        torch.distributed.barrier()
    return text


def supported_corpus(corpus_name):
    """checks if corpus name is defined in `corpora.py`"""
    return corpus_name in corpora.NAMED_CORPORA


def make_dataset(path, seq_length, mem_length, shuffle=True, split=None, tokenizer=None,
                 sample_one_document=False, pre_tokenize=False, ds_type='', save_splits=None, load_splits=None,
                 save_test_data=None, no_lazy_loader=False, loader_scatter=None, data_parallel_rank=None,
                 filter_english=False, non_sentence_start=0.0, half_lazy_loader=False, **kwargs):
    """function to create datasets+tokenizers for common options"""
    if split is None:
        split = [1.]

    # get one or multiple datasets and concatenate
    paths = [path] if isinstance(path, str) else path
    ds = [get_language_datasets(tokenizer=tokenizer, pre_tokenize=pre_tokenize, data_parallel_rank=data_parallel_rank,
                                loader_scatter=loader_scatter, no_lazy_loader=no_lazy_loader) if p == "multilingual"
          else get_dataset(p, tokenizer=tokenizer, pre_tokenize=pre_tokenize, no_lazy_loader=no_lazy_loader,
                           loader_scatter=loader_scatter, data_parallel_rank=data_parallel_rank,
                           half_lazy_loader=half_lazy_loader)
          for p in paths]
    ds = ConcatDataset(ds) if len(ds) > 1 else ds[0]

    # Split dataset into train/val/test (and wrap bert dataset)
    def wrap_dataset(dataset):
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs['presplit_sentences'] if 'presplit_sentences' in kwargs else False
            dataset = BertSentencepairDataset(dataset, max_seq_len=seq_length, presplit_sentences=presplit_sentences)
        elif ds_type.lower() == 'gpt-xl':
            assert pre_tokenize
            dataset = XLDataset(dataset, tokenizer, max_seq_len=seq_length, mem_len=mem_length,
                                sample_across_doc=not sample_one_document)
        elif ds_type.lower() == 'gpt2':
            dataset = GPT2Dataset(dataset, tokenizer, max_seq_len=seq_length, sample_across_doc=not sample_one_document)
        elif ds_type.lower() == 'block':
            dataset = BlockDataset(dataset, tokenizer, max_seq_len=seq_length,
                                   sample_across_doc=not sample_one_document, filter_english=filter_english,
                                   non_sentence_start=non_sentence_start)
        return dataset

    if should_split(split):
        ds = split_ds(ds, split, shuffle=shuffle, save_splits=save_splits, load_splits=load_splits)
        if save_test_data is not None and torch.distributed.get_rank() == 0:
            test_ds = ds[-1]
            with open(save_test_data, "w", encoding='utf-8') as output:
                for data in test_ds:
                    text = data['tokens']
                    text = tokenizer.DecodeIds(text)
                    output.write(text)
                    output.write("\n")
            print(f"Write test data to {save_test_data}")
        ds = [wrap_dataset(d) if d is not None else None for d in ds]
    else:
        ds = wrap_dataset(ds)
    return ds
