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

from .samplers import DistributedBatchSampler
from .datasets import json_dataset, csv_dataset, split_ds, ConcatDataset, SplitDataset, bert_sentencepair_dataset, \
    GPT2Dataset, ShuffleDataset, XLDataset
from .lazy_loader import exists_lazy, LazyWriter, lazy_array_loader
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


def get_dataset(path, **kwargs):
    """gets dataset object based on keyword args and file at `path`"""
    if supported_corpus(path):
        return corpora.NAMED_CORPORA[path](**kwargs)
    ext = get_ext(path)
    if '.json' in ext:
        text = json_dataset(path, **kwargs)
    elif ext in ['.csv', '.tsv']:
        text = csv_dataset(path, **kwargs)
    else:
        raise NotImplementedError('data file type %s is not supported' % (ext))
    return text


def supported_corpus(corpus_name):
    """checks if corpus name is defined in `corpora.py`"""
    return corpus_name in corpora.NAMED_CORPORA


def make_dataset(path, seq_length, mem_length, text_key, label_key, local_rank, lazy=False, xl_style=False,
                 shuffle=True, split=[1.], delim=',', loose=False, binarize_sent=False, drop_unlabeled=False,
                 tokenizer=None, tokenizer_type='CharacterLevelTokenizer', tokenizer_model_path=None, vocab_size=None,
                 model_type='bpe', pad_token=0, character_converage=1.0, non_binary_cols=None,sample_one_document=False, pre_tokenize=False, **kwargs):
    """function to create datasets+tokenizers for common options"""
    if non_binary_cols is not None:
        # multilabel dataset support (only for csvs)
        label_key = non_binary_cols

        # make tokenizer for dataset
    if tokenizer is None:
        tokenizer = make_tokenizer(tokenizer_type, None, tokenizer_model_path, vocab_size, model_type,
                                   pad_token, character_converage, **kwargs)

    def get_dataset_from_path(path_):
        if lazy:
            # get lazily loaded dataset
            if supported_corpus(path_):
                name = path_
                path_ = corpora.NAMED_CORPORA[path_].PATH
            else:
                raise NotImplementedError
            if not (exists_lazy(path_, data_type='prompt') and exists_lazy(path_, data_type='text')):
                # create cached version of dataset for lazy loading if it doesn't exist
                if local_rank == 0:
                    prompt_writer = LazyWriter(path_, data_type='prompt', is_array=pre_tokenize)
                    text_writer = LazyWriter(path_, data_type='text', is_array=pre_tokenize)
                    get_dataset(name, text_key=text_key, label_key=label_key, prompt_writer=prompt_writer,
                                       text_writer=text_writer, binarize_sent=binarize_sent, tokenizer=tokenizer,
                                       delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose, tokenize=pre_tokenize)
                    prompt_writer.close()
                    text_writer.close()
                else:
                    while not os.path.exists(LazyWriter.get_len_path(path_, data_type='prompt')):
                        time.sleep(1)
            map_fn = (lambda x: x.tolist()) if pre_tokenize else None
            prompts = lazy_array_loader(path_, data_type='prompt', map_fn=map_fn, mem_map=True,
                                        is_array=pre_tokenize)
            texts = lazy_array_loader(path_, data_type='text', map_fn=map_fn, mem_map=True,
                                      is_array=pre_tokenize)
            text = corpora.PromptDataset(prompt_loader=prompts, text_loader=texts, tokenizer=tokenizer,
                                         to_tokenize=not pre_tokenize)
        else:
            # get dataset
            text = get_dataset(path_, text_key=text_key, label_key=label_key, binarize_sent=binarize_sent,
                               tokenizer=tokenizer, delim=delim, drop_unlabeled=drop_unlabeled, loose_json=loose)
            text = corpora.PromptDataset(prompt_loader=text.prompts, text_loader=text.texts)
        return text

    # get one or multiple datasets and concatenate
    if isinstance(path, str):
        ds = get_dataset_from_path(path)
    else:
        ds = [get_dataset_from_path(p) for p in path]
        ds = ConcatDataset(ds)

    ds_type = ''
    if 'ds_type' in kwargs:
        ds_type = kwargs['ds_type']
    # Split dataset into train/val/test (and wrap bert dataset)
    if should_split(split):
        ds = split_ds(ds, split, shuffle=shuffle)
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs['presplit_sentences'] if 'presplit_sentences' in kwargs else False
            ds = [bert_sentencepair_dataset(d, max_seq_len=seq_length,
                                            presplit_sentences=presplit_sentences) if d is not None else None for d in
                  ds]
        elif ds_type.lower() == 'gpt2':
            if xl_style:
                ds = [XLDataset(d, tokenizer, max_seq_len=seq_length, mem_len=mem_length,
                                sample_across_doc=not sample_one_document) if d is not None else None for d in ds]
            else:
                ds = [GPT2Dataset(d, tokenizer, max_seq_len=seq_length,
                                  sample_across_doc=not sample_one_document) if d is not None else None for d in ds]
    else:
        if ds_type.lower() == 'bert':
            presplit_sentences = kwargs['presplit_sentences'] if 'presplit_sentences' in kwargs else False
            ds = bert_sentencepair_dataset(ds, max_seq_len=seq_length, presplit_sentences=presplit_sentences)
        elif ds_type.lower() == 'gpt2':
            if xl_style:
                ds = XLDataset(ds, tokenizer, max_seq_len=seq_length, mem_len=mem_length,
                               sample_across_doc=not sample_one_document)
            else:
                ds = GPT2Dataset(ds, tokenizer, max_seq_len=seq_length, sample_across_doc=not sample_one_document)
    return ds, tokenizer
