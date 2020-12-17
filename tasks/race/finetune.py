# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Race."""

from utils import print_rank_0
from model.multiple_choice import MultipleChoice
from tasks.eval_utils import accuracy_func_provider
from finetune_gpt2 import finetune
from tasks.race.dataset import RaceDataset


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    train_dataset = RaceDataset('training', args.train_data, tokenizer, args.seq_length, is_bert=args.pretrained_bert,
                                pool_token=args.pool_token)
    valid_dataset = RaceDataset('validation', args.valid_data, tokenizer, args.seq_length, is_bert=args.pretrained_bert,
                                pool_token=args.pool_token)

    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Privde metrics callback function."""

    def single_dataset_provider(datapath):
        name = datapath.split('RACE')[-1].strip('/').replace('/', '-')
        return RaceDataset(name, [datapath], tokenizer, args.seq_length, is_bert=args.pretrained_bert,
                           pool_token=args.pool_token)

    return accuracy_func_provider(single_dataset_provider, args, is_test=is_test)


def main(args):
    finetune(args, train_valid_datasets_provider, "multiple_choice",
             end_of_epoch_callback_provider=metrics_func_provider)
