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

from tasks.eval_utils import accuracy_func_provider
from finetune_gpt2 import finetune
from tasks.superglue.dataset import GlueDataset, SINGLE_TOKEN_DATASETS, MULTI_TOKEN_DATASETS, PROCESSORS
from tasks.superglue.evaluate import exact_match_metric, f1_metric
from collections import OrderedDict
from tasks.eval_utils import accuracy_metric

default_metrics = {
    "record": [("EM", exact_match_metric), ("F1", f1_metric)],
    "copa": [("accuracy", accuracy_metric)],
    "rte": [("accuracy", accuracy_metric)],
    "boolq": [("accuracy", accuracy_metric)],
    "wic":  [("accuracy", accuracy_metric)]
}


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    train_dataset = GlueDataset(args.task.lower(), "train", args.data_dir, tokenizer, max_seq_length=args.seq_length,
                                cloze_format=args.cloze_eval, for_bert=args.pretrained_bert, pattern_id=args.pattern_id)
    valid_dataset = GlueDataset(args.task.lower(), "dev", args.data_dir, tokenizer, max_seq_length=args.seq_length,
                                for_train=True, cloze_format=args.cloze_eval, for_bert=args.pretrained_bert,
                                pattern_id=args.pattern_id)

    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Privde metrics callback function."""

    def single_dataset_provider(split):
        return GlueDataset(args.task.lower(), split, args.data_dir, tokenizer, max_seq_length=args.seq_length,
                           cloze_format=args.cloze_eval, for_bert=args.pretrained_bert, pattern_id=args.pattern_id)

    metric_dict = OrderedDict(default_metrics[args.task.lower()])
    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test)


def main(args):
    model_kwargs = {}
    if args.task.lower() in SINGLE_TOKEN_DATASETS:
        model_kwargs["model_type"] = "multiple_choice" if args.cloze_eval else "classification"
        model_kwargs["multi_token"] = False
        model_kwargs["num_labels"] = len(PROCESSORS[args.task.lower()]().get_labels())
    elif args.task.lower() in MULTI_TOKEN_DATASETS:
        model_kwargs["model_type"] = "multiple_choice"
        model_kwargs["multi_token"] = True
    else:
        raise NotImplementedError(args.task)
    finetune(args, train_valid_datasets_provider, model_kwargs,
             end_of_epoch_callback_provider=metrics_func_provider)
