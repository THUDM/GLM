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
import torch
import mpu
import functools
from tasks.eval_utils import accuracy_func_provider
from finetune_gpt2 import finetune
from pretrain_gpt2 import get_batch
from collections import OrderedDict
from tasks.seq2seq.dataset import Seq2SeqDataset
from tasks.seq2seq.evaluate import rouge_metric, DecoderEvaluater

global_tokenizer = None


def seq2seq_forward_step(data, model, args, timers, mems):
    """Forward step."""

    # Get the batch.
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    logits, loss_mask = logits[:, args.src_seq_length:], loss_mask[:, args.src_seq_length:]
    labels = labels[:, args.src_seq_length:]
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    if args.label_smoothing > 0.0:
        epsilon = args.label_smoothing
        smooth_loss = -torch.nn.functional.log_softmax(logits, dim=-1).mean(dim=-1)
        losses = (1 - epsilon) * losses + epsilon * smooth_loss
    loss_mask = loss_mask.reshape(-1)
    # The loss is not normalized for fair comparison
    loss = torch.sum(losses.reshape(-1) * loss_mask) / loss_mask.sum()
    return loss, mems, 'bert'


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    train_dataset = Seq2SeqDataset(args, split='train', tokenizer=tokenizer)
    valid_dataset = None
    global global_tokenizer
    global_tokenizer = tokenizer
    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Privde metrics callback function."""
    if not is_test:
        return None

    def single_dataset_provider(split):
        return Seq2SeqDataset(args, split=split, tokenizer=tokenizer)

    evaluater = DecoderEvaluater(args, tokenizer)
    eval_func = evaluater.evaluate
    metric_dict = OrderedDict({"rouge-1": functools.partial(rouge_metric, metric="rouge-1"),
                               "rouge-2": functools.partial(rouge_metric, metric="rouge-2"),
                               "rouge-l": functools.partial(rouge_metric, metric="rouge-l")})

    def output_func(predictions, examples, output_file):
        with open(output_file + ".hyps", "w") as output:
            for prediction in predictions:
                output.write(prediction)
                output.write("\n")
        with open(output_file + ".refs", "w") as output:
            for example in examples:
                output.write(example.meta["ref"])
                output.write("\n")

    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test, eval_func=eval_func,
                                  output_func=output_func, only_rank0=False)


def main(args):
    if args.src_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.src_seq_length
    if args.task.lower() in ['cnn_dm', 'gigaword']:
        finetune(args, train_valid_datasets_provider, {}, end_of_epoch_callback_provider=metrics_func_provider,
                 forward_step=seq2seq_forward_step)
    else:
        raise NotImplementedError(args.task)
