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

"""Evaluation utilities."""

import os
import time
import json
import torch

from utils import print_rank_0
import mpu
from tasks.data_utils import build_data_loader
from finetune_gpt2 import process_batch
from collections import OrderedDict
from typing import List
from tasks.data_utils import InputExample


def accuracy_metric(predictions, labels, examples):
    count = 0
    assert len(predictions) == len(labels)
    for prediction, label in zip(predictions, labels):
        count += prediction == label
    return count * 100.0


def accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=False):
    """Provide function that calculates accuracies."""
    # Build dataloaders.
    if is_test:
        datapaths = args.test_data if args.test_data is not None else ['test']
    else:
        datapaths = args.valid_data if args.valid_data is not None else ['dev']
    dataloaders = []
    for datapath in datapaths:
        dataset = single_dataset_provider(datapath)
        dataloader = build_data_loader(
            dataset, args.batch_size, num_workers=args.num_workers,
            drop_last=False, shuffle=False, only_rank0=is_test)
        dataloaders.append((dataset.dataset_name, dataloader))

    def metrics_func(model, epoch, output_predictions=False, summary_writer=None):
        print_rank_0('calculating metrics ...')
        score_dict = OrderedDict([(key, 0.0) for key in metric_dict])
        total = 0
        for name, dataloader in dataloaders:
            examples = None
            if hasattr(dataloader.dataset, "examples"):
                examples = dataloader.dataset.examples
            output = evaluate_metrics(name, model, dataloader, metric_dict, examples, epoch, output_predictions,
                                      args, labeled=dataloader.dataset.labeled)
            if not output_predictions:
                single_dict, total_count = output
            elif torch.distributed.get_rank() == 0:
                save_dir = args.load if args.load is not None else args.log_dir
                single_dict, total_count, predictions = output
                filename = os.path.join(save_dir, name + '.jsonl')
                ids, predictions = predictions
                with open(filename, "w") as output:
                    for idx, prediction in zip(ids, predictions):
                        data = {"idx": idx, "label": prediction}
                        output.write(json.dumps(data) + "\n")
            for key in score_dict:
                score_dict[key] += single_dict[key]
            total += total_count
        score_dict = {key: score / float(total) for key, score in score_dict.items()}
        output_str = ' >> |epoch: {}| overall: total = {}'.format(epoch, total)
        for key, score in score_dict.items():
            output_str += " {} = {:.4f}".format(key, score)
            if summary_writer is not None and epoch >= 0 and not is_test:
                summary_writer.add_scalar(f'Train/valid_{key}', score, epoch)
        print_rank_0(output_str)

        return score_dict

    return metrics_func


segment_length = 10


def evaluate_metrics(name, model, dataloader, metric_dict, examples: List[InputExample], epoch, output_predictions,
                     args, labeled=True, translate_predictions=False):
    """Calculate correct over total answers and return prediction if the
    `output_predictions` is true."""

    start_time = time.time()
    model.eval()
    with torch.no_grad():
        # For all the batches in the dataset.
        total = 0
        score_dict = {key: 0.0 for key in metric_dict}
        if output_predictions:
            ids, predictions = [], []
        for _, batch in enumerate(dataloader):
            # Run the model forward.
            data = process_batch(batch, args)
            if args.pretrained_bert:
                tokens, types, labels_, attention_mask = data['text'], data['types'], data['label'], data[
                    'attention_mask']
                inputs = [tokens, types, attention_mask]
            elif args.cloze_eval:
                tokens, labels_, position_ids = data['text'], data['label'], data['position']
                attention_mask, target_ids, logit_mask = data['attention_mask'], data['target'], data['logit_mask']
                inputs = [tokens, position_ids, attention_mask, target_ids, logit_mask]
            else:
                tokens, labels_, position_ids, attention_mask = data['text'], data['label'], data['position'], data[
                    'attention_mask']
                inputs = [tokens, position_ids, attention_mask]
            if inputs[0].size(1) > segment_length:
                logit_list = []
                for i in range((inputs[0].size(1) - 1) // segment_length + 1):
                    input_batch = [arg[:, i * segment_length: (i + 1) * segment_length] for arg in inputs]
                    if args.pretrained_bert:
                        logits = model(*input_batch)
                    else:
                        logits, *mems = model(*input_batch)
                    logit_list.append(logits)
                logits = torch.cat(logit_list, dim=1)
            else:
                if args.pretrained_bert:
                    logits = model(*inputs)
                else:
                    logits, *mems = model(*inputs)
            if "loss_mask" in data:
                loss_mask = data["loss_mask"]
                logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)
            uid_list = batch['uid']
            if isinstance(uid_list, torch.Tensor):
                uid_list = uid_list.cpu().numpy().tolist()
            example_batch = None
            if examples is not None:
                example_batch = [examples[uid] for uid in uid_list]
            # Compute the correct answers.
            predicted = torch.argmax(logits, dim=-1).tolist()
            # Add output predictions.
            if output_predictions:
                if translate_predictions:
                    predictions.extend(
                        [example.meta["candidates"][idx] for example, idx in zip(example_batch, predicted)])
                else:
                    predictions.extend(predicted)
                ids_list = [example.idx for example in example_batch]
                ids.extend(ids_list)
            if labeled:
                for key, metric in metric_dict.items():
                    score_dict[key] += metric(predicted, labels_.tolist(), example_batch)
            # Add to the counters.
            total += labels_.size(0)
    model.train()
    print("here")
    # Reduce.
    if not output_predictions:
        keys = list(score_dict.keys())
        keys.sort()
        unreduced = [score_dict[key] for key in keys] + [total]
        unreduced = torch.cuda.FloatTensor(unreduced)
        torch.distributed.all_reduce(unreduced, group=mpu.get_data_parallel_group())
        # Print on screen.
        unreduced = unreduced.tolist()
        for i, key in enumerate(keys):
            score_dict[key] = unreduced[i]
        total = unreduced[-1]
    elapsed_time = time.time() - start_time
    output_str = ' > |epoch: {}| metrics for {}: total {}'.format(epoch, name, total)
    for key, value in score_dict.items():
        output_str += " {} = {:.4f} %".format(key, value / total)
    output_str += ' elapsed time (sec): {:.3f}'.format(elapsed_time)
    if output_predictions:
        return score_dict, total, (ids, predictions)
    return score_dict, total
