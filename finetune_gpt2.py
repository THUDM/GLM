import os
import json

from tasks.data_utils import build_data_loader
from utils import get_sample_writer, get_log_dir, print_and_save_args
from model import GPT2Model
from arguments import get_args

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

"""Finetune utilities."""

import torch
import torch.utils.data
from torch_scatter import scatter_sum
from configure_data import prepare_tokenizer

from utils import print_rank_0
from utils import Timers
from train_utils import setup_model_and_optimizer, train_step
from utils import load_checkpoint, save_checkpoint
from pretrain_gpt2 import report_iteration_metrics
from pretrain_gpt2 import evaluate_and_print_results
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from model import PyTorchDistributedDataParallel as TorchDDP
from model import DistributedDataParallel as LocalDDP
from fp16 import FP16_Module


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    new_batch = {'text': batch['text'].long().cuda().contiguous(), 'label': batch['label'].long().cuda().contiguous()}
    if "types" in batch:
        new_batch["types"] = batch['types'].long().cuda().contiguous()
    if "padding_mask" in batch:
        attention_mask = batch['padding_mask'].float().cuda().contiguous()
        if args.fp16:
            attention_mask = attention_mask.half()
        new_batch["attention_mask"] = attention_mask
    elif "mask" in batch:
        attention_mask = batch['mask'].long().cuda().contiguous()
        new_batch["attention_mask"] = attention_mask
    if "target" in batch:
        new_batch["target"] = batch['target'].long().cuda().contiguous()
    if "logit_mask" in batch:
        new_batch["logit_mask"] = batch['logit_mask'].long().cuda().contiguous()
    if "position" in batch:
        new_batch["position"] = batch['position'].long().cuda().contiguous()
    if "loss_mask" in batch:
        new_batch["loss_mask"] = batch["loss_mask"].float().cuda().contiguous()
        if args.fp16:
            new_batch['loss_mask'] = new_batch['loss_mask'].half()
    if "segment_id" in batch:
        new_batch["segment_id"] = batch["segment_id"].long().cuda().contiguous()
    return new_batch
    # if args.fp16:
    #     attention_mask = attention_mask.half()
    # position_ids = torch.arange(tokens.size(-1), dtype=torch.long, device=tokens.device)
    # block_position_ids = tokens.new_zeros(tokens.size(-1)).unsqueeze(0).unsqueeze(0).expand_as(tokens)
    # position_ids = position_ids.unsqueeze(0).unsqueeze(0).expand_as(tokens)
    # position_ids = torch.stack((position_ids, block_position_ids), dim=2)


tokenizer = None


def finetune_forward_step(batch, model, args, timers, mems):
    """Simple forward step with cross-entropy loss."""
    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    data = process_batch(batch_, args)
    timers('batch generator').stop()

    # Forward model.
    if args.pretrained_bert:
        tokens, types, labels, attention_mask = data['text'], data['types'], data['label'], data['attention_mask']
        logits = model(tokens, token_type_ids=types, attention_mask=attention_mask, checkpoint_activations=True)
    elif args.cloze_eval:
        tokens, labels, position_ids = data['text'], data['label'], data['position']
        attention_mask, target_ids, logit_mask = data['attention_mask'], data['target'], data['logit_mask']

        def print_masked_text(batch_id):
            output_tokens = []
            sep = attention_mask[batch_id].item()
            for i, token in enumerate(tokens[batch_id][:sep].tolist()):
                token = tokenizer.IdToToken(token)
                if token == '[MASK]':
                    token = f"[{position_ids[batch_id][0, i].item()}]"
                output_tokens.append(token)
            print(" ".join(output_tokens))
            target_positions = []
            for i in range(sep, tokens.size(-1)):
                if logit_mask[batch_id][i]:
                    target_positions.append(i)
            print(target_positions)
            print(tokenizer.DecodeIds(tokens[batch_id][target_positions].tolist()))
            print(tokenizer.DecodeIds(target_ids[batch_id][target_positions].tolist()))
            print(position_ids[batch_id][:, target_positions])

        logits, *mems = model(tokens, position_ids, attention_mask, target_ids, logit_mask)
    else:
        tokens, labels, position_ids, attention_mask = data['text'], data['label'], data['position'], data[
            'attention_mask']
        logits, *mems = model(tokens, position_ids, attention_mask)
    if "segment_id" in data:
        if "loss_mask" in data:
            logits = logits * data["loss_mask"]
        logits = scatter_sum(logits, data["segment_id"], dim=1)
    elif "loss_mask" in data:
        loss_mask = data["loss_mask"]
        logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)
    if args.loss_func == "cross_entropy":
        # Cross-entropy loss.
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(logits.contiguous().float(), labels)
    elif args.loss_func == "hinge":
        correct_logits = logits[range(logits.size(0)), labels]
        hinge_loss = 1 + logits - correct_logits.unsqueeze(1)
        hinge_loss[hinge_loss < 0.0] = 0.0
        loss = hinge_loss.sum(dim=1).mean() - 1.0
    else:
        raise NotImplementedError

    # Reduce loss for logging.

    return loss, mems, 'bert'


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset, args):
    """Traing and validation dataloaders."""
    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset, args.batch_size, args.num_workers, drop_last=False)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader = None
    if valid_dataset is not None:
        valid_dataloader_ = build_data_loader(valid_dataset, args.batch_size,
                                              args.num_workers, drop_last=False)
        valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

    return train_dataloader, valid_dataloader


def _train(model, optimizer, lr_scheduler, forward_step,
           train_dataloader, valid_dataloader, end_of_epoch_callback, args, timers, summary_writer=None):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    args.iteration = 0
    total_lm_loss = 0.0
    best_score, best_iteration = 0, None
    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch

    # For each remaining epoch
    timers('interval time').start()
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(args.seed + epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Train for one step.
            lm_loss, skipped_iter, _ = train_step(batch, model, optimizer, lr_scheduler, args, timers,
                                                  forward_step_func=forward_step)
            args.iteration += 1
            total_lm_loss += lm_loss.data.detach().float()

            # Logging.
            if args.iteration % args.log_interval == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                avg_lm_loss = total_lm_loss.item() / args.log_interval
                elapsed_time = timers('interval time').elapsed()
                report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                         elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters,
                                         args)
                total_lm_loss = 0.0

            # Evaluation
            if args.eval_interval and valid_dataloader is not None and args.iteration % args.eval_interval == 0:
                prefix = 'iteration {}'.format(args.iteration)
                evaluate_and_print_results(prefix, valid_dataloader, model, args, timers, step=args.iteration,
                                           verbose=False, forward_step_func=finetune_forward_step,
                                           summary_writer=summary_writer)

        # Checkpointing at the end of each epoch.
        if args.save and (epoch + 1) % args.save_epoch == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None and (epoch + 1) % args.eval_epoch == 0:
            score_dict = end_of_epoch_callback(model, epoch, summary_writer=summary_writer)
            validation_metric = args.validation_metric if args.validation_metric else list(score_dict.keys())[0]
            validation_score = score_dict[validation_metric]
            if best_iteration is None or validation_score > best_score:
                best_iteration = args.iteration
                best_score = validation_score
                print_rank_0(f"Found best {validation_metric} {best_score} at {best_iteration}")
                if torch.distributed.get_rank() == 0:
                    save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args, tag="best", barrier=False)
                    score_dict.update({"type": "validation", "epoch": epoch})
                    with open(os.path.join(args.log_dir, "results.json"), "w") as output:
                        output.write(json.dumps(score_dict) + "\n")
                    with open(os.path.join(args.save, "best_checkpointed_iteration.txt"), "w") as output:
                        output.write(str(best_iteration))
    return best_iteration


def finetune(args, train_valid_datasets_provider, model_kwargs,
             forward_step=finetune_forward_step,
             end_of_epoch_callback_provider=None):
    """Main finetune function used across all tasks."""
    global tokenizer
    timers = Timers()
    tokenizer = prepare_tokenizer(args)
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder').start()
    train_dataloader, valid_dataloader = None, None
    if train_valid_datasets_provider is not None and args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider(args, tokenizer)
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset, args)
    timers('train/valid/test dataset/dataloder').stop()
    # Build calback function.
    timers('callback function').start()
    end_of_epoch_callback, end_of_train_callback = None, None
    if end_of_epoch_callback_provider is not None:
        if train_valid_datasets_provider is not None and args.epochs > 0:
            end_of_epoch_callback = end_of_epoch_callback_provider(args, tokenizer, is_test=False)
        end_of_train_callback = end_of_epoch_callback_provider(args, tokenizer, is_test=True)
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, **model_kwargs)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint').start()
    if args.load_pretrained is not None and not args.pretrained_bert and not args.load:
        module = model
        if isinstance(module, (LocalDDP, TorchDDP)):
            module = module.module
        if isinstance(module, FP16_Module):
            module = module.module
        if not isinstance(module, GPT2Model):
            module = module.model
        args.load = args.load_pretrained
        load_checkpoint(module, optimizer, lr_scheduler, args)
        args.load = None
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16:
            optimizer._model_params_to_master_params()
    if args.load is not None:
        load_checkpoint(model, optimizer, lr_scheduler, args)
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16:
            optimizer._model_params_to_master_params()
    timers('pretrained checkpoint').stop()
    args.iteration = 0
    summary_writer = None
    if torch.distributed.get_rank() == 0:
        args.log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
        if os.path.exists(args.log_dir) and args.load is None:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.log_dir))
        summary_writer = get_sample_writer(log_dir=args.log_dir, iteration=args.iteration)
        print_and_save_args(args, verbose=False, log_dir=args.log_dir)

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder', 'callback function',
                'model and optimizer', 'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    score_dict = None
    if train_dataloader is not None and args.epochs > 0:
        best_iteration = _train(model, optimizer, lr_scheduler, forward_step,
                                train_dataloader, valid_dataloader, end_of_epoch_callback, args, timers,
                                summary_writer=summary_writer)
        if best_iteration is not None and end_of_train_callback is not None:
            args.load = os.path.join(args.save, "best")
            load_checkpoint(model, optimizer, lr_scheduler, args)
            args.load = None
        if end_of_train_callback is not None:
            score_dict = end_of_train_callback(model, epoch=-1, output_predictions=True)
    # Or just evaluate.
    else:
        if end_of_train_callback is not None:
            print_rank_0('evaluation only mode, setting epoch to -1')
            score_dict = end_of_train_callback(model, epoch=-1, output_predictions=True)
    if score_dict is not None and torch.distributed.get_rank() == 0:
        score_dict.update({"type": "test"})
        with open(os.path.join(args.log_dir, "results.json"), "a") as output:
            output.write(json.dumps(score_dict) + "\n")

    print_rank_0('done :-)')


if __name__ == '__main__':
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    assert args.finetune

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    from tasks.superglue.dataset import PROCESSORS

    superglue_tasks = list(PROCESSORS.keys())
    if args.task.lower() == 'race':
        from tasks.race.finetune import main
    elif args.task.lower() in superglue_tasks:
        from tasks.superglue.finetune import main
    elif args.task.lower() in ['lambda', 'wikitext', 'language_model']:
        from tasks.language_model.finetune import main
    elif args.task.lower() in ['cnn_dm', 'gigaword']:
        from tasks.seq2seq.finetune import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(args.task))

    main(args)
