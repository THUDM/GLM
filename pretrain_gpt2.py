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

"""Pretrain GPT2"""

# Flag to use Pytorch ddp which uses overlapping communication and computation.
from datetime import datetime
import os
import random
import math
from filelock import FileLock
import numpy as np
import torch

import deepspeed
from contextlib import ExitStack
from arguments import get_args
from configure_data import configure_data, prepare_tokenizer
import mpu

from train_utils import setup_model_and_optimizer, train_step
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_and_save_args
from utils import print_rank_0
from utils import get_sample_writer, get_log_dir
import torch.distributed as dist


def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask,
                               loss_mask=None,
                               attention_mask=None,
                               transformer_xl=False,
                               mem_length=None):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if transformer_xl:
        if attention_mask is None:
            attention_mask = torch.ones((1, seq_length, seq_length + mem_length), device=data.device)
        attention_mask = torch.tril(torch.triu(attention_mask, 1 - seq_length + mem_length), mem_length)
    else:
        if reset_attention_mask:
            att_mask_batch = batch_size
        else:
            att_mask_batch = 1
        if attention_mask is None:
            attention_mask = torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
        attention_mask = torch.tril(attention_mask)
    attention_mask = attention_mask.unsqueeze(1)

    # Loss mask.
    if loss_mask is None:
        loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    if not transformer_xl:
        loss_mask[data == eod_token] = 0.0
        # We need to clone as the ids will be modifed based on batch index.
        if reset_position_ids:
            position_ids = position_ids.clone()

        if reset_position_ids or reset_attention_mask:
            # Loop through the batches:
            for b in range(batch_size):

                # Find indecies where EOD token is.
                eod_index = position_ids[b, data[b] == eod_token]
                # Detach indecies from positions if going to modify positions.
                if reset_position_ids:
                    eod_index = eod_index.clone()

                # Loop through EOD indecies:
                prev_index = 0
                for j in range(eod_index.size()[0]):
                    i = eod_index[j]
                    # Mask attention loss.
                    if reset_attention_mask:
                        attention_mask[b, 0, (i + 1):, :(i + 1)] = 0
                    # Reset positions.
                    if reset_position_ids:
                        position_ids[b, (i + 1):] -= (i + 1 - prev_index)
                        prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(data_iterator, args, timers):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    keys = ['text', 'loss_mask']
    if args.transformer_xl or args.block_lm:
        keys += ['target', 'attention_mask']
    if args.block_lm:
        keys += ['position_id']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    if args.transformer_xl:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].float()
        loss_mask = data_b['loss_mask'].float()
    elif args.block_lm:
        tokens = data_b['text'].long()
        labels = data_b['target'].long()
        attention_mask = data_b['attention_mask'].long()
        loss_mask = data_b['loss_mask'].float()
        position_ids = data_b['position_id'].long()
    else:
        tokens_ = data_b['text'].long()
        loss_mask = data_b['loss_mask'].float()
        labels = tokens_[:, 1:].contiguous()
        loss_mask = loss_mask[:, 1:].contiguous()
        tokens = tokens_[:, :-1].contiguous()
        attention_mask = None

    # Get the masks and postition ids.
    if not args.block_lm:
        attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
            tokens,
            args.eod_token,
            args.reset_position_ids,
            args.reset_attention_mask,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            transformer_xl=args.transformer_xl,
            mem_length=args.mem_length)
        # Convert
        if args.fp16:
            attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids


tokenizer = None


def forward_step(data_iterator, model, args, timers, mems):
    """Forward step."""

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    def print_masked_text(batch_id):
        if not args.no_block_position:
            block_position_ids = position_ids[:, 1]
            position_ids_ = position_ids[:, 0]
        else:
            position_ids_ = position_ids
        output_tokens = []
        sep = attention_mask.item()
        for i, token in enumerate(tokens[batch_id, :sep].tolist()):
            token = tokenizer.IdToToken(token)
            if token == '[MASK]':
                token = f"[{position_ids_[batch_id, i].item()}]"
            output_tokens.append(token)
        print(" ".join(output_tokens))
        last_index = None
        last_position = None
        for i in range(sep, tokens.size(1)):
            if (not args.no_block_position and block_position_ids[batch_id, i] == 1) or (args.no_block_position and (
                    last_position is None or position_ids_[batch_id, i] != last_position + 1)):
                if last_index is not None:
                    print(tokenizer.DecodeIds(tokens[batch_id, last_index: i].tolist()), ";",
                          tokenizer.DecodeIds(labels[batch_id, last_index: i].tolist()),
                          position_ids_[batch_id, last_index: i].tolist())
                last_index = i
            last_position = position_ids_[batch_id, i]
        if last_index is not None:
            print(tokenizer.DecodeIds(tokens[batch_id, last_index:].tolist()), ";",
                  tokenizer.DecodeIds(labels[batch_id, last_index:].tolist()),
                  position_ids_[batch_id, last_index:].tolist())

    if tokens.size(1) <= args.seq_length + 2:
        mode = 'gpt'
    else:
        mode = 'bert'

    # Forward model.
    logits, *mems = model(tokens, position_ids, attention_mask, *mems)
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(),
                                              labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    return loss, mems, mode


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' lm loss {:.6E} |'.format(loss)
    if args.fp16:
        log_string += ' loss scale {:.1f} |'.format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
    print_rank_0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, gpt_loss, bert_loss, step):
    string = ' validation loss at {}'.format(prefix)
    string += ' | LM loss: {:.6E}'.format(loss)
    string += ' | LM PPL: {:.6E}'.format(ppl)
    if gpt_loss != 0 and bert_loss != 0:
        string += ' | GPT loss: {:.6E}'.format(gpt_loss)
        string += ' | BERT loss: {:.6E}'.format(bert_loss)
    length = len(string) + 1
    print_rank_0('-' * 100)
    print_rank_0('-' * length)
    print_rank_0(string)
    print_rank_0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        if gpt_loss != 0:
            summary_writer.add_scalar(f'Train/valid_gpt_loss', gpt_loss, step)
        if bert_loss != 0:
            summary_writer.add_scalar(f'Train/valid_bert_loss', bert_loss, step)


def train(model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args, summary_writer=None):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    mems = []
    while args.iteration < args.train_iters:

        lm_loss, skipped_iter, mems = train_step(train_data_iterator,
                                                 model,
                                                 optimizer,
                                                 lr_scheduler,
                                                 args, timers, mems=mems, forward_step_func=forward_step)
        skipped_iters += skipped_iter
        args.iteration += 1

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False
            if args.deepspeed or args.DDP_impl == 'torch':
                timers.log(['forward', 'backward', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
            else:
                timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                            'batch generator', 'data loader'],
                           normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, args, timers, verbose=False, step=args.iteration,
                summary_writer=summary_writer, forward_step_func=forward_step)

    return args.iteration, skipped_iters


def evaluate(data_iterator, model, args, timers, forward_step_func, verbose=False):
    """Evaluation."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss, total_gpt_loss, total_mask_loss = 0, 0, 0
    gpt_iters, mask_iters = 0, 0
    mems = []
    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank_0('Evaluating iter {}/{}'.format(iteration, args.eval_iters))
            # Forward evaluation.
            lm_loss, mems, mode = forward_step_func(data_iterator, model, args, timers, mems=mems)

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            # Reduce across processes.
            torch.distributed.all_reduce(lm_loss.data)
            lm_loss.data = lm_loss.data / args.world_size

            lm_loss = lm_loss.data.detach().float().item()
            total_lm_loss += lm_loss
            if mode == 'gpt':
                total_gpt_loss += lm_loss
                gpt_iters += 1
            elif mode == 'bert':
                total_mask_loss += lm_loss
                mask_iters += 1

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= args.eval_iters
    total_gpt_loss = total_gpt_loss / gpt_iters if gpt_iters > 0 else 0
    total_mask_loss = total_mask_loss / mask_iters if mask_iters > 0 else 0
    return total_lm_loss, total_gpt_loss, total_mask_loss


def evaluate_and_print_results(prefix, data_iterator, model,
                               args, timers, forward_step_func, verbose=False, step=None, summary_writer=None):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, gpt_loss, bert_loss = evaluate(data_iterator, model, args, timers, verbose=verbose,
                                            forward_step_func=forward_step_func)

    lm_ppl = math.exp(min(20, lm_loss))
    report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, gpt_loss, bert_loss, step)

    return lm_loss


'''
    Optional DeepSpeed Activation Checkpointing features
    Gives access to partition activations, contiguous memory optimizations
    and cpu checkpointing.

    Activation checkpoint requires keep track of the random states
    and setting the random seed for each MP process. Megatron uses
    mpu.get_cuda_rng_tracker and mpu.model_parallel_cuda_manual_seed
    for keeping track of the random states and setting the random seeds.
    Since they are used in places outside of activation checkpointing,
    we overwrite them to maintain consistency.

    This must be done before all the calls to mpu.model_parallel_cuda_manual_seed
    '''


def set_deepspeed_activation_checkpointing(args):
    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size, rank=args.rank,
        init_method=init_method)

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if hasattr(args, "deepspeed") and args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def get_train_val_test_data(args, tokenizer):
    """Load the data on rank zero and boradcast number of tokens to all GPUS."""

    (train_data, val_data, test_data) = (None, None, None)
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        data_config = configure_data()
        data_config.set_defaults(data_set_type='Block' if args.block_lm else 'GPT2', transpose=False)
        train_data, val_data, test_data = data_config.apply(args, tokenizer)

        data_counts = torch.cuda.LongTensor([int(args.do_train), int(args.do_valid), int(args.do_test)])
    else:
        data_counts = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(data_counts,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = data_counts[0].item()
    args.do_valid = data_counts[1].item()
    args.do_test = data_counts[2].item()

    return train_data, val_data, test_data


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False
    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()
    args.mem_length = args.mem_length if args.transformer_xl else 0
    if args.load:
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + datetime.now().strftime("%m-%d-%H-%M")
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # Data stuff.
    global tokenizer
    tokenizer = prepare_tokenizer(args)
    train_data, val_data, test_data, = get_train_val_test_data(args, tokenizer)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)

    if args.load is not None:
        with FileLock("/root/checkpoint_lock", timeout=-1):
            args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0
    torch.distributed.barrier()

    summary_writer = None
    if torch.distributed.get_rank() == 0:
        print('Pretrain GPT2 model')
        log_dir = None
        if args.train_iters > 0:
            log_dir = get_log_dir(base=args.summary_dir, name=args.experiment_name)
            summary_writer = get_sample_writer(log_dir=log_dir, iteration=args.iteration)
        print_and_save_args(args, verbose=True, log_dir=log_dir)

    # Resume data loader if necessary.
    if args.resume_dataloader:
        if train_data is not None:
            train_data.batch_sampler.start_iter = args.iteration % \
                                                  len(train_data)
        if val_data is not None:
            start_iter_val = (args.train_iters // args.save_interval) * \
                             args.eval_interval
            val_data.batch_sampler.start_iter = start_iter_val % \
                                                len(val_data)
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

    # TODO: figure out how to properly set this especially when resuming training
    iteration = 0
    if args.train_iters > 0:
        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)

                # stack.callback(save_on_exit, args, model, optimizer, lr_scheduler)
                iteration, skipped = train(model, optimizer,
                                           lr_scheduler,
                                           train_data_iterator,
                                           val_data_iterator,
                                           timers, args, summary_writer=summary_writer)

        if args.do_valid:
            prefix = 'the end of training for val data'
            val_loss = evaluate_and_print_results(prefix, val_data_iterator,
                                                  model, args, timers, verbose=False, forward_step_func=forward_step)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    if test_data is not None:
        test_data_iterator = iter(test_data)
    else:
        test_data_iterator = None

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(prefix, test_data_iterator,
                                   model, args, timers, verbose=True, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
