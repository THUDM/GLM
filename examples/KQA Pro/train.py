import os
import sys
import time
from datetime import timedelta
import logging
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding

import transformers.utils.logging as transformers_logging

from utils.misc import seed_everything, ProgressBar
from utils.lr_scheduler import get_linear_schedule_with_warmup
from utils.data import DataLoader, DistributedDataLoader, prepare_dataset

from inference import validate

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()
import warnings
warnings.simplefilter("ignore")


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.local_rank in [-1, 0]:
        logging.info("Create train_loader and val_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    train_json = os.path.join(args.input_dir, 'train.json')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    val_json = os.path.join(args.input_dir, 'val.json')

    if args.n_gpus > 1:
        train_dataset, train_vocab = prepare_dataset(vocab_json, train_pt, train_json, training=True)
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DistributedDataLoader(train_dataset, train_vocab, args.batch_size//args.n_gpus, train_sampler)
    else:
        train_loader = DataLoader(vocab_json, train_pt, train_json, args.batch_size, training=True)

    val_loader = DataLoader(vocab_json, val_pt, val_json, args.batch_size // args.n_gpus * 2, training=False)
    if args.local_rank in [-1, 0]:
        logging.info("Create model.........")

    _, model_class, tokenizer_class = (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, trust_remote_code=True, revision='main')

    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        task_special_tokens = config.special_tokens
        tokenizer.add_tokens(task_special_tokens)
        if args.local_rank in [-1, 0]:
            logging.info("Add {} special tokens.".format(len(task_special_tokens)))
    except:
        raise Exception('Error loading config file')

    transformers_logging.set_verbosity_error()
    if args.local_rank in [-1, 0]:
        logging.info("Initiating model parameters.........")

    model = model_class.from_pretrained(args.ckpt) if args.ckpt else model_class.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, revision='main')

    if args.n_gpus > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    else:
        model = model.to(device)

    # Prepare optimizer and schedule (linear warmup and decay)
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = []
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train
    if args.local_rank in [-1, 0]:
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_loader.dataset))
        logging.info("  Num Epochs = %d", args.num_train_epochs)
        logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logging.info("  Total optimization steps = %d", t_total)

    # Check if continuing training from a checkpoint
    if args.ckpt and args.local_rank in [-1, 0]:
        logging.info("Continuing training from checkpoint, will skip to saved global_step")

    global_step = 0
    total_loss = 0.0
    best_acc, current_acc = 0.0, 0.0

    model.zero_grad()
    # Run validate() once to ensure correct implementation of inference
    '''if args.local_rank in [-1, 0]:
        current_acc, _ = validate(args, model, val_loader, device, tokenizer)
        print("Current performance on validation set: %f" % (current_acc))'''

    epochs_not_improving = 0

    for epoch_i in range(int(args.num_train_epochs)):
        if args.n_gpus > 1:
            train_loader.sampler.set_epoch(epoch_i)
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        if args.local_rank in [-1, 0]:
            epochs_not_improving += 1
        for step, batch in enumerate(train_loader):
            model.train()
            source_ids, source_masks, _, target = batch
            inputs = BatchEncoding(data={"input_ids": source_ids, "attention_mask": source_masks})
            if args.classification:
                choices = [args.choices for _ in range(len(source_ids))]
                inputs = tokenizer.build_inputs_for_generation(inputs, choices, targets=target, padding=False)
            else:
                inputs = tokenizer.build_inputs_for_generation(inputs, targets=target, padding=False)
            inputs = inputs.to(device)
            outputs = model(**inputs)

            loss = outputs.loss
            if args.n_gpus > 1:
                dist.barrier()

            loss.backward()
            loss_num = loss.item()
            pbar_info = {'loss': loss_num}
            pbar(step, pbar_info)
            total_loss += loss_num
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        if args.n_gpus > 1:
            dist.barrier()

        if args.local_rank in [-1, 0]:
            logging.info("Epoch %d loss: %.3f" % (epoch_i, loss_num))
            current_acc, _ = validate(args, model, val_loader, device, tokenizer)

        if args.local_rank in [-1, 0] and current_acc > best_acc:
            epochs_not_improving = 0
            best_acc = current_acc
            print("Best performance on validation set updated: %f" % (best_acc))
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-best")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # Take care of distributed/parallel training
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logging.info("Saving model checkpoint to %s", output_dir)
            tokenizer.save_vocabulary(output_dir)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logging.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.n_gpus > 1:
            dist.barrier()

        if 'cuda' in str(device):
            torch.cuda.empty_cache()

        if epochs_not_improving > args.early_stopping:
            logging.info("%d epochs not improving, training early stopped" % epochs_not_improving)
            dist.destroy_process_group()
            return global_step, total_loss / global_step

    return global_step, total_loss / global_step


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', default=None)

    parser.add_argument('--classification', default=False, type=bool)
    parser.add_argument('--choices', default=["1", "2", "3", "4", "5"], type=list)

    # training parameters
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--learning_rate', default=3e-5, type=float)
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--early_stopping', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1=10% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--eval_max_length", default=500, type=int,
                        help="Eval max length.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="Beam size for inference.")

    # special parameters
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--port', default=12355, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, '{}.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    if args.classification:
        assert args.choices is not None

    seed_everything(args.seed)
    args.inference = False

    # distributed data parallel
    args.n_gpus = torch.cuda.device_count()
    if args.n_gpus > 1:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(args.port)
        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
        dist.init_process_group(backend='nccl', world_size=args.n_gpus, timeout=timedelta(5400000))
        torch.cuda.set_device(args.local_rank)

    # args display
    if args.local_rank in [-1, 0]:
        for k, v in vars(args).items():
            logging.info(k + ':' + str(v))

    train(args)

    if args.n_gpus > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

