import os
import sys

import torch
import numpy as np


import argparse
import importlib.util
from tqdm import tqdm

from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, BatchEncoding
from utils.misc import seed_everything


import logging
import time
from utils.data import DataLoader


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()


def validate(args, model, data, device, tokenizer):
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    except:
        raise Exception('Error loading config file')

    args.tokenizer = tokenizer
    model.eval()
    model = model.module if hasattr(model, "module") else model

    all_outputs = []
    all_targets = []
    all_answer_ids = []

    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            source_ids, source_masks, answer_ids, target = batch

            inputs = BatchEncoding(data={"input_ids": source_ids, "attention_mask": source_masks})
            if args.classification:
                choices = [args.choices for _ in range(len(source_ids))]
                inputs = tokenizer.build_inputs_for_generation(inputs, choices, max_gen_length=args.eval_max_length, padding=False)
            else:
                inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=args.eval_max_length, padding=False)
            inputs = inputs.to(device)

            outputs = model.module.generate(
                **inputs, max_length=512, eos_token_id=tokenizer.eop_token_id
            ) if hasattr(model, "module") else model.generate(
                **inputs, max_length=512, eos_token_id=tokenizer.eop_token_id
            )

            outputs = list(outputs.cpu())
            for i in range(len(outputs)):
                outputs[i] = outputs[i].tolist()
                try:
                    outputs[i] = outputs[i][
                                 outputs[i].index(tokenizer.sop_token_id) + 1:outputs[i].index(tokenizer.eop_token_id)]
                except ValueError:
                    continue

            all_outputs.extend(np.array(outputs))
            all_targets.extend(target)
            all_answer_ids.extend(answer_ids.cpu().numpy())

        assert len(all_outputs) == len(all_targets)
        outputs = tokenizer.batch_decode(all_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        targets = all_targets

        print("Target sample sequence: %s " % targets[-1])
        print("Output sample sequence: %s " % outputs[-1])

    with open(os.path.join(args.output_dir, 'output.txt'), 'w') as f:
        for output in outputs:
            f.write(output + '\n')

    str_matching = np.mean([1 if p.strip() == g.strip() else 0 for p, g in zip(outputs, targets)])
    lf_matching = config.evaluate(args, outputs, targets, all_answer_ids, data)
    logging.info('Execution accuracy: {}, String matching accuracy: {}'.format(lf_matching, str_matching))

    return lf_matching, outputs


def inference(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader and test_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    test_json = os.path.join(args.input_dir, 'test.json')
    test_loader = DataLoader(vocab_json, test_pt, test_json, args.batch_size, training=False)

    logging.info("Create model.........")
    config_class, model_class, tokenizer_class = AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, trust_remote_code=True, revision='main')
    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        task_special_tokens = config.special_tokens
        tokenizer.add_tokens(task_special_tokens)
    except:
        raise Exception('Error loading config file')

    model = model_class.from_pretrained(args.ckpt, trust_remote_code=True, revision='main')
    model = model.to(device)

    _, outputs = validate(args, model, test_loader, device, tokenizer)
    with open("output.txt", "w") as f:
        for output in outputs:
            f.write(output + "\n")


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--ckpt', required=True)

    parser.add_argument('--classification', default=False, type=bool)
    parser.add_argument('--choices', default=["1", "2", "3", "4", "5"], type=list)

    # inference parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument("--eval_max_length", default=500, type=int,
                        help="Eval max length.")
    parser.add_argument("--beam_size", default=1, type=int,
                        help="Beam size for inference.")

    args = parser.parse_args()
    args.inference = True
    args.local_rank = -1

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    fileHandler = logging.FileHandler(os.path.join(args.output_dir, '{}.predict.log'.format(time_)))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    if args.classification:
        assert args.choices is not None

    for k, v in vars(args).items():
        logging.info(k + ':' + str(v))

    seed_everything(args.seed)
    inference(args)


if __name__ == '__main__':
    main()



