import os
import re
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import importlib.util

from transformers import AutoTokenizer, set_seed


def preprocess(args, dataset, tokenizer):
    inputs, targets, extra_id = [], [], []

    for item in tqdm(dataset):
        if item['input'] is None or item['target'] is None:
            continue
        inputs.append(item['input'])
        targets.append(item['target'])
        if 'answer' in item.keys():
            extra_id.append(item['answer'])

    assert len(inputs) == len(targets)

    print("Computing shortest max_len...")
    sequences = inputs + targets
    encoded_inputs = tokenizer.batch_encode_plus(sequences, padding='longest')
    max_seq_length = min(len(encoded_inputs['input_ids'][0]), args.max_length)

    print("Tokenizing...")
    input_ids = tokenizer.batch_encode_plus(inputs, max_length=max_seq_length, padding='max_length', truncation=True)
    source_ids = np.array(input_ids['input_ids'], dtype=np.int32)
    source_mask = np.array(input_ids['attention_mask'], dtype=np.int32)

    # target_ids = tokenizer.batch_encode_plus(targets, max_length=max_seq_length, padding='max_length', truncation=True)
    # target_ids = np.array(target_ids['input_ids'], dtype=np.int32)

    extra_id = np.array(extra_id) if extra_id else np.array([0] * len(inputs), dtype=np.int32)

    return source_ids, source_mask, targets, extra_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--model_name_or_path', required=True)

    parser.add_argument('--max_length', default=512, type=int)

    args = parser.parse_args()
    set_seed(42)

    for k, v in vars(args).items():
        print(k + ':' + str(v))

    try:
        spec = importlib.util.spec_from_file_location("config", args.config)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        train_set, val_set, test_set, *xargs = config.load_data(args)
        task_special_tokens = config.special_tokens
    except:
        raise Exception('Error loading config file')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if xargs:
        fn = os.path.join(args.output_dir, 'vocab.json')
        with open(fn, 'w') as f:
            json.dump(xargs[0], f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, revision='main')
    tokenizer.add_tokens(task_special_tokens)
    print('Tokenizer loaded with domain specific special tokens added:')
    print(tokenizer.get_added_vocab())

    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = preprocess(args, dataset, tokenizer)
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                if type(o) == np.ndarray:
                    pickle.dump(o, f)
                else:
                    json.dump(o, open(os.path.join(args.output_dir, '{}.json'.format(name)), 'w'))


if __name__ == '__main__':
    main()
