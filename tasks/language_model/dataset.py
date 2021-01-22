import torch
import json
import math
import numpy as np
from utils import print_rank_0
from tasks.data_utils import build_input_from_ids, num_special_tokens_to_add
from tasks.language_model.detokenizer import get_detokenizer
from bisect import bisect_right
from itertools import accumulate


class LMDataset(torch.utils.data.Dataset):

    def __init__(self, documents, max_seq_length, tokenizer, num_original_tokens,
                 num_tokenized_tokens, overalapping_eval=None):
        self.documents = documents
        self.max_seq_len = max_seq_length
        self.tokenizer = tokenizer
        self.overalapping_eval = overalapping_eval
        if self.overalapping_eval is None:
            self.overalapping_eval = self.max_seq_len
        self.overalapping_eval = max(1, self.overalapping_eval)
        self.num_original_tokens = num_original_tokens
        self.num_tokenized_tokens = num_tokenized_tokens
        # remove first sequence tokens
        targets = [max(len(tokens) - self.max_seq_len, 0) for tokens in self.documents]
        self.num_sequences = [max(math.ceil(target / self.overalapping_eval) + 1, 1) for target in targets]
        self.weights = list(accumulate(self.num_sequences))
        self.left_weights = [0] + self.weights[:-1]

    def __len__(self):
        return sum(self.num_sequences)

    def __getitem__(self, idx):
        document_idx = bisect_right(self.weights, idx)
        idx = idx - self.left_weights[document_idx]
        start_idx = idx * self.overalapping_eval
        end_idx = start_idx + self.max_seq_len
        tokens = self.documents[document_idx][start_idx:end_idx]
        if idx == 0:
            prompt, text = tokens[:1], tokens[1:]
        else:
            prompt_length = self.max_seq_len - self.overalapping_eval
            prompt, text = tokens[:prompt_length], tokens[prompt_length:]
        prompt = prompt + [self.tokenizer.get_command('MASK').Id]
        num_special_tokens = num_special_tokens_to_add(prompt, None, text, add_cls=False, add_sep=False, add_piece=True,
                                                       add_eos=False)
        data = build_input_from_ids(prompt, None, text, self.max_seq_len + num_special_tokens + 1, self.tokenizer,
                                    add_cls=False, add_sep=False, add_piece=True, add_eos=False)
        ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
        return {'text': np.array(ids, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64), 'loss_mask': np.array(loss_masks, dtype=np.int64),
                "position_id": np.array(position_ids, dtype=np.int64)}


class LambadaDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, max_seq_length, strict=True):
        print_rank_0('> building lambada dataset from {} ...'.format(data_path))
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.get_command('pad').Id
        self.strict = strict

        self.tokens = []
        self.labels = []
        with open(data_path, 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = self.get_tokens(text)
                self.tokens.append(tokens)
                self.labels.append(labels)

    def get_tokens(self, text):
        if not self.strict:
            tokens = self.tokenizer.EncodeAsIds(text).tokenization
            return tokens[:-1], [tokens[-1]]
        last_token = text.split()[-1]
        start_idx = text.rfind(last_token)
        beginning_tokens = self.tokenizer.EncodeAsIds(text[:start_idx].strip()).tokenization
        last_token = self.tokenizer.EncodeAsIds(' ' + last_token).tokenization
        return beginning_tokens, last_token

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens, answer = self.tokens[idx], self.labels[idx]
        tokens = tokens + [self.tokenizer.get_command('MASK').Id]
        data = build_input_from_ids(tokens, None, answer, self.max_seq_length, self.tokenizer,
                                    add_cls=True, add_sep=False, add_piece=True)
        ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
        return {'text': np.array(ids, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                'attention_mask': np.array(sep, dtype=np.int64), 'loss_mask': np.array(loss_masks, dtype=np.int64),
                "position_id": np.array(position_ids, dtype=np.int64)}


def build_lambada_dataset(tokenizer, args):
    """Build lambada dataset."""
    assert len(args.valid_data) == 1
    val_dataset = LambadaDataset(args.valid_data[0], tokenizer, args.seq_length, True)
    print_rank_0(' > found {} samples.'.format(len(val_dataset)))
    return val_dataset


def build_lm_dataset(tokenizer, args):
    documents = []
    num_tokens, num_original_tokens = 0, 0
    with open(args.valid_data[0], encoding='utf-8') as file:
        for line in file:
            tokens = tokenizer.EncodeAsIds(line.strip()).tokenization
            num_tokens += len(tokens)
            num_original_tokens += len(line.strip().split(" "))
            documents.append(tokens)
    val_dataset = LMDataset(documents, args.seq_length, tokenizer, num_original_tokens, num_tokens,
                            args.overlapping_eval)
    print_rank_0(
        ' > number of document: {}, number of original tokens {}, number of detokenized tokens: {}'.format(
            len(documents), num_original_tokens, num_tokens))
    return val_dataset


def build_wikitext103_dataset(tokenizer, args):
    """"""

    assert len(args.valid_data) == 1
    with open(args.valid_data[0], "rb") as reader:
        entire_data = reader.read().decode('utf-8')
    num_original_tokens = len(entire_data.strip().split(" "))
    entire_data = get_detokenizer('wikitext')(entire_data)
    tokenized_data = tokenizer.EncodeAsIds(entire_data).tokenization
    num_tokenized_tokens = len(tokenized_data)

    val_dataset = LMDataset([tokenized_data], args.seq_length, tokenizer,
                            num_original_tokens, num_tokenized_tokens,
                            args.overlapping_eval)
    print_rank_0(' > number of original tokens: {}, number of detokenized '
                 'tokens: {}'.format(num_original_tokens, num_tokenized_tokens))
    return val_dataset
