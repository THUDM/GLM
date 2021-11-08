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
"""Utilities for using and training tokenizers (char, wordpiece, sentencepiece)"""

from collections import namedtuple
import random
import os
import csv
import torch
import itertools


from .tokenization_gpt2 import GPT2Tokenizer
from .sp_tokenizer import SentencePieceTokenizer

class Tokenization(object):
    """
    Tokenization object to hold tokenization, (processed text),and original
    text. Can hold tokenization as Ids or tokens.

    It also holds command tokens (pad, unk, etc.) for the tokenization.
    This allows functions to pad/operate on tokenization without having
    access to the full tokenizer, just the tokenization.

    Several standard array operations are implemented (insert, append, extend).
    """

    def __init__(self, tokenization, text=None, original_text=None, command_tokens=None, asIds=True):
        self.tokenization = tokenization
        self.text = text
        if self.text is None:
            self.text = self.tokenization
        self.original_text = original_text
        if self.original_text is None:
            self.original_text = self.text
        self.command_tokens = command_tokens
        self.asIds = asIds
        self.parse_command_tokens()

    def set_command_tokens(self, command_tokens):
        self.command_tokens = command_tokens
        return self.parse_command_tokens()

    def parse_command_tokens(self):
        if self.command_tokens is None:
            return
        for command_token in self.command_tokens:
            if self.asIds:
                setattr(self, command_token.name, command_token.Id)
            else:
                setattr(self, command_token.name, command_token.token)

    def __getitem__(self, index):
        return self.tokenization[index]

    def __len__(self):
        return len(self.tokenization)

    def __str__(self):
        return f"Tokenization = {self.tokenization}, Text = {self.text}"

    def insert(self, idx, other):
        if isinstance(other, CommandToken):
            self.tokenization.insert(idx, other.Id)
            if idx == 0:
                self.text = other.token + self.text
                self.original_text = other.token + self.original_text
            elif idx == len(self.tokenization) - 1:
                self.text += other.token
                self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]
        else:
            self.tokenization = self.tokenization[:idx] + other.tokenization + self.tokenization[idx:]

    def append(self, other):
        if isinstance(other, CommandToken):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.append(other)
        return self

    def extend(self, other):
        if isinstance(other, CommandToken):
            self.tokenization.append(other.Id)
            self.text += other.token
            self.original_text += other.token
        elif isinstance(other, list) and isinstance(other[0], CommandToken):
            self.tokenization.extend([o.Id for o in other])
            self.text += [o.token for o in other]
            self.original_text += [o.token for o in other]
        elif isinstance(other, Tokenization):
            self.tokenization.extend(other.tokenization)
            self.text += other.text
            self.original_text += other.original_text
        else:
            self.tokenization.extend(other)
        return self


"""define some default command tokens for the tokenizer to use"""
token_format = "<{0}>"

COMMAND_TUPLE = namedtuple('CommandToken', ('name', 'token', 'Id'))


def prep_command_tokens(tokenlist, token_format=token_format):
    return [CommandToken(tok[0], token_format.format(tok[0]), tok[1]) for tok in tokenlist]


class CommandToken(object):
    def __init__(self, name, token, Id, lstrip=False, rstrip=False):
        self.name = name
        self.token = token
        self.Id = Id
        self.lstrip = lstrip
        self.rstrip = rstrip

    def __repr__(self):
        return str(COMMAND_TUPLE(self.name, self.token, self.Id))


class Tokenizer(object):
    """
    Tokenizer object that handles text tokenization, command tokens, and type tokens.

    Command tokens and text tokens are stored together in one mapping of size
    `len(text_tokenizer)+len(command_tokens)`. Command tokens are stored as first
    `len(command_tokens)` tokens. Token idx is stored at `idx+len(command_tokens)`.

    Token types are stored in a separate mapping of size `len(type_tokens)`.
    """

    def __init__(self, text_tokenizer, command_tokens=None):
        # set text tokenizer
        self.text_tokenizer = text_tokenizer
        if not hasattr(self, 'num_text_tokens'):
            self.num_text_tokens = len(self.text_tokenizer)
        print(command_tokens)
        self._command_tokens = command_tokens
        self.command_name_map = {tok.name: tok for tok in self.command_tokens}
        self.command_token_map = {tok.token: tok for tok in self.command_tokens}
        self.command_id_map = {tok.Id: tok for tok in self.command_tokens}

        # parse tokens and vocabs from tokenizer
        max_token_id = max(len(self.text_tokenizer.tokens) - 1, max(self.command_id_map.keys()))
        self._tokens = [self.text_tokenizer.tokens[i] if i < len(self.text_tokenizer.tokens) else f'[UNUSED{i}]' for i
                        in range(max_token_id + 1)]
        for idx, token in self.command_id_map.items():
            self._tokens[idx] = token.token
        self._vocab = {t.token: Id for Id, t in self.command_id_map.items()}
        self._vocab.update(self.text_tokenizer.vocab)

        if not hasattr(self, 'num_command_tokens'):
            self.num_command_tokens = len(self.command_tokens)
        if not hasattr(self, 'num_tokens'):
            self.num_tokens = len(self.tokens)

        self._text_tokens = list(self.text_tokenizer.tokens)
        self._text_token_vocab = {t: Id for t, Id in self.text_tokenizer.vocab.items()}

        self._command_token_tokens = list(self.command_token_map.keys())
        self._command_token_vocab = {t: Id for Id, t in self.command_id_map.items()}

        self.spaces_between_special_tokens = True

    @property
    def command_tokens(self):
        return self._command_tokens

    def __call__(self, text, process_fn=None):
        """run preprocessing and encode text as Ids"""
        return self.EncodeAsIds(text, process_fn=process_fn)

    def __len__(self):
        """total number of tokens"""
        return self.num_tokens

    def get_command(self, name):
        """get command token corresponding to `name`"""
        return self.command_name_map[name]

    @property
    def tokens(self):
        """list (or iterable) of all tokens for tokenizer"""
        return self._tokens

    @property
    def vocab(self):
        """dictionary mapping tokens to ids for tokenizer"""
        return self._vocab

    @property
    def command_token_vocab(self):
        """dictionary mapping command tokens to ids for tokenizer"""
        return self._command_token_vocab

    @property
    def text_tokens(self):
        """list (or iterable) of text tokens for text tokenizer"""
        return self._text_tokens

    @property
    def text_token_vocab(self):
        """dictionary mapping text tokens to ids for text tokenizer"""
        return self._text_token_vocab

    def EncodeAsIds(self, text, process_fn=None):
        """
        encode text using text tokenizer and shift Id values for command tokens
        """
        processed_text = text
        if process_fn is not None:
            processed_text = process_fn(processed_text)

        def split_on_token(tok_extended: CommandToken, text):
            result = []
            tok = tok_extended.token
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # CommandToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # Strip white spaces on the right
                if tok_extended.rstrip and i > 0:
                    # A bit counter-intuitive but we strip the left of the string
                    # since tok_extended.rstrip means the special token is eating all white spaces on its right
                    sub_text = sub_text.lstrip()
                # Strip white spaces on the left
                if tok_extended.lstrip and i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()  # Opposite here

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self.text_tokenizer.encode(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self._command_token_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._encode(token) if token not in self._command_token_tokens else [
                            self.command_token_map[token].Id] for token in tokenized_text
                    )
                )
            )

        no_split_tokens = self._command_tokens
        Ids = split_on_tokens(no_split_tokens, processed_text)
        tokenization = Tokenization(Ids, processed_text, text)
        tokenization.set_command_tokens(self._command_tokens)
        return tokenization

    def _encode(self, text):
        raise NotImplementedError

    def _decode(self, ids):
        raise NotImplementedError

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        return out_string

    def EncodeAsTokens(self, text, process_fn=None):
        """
        encode text as tokens using text tokenizer
        """
        tokenization = self.EncodeAsIds(text, process_fn=process_fn)
        tokenization.tokenization = [self.IdToToken(idx) for idx in tokenization.tokenization]
        return tokenization

    def IdToToken(self, Id):
        """convert Id to token accounting for command tokens"""
        if isinstance(Id, CommandToken):
            return Id.token
        return self.tokens[Id]

    def TokenToId(self, token):
        """convert token to Id accounting for command tokens"""
        if isinstance(token, CommandToken):
            return token.Id
        return self.vocab[token]

    def DecodeIds(self, Ids):
        """
        convert Ids to tokens accounting for command tokens, tokens
        are joined and returned as a string.
        """
        rtn_strs = []
        current_str = []
        if isinstance(Ids, Tokenization):
            Ids = Ids.tokenization
        for Id in Ids:
            if isinstance(Id, CommandToken):
                rtn_strs.append(self._decode(current_str))
                current_str = []
                rtn_strs.append(Id.token)
            elif Id in self.command_id_map:
                rtn_strs.append(self._decode(current_str))
                current_str = []
                rtn_strs.append(self.command_id_map[Id].token)
            else:
                current_str.append(Id)
        if current_str:
            rtn_strs.append(self._decode(current_str))
        if self.spaces_between_special_tokens:
            output = ' '.join(rtn_strs)
        else:
            output = "".join(rtn_strs)
        output = self.clean_up_tokenization(output)
        return output

    def DecodeTokens(self, Tokens):
        """
        convert tokens to a string accounting for command and type tokens.
        """
        Ids = [self.TokenToId(token) for token in Tokens]
        return self.DecodeIds(Ids)


class GPT2BPETokenizer(Tokenizer):
    def __init__(self, model_type_or_path, cache_dir=None, add_block_symbols=False, add_task_mask=False,
                 add_decoder_mask=False, **kwargs):
        text_tokenizer = GPT2Tokenizer.from_pretrained(model_type_or_path,
                                                       cache_dir=cache_dir)

        # disable max len warnings by increasing max len
        text_tokenizer.max_len = int(1e12)
        num_tokens = len(text_tokenizer.encoder)
        if model_type_or_path.startswith('roberta'):
            command_tokens = [
                CommandToken('pad', '<|endoftext|>', text_tokenizer.encoder['</s>']),
                CommandToken('eos', '<|endoftext|>', text_tokenizer.encoder['</s>']),
                CommandToken('sep', '[SEP]', text_tokenizer.encoder['<pad>']),
                CommandToken('ENC', '[CLS]', text_tokenizer.encoder['<s>']),
                CommandToken('MASK', '[MASK]', text_tokenizer.encoder['<mask>'], lstrip=True),
                CommandToken('unk', '[UNK]', text_tokenizer.encoder['<unk>'])
            ]
            if add_block_symbols:
                command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>', num_tokens),
                    CommandToken('eop', '<|endofpiece|>', num_tokens + 1)
                ])
                num_tokens += 2
        else:
            command_tokens = [
                CommandToken('pad', '<|endoftext|>', text_tokenizer.encoder['<|endoftext|>']),
                CommandToken('eos', '<|endoftext|>', text_tokenizer.encoder['<|endoftext|>'])
            ]
            if add_block_symbols:
                command_tokens.extend([
                    CommandToken('sop', '<|startofpiece|>', num_tokens),
                    CommandToken('eop', '<|endofpiece|>', num_tokens + 1),
                    CommandToken('ENC', '[CLS]', num_tokens + 2),
                    CommandToken('MASK', '[MASK]', num_tokens + 3, lstrip=True),
                    CommandToken('sep', '[SEP]', num_tokens + 4),
                    CommandToken('unk', '[UNK]', num_tokens + 5)
                ])
                num_tokens += 6
        if add_block_symbols:
            if add_task_mask:
                command_tokens.extend([
                    CommandToken('gMASK', '[gMASK]', num_tokens, lstrip=True),
                    CommandToken('sMASK', '[sMASK]', num_tokens + 1, lstrip=True)
                ])
                num_tokens += 2
            if add_decoder_mask:
                command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', num_tokens)
                ])
                num_tokens += 1
        super().__init__(text_tokenizer, command_tokens=command_tokens)

    def _encode(self, text):
        return self.text_tokenizer.encode(text)

    def _decode(self, ids):
        return self.text_tokenizer.decode(ids)


class ChineseSPTokenizer(Tokenizer):
    def __init__(self, model_type_or_path, add_block_symbols=False, add_task_mask=False, add_decoder_mask=False,
                 **kwargs):
        text_tokenizer = SentencePieceTokenizer.from_pretrained(model_type_or_path)
        num_tokens = len(text_tokenizer.tokens)

        command_tokens = [
            CommandToken('pad', '<|endoftext|>', num_tokens),
            CommandToken('eos', '<|endoftext|>', num_tokens),
            CommandToken('sep', '[SEP]', num_tokens + 1),
            CommandToken('ENC', '[CLS]', num_tokens + 2),
            CommandToken('MASK', '[MASK]', num_tokens + 3, lstrip=True),
            CommandToken('unk', '[UNK]', num_tokens + 4)
        ]
        num_tokens += 5
        if add_block_symbols:
            command_tokens.extend([
                CommandToken('sop', '<|startofpiece|>', num_tokens + 1),
                CommandToken('eop', '<|endofpiece|>', num_tokens + 2)
            ])
            if model_type_or_path != 'glm-10b':
                num_tokens += 3
            else:
                num_tokens += 2
            if add_task_mask:
                if model_type_or_path != 'glm-10b':
                    command_tokens.extend([
                        CommandToken('sMASK', '[sMASK]', num_tokens, lstrip=True),
                        CommandToken('gMASK', '[gMASK]', num_tokens + 1, lstrip=True)
                    ])
                else:
                    command_tokens.extend([
                        CommandToken('gMASK', '[gMASK]', num_tokens, lstrip=True),
                        CommandToken('sMASK', '[sMASK]', num_tokens + 1, lstrip=True)
                    ])
                num_tokens += 2
            if add_decoder_mask:
                command_tokens.extend([
                    CommandToken('dBLOCK', '[dBLOCK]', num_tokens)
                ])
                num_tokens += 1
        super().__init__(text_tokenizer, command_tokens=command_tokens)
        if model_type_or_path in ['glm-large', 'glm-10b']:
            self.spaces_between_special_tokens = False

    def _encode(self, text):
        ids = self.text_tokenizer.encode(text)
        return ids

    def _decode(self, ids):
        text = self.text_tokenizer.decode(ids)
        return text