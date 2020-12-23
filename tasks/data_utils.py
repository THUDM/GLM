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

""" Tasks data utility."""
import copy
import json
import pickle

import re
from typing import Dict, List, Optional

import numpy as np


def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    for _ in range(3):
        text = text.replace(' . ', '. ')

    return text


def build_sample(ids, types=None, paddings=None, positions=None, masks=None, label=None, unique_id=None, target=None,
                 logit_mask=None):
    """Convert to numpy and return a sample consumed by the batch producer."""

    ids_np = np.array(ids, dtype=np.int64)
    sample = {'text': ids_np, 'label': int(label)}
    if types is not None:
        types_np = np.array(types, dtype=np.int64)
        sample['types'] = types_np
    if paddings is not None:
        paddings_np = np.array(paddings, dtype=np.int64)
        sample['padding_mask'] = paddings_np
    if positions is not None:
        positions_np = np.array(positions, dtype=np.int64)
        sample['position'] = positions_np
    if masks is not None:
        masks_np = np.array(masks, dtype=np.int64)
        sample['mask'] = masks_np
    if target is not None:
        target_np = np.array(target, dtype=np.int64)
        sample['target'] = target_np
    if logit_mask is not None:
        logit_mask_np = np.array(logit_mask, dtype=np.int64)
        sample['logit_mask'] = logit_mask_np
    if unique_id is not None:
        sample['uid'] = int(unique_id)
    return sample


def build_tokens_types_paddings_from_text(text_a, text_b,
                                          tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    text_a_ids = tokenizer.tokenize(text_a)
    text_b_ids = None
    if text_b is not None:
        text_b_ids = tokenizer.tokenize(text_b)

    return build_block_input_from_ids(text_a_ids, text_b_ids,
                                      max_seq_length, tokenizer.cls,
                                      tokenizer.sep, tokenizer.pad)


def build_block_input_from_ids(text_a_ids, max_seq_length, mask_id=None, start_id=None, pad_id=None, cls_id=None,
                               pool_token='start'):
    """Build token types and paddings, trim if needed, and pad if needed."""

    if pool_token not in ['start', 'cls', 'pad']:
        raise NotImplementedError(pool_token)
    ids = []

    if pool_token == 'cls':
        assert cls_id is not None
    # [CLS].
    if cls_id is not None:
        ids.append(cls_id)

    # A.
    ids.extend(text_a_ids)
    if pool_token == 'start':
        # Cap the size.
        if len(ids) > max_seq_length - 3:
            max_seq_length_m1 = max_seq_length - 3
            ids = ids[0:max_seq_length_m1]
        # Mask
        mask_position = len(ids)
        ids.append(mask_id)
    elif pool_token == 'pad' or pool_token == 'cls':
        if len(ids) > max_seq_length - 1:
            max_seq_length_m1 = max_seq_length - 1
            ids = ids[:max_seq_length_m1]
    ids.append(pad_id)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
    sep = len(ids)
    if pool_token == 'start':
        ids.append(start_id)
        position_ids.append(mask_position)
        block_position_ids.append(1)
    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([pad_id] * padding_length)
        position_ids.extend([position_ids[-1]] * padding_length)
        block_position_ids.extend(range(2, padding_length + 2))

    position_ids = [position_ids, block_position_ids]
    return ids, position_ids, sep


def build_input_from_ids(text_a_ids, text_b_ids, answer_ids, max_seq_length, tokenizer, add_cls=True, add_sep=False,
                         add_piece=False):
    mask_id = tokenizer.get_command('MASK').Id
    eos_id = tokenizer.get_command('eos').Id
    sop_id = tokenizer.get_command('sop').Id
    cls_id = tokenizer.get_command('ENC').Id
    sep_id = tokenizer.get_command('sep').Id
    ids = []
    types = []
    paddings = []
    # CLS
    if add_cls:
        ids.append(cls_id)
        types.append(0)
        paddings.append(1)
    # A
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)
    # SEP
    if add_sep:
        ids.append(sep_id)
        types.append(0)
        paddings.append(1)
    # B
    if text_b_ids is not None:
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)
    # Cap the size.
    if len(ids) >= max_seq_length - 1:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
    ids.append(eos_id)
    end_type = 0 if text_b_ids is None else 1
    types.append(end_type)
    paddings.append(1)
    sep = len(ids)
    target_ids = [0] * len(ids)
    loss_masks = [0] * len(ids)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
    # Piece
    if add_piece or answer_ids is not None:
        mask_position = ids.index(mask_id)
        ids.append(sop_id)
        types.append(end_type)
        paddings.append(1)
        position_ids.append(mask_position)
        block_position_ids.append(1)
        if answer_ids is not None:
            len_answer = len(answer_ids)
            ids.extend(answer_ids[:-1])
            types.extend([end_type] * (len_answer - 1))
            paddings.extend([1] * (len_answer - 1))
            position_ids.extend([mask_position] * (len_answer - 1))
            block_position_ids.extend(range(2, len(answer_ids) + 1))
            target_ids.extend(answer_ids)
            loss_masks.extend([1] * len(answer_ids))
        else:
            target_ids.append(0)
            loss_masks.append(0)
    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([eos_id] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    position_ids = [position_ids, block_position_ids]
    return ids, types, paddings, position_ids, sep, target_ids, loss_masks


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta: Optional[Dict] = None, idx=-1):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)
