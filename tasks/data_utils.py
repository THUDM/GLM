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

import re
import numpy as np


def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    for _ in range(3):
        text = text.replace(' . ', '. ')

    return text


def build_sample(ids, positions, masks, label, unique_id):
    """Convert to numpy and return a sample consumed by the batch producer."""

    ids_np = np.array(ids, dtype=np.int64)
    positions_np = np.array(positions, dtype=np.int64)
    masks_np = np.array(masks, dtype=np.int64)
    sample = ({'text': ids_np,
               'position': positions_np,
               'mask': masks_np,
               'label': int(label),
               'uid': int(unique_id)})

    return sample


def build_tokens_types_paddings_from_text(text_a, text_b,
                                          tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    text_a_ids = tokenizer.tokenize(text_a)
    text_b_ids = None
    if text_b is not None:
        text_b_ids = tokenizer.tokenize(text_b)

    return build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids,
                                                max_seq_length, tokenizer.cls,
                                                tokenizer.sep, tokenizer.pad)


def build_tokens_types_paddings_from_ids(text_a_ids, max_seq_length, mask_id, start_id, pad_id, cls_id=None):
    """Build token types and paddings, trim if needed, and pad if needed."""

    ids = []

    # [CLS].
    if cls_id is not None:
        ids.append(cls_id)

    # A.
    ids.extend(text_a_ids)

    # Cap the size.
    if len(ids) > max_seq_length - 3:
        max_seq_length_m1 = max_seq_length - 3
        ids = ids[0:max_seq_length_m1]

    # Mask
    mask_position = len(ids)
    ids.append(mask_id)
    ids.append(pad_id)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)

    mask = len(ids)
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
    return ids, position_ids, mask
