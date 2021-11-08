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

"""GPT-2 model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from SwissArmyTransformer import mpu
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.model.mixins import BaseMixin
from SwissArmyTransformer.mpu.transformer import standard_attention, split_tensor_along_last_dim
from utils import print_rank_0


class CachedAutoregressiveMixin(BaseMixin):
    def __init__(self):
        super().__init__()

    def attention_forward(self, hidden_states, mask, prefixs=None, layer_id=None, log_attention_weights=None, **kwargs):
        attn_module = self.transformer.layers[layer_id].attention
        prefix = prefixs[layer_id] if prefixs is not None else None
        query_length = hidden_states.size(1)
        if prefix is None:  # the first time, mem is None
            mixed_raw_layer = attn_module.query_key_value(hidden_states)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
        else:
            cat = torch.cat((prefix, hidden_states), 1)
            mixed_raw_layer = attn_module.query_key_value(cat)
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
            mixed_query_layer = mixed_query_layer[:, -query_length:]
        # same as training
        query_layer = attn_module._transpose_for_scores(mixed_query_layer)
        key_layer = attn_module._transpose_for_scores(mixed_key_layer)
        value_layer = attn_module._transpose_for_scores(mixed_value_layer)
        context_layer = standard_attention(query_layer, key_layer, value_layer, mask, None,
                                           log_attention_weights=log_attention_weights)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = attn_module.dense(context_layer)

        # new mem this layer
        new_mem = mixed_raw_layer.detach()[..., -(mixed_raw_layer.shape[-1] // 3 * 2):].contiguous()

        return output, new_mem


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, layers*hidden)
    """

    def __init__(self, prefix_length, num_layers, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(prefix_length, num_layers * hidden_size)

    def forward(self, prefix: torch.Tensor):
        past_key_values = self.embedding(prefix)
        return past_key_values


class GLMFPrefixModel(GLMModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super(GLMFPrefixModel, self).__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('auto-regressive', CachedAutoregressiveMixin())
        print_rank_0(f"Create prefix prompt of length {args.prefix_prompt}")
        self.prefix_length = args.prefix_prompt
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.prefix_encoder = PrefixEncoder(prefix_length=args.prefix_prompt, num_layers=args.num_layers,
                                            hidden_size=args.hidden_size)

    def get_prompt(self, batch_size):
        prefix_hidden_states = self.prefix_encoder.embedding.weight.half()
        prefix_hidden_states = prefix_hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
        # bsz, seqlen, _ = past_key_values.shape
        prefix_hidden_states = prefix_hidden_states.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            self.hidden_size
        )
        prefix_hidden_states = prefix_hidden_states.permute([2, 0, 1, 3]).split(1)
        prefix_hidden_states = [value[0] for value in prefix_hidden_states]
        return prefix_hidden_states

    def forward(self, input_ids, position_ids, attention_mask, *, branch_input=None, **kw_args):
        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        prompt_hidden_states = self.get_prompt(batch_size=batch_size)
        attention_mask = torch.cat(
            (attention_mask.new_ones(batch_size, 1, seq_length, self.prefix_length), attention_mask), dim=-1)
        output, *mems = super().forward(input_ids, position_ids, attention_mask, prefixs=prompt_hidden_states)
        return (output, *mems)
