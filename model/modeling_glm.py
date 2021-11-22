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
from SwissArmyTransformer.model.cached_autoregressive_model import CachedAutoregressiveMixin
from utils import print_rank_0


class ExtendBlockPositionEmbeddingMixin(BaseMixin):
    def __init__(self, max_sequence_length, hidden_size, additional_sequence_length, init_method_std=0.02):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        self.additional_position_embeddings = torch.nn.Embedding(additional_sequence_length, hidden_size)
        self.additional_block_position_embeddings = torch.nn.Embedding(additional_sequence_length, hidden_size)
        torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
        torch.nn.init.normal_(self.additional_position_embeddings.weight, mean=0.0, std=init_method_std)
        torch.nn.init.normal_(self.additional_block_position_embeddings.weight, mean=0.0, std=init_method_std)

    def position_embedding_forward(self, position_ids, **kwargs):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = torch.cat(
            (self.transformer.position_embeddings.weight, self.additional_position_embeddings.weight), dim=0)
        position_embeddings = F.embedding(position_ids, position_embeddings)
        block_position_embeddings = torch.cat(
            (self.block_position_embeddings.weight, self.additional_block_position_embeddings.weight), dim=0)
        block_position_embeddings = F.embedding(block_position_ids, block_position_embeddings)
        return position_embeddings + block_position_embeddings


class GLMCustomModel(GLMModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        if hasattr(args, "additional_sequence_length"):
            self.del_mixin('block_position_embedding')
            self.add_mixin('block_position_embedding',
                           ExtendBlockPositionEmbeddingMixin(args.max_sequence_length, args.hidden_size,
                                                             args.additional_sequence_length))
            print_rank_0(f"Extend additional sequence length {args.additional_sequence_length}")


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, layers*hidden)
    """

    def __init__(self, prefix_length, num_layers, hidden_size, prompt_func='none'):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(prefix_length, num_layers * hidden_size)
        self.prompt_func = prompt_func
        if prompt_func == 'lstm':
            self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                           hidden_size=self.hidden_size,
                                           num_layers=2,
                                           bidirectional=True,
                                           batch_first=False)
            self.mlp_head = torch.nn.Sequential(torch.nn.Linear(2 * self.hidden_size, self.hidden_size),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.hidden_size, self.hidden_size))
        elif prompt_func == 'mlp':
            self.mlp_head = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, self.hidden_size),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(self.hidden_size, self.hidden_size))
        elif prompt_func != 'none':
            raise NotImplementedError(prompt_func)

    def forward(self, batch_size, layer_id):
        prefix_hidden_states = self.embedding.weight
        prefix_hidden_states = prefix_hidden_states.view(self.prefix_length,
                                                         self.num_layers,
                                                         self.hidden_size)
        prefix_hidden_states = prefix_hidden_states[:, layer_id: layer_id + 1]
        if self.prompt_func == 'lstm':
            prefix_hidden_states = self.lstm_head(prefix_hidden_states)[0]
            prefix_hidden_states = self.mlp_head(prefix_hidden_states)
        elif self.prompt_func == 'mlp':
            prefix_hidden_states = self.mlp_head(prefix_hidden_states)
        else:
            pass
        prefix_hidden_states = prefix_hidden_states.squeeze(1)
        prefix_hidden_states = prefix_hidden_states.unsqueeze(0).expand(batch_size, -1, -1)
        return prefix_hidden_states


class GLMFPrefixModel(GLMCustomModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        self.autoregressive_hooks = set()
        super(GLMFPrefixModel, self).__init__(args, transformer=transformer, parallel_output=parallel_output)
        print_rank_0(f"Create prefix prompt of length {args.prefix_prompt}")
        self.prefix_length = args.prefix_prompt
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.freeze_transformer = args.freeze_transformer
        self.prefix_encoder = PrefixEncoder(prefix_length=args.prefix_prompt, num_layers=args.num_layers,
                                            hidden_size=args.hidden_size, prompt_func=args.prompt_func)

    def disable_untrainable_params(self):
        if self.freeze_transformer:
            print_rank_0("Freeze transformer model")
            self.transformer.requires_grad_(False)
            self.mixins['block_position_embedding'].block_position_embeddings.requires_grad_(False)

    def add_mixin(self, name, new_mixin, reinit=False):
        if hasattr(new_mixin, "attention_forward"):
            if isinstance(new_mixin, CachedAutoregressiveMixin):
                self.autoregressive_hooks.add(name)
            else:
                raise ValueError(f'Hook {name} conflicts at attention forward with prefix model.')
        else:
            super().add_mixin(name, new_mixin, reinit=reinit)

    def del_mixin(self, name):
        if name in self.autoregressive_hooks:
            self.autoregressive_hooks.remove(name)
        else:
            super().del_mixin(name)

    def attention_forward(self, hidden_states, mask, layer_id=None, log_attention_weights=None, mems=None, **kwargs):
        attn_module = self.transformer.layers[layer_id].attention
        prefix = self.prefix_encoder(hidden_states.size(0), layer_id)
        mem = mems[layer_id] if mems is not None and self.autoregressive_hooks else None
        batch_size, query_length = hidden_states.size(0), hidden_states.size(1)
        cat = torch.cat((prefix, hidden_states), 1)
        mixed_raw_layer = attn_module.query_key_value(cat)
        (mixed_query_layer,
         mixed_key_layer,
         mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
        mixed_query_layer = mixed_query_layer[:, -query_length:]
        if mem is not None:
            memk, memv = split_tensor_along_last_dim(mem.expand(batch_size, -1, -1), 2)
            mixed_key_layer = torch.cat((memk, mixed_key_layer), dim=1)
            mixed_value_layer = torch.cat((memv, mixed_value_layer), dim=1)
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
        if self.autoregressive_hooks:
            new_mem = mixed_raw_layer.detach()[:, -query_length:, -(mixed_raw_layer.shape[-1] // 3 * 2):].contiguous()
            return output, new_mem
        else:
            return output, None

    def forward(self, input_ids, position_ids, attention_mask, *args, **kw_args):
        batch_size, seq_length = attention_mask.shape[0], input_ids.shape[1]
        memory_mask, input_mask = attention_mask[:, :, :, :-seq_length], attention_mask[:, :, :, -seq_length:]
        attention_mask = torch.cat(
            (memory_mask, attention_mask.new_ones(batch_size, 1, seq_length, self.prefix_length), input_mask), dim=-1)
        output, *mems = super().forward(input_ids, position_ids, attention_mask, *args, **kw_args)
        return (output, *mems)
