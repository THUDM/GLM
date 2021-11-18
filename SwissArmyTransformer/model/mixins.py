# -*- encoding: utf-8 -*-
'''
@File    :   mixins.py
@Time    :   2021/10/01 17:52:40
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import torch
from SwissArmyTransformer.mpu import ColumnParallelLinear, RowParallelLinear
from SwissArmyTransformer.mpu.transformer import unscaled_init_method
from .base_model import BaseMixin
from .cached_autoregressive_model import CachedAutoregressiveMixin

class PositionEmbeddingMixin(BaseMixin):
    def __init__(self, additional_sequence_length, hidden_size, 
                init_method_std=0.02, reinit_slice=slice(-1024, None)
        ):
        super(PositionEmbeddingMixin, self).__init__()
        self.reinit_slice = reinit_slice
        self.position_embeddings = torch.nn.Embedding(additional_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)
    def reinit(self, *pre_mixins):
        old_weights = self.transformer.position_embeddings.weight.data[self.reinit_slice]
        old_len, hidden_size = old_weights.shape
        assert hidden_size == self.position_embeddings.weight.shape[-1]
        self.position_embeddings.weight.data.view(-1, old_len, hidden_size).copy_(old_weights)

class AttentionMixin(BaseMixin):
    def __init__(self, num_layers,
                hidden_size, 
                init_method=unscaled_init_method(0.02),
                output_layer_init_method=unscaled_init_method(0.02)
        ):
        super(AttentionMixin, self).__init__()
        self.num_layers = num_layers # replace attention in the LAST n layers
        self.query_key_value = torch.nn.ModuleList(
            [ColumnParallelLinear(hidden_size, 3*hidden_size,stride=3,
                gather_output=False,init_method=init_method)
                for layer_id in range(num_layers)
            ])
        self.dense = torch.nn.ModuleList(
            [RowParallelLinear(hidden_size,
                hidden_size,
                input_is_parallel=True,
                init_method=output_layer_init_method)
                for layer_id in range(num_layers)
            ])
    def reinit(self, *pre_mixins):
        start_layer = len(self.transformer.layers) - self.num_layers
        assert start_layer >= 0
        for layer_id in range(self.num_layers):
            old_attention = self.transformer.layers[start_layer + layer_id].attention
            self.query_key_value[layer_id].weight.data.copy_(old_attention.query_key_value.weight.data)
            self.query_key_value[layer_id].bias.data.copy_(old_attention.query_key_value.bias.data)
            self.dense[layer_id].weight.data.copy_(old_attention.dense.weight.data)
            self.dense[layer_id].bias.data.copy_(old_attention.dense.bias.data)
