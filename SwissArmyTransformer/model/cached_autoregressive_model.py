# -*- encoding: utf-8 -*-
'''
@File    :   cached_autoregressive_model.py
@Time    :   2021/10/02 01:36:24
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

from .base_model import BaseModel, BaseMixin
from SwissArmyTransformer.mpu.transformer import standard_attention, split_tensor_along_last_dim

class CachedAutoregressiveMixin(BaseMixin):
    def __init__(self):
        super().__init__()
        
    def attention_forward(self, hidden_states, mask, mems=None, layer_id=None, log_attention_weights=None, **kwargs):
        attn_module = self.transformer.layers[layer_id].attention
        mem = mems[layer_id] if mems is not None else None
        
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        (mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)
        
        if mem is not None: # the first time, mem is None
            b = mixed_key_layer.shape[0] # might change batch_size
            memk, memv = split_tensor_along_last_dim(mem.expand(b, -1, -1), 2)
            mixed_key_layer = torch.cat((memk, mixed_key_layer), dim=1)
            mixed_value_layer = torch.cat((memv, mixed_value_layer), dim=1)

        # same as training
        query_layer = attn_module._transpose_for_scores(mixed_query_layer)
        key_layer = attn_module._transpose_for_scores(mixed_key_layer)
        value_layer = attn_module._transpose_for_scores(mixed_value_layer)
        context_layer = standard_attention(query_layer, key_layer, value_layer, mask, None, log_attention_weights=log_attention_weights)
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (attn_module.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        output = attn_module.dense(context_layer)
        
        # new mem this layer
        new_mem = mixed_raw_layer.detach()[..., -(mixed_raw_layer.shape[-1] // 3 * 2):].contiguous()
            
        return output, new_mem

class CachedAutoregressiveModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.add_mixin('auto-regressive', CachedAutoregressiveMixin())
