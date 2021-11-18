# -*- encoding: utf-8 -*-
'''
@File    :   cuda2d_model.py
@Time    :   2021/10/02 01:36:32
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import torch.nn.functional as F


from .base_model import BaseModel
from .mixins import PositionEmbeddingMixin, AttentionMixin

from SwissArmyTransformer.mpu.transformer import split_tensor_along_last_dim
from SwissArmyTransformer.mpu.utils import sqrt
from deepspeed.runtime.activation_checkpointing.checkpointing import get_cuda_rng_tracker


class Cuda2dModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        additional_seqlen = args.new_sequence_length - args.max_sequence_length
        self.add_mixin('extra_position_embedding', PositionEmbeddingMixin(
            additional_seqlen, args.hidden_size
        ))
        self.add_mixin('attention_plus', AttentionMixin(
            num_layers=args.num_layers,
            hidden_size=args.hidden_size
        ))
        self.layout = args.layout
        # [PAD]... [ROI1] text ... [BOI1] {layout[0]} 1024 {layout[1]} [EOI1] 4095 {layout[2]}
        self.kernel_size = args.kernel_size
        self.kernel_size2 = args.kernel_size2
        self.log_attention_weights = None
    
    def position_embedding_forward(self, position_ids, **kw_args):
        position = position_ids[..., :self.layout[1]]
        position_plus = position_ids[..., self.layout[1]:]
        position_embeddings = torch.cat(
                (
                    self.transformer.position_embeddings(position),
                    self.get_mixin('extra_position_embedding').position_embeddings(position_plus)
                ),
                dim=-2
            )
        return position_embeddings
        
    def attention_forward(self, hidden_states, mask, 
                        layer_id=None, log_attention_weights=None, **kw_args):
        attn_module = self.transformer.layers[layer_id].attention
        # attention_plus on all layers
        query_key_value_plus = self.get_mixin('attention_plus').query_key_value[layer_id] 
        dense_plus = self.get_mixin('attention_plus').dense[layer_id]
        
        # split two parts
        hidden_states_plus = hidden_states[:, self.layout[1]:]
        hidden_states = hidden_states[:, :self.layout[1]]
        # base model qkv
        mixed_raw_layer = attn_module.query_key_value(hidden_states)
        q0, k0, v0 = split_tensor_along_last_dim(mixed_raw_layer, 3)
        # cuda2d model qkv
        mixed_raw_layer = query_key_value_plus(hidden_states_plus)
        q1, k1, v1 = split_tensor_along_last_dim(mixed_raw_layer, 3)
        
        dropout_fn = attn_module.attention_dropout if self.training else None

        # cuda2d attention
        context_layer0, context_layer1 = sparse_attention_2d_light(
                q0, k0, v0,
                q1, k1, v1,
                mask,
                n_head=attn_module.num_attention_heads_per_partition,
                text_len=self.layout[0],
                kernel_size=self.kernel_size,
                kernel_size2=self.kernel_size2,
                attention_dropout=dropout_fn,
                log_attention_weights=log_attention_weights
            )

        output_0 = attn_module.dense(context_layer0)
        output_1 = dense_plus(context_layer1)
        output = torch.cat((output_0, output_1), dim=1)
        
        return output, None
    
    def disable_untrainable_params(self):
        self.transformer.requires_grad_(False)
    
    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('Cuda2dModel', 'cuda2d model configurations')
        group.add_argument("--kernel-size", type=int, default=9)
        group.add_argument("--kernel-size2", type=int, default=7)
        group.add_argument("--layout", type=str, default='64,1088,5184')
        group.add_argument("--new-sequence-length", type=int, default=5185)
        return parser

def sparse_attention_2d_light(q0, k0, v0, q1, k1, v1, attention_mask, n_head, text_len, kernel_size=9, kernel_size2=7, attention_dropout=None, log_attention_weights = None, **kwargs):
    '''
    q0, k0, v0: [batch_size, 1088, hidden_size]
    q1, k1, v1: [batch_size, 4096, h2]
    n_head: int
    attention_mask: [batch_size, 1088, 1088]
    '''
    from SwissArmyTransformer.mpu.local_attention_function import f_similar, f_weighting

    b, s0, h0 = q0.shape
    b, s1, h1 = q1.shape
    h, l0, l1 = h0 // n_head, sqrt(s0-text_len), sqrt(s1)

    q0 = q0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    v0 = v0.reshape(b, s0, n_head, h).permute(0, 2, 1, 3)
    k0T = k0.reshape(b, s0, n_head, h).permute(0, 2, 3, 1)
    
    # standard attention for level 0
    attention_scores = torch.matmul(q0 / math.sqrt(q0.shape[-1]), k0T)
    
    if log_attention_weights is not None:
        attention_scores += log_attention_weights

    attention_scores = torch.mul(attention_scores, attention_mask) - \
                    10000.0 * (1.0 - attention_mask)
    
    attention_probs0 = F.softmax(attention_scores, dim=-1)
    
    # local attention for level 1
    q1 = (q1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1) / math.sqrt(h1//n_head)).contiguous().view(b*n_head, h1//n_head, l1, l1)
    k1 = k1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    v1 = v1.view(b, s1, n_head, h1 // n_head).permute(0, 2, 3, 1).contiguous().view(b*n_head, h1//n_head, l1, l1)
    scores_1_to_1 = f_similar(q1, k1, kernel_size*2-1, kernel_size, True)    

    # cross attention
    k0T = k0T[..., -l0**2:].reshape(b*n_head, h, l0, l0).contiguous()
    scores_1_to_0 = f_similar(q1, k0T, kernel_size2, kernel_size2, False) # [b*n_head, l1, l1, field]
    scores_1 = torch.cat(
        (
            scores_1_to_0.view(b*n_head, -1, scores_1_to_0.shape[3]),
            scores_1_to_1.view(b*n_head, -1, scores_1_to_1.shape[3])
        ),
        dim=-1)
    attention_probs1 = F.softmax(scores_1, dim=-1)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs0 = attention_dropout(attention_probs0)
            attention_probs1 = attention_dropout(attention_probs1)
        
    # weighting for level 0
    context0 = torch.matmul(attention_probs0, v0) # [b, n_head, s0, h]
    # weighting for level 1
    probs_1_to_1 = attention_probs1[:, :, -scores_1_to_1.shape[3]:].view_as(scores_1_to_1)
    context1_to_1 = f_weighting(v1, probs_1_to_1.contiguous(), kernel_size*2-1, kernel_size, True)
    context1 = context1_to_1.view(b, n_head * h, l1**2)
    # weighting for cross attention
    probs_1_to_0 = attention_probs1[:, :, :scores_1_to_0.shape[3]].view_as(scores_1_to_0)
    v0_part = v0[:, :, -l0**2:].transpose(-1, -2).contiguous().view(b*n_head, h, l0, l0)
    context1_to_0 = f_weighting(v0_part, probs_1_to_0.contiguous(), kernel_size2, kernel_size2, False)
    context1_to_0 = context1_to_0.view(b, n_head * h, l1**2)
    context1 = context1 + context1_to_0
    return context0.transpose(1, 2).reshape(b, s0, h0), context1.transpose(-1, -2)