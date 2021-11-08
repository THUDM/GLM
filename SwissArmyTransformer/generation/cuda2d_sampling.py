# -*- encoding: utf-8 -*-
'''
@File    :   cuda2d_sampling.py
@Time    :   2021/10/09 00:46:04
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from .sampling_strategies import IterativeEntfilterStrategy

def filling_sequence_cuda2d(
        model, 
        seq0,
        seq1, 
        warmup_steps=3,
        block_hw=(4, 4),
        strategy=IterativeEntfilterStrategy(topk=10)
        ):
    '''
        seq: [PAD]... [ROI1] text ... [BOI1] {layout[0]} 1024 {layout[1]} [EOI1]
            4095 {layout[2]} final_token.

        Attention:
        The sampling temperature are changing, temporally we hard code them here.
        The temperature in the strategy is not used.
    '''
    assert hasattr(model, 'layout')
    layout = model.layout
    assert len(seq0.shape) == 2 and len(seq1.shape) == 2 \
        and seq0.shape[0] == seq1.shape[0]
    assert len(layout) == 3
    assert seq1.shape[1] == layout[-1] - layout[-2]
    assert (seq1 >= 0).all() and (seq0 >= 0).all()
    device = seq0.device
    # concat and pad sequences
    batch_size = seq0.shape[0]
    n_pad = layout[1] + 1 - seq0.shape[1] # +1 for [EOI1]
    assert n_pad > 0, "You should truncate long input before filling."
    seq = torch.cat((
        torch.tensor([0]*n_pad, device=device, dtype=seq0.dtype)
            .unsqueeze(0).expand(batch_size, n_pad),
        seq0, seq1), dim=1) # [b, layout[-1]+1]
    assert seq.shape[1] == layout[-1] + 1

    # build initial tokens, attention_mask, and position_ids
    tokens = seq.clone()
    attention_mask = torch.ones(layout[1], layout[1]).tril().to(device)
    attention_mask[n_pad:, :n_pad] = 0
    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    position_ids = torch.cat((
        torch.zeros(n_pad, dtype=torch.long),
        torch.arange(0, layout[1] - n_pad), 
        torch.arange(0, layout[2]-layout[1]))).to(device)

    # prepare for interation
    unfixed = (tokens < 0)
    unfixed[:, -layout[-1] + layout[-2]:] = True
    ll, rr = block_hw
    edge_len = int(math.sqrt(layout[-1] - layout[-2]) + 1e-4)
    num_steps = warmup_steps + ll + rr - 2
    # interative refining
    for step_cnt in range(1, num_steps+1):
        logits, *_dump = model(tokens[:,:-1], position_ids, attention_mask)
        if step_cnt <= warmup_steps:
            real_temp = 0.1
            new_tokens = strategy.forward(logits, tokens, real_temp)
            tokens[unfixed] = new_tokens[unfixed]
        else:
            real_temp = 1.05
            new_tokens = strategy.forward(
                logits, tokens, real_temp,
                entfilter=1.3,
                filter_topk=5,
                temperature2=0.6
            )
            tokens[unfixed] = new_tokens[unfixed]
            # fixed tokens (update unfixed)
            for x in range(min(ll, step_cnt - warmup_steps)):
                y = step_cnt - warmup_steps - x - 1
                if y < rr:
                    unfixed[..., -(layout[-1] - layout[-2]):].view(
                        batch_size, edge_len//ll, ll, edge_len//rr, rr)[:, :, x, :, y] = False

    return tokens[:, n_pad:]