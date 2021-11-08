# -*- encoding: utf-8 -*-
'''
@File    :   iterative_entfilter_strategy.py
@Time    :   2021/10/09 14:32:29
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

def top_k_logits_(logits, top_k=0, filter_value=-float('Inf')):
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value     
    return logits

class IterativeEntfilterStrategy:
    def __init__(self, invalid_slices=[], temperature=1., topk=10):
        self.invalid_slices = invalid_slices
        self.temperature = temperature
        self.topk = topk

    def forward(self, logits, tokens, temperature=None, entfilter=None, filter_topk=5, temperature2=None):
        # In interative strategy, logits are of shape [batch_size, seq_length, hidden_size]
        if temperature is None:
            temperature = self.temperature 
        # check entropy filter
        if entfilter is not None:
            assert temperature2 is not None
            topraw = (torch.topk(logits, filter_topk, dim=-1)[0]).softmax(dim=-1)
            ent = -(topraw * topraw.log()).sum(dim=-1) # [batch_size, seq_length]
            temperature = torch.tensor([[[temperature - temperature2]]], device=logits.device).expand(*logits.shape[:2], 1) * (ent > entfilter).unsqueeze(-1) + temperature2
        logits = logits.float() / temperature
        for invalid_slice in self.invalid_slices:
            logits[..., invalid_slice] = -float('Inf')
        
        # debiased topk
        probs = F.softmax(logits, dim=-1)
        tk_value, tk_idx = torch.topk(probs, self.topk, dim=-1)
        pred = torch.multinomial(probs.view(-1, logits.shape[-1]), num_samples=1).view(*logits.shape[:2], 1)
        edge_idx = tk_idx[:, :, -1:]
        edge_value = tk_value[:, :, -1:]
        edge_mask = probs.gather(dim=-1, index=pred) < edge_value
        pred[edge_mask] = edge_idx[edge_mask] # replace outliers as the "filter_topk"-th token
        pred.squeeze_(-1) # [batch_size, seq_length]
        
        assert tokens.shape[1] == pred.shape[1] + 1
        tokens = torch.cat((tokens[:, :1], pred), dim=1)
        return tokens