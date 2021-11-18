# -*- encoding: utf-8 -*-
'''
@File    :   autoregressive_sampling.py
@Time    :   2021/10/08 15:43:59
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
from .sampling_strategies import BaseStrategy

def get_masks_and_position_ids_default(seq):
    tokens = seq.unsqueeze(0)

    attention_mask = torch.ones((1, len(seq), len(seq)), device=tokens.device)
    attention_mask.tril_()
    attention_mask.unsqueeze_(1)

    position_ids = torch.arange(len(seq), dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0)
    return tokens, attention_mask, position_ids

def update_mems(hiddens, mems, max_memory_length):
    '''
        hiddens: list (num_layers) of [batch, query_length, 2d]
        mems: None or [num_layers, batch, memory_length, 2d]
    '''
    if hiddens is None:
        return None
    hiddens = torch.stack(hiddens)
    memory_length = mems.shape[2] if mems is not None else 0
    query_length = hiddens.shape[2]
    new_memory_length = min(max_memory_length, memory_length + query_length)
    with torch.no_grad():
        if new_memory_length <= query_length:
            return hiddens[:, :, -new_memory_length:]
        else:
            if mems.shape[1] < hiddens.shape[1]:
                mems = mems.expand(-1, hiddens.shape[1], -1, -1)
            return torch.cat(
                (mems[:, :, -new_memory_length+query_length:], hiddens),
                dim=2
            )


def filling_sequence(
        model, 
        seq, 
        batch_size,
        strategy=BaseStrategy(),
        max_memory_length=100000,
        log_attention_weights=None,
        get_masks_and_position_ids=get_masks_and_position_ids_default,
        mems=None
        ):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -1, ...]
        mems: [num_layers, batch_size, len_mems(index), mem_hidden_size]
            cache, should be first mems.shape[1] parts of context_tokens.
            mems are the first-level citizens here, but we don't assume what is memorized.
            input mems are used when multi-phase generation.
    '''
    assert len(seq.shape) == 1

    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1 # [0, context_length-1] are given
    assert context_length > 0
    tokens, attention_mask, position_ids = get_masks_and_position_ids(seq)
    tokens = tokens[..., :context_length]
    attention_mask = attention_mask.type_as(next(model.parameters())) # if fp16
    # initialize generation
    counter = context_length - 1 # Last fixed index is ``counter'' 
    index = 0 if mems is None else mems.shape[2] # Next forward starting index, also the length of cache.
    # step-by-step generation
    while counter < len(seq) - 1:
        # Now, we want to generate seq[counter + 1],
        # token[:, index: counter+1] needs forwarding.

        if seq[counter + 1] >= 0: # provided
            tokens = torch.cat(
                (
                tokens, 
                    seq[counter+1: counter+2].expand(tokens.shape[0], 1)
                ), dim=1
            )
            counter += 1
            continue

        # forward
        if log_attention_weights is not None:
            log_attention_weights_part = log_attention_weights[..., index: counter+1, :counter+1] # TODO memlen
        else:
            log_attention_weights_part = None

        logits, *mem_kv = model(
            tokens[:, index:], 
            position_ids[..., index: counter+1],
            attention_mask[..., index: counter+1, :counter+1], # TODO memlen
            mems=mems,
            log_attention_weights=log_attention_weights_part
        )
        mems = update_mems(mem_kv, mems, max_memory_length=max_memory_length)
        counter += 1
        index = counter
        # sampling
        logits = logits[:, -1].expand(batch_size, -1) # [batch size, vocab size]
        tokens = tokens.expand(batch_size, -1)
        tokens, mems = strategy.forward(logits, tokens, mems)
        if strategy.is_done:
            break
    return strategy.finalize(tokens, mems)



def evaluate_perplexity(model, tokens, attention_mask, position_ids, loss_mask, invalid_slices=[], reduction='mean'):
    # sanity check
    assert len(tokens.shape) <= 2 and len(loss_mask.shape)
    if len(tokens.shape) == 1:
        tokens = tokens.unsqueeze(0)
    if len(loss_mask.shape) == 1:
        loss_mask = loss_mask.unsqueeze(0).expand(tokens.shape)
    pad_pos = tokens < 0
    if pad_pos.any():
        print('Find -1 in tokens, automatically ignore them.')
        tokens[pad_pos] = 0
        loss_mask[pad_pos] = 0

    attention_mask = attention_mask.type_as(next(model.parameters()))
    logits = model(tokens, position_ids, attention_mask)[0]
    logits = logits.float()
    for slc in invalid_slices:
        logits[..., slc] = -float('Inf')
    log_probs = torch.log(torch.nn.functional.softmax(logits, dim=-1))

    pred = log_probs[:, :-1, :] 
    target = tokens[:, 1:].unsqueeze(-1) 
    loss_mask = loss_mask[..., 1:]
    scores = torch.gather(pred, dim=2, index=target).squeeze(-1) # [batch_size, seq_len-1]
    if reduction == 'mean':
        return (scores * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1)
    elif reduction == 'none':
        return (scores * loss_mask)
    else:
        raise ValueError('Unknown reduction type')