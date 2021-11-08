# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2021/10/06 17:58:04
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

from SwissArmyTransformer.training.utils import print_rank_0


def _export_vocab_size_to_args(args, original_num_tokens):
    tokenizer = get_tokenizer(args)
    num_tokens = original_num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by
    # you should control args to let it divided by 
    # mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))
    args.vocab_size = after
    print_rank_0("prepare tokenizer done")
    return tokenizer


def get_tokenizer(args=None, outer_tokenizer=None):
    '''
        If you're using outer_tokenizer, call `get_tokenizer(args, outer_tokenizer)`
        before `training_main`.
    '''
    if outer_tokenizer is not None:
        assert hasattr(outer_tokenizer, 'num_tokens')
        assert not hasattr(get_tokenizer, 'tokenizer')
        get_tokenizer.tokenizer = outer_tokenizer
        _export_vocab_size_to_args(args, get_tokenizer.tokenizer.num_tokens)
        return outer_tokenizer
    if not hasattr(get_tokenizer, 'tokenizer'):
        # the first time to load the tokenizer
        if args.tokenizer_type == 'cogview':
            from .cogview import UnifiedTokenizer
            get_tokenizer.tokenizer = UnifiedTokenizer(
                args.img_tokenizer_path,
                device=torch.cuda.current_device()
            )
        elif args.tokenizer_type.startswith('glm_'):
            kwargs = {"add_block_symbols": True, "add_task_mask": args.task_mask,
                      "add_decoder_mask": args.block_mask_prob > 0.0}
            if args.tokenizer_type == "glm_GPT2BPETokenizer":
                from .glm import GPT2BPETokenizer
                get_tokenizer.tokenizer = GPT2BPETokenizer(args.tokenizer_model_type, **kwargs)
            elif args.tokenizer_type == "glm_ChineseSPTokenizer":
                from .glm import ChineseSPTokenizer
                get_tokenizer.tokenizer = ChineseSPTokenizer(args.tokenizer_model_type, **kwargs)
        else:
            assert args.vocab_size > 0
            get_tokenizer.tokenizer = FakeTokenizer(args.vocab_size)
        _export_vocab_size_to_args(args, get_tokenizer.tokenizer.num_tokens)
    return get_tokenizer.tokenizer


class FakeTokenizer(object):
    def __init__(self, num_tokens):
        self.num_tokens = num_tokens

    def __len__(self):
        return self.num_tokens
