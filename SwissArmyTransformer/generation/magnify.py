# -*- encoding: utf-8 -*-
'''
@File    :   magnify.py
@Time    :   2021/01/14 00:41:40
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from .sampling import filling_sequence


def magnify(model, tokenizer, tokens_list, text_token_list, args):
        # 32 * 32 to 4 16 * 16
        s = int(math.sqrt(len(tokens_list)+ 1e-6))
        assert s == 32
        code = tokens_list.view(s, s)

        midfix = torch.tensor([tokenizer['[EOI1]'], tokenizer['[ROI2]'], tokenizer['[POS0]'], tokenizer['[BASE]'], tokenizer['[BOI2]']], device=code.device)

        magnified_code = code.new_zeros((s * 2, s * 2), dtype=torch.long) - 1

        windows = [(0, 0, 18), (0, 1, 30), (0, 2, 30), (1, 1, 30), (1, 0, 30), (1, 2, 30), (2, 0, 32), (2, 1, 32), (2, 2, 32)]
        for i, j, line in windows:
                code_part = code[8 * i: 8 * (i+2), 8 * j: 8 * (j+2)].reshape(-1)

                magnified_code_part = magnified_code[16 * i: 16 * i + line, 16 * j: 16 * (j+2)].reshape(-1)
                context_tokens_tensor = torch.cat([text_token_list, code_part, midfix], dim=0)
                context_len = len(context_tokens_tensor)
                seq = torch.cat([context_tokens_tensor, magnified_code_part], dim=0)

                magnified_code_part_completed = filling_sequence(model, seq, args, invalid_slices=[slice(tokenizer.img_tokenizer.num_tokens, None)])
                magnified_code[16 * i: 16 * i + line, 16 * j: 16 * (j+2)] = magnified_code_part_completed[0, context_len:].view(line, 32)
        return magnified_code.view(1, s * s * 4)
