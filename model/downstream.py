# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Multiple choice model."""

import torch
import torch.nn

from utils import print_rank_0
from .modeling import bert_extended_attention_mask
from .gpt2_modeling import init_method_normal


class MultipleChoice(torch.nn.Module):

    def __init__(self, language_model, hidden_size, hidden_dropout, pool_token, cloze_format):
        super(MultipleChoice, self).__init__()
        self.pool_token = pool_token
        self.cloze_foramt = cloze_format
        self.model = language_model

        # Multi-choice head.
        if not self.cloze_foramt:
            self.pool_layer = torch.nn.Linear(hidden_size, hidden_size)
            self.multichoice_dropout = torch.nn.Dropout(hidden_dropout)
            self.multichoice_head = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, position_ids, attention_mask, target_ids=None, logit_mask=None):
        # [batch, choices, sequence] --> [batch * choices, sequence] -->
        #    transformer --> [batch, choices] --> softmax

        # Ensure the shape is [batch-size, choices, sequence]
        assert len(input_ids.shape) == 3
        assert len(position_ids.shape) == 4

        # Reshape and treat choice dimension the same as batch.
        batch_size, num_choices = input_ids.shape[:2]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, *attention_mask.size()[2:])
        position_ids = position_ids.view(-1, *position_ids.size()[2:])

        outputs, *mems = self.model(input_ids, position_ids, attention_mask)
        # Output.
        if self.cloze_foramt:
            target_ids = target_ids.view(-1, target_ids.size(-1))
            logit_mask = logit_mask.view(-1, logit_mask.size(-1))
            seq_ids = torch.arange(target_ids.size(-1), dtype=torch.long, device=target_ids.device)
            seq_ids = seq_ids.unsqueeze(0).expand_as(target_ids)
            batch_ids = torch.arange(target_ids.size(0), dtype=torch.long, device=target_ids.device)
            batch_ids = batch_ids.unsqueeze(1).expand_as(target_ids)
            logits = outputs[batch_ids, seq_ids, target_ids]
            multichoice_logits = (logits * logit_mask).sum(dim=1)
            # multichoice_logits = multichoice_logits / logit_mask.sum(dim=1)
            multichoice_logits = multichoice_logits / logit_mask.sum(dim=1).max(dim=0, keepdim=True).values
        else:
            if self.pool_token == 'start':
                output = outputs[
                    torch.arange(batch_size * num_choices, dtype=attention_mask.dtype,
                                 device=attention_mask.device), attention_mask]
            elif self.pool_token == 'pad':
                output = outputs[
                    torch.arange(batch_size * num_choices, dtype=attention_mask.dtype,
                                 device=attention_mask.device), attention_mask - 1]
            elif self.pool_token == 'cls':
                output = outputs[:, 0]
            else:
                raise NotImplementedError
            output = torch.tanh(self.pool_layer(output))
            multichoice_output = self.multichoice_dropout(output)
            multichoice_logits = self.multichoice_head(multichoice_output)

        # Reshape back to separate choices.
        multichoice_logits = multichoice_logits.view(-1, num_choices)

        return (multichoice_logits, *mems)
