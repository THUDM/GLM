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

import mpu
from model.prompt import PromptSpell
from utils import print_rank_0


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


class GLMModel(torch.nn.Module):
    """GLM Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 max_memory_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True,
                 relative_encoding=False,
                 block_position_encoding=False,
                 output_predict=True,
                 spell_length=None,
                 spell_func='lstm',
                 attention_scale=1.0,
                 ):

        super(GLMModel, self).__init__()

        self.parallel_output = parallel_output
        self.output_predict = output_predict
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        # Transformer
        self.transformer = mpu.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       max_sequence_length,
                                                       max_memory_length,
                                                       embedding_dropout_prob,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers,
                                                       attention_scale=attention_scale,
                                                       relative_encoding=relative_encoding,
                                                       block_position_encoding=block_position_encoding)
        if spell_length is not None:
            self.prompt_spell = PromptSpell(spell_length, self.hidden_size, spell_func)

    def freeze_transformer(self, tune_prefix_layers=None):
        log_str = "Freeze transformer"
        self.word_embeddings.requires_grad_(False)
        self.transformer.requires_grad_(False)
        if tune_prefix_layers is not None:
            log_str += f" tune {tune_prefix_layers} prefix layers"
            for i in range(tune_prefix_layers):
                self.transformer.layers[i].requires_grad_(True)
        print_rank_0(log_str)

    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None):
        # Embeddings.
        batch_size = input_ids.size(0)
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        if prompt_pos is not None:
            embeddings = embeddings.clone()
            prompt_embeds = self.prompt_spell()
            batch_index = torch.arange(batch_size, device=input_ids.device).unsqueeze(1)
            embeddings[batch_index, prompt_pos] = prompt_embeds
        # Transformer.
        transformer_output = self.transformer(embeddings, position_ids, attention_mask, mems,
                                              return_memory=return_memory, detach_memory=detach_memory)
        logits, hidden_layers = transformer_output
        outputs = hidden_layers

        if self.output_predict:
            # Parallel logits.
            logits_parallel = mpu.copy_to_model_parallel_region(
                logits)
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)

            if self.parallel_output:
                return (logits_parallel, *outputs)

            return (mpu.gather_from_model_parallel_region(logits_parallel), *outputs)
        else:
            return (logits, *outputs)


class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, layers*hidden)
    """

    def __init__(self, prefix_length, num_layers, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(prefix_length, num_layers * hidden_size)

    def forward(self, prefix: torch.Tensor):
        past_key_values = self.embedding(prefix)
        return past_key_values


class GLMFPrefixModel(GLMModel):
    def __init__(self, prefix_prompt, num_layers, hidden_size, **kwargs):
        super(GLMFPrefixModel, self).__init__(num_layers=num_layers, hidden_size=hidden_size, **kwargs)
        print(f"Create prefix prompt of length {prefix_prompt}")
        self.prefix_length = prefix_prompt
        self.prefix_tokens = torch.arange(self.prefix_length).long()
        self.prefix_encoder = PrefixEncoder(prefix_length=prefix_prompt, num_layers=num_layers, hidden_size=hidden_size)

    def get_prompt(self, batch_size, device):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        prefix_hidden_states = self.prefix_encoder(prefix_tokens).half()
        # bsz, seqlen, _ = past_key_values.shape
        prefix_hidden_states = prefix_hidden_states.view(
            batch_size,
            self.prefix_length,
            self.num_layers,
            self.hidden_size
        )
        prefix_hidden_states = prefix_hidden_states.permute([2, 0, 1, 3]).split(1)
        prefix_hidden_states = [value[0] for value in prefix_hidden_states]
        return prefix_hidden_states

    def forward(self, input_ids, position_ids, attention_mask, *mems, return_memory=False, detach_memory=True,
                prompt_pos=None
                ):
        batch_size = input_ids.shape[0]
        prompt_hidden_states = self.get_prompt(batch_size=batch_size, device=input_ids.device)
        mems = [torch.cat((mem, prompt), dim=1) for mem, prompt in
                zip(mems, prompt_hidden_states)] if mems else prompt_hidden_states
        output, *mems = super().forward(input_ids, position_ids, attention_mask, *mems, return_memory=return_memory,
                                       detach_memory=detach_memory)
        if return_memory:
            mems = [mem[:, self.prefix_length:] for mem in mems]
        return (output, *mems)


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n == 'bias'])

    if len(weight_decay_params['params']) == 0:
        return (no_weight_decay_params,)
    elif len(no_weight_decay_params['params']) == 0:
        return (weight_decay_params,)

    return weight_decay_params, no_weight_decay_params
