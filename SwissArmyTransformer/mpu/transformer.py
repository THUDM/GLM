# coding=utf-8
# rewritten, Copyright (c) 2021, Ming Ding.  All rights reserved.
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

"""Transformer."""

import math
import copy
import torch
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm

from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from .mappings import gather_from_model_parallel_region, copy_to_model_parallel_region

from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint, get_cuda_rng_tracker

from .utils import divide, sqrt, scaled_init_method, unscaled_init_method, gelu
from .utils import split_tensor_along_last_dim

class LayerNorm(FusedLayerNorm):
    def __init__(self, *args, pb_relax=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pb_relax = pb_relax
    def forward(self, x):
        if not self.pb_relax:
            return super().forward(x)
        return super().forward(x / (x.abs().max().detach()/8))

def standard_attention(query_layer, key_layer, value_layer, attention_mask,
                    attention_dropout=None, log_attention_weights=None):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training.
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers.

    attention_scores = torch.matmul(
        query_layer / math.sqrt(query_layer.shape[-1]),
        key_layer.transpose(-1, -2)
    )
    if log_attention_weights is not None:
        attention_scores += log_attention_weights

    # if attention_mask.shape[-2] > 1: # if auto-regressive, skip
    attention_scores = torch.mul(attention_scores, attention_mask) - \
                10000.0 * (1.0 - attention_mask)

    attention_probs = F.softmax(attention_scores, dim=-1)

    if attention_dropout is not None:
        with get_cuda_rng_tracker().fork():
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                attention_dropout_prob, output_dropout_prob,
                init_method, layer_id, output_layer_init_method=None,
                hooks={}):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size()
        self.hidden_size_per_partition = divide(hidden_size, world_size)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)

        # Strided linear layer.
        self.query_key_value = ColumnParallelLinear(
            hidden_size,
            3*hidden_size,
            stride=3,
            gather_output=False,
            init_method=init_method
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = RowParallelLinear(
            hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)


    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                            (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask, **kw_args):
        if 'attention_forward' in self.hooks:
            return self.hooks['attention_forward'](hidden_states, mask, **kw_args, layer_id=self.layer_id)
        else:
            mixed_raw_layer = self.query_key_value(hidden_states)
            (mixed_query_layer,
                mixed_key_layer,
                mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

            dropout_fn = self.attention_dropout if self.training else None

            query_layer = self._transpose_for_scores(mixed_query_layer)
            key_layer = self._transpose_for_scores(mixed_key_layer)
            value_layer = self._transpose_for_scores(mixed_value_layer)

            context_layer = standard_attention(query_layer, key_layer, value_layer, mask, dropout_fn)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
            context_layer = context_layer.view(*new_context_layer_shape)
            output = self.dense(context_layer)

            if self.training:
                output = self.output_dropout(output)

            return output, None


class MLP(torch.nn.Module):
    def __init__(self, hidden_size, output_dropout_prob, init_method,
                output_layer_init_method=None, hooks={}):
        super(MLP, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size,
            4*hidden_size,
            gather_output=False,
            init_method=init_method
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            4*hidden_size,
            hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method
        )
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, hidden_states, **kw_args):
        if 'mlp_forward' in self.hooks:
            output = self.hooks['mlp_forward'](hidden_states, **kw_args, layer_id=self.layer_id)
        else:
            intermediate_parallel = self.dense_h_to_4h(hidden_states)
            intermediate_parallel = gelu(intermediate_parallel)
            output = self.dense_4h_to_h(intermediate_parallel)

        if self.training:
            output = self.dropout(output)
        return output


class BaseTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        layernorm_epsilon,
        init_method,
        layer_id,
        output_layer_init_method=None,
        sandwich_ln=True,
        hooks={}
    ):
        super(BaseTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.hooks = hooks

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            output_layer_init_method=output_layer_init_method,
            hooks=hooks
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.sandwich_ln = sandwich_ln
        if sandwich_ln:
            self.third_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            output_layer_init_method=output_layer_init_method,
            hooks=hooks
        )

    def forward(self, hidden_states, mask, **kw_args):
        '''
            hidden_states: [batch, seq_len, hidden_size]
            mask: [(1, 1), seq_len, seq_len]
        '''

        # Layer norm at the begining of the transformer layer.
        layernorm_output1 = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, output_this_layer = self.attention(layernorm_output1, mask, **kw_args)

        # Third LayerNorm
        if self.sandwich_ln:
            attention_output = self.third_layernorm(attention_output)

        # Residual connection.
        layernorm_input = hidden_states + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # MLP.
        mlp_output = self.mlp(layernorm_output, **kw_args)

        # Fourth LayerNorm
        if self.sandwich_ln:
            mlp_output = self.fourth_layernorm(mlp_output)

        # Second residual connection.
        output = layernorm_input + mlp_output

        return output, output_this_layer # temporally, output_this_layer is only from attention

class BaseTransformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 sandwich_ln=True,
                 parallel_output=True,
                 hooks={}
                 ):
        super(BaseTransformer, self).__init__()

        # recording parameters
        self.parallel_output = parallel_output
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_sequence_length = max_sequence_length
        self.hooks = copy.copy(hooks) # hooks will be updated each forward

        # create embedding parameters
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        self.word_embeddings = VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=unscaled_init_method(0.02))

        self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        # create all layers
        self.output_layer_init_method = scaled_init_method(init_method_std, num_layers)
        self.init_method = unscaled_init_method(init_method_std)
        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                output_layer_init_method=self.output_layer_init_method,
                sandwich_ln=sandwich_ln,
                hooks=self.hooks
                )
        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        self.final_layernorm = LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, input_ids, position_ids, attention_mask, *, branch_input=None, output_hidden_states=False,
                **kw_args):
        # sanity check
        assert len(input_ids.shape) == 2
        batch_size, query_length = input_ids.shape
        assert len(attention_mask.shape) == 2 or \
            len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1
        assert branch_input is None or 'layer_forward' in self.hooks and isinstance(branch_input, torch.Tensor)
        # branch_input is a new part of input need layer-by-layer update,
        #   but with different hidden_dim and computational routine.
        #   In most cases, you can just ignore it.

        # embedding part
        if 'word_embedding_forward' in self.hooks:
            hidden_states = self.hooks['word_embedding_forward'](input_ids, **kw_args)
        else: # default
            hidden_states = self.word_embeddings(input_ids)

        if 'position_embedding_forward' in self.hooks:
            position_embeddings = self.hooks['position_embedding_forward'](position_ids, **kw_args)
        else:
            assert len(position_ids.shape) <= 2
            assert position_ids.shape[-1] == query_length
            position_embeddings = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        hidden_states_outputs = [hidden_states] if output_hidden_states else []
        # branch related embedding
        if branch_input is None and 'branch_embedding_forward' in self.hooks:
            branch_input = self.hooks['branch_embedding_forward'](branch_input, **kw_args)

        # define custom_forward for checkpointing
        output_per_layers = []
        if self.checkpoint_activations:
            def custom(start, end):
                def custom_forward(*inputs):
                    layers_ = self.layers[start:end]
                    x_, mask = inputs[0], inputs[1]
                    if len(inputs) > 2: # have branch_input
                        branch_ = inputs[2]
                    output_per_layers_part = []
                    for i, layer in enumerate(layers_):
                        if len(inputs) > 2:
                            x_, branch_, output_this_layer = self.hooks['layer_forward'](
                                x_, mask, layer_id=layer.layer_id, branch_input=branch_, **kw_args
                            )
                        elif 'layer_forward' in self.hooks:
                            x_, output_this_layer = self.hooks['layer_forward'](
                                x_, mask, layer_id=layer.layer_id, **kw_args
                            )
                        else:
                            x_, output_this_layer = layer(x_, mask, **kw_args)
                        output_per_layers_part.append(output_this_layer)
                    return x_, output_per_layers_part
                return custom_forward

            l, num_layers = 0, len(self.layers)
            chunk_length = self.checkpoint_num_layers
            hidden_states.requires_grad_(True)
            while l < num_layers:
                args = [hidden_states, attention_mask]
                if branch_input is not None:
                    hidden_states, branch_input, output_per_layers_part = checkpoint(custom(l, l + chunk_length), *args, branch_input)
                else:
                    hidden_states, output_per_layers_part = checkpoint(custom(l, l + chunk_length), *args)
                if output_hidden_states:
                    hidden_states_outputs.append(hidden_states)
                output_per_layers.extend(output_per_layers_part)
                l += chunk_length
        else:
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]
                if branch_input is not None: # customized layer_forward with branch_input
                    hidden_states, branch_input, output_this_layer = self.hooks['layer_forward'](*args, layer_id=torch.tensor(i), branch_input=branch_input, **kw_args)
                elif 'layer_forward' in self.hooks: # customized layer_forward
                    hidden_states, output_this_layer = self.hooks['layer_forward'](*args, layer_id=torch.tensor(i), **kw_args)
                else:
                    hidden_states, output_this_layer = layer(*args, **kw_args)
                if output_hidden_states:
                    hidden_states_outputs.append(hidden_states)
                output_per_layers.append(output_this_layer)

        # Final layer norm.
        logits = self.final_layernorm(hidden_states)

        if 'final_forward' in self.hooks:
            logits_parallel = self.hooks['final_forward'](logits, **kw_args)
        else:
            logits_parallel = copy_to_model_parallel_region(logits)
            logits_parallel = F.linear(logits_parallel, self.word_embeddings.weight)

        # branch related embedding
        if branch_input is None and 'branch_final_forward' in self.hooks:
            branch_input = self.hooks['branch_final_forward'](branch_input, **kw_args)

        if not self.parallel_output:
            logits_parallel = gather_from_model_parallel_region(logits_parallel)

        outputs = [logits_parallel]
        if branch_input is not None:
            outputs.append(branch_input)
        if output_hidden_states:
            outputs.append(hidden_states_outputs)
        outputs.extend(output_per_layers)

        return outputs

