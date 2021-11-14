# -*- encoding: utf-8 -*-
'''
@File    :   base_model.py
@Time    :   2021/10/01 22:40:33
@Author  :   Ming Ding 
@Contact :   dm18@mail.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch

from SwissArmyTransformer.mpu import BaseTransformer

class BaseMixin(torch.nn.Module):
    def __init__(self):
        super(BaseMixin, self).__init__()
        # define new params
    def reinit(self, *pre_mixins):
        # reload the initial params from previous trained modules
        pass
    # can also define hook-functions here
    # ...

class BaseModel(torch.nn.Module):
    def __init__(self, args, transformer=None, parallel_output=True):
        super(BaseModel, self).__init__()
        self.mixins = torch.nn.ModuleDict()
        self.collect_hooks_()
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = BaseTransformer(
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
                hidden_size=args.hidden_size,
                num_attention_heads=args.num_attention_heads,
                max_sequence_length=args.max_sequence_length,
                embedding_dropout_prob=args.hidden_dropout,
                attention_dropout_prob=args.attention_dropout,
                output_dropout_prob=args.hidden_dropout,
                checkpoint_activations=args.checkpoint_activations,
                checkpoint_num_layers=args.checkpoint_num_layers,
                sandwich_ln=args.sandwich_ln,
                parallel_output=parallel_output,
                hooks=self.hooks
            )
        
    def reinit(self): # will be called when loading model
        # if some mixins are loaded, overrides this function
        for m in self.mixins.values(): 
            m.reinit(self.transformer)
            
    def add_mixin(self, name, new_mixin, reinit=False):
        assert name not in self.mixins
        assert isinstance(new_mixin, BaseMixin)
        
        self.mixins[name] = new_mixin # will auto-register parameters
        object.__setattr__(new_mixin, 'transformer', self.transformer) # cannot use pytorch set_attr
        
        if reinit:
            new_mixin.reinit(self.transformer, **self.mixins) # also pass current mixins
        self.collect_hooks_()

    def del_mixin(self, name):
        assert name in self.mixins
        del self.mixins[name]
        self.collect_hooks_()
        
    def get_mixin(self, name):
        return self.mixins[name]
    
    def forward(self, *args, **kwargs):
        # update hooks as the current model (overrided forwards)
        # Attention! the transformer might be shared by multiple models
        self.transformer.hooks.clear()
        self.transformer.hooks.update(self.hooks)
        return self.transformer(*args, **kwargs)
        
    def collect_hooks_(self):
        names = ['word_embedding_forward', 'position_embedding_forward',
                'attention_forward', 'mlp_forward', 'final_forward', 'layer_forward',
                'branch_embedding_forward', 'branch_final_forward'
                ]
        hooks = {}
        hook_origins = {}
        for name in names:
            for mixin_name, m in self.mixins.items():
                if hasattr(m, name):
                    if name in hooks: # conflict
                        raise ValueError(f'Hook {name} conflicts at {mixin_name} and {hook_origins[name]}.')
                    hooks[name] = getattr(m, name)
                    hook_origins[name] = mixin_name
            if hasattr(self, name):
                # if name in hooks: # defined in mixins, can override
                #     print(f'Override {name} in {hook_origins[name]}...')
                hooks[name] = getattr(self, name)
                hook_origins[name] = 'model'
        self.hooks = hooks
        self.hook_origins = hook_origins
        return hooks
    
    def disable_untrainable_params(self):
        pass