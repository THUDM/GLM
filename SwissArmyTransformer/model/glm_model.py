import torch
import torch.nn as nn

from .base_model import BaseModel
from .mixins import BaseMixin

class BlockPositionEmbeddingMixin(BaseMixin):
    def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
        super(BlockPositionEmbeddingMixin, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
    
    def position_embedding_forward(self, position_ids, **kwargs):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.transformer.position_embeddings(position_ids)
        block_position_embeddings = self.block_position_embeddings(block_position_ids)
        return position_embeddings + block_position_embeddings

class GLMModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.add_mixin('block_position_embedding', 
            BlockPositionEmbeddingMixin(args.max_sequence_length, args.hidden_size)
        )
    
