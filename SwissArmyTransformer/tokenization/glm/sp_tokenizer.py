"""
from https://github.com/openai/gpt-2/, changed for chinese
"""
import json
import os
import csv
import nltk
import random

from nltk import tokenize as nltk_tokenize
import sentencepiece as spm

"""
SentencePiece is an unsupervised text tokenizer and detokenizer mainly for Neural Network-based text generation 
systems where the vocabulary size is predetermined prior to the neural model training. SentencePiece implements 
subword units (e.g., byte-pair-encoding (BPE) [Sennrich et al.]) and unigram language model [Kudo.]) with the 
extension of direct training from raw sentences. SentencePiece allows us to make a purely end-to-end 
system that does not depend on language-specific pre/postprocessing.
https://github.com/google/sentencepiece

pip install sentencepiece

or  git clone https://github.com/google/sentencepiece.git
python setup.py install

"""
PRETRAINED_MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
     'embed_assets', 'chinese_sentencepiece/cog-pretrain.model')

class SentencePieceTokenizer:
    """Trains and uses sentencepiece for text tokenization"""

    def __init__(self, model_path=None, **kwargs):
        self.spm_model = model_path
        self._tokens = []
        self._vocab = {}
        self.sp, self.vocab_size = None, 0
        self.load_spm_model()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        if pretrained_model_name_or_path in ['glm-large', 'glm-10b']:
            return cls(model_path=PRETRAINED_MODEL_FILE)
        else:
            return cls(model_path=pretrained_model_name_or_path)

    def __len__(self):
        return self.num_text_tokens

    def load_spm_model(self):
        """load sentencepiece model and parse vocab"""
        if not os.path.exists(self.spm_model) and not self.spm_model.endswith('.model'):
            self.spm_model = self.spm_model + '.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.spm_model)
        self.vocab_size = self.num_text_tokens = len(self.sp)
        self._tokens = [self.IdToToken(t) for t in range(self.vocab_size)]
        self._vocab = {t: i for i, t in enumerate(self._tokens)}

    @property
    def tokens(self):
        return self._tokens

    @property
    def vocab(self):
        return self._vocab

    @staticmethod
    def exists(model_path):
        if model_path is None:
            return False
        # check if path exists
        dne = not os.path.exists(model_path)
        # check if path.model exists
        if dne and not model_path.endswith('.model'):
            dne = not os.path.exists(model_path + '.model')
        return not dne

    def encode(self, text):
        """convert text to sentencepiece Ids"""
        tokens = self.sp.EncodeAsIds(text)
        return tokens

    def IdToToken(self, Id):
        """convert Id to sentencpiece token"""
        return self.sp.IdToPiece(Id)

    def TokenToId(self, token):
        """convert sentencpiece token to Id"""
        return self.sp.PieceToId(token)

    def decode(self, Ids):
        """converts ids to a text string"""
        return self.sp.DecodeIds(Ids)


def from_pretrained():
    return SentencePieceTokenizer(model_path=PRETRAINED_MODEL_FILE)