import os
from typing import Optional, Tuple, List, Union
from shutil import copyfile
import torch

from transformers import PreTrainedTokenizer, RobertaTokenizer, GPT2Tokenizer, BertTokenizer
from transformers.utils import logging
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.auto.tokenization_auto import get_tokenizer_config
from transformers.utils import torch_required
from transformers.utils.generic import _is_torch_device
import sentencepiece as spm

logger = logging.get_logger(__name__)


class GLMBatchEncoding(BatchEncoding):
    @torch_required
    def to(self, device: Union[str, "torch.device"]) -> "BatchEncoding":
        """
        Send all values to device by calling `v.to(device)` (PyTorch only).

        Args:
            device (`str` or `torch.device`): The device to put the tensors on.

        Returns:
            [`BatchEncoding`]: The same instance after modification.
        """

        # This check catches things like APEX blindly calling "to" on all inputs to a module
        # Otherwise it passes the casts down and casts the LongTensor containing the token idxs
        # into a HalfTensor
        if isinstance(device, str) or _is_torch_device(device) or isinstance(device, int):
            self.data = {k: v.to(device=device) if torch.is_tensor(v) else v for k, v in self.data.items()}
        else:
            logger.warning(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self


class GLMTokenizerMixin:
    @property
    def sop_token(self) -> Optional[str]:
        return "<|startofpiece|>"

    @property
    def sop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the start token in the vocabulary, used when training a model with autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.sop_token)

    @property
    def eop_token(self) -> Optional[str]:
        return "<|endofpiece|>"

    @property
    def eop_token_id(self) -> Optional[int]:
        """
        `Optional[int]`: Id of the end token in the vocabulary, used when training a model with autoregressive blank filling.
        """
        return self.convert_tokens_to_ids(self.eop_token)

    @property
    def gmask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[gMASK]")

    @property
    def smask_token_id(self) -> int:
        return self.convert_tokens_to_ids("[sMASK]")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id, self.smask_token_id, self.gmask_token_id]

    def _build_input_for_multiple_choice(self, context, choices):
        context_id = context["input_ids"]
        if torch.is_tensor(context_id):
            context_id = context_id.tolist()

        division = len(context_id)
        mask_position = context_id.index(self.mask_token_id)

        token = torch.tensor(context_id, dtype=torch.long)
        attention_mask = [context["attention_mask"].expand(division, -1)]
        position_id = torch.arange(division, dtype=torch.long)
        block_position_id = torch.zeros(division, dtype=torch.long)

        choice_ids, choice_indices = [], []

        for choice_str in choices:
            choice = torch.tensor(self(choice_str, add_special_tokens=False, padding=False)['input_ids'],
                                  dtype=torch.long)
            choice_ids.append(choice)
            choice_indices.append(torch.arange(len(token), len(token) + len(choice), dtype=torch.long))
            attention_mask.append(torch.tril(torch.ones((len(choice), len(choice)), dtype=torch.long)))

            token = torch.cat((token, torch.tensor([self.sop_token_id], dtype=torch.long), choice[:-1]))
            position_id = torch.cat((position_id, torch.tensor([mask_position] * len(choice), dtype=torch.long)))
            block_position_id = torch.cat((block_position_id, torch.arange(1, 1 + len(choice), dtype=torch.long)))

        attention_mask = torch.block_diag(*attention_mask)
        attention_mask[division:, :division] = context["attention_mask"].unsqueeze(0)

        return {
            "input_ids": token,
            "position_ids": torch.stack((position_id, block_position_id)),
            "attention_mask": attention_mask,
            "choice_ids": choice_ids,
            "choice_indices": choice_indices
        }

    def _pad_batch(self, tokens, position_ids, attention_mask, max_seq_length):
        pad_length = max_seq_length - len(tokens)
        attention_mask = torch.nn.functional.pad(
            attention_mask,
            (0, pad_length, 0, pad_length),
            mode="constant",
            value=0,
        )
        tokens = torch.cat((tokens, torch.zeros(pad_length, dtype=torch.long)))
        position_ids = torch.cat((position_ids, position_ids[..., -1:].expand(-1, pad_length)), dim=-1)
        return tokens, position_ids, attention_mask

    def _collate(self, samples):
        TILE = 1
        length_to_pad = (max(map(lambda spl: len(spl["input_ids"]), samples)) + TILE - 1) // TILE * TILE

        token_batch, position_id_batch, attention_mask_batch = [], [], []
        choices_batch, choice_target_ids_batch = [], []

        for sample in samples:
            token, position_id, attention_mask = self._pad_batch(
                sample["input_ids"], sample["position_ids"], sample["attention_mask"], length_to_pad
            )
            token_batch.append(token)
            position_id_batch.append(position_id)
            attention_mask_batch.append(attention_mask)
            choices_batch.append(sample["choice_ids"])
            choice_target_ids_batch.append(sample["choice_indices"])
        return {
            "input_ids": torch.stack(token_batch),
            "position_ids": torch.stack(position_id_batch),
            "attention_mask": torch.stack(attention_mask_batch).unsqueeze(1),
            "choice_ids": choices_batch,
            "choice_indices": choice_target_ids_batch,
        }

    def build_inputs_for_multiple_choice(self, model_input: BatchEncoding, choices, max_length=None):
        samples = [{key: value[i] for key, value in model_input.items()} for i in range(len(model_input["input_ids"]))]
        samples = [self._build_input_for_multiple_choice(sample, choice) for sample, choice in
                   zip(samples, choices)]
        inputs = self._collate(samples)
        return GLMBatchEncoding(inputs)

    def build_inputs_for_generation(self, model_input: BatchEncoding, max_gen_length=512, targets=None, padding=False):
        mask_ids = self.mask_token_ids
        input_ids = model_input.input_ids
        batch_size, seq_length = input_ids.shape[:2]
        position_id, block_position_id = list(range(seq_length)), [0 for _ in range(seq_length)]
        position_ids, block_position_ids = [], []
        labels = None
        if targets is not None:
            is_batched = isinstance(targets, (list, tuple))
            targets = self(targets, add_special_tokens=False, padding=False).input_ids
            if not is_batched:
                targets = [targets]
            assert len(targets) == len(input_ids)
            targets = [(target + [self.eop_token_id])[:max_gen_length] for target in targets]
            if not padding:
                max_gen_length = max(map(len, targets))
            targets = [[self.sop_token_id] + target for target in targets]
            labels = [target[1:] for target in targets]
            targets = [target + [self.pad_token_id] * (max_gen_length + 1 - len(target)) for target in targets]
            labels = [label + [-100] * (max_gen_length - len(label)) for label in labels]
            targets = torch.tensor(targets, dtype=input_ids.dtype, device=input_ids.device)
            labels = torch.tensor(labels, dtype=input_ids.dtype, device=input_ids.device)
            labels = torch.cat((input_ids.new_full((batch_size, seq_length), -100), labels), dim=1)
        for i in range(batch_size):
            mask_positions = []
            for mask_id in mask_ids:
                mask_positions += (input_ids[i] == mask_id).nonzero(as_tuple=True)[0].tolist()
            if not mask_positions:
                raise ValueError("Cannot find mask token in the input")
            mask_positions.sort()
            mask_pos = mask_positions[0]
            position_ids.append(position_id + [mask_pos] * max_gen_length)
            block_position_ids.append(block_position_id + list(range(1, max_gen_length + 1)))
        position_ids = torch.tensor(position_ids, dtype=input_ids.dtype, device=input_ids.device)
        block_position_ids = torch.tensor(block_position_ids, dtype=input_ids.dtype, device=input_ids.device)
        position_ids = torch.stack((position_ids, block_position_ids), dim=1)
        attention_mask = model_input.attention_mask
        attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length + max_gen_length, -1)
        generation_attention_mask = torch.cat([attention_mask.new_zeros((seq_length, max_gen_length)),
                                               torch.tril(attention_mask.new_ones((max_gen_length, max_gen_length)))],
                                              dim=0).unsqueeze(0).expand(batch_size, -1, -1)
        attention_mask = torch.cat((attention_mask, generation_attention_mask), dim=2)
        attention_mask = attention_mask.unsqueeze(1)
        if targets is None:
            input_ids = torch.cat((input_ids, input_ids.new_full((batch_size, 1), self.sop_token_id)), dim=-1)
        else:
            input_ids = torch.cat((input_ids, targets[:, :-1]), dim=1)
        batch = {"input_ids": input_ids, "position_ids": position_ids}
        if labels is None:
            batch["generation_attention_mask"] = attention_mask
        else:
            batch["attention_mask"] = attention_mask
            batch["labels"] = labels
        return BatchEncoding(batch)


class GLMRobertaTokenizer(RobertaTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]


class GLMChineseTokenizer(PreTrainedTokenizer, GLMTokenizerMixin):
    vocab_files_names = {"vocab_file": "cog-pretrain.model"}
    truncation_side: str = "left"

    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text, **kwargs):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.decode(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMGPT2Tokenizer(GPT2Tokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A BERT sequence has the following format:

        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        assert token_ids_1 is None
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + eos


class GLMBertTokenizer(BertTokenizer, GLMTokenizerMixin):
    model_input_names = ["input_ids", "position_ids", "attention_mask"]
    truncation_side: str = "left"

    @property
    def gmask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support gMASK")

    @property
    def smask_token_id(self) -> int:
        raise NotImplementedError("The model doesn't support sMASK")

    @property
    def mask_token_ids(self):
        return [self.mask_token_id]


class GLMTokenizer:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
        config_tokenizer_class = tokenizer_config.get("tokenizer_class")
        if config_tokenizer_class == "GLMRobertaTokenizer":
            tokenizer_class = GLMRobertaTokenizer
        elif config_tokenizer_class == "GLMChineseTokenizer":
            tokenizer_class = GLMChineseTokenizer
        elif config_tokenizer_class == "GLMGPT2Tokenizer":
            tokenizer_class = GLMGPT2Tokenizer
        elif config_tokenizer_class == "GLMBertTokenizer":
            tokenizer_class = GLMBertTokenizer
        else:
            raise NotImplementedError("Not implemented tokenizer type:", config_tokenizer_class)
        return tokenizer_class.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
