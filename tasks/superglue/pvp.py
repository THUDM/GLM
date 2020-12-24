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

"""
This file contains the pattern-verbalizer pairs (PVPs) for all tasks.
"""
import random
import string
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Union, Dict

import torch
from tasks.data_utils import InputExample
from utils import print_rank_0

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    This class contains functions to apply patterns and verbalizers as required by PET. Each task requires its own
    custom implementation of a PVP.
    """

    def __init__(self, wrapper, pattern_id: int = 0, verbalizer_file: str = None, seed: int = 42):
        """
        Create a new PVP.

        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        :param verbalizer_file: an optional file that contains the verbalizer to be used
        :param seed: a seed to be used for generating random numbers if necessary
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.rng = random.Random(seed)

        if verbalizer_file:
            self.verbalize = PVP._load_verbalizer_from_file(verbalizer_file, self.pattern_id)

        self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """Return the maximum number of verbalizers across all labels"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def remove_final_punc(s: Union[str, Tuple[str, bool]]):
        """Remove the final punctuation mark"""
        if isinstance(s, tuple):
            return PVP.remove_final_punc(s[0]), s[1]
        return s.rstrip(string.punctuation)

    @staticmethod
    def lowercase_first(s: Union[str, Tuple[str, bool]]):
        """Lowercase the first character"""
        if isinstance(s, tuple):
            return PVP.lowercase_first(s[0]), s[1]
        return s[0].lower() + s[1:]

    def encode(self, example: InputExample, priming: bool = False, labeled: bool = False) \
            -> Tuple[List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param priming: whether to use this example for priming
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, consisting of a list of input ids and a list of token type ids
        """

        if not priming:
            assert not labeled, "'labeled' can only be set to true if 'priming' is also set to true"

        tokenizer = self.wrapper.tokenizer  # type: PreTrainedTokenizer
        parts_a, parts_b = self.get_parts(example)

        parts_a = [x if isinstance(x, tuple) else (x, False) for x in parts_a]
        parts_a = [(tokenizer.encode(x), s) for x, s in parts_a if x]

        if parts_b:
            parts_b = [x if isinstance(x, tuple) else (x, False) for x in parts_b]
            parts_b = [(tokenizer.encode(x, add_special_tokens=False), s) for x, s in parts_b if x]

        self.truncate(parts_a, parts_b, max_length=self.wrapper.config.max_seq_length)

        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        tokens_b = [token_id for part, _ in parts_b for token_id in part] if parts_b else None

        if priming:
            input_ids = tokens_a
            if tokens_b:
                input_ids += tokens_b
            if labeled:
                mask_idx = input_ids.index(self.mask_id)
                assert mask_idx >= 0, 'sequence of input_ids must contain a mask token'
                assert len(self.verbalize(example.label)) == 1, 'priming only supports one verbalization per label'
                verbalizer = self.verbalize(example.label)[0]
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                input_ids[mask_idx] = verbalizer_id
            return input_ids, []

        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)

        return input_ids, token_type_ids

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        Return all verbalizations for a given label.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0]
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # filler_len.shape() == max_fillers
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # cls_logits.shape() == num_labels x max_fillers  (and 0 when there are not as many fillers).
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        cls_logits = cls_logits * (m2c > 0).float()

        # cls_logits.shape() == num_labels
        cls_logits = cls_logits.sum(axis=1) / filler_len
        return cls_logits

    def convert_plm_logits_to_cls_logits(self, logits: torch.Tensor) -> torch.Tensor:
        assert logits.shape[1] == 1
        logits = torch.squeeze(logits, 1)  # remove second dimension as we always have exactly one <mask> per example
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(lgt) for lgt in logits])
        return cls_logits

    @staticmethod
    def _load_verbalizer_from_file(path: str, pattern_id: int):

        verbalizers = defaultdict(dict)  # type: Dict[int, Dict[str, List[str]]]
        current_pattern_id = None

        with open(path, 'r') as fh:
            for line in fh.read().splitlines():
                if line.isdigit():
                    current_pattern_id = int(line)
                elif line:
                    label, *realizations = line.split()
                    verbalizers[current_pattern_id][label] = realizations

        print_rank_0("Automatically loaded the following verbalizer: \n {}".format(verbalizers[pattern_id]))

        def verbalize(label) -> List[str]:
            return verbalizers[pattern_id][label]

        return verbalize


class AgnewsPVP(PVP):
    VERBALIZER = {
        "1": ["World"],
        "2": ["Sports"],
        "3": ["Business"],
        "4": ["Tech"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'News:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return AgnewsPVP.VERBALIZER[label]


class YahooPVP(PVP):
    VERBALIZER = {
        "1": ["Society"],
        "2": ["Science"],
        "3": ["Health"],
        "4": ["Education"],
        "5": ["Computer"],
        "6": ["Sports"],
        "7": ["Business"],
        "8": ["Entertainment"],
        "9": ["Relationship"],
        "10": ["Politics"],
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0:
            return [self.mask, ':', text_a, text_b], []
        elif self.pattern_id == 1:
            return [self.mask, 'Question:', text_a, text_b], []
        elif self.pattern_id == 2:
            return [text_a, '(', self.mask, ')', text_b], []
        elif self.pattern_id == 3:
            return [text_a, text_b, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text_a, text_b], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text_a, text_b], []
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YahooPVP.VERBALIZER[label]


class MnliPVP(PVP):
    VERBALIZER_A = {
        "contradiction": ["Wrong"],
        "entailment": ["Right"],
        "neutral": ["Maybe"]
    }
    VERBALIZER_B = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(self.remove_final_punc(example.text_a))
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2:
            return ['"', text_a, '" ?'], [self.mask, ', "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3:
            return [text_a, '?'], [self.mask, ',', text_b]

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 1:
            return MnliPVP.VERBALIZER_A[label]
        return MnliPVP.VERBALIZER_B[label]


class YelpPolarityPVP(PVP):
    VERBALIZER = {
        "1": ["bad"],
        "2": ["good"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text = self.shortenable(example.text_a)

        if self.pattern_id == 0:
            return ['It was', self.mask, '.', text], []
        elif self.pattern_id == 1:
            return [text, '. All in all, it was', self.mask, '.'], []
        elif self.pattern_id == 2:
            return ['Just', self.mask, "!"], [text]
        elif self.pattern_id == 3:
            return [text], ['In summary, the restaurant is', self.mask, '.']
        else:
            raise ValueError("No pattern implemented for id {}".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return YelpPolarityPVP.VERBALIZER[label]


class YelpFullPVP(YelpPolarityPVP):
    VERBALIZER = {
        "1": ["terrible"],
        "2": ["bad"],
        "3": ["okay"],
        "4": ["good"],
        "5": ["great"]
    }

    def verbalize(self, label) -> List[str]:
        return YelpFullPVP.VERBALIZER[label]


class XStancePVP(PVP):
    VERBALIZERS = {
        'en': {"FAVOR": ["Yes"], "AGAINST": ["No"]},
        'de': {"FAVOR": ["Ja"], "AGAINST": ["Nein"]},
        'fr': {"FAVOR": ["Oui"], "AGAINST": ["Non"]}
    }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)

        if self.pattern_id == 0 or self.pattern_id == 2 or self.pattern_id == 4:
            return ['"', text_a, '"'], [self.mask, '. "', text_b, '"']
        elif self.pattern_id == 1 or self.pattern_id == 3 or self.pattern_id == 5:
            return [text_a], [self.mask, '.', text_b]

    def verbalize(self, label) -> List[str]:
        lang = 'de' if self.pattern_id < 2 else 'en' if self.pattern_id < 4 else 'fr'
        return XStancePVP.VERBALIZERS[lang][label]


class RtePVP(PVP):
    VERBALIZER = {
        "not_entailment": ["No"],
        "entailment": ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        # switch text_a and text_b to get the correct order
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b.rstrip(string.punctuation))

        if self.pattern_id == 0:
            return ['"', text_b, '" ?'], [self.mask, ', "', text_a, '"']
        elif self.pattern_id == 1:
            return [text_b, '?'], [self.mask, ',', text_a]
        if self.pattern_id == 2:
            return ['"', text_b, '" ?'], [self.mask, '. "', text_a, '"']
        elif self.pattern_id == 3:
            return [text_b, '?'], [self.mask, '.', text_a]
        elif self.pattern_id == 4:
            return [text_a, ' question: ', self.shortenable(example.text_b), ' True or False? answer:', self.mask], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return ['true'] if label == 'entailment' else ['false']
        return RtePVP.VERBALIZER[label]


class CbPVP(RtePVP):
    VERBALIZER = {
        "contradiction": ["No"],
        "entailment": ["Yes"],
        "neutral": ["Maybe"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        if self.pattern_id == 4:
            text_a = self.shortenable(example.text_a)
            text_b = self.shortenable(example.text_b)
            return [text_a, ' question: ', text_b, ' true, false or neither? answer:', self.mask], []
        return super().get_parts(example)

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 4:
            return ['true'] if label == 'entailment' else ['false'] if label == 'contradiction' else ['neither']
        return CbPVP.VERBALIZER[label]


class CopaPVP(PVP):

    def get_parts(self, example: InputExample) -> FilledPattern:

        premise = self.remove_final_punc(self.shortenable(example.text_a))
        choice1 = self.remove_final_punc(self.lowercase_first(example.meta['choice1']))
        choice2 = self.remove_final_punc(self.lowercase_first(example.meta['choice2']))

        question = example.meta['question']
        assert question in ['cause', 'effect']

        example.meta['choice1'], example.meta['choice2'] = choice1, choice2
        num_masks = max(len(get_verbalization_ids(c, self.wrapper.tokenizer, False)) for c in [choice1, choice2])

        if question == 'cause':
            if self.pattern_id == 0:
                return ['"', choice1, '" or "', choice2, '"?', premise, 'because', self.mask * num_masks, '.'], []
            elif self.pattern_id == 1:
                return [choice1, 'or', choice2, '?', premise, 'because', self.mask * num_masks, '.'], []
        else:
            if self.pattern_id == 0:
                return ['"', choice1, '" or "', choice2, '"?', premise, ', so', self.mask * num_masks, '.'], []
            elif self.pattern_id == 1:
                return [choice1, 'or', choice2, '?', premise, ', so', self.mask * num_masks, '.'], []

    def verbalize(self, label) -> List[str]:
        return []


class WscPVP(PVP):

    def get_parts(self, example: InputExample) -> FilledPattern:
        pronoun = example.meta['span2_text']
        target = example.meta['span1_text']
        pronoun_idx = example.meta['span2_index']

        words_a = example.text_a.split()
        words_a[pronoun_idx] = '*' + words_a[pronoun_idx] + '*'
        text_a = ' '.join(words_a)
        text_a = self.shortenable(text_a)

        num_pad = self.rng.randint(0, 3) if 'train' in example.guid else 1
        num_masks = len(get_verbalization_ids(target, self.wrapper.tokenizer, force_single_token=False)) + num_pad
        masks = self.mask * num_masks

        if self.pattern_id == 0:
            return [text_a, "The pronoun '*" + pronoun + "*' refers to", masks + '.'], []
        elif self.pattern_id == 1:
            return [text_a, "In the previous sentence, the pronoun '*" + pronoun + "*' refers to", masks + '.'], []
        elif self.pattern_id == 2:
            return [text_a,
                    "Question: In the passage above, what does the pronoun '*" + pronoun + "*' refer to? Answer: ",
                    masks + '.'], []

    def verbalize(self, label) -> List[str]:
        return []


class BoolQPVP(PVP):
    VERBALIZER_A = {
        "False": ["No"],
        "True": ["Yes"]
    }

    VERBALIZER_B = {
        "False": ["false"],
        "True": ["true"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = self.shortenable(example.text_a)
        question = self.shortenable(example.text_b)

        if self.pattern_id < 2:
            return [passage, '. Question: ', question, '? Answer: ', self.mask, '.'], []
        elif self.pattern_id < 4:
            return [passage, '. Based on the previous passage, ', question, '?', self.mask, '.'], []
        else:
            return ['Based on the following passage, ', question, '?', self.mask, '.', passage], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 0 or self.pattern_id == 2 or self.pattern_id == 4:
            return BoolQPVP.VERBALIZER_A[label]
        else:
            return BoolQPVP.VERBALIZER_B[label]


class MultiRcPVP(PVP):
    VERBALIZER = {
        "0": ["No"],
        "1": ["Yes"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        passage = self.shortenable(example.text_a)
        question = example.text_b
        answer = example.meta['answer']

        if self.pattern_id == 0:
            return [passage, '. Question: ', question, '? Is it ', answer, '?', self.mask, '.'], []
        if self.pattern_id == 1:
            return [passage, '. Question: ', question, '? Is the correct answer "', answer, '"?', self.mask, '.'], []
        if self.pattern_id == 2:
            return [passage, '. Based on the previous passage, ', question, '? Is "', answer, '" a correct answer?',
                    self.mask, '.'], []
        if self.pattern_id == 3:
            return [passage, question, '- [', self.mask, ']', answer], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 3:
            return ['False'] if label == "0" else ['True']
        return MultiRcPVP.VERBALIZER[label]


class WicPVP(PVP):
    VERBALIZER_A = {
        "F": ["No"],
        "T": ["Yes"]
    }
    VERBALIZER_B = {
        "F": ["2"],
        "T": ["b"]
    }

    def get_parts(self, example: InputExample) -> FilledPattern:
        text_a = self.shortenable(example.text_a)
        text_b = self.shortenable(example.text_b)
        word = example.meta['word']

        if self.pattern_id == 0:
            return ['"', text_a, '" / "', text_b, '" Similar sense of "' + word + '"?', self.mask, '.'], []
        if self.pattern_id == 1:
            return [text_a, text_b, 'Does ' + word + ' have the same meaning in both sentences?', self.mask], []
        if self.pattern_id == 2:
            return [word, ' . Sense (1) (a) "', text_a, '" (', self.mask, ') "', text_b, '"'], []

    def verbalize(self, label) -> List[str]:
        if self.pattern_id == 2:
            return WicPVP.VERBALIZER_B[label]
        return WicPVP.VERBALIZER_A[label]


class RecordPVP(PVP):

    def get_parts(self, example: InputExample) -> FilledPattern:
        premise = self.shortenable(example.text_a)
        choices = example.meta['candidates']

        assert '@placeholder' in example.text_b, f'question "{example.text_b}" does not contain a @placeholder token'
        num_masks = max(len(get_verbalization_ids(c, self.wrapper.tokenizer, False)) for c in choices)
        question = example.text_b.replace('@placeholder', self.mask * num_masks)
        return [premise, question], []

    def verbalize(self, label) -> List[str]:
        return []


def get_verbalization_ids(word: str, tokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    ids = tokenizer.EncodeAsIds(word)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.DecodeIds(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.IdToToken(verbalization_id)}'
    return verbalization_id


PVPS = {
    'agnews': AgnewsPVP,
    'mnli': MnliPVP,
    'yelp-polarity': YelpPolarityPVP,
    'yelp-full': YelpFullPVP,
    'yahoo': YahooPVP,
    'xstance': XStancePVP,
    'xstance-de': XStancePVP,
    'xstance-fr': XStancePVP,
    'rte': RtePVP,
    'wic': WicPVP,
    'cb': CbPVP,
    'wsc': WscPVP,
    'boolq': BoolQPVP,
    'copa': CopaPVP,
    'multirc': MultiRcPVP,
    'record': RecordPVP,
    'ax-b': RtePVP,
    'ax-g': RtePVP,
}
