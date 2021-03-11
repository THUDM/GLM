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
This file contains the logic for loading training and test data for all tasks.
"""

import csv
import json
import os
import random
import copy
from abc import ABC, abstractmethod
from collections import Counter
from typing import List, Dict, Callable
from torch.utils.data import Dataset
from tqdm import tqdm

from tasks.data_utils import InputExample
from utils import print_rank_0
from tasks.superglue.pvp import PVPS
from tasks.data_utils import build_input_from_ids, build_sample, num_special_tokens_to_add
from collections import defaultdict

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
TRUE_DEV_SET = "true_dev"
UNLABELED_SET = "unlabeled"

SPLIT_TYPES = [TRAIN_SET, DEV_SET, TEST_SET, TRUE_DEV_SET, UNLABELED_SET]


def get_output_func(task_name):
    return PROCESSORS[task_name]().output_prediction


class GlueDataset(Dataset):

    def __init__(self, args, split, tokenizer, for_train=False):
        task_name = args.task.lower()
        data_dir = args.data_dir
        processor = PROCESSORS[task_name]()
        print_rank_0(
            f"Creating {task_name} dataset from file at {data_dir} (split={split})"
        )
        self.dataset_name = f"{task_name}-{split}"
        if split == DEV_SET:
            examples = processor.get_dev_examples(data_dir, for_train=for_train)
        elif split == TEST_SET:
            examples = processor.get_test_examples(data_dir)
        elif split == TRUE_DEV_SET:
            examples = processor.get_true_dev_examples(data_dir)
        elif split == TRAIN_SET:
            if task_name == "wsc":
                examples = processor.get_train_examples(data_dir, cloze_eval=args.cloze_eval)
            else:
                examples = processor.get_train_examples(data_dir)
        elif split == UNLABELED_SET:
            examples = processor.get_unlabeled_examples(data_dir)
            for example in examples:
                example.label = processor.get_labels()[0]
        else:
            raise ValueError(f"'split' must be one of {SPLIT_TYPES}, got '{split}' instead")
        if split == TEST_SET:
            self.labeled = False
        else:
            self.labeled = True

        label_distribution = Counter(example.label for example in examples)
        print_rank_0(
            f"Returning {len(examples)} {split} examples with label dist.: {list(label_distribution.items())}")
        self.samples = []
        examples.sort(key=lambda x: x.num_choices)
        if args.cloze_eval:
            pvp = PVPS[task_name](args, tokenizer, processor.get_labels(), args.seq_length, pattern_id=args.pattern_id,
                                  is_multi_token=args.multi_token, max_segment_length=args.segment_length, 
                                  fast_decode=args.fast_decode, split=split)
            for example in examples:
                sample = pvp.encode(example)
                self.samples.append(sample)
            print_rank_0(f"Truncate {pvp.num_truncated} examples")
        else:
            for example in examples:
                sample = processor.encode(example, tokenizer, args)
                self.samples.append(sample)
            print_rank_0(f"Truncate {processor.num_truncated} examples")
        print_rank_0(f"Creating {len(self.samples)} samples")
        self.examples = {example.guid: example for example in examples}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    def __init__(self):
        self.num_truncated = 0

    def output_prediction(self, predictions, examples, output_file):
        with open(output_file, "w") as output:
            for prediction, example in zip(predictions, examples):
                prediction = self.get_labels()[prediction]
                data = {"idx": example.idx, "label": prediction}
                output.write(json.dumps(data) + "\n")

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir, for_train=False) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the unlabeled set."""
        pass

    def get_true_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the true dev set."""
        return self._create_examples(os.path.join(data_dir, "true_dev.jsonl"), "true_dev")

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

    def get_classifier_input(self, example: InputExample, tokenizer):
        return example.text_a, example.text_b

    def encode(self, example: InputExample, tokenizer, args):
        text_a, text_b = self.get_classifier_input(example, tokenizer)
        tokens_a = tokenizer.EncodeAsIds(text_a).tokenization
        tokens_b = tokenizer.EncodeAsIds(text_b).tokenization
        num_special_tokens = num_special_tokens_to_add(tokens_a, tokens_b, None, add_cls=True, add_sep=True,
                                                       add_piece=False)
        if len(tokens_a) + len(tokens_b) + num_special_tokens > args.seq_length:
            self.num_truncated += 1
        data = build_input_from_ids(tokens_a, tokens_b, None, args.seq_length, tokenizer, args=args,
                                    add_cls=True, add_sep=True, add_piece=False)
        ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
        label = 0
        if example.label is not None:
            label = example.label
            label = self.get_labels().index(label)
        if args.pretrained_bert:
            sample = build_sample(ids, label=label, types=types, paddings=paddings,
                                  unique_id=example.guid)
        else:
            sample = build_sample(ids, positions=position_ids, masks=sep, label=label,
                                  unique_id=example.guid)
        return sample


class RteProcessor(DataProcessor):
    """Processor for the RTE data set."""

    def __init__(self):
        super().__init__()
        self.mnli_processor = MnliProcessor()

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["entailment", "not_entailment"]

    def _create_examples(self, path: str, set_type: str, hypothesis_name: str = "hypothesis",
                         premise_name: str = "premise") -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line_idx, line in enumerate(f):
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    try:
                        idx = int(idx)
                    except ValueError:
                        idx = line_idx
                label = example_json.get('label')
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json[premise_name]
                text_b = example_json[hypothesis_name]

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples


class AxGProcessor(RteProcessor):
    """Processor for the AX-G diagnostic data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-g.jsonl"), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-g.jsonl"), "test")


class AxBProcessor(RteProcessor):
    """Processor for the AX-B diagnostic data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-b.jsonl"), "train")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "AX-b.jsonl"), "test")

    def _create_examples(self, path, set_type, hypothesis_name="sentence2", premise_name="sentence1"):
        return super()._create_examples(path, set_type, hypothesis_name, premise_name)


class CbProcessor(RteProcessor):
    """Processor for the CB data set."""

    def get_labels(self):
        return ["entailment", "contradiction", "neutral"]


class WicProcessor(DataProcessor):
    """Processor for the WiC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["false", "true"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []
        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                if isinstance(idx, str):
                    idx = int(idx)
                label = "true" if example_json.get('label') else "false"
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['sentence1']
                text_b = example_json['sentence2']
                meta = {'word': example_json['word']}
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx, meta=meta)
                examples.append(example)
        return examples

    def get_classifier_input(self, example: InputExample, tokenizer):
        text_a = example.meta['word'] + ': ' + example.text_a
        return text_a, example.text_b


class WscProcessor(DataProcessor):
    """Processor for the WSC data set."""

    def get_train_examples(self, data_dir, cloze_eval=True):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train", cloze_eval=cloze_eval)

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["False", "True"]

    def get_classifier_input(self, example: InputExample, tokenizer):
        target = example.meta['span1_text']
        pronoun_idx = example.meta['span2_index']

        # mark the pronoun with asterisks
        words_a = example.text_a.split()
        words_a[pronoun_idx] = '*' + words_a[pronoun_idx] + '*'
        text_a = ' '.join(words_a)
        text_b = target
        return text_a, text_b

    @staticmethod
    def _create_examples(path: str, set_type: str, cloze_eval=True) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']) if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['text']
                meta = {
                    'span1_text': example_json['target']['span1_text'],
                    'span2_text': example_json['target']['span2_text'],
                    'span1_index': example_json['target']['span1_index'],
                    'span2_index': example_json['target']['span2_index']
                }
                if 'candidates' in example_json:
                    candidates = [cand['text'] for cand in example_json['candidates']]
                    # candidates = list(set(candidates))
                    filtered = []
                    for i, cand in enumerate(candidates):
                        if not cand in candidates[:i]:
                            filtered.append(cand)
                    candidates = filtered

                # the indices in the dataset are wrong for some examples, so we manually fix them
                span1_index, span1_text = meta['span1_index'], meta['span1_text']
                span2_index, span2_text = meta['span2_index'], meta['span2_text']
                words_a = text_a.split()
                words_a_lower = text_a.lower().split()
                words_span1_text = span1_text.lower().split()
                span1_len = len(words_span1_text)

                if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                    for offset in [-1, +1]:
                        if words_a_lower[span1_index + offset:span1_index + span1_len + offset] == words_span1_text:
                            span1_index += offset

                # if words_a_lower[span1_index:span1_index + span1_len] != words_span1_text:
                #     print_rank_0(f"Got '{words_a_lower[span1_index:span1_index + span1_len]}' but expected "
                #                  f"'{words_span1_text}' at index {span1_index} for '{words_a}'")

                if words_a[span2_index] != span2_text:
                    for offset in [-1, +1]:
                        if words_a[span2_index + offset] == span2_text:
                            span2_index += offset

                    if words_a[span2_index] != span2_text and words_a[span2_index].startswith(span2_text):
                        words_a = words_a[:span2_index] \
                                  + [words_a[span2_index][:len(span2_text)], words_a[span2_index][len(span2_text):]] \
                                  + words_a[span2_index + 1:]

                assert words_a[span2_index] == span2_text, \
                    f"Got '{words_a[span2_index]}' but expected '{span2_text}' at index {span2_index} for '{words_a}'"

                text_a = ' '.join(words_a)
                meta['span1_index'], meta['span2_index'] = span1_index, span2_index

                if cloze_eval and set_type == 'train' and label != 'True':
                    continue
                if set_type == 'train' and 'candidates' in example_json and len(candidates) > 9:
                    for i in range(0, len(candidates), 9):
                        _meta = copy.deepcopy(meta)
                        _meta['candidates'] = candidates[i:i+9]
                        if len(_meta['candidates']) < 9:
                            _meta['candidates'] += candidates[:9-len(_meta['candidates'])]
                        example = InputExample(guid=guid, text_a=text_a, label=label, meta=_meta, idx=idx)
                        examples.append(example)
                else:
                    if 'candidates' in example_json:
                        meta['candidates'] = candidates
                    example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                    examples.append(example)

        return examples


class BoolQProcessor(DataProcessor):
    """Processor for the BoolQ data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["false", "true"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                idx = example_json['idx']
                label = str(example_json['label']).lower() if 'label' in example_json else None
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['passage']
                text_b = example_json['question']
                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, idx=idx)
                examples.append(example)

        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return [0, 1]

    def encode(self, example: InputExample, tokenizer, args):
        if args.pretrained_bert:
            ids_list, types_list, paddings_list = [], [], []
        else:
            ids_list, positions_list, sep_list = [], [], []
        question = example.meta['question']
        joiner = 'because' if question == 'cause' else 'so'
        text_a = example.text_a + " " + joiner
        tokens_a = tokenizer.EncodeAsIds(text_a).tokenization
        for choice in [example.meta["choice1"], example.meta["choice2"]]:
            tokens_b = tokenizer.EncodeAsIds(choice).tokenization
            num_special_tokens = num_special_tokens_to_add(tokens_a, tokens_b, None, add_cls=True, add_sep=True,
                                                           add_piece=False)
            if len(tokens_a) + len(tokens_b) + num_special_tokens > args.seq_length:
                self.num_truncated += 1
            data = build_input_from_ids(tokens_a, tokens_b, None, args.seq_length, tokenizer,
                                        add_cls=True, add_sep=True, add_piece=False)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            if args.pretrained_bert:
                ids_list.append(ids)
                types_list.append(types)
                paddings_list.append(paddings)
            else:
                ids_list.append(ids)
                positions_list.append(position_ids)
                sep_list.append(sep)
        label = 0
        if example.label is not None:
            label = example.label
            label = self.get_labels().index(label)
        if args.pretrained_bert:
            sample = build_sample(ids_list, label=label, types=types_list, paddings=paddings_list,
                                  unique_id=example.guid)
        else:
            sample = build_sample(ids_list, positions=positions_list, masks=sep_list, label=label,
                                  unique_id=example.guid)
        return sample

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = example_json['label'] if 'label' in example_json else None
                idx = example_json['idx']
                guid = "%s-%s" % (set_type, idx)
                text_a = example_json['premise']
                meta = {
                    'choice1': example_json['choice1'],
                    'choice2': example_json['choice2'],
                    'question': example_json['question']
                }
                example = InputExample(guid=guid, text_a=text_a, label=label, meta=meta, idx=idx)
                examples.append(example)

        if set_type == 'train' or set_type == 'unlabeled':
            mirror_examples = []
            for ex in examples:
                label = 1 if ex.label == 0 else 0
                meta = {
                    'choice1': ex.meta['choice2'],
                    'choice2': ex.meta['choice1'],
                    'question': ex.meta['question']
                }
                mirror_example = InputExample(guid=ex.guid + 'm', text_a=ex.text_a, label=label, meta=meta)
                mirror_examples.append(mirror_example)
            examples += mirror_examples
            print_rank_0(f"Added {len(mirror_examples)} mirror examples, total size is {len(examples)}...")
        return examples


class MultiRcProcessor(DataProcessor):
    """Processor for the MultiRC data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return [0, 1]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)

                passage_idx = example_json['idx']
                text = example_json['passage']['text']
                questions = example_json['passage']['questions']
                for question_json in questions:
                    question = question_json["question"]
                    question_idx = question_json['idx']
                    answers = question_json["answers"]
                    for answer_json in answers:
                        label = answer_json["label"] if 'label' in answer_json else None
                        answer_idx = answer_json["idx"]
                        guid = f'{set_type}-p{passage_idx}-q{question_idx}-a{answer_idx}'
                        meta = {
                            'passage_idx': passage_idx,
                            'question_idx': question_idx,
                            'answer_idx': answer_idx,
                            'answer': answer_json["text"]
                        }
                        idx = [passage_idx, question_idx, answer_idx]
                        example = InputExample(guid=guid, text_a=text, text_b=question, label=label, meta=meta, idx=idx)
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        print_rank_0(
            f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
            f"distribution {list(label_distribution.items())}")
        return examples

    def output_prediction(self, predictions, examples, output_file):
        with open(output_file, "w") as output:
            passage_dict = defaultdict(list)
            for prediction, example in zip(predictions, examples):
                passage_dict[example.meta["passage_idx"]].append((prediction, example))
            for passage_idx, data in passage_dict.items():
                question_dict = defaultdict(list)
                passage_data = {"idx": passage_idx, "passage": {"questions": []}}
                for prediction, example in data:
                    question_dict[example.meta["question_idx"]].append((prediction, example))
                for question_idx, data in question_dict.items():
                    question_data = {"idx": question_idx, "answers": []}
                    for prediction, example in data:
                        prediction = self.get_labels()[prediction]
                        question_data["answers"].append({"idx": example.meta["answer_idx"], "label": prediction})
                    passage_data["passage"]["questions"].append(question_data)
                output.write(json.dumps(passage_data) + "\n")

    def get_classifier_input(self, example: InputExample, tokenizer):
        text_a = example.text_a
        text_b = ' '.join([example.text_b, "answer:", example.meta['answer']])
        return text_a, text_b


class RecordProcessor(DataProcessor):
    """Processor for the ReCoRD data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev", for_train=for_train)

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")

    def get_unlabeled_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "unlabeled.jsonl"), "unlabeled")

    def get_labels(self):
        return ["0", "1"]

    def output_prediction(self, predictions, examples, output_file):
        with open(output_file, "w") as output:
            for prediction, example in zip(predictions, examples):
                prediction = example.meta["candidates"][prediction]
                data = {"idx": example.idx, "label": prediction}
                output.write(json.dumps(data) + "\n")

    def encode(self, example: InputExample, tokenizer, args):
        if args.pretrained_bert:
            ids_list, types_list, paddings_list = [], [], []
        else:
            ids_list, positions_list, sep_list = [], [], []
        tokens_a = tokenizer.EncodeAsIds(example.text_a).tokenization
        tokens_b = tokenizer.EncodeAsIds(example.text_b).tokenization if example.text_b else None
        for answer in example.meta["candidates"]:
            answer_ids = tokenizer.EncodeAsIds(answer).tokenization
            total_length = len(tokens_a) + len(tokens_b) + len(answer_ids)
            total_length += num_special_tokens_to_add(tokens_a, tokens_b + answer_ids, None, add_cls=True, add_sep=True,
                                                      add_piece=False)
            if total_length > args.seq_length:
                self.num_truncated += 1
            data = build_input_from_ids(tokens_a, tokens_b + answer_ids, None, args.seq_length, tokenizer,
                                        add_cls=True, add_sep=True, add_piece=False)
            ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
            if args.pretrained_bert:
                ids_list.append(ids)
                types_list.append(types)
                paddings_list.append(paddings)
            else:
                ids_list.append(ids)
                positions_list.append(position_ids)
                sep_list.append(sep)
        label = example.label
        label = self.get_labels().index(label)
        if args.pretrained_bert:
            sample = build_sample(ids_list, label=label, types=types_list, paddings=paddings_list,
                                  unique_id=example.guid)
        else:
            sample = build_sample(ids_list, positions=positions_list, masks=sep_list, label=label,
                                  unique_id=example.guid)
        return sample

    @staticmethod
    def _create_examples(path, set_type, seed=42, max_train_candidates_per_question: int = 10, for_train=False) -> List[
        InputExample]:
        examples = []

        entity_shuffler = random.Random(seed)

        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                example_json = json.loads(line)

                idx = example_json['idx']
                text = example_json['passage']['text']
                entities = set()

                for entity_json in example_json['passage']['entities']:
                    start = entity_json['start']
                    end = entity_json['end']
                    entity = text[start:end + 1]
                    entities.add(entity)

                entities = list(entities)

                text = text.replace("@highlight\n", "- ")  # we follow the GPT-3 paper wrt @highlight annotations
                questions = example_json['qas']

                for question_json in questions:
                    question = question_json['query']
                    question_idx = question_json['idx']
                    answers = set()

                    for answer_json in question_json.get('answers', []):
                        answer = answer_json['text']
                        answers.add(answer)

                    answers = list(answers)

                    if set_type == 'train' or for_train:
                        # create a single example per *correct* answer
                        for answer_idx, answer in enumerate(answers):
                            candidates = [ent for ent in entities if ent not in answers]
                            if len(candidates) > max_train_candidates_per_question - 1:
                                entity_shuffler.shuffle(candidates)
                                candidates = candidates[:max_train_candidates_per_question - 1]

                            guid = f'{set_type}-p{idx}-q{question_idx}-a{answer_idx}'
                            meta = {
                                'passage_idx': idx,
                                'question_idx': question_idx,
                                'candidates': [answer] + candidates,
                                'answers': [answer]
                            }
                            ex_idx = [idx, question_idx, answer_idx]
                            example = InputExample(guid=guid, text_a=text, text_b=question, label="0", meta=meta,
                                                   idx=ex_idx, num_choices=len(candidates) + 1)
                            examples.append(example)

                    else:
                        # create just one example with *all* correct answers and *all* answer candidates
                        guid = f'{set_type}-p{idx}-q{question_idx}'
                        meta = {
                            'passage_idx': idx,
                            'question_idx': question_idx,
                            'candidates': entities,
                            'answers': answers
                        }
                        example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta,
                                               idx=question_idx, num_choices=len(entities))
                        examples.append(example)

        question_indices = list(set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        print_rank_0(
            f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
            f"distribution {list(label_distribution.items())}")
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(MnliProcessor._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["contradiction", "entailment", "neutral"]

    @staticmethod
    def _create_examples(lines: List[List[str]], set_type: str) -> List[InputExample]:
        examples = []

        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]

            example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            examples.append(example)

        return examples

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()


class AgnewsProcessor(DataProcessor):
    """Processor for the AG news data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2", "3", "4"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class YahooAnswersProcessor(DataProcessor):
    """Processor for the Yahoo Answers data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, question_title, question_body, answer = row
                guid = "%s-%s" % (set_type, idx)
                text_a = ' '.join([question_title.replace('\\n', ' ').replace('\\', ' '),
                                   question_body.replace('\\n', ' ').replace('\\', ' ')])
                text_b = answer.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


class YelpPolarityProcessor(DataProcessor):
    """Processor for the YELP binary classification set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "test.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["1", "2"]

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, body = row
                guid = "%s-%s" % (set_type, idx)
                text_a = body.replace('\\n', ' ').replace('\\', ' ')

                example = InputExample(guid=guid, text_a=text_a, label=label)
                examples.append(example)

        return examples


class YelpFullProcessor(YelpPolarityProcessor):
    """Processor for the YELP full classification set."""

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_labels(self):
        return ["1", "2", "3", "4", "5"]


class XStanceProcessor(DataProcessor):
    """Processor for the X-Stance data set."""

    def __init__(self, language: str = None):
        super().__init__()
        if language is not None:
            assert language in ['de', 'fr']
        self.language = language

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"))

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"))

    def get_test_examples(self, data_dir) -> List[InputExample]:
        raise NotImplementedError()

    def get_unlabeled_examples(self, data_dir) -> List[InputExample]:
        return self.get_train_examples(data_dir)

    def get_labels(self):
        return ["FAVOR", "AGAINST"]

    def _create_examples(self, path: str) -> List[InputExample]:
        examples = []

        with open(path, encoding='utf8') as f:
            for line in f:
                example_json = json.loads(line)
                label = example_json['label']
                id_ = example_json['id']
                text_a = example_json['question']
                text_b = example_json['comment']
                language = example_json['language']

                if self.language is not None and language != self.language:
                    continue

                example = InputExample(guid=id_, text_a=text_a, text_b=text_b, label=label)
                examples.append(example)

        return examples


CLASSIFICATION_DATASETS = {"wic", "rte", "cb", "boolq", "multirc", "wsc"}
MULTI_CHOICE_DATASETS = {"copa", "record"}

PROCESSORS = {
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "agnews": AgnewsProcessor,
    "yahoo": YahooAnswersProcessor,
    "yelp-polarity": YelpPolarityProcessor,
    "yelp-full": YelpFullProcessor,
    "xstance-de": lambda: XStanceProcessor("de"),
    "xstance-fr": lambda: XStanceProcessor("fr"),
    "xstance": XStanceProcessor,
    "wic": WicProcessor,
    "rte": RteProcessor,
    "cb": CbProcessor,
    "wsc": WscProcessor,
    "boolq": BoolQProcessor,
    "copa": CopaProcessor,
    "multirc": MultiRcProcessor,
    "record": RecordProcessor,
    "ax-g": AxGProcessor,
    "ax-b": AxBProcessor,
}  # type: Dict[str,Callable[[],DataProcessor]]
