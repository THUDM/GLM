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

"""several datasets with preset arguments"""
import glob
import gzip
import zlib
from abc import ABC

import os
import json
import random
import tqdm
from multiprocessing import Queue, Process
from queue import Empty
from collections import defaultdict
from torch.utils import data
from .lazy_loader import LazyLoader

NUM_PROCESSES = 100


def tqdm_print(s):
    tqdm.tqdm.write(s)


def punctuation_standardization(string: str):
    punctuation_dict = {"\u201c": "\"", "\u201d": "\"", "\u2019": "'", "\u2018": "'", "\u2013": "-"}
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string


class PromptDataset(data.Dataset):
    def __init__(self, prompt_loader, text_loader, tokenizer=None, to_tokenize=False, name="", **kwargs):
        self._name = name
        self.prompts = prompt_loader
        self.texts = text_loader
        self._tokenizer = tokenizer
        self.to_tokenize = to_tokenize
        if (self.prompts is None or isinstance(self.prompts, LazyLoader)) and isinstance(self.texts, LazyLoader):
            self.prompt_lens = self.prompts.lens if self.prompts else None
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def get_text_len(self, idx):
        prompt_length = self.prompt_lens[idx] if self.prompt_lens else 0
        return prompt_length + self.text_lens[idx]

    @property
    def name(self):
        return self._name

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer):
        self._tokenizer = tokenizer

    def __getitem__(self, index):
        prompt = self.prompts[index] if self.prompts else ""
        text = self.texts[index]
        if self.to_tokenize:
            prompt = self.tokenizer.EncodeAsIds(prompt).tokenization if prompt else []
            text = self.tokenizer.EncodeAsIds(text).tokenization
        return {"tokens": prompt + text, "loss_masks": [0] * len(prompt) + [1] * len(text)}

    def __len__(self):
        return len(self.texts)


class DataReader:
    PATH = None
    assert_str = None
    reserve_punct = False

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        raise NotImplementedError

    def print_info(self, info):
        pass

    @classmethod
    def path(cls):
        return cls.PATH

    def __init__(self, writer, tokenizer=None, tokenize=False, **kwargs):
        assert os.path.exists(self.PATH), self.assert_str
        tqdm_print(f"Creating dataset from {self.PATH} with pre-tokenization {tokenize}")
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        self.writer = writer

    def read_input_to_queue(self, task_queue):
        raise NotImplementedError

    @staticmethod
    def write_result(data, writer):
        writer.write(data)

    def process(self):
        task_queue, done_queue, info_queue = Queue(maxsize=10000000), Queue(maxsize=10000000), Queue()
        processes = []
        for i in range(NUM_PROCESSES):
            process = Process(target=self.tokenize_worker,
                              args=(task_queue, done_queue, info_queue, self.tokenizer, self.tokenize))
            process.start()
            processes.append(process)

        def read_func():
            self.read_input_to_queue(task_queue)
            tqdm_print("Read input complete")
            for _ in range(len(processes)):
                task_queue.put('STOP')

        process = Process(target=read_func)
        process.start()
        count = len(processes)
        progress_bar = tqdm.tqdm()
        while True:
            data = done_queue.get()
            if data == 'COMPLETE':
                count -= 1
                if count == 0:
                    break
            else:
                self.write_result(data, self.writer)
                progress_bar.update()
        progress_bar.close()
        self.print_info(info_queue)

    @staticmethod
    def get_token_count(contents):
        return sum(map(len, contents))

    @classmethod
    def process_sample(cls, text, tokenizer, tokenize):
        if isinstance(text, str) and tokenize:
            if not cls.reserve_punct:
                text = punctuation_standardization(text)
            text = tokenizer.EncodeAsIds(text).tokenization if text else []
        return text

    @staticmethod
    def trim_field(content, max_length):
        if len(content) > max_length:
            content = content[:max_length]
            content += "......"
        return content


class LineReader(DataReader, ABC):
    is_json = True  # Take as jsonline file by default

    def get_paths(self):
        if os.path.isdir(self.PATH):
            paths = [entry.path for entry in os.scandir(self.PATH) if
                     not entry.is_dir() and not entry.name.endswith("bz2")]
        else:
            paths = [self.PATH]
        return paths

    def read_input_to_queue(self, task_queue):
        for path in self.get_paths():
            tqdm_print(f"Start reading {path}")
            with open(path) as file:
                for row in file:
                    task_queue.put(row)

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        for row in iter(input.get, 'STOP'):
            row = row.rstrip()
            if row:
                if self.is_json:
                    row = json.loads(row)
                data = self.process_line(row, tokenizer, tokenize)
                for item in data:
                    output.put(item)
        output.put("COMPLETE")

    def process_line(self, data, tokenizer, tokenize):
        if data:
            prompt, text = "", data
            prompt, text = self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
            return [{"prompt": prompt, "text": text}]
        else:
            return []


class FileReader(DataReader):

    def get_paths(self):
        raise NotImplementedError

    def read_input_to_queue(self, task_queue):
        for path in self.get_paths():
            task_queue.put(path)

    def process_file(self, path, output, tokenizer, tokenize):
        raise NotImplementedError

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        for path in iter(input.get, 'STOP'):
            tqdm_print(f"Start reading {path}")
            self.process_file(path, output, tokenizer=tokenizer, tokenize=tokenize)
        output.put("COMPLETE")


def create_multilingual_reader(language=None):
    class MultilingualReader(FileReader):
        PATH = "/dataset/fd5061f6/english_data/xiaoice"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @classmethod
        def path(cls):
            if language is None:
                return cls.PATH
            else:
                return os.path.join(cls.PATH + ".lazy", language)

        @classmethod
        def get_languages(cls):
            languages = set()
            pattern = os.path.join(cls.PATH, "**", "*.json.gz")
            for path in glob.glob(pattern, recursive=True):
                filename = os.path.basename(path)
                lang = filename.split('_')[0]
                languages.add(lang)
            return languages

        def get_paths(self):
            if language is not None:
                pattern = f"{language}_*.json.gz"
            else:
                pattern = "*.json.gz"
            found_files = set()
            for path in glob.glob(os.path.join(self.PATH, "**", pattern), recursive=True):
                filename = os.path.basename(path)
                if filename not in found_files:
                    yield path
                    found_files.add(filename)

        def process_file(self, path, output, tokenizer, tokenize):
            lines = []
            try:
                with gzip.open(path, mode='rt') as file:
                    for row in file:
                        row = row.rstrip()
                        if row:
                            lines.append(row)
                        else:
                            if lines:
                                text = "\n".join(lines)
                                prompt = self.process_sample("", tokenizer, tokenize)
                                text = self.process_sample(text, tokenizer, tokenize)
                                output.put({"prompt": prompt, "text": text})
                            lines = []
            except (zlib.error, gzip.BadGzipFile) as e:
                tqdm_print(f"Compression error when reading {path}")

    return MultilingualReader


class zhihu(LineReader):
    PATH = "/root/data/zhihu/zhihu"
    reserve_punct = True
    assert_str = "make sure to set PATH for zhihu data_utils/corpora.py"
    qtitle_prefix = "问题："
    qcontent_prefix = "问题描述："
    user_prefix = "回答用户："
    answer_prefix = " 回答："

    def process_line(self, data, tokenizer, tokenize):
        outputs = []
        ans_length = len(data.get("ans-content", ""))
        ans_up = data.get("ans-up-num", "")
        ans_up = int(ans_up) if ans_up else 0
        if ans_length > 100 or ans_up > 1000:
            qtitle = data["q_title"]
            qcontent = data["q-content"]
            if qcontent is None:
                qcontent = ""
            qcontent = self.trim_field(qcontent, max_length=100)
            user = data.get("user-signature", "")
            prompt = self.qtitle_prefix + qtitle + self.qcontent_prefix + qcontent + self.user_prefix + user + self.answer_prefix
            text = data["ans-content"]
            prompt, text = self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
            outputs.append({"prompt": prompt, "text": text})
        return outputs


class zhidao(LineReader):
    PATH = "/root/data/zhidao/zhidao"
    reserve_punct = True
    assert_str = "make sure to set PATH for zhidao data_utils/corpora.py"
    qtitle_prefix = "问题："
    qcontent_prefix = "问题描述："
    answer_prefix = "回答："

    def process_line(self, data, tokenizer, tokenize):
        if "title" not in data:
            return [], []
        outputs = []
        qtitle = data["title"]
        qcontent = data.get("content", "")
        qcontent = self.trim_field(qcontent, max_length=100)
        prompt = self.qtitle_prefix + qtitle + self.qcontent_prefix + qcontent + self.answer_prefix
        prompt = self.process_sample(prompt, tokenizer, tokenize)
        if "best_answer" in data:
            text = data["best_answer"]["content"]
            if len(text) > 10:
                text = self.process_sample(text, tokenizer, tokenize)
                outputs.append({"prompt": prompt, "text": text})
        for answer in data.get("other_answers", []):
            text = answer["content"]
            if len(text) > 100:
                text = self.process_sample(text, tokenizer, tokenize)
                outputs.append({"prompt": prompt, "text": text})
        return outputs


class baike(LineReader):
    PATH = "/root/data/baike/baike"
    reserve_punct = True
    assert_str = "make sure to set PATH for baike data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        outputs = []
        text = data.get("title", "") + data.get("abstract", "") + data.get("content", "")
        if text:
            p, t = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
            outputs.append({"prompt": p, "text": t})
        return outputs


class wikipedia(LineReader):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    is_json = False
    PATH = '/root/data/bert_data/wiki.txt'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"


class TestDataset(LineReader):
    PATH = '/root/data/test.json'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        prompt, text = data['prompt'], data['text']
        prompt, text = self.process_sample(prompt, tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
        return [{"prompt": prompt, "text": text}]


class OpenWebText(LineReader):
    PATH = '/dataset/fd5061f6/english_data/openwebtext2'
    assert_str = "make sure to set PATH for openwebtext data_utils/corpora.py"

    def __init__(self, *args, **kwargs):
        import fasttext
        super().__init__(*args, **kwargs)
        self.model = fasttext.load_model('/dataset/fd5061f6/english_data/lid.176.bin')
        print("Load language detection model")

    def process_line(self, data, tokenizer, tokenize):
        text = data['text']
        if len(text) > 100:
            lang = self.model.predict(text.replace('\n', ''))[0][0]
            if lang == '__label__en':
                prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
                return [{"prompt": prompt, "text": text}]
        return []


class CCNews(LineReader):
    # PATH = "/dataset/fd5061f6/english_data/cc_news.json"  # For GPT2 tokenizer
    PATH = "/dataset/fd5061f6/english_data/cc_news_new.json"  # For RoBERTa tokenizer
    assert_str = "make sure to set PATH for cc-news data_utils/corpora.py"

    def process_line(self, data, tokenizer, tokenize):
        text = ""
        title = data.get("title", None)
        description = data.get("description", None)
        maintext = data.get("maintext", None)
        if title:
            text += title.strip() + " "
        if description and (not maintext or not maintext.startswith(description)):
            text += description.strip() + " "
        if maintext:
            text += maintext
        if len(text) > 100:
            prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer, tokenize)
            return [{"prompt": prompt, "text": text}]
        else:
            return []


class BertData(LineReader):
    is_json = False
    PATH = '/dataset/fd5061f6/english_data/wikibook'


class Pile(LineReader):
    is_json = True
    PATH = "/dataset/fd5061f6/english_data/pile/train"
    filtered_sources = ["Github", "StackExchange", "DM Mathematics", "Ubuntu IRC", "EuroParl", "YoutubeSubtitles",
                        "Enron Emails"]
    downsample_sources = {"PubMed Central": 0.3, "ArXiv": 0.3, "FreeLaw": 0.3}

    def print_info(self, info):
        total_dict = defaultdict(int)
        while True:
            try:
                source_dict = info.get(block=False)
                for source, length in source_dict.items():
                    total_dict[source] += length
            except Empty:
                break
        tqdm_print(total_dict)

    def tokenize_worker(self, input, output, info, tokenizer, tokenize):
        source_dict = defaultdict(int)
        for row in iter(input.get, 'STOP'):
            row = row.rstrip()
            if row:
                if self.is_json:
                    row = json.loads(row)
                data, source = self.process_line(row, tokenizer, tokenize)
                length = 0
                for item in data:
                    length += len(item['text'])
                    output.put(item)
                if source:
                    source_dict[source] += length
        output.put("COMPLETE")
        info.put(source_dict)

    def process_line(self, data, tokenizer, tokenize):
        source = data["meta"].get("pile_set_name", None)
        text = data.get("text", None)
        if source and text:
            if source in self.filtered_sources:
                return [], [], None
            elif source in self.downsample_sources and random.random() > self.downsample_sources[source]:
                return [], [], None
            else:
                prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                                 tokenize)
                return [{"prompt": prompt, "text": text}], source
        else:
            return [], None


class Stories(LineReader):
    is_json = True
    PATH = "/dataset/fd5061f6/english_data/stories_31G.jsonl"

    def process_line(self, data, tokenizer, tokenize):
        text = data.get("text", None)
        if text:
            prompt, text = self.process_sample("", tokenizer, tokenize), self.process_sample(text, tokenizer,
                                                                                             tokenize)
            return [{"prompt": prompt, "text": text}]
        else:
            return []


class BertBaseData(BertData):
    PATH = '/root/data/formatted_one_article_per_line'


class BertLargeData(BertData):
    PATH = '/dataset/c07bd62b/cognitive/zhengxiao/formatted_one_article_per_line_large'


NAMED_CORPORA = {
    'wikipedia': wikipedia,
    'openwebtext': OpenWebText,
    "zhihu": zhihu,
    "zhidao": zhidao,
    "baike": baike,
    "test": TestDataset,
    'wikibook': BertData,
    "bert-base": BertBaseData,
    "bert-large": BertLargeData,
    'cc-news': CCNews,
    'pile': Pile,
    'stories': Stories
}


def get_corpora_class(corpus_name):
    if corpus_name.startswith('multilingual'):
        if corpus_name == 'multilingual':
            return create_multilingual_reader(None)
        else:
            lang = corpus_name.split('-')[1]
            return create_multilingual_reader(language=lang)
    elif corpus_name in NAMED_CORPORA:
        return NAMED_CORPORA[corpus_name]
    else:
        raise NotImplementedError('dataset %s is not supported' % corpus_name)
