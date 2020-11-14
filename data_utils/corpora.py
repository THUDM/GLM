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
from .datasets import json_dataset, csv_dataset
import os
import json
import random
# import ray
import tqdm
from torch.utils import data
from .lazy_loader import lazy_array_loader


class webtext(json_dataset):
    """
    dataset for webtext with arguments configured for convenience

    command line usage: `--train-data webtext`
    """
    PATH = 'data/webtext/data.json'
    assert_str = "make sure to set PATH for webtext data_utils/corpora.py"

    def __init__(self, **kwargs):
        assert os.path.exists(webtext.PATH), \
            webtext.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(webtext, self).__init__(webtext.PATH, **kwargs)


class ChineseDataset(data.Dataset):
    def __init__(self, prompt_loader, text_loader, **kwargs):
        self.prompts = prompt_loader
        self.texts = text_loader
        if isinstance(self.prompts, lazy_array_loader) and isinstance(self.texts, lazy_array_loader):
            self.prompt_lens = self.prompts.lens
            self.text_lens = self.texts.lens
            self.is_lazy = True

    def process_line(self, data):
        raise NotImplementedError

    def __getitem__(self, index):
        return {"prompt": self.prompts[index], "text": self.texts[index]}

    def __len__(self):
        return len(self.prompts)


# @ray.remote
def read_file(path, reader, tokenizer, tokenize):
    prompts, texts = [], []
    with open(path) as file:
        for row in file:
            data = json.loads(row)
            prompt, text = reader.process_line(data, tokenizer, tokenize)
            prompts += prompt
            texts += text
    return prompts, texts


class DataReader:
    PATH = None
    assert_str = None

    def __init__(self, prompt_writer, text_writer, tokenizer=None, tokenize=False, **kwargs):
        assert os.path.exists(self.PATH), self.assert_str
        # ray.init(num_cpus=90)
        self.tokenizer = tokenizer
        self.tokenize = tokenize
        if os.path.isdir(self.PATH):
            paths = [entry.path for entry in os.scandir(self.PATH) if not entry.is_dir() and not entry.name.endswith("bz2")]
        else:
            paths = [self.PATH]
        results = []
        for path in paths:
            results.append(read_file(path, type(self), tokenizer, tokenize))
        for result in tqdm.tqdm(results):
            prompts, texts = result
            prompt_writer.write(prompts)
            text_writer.write(texts)

    @staticmethod
    def get_token_count(contents):
        return sum(map(len, contents))

    @staticmethod
    def process_sample(prompt, text, tokenizer, tokenize):
        if isinstance(prompt, str) and tokenize:
            prompt = tokenizer.EncodeAsIds(prompt).tokenization if prompt else []
        if isinstance(text, str) and tokenize:
            text = tokenizer.EncodeAsIds(text).tokenization if text else []
        return prompt, text

    @staticmethod
    def trim_field(content, max_length):
        if len(content) > max_length:
            content = content[:max_length]
            content += "......"
        return content

    @staticmethod
    def process_line(data, tokenizer, tokenize):
        raise NotImplementedError


class zhihu(DataReader):
    PATH = "/root/data/zhihu/zhihu"
    assert_str = "make sure to set PATH for zhihu data_utils/corpora.py"
    qtitle_prefix = "问题："
    qcontent_prefix = "问题描述："
    user_prefix = "回答用户："
    answer_prefix = " 回答："

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        prompts, texts = [], []
        ans_length = len(data.get("ans-content", ""))
        ans_up = data.get("ans-up-num", "")
        ans_up = int(ans_up) if ans_up else 0
        if ans_length > 100 or ans_up > 1000:
            qtitle = data["q_title"]
            qcontent = data["q-content"]
            if qcontent is None:
                qcontent = ""
            qcontent = cls.trim_field(qcontent, max_length=100)
            user = data.get("user-signature", "")
            prompt = cls.qtitle_prefix + qtitle + cls.qcontent_prefix + qcontent + cls.user_prefix + user + cls.answer_prefix
            text = data["ans-content"]
            prompt, text = cls.process_sample(prompt, text, tokenizer, tokenize)
            prompts.append(prompt)
            texts.append(text)
        return prompts, texts


class zhidao(DataReader):
    PATH = "/root/data/baike_zhidao/zhidao"
    assert_str = "make sure to set PATH for zhidao data_utils/corpora.py"
    qtitle_prefix = "问题："
    qcontent_prefix = "问题描述："
    answer_prefix = "回答："

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        if "title" not in data:
            return [], []
        prompts, texts = [], []
        qtitle = data["title"]
        qcontent = data.get("content", "")
        qcontent = cls.trim_field(qcontent, max_length=100)
        prompt = cls.qtitle_prefix + qtitle + cls.qcontent_prefix + qcontent + cls.answer_prefix
        if "best_answer" in data:
            text = data["best_answer"]["content"]
            if len(text) > 10:
                p, t = cls.process_sample(prompt, text, tokenizer, tokenize)
                prompts.append(p)
                texts.append(t)
        for answer in data.get("other_answers", []):
            text = answer["content"]
            if len(text) > 100:
                p, t = cls.process_sample(prompt, text, tokenizer, tokenize)
                prompts.append(p)
                texts.append(t)
        return prompts, texts


class baike(DataReader):
    PATH = "/root/data/baike_zhidao/baike"
    assert_str = "make sure to set PATH for baike data_utils/corpora.py"

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        prompts, texts = [], []
        text = data.get("title", "") + data.get("abstract", "") + data.get("content", "")
        if text:
            p, t = cls.process_sample("", text, tokenizer, tokenize)
            prompts.append(p)
            texts.append(t)
        return prompts, texts


class wikipedia(DataReader):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    # PATH = '/dataset/data/wiki.txt'
    PATH = 'data/wiki.txt'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"

    @classmethod
    def process_line(cls, data, tokenizer, tokenize):
        text = data['text']
        prompt, text = cls.process_sample("", text)
        return [prompt], [text]



NAMED_CORPORA = {
    'wikipedia': wikipedia,
    'webtext': webtext,
    "zhihu": zhihu,
    "zhidao": zhidao,
    "baike": baike
}
