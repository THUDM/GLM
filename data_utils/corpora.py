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
from torch.utils import data
from .lazy_loader import lazy_array_loader


class wikipedia(json_dataset):
    """
    dataset for wikipedia with arguments configured for convenience

    command line usage: `--train-data wikipedia`
    """
    PATH = 'data/wiki.txt'
    assert_str = "make sure to set PATH for wikipedia data_utils/corpora.py"

    def __init__(self, **kwargs):
        assert os.path.exists(wikipedia.PATH), \
            wikipedia.assert_str
        if not kwargs:
            kwargs = {}
        kwargs['text_key'] = 'text'
        kwargs['loose_json'] = True
        super(wikipedia, self).__init__(wikipedia.PATH, **kwargs)


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


class DataReader:
    PATH = None
    assert_str = None

    def __init__(self, shuffle=False, **kwargs):
        assert os.path.exists(self.PATH), self.assert_str
        self.prompts, self.texts = [], []
        with open(self.PATH) as file:
            for row in file:
                data = json.loads(row)
                self.process_line(data)
        if shuffle:
            shuffle_idx = list(range(len(self.prompts)))
            random.shuffle(shuffle_idx)
            self.prompts = [self.prompts[idx] for idx in shuffle_idx]
            self.texts = [self.texts[idx] for idx in shuffle_idx]

    def process_line(self, data):
        raise NotImplementedError


class zhihu(DataReader):
    PATH = "data/zhihu/data.json"
    assert_str = "make sure to set PATH for zhihu data_utils/corpora.py"
    qtitle_prefix = []
    qcontent_prefix = []
    user_prefix = []
    answer_prefix = []

    def process_line(self, data):
        if data.get("ans-content", []):
            prompt = zhihu.qtitle_prefix + data["q_title"] + zhihu.qcontent_prefix + data[
                "q-content"] + zhihu.user_prefix + data.get("user-signature", []) + zhihu.answer_prefix
            text = data["ans-content"]
            self.prompts.append(prompt)
            self.texts.append(text)


class zhidao(DataReader):
    PATH = "data/zhidao/data.json"
    assert_str = "make sure to set PATH for zhidao data_utils/corpora.py"
    qtitle_prefix = []
    qcontent_prefix = []
    answer_prefix = []

    def process_line(self, data):
        prompt = self.qtitle_prefix + data["title"] + self.qcontent_prefix + data[
            "content"] + self.answer_prefix
        if "best_answer" in data:
            text = data["best_answer"]["content"]
            self.prompts.append(prompt)
            self.texts.append(text)
        for answer in data.get("other_answers", []):
            text = answer["content"]
            self.prompts.append(prompt)
            self.texts.append(text)


class baike(DataReader):
    PATH = "data/baike/data.json"
    assert_str = "make sure to set PATH for baike data_utils/corpora.py"

    def process_line(self, data):
        self.prompts.append([])
        text = data.get("title", []) + data.get("abstract", []) + data.get("content", [])
        if text:
            self.prompts.append([])
            self.texts.append(text)


NAMED_CORPORA = {
    'wikipedia': wikipedia,
    'webtext': webtext,
    "zhihu": zhihu,
    "zhidao": zhidao,
    "baike": baike
}
