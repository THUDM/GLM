import json
import pickle
import torch
from utils.misc import invert_dict


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.source_ids, self.source_mask, self.extra_ids, self.target = inputs

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.source_ids[index])
        source_mask = torch.LongTensor(self.source_mask[index])
        extra_ids = torch.LongTensor([self.extra_ids[index]])
        target = self.target[index]
        return source_ids, source_mask, extra_ids, target

    def __len__(self):
        return len(self.source_ids)


def load_vocab(path):
    try:
        vocab = json.load(open(path))
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    except:
        vocab = None
    return vocab


def collate(batch):
    batch = list(zip(*batch))
    source_ids = torch.stack(batch[0])
    source_mask = torch.stack(batch[1])
    if batch[-1][0] is None:
        target, answer_ids = None, None
    else:
        target = list(batch[3])
        answer_ids = torch.cat(batch[2])
    return source_ids, source_mask, answer_ids, target


def prepare_dataset(vocab_json, question_pt, question_json, training=False, **kwargs):
    vocab = load_vocab(vocab_json)
    inputs = []
    input_len = 3
    with open(question_pt, 'rb') as f:
        for _ in range(input_len):
            inputs.append(pickle.load(f))
    inputs.append(json.load(open(question_json, "r")))
    dataset = Dataset(inputs)
    return dataset, vocab


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, question_json, batch_size, training=False, **kwargs):
        dataset, vocab = prepare_dataset(vocab_json, question_pt, question_json, training)
        super().__init__(dataset, batch_size=batch_size, shuffle=training, collate_fn=collate)
        self.vocab = vocab


class DistributedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, vocab, batch_size, sampler, **kwargs):
        self.vocab = vocab
        self.sampler = sampler
        super().__init__(dataset, batch_size=batch_size, sampler=self.sampler, pin_memory=True, collate_fn=collate)
