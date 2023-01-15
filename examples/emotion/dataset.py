import torch
from tqdm import tqdm
from datasets import load_dataset
from promptsource.templates import DatasetTemplates


class MultipleChoiceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split, prompt_name, tokenizer):
        super(MultipleChoiceDataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.prompt = DatasetTemplates(self.dataset_name)[prompt_name]
        self.tokenizer = tokenizer

        self.data = []
        if '/' in self.dataset_name:
            iters = load_dataset(self.dataset_name.split('/')[0], self.dataset_name.split('/')[1], split=self.split)
        else:
            iters = load_dataset(self.dataset_name, split=self.split)
        for sample in tqdm(iters):
            self.data.append(dict(zip(
                ['inputs_pretokenized', 'choices_pretokenized', 'label'],
                self.prompting_single_sample(sample)
            )))

    def prompting_single_sample(self, sample):
        inputs_pretokenized, _ = tuple(self.prompt.apply(sample))
        choices_pretokenized = self.prompt.answer_choices.split(' ||| ')
        return inputs_pretokenized + f" {self.tokenizer.mask_token}", choices_pretokenized, sample['label']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
