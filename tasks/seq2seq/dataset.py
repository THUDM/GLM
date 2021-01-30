import os
import torch
import torch.utils.data
import numpy as np
from tasks.data_utils import InputExample
from tqdm import tqdm
from utils import print_rank_0


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, tokenizer, max_src_length, max_tgt_length):
        if split == "train":
            filename = "train"
        elif split == "dev":
            filename = "dev"
        elif split == "test":
            filename = "test"
        else:
            raise NotImplementedError(split)
        self.dataset_name = split
        source_texts, target_texts = [], []
        with open(os.path.join(data_dir, f"{filename}.source")) as file:
            for line in file:
                source_texts.append(line.strip())
        with open(os.path.join(data_dir, f"{filename}.target")) as file:
            for line in file:
                target_texts.append(line.strip())
        assert len(source_texts) == len(target_texts)
        self.examples, self.samples = {}, []
        num_source_truncated, num_target_truncated = 0, 0
        cls_id = tokenizer.get_command('ENC').Id
        mask_id = tokenizer.get_command('MASK').Id
        pad_id = tokenizer.get_command('pad').Id
        sop_id = tokenizer.get_command('sop').Id
        eop_id = tokenizer.get_command('eop').Id
        for idx, (source_text, target_text) in enumerate(tqdm(zip(source_texts, target_texts))):
            guid = "%s-%s" % (split, idx)
            source_truncated, target_truncated = False, False
            meta = {"ref": tokenizer.DecodeIds(tokenizer.EncodeAsIds(target_text).tokenization)}
            example = InputExample(guid=guid, text_a=source_text, text_b=target_text, meta=meta)
            self.examples[guid] = example
            source_tokens = tokenizer.EncodeAsIds(source_text).tokenization
            source_tokens = [cls_id] + source_tokens
            prompt = tokenizer.EncodeAsIds(" Summary:").tokenization
            prompt = prompt + [mask_id]
            if len(source_tokens) > max_src_length - len(prompt):
                source_tokens = source_tokens[:max_src_length - len(prompt)]
                source_truncated = True
            source_tokens = source_tokens + prompt
            if len(source_tokens) < max_src_length:
                source_tokens = source_tokens + [pad_id] * (max_src_length - len(source_tokens))
            sep = len(source_tokens)
            position_ids = list(range(len(source_tokens)))
            block_position_ids = [0] * len(source_tokens)
            mask_pos = source_tokens.index(mask_id)
            if split == 'train':
                target_tokens = tokenizer.EncodeAsIds(" " + target_text).tokenization
                target_tokens = target_tokens + [eop_id]
                if len(target_tokens) > max_tgt_length:
                    target_tokens = target_tokens[:max_tgt_length]
                    target_truncated = True
                loss_mask = [1] * len(target_tokens)
                if len(target_tokens) < max_tgt_length:
                    loss_mask += [0] * (max_tgt_length - len(target_tokens))
                    target_tokens += [pad_id] * (max_tgt_length - len(target_tokens))
                tokens = source_tokens + [sop_id] + target_tokens[:-1]
                loss_mask = [0] * len(source_tokens) + loss_mask
                target_ids = [0] * len(source_tokens) + target_tokens
                position_ids += [mask_pos] * len(target_tokens)
                block_position_ids += list(range(1, len(target_tokens) + 1))
                position_ids = [position_ids, block_position_ids]
                sample = {'text': np.array(tokens, dtype=np.int64), 'target': np.array(target_ids, dtype=np.int64),
                          'attention_mask': np.array(sep, dtype=np.int64),
                          'loss_mask': np.array(loss_mask, dtype=np.int64),
                          "position_id": np.array(position_ids, dtype=np.int64), "uid": guid}
                self.samples.append(sample)
            else:
                tokens = source_tokens + [sop_id]
                position_ids = position_ids + [mask_pos]
                block_position_ids = block_position_ids + [1]
                position_ids = [position_ids, block_position_ids]
                sample = {'text': np.array(tokens, dtype=np.int64), 'attention_mask': np.array(sep, dtype=np.int64),
                          "position_id": np.array(position_ids, dtype=np.int64), "uid": guid}
                self.samples.append(sample)
            if source_truncated:
                num_source_truncated += 1
            if target_truncated:
                num_target_truncated += 1
        print_rank_0(
            f"Return {len(self.samples)} {split} examples, {num_source_truncated} examples source truncated, {num_target_truncated} examples target truncated")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
