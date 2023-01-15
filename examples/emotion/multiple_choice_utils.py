'''
Acknowledgement: Code adapted from Aohan Zeng and Xiao Liu.
'''

import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from scipy.linalg import block_diag


def flatten_labels(compacted_labels):
    batch_size = len(compacted_labels[0])
    num_of_classes = len(compacted_labels)
    return [[compacted_labels[i][idx] for i in range(num_of_classes)] for idx in range(batch_size)]


def build_multiple_choice_sample(tokenizer, context, choices):
    context_id = tokenizer(context)['input_ids']

    division = len(context_id)
    mask_position = context_id.index(tokenizer.mask_token_id)

    token = np.array(context_id, dtype=np.int64)
    attention_mask = [np.ones((division, division), dtype=np.int64)]
    position_id = np.arange(division, dtype=np.int64)
    block_position_id = np.zeros(division, dtype=np.int64)

    choice_target_id = []
    choice_id = []

    for choice_str in choices:
        choice = np.array(tokenizer(choice_str)['input_ids'][1:-1], dtype=np.int64)

        choice_id.append(choice)
        choice_target_id.append(np.arange(len(token), len(token) + len(choice), dtype=np.int64))
        attention_mask.append(np.tril(np.ones((len(choice), len(choice)), dtype=np.int64)))

        token = np.concatenate((token, [tokenizer.sop_token_id], choice[:-1]))
        position_id = np.concatenate((position_id, [mask_position] * len(choice)))
        block_position_id = np.concatenate((block_position_id, np.arange(1, 1 + len(choice), dtype=np.int64)))

    attention_mask = block_diag(*attention_mask)
    attention_mask[division:, :division] = 1

    return {
        "token": token,
        "position_id": np.stack((position_id, block_position_id)),
        "attention_mask": attention_mask,
        "choices": choice_id,
        "choice_target_ids": choice_target_id
    }


def pad_batch(tokens, position_ids, attention_mask, max_seq_length):
    pad_length = max_seq_length - len(tokens)
    attention_mask = np.pad(
        attention_mask,
        pad_width=((0, pad_length),),
        mode="constant",
        constant_values=0,
    )
    tokens = np.concatenate((tokens, np.zeros(pad_length, dtype=np.int64)))
    position_ids = np.concatenate((position_ids, position_ids[..., -1:].repeat(pad_length, -1)), axis=-1)
    return tokens, position_ids, attention_mask
    
    
def collate_fn(samples):
    TILE = 16
    length_to_pad = (max(map(lambda spl: len(spl["token"]), samples)) + TILE - 1) // TILE * TILE

    token_batch, position_id_batch, attention_mask_batch = [], [], []
    choices_batch, choice_target_ids_batch = [], []

    for sample in samples:
        token, position_id, attention_mask = pad_batch(
            sample["token"], sample["position_id"], sample["attention_mask"], length_to_pad
        )
        token_batch.append(token)
        position_id_batch.append(position_id)
        attention_mask_batch.append(attention_mask)
        choices_batch.append(sample["choices"])
        choice_target_ids_batch.append(sample["choice_target_ids"])

    return {
        "tokens": torch.tensor(np.array(token_batch), dtype=torch.int64),
        "position_ids": torch.tensor(np.array(position_id_batch), dtype=torch.int64),
        "attention_mask": torch.tensor(np.array(attention_mask_batch), dtype=torch.int64),
        "choices": choices_batch,
        "choice_target_ids": choice_target_ids_batch,
    }


def cond_log_prob(model, tokenizer, context: List[str], choices: List[List[str]]) -> List[List[float]]:
    """
    Compute conditonal probability for one or more continuation/infilling options.
    :return The log probablity of each option.
    """
    if not isinstance(context, list):
        context = [context]
        choices = [choices]
    choices = [[(' ' + choice) for choice in choice_pair] for choice_pair in choices]  # Feature of SentencePiece tokenizer
        
    samples = [build_multiple_choice_sample(tokenizer, ctx, ch) for ctx, ch in zip(context, choices)]
    
    batch = collate_fn(samples)
    
    logits = model.forward(input_ids=batch['tokens'].cuda(),
                        attention_mask=batch['attention_mask'].cuda().unsqueeze(1),
                        position_ids=batch['position_ids'].cuda())['logits']
    
    log_probs = []
    
    for output, choices, choice_target_ids in zip(F.log_softmax(logits, dim=-1), batch['choices'], batch['choice_target_ids']):
        log_probs_single = []
        for choice, choice_target_id in zip(choices, choice_target_ids):
            tmp = output[choice_target_id, choice]
            log_probs_single.append(tmp.sum())
        log_probs.append(torch.stack(log_probs_single))
    
    return torch.stack(log_probs)
