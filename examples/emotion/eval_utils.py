import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from multiple_choice_utils import cond_log_prob, flatten_labels


def evaluate(model, tokenizer, data_loader, split):
    valid_loss = 0.
    valid_labels = []
    valid_preds = []

    model.eval()

    with torch.no_grad():
        for _, sample in tqdm(enumerate(data_loader, start=1), desc=split, total=len(data_loader)):
            logits = cond_log_prob(model, tokenizer, sample["inputs_pretokenized"], flatten_labels(sample['choices_pretokenized']))

            labels = sample["label"].cuda()
            loss = F.nll_loss(logits, labels)
            valid_loss += loss.item()
            valid_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            valid_labels.extend(np.array(sample["label"]).tolist())

    valid_loss = valid_loss / len(data_loader)
    valid_acc = accuracy_score(valid_preds, valid_labels)
    print(f"[{split.upper()}] loss={valid_loss}, acc={valid_acc}")

    return valid_loss, valid_acc
