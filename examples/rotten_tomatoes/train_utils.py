import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from eval_utils import evaluate
from multiple_choice_utils import cond_log_prob, flatten_labels


def train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, ckpt_path, epoch_num, early_stopping=-1):
    
    best_acc = 0.
    early_stopping_counter = early_stopping

    for e in range(1, epoch_num + 1):
        print(f"EPOCH {e}")
        train_loss_value = 0.
        tqdm_vars = {"lr": np.nan, "loss": np.nan}
        tbar = tqdm(enumerate(train_loader, start=1), desc="train", total=len(train_loader),
                    postfix=tqdm_vars)

        model.train()

        for _, sample in tbar:
            logits = cond_log_prob(model, tokenizer, sample["inputs_pretokenized"], flatten_labels(sample['choices_pretokenized']))
            labels = sample["label"].cuda()
            loss = F.nll_loss(logits, labels)
            train_loss_value += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            tqdm_vars["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
            tqdm_vars["loss"] = train_loss_value
            tbar.set_postfix(tqdm_vars)
            train_loss_value = 0.

        _, valid_acc = evaluate(model, tokenizer, valid_loader, 'valid')

        if early_stopping >= 0:
            if valid_acc > best_acc:
                best_acc = valid_acc
                early_stopping_counter = early_stopping
                torch.save(model, ckpt_path)
            else:
                early_stopping_counter -= 1

            if early_stopping_counter <= 0:
                print('EARLY STOPPING...')
                break

    return torch.load(ckpt_path)
