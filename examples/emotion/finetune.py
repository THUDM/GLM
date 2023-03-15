import torch
import argparse
import warnings
from train_utils import train
from eval_utils import evaluate
from dataset import MultipleChoiceDataset
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, get_linear_schedule_with_warmup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', '--model_type', type=str, default='BAAI/glm-roberta-large')
    parser.add_argument('-dn', '--dataset_name', type=str, default='emotion')
    parser.add_argument('-pn', '--prompt_name', type=str, default='select_emotion_label_from_list')
    parser.add_argument('-bs', '--batch_size', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-en', '--epoch_num', type=int, default=10)
    parser.add_argument('-es', '--early_stopping', type=int, default=2)
    parser.add_argument('-cd', '--ckpt_dir', type=str, default='./')
    args = parser.parse_args()
    print(args)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_type, trust_remote_code=True, revision='main')
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_type, trust_remote_code=True, revision='main').cuda()

    # Load data
    train_dataset = MultipleChoiceDataset(args.dataset_name, 'train', args.prompt_name, tokenizer)
    valid_dataset = MultipleChoiceDataset(args.dataset_name, 'validation', args.prompt_name, tokenizer)
    test_dataset = MultipleChoiceDataset(args.dataset_name, 'test', args.prompt_name, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Configure training model, optimizer, and scheduler
    model = model.float()
    model.train()
    num_training_steps = args.epoch_num * (len(train_dataset) // args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(num_training_steps * 0.06),
                                                num_training_steps=num_training_steps)

    print('Performance on test set BEFORE fine-tuning:')
    evaluate(model, tokenizer, test_loader, 'test')
    
    print('TRAINING...')
    ckpt_path = args.ckpt_dir + \
                f"{args.model_type.split('/')[1] if '/' in args.model_type else args.model_type}-" + \
                f"{args.dataset_name.split('/')[1] if '/' in args.dataset_name else args.dataset_name}.ckpt"
    model = train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, ckpt_path,
                  args.epoch_num, args.early_stopping)
    
    print('Performance on test set AFTER fine-tuning:')
    evaluate(model, tokenizer, test_loader, 'test')

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()
