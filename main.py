"""
Language Identification (LID) Training Script
============================================

This script trains and evaluates multilingual transformer models on a language identification task.

0--> Only Codemix
1--> English + Codemix
2--> Hindi + Codemix
3--> Hi +En + CM
Usage Example:
--------------
python main.py --model MODEL_NAME --train train_csv_file --test test_csv_file --output output_txt_file --train_config N[0/1/2/3] --sample_size_en N1 --sample_size_hi N2 --train_sample_size N3[Total_training_examples] --patience[for early stopping] 

Eg 1 : python main.py --model XLMR --train lince_train.csv --test lince_test.csv --output XLMR_config3_es.txt --train_config 3  --sample_size_hi 4000 --sample_size_en 4000 --train_sample_size 12000 --patience 3
"""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_scheduler

MODEL_PATHS = {
    'XLMR': 'xlm-roberta-base',
    'mBERT': 'bert-base-multilingual-cased',
    'MuRIL': 'google/muril-base-cased',
    'IndicBERT': 'ai4bharat/indic-bert',
    'XLM': 'xlm-mlm-100-1280'
}

def parse_args():
    '''to take input all arguments
       --model [XLMR, mBERT, MuRIL, IndicBERT, XLM ]
       --train [train csv path]
       --test [test csv path]
       --output [output results txt path]
       --batch_size [integer] default: 16
       --epochs [integer] default: 50
       --train_config [0--> CM, 1--> CM+EN, 2-->CM+HI, 3-->CM+EN+HI]
       --sample_size_en [number of samples to be picked from native english dataset]
       --sample_size_hi [number of samples to be picked from native hindi dataset]
       --train_sample_size [Total number of training samples to use after combining]
       --patience [Early stopping patience]
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODEL_PATHS.keys())
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)  # changed default max epochs to 50
    parser.add_argument('--train_config', type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument('--sample_size_en', type=int, default=0, help='Number of English samples to add')
    parser.add_argument('--sample_size_hi', type=int, default=0, help='Number of Hindi samples to add')
    parser.add_argument('--train_sample_size', type=int, default=0, help='Total number of training samples to use after combining')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    return parser.parse_args()

def parse_token_label_column(json_str):
    try:
        json_str = json_str.replace('""', '"')
        items = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"❌ JSONDecodeError: {e}")
        print(f"Problematic row: {json_str}")
        return [], []
    tokens, labels = [], []
    for obj in items:
        token = str(obj['key']).strip()
        label = obj['value'].replace('eng', 'en')
        tokens.append(token)
        labels.append(label)
    return tokens, labels

def load_dataset(csv_path):
    '''for reading dataset and corresponding labels'''
    df = pd.read_csv(csv_path)
    token_lists, label_lists = [], []
    for _, row in df.iterrows():
        tokens, labels = parse_token_label_column(row['Tags'])
        token_lists.append(tokens)
        label_lists.append(labels)
    return token_lists, label_lists

def load_and_sample_datasets(base_path, config, sample_size_en, sample_size_hi, train_sample_size):
    base_tokens, base_labels = load_dataset(base_path)
    additional_tokens, additional_labels = [], []

    if config in [1, 3] and sample_size_en > 0:
        eng_tokens, eng_labels = load_dataset('english_dataset_cleaned.csv')
        indices = random.sample(range(len(eng_tokens)), min(sample_size_en, len(eng_tokens)))
        additional_tokens += [eng_tokens[i] for i in indices]
        additional_labels += [eng_labels[i] for i in indices]

    if config in [2, 3] and sample_size_hi > 0:
        hin_tokens, hin_labels = load_dataset('hindi_dataset_cleaned.csv')
        indices = random.sample(range(len(hin_tokens)), min(sample_size_hi, len(hin_tokens)))
        additional_tokens += [hin_tokens[i] for i in indices]
        additional_labels += [hin_labels[i] for i in indices]

    combined = list(zip(base_tokens + additional_tokens, base_labels + additional_labels))
    random.shuffle(combined)

    if train_sample_size > 0 and train_sample_size < len(combined):
        combined = random.sample(combined, train_sample_size)

    combined_tokens, combined_labels = zip(*combined)
    return list(combined_tokens), list(combined_labels)

unique_labels = ['hi', 'en', 'un']
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}

class LIDTokenDataset(Dataset):
    def __init__(self, token_lists, label_lists, tokenizer, max_len=128):
        self.token_lists = token_lists
        self.label_lists = label_lists
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        tokens = list(map(str, self.token_lists[idx]))
        labels = self.label_lists[idx]
        text = " ".join(tokens)

        tokenizer_kwargs = {
            "text": text,
            "truncation": True,
            "max_length": self.max_len,
            "padding": "max_length",
            "return_tensors": "pt",
        }

        # Only add return_offsets_mapping if tokenizer supports it
        if hasattr(self.tokenizer, "is_fast") and self.tokenizer.is_fast:
            tokenizer_kwargs["return_offsets_mapping"] = True
        else:
            # If slow tokenizer, skip offset mapping and align labels differently (or raise error)
            # Here, simplest is to raise error or handle differently
            # For now, let's skip offset mapping and create dummy aligned_labels
            offset_mapping = None

        encoding = self.tokenizer(**tokenizer_kwargs)

        if "offset_mapping" in encoding:
            offset_mapping = encoding.pop("offset_mapping").squeeze()
            aligned_labels = torch.full((self.max_len,), -100, dtype=torch.long)

            token_idx = 0
            for i, (start, end) in enumerate(offset_mapping.tolist()):
                if start == end == 0:
                    continue
                if token_idx < len(labels):
                    aligned_labels[i] = label2id[labels[token_idx]]
                    token_idx += 1
        else:
            # If offset mapping not available, fallback: assign first N tokens label and rest -100
            aligned_labels = torch.full((self.max_len,), -100, dtype=torch.long)
            length = min(len(labels), self.max_len)
            for i in range(length):
                aligned_labels[i] = label2id[labels[i]]

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": aligned_labels,
        }


def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)

            for pred, true in zip(preds, labels):
                for p, l in zip(pred.cpu().numpy(), true.cpu().numpy()):
                    if l != -100:
                        all_preds.append(id2label[p])
                        all_trues.append(id2label[l])

    avg_loss = total_loss / len(data_loader)
    report = classification_report(all_trues, all_preds, labels=unique_labels)
    return avg_loss, report

def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fast = False if 'IndicBERT' or 'XLM' in args.model else True
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model], use_fast=use_fast)


    train_tokens, train_labels = load_and_sample_datasets(
        args.train, args.train_config, args.sample_size_en, args.sample_size_hi, args.train_sample_size
    )
    test_tokens, test_labels = load_dataset(args.test)

    train_dataset = LIDTokenDataset(train_tokens, train_labels, tokenizer)
    test_dataset = LIDTokenDataset(test_tokens, test_labels, tokenizer)

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_PATHS[args.model],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = get_scheduler("linear", optimizer, 0, num_training_steps)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        val_loss, val_report = evaluate_model(model, val_loader, device)
        print(f"\nEpoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        print(val_report)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_config_{args.train_config}.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

    print(f"\nLoading best model from epoch {best_epoch+1} with val loss {best_val_loss:.4f}")
    model.load_state_dict(torch.load(f"best_model_config_{args.train_config}.pt"))
    test_loss, test_report = evaluate_model(model, test_loader, device)

    with open(args.output, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Training Config: {args.train_config}\n")
        f.write(f"Batch Size: {args.batch_size}, Epochs: {epoch+1}\n")
        f.write(f"Sample Size - English: {args.sample_size_en}, Hindi: {args.sample_size_hi}, Total Train Samples: {len(train_tokens)}\n")
        f.write("="*50 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write("Test Report:\n")
        f.write(test_report)

    print(f"\n✅ Final test results saved to {args.output}")
    os.remove(f"best_model_config_{args.train_config}.pt")

if __name__ == "__main__":
    main()
