# Language Identification (LID) Training Script
# ================================
# Supports training on CM, EN, HI or combinations based on --train_config

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODEL_PATHS.keys())
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--train_config', type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('--sample_size_en', type=int, default=0)
    parser.add_argument('--sample_size_hi', type=int, default=0)
    parser.add_argument('--train_sample_size', type=int, default=0)
    parser.add_argument('--patience', type=int, default=3)
    return parser.parse_args()

def parse_token_label_column(json_str):
    try:
        json_str = json_str.replace('""', '"')
        items = json.loads(json_str)
    except json.JSONDecodeError:
        return [], []
    tokens, labels = [], []
    for obj in items:
        token = str(obj['key']).strip()
        label = obj['value'].replace('eng', 'en')
        tokens.append(token)
        labels.append(label)
    return tokens, labels

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    token_lists, label_lists = [], []
    for _, row in df.iterrows():
        tokens, labels = parse_token_label_column(row['Tags'])
        token_lists.append(tokens)
        label_lists.append(labels)
    return token_lists, label_lists

def load_and_sample_datasets(base_path, config, sample_size_en, sample_size_hi, train_sample_size):
    base_tokens, base_labels = [], []
    eng_tokens, eng_labels = [], []
    hin_tokens, hin_labels = [], []

    if config in [0, 1, 2, 3]:
        if os.path.exists(base_path):
            base_tokens, base_labels = load_dataset(base_path)
            print(f"✅ Loaded {len(base_tokens)} CM samples from {base_path}")

    if config in [1, 3, 4] and sample_size_en > 0:
        eng_tokens, eng_labels = load_dataset('english_dataset_cleaned.csv')
        print(f"✅ Loaded {len(eng_tokens)} English samples")

    if config in [2, 3, 4] and sample_size_hi > 0:
        hin_tokens, hin_labels = load_dataset('hindi_dataset_cleaned.csv')
        print(f"✅ Loaded {len(hin_tokens)} Hindi samples")

    selected_tokens, selected_labels = [], []

    if config == 0:
        selected_tokens += base_tokens
        selected_labels += base_labels
        print("📌 Using only Code-Mixed samples.")
    elif config == 1:
        if train_sample_size == sample_size_en:
            indices = random.sample(range(len(eng_tokens)), min(sample_size_en, len(eng_tokens)))
            selected_tokens = [eng_tokens[i] for i in indices]
            selected_labels = [eng_labels[i] for i in indices]
            print("📌 Using only English samples (pure English training).")
        else:
            selected_tokens = base_tokens
            selected_labels = base_labels
            indices = random.sample(range(len(eng_tokens)), min(sample_size_en, len(eng_tokens)))
            selected_tokens += [eng_tokens[i] for i in indices]
            selected_labels += [eng_labels[i] for i in indices]
            print("📌 Using CM + English samples.")
    elif config == 2:
        if train_sample_size == sample_size_hi:
            indices = random.sample(range(len(hin_tokens)), min(sample_size_hi, len(hin_tokens)))
            selected_tokens = [hin_tokens[i] for i in indices]
            selected_labels = [hin_labels[i] for i in indices]
            print("📌 Using only Hindi samples (pure Hindi training).")
        else:
            selected_tokens = base_tokens
            selected_labels = base_labels
            indices = random.sample(range(len(hin_tokens)), min(sample_size_hi, len(hin_tokens)))
            selected_tokens += [hin_tokens[i] for i in indices]
            selected_labels += [hin_labels[i] for i in indices]
            print("📌 Using CM + Hindi samples.")
    elif config == 3:
        selected_tokens = base_tokens
        selected_labels = base_labels
        indices_en = random.sample(range(len(eng_tokens)), min(sample_size_en, len(eng_tokens)))
        indices_hi = random.sample(range(len(hin_tokens)), min(sample_size_hi, len(hin_tokens)))
        selected_tokens += [eng_tokens[i] for i in indices_en] + [hin_tokens[i] for i in indices_hi]
        selected_labels += [eng_labels[i] for i in indices_en] + [hin_labels[i] for i in indices_hi]
        print("📌 Using CM + English + Hindi samples.")
    elif config == 4:
        indices_en = random.sample(range(len(eng_tokens)), min(sample_size_en, len(eng_tokens)))
        indices_hi = random.sample(range(len(hin_tokens)), min(sample_size_hi, len(hin_tokens)))
        selected_tokens += [eng_tokens[i] for i in indices_en] + [hin_tokens[i] for i in indices_hi]
        selected_labels += [eng_labels[i] for i in indices_en] + [hin_labels[i] for i in indices_hi]
        print("📌 Using only English + Hindi samples (no CM).")

    combined = list(zip(selected_tokens, selected_labels))
    if len(combined) == 0:
        raise ValueError("❌ No data selected for training. Check sample sizes or dataset availability.")

    random.shuffle(combined)

    if train_sample_size > 0 and train_sample_size < len(combined):
        combined = random.sample(combined, train_sample_size)

    final_tokens, final_labels = zip(*combined)
    return list(final_tokens), list(final_labels)

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

        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
            padding="max_length", return_tensors="pt", return_offsets_mapping=True
        )

        offset_mapping = encoding.pop("offset_mapping").squeeze()
        aligned_labels = torch.full((self.max_len,), -100, dtype=torch.long)

        token_idx = 0
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start == end == 0:
                continue
            if token_idx < len(labels):
                aligned_labels[i] = label2id[labels[token_idx]]
                token_idx += 1

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": aligned_labels
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model], use_fast=True)

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
        MODEL_PATHS[args.model], num_labels=len(label2id),
        id2label=id2label, label2id=label2id
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
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
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
                print(f"\n⛔ Early stopping triggered at epoch {epoch+1}")
                break

    print(f"\n✅ Loading best model from epoch {best_epoch+1}")
    model.load_state_dict(torch.load(f"best_model_config_{args.train_config}.pt"))
    test_loss, test_report = evaluate_model(model, test_loader, device)

    with open(args.output, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Train Config: {args.train_config}\n")
        f.write(f"Batch Size: {args.batch_size}, Epochs: {epoch+1}\n")
        f.write(f"Train Size: {len(train_tokens)}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(test_report)

    print(f"\n📁 Results saved to {args.output}")
    os.remove(f"best_model_config_{args.train_config}.pt")

if __name__ == "__main__":
    main()
