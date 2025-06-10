"""
Language Identification (LID) Training Script
============================================
Trains and evaluates multilingual transformer models on a language identification task.

0 --> Only Codemix
1 --> English + Codemix
2 --> Hindi + Codemix
3 --> Hi + En + CM
4 --> Only English + Hindi (no CM)

Usage Example:
--------------
python main.py --model XLMR --train lince_train.csv --test lince_test.csv --output result.txt \
--train_config 3 --sample_size_hi 4000 --sample_size_en 4000 --train_sample_size 12000 --patience 3
"""

import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report
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

def clean_unicode_text(text):
    """Clean text by removing or replacing invalid Unicode characters."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove or replace surrogate characters
    # Method 1: Use encode with 'ignore' to skip bad characters
    try:
        # First try to encode/decode to catch issues early
        cleaned = text.encode('utf-8', errors='ignore').decode('utf-8')
        return cleaned
    except UnicodeError:
        # Fallback: manually filter out surrogate characters
        cleaned_chars = []
        for char in text:
            try:
                char.encode('utf-8')
                cleaned_chars.append(char)
            except UnicodeEncodeError:
                # Skip this character or replace with space
                cleaned_chars.append(' ')
        return ''.join(cleaned_chars)

def parse_token_label_column(json_str):
    try:
        json_str = str(json_str).strip()
        
        # Clean unicode issues first
        json_str = clean_unicode_text(json_str)
        
        # Handle quotes around the entire string
        if json_str.startswith('"') and json_str.endswith('"'):
            json_str = json_str[1:-1]
        
        # Handle escaped quotes within the string
        json_str = json_str.replace('\\"', '"')
        
        # Parse the list
        items = ast.literal_eval(json_str)
        assert isinstance(items, list), f"Expected list, got {type(items)}"
        
    except Exception as e:
        print(f"❌ Error parsing row: {e}")
        print(f"Row content (first 200 chars): {str(json_str)[:200]}...")
        return [], []

    tokens, labels = [], []
    try:
        for i, obj in enumerate(items):
            if not isinstance(obj, dict):
                print(f"❌ Item {i} is not a dict: {obj}")
                continue
                
            if 'key' not in obj or 'value' not in obj:
                print(f"❌ Item {i} missing key/value: {obj}")
                continue
            
            # Clean unicode for both token and label
            token = clean_unicode_text(str(obj['key'])).strip()
            label = clean_unicode_text(str(obj['value'])).replace('eng', 'en').strip()
            
            # Skip empty tokens
            if token:
                tokens.append(token)
                labels.append(label)
                
    except Exception as e:
        print(f"❌ Error processing items: {e}")
        return [], []
    
    return tokens, labels

def load_dataset(csv_path):
    print(f"📁 Loading dataset from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded CSV with {len(df)} rows")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Check if 'Tags' column exists
        if 'Tags' not in df.columns:
            print(f"❌ 'Tags' column not found. Available columns: {list(df.columns)}")
            return [], []
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return [], []
    
    token_lists, label_lists = [], []
    failed_rows = 0
    
    for idx, row in df.iterrows():
        try:
            tokens, labels = parse_token_label_column(row['Tags'])
            if tokens and labels:  # Only add non-empty sequences
                token_lists.append(tokens)
                label_lists.append(labels)
            else:
                failed_rows += 1
                if failed_rows <= 5:  # Show first 5 failures
                    print(f"⚠️  Row {idx} failed to parse or is empty")
        except Exception as e:
            failed_rows += 1
            if failed_rows <= 5:
                print(f"❌ Error processing row {idx}: {e}")
    
    print(f"✅ Successfully parsed {len(token_lists)} sequences")
    if failed_rows > 0:
        print(f"⚠️  Failed to parse {failed_rows} rows")
    
    # Show sample data
    if token_lists:
        print(f"📝 Sample tokens: {token_lists[0][:5]}...")
        print(f"🏷️  Sample labels: {label_lists[0][:5]}...")
    
    return token_lists, label_lists

def load_and_sample_datasets(base_path, config, sample_size_en, sample_size_hi, train_sample_size):
    base_tokens, base_labels = [], []
    eng_tokens, eng_labels = [], []
    hin_tokens, hin_labels = [], []

    # Load base codemix data for configs 0, 1, 2, 3
    if config in [0, 1, 2, 3]:
        if os.path.exists(base_path):
            base_tokens, base_labels = load_dataset(base_path)
            print(f"✅ Loaded {len(base_tokens)} CM samples from {base_path}")

    # Load English data for configs 1, 3, 4
    if config in [1, 3, 4] and sample_size_en > 0:
        eng_tokens, eng_labels = load_dataset('english_dataset_cleaned.csv')
        print(f"✅ Loaded {len(eng_tokens)} English samples")

    # Load Hindi data for configs 2, 3, 4
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
    def __init__(self, token_lists, label_lists, tokenizer, label2id, max_len=128):
        self.token_lists = token_lists
        self.label_lists = label_lists
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        tokens = self.token_lists[idx]
        labels = self.label_lists[idx]
        
        # Ensure tokens and labels are strings and equal length
        clean_tokens = []
        clean_labels = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # Clean unicode issues in tokens and labels
            token_str = clean_unicode_text(str(token)).strip()
            label_str = clean_unicode_text(str(label)).strip()
            
            if token_str:
                clean_tokens.append(token_str)
                clean_labels.append(label_str)
        
        # Handle empty sequences
        if not clean_tokens:
            clean_tokens = [""]
            clean_labels = ["un"]
        
        # Create text by joining tokens with spaces - ENSURE IT'S A STRING
        text = " ".join(clean_tokens)
        
        # Clean the final text as well
        text = clean_unicode_text(text)
        
        # Validate that text is a string
        if not isinstance(text, str):
            print(f"❌ Text is not a string: {type(text)} - {text}")
            text = str(text)
        
        # Ensure text is not empty
        if not text.strip():
            text = " "  # Single space as fallback
        
        try:
            # Standard tokenization - pass string directly
            inputs = self.tokenizer.encode_plus(
                text,
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=True
            )
        except Exception as e:
            print(f"❌ Tokenization error for text: '{text}' (type: {type(text)})")
            print(f"Error: {e}")
            # Fallback to a simple string
            inputs = self.tokenizer.encode_plus(
                " ",
                truncation=True,
                max_length=self.max_len,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=True
            )
        
        # Initialize labels
        labels_tensor = torch.full((self.max_len,), -100, dtype=torch.long)
        
        # Simple alignment strategy
        # Tokenize individual words to understand subword structure
        word_boundaries = []
        current_pos = 1  # Start after [CLS]
        
        for word in clean_tokens:
            try:
                # Clean unicode in word before tokenizing
                clean_word = clean_unicode_text(str(word))
                word_tokens = self.tokenizer.tokenize(clean_word)
                word_boundaries.append((current_pos, current_pos + len(word_tokens)))
                current_pos += len(word_tokens)
            except Exception as e:
                print(f"❌ Error tokenizing word '{word}': {e}")
                continue
        
        # Assign labels
        for i, (start, end) in enumerate(word_boundaries):
            if i < len(clean_labels) and start < self.max_len:
                # Only label the first subtoken of each word
                labels_tensor[start] = self.label2id.get(clean_labels[i], self.label2id['un'])
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels_tensor,
        }


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_trues = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            predictions = torch.argmax(outputs.logits, dim=-1)
            for pred, true in zip(predictions, labels):
                for p, t in zip(pred.cpu().numpy(), true.cpu().numpy()):
                    if t != -100:
                        all_preds.append(id2label[p])
                        all_trues.append(id2label[t])

    avg_loss = total_loss / len(loader)
    report = classification_report(all_trues, all_preds, labels=unique_labels)
    return avg_loss, report

def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[args.model], use_fast=True)

    train_tokens, train_labels = load_and_sample_datasets(
        args.train, args.train_config, args.sample_size_en, args.sample_size_hi, args.train_sample_size
    )
    test_tokens, test_labels = load_dataset(args.test)

    train_dataset = LIDTokenDataset(train_tokens, train_labels, tokenizer, label2id)
    test_dataset = LIDTokenDataset(test_tokens, test_labels, tokenizer, label2id)

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
    scheduler = get_scheduler("linear", optimizer, 0, num_training_steps)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pbar.set_postfix(loss=loss.item())

        val_loss, val_report = evaluate_model(model, val_loader, device)
        print(f"\nEpoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
        print(val_report)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_config_{args.train_config}_{args.output}.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("\nEarly stopping triggered.")
                break

    model.load_state_dict(torch.load(f"best_model_config_{args.train_config}_{args.output}.pt"))
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

    print(f"\n✅ Test results written to {args.output}")
    os.remove(f"best_model_config_{args.train_config}_{args.output}.pt")

if __name__ == "__main__":
    main()