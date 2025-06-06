import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ast
import numpy as np
import random

# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the dataset
df = pd.read_csv('LID_train.csv')
print(f"Initial DataFrame shape: {df.shape}")
print("\nSample of raw data:")
print(df['Annotated by: Annotator 1 '].head())

# Function to extract tokens and labels from annotator data
def extract_tokens_labels(row):
    try:
        entries = ast.literal_eval(row['Annotated by: Annotator 1 '])
        if not entries:  # Check if entries is empty
            return [], []
        if isinstance(entries, list):
            tokens = [entry['key'] for entry in entries]
            labels = [entry['value'] for entry in entries]
            return tokens, labels
        return [], []
    except (ValueError, SyntaxError, KeyError) as e:
        print(f"Error processing row: {e}")
        return [], []

df[['tokens', 'labels']] = df.apply(lambda row: pd.Series(extract_tokens_labels(row)), axis=1)

print("\nSample of first few rows after extraction:")
print(df[['tokens', 'labels']].head())
print(f"Number of rows with tokens: {len(df[df['tokens'].map(len) > 0])}")

# Filter out rows with empty tokens
df_filtered = df[df['tokens'].map(len) > 0].reset_index(drop=True)
print(f"\nShape after filtering: {df_filtered.shape}")

if df_filtered.empty:
    raise ValueError("No valid data found after filtering. Please check your input data format.")

df = df_filtered  # Update the main dataframe with filtered data

# Define label mappings
unique_labels = set(label for sublist in df['labels'] for label in sublist)
label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(label2id)

# Initialize tokenizer
model_name = "google/muril-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Custom Dataset class
class LIDTokenDataset(Dataset):
    def __init__(self, tokens_list, labels_list, tokenizer, max_len):
        self.tokens_list = tokens_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        labels = self.labels_list[idx]
        label_ids = [label2id[label] for label in labels]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        word_ids = encoding.word_ids(batch_index=0)

        # Initialize labels with -100
        aligned_labels = [-100] * len(word_ids)
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                aligned_labels[i] = label_ids[word_idx]
            previous_word_idx = word_idx

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(aligned_labels)
        }

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['tokens'], df['labels'], test_size=0.2, random_state=random_seed
)

# Create datasets
max_len = 128
train_dataset = LIDTokenDataset(train_texts.tolist(), train_labels.tolist(), tokenizer, max_len)
val_dataset = LIDTokenDataset(val_texts.tolist(), val_labels.tolist(), tokenizer, max_len)

# Create dataloaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the model
class LIDTokenClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(LIDTokenClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        return loss, logits

# Initialize the model
model = LIDTokenClassifier(model_name, num_labels)
model.to(device)

# Define optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        _, logits = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(logits, dim=-1)

        for pred, label in zip(predictions, labels):
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()
            for p, l in zip(pred, label):
                if l != -100:
                    all_preds.append(p)
                    all_labels.append(l)

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(num_labels)])
print(report)
