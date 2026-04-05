import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_loader import create_pairs, load_labels
from src.matcher_training import ResumeJDDataset

import json

# =========================================================
# 🔴 HOOKS
# =========================================================
USE_BERT = True   # 🔥 turn OFF to test only structured features

# -----------------------------
# Config
# -----------------------------
model_name = "bert-base-uncased"
batch_size = 4
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load data
# -----------------------------
'''
label_map = load_labels("data/updated_Data.csv")
pairs = create_pairs("data/raw", label_map)
'''

import os

# Get project root (2 levels up from src/training.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(BASE_DIR, "data", "raw")
csv_path = os.path.join(BASE_DIR, "data", "updated_Data.csv")

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", data_dir)
print("Exists:", os.path.exists(data_dir))

label_map = load_labels(csv_path)
pairs = create_pairs(data_dir, label_map)

# print(pairs[0].keys()) # For debugging

train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(model_name)

# -----------------------------
# Load skill files
# -----------------------------
with open("data/annotations/skill_dict.json") as f:
    skill_dict = json.load(f)

with open("data/annotations/skill_graph.json") as f:
    skill_graph = json.load(f)

# -----------------------------
# Dataset
# -----------------------------
train_dataset = ResumeJDDataset(train_pairs, tokenizer, skill_dict, skill_graph)
test_dataset = ResumeJDDataset(test_pairs, tokenizer, skill_dict, skill_graph)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -----------------------------
# Get feature size dynamically
# -----------------------------
sample = next(iter(train_loader))
extra_dim = sample['extra_features'].shape[1]

# -----------------------------
# Model
# -----------------------------
class HybridModel(nn.Module):
    def __init__(self, extra_dim, num_labels):
        super().__init__()

        if USE_BERT:
            self.bert = BertModel.from_pretrained(model_name)
            input_dim = 768 + extra_dim
        else:
            self.bert = None
            input_dim = extra_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, extra_features):

        if USE_BERT:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :]
            combined = torch.cat((cls, extra_features), dim=1)
        else:
            combined = extra_features

        return self.classifier(combined)


model = HybridModel(extra_dim, num_labels=4).to(device)

# -----------------------------
# Training
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        extra_features = batch['extra_features'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, extra_features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        extra_features = batch['extra_features'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, extra_features)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nAccuracy:", accuracy_score(all_labels, all_preds))
print("F1:", f1_score(all_labels, all_preds, average="weighted"))

print("\nDetailed Report:\n")
print(classification_report(all_labels, all_preds))

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), "models/hybrid_model.pt")