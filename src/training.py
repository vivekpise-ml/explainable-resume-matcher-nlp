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


USE_FUSION_MODEL = False # to switch between BertWithFeatureFusion model or HybridModel
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
#for sap_ta_abap_1 dataset commenting below as this is for sap_functional
#csv_path = os.path.join(BASE_DIR, "data", "updated_Data.csv")
csv_path = os.path.join(BASE_DIR, "data", "sap_ta_abap_1_Data.csv")

print("BASE_DIR:", BASE_DIR)
print("DATA_DIR:", data_dir)
print("Exists:", os.path.exists(data_dir))

label_map = load_labels(csv_path)
pairs = create_pairs(data_dir, label_map)

# Extract labels from pairs
labels = [item["label"] for item in pairs] # for using stratify

# print(pairs[0].keys()) # For debugging

#train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42) # for debugging

# for equi distribution of labels for testing
train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42, stratify=labels)

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

### To verify if CSV is off ####
sample = next(iter(train_loader))

print("\nSample extra_features:")
#print(sample['extra_features'][0])


'''
# earlier debugs now enhancing the debugs
print("\nFeature breakdown:")
print("Dynamic:", sample['extra_features'][0][:4])
print("CSV:", sample['extra_features'][0][4:7])
print("Missing flags:", sample['extra_features'][0][7:])
'''

feature_names = [
    "exact_ratio",
    #"exact_count_norm",
    "related_ratio",
    #"related_count_norm",
    "coverage_ratio",
    #"missing_ratio",
    "resume_skill_count_norm",
    "jd_skill_count_norm",
    "exact_vs_related",
    "match_density",
    #"jd_unmatched_pressure",
    "balance_score"
]

features = sample['extra_features'][0]

print("\n===== FEATURE BREAKDOWN =====")
for name, value in zip(feature_names, features):
    print(f"{name:30s}: {value:.4f}")
print("============================\n")


test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -----------------------------
# Get feature size dynamically
# -----------------------------
sample = next(iter(train_loader))
extra_dim = sample['extra_features'].shape[1]

import torch
import torch.nn as nn
from transformers import BertModel

# =========================================================
# 🔴 NEW: Controlled Fusion Model (add-on, not replacement)
# =========================================================
class BertWithFeatureFusion(nn.Module):
    def __init__(self, num_labels, feature_dim):
        super().__init__()

        # -----------------------------
        # BERT backbone (unchanged)
        # -----------------------------
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # -----------------------------
        # Projection layers (NEW)
        # -----------------------------
        self.bert_fc = nn.Linear(768, 128)
        self.feature_fc = nn.Linear(feature_dim, 128)

        # -----------------------------
        # Fusion gate (NEW - key part)
        # -----------------------------
        self.alpha_layer = nn.Linear(256, 1)

        # -----------------------------
        # Classifier (slightly modified)
        # -----------------------------
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels)
        )

    def forward(self, input_ids, attention_mask, extra_features):
        
        # -----------------------------
        # BERT forward (unchanged)
        # -----------------------------
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.pooler_output  # [batch, 768]

        # -----------------------------
        # Project both modalities
        # -----------------------------
        bert_embed = self.bert_fc(cls_output)             # [batch, 128]
        feature_embed = self.feature_fc(extra_features)   # [batch, 128]

        # -----------------------------
        # 🔥 Learnable fusion (KEY)
        # -----------------------------
        fusion_input = torch.cat([bert_embed, feature_embed], dim=1)

        alpha = torch.sigmoid(self.alpha_layer(fusion_input))  # [batch, 1]

        fused = alpha * bert_embed + (1 - alpha) * feature_embed

        # -----------------------------
        # Classification
        # -----------------------------
        logits = self.classifier(fused)

        return logits

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



if USE_FUSION_MODEL:
    # --------------------------------------------
    # Trying with BertWithFeatureFusion here
    # --------------------------------------------
    model = BertWithFeatureFusion(num_labels=4, feature_dim = extra_dim).to(device)
else:
    # --------------------------------------------
    # trivial HybridModel for concatenation of extra features with Bert
    # ---------------------------------------------
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
# Train Evaluation (NEW)
# -----------------------------
train_preds, train_labels = [], []

with torch.no_grad():
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        extra_features = batch['extra_features'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, extra_features)
        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

train_acc = accuracy_score(train_labels, train_preds)
print("\nTrain Accuracy:", train_acc)

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), "models/hybrid_model.pt")