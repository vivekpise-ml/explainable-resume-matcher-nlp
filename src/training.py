import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.data_loader import create_pairs, load_labels
from src.matcher_training import ResumeJDDataset


# -----------------------------
# Config
# -----------------------------
data_dir = "data/raw"
csv_path = "data/Data.csv"
model_name = "bert-base-uncased"


# -----------------------------
# Load labels + pairs
# -----------------------------
label_map = load_labels(csv_path)
pairs = create_pairs(data_dir, label_map)

print("Total samples:", len(pairs))


# -----------------------------
# Train/Test split
# -----------------------------
train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)


# -----------------------------
# Tokenizer + Dataset
# -----------------------------
tokenizer = BertTokenizer.from_pretrained(model_name)

train_dataset = ResumeJDDataset(train_pairs, tokenizer)
test_dataset = ResumeJDDataset(test_pairs, tokenizer)


# -----------------------------
# Model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    return {"accuracy": acc, "f1": f1}


# -----------------------------
# Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="models/",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="logs/",
)


# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# -----------------------------
# Train
# -----------------------------
trainer.train()


# -----------------------------
# Evaluate
# -----------------------------
results = trainer.evaluate()
print("Evaluation:", results)


# -----------------------------
# Save model
# -----------------------------
trainer.save_model("models/matcher_model")
tokenizer.save_pretrained("models/matcher_model")

# -----------------------------
# Optionally can call the classification_report in evaluate.py
#   This prediction otherwise is already happening 
#   in compute_metrics() 
#  THIS IS OPTIONAL (but by default still called)
# -----------------------------
from sklearn.metrics import classification_report

predictions = trainer.predict(test_dataset)

y_true = predictions.label_ids
y_pred = predictions.predictions.argmax(axis=1)

print("\nDetailed Report:\n")
print(classification_report(y_true, y_pred))