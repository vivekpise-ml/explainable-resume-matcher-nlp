'''
from src.data_loader import create_pairs

data_dir = "data/raw"

pairs = create_pairs(data_dir)

print("Total pairs:", len(pairs))
'''

from src.data_loader import create_pairs
from src.skill_extraction import load_skill_dict
from src.inference import run_inference, load_ner_model

# (Assume you already have these)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -----------------------------
# Config
# -----------------------------
data_dir = "data/raw"
skill_dict_path = "data/annotations/skill_dict.json"
model_name = "distilbert-base-uncased"


# -----------------------------
# Step 1: Load data
# -----------------------------
pairs = create_pairs(data_dir)
print("Total pairs:", len(pairs))


# -----------------------------
# Step 2: Load skill dictionary
# -----------------------------
skill_dict = load_skill_dict(skill_dict_path)


# -----------------------------
# Step 3: Load NER model (optional)
# -----------------------------
ner_model = load_ner_model()


# -----------------------------
# Step 4: Load transformer model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
matcher_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# -----------------------------
# Step 5: Run inference (sample)
# -----------------------------
for i, pair in enumerate(pairs[:3]):  # test on first 3

    resume_text = pair["resume_text"]
    jd_text = pair["jd_text"]

    result = run_inference(
        resume_text=resume_text,
        jd_text=jd_text,
        matcher_model=matcher_model,
        tokenizer=tokenizer,
        skill_dict=skill_dict,
        ner_model=ner_model,
        row_data=None
    )

    print(f"\n--- Result {i+1} ---")
    print("Match Score:", result["match_score"])
    print("Matched Skills:", result["skill_analysis"]["matched"])
    print("Missing Skills:", result["skill_analysis"]["missing"])