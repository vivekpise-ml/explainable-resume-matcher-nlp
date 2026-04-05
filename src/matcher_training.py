import torch
from torch.utils.data import Dataset
import re
import math

# =========================================================
# 🔴 HOOKS (toggle easily for experiments)
# =========================================================
USE_DYNAMIC_SKILLS = True      # skill_dict + skill_graph
#USE_CSV_FEATURES = False        # CSV scores
#USE_EXPERIENCE = True
#USE_QUALIFICATION = True


# -----------------------------
# Skill extraction
# -----------------------------
def extract_skills(text, skill_dict):
    text = text.lower()
    found = set()

    for skill in skill_dict:
        if skill.lower() in text:
            found.add(skill)

    return found


# -----------------------------
# Graph matching
# -----------------------------
def is_related(skill, resume_skills, skill_graph):
    related = skill_graph.get(skill, [])
    return any(rs in related for rs in resume_skills)


# -----------------------------
# Experience extraction (fallback)
# -----------------------------
def extract_experience(text):
    matches = re.findall(r'(\d+)\+?\s*years', text.lower())
    return max([int(m) for m in matches], default=0)


# -----------------------------
# Safe value
# -----------------------------
def safe_value(val, fallback):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return fallback, 1.0  # value, missing_flag
    return val, 0.0


# -----------------------------
# Feature computation
# -----------------------------
def compute_features(item, skill_dict, skill_graph):
    resume = item['resume']
    jd = item['jd']

    # -----------------------------
    # Dynamic skill features
    # -----------------------------
    if USE_DYNAMIC_SKILLS:
        resume_skills = extract_skills(resume, skill_dict)
        jd_skills = extract_skills(jd, skill_dict)

        required = len(jd_skills)

        exact = resume_skills & jd_skills

        related = set()
        for s in jd_skills:
            if s not in exact and is_related(s, resume_skills, skill_graph):
                related.add(s)

        matched = len(exact) + len(related)
        missing = required - matched

        match_ratio = matched / required if required > 0 else 0.0
    else:
        match_ratio = matched = missing = required = 0.0

    # -----------------------------
    # CSV skills (clean handling)
    # -----------------------------
    skill_score = item.get('skill_score')
    skill_score = 0.0 if skill_score is None else skill_score / 100  # normalize
    skill_missing = 1 if skill_score is None else 0

    
    # -----------------------------
    # CSV features - qualification and experience
    #-------------------------------
    #exp_csv, exp_missing = safe_value(item.get('experience_score'), extract_experience(resume))
    #qual_csv, qual_missing = safe_value(item.get('qualification_score'), 0.0)

    exp_csv = item.get('experience_score')
    exp_missing = 1 if exp_csv is None else 0
    exp_csv = 0.0 if exp_csv is None else exp_csv / 100

    qual_csv = item.get('qualification_score')
    qual_missing = 1 if qual_csv is None else 0
    qual_csv = 0.0 if qual_csv is None else qual_csv / 100

    # -----------------------------
    # Optional weighting (simple static fallback)
    # -----------------------------
    EXP_WEIGHT = 0.05
    QUAL_WEIGHT = 0.05

    # -----------------------------
    # Build feature vector
    # -----------------------------
    features = []

    # Dynamic
    if USE_DYNAMIC_SKILLS:
        features += [
            match_ratio,
            matched / (required + 1e-5),   # better normalization,
            missing / (required + 1e-5),
            required / 20  # optional max requirements cap of 20 in JD
        ]
    else:
        features += [0.0, 0.0, 0.0]

    # CSV features
    
    features.append(skill_score)

    features.append(exp_csv * EXP_WEIGHT)

    features.append(qual_csv * QUAL_WEIGHT)

    # Missing flags (VERY important)
    features += [skill_missing, exp_missing, qual_missing]

    return torch.tensor(features, dtype=torch.float)


# -----------------------------
# Dataset
# -----------------------------
class ResumeJDDataset(Dataset):
    def __init__(self, pairs, tokenizer, skill_dict, skill_graph):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.skill_dict = skill_dict
        self.skill_graph = skill_graph

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]

        encoding = self.tokenizer(
            item['resume'],
            item['jd'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )

        extra_features = compute_features(item, self.skill_dict, self.skill_graph)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'extra_features': extra_features,
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }