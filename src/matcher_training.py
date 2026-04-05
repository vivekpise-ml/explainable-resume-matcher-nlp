import torch
from torch.utils.data import Dataset
import re
import math

# =========================================================
# 🔴 HOOKS (toggle easily for experiments)
# =========================================================
USE_DYNAMIC_SKILLS = True      # skill_dict + skill_graph
USE_CSV_FEATURES = False        # CSV scores
USE_EXPERIENCE = True
USE_QUALIFICATION = True


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
    # CSV features (with fallback)
    # -----------------------------
    skill_score, skill_missing = safe_value(item.get('skill_score'), match_ratio)

    exp_csv, exp_missing = safe_value(item.get('experience_score'), extract_experience(resume))
    qual_csv, qual_missing = safe_value(item.get('qualification_score'), 0.0)

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
            matched / 20,
            missing / 20,
            required / 20
        ]

    # CSV
    if USE_CSV_FEATURES:
        features.append(skill_score)

    if USE_EXPERIENCE:
        features.append(exp_csv * EXP_WEIGHT)

    if USE_QUALIFICATION:
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