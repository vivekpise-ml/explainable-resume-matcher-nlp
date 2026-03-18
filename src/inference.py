'''
def run_inference(resume_text, jd_text, ner_model, matcher_model, tokenizer):
    
    # Extract skills
    doc_resume = ner_model(resume_text)
    resume_skills = [ent.text.lower() for ent in doc_resume.ents]

    doc_jd = ner_model(jd_text)
    jd_skills = [ent.text.lower() for ent in doc_jd.ents]

    # Match score
    inputs = tokenizer(resume_text, jd_text, return_tensors="pt", truncation=True)
    outputs = matcher_model(**inputs)
    score = outputs.logits.softmax(dim=1)[0][1].item()

    return resume_skills, jd_skills, score
'''

import torch
from src.skill_extraction import load_skill_dict, extract_skills
from src.feature_extraction import extract_structured_features


# -----------------------------
# Optional: Load spaCy model
# -----------------------------
def load_ner_model():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except:
        print("spaCy model not found. Running without NER support.")
        return None


# -----------------------------
# Main Inference Function
# -----------------------------
def run_inference(
    resume_text,
    jd_text,
    matcher_model,
    tokenizer,
    skill_dict,
    ner_model=None,
    row_data=None
):
    
    # -----------------------------
    # Skill Extraction (Hybrid)
    # -----------------------------
    resume_tech, resume_soft = extract_skills(
        resume_text, skill_dict, ner_model
    )

    jd_tech, jd_soft = extract_skills(
        jd_text, skill_dict, ner_model
    )

    # -----------------------------
    # Structured Features
    # -----------------------------
    structured_features = {}
    if row_data is not None:
        structured_features = extract_structured_features(row_data)

    # -----------------------------
    # Transformer Matching
    # -----------------------------
    inputs = tokenizer(
        resume_text,
        jd_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = matcher_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        score = probs[0][1].item()

    # -----------------------------
    # Skill Gap Analysis (basic)
    # -----------------------------
    matched_skills = list(set(resume_tech) & set(jd_tech))
    missing_skills = list(set(jd_tech) - set(resume_tech))

    # -----------------------------
    # Final Output
    # -----------------------------
    return {
        "match_score": round(score, 3),

        "resume": {
            "tech_skills": resume_tech,
            "soft_skills": resume_soft
        },

        "jd": {
            "tech_skills": jd_tech,
            "soft_skills": jd_soft
        },

        "skill_analysis": {
            "matched": matched_skills,
            "missing": missing_skills
        },

        "features": structured_features
    }