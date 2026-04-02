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

# ------------------------------
# Loading skill graph
#-------------------------------
def load_skill_graph(path):
    import json
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Add graph based matching function
# -----------------------------
def graph_based_match(jd_skills, resume_skills, skill_graph):
    matched = set()
    graph_matches = {}

    resume_set = set(resume_skills)

    for jd_skill in jd_skills:

        # -----------------------------
        # Direct Match
        # -----------------------------
        if jd_skill in resume_set:
            matched.add(jd_skill)
            continue

        # -----------------------------
        # Graph Match
        # -----------------------------
        if jd_skill in skill_graph:
            related = skill_graph[jd_skill]

            for rel_skill in related:
                if rel_skill in resume_set:
                    matched.add(jd_skill)
                    graph_matches[jd_skill] = rel_skill
                    break

    return matched, graph_matches


# -----------------------------
# Main Inference Function
# -----------------------------
'''
def run_inference(
    resume_text,
    jd_text,
    matcher_model,
    tokenizer,
    skill_dict,
    ner_model=None,
    row_data=None
):
'''
def run_inference(
    resume_text,
    jd_text,
    matcher_model,
    tokenizer,
    skill_dict,
    skill_graph,
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
    
    # matched_skills = list(set(resume_tech) & set(jd_tech))
    # missing_skills = list(set(jd_tech) - set(resume_tech))

    resume_tech = list(set(resume_tech)) # To remove duplicates using set
    jd_tech = list(set(jd_tech)) # To remove duplicates using set

    # -----------------------------
    # Graph-based Skill Matching
    # -----------------------------
    matched_set, graph_matches = graph_based_match(
        jd_tech, resume_tech, skill_graph
    )

    missing_set = set(jd_tech) - matched_set

    matched_skills = list(matched_set)
    missing_skills = list(missing_set)

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
       
       ''' 
        "skill_analysis": {
             "matched": matched_skills,
            "missing": missing_skills
        },
       '''
        "skill_analysis": {
             "matched": matched_skills,
            "missing": missing_skills,
            "graph_matches": graph_matches
        },

        "features": structured_features
    }