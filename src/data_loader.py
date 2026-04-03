import os
#from text_extractor import extract_text
from src.text_extractor import extract_text

def normalize_filename(name):
    name = str(name).strip().lower()
    name = name.replace(".pdf", "").replace(".docx", "")
    return name

def create_pairs(data_dir, label_map):
    pairs = []

    for jd_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, jd_folder)

        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)

         # -----------------------------
        # Find JD file
        # -----------------------------
        jd_files = [
            f for f in files
            if f.lower().startswith(jd_folder.lower())
        ]

        if not jd_files:
            raise ValueError(f"No JD file found in {folder_path}")

        jd_file = jd_files[0]
        jd_text = extract_text(os.path.join(folder_path, jd_file))

        # -----------------------------
        # Process resumes
        # -----------------------------
        for f in files:

            if f == jd_file:
                continue

            if not f.lower().startswith("candidate"):
                continue

            resume_text = extract_text(os.path.join(folder_path, f))

            #key = (jd_folder.lower(), f.lower())
            key = (jd_folder.lower(), normalize_filename(f))
            label = label_map.get(key, None)

            if key not in label_map:
                print("❌ No label for:", key)
                continue

            if label is None:
                continue

            pairs.append({
                "resume": resume_text,
                "jd": jd_text,
                "label": label
            })

    return pairs

import pandas as pd

def load_labels(csv_path):
    df = pd.read_csv(csv_path, header=2)

    '''
    # For debugging
    print("Columns in CSV:", df.columns.tolist())
    '''

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    label_map = {}

    for _, row in df.iterrows():
        jd = str(row["jd title ( folder name)"]).strip().lower()
        #resume = str(row["resume files"]).strip().lower()
        resume = normalize_filename(row["resume files"])
        score = float(row["matching score"])

        '''
        # For training experimentation following is commented. Will 
        # uncomment once more samples we get
        '''       
        # Convert score → label
        if score <= 25:
            label = 0
        elif score <= 50:
            label = 1
        elif score <= 75:
            label = 2
        else:
            label = 3
                 
        '''
        if score >= 50:
            label = 1
        else:
            label = 0
        '''

        key = (jd, resume)
        label_map[key] = label

    return label_map