import os
from text_extractor import extract_text

def create_pairs(data_dir):
    pairs = []

    for jd_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, jd_folder)

        if not os.path.isdir(folder_path):
            continue

        files = os.listdir(folder_path)

        jd_file = [f for f in files if "jd" in f.lower()][0]
        jd_text = extract_text(os.path.join(folder_path, jd_file))

        for f in files:
            if f == jd_file:
                continue

            resume_path = os.path.join(folder_path, f)
            resume_text = extract_text(resume_path)

            pairs.append({
                "resume": resume_text,
                "jd": jd_text,
                "label": None  # you will fill this from Data.csv
            })

    return pairs