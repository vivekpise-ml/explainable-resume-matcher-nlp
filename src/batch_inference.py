import os
from src.inference import run_inference

JD_PATH = "sample_jd.txt"
RESUME_FOLDER = "batch_resumes"


# Load JD once
with open(JD_PATH, "r", encoding="utf-8") as f:
    jd_text = f.read()


results = []

for file in os.listdir(RESUME_FOLDER):

    path = os.path.join(RESUME_FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    result = run_inference(resume_text, jd_text)

    results.append((file, result))


# Print results
for file, res in results:
    print("\n====================")
    print("Resume:", file)
    print("Prediction:", res["prediction"])
    print("Confidence:", res["confidence"])