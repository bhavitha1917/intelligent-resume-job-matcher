import pandas as pd
from datasets import load_dataset

# -----------------------------
# 1. Kaggle Resume Dataset
# -----------------------------

resume_path = "data\\raw\\Resume\\Resume.csv"

resume_df = pd.read_csv(resume_path)

print("\n========== RESUME DATASET ==========")
print("Shape:", resume_df.shape)
print("Columns:", resume_df.columns.tolist())
print(resume_df.head())


# -----------------------------
# 2. Hugging Face Resume-JD Dataset
# -----------------------------

print("\n========== RESUME-JOB FIT DATASET ==========")

dataset = load_dataset("batuhanmtl/job_resume_fit")

print(dataset)

train_data = dataset["train"]

print("Columns:", train_data.column_names)
print("Number of rows:", len(train_data))

fit_df = train_data.to_pandas()

print(fit_df.head())
print(fit_df.shape)