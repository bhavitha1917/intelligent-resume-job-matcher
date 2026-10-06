"""EDA: Kaggle Resume Dataset (data/raw/Resume/Resume.csv).

Run with: .venv/Scripts/python.exe notebooks/01_eda_resume_dataset.py
Written as a plain, reproducible script (spec.md Phase 3 explicitly allows
"notebooks/scripts") rather than a .ipynb so there is no notebook-output
diff noise in Git and no extra jupyter/ipykernel dependency is required.

Outputs:
- reports/figures/resume_category_distribution.png
- reports/figures/resume_length_distribution.png
- reports/figures/resume_top_skills.png
- reports/results/resume_eda_summary.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless, figure-saving only
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.text_cleaning import clean_text, handle_missing
from src.skills.skill_extractor import extract_skills
from src.utils.paths import FIGURES_DIR, RESULTS_DIR, RESUME_CSV_PATH, require_path

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_resume_dataset() -> pd.DataFrame:
    require_path(
        RESUME_CSV_PATH,
        "Download the Kaggle Resume Dataset and place Resume.csv at this path.",
    )
    return pd.read_csv(RESUME_CSV_PATH)


def main() -> None:
    df = load_resume_dataset()
    summary = {}

    summary["num_rows"] = int(len(df))
    summary["num_columns"] = int(df.shape[1])
    summary["columns"] = df.columns.tolist()

    # --- missing values & duplicates ---
    summary["missing_values_per_column"] = df.isna().sum().to_dict()
    summary["num_duplicate_resume_str"] = int(df["Resume_str"].duplicated().sum())
    summary["num_duplicate_ids"] = int(df["ID"].duplicated().sum())

    # --- category distribution ---
    category_counts = df["Category"].value_counts()
    summary["num_categories"] = int(category_counts.shape[0])
    summary["category_distribution"] = category_counts.to_dict()

    plt.figure(figsize=(10, 7))
    sns.barplot(y=category_counts.index, x=category_counts.values, color="#4C72B0")
    plt.title("Resume Count by Category (Kaggle Resume Dataset)")
    plt.xlabel("Number of resumes")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "resume_category_distribution.png", dpi=150)
    plt.close()

    # --- resume length (word count) distribution ---
    cleaned_text = df["Resume_str"].apply(lambda t: clean_text(handle_missing(t)))
    word_counts = cleaned_text.apply(lambda t: len(t.split()))
    summary["resume_length_words"] = {
        "mean": float(word_counts.mean()),
        "median": float(word_counts.median()),
        "min": int(word_counts.min()),
        "max": int(word_counts.max()),
        "std": float(word_counts.std()),
    }

    plt.figure(figsize=(8, 5))
    sns.histplot(word_counts, bins=40, color="#4C72B0")
    plt.title("Resume Length Distribution (word count)")
    plt.xlabel("Words per resume")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "resume_length_distribution.png", dpi=150)
    plt.close()

    # --- common skills (dictionary-based extraction across the corpus) ---
    print(f"Extracting skills from {len(df)} resumes (this may take a moment)...")
    all_skills = cleaned_text.apply(extract_skills)
    skill_counter = Counter(skill for skills in all_skills for skill in skills)
    top_skills = skill_counter.most_common(20)
    summary["top_20_detected_skills"] = dict(top_skills)
    summary["pct_resumes_with_zero_detected_skills"] = float(
        (all_skills.apply(len) == 0).mean() * 100
    )

    plt.figure(figsize=(8, 7))
    skill_names, skill_freqs = zip(*top_skills)
    sns.barplot(y=list(skill_names), x=list(skill_freqs), color="#55A868")
    plt.title("Top 20 Detected Skills Across All Resumes")
    plt.xlabel("Number of resumes mentioning this skill")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "resume_top_skills.png", dpi=150)
    plt.close()

    # --- category-specific vocabulary: top detected skills per category ---
    df_with_skills = df.copy()
    df_with_skills["_skills"] = all_skills
    category_top_skills = {}
    for category, group in df_with_skills.groupby("Category"):
        counter = Counter(skill for skills in group["_skills"] for skill in skills)
        category_top_skills[category] = dict(counter.most_common(5))
    summary["top_5_skills_per_category"] = category_top_skills

    out_path = RESULTS_DIR / "resume_eda_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary -> {out_path}")
    print(f"Saved figures -> {FIGURES_DIR}")
    print(f"Rows: {summary['num_rows']}, Categories: {summary['num_categories']}")
    print(f"Duplicate Resume_str rows: {summary['num_duplicate_resume_str']}")
    print(f"Mean resume length (words): {summary['resume_length_words']['mean']:.1f}")
    print(f"Resumes with zero detected dictionary skills: {summary['pct_resumes_with_zero_detected_skills']:.1f}%")


if __name__ == "__main__":
    main()
