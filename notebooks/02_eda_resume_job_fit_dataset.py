"""EDA: Hugging Face `batuhanmtl/job_resume_fit` dataset.

This dataset also stands in for the "job dataset" EDA required by spec.md
Phase 3, because the Kaggle LinkedIn Job Postings dataset has not been
downloaded yet (see data/README.md). `job_text` / `job_required_skills`
are analyzed here as the job-posting-side corpus; this substitution is
documented, not silently assumed.

IMPORTANT: ai_match_score / skill_string_match_score / fuzzy_match_score
are dataset-provided matching signals, not human recruiter decisions
(spec.md section 4.3). This script treats them only as such.

Run with: .venv/Scripts/python.exe notebooks/02_eda_resume_job_fit_dataset.py

Outputs:
- reports/figures/fit_category_distribution.png
- reports/figures/fit_text_length_distributions.png
- reports/figures/fit_score_distributions.png
- reports/figures/fit_score_correlation_heatmap.png
- reports/figures/fit_top_job_required_skills.png
- reports/results/resume_job_fit_eda_summary.json
"""

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import FIGURES_DIR, RESULTS_DIR

sns.set_theme(style="whitegrid")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET_NAME = "batuhanmtl/job_resume_fit"


def _parse_stringified_list(value) -> list:
    if isinstance(value, list):
        return value
    if not value or not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def load_fit_dataset() -> pd.DataFrame:
    dataset = load_dataset(HF_DATASET_NAME)
    return dataset["train"].to_pandas()


def main() -> None:
    df = load_fit_dataset()
    summary = {"dataset_name": HF_DATASET_NAME}

    summary["num_rows"] = int(len(df))
    summary["num_columns"] = int(df.shape[1])
    summary["columns"] = df.columns.tolist()
    summary["missing_values_per_column"] = df.isna().sum().to_dict()
    summary["num_duplicate_ids"] = int(df["ID"].duplicated().sum())
    summary["num_duplicate_resume_text"] = int(df["resume_text"].duplicated().sum())

    # --- category distribution ---
    category_counts = df["category"].value_counts()
    summary["num_categories"] = int(category_counts.shape[0])
    summary["category_distribution"] = category_counts.to_dict()

    plt.figure(figsize=(10, 7))
    sns.barplot(y=category_counts.index, x=category_counts.values, color="#4C72B0")
    plt.title("Resume-Job Pair Count by Category (HF job_resume_fit)")
    plt.xlabel("Number of pairs")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fit_category_distribution.png", dpi=150)
    plt.close()

    # --- text length distributions (resume_text and job_text) ---
    resume_lengths = df["resume_text"].fillna("").apply(lambda t: len(str(t).split()))
    job_lengths = df["job_text"].fillna("").apply(lambda t: len(str(t).split()))
    summary["resume_text_length_words"] = {
        "mean": float(resume_lengths.mean()), "median": float(resume_lengths.median()),
        "min": int(resume_lengths.min()), "max": int(resume_lengths.max()),
    }
    summary["job_text_length_words"] = {
        "mean": float(job_lengths.mean()), "median": float(job_lengths.median()),
        "min": int(job_lengths.min()), "max": int(job_lengths.max()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(resume_lengths, bins=40, color="#4C72B0", ax=axes[0])
    axes[0].set_title("resume_text Length (words)")
    sns.histplot(job_lengths, bins=40, color="#DD8452", ax=axes[1])
    axes[1].set_title("job_text Length (words)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fit_text_length_distributions.png", dpi=150)
    plt.close()

    # --- matching score distributions ---
    score_cols = ["ai_match_score", "skill_string_match_score", "fuzzy_match_score"]
    summary["score_statistics"] = {
        col: {
            "mean": float(df[col].mean()), "median": float(df[col].median()),
            "std": float(df[col].std()), "min": float(df[col].min()), "max": float(df[col].max()),
        }
        for col in score_cols
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, score_cols):
        sns.histplot(df[col], bins=30, color="#55A868", ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fit_score_distributions.png", dpi=150)
    plt.close()

    # --- relationship between the three matching scores ---
    correlation = df[score_cols].corr()
    summary["score_correlation"] = correlation.round(3).to_dict()

    plt.figure(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Dataset-Provided Matching Scores")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fit_score_correlation_heatmap.png", dpi=150)
    plt.close()

    # --- most common required skills requested across job postings ---
    required_skills_lists = df["job_required_skills"].apply(_parse_stringified_list)
    skill_counter = Counter(skill for skills in required_skills_lists for skill in skills)
    top_skills = skill_counter.most_common(20)
    summary["top_20_job_required_skills"] = dict(top_skills)

    plt.figure(figsize=(8, 7))
    names, freqs = zip(*top_skills)
    sns.barplot(y=list(names), x=list(freqs), color="#C44E52")
    plt.title("Top 20 Required Skills Across Job Postings (dataset-provided field)")
    plt.xlabel("Number of job postings requiring this skill")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fit_top_job_required_skills.png", dpi=150)
    plt.close()

    out_path = RESULTS_DIR / "resume_job_fit_eda_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary -> {out_path}")
    print(f"Rows: {summary['num_rows']}, Categories: {summary['num_categories']}")
    print(f"ai_match_score mean/std: {summary['score_statistics']['ai_match_score']['mean']:.1f} / "
          f"{summary['score_statistics']['ai_match_score']['std']:.1f}")
    print("Score correlation (ai_match_score vs fuzzy_match_score): "
          f"{correlation.loc['ai_match_score', 'fuzzy_match_score']:.3f}")


if __name__ == "__main__":
    main()
