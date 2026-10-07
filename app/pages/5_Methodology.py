import _bootstrap  # noqa: F401
import streamlit as st

st.set_page_config(page_title="Methodology", page_icon="🧭", layout="wide")
st.title("Methodology / About")

st.markdown(
    """
## Datasets

1. **Kaggle Resume Dataset** (`data/raw/Resume/Resume.csv`) — 2,484 resumes
   across 24 professional categories. Used for resume-side EDA, parsing
   validation, and the bundled demo examples.
2. **Kaggle LinkedIn Job Postings 2023-2024** — specified by the project
   brief but **not downloaded** in this environment (see `data/README.md`).
   Job-side analysis currently falls back to the Hugging Face dataset's
   `job_text` field instead.
3. **Hugging Face `batuhanmtl/job_resume_fit`** — 2,385 paired resume/job
   examples with dataset-provided matching signals
   (`ai_match_score`, `skill_string_match_score`, `fuzzy_match_score`).
   **These are not human recruiter decisions** — they are the dataset's
   own generated signals, used here only to supervise a classifier.

## Preprocessing

Unicode/whitespace normalization and HTML stripping, deliberately **without**
removing punctuation — technical tokens like `C++`, `C#`, `.NET`, `Node.js`
must survive cleaning (`src/preprocessing/text_cleaning.py`).

## Skill extraction

A curated JSON dictionary (`data/skills/skill_dictionary.json`) with
canonical names and aliases (e.g. `ML` → `Machine Learning`), matched with
word-boundary-aware regex that correctly handles punctuation-containing
terms (`src/skills/skill_extractor.py`). This is deterministic and
testable — not an ML-based extractor — and it is a documented limitation
that it has no notion of negation (e.g. "no formal machine learning
experience" still contains the phrase "machine learning").

## Matching methods compared

1. **Keyword baseline** — skill coverage = matched required skills / total required skills.
2. **TF-IDF + cosine similarity** — vectorizer fit only on the training split.
3. **Sentence-Transformer semantic similarity** — `all-MiniLM-L6-v2`, pretrained and
   frozen (never fine-tuned), embeddings cached, cosine similarity between resume and job.
4. **Supervised models** — Logistic Regression, Random Forest, SVM, trained
   on engineered features (skill coverage, TF-IDF similarity, semantic
   similarity, resume/job length, experience/education compatibility — no
   PII, no protected attributes).
5. **Final hybrid** — a transparent, documented weighted combination
   (`src/matching/hybrid.py`) compared against the learned model.

See the **Model Comparison** page for the actual experimental numbers and
why Random Forest was selected as the final model.

## Target definition (spec.md section 18)

`ai_match_score` (0-100) is bucketed into `weak` / `moderate` / `strong`
using documented, configurable thresholds in `src/models/config.py`
(`STRONG_MATCH_THRESHOLD = 70`, `MODERATE_MATCH_THRESHOLD = 40`). A secondary
regression experiment predicts the continuous score directly.

## Leakage prevention (spec.md section 28)

- Exact-duplicate resumes are dropped before splitting.
- The TF-IDF vectorizer and feature scaler are fit **only** on the training split.
- Sentence-transformer embeddings come from a frozen pretrained model (not
  fit on this dataset), so precomputing them for all splits does not leak
  target information.

## Evaluation

Classification metrics (accuracy, precision/recall/F1 macro, ROC-AUC
one-vs-rest) plus ranking metrics (Precision@K, Recall@K, MRR, NDCG@K) on a
synthetic multi-candidate ranking task constructed from the test split
(true pair + 9 sampled distractors per query) — see
`src/evaluation/ranking_metrics.py` and
`reports/results/ranking_evaluation.json`.

## Limitations

- The skill dictionary is biased toward technical/data roles; ~38% of the
  Kaggle resume corpus (which spans categories like CHEF, TEACHER,
  AVIATION) has zero detected skills under it — see
  `reports/results/resume_eda_summary.json`.
- Dictionary-based skill extraction has no negation handling.
- The job-posting side of the pipeline currently substitutes the HF
  dataset's `job_text` field for the (not yet downloaded) LinkedIn
  postings dataset.
- `ai_match_score` reflects the HF dataset's own generation pipeline, not
  verified recruiter outcomes — all downstream metrics inherit that
  limitation.
- Resume datasets are not a representative sample of the real labor
  market or candidate population; results should not be assumed to
  generalize across industries, regions, or demographic groups.

## Ethical considerations

- No protected attribute (name, gender, age, photo, email, phone, address,
  religion, caste, nationality, marital status) is used as a model feature.
- The system is a **decision-support tool**. It does not and must not make
  autonomous hiring decisions.
- "Missing" a skill is always reported as *"Not detected in submitted
  resume"* rather than a claim that the candidate lacks that skill.
"""
)
