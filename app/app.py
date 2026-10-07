import _bootstrap  # noqa: F401  (adds project root to sys.path)
import streamlit as st

st.set_page_config(
    page_title="Resume-Job Matching | Home",
    page_icon="🧭",
    layout="wide",
)

st.title("Intelligent Resume–Job Matching, Candidate Ranking and Skill Gap Analysis")
st.caption("A decision-support NLP/ML system — not an autonomous hiring system.")

st.markdown(
    """
### What this project does

Given **one job description** and **one or more resumes**, this system:

1. Parses resume and job text (PDF or plain text).
2. Extracts skills using a curated, extensible skill dictionary.
3. Computes lexical similarity (keyword coverage, TF-IDF cosine) and
   semantic similarity (a pretrained Sentence Transformer).
4. Combines these into engineered features and scores each resume with a
   trained supervised model (Random Forest), selected experimentally over
   several alternatives — see the **Model Comparison** page.
5. Produces a match score, a ranked candidate table, a skill-gap analysis
   (matched vs. missing skills), and a feature-grounded explanation.

### How the system works (pipeline)

```
Resume (PDF/TXT)              Job description (text)
      │                              │
      ▼                              ▼
  Resume parser                 Job parser
      │                              │
      └─────────────┬────────────────┘
                     ▼
            Skill extraction (curated dictionary)
                     │
                     ▼
     Keyword coverage · TF-IDF similarity · Semantic similarity
                     │
                     ▼
            Feature engineering (no PII, no protected attributes)
                     │
                     ▼
   Trained supervised model (Random Forest) ──or── transparent hybrid score
                     │
                     ▼
        Match score · Ranking · Skill gap · Explanation
```

### Responsible use — please read

- This tool is a **decision-support aid**, not an autonomous hiring system.
  It must not be the sole basis for any hiring decision.
- It never uses name, gender, age, photo, email, phone number, address,
  religion, caste, nationality, or marital status for scoring.
- "Missing" a skill means **"not detected in the submitted resume,"** not
  "the candidate does not have this skill." Resumes are noisy, partial
  documents.
- The underlying training data (`ai_match_score` and related fields from
  the Hugging Face `job_resume_fit` dataset) are **dataset-provided
  matching signals**, not verified human recruiter decisions. Treat all
  scores as indicative, not authoritative.
- Resume datasets are known to be imperfect samples of the real labor
  market; results should not be assumed to generalize to every industry,
  region, or demographic group. See the **Methodology** page for a fuller
  discussion of limitations and fairness considerations.

### Where to go next

- **Single Resume Analysis** — score one resume against one job description.
- **Candidate Ranking** — rank several resumes for the same job.
- **Candidate Details** — drill into one ranked candidate's full breakdown.
- **Model Comparison** — see the experimental evidence behind model choice.
- **Methodology** — datasets, preprocessing, modeling, and evaluation details.
"""
)
