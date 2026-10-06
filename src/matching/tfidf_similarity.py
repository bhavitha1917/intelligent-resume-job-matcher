"""Baseline 2 (spec.md section 14): TF-IDF + cosine similarity.

Leakage rule (spec.md section 28): the vectorizer must be FIT only on
training data. `fit_tfidf_vectorizer` / `save_vectorizer` / `load_vectorizer`
support that training-time workflow; `compute_tfidf_similarity` only ever
TRANSFORMS with an already-fitted vectorizer, it never re-fits.
"""

from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.paths import MODELS_DIR

DEFAULT_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"


def fit_tfidf_vectorizer(
    corpus: Iterable[str],
    max_features: int = 20000,
    ngram_range: tuple = (1, 2),
) -> TfidfVectorizer:
    """Fit a TF-IDF vectorizer on a TRAINING corpus only.

    `sublinear_tf=True` dampens the effect of very long resumes/postings
    repeating the same term many times.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
    )
    vectorizer.fit(list(corpus))
    return vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path = DEFAULT_VECTORIZER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path = DEFAULT_VECTORIZER_PATH) -> TfidfVectorizer:
    if not path.exists():
        raise FileNotFoundError(
            f"No fitted TF-IDF vectorizer found at {path}. "
            "Run the training pipeline (notebooks/03_tfidf_and_models) first."
        )
    return joblib.load(path)


def compute_tfidf_similarity(resume_text: str, job_text: str, vectorizer: TfidfVectorizer) -> float:
    """Cosine similarity between a resume and a job description, using an
    already-fitted vectorizer. Returns 0.0 for empty input instead of NaN.
    """
    if not resume_text or not job_text or not resume_text.strip() or not job_text.strip():
        return 0.0
    vectors = vectorizer.transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(round(similarity, 4))


def compute_tfidf_similarity_adhoc(resume_text: str, job_text: str) -> float:
    """Fallback for interactive use when no pretrained vectorizer artifact
    is available: fits a throwaway vectorizer on just this one resume/job
    pair. This is NOT used for model evaluation (would leak), only for a
    standalone single-pair similarity number in the app when
    `models/tfidf_vectorizer.joblib` has not been produced yet.
    """
    if not resume_text or not job_text or not resume_text.strip() or not job_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform([resume_text, job_text])
    similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
    return float(round(similarity, 4))
