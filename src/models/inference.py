"""Inference-time loading of trained artifacts (spec.md section 30/36).

The deployed app NEVER retrains anything -- it only loads what
src/models/train_supervised.py already saved under models/. If an
artifact is missing (e.g. a fresh clone before training has been run),
callers fall back to the transparent weighted hybrid score instead of
crashing.
"""

from functools import lru_cache
from typing import Optional

import joblib
import pandas as pd

from src.matching.hybrid import compute_weighted_hybrid_score
from src.utils.paths import MODELS_DIR

FEATURE_COLUMNS = [
    "skill_coverage",
    "num_matched_skills",
    "num_missing_skills",
    "tfidf_similarity",
    "semantic_similarity",
    "resume_length",
    "job_description_length",
    "experience_compatibility",
    "education_compatibility",
]

RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"


@lru_cache(maxsize=1)
def load_final_model_artifacts() -> Optional[dict]:
    """Load the final selected supervised model (Random Forest -- chosen
    in src/models/train_supervised.py based on validation F1/ROC-AUC, see
    reports/results/model_comparison.json) plus its feature scaler.
    Returns None if training has not been run yet in this environment.
    """
    if not RF_MODEL_PATH.exists() or not SCALER_PATH.exists():
        return None
    return {
        "model": joblib.load(RF_MODEL_PATH),
        "scaler": joblib.load(SCALER_PATH),
    }


def predict_match(features: dict) -> dict:
    """Return a 0-100 match score and predicted label for one feature dict.

    Uses the trained Random Forest when available (score = P(strong)*100 +
    P(moderate)*50, matching the scoring convention used during ranking
    evaluation); otherwise falls back to the documented weighted hybrid
    score so the app still functions on a machine where training hasn't
    been run.
    """
    artifacts = load_final_model_artifacts()

    if artifacts is None:
        hybrid = compute_weighted_hybrid_score(features)
        return {
            "score_percent": round(hybrid * 100, 1),
            "predicted_label": None,
            "method": "weighted_hybrid_fallback",
        }

    model = artifacts["model"]
    scaler = artifacts["scaler"]

    row = pd.DataFrame([{col: features.get(col, 0.0) or 0.0 for col in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    row_scaled = scaler.transform(row)

    probs = model.predict_proba(row_scaled)[0]
    classes = list(model.classes_)
    strong_p = probs[classes.index("strong")] if "strong" in classes else 0.0
    moderate_p = probs[classes.index("moderate")] if "moderate" in classes else 0.0
    score = strong_p * 100 + moderate_p * 50

    predicted_label = model.predict(row_scaled)[0]

    return {
        "score_percent": round(float(score), 1),
        "predicted_label": predicted_label,
        "method": "random_forest",
        "class_probabilities": {c: round(float(p), 4) for c, p in zip(classes, probs)},
    }
