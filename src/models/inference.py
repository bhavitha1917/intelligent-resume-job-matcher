"""Inference-time loading of the (optional, experimental) v2 learned models.

What the learned model output IS: an estimate of which bucket
(weak / moderate / strong) the dataset's own algorithmic `ai_match_score`
would fall into for this pair, from a model trained on 23 job
descriptions. It is shown separately from the rubric score, as a
diagnostic, and is never used for the displayed score or the ranking.

What it is NOT: a match percentage, a hiring probability, a calibrated
probability, or the model's accuracy. The v1 display formula
(P(strong)*100 + P(moderate)*50) has been removed.

Compatibility: artifacts are loaded only from models/v2/ and only if
models/v2/manifest.json declares exactly the feature schema this code
computes. The v1 artifacts in models/*.joblib were trained on features
with different semantics (see src/evaluation/legacy_v1.py) and are never
loaded by the application.
"""

import json
from functools import lru_cache
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, model_features_from_report
from src.utils.paths import MODELS_DIR, MODELS_V2_DIR

CLASS_LABELS = ["weak", "moderate", "strong"]
MODEL_DISCLAIMER = (
    "Experimental: estimate of the dataset's algorithmic weak/moderate/strong label, from a model trained on "
    "only 23 job descriptions. Not used for the score or ranking. Not a hiring probability or an accuracy figure."
)
LEGACY_V1_FILES = ["random_forest.joblib", "logistic_regression.joblib", "svm.joblib", "feature_scaler.joblib"]


def legacy_artifact_status() -> dict:
    """Explain why the v1 artifacts present in models/ are not used."""
    present = [f for f in LEGACY_V1_FILES if (MODELS_DIR / f).exists()]
    return {
        "present": present,
        "used": False,
        "reason": "v1 models were trained on features whose semantics changed (skill coverage computed "
                  "inconsistently, 0.5 'unknown' fallbacks, job-identifying length feature). Loading them with "
                  "v2 features would silently produce meaningless predictions.",
    }


@lru_cache(maxsize=1)
def load_v2_artifacts(model_name: Optional[str] = None) -> dict:
    """Returns {"ok": bool, "reason": str, ...artifacts}. Never raises."""
    manifest_path = MODELS_V2_DIR / "manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "reason": "No v2 model artifacts found (run `python -m src.models.train_supervised`)."}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"ok": False, "reason": f"Unreadable model manifest: {exc}"}

    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or manifest.get("feature_columns") != FEATURE_COLUMNS:
        return {"ok": False, "reason": "Model manifest feature schema does not match the current code; refusing to load."}
    if manifest.get("class_labels") != CLASS_LABELS:
        return {"ok": False, "reason": "Model manifest class labels do not match; refusing to load."}

    name = model_name or manifest.get("display_model", "random_forest")
    model_path = MODELS_V2_DIR / f"{name}.joblib"
    vec_path = MODELS_V2_DIR / "tfidf_vectorizer.joblib"
    if not model_path.exists() or not vec_path.exists():
        return {"ok": False, "reason": f"Missing v2 artifact(s) for '{name}'."}
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vec_path)
    except Exception as exc:  # version mismatch, corrupted file, ...
        return {"ok": False, "reason": f"Could not load v2 artifacts: {type(exc).__name__}: {exc}"}
    return {"ok": True, "reason": "", "model": model, "vectorizer": vectorizer, "manifest": manifest, "model_name": name}


def ordered_probabilities(model, row: pd.DataFrame) -> dict:
    """Class probabilities keyed by class name, using model.classes_ (never
    assuming column order)."""
    proba = model.predict_proba(row)[0]
    classes = list(model.classes_)
    return {label: float(proba[classes.index(label)]) if label in classes else 0.0 for label in CLASS_LABELS}


def predict_match(features: dict, artifacts: Optional[dict] = None) -> dict:
    """Predict the dataset-label bucket for one v2 feature dict.

    Raises ValueError if any feature is missing or None -- missing values
    must be encoded explicitly (see feature_engineering), never silently
    defaulted. The predicted label is always the argmax of the returned
    probabilities, so the two can never disagree.
    """
    artifacts = artifacts or load_v2_artifacts()
    if not artifacts.get("ok"):
        return {"available": False, "reason": artifacts.get("reason")}

    missing = [c for c in FEATURE_COLUMNS if features.get(c) is None]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    row = pd.DataFrame([{c: float(features[c]) for c in FEATURE_COLUMNS}], columns=FEATURE_COLUMNS)
    probabilities = ordered_probabilities(artifacts["model"], row)
    predicted = CLASS_LABELS[int(np.argmax([probabilities[c] for c in CLASS_LABELS]))]
    return {
        "available": True,
        "model": artifacts["model_name"],
        "predicted_label": predicted,
        "class_probabilities": {c: round(p, 4) for c, p in probabilities.items()},
        "disclaimer": MODEL_DISCLAIMER,
    }


def model_estimate_for_report(report: dict, resume_sanitized_text: str, job_text: str) -> dict:
    """Experimental learned-model estimate for an already-scored report."""
    artifacts = load_v2_artifacts()
    if not artifacts.get("ok"):
        return {"available": False, "reason": artifacts.get("reason")}
    if report.get("semantic", {}).get("raw") is None:
        return {"available": False, "reason": "Semantic similarity was unavailable, which the model requires."}
    vec = artifacts["vectorizer"].transform([resume_sanitized_text, job_text])
    tfidf = float(vec[0].multiply(vec[1]).sum())
    return predict_match(model_features_from_report(report, tfidf), artifacts)
