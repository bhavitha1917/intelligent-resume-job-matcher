"""Direct tests of src/models/inference.py (predict_match had no tests in v1)
and of the training/inference feature contract.

Regression fixture: the feature vector reported by the earlier audit (the
original resume / job description are not available and are NOT
reconstructed here):
    skill_coverage 0.3333, num_matched 2, num_missing 4, tfidf 0.3566,
    semantic 0.2972, experience 0.5, education 0.5, title/category 0.5
"""

import json

import numpy as np
import pandas as pd
import pytest
from _fixtures import REFERENCE_DATE, RESUME_PARTIAL, RESUME_STRONG, job, parsed

import src.models.inference as inference
from src.evaluation.legacy_v1 import legacy_blend_score, legacy_skill_features
from src.features.feature_engineering import FEATURE_COLUMNS, model_features_from_report
from src.scoring.requirements import requirements_from_dataset_record
from src.scoring.scorer import score_candidate
from src.utils.paths import REPORTS_DIR

AUDIT_FIXTURE = {
    "skill_coverage": 0.3333, "num_matched_skills": 2, "num_missing_skills": 4, "tfidf_similarity": 0.3566,
    "semantic_similarity": 0.2972, "experience_compatibility": 0.5, "education_compatibility": 0.5,
    "title_category_compatibility": 0.5,
}


class FakeModel:
    """Classes deliberately in a non-alphabetical, non-label order."""
    classes_ = np.array(["strong", "weak", "moderate"])

    def predict_proba(self, X):
        return np.array([[0.2, 0.5, 0.3]])


def _fixture_as_v2_features():
    """The audit vector expressed in the v2 schema: unknown experience /
    education become credit 0 + applicable 0, not a silent 0.5."""
    return {
        "required_skill_credit": 0.3333, "required_skill_detected_fraction": 2 / 6, "required_skill_applied_fraction": 0.0,
        "experience_credit": 0.0, "experience_applicable": 0.0, "education_credit": 0.0, "education_applicable": 0.0,
        "certification_credit": 0.0, "certification_applicable": 0.0, "preferred_credit": 0.0, "preferred_applicable": 0.0,
        "semantic_similarity": AUDIT_FIXTURE["semantic_similarity"], "tfidf_similarity": AUDIT_FIXTURE["tfidf_similarity"],
    }


def test_v1_display_formula_regression_documented():
    """The baseline recorded P(moderate)=0.515, P(strong)=0.325 for the
    fixture and displayed 58.2 'match'. That formula is kept only in the
    legacy module; the app no longer computes it."""
    baseline = json.loads((REPORTS_DIR / "baseline" / "baseline_scores.json").read_text(encoding="utf-8"))
    probs = baseline["fixture_predict_match"]["class_probabilities"]
    assert legacy_blend_score(probs) == pytest.approx(baseline["fixture_predict_match"]["score_percent"], abs=0.06)
    assert baseline["fixture_features"] == AUDIT_FIXTURE


def test_class_probabilities_follow_model_classes_not_column_position():
    probs = inference.ordered_probabilities(FakeModel(), pd.DataFrame([[0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS))
    assert probs == {"weak": 0.5, "moderate": 0.3, "strong": 0.2}


def test_predicted_label_is_argmax_and_no_percentage_is_produced():
    artifacts = {"ok": True, "model": FakeModel(), "model_name": "fake"}
    out = inference.predict_match(_fixture_as_v2_features(), artifacts)
    assert out["predicted_label"] == "weak"
    assert "score_percent" not in out and "accuracy" not in out
    assert "Not a hiring probability" in out["disclaimer"]


def test_missing_features_raise_instead_of_silently_defaulting():
    features = _fixture_as_v2_features()
    features["semantic_similarity"] = None
    with pytest.raises(ValueError):
        inference.predict_match(features, {"ok": True, "model": FakeModel(), "model_name": "fake"})
    del features["semantic_similarity"]
    with pytest.raises(ValueError):
        inference.predict_match(features, {"ok": True, "model": FakeModel(), "model_name": "fake"})


def test_incompatible_manifest_is_refused(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "feature_schema_version": "v1", "feature_columns": list(AUDIT_FIXTURE), "class_labels": inference.CLASS_LABELS}))
    monkeypatch.setattr(inference, "MODELS_V2_DIR", tmp_path)
    inference.load_v2_artifacts.cache_clear()
    try:
        status = inference.load_v2_artifacts()
        assert not status["ok"] and "schema" in status["reason"]
        assert inference.predict_match(_fixture_as_v2_features())["available"] is False
    finally:
        inference.load_v2_artifacts.cache_clear()


def test_legacy_v1_artifacts_are_reported_as_unused():
    status = inference.legacy_artifact_status()
    assert status["used"] is False


def test_trained_v2_model_on_audit_fixture_if_available():
    artifacts = inference.load_v2_artifacts()
    if not artifacts["ok"]:
        pytest.skip(artifacts["reason"])
    out = inference.predict_match(_fixture_as_v2_features(), artifacts)
    assert out["predicted_label"] in inference.CLASS_LABELS
    assert sum(out["class_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["predicted_label"] == max(out["class_probabilities"], key=out["class_probabilities"].get)


def test_model_features_have_no_missing_values_and_explicit_indicators():
    report = score_candidate(job(), parsed(RESUME_PARTIAL), semantic=lambda a, b: 0.4, reference_date=REFERENCE_DATE)
    features = model_features_from_report(report, tfidf_similarity=0.2)
    assert list(features) == FEATURE_COLUMNS
    assert all(isinstance(v, float) for v in features.values())
    no_cert = score_candidate(job(), parsed(RESUME_PARTIAL), semantic=lambda a, b: 0.4, reference_date=REFERENCE_DATE)
    f = model_features_from_report(no_cert, 0.2)
    assert f["certification_applicable"] == 0.0 and f["certification_credit"] == 0.0


def test_model_features_require_semantic_similarity():
    report = score_candidate(job(), parsed(RESUME_PARTIAL), semantic=None, reference_date=REFERENCE_DATE)
    with pytest.raises(ValueError):
        model_features_from_report(report, 0.1)


def test_training_and_inference_skill_coverage_agree():
    """The training pipeline builds requirements from the dataset's raw
    skill strings; the app builds them from a typed job description. For
    the same skills, coverage must be identical. (v1 got 0.0 here.)"""
    raw = ["sql", "python", "tableau", "statistics"]
    train_job = {"title": "Data Analyst", "text": "", "requirements": requirements_from_dataset_record("", raw)}
    app_job = job("Data Analyst\nRequirements:\n- SQL\n- Python\n- Tableau\n- Statistics\n")
    train_report = score_candidate(train_job, parsed(RESUME_STRONG), semantic=None, reference_date=REFERENCE_DATE)
    app_report = score_candidate(app_job, parsed(RESUME_STRONG), semantic=None, reference_date=REFERENCE_DATE)
    train_cov = next(c for c in train_report["components"] if c["name"] == "required_skills")["value"]
    app_cov = next(c for c in app_report["components"] if c["name"] == "required_skills")["value"]
    assert train_cov == app_cov == 1.0
    assert legacy_skill_features(RESUME_STRONG, raw)["skill_coverage"] == 0.0  # the documented v1 defect
