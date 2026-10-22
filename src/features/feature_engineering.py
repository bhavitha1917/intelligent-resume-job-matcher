"""Feature engineering for the (optional, experimental) learned models.

v2 features are derived ONLY from a scoring report produced by
src.scoring.scorer.score_candidate, plus a TF-IDF similarity. Because the
training pipeline and the app both build reports with the same function,
required-skill canonicalization, matching, counting and coverage are
identical at training and inference time by construction.

Design rules:
- No PII / protected attributes: reports are computed on sanitized text.
- No job-identifying features: the v1 `job_description_length` and raw
  skill counts (num_missing_skills varies mainly with the job's list
  length) are dropped; only per-job-normalized fractions are used.
- Missing information is explicit: every component that may be
  inapplicable has a value (0.0 when inapplicable) AND an `_applicable`
  0/1 indicator, instead of v1's silent 0.5 fallback.
"""

FEATURE_SCHEMA_VERSION = "v2"

FEATURE_COLUMNS = [
    "required_skill_credit",
    "required_skill_detected_fraction",
    "required_skill_applied_fraction",
    "experience_credit",
    "experience_applicable",
    "education_credit",
    "education_applicable",
    "certification_credit",
    "certification_applicable",
    "preferred_credit",
    "preferred_applicable",
    "semantic_similarity",
    "tfidf_similarity",
]

_COMPONENT_FEATURES = {
    "experience": "experience",
    "education": "education",
    "certifications": "certification",
    "preferred_qualifications": "preferred",
}


def model_features_from_report(report: dict, tfidf_similarity: float) -> dict:
    """Map one scoring report to the v2 feature dict (all keys of
    FEATURE_COLUMNS, no None values)."""
    components = {c["name"]: c for c in report["components"]}
    required_skills = [r for r in report["requirements"] if r["kind"] == "skill" and r["importance"] == "required"]
    n = len(required_skills)
    detected = sum(r["result"]["status"] == "met" for r in required_skills)
    applied = sum(r["result"].get("evidence_level") == "applied" for r in required_skills)

    skill_component = components["required_skills"]
    features = {
        "required_skill_credit": float(skill_component["value"]) if skill_component["applicable"] else 0.0,
        "required_skill_detected_fraction": detected / n if n else 0.0,
        "required_skill_applied_fraction": applied / n if n else 0.0,
    }
    for component_name, prefix in _COMPONENT_FEATURES.items():
        comp = components[component_name]
        features[f"{prefix}_credit"] = float(comp["value"]) if comp["applicable"] else 0.0
        features[f"{prefix}_applicable"] = 1.0 if comp["applicable"] else 0.0

    semantic_raw = report["semantic"]["raw"]
    if semantic_raw is None:
        raise ValueError("The v2 learned models require semantic similarity; it was unavailable for this report.")
    features["semantic_similarity"] = float(semantic_raw)
    features["tfidf_similarity"] = float(tfidf_similarity)
    return features
