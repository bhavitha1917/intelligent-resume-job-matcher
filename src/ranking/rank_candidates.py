"""Phase 15 -- candidate ranking.

Ties together parsing, skill extraction, feature engineering, scoring,
skill-gap analysis and explanation generation into one function used by
both the Streamlit app and the evaluation notebooks.
"""

from typing import Callable, Optional

from src.data.job_parser import parse_job_description_text
from src.data.resume_parser import parse_resume
from src.features.feature_engineering import build_feature_vector
from src.matching.explainability import generate_explanation
from src.matching.hybrid import compute_weighted_hybrid_score
from src.matching.skill_gap import get_skill_gap
from src.models.inference import load_final_model_artifacts, predict_match


def _default_scorer(features: dict) -> float:
    """Use the trained Random Forest (the final model selected in
    src/models/train_supervised.py on validation performance) when its
    artifacts are available; otherwise fall back to the transparent
    weighted hybrid score so the app still works before training has run.
    """
    if load_final_model_artifacts() is not None:
        return predict_match(features)["score_percent"] / 100.0
    return compute_weighted_hybrid_score(features)


def _default_feature_importance() -> Optional[dict]:
    artifacts = load_final_model_artifacts()
    if artifacts is None:
        return None
    from src.models.inference import FEATURE_COLUMNS

    model = artifacts["model"]
    return dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))


def rank_candidates(
    job_description: str,
    resumes: list,
    tfidf_vectorizer=None,
    compute_semantic: bool = True,
    scorer: Optional[Callable[[dict], float]] = None,
) -> list:
    """Rank N resumes against one job description.

    `resumes`: list of dicts, each with:
        - "candidate_id": any hashable identifier for display
        - "source": a file path, raw bytes, or raw text (see parse_resume)
        - "filename": optional, required if source is bytes
    `scorer`: optional callable(features_dict) -> float in [0, 1]; defaults
    to the transparent weighted hybrid score. Pass a wrapped supervised
    model's predict function here to rank with the learned model instead.

    Returns a list of result dicts sorted by descending overall_score, each
    with an added "rank" field (1 = best match).
    """
    job_parsed = parse_job_description_text(job_description)
    score_fn = scorer or _default_scorer
    feature_importance = _default_feature_importance() if scorer is None else None

    results = []
    for resume_entry in resumes:
        candidate_id = resume_entry.get("candidate_id")
        resume_parsed = parse_resume(resume_entry["source"], resume_entry.get("filename"))

        feature_result = build_feature_vector(
            resume_full_text=resume_parsed["full_text"],
            job_description_text=job_parsed["description"],
            required_skills=job_parsed["required_skills"],
            preferred_skills=job_parsed["preferred_skills"],
            resume_experience_text=resume_parsed["experience"],
            job_experience_requirement=job_parsed["experience_requirement"],
            resume_education_text=resume_parsed["education"],
            job_education_requirement=job_parsed["education_requirement"],
            tfidf_vectorizer=tfidf_vectorizer,
            compute_semantic=compute_semantic,
        )
        features = feature_result["features"]
        overall_score = score_fn(features)

        skill_gap = get_skill_gap(
            resume_parsed["full_text"], job_parsed["required_skills"], job_parsed["preferred_skills"]
        )
        explanation = generate_explanation(
            features,
            match_score=overall_score,
            missing_required_skills=[item["skill"] for item in skill_gap["missing_required_skills"]],
            feature_importance=feature_importance,
        )

        results.append({
            "candidate_id": candidate_id,
            "overall_score": overall_score,
            "skill_score": features["skill_coverage"],
            "semantic_score": features["semantic_similarity"],
            "tfidf_score": features["tfidf_similarity"],
            "matched_skills": feature_result["matched_skills"],
            "missing_skills": feature_result["missing_skills"],
            "skill_gap": skill_gap,
            "explanation": explanation,
            "parse_warnings": resume_parsed["parse_warnings"],
        })

    results.sort(key=lambda r: r["overall_score"], reverse=True)
    for i, result in enumerate(results, start=1):
        result["rank"] = i

    return results
