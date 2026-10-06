"""Phase 17 -- explainability.

Every statement generated here is derived directly from an already-computed
feature value passed in by the caller. Nothing here invents a reason that
isn't backed by a number (spec.md section 23: "Do not generate fabricated
explanations").
"""

from typing import Optional

# (feature_key, threshold, message_template) -- checked in this order.
_POSITIVE_RULES = [
    ("skill_coverage", 0.8, "High required-skill coverage ({value:.0%})"),
    ("skill_coverage", 0.5, "Moderate required-skill coverage ({value:.0%})"),
    ("semantic_similarity", 0.7, "Strong semantic similarity to the job description ({value:.2f})"),
    ("semantic_similarity", 0.5, "Reasonable semantic similarity to the job description ({value:.2f})"),
    ("tfidf_similarity", 0.3, "Notable lexical overlap with the job description ({value:.2f})"),
    ("experience_compatibility", 0.8, "Experience mentions appear to meet the stated requirement"),
    ("education_compatibility", 0.8, "Education level appears to meet the stated requirement"),
    ("preferred_skill_coverage", 0.5, "Covers a meaningful share of preferred (nice-to-have) skills ({value:.0%})"),
]


def generate_explanation(
    features: dict,
    match_score: float,
    missing_required_skills: Optional[list] = None,
    feature_importance: Optional[dict] = None,
) -> dict:
    """Build a feature-grounded explanation.

    `feature_importance`: optional {feature_name: importance} from the
    final trained supervised model (e.g. RandomForest.feature_importances_
    mapped back to names); when provided, the top contributing features
    are listed explicitly instead of only fixed threshold rules.
    """
    positive_factors = []
    seen_features = set()

    for feature_key, threshold, template in _POSITIVE_RULES:
        if feature_key in seen_features:
            continue
        value = features.get(feature_key)
        if value is None:
            continue
        if value >= threshold:
            positive_factors.append(template.format(value=value))
            seen_features.add(feature_key)

    missing_requirements = list(missing_required_skills or [])

    result = {
        "match_score": match_score,
        "positive_factors": positive_factors,
        "missing_requirements": missing_requirements,
    }

    if feature_importance:
        ranked = sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)[:5]
        result["top_model_features"] = [
            {"feature": name, "importance": round(float(importance), 4)} for name, importance in ranked
        ]

    if not positive_factors and not missing_requirements:
        result["note"] = "Insufficient signal to generate a detailed explanation for this pair."

    return result
