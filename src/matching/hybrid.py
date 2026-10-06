"""Phase 14 -- final hybrid matching score.

Two strategies are provided and compared experimentally in
notebooks/05_model_comparison.ipynb (spec.md section 21):

1. `compute_weighted_hybrid_score`: a transparent, manually-weighted
   combination of engineered features. Weights are documented constants
   below, not hidden magic numbers.
2. A learned supervised model (src/models/train_supervised.py), loaded and
   applied separately -- see src/ranking/rank_candidates.py which can use
   either strategy via the `scorer` argument.

Weight documentation: skill coverage is weighted highest because it is the
most directly interpretable and literally required by most job postings;
semantic similarity is weighted second because it captures paraphrased
requirements that lexical/skill matching misses; TF-IDF, experience and
education contribute smaller, supporting signal.
"""

HYBRID_WEIGHTS = {
    "skill_coverage": 0.35,
    "semantic_similarity": 0.30,
    "tfidf_similarity": 0.15,
    "experience_compatibility": 0.10,
    "education_compatibility": 0.10,
}


def compute_weighted_hybrid_score(features: dict) -> float:
    """Weighted sum of available features, renormalized over whatever
    subset of HYBRID_WEIGHTS keys has a non-None value in `features` (e.g.
    semantic_similarity may be skipped in fast/offline contexts).
    """
    available = {
        key: features[key]
        for key in HYBRID_WEIGHTS
        if features.get(key) is not None
    }
    if not available:
        return 0.0

    weight_sum = sum(HYBRID_WEIGHTS[key] for key in available)
    score = sum(HYBRID_WEIGHTS[key] * available[key] for key in available) / weight_sum
    return round(max(0.0, min(1.0, score)), 4)
