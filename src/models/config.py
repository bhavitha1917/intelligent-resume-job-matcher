"""Model / target configuration (spec.md sections 18, 29).

Centralized here (rather than hardcoded inside the training script) so the
thresholds are documented, easy to find, and easy to change. A copy of
these values is also written to models/training_config.json at training
time for reproducibility.
"""

RANDOM_STATE = 42

# ai_match_score is on a 0-100 scale (see reports/results/resume_job_fit_eda_summary.json).
# Thresholds chosen from the observed distribution (mean ~44, std ~25) to
# produce three non-trivial classes rather than an arbitrary 50/50 split.
STRONG_MATCH_THRESHOLD = 70  # ai_match_score >= 70  -> "strong"
MODERATE_MATCH_THRESHOLD = 40  # 40 <= ai_match_score < 70 -> "moderate"; below -> "weak"

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)


def score_to_label(score: float) -> str:
    """Map a continuous 0-100 dataset-provided match score to a 3-class
    label. This is a documented, configurable strategy -- NOT a claim that
    the dataset score represents an actual human hiring decision
    (spec.md section 4.3 / section 18).
    """
    if score >= STRONG_MATCH_THRESHOLD:
        return "strong"
    if score >= MODERATE_MATCH_THRESHOLD:
        return "moderate"
    return "weak"
