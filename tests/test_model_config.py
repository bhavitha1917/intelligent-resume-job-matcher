from src.models.config import MODERATE_MATCH_THRESHOLD, STRONG_MATCH_THRESHOLD, score_to_label


def test_score_to_label_boundaries():
    assert score_to_label(STRONG_MATCH_THRESHOLD) == "strong"
    assert score_to_label(STRONG_MATCH_THRESHOLD - 0.01) == "moderate"
    assert score_to_label(MODERATE_MATCH_THRESHOLD) == "moderate"
    assert score_to_label(MODERATE_MATCH_THRESHOLD - 0.01) == "weak"
    assert score_to_label(0) == "weak"
    assert score_to_label(100) == "strong"
