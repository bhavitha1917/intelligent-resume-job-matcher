from src.data.job_parser import parse_job_description_record, parse_job_description_text
from src.features.feature_engineering import build_feature_vector
from src.matching.explainability import generate_explanation
from src.matching.hybrid import compute_weighted_hybrid_score
from src.matching.keyword_baseline import keyword_match
from src.matching.skill_gap import NOT_DETECTED_PHRASE, get_skill_gap
from src.matching.tfidf_similarity import compute_tfidf_similarity_adhoc, fit_tfidf_vectorizer


# ---- job_parser ----

def test_parse_job_description_text_extracts_sections():
    jd = (
        "Data Scientist\n\n"
        "Required skills:\nPython, SQL\n\n"
        "Preferred skills:\nAWS, Docker\n\n"
        "Experience:\n3+ years of experience required.\n\n"
        "Education:\nBachelor's degree in CS.\n"
    )
    result = parse_job_description_text(jd)
    assert result["title"] == "Data Scientist"
    assert "Python" in result["required_skills"]
    assert "SQL" in result["required_skills"]
    assert "AWS" in result["preferred_skills"]
    assert result["experience_requirement"].startswith("3+")
    assert "bachelor" in result["education_requirement"].lower()


def test_parse_job_description_text_empty_is_flagged():
    result = parse_job_description_text("")
    assert result["parse_warnings"]


def test_parse_job_description_record_from_structured_fields():
    result = parse_job_description_record(
        title="ML Engineer",
        description="Build ML systems",
        required_skills=["Python", "PyTorch"],
    )
    assert result["required_skills"] == ["PyTorch", "Python"]


# ---- keyword_baseline ----

def test_keyword_match_coverage():
    result = keyword_match("Python, SQL, PyTorch", ["Python", "SQL", "Docker", "AWS", "PyTorch"])
    assert result["skill_coverage"] == 0.6
    assert set(result["matched_skills"]) == {"Python", "SQL", "PyTorch"}
    assert set(result["missing_skills"]) == {"Docker", "AWS"}


def test_keyword_match_no_required_skills_is_flagged_not_crash():
    result = keyword_match("Python", [])
    assert result["skill_coverage"] == 0.0
    assert result["warning"] is not None


# ---- tfidf ----

def test_tfidf_similarity_identical_text_is_high():
    sim = compute_tfidf_similarity_adhoc("Python Machine Learning SQL", "Python Machine Learning SQL")
    assert sim > 0.9


def test_tfidf_similarity_empty_text_returns_zero():
    assert compute_tfidf_similarity_adhoc("", "Python SQL") == 0.0


def test_fit_tfidf_vectorizer_only_on_training_corpus():
    vectorizer = fit_tfidf_vectorizer(["Python SQL", "Java Spring", "Python SQL Python SQL"])
    assert vectorizer is not None
    # vectorizer must already be fit (transform should not raise)
    vectorizer.transform(["new unseen text"])


# ---- hybrid ----

def test_hybrid_score_is_bounded_and_weighted():
    features = {
        "skill_coverage": 1.0,
        "semantic_similarity": 1.0,
        "tfidf_similarity": 1.0,
        "experience_compatibility": 1.0,
        "education_compatibility": 1.0,
    }
    assert compute_weighted_hybrid_score(features) == 1.0


def test_hybrid_score_handles_missing_semantic_similarity():
    features = {
        "skill_coverage": 1.0,
        "semantic_similarity": None,
        "tfidf_similarity": 1.0,
        "experience_compatibility": 1.0,
        "education_compatibility": 1.0,
    }
    score = compute_weighted_hybrid_score(features)
    assert score == 1.0  # renormalized over available features, still all perfect


# ---- skill_gap ----

def test_skill_gap_uses_not_detected_wording():
    gap = get_skill_gap("Python SQL", ["Python", "AWS"])
    assert gap["missing_required_skills"][0]["skill"] == "AWS"
    assert gap["missing_required_skills"][0]["status"] == NOT_DETECTED_PHRASE
    assert gap["skill_coverage_percentage"] == 50.0


# ---- explainability ----

def test_explanation_only_uses_provided_features():
    features = {"skill_coverage": 0.9, "semantic_similarity": 0.8, "tfidf_similarity": 0.1,
                "experience_compatibility": 0.5, "education_compatibility": 0.5,
                "preferred_skill_coverage": None}
    explanation = generate_explanation(features, match_score=0.85, missing_required_skills=["AWS"])
    assert explanation["missing_requirements"] == ["AWS"]
    assert any("skill" in factor.lower() for factor in explanation["positive_factors"])


def test_explanation_no_signal_gets_note():
    features = {"skill_coverage": 0.0, "semantic_similarity": None, "tfidf_similarity": 0.0,
                "experience_compatibility": 0.5, "education_compatibility": 0.5,
                "preferred_skill_coverage": None}
    explanation = generate_explanation(features, match_score=0.0, missing_required_skills=[])
    assert "note" in explanation


# ---- feature_engineering (no semantic model needed) ----

def test_build_feature_vector_without_semantic_model():
    result = build_feature_vector(
        resume_full_text="Python SQL Machine Learning",
        job_description_text="Looking for Python SQL expert",
        required_skills=["Python", "SQL", "AWS"],
        compute_semantic=False,
    )
    features = result["features"]
    assert features["semantic_similarity"] is None
    assert features["skill_coverage"] == round(2 / 3, 4)
    assert "AWS" in result["missing_skills"]
