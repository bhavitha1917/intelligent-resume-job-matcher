"""Static checks that the Streamlit pages do not mislabel the score as an
accuracy, a hiring probability, or a classifier-derived match percentage,
and that every page renders reports through the shared component."""

import re
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
PAGES = sorted((APP_DIR / "pages").glob("*.py")) + [APP_DIR / "app.py", APP_DIR / "components" / "ui.py"]


def _source(path):
    return path.read_text(encoding="utf-8")


def test_no_page_presents_a_match_percentage_or_hiring_probability():
    # Pages may mention "probability of being hired" only to say the score is NOT one.
    forbidden = [r"(?i)match score", r"% match", r"(?i)model accuracy:", r"P\(strong\)\s*\*\s*100",
                 r"score_percent", r"predict_match\("]
    for page in PAGES:
        if page.name == "4_Evaluation.py":
            continue  # quotes the removed v1 formula when explaining it
        src = _source(page)
        for pattern in forbidden:
            assert not re.search(pattern, src), f"{page.name} contains '{pattern}'"


def test_score_shown_with_its_interpretation():
    ui = _source(APP_DIR / "components" / "ui.py")
    assert "Rubric alignment score" in ui and 'report["interpretation"]' in ui


def test_both_analysis_pages_use_the_shared_scorer_and_renderer():
    single = _source(APP_DIR / "pages" / "1_Single_Resume_Analysis.py")
    ranking = _source(APP_DIR / "pages" / "2_Candidate_Ranking.py")
    assert "score_candidate(" in single and "render_report(" in single
    assert "rank_candidates(" in ranking and "render_report(" in ranking
    for src in (single, ranking):
        assert "today()" in src and "semantic_arg()" in src
