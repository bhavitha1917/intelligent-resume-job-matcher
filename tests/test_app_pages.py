"""Headless smoke tests of the real Streamlit pages (streamlit.testing).
They run each page script, drive the widgets with the bundled demo data
(semantic component off, so no model download is needed) and assert that
no exception occurs and that the expected report elements appear.
They do not test visual layout on a phone."""

import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP_DIR))

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _run(page, timeout=60):
    at = AppTest.from_file(str(APP_DIR / page), default_timeout=timeout)
    at.session_state["use_semantic"] = False
    return at.run()


def _no_exceptions(at):
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.parametrize("page", ["app.py", "pages/4_Evaluation.py", "pages/5_Methodology.py",
                                  "pages/3_Candidate_Details.py"])
def test_static_pages_render(page):
    _no_exceptions(_run(page))


def test_single_resume_page_end_to_end_with_demo_data():
    at = _run("pages/1_Single_Resume_Analysis.py")
    at.radio(key="single_jd_source").set_value("Demo job").run()
    at.radio(key="single_resume_mode").set_value("Demo resume").run()
    at.button[0].click().run()
    _no_exceptions(at)
    metrics = {m.label: m.value for m in at.metric}
    assert "Rubric alignment score" in metrics and metrics["Rubric alignment score"].endswith("/ 100")
    assert at.session_state["single_result"]["report"]["status"] == "scored"


def test_ranking_page_end_to_end_and_matches_single_page():
    at = _run("pages/2_Candidate_Ranking.py")
    at.radio(key="rank_jd_source").set_value("Demo job").run()
    at.toggle(key="rank_use_demo").set_value(True).run()
    at.button[0].click().run()
    _no_exceptions(at)
    rows = at.session_state["ranking_results"]
    assert [r["rank"] for r in rows] == [1, 2, 3]

    single = _run("pages/1_Single_Resume_Analysis.py")
    single.radio(key="single_jd_source").set_value("Demo job").run()
    single.radio(key="single_resume_mode").set_value("Demo resume").run()
    single.selectbox[0].set_value(rows[0]["candidate_id"]).run()
    single.button[0].click().run()
    assert single.session_state["single_result"]["report"]["score"] == rows[0]["score"]
