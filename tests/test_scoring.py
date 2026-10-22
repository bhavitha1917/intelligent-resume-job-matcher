"""Tests of the single scoring interface (src/scoring/scorer.py).

What these establish: the documented rubric rules are implemented as
written, scores are reproducible from their components, and evidence is
classified as specified. What they do NOT establish: that the rubric's
weights or outputs agree with human recruiters (no such data exists here).
"""

import copy

import pytest
from _fixtures import (
    JOB_DATA_ANALYST,
    REFERENCE_DATE,
    RESUME_PARTIAL,
    RESUME_STRONG,
    RESUME_WEAK,
    fake_semantic,
    job,
    parsed,
)

from src.scoring.config import load_rubric, with_overrides
from src.scoring.requirements import make_experience_requirement, make_skill_requirement
from src.scoring.scorer import aggregate, recompute_score, score_candidate


def score(resume_text, job_dict=None, **kw):
    kw.setdefault("semantic", None)
    kw.setdefault("reference_date", REFERENCE_DATE)
    return score_candidate(job_dict or job(), parsed(resume_text), **kw)


def req_status(report, label):
    return next(r["result"] for r in report["requirements"] if r["label"] == label)


# ---------------------------------------------------------------- reproducibility
def test_identical_input_gives_identical_report():
    a = score(RESUME_STRONG, semantic=fake_semantic)
    b = score(RESUME_STRONG, semantic=fake_semantic)
    assert a == b


def test_score_is_reproducible_from_displayed_components():
    for text in (RESUME_STRONG, RESUME_PARTIAL, RESUME_WEAK):
        report = score(text, semantic=fake_semantic)
        assert recompute_score(report) == report["score"]
        applicable = [c for c in report["components"] if c["applicable"]]
        if report["status"] == "scored" and not report["cap_applied"]:
            assert sum(c["contribution"] for c in applicable) == pytest.approx(report["score"], abs=0.15)


def test_report_records_rubric_fingerprint_and_reference_date():
    report = score(RESUME_STRONG)
    assert report["config"]["fingerprint"] == load_rubric()["_fingerprint"]
    assert report["reference_date"] == REFERENCE_DATE.isoformat()


# ---------------------------------------------------------------- strong / partial / weak
def test_realistic_strong_partial_weak_ordering_and_bands():
    """Establishes: on hand-written resumes with clearly different evidence,
    the rubric orders them as their evidence suggests. Does not establish
    agreement with recruiters on real resumes."""
    strong, partial, weak = (score(t) for t in (RESUME_STRONG, RESUME_PARTIAL, RESUME_WEAK))
    assert strong["score"] > partial["score"] > weak["score"]
    assert strong["band"] == "Strong alignment"
    assert weak["band"] == "Limited alignment"
    assert strong["summary"]["required_skills_met"] == strong["summary"]["required_skills_total"] == 4


# ---------------------------------------------------------------- required-skill coverage
def test_required_skill_coverage_counts_and_credits():
    report = score(RESUME_PARTIAL)
    s = report["summary"]
    # SQL (applied), Power BI (listed) detected; Python and Statistics not.
    assert (s["required_skills_met"], s["required_skills_total"]) == (2, 4)
    comp = next(c for c in report["components"] if c["name"] == "required_skills")
    credit = load_rubric()["skill_evidence_credit"]
    assert comp["value"] == pytest.approx((credit["applied"] + credit["listed"]) / 4)


def test_alternatives_count_as_one_requirement():
    report = score(RESUME_STRONG)
    result = req_status(report, "Power BI or Tableau")
    assert result["status"] == "met" and result["matched"] == "Tableau"


def test_duplicate_mentions_do_not_inflate_coverage():
    repeated = RESUME_PARTIAL.replace("SKILLS\nExcel, SQL, Power BI", "SKILLS\nExcel, SQL, SQL, sql, SQL, Power BI, SQL")
    a, b = score(RESUME_PARTIAL), score(repeated)
    assert a["summary"]["required_skills_met"] == b["summary"]["required_skills_met"]
    assert a["components"][0]["value"] == b["components"][0]["value"]


# ---------------------------------------------------------------- missing vs negative evidence
def test_not_detected_is_distinguished_from_contradicted():
    base = "SKILLS\nSQL\n\nEXPERIENCE\nAnalyst (2020 - 2024)\n- Wrote SQL reports for finance. " * 3
    negated = base + "\nI have no experience with Python."
    job_dict = {"title": "Analyst", "text": "Requirements: SQL, Python",
                "requirements": [make_skill_requirement(["SQL"], rid="R1"), make_skill_requirement(["Python"], rid="R2")]}
    missing = score(base, job_dict)
    contra = score(negated, job_dict)
    assert req_status(missing, "Python")["status"] == "not_detected"
    assert req_status(contra, "Python")["status"] == "contradicted"
    assert contra["summary"]["contradicted"] == ["Python"]
    assert missing["summary"]["contradicted"] == []


def test_wrapped_line_negation_from_demo_resume_is_detected():
    from src.utils.paths import DEMO_DATA_DIR
    text = (DEMO_DATA_DIR / "resume_2.txt").read_text(encoding="utf-8")
    report = score(text, job((DEMO_DATA_DIR / "job_description_1.txt").read_text(encoding="utf-8")))
    assert req_status(report, "Machine Learning")["status"] == "contradicted"


def test_quoted_job_requirements_inside_resume_are_ignored():
    text = RESUME_WEAK + "\nJOB DESCRIPTION\nRequired skills: SQL, Python, Statistics\n"
    report = score(text)
    assert req_status(report, "SQL")["status"] == "not_detected"
    assert any("quoted" in n for n in req_status(report, "SQL")["notes"])


def test_missing_experience_information_is_not_a_neutral_half_score():
    """Regression for the v1 0.5 fallback: no detectable experience gives
    status not_detected and credit 0, never 0.5."""
    report = score("SKILLS\nSQL, Python, Tableau, Statistics\n" + "Analytical and curious person. " * 10)
    exp = next(r["result"] for r in report["requirements"] if r["kind"] == "experience")
    assert exp["status"] == "not_detected" and exp["credit"] == 0.0


# ---------------------------------------------------------------- documented contribution rules
def test_project_evidence_only_upgrades_listed_to_applied():
    job_dict = {"title": "X", "text": "Requirements: Docker",
                "requirements": [make_skill_requirement(["Docker"], rid="R1")]}
    listed = score("SKILLS\nDocker\n\nSUMMARY\n" + "Reliable engineer. " * 20, job_dict)
    applied = score("SKILLS\nDocker\n\nPROJECTS\n- Containerised a web app with Docker.\n\nSUMMARY\n" + "Reliable engineer. " * 20, job_dict)
    credit = load_rubric()["skill_evidence_credit"]
    assert req_status(listed, "Docker")["credit"] == credit["listed"]
    assert req_status(applied, "Docker")["credit"] == credit["applied"]
    # the project adds nothing else: no separate project component exists
    assert {c["name"] for c in applied["components"]} == {c["name"] for c in listed["components"]}


def test_internship_counts_toward_experience_only_when_configured():
    job_dict = {"title": "Data Analyst", "text": "x",
                "requirements": [make_experience_requirement(1, rid="R1"), make_skill_requirement(["SQL"], rid="R2")]}
    text = "INTERNSHIPS\nData Analyst Intern, Acme (Jan 2024 - Jan 2025)\n- Wrote SQL reports.\n\nSKILLS\nSQL\n" + "Hard working. " * 15
    on = score(text, job_dict)
    rubric = load_rubric()
    rubric_off = copy.deepcopy(rubric)
    rubric_off["experience"]["count_internships"] = False
    off = score(text, job_dict, rubric=rubric_off)
    exp_on = next(r["result"] for r in on["requirements"] if r["kind"] == "experience")
    exp_off = next(r["result"] for r in off["requirements"] if r["kind"] == "experience")
    assert exp_on["detected_years"] == pytest.approx(1.0) and exp_on["credit"] == 1.0
    assert exp_off["credit"] == 0.0


def test_education_scrum_master_and_ms_office_are_not_degrees():
    report = score("SKILLS\nMS Office, Scrum Master certified\n\n" + "Organised team player. " * 20)
    edu = next(r["result"] for r in report["requirements"] if r["kind"] == "education")
    assert edu["status"] == "not_detected"


def test_inapplicable_components_are_excluded_and_renormalized():
    job_dict = {"title": "X", "text": "Requirements: SQL", "requirements": [make_skill_requirement(["SQL"], rid="R1")]}
    report = score("EXPERIENCE\nAnalyst (2020 - 2024)\n- Wrote SQL daily.\n" + "Detail oriented. " * 20, job_dict)
    active = [c for c in report["components"] if c["applicable"]]
    assert [c["name"] for c in active] == ["required_skills"]
    assert active[0]["effective_weight"] == 1.0
    assert report["score"] == 100.0


# ---------------------------------------------------------------- gating / insufficient evidence
def test_unmet_must_have_caps_an_otherwise_high_score():
    job_dict = job()
    for r in job_dict["requirements"]:
        if r["label"] == "Statistics":
            r["essential"] = True
    no_stats = RESUME_STRONG.replace("Statistics, ", "").replace("Applied statistics to evaluate", "Evaluated") \
                            .replace("B.S. in Statistics", "B.S. in Computer Science")
    report = score(no_stats, job_dict)
    cap = load_rubric()["essential_requirements"]["score_cap_when_unmet"]
    assert report["uncapped_score"] > cap
    assert report["score"] == cap and report["cap_applied"]
    assert report["band"] != "Strong alignment"


def test_empty_or_tiny_resume_gives_insufficient_evidence_not_a_number():
    for text in ("", "Python SQL"):
        report = score(text)
        assert report["status"] == "insufficient_evidence"
        assert report["score"] is None and report["band"] == "Insufficient evidence"


def test_job_without_requirements_gives_insufficient_evidence():
    report = score(RESUME_STRONG, {"title": None, "text": "We are hiring!", "requirements": []}, semantic=fake_semantic)
    assert report["status"] == "insufficient_evidence"


def test_semantic_failure_degrades_gracefully():
    def broken(a, b):
        raise OSError("model download failed")
    report = score(RESUME_STRONG, semantic=broken)
    assert report["status"] == "scored"
    assert any("Semantic similarity unavailable" in w for w in report["warnings"])
    assert not next(c for c in report["components"] if c["name"] == "semantic_relevance")["applicable"]


# ---------------------------------------------------------------- configuration
def test_weights_come_from_config_and_must_sum_to_one():
    with pytest.raises(ValueError):
        with_overrides(load_rubric(), required_skills=0.9)
    custom = with_overrides(load_rubric(), required_skills=0.55, semantic_relevance=0.05)
    assert custom["_fingerprint"].startswith("custom-")
    report = score(RESUME_PARTIAL, rubric=custom, semantic=fake_semantic)
    assert recompute_score(report, custom) == report["score"]


def test_aggregate_ignores_inapplicable_components():
    comps = [{"weight": 0.5, "applicable": True, "value": 1.0}, {"weight": 0.5, "applicable": False, "value": None}]
    assert aggregate(comps) == 100.0


# ---------------------------------------------------------------- wording
def test_score_is_not_described_as_probability_or_accuracy():
    report = score(RESUME_STRONG)
    text = report["interpretation"]
    assert "NOT a probability of being hired" in text and "NOT a model accuracy" in text
    assert not any(k in report for k in ("accuracy", "probability", "score_percent", "match_probability"))


def test_job_description_text_is_used_unchanged():
    assert job()["text"] == JOB_DATA_ANALYST


def test_rubric_fingerprint_is_independent_of_line_endings(tmp_path):
    from src.utils.paths import SCORING_RUBRIC_PATH
    crlf = tmp_path / "rubric.json"
    crlf.write_bytes(SCORING_RUBRIC_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert load_rubric(crlf)["_fingerprint"] == load_rubric()["_fingerprint"]
