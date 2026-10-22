from _fixtures import JOB_DATA_ANALYST

from src.scoring.requirements import (
    extract_job_requirements,
    requirements_from_dataset_record,
    requirements_from_table,
    requirements_to_table,
)


def by_kind(reqs, kind):
    return [r for r in reqs if r["kind"] == kind]


def test_required_and_preferred_sections_are_separated():
    reqs = extract_job_requirements(JOB_DATA_ANALYST)["requirements"]
    skills = {r["label"]: r["importance"] for r in by_kind(reqs, "skill")}
    assert skills == {"SQL": "required", "Python": "required", "Power BI or Tableau": "required",
                      "Statistics": "required", "AWS": "preferred", "Docker": "preferred"}


def test_responsibilities_text_does_not_become_requirements():
    reqs = extract_job_requirements(JOB_DATA_ANALYST)["requirements"]
    assert "Data Visualization" not in {r["label"] for r in reqs}


def test_experience_and_education_requirements():
    reqs = extract_job_requirements(JOB_DATA_ANALYST)["requirements"]
    (exp,) = by_kind(reqs, "experience")
    (edu,) = by_kind(reqs, "education")
    assert exp["min_years"] == 2 and exp["importance"] == "required"
    assert edu["level"] == 2 and edu["fields"] == ["statistics", "computer science"] and edu["allow_related_field"]


def test_preferred_cue_inside_a_line_and_essential_cues():
    text = "Requirements:\n- Must have Python\n- Master's degree preferred\n- Bachelor's degree in Mathematics\n"
    reqs = extract_job_requirements(text)["requirements"]
    python = next(r for r in reqs if r["label"] == "Python")
    master = next(r for r in reqs if r["kind"] == "education" and r["level"] == 3)
    bachelor = next(r for r in reqs if r["kind"] == "education" and r["level"] == 2)
    assert python["essential"]
    assert master["importance"] == "preferred" and not master["essential"]
    assert bachelor["importance"] == "required"


def test_unstructured_job_text_warns_and_uses_all_detected_skills():
    out = extract_job_requirements("We need someone good with SQL and Tableau to build dashboards.")
    assert {r["label"] for r in out["requirements"]} == {"SQL", "Tableau"}
    assert out["warnings"] and not out["structured"]


def test_certification_alternatives_become_one_requirement():
    out = extract_job_requirements("Requirements:\n- HR certification such as SHRM-CP, PHR, or SPHR preferred")
    (cert,) = by_kind(out["requirements"], "certification")
    assert cert["alternatives"] == ["SHRM-CP", "PHR", "SPHR"] and cert["importance"] == "preferred"


def test_table_round_trip_preserves_requirements():
    reqs = extract_job_requirements(JOB_DATA_ANALYST)["requirements"]
    rebuilt, warnings = requirements_from_table(requirements_to_table(reqs), reqs)
    assert not warnings
    strip = lambda rs: [(r["kind"], r["label"], r["importance"], r["essential"]) for r in rs]  # noqa: E731
    assert strip(rebuilt) == strip(reqs)


def test_user_edits_are_applied_and_bad_rows_are_reported():
    rows = [
        {"Include": True, "Type": "Skill", "Requirement": "UI/UX Design", "Importance": "Required", "Must-have": True},
        {"Include": True, "Type": "Skill", "Requirement": "Python / R", "Importance": "Preferred", "Must-have": True},
        {"Include": False, "Type": "Skill", "Requirement": "SQL", "Importance": "Required", "Must-have": False},
        {"Include": True, "Type": "Education", "Requirement": "a good degree", "Importance": "Required", "Must-have": False},
        {"Include": True, "Type": "Experience", "Requirement": "3 years", "Importance": "Required", "Must-have": False},
    ]
    reqs, warnings = requirements_from_table(rows)
    assert reqs[0]["alternatives"] == ["UI/UX Design"] and reqs[0]["essential"]
    assert reqs[1]["alternatives"] == ["Python", "R"] and not reqs[1]["essential"]  # preferred is never must-have
    assert all(r["label"] != "SQL" for r in reqs)
    assert len(warnings) == 1 and "degree level" in warnings[0]
    assert reqs[-1]["min_years"] == 3


def test_training_and_inference_requirement_canonicalization_agree():
    """The dataset's structured skill strings and the same skills typed in a
    job description must canonicalize identically (the v1 defect)."""
    raw = ["machine learning", "Python", "problem-solving", "menu costing"]
    from_dataset = [r["alternatives"] for r in requirements_from_dataset_record("", raw) if r["kind"] == "skill"]
    from_table, _ = requirements_from_table(
        [{"Include": True, "Type": "Skill", "Requirement": s, "Importance": "Required", "Must-have": False} for s in raw])
    assert from_dataset == [r["alternatives"] for r in from_table]
    assert from_dataset == [["Machine Learning"], ["Python"], ["Problem Solving"], ["menu costing"]]
