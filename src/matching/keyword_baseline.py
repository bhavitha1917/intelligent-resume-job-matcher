"""Baseline 1 (spec.md section 13): simple keyword/skill-coverage matching.

skill_coverage = matched required skills / total required skills

This is intentionally the simplest possible method and is retained
throughout the project as the floor that every other approach is compared
against.
"""

from typing import Iterable, Union

from src.skills.skill_extractor import extract_skills


def _as_skill_set(skills_or_text: Union[str, Iterable[str]]) -> set:
    if isinstance(skills_or_text, str):
        return set(extract_skills(skills_or_text))
    return set(skills_or_text or [])


def keyword_match(
    resume: Union[str, Iterable[str]],
    required_skills: Union[str, Iterable[str]],
) -> dict:
    """Compare a resume (raw text or pre-extracted skill list) against a
    job's required skills (raw text or pre-extracted skill list).

    Returns matched skills, missing skills, and skill coverage in [0, 1].
    Coverage is 0.0 (not undefined) when there are no required skills, and
    this is flagged explicitly so callers don't mistake it for a real score.
    """
    resume_skills = _as_skill_set(resume)
    required = _as_skill_set(required_skills)

    if not required:
        return {
            "matched_skills": [],
            "missing_skills": [],
            "skill_coverage": 0.0,
            "warning": "Job posting has no detected required skills; coverage is undefined and reported as 0.0.",
        }

    matched = sorted(resume_skills & required)
    missing = sorted(required - resume_skills)
    coverage = len(matched) / len(required)

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "skill_coverage": round(coverage, 4),
        "warning": None,
    }
