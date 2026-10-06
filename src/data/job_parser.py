"""Job description parsing.

Supports two input shapes (spec.md section 11):
1. Raw free text typed/pasted by a user in the Streamlit app.
2. Structured records from a dataset (e.g. the HF resume-job-fit dataset's
   `job_text` / `job_required_skills` columns, or LinkedIn postings' `title`
   / `description` / `formatted_experience_level`).
"""

import re
from typing import Optional

from src.skills.skill_extractor import extract_skills

_REQUIRED_HEADER_RE = re.compile(
    r"required\s+skills\s*:?\s*(.*?)(?=\n\s*\n|preferred\s+skills|experience\s*:|education\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_PREFERRED_HEADER_RE = re.compile(
    r"preferred\s+skills\s*:?\s*(.*?)(?=\n\s*\n|required\s+skills|experience\s*:|education\s*:|$)",
    re.IGNORECASE | re.DOTALL,
)
_EXPERIENCE_RE = re.compile(
    r"(\d+\+?\s*(?:-\s*\d+\s*)?years?)[^\n.]*experience",
    re.IGNORECASE,
)
_EDUCATION_RE = re.compile(
    r"(bachelor'?s|master'?s|ph\.?d\.?|associate'?s)[^\n.]*degree?",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /\-&()]{2,80})\s*\n")


def _empty_result() -> dict:
    return {
        "title": None,
        "description": "",
        "required_skills": [],
        "preferred_skills": [],
        "experience_requirement": None,
        "education_requirement": None,
        "parse_warnings": [],
    }


def parse_job_description_text(raw_text: str) -> dict:
    """Heuristically parse a raw job-description text blob."""
    result = _empty_result()
    if not raw_text or not raw_text.strip():
        result["parse_warnings"].append("Job description text is empty.")
        return result

    result["description"] = raw_text.strip()

    title_match = _TITLE_RE.match(raw_text.strip())
    if title_match:
        result["title"] = title_match.group(1).strip()

    required_match = _REQUIRED_HEADER_RE.search(raw_text)
    if required_match:
        result["required_skills"] = extract_skills(required_match.group(1))
    preferred_match = _PREFERRED_HEADER_RE.search(raw_text)
    if preferred_match:
        result["preferred_skills"] = extract_skills(preferred_match.group(1))

    # Fallback: if no explicit "required skills" section was found, treat
    # all skills detected anywhere in the description as required, since
    # that is the only signal available (documented limitation).
    if not result["required_skills"] and not preferred_match:
        result["required_skills"] = extract_skills(raw_text)
        result["parse_warnings"].append(
            "No explicit 'Required skills' section detected; using all "
            "skills found in the full description text as required skills."
        )

    exp_match = _EXPERIENCE_RE.search(raw_text)
    if exp_match:
        result["experience_requirement"] = exp_match.group(1).strip()

    edu_match = _EDUCATION_RE.search(raw_text)
    if edu_match:
        result["education_requirement"] = edu_match.group(0).strip()

    return result


def parse_job_description_record(
    title: Optional[str] = None,
    description: str = "",
    required_skills: Optional[list] = None,
    preferred_skills: Optional[list] = None,
    experience_level: Optional[str] = None,
    education_requirement: Optional[str] = None,
) -> dict:
    """Build the same normalized structure from an already-structured dataset
    record instead of re-deriving fields from raw text via regex.
    """
    result = _empty_result()
    result["title"] = title
    result["description"] = description or ""
    result["experience_requirement"] = experience_level
    result["education_requirement"] = education_requirement

    if required_skills:
        if isinstance(required_skills, str):
            result["required_skills"] = extract_skills(required_skills)
        else:
            result["required_skills"] = sorted(set(required_skills))
    elif description:
        result["required_skills"] = extract_skills(description)
        result["parse_warnings"].append(
            "No structured required_skills field provided; derived from description text."
        )

    if preferred_skills:
        if isinstance(preferred_skills, str):
            result["preferred_skills"] = extract_skills(preferred_skills)
        else:
            result["preferred_skills"] = sorted(set(preferred_skills))

    if not result["description"].strip():
        result["parse_warnings"].append("Job description text is empty.")

    return result
