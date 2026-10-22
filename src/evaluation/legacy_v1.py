"""Faithful reconstruction of the ORIGINAL (v1, tag `baseline-v1-deployed`)
training-time feature computation, kept ONLY so the v2 evaluation can
compare old and new features on identical data splits.

Nothing in the application imports this module. Known defects are kept
on purpose (they are what is being measured):
- skill_coverage intersects dictionary-extracted resume skills with the
  dataset's raw required-skill strings, so most rows get 0.0
- experience/education compatibility return 0.5 when unknown and use
  substring matching ("scrum master" counts as a master's degree)
- job_description_length is included (it identifies the job)

The v1 skill dictionary is loaded from reports/baseline/skill_dictionary_v1.json.
"""

import re
from functools import lru_cache
from typing import Optional

from src.skills.skill_extractor import SkillExtractor
from src.utils.paths import REPORTS_DIR

LEGACY_DICTIONARY_PATH = REPORTS_DIR / "baseline" / "skill_dictionary_v1.json"

LEGACY_FEATURE_COLUMNS = [
    "skill_coverage", "num_matched_skills", "num_missing_skills", "tfidf_similarity", "semantic_similarity",
    "resume_length", "job_description_length", "experience_compatibility", "education_compatibility",
]

_YEARS_RE = re.compile(r"(\d+)\+?\s*years?", re.IGNORECASE)
_EDUCATION_LEVELS = {
    "phd": 4, "ph.d": 4, "doctorate": 4,
    "master": 3, "master's": 3, "msc": 3, "m.s.": 3,
    "bachelor": 2, "bachelor's": 2, "bsc": 2, "b.s.": 2,
    "associate": 1, "associate's": 1,
}


@lru_cache(maxsize=1)
def legacy_extractor() -> SkillExtractor:
    return SkillExtractor(LEGACY_DICTIONARY_PATH)


def _max_years(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    return max((int(m) for m in _YEARS_RE.findall(text)), default=None)


def _education_level(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    lowered = text.lower()
    return max((lvl for kw, lvl in _EDUCATION_LEVELS.items() if kw in lowered), default=None)


def experience_compatibility(resume_text: Optional[str], job_text: Optional[str]) -> float:
    required, candidate = _max_years(job_text), _max_years(resume_text)
    if required is None or candidate is None:
        return 0.5
    if required == 0:
        return 1.0
    return min(1.0, candidate / required)


def education_compatibility(resume_text: Optional[str], job_text: Optional[str]) -> float:
    required, candidate = _education_level(job_text), _education_level(resume_text)
    if required is None or candidate is None:
        return 0.5
    if candidate >= required:
        return 1.0
    return max(0.0, candidate / required)


def legacy_skill_features(resume_clean_text: str, raw_required_skills: list) -> dict:
    resume_skills = set(legacy_extractor().extract_skills(resume_clean_text))
    required = set(raw_required_skills or [])
    if not required:
        return {"skill_coverage": 0.0, "num_matched_skills": 0, "num_missing_skills": 0}
    matched = resume_skills & required
    return {
        "skill_coverage": round(len(matched) / len(required), 4),
        "num_matched_skills": len(matched),
        "num_missing_skills": len(required - resume_skills),
    }


def legacy_blend_score(class_probabilities: dict) -> float:
    """The v1 displayed 'match score': P(strong)*100 + P(moderate)*50."""
    return class_probabilities.get("strong", 0.0) * 100 + class_probabilities.get("moderate", 0.0) * 50
