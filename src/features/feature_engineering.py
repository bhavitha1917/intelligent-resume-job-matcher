"""Feature engineering (spec.md section 16 / Phase 11).

Builds a flat, structured feature dict per (resume, job) pair. Deliberately
excludes any protected/PII attribute (name, gender, age, email, phone,
address, religion, caste, nationality, marital status, photo) -- see
spec.md section 27. All features here are derived purely from resume/job
TEXT content.
"""

import re
from typing import Optional

from src.matching.keyword_baseline import keyword_match
from src.matching.semantic_similarity import semantic_similarity as compute_semantic_similarity
from src.matching.tfidf_similarity import compute_tfidf_similarity

_YEARS_RE = re.compile(r"(\d+)\+?\s*years?", re.IGNORECASE)

_EDUCATION_LEVELS = {
    "phd": 4, "ph.d": 4, "doctorate": 4,
    "master": 3, "master's": 3, "msc": 3, "m.s.": 3,
    "bachelor": 2, "bachelor's": 2, "bsc": 2, "b.s.": 2,
    "associate": 1, "associate's": 1,
}


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _max_years_mentioned(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    matches = _YEARS_RE.findall(text)
    return max((int(m) for m in matches), default=None)


def _education_level(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    lowered = text.lower()
    levels_found = [level for keyword, level in _EDUCATION_LEVELS.items() if keyword in lowered]
    return max(levels_found, default=None)


def experience_compatibility(resume_experience_text: Optional[str], job_experience_requirement: Optional[str]) -> float:
    """1.0 if candidate's mentioned years meet/exceed the requirement, a
    partial score if below, or 0.5 (neutral/unknown) when either side
    cannot be detected -- this is an explicit, documented assumption, not a
    fabricated certainty.
    """
    required_years = _max_years_mentioned(job_experience_requirement)
    candidate_years = _max_years_mentioned(resume_experience_text)

    if required_years is None or candidate_years is None:
        return 0.5
    if required_years == 0:
        return 1.0
    return min(1.0, candidate_years / required_years)


def education_compatibility(resume_education_text: Optional[str], job_education_requirement: Optional[str]) -> float:
    """Same neutral-when-unknown convention as experience_compatibility."""
    required_level = _education_level(job_education_requirement)
    candidate_level = _education_level(resume_education_text)

    if required_level is None or candidate_level is None:
        return 0.5
    if candidate_level >= required_level:
        return 1.0
    return max(0.0, candidate_level / required_level)


def title_category_compatibility(resume_category: Optional[str], job_title: Optional[str]) -> float:
    """Token-overlap compatibility between a resume's dataset category
    label (e.g. 'INFORMATION-TECHNOLOGY') and the job title. 0.5 (neutral)
    when either is unavailable -- this feature is explicitly "where
    reliable" per spec.md section 16, not a hard requirement.
    """
    if not resume_category or not job_title:
        return 0.5
    cat_tokens = set(re.split(r"[-_\s]+", resume_category.lower()))
    title_tokens = set(re.split(r"[-_\s/]+", job_title.lower()))
    if not cat_tokens or not title_tokens:
        return 0.5
    overlap = cat_tokens & title_tokens
    return round(len(overlap) / len(title_tokens), 4) if title_tokens else 0.5


def build_feature_vector(
    resume_full_text: str,
    job_description_text: str,
    required_skills: list,
    preferred_skills: Optional[list] = None,
    resume_experience_text: Optional[str] = None,
    job_experience_requirement: Optional[str] = None,
    resume_education_text: Optional[str] = None,
    job_education_requirement: Optional[str] = None,
    resume_category: Optional[str] = None,
    job_title: Optional[str] = None,
    tfidf_vectorizer=None,
    compute_semantic: bool = True,
) -> dict:
    """Assemble the full matching feature vector for one (resume, job) pair.

    `tfidf_vectorizer`: a pre-fit TfidfVectorizer (required for leakage-safe
    evaluation; falls back to an ad-hoc single-pair fit if None, which is
    acceptable only for interactive single-pair inference, not training).
    `compute_semantic`: set False to skip the (comparatively slow)
    sentence-transformer call, e.g. in unit tests.
    """
    required_match = keyword_match(resume_full_text, required_skills)
    preferred_match = (
        keyword_match(resume_full_text, preferred_skills) if preferred_skills else None
    )

    if tfidf_vectorizer is not None:
        tfidf_sim = compute_tfidf_similarity(resume_full_text, job_description_text, tfidf_vectorizer)
    else:
        from src.matching.tfidf_similarity import compute_tfidf_similarity_adhoc
        tfidf_sim = compute_tfidf_similarity_adhoc(resume_full_text, job_description_text)

    semantic_sim = (
        compute_semantic_similarity(resume_full_text, job_description_text) if compute_semantic else None
    )

    features = {
        "skill_coverage": required_match["skill_coverage"],
        "num_matched_skills": len(required_match["matched_skills"]),
        "num_missing_skills": len(required_match["missing_skills"]),
        "preferred_skill_coverage": preferred_match["skill_coverage"] if preferred_match else None,
        "tfidf_similarity": tfidf_sim,
        "semantic_similarity": semantic_sim,
        "resume_length": _word_count(resume_full_text),
        "job_description_length": _word_count(job_description_text),
        "experience_compatibility": experience_compatibility(resume_experience_text, job_experience_requirement),
        "education_compatibility": education_compatibility(resume_education_text, job_education_requirement),
        "title_category_compatibility": title_category_compatibility(resume_category, job_title),
    }

    return {
        "features": features,
        "matched_skills": required_match["matched_skills"],
        "missing_skills": required_match["missing_skills"],
        "matched_preferred_skills": preferred_match["matched_skills"] if preferred_match else [],
        "missing_preferred_skills": preferred_match["missing_skills"] if preferred_match else [],
    }
