"""Phase 16 -- skill gap analysis.

Wording rule (spec.md section 22): never claim a candidate "does not
possess" a skill -- only that it was "Not detected in submitted resume",
since absence from the parsed resume text is not proof of absence of the
underlying ability.
"""

NOT_DETECTED_PHRASE = "Not detected in submitted resume."

from src.matching.keyword_baseline import keyword_match


def get_skill_gap(resume_full_text: str, required_skills, preferred_skills=None) -> dict:
    required_result = keyword_match(resume_full_text, required_skills)
    preferred_result = (
        keyword_match(resume_full_text, preferred_skills) if preferred_skills else None
    )

    return {
        "matched_required_skills": required_result["matched_skills"],
        "missing_required_skills": [
            {"skill": skill, "status": NOT_DETECTED_PHRASE}
            for skill in required_result["missing_skills"]
        ],
        "matched_preferred_skills": preferred_result["matched_skills"] if preferred_result else [],
        "missing_preferred_skills": [
            {"skill": skill, "status": NOT_DETECTED_PHRASE}
            for skill in (preferred_result["missing_skills"] if preferred_result else [])
        ],
        "skill_coverage_percentage": round(required_result["skill_coverage"] * 100, 1),
    }
