"""The single scoring interface used by Single Resume Analysis, Candidate
Ranking, the CSV/report exports, the training pipeline and the tests.

    report = score_candidate(job, parsed_resume)

`job` is a dict {"title": str|None, "text": str, "requirements": [...]}
(see src/scoring/requirements.py); `parsed_resume` is the output of
src.data.resume_parser.parse_resume.

Score definition (all numbers from config/scoring_rubric.json):

    component value v_c in [0, 1] for each APPLICABLE component c
    uncapped = 100 * sum(w_c * v_c) / sum(w_c)          (applicable c only)
    score    = min(uncapped, cap) if an essential requirement is unmet or
               required-skill coverage is below the "strong" minimum,
               else uncapped

A component is applicable only if the job states something for it to be
measured against (e.g. "experience" needs a minimum-years requirement).
Inapplicable components are excluded and the remaining weights are
renormalized -- so a job with no certification requirement does not
penalize anyone for having no certification.

The score is an evidence-alignment estimate under this rubric. It is not
a probability of being hired, not a model accuracy, and not a recruiter
judgment.
"""

from datetime import date
from typing import Callable, Optional, Union

from src.data.resume_parser import parse_resume
from src.scoring.config import COMPONENT_ORDER, load_rubric
from src.scoring.evidence import (
    ResumeDoc,
    detect_degrees,
    evaluate_education_requirement,
    evaluate_experience_requirement,
    evaluate_skill_requirement,
    project_evidence,
)
from src.scoring.privacy import sanitize_for_scoring

COMPONENT_LABELS = {
    "required_skills": "Required skills",
    "experience": "Experience (employment + internships)",
    "education": "Education",
    "certifications": "Required certifications",
    "preferred_qualifications": "Preferred qualifications",
    "semantic_relevance": "Contextual relevance (semantic similarity)",
}

INTERPRETATION = (
    "This score summarises how much of the job's stated requirements are evidenced in the submitted "
    "resume, using the documented rubric. It is a decision-support estimate for human review. It is NOT a "
    "probability of being hired, NOT a model accuracy, NOT a calibrated probability and NOT a recruiter's "
    "judgment. 'Not detected' means the resume text does not show the item -- not that the candidate lacks it."
)

DEFAULT_SEMANTIC = "default"
SemanticArg = Union[str, None, Callable[[str, str], float]]


def _default_semantic(resume_text: str, job_text: str) -> float:
    from src.matching.semantic_similarity import semantic_similarity
    return semantic_similarity(resume_text, job_text)


def normalize_semantic(raw: float, rubric: dict) -> float:
    low = rubric["semantic_normalization"]["low"]
    high = rubric["semantic_normalization"]["high"]
    return round(max(0.0, min(1.0, (raw - low) / (high - low))), 4)


def band_for(score: Optional[float], rubric: dict) -> str:
    if score is None:
        return "Insufficient evidence"
    for band in rubric["bands"]:
        if score >= band["min_score"]:
            return band["label"]
    return rubric["bands"][-1]["label"]


def aggregate(components: list[dict]) -> Optional[float]:
    """uncapped score from component dicts (name, weight, applicable, value)."""
    active = [c for c in components if c["applicable"]]
    total_weight = sum(c["weight"] for c in active)
    if not active or total_weight <= 0:
        return None
    return round(100 * sum(c["weight"] * c["value"] for c in active) / total_weight, 1)


def recompute_score(report: dict, rubric: Optional[dict] = None) -> Optional[float]:
    """Reproduce a report's displayed score from its own component values
    and the rubric -- used by tests and shown in the methodology."""
    rubric = rubric or load_rubric()
    if report["status"] != "scored":
        return None
    uncapped = aggregate(report["components"])
    if report["cap_applied"]:
        return min(uncapped, rubric["essential_requirements"]["score_cap_when_unmet"])
    return uncapped


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def score_candidate(
    job: dict,
    parsed_resume: dict,
    rubric: Optional[dict] = None,
    semantic: SemanticArg = DEFAULT_SEMANTIC,
    semantic_raw: Optional[float] = None,
    reference_date: Optional[date] = None,
    candidate_id: Optional[str] = None,
) -> dict:
    """Score one parsed resume against one job. Deterministic for fixed
    inputs, rubric, reference_date and semantic model.

    `semantic`: "default" (sentence-transformer), a callable
    (resume_text, job_text) -> cosine, or None to disable the component.
    `semantic_raw`: a precomputed cosine (training pipeline); overrides
    `semantic`.
    `reference_date`: date used for "Present" in date ranges (defaults to
    today; recorded in the report).
    """
    rubric = rubric or load_rubric()
    reference_date = reference_date or date.today()
    requirements = list(job.get("requirements") or [])
    warnings: list[str] = list(parsed_resume.get("parse_warnings") or [])

    sanitized, removed = sanitize_for_scoring(parsed_resume.get("full_text") or "", parsed_resume.get("section_spans"))
    doc = ResumeDoc(sanitized, parsed_resume.get("section_spans") or [])
    word_count = len(sanitized.split())

    skill_terms = [alt for r in requirements if r["kind"] == "skill" for alt in r["alternatives"]]

    evaluated = []
    for req in requirements:
        if req["kind"] in ("skill", "certification"):
            result = evaluate_skill_requirement(req, doc, rubric)
        elif req["kind"] == "experience":
            result = evaluate_experience_requirement(req, doc, rubric, reference_date, skill_terms, job.get("title"))
        elif req["kind"] == "education":
            result = evaluate_education_requirement(req, doc, rubric, reference_date)
        else:
            continue
        evaluated.append({**req, "result": result})

    def credits(kind=None, importance=None):
        return [r["result"]["credit"] for r in evaluated
                if (kind is None or r["kind"] == kind) and (importance is None or r["importance"] == importance)]

    def first_credit(kind):
        items = [r for r in evaluated if r["kind"] == kind and r["importance"] == "required"]
        return items[0]["result"]["credit"] if items else None

    values = {
        "required_skills": _mean(credits("skill", "required")),
        "experience": first_credit("experience"),
        "education": first_credit("education"),
        "certifications": _mean(credits("certification", "required")),
        "preferred_qualifications": _mean(credits(importance="preferred")),
        "semantic_relevance": None,
    }

    semantic_info = {"available": False, "raw": None, "normalized": None}
    job_text = job.get("text") or ""
    if semantic_raw is None and semantic is not None and sanitized.strip() and job_text.strip():
        fn = _default_semantic if semantic == DEFAULT_SEMANTIC else semantic
        try:
            semantic_raw = fn(sanitized, job_text)
        except Exception as exc:  # model download / load failure must not break scoring
            warnings.append(f"Semantic similarity unavailable ({type(exc).__name__}); score computed without it.")
            semantic_raw = None
    if semantic_raw is not None:
        semantic_info = {"available": True, "raw": round(float(semantic_raw), 4),
                         "normalized": normalize_semantic(float(semantic_raw), rubric)}
        values["semantic_relevance"] = semantic_info["normalized"]
    elif semantic is None:
        warnings.append("Semantic-similarity component disabled for this run.")

    weights = rubric["component_weights"]
    components = [
        {"name": name, "label": COMPONENT_LABELS[name], "weight": weights[name],
         "applicable": values[name] is not None, "value": values[name]}
        for name in COMPONENT_ORDER
    ]
    applicable_weight = sum(c["weight"] for c in components if c["applicable"])
    for c in components:
        c["effective_weight"] = round(c["weight"] / applicable_weight, 4) if c["applicable"] and applicable_weight else 0.0
        c["contribution"] = round(100 * c["effective_weight"] * c["value"], 2) if c["applicable"] else None

    scorable = [r for r in evaluated]
    insufficient = []
    min_words = rubric["insufficient_evidence"]["min_resume_words"]
    if word_count < min_words:
        insufficient.append(f"Only {word_count} words of resume text could be used (minimum {min_words}).")
    if len(scorable) < rubric["insufficient_evidence"]["min_scorable_requirements"]:
        insufficient.append("The job description has no scorable requirements (skills, experience, education or "
                            "certifications). Add requirements in the review step.")

    ess = rubric["essential_requirements"]
    unmet_essential = [
        r for r in evaluated if r["essential"]
        and not r["result"].get("gate_met", r["result"]["credit"] >= ess["min_credit_to_count_as_met"])
    ]
    cap_reasons = [f"Must-have requirement not evidenced: {r['label']} ({r['result']['status'].replace('_', ' ')})"
                   for r in unmet_essential]
    if values["required_skills"] is not None and values["required_skills"] < ess["min_required_skill_component_for_strong"]:
        cap_reasons.append(
            f"Required-skill evidence ({values['required_skills']:.0%}) is below the "
            f"{ess['min_required_skill_component_for_strong']:.0%} minimum for a strong result.")

    uncapped = aggregate(components)
    if insufficient or uncapped is None:
        status, score, cap_applied = "insufficient_evidence", None, False
    else:
        status = "scored"
        cap_applied = bool(cap_reasons) and uncapped > ess["score_cap_when_unmet"]
        score = min(uncapped, ess["score_cap_when_unmet"]) if cap_reasons else uncapped

    def count(kind, importance, statuses=("met",)):
        items = [r for r in evaluated if r["kind"] == kind and r["importance"] == importance]
        return sum(r["result"]["status"] in statuses for r in items), len(items)

    req_met, req_total = count("skill", "required")
    pref_items = [r for r in evaluated if r["importance"] == "preferred"]
    exp_req = next((r for r in evaluated if r["kind"] == "experience"), None)
    degrees = detect_degrees(doc, reference_date)
    top_degree = max(degrees, key=lambda d: d["level"]) if degrees else None

    if not doc.has_sections:
        warnings.append("No resume section headers were recognised; evidence context (project vs. skills list) "
                        "could not be determined, so skill mentions are classified by sentence wording only.")

    return {
        "candidate_id": candidate_id,
        "status": status,
        "score": score,
        "uncapped_score": uncapped,
        "band": band_for(score, rubric),
        "cap_applied": cap_applied,
        "cap_reasons": cap_reasons,
        "insufficient_reasons": insufficient,
        "components": components,
        "requirements": evaluated,
        "summary": {
            "required_skills_met": req_met,
            "required_skills_total": req_total,
            "preferred_met": sum(r["result"]["status"] == "met" for r in pref_items),
            "preferred_total": len(pref_items),
            "experience_years_detected": (exp_req["result"].get("detected_years") if exp_req else None),
            "highest_degree": top_degree["level_name"] if top_degree else None,
            "contradicted": [r["label"] for r in evaluated if r["result"]["status"] == "contradicted"],
            "not_detected": [r["label"] for r in evaluated if r["result"]["status"] == "not_detected"],
        },
        "projects": project_evidence(doc, skill_terms),
        "degrees": [{k: v for k, v in d.items() if k != "start"} for d in degrees],
        "semantic": semantic_info,
        "resume_word_count": word_count,
        "warnings": warnings,
        "privacy": {"removed_before_scoring": removed},
        "config": {"version": rubric["version"], "fingerprint": rubric["_fingerprint"]},
        "reference_date": reference_date.isoformat(),
        "interpretation": INTERPRETATION,
    }


def score_resume_source(job: dict, source, filename: Optional[str] = None, **kwargs) -> dict:
    """Parse + score in one call (file path, bytes or raw text)."""
    return score_candidate(job, parse_resume(source, filename), **kwargs)
