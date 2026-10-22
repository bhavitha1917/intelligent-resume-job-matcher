"""Job-requirement extraction.

Turns a job description into an explicit, reviewable list of requirement
dicts. Each requirement has:

    id           stable identifier ("R1", "R2", ...)
    kind         "skill" | "experience" | "education" | "certification"
    label        human-readable text (editable by the user)
    importance   "required" | "preferred"
    essential    True if the job states it as a must-have (gates the score)
    source_text  the job-description line it came from ("" if user-added)
  plus kind-specific fields:
    skill/certification: alternatives  (list of canonical names; ANY one counts)
    experience:          min_years
    education:           level, level_name, fields, allow_related_field

Extraction is conservative and rule-based (documented in
docs/SCORING.md). The app always shows the extracted list for review and
lets the user correct it before scoring (`requirements_from_table`).
"""

import re
from typing import Iterable, Optional

from src.skills.skill_extractor import canonicalize_skill, get_default_extractor, normalize_phrase

# ----------------------------------------------------------------------------
# Section headers in job descriptions
# ----------------------------------------------------------------------------
_REQUIRED_HEADERS = {
    "required skills", "requirements", "required qualifications", "qualifications", "minimum qualifications",
    "basic qualifications", "must have", "must-have", "must have skills", "must-have skills", "what you need",
    "what we are looking for", "what we're looking for", "who you are", "skills", "key skills",
    "technical skills", "skills required", "skills and qualifications", "experience", "education",
    "eligibility", "eligibility criteria", "required experience", "job requirements", "your profile",
}
_PREFERRED_HEADERS = {
    "preferred skills", "preferred qualifications", "nice to have", "nice-to-have", "bonus", "bonus points",
    "desired skills", "good to have", "preferred", "desirable", "additional qualifications", "pluses",
}
_OTHER_HEADERS = {
    "responsibilities", "key responsibilities", "about the role", "about us", "about the company",
    "what you'll do", "what you will do", "duties", "benefits", "perks", "the role", "role overview",
    "job summary", "overview", "description", "how to apply", "location", "compensation",
}
_MUST_HEADERS = {"must have", "must-have", "must have skills", "must-have skills"}

_PREFERRED_CUE_RE = re.compile(
    r"\b(preferred|a plus|is a plus|nice[- ]to[- ]have|bonus|desirable|desired|ideally|advantageous|good to have|optional)\b",
    re.IGNORECASE,
)
_ESSENTIAL_CUE_RE = re.compile(r"\b(must|mandatory|essential|required|requirement|minimum|at least)\b", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*(?:[-*•●▪◦]|\d+[.)])\s*")

# ----------------------------------------------------------------------------
# Experience
# ----------------------------------------------------------------------------
_YEARS_REQ_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:(?:-|–|to)\s*\d+(?:\.\d+)?\s*)?\+?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)

# ----------------------------------------------------------------------------
# Education levels (word-boundary aware: "Scrum Master" and "MS Office" are
# NOT degrees)
# ----------------------------------------------------------------------------
EDUCATION_LEVEL_NAMES = {0: "High school", 1: "Diploma / Associate", 2: "Bachelor's", 3: "Master's", 4: "Doctorate"}
EDUCATION_PATTERNS = [
    (4, re.compile(r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral degree\b", re.IGNORECASE)),
    (3, re.compile(
        r"\bmaster'?s\b|\bmasters\b|\bmaster of\b|\bm\.s\.?(?=[\s,;)]|$)|\bms\s+in\b|\bm\.?sc\b|\bm\.?\s?tech\b|"
        r"\bmca\b|\bmba\b|\bm\.e\.(?=[\s,;)]|$)|\bpost[- ]?graduate degree\b|\bpost[- ]?graduation\b",
        re.IGNORECASE,
    )),
    (2, re.compile(
        r"\bbachelor'?s\b|\bbachelors\b|\bbachelor of\b|\bb\.s\.?(?=[\s,;)]|$)|\bbs\s+in\b|\bb\.?sc\b|\bb\.?\s?tech\b|"
        r"\bb\.e\.(?=[\s,;)]|$)|\bbca\b|\bbba\b|\bb\.?com\b|\bb\.a\.(?=[\s,;)]|$)|\bundergraduate degree\b|"
        r"\bfour[- ]year degree\b|\bcollege degree\b",
        re.IGNORECASE,
    )),
    (1, re.compile(r"\bassociate'?s degree\b|\bassociate degree\b|\bdiploma\b", re.IGNORECASE)),
    (0, re.compile(
        r"\bhigh school (?:diploma|degree|graduate|education|certificate)\b|\bged\b|"
        r"\bsecondary school certificate\b|\b12th (?:grade|pass|standard)\b|\bhsc\b",
        re.IGNORECASE,
    )),
]
_FIELD_AFTER_IN_RE = re.compile(r"\b(?:in|of)\s+(.+)$", re.IGNORECASE)
_RELATED_FIELD_RE = re.compile(r"\b(?:or\s+)?(?:a\s+)?(?:related|relevant|equivalent|similar)(?:\s+[a-z]+)?\s+(?:field|discipline|area|subject)s?\b|\bor equivalent\b", re.IGNORECASE)


def education_levels_in(text: str) -> list[tuple[int, re.Match]]:
    found = []
    for level, pattern in EDUCATION_PATTERNS:
        for match in pattern.finditer(text or ""):
            found.append((level, match))
    return sorted(found, key=lambda lm: lm[1].start())


# ----------------------------------------------------------------------------
# Line classification
# ----------------------------------------------------------------------------
def _header_of(line: str) -> tuple[Optional[str], str]:
    """If `line` is (or starts with) a section header, return
    (section_kind, remaining_text_on_line); else (None, line)."""
    stripped = _BULLET_RE.sub("", line).strip()
    head, sep, rest = stripped.partition(":")
    candidates = [(head, rest)] if sep else []
    candidates.append((stripped, ""))
    for text, remainder in candidates:
        key = normalize_phrase(text.replace("’", "'"))
        if len(key.split()) > 6:
            continue
        if key in _MUST_HEADERS:
            return "must", remainder
        if key in _PREFERRED_HEADERS:
            return "preferred", remainder
        if key in _REQUIRED_HEADERS:
            return "required", remainder
        if key in _OTHER_HEADERS:
            return "other", remainder
    return None, line


def _clauses(line: str) -> list[str]:
    parts = re.split(r";|(?<=[a-z0-9)])\.\s+(?=[A-Z])", line)
    return [p.strip() for p in parts if p and p.strip()]


def _new_id(reqs: list) -> str:
    return f"R{len(reqs) + 1}"


def _importance_for(section: str, clause: str) -> str:
    if section == "preferred" or _PREFERRED_CUE_RE.search(clause):
        return "preferred"
    return "required"


def _essential_for(section: str, clause: str, importance: str) -> bool:
    if importance != "required":
        return False
    return section == "must" or bool(_ESSENTIAL_CUE_RE.search(clause))


def _skill_label(alternatives: list[str]) -> str:
    return " or ".join(alternatives)


def make_skill_requirement(alternatives: list[str], importance="required", essential=False, source_text="", kind="skill", rid="") -> dict:
    return {
        "id": rid, "kind": kind, "label": _skill_label(alternatives), "importance": importance,
        "essential": essential, "source_text": source_text, "alternatives": alternatives,
    }


def make_experience_requirement(min_years: float, importance="required", essential=False, source_text="", rid="") -> dict:
    years = int(min_years) if float(min_years).is_integer() else min_years
    return {
        "id": rid, "kind": "experience", "label": f"At least {years} year(s) of relevant experience",
        "importance": importance, "essential": essential, "source_text": source_text, "min_years": float(min_years),
    }


def make_education_requirement(level: int, fields: list[str], allow_related: bool, importance="required",
                               essential=False, source_text="", rid="") -> dict:
    label = f"{EDUCATION_LEVEL_NAMES[level]} degree or higher"
    if fields:
        label += " in " + ", ".join(fields) + (" (or related field)" if allow_related else "")
    return {
        "id": rid, "kind": "education", "label": label, "importance": importance, "essential": essential,
        "source_text": source_text, "level": level, "level_name": EDUCATION_LEVEL_NAMES[level],
        "fields": fields, "allow_related_field": allow_related,
    }


def parse_education_clause(clause: str) -> Optional[tuple[int, list[str], bool]]:
    levels = education_levels_in(clause)
    if not levels:
        return None
    level, first_match = levels[0]
    if len({lvl for lvl, _ in levels}) > 1 and re.search(r"\bor\b", clause[: levels[-1][1].end()], re.IGNORECASE):
        # "Bachelor's or Master's degree" -> the minimum of the alternatives.
        level = min(lvl for lvl, _ in levels)
    after = clause[first_match.end():]
    allow_related = bool(_RELATED_FIELD_RE.search(after))
    fields: list[str] = []
    field_match = _FIELD_AFTER_IN_RE.search(after)
    if field_match:
        field_text = _RELATED_FIELD_RE.sub("", field_match.group(1))
        field_text = re.split(r"\b(?:with|and\s+\d|preferred|required|from)\b|\(", field_text, flags=re.IGNORECASE)[0]
        for part in re.split(r",|/|\bor\b|\band\b", field_text):
            part = normalize_phrase(part)
            part = re.sub(r"^(?:a|an|the)\s+", "", part)
            if part and len(part.split()) <= 5 and not education_levels_in(part):
                fields.append(part)
    return level, fields, allow_related


def _skills_in_clause(clause: str) -> tuple[list[list[str]], list[list[str]]]:
    """Return (skill_groups, certification_groups). Each group is a list of
    alternatives. Certifications listed together ("SHRM-CP, PHR, or SPHR")
    are one any-of group; skills joined by "or" inside one comma-item are
    one group; everything else is one skill per group."""
    extractor = get_default_extractor()
    mentions = extractor.find_mentions(clause)
    certs = [m.canonical for m in mentions if extractor.skill_type(m.canonical) == "certification"]
    cert_groups = [list(dict.fromkeys(certs))] if certs else []

    skill_groups: list[list[str]] = []
    seen: set[str] = set()
    for item in re.split(r",|;", clause):
        item_mentions = [m.canonical for m in extractor.find_mentions(item)
                         if extractor.skill_type(m.canonical) != "certification"]
        item_mentions = [s for s in dict.fromkeys(item_mentions) if s not in seen]
        if not item_mentions:
            continue
        if len(item_mentions) > 1 and re.search(r"\bor\b|/", item, re.IGNORECASE):
            skill_groups.append(item_mentions)
        else:
            skill_groups.extend([s] for s in item_mentions)
        seen.update(item_mentions)
    return skill_groups, cert_groups


def _free_text_items(clause: str) -> list[str]:
    """Short list items with no dictionary skill (e.g. "Kubeflow", "menu
    costing") are kept as literal requirement phrases. Long sentences are
    not turned into requirements -- they are left to the user to add."""
    items = []
    body = _BULLET_RE.sub("", clause)
    for item in re.split(r",|;", body):
        item = item.strip(" .:")
        item = re.sub(r"^(?:and|or)\s+", "", item, flags=re.IGNORECASE)
        words = item.split()
        if not item or len(words) > 4 or _YEARS_REQ_RE.search(item) or education_levels_in(item):
            continue
        if _PREFERRED_CUE_RE.search(item) or _ESSENTIAL_CUE_RE.search(item):
            continue
        if not re.search(r"[A-Za-z]", item):
            continue
        items.append(item)
    return items


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def extract_job_requirements(job_text: str, kinds: Iterable[str] = ("skill", "experience", "education", "certification")) -> dict:
    """Extract requirements from free-text job description.

    Returns {"title", "requirements": [...], "warnings": [...],
    "structured": bool} where `structured` says whether explicit
    requirement/preferred sections were found.
    """
    kinds = set(kinds)
    result = {"title": None, "requirements": [], "warnings": [], "structured": False}
    if not job_text or not job_text.strip():
        result["warnings"].append("Job description text is empty.")
        return result

    lines = [ln.rstrip() for ln in job_text.replace("\r", "").split("\n")]
    first = next((ln.strip() for ln in lines if ln.strip()), "")
    if first and len(first.split()) <= 10 and _header_of(first)[0] is None:
        result["title"] = first.strip("-= ")

    reqs: list = result["requirements"]
    skill_seen: set[tuple] = set()
    section = None
    any_section = any(_header_of(ln)[0] in ("required", "preferred", "must") for ln in lines if ln.strip())
    experience_found = {"required": False, "preferred": False}
    education_found = {"required": False, "preferred": False}

    for raw_line in lines:
        if not raw_line.strip() or set(raw_line.strip()) <= set("-=_*"):
            continue
        header, remainder = _header_of(raw_line)
        if header is not None:
            section = header
            if not remainder.strip():
                continue
            line = remainder
        else:
            line = raw_line

        for clause in _clauses(_BULLET_RE.sub("", line)):
            importance = _importance_for(section or "", clause)
            essential = _essential_for(section or "", clause, importance)
            in_req_section = section in ("required", "preferred", "must")

            if "experience" in kinds and not experience_found[importance]:
                years = _YEARS_REQ_RE.search(clause)
                if years and (re.search(r"experience", clause, re.IGNORECASE) or in_req_section):
                    min_years = float(years.group(1))
                    if min_years > 0:
                        reqs.append(make_experience_requirement(
                            min_years, importance, essential or in_req_section and importance == "required",
                            clause, _new_id(reqs)))
                        experience_found[importance] = True

            if "education" in kinds and not education_found[importance]:
                parsed = parse_education_clause(clause)
                if parsed:
                    level, fields, related = parsed
                    reqs.append(make_education_requirement(
                        level, fields, related, importance,
                        essential or (in_req_section and importance == "required"), clause, _new_id(reqs)))
                    education_found[importance] = True

            if any_section and section not in ("required", "preferred", "must"):
                continue  # responsibilities/about text is context, not a requirement list

            skill_groups, cert_groups = _skills_in_clause(clause)
            if "certification" in kinds:
                for group in cert_groups:
                    key = ("certification", tuple(group))
                    if key not in skill_seen:
                        skill_seen.add(key)
                        reqs.append(make_skill_requirement(group, importance, essential, clause, "certification", _new_id(reqs)))
            if "skill" in kinds:
                groups = skill_groups
                if not groups and in_req_section:
                    groups = [[canonicalize_skill(item)] for item in _free_text_items(clause)]
                for group in groups:
                    key = ("skill", tuple(group))
                    if key not in skill_seen and not any(("skill", (s,)) in skill_seen for s in group if len(group) > 1):
                        skill_seen.add(key)
                        reqs.append(make_skill_requirement(group, importance, essential, clause, "skill", _new_id(reqs)))

    result["structured"] = any_section
    if not any_section:
        result["warnings"].append(
            "No explicit 'Requirements' / 'Required skills' / 'Preferred' section was detected. "
            "All dictionary skills found in the text were treated as required, non-essential "
            "requirements. Please review and correct the list before scoring."
        )
    if not any(r["kind"] in ("skill", "certification") for r in reqs):
        result["warnings"].append("No skills could be extracted from the job description; add them manually.")
    return result


def requirements_from_dataset_record(job_text: str, required_skills: list) -> list:
    """Requirements for a row of the HF `job_resume_fit` dataset.

    Skills come from the dataset's structured `job_required_skills` list
    (canonicalized with the same `canonicalize_skill` the app uses);
    experience / education / certification requirements are extracted
    from `job_text` with exactly the same extractor the app uses.
    """
    reqs: list = []
    seen = set()
    for name in required_skills or []:
        canonical = canonicalize_skill(str(name))
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        reqs.append(make_skill_requirement([canonical], "required", False, str(name), "skill", _new_id(reqs)))
    other = extract_job_requirements(job_text, kinds=("experience", "education", "certification"))["requirements"]
    for req in other:
        req["id"] = _new_id(reqs)
        reqs.append(req)
    return reqs


# ----------------------------------------------------------------------------
# Editable-table round trip (used by the Streamlit requirement editor)
# ----------------------------------------------------------------------------
TABLE_COLUMNS = ["Include", "Type", "Requirement", "Importance", "Must-have"]
_KIND_TO_TYPE = {"skill": "Skill", "experience": "Experience", "education": "Education", "certification": "Certification"}
_TYPE_TO_KIND = {v: k for k, v in _KIND_TO_TYPE.items()}


def requirements_to_table(requirements: list) -> list[dict]:
    return [
        {
            "Include": True,
            "Type": _KIND_TO_TYPE[r["kind"]],
            "Requirement": r["label"],
            "Importance": r["importance"].title(),
            "Must-have": bool(r["essential"]),
        }
        for r in requirements
    ]


def requirements_from_table(rows: list[dict], original: Optional[list] = None) -> tuple[list, list[str]]:
    """Rebuild requirement dicts from (possibly user-edited) table rows.
    Returns (requirements, warnings). Rows that cannot be interpreted are
    skipped with a warning rather than guessed."""
    original = original or []
    source_by_label = {(r["kind"], r["label"]): r.get("source_text", "") for r in original}
    reqs: list = []
    warnings: list[str] = []
    for i, row in enumerate(rows, start=1):
        if not row.get("Include", True):
            continue
        text = str(row.get("Requirement") or "").strip()
        kind = _TYPE_TO_KIND.get(str(row.get("Type") or "Skill"), "skill")
        if not text:
            continue
        importance = "preferred" if str(row.get("Importance", "Required")).lower().startswith("pref") else "required"
        essential = bool(row.get("Must-have")) and importance == "required"
        source = source_by_label.get((kind, text), "")
        rid = _new_id(reqs)

        if kind in ("skill", "certification"):
            # "Python or R", "AWS / GCP", "A | B" -> alternatives; "UI/UX", "CI/CD" stay whole.
            alternatives = [canonicalize_skill(p) for p in re.split(r"\s+or\s+|\s+/\s+|\s*\|\s*", text) if p.strip()]
            alternatives = [a for a in dict.fromkeys(alternatives) if a]
            reqs.append(make_skill_requirement(alternatives, importance, essential, source, kind, rid))
        elif kind == "experience":
            years = _YEARS_REQ_RE.search(text) or re.search(r"(\d+(?:\.\d+)?)", text)
            if not years:
                warnings.append(f"Row {i}: could not read a number of years from '{text}'; row ignored.")
                continue
            reqs.append(make_experience_requirement(float(years.group(1)), importance, essential, source, rid))
        else:
            parsed = parse_education_clause(text)
            if not parsed:
                warnings.append(
                    f"Row {i}: could not recognise a degree level in '{text}' "
                    "(e.g. Bachelor's, Master's, PhD, Diploma); row ignored."
                )
                continue
            level, fields, related = parsed
            if "(or related field)" in text:
                related = True
            reqs.append(make_education_requirement(level, fields, related, importance, essential, source, rid))
    return reqs, warnings
