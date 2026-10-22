"""Resume evidence extraction, one function per requirement kind.

Every result carries a `status`:

    "met"           evidence detected in the resume
    "partial"       evidence is incomplete or ambiguous (e.g. 1 of 2 required
                    years, degree field not matched, degree in progress)
    "not_detected"  nothing found in the resume text. This is NOT proof the
                    candidate lacks the qualification -- only that the
                    submitted resume does not show it.
    "contradicted"  the resume explicitly states a lack ("no experience
                    with Python")

plus a `credit` in [0, 1] taken from the rubric, and `evidence` items
quoting the resume text that justified the decision. Nothing is inferred
beyond what is quoted.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from datetime import date
from typing import Optional

from src.scoring.requirements import EDUCATION_LEVEL_NAMES, education_levels_in
from src.skills.skill_extractor import phrase_pattern, skill_patterns

# Clause boundaries. A bare newline is NOT a boundary (resumes wrap long
# lines: "(no formal\n machine learning framework used)"); a newline that
# starts a bullet, or a blank line, is.
_SENTENCE_BOUNDARY_RE = re.compile(r"[.;:!?•●▪|]|\n\s*\n|\n\s*[-*]\s|[ \t]{3,}")
_NEGATION_TOKENS = {"no", "not", "never", "without", "lack", "lacks", "lacking", "none", "nor", "unfamiliar", "zero"}
_NEGATION_PHRASES_RE = re.compile(r"\b(?:no|little|zero)\s+(?:prior\s+|formal\s+|hands-on\s+|professional\s+)?(?:experience|exposure|knowledge)\s+(?:with|in|of|using)\b", re.IGNORECASE)
_QUOTED_LINE_RE = re.compile(
    r"^\s*(?:required skills|requirements|job requirements|job description|the (?:job|role|position) requires|"
    r"target role requires|must[- ]have skills|jd)\s*[:\-]",
    re.IGNORECASE,
)


@dataclass
class ResumeDoc:
    """Sanitized resume text plus section offsets (see resume_parser)."""
    text: str
    section_spans: list = field(default_factory=list)

    @property
    def has_sections(self) -> bool:
        return any(name not in ("header",) for name, _, _ in self.section_spans)

    def section_at(self, pos: int) -> str:
        for name, start, end in self.section_spans:
            if start <= pos < end:
                return name
        return "unsectioned"

    def spans_of(self, *names: str) -> list[tuple[int, int]]:
        return [(s, e) for n, s, e in self.section_spans if n in names]

    def line_at(self, start: int, end: int, max_chars: int = 220) -> str:
        line_start = self.text.rfind("\n", 0, start) + 1
        line_end = self.text.find("\n", end)
        line_end = len(self.text) if line_end == -1 else line_end
        if line_end - line_start > max_chars:
            line_start = max(line_start, start - max_chars // 2)
            line_end = min(line_end, end + max_chars // 2)
        snippet = re.sub(r"\s+", " ", self.text[line_start:line_end]).strip()
        return snippet


def _clause_before(text: str, pos: int, max_chars: int = 120) -> str:
    window = text[max(0, pos - max_chars):pos]
    parts = _SENTENCE_BOUNDARY_RE.split(window)
    return (parts[-1] if parts else window).replace("\n", " ")


def is_negated(text: str, start: int, window_tokens: int) -> bool:
    """True if a negation cue precedes the mention within the same clause,
    at most `window_tokens` tokens earlier ("no formal machine learning",
    "not familiar with Docker", "no experience with SQL"). Documented
    limitation: negation that FOLLOWS the mention ("Python: none") and
    double negatives are not handled."""
    clause = _clause_before(text, start)
    tokens = re.findall(r"[A-Za-z']+", clause.lower())[-window_tokens:]
    if any(t in _NEGATION_TOKENS or t.endswith("n't") for t in tokens):
        return True
    return bool(_NEGATION_PHRASES_RE.search(clause))


def _is_quoted_requirement(doc: ResumeDoc, start: int) -> bool:
    if doc.section_at(start) == "quoted_requirements":
        return True
    line_start = doc.text.rfind("\n", 0, start) + 1
    return bool(_QUOTED_LINE_RE.match(doc.text[line_start:start + 1]))


# ----------------------------------------------------------------------------
# Skills and certifications
# ----------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _action_verb_re(verbs: tuple) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(map(re.escape, verbs)) + r")\b", re.IGNORECASE)


def skill_mentions(doc: ResumeDoc, alternative: str, rubric: dict) -> list[dict]:
    """Every textual mention of one skill, classified."""
    action_verb_re = _action_verb_re(tuple(rubric["action_verbs"]))
    applied_sections = set(rubric["applied_evidence_sections"])
    mentions = []
    seen_spans = set()
    for pattern in skill_patterns(alternative):
        for match in pattern.finditer(doc.text):
            if match.span() in seen_spans:
                continue
            seen_spans.add(match.span())
            start, end = match.span()
            section = doc.section_at(start)
            line = doc.line_at(start, end)
            if _is_quoted_requirement(doc, start):
                kind = "quoted"
            elif is_negated(doc.text, start, rubric["negation_window_tokens"]):
                kind = "contradicted"
            elif section in applied_sections or action_verb_re.search(_clause_before(doc.text, start, 160) + doc.text[end:end + 60].split("\n")[0]):
                kind = "applied"
            else:
                kind = "listed"
            mentions.append({"skill": alternative, "kind": kind, "section": section, "text": line, "start": start})
    return sorted(mentions, key=lambda m: m["start"])


_KIND_RANK = {"applied": 3, "listed": 2, "contradicted": 1, "quoted": 0}


def evaluate_skill_requirement(req: dict, doc: ResumeDoc, rubric: dict) -> dict:
    credit_table = rubric["skill_evidence_credit"]
    all_mentions = [m for alt in req["alternatives"] for m in skill_mentions(doc, alt, rubric)]
    usable = [m for m in all_mentions if m["kind"] != "quoted"]
    notes = []
    if any(m["kind"] == "quoted" for m in all_mentions):
        notes.append("A mention inside a quoted job requirement / job description in the resume was ignored.")

    if not usable:
        return {"status": "not_detected", "credit": credit_table["not_detected"], "evidence": [],
                "evidence_level": None, "matched": None, "notes": notes}

    best = max(usable, key=lambda m: _KIND_RANK[m["kind"]])
    positive = [m for m in usable if m["kind"] in ("applied", "listed")]
    negative = [m for m in usable if m["kind"] == "contradicted"]
    if positive and negative:
        notes.append("The resume contains both a positive mention and a negated statement about this skill; review manually.")

    if best["kind"] == "contradicted":
        status, level = "contradicted", "contradicted"
    else:
        status, level = "met", best["kind"]

    if req["kind"] == "certification":
        credit = 1.0 if status == "met" else 0.0
    else:
        credit = credit_table[level]

    shown = sorted(usable, key=lambda m: (-_KIND_RANK[m["kind"]], m["start"]))[:3]
    return {
        "status": status, "credit": credit, "evidence_level": level, "matched": best["skill"],
        "evidence": [{"text": m["text"], "section": m["section"], "kind": m["kind"]} for m in shown],
        "notes": notes,
    }


# ----------------------------------------------------------------------------
# Education
# ----------------------------------------------------------------------------
_IN_PROGRESS_RE = re.compile(r"\b(pursuing|expected|ongoing|currently enrolled|candidate|anticipated|in progress|final[- ]year|(?:1st|2nd|3rd|4th|first|second|third|fourth)[- ]year)\b", re.IGNORECASE)
_FIELD_SYNONYMS = {
    "computer science": ["computer science", "computer science and engineering", "cse", "computing"],
    "information technology": ["information technology"],
    "statistics": ["statistics", "statistical science"],
    "mathematics": ["mathematics", "maths", "applied mathematics"],
    "business administration": ["business administration", "business management", "mba", "bba"],
    "electrical engineering": ["electrical engineering", "electrical and electronics engineering", "eee"],
    "electronics": ["electronics", "electronics and communication", "ece"],
}


def _field_patterns(field_name: str) -> list:
    names = _FIELD_SYNONYMS.get(field_name, [field_name])
    return [phrase_pattern(n, allow_plural=True) for n in names]


def detect_degrees(doc: ResumeDoc, reference_date: date) -> list[dict]:
    spans = doc.spans_of("education") or [(0, len(doc.text))]
    degrees = []
    for start, end in spans:
        segment = doc.text[start:end]
        for level, match in education_levels_in(segment):
            abs_start = start + match.start()
            line = doc.line_at(abs_start, start + match.end(), max_chars=200)
            years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", line)]
            in_progress = bool(_IN_PROGRESS_RE.search(line)) or (bool(years) and max(years) > reference_date.year)
            degrees.append({"level": level, "level_name": EDUCATION_LEVEL_NAMES[level], "text": line,
                            "in_progress": in_progress, "start": abs_start})
    return degrees


def evaluate_education_requirement(req: dict, doc: ResumeDoc, rubric: dict, reference_date: date) -> dict:
    cfg = rubric["education"]
    degrees = detect_degrees(doc, reference_date)
    if not degrees:
        return {"status": "not_detected", "credit": 0.0, "evidence": [],
                "notes": ["No degree or qualification level was recognised in the resume."]}

    qualifying = [d for d in degrees if d["level"] >= req["level"]]
    if not qualifying:
        top = max(degrees, key=lambda d: d["level"])
        return {"status": "partial", "credit": 0.0, "gate_met": False,
                "evidence": [{"text": top["text"], "section": "education", "kind": "degree"}],
                "notes": [f"Highest qualification detected: {top['level_name']}; the job asks for {req['level_name']} or higher."]}

    def field_ok(d):
        if not req.get("fields"):
            return True
        return any(p.search(d["text"]) for f in req["fields"] for p in _field_patterns(f))

    best = sorted(qualifying, key=lambda d: (not d["in_progress"], field_ok(d), d["level"]), reverse=True)[0]
    notes, credit, status = [], 1.0, "met"
    if not field_ok(best):
        credit *= cfg["unmatched_field_credit"]
        status = "partial"
        notes.append(
            "Degree level meets the requirement but none of the requested fields ("
            + ", ".join(req["fields"]) + ") was recognised"
            + ("; the job accepts related fields, so verify manually." if req.get("allow_related_field") else ".")
        )
    if best["in_progress"]:
        credit *= cfg["in_progress_credit"]
        status = "partial"
        notes.append("The qualifying degree appears to be in progress / not yet completed.")
    # For must-have gating, a degree at the required level counts as meeting the
    # requirement when its field matches OR the job accepts related fields
    # (a related field cannot be ruled out by keyword matching).
    gate_met = not best["in_progress"] and (field_ok(best) or bool(req.get("allow_related_field")))
    return {"status": status, "credit": round(credit, 4), "gate_met": gate_met,
            "evidence": [{"text": best["text"], "section": "education", "kind": "degree"}], "notes": notes}


# ----------------------------------------------------------------------------
# Experience (employment + internships)
# ----------------------------------------------------------------------------
_MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH_RE = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_RE = rf"(?:{_MONTH_RE}\s*,?\s*(?:19|20)\d{{2}}|\d{{1,2}}\s*[/\-.]\s*(?:19|20)\d{{2}}|(?:19|20)\d{{2}})"
_END_RE = rf"(?:{_DATE_RE}|present|current|currently|now|till date|to date|date|ongoing)"
DATE_RANGE_RE = re.compile(rf"({_DATE_RE})\s*(?:-|–|—|to|till|until)\s*({_END_RE})", re.IGNORECASE)
_CLAIMED_YEARS_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[a-z\-]+\s+){0,3}?(?:experience|exp)\b", re.IGNORECASE)
_INTERN_RE = re.compile(r"\bintern(?:ship)?s?\b|\btrainee\b|\bapprentice(?:ship)?\b", re.IGNORECASE)
_GENERIC_TITLE_WORDS = {
    "senior", "junior", "mid", "level", "lead", "intern", "trainee", "associate", "entry", "i", "ii", "iii",
    "the", "and", "of", "for", "to", "a", "an", "in", "with", "remote", "hybrid", "full", "time", "part",
    "sr", "jr", "principal", "staff", "head",
}


def _parse_date(text: str, is_end: bool, reference_date: date) -> Optional[tuple[int, int]]:
    t = text.strip().lower()
    if re.fullmatch(r"present|current|currently|now|till date|to date|date|ongoing", t):
        return reference_date.year, reference_date.month
    year_match = re.search(r"(?:19|20)\d{2}", t)
    if not year_match:
        return None
    year = int(year_match.group(0))
    month_name = re.match(r"[a-z]{3}", t)
    if month_name and month_name.group(0) in _MONTHS:
        return year, _MONTHS[month_name.group(0)]
    numeric = re.match(r"(\d{1,2})\s*[/\-.]", t)
    if numeric and 1 <= int(numeric.group(1)) <= 12:
        return year, int(numeric.group(1))
    return year, 1  # year only: counted from/to January (conservative, documented)


def _merge_months(intervals: list[tuple[int, int]]) -> int:
    total, current = 0, None
    for start, end in sorted(intervals):
        if current is None or start > current[1]:
            if current:
                total += current[1] - current[0]
            current = [start, end]
        else:
            current[1] = max(current[1], end)
    if current:
        total += current[1] - current[0]
    return total


def _title_terms(job_title: Optional[str]) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z+#.]{2,}", job_title or "")
    return [w for w in words if w.lower() not in _GENERIC_TITLE_WORDS]


def extract_experience_entries(doc: ResumeDoc, reference_date: date, relevance_terms: list, job_title: Optional[str]) -> dict:
    """Dated employment/internship entries and their (merged) duration."""
    work_spans = doc.spans_of("experience", "internships")
    ambiguous = False
    if not work_spans:
        if doc.has_sections:
            excluded = doc.spans_of("education", "certifications", "achievements", "personal", "header")
            work_spans = [(0, len(doc.text))]
        else:
            excluded = []
            work_spans = [(0, len(doc.text))]
        ambiguous = True
    else:
        excluded = []

    relevance_patterns = []
    for term in relevance_terms:
        relevance_patterns.extend(skill_patterns(term))
    relevance_patterns.extend(phrase_pattern(t) for t in _title_terms(job_title))

    entries = []
    for span_start, span_end in work_spans:
        matches = [m for m in DATE_RANGE_RE.finditer(doc.text, span_start, span_end)
                   if not any(s <= m.start() < e for s, e in excluded)]
        for i, match in enumerate(matches):
            start = _parse_date(match.group(1), False, reference_date)
            end = _parse_date(match.group(2), True, reference_date)
            if not start or not end:
                continue
            start_m, end_m = start[0] * 12 + start[1], end[0] * 12 + end[1]
            if end_m < start_m or end_m - start_m > 50 * 12:
                continue
            line_start = doc.text.rfind("\n", 0, match.start()) + 1
            entry_start = max(span_start, line_start)
            entry_end = matches[i + 1].start() if i + 1 < len(matches) else span_end
            entry_text = doc.text[entry_start:entry_end]
            heading = re.sub(r"\s+", " ", doc.text[max(span_start, match.start() - 120):match.end() + 160]).strip()
            section = doc.section_at(match.start())
            is_internship = section == "internships" or bool(_INTERN_RE.search(heading))
            relevant = (not relevance_patterns) or any(p.search(entry_text) for p in relevance_patterns)
            entries.append({
                "dates": match.group(0).strip(), "months": end_m - start_m, "interval": (start_m, end_m),
                "kind": "internship" if is_internship else "employment", "relevant": relevant,
                "text": re.sub(r"\s+", " ", entry_text).strip()[:240], "section": section,
            })

    claimed = []
    for match in _CLAIMED_YEARS_RE.finditer(doc.text):
        if not is_negated(doc.text, match.start(), 4):
            claimed.append({"years": float(match.group(1)), "text": doc.line_at(match.start(), match.end())})

    return {"entries": entries, "claimed": claimed, "ambiguous_sections": ambiguous}


def evaluate_experience_requirement(req: dict, doc: ResumeDoc, rubric: dict, reference_date: date,
                                    relevance_terms: list, job_title: Optional[str]) -> dict:
    cfg = rubric["experience"]
    found = extract_experience_entries(doc, reference_date, relevance_terms, job_title)
    entries = found["entries"]
    counted = [e for e in entries
               if (cfg["count_internships"] or e["kind"] != "internship")
               and (e["relevant"] or not cfg["count_only_relevant_entries"])]
    months = _merge_months([e["interval"] for e in counted])
    years = round(months / 12, 2)
    min_years = req["min_years"]
    notes = []
    evidence = [{"text": f"{e['dates']}: {e['text']}", "section": e["section"], "kind": e["kind"]} for e in counted[:4]]

    skipped = [e for e in entries if e not in counted]
    if skipped:
        notes.append(f"{len(skipped)} dated entr{'y was' if len(skipped) == 1 else 'ies were'} not counted "
                     "(no job-relevant terms detected, or internship counting disabled).")
    if found["ambiguous_sections"] and entries:
        notes.append("No experience/internship section header was detected, so date ranges were read from the "
                     "whole resume (excluding detected education sections); durations may be inaccurate.")

    if years > 0:
        credit = min(1.0, years / min_years)
        status = "met" if years >= min_years and not found["ambiguous_sections"] else "partial"
        notes.insert(0, f"About {years:g} year(s) of relevant dated experience detected (overlapping periods merged).")
        return {"status": status, "credit": round(credit, 4), "evidence": evidence, "notes": notes,
                "detected_years": years, "entries": entries}

    if found["claimed"]:
        claimed_years = max(c["years"] for c in found["claimed"])
        credit = min(1.0, claimed_years / min_years) * cfg["claimed_years_credit_factor"]
        notes.insert(0, f"{claimed_years:g} year(s) of experience is stated in the text but could not be "
                        "verified from dated entries.")
        return {"status": "met" if claimed_years >= min_years else "partial", "credit": round(credit, 4),
                "evidence": [{"text": c["text"], "section": "stated", "kind": "claimed"} for c in found["claimed"][:2]],
                "notes": notes, "detected_years": None, "claimed_years": claimed_years, "entries": entries}

    if entries:
        notes.insert(0, "Dated entries were found but none could be counted as relevant experience.")
        return {"status": "partial", "credit": 0.0, "evidence": evidence or
                [{"text": f"{e['dates']}: {e['text']}", "section": e["section"], "kind": e["kind"]} for e in entries[:3]],
                "notes": notes, "detected_years": 0.0, "entries": entries}

    has_section = bool(doc.spans_of("experience", "internships"))
    notes.insert(0, "An experience section was found but no durations could be extracted." if has_section
                 else "No dated employment or internship entries were detected.")
    return {"status": "partial" if has_section else "not_detected", "credit": 0.0, "evidence": [],
            "notes": notes, "detected_years": None, "entries": entries}


# ----------------------------------------------------------------------------
# Projects (display-only evidence; contributes to the score only through the
# "applied" skill-evidence credit, so it is never counted twice)
# ----------------------------------------------------------------------------
def project_evidence(doc: ResumeDoc, skill_alternatives: list[str]) -> list[dict]:
    items = []
    for start, end in doc.spans_of("projects"):
        block = doc.text[start:end]
        parts = [p for p in re.split(r"\n\s*(?=[-*•●▪]|\d+[.)])|\n{2,}", block) if p.strip()]
        for part in parts:
            matched = sorted({alt for alt in skill_alternatives if any(p.search(part) for p in skill_patterns(alt))})
            items.append({"text": re.sub(r"\s+", " ", part).strip(" -*•")[:240], "skills": matched})
    return items
