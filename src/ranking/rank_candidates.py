"""Multi-resume candidate ranking.

Every resume is scored with exactly the same function, job requirements,
rubric and reference date as the Single Resume Analysis page
(src.scoring.scorer.score_candidate), so a candidate's score here is
identical to the score on their individual report.

Every uploaded file produces exactly one row; nothing is dropped silently:
    status "scored"                 -> ranked
    status "insufficient_evidence"  -> listed after ranked rows, no rank
    status "parse_failed"           -> listed, with the parser's reason
    status "duplicate"              -> listed, pointing at the identical file
                                       that was scored (scored once only)

Tie rule (also in config/scoring_rubric.json): equal scores at the
displayed precision share a rank (1, 2, 2, 4). Within a tie, rows are
listed alphabetically by file name; that order carries no meaning.
"""

import hashlib
from datetime import date
from pathlib import Path
from typing import Callable, Optional

from src.data.resume_parser import parse_resume
from src.scoring.config import load_rubric
from src.scoring.requirements import extract_job_requirements
from src.scoring.scorer import DEFAULT_SEMANTIC, score_candidate


def _content_hash(source) -> str:
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists() and "\n" not in source):
        data = Path(source).read_bytes()
    elif isinstance(source, bytes):
        data = source
    else:
        data = str(source).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_job(job_description: str, requirements: Optional[list] = None, title: Optional[str] = None) -> dict:
    """Job dict for the scorer. If `requirements` is None they are extracted
    from the text (the app passes the user-reviewed list instead)."""
    extracted = extract_job_requirements(job_description)
    return {
        "title": title or extracted["title"],
        "text": job_description or "",
        "requirements": extracted["requirements"] if requirements is None else requirements,
        "extraction_warnings": extracted["warnings"],
    }


def assign_ranks(rows: list, decimals: int = 1) -> list:
    """Standard competition ranking over scored rows; sorts in place."""
    scored = sorted((r for r in rows if r["status"] == "scored"),
                    key=lambda r: (-round(r["score"], decimals), str(r["candidate_id"]).lower()))
    previous, rank = None, 0
    for position, row in enumerate(scored, start=1):
        key = round(row["score"], decimals)
        if key != previous:
            rank, previous = position, key
        row["rank"] = rank
        row["tied"] = False
    for row in scored:
        row["tied"] = sum(1 for other in scored if other["rank"] == row["rank"]) > 1
    order = {"insufficient_evidence": 1, "parse_failed": 2, "duplicate": 3}
    rest = sorted((r for r in rows if r["status"] != "scored"),
                  key=lambda r: (order.get(r["status"], 9), str(r["candidate_id"]).lower()))
    for row in rest:
        row["rank"] = None
        row["tied"] = False
    return scored + rest


def rank_candidates(
    job_description: str,
    resumes: list,
    requirements: Optional[list] = None,
    rubric: Optional[dict] = None,
    semantic=DEFAULT_SEMANTIC,
    compute_semantic: Optional[bool] = None,
    reference_date: Optional[date] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> list:
    """Score and rank N resumes against one job.

    `resumes`: list of {"candidate_id", "source" (path / bytes / text),
    "filename" (needed when source is bytes)}.
    `compute_semantic=False` is a shorthand for `semantic=None`.
    `progress(i, n, candidate_id)` is called before each file is processed.
    """
    rubric = rubric or load_rubric()
    reference_date = reference_date or date.today()
    if compute_semantic is False:
        semantic = None
    job = build_job(job_description, requirements)

    rows = []
    seen_hashes: dict[str, str] = {}
    for i, entry in enumerate(resumes):
        candidate_id = entry.get("candidate_id") or entry.get("filename") or f"candidate_{i + 1}"
        if progress:
            progress(i, len(resumes), candidate_id)
        digest = _content_hash(entry["source"])
        if digest in seen_hashes:
            rows.append({"candidate_id": candidate_id, "status": "duplicate", "duplicate_of": seen_hashes[digest],
                         "score": None, "band": "Duplicate file", "report": None,
                         "parse_warnings": [f"Identical content to '{seen_hashes[digest]}'; scored once under that name."]})
            continue
        seen_hashes[digest] = candidate_id

        parsed = parse_resume(entry["source"], entry.get("filename"))
        if not (parsed["full_text"] or "").strip():
            rows.append({"candidate_id": candidate_id, "status": "parse_failed", "score": None,
                         "band": "Could not read file", "report": None, "parse_warnings": parsed["parse_warnings"]})
            continue

        report = score_candidate(job, parsed, rubric=rubric, semantic=semantic,
                                 reference_date=reference_date, candidate_id=candidate_id)
        rows.append({"candidate_id": candidate_id, "status": report["status"], "score": report["score"],
                     "band": report["band"], "report": report, "parse_warnings": parsed["parse_warnings"]})

    if progress:
        progress(len(resumes), len(resumes), "")
    return assign_ranks(rows, rubric["ranking"]["score_decimals"])
