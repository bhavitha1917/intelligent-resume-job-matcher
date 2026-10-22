"""CSV and per-candidate report exports. Exports contain scores, statuses
and the short evidence snippets already shown on screen -- never the full
resume text or contact details."""

import pandas as pd

from src.scoring.scorer import INTERPRETATION

STATUS_SYMBOLS = {"met": "Detected", "partial": "Partial / ambiguous", "not_detected": "Not detected",
                  "contradicted": "Contradicted (resume states a lack)"}

CSV_COLUMNS = [
    "rank", "tied", "candidate", "status", "score", "band", "required_skills_detected", "required_skills_total",
    "preferred_detected", "preferred_total", "experience_years_detected", "highest_degree_detected",
    "must_have_gaps", "contradicted", "not_detected", "warnings", "rubric_version", "rubric_fingerprint", "reference_date",
]


def results_to_dataframe(rows: list) -> pd.DataFrame:
    records = []
    for row in rows:
        report = row.get("report")
        summary = report["summary"] if report else {}
        records.append({
            "rank": row.get("rank"),
            "tied": row.get("tied", False),
            "candidate": row["candidate_id"],
            "status": row["status"],
            "score": row.get("score"),
            "band": row.get("band"),
            "required_skills_detected": summary.get("required_skills_met"),
            "required_skills_total": summary.get("required_skills_total"),
            "preferred_detected": summary.get("preferred_met"),
            "preferred_total": summary.get("preferred_total"),
            "experience_years_detected": summary.get("experience_years_detected"),
            "highest_degree_detected": summary.get("highest_degree"),
            "must_have_gaps": "; ".join(report["cap_reasons"]) if report else "",
            "contradicted": "; ".join(summary.get("contradicted", [])),
            "not_detected": "; ".join(summary.get("not_detected", [])),
            "warnings": "; ".join(report["warnings"] if report else (row.get("parse_warnings") or [])),
            "rubric_version": report["config"]["version"] if report else None,
            "rubric_fingerprint": report["config"]["fingerprint"] if report else None,
            "reference_date": report["reference_date"] if report else None,
        })
    return pd.DataFrame(records, columns=CSV_COLUMNS)


def results_to_csv(rows: list) -> bytes:
    return results_to_dataframe(rows).to_csv(index=False).encode("utf-8")


def report_to_markdown(report: dict) -> str:
    lines = [f"# Candidate report: {report.get('candidate_id') or 'resume'}", ""]
    if report["status"] == "scored":
        lines.append(f"**Rubric alignment score:** {report['score']:.1f} / 100 — {report['band']}")
        if report["cap_applied"]:
            lines.append(f"(Uncapped weighted score {report['uncapped_score']:.1f}; capped because a must-have "
                         "requirement or minimum required-skill evidence was not met.)")
    else:
        lines.append("**Insufficient evidence — no score shown.** " + " ".join(report["insufficient_reasons"]))
    lines += ["", f"_{INTERPRETATION}_", "", "## Score components", "",
              "| Component | Weight | Applies? | Value | Contribution |", "|---|---|---|---|---|"]
    for c in report["components"]:
        value = f"{c['value']:.2f}" if c["applicable"] else "—"
        contrib = f"{c['contribution']:.1f}" if c["applicable"] else "—"
        lines.append(f"| {c['label']} | {c['weight']:.2f} (eff. {c['effective_weight']:.2f}) | "
                     f"{'yes' if c['applicable'] else 'no'} | {value} | {contrib} |")
    if report["cap_reasons"]:
        lines += ["", "**Gating rules triggered:**"] + [f"- {r}" for r in report["cap_reasons"]]
    lines += ["", "## Requirements", ""]
    for r in report["requirements"]:
        res = r["result"]
        tag = f"{r['importance']}{', must-have' if r['essential'] else ''}"
        lines.append(f"- **{r['label']}** ({r['kind']}, {tag}) — {STATUS_SYMBOLS[res['status']]}, credit {res['credit']:.2f}")
        for ev in res.get("evidence", [])[:2]:
            lines.append(f"    - “{ev['text']}” _(section: {ev['section']}, {ev['kind']})_")
        for note in res.get("notes", []):
            lines.append(f"    - Note: {note}")
    if report["projects"]:
        lines += ["", "## Projects (evidence context)"]
        for p in report["projects"]:
            lines.append(f"- {p['text']}" + (f" — job skills: {', '.join(p['skills'])}" if p["skills"] else ""))
    if report["warnings"]:
        lines += ["", "## Extraction warnings"] + [f"- {w}" for w in report["warnings"]]
    removed = report["privacy"]["removed_before_scoring"]
    lines += ["", "## Reproducibility", f"- Rubric version {report['config']['version']}, fingerprint "
              f"`{report['config']['fingerprint']}`", f"- Reference date for 'Present': {report['reference_date']}",
              f"- Removed before scoring: {', '.join(removed) if removed else 'nothing detected'}", ""]
    return "\n".join(lines)
