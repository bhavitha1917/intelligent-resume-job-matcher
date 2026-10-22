"""Shared Streamlit UI pieces. Every page renders reports and edits
requirements through these functions, so the single-resume page, the
ranking page and the details page cannot drift apart."""

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.resume_parser import SUPPORTED_EXTENSIONS, parse_resume
from src.scoring.config import load_rubric
from src.scoring.export import STATUS_SYMBOLS, report_to_markdown
from src.scoring.requirements import TABLE_COLUMNS, extract_job_requirements, requirements_from_table, requirements_to_table
from src.utils.paths import DEMO_DATA_DIR

APP_DIR = Path(__file__).resolve().parents[1]
ICON_PATH = APP_DIR / "static" / "icon-192.png"
UPLOAD_TYPES = [ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS]
STATUS_ICON = {"met": "✅", "partial": "🟡", "not_detected": "⚪", "contradicted": "⛔"}

STYLES_PATH = Path(__file__).resolve().parent / "styles.css"

DECISION_SUPPORT_NOTE = (
    "Decision support only: scores summarise evidence found in resume text under a documented rubric. "
    "They require human review and must not be the sole basis for a hiring decision."
)


def setup_page(title: str, layout: str = "centered") -> None:
    st.set_page_config(page_title=f"{title} · Resume–Job Matching",
                       page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🧭", layout=layout)
    st.markdown(f"<style>{_stylesheet()}</style>", unsafe_allow_html=True)


def _stylesheet() -> str:
    return STYLES_PATH.read_text(encoding="utf-8")


def semantic_setting() -> bool:
    """One app-wide setting, shared by all pages so scores agree."""
    if "use_semantic" not in st.session_state:
        st.session_state["use_semantic"] = True
    with st.sidebar:
        st.session_state["use_semantic"] = st.toggle(
            "Include semantic-similarity component",
            value=st.session_state["use_semantic"],
            help="Uses a pretrained sentence-transformer (downloaded on first use, ~90 MB). Turning it off "
                 "removes that component and renormalises the others; the setting applies to every page.",
        )
        st.caption(f"Rubric v{load_rubric()['version']} · fingerprint `{load_rubric()['_fingerprint']}`")
    return st.session_state["use_semantic"]


def semantic_arg():
    from src.scoring.scorer import DEFAULT_SEMANTIC
    return DEFAULT_SEMANTIC if semantic_setting() else None


def read_upload(uploaded) -> tuple[bytes, str]:
    return uploaded.getvalue(), uploaded.name


def job_input(prefix: str) -> str:
    """Job description from demo / paste / upload. Returns the text."""
    source = st.radio("Job description source", ["Paste text", "Upload file", "Demo job"], horizontal=True,
                      key=f"{prefix}_jd_source")
    if source == "Demo job":
        text = (DEMO_DATA_DIR / "job_description_1.txt").read_text(encoding="utf-8")
        with st.expander("Show demo job description"):
            st.text(text)
        return text
    if source == "Upload file":
        uploaded = st.file_uploader("Job description file", type=UPLOAD_TYPES, key=f"{prefix}_jd_file")
        if uploaded is None:
            return ""
        data, name = read_upload(uploaded)
        parsed = parse_resume(data, name)  # same text extraction as resumes
        for w in parsed["parse_warnings"]:
            if "section" not in w:
                st.warning(w)
        return parsed["full_text"]
    return st.text_area("Job description", height=220, key=f"{prefix}_jd_text",
                        placeholder="Paste the job description. Sections such as 'Requirements:' / "
                                    "'Preferred:' help the extractor separate must-haves from nice-to-haves.")


def requirement_editor(job_text: str, prefix: str) -> tuple[list, list]:
    """Show extracted requirements for review/correction. Returns
    (requirements, warnings)."""
    digest = hashlib.sha256(job_text.encode("utf-8")).hexdigest()[:10]
    extracted = extract_job_requirements(job_text)
    for w in extracted["warnings"]:
        st.warning(w)
    st.caption("Review the requirements extracted from the job description. Edit text, change Required ↔ "
               "Preferred, tick **Must-have** for essentials, untick **Include** to drop a row, or add rows. "
               "For several acceptable options write e.g. `Python or R`.")
    edited = st.data_editor(
        pd.DataFrame(requirements_to_table(extracted["requirements"]), columns=TABLE_COLUMNS),
        key=f"{prefix}_req_editor_{digest}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "Include": st.column_config.CheckboxColumn(width="small", default=True),
            "Type": st.column_config.SelectboxColumn(options=["Skill", "Experience", "Education", "Certification"],
                                                     required=True, default="Skill", width="small"),
            "Requirement": st.column_config.TextColumn(required=True, width="large"),
            "Importance": st.column_config.SelectboxColumn(options=["Required", "Preferred"], required=True,
                                                           default="Required", width="small"),
            "Must-have": st.column_config.CheckboxColumn(width="small", default=False),
        },
    )
    rows = edited.replace({float("nan"): None}).to_dict("records")
    requirements, warnings = requirements_from_table(rows, extracted["requirements"])
    for w in warnings:
        st.warning(w)
    return requirements, extracted.get("title")


def _pct(x):
    return f"{x * 100:.0f}%"


def _status_chip(status: str) -> str:
    """Icon + text label in a tinted pill; the colour is only a secondary cue."""
    return f"<span class='chip chip-{status}'>{STATUS_ICON[status]} {STATUS_SYMBOLS[status]}</span>"


def _render_requirement(r: dict) -> None:
    res = r["result"]
    tags = [r["importance"]] + (["must-have"] if r["essential"] else [])
    level = f" · {res['evidence_level']}" if res.get("evidence_level") in ("applied", "listed") else ""
    st.markdown(
        f"<div class='req-line'>{_status_chip(res['status'])}<b>{_escape(r['label'])}</b> "
        f"<span class='req-meta'>({', '.join(tags)}){level} · credit {res['credit']:.2f}</span></div>",
        unsafe_allow_html=True,
    )
    for ev in res.get("evidence", [])[:2]:
        st.markdown(f"<div class='evidence'>“{_escape(ev['text'])}” <i>({ev['section']}, {ev['kind']})</i></div>",
                    unsafe_allow_html=True)
    for note in res.get("notes", []):
        st.markdown(f"<div class='evidence note'>ℹ️ {_escape(note)}</div>", unsafe_allow_html=True)


def _escape(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_report(report: dict, key: str, model_estimate: dict | None = None) -> None:
    """Full explainable candidate report (mobile-friendly, vertical)."""
    if report["status"] != "scored":
        st.warning("**Insufficient evidence — no score is shown.** " + " ".join(report["insufficient_reasons"]))
    else:
        c1, c2 = st.columns(2)
        c1.metric("Rubric alignment score", f"{report['score']:.1f} / 100")
        c2.metric("Band", report["band"])
        st.progress(min(1.0, report["score"] / 100))
        if report["cap_reasons"]:
            msg = "**Gating rule:** " + " ".join(report["cap_reasons"])
            if report["cap_applied"]:
                msg += f" The weighted score {report['uncapped_score']:.1f} was capped at " \
                       f"{load_rubric()['essential_requirements']['score_cap_when_unmet']:.0f}."
            st.info(msg)
    st.caption(report["interpretation"])

    st.markdown("#### How this score was calculated")
    applicable = [c for c in report["components"] if c["applicable"]]
    for c in applicable:
        st.markdown(f"<div class='component-line'><b>{c['label']}</b> — value {_pct(c['value'])} × weight "
                    f"{c['effective_weight']:.2f} = <span class='component-points'>{c['contribution']:.1f} points</span>"
                    f"</div>", unsafe_allow_html=True)
        st.progress(min(1.0, c["value"]))
    if applicable and report["status"] == "scored":
        total = sum(c["contribution"] for c in applicable)
        st.caption(f"Sum of contributions = {total:.1f} (rounding may differ by ±0.1)"
                   + (f"; capped to {report['score']:.1f}." if report["cap_applied"] else "."))
    skipped = [c["label"] for c in report["components"] if not c["applicable"]]
    if skipped:
        st.caption("Not applicable for this job (excluded, weights renormalised): " + ", ".join(skipped) + ".")

    s = report["summary"]
    reqs = report["requirements"]
    groups = [
        (f"Required skills — {s['required_skills_met']}/{s['required_skills_total']} detected",
         [r for r in reqs if r["kind"] == "skill" and r["importance"] == "required"]),
        ("Experience (employment + internships)", [r for r in reqs if r["kind"] == "experience"]),
        ("Education", [r for r in reqs if r["kind"] == "education"]),
        ("Certifications", [r for r in reqs if r["kind"] == "certification"]),
        (f"Preferred qualifications — {s['preferred_met']}/{s['preferred_total']} detected",
         [r for r in reqs if r["kind"] == "skill" and r["importance"] == "preferred"]),
    ]
    for title, items in groups:
        if not items:
            continue
        with st.expander(title, expanded=title.startswith("Required")):
            for r in items:
                _render_requirement(r)
            if title.startswith("Experience"):
                entries = items[0]["result"].get("entries") or []
                for e in entries[:6]:
                    counted = "counted" if e["relevant"] else "not counted (no job-relevant terms)"
                    st.markdown(f"<div class='evidence'>{e['kind'].title()} · {e['dates']} · {e['months']} months · "
                                f"{counted}</div>", unsafe_allow_html=True)

    with st.expander("Projects (practical work)"):
        if report["projects"]:
            for p in report["projects"]:
                skills = f" — job skills: {', '.join(p['skills'])}" if p["skills"] else " — no job skills detected"
                st.markdown(f"<div class='req-line'>• {_escape(p['text'])}<small>{skills}</small></div>",
                            unsafe_allow_html=True)
            st.caption("Projects affect the score only by upgrading a skill's evidence from 'listed' to 'applied'; "
                       "they are not counted a second time.")
        else:
            st.write("No projects section detected.")

    with st.expander("Missing or unverified evidence"):
        missing = [r for r in reqs if r["result"]["status"] in ("not_detected", "contradicted", "partial")]
        if not missing:
            st.write("Every listed requirement has supporting evidence.")
        for r in missing:
            st.markdown(f"<div class='req-line'>{_status_chip(r['result']['status'])}{_escape(r['label'])}</div>",
                        unsafe_allow_html=True)
        st.caption("'Not detected' means the submitted resume text does not show it — not that the candidate "
                   "lacks it. 'Contradicted' means the resume itself states a lack.")

    with st.expander("Limitations and extraction warnings"):
        for w in report["warnings"]:
            st.markdown(f"<div class='evidence note'>⚠️ {_escape(w)}</div>", unsafe_allow_html=True)
        removed = report["privacy"]["removed_before_scoring"]
        st.markdown(f"- Removed from scoring input: {', '.join(removed) if removed else 'no contact / personal fields detected'}.")
        st.markdown("- Skill detection uses a curated dictionary plus literal phrases from the job; synonyms "
                    "outside the dictionary are only partly covered by the semantic component.")
        st.markdown("- Negation is detected only when a cue (no / not / without …) shortly precedes the skill.")
        st.markdown(f"- Reproducibility: rubric v{report['config']['version']}, fingerprint "
                    f"`{report['config']['fingerprint']}`, 'Present' = {report['reference_date']}.")

    if model_estimate is not None:
        with st.expander("Experimental: learned-model estimate (not used for the score)"):
            if model_estimate.get("available"):
                st.write(f"Predicted dataset label: **{model_estimate['predicted_label']}**")
                st.dataframe(pd.DataFrame([model_estimate["class_probabilities"]]), hide_index=True,
                             width="stretch")
                st.caption(model_estimate["disclaimer"])
            else:
                st.write(model_estimate.get("reason"))

    st.download_button("Download this report (Markdown)", report_to_markdown(report).encode("utf-8"),
                       file_name=f"report_{(report.get('candidate_id') or 'resume').rsplit('.', 1)[0]}.md",
                       mime="text/markdown", key=f"dl_{key}")


def today() -> date:
    return date.today()
