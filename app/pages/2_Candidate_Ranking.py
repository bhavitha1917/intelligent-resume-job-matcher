import _bootstrap  # noqa: F401
import streamlit as st
from components.ui import (
    DECISION_SUPPORT_NOTE,
    UPLOAD_TYPES,
    job_input,
    render_report,
    requirement_editor,
    semantic_arg,
    setup_page,
    today,
)

from src.ranking.rank_candidates import rank_candidates
from src.scoring.config import load_rubric
from src.scoring.export import results_to_csv, results_to_dataframe
from src.utils.paths import DEMO_DATA_DIR

setup_page("Candidate Ranking", layout="wide")
st.title("Candidate Ranking")
st.caption(DECISION_SUPPORT_NOTE)
semantic = semantic_arg()

st.subheader("1 · Job description")
job_text = job_input("rank")
requirements = []
if job_text.strip():
    st.subheader("2 · Review the extracted requirements")
    requirements, _ = requirement_editor(job_text, "rank")

st.subheader("3 · Resumes")
use_demo = st.toggle("Use the 3 bundled demo resumes", value=False, key="rank_use_demo")
resumes = []
if use_demo:
    resumes = [{"candidate_id": f, "source": (DEMO_DATA_DIR / f).read_bytes(), "filename": f}
               for f in ["resume_1.txt", "resume_2.txt", "resume_3.txt"]]
else:
    files = st.file_uploader("Resumes (.pdf, .docx, .txt) — select several", type=UPLOAD_TYPES,
                             accept_multiple_files=True, key="rank_files")
    names_seen = {}
    for f in files or []:
        name = f.name
        if name in names_seen:  # same file name uploaded twice: keep both, disambiguate
            names_seen[name] += 1
            name = f"{name} ({names_seen[f.name]})"
        else:
            names_seen[name] = 1
        resumes.append({"candidate_id": name, "source": f.getvalue(), "filename": f.name})
st.caption(f"{len(resumes)} file(s) selected. Files are processed in memory and are not stored.")

if st.button("Rank candidates", type="primary", disabled=not (job_text.strip() and resumes)):
    bar = st.progress(0.0, text="Starting...")

    def on_progress(i, n, name):
        bar.progress(i / n if n else 1.0, text=f"Scoring {i + 1}/{n}: {name}" if name else "Done")

    rows = rank_candidates(job_text, resumes, requirements=requirements, semantic=semantic,
                           reference_date=today(), progress=on_progress)
    bar.empty()
    st.session_state["ranking_results"] = rows
    st.session_state["ranking_requirements"] = requirements

rows = st.session_state.get("ranking_results")
if not rows:
    st.info("Run a ranking to see results here.")
    st.stop()

st.divider()
counts = {s: sum(r["status"] == s for r in rows) for s in ("scored", "insufficient_evidence", "parse_failed", "duplicate")}
st.markdown(f"**{counts['scored']} ranked** · {counts['insufficient_evidence']} insufficient evidence · "
            f"{counts['parse_failed']} unreadable · {counts['duplicate']} duplicate")
for r in rows:
    if r["status"] == "parse_failed":
        st.error(f"{r['candidate_id']}: could not be read — " + " ".join(r["parse_warnings"]))
    elif r["status"] == "duplicate":
        st.warning(f"{r['candidate_id']}: " + " ".join(r["parse_warnings"]))

with st.expander("Filter and sort", expanded=False):
    min_score = st.slider("Minimum score", 0, 100, 0)
    bands = sorted({r["band"] for r in rows if r["status"] == "scored"})
    band_filter = st.multiselect("Band", bands, default=bands)
    skill_labels = [q["label"] for q in st.session_state.get("ranking_requirements", []) if q["kind"] == "skill"]
    must_skills = st.multiselect("Only candidates with these skills detected", skill_labels)
    sort_by = st.selectbox("Sort by", ["Rank (score)", "Required skills detected", "Experience detected", "Name"])
    show_unscored = st.checkbox("Show unscored files (insufficient / unreadable / duplicate)", value=True)


def keep(r):
    if r["status"] != "scored":
        return show_unscored
    if r["score"] < min_score or r["band"] not in band_filter:
        return False
    met = {q["label"] for q in r["report"]["requirements"] if q["result"]["status"] == "met"}
    return all(s in met for s in must_skills)


visible = [r for r in rows if keep(r)]
sort_keys = {
    "Required skills detected": lambda r: -(r["report"]["summary"]["required_skills_met"] if r["report"] else -1),
    "Experience detected": lambda r: -((r["report"]["summary"]["experience_years_detected"] or 0) if r["report"] else -1),
    "Name": lambda r: str(r["candidate_id"]).lower(),
}
if sort_by in sort_keys:
    visible = sorted(visible, key=sort_keys[sort_by])

table = results_to_dataframe(visible)[["rank", "candidate", "score", "band", "required_skills_detected",
                                       "required_skills_total", "experience_years_detected", "highest_degree_detected", "status"]]
st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={
        "rank": st.column_config.NumberColumn("Rank", width="small"),
        "candidate": st.column_config.TextColumn("Candidate"),
        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "band": st.column_config.TextColumn("Band"),
        "required_skills_detected": st.column_config.NumberColumn("Req. skills ✓", width="small"),
        "required_skills_total": st.column_config.NumberColumn("of", width="small"),
        "experience_years_detected": st.column_config.NumberColumn("Exp. yrs", format="%.1f", width="small"),
        "highest_degree_detected": st.column_config.TextColumn("Degree"),
        "status": st.column_config.TextColumn("Status"),
    },
)
if any(r.get("tied") for r in visible):
    st.caption("Tied ranks: " + load_rubric()["ranking"]["tie_rule"])

st.download_button("Download results (CSV)", results_to_csv(rows), file_name="candidate_ranking.csv",
                   mime="text/csv")

st.subheader("Candidate reports")
st.caption("Each score below is exactly the score shown in the table, computed with the same rubric.")
for i, r in enumerate(visible):
    if r["status"] not in ("scored", "insufficient_evidence"):
        continue
    label = (f"#{r['rank']} · {r['candidate_id']} — {r['score']:.1f} · {r['band']}" if r["status"] == "scored"
             else f"{r['candidate_id']} — insufficient evidence")
    with st.expander(label):
        render_report(r["report"], key=f"rank_{i}")
