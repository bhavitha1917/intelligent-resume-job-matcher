import _bootstrap  # noqa: F401
import pandas as pd
import streamlit as st

from src.ranking.rank_candidates import rank_candidates
from src.utils.paths import DEMO_DATA_DIR

st.set_page_config(page_title="Candidate Ranking", page_icon="🧭", layout="wide")
st.title("Candidate Ranking")
st.caption("Rank multiple resumes against one job description.")

st.subheader("Job description")
use_demo_jd = st.checkbox("Use bundled demo job description", value=True)
if use_demo_jd:
    job_text = (DEMO_DATA_DIR / "job_description_1.txt").read_text(encoding="utf-8")
    st.text_area("Job description text", value=job_text, height=200, disabled=True, key="rank_jd_demo")
else:
    job_text = st.text_area("Paste the job description text", height=200, key="rank_jd_input")

st.subheader("Resumes")
use_demo_resumes = st.checkbox("Use the 3 bundled demo resumes", value=True)

resumes = []
if use_demo_resumes:
    for fname in ["resume_1.txt", "resume_2.txt", "resume_3.txt"]:
        resumes.append({"candidate_id": fname, "source": DEMO_DATA_DIR / fname})
else:
    uploaded_files = st.file_uploader(
        "Upload resumes (.pdf or .txt) — select multiple files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    for f in uploaded_files or []:
        resumes.append({"candidate_id": f.name, "source": f.read(), "filename": f.name})

run = st.button("Rank candidates", type="primary", disabled=not (job_text and resumes))

if run:
    with st.spinner(f"Scoring {len(resumes)} resume(s)..."):
        results = rank_candidates(job_text, resumes, compute_semantic=True)
    st.session_state["ranking_results"] = results
    st.session_state["ranking_job_text"] = job_text

results = st.session_state.get("ranking_results")
if results:
    st.divider()
    table_rows = [
        {
            "Rank": r["rank"],
            "Candidate": r["candidate_id"],
            "Match Score": f"{r['overall_score'] * 100:.0f}%",
            "Skill Match": f"{r['skill_score'] * 100:.0f}%",
            "Semantic Match": f"{r['semantic_score']:.2f}" if r["semantic_score"] is not None else "n/a",
        }
        for r in results
    ]
    df = pd.DataFrame(table_rows)

    min_score = st.slider("Filter: minimum match score (%)", 0, 100, 0)
    filtered = df[df["Match Score"].str.rstrip("%").astype(float) >= min_score]
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.caption(
        "Open the **Candidate Details** page and pick a candidate from the "
        "dropdown there to see the full skill-gap and explanation breakdown."
    )

    for r in results:
        if r["parse_warnings"]:
            st.warning(f"{r['candidate_id']}: " + "; ".join(r["parse_warnings"]))
else:
    st.info("Run a ranking to see results here.")
