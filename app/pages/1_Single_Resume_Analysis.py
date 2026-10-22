import _bootstrap  # noqa: F401
import hashlib
import json

import streamlit as st
from components.ui import (
    DECISION_SUPPORT_NOTE,
    UPLOAD_TYPES,
    job_input,
    read_upload,
    render_report,
    requirement_editor,
    semantic_arg,
    setup_page,
    today,
)

from src.data.resume_parser import parse_resume
from src.models.inference import model_estimate_for_report
from src.scoring.privacy import sanitize_for_scoring
from src.scoring.scorer import score_candidate
from src.utils.paths import DEMO_DATA_DIR

setup_page("Single Resume Analysis")
st.title("Single Resume Analysis")
st.caption(DECISION_SUPPORT_NOTE)
semantic = semantic_arg()

st.subheader("1 · Job description")
job_text = job_input("single")

requirements, title = [], None
if job_text.strip():
    st.subheader("2 · Review the extracted requirements")
    requirements, title = requirement_editor(job_text, "single")

st.subheader("3 · Resume")
resume_mode = st.radio("Resume source", ["Upload file", "Demo resume"], horizontal=True, key="single_resume_mode")
resume_source, resume_name = None, None
if resume_mode == "Demo resume":
    resume_name = st.selectbox("Demo resume", ["resume_1.txt", "resume_2.txt", "resume_3.txt"])
    resume_source = (DEMO_DATA_DIR / resume_name).read_bytes()
else:
    uploaded = st.file_uploader("Resume (.pdf, .docx or .txt)", type=UPLOAD_TYPES, key="single_resume_file")
    if uploaded is not None:
        resume_source, resume_name = read_upload(uploaded)

inputs_key = hashlib.sha256(
    (job_text + json.dumps(requirements, sort_keys=True, default=str) + str(resume_name)
     + str(hashlib.sha256(resume_source).hexdigest() if resume_source else "") + str(semantic)).encode("utf-8")
).hexdigest()

if st.button("Analyze", type="primary", disabled=not (job_text.strip() and resume_source)):
    parsed = parse_resume(resume_source, resume_name)
    if not parsed["full_text"].strip():
        st.error("No text could be extracted from this resume. " + " ".join(parsed["parse_warnings"]))
        st.session_state.pop("single_result", None)
    else:
        with st.spinner("Scoring (the first run may take ~30 s while the semantic model loads)..."):
            job = {"title": title, "text": job_text, "requirements": requirements}
            report = score_candidate(job, parsed, semantic=semantic, reference_date=today(), candidate_id=resume_name)
            estimate = None
            if semantic is not None:
                sanitized, _ = sanitize_for_scoring(parsed["full_text"], parsed["section_spans"])
                estimate = model_estimate_for_report(report, sanitized, job_text)
        st.session_state["single_result"] = {"key": inputs_key, "report": report, "estimate": estimate}

result = st.session_state.get("single_result")
if result:
    st.divider()
    if result["key"] != inputs_key:
        st.warning("The inputs have changed since this analysis was run. Press **Analyze** to update it.")
    st.subheader(f"Report: {result['report'].get('candidate_id') or 'resume'}")
    render_report(result["report"], key="single", model_estimate=result["estimate"])
