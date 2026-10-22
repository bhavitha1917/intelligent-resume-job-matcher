import _bootstrap  # noqa: F401  (adds project root to sys.path)
import streamlit as st
from components.ui import DECISION_SUPPORT_NOTE, setup_page

from src.scoring.config import load_rubric

setup_page("Home")
rubric = load_rubric()

st.title("Resume–Job Matching")
st.caption(DECISION_SUPPORT_NOTE)

st.markdown(
    """
Compare resumes against a job description and see **why** each one scores the way it does.

- **Single Resume Analysis**: one resume against one job, with a full evidence report.
- **Candidate Ranking**: many resumes against one job, ranked by the same score, with CSV export.
- **Candidate Details**: open any ranked candidate's report.
- **Evaluation**: what has been measured, and what the measurements can and cannot support.
- **Methodology**: the exact scoring formula and its limitations.

### What the score is

A **rubric alignment score (0–100)**: how much of the job's stated requirements (required skills,
experience, education, certifications, preferred items) is evidenced in the resume text, plus a small
contextual-similarity component. Every point can be traced to a component and to quoted resume text.

It is **not** a probability of being hired, **not** a model accuracy, and **not** a recruiter's judgment.
"Not detected" means the resume does not show something, not that the candidate lacks it.

### Privacy

Uploaded files are processed in memory for your session only; they are not saved or logged by the app.
Contact details and personal fields (date of birth, gender, marital status, nationality, religion, caste,
address …) are removed before scoring, and names are never used as features.
"""
)
st.caption(f"Scoring rubric v{rubric['version']} · fingerprint `{rubric['_fingerprint']}`")

with st.expander("Using this on an Android phone"):
    st.markdown(
        """
1. Open the app's URL in **Chrome** on the phone.
2. Tap the **⋮** menu → **Add to Home screen** (on some versions: **Install app** / **Add to Home screen → Add**).
3. A shortcut icon appears on the home screen and opens the app in Chrome.

This is a browser shortcut. It is not a native app and does not work offline, because the analysis runs on the
server.
"""
    )
