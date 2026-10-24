import _bootstrap  # noqa: F401  (adds project root to sys.path)
import streamlit as st
from components.ui import DECISION_SUPPORT_NOTE, setup_page

from src.scoring.config import load_rubric

setup_page("Home")

# Welcome screen: shown once per browser session before the homepage. Only this
# script is gated; the sidebar pages work as before, and returning to Home later
# in the session goes straight to the homepage.
WELCOME_ICON = (
    "<svg class='welcome-icon' viewBox='0 0 64 64' aria-hidden='true'>"
    "<rect x='12' y='6' width='32' height='42' rx='5' fill='#FFFFFF' stroke='#6E62A8' stroke-width='2.5'/>"
    "<path d='M19 17h18M19 24h18M19 31h11' stroke='#B9B1DC' stroke-width='2.5' stroke-linecap='round'/>"
    "<circle cx='42' cy='40' r='10' fill='#EAF1F9' stroke='#7C9CC4' stroke-width='2.5'/>"
    "<path d='M49.5 47.5L57 55' stroke='#7C9CC4' stroke-width='3' stroke-linecap='round'/></svg>"
)

if not st.session_state.get("entered_app", False):
    st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
                "[data-testid='stExpandSidebarButton'] "
                "{ display: none; }</style>", unsafe_allow_html=True)
    with st.container(key="welcome_screen"):
        st.markdown(
            f"<div class='welcome'>{WELCOME_ICON}"
            "<p class='welcome-eyebrow'>Intelligent hiring tools</p>"
            "<h1>Resume–Job Matching</h1>"
            "<p class='welcome-tagline'>Every skill matters. Every match has a reason.</p>"
            "<p class='welcome-text'>Understand resume alignment through transparent scoring and "
            "evidence-based insights.</p></div>",
            unsafe_allow_html=True,
        )
        if st.button("Enter the App", type="primary", key="enter_app"):
            st.session_state["entered_app"] = True
            st.rerun()
        st.markdown("<p class='welcome-footer'>Evidence-based insights · Transparent scoring · "
                    "Human-led decisions</p>", unsafe_allow_html=True)
    st.stop()

rubric = load_rubric()

st.markdown(
    "<div class='hero'>"
    "<p class='hero-eyebrow'>Resume intelligence</p>"
    "<h1>Find the right fit. Understand every match.</h1>"
    "<p class='hero-text'>Compare resumes against job requirements with transparent, evidence-based scoring. "
    "Explore candidate strengths, identify missing evidence, and make more informed decisions.</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Two entry points. Columns stack vertically on narrow screens; the keyed
# containers give each card its tint (see .st-key-home_card_* in styles.css).
analyze_col, compare_col = st.columns(2, gap="medium")
with analyze_col:
    with st.container(key="home_card_analyze"):
        st.markdown("<p class='card-title'>Analyze a resume</p>"
                    "<p class='card-text'>Compare one resume with a job description and explore the evidence "
                    "behind the score.</p>", unsafe_allow_html=True)
        st.page_link("pages/1_Single_Resume_Analysis.py", label="Start single analysis", icon=":material/arrow_forward:")
with compare_col:
    with st.container(key="home_card_compare"):
        st.markdown("<p class='card-title'>Compare candidates</p>"
                    "<p class='card-text'>Review multiple candidates against one job using a consistent "
                    "scoring rubric.</p>", unsafe_allow_html=True)
        st.page_link("pages/2_Candidate_Ranking.py", label="Rank candidates", icon=":material/arrow_forward:")

st.caption(DECISION_SUPPORT_NOTE)

st.markdown("### What the score means")
st.markdown(
    "A **rubric alignment score (0–100)** shows how much of the job's stated requirements (required skills, "
    "experience, education, certifications and preferred items) is evidenced in the resume text, plus a small "
    "contextual-similarity component. Every point can be traced to a component and to quoted resume text.\n\n"
    "It is **not** a probability of being hired, **not** a model accuracy, and **not** a recruiter's judgment. "
    "*Not detected* means the resume does not show something, not that the candidate lacks it."
)

st.markdown("### More in this app")
st.page_link("pages/3_Candidate_Details.py", label="Candidate Details: open any ranked candidate's report")
st.page_link("pages/4_Evaluation.py", label="Evaluation: what has been measured, and its limits")
st.page_link("pages/5_Methodology.py", label="Methodology: the exact scoring formula")

st.markdown("### Privacy")
st.markdown(
    "Uploaded files are processed in memory for your session only; they are not saved or logged by the app. "
    "Contact details and personal fields (date of birth, gender, marital status, nationality, religion, caste, "
    "address …) are removed before scoring, and names are never used as features."
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
