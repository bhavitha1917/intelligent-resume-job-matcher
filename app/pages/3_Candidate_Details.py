import _bootstrap  # noqa: F401
import streamlit as st
from components.ui import DECISION_SUPPORT_NOTE, render_report, setup_page

setup_page("Candidate Details")
st.title("Candidate Details")
st.caption(DECISION_SUPPORT_NOTE)

rows = st.session_state.get("ranking_results")
reportable = [r for r in rows or [] if r.get("report")]
if not reportable:
    st.info("No ranking available yet. Run one on **Candidate Ranking**, then come back here.")
    st.stop()


def label(r):
    return f"#{r['rank']} · {r['candidate_id']}" if r["rank"] else f"{r['candidate_id']} (unranked)"


choice = st.selectbox("Candidate", reportable, format_func=label)
st.divider()
render_report(choice["report"], key="details")
