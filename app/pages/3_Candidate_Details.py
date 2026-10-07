import _bootstrap  # noqa: F401
import streamlit as st

st.set_page_config(page_title="Candidate Details", page_icon="🧭", layout="wide")
st.title("Candidate Details")

results = st.session_state.get("ranking_results")

if not results:
    st.info("No ranking available yet. Go to **Candidate Ranking**, run a ranking, then come back here.")
else:
    candidate_ids = [r["candidate_id"] for r in results]
    selected_id = st.selectbox("Select a candidate", candidate_ids)
    result = next(r for r in results if r["candidate_id"] == selected_id)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Rank", f"#{result['rank']}")
    col2.metric("Match score", f"{result['overall_score'] * 100:.0f}%")
    col3.metric("Skill coverage", f"{result['skill_score'] * 100:.0f}%")

    skill_col1, skill_col2 = st.columns(2)
    with skill_col1:
        st.markdown("**✅ Matched skills**")
        st.write(", ".join(result["matched_skills"]) or "None detected.")
    with skill_col2:
        st.markdown("**❌ Missing required skills**")
        missing = result["skill_gap"]["missing_required_skills"]
        if missing:
            for item in missing:
                st.write(f"- {item['skill']} — _{item['status']}_")
        else:
            st.write("None — all required skills detected.")

    st.markdown("**Explanation**")
    explanation = result["explanation"]
    if explanation["positive_factors"]:
        for factor in explanation["positive_factors"]:
            st.write(f"- {factor}")
    else:
        st.write("No strong positive factors were detected for this pair.")

    if explanation.get("top_model_features"):
        st.markdown("**Top model features for this prediction (Random Forest importances)**")
        for item in explanation["top_model_features"]:
            st.write(f"- {item['feature']}: {item['importance']:.3f}")

    if result["parse_warnings"]:
        st.markdown("**Parsing notes**")
        for w in result["parse_warnings"]:
            st.warning(w)

    with st.expander("Raw result (debug)"):
        st.json(result)
