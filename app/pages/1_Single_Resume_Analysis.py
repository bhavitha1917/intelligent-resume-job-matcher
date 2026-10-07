import _bootstrap  # noqa: F401
import streamlit as st

from src.data.job_parser import parse_job_description_text
from src.data.resume_parser import parse_resume
from src.features.feature_engineering import build_feature_vector
from src.matching.explainability import generate_explanation
from src.matching.skill_gap import get_skill_gap
from src.models.inference import load_final_model_artifacts, predict_match
from src.utils.paths import DEMO_DATA_DIR

st.set_page_config(page_title="Single Resume Analysis", page_icon="🧭", layout="wide")
st.title("Single Resume Analysis")
st.caption("Upload one resume and enter a job description to get a match score, skill gap, and explanation.")


@st.cache_resource
def _model_status():
    return load_final_model_artifacts() is not None


col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Job description")
    use_demo_jd = st.checkbox("Use bundled demo job description", value=False)
    if use_demo_jd:
        job_text = (DEMO_DATA_DIR / "job_description_1.txt").read_text(encoding="utf-8")
        st.text_area("Job description text", value=job_text, height=260, disabled=True, key="jd_demo")
    else:
        job_text = st.text_area(
            "Paste the job description text",
            height=260,
            placeholder="e.g. Required skills: Python, SQL, Machine Learning...",
            key="jd_input",
        )

with col_right:
    st.subheader("Resume")
    use_demo_resume = st.checkbox("Use a bundled demo resume", value=False)
    resume_source = None
    resume_filename = None
    if use_demo_resume:
        demo_choice = st.selectbox("Demo resume", ["resume_1.txt", "resume_2.txt", "resume_3.txt"])
        resume_source = (DEMO_DATA_DIR / demo_choice).read_text(encoding="utf-8")
    else:
        uploaded = st.file_uploader("Upload a resume (.pdf or .txt)", type=["pdf", "txt"])
        if uploaded is not None:
            resume_source = uploaded.read()
            resume_filename = uploaded.name

analyze = st.button("Analyze match", type="primary", disabled=not (job_text and resume_source))

if not _model_status():
    st.info(
        "Trained model artifacts were not found under `models/`. Run "
        "`python -m src.models.train_supervised` first; until then, scores "
        "use the transparent weighted-hybrid fallback instead of the "
        "trained Random Forest model.",
        icon="ℹ️",
    )

if analyze:
    job_parsed = parse_job_description_text(job_text)
    resume_parsed = parse_resume(resume_source, resume_filename)

    if resume_parsed["parse_warnings"]:
        for w in resume_parsed["parse_warnings"]:
            st.warning(w)

    if not job_parsed["required_skills"] and not job_parsed["description"].strip():
        st.error("Could not analyze: job description is empty.")
    elif not resume_parsed["full_text"].strip():
        st.error("Could not analyze: no text could be extracted from the resume.")
    else:
        with st.spinner("Computing similarity and generating the match score..."):
            feature_result = build_feature_vector(
                resume_full_text=resume_parsed["full_text"],
                job_description_text=job_parsed["description"],
                required_skills=job_parsed["required_skills"],
                preferred_skills=job_parsed["preferred_skills"],
                resume_experience_text=resume_parsed["experience"],
                job_experience_requirement=job_parsed["experience_requirement"],
                resume_education_text=resume_parsed["education"],
                job_education_requirement=job_parsed["education_requirement"],
                compute_semantic=True,
            )
            features = feature_result["features"]
            prediction = predict_match(features)
            skill_gap = get_skill_gap(
                resume_parsed["full_text"], job_parsed["required_skills"], job_parsed["preferred_skills"]
            )
            explanation = generate_explanation(
                features,
                match_score=prediction["score_percent"] / 100,
                missing_required_skills=[item["skill"] for item in skill_gap["missing_required_skills"]],
            )

        st.divider()
        score_col, label_col, semantic_col = st.columns(3)
        score_col.metric("Match score", f"{prediction['score_percent']:.0f}%")
        label_col.metric("Predicted category", (prediction["predicted_label"] or "n/a").title())
        semantic_col.metric("Semantic similarity", f"{features['semantic_similarity']:.2f}")

        st.progress(min(1.0, prediction["score_percent"] / 100))

        skill_col1, skill_col2 = st.columns(2)
        with skill_col1:
            st.markdown("**✅ Matched skills**")
            if feature_result["matched_skills"]:
                st.write(", ".join(feature_result["matched_skills"]))
            else:
                st.write("None detected.")
        with skill_col2:
            st.markdown("**❌ Missing required skills**")
            if skill_gap["missing_required_skills"]:
                for item in skill_gap["missing_required_skills"]:
                    st.write(f"- {item['skill']} — _{item['status']}_")
            else:
                st.write("None — all required skills detected.")

        st.markdown("**Explanation**")
        if explanation["positive_factors"]:
            for factor in explanation["positive_factors"]:
                st.write(f"- {factor}")
        else:
            st.write("No strong positive factors were detected for this pair.")
        if explanation.get("note"):
            st.caption(explanation["note"])

        with st.expander("Full feature breakdown"):
            st.json(features)
