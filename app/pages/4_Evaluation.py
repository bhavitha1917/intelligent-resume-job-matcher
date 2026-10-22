import json

import _bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from components.ui import setup_page

from src.models.inference import legacy_artifact_status
from src.utils.paths import FIGURES_DIR, RESULTS_DIR, RESULTS_V2_DIR

setup_page("Evaluation", layout="wide")
st.title("Evaluation & Model Comparison")
st.warning(
    "All metrics on this page measure agreement with the Hugging Face dataset's **algorithmically generated** "
    "`ai_match_score` (bucketed into weak / moderate / strong). They are **not** measurements of real hiring "
    "suitability, recruiter agreement, or the accuracy of the score shown on the analysis pages."
)

path = RESULTS_V2_DIR / "evaluation.json"
if not path.exists():
    st.info("No v2 evaluation found. Run `python -m src.models.train_supervised`.")
    st.stop()
ev = json.loads(path.read_text(encoding="utf-8"))
ds = ev["dataset"]

st.subheader("Dataset facts that shape the evaluation")
st.markdown(
    f"- {ds['rows_after_exact_resume_dedupe']} resume–job pairs after removing exact duplicates, but only "
    f"**{ds['unique_jobs']} distinct job descriptions** (one per category, {ds['rows_per_job']['min']}–"
    f"{ds['rows_per_job']['max']} resumes each).\n"
    f"- The mean dataset score differs strongly by job (e.g. "
    + ", ".join(f"{k} {v}" for k, v in list(sorted(ds["mean_ai_match_score_by_category"].items(), key=lambda kv: kv[1]))[:2])
    + " … " + ", ".join(f"{k} {v}" for k, v in list(sorted(ds["mean_ai_match_score_by_category"].items(), key=lambda kv: kv[1]))[-2:])
    + "), so knowing *which job* a row belongs to already predicts much of its label.\n"
    f"- Rows with required-skill coverage of 0: v1 features **{ds['skill_coverage_zero_fraction']['v1_training_features']:.1%}** → "
    f"v2 shared matcher **{ds['skill_coverage_zero_fraction']['v2_shared_matcher']:.1%}**.\n"
    f"- Near-duplicate resume pairs (TF-IDF cosine ≥ {ds['near_duplicates']['threshold_tfidf_cosine']}): "
    f"{ds['near_duplicates']['near_duplicate_resume_pairs']} "
    f"({ds['near_duplicates']['pairs_across_different_jobs']} across different jobs)."
)


def method_table(methods: dict) -> pd.DataFrame:
    rows = []
    for name, m in methods.items():
        rows.append({"method": name, "macro F1": m["f1_macro"], "accuracy": m["accuracy"],
                     "macro precision": m["precision_macro"], "macro recall": m["recall_macro"],
                     "ROC-AUC (OvR)": m.get("roc_auc_macro_ovr"), "n": m["n"]})
    return pd.DataFrame(rows).sort_values("macro F1", ascending=False)


a = ev["protocols"]["A_row_level_split"]
b = ev["protocols"]["B_leave_one_job_out"]
tab_b, tab_a, tab_rank = st.tabs(["Unseen jobs (leave-one-job-out)", "Row-level split (v1 protocol)", "Within-job ranking"])
with tab_b:
    st.caption(f"{b['folds']} folds; each fold holds out one complete job description. TF-IDF and scalers are fit "
               "on the 22 training jobs only; hyperparameters are fixed in advance (no tuning on test data).")
    st.dataframe(method_table(b["methods"]), hide_index=True, width="stretch")
    fig = FIGURES_DIR / "v2" / "confusion_matrix_rf_v2_lojo.png"
    if fig.exists():
        st.image(str(fig), width=360)
    pj = b["per_job_accuracy"].get("random_forest__v2_features")
    if pj:
        st.caption(f"Random Forest (v2) per-job accuracy ranges from {pj['min']:.2f} to {pj['max']:.2f} "
                   f"(median {pj['median']:.2f}) — with 23 jobs, pooled numbers hide large per-job variation.")
with tab_a:
    st.caption(f"Train {a['train_rows']} / test {a['test_rows']} rows; **{a['jobs_in_both_train_and_test']} of 23 jobs "
               "appear in both train and test**, so these numbers include job memorisation. "
               "`job_identity_only` predicts each test row's label from its job's most common training label.")
    st.dataframe(method_table(a["methods"]), hide_index=True, width="stretch")
    fig = FIGURES_DIR / "v2" / "protocol_comparison_rf.png"
    if fig.exists():
        st.image(str(fig))
with tab_rank:
    st.caption(ev["within_job_ranking"]["design"])
    st.dataframe(pd.DataFrame(ev["within_job_ranking"]["methods"]).T, width="stretch")

st.subheader("Rubric score vs. dataset score (no fitting)")
ra = ev["rubric_agreement_with_dataset_score"]
st.markdown(f"Spearman correlation pooled **{ra['spearman_pooled']}**, mean within-job **{ra['spearman_mean_within_job']}** "
            f"over {ra['rows_scored']} rows ({ra['rows_insufficient_evidence']} rows had insufficient evidence). "
            + ra["note"])

st.subheader("How the learned model is used in the app")
st.markdown(
    "The rubric score is the only score used for display and ranking. The v2 Random Forest's predicted dataset "
    "label is shown in a collapsed, clearly-labelled *experimental* panel. The v1 display formula "
    "`P(strong)×100 + P(moderate)×50` has been removed."
)
legacy = legacy_artifact_status()
with st.expander("Legacy v1 artifacts and results"):
    st.write(f"v1 model files kept for reproducibility: {', '.join(legacy['present']) or 'none'} — not loaded. {legacy['reason']}")
    old = RESULTS_DIR / "model_comparison.json"
    if old.exists():
        st.caption("Original v1 results (row-level split, features with the defects listed above):")
        st.json(json.loads(old.read_text(encoding="utf-8")), expanded=False)
