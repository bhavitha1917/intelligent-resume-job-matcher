import json

import _bootstrap  # noqa: F401
import pandas as pd
import streamlit as st

from src.utils.paths import FIGURES_DIR, RESULTS_DIR

st.set_page_config(page_title="Model Comparison", page_icon="🧭", layout="wide")
st.title("Model Comparison")
st.caption("Experimental evidence behind the choice of final model (spec.md section 21: no model is claimed best without evidence).")

comparison_path = RESULTS_DIR / "model_comparison.json"
ranking_path = RESULTS_DIR / "ranking_evaluation.json"

if not comparison_path.exists():
    st.warning(
        "No results found. Run `python -m src.models.train_supervised` to "
        "generate reports/results/model_comparison.json first."
    )
else:
    with open(comparison_path, encoding="utf-8") as f:
        comparison = json.load(f)

    st.subheader("Classification comparison (test set)")
    st.caption(
        "Target: a 3-class label (weak/moderate/strong) derived from the "
        "dataset-provided ai_match_score using documented thresholds "
        "(see src/models/config.py) — not a human recruiter decision."
    )
    rows = []
    for name, metrics in comparison.items():
        row = {"Method": name}
        row.update({k: v for k, v in metrics.items() if k != "note"})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown(
        "**Finding:** the keyword and TF-IDF baselines perform close to "
        "random on this 3-class task, semantic similarity alone improves "
        "substantially, and the supervised models (especially Random "
        "Forest) perform best overall — directly supporting the project's "
        "research question. Random Forest was selected as the final model "
        "based on this validation/test evidence, not by assumption."
    )

    f1_chart = FIGURES_DIR / "model_comparison_f1.png"
    if f1_chart.exists():
        st.image(str(f1_chart), caption="Macro F1-score by method (test set)")

    col1, col2 = st.columns(2)
    for col, name in zip(
        [col1, col2],
        ["confusion_matrix_5_random_forest.png", "feature_importance_random_forest.png"],
    ):
        path = FIGURES_DIR / name
        if path.exists():
            col.image(str(path))

    if ranking_path.exists():
        st.subheader("Ranking evaluation (synthetic multi-candidate queries, test set)")
        st.caption(
            "Each test job is paired with its true matched resume plus 9 "
            "randomly sampled distractor resumes; candidates are ranked by "
            "each method's score and compared against the known match label."
        )
        with open(ranking_path, encoding="utf-8") as f:
            ranking = json.load(f)
        st.dataframe(pd.DataFrame(ranking).T, use_container_width=True)

    st.subheader("Secondary experiment: regression on the continuous score")
    reg = comparison.get("regression_experiment_random_forest")
    if reg:
        st.write(f"RMSE: {reg['rmse']} · MAE: {reg['mae']} · R²: {reg['r2']}")
        st.caption(reg["note"])
