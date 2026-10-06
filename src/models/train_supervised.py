"""Phase 12/13/14/19/20 -- supervised model training, baseline comparison,
and ranking evaluation, all on the HF `batuhanmtl/job_resume_fit` dataset.

Target strategy (documented, configurable -- see src/models/config.py):
ai_match_score (0-100, dataset-provided) is bucketed into
strong/moderate/weak. This is NOT a human recruiter decision (spec.md
4.3) -- it is the dataset's own matching signal, used here only to
supervise a classifier that reproduces/improves on it using resume+job
TEXT features (no PII).

Run with: .venv/Scripts/python.exe -m src.models.train_supervised

Leakage safeguards (spec.md section 28):
- exact-duplicate resumes are dropped before splitting
- TF-IDF vectorizer is fit ONLY on the train split
- the StandardScaler is fit ONLY on the train split
- sentence-transformer embeddings come from a FROZEN pretrained model
  (not fit on this dataset), so computing them for all splits up front
  does not leak target information
"""

import ast
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from src.features.feature_engineering import education_compatibility, experience_compatibility
from src.matching.hybrid import compute_weighted_hybrid_score
from src.matching.keyword_baseline import keyword_match
from src.matching.semantic_similarity import batch_encode
from src.matching.tfidf_similarity import fit_tfidf_vectorizer, save_vectorizer
from src.models.config import (
    MODERATE_MATCH_THRESHOLD,
    RANDOM_STATE,
    STRONG_MATCH_THRESHOLD,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    score_to_label,
)
from src.preprocessing.text_cleaning import clean_text, handle_missing
from src.skills.skill_extractor import extract_skills
from src.evaluation.ranking_metrics import evaluate_ranking_queries
from src.utils.paths import FIGURES_DIR, MODELS_DIR, RESULTS_DIR

HF_DATASET_NAME = "batuhanmtl/job_resume_fit"
FEATURE_COLUMNS = [
    "skill_coverage",
    "num_matched_skills",
    "num_missing_skills",
    "tfidf_similarity",
    "semantic_similarity",
    "resume_length",
    "job_description_length",
    "experience_compatibility",
    "education_compatibility",
]
CLASS_LABELS = ["weak", "moderate", "strong"]
NUM_RANKING_NEGATIVES = 9
RANKING_K = 5


def _parse_skill_list(value) -> list:
    if isinstance(value, list):
        return value
    if not value or not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def load_and_prepare_dataset() -> pd.DataFrame:
    print(f"Loading {HF_DATASET_NAME} ...")
    df = load_dataset(HF_DATASET_NAME)["train"].to_pandas()

    df["resume_text_clean"] = df["resume_text"].apply(lambda t: clean_text(handle_missing(t)))
    df["job_text_clean"] = df["job_text"].apply(lambda t: clean_text(handle_missing(t)))
    df["required_skills_list"] = df["job_required_skills"].apply(_parse_skill_list)

    before = len(df)
    df = df.drop_duplicates(subset="resume_text", keep="first").reset_index(drop=True)
    print(f"Dropped {before - len(df)} exact-duplicate-resume rows (leakage prevention).")

    df["match_label"] = df["ai_match_score"].apply(score_to_label)
    df["match_score_norm"] = df["ai_match_score"] / 100.0

    print("Extracting resume skills with our dictionary-based extractor...")
    df["resume_skills_extracted"] = df["resume_text_clean"].apply(extract_skills)

    return df


def split_dataset(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["match_label"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["match_label"], random_state=RANDOM_STATE
    )
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return train_df, val_df, test_df


def build_pair_features(resume_pos: int, job_pos: int, data: dict) -> dict:
    """Feature vector for an (resume at resume_pos, job at job_pos) pair.
    resume_pos == job_pos for the dataset's true pairs; ranking evaluation
    also calls this with resume_pos != job_pos for sampled negatives.
    """
    resume_skills = data["resume_skills"][resume_pos]
    required_skills = data["required_skills"][job_pos]
    km = keyword_match(resume_skills, required_skills)

    tfidf_sim = float(
        (data["resume_tfidf"][resume_pos] @ data["job_tfidf"][job_pos].T).toarray()[0, 0]
    )
    semantic_sim = float(np.dot(data["resume_emb"][resume_pos], data["job_emb"][job_pos]))

    resume_text = data["resume_text"][resume_pos]
    job_text = data["job_text"][job_pos]

    return {
        "skill_coverage": km["skill_coverage"],
        "num_matched_skills": len(km["matched_skills"]),
        "num_missing_skills": len(km["missing_skills"]),
        "tfidf_similarity": round(tfidf_sim, 4),
        "semantic_similarity": round(semantic_sim, 4),
        "resume_length": len(resume_text.split()),
        "job_description_length": len(job_text.split()),
        "experience_compatibility": experience_compatibility(resume_text, job_text),
        "education_compatibility": education_compatibility(resume_text, job_text),
    }


def precompute_pair_data(df: pd.DataFrame, vectorizer) -> dict:
    print("Computing TF-IDF vectors for all rows (transform only, vectorizer already fit on train)...")
    resume_tfidf = vectorizer.transform(df["resume_text_clean"].tolist())
    job_tfidf = vectorizer.transform(df["job_text_clean"].tolist())

    print("Computing sentence-transformer embeddings for all rows (frozen pretrained model)...")
    resume_emb = batch_encode(df["resume_text_clean"].tolist())
    job_emb = batch_encode(df["job_text_clean"].tolist())

    return {
        "resume_skills": df["resume_skills_extracted"].tolist(),
        "required_skills": df["required_skills_list"].tolist(),
        "resume_tfidf": resume_tfidf,
        "job_tfidf": job_tfidf,
        "resume_emb": resume_emb,
        "job_emb": job_emb,
        "resume_text": df["resume_text_clean"].tolist(),
        "job_text": df["job_text_clean"].tolist(),
    }


def build_feature_matrix(positions: list, data: dict) -> pd.DataFrame:
    rows = [build_pair_features(p, p, data) for p in positions]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def continuous_score_to_label(score: float) -> str:
    """Apply the SAME relative thresholds (on a 0-1 scale) to baseline
    similarity scores so keyword/TF-IDF/semantic baselines can be compared
    in the same classification table as the supervised models.
    """
    if score >= STRONG_MATCH_THRESHOLD / 100:
        return "strong"
    if score >= MODERATE_MATCH_THRESHOLD / 100:
        return "moderate"
    return "weak"


def classification_metrics(y_true, y_pred, y_score_matrix=None) -> dict:
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }
    if y_score_matrix is not None:
        try:
            y_true_bin = label_binarize(y_true, classes=CLASS_LABELS)
            metrics["roc_auc_macro_ovr"] = round(
                roc_auc_score(y_true_bin, y_score_matrix, average="macro", multi_class="ovr"), 4
            )
        except ValueError:
            metrics["roc_auc_macro_ovr"] = None
    return metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_dataset()
    train_df, val_df, test_df = split_dataset(df)

    # --- TF-IDF: fit ONLY on train ---
    train_corpus = train_df["resume_text_clean"].tolist() + train_df["job_text_clean"].tolist()
    vectorizer = fit_tfidf_vectorizer(train_corpus, max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE)
    save_vectorizer(vectorizer)
    print(f"Saved TF-IDF vectorizer -> {MODELS_DIR / 'tfidf_vectorizer.joblib'}")

    # --- precompute pair-level data for the whole dataset (transform/embedding only, no fitting) ---
    data = precompute_pair_data(df, vectorizer)
    position_of_index = {idx: pos for pos, idx in enumerate(df.index)}

    train_pos = [position_of_index[i] for i in train_df.index]
    val_pos = [position_of_index[i] for i in val_df.index]
    test_pos = [position_of_index[i] for i in test_df.index]

    X_train = build_feature_matrix(train_pos, data)
    X_val = build_feature_matrix(val_pos, data)
    X_test = build_feature_matrix(test_pos, data)

    y_train = train_df["match_label"].to_numpy()
    y_val = val_df["match_label"].to_numpy()
    y_test = test_df["match_label"].to_numpy()
    y_train_reg = train_df["match_score_norm"].to_numpy()
    y_test_reg = test_df["match_score_norm"].to_numpy()

    # --- scale features (fit ONLY on train) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / "feature_scaler.joblib")

    comparison = {}

    # ---------- Baseline 1: keyword / skill coverage ----------
    baseline_preds = X_test["skill_coverage"].apply(continuous_score_to_label).to_numpy()
    comparison["1_keyword_baseline"] = classification_metrics(y_test, baseline_preds)

    # ---------- Baseline 2: TF-IDF cosine similarity ----------
    tfidf_preds = X_test["tfidf_similarity"].apply(continuous_score_to_label).to_numpy()
    comparison["2_tfidf_baseline"] = classification_metrics(y_test, tfidf_preds)

    # ---------- Baseline 3: sentence-transformer semantic similarity ----------
    semantic_preds = X_test["semantic_similarity"].apply(continuous_score_to_label).to_numpy()
    comparison["3_semantic_baseline"] = classification_metrics(y_test, semantic_preds)

    # ---------- Supervised models ----------
    models = {
        "4_logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "5_random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "6_svm": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
    }

    trained_models = {}
    inference_times = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)
        per_pair_time = (time.perf_counter() - t0) / len(X_test_scaled)

        class_order = list(model.classes_)
        probs_ordered = probs[:, [class_order.index(c) for c in CLASS_LABELS]]
        metrics = classification_metrics(y_test, preds, probs_ordered)
        metrics["train_time_seconds"] = round(fit_time, 4)
        metrics["inference_time_seconds_per_pair"] = round(per_pair_time, 6)
        comparison[name] = metrics

        trained_models[name] = model
        inference_times[name] = per_pair_time

        model_filename = name.split("_", 1)[1] + ".joblib"
        joblib.dump(model, MODELS_DIR / model_filename)

        cm = confusion_matrix(y_test, preds, labels=CLASS_LABELS)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{name}.png", dpi=150)
        plt.close()

    # ---------- Final hybrid (manually-weighted) ----------
    hybrid_scores = X_test.apply(lambda row: compute_weighted_hybrid_score(row.to_dict()), axis=1)
    hybrid_preds = hybrid_scores.apply(continuous_score_to_label).to_numpy()
    comparison["7_final_hybrid_weighted"] = classification_metrics(y_test, hybrid_preds)

    # ---------- Secondary experiment: regression on continuous score ----------
    regressor = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    regressor.fit(X_train_scaled, y_train_reg)
    reg_preds = regressor.predict(X_test_scaled)
    rmse = float(np.sqrt(np.mean((reg_preds - y_test_reg) ** 2)))
    mae = float(np.mean(np.abs(reg_preds - y_test_reg)))
    ss_res = float(np.sum((y_test_reg - reg_preds) ** 2))
    ss_tot = float(np.sum((y_test_reg - np.mean(y_test_reg)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    comparison["regression_experiment_random_forest"] = {
        "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4),
        "note": "Secondary experiment predicting the continuous ai_match_score (spec.md section 18).",
    }
    joblib.dump(regressor, MODELS_DIR / "random_forest_regressor.joblib")

    # ---------- feature importance (Random Forest classifier) ----------
    rf_model = trained_models["5_random_forest"]
    importances = dict(zip(FEATURE_COLUMNS, rf_model.feature_importances_.tolist()))
    plt.figure(figsize=(7, 5))
    sorted_items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    names, vals = zip(*sorted_items)
    sns.barplot(x=list(vals), y=list(names), color="#4C72B0")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance_random_forest.png", dpi=150)
    plt.close()

    # ---------- comparison bar chart (F1 macro across all methods) ----------
    f1_by_method = {k: v["f1_macro"] for k, v in comparison.items() if "f1_macro" in v}
    plt.figure(figsize=(9, 5))
    sorted_f1 = dict(sorted(f1_by_method.items()))
    sns.barplot(x=list(sorted_f1.values()), y=list(sorted_f1.keys()), color="#55A868")
    plt.title("Macro F1-score by Method (test set)")
    plt.xlabel("F1 (macro)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_f1.png", dpi=150)
    plt.close()

    # ---------- ranking evaluation (Precision@K, Recall@K, MRR, NDCG@K) ----------
    print("Running ranking evaluation (synthetic multi-candidate queries from the test split)...")
    rng = np.random.default_rng(RANDOM_STATE)
    n_test = len(test_pos)

    def score_pair_hybrid(resume_pos, job_pos):
        feats = build_pair_features(resume_pos, job_pos, data)
        return compute_weighted_hybrid_score(feats)

    def score_pair_rf(resume_pos, job_pos):
        feats = build_pair_features(resume_pos, job_pos, data)
        row = pd.DataFrame([feats], columns=FEATURE_COLUMNS)
        row_scaled = scaler.transform(row)
        probs = rf_model.predict_proba(row_scaled)[0]
        class_order = list(rf_model.classes_)
        # score = P(strong) + 0.5 * P(moderate), a single ranking scalar
        strong_p = probs[class_order.index("strong")] if "strong" in class_order else 0.0
        moderate_p = probs[class_order.index("moderate")] if "moderate" in class_order else 0.0
        return strong_p + 0.5 * moderate_p

    def score_pair_keyword(resume_pos, job_pos):
        return build_pair_features(resume_pos, job_pos, data)["skill_coverage"]

    def score_pair_semantic(resume_pos, job_pos):
        return build_pair_features(resume_pos, job_pos, data)["semantic_similarity"]

    scorers = {
        "keyword_baseline": score_pair_keyword,
        "semantic_baseline": score_pair_semantic,
        "final_hybrid_weighted": score_pair_hybrid,
        "random_forest": score_pair_rf,
    }

    ranking_results = {}
    for method_name, score_fn in scorers.items():
        queries = []
        for qi, job_pos in enumerate(test_pos):
            candidate_positions = [job_pos] + list(
                rng.choice([p for p in test_pos if p != job_pos], size=NUM_RANKING_NEGATIVES, replace=False)
            )
            relevance = [
                1 if (cand == job_pos and df.iloc[job_pos]["match_label"] != "weak") else 0
                for cand in candidate_positions
            ]
            scores = [score_fn(cand, job_pos) for cand in candidate_positions]
            order = np.argsort(scores)[::-1]
            relevance_sorted = [relevance[i] for i in order]
            queries.append(relevance_sorted)
        ranking_results[method_name] = evaluate_ranking_queries(queries, k=RANKING_K)
        print(f"  {method_name}: {ranking_results[method_name]}")

    # ---------- save everything ----------
    with open(RESULTS_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    with open(RESULTS_DIR / "ranking_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(ranking_results, f, indent=2)

    training_config = {
        "random_state": RANDOM_STATE,
        "strong_match_threshold": STRONG_MATCH_THRESHOLD,
        "moderate_match_threshold": MODERATE_MATCH_THRESHOLD,
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_ngram_range": list(TFIDF_NGRAM_RANGE),
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": CLASS_LABELS,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "hybrid_weights": __import__("src.matching.hybrid", fromlist=["HYBRID_WEIGHTS"]).HYBRID_WEIGHTS,
    }
    with open(MODELS_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2)

    print("\n=== MODEL COMPARISON (test set) ===")
    for name, metrics in comparison.items():
        print(name, metrics)
    print(f"\nSaved comparison table -> {RESULTS_DIR / 'model_comparison.json'}")
    print(f"Saved ranking evaluation -> {RESULTS_DIR / 'ranking_evaluation.json'}")
    print(f"Saved trained models -> {MODELS_DIR}")


if __name__ == "__main__":
    main()
