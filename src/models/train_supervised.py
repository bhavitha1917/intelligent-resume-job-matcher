"""v2 training + honest evaluation on the HF `batuhanmtl/job_resume_fit` dataset.

Run with:  .venv/Scripts/python.exe -m src.models.train_supervised

What the labels are: `ai_match_score` (0-100) is produced by the dataset
authors' own automated pipeline. It is NOT a recruiter decision. It is
bucketed into weak / moderate / strong with the documented thresholds in
src/models/config.py. Every metric below measures agreement with that
algorithmic label -- not real hiring suitability.

The dataset has 2,385 rows but only 23 distinct job descriptions (one job
per category), so this script reports two protocols side by side:

  A. row-level stratified split (the v1 protocol). The same 23 jobs occur
     in train and test, so a model can exploit "which job is this"
     (mean label differs a lot by job). Kept only for comparison with v1.
  B. leave-one-job-out (LOJO, 23 folds). The test job never appears in
     training. This is the protocol that estimates generalization to an
     unseen job description -- with only 23 groups, so results are noisy.

Leakage rules (both protocols): TF-IDF vectorizers and scalers are fit
on the training part only (scalers live inside sklearn Pipelines); model
hyperparameters are fixed in advance (the same values as v1) -- nothing
is tuned on test data, so no separate validation set is used for
selection; the sentence-transformer is frozen and pretrained.

After evaluation, final models are refit on all rows and saved under
models/v2/ together with a manifest describing the exact feature schema.
v1 artifacts in models/ are left untouched (and are refused by the v2
loader in src/models/inference.py).
"""

import ast
import hashlib
import json
import time
import unicodedata
from datetime import date

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

from src.data.resume_parser import parse_resume_text
from src.evaluation.legacy_v1 import (
    LEGACY_FEATURE_COLUMNS,
    education_compatibility,
    experience_compatibility,
    legacy_skill_features,
)
from src.evaluation.ranking_metrics import ndcg_at_k, precision_at_k, reciprocal_rank
from src.features.feature_engineering import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, model_features_from_report
from src.matching.semantic_similarity import DEFAULT_MODEL_NAME, batch_encode
from src.models.config import (
    MODERATE_MATCH_THRESHOLD,
    RANDOM_STATE,
    STRONG_MATCH_THRESHOLD,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    score_to_label,
)
from src.preprocessing.text_cleaning import clean_text, handle_missing
from src.scoring.config import load_rubric
from src.scoring.privacy import sanitize_for_scoring
from src.scoring.requirements import requirements_from_dataset_record
from src.scoring.scorer import score_candidate
from src.utils.paths import FIGURES_DIR, MODELS_V2_DIR, PROCESSED_DATA_DIR, RESULTS_V2_DIR

HF_DATASET_NAME = "batuhanmtl/job_resume_fit"
CLASS_LABELS = ["weak", "moderate", "strong"]
LABEL_GAIN = {"weak": 0, "moderate": 1, "strong": 2}
# "Present" in resume date ranges is resolved against this fixed date so the
# dataset features are reproducible (the app uses the analysis date instead).
DATASET_REFERENCE_DATE = date(2025, 1, 1)
RANK_K = 10
FIGURES_V2_DIR = FIGURES_DIR / "v2"


def make_models() -> dict:
    """Fixed hyperparameters (identical to v1); scaler inside the pipeline
    so it is always fit on the training part only."""
    return {
        "logistic_regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "random_forest": Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))]),
        # Platt-scaled RBF SVM (what v1's SVC(probability=True) did; that flag is deprecated in sklearn 1.9).
        "svm": Pipeline([("scaler", StandardScaler()), ("model", CalibratedClassifierCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE),
            method="sigmoid", cv=5, ensemble=False))]),
    }


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE, sublinear_tf=True, min_df=2)


def _parse_list(value) -> list:
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else []
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


# ----------------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------------
def load_dataset_frame() -> pd.DataFrame:
    from datasets import load_dataset

    df = load_dataset(HF_DATASET_NAME)["train"].to_pandas()
    report = {"rows_raw": int(len(df)), "unique_jobs": int(df["job_text"].nunique()),
              "unique_resumes": int(df["resume_text"].nunique())}
    df = df.drop_duplicates(subset="resume_text", keep="first").reset_index(drop=True)
    report["rows_after_exact_resume_dedupe"] = int(len(df))
    df["job_id"] = pd.factorize(df["job_text"])[0]
    df["match_label"] = df["ai_match_score"].apply(score_to_label)
    df["required_skills_list"] = df["job_required_skills"].apply(_parse_list)
    df["resume_raw"] = df["resume_text"].apply(lambda t: unicodedata.normalize("NFKC", handle_missing(t)))
    df["job_raw"] = df["job_text"].apply(lambda t: unicodedata.normalize("NFKC", handle_missing(t)))
    df["resume_clean_v1"] = df["resume_text"].apply(lambda t: clean_text(handle_missing(t)))
    df["job_clean_v1"] = df["job_text"].apply(lambda t: clean_text(handle_missing(t)))
    return df, report


def _cached_embeddings(texts: list, tag: str) -> np.ndarray:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(("\x1e".join(texts) + DEFAULT_MODEL_NAME).encode("utf-8")).hexdigest()[:16]
    path = PROCESSED_DATA_DIR / f"embeddings_{tag}_{digest}.npy"
    if path.exists():
        return np.load(path)
    emb = batch_encode(texts)
    np.save(path, emb)
    return emb


def near_duplicate_report(texts: list, job_ids: np.ndarray, threshold: float = 0.95) -> dict:
    vec = TfidfVectorizer(min_df=1).fit_transform(texts)
    sims = (vec @ vec.T).tocoo()
    pairs = [(i, j) for i, j, v in zip(sims.row, sims.col, sims.data) if i < j and v >= threshold]
    return {
        "threshold_tfidf_cosine": threshold,
        "near_duplicate_resume_pairs": len(pairs),
        "pairs_across_different_jobs": int(sum(job_ids[i] != job_ids[j] for i, j in pairs)),
        "note": "Computed on the resume text only. Pairs within the same job cannot leak across "
                "leave-one-job-out folds; cross-job pairs are reported for transparency.",
    }


def build_reports(df: pd.DataFrame, rubric: dict, semantic_raw: np.ndarray) -> list:
    reports = []
    requirement_cache: dict = {}  # only 23 distinct jobs
    for i, row in enumerate(df.itertuples(index=False)):
        parsed = parse_resume_text(row.resume_raw)
        key = (row.job_raw, tuple(row.required_skills_list))
        if key not in requirement_cache:
            requirement_cache[key] = requirements_from_dataset_record(row.job_raw, row.required_skills_list)
        job = {"title": row.job_raw.strip().split("\n")[0][:120], "text": row.job_raw,
               "requirements": requirement_cache[key]}
        reports.append(score_candidate(job, parsed, rubric=rubric, semantic_raw=float(semantic_raw[i]),
                                       reference_date=DATASET_REFERENCE_DATE))
    return reports


def tfidf_pair_similarity(vectorizer, resume_texts: list, job_texts: list) -> np.ndarray:
    """Row-wise cosine (vectors are L2-normalised). Each distinct job text is
    transformed once (there are only 23)."""
    r = vectorizer.transform(resume_texts)
    unique_jobs = list(dict.fromkeys(job_texts))
    job_vecs = vectorizer.transform(unique_jobs)
    position = {t: i for i, t in enumerate(unique_jobs)}
    j = job_vecs[[position[t] for t in job_texts]]
    return np.asarray(r.multiply(j).sum(axis=1)).ravel()


def fit_fold_vectorizer(train_idx, resume_texts, job_texts):
    corpus = [resume_texts[i] for i in train_idx] + list(dict.fromkeys(job_texts[i] for i in train_idx))
    return make_vectorizer().fit(corpus)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, proba=None) -> dict:
    out = {
        "n": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0), 4),
        "confusion_matrix": {"labels": CLASS_LABELS,
                             "matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist()},
    }
    if proba is not None:
        try:
            out["roc_auc_macro_ovr"] = round(roc_auc_score(label_binarize(y_true, classes=CLASS_LABELS), proba,
                                                           average="macro", multi_class="ovr"), 4)
        except ValueError:
            out["roc_auc_macro_ovr"] = None
    return out


def threshold_labels(scores_0_100) -> np.ndarray:
    return np.array([score_to_label(s) for s in scores_0_100])


def ranking_metrics_by_job(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Each job is one query; its paired resumes are the candidates; graded
    relevance = dataset label (weak 0 / moderate 1 / strong 2). Ties are
    broken by row order (stable sort) -- the same for every method."""
    ndcgs, p_at_k, rrs, rhos = [], [], [], []
    jobs_without_strong = 0
    for job_id, idx in df.groupby("job_id").indices.items():
        order = np.argsort(-scores[idx], kind="stable")
        gains = [LABEL_GAIN[df["match_label"].iloc[idx[o]]] for o in order]
        ndcgs.append(ndcg_at_k(gains, RANK_K))
        strong = [1 if g == 2 else 0 for g in gains]
        if sum(strong):
            p_at_k.append(precision_at_k(strong, RANK_K))
            rrs.append(reciprocal_rank(strong))
        else:
            jobs_without_strong += 1
        rho = spearmanr(scores[idx], df["ai_match_score"].iloc[idx]).correlation
        if not np.isnan(rho):
            rhos.append(rho)
    return {
        "queries": int(df["job_id"].nunique()),
        f"ndcg_at_{RANK_K}_graded": round(float(np.mean(ndcgs)), 4),
        f"precision_at_{RANK_K}_strong": round(float(np.mean(p_at_k)), 4) if p_at_k else None,
        "mrr_first_strong": round(float(np.mean(rrs)), 4) if rrs else None,
        "mean_within_job_spearman_vs_ai_match_score": round(float(np.mean(rhos)), 4) if rhos else None,
        "jobs_without_any_strong_candidate": jobs_without_strong,
    }


def expected_gain(proba: np.ndarray) -> np.ndarray:
    """Ranking scalar from class probabilities (columns in CLASS_LABELS
    order): E[gain] = P(moderate) + 2 P(strong). Used for ranking metrics
    only -- never displayed as a 'match percentage'."""
    return proba[:, 1] + 2 * proba[:, 2]


def ordered_proba(model, X) -> np.ndarray:
    classes = list(model.classes_)
    proba = model.predict_proba(X)
    out = np.zeros((len(X), len(CLASS_LABELS)))
    for k, label in enumerate(CLASS_LABELS):
        if label in classes:
            out[:, k] = proba[:, classes.index(label)]
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    for d in (MODELS_V2_DIR, RESULTS_V2_DIR, FIGURES_V2_DIR):
        d.mkdir(parents=True, exist_ok=True)
    rubric = load_rubric()

    print("Loading dataset ...")
    df, data_report = load_dataset_frame()
    y = df["match_label"].to_numpy()
    groups = df["job_id"].to_numpy()
    data_report["label_distribution"] = df["match_label"].value_counts().to_dict()
    data_report["rows_per_job"] = {"min": int(df.groupby("job_id").size().min()), "max": int(df.groupby("job_id").size().max())}
    data_report["mean_ai_match_score_by_category"] = df.groupby("category")["ai_match_score"].mean().round(1).to_dict()
    data_report["near_duplicates"] = near_duplicate_report(df["resume_raw"].tolist(), groups)

    print("Sanitizing resumes and computing frozen sentence-transformer embeddings (cached) ...")
    sanitized = [sanitize_for_scoring(t, parse_resume_text(t)["section_spans"])[0] for t in df["resume_raw"]]
    job_unique = list(dict.fromkeys(df["job_raw"]))
    job_emb = dict(zip(job_unique, _cached_embeddings(job_unique, "job_v2")))
    res_emb = _cached_embeddings(sanitized, "resume_v2")
    semantic_v2 = np.array([float(np.dot(res_emb[i], job_emb[j])) for i, j in enumerate(df["job_raw"])])

    bounds = {"p05": round(float(np.percentile(semantic_v2, 5)), 4), "p95": round(float(np.percentile(semantic_v2, 95)), 4),
              "n_pairs": int(len(semantic_v2)), "labels_used": False,
              "config_low": rubric["semantic_normalization"]["low"], "config_high": rubric["semantic_normalization"]["high"]}
    (RESULTS_V2_DIR / "semantic_bounds.json").write_text(json.dumps(bounds, indent=2), encoding="utf-8")
    print(f"  semantic similarity P5={bounds['p05']} P95={bounds['p95']} (config uses {bounds['config_low']}/{bounds['config_high']})")

    print("Scoring every dataset pair with the SAME scorer the app uses ...")
    reports = build_reports(df, rubric, semantic_v2)
    rubric_scores = np.array([r["score"] if r["score"] is not None else np.nan for r in reports])
    insufficient = int(np.isnan(rubric_scores).sum())
    rubric_scores_filled = np.nan_to_num(rubric_scores, nan=0.0)

    skill_reqs = [[q for q in r["requirements"] if q["kind"] == "skill"] for r in reports]
    coverage_v2 = np.array([np.mean([q["result"]["status"] == "met" for q in s]) if s else 0.0 for s in skill_reqs])
    legacy_skills = [legacy_skill_features(t, s) for t, s in zip(df["resume_clean_v1"], df["required_skills_list"])]
    coverage_v1 = np.array([f["skill_coverage"] for f in legacy_skills])
    data_report["skill_coverage_zero_fraction"] = {
        "v1_training_features": round(float((coverage_v1 == 0).mean()), 4),
        "v2_shared_matcher": round(float((coverage_v2 == 0).mean()), 4),
    }
    data_report["rubric_insufficient_evidence_rows"] = insufficient
    print(f"  skill_coverage==0: v1 {data_report['skill_coverage_zero_fraction']['v1_training_features']:.1%}"
          f" -> v2 {data_report['skill_coverage_zero_fraction']['v2_shared_matcher']:.1%}")

    print("Legacy (v1) semantic embeddings for the comparison ...")
    job_unique_v1 = list(dict.fromkeys(df["job_clean_v1"]))
    job_emb_v1 = dict(zip(job_unique_v1, _cached_embeddings(job_unique_v1, "job_v1")))
    res_emb_v1 = _cached_embeddings(df["resume_clean_v1"].tolist(), "resume_v1")
    semantic_v1 = np.array([float(np.dot(res_emb_v1[i], job_emb_v1[j])) for i, j in enumerate(df["job_clean_v1"])])

    resume_v2_texts, job_v2_texts = sanitized, df["job_raw"].tolist()
    resume_v1_texts, job_v1_texts = df["resume_clean_v1"].tolist(), df["job_clean_v1"].tolist()

    # Everything except the TF-IDF column is fold-independent: build once.
    v2_base = pd.DataFrame([model_features_from_report(r, 0.0) for r in reports], columns=FEATURE_COLUMNS)
    v1_base = pd.DataFrame([
        {**legacy_skills[i], "tfidf_similarity": 0.0, "semantic_similarity": semantic_v1[i],
         "resume_length": len(row.resume_clean_v1.split()), "job_description_length": len(row.job_clean_v1.split()),
         "experience_compatibility": experience_compatibility(row.resume_clean_v1, row.job_clean_v1),
         "education_compatibility": education_compatibility(row.resume_clean_v1, row.job_clean_v1)}
        for i, row in enumerate(df.itertuples(index=False))
    ], columns=LEGACY_FEATURE_COLUMNS)

    def v2_matrix(tfidf_sims) -> pd.DataFrame:
        X = v2_base.copy()
        X["tfidf_similarity"] = tfidf_sims
        return X

    def v1_matrix(tfidf_sims) -> pd.DataFrame:
        X = v1_base.copy()
        X["tfidf_similarity"] = tfidf_sims
        return X

    results = {"dataset": data_report, "protocols": {}}

    # ---------------- Protocol A: row-level split (v1 protocol) ----------------
    print("Protocol A: row-level stratified split (v1 protocol) ...")
    idx_all = np.arange(len(df))
    train_idx, temp_idx = train_test_split(idx_all, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    _, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=RANDOM_STATE)
    proto_a = {"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
               "jobs_in_both_train_and_test": int(len(set(groups[train_idx]) & set(groups[test_idx]))), "methods": {}}
    vec_v2 = fit_fold_vectorizer(train_idx, resume_v2_texts, job_v2_texts)
    vec_v1 = fit_fold_vectorizer(train_idx, resume_v1_texts, job_v1_texts)
    X_v2 = v2_matrix(tfidf_pair_similarity(vec_v2, resume_v2_texts, job_v2_texts))
    X_v1 = v1_matrix(tfidf_pair_similarity(vec_v1, resume_v1_texts, job_v1_texts))
    majority = pd.Series(y[train_idx]).mode()[0]
    proto_a["methods"]["majority_class"] = classification_metrics(y[test_idx], np.full(len(test_idx), majority))
    job_mode = pd.DataFrame({"g": groups[train_idx], "y": y[train_idx]}).groupby("g")["y"].agg(lambda s: s.mode()[0])
    proto_a["methods"]["job_identity_only"] = classification_metrics(
        y[test_idx], np.array([job_mode.get(g, majority) for g in groups[test_idx]]))
    proto_a["methods"]["rubric_score_thresholded"] = classification_metrics(y[test_idx], threshold_labels(rubric_scores_filled[test_idx]))
    for feat_name, X in (("v1_features", X_v1), ("v2_features", X_v2)):
        for name, model in make_models().items():
            m = clone(model).fit(X.iloc[train_idx], y[train_idx])
            proba = ordered_proba(m, X.iloc[test_idx])
            proto_a["methods"][f"{name}__{feat_name}"] = classification_metrics(y[test_idx], m.predict(X.iloc[test_idx]), proba)
    results["protocols"]["A_row_level_split"] = proto_a

    # ---------------- Protocol B: leave-one-job-out ----------------
    print("Protocol B: leave-one-job-out (23 folds) ...")
    lojo_pred = {}
    lojo_proba = {}
    tfidf_v2_lojo = np.zeros(len(df))
    tfidf_v1_lojo = np.zeros(len(df))
    majority_pred = np.empty(len(df), dtype=object)
    for g in np.unique(groups):
        test = np.where(groups == g)[0]
        train = np.where(groups != g)[0]
        v2v = fit_fold_vectorizer(train, resume_v2_texts, job_v2_texts)
        v1v = fit_fold_vectorizer(train, resume_v1_texts, job_v1_texts)
        tfidf_v2_lojo[test] = tfidf_pair_similarity(v2v, [resume_v2_texts[i] for i in test], [job_v2_texts[i] for i in test])
        tfidf_v1_lojo[test] = tfidf_pair_similarity(v1v, [resume_v1_texts[i] for i in test], [job_v1_texts[i] for i in test])
        # the fold's training-feature tf-idf values must also come from the fold vectorizer
        tf_v2_train = tfidf_pair_similarity(v2v, [resume_v2_texts[i] for i in train], [job_v2_texts[i] for i in train])
        tf_v1_train = tfidf_pair_similarity(v1v, [resume_v1_texts[i] for i in train], [job_v1_texts[i] for i in train])
        majority_pred[test] = pd.Series(y[train]).mode()[0]
        for feat_name, builder, tf_test, tf_train in (
            ("v2_features", v2_matrix, tfidf_v2_lojo, tf_v2_train),
            ("v1_features", v1_matrix, tfidf_v1_lojo, tf_v1_train),
        ):
            tf_full = np.zeros(len(df))
            tf_full[train] = tf_train
            tf_full[test] = tf_test[test]
            X = builder(tf_full)
            for name, model in make_models().items():
                key = f"{name}__{feat_name}"
                m = clone(model).fit(X.iloc[train], y[train])
                lojo_pred.setdefault(key, np.empty(len(df), dtype=object))[test] = m.predict(X.iloc[test])
                lojo_proba.setdefault(key, np.zeros((len(df), 3)))[test] = ordered_proba(m, X.iloc[test])
        print(f"  fold job {g + 1}/23 done")

    proto_b = {"folds": int(len(np.unique(groups))), "methods": {}, "per_job_accuracy": {}}
    proto_b["methods"]["majority_class"] = classification_metrics(y, majority_pred.astype(str))
    proto_b["methods"]["rubric_score_thresholded"] = classification_metrics(y, threshold_labels(rubric_scores_filled))
    proto_b["methods"]["semantic_only_thresholded_v1_rule"] = classification_metrics(y, threshold_labels(semantic_v2 * 100))
    for key in lojo_pred:
        proto_b["methods"][key] = classification_metrics(y, lojo_pred[key].astype(str), lojo_proba[key])
    for key in ("random_forest__v2_features", "random_forest__v1_features"):
        per_job = pd.DataFrame({"g": df["category"], "ok": lojo_pred[key].astype(str) == y}).groupby("g")["ok"].mean()
        proto_b["per_job_accuracy"][key] = {"min": round(float(per_job.min()), 4), "median": round(float(per_job.median()), 4),
                                            "max": round(float(per_job.max()), 4), "by_job": per_job.round(4).to_dict()}
    results["protocols"]["B_leave_one_job_out"] = proto_b

    # ---------------- Rubric vs dataset label (no fitting involved) ----------------
    valid = ~np.isnan(rubric_scores)
    within = []
    for _, idx in df[valid].groupby("job_id").indices.items():
        rho = spearmanr(rubric_scores[valid][idx], df["ai_match_score"].to_numpy()[valid][idx]).correlation
        if not np.isnan(rho):
            within.append(rho)
    results["rubric_agreement_with_dataset_score"] = {
        "spearman_pooled": round(float(spearmanr(rubric_scores[valid], df["ai_match_score"][valid]).correlation), 4),
        "spearman_mean_within_job": round(float(np.mean(within)), 4),
        "rows_scored": int(valid.sum()), "rows_insufficient_evidence": insufficient,
        "note": "The rubric was not fitted to these labels; agreement is reported, not optimized.",
    }

    # ---------------- Within-job ranking (unseen-job predictions only) ----------------
    print("Within-job ranking evaluation ...")
    ranking = {
        "rubric_score": ranking_metrics_by_job(df, rubric_scores_filled),
        "required_skill_coverage_v2": ranking_metrics_by_job(df, coverage_v2),
        "required_skill_coverage_v1": ranking_metrics_by_job(df, coverage_v1),
        "semantic_similarity": ranking_metrics_by_job(df, semantic_v2),
        "tfidf_similarity_lojo": ranking_metrics_by_job(df, tfidf_v2_lojo),
    }
    for key in ("random_forest__v2_features", "random_forest__v1_features",
                "logistic_regression__v2_features", "svm__v2_features"):
        ranking[f"{key}__lojo_expected_gain"] = ranking_metrics_by_job(df, expected_gain(lojo_proba[key]))
    rng = np.random.default_rng(RANDOM_STATE)
    ranking["random_order_mean_of_20"] = {
        k: round(float(np.mean([ranking_metrics_by_job(df, rng.random(len(df)))[k] for _ in range(20)])), 4)
        for k in (f"ndcg_at_{RANK_K}_graded", f"precision_at_{RANK_K}_strong", "mrr_first_strong")
    }
    results["within_job_ranking"] = {
        "design": f"One query per job (23). Candidates = that job's paired resumes (22-120). Graded relevance = "
                  f"dataset label (weak 0, moderate 1, strong 2). Learned-model scores come from leave-one-job-out "
                  f"predictions, so the ranked job was never seen in training. K={RANK_K}. Relevance labels are "
                  f"algorithmic, not human judgments.",
        "methods": ranking,
    }

    # ---------------- Figures ----------------
    cm = np.array(proto_b["methods"]["random_forest__v2_features"]["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Random Forest (v2 features)\nleave-one-job-out, pooled")
    plt.tight_layout()
    plt.savefig(FIGURES_V2_DIR / "confusion_matrix_rf_v2_lojo.png", dpi=150)
    plt.close()

    compare = {
        "v1 features\nrow split": proto_a["methods"]["random_forest__v1_features"]["f1_macro"],
        "v1 features\nleave-one-job-out": proto_b["methods"]["random_forest__v1_features"]["f1_macro"],
        "v2 features\nrow split": proto_a["methods"]["random_forest__v2_features"]["f1_macro"],
        "v2 features\nleave-one-job-out": proto_b["methods"]["random_forest__v2_features"]["f1_macro"],
        "job identity only\nrow split": proto_a["methods"]["job_identity_only"]["f1_macro"],
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(list(compare.keys()), list(compare.values()), color="#4C72B0")
    ax.set_xlabel("Macro F1 vs. dataset label (weak/moderate/strong)")
    ax.set_xlim(0, 1)
    ax.set_title("Random Forest: effect of evaluation protocol")
    plt.tight_layout()
    plt.savefig(FIGURES_V2_DIR / "protocol_comparison_rf.png", dpi=150)
    plt.close()

    # ---------------- Final refit on all rows + manifest ----------------
    print("Refitting final v2 models on all rows ...")
    final_vec = fit_fold_vectorizer(idx_all, resume_v2_texts, job_v2_texts)
    X_final = v2_matrix(tfidf_pair_similarity(final_vec, resume_v2_texts, job_v2_texts))
    joblib.dump(final_vec, MODELS_V2_DIR / "tfidf_vectorizer.joblib")
    for name, model in make_models().items():
        joblib.dump(clone(model).fit(X_final, y), MODELS_V2_DIR / f"{name}.joblib")
    manifest = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "class_labels": CLASS_LABELS,
        "label_definition": f"ai_match_score >= {STRONG_MATCH_THRESHOLD} strong; >= {MODERATE_MATCH_THRESHOLD} moderate; else weak "
                            f"(dataset-generated score, not a recruiter decision)",
        "display_model": "random_forest",
        "display_model_choice": "Random Forest was kept as the displayed model because it was the v1 choice; it was "
                                "NOT re-selected using the v2 evaluation results.",
        "trained_on": {"dataset": HF_DATASET_NAME, "rows": int(len(df)), "jobs": int(df["job_id"].nunique())},
        "dataset_reference_date": DATASET_REFERENCE_DATE.isoformat(),
        "rubric_version": rubric["version"], "rubric_fingerprint": rubric["_fingerprint"],
        "semantic_model": DEFAULT_MODEL_NAME, "sklearn_version": sklearn.__version__,
        "trained_at": date.today().isoformat(),
    }
    (MODELS_V2_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    results["meta"] = {"runtime_seconds": round(time.perf_counter() - t_start, 1), **manifest}
    (RESULTS_V2_DIR / "evaluation.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n=== SUMMARY (macro F1 vs dataset label) ===")
    for proto_name, proto in results["protocols"].items():
        print(proto_name)
        for k, v in proto["methods"].items():
            print(f"  {k:40s} F1={v['f1_macro']:.4f} acc={v['accuracy']:.4f} auc={v.get('roc_auc_macro_ovr')}")
    print("Ranking:", json.dumps(ranking, indent=1))
    print(f"Saved -> {RESULTS_V2_DIR / 'evaluation.json'} and models -> {MODELS_V2_DIR}")


if __name__ == "__main__":
    main()
