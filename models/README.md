# Model artifacts

## `models/v2/` (current, produced by `python -m src.models.train_supervised`)

| File | Contents |
|---|---|
| `manifest.json` | feature schema version, exact feature column list, class labels, label definition, training data summary, rubric fingerprint, scikit-learn version |
| `random_forest.joblib`, `logistic_regression.joblib`, `svm.joblib` | `sklearn.pipeline.Pipeline(StandardScaler → model)`, so the scaler is always stored with its model |
| `tfidf_vectorizer.joblib` | TF-IDF vectorizer used for the `tfidf_similarity` feature |

`src/models/inference.py` loads these only if the manifest's `feature_schema_version` and
`feature_columns` exactly match the code (`src/features/feature_engineering.py`). Otherwise it refuses and
the app shows why.

The models predict the dataset's **algorithmic** weak / moderate / strong label. The app shows the result
only in a collapsed "experimental" panel. **The models are not used for the displayed score or the
ranking**; the transparent rubric is (`docs/SCORING.md`). Final models are refit on all 2,383 rows *after*
the leave-one-job-out evaluation; the metrics in `reports/results/v2/evaluation.json` come from the
cross-validation folds, not from these refit models.

## `models/*.joblib` and `models/training_config.json` (v1, legacy, NOT loaded)

These are the artifacts of the originally deployed version (git tag `baseline-v1-deployed`). They are kept
unchanged so that version can be reproduced and rolled back to. They were trained on features with known
defects: skill coverage computed inconsistently with inference (0 for 92.9% of rows), 0.5 fallbacks for
unknown experience and education, and a job-identifying `job_description_length` feature. They are
incompatible with the v2 feature schema, and the v2 code never loads them.
