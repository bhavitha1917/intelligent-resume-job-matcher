# Deploying to Streamlit Community Cloud

This project has not been deployed from the development environment it was
built in — deploying requires interactive GitHub/Streamlit Cloud
authentication in a browser, which that environment does not have. These
are the exact steps for whoever owns the GitHub account to finish
deployment; everything up to the "click Deploy" step is already done.

## Prerequisites (already satisfied in this repo)

- `app/app.py` is a valid multipage Streamlit entry point (`app/pages/` holds
  the other 5 pages).
- `requirements.txt` is complete and version-pinned.
- Trained model artifacts are committed under `models/` — the app loads them
  at runtime and never retrains (see `src/models/inference.py`).
- No hardcoded absolute paths — everything resolves via `src/utils/paths.py`
  relative to the repo root, so the app works regardless of clone location.
- No secrets are required (`.env.example` documents this).
- `.streamlit/config.toml` sets headless mode and disables usage-stat collection.

## Steps

1. Push this repository to GitHub (if not already there):
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin master
   ```
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click **New app**.
4. Select this repository, the branch (`master`), and set:
   - **Main file path:** `app/app.py`
5. Click **Deploy**.

Streamlit Cloud will:
- Install everything in `requirements.txt`.
- Download and cache `all-MiniLM-L6-v2` on first use (not bundled in Git —
  see `src/matching/semantic_similarity.py`).
- Load the committed `models/*.joblib` artifacts directly.

## After deploying

- First request may take anywhere from ~20 seconds to a few minutes (cold
  start + sentence-transformer model download/load — during local testing
  this varied significantly run-to-run, even with the model already
  cached). Subsequent requests in the same running instance are fast.
  If it consistently takes longer than a few minutes on Streamlit Cloud,
  check the app logs for network errors reaching huggingface.co.
- If you later re-run `python -m src.models.train_supervised` and commit new
  model artifacts, Streamlit Cloud redeploys automatically on push.
- To update secrets (none required currently), use the app's **Settings →
  Secrets** panel in the Streamlit Cloud dashboard — never commit a
  `.streamlit/secrets.toml` file (already gitignored).

## Verifying a deployment

Once deployed, repeat the same manual check used during development:
1. Open the app URL — Home page should load.
2. Go to **Single Resume Analysis**, check both "bundled demo" boxes, click
   **Analyze match** — expect a match score, matched/missing skills, and an
   explanation within ~30 seconds on first run.
3. Go to **Candidate Ranking**, click **Rank candidates** — expect a ranked
   table of the 3 demo resumes.
4. Go to **Model Comparison** — expect the comparison table and charts to
   render from `reports/results/model_comparison.json`.
