# Deploying, reviewing and rolling back (Streamlit Community Cloud)

The production app is deployed from the **`main`** branch, with main file **`app/app.py`**.

The v2 rework lives on the branch **`improve/transparent-scoring`**. The commit that was in production
before the rework is tagged **`baseline-v1-deployed`** (commit `9b2c646`). Nothing in this guide changes
production until you merge into `main` yourself.

## 1. Review the new version before it reaches production

### Option A: run it locally (recommended first)

```bash
git fetch origin
git switch improve/transparent-scoring
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m pytest -q               # full suite; should report 0 failures
streamlit run app/app.py          # http://localhost:8501
```

### Option B: a separate preview deployment (production untouched)

1. Push the branch: `git push -u origin improve/transparent-scoring`.
2. On https://share.streamlit.io choose **Create app → Deploy a public app from GitHub**.
3. Pick the same repository, branch **`improve/transparent-scoring`**, main file `app/app.py`, and a
   *different* app URL (e.g. `…-preview`).
4. Test it, including on your phone (section 3). Delete the preview app when done.

### Checklist before promoting

- [ ] `python -m pytest -q` passes locally.
- [ ] Single Resume Analysis: demo job + each demo resume gives a report with components, evidence and
      no errors. `resume_2.txt` shows *Machine Learning — Contradicted*.
- [ ] Candidate Ranking: demo job + 3 demo resumes gives ranks 1–3, and each expander shows the same score
      as Single Resume Analysis for that resume; the CSV downloads.
- [ ] Upload one real PDF and one DOCX of your own (do not commit them); check the evidence quotes.
- [ ] Evaluation page loads `reports/results/v2/evaluation.json`.
- [ ] First load on Streamlit Cloud: the semantic model downloads (~90 MB). If it fails, the app still
      scores without that component and says so.

## 2. Promote to production

```bash
git switch main
git merge --no-ff improve/transparent-scoring
git push origin main               # Streamlit Cloud redeploys main automatically
```

## 3. Android access

1. Open the deployed URL in **Chrome on Android**.
2. Check the pages: the layout stacks into one column, buttons are full width, and reports use expandable
   sections instead of wide tables. The ranking table scrolls horizontally inside its own box; use the
   candidate report expanders below it on narrow screens.
3. **⋮ menu → Add to Home screen** (or **Install app**, depending on Chrome version) → **Add**. The
   shortcut opens the site in Chrome.

This is a home-screen **shortcut to the website**, not an installable PWA or native app. Streamlit serves
its own HTML page and does not let an app register a web manifest or service worker at the site root, which
proper PWA installation requires. Streamlit Community Cloud also wraps apps in its own domain shell.
Faking this with injected scripts would be brittle, so it was not done. The app needs the server for every
analysis, so it does not work offline.

## 4. Roll back

If something is wrong after promoting:

```bash
git switch main
git revert -m 1 <merge-commit-sha>   # keeps history; Streamlit redeploys the previous behaviour
git push origin main
```

Or, to restore exactly the previously deployed code (rewrites `main`; only if you are sure):

```bash
git switch main
git reset --hard baseline-v1-deployed
git push --force-with-lease origin main
```

The v1 model files (`models/*.joblib`) are kept unchanged on every branch, so either rollback restores a
working v1 app.

## 5. Runtime notes

- No secrets are needed. Never commit `.streamlit/secrets.toml`, `.env`, resumes or datasets.
- Uploads are limited to 10 MB per file (`.streamlit/config.toml`). Files are processed in memory and are
  not written to disk by the app. Streamlit Cloud keeps its own server logs; the app does not log resume
  content.
- Model artifacts are committed (`models/v2/`), and the app never trains at runtime.
