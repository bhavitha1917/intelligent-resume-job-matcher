import _bootstrap  # noqa: F401
import pandas as pd
import streamlit as st
from components.ui import setup_page

from src.scoring.config import load_rubric
from src.scoring.scorer import COMPONENT_LABELS, INTERPRETATION

setup_page("Methodology")
st.title("Methodology")
rubric = load_rubric()

st.markdown("## What the score means")
st.info(INTERPRETATION)

st.markdown("## The scoring rubric")
st.caption(f"Rubric v{rubric['version']} · fingerprint `{rubric['_fingerprint']}` · source: `config/scoring_rubric.json`")
st.dataframe(pd.DataFrame([{"Component": COMPONENT_LABELS[k], "Weight": w,
                            "Applies when the job states…": {
                                "required_skills": "at least one required skill",
                                "experience": "a minimum number of years",
                                "education": "a required degree level",
                                "certifications": "a required certification",
                                "preferred_qualifications": "any preferred item",
                                "semantic_relevance": "always (if the model is enabled)",
                            }[k]} for k, w in rubric["component_weights"].items()]),
             hide_index=True, width="stretch")

cap = rubric["essential_requirements"]
credit = rubric["skill_evidence_credit"]
st.markdown(f"""
**Aggregation**

`uncapped = 100 × Σ (wᶜ × vᶜ) / Σ wᶜ` over *applicable* components only (weights renormalised), each value vᶜ ∈ [0, 1].

`score = min(uncapped, {cap['score_cap_when_unmet']:.0f})` if a **must-have** requirement is not evidenced, or required-skill
evidence is below {cap['min_required_skill_component_for_strong']:.0%}; otherwise `score = uncapped`.
Bands: {", ".join(f"≥ {b['min_score']:.0f} {b['label']}" for b in rubric['bands'])}.

**Component values**

- *Required skills / certifications*: mean credit per requirement. A skill is **applied** ({credit['applied']}) when it appears
  in an experience, internship or project section or next to an action verb ("built", "deployed" …), **listed**
  ({credit['listed']}) when it only appears in a skills list or summary, **contradicted** ({credit['contradicted']}) when the
  resume negates it ("no experience with X"), **not detected** ({credit['not_detected']}) otherwise. Each requirement counts
  once, however many times it is mentioned.
- *Experience*: dated employment and internship entries that mention a job skill or title word are merged (overlaps counted
  once); value = min(1, detected years / required years). Stated-but-undated years ("5 years of experience") are used only
  when no dated entry exists, and are flagged.
- *Education*: 1.0 if the highest detected degree meets the level and field; ×{rubric['education']['unmatched_field_credit']}
  if the field is not recognised; ×{rubric['education']['in_progress_credit']} if in progress; 0 if below the level or not detected.
- *Preferred qualifications*: mean credit over all preferred items of any kind.
- *Contextual relevance*: cosine similarity of `all-MiniLM-L6-v2` embeddings of the (PII-redacted) resume and the job,
  rescaled linearly from [{rubric['semantic_normalization']['low']}, {rubric['semantic_normalization']['high']}] to [0, 1]
  (5th/95th percentiles over the dataset, computed without labels).

**Insufficient evidence**: no score is shown if fewer than {rubric['insufficient_evidence']['min_resume_words']} words of
resume text are usable or the job has no scorable requirement.

**Why these weights?** They are a documented initial judgment, not fitted to the dataset labels: explicit required skills
carry the most weight because they are what the job states it needs; experience and education are standard screening criteria
but are extracted less reliably; semantic similarity is limited to {rubric['component_weights']['semantic_relevance']:.0%} so
generic text overlap cannot outweigh missing requirements. See `docs/SCORING.md` for the full rationale and limitations.
""")

st.markdown("""
## Pipeline

1. **Parsing** — PDF (PyMuPDF), DOCX (standard-library XML reader) and TXT. Scanned PDFs are not OCR'd. Sections are
   detected from header lines (or inline headers for single-line text).
2. **Privacy** — e-mail, phone numbers, URLs, and lines stating date of birth, age, gender, marital status, nationality,
   religion, caste, family names, IDs or address are blanked before any scoring step. Names are not removed from the text
   but are never used as features.
3. **Requirements** — extracted from "Requirements / Preferred / Must have" sections; you review and edit them before
   scoring. Skills use a curated dictionary of ~230 skills and certifications with aliases (ambiguous abbreviations
   such as "ML", "CV", "TS" must appear in capitals), plus literal phrases for anything not in the dictionary.
4. **Evidence** — every requirement gets a status (detected / partial / not detected / contradicted), a credit and the
   resume text that justified it.
5. **Score** — the rubric above. Single-resume and ranking pages call the same function.

## Data and evaluation

The Hugging Face `batuhanmtl/job_resume_fit` dataset (2,385 pairs) has **only 23 distinct job descriptions**, and its
`ai_match_score` is generated by the dataset authors' automated pipeline — not by recruiters. The learned models are
therefore evaluated with **leave-one-job-out** cross-validation, and are shown only as an experimental diagnostic. See the
**Evaluation** page for measured results and their limits. No human-labelled relevance set exists for this project; building
one is the necessary next step before any claim about real-world ranking quality.

## Known limitations

- Rule-based extraction: unusual resume layouts, tables and multi-column PDFs can be mis-sectioned.
- Negation handling covers short preceding cues only; synonyms outside the dictionary are only partly captured by the
  semantic component.
- Experience durations use month/year ranges; year-only ranges are counted from January; "Present" is the analysis date.
- Excluding protected attributes does not make the system bias-free: wording, school names, career gaps and the
  dictionary's coverage of different professions can still correlate with protected characteristics.
- Uploaded files are processed in memory for the session and are not written to disk or logged by this app.
""")
