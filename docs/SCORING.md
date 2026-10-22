# Scoring rubric (v2)

Source of truth: [`config/scoring_rubric.json`](../config/scoring_rubric.json). Implementation:
[`src/scoring/scorer.py`](../src/scoring/scorer.py). Every report shows the rubric version and the first 12
hex characters of the config file's SHA-256 (its *fingerprint*), so a score can be reproduced with exactly
the same configuration.

## 1. What the score means

The **rubric alignment score (0–100)** summarises how much of a job description's *stated* requirements is
*evidenced in the text of the submitted resume*, under the rules below.

It is **not**:

- a probability of being hired, or of anything else (nothing is calibrated);
- a model accuracy;
- a recruiter's judgment. The dataset used for the experimental models has algorithmically generated labels;
  no recruiter has validated this rubric.

"Not detected" means the resume text does not show an item, not that the candidate lacks it.

## 2. Formula

For each **applicable** component *c*, a value v_c ∈ [0, 1] and a weight w_c are combined:

```
uncapped = 100 × Σ_c (w_c × v_c) / Σ_c w_c        (sum over applicable components only)

score    = min(uncapped, 69)   if  any must-have requirement is not met
                               or  required-skill value < 0.50
         = uncapped            otherwise
```

The final score is rounded to one decimal place. The bands are: ≥ 70 *Strong alignment*, ≥ 45 *Partial
alignment*, otherwise *Limited alignment*. No score is shown (*Insufficient evidence*) if fewer than 40
words of resume text are usable, or if the job has no scorable requirement.

`recompute_score(report)` reproduces the displayed score from the report's own component values, and a
test checks this for every fixture.

### Components and default weights

| Component | Weight | Applicable when | Value |
|---|---|---|---|
| Required skills | 0.45 | job lists ≥ 1 required skill | mean credit over required skills |
| Experience | 0.15 | job states a minimum number of years | min(1, detected years / required years) |
| Education | 0.10 | job states a required degree level | see §3.3 |
| Required certifications | 0.05 | job lists a required certification | mean credit (1 if detected, else 0) |
| Preferred qualifications | 0.10 | job lists any preferred item | mean credit over *all* preferred items |
| Contextual relevance | 0.15 | semantic model enabled and available | rescaled cosine similarity, see §3.5 |

When a component is not applicable, its weight is redistributed proportionally. For example, a job with no
certification requirement does not lower anyone's score for lacking a certification. The report lists the
*effective* weight of each component and its contribution in points.

## 3. Component rules

### 3.1 Skills (required and preferred)

Each requirement is a list of acceptable alternatives (`Power BI or Tableau`). It is evaluated once,
however many times it is mentioned, so repeating a skill cannot raise the score.

| Evidence level | Credit | Rule |
|---|---|---|
| applied | 1.00 | mentioned in an experience, internship or project section, or in the same clause as an action verb ("built", "deployed", "analysed" …, list in config) |
| listed | 0.75 | mentioned only in a skills list, summary or other section |
| contradicted | 0.00 | a negation cue ("no", "not", "never", "without", "lack of", "no experience with" …) precedes the skill within 4 tokens in the same clause |
| not detected | 0.00 | no mention |

Mentions inside a quoted job description or requirement list pasted into the resume are ignored.

**Why projects and internships are not separate scored components:** a project that uses Python is evidence
*about Python*. Scoring it again as "project alignment" would count the same evidence twice. Projects and
internships therefore raise a skill from *listed* to *applied*, and internship durations count toward
experience. They are displayed with their matched skills, but contribute only through those two rules.

**Matching rules** (`src/skills/skill_extractor.py`, shared by training and inference):

- A curated dictionary (`data/skills/skill_dictionary.json`, ~230 technical skills, professional skills and
  certifications) maps aliases to canonical names (`postgres` → PostgreSQL).
- Ambiguous abbreviations are case-sensitive: `ML`, `DL`, `CV`, `JS`, `TS` and `Excel` must appear exactly so
  written. A bare `R`, `C` or `Go` is never accepted; only `R programming language`, `C programming` or
  `Golang` count.
- Matches need non-alphanumeric boundaries: `Java` ≠ `JavaScript`, `SQL` ≠ `NoSQL`.
- Requirements not in the dictionary (e.g. `menu costing`) are matched as literal phrases, tolerant of
  hyphens and a plural *s*.

### 3.2 Experience

Date ranges (`Jan 2020 – Mar 2022`, `06/2019 - Present`, `2018 - 2021`) are read from experience and
internship sections. Each range starts an entry. An entry counts only if it mentions a job skill or a
title word, so a 10-year unrelated career does not satisfy "2 years as a data analyst". Overlapping periods
are merged. Year-only ranges are counted from January. "Present" means the analysis date, which is
recorded in the report. If no dated entry exists, a stated figure ("5 years of experience") is used and
flagged as not date-verified. Internships count by default (`experience.count_internships`).

### 3.3 Education

Degree levels are recognised with word boundaries (Bachelor's / B.S. / B.Tech / B.E. / BCA …, Master's /
M.S. / M.Tech / MBA …, PhD; "Scrum Master" and "MS Office" are not degrees). Credit:

- 1.0: highest detected degree ≥ required level, and field matched (or no field required);
- × 0.5: the field is not recognised (the job's "or related field" cannot be verified by keyword);
- × 0.5: the degree appears in progress ("expected 2027", "pursuing");
- 0: below the required level, or no degree detected.

For must-have gating, a degree at the right level counts as met if the job accepts related fields.

### 3.4 Certifications

Certifications come from the dictionary's `certification` entries. Alternatives written together in the
job ("SHRM-CP, PHR or SPHR") form one any-of requirement.

### 3.5 Contextual relevance (semantic similarity)

The cosine similarity of `all-MiniLM-L6-v2` embeddings of the PII-redacted resume and the job text is
rescaled linearly from [0.23, 0.70] to [0, 1] and clipped. The bounds are the 5th and 95th percentiles over
the 2,383 de-duplicated dataset pairs, computed without labels (`reports/results/v2/semantic_bounds.json`).
The weight is capped at 15% so generic topical similarity cannot compensate for missing requirements. If
the model cannot be loaded, the component is dropped, the remaining weights are renormalised and the
report says so.

## 4. Must-have gating

A requirement is a **must-have** when the job marks it as such ("must", "mandatory", "required",
"minimum", or a *Must have* section), or when the user ticks *Must-have* in the review table. Experience
and education requirements from a requirements section default to must-have. If a must-have has credit
below 0.75, or required-skill evidence is below 50%, the score is capped at 69. That keeps the result
below the *Strong* band however much generic text similarity it has. The report shows both numbers and
the reason.

## 5. Why these weights?

They are a documented initial judgment, not fitted to any labels, and are not tuned to make an example
score look plausible:

- Explicitly required skills are what the job says it needs, and are the most reliably extractable signal,
  so they carry the most weight.
- Experience and education are standard screening criteria but are extracted less reliably from free text,
  so they get smaller weights.
- Preferred items should help, but missing them should cost little.
- Semantic similarity catches paraphrases the dictionary misses, but it is also high for any two documents
  on the same topic, so it is limited.

Changing a weight changes the fingerprint shown on every report. `with_overrides()` supports sensitivity
checks.

## 6. Evidence statuses

| Status | Meaning |
|---|---|
| ✅ Detected | the resume text supports it (evidence quoted) |
| 🟡 Partial / ambiguous | incomplete: fewer years than required, field unverified, degree in progress, dates unreadable |
| ⚪ Not detected | not found in the resume text; **not** proof of absence |
| ⛔ Contradicted | the resume itself states a lack |

## 7. Known limitations

- Rule-based parsing: multi-column PDFs, tables, images and unusual headers can be mis-sectioned. Scanned
  PDFs are not OCR'd.
- Negation handling covers short preceding cues only. "Python: none" and double negatives are missed.
- Synonyms not in the dictionary are only partly covered (by the semantic component).
- Requirement extraction from unstructured job text is a heuristic, which is why the app asks the user to
  review and correct the list before scoring.
- The weights have not been validated against human relevance judgments.
