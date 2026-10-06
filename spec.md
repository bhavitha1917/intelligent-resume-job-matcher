# Intelligent Resume–Job Matching, Candidate Ranking and Skill Gap Analysis using NLP

## 1. Project Overview

Build a complete end-to-end Natural Language Processing and Machine Learning project for intelligent resume–job matching.

The system should accept a job description and one or more resumes, calculate how well each candidate matches the job, rank candidates, identify matched and missing skills, and provide interpretable reasons for the ranking.

The project must demonstrate:

- Exploratory Data Analysis
- Data preprocessing
- Natural Language Processing
- Feature engineering
- Classical machine learning
- Semantic text similarity
- Candidate ranking
- Skill extraction
- Model evaluation
- Explainability
- Web application development
- Deployment

The system must NOT make autonomous hiring decisions. It should be presented as a candidate-job relevance and decision-support system.

---

# 2. Final Project Title

Preferred title:

"Intelligent Resume–Job Matching, Candidate Ranking and Skill Gap Analysis using NLP"

Alternative shorter title:

"Intelligent Resume–Job Matching System"

---

# 3. Main Research Question

The main research question is:

"Can combining lexical similarity, semantic similarity, skill matching, and candidate-specific features improve resume–job matching and candidate ranking compared with simple keyword-based matching?"

The project must experimentally compare multiple approaches rather than presenting only one model.

---

# 4. Dataset Requirements

## 4.1 Primary Resume Dataset

Use the Kaggle Resume Dataset:

https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

Expected information includes:

- ID
- Resume_str
- Resume_html
- Category

The dataset contains more than 2,400 resumes across multiple professional categories.

Use it primarily for:

- resume corpus analysis
- resume preprocessing
- resume category analysis
- skill extraction
- PDF/text parsing validation
- application testing

Do not upload the raw dataset to GitHub.

---

## 4.2 Primary Job Dataset

Use the Kaggle LinkedIn Job Postings 2023–2024 dataset:

https://www.kaggle.com/datasets/arshkon/linkedin-job-postings-2023-2024

Use relevant fields such as:

- job_id
- title
- description
- formatted_experience_level
- location
- work type
- skills_desc
- company_id
- other relevant job metadata

Do not require the entire dataset to be loaded into memory.

Create a preprocessing pipeline that can sample/filter relevant records.

Do not upload the raw dataset to GitHub.

---

## 4.3 Paired Resume–Job Matching Dataset

Use:

https://huggingface.co/datasets/batuhanmtl/job_resume_fit

This dataset contains paired resume/job examples and matching-related scores.

Use it primarily for:

- supervised matching experiments
- feature engineering
- model training
- model comparison
- validation of matching approaches

Important:

The existing matching scores should not automatically be treated as objective human hiring decisions.

Document clearly that they are dataset-provided matching signals.

---

# 5. Dataset Download Policy

The project must NOT automatically scrape websites.

The system should expect datasets to be downloaded separately by the user.

Provide:

- `data/README.md`
- clear dataset download instructions
- expected filenames
- expected folder structure
- preprocessing instructions

Raw datasets must be excluded from Git using `.gitignore`.

The project should still run using a small sample/demo dataset if the full datasets are unavailable.

## Current Verified Dataset Structure

### Kaggle Resume Dataset

Local path:

`data/raw/Resume/Resume.csv`

Verified shape:

- 2484 rows
- 4 columns

Columns:

- ID
- Resume_str
- Resume_html
- Category

### Hugging Face Resume–Job Fit Dataset

Dataset:

`batuhanmtl/job_resume_fit`

Verified split:

`train`

Verified size:

- 2385 rows
- 10 columns

Columns:

- ID
- resume_text
- job_text
- category
- job_required_skills
- resume_skill_list
- ai_matched_skills
- ai_match_score
- skill_string_match_score
- fuzzy_match_score

The implementation must inspect and validate the actual dataset schema programmatically rather than assuming a schema.

---

# 6. Problem Formulation

Given:

- one job description
- N candidate resumes

the system should produce:

1. Overall candidate-job match score
2. Candidate ranking
3. Matched skills
4. Missing skills
5. Semantic similarity
6. Skill overlap
7. Experience relevance where available
8. Explanation of major matching factors

Example:

Candidate 1
- Match score: 87%
- Rank: 1
- Matched skills: Python, SQL, Machine Learning, PyTorch
- Missing skills: AWS, Docker
- Semantic similarity: 0.89
- Skill coverage: 0.83

---

# 7. System Architecture

The overall architecture should be:

Resume PDF/Text
        |
        v
Resume Parser
        |
        v
Text Preprocessing
        |
        +--------------------+
        |                    |
        v                    v
Skill Extraction       Full Resume Text
        |                    |
        v                    v
Skill Features       Text Representation
        |                    |
        +---------+----------+
                  |
                  v
           Matching Engine
                  |
       +----------+----------+
       |          |           |
       v          v           v
 Keyword      TF-IDF      Semantic
 Matching     Similarity   Embedding
       |          |           |
       +----------+-----------+
                  |
                  v
          Feature Engineering
                  |
                  v
       Supervised ML Models
                  |
                  v
          Candidate Ranking
                  |
                  v
       Skill Gap + Explanation
                  |
                  v
          Streamlit Web App

---

# 8. Development Stages

## Stage 1 — Data Exploration

Perform EDA on:

### Resume dataset

Analyze:

- number of resumes
- category distribution
- resume length
- token counts
- common skills/terms
- category-specific vocabulary
- missing values
- duplicate records

### Job dataset

Analyze:

- number of jobs
- job categories
- job title distribution
- experience-level distribution
- description length
- common skills
- most frequently requested technologies

### Resume-job fit dataset

Analyze:

- number of pairs
- category distribution
- matching score distributions
- skill matching distributions
- relationship between different matching scores

Save all important figures in:

`reports/figures/`

---

# 9. Text Preprocessing

Implement reusable preprocessing functions.

Requirements:

- Unicode normalization
- whitespace normalization
- optional lowercasing
- removal of unnecessary formatting
- preservation of technical tokens
- preservation of terms such as:
  - C++
  - C#
  - .NET
  - AWS
  - SQL
  - PyTorch
  - TensorFlow
  - Node.js

Do NOT blindly remove punctuation because technical skill names often contain punctuation.

Use section-aware processing where useful:

- education
- experience
- skills
- projects
- certifications

---

# 10. Resume Parsing

The application must support:

- `.pdf`
- `.txt`

PDF extraction should use PyMuPDF or an equivalent lightweight reliable parser.

Extract:

- full text
- skills
- education section where detectable
- experience section where detectable
- projects where detectable
- certifications where detectable

The parser should degrade gracefully when a section cannot be identified.

Do not require OCR unless explicitly needed.

---

# 11. Job Description Parsing

Extract:

- title
- description
- required skills
- preferred skills where available
- experience requirements
- education requirements where detectable

The system should support both structured job records and raw job-description text.

---

# 12. Skill Extraction

Create a reusable skill extraction module.

Initial approach:

- curated skill vocabulary
- normalized aliases
- phrase matching
- regex where appropriate

Support aliases such as:

Python / python programming

Machine Learning / ML

Deep Learning / DL

Natural Language Processing / NLP

Scikit-learn / sklearn

PyTorch / torch

PostgreSQL / Postgres

JavaScript / JS

Use word-boundary-aware matching to reduce false positives.

Do not count generic words such as "data" or "experience" as technical skills unless explicitly defined.

The skill vocabulary should be stored separately, for example:

`data/skills/skill_dictionary.json`

Make it easy to extend.

---

# 13. Baseline 1 — Keyword Matching

Implement a simple baseline.

For a job requiring:

Python, SQL, Docker, AWS, PyTorch

and a resume containing:

Python, SQL, PyTorch

calculate:

skill coverage = matched required skills / total required skills

This is the simplest baseline and must be included for comparison.

---

# 14. Baseline 2 — TF-IDF + Cosine Similarity

Implement:

- TF-IDF vectorization
- cosine similarity

Compute similarity between:

- complete resume text
- complete job-description text

Also experiment with section-weighted text if practical.

Store the resulting similarity score.

---

# 15. Model 3 — Sentence Transformer Semantic Similarity

Use a pretrained Sentence Transformer.

Preferred initial model:

`all-MiniLM-L6-v2`

Do NOT train a transformer from scratch.

Generate embeddings for:

- resume
- job description

Calculate cosine similarity.

Store:

`semantic_similarity`

The model should be downloaded/cached rather than bundled into the Git repository.

---

# 16. Feature Engineering

Create a structured feature vector containing useful matching signals.

Possible features:

- skill coverage
- number of matched skills
- number of missing skills
- TF-IDF cosine similarity
- semantic similarity
- resume length
- job description length
- experience keyword overlap
- education keyword overlap
- title/category compatibility
- required-skill coverage
- preferred-skill coverage

Avoid features that encode protected characteristics or obvious PII.

Do NOT use:

- name
- gender
- age
- photograph
- email
- phone number
- address
- nationality
- religion
- caste
- marital status
- other protected personal attributes

for ranking.

---

# 17. Supervised ML Models

Train and compare:

1. Logistic Regression
2. Random Forest
3. SVM

Optional:

4. Gradient Boosting / XGBoost if useful

The primary supervised models should operate on engineered matching features.

Do not use large deep neural networks unnecessarily.

---

# 18. Target Definition

Where the paired dataset provides a matching score, define a transparent target strategy.

Possible formulation:

Strong Match:
score >= high threshold

Moderate Match:
middle range

Weak Match:
low range

The thresholds must be documented and configurable.

Also experiment with regression if the matching score is sufficiently continuous.

Do NOT claim that these labels represent actual recruiter decisions.

---

# 19. Model Comparison

Compare:

1. Keyword baseline
2. TF-IDF similarity
3. Sentence Transformer similarity
4. Logistic Regression
5. Random Forest
6. SVM
7. Final hybrid approach

Create a comparison table containing:

- accuracy where applicable
- precision
- recall
- F1-score
- ROC-AUC where applicable
- ranking metrics where applicable
- inference time
- model complexity

Use appropriate metrics rather than accuracy alone.

---

# 20. Ranking Evaluation

Because the final application ranks candidates, include ranking metrics where possible.

Evaluate:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- NDCG@K

Explain each metric in the report.

Do not rely only on classification accuracy.

---

# 21. Final Hybrid Matching Score

Create a final hybrid scoring system.

Possible components:

- skill match
- TF-IDF similarity
- semantic similarity
- experience compatibility
- education compatibility

Do not hardcode arbitrary weights without documenting them.

Compare:

1. manually weighted baseline
2. learned supervised model

Select the final approach based on validation performance and interpretability.

---

# 22. Skill Gap Analysis

For each candidate, show:

### Matched skills

Skills required by the job that appear in the resume.

### Missing skills

Important job requirements not found in the resume.

### Optional skills

Preferred skills that are present/absent.

Example:

Matched:
- Python
- SQL
- PyTorch

Missing:
- AWS
- Docker

The system must clearly distinguish:

"not detected in resume"

from

"candidate definitely does not possess the skill."

Use the wording:

"Not detected in the submitted resume."

---

# 23. Explainability

Every final candidate result should provide an explanation.

Example:

Match score: 87%

Positive factors:
- High required-skill coverage
- Strong semantic similarity
- Relevant experience terms

Missing requirements:
- AWS
- Docker

The explanation should be based on actual calculated features.

Do not generate fabricated explanations.

---

# 24. Candidate Ranking

For one job and multiple resumes:

Input:

- Job description
- N resumes

Output:

| Rank | Candidate | Match Score | Skill Match | Semantic Match |
|------|-----------|-------------|-------------|----------------|
| 1 | Candidate A | 91% | 95% | 0.91 |
| 2 | Candidate C | 84% | 80% | 0.87 |
| 3 | Candidate B | 76% | 72% | 0.78 |

Clicking a candidate should show detailed analysis.

---

# 25. Web Application

Use Streamlit.

The application must provide:

## Page 1 — Home

Explain:

- project objective
- how the system works
- limitations
- ethical considerations

## Page 2 — Single Resume Analysis

Upload:

- one resume
- one job description

Display:

- match score
- matched skills
- missing skills
- semantic similarity
- explanation

## Page 3 — Candidate Ranking

Upload:

- one job description
- multiple resumes

Display:

- ranked candidate table
- filters
- sortable scores

## Page 4 — Detailed Candidate Analysis

Show:

- resume summary
- match score
- skill coverage
- semantic score
- matched skills
- missing skills
- explanation

## Page 5 — Model Comparison

Display experimental results:

- baseline
- TF-IDF
- Sentence Transformer
- supervised models
- final model

Use charts/tables.

## Page 6 — About / Methodology

Explain:

- datasets
- preprocessing
- models
- evaluation
- limitations
- ethical considerations

---

# 26. Application Design

Keep the UI clean and academic.

Do not build an overly complicated enterprise dashboard.

Use:

- clear cards
- tables
- progress indicators
- charts
- readable typography
- consistent layout

The app should work on desktop and reasonably on smaller screens.

---

# 27. Privacy and Fairness

This is a recruitment-related system and therefore fairness is an important academic component.

The system must:

- remove or ignore PII for model scoring
- never rank candidates using names, gender, age, photograph, email, phone number, address, religion, caste, nationality or similar protected information
- document dataset bias
- explain that resume datasets may not represent the real hiring population
- clearly state that the system is decision support and not an autonomous hiring system

If suitable attributes exist in a dataset, perform a basic fairness/bias discussion.

Do not attempt to infer protected attributes from names or resumes.

---

# 28. Data Leakage Prevention

This is mandatory.

Ensure:

- train/validation/test splits are performed before fitting vectorizers/models
- TF-IDF is fitted only on training data
- supervised models never see test labels
- embeddings do not leak target information
- duplicate or near-duplicate resumes are not present across train and test where detectable

Document all splitting decisions.

---

# 29. Reproducibility

Set random seeds where applicable.

Save:

- preprocessing configuration
- model configuration
- trained model files
- vectorizers
- evaluation results

Use configuration files rather than hardcoded paths.

---

# 30. Model Saving

Save final artifacts under:

`models/`

Examples:

- TF-IDF vectorizer
- Logistic Regression model
- Random Forest model
- SVM model
- skill dictionary
- configuration

Do not commit huge pretrained transformer model files to GitHub.

Download/cache them during setup when appropriate.

---

# 31. Testing

Create unit tests for:

- text cleaning
- skill extraction
- PDF parsing
- TF-IDF similarity
- semantic similarity
- score calculation
- missing-skill detection
- ranking
- malformed input handling

The application must not crash when:

- PDF extraction fails
- resume has very little text
- job description is empty
- no skills are detected
- a resume has unusual formatting

Display useful error messages.

---

# 32. Project Documentation

Create a high-quality README containing:

1. Project overview
2. Problem statement
3. Objectives
4. Dataset sources
5. Dataset setup
6. Architecture
7. Methodology
8. Models
9. Evaluation
10. Results
11. Screenshots
12. Local setup
13. Running the application
14. Deployment
15. Limitations
16. Ethical considerations
17. Future work

---

# 33. Research Report Outputs

Generate reproducible outputs for:

- dataset statistics
- category distribution
- common skills
- resume length distribution
- job description length distribution
- similarity distributions
- model comparison
- confusion matrices
- ROC curves where appropriate
- ranking metrics
- feature importance

Save figures under:

`reports/figures/`

Save numerical results under:

`reports/results/`

---

# 34. Deployment

Deploy the final Streamlit application.

Preferred deployment:

Streamlit Community Cloud using GitHub.

The repository must contain:

- `app.py` or correct Streamlit entry point
- `requirements.txt`
- `.gitignore`
- README
- model-loading code
- deployment instructions

Do not require raw datasets to be uploaded to GitHub.

The deployed application should use:

- a small bundled demo dataset OR
- user-uploaded resumes and job descriptions

for interactive inference.

Large training datasets should not be loaded during application startup.

---

# 35. Deployment Architecture

Training:

Local environment / notebook
        ↓
Dataset preprocessing
        ↓
Model training
        ↓
Saved model artifacts
        ↓
GitHub repository
        ↓
Streamlit deployment
        ↓
User uploads resume + JD
        ↓
Inference
        ↓
Results

The deployed app should NOT retrain models.

---

# 36. Deployment Requirements

The final deployment must:

- start successfully from a clean environment
- install dependencies from requirements.txt
- load trained artifacts
- load/call the sentence-transformer model appropriately
- accept PDF resumes
- accept text job descriptions
- produce a match score
- produce skill-gap analysis
- rank multiple candidates
- handle invalid input gracefully

---

# 37. GitHub Requirements

Create a clean Git repository.

Never commit:

- raw datasets
- API keys
- passwords
- `.env`
- personal resumes
- personal information
- huge model checkpoints
- cache directories

Include:

`.gitignore`

and:

`.env.example`

if environment variables are required.

---

# 38. Final Deliverables

The completed project must contain:

1. `spec.md`
2. source code
3. notebooks
4. preprocessing pipeline
5. trained models
6. evaluation results
7. figures
8. Streamlit application
9. README
10. requirements.txt
11. tests
12. deployment configuration
13. GitHub repository
14. deployed application URL

---

# 39. Scope Control

This is a solo undergraduate semester project.

Do NOT unnecessarily implement:

- custom large language models
- model pretraining
- custom transformer architecture
- distributed training
- microservices
- Kubernetes
- complex databases
- authentication
- payment systems
- enterprise infrastructure

Prefer a clean, reproducible and academically rigorous implementation.

If an advanced feature would significantly increase complexity without improving the academic objective, make it optional.

---

# 40. Development Priority

Implement in this exact order:

Phase 1:
Dataset loading and EDA

Phase 2:
Text preprocessing

Phase 3:
Skill extraction

Phase 4:
Keyword baseline

Phase 5:
TF-IDF similarity

Phase 6:
Sentence Transformer similarity

Phase 7:
Supervised matching models

Phase 8:
Model evaluation

Phase 9:
Hybrid ranking

Phase 10:
Skill-gap analysis

Phase 11:
Explainability

Phase 12:
Streamlit application

Phase 13:
Testing

Phase 14:
Documentation

Phase 15:
GitHub preparation

Phase 16:
Deployment

Do not jump to deployment before the core pipeline is tested.

---

# 41. Definition of Done

The project is complete only when:

- datasets can be loaded reproducibly
- preprocessing works
- skills are extracted
- baseline is implemented
- TF-IDF model works
- semantic model works
- supervised models work
- models are compared
- candidate ranking works
- skill-gap analysis works
- explanations are generated from actual features
- Streamlit app works locally
- tests pass
- GitHub repository is clean
- deployment succeeds
- deployed application can perform inference
- README explains the entire system
- limitations and ethical considerations are documented

The final result should look like a complete academic Data Science/NLP project, not merely a notebook or keyword-matching script.