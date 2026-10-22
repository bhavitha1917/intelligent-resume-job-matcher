"""Central, relative path configuration.

All other modules import paths from here instead of hardcoding strings, so
the project can be moved/cloned anywhere (including Streamlit Cloud) without
code changes.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
DEMO_DATA_DIR = EXTERNAL_DATA_DIR / "demo"
SKILLS_DIR = DATA_DIR / "skills"
SKILL_DICTIONARY_PATH = SKILLS_DIR / "skill_dictionary.json"
CONFIG_DIR = PROJECT_ROOT / "config"
SCORING_RUBRIC_PATH = CONFIG_DIR / "scoring_rubric.json"

RESUME_CSV_PATH = RAW_DATA_DIR / "Resume" / "Resume.csv"
RESUME_PDF_ROOT = RAW_DATA_DIR / "data" / "data"
LINKEDIN_POSTINGS_PATH = RAW_DATA_DIR / "linkedin_job_postings" / "postings.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
RESULTS_V2_DIR = RESULTS_DIR / "v2"
MODELS_V2_DIR = MODELS_DIR / "v2"


def require_path(path: Path, hint: str) -> Path:
    """Raise a clear, actionable error if an expected dataset file is missing.

    Used instead of a bare FileNotFoundError so the user is told exactly
    where to put the file (per spec.md section 5: no auto-scraping, the
    system must explain where datasets are expected).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found: {path}\n{hint}\n"
            f"See data/README.md for download and placement instructions."
        )
    return path
