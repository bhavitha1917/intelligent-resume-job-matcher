from _fixtures import REFERENCE_DATE, RESUME_STRONG, job, parsed

from src.scoring.privacy import sanitize_for_scoring
from src.scoring.scorer import score_candidate

PERSONAL = """Priya Example
priya.example@mail.com | +91 98765 43210 | linkedin.com/in/priya-example
Date of Birth: 01/01/2000
Gender: Female
Marital Status: Single
Nationality: Indian
Religion: Hindu
Caste: General
Address: 12 Example Street, Hyderabad
"""


def test_contact_and_protected_fields_are_removed():
    sanitized, removed = sanitize_for_scoring(PERSONAL + RESUME_STRONG)
    for leaked in ("priya.example@mail.com", "98765", "linkedin.com", "Female", "Single", "Indian", "Hindu",
                   "General", "Example Street", "01/01/2000"):
        assert leaked not in sanitized
    assert "e-mail" in removed and "phone number" in removed
    assert len(sanitized) == len(PERSONAL + RESUME_STRONG)  # offsets preserved


def test_date_ranges_are_not_mistaken_for_phone_numbers():
    sanitized, _ = sanitize_for_scoring("Analyst (2019 - 2022) and 2015-2019, Jan 2020 - Mar 2023")
    assert "2019 - 2022" in sanitized and "2015-2019" in sanitized


def test_privacy_sensitive_fields_do_not_reach_scoring_inputs_or_change_the_score():
    seen = []

    def spy(resume_text, job_text):
        seen.append(resume_text)
        return 0.5

    with_pii = score_candidate(job(), parsed(PERSONAL + RESUME_STRONG), semantic=spy, reference_date=REFERENCE_DATE)
    without = score_candidate(job(), parsed(RESUME_STRONG), semantic=spy, reference_date=REFERENCE_DATE)
    assert "Female" not in seen[0] and "Hindu" not in seen[0] and "@" not in seen[0]
    assert with_pii["score"] == without["score"]
    assert with_pii["components"] == without["components"]
