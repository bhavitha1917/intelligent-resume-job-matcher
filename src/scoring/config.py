"""Loads the centralized scoring rubric (config/scoring_rubric.json)."""

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from src.utils.paths import SCORING_RUBRIC_PATH

COMPONENT_ORDER = (
    "required_skills",
    "experience",
    "education",
    "certifications",
    "preferred_qualifications",
    "semantic_relevance",
)


@lru_cache(maxsize=4)
def _load(path_str: str) -> tuple[str, str]:
    # Normalise line endings so the fingerprint is identical on Windows (CRLF
    # checkout) and Linux / Streamlit Cloud (LF).
    raw = Path(path_str).read_bytes().replace(b"\r\n", b"\n")
    return raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()


def load_rubric(path: Optional[Path] = None) -> dict:
    """Return a fresh copy of the rubric with `_fingerprint` (first 12 hex
    chars of the file's SHA-256) added. Callers may modify the copy, e.g.
    in tests, without affecting other callers."""
    text, digest = _load(str(path or SCORING_RUBRIC_PATH))
    rubric = json.loads(text)
    validate_rubric(rubric)
    rubric["_fingerprint"] = digest[:12]
    return rubric


def validate_rubric(rubric: dict) -> None:
    weights = rubric["component_weights"]
    missing = set(COMPONENT_ORDER) - set(weights)
    if missing:
        raise ValueError(f"Rubric is missing weights for: {sorted(missing)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError("Rubric weights must be non-negative.")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"Rubric weights must sum to 1.0 (got {sum(weights.values())}).")
    low, high = rubric["semantic_normalization"]["low"], rubric["semantic_normalization"]["high"]
    if not low < high:
        raise ValueError("semantic_normalization.low must be < high.")


def with_overrides(rubric: dict, **weight_overrides) -> dict:
    """Copy of `rubric` with some component weights replaced (used by tests
    and sensitivity analysis; the result is re-validated)."""
    updated = copy.deepcopy(rubric)
    updated["component_weights"].update(weight_overrides)
    validate_rubric(updated)
    updated["_fingerprint"] = "custom-" + hashlib.sha256(
        json.dumps({k: v for k, v in updated.items() if k != "_fingerprint"}, sort_keys=True).encode()
    ).hexdigest()[:8]
    return updated
