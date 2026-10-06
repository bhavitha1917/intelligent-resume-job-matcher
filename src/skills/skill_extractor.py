"""Deterministic, dictionary-based skill extraction.

Approach (spec.md section 12): curated vocabulary + normalized aliases +
word-boundary-aware phrase matching. No ML model is used for extraction so
behavior is fully testable and explainable.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List

from src.utils.paths import SKILL_DICTIONARY_PATH

# Characters that do NOT count as "word" characters for boundary purposes,
# but which must be allowed to appear *inside* an alias (C++, C#, .NET, ...).
_WORD_CHAR_RE = re.compile(r"[A-Za-z0-9]")


def _is_word_char(ch: str) -> bool:
    return bool(ch) and bool(_WORD_CHAR_RE.match(ch))


class SkillExtractor:
    """Loads the skill dictionary once and extracts canonical skill names
    from free text using longest-alias-first, non-overlapping matching.
    """

    def __init__(self, dictionary_path: Path = SKILL_DICTIONARY_PATH):
        self.dictionary_path = Path(dictionary_path)
        self._alias_to_canonical: dict[str, str] = {}
        self._patterns: list[tuple[re.Pattern, str]] = []
        self._load()

    def _load(self) -> None:
        with open(self.dictionary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = []
        for item in data["skills"]:
            canonical = item["canonical"]
            for alias in item["aliases"]:
                alias_norm = alias.strip().lower()
                if not alias_norm:
                    continue
                self._alias_to_canonical[alias_norm] = canonical
                entries.append((alias_norm, canonical))

        # Longest alias first so multi-word aliases (e.g. "machine learning")
        # are matched before any shorter overlapping alias could consume part
        # of the same span.
        entries.sort(key=lambda pair: len(pair[0]), reverse=True)

        for alias_norm, canonical in entries:
            escaped = re.escape(alias_norm)
            # Custom boundary: previous/next character (if any) must not be
            # alphanumeric. This correctly handles aliases that themselves
            # start/end with punctuation (c++, c#, .net) where \b would fail,
            # while still preventing "javascript" from matching inside
            # "hyperjavascriptx".
            pattern = re.compile(
                r"(?<![A-Za-z0-9])" + escaped + r"(?![A-Za-z0-9])",
                re.IGNORECASE,
            )
            self._patterns.append((pattern, canonical))

    def extract_skills(self, text: str) -> List[str]:
        """Return the sorted, de-duplicated list of canonical skills found
        in `text`. Matching is case-insensitive; already-consumed spans are
        not re-matched by shorter/overlapping aliases.
        """
        if not text:
            return []

        found: dict[str, None] = {}
        consumed = [False] * len(text)

        for pattern, canonical in self._patterns:
            if canonical in found:
                continue
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(consumed[start:end]):
                    continue
                for i in range(start, end):
                    consumed[i] = True
                found[canonical] = None
                break  # one hit is enough to mark this skill as present

        return sorted(found.keys())


@lru_cache(maxsize=1)
def get_default_extractor() -> SkillExtractor:
    """Module-level cached extractor so the dictionary is parsed only once."""
    return SkillExtractor()


def extract_skills(text: str) -> List[str]:
    """Convenience function used throughout the project."""
    return get_default_extractor().extract_skills(text)
