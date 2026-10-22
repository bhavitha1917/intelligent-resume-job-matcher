"""Deterministic, dictionary-based skill extraction.

Approach (spec.md section 12): curated vocabulary + normalized aliases +
word-boundary-aware phrase matching. No ML model is used for extraction so
behavior is fully testable and explainable.

This module is the single source of truth for skill normalization and
matching. Training (src/models/train_supervised.py) and inference
(src/scoring/) both go through `canonicalize_skill` and `phrase_pattern`
so a required skill is matched identically in both paths.

Conventions:
- Matching is case-insensitive, except for `case_sensitive_aliases` in the
  dictionary (ambiguous abbreviations such as "ML", "CV", "TS", "Excel"
  which are only accepted when written exactly that way).
- A match must not be preceded or followed by a letter/digit, so "R" never
  matches inside "React" and "Java" never matches inside "JavaScript".
- Hyphens and runs of whitespace inside a phrase are interchangeable
  ("problem-solving" == "problem solving").
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from src.utils.paths import SKILL_DICTIONARY_PATH

_BOUNDARY_BEFORE = r"(?<![A-Za-z0-9])"
_BOUNDARY_AFTER = r"(?![A-Za-z0-9])"
_SEPARATOR_RE = re.compile(r"[\s\-]+")


@dataclass(frozen=True)
class SkillMention:
    canonical: str
    start: int
    end: int
    matched_text: str


def normalize_phrase(text: str) -> str:
    """Lowercase, trim, unify hyphen/whitespace runs to a single space."""
    return _SEPARATOR_RE.sub(" ", (text or "").strip().lower()).strip(" .,;:")


@lru_cache(maxsize=8192)
def phrase_pattern(phrase: str, case_sensitive: bool = False, allow_plural: bool = False) -> re.Pattern:
    """Word-boundary-aware regex for a skill phrase.

    `allow_plural` lets the final word take an optional "s" (used for
    free-text requirement phrases such as "financial statement").
    """
    words = [re.escape(w) for w in _SEPARATOR_RE.split(phrase.strip()) if w]
    body = r"[\s\-]+".join(words)
    if allow_plural and words and words[-1][-1:].isalpha():
        body += r"s?"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(_BOUNDARY_BEFORE + body + _BOUNDARY_AFTER, flags)


class SkillExtractor:
    """Loads the skill dictionary once and extracts canonical skill names
    from free text using longest-alias-first, non-overlapping matching.
    """

    def __init__(self, dictionary_path: Path = SKILL_DICTIONARY_PATH):
        self.dictionary_path = Path(dictionary_path)
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_lookup: dict[str, str] = {}
        self._canonical_aliases: dict[str, list[tuple[str, bool]]] = {}
        self._types: dict[str, str] = {}
        self._patterns: list[tuple[re.Pattern, str]] = []
        self._load()

    def _load(self) -> None:
        with open(self.dictionary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = []
        for item in data["skills"]:
            canonical = item["canonical"]
            self._types[canonical] = item.get("type", "technical")
            self._canonical_lookup[normalize_phrase(canonical)] = canonical
            aliases = self._canonical_aliases.setdefault(canonical, [])
            for alias in item.get("aliases", []):
                alias_norm = normalize_phrase(alias)
                if not alias_norm:
                    continue
                self._alias_to_canonical[alias_norm] = canonical
                aliases.append((alias_norm, False))
                entries.append((alias_norm, canonical, False))
            for alias in item.get("case_sensitive_aliases", []):
                aliases.append((alias, True))
                entries.append((alias, canonical, True))

        # Longest alias first so multi-word aliases (e.g. "machine learning")
        # are matched before any shorter overlapping alias could consume part
        # of the same span.
        entries.sort(key=lambda e: len(e[0]), reverse=True)
        self._patterns = [(phrase_pattern(alias, case_sensitive=cs), canonical) for alias, canonical, cs in entries]

    def skill_type(self, canonical: str) -> Optional[str]:
        return self._types.get(canonical)

    def canonicalize(self, name: str) -> Optional[str]:
        """Dictionary canonical name for `name` (alias or canonical
        spelling, case/hyphen-insensitive), or None if not in the dictionary."""
        key = normalize_phrase(name)
        return self._alias_to_canonical.get(key) or self._canonical_lookup.get(key)

    def aliases_for(self, canonical: str) -> list[tuple[str, bool]]:
        return list(self._canonical_aliases.get(canonical, []))

    def find_mentions(self, text: str) -> List[SkillMention]:
        """All non-overlapping dictionary skill mentions, in text order.
        Unlike `extract_skills`, repeated mentions are all returned so
        callers can inspect the surrounding context of each one.
        """
        if not text:
            return []
        consumed = [False] * len(text)
        mentions = []
        for pattern, canonical in self._patterns:
            for match in pattern.finditer(text):
                start, end = match.span()
                if any(consumed[start:end]):
                    continue
                for i in range(start, end):
                    consumed[i] = True
                mentions.append(SkillMention(canonical, start, end, match.group(0)))
        return sorted(mentions, key=lambda m: m.start)

    def extract_skills(self, text: str) -> List[str]:
        """Return the sorted, de-duplicated list of canonical skills found
        in `text`."""
        return sorted({m.canonical for m in self.find_mentions(text)})


@lru_cache(maxsize=1)
def get_default_extractor() -> SkillExtractor:
    """Module-level cached extractor so the dictionary is parsed only once."""
    return SkillExtractor()


def extract_skills(text: str) -> List[str]:
    """Convenience function used throughout the project."""
    return get_default_extractor().extract_skills(text)


def canonicalize_skill(name: str) -> str:
    """Canonical display form of a skill name.

    Dictionary skills map to their canonical name ("ml" -> "Machine
    Learning" only via the exact case-sensitive alias, "postgres" ->
    "PostgreSQL"). Anything not in the dictionary is kept as a free-text
    requirement phrase, normalized to lowercase with single spaces, so it
    can still be matched literally against resume text.
    """
    stripped = (name or "").strip()
    canonical = get_default_extractor().canonicalize(stripped) or _case_sensitive_lookup().get(stripped)
    return canonical or normalize_phrase(stripped)


@lru_cache(maxsize=1)
def _case_sensitive_lookup() -> dict:
    extractor = get_default_extractor()
    return {
        alias: canonical
        for canonical in extractor._canonical_aliases
        for alias, cs in extractor.aliases_for(canonical)
        if cs
    }


@lru_cache(maxsize=8192)
def skill_patterns(canonical_or_phrase: str) -> tuple:
    """Regex patterns that count as evidence for a (canonicalized) skill.
    Dictionary skills use exactly their dictionary aliases (the same
    patterns `find_mentions` uses, so a bare "R" or "C" is never accepted
    as evidence); free-text phrases match literally (plural-tolerant)."""
    aliases = get_default_extractor().aliases_for(canonical_or_phrase)
    if aliases:
        return tuple(phrase_pattern(a, case_sensitive=cs) for a, cs in aliases)
    return (phrase_pattern(canonical_or_phrase, allow_plural=True),)
