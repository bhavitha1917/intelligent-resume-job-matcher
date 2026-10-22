"""Separate personal / protected information from scoring inputs.

`sanitize_for_scoring` returns a copy of the resume text in which contact
details and protected-attribute lines are blanked out (replaced by spaces
of the same length, so character offsets used for section detection stay
valid). Only the sanitized text is passed to skill matching, TF-IDF and
the sentence-transformer.

What is removed:
- e-mail addresses, phone numbers, URLs (incl. LinkedIn/GitHub profiles)
- lines that state date of birth, age, gender/sex, marital status,
  nationality, religion, caste/category, father's/mother's/spouse's name,
  photo, passport/Aadhaar/PAN/SSN numbers, or a postal address
- the whole "personal details" section, if one is detected

Limitations (documented, not hidden): the candidate's name is not reliably
detectable and is not removed from the text itself. It is never used as a
feature, but it is still part of the text the semantic model embeds. Free-
text mentions of protected characteristics elsewhere (e.g. in a summary
paragraph) are not detected.
"""

import re

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b(?:linkedin|github)\.com/\S+", re.IGNORECASE)
_PHONE_CANDIDATE_RE = re.compile(r"(?<![\w/])\+?\(?\d[\d \t().\-]{7,}\d(?![\w/])")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


class _PhonePattern:
    """Phone-number detector: 10-13 digits in one run of digits/separators,
    and not just a sequence of years (so "2019 - 2022" is never blanked)."""

    @staticmethod
    def finditer(text):
        for match in _PHONE_CANDIDATE_RE.finditer(text):
            digits = re.sub(r"\D", "", match.group(0))
            groups = re.findall(r"\d+", match.group(0))
            if not 10 <= len(digits) <= 13:
                continue
            if all(len(g) == 4 and _YEAR_RE.fullmatch(g) for g in groups):
                continue
            yield match


_PHONE_RE = _PhonePattern()

_PROTECTED_FIELD_LINE_RE = re.compile(
    r"^[ \t\-•*]*(?:date\s+of\s+birth|d\.?o\.?b\.?|birth\s*date|age|gender|sex|marital\s+status|"
    r"nationality|citizenship|religion|caste|category|father'?s\s+name|mother'?s\s+name|"
    r"husband'?s\s+name|wife'?s\s+name|spouse(?:'s)?\s+name|photo(?:graph)?|passport(?:\s+no\.?|\s+number)?|"
    r"aadhaa?r(?:\s+no\.?|\s+number)?|pan(?:\s+no\.?|\s+number)|ssn|social\s+security(?:\s+number)?|"
    r"(?:permanent\s+|current\s+|postal\s+)?address|place\s+of\s+birth|blood\s+group|height|weight)"
    r"\s*[:\-–]\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)

PROTECTED_FIELD_NAMES = (
    "date of birth / age", "gender", "marital status", "nationality", "religion", "caste",
    "family names", "photo", "identity numbers", "address", "e-mail", "phone", "profile URLs",
)


def _blank(text: str, start: int, end: int) -> str:
    segment = text[start:end]
    return text[:start] + re.sub(r"[^\n]", " ", segment) + text[end:]


def sanitize_for_scoring(text: str, section_spans=None) -> tuple[str, list[str]]:
    """Return (sanitized_text, removed_categories). Offsets are preserved."""
    if not text:
        return "", []
    removed: list[str] = []
    sanitized = text

    for field, start, end in section_spans or []:
        if field == "personal":
            sanitized = _blank(sanitized, start, end)
            removed.append("personal details section")

    for name, pattern in (("protected/personal field line", _PROTECTED_FIELD_LINE_RE),
                          ("e-mail", _EMAIL_RE), ("URL", _URL_RE), ("phone number", _PHONE_RE)):
        matches = list(pattern.finditer(sanitized))
        for match in reversed(matches):
            sanitized = _blank(sanitized, match.start(), match.end())
        if matches:
            removed.append(name)

    return sanitized, removed
