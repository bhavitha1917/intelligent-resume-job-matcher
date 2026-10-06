"""Reusable text-cleaning functions shared by resume and job-description
preprocessing.

Design constraint (spec.md section 9): technical tokens such as C++, C#,
.NET, AWS, SQL, PyTorch, TensorFlow, Node.js must survive cleaning, so we do
NOT blindly strip punctuation. Only whitespace/HTML noise is normalized.
"""

import html
import re
import unicodedata

# Tokens whose internal punctuation must never be treated as noise.
# Used only to sanity-check cleaning behavior in tests, not for extraction.
PROTECTED_TOKEN_EXAMPLES = [
    "C++", "C#", ".NET", "AWS", "SQL", "PyTorch", "TensorFlow", "Node.js",
]

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")
_BULLET_RE = re.compile(r"^[•●▪\-\*]\s+", re.MULTILINE)


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities (used for Resume_html)."""
    if not text:
        return ""
    text = html.unescape(text)
    return _TAG_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """NFKC-normalize unicode (curly quotes, ligatures, full-width chars)."""
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs and excessive blank lines.

    Newlines are preserved (single) so section-aware splitting downstream
    can still rely on line boundaries.
    """
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line != "") if lines else text


def strip_bullets(text: str) -> str:
    """Remove leading bullet glyphs ('-', '*', bullet chars) but keep the
    rest of the line (including any internal punctuation like C++)."""
    return _BULLET_RE.sub("", text)


def clean_text(text: str, lowercase: bool = False) -> str:
    """Full cleaning pipeline: HTML -> unicode normalize -> bullets ->
    whitespace normalize -> optional lowercase.

    Punctuation is intentionally left untouched so tokens like C++, C#,
    .NET, Node.js remain intact for downstream skill extraction.
    """
    if text is None:
        return ""
    text = strip_html(text)
    text = normalize_unicode(text)
    text = strip_bullets(text)
    text = normalize_whitespace(text)
    if lowercase:
        text = text.lower()
    return text.strip()


def handle_missing(value, default: str = "") -> str:
    """Normalize missing/NaN-like values (used when cleaning DataFrame cells)."""
    if value is None:
        return default
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return default
    except TypeError:
        pass
    text = str(value)
    return default if text.strip().lower() in {"", "nan", "none"} else text
