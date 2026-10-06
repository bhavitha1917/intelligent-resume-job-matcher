"""Resume parsing for .pdf and .txt files.

Returns a structured dict and NEVER raises on malformed input — callers
(tests, Streamlit app) always get a dict back, with `parse_warnings`
explaining anything that went wrong or any section that could not be
detected (spec.md section 10: "degrade gracefully").
"""

import re
from pathlib import Path
from typing import Optional, Union

# Section headers we look for, in a fixed canonical order. Keys are the
# output field names; values are the header spellings we recognize.
_SECTION_ALIASES = {
    "education": ["education", "academic background", "academic qualifications"],
    "experience": [
        "experience", "work experience", "professional experience",
        "employment history", "work history",
    ],
    "projects": ["projects", "academic projects", "personal projects"],
    "certifications": [
        "certifications", "certificates", "licenses & certifications",
        "licenses and certifications",
    ],
    "skills": ["skills", "technical skills", "skills & abilities", "core skills"],
}

_HEADER_LINE_RE = re.compile(r"^\s*([A-Za-z &/]{3,40})\s*:?\s*$")


def _build_header_lookup() -> dict:
    lookup = {}
    for field, aliases in _SECTION_ALIASES.items():
        for alias in aliases:
            lookup[alias.lower()] = field
    return lookup


_HEADER_LOOKUP = _build_header_lookup()


def _empty_result(full_text: str = "") -> dict:
    return {
        "full_text": full_text,
        "skills_section": None,
        "education": None,
        "experience": None,
        "projects": None,
        "certifications": None,
        "parse_warnings": [],
    }


def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, list[str]]:
    """Extract raw text from PDF bytes using PyMuPDF. Returns (text, warnings).

    Never raises: any extraction failure is captured as a warning and an
    empty string is returned so downstream code can still run.
    """
    warnings: list[str] = []
    try:
        import pymupdf as fitz  # PyMuPDF (new import name; "fitz" alias is deprecated)
    except ImportError:
        return "", ["PyMuPDF (pymupdf) is not installed; cannot parse PDF files."]

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as exc:  # malformed / encrypted / not actually a PDF
        return "", [f"Failed to open PDF: {exc}"]

    text_parts = []
    try:
        for page in doc:
            text_parts.append(page.get_text())
    except Exception as exc:
        warnings.append(f"Partial PDF extraction failure: {exc}")
    finally:
        doc.close()

    text = "\n".join(text_parts).strip()
    if not text:
        warnings.append(
            "No extractable text found in PDF (it may be a scanned/image-only "
            "document). OCR is not performed by this system."
        )
    return text, warnings


def split_sections(full_text: str) -> dict:
    """Heuristically split resume text into named sections based on
    recognizable header lines. Any section not found is left as None so
    the caller can distinguish "absent" from "empty".
    """
    lines = full_text.split("\n")
    sections: dict[str, list[str]] = {}
    current_field: Optional[str] = None

    for line in lines:
        header_match = _HEADER_LINE_RE.match(line)
        matched_field = None
        if header_match:
            candidate = header_match.group(1).strip().lower()
            matched_field = _HEADER_LOOKUP.get(candidate)

        if matched_field:
            current_field = matched_field
            sections.setdefault(current_field, [])
            continue

        if current_field:
            sections[current_field].append(line)

    return {
        field: ("\n".join(content_lines).strip() or None)
        for field, content_lines in sections.items()
    }


def parse_resume_text(full_text: str) -> dict:
    """Parse already-extracted plain text (shared by .pdf and .txt paths)."""
    result = _empty_result(full_text)
    if not full_text or not full_text.strip():
        result["parse_warnings"].append("Resume text is empty or whitespace-only.")
        return result

    detected = split_sections(full_text)
    for field in ("education", "experience", "projects", "certifications"):
        result[field] = detected.get(field)
    result["skills_section"] = detected.get("skills")

    missing = [f for f in ("education", "experience", "skills") if not detected.get(f if f != "skills" else "skills")]
    if missing:
        result["parse_warnings"].append(
            f"Could not confidently detect section(s): {', '.join(missing)}. "
            "Full text is still available for skill/feature extraction."
        )
    return result


def parse_resume(source: Union[str, Path, bytes], filename: Optional[str] = None) -> dict:
    """Parse a resume from a file path, raw bytes, or raw text string.

    `filename` is used to determine the file type when `source` is bytes.
    Returns a dict with keys: full_text, skills_section, education,
    experience, projects, certifications, parse_warnings.
    """
    warnings: list[str] = []

    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            file_bytes = path.read_bytes()
            text, pdf_warnings = extract_text_from_pdf(file_bytes)
            warnings.extend(pdf_warnings)
        elif suffix == ".txt":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                text = ""
                warnings.append(f"Failed to read text file: {exc}")
        else:
            text = ""
            warnings.append(f"Unsupported file extension '{suffix}'. Only .pdf and .txt are supported.")
    elif isinstance(source, bytes):
        name = (filename or "").lower()
        if name.endswith(".pdf"):
            text, pdf_warnings = extract_text_from_pdf(source)
            warnings.extend(pdf_warnings)
        else:
            try:
                text = source.decode("utf-8", errors="replace")
            except Exception as exc:
                text = ""
                warnings.append(f"Failed to decode bytes as text: {exc}")
    else:
        # Treat as raw text already (e.g. pasted resume content).
        text = str(source) if source is not None else ""

    result = parse_resume_text(text)
    result["parse_warnings"] = warnings + result["parse_warnings"]
    return result
