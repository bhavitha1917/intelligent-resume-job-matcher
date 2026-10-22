"""Resume parsing for .pdf, .docx and .txt files.

Returns a structured dict and NEVER raises on malformed input — callers
(tests, Streamlit app) always get a dict back, with `parse_warnings`
explaining anything that went wrong or any section that could not be
detected (spec.md section 10: "degrade gracefully").

Besides the per-section text fields, the result carries `section_spans`:
(section_name, start, end) character offsets into `full_text`, so the
scoring layer can tell whether a skill mention sits in a project,
internship, employment, skills-list or personal-details section.
"""

import io
import re
import zipfile
from pathlib import Path
from typing import Optional, Union
from xml.etree import ElementTree

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")

# Section headers we recognize. Keys are canonical section names; values
# are the header spellings that introduce them.
_SECTION_ALIASES = {
    "summary": ["summary", "professional summary", "profile", "career objective", "objective", "about me"],
    "education": [
        "education", "academic background", "academic qualifications", "educational qualifications",
        "education and training", "qualifications",
    ],
    "experience": [
        "experience", "work experience", "professional experience", "employment history",
        "work history", "employment", "relevant experience",
    ],
    "internships": ["internships", "internship", "internship experience", "internships and training"],
    "projects": ["projects", "academic projects", "personal projects", "key projects", "project work"],
    "certifications": [
        "certifications", "certificates", "licenses & certifications", "licenses and certifications",
        "certifications and licenses", "courses and certifications",
    ],
    "skills": [
        "skills", "technical skills", "skills & abilities", "core skills", "key skills",
        "highlights", "skill highlights", "core competencies", "competencies", "tools and technologies",
    ],
    "achievements": ["achievements", "accomplishments", "awards", "honors", "awards and honors"],
    "personal": ["personal details", "personal information", "personal profile", "personal data"],
    "other": ["interests", "hobbies", "languages", "references", "extracurricular activities", "activities"],
    # A pasted job advert / requirement list inside a resume: its skills are
    # NOT evidence about the candidate (see src/scoring/evidence.py).
    "quoted_requirements": ["job description", "job requirements", "target job requirements", "role requirements"],
}

_HEADER_LINE_RE = re.compile(r"^\s*([A-Za-z &/]{3,40}?)\s*:?\s*$")


def _build_header_lookup() -> dict:
    return {alias.lower(): field for field, aliases in _SECTION_ALIASES.items() for alias in aliases}


_HEADER_LOOKUP = _build_header_lookup()

# Inline-header fallback (e.g. the HF dataset's resume_text, where sections
# are separated by runs of spaces instead of newlines): a known header
# surrounded by 3+ spaces on both sides.
_INLINE_HEADER_RE = re.compile(
    r"(?:^|\s{3,})(" + "|".join(sorted((re.escape(a) for a in _HEADER_LOOKUP), key=len, reverse=True)) + r")(?=\s{3,})",
    re.IGNORECASE,
)


def _empty_result(full_text: str = "") -> dict:
    return {
        "full_text": full_text,
        "skills_section": None,
        "education": None,
        "experience": None,
        "internships": None,
        "projects": None,
        "certifications": None,
        "section_spans": [],
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
        if doc.needs_pass:
            return "", ["PDF is password-protected; text could not be extracted."]
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


_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_text_from_docx(file_bytes: bytes) -> tuple[str, list[str]]:
    """Extract paragraph text from a .docx using only the standard library
    (a .docx is a zip archive containing word/document.xml). Tables are
    read cell by cell; headers/footers, text boxes and images are not.
    Never raises."""
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            xml_bytes = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError, OSError) as exc:
        return "", [f"Failed to open DOCX: {exc}"]

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        return "", [f"Failed to read DOCX content: {exc}"]

    paragraphs = []
    for paragraph in root.iter(f"{_W_NS}p"):
        parts = []
        for node in paragraph.iter():
            if node.tag == f"{_W_NS}t" and node.text:
                parts.append(node.text)
            elif node.tag == f"{_W_NS}tab":
                parts.append("\t")
            elif node.tag in (f"{_W_NS}br", f"{_W_NS}cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))

    text = "\n".join(paragraphs).strip()
    warnings = [] if text else ["No extractable text found in DOCX."]
    return text, warnings


def section_spans(full_text: str) -> list[tuple[str, int, int]]:
    """(section_name, start, end) offsets of detected sections, in order.

    Header lines (a whole line that is a known header) are used first; if
    none are found, the inline-header fallback is tried. Text before the
    first header is reported as section "header" (typically name/contact).
    """
    headers: list[tuple[str, int, int]] = []  # (field, header_start, content_start)
    offset = 0
    for line in full_text.split("\n"):
        match = _HEADER_LINE_RE.match(line)
        if match:
            field = _HEADER_LOOKUP.get(match.group(1).strip().lower())
            if field:
                headers.append((field, offset, offset + len(line) + 1))
        offset += len(line) + 1

    if not headers:
        for match in _INLINE_HEADER_RE.finditer(full_text):
            field = _HEADER_LOOKUP[match.group(1).lower()]
            headers.append((field, match.start(1), match.end(1)))

    if not headers:
        return []

    spans = []
    if headers[0][1] > 0:
        spans.append(("header", 0, headers[0][1]))
    for i, (field, _, content_start) in enumerate(headers):
        end = headers[i + 1][1] if i + 1 < len(headers) else len(full_text)
        spans.append((field, min(content_start, len(full_text)), end))
    return spans


def split_sections(full_text: str) -> dict:
    """Heuristically split resume text into named sections. Any section not
    found is absent from the dict so the caller can distinguish "absent"
    from "empty". Repeated sections are concatenated."""
    collected: dict[str, list[str]] = {}
    for field, start, end in section_spans(full_text):
        if field == "header":
            continue
        collected.setdefault(field, []).append(full_text[start:end].strip())
    return {field: ("\n".join(p for p in parts if p).strip() or None) for field, parts in collected.items()}


def parse_resume_text(full_text: str) -> dict:
    """Parse already-extracted plain text (shared by all file types)."""
    result = _empty_result(full_text)
    if not full_text or not full_text.strip():
        result["parse_warnings"].append("Resume text is empty or whitespace-only.")
        return result

    result["section_spans"] = section_spans(full_text)
    detected = split_sections(full_text)
    for field in ("education", "experience", "internships", "projects", "certifications"):
        result[field] = detected.get(field)
    result["skills_section"] = detected.get("skills")

    missing = [f for f in ("education", "experience", "skills") if not detected.get(f)]
    if missing:
        result["parse_warnings"].append(
            f"Could not confidently detect section(s): {', '.join(missing)}. "
            "Full text is still available for skill/feature extraction."
        )
    return result


def _decode_text_bytes(data: bytes) -> tuple[str, list[str]]:
    for encoding in ("utf-8-sig", "utf-16"):
        try:
            text = data.decode(encoding)
            if encoding == "utf-16" and "\x00" in text:
                continue
            return text, []
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace"), ["Text file is not valid UTF-8; undecodable characters were replaced."]


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:4096]


def parse_resume(source: Union[str, Path, bytes], filename: Optional[str] = None) -> dict:
    """Parse a resume from a file path, raw bytes, or raw text string.

    `filename` is used to determine the file type when `source` is bytes.
    Returns a dict with keys: full_text, skills_section, education,
    experience, internships, projects, certifications, section_spans,
    parse_warnings.
    """
    warnings: list[str] = []
    data: Optional[bytes] = None
    suffix = ""

    if isinstance(source, Path) or (isinstance(source, str) and len(source) < 1024 and "\n" not in source and Path(source).exists()):
        path = Path(source)
        suffix = path.suffix.lower()
        try:
            data = path.read_bytes()
        except OSError as exc:
            data = b""
            warnings.append(f"Failed to read file: {exc}")
    elif isinstance(source, bytes):
        data = source
        suffix = Path(filename or "").suffix.lower()
    else:
        # Treat as raw text already (e.g. pasted resume content).
        text = str(source) if source is not None else ""

    if data is not None:
        if suffix == ".pdf":
            text, extra = extract_text_from_pdf(data)
        elif suffix == ".docx":
            text, extra = extract_text_from_docx(data)
        elif suffix == ".txt" or (suffix == "" and not _looks_binary(data)):
            text, extra = _decode_text_bytes(data)
        else:
            text, extra = "", [
                f"Unsupported file type '{suffix or 'unknown'}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
            ]
        warnings.extend(extra)

    result = parse_resume_text(text)
    result["parse_warnings"] = warnings + result["parse_warnings"]
    return result
