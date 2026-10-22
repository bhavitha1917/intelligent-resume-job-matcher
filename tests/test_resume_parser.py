import io
import zipfile
from pathlib import Path

from src.data.resume_parser import parse_resume, section_spans, split_sections

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "demo"


def _docx_bytes(paragraphs):
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs)
    xml = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{ns}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def test_parses_txt_resume_from_path():
    result = parse_resume(DEMO_DIR / "resume_1.txt")
    assert "Python" in result["full_text"]
    assert result["skills_section"] is not None
    assert result["parse_warnings"] == []


def test_parses_raw_text_string_directly():
    result = parse_resume("SKILLS\nPython, SQL\n\nEDUCATION\nB.S. Computer Science")
    assert result["skills_section"] == "Python, SQL"
    assert result["education"] == "B.S. Computer Science"


def test_parses_docx_bytes():
    data = _docx_bytes(["SKILLS", "Python, SQL", "EDUCATION", "B.Tech in Computer Science"])
    result = parse_resume(data, filename="cv.docx")
    assert result["skills_section"] == "Python, SQL"
    assert "B.Tech" in result["education"]


def test_malformed_docx_and_pdf_do_not_crash():
    for data, name in ((b"not a zip", "cv.docx"), (b"not a real pdf", "cv.pdf")):
        result = parse_resume(data, filename=name)
        assert result["full_text"] == ""
        assert result["parse_warnings"]


def test_empty_resume_produces_warning_not_crash():
    result = parse_resume("")
    assert result["full_text"] == ""
    assert any("empty" in w.lower() for w in result["parse_warnings"])


def test_unsupported_extension_does_not_crash(tmp_path):
    bad_file = tmp_path / "resume.odt"
    bad_file.write_bytes(b"PK\x03\x04\x00\x00binary")
    result = parse_resume(bad_file)
    assert any("unsupported" in w.lower() for w in result["parse_warnings"])


def test_split_sections_missing_section_is_none():
    sections = split_sections("SKILLS\nPython\nSQL")
    assert sections.get("education") is None
    assert sections.get("skills") == "Python\nSQL"


def test_internships_and_personal_sections_are_recognised():
    names = [n for n, _, _ in section_spans("Name\nINTERNSHIPS\nIntern\nPERSONAL DETAILS\nGender: x\nPROJECTS\nApp")]
    assert names == ["header", "internships", "personal", "projects"]


def test_inline_headers_fallback_for_single_line_text():
    text = "JOHN   Summary   Analyst.   Experience   01/2019 to 02/2021 Analyst   Education   B.S. Math   Skills   SQL"
    names = [n for n, _, _ in section_spans(text)]
    assert names[:5] == ["header", "summary", "experience", "education", "skills"]
