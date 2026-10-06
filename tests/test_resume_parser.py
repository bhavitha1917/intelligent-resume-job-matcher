from pathlib import Path

from src.data.resume_parser import parse_resume, split_sections

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "demo"


def test_parses_txt_resume_from_path():
    result = parse_resume(DEMO_DIR / "resume_1.txt")
    assert "Python" in result["full_text"]
    assert result["skills_section"] is not None
    assert result["parse_warnings"] == []


def test_parses_raw_text_string_directly():
    result = parse_resume("SKILLS\nPython, SQL\n\nEDUCATION\nB.S. Computer Science")
    assert result["skills_section"] == "Python, SQL"
    assert result["education"] == "B.S. Computer Science"


def test_empty_resume_produces_warning_not_crash():
    result = parse_resume("")
    assert result["full_text"] == ""
    assert any("empty" in w.lower() for w in result["parse_warnings"])


def test_unsupported_extension_does_not_crash(tmp_path):
    bad_file = tmp_path / "resume.docx"
    bad_file.write_text("some content")
    result = parse_resume(bad_file)
    assert any("unsupported" in w.lower() for w in result["parse_warnings"])


def test_malformed_pdf_bytes_do_not_crash():
    result = parse_resume(b"not a real pdf", filename="resume.pdf")
    assert result["full_text"] == ""
    assert len(result["parse_warnings"]) >= 1


def test_split_sections_missing_section_is_none():
    sections = split_sections("SKILLS\nPython\nSQL")
    assert sections.get("education") is None
    assert sections.get("skills") == "Python\nSQL"
