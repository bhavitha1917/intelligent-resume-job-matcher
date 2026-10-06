from src.preprocessing.text_cleaning import (
    clean_text,
    handle_missing,
    normalize_whitespace,
    strip_html,
)


def test_strip_html_removes_tags_keeps_text():
    assert strip_html("<p>Hello <b>World</b></p>") == " Hello  World  "


def test_clean_text_preserves_technical_tokens():
    text = "Skilled in C++, C#, .NET, AWS, SQL, PyTorch, TensorFlow and Node.js."
    cleaned = clean_text(text)
    for token in ["C++", "C#", ".NET", "AWS", "SQL", "PyTorch", "TensorFlow", "Node.js"]:
        assert token in cleaned


def test_clean_text_lowercase_flag():
    assert clean_text("Python SQL", lowercase=True) == "python sql"


def test_normalize_whitespace_collapses_blank_lines():
    text = "Line1\n\n\n\nLine2"
    assert normalize_whitespace(text) == "Line1\nLine2"


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_handle_missing_normalizes_nan_like_values():
    assert handle_missing(None) == ""
    assert handle_missing(float("nan")) == ""
    assert handle_missing("NaN") == ""
    assert handle_missing("  ") == ""
    assert handle_missing("Python") == "Python"
