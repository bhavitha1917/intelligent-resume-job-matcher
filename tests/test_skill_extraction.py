from src.skills.skill_extractor import extract_skills


def test_extracts_known_skills():
    text = "Experience with Python, SQL, Machine Learning and PyTorch."
    skills = extract_skills(text)
    assert "Python" in skills
    assert "SQL" in skills
    assert "Machine Learning" in skills
    assert "PyTorch" in skills


def test_extracts_punctuation_containing_skills():
    text = "Worked with C++, C#, .NET and Node.js daily."
    skills = extract_skills(text)
    assert "C++" in skills
    assert "C#" in skills
    assert ".NET" in skills
    assert "Node.js" in skills


def test_aliases_map_to_canonical_name():
    assert "Machine Learning" in extract_skills("5 years of ML experience")
    assert "PostgreSQL" in extract_skills("Database: Postgres")
    assert "PyTorch" in extract_skills("Built models using torch")


def test_does_not_match_substrings():
    # "js" must not match inside an unrelated word
    skills = extract_skills("projects require strong judgement")
    assert "JavaScript" not in skills


def test_generic_words_are_not_skills():
    skills = extract_skills("I have a lot of experience and data.")
    assert skills == []


def test_empty_text_returns_empty_list():
    assert extract_skills("") == []
    assert extract_skills(None) == []


def test_deterministic_and_deduplicated():
    text = "Python python PYTHON"
    assert extract_skills(text) == ["Python"]
