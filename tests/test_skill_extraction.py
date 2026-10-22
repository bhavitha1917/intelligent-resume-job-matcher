from src.skills.skill_extractor import canonicalize_skill, extract_skills, skill_patterns


def _found(phrase, text):
    return any(p.search(text) for p in skill_patterns(canonicalize_skill(phrase)))


def test_extracts_known_skills():
    skills = extract_skills("Experience with Python, SQL, Machine Learning and PyTorch.")
    assert {"Python", "SQL", "Machine Learning", "PyTorch"} <= set(skills)


def test_extracts_punctuation_containing_skills():
    skills = extract_skills("Worked with C++, C#, .NET and Node.js daily.")
    assert {"C++", "C#", ".NET", "Node.js"} <= set(skills)


def test_aliases_map_to_canonical_name():
    assert "Machine Learning" in extract_skills("5 years of ML experience")
    assert "PostgreSQL" in extract_skills("Database: Postgres")
    assert "PyTorch" in extract_skills("Built models using torch")
    assert canonicalize_skill("postgres") == "PostgreSQL"
    assert canonicalize_skill("Problem-Solving") == "Problem Solving"


def test_ambiguous_abbreviations_are_case_sensitive():
    assert "Machine Learning" not in extract_skills("Add 5 ml of reagent")
    assert "Computer Vision" not in extract_skills("Please find my cv attached")
    assert "Computer Vision" in extract_skills("Projects in CV and NLP")
    assert "Excel" not in extract_skills("I excel at teamwork")
    assert "Excel" in extract_skills("Advanced Excel and VBA")


def test_does_not_match_substrings():
    assert "JavaScript" not in extract_skills("projects require strong judgement")
    assert not _found("Java", "Expert in JavaScript")
    assert not _found("R", "Built React apps; led R&D team")       # bare "R" is never evidence
    assert not _found("Go", "Good communicator, go-getter")
    assert not _found("SQL", "NoSQLish")
    assert _found("SQL", "SQL Server and SQL")


def test_free_text_requirement_phrases_match_literally_with_plural_tolerance():
    assert canonicalize_skill("Menu Costing") == "menu costing"
    assert _found("menu costing", "Responsible for menu costings and ordering")
    assert not _found("menu costing", "menu planning")


def test_spring_season_is_not_spring_boot():
    assert "Spring Boot" not in extract_skills("Internship, Spring 2023")
    assert "Spring Boot" in extract_skills("REST services in Spring Boot")


def test_generic_words_are_not_skills():
    assert extract_skills("I have a lot of experience and data.") == []


def test_empty_text_returns_empty_list():
    assert extract_skills("") == []
    assert extract_skills(None) == []


def test_deterministic_and_deduplicated():
    assert extract_skills("Python python PYTHON") == ["Python"]
