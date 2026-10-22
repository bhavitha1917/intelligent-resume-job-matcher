import io

import pandas as pd
from _fixtures import JOB_DATA_ANALYST, REFERENCE_DATE, RESUME_PARTIAL, RESUME_STRONG, RESUME_WEAK, fake_semantic, job

from src.data.resume_parser import parse_resume
from src.ranking.rank_candidates import assign_ranks, rank_candidates
from src.scoring.export import CSV_COLUMNS, report_to_markdown, results_to_csv
from src.scoring.scorer import score_candidate


def _batch():
    return [
        {"candidate_id": "weak.txt", "source": RESUME_WEAK.encode(), "filename": "weak.txt"},
        {"candidate_id": "strong.txt", "source": RESUME_STRONG.encode(), "filename": "strong.txt"},
        {"candidate_id": "partial.txt", "source": RESUME_PARTIAL.encode(), "filename": "partial.txt"},
    ]


def _rank(resumes, **kw):
    kw.setdefault("semantic", fake_semantic)
    return rank_candidates(JOB_DATA_ANALYST, resumes, requirements=job()["requirements"],
                           reference_date=REFERENCE_DATE, **kw)


def test_single_and_multi_resume_scores_agree_exactly():
    rows = _rank(_batch())
    for entry in _batch():
        single = score_candidate(job(), parse_resume(entry["source"], entry["filename"]), semantic=fake_semantic,
                                 reference_date=REFERENCE_DATE, candidate_id=entry["candidate_id"])
        row = next(r for r in rows if r["candidate_id"] == entry["candidate_id"])
        assert row["score"] == single["score"]
        assert row["report"] == single


def test_ranking_is_consistent_with_scores():
    rows = _rank(_batch())
    assert [r["candidate_id"] for r in rows] == ["strong.txt", "partial.txt", "weak.txt"]
    assert [r["rank"] for r in rows] == [1, 2, 3]
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)


def test_ties_share_a_rank_and_are_listed_by_name():
    rows = [{"candidate_id": n, "status": "scored", "score": s} for n, s in (("b", 70.0), ("a", 70.04), ("c", 50.0))]
    ranked = assign_ranks(rows, decimals=1)
    assert [(r["candidate_id"], r["rank"], r["tied"]) for r in ranked] == [("a", 1, True), ("b", 1, True), ("c", 3, False)]


def test_partial_batch_failures_are_reported_not_dropped():
    batch = _batch() + [
        {"candidate_id": "broken.pdf", "source": b"%PDF-garbage", "filename": "broken.pdf"},
        {"candidate_id": "copy_of_strong.txt", "source": RESUME_STRONG.encode(), "filename": "copy_of_strong.txt"},
        {"candidate_id": "tiny.txt", "source": b"Python SQL", "filename": "tiny.txt"},
    ]
    rows = _rank(batch)
    status = {r["candidate_id"]: r["status"] for r in rows}
    assert len(rows) == len(batch)
    assert status["broken.pdf"] == "parse_failed"
    assert status["copy_of_strong.txt"] == "duplicate"
    assert next(r for r in rows if r["candidate_id"] == "copy_of_strong.txt")["duplicate_of"] == "strong.txt"
    assert status["tiny.txt"] == "insufficient_evidence"
    assert all(r["rank"] is None for r in rows if r["status"] != "scored")


def test_progress_callback_reports_every_file():
    calls = []
    _rank(_batch(), progress=lambda i, n, name: calls.append((i, n)))
    assert calls[0] == (0, 3) and calls[-1] == (3, 3)


def test_csv_export_contains_every_candidate_and_expected_columns():
    rows = _rank(_batch() + [{"candidate_id": "broken.pdf", "source": b"x", "filename": "broken.pdf"}])
    df = pd.read_csv(io.BytesIO(results_to_csv(rows)))
    assert list(df.columns) == CSV_COLUMNS
    assert set(df["candidate"]) == {"strong.txt", "partial.txt", "weak.txt", "broken.pdf"}
    strong = df[df["candidate"] == "strong.txt"].iloc[0]
    assert strong["rank"] == 1 and strong["score"] == rows[0]["score"]
    assert strong["required_skills_detected"] == 4 and strong["required_skills_total"] == 4
    assert "@" not in results_to_csv(rows).decode()  # no contact details / full text in exports


def test_markdown_report_states_interpretation_and_reproducibility():
    rows = _rank(_batch())
    md = report_to_markdown(rows[0]["report"])
    assert "NOT a probability of being hired" in md and "fingerprint" in md
    assert f"{rows[0]['score']:.1f} / 100" in md
