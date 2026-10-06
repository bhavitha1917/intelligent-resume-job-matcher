from pathlib import Path

from src.ranking.rank_candidates import rank_candidates

DEMO_DIR = Path(__file__).resolve().parents[1] / "data" / "external" / "demo"


def _job_text():
    return (DEMO_DIR / "job_description_1.txt").read_text(encoding="utf-8")


def test_rank_candidates_orders_best_match_first():
    resumes = [
        {"candidate_id": "candidate_1", "source": DEMO_DIR / "resume_1.txt"},
        {"candidate_id": "candidate_2", "source": DEMO_DIR / "resume_2.txt"},
        {"candidate_id": "candidate_3", "source": DEMO_DIR / "resume_3.txt"},
    ]
    results = rank_candidates(_job_text(), resumes, compute_semantic=False)

    assert len(results) == 3
    assert [r["rank"] for r in results] == [1, 2, 3]
    # scores must be non-increasing down the ranking
    scores = [r["overall_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    # resume_2 (backend engineer, little overlap) should not rank first
    assert results[0]["candidate_id"] != "candidate_2"


def test_rank_candidates_handles_empty_job_description_gracefully():
    resumes = [{"candidate_id": "c1", "source": DEMO_DIR / "resume_1.txt"}]
    results = rank_candidates("", resumes, compute_semantic=False)
    assert len(results) == 1
    assert results[0]["overall_score"] is not None


def test_rank_candidates_handles_malformed_resume_gracefully():
    resumes = [{"candidate_id": "bad", "source": b"not a pdf", "filename": "bad.pdf"}]
    results = rank_candidates(_job_text(), resumes, compute_semantic=False)
    assert len(results) == 1
    assert results[0]["parse_warnings"]
