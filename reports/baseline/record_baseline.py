"""Records the behaviour of the ORIGINAL (baseline-v1-deployed) scoring path
so later changes can be compared against it. It uses the v1 APIs, so it only
runs on the `baseline-v1-deployed` tag (git switch --detach baseline-v1-deployed):
    python -m reports.baseline.record_baseline
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.inference import predict_match
from src.ranking.rank_candidates import rank_candidates
from src.utils.paths import DEMO_DATA_DIR

fixture = {
    "skill_coverage": 0.3333, "num_matched_skills": 2, "num_missing_skills": 4,
    "tfidf_similarity": 0.3566, "semantic_similarity": 0.2972,
    "experience_compatibility": 0.5, "education_compatibility": 0.5,
    "title_category_compatibility": 0.5,
}
out = {"fixture_features": fixture, "fixture_predict_match": predict_match(fixture)}
job = (DEMO_DATA_DIR / "job_description_1.txt").read_text(encoding="utf-8")
res = rank_candidates(job, [{"candidate_id": f, "source": DEMO_DATA_DIR / f} for f in ["resume_1.txt", "resume_2.txt", "resume_3.txt"]])
out["demo_ranking"] = [
    {k: r[k] for k in ("rank", "candidate_id", "overall_score", "skill_score", "semantic_score", "tfidf_score", "matched_skills", "missing_skills")}
    for r in res
]
(Path(__file__).parent / "baseline_scores.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print(json.dumps(out, indent=2, default=str))
