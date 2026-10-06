"""Phase 13/20 -- ranking evaluation metrics (spec.md section 20).

Each function takes `relevance_sorted`: a list of relevance labels (0/1, or
graded) already ORDERED by descending predicted score (i.e. the caller does
the ranking; these functions only score a given ranking).
"""

import math
from typing import List


def precision_at_k(relevance_sorted: List[float], k: int) -> float:
    top_k = relevance_sorted[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r > 0) / len(top_k)


def recall_at_k(relevance_sorted: List[float], k: int) -> float:
    total_relevant = sum(1 for r in relevance_sorted if r > 0)
    if total_relevant == 0:
        return 0.0
    top_k = relevance_sorted[:k]
    return sum(1 for r in top_k if r > 0) / total_relevant


def reciprocal_rank(relevance_sorted: List[float]) -> float:
    for i, r in enumerate(relevance_sorted, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevance_sorted: List[float], k: int) -> float:
    return sum(
        r / math.log2(i + 1) for i, r in enumerate(relevance_sorted[:k], start=1)
    )


def ndcg_at_k(relevance_sorted: List[float], k: int) -> float:
    ideal = sorted(relevance_sorted, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevance_sorted, k) / ideal_dcg


def evaluate_ranking_queries(queries: List[List[float]], k: int = 5) -> dict:
    """Average Precision@K / Recall@K / MRR / NDCG@K over many queries.

    `queries`: list of per-query relevance lists, each already sorted by
    descending predicted score for that query.
    """
    if not queries:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0, "num_queries": 0}

    precisions = [precision_at_k(q, k) for q in queries]
    recalls = [recall_at_k(q, k) for q in queries]
    rrs = [reciprocal_rank(q) for q in queries]
    ndcgs = [ndcg_at_k(q, k) for q in queries]

    return {
        "precision_at_k": round(sum(precisions) / len(precisions), 4),
        "recall_at_k": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(rrs) / len(rrs), 4),
        "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4),
        "k": k,
        "num_queries": len(queries),
    }
