from src.evaluation.ranking_metrics import (
    evaluate_ranking_queries,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_recall_at_k_perfect_ranking():
    relevance = [1, 1, 0, 0, 0]
    assert precision_at_k(relevance, 2) == 1.0
    assert recall_at_k(relevance, 2) == 1.0


def test_precision_at_k_worst_ranking():
    relevance = [0, 0, 0, 1, 1]
    assert precision_at_k(relevance, 2) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([0, 0, 1, 0]) == 1 / 3
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 1, 0], 3) == 1.0


def test_ndcg_no_relevant_items_is_zero():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_evaluate_ranking_queries_aggregates():
    queries = [[1, 0, 0], [0, 1, 0]]
    result = evaluate_ranking_queries(queries, k=2)
    assert result["num_queries"] == 2
    assert 0.0 <= result["mrr"] <= 1.0


def test_evaluate_ranking_queries_empty_input():
    result = evaluate_ranking_queries([], k=5)
    assert result["num_queries"] == 0
