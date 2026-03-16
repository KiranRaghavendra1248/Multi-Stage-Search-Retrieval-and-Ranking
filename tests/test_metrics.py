import pytest
from src.evaluation.metrics import mrr_at_k, recall_at_k, ndcg_at_k


def test_mrr_perfect():
    ranked = [["p1", "p2", "p3"]]
    gold = ["p1"]
    assert mrr_at_k(ranked, gold, k=10) == 1.0


def test_mrr_second_rank():
    ranked = [["p0", "p1", "p2"]]
    gold = ["p1"]
    assert mrr_at_k(ranked, gold, k=10) == pytest.approx(0.5)


def test_mrr_not_in_top_k():
    ranked = [["p0", "p1", "p2"]]
    gold = ["p3"]
    assert mrr_at_k(ranked, gold, k=10) == 0.0


def test_mrr_cutoff_respected():
    ranked = [["p0", "p1", "p2", "p3"]]
    gold = ["p3"]
    assert mrr_at_k(ranked, gold, k=2) == 0.0


def test_mrr_multiple_queries():
    ranked = [["p1", "p2"], ["p0", "p1"]]
    gold = ["p1", "p1"]
    # query 1: RR = 1.0, query 2: RR = 0.5  → MRR = 0.75
    assert mrr_at_k(ranked, gold, k=10) == pytest.approx(0.75)


def test_recall_at_k_hit():
    ranked = [["p0", "p1", "p2"]]
    gold = ["p2"]
    assert recall_at_k(ranked, gold, k=3) == 1.0


def test_recall_at_k_miss():
    ranked = [["p0", "p1", "p2"]]
    gold = ["p2"]
    assert recall_at_k(ranked, gold, k=2) == 0.0


def test_recall_at_k_partial():
    ranked = [["p1", "p2"], ["p0", "p3"]]
    gold = ["p1", "p1"]
    # query 1 hit, query 2 miss → 0.5
    assert recall_at_k(ranked, gold, k=2) == pytest.approx(0.5)


def test_ndcg_perfect():
    ranked = [["p1"]]
    gold = ["p1"]
    assert ndcg_at_k(ranked, gold, k=10) == pytest.approx(1.0)


def test_ndcg_miss():
    ranked = [["p0", "p1"]]
    gold = ["p2"]
    assert ndcg_at_k(ranked, gold, k=10) == 0.0


def test_empty_inputs():
    assert mrr_at_k([], [], k=10) == 0.0
    assert recall_at_k([], [], k=10) == 0.0
    assert ndcg_at_k([], [], k=10) == 0.0
