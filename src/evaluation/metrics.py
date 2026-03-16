import math


def reciprocal_rank(ranked_passage_ids: list, gold_id: str) -> float:
    """Reciprocal rank of the first relevant result. 0 if not found."""
    for rank, pid in enumerate(ranked_passage_ids, start=1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0


def mrr_at_k(
    ranked_lists: list[list[str]],
    gold_ids: list[str],
    k: int = 10,
) -> float:
    """
    Mean Reciprocal Rank @ k.

    Args:
        ranked_lists: list of N ranked lists of passage strings
        gold_ids:     list of N gold passage strings (one per query)
        k:            cutoff rank

    Returns:
        MRR@k score
    """
    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_ids):
        rr = reciprocal_rank(ranked[:k], gold)
        total += rr
    return total / len(ranked_lists) if ranked_lists else 0.0


def recall_at_k(
    ranked_lists: list[list[str]],
    gold_ids: list[str],
    k: int = 100,
) -> float:
    """
    Recall @ k — fraction of queries where gold is in the top-k results.
    """
    hits = sum(
        1 for ranked, gold in zip(ranked_lists, gold_ids)
        if gold in ranked[:k]
    )
    return hits / len(ranked_lists) if ranked_lists else 0.0


def ndcg_at_k(
    ranked_lists: list[list[str]],
    gold_ids: list[str],
    k: int = 10,
) -> float:
    """
    NDCG @ k with binary relevance (1 if passage == gold, else 0).
    """
    def dcg(ranked: list[str], gold: str, k: int) -> float:
        score = 0.0
        for rank, pid in enumerate(ranked[:k], start=1):
            if pid == gold:
                score += 1.0 / math.log2(rank + 1)
        return score

    # Ideal DCG: gold is at rank 1
    idcg = 1.0 / math.log2(2)  # log2(1+1) = 1

    total = 0.0
    for ranked, gold in zip(ranked_lists, gold_ids):
        total += dcg(ranked, gold, k) / idcg

    return total / len(ranked_lists) if ranked_lists else 0.0
