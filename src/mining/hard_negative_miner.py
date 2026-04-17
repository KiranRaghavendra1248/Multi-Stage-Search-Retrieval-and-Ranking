from __future__ import annotations

from typing import Iterable, Optional, Union

from tqdm import tqdm

from src.indexing.bm25_index import BM25Index
from src.mining.triplet_writer import TripletWriter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _apply_positive_aware_filter(
    candidates: list[tuple[str, float]],
    pos_score: float,
    method: str,
    margin_pos: float,
    perc_pos: float,
) -> list[tuple[str, float]]:
    """
    Filter hard negative candidates using the positive passage score as an anchor.

    Removes passages that are too similar to the positive — these are likely
    false negatives (relevant but unlabeled). Based on NV-Retriever (arxiv:2407.15831).

    Methods:
        topk_margin_pos:  threshold = pos_score - margin_pos
                          keep candidates where sim(q, c) <= threshold
        topk_perc_pos:    threshold = perc_pos * pos_score
                          keep candidates where sim(q, c) <= threshold
                          (recommended: wins on 13/15 BEIR datasets)

    Args:
        candidates:  list of (text, score) sorted by score descending
        pos_score:   similarity score of the gold positive passage
        method:      "topk_margin_pos" | "topk_perc_pos"
        margin_pos:  absolute margin (used by topk_margin_pos, e.g. 0.05)
        perc_pos:    percentage of pos_score (used by topk_perc_pos, e.g. 0.95)

    Returns:
        Filtered list of (text, score), preserving original order.
    """
    if method == "topk_margin_pos":
        threshold = pos_score - margin_pos
    elif method == "topk_perc_pos":
        threshold = perc_pos * pos_score
    else:
        raise ValueError(f"Unknown positive_aware_method: {method!r}")

    filtered = [(text, score) for text, score in candidates if score <= threshold]
    return filtered


def mine_hard_negatives(
    records: Iterable[dict],
    index,  # BM25Index | DenseTeacher
    writer: TripletWriter,
    seen_queries: set[str],
    n_hard_negatives: int = 5,
    top_k_retrieve: int = 100,
    total: Optional[int] = None,
    max_triplets: Optional[int] = None,
    positive_aware_method: Optional[str] = None,
    margin_pos: float = 0.05,
    perc_pos: float = 0.95,
) -> dict:
    """
    Mine hard negatives for each record and write triplets via writer.

    Supports two teacher modes:
      - BM25Index:     retrieval by keyword overlap (fast, no GPU)
      - DenseTeacher:  retrieval by semantic similarity (slower, requires GPU)
                       Enables positive-aware filtering (topk_margin_pos / topk_perc_pos)
                       to remove false negatives before storing.

    Args:
        records:                 Iterable of MS MARCO records (from ms_marco_loader).
        index:                   Pre-built BM25Index or DenseTeacher with .search() interface.
        writer:                  TripletWriter in append mode.
        seen_queries:            Set of query strings already written (for resume).
        n_hard_negatives:        How many hard negatives to store per triplet.
        top_k_retrieve:          How many candidates to retrieve before filtering.
        total:                   Optional total count for tqdm progress bar.
        max_triplets:            Stop after writing this many triplets (None = no cap).
        positive_aware_method:   null | "topk_margin_pos" | "topk_perc_pos"
                                 Only applied when index is a DenseTeacher.
        margin_pos:              Absolute margin for topk_margin_pos (default 0.05).
        perc_pos:                Percentage multiplier for topk_perc_pos (default 0.95).

    Returns:
        Stats dict: {written, skipped_seen, skipped_no_positive, skipped_few_negatives}
    """
    is_dense = not isinstance(index, BM25Index)

    stats = {
        "written": 0,
        "skipped_seen": 0,
        "skipped_no_positive": 0,
        "skipped_few_negatives": 0,
    }

    already_written = len(seen_queries)
    for record in tqdm(records, total=total, desc="Mining hard negatives", unit="query"):
        if max_triplets is not None and (already_written + stats["written"]) >= max_triplets:
            break

        query: str = record["query"]
        positive: Optional[str] = record["positive_passage"]

        # Resume: skip already-processed queries
        if query in seen_queries:
            stats["skipped_seen"] += 1
            continue

        # Skip queries with no labeled positive
        if not positive:
            stats["skipped_no_positive"] += 1
            continue

        # Retrieve candidates — both BM25Index and DenseTeacher expose .search(query, top_k)
        raw_results: list[tuple[str, float]] = index.search(query, top_k=top_k_retrieve)

        # Filter: remove the gold positive by exact text match
        candidates = [(text, score) for text, score in raw_results if text != positive]

        # Positive-aware filtering (dense teacher only — BM25 scores are unnormalized)
        if is_dense and positive_aware_method is not None and candidates:
            pos_score = index.score_pair(query, positive)
            before = len(candidates)
            candidates = _apply_positive_aware_filter(
                candidates, pos_score, positive_aware_method, margin_pos, perc_pos
            )
            logger.debug(
                "Positive-aware filter (%s): %d → %d candidates (pos_score=%.4f)",
                positive_aware_method, before, len(candidates), pos_score,
            )

        if len(candidates) < 1:
            stats["skipped_few_negatives"] += 1
            continue

        # Take up to n_hard_negatives (sorted by score descending)
        hard_negatives = [text for text, _score in candidates[:n_hard_negatives]]

        writer.write(query, positive, hard_negatives)
        seen_queries.add(query)
        stats["written"] += 1

    logger.info(
        "Mining complete. written=%d skipped_seen=%d skipped_no_positive=%d skipped_few=%d",
        stats["written"],
        stats["skipped_seen"],
        stats["skipped_no_positive"],
        stats["skipped_few_negatives"],
    )
    return stats
