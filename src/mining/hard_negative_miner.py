from typing import Iterable
from tqdm import tqdm
from src.indexing.bm25_index import BM25Index
from src.mining.triplet_writer import TripletWriter
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def mine_hard_negatives(
    records: Iterable[dict],
    index: BM25Index,
    writer: TripletWriter,
    seen_queries: set[str],
    n_hard_negatives: int = 5,
    bm25_top_k: int = 100,
    total: int | None = None,
    max_triplets: int | None = None,
) -> dict:
    """
    Mine hard negatives for each record and write triplets via writer.

    Args:
        records:          Iterable of MS MARCO records (from ms_marco_loader).
        index:            Pre-built BM25Index.
        writer:           TripletWriter in append mode.
        seen_queries:     Set of query strings already written (for resume).
        n_hard_negatives: How many hard negatives to store per triplet (default 5).
        bm25_top_k:       How many BM25 results to retrieve before filtering.
        total:            Optional total count for tqdm progress bar.
        max_triplets:     Stop after writing this many triplets (None = no cap).

    Returns:
        Stats dict: {written, skipped_seen, skipped_no_positive, skipped_few_negatives}
    """
    stats = {
        "written": 0,
        "skipped_seen": 0,
        "skipped_no_positive": 0,
        "skipped_few_negatives": 0,
    }

    already_written = len(seen_queries)
    for record in tqdm(records, total=total, desc="Mining hard negatives", unit="query"):
        if max_triplets is not None and (already_written + stats["written"]) >= max_triplets:
            logger.info("Reached max_triplets=%d — stopping.", max_triplets)
            break
        query: str = record["query"]
        positive: str | None = record["positive_passage"]

        # Resume: skip already-processed queries
        if query in seen_queries:
            stats["skipped_seen"] += 1
            continue

        # Skip queries with no labeled positive
        if not positive:
            stats["skipped_no_positive"] += 1
            continue

        # BM25 retrieval
        bm25_results = index.search(query, top_k=bm25_top_k)  # [(text, score), ...]

        # Filter: remove the gold positive by exact text match
        hard_negatives = [
            text for text, _score in bm25_results
            if text != positive
        ]

        if len(hard_negatives) < 1:
            stats["skipped_few_negatives"] += 1
            continue

        # Take up to n_hard_negatives (BM25 score already sorted descending)
        hard_negatives = hard_negatives[:n_hard_negatives]

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
