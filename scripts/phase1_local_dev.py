"""
Phase 1: Local Development Prototype (Zero-Disk)

Validates the full data pipeline on 1,000 MS MARCO rows locally:
  - Stream 1k rows
  - Chunk all passages
  - Build BM25 index
  - Mine hard negatives for each query
  - Print 10 example triplets to stdout
No disk writes. Completes in < 2 minutes.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.data.ms_marco_loader import load_msmarco_stream
from src.data.chunker import chunk_document
from src.indexing.bm25_index import BM25Index

logger = get_logger("phase1")


def main():
    cfg = load_config()
    logger.info("Environment: %s | sample_cap: %s", cfg.environment, cfg.data.sample_cap)

    # 1. Stream 1k records
    records = load_msmarco_stream(cfg, split=cfg.data.split_train)
    logger.info("Loaded %d records.", len(records))

    # 2. Chunk all passages
    all_passages: list[str] = []
    passage_to_query_id: dict[str, str] = {}

    for rec in records:
        for pt in rec["passages"].get("passage_text", []):
            chunks = chunk_document(
                pt,
                tokenizer_name=cfg.chunking.tokenizer,
                max_tokens=cfg.chunking.max_tokens,
                min_tokens_merge=cfg.chunking.min_tokens_merge,
            )
            for chunk in chunks:
                all_passages.append(chunk)
                passage_to_query_id[chunk] = rec["query_id"]

    logger.info("Total chunks: %d (avg %.1f per passage)", len(all_passages), len(all_passages) / max(len(records), 1))

    # 3. Build BM25 index (in-memory, no save)
    index = BM25Index()
    index.build(all_passages)

    # 4. Mine hard negatives
    triplets = []
    skipped_no_positive = 0

    for rec in records:
        if rec["positive_passage"] is None:
            skipped_no_positive += 1
            continue

        query = rec["query"]
        positive = rec["positive_passage"]

        bm25_results = index.search(query, top_k=cfg.bm25.top_k_retrieve)
        hard_negatives = [
            text for text, _score in bm25_results
            if text != positive
        ][:cfg.bm25.n_hard_negatives]

        if not hard_negatives:
            continue

        triplets.append({
            "query": query,
            "positive": positive,
            "hard_negatives": hard_negatives,
        })

    logger.info(
        "Mined %d triplets. Skipped %d queries (no positive).",
        len(triplets), skipped_no_positive,
    )

    # 5. Print 10 example triplets
    print("\n" + "=" * 70)
    print("SAMPLE TRIPLETS (10 of %d)" % len(triplets))
    print("=" * 70)
    for i, t in enumerate(triplets[:10]):
        print(f"\n[{i+1}] Query:    {t['query'][:100]}")
        print(f"    Positive: {t['positive'][:120]}")
        print(f"    HN #1:    {t['hard_negatives'][0][:120]}")

    # 6. Stats
    print("\n" + "=" * 70)
    print("STATS")
    print("=" * 70)
    print(f"  Records loaded:     {len(records)}")
    print(f"  Total chunks:       {len(all_passages)}")
    print(f"  Triplets mined:     {len(triplets)}")
    print(f"  No-positive skip:   {skipped_no_positive}")
    avg_hn = sum(len(t["hard_negatives"]) for t in triplets) / max(len(triplets), 1)
    print(f"  Avg HN per query:   {avg_hn:.2f}")


if __name__ == "__main__":
    main()
