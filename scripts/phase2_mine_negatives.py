"""
Phase 2: Hard Negative Mining (Vast.ai)

1. Stream full 3.2M MS MARCO documents → chunk → build bm25s index → save to disk
2. Stream 500k train queries → mine 5 hard negatives per query → write JSONL
3. Supports crash-safe resume: re-run picks up from where it left off
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.data.ms_marco_loader import iter_msmarco_stream
from src.data.chunker import chunk_document
from src.indexing.bm25_index import BM25Index
from src.mining.hard_negative_miner import mine_hard_negatives
from src.mining.triplet_writer import TripletWriter

logger = get_logger("phase2")


def main():
    cfg = load_config()
    logger.info("Environment: %s", cfg.environment)

    bm25_index_dir = cfg.paths.bm25_index_dir
    triplets_file = cfg.paths.triplets_file

    # --- Step 1: Build BM25 index (skip if already exists) ---
    if Path(bm25_index_dir).exists() and (Path(bm25_index_dir) / "passages.pkl").exists():
        logger.info("BM25 index already exists at %s — loading.", bm25_index_dir)
        index = BM25Index.load(bm25_index_dir)
    else:
        logger.info("Building BM25 index from full MS MARCO corpus...")
        all_passages: list[str] = []

        for rec in iter_msmarco_stream(cfg, split=cfg.data.split_train):
            for pt in rec["passages"].get("passage_text", []):
                chunks = chunk_document(
                    pt,
                    tokenizer_name=cfg.chunking.tokenizer,
                    max_tokens=cfg.chunking.max_tokens,
                    min_tokens_merge=cfg.chunking.min_tokens_merge,
                )
                all_passages.extend(chunks)

            if len(all_passages) % 100_000 == 0 and len(all_passages) > 0:
                logger.info("  Processed %dM passages so far...", len(all_passages) // 1_000_000)

        logger.info("Total passages collected: %d", len(all_passages))
        index = BM25Index()
        index.build(all_passages)
        index.save(bm25_index_dir)
        logger.info("BM25 index saved to %s", bm25_index_dir)

    # --- Step 2: Mine hard negatives ---
    seen_queries = TripletWriter.load_seen_query_ids(triplets_file)
    logger.info("Resuming mining. Already done: %d queries.", len(seen_queries))

    with TripletWriter(triplets_file, flush_every=100) as writer:
        stats = mine_hard_negatives(
            records=iter_msmarco_stream(cfg, split=cfg.data.split_train),
            index=index,
            writer=writer,
            seen_queries=seen_queries,
            n_hard_negatives=cfg.bm25.n_hard_negatives,
            bm25_top_k=cfg.bm25.top_k_retrieve,
        )

    logger.info("Phase 2 complete. Stats: %s", stats)
    logger.info("Triplets written to: %s", triplets_file)


if __name__ == "__main__":
    main()
