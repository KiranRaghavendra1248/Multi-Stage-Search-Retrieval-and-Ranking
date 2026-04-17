"""
Phase 2: Hard Negative Mining (Vast.ai)

Teacher options (set via cfg.mining.teacher):

  "bm25" (default):
    1. Stream BeIR/msmarco corpus (~8.8M passages) → build bm25s index → save to disk
    2. Stream train queries → mine hard negatives by BM25 retrieval → write JSONL
    No positive-aware filtering (BM25 scores are unnormalized).

  "intfloat/e5-large-unsupervised":
    1. Build BM25 index as above (still needed for eval later)
    2. Load TensorRTDenseTeacher, encode 8.8M passages, build dense FAISS index
    3. Stream train queries → mine via dense retrieval + positive-aware filtering → write JSONL

  "intfloat/e5-mistral-7b-instruct":
    Same as above but uses VLLMDenseTeacher (vLLM HTTP server at port 8001).
    Start with: make start-vllm-teacher  (separate process, stop before make start-vllm)

All modes support crash-safe resume: re-run picks up from where it left off.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.data.ms_marco_loader import iter_msmarco_stream
from src.data.beir_loader import iter_beir_corpus
from src.indexing.bm25_index import BM25Index
from src.mining.hard_negative_miner import mine_hard_negatives
from src.mining.triplet_writer import TripletWriter

logger = get_logger("phase2")


def main():
    cfg = load_config()
    logger.info("Environment: %s", cfg.environment)

    bm25_index_dir = cfg.paths.bm25_index_dir
    triplets_file = cfg.paths.triplets_file
    teacher_name: str = cfg.mining.teacher

    # --- Step 1: Build BM25 index (always — needed at eval time regardless of teacher) ---
    if Path(bm25_index_dir).exists() and (Path(bm25_index_dir) / "passages.pkl").exists():
        logger.info("BM25 index already exists at %s — loading.", bm25_index_dir)
        bm25_index = BM25Index.load(bm25_index_dir)
    else:
        logger.info("Building BM25 index from BeIR/msmarco corpus (~8.8M passages)...")
        all_passages: list[str] = []
        for rec in iter_beir_corpus(cfg):
            all_passages.append(rec["text"])
            if len(all_passages) % 1_000_000 == 0 and len(all_passages) > 0:
                logger.info("  Collected %dM passages so far...", len(all_passages) // 1_000_000)
        logger.info("Total passages collected: %d", len(all_passages))
        bm25_index = BM25Index()
        bm25_index.build(all_passages)
        bm25_index.save(bm25_index_dir)
        logger.info("BM25 index saved to %s", bm25_index_dir)
        # Free the passages list; BM25 index holds its own reference
        del all_passages

    # --- Step 2: Prepare the mining teacher ---
    if teacher_name == "bm25":
        logger.info("Mining teacher: BM25 (no positive-aware filtering)")
        mining_index = bm25_index
    else:
        from src.mining.dense_teacher import build_dense_teacher

        logger.info("Mining teacher: %s", teacher_name)
        teacher = build_dense_teacher(cfg)

        logger.info("Encoding BeIR corpus with dense teacher and building FAISS index...")
        all_passages = [rec["text"] for rec in iter_beir_corpus(cfg)]
        teacher.build_index(all_passages)
        del all_passages  # free RAM — FAISS holds the index separately
        mining_index = teacher

    # --- Step 3: Mine hard negatives ---
    seen_queries = TripletWriter.load_seen_query_ids(triplets_file)
    logger.info("Resuming mining. Already done: %d queries.", len(seen_queries))

    pa_method = cfg.mining.get("positive_aware_method", None)
    if pa_method and teacher_name == "bm25":
        logger.warning(
            "positive_aware_method=%r is set but teacher=bm25 — skipping (BM25 scores are unnormalized).",
            pa_method,
        )
        pa_method = None

    with TripletWriter(triplets_file, flush_every=100) as writer:
        stats = mine_hard_negatives(
            records=iter_msmarco_stream(cfg, split=cfg.data.split_train),
            index=mining_index,
            writer=writer,
            seen_queries=seen_queries,
            n_hard_negatives=cfg.mining.n_hard_negatives,
            top_k_retrieve=cfg.mining.top_k_retrieve,
            max_triplets=cfg.mining.get("max_triplets", None),
            positive_aware_method=pa_method,
            margin_pos=cfg.mining.get("margin_pos", 0.05),
            perc_pos=cfg.mining.get("perc_pos", 0.95),
        )

    logger.info("Phase 2 complete. Stats: %s", stats)
    logger.info("Triplets written to: %s", triplets_file)


if __name__ == "__main__":
    main()
