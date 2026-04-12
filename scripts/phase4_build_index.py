"""
Phase 4: Build FAISS Index

Encodes all ~8.8M MS MARCO v1.1 passages using the fine-tuned bi-encoder
and builds a FAISS index for dense retrieval in Phase 5 and 6.

Saves:
    - data/index/passages.faiss       — FAISS index (fine-tuned model)
    - data/index/passage_store.pkl    — passage text store (shared with pretrained index)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.data.beir_loader import iter_beir_corpus
from src.inference.stage1_dense import DenseRetriever

logger = get_logger("phase4")


def main():
    cfg = load_config()
    logger.info("Environment: %s", cfg.environment)

    faiss_path = Path(cfg.paths.faiss_index_path)
    if faiss_path.exists():
        logger.info("FAISS index already exists at %s — skipping build.", faiss_path)
        return

    logger.info("Loading fine-tuned bi-encoder from %s", cfg.paths.best_model_dir)
    retriever = DenseRetriever.from_config(cfg)

    # Index all 8.8M passages from BeIR/msmarco corpus.
    # This is the standard collection used in published MS MARCO benchmarks,
    # making our MRR@10 and Recall@100 directly comparable to reported results.
    logger.info("Streaming BeIR/msmarco corpus (~8.8M passages)...")
    all_passages = []
    seen: set[str] = set()
    for rec in tqdm(iter_beir_corpus(cfg), desc="Streaming BeIR corpus", unit="passage", leave=True):
        text = rec["text"]
        if text not in seen:
            seen.add(text)
            all_passages.append(text)

    logger.info("Total unique passages: %d", len(all_passages))
    retriever.build_index(all_passages)
    retriever.save()
    logger.info("FAISS index saved to %s", cfg.paths.faiss_index_path)
    logger.info("Passage store saved to %s", cfg.paths.passage_store_path)


if __name__ == "__main__":
    main()
