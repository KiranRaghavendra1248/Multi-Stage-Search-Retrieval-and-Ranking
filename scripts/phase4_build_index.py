"""
Phase 4: Build FAISS Indexes

Encodes all ~8.8M MS MARCO passages using both the fine-tuned bi-encoder and
the pretrained MS MARCO bi-encoder, building two FAISS indexes for Phase 6 eval.

Saves:
    - data/index/passages.faiss           — FAISS index (fine-tuned model)
    - data/index/passages_pretrained.faiss — FAISS index (pretrained model, Variant 2)
    - data/index/passage_store.pkl         — passage text store (shared by both indexes)
"""
import pickle
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.data.beir_loader import iter_beir_corpus
from src.inference.stage1_dense import DenseRetriever
from src.training.bi_encoder import BiEncoder
from src.indexing.faiss_index import build_faiss_index, save_faiss_index

logger = get_logger("phase4")


def main():
    cfg = load_config()
    logger.info("Environment: %s", cfg.environment)

    faiss_path = Path(cfg.paths.faiss_index_path)
    pretrained_faiss_path = Path(cfg.paths.faiss_index_pretrained_path)
    passage_store_path = Path(cfg.paths.passage_store_path)

    both_exist = faiss_path.exists() and pretrained_faiss_path.exists() and passage_store_path.exists()
    if both_exist:
        logger.info("Both FAISS indexes already exist — skipping build.")
        return

    # Stream and deduplicate all 8.8M passages (shared by both indexes)
    logger.info("Streaming BeIR/msmarco corpus (~8.8M passages)...")
    all_passages = []
    seen: set[str] = set()
    for rec in tqdm(iter_beir_corpus(cfg), desc="Streaming BeIR corpus", unit="passage", leave=True):
        text = rec["text"]
        if text not in seen:
            seen.add(text)
            all_passages.append(text)
    logger.info("Total unique passages: %d", len(all_passages))

    # Save passage store (shared by both indexes)
    passage_store_path.parent.mkdir(parents=True, exist_ok=True)
    with open(passage_store_path, "wb") as f:
        pickle.dump(all_passages, f)
    logger.info("Passage store saved to %s", passage_store_path)

    # --- Fine-tuned bi-encoder FAISS index ---
    if faiss_path.exists():
        logger.info("Fine-tuned FAISS index already exists at %s — skipping.", faiss_path)
    else:
        logger.info("Building fine-tuned FAISS index from %s", cfg.paths.best_model_dir)
        retriever = DenseRetriever.from_config(cfg)
        retriever._passages = all_passages
        retriever.build_index(all_passages)
        save_faiss_index(retriever._index, str(faiss_path))
        logger.info("Fine-tuned FAISS index saved to %s", faiss_path)

    # --- Pretrained MS MARCO bi-encoder FAISS index (Variant 2 benchmark) ---
    if pretrained_faiss_path.exists():
        logger.info("Pretrained FAISS index already exists at %s — skipping.", pretrained_faiss_path)
    else:
        logger.info("Building pretrained FAISS index using %s", cfg.model.pretrained_msmarco_biencoder)
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pretrained_model = BiEncoder(cfg.model.pretrained_msmarco_biencoder)
        embs = pretrained_model.encode(all_passages, batch_size=512, device=device)
        embs = embs.astype(np.float32)
        pretrained_index = build_faiss_index(embs, cfg)
        save_faiss_index(pretrained_index, str(pretrained_faiss_path))
        logger.info("Pretrained FAISS index saved to %s", pretrained_faiss_path)


if __name__ == "__main__":
    main()
