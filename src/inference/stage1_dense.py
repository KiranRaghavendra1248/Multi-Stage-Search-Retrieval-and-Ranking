import pickle
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig

from src.training.bi_encoder import BiEncoder
from src.indexing.faiss_index import build_faiss_index, save_faiss_index, load_faiss_index, search_faiss
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DenseRetriever:
    """
    Stage 1: encode passages with the fine-tuned bi-encoder and retrieve top-k via FAISS.

    Usage:
        retriever = DenseRetriever.from_config(cfg)
        retriever.build_index(passages)          # once, then save
        results = retriever.retrieve(query, top_k=1000)
    """

    def __init__(self, model: BiEncoder, cfg: DictConfig):
        self.model = model
        self.cfg = cfg
        self._index = None
        self._passages: list[str] = []
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "DenseRetriever":
        model = BiEncoder.load(cfg.paths.best_model_dir)
        return cls(model, cfg)

    def build_index(self, passages: list[str], batch_size: int = 512) -> None:
        logger.info("Encoding %d passages for FAISS index...", len(passages))
        self._passages = passages
        embs = self.model.encode(passages, batch_size=batch_size, device=self._device)
        embs = embs.astype(np.float32)
        self._index = build_faiss_index(embs, self.cfg)

    def save(self) -> None:
        save_faiss_index(self._index, self.cfg.paths.faiss_index_path)
        with open(self.cfg.paths.passage_store_path, "wb") as f:
            pickle.dump(self._passages, f)
        logger.info("DenseRetriever index and passage store saved.")

    def load(self) -> None:
        self._index = load_faiss_index(self.cfg.paths.faiss_index_path, self.cfg)
        with open(self.cfg.paths.passage_store_path, "rb") as f:
            self._passages = pickle.load(f)
        logger.info("DenseRetriever loaded. Passages: %d", len(self._passages))

    def retrieve(self, query: str, top_k: int = 1000) -> list[dict]:
        """
        Encode query and retrieve top-k passages.

        Args:
            query: HyDE-generated hypothetical passage (or raw query as fallback).
            top_k: number of candidates to return.

        Returns:
            list of {"passage": str, "score": float}
        """
        q_emb = self.model.encode([query], batch_size=1, device=self._device).astype(np.float32)
        indices, scores = search_faiss(self._index, q_emb, top_k=top_k)

        seen = set()
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            passage = self._passages[idx]
            if passage in seen:
                continue
            seen.add(passage)
            results.append({"passage": passage, "score": float(score)})
        return results
