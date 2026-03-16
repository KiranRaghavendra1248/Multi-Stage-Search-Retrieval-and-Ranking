import pickle
from pathlib import Path
import bm25s
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    """
    Thin wrapper around bm25s.BM25.

    Usage:
        Phase 1 (in-memory):
            idx = BM25Index()
            idx.build(passages)
            results = idx.search("query text", top_k=10)

        Phase 2 (build once, reuse across restarts):
            idx = BM25Index()
            idx.build(passages)
            idx.save("data/index/bm25")
            # --- later / after crash ---
            idx = BM25Index.load("data/index/bm25")
            results = idx.search("query text", top_k=100)
    """

    def __init__(self):
        self._model: bm25s.BM25 | None = None
        self._passages: list[str] = []

    def build(self, passages: list[str]) -> None:
        logger.info("Building BM25 index over %d passages...", len(passages))
        self._passages = passages
        tokenized = [_tokenize(p) for p in passages]
        self._model = bm25s.BM25()
        self._model.index(tokenized)
        logger.info("BM25 index built.")

    def save(self, path: str) -> None:
        dirpath = Path(path)
        dirpath.mkdir(parents=True, exist_ok=True)
        self._model.save(str(dirpath / "bm25_model"))
        with open(dirpath / "passages.pkl", "wb") as f:
            pickle.dump(self._passages, f)
        logger.info("BM25 index saved to %s", dirpath)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        dirpath = Path(path)
        instance = cls()
        instance._model = bm25s.BM25.load(str(dirpath / "bm25_model"), load_corpus=False)
        with open(dirpath / "passages.pkl", "rb") as f:
            instance._passages = pickle.load(f)
        logger.info("BM25 index loaded from %s (%d passages)", dirpath, len(instance._passages))
        return instance

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Returns list of (passage_text, bm25_score) sorted by score descending."""
        if self._model is None:
            raise RuntimeError("Index not built or loaded.")
        # bm25s.tokenize expects raw strings, not pre-tokenized lists
        tokenized = bm25s.tokenize([query], show_progress=False)
        results, scores = self._model.retrieve(
            tokenized,
            corpus=self._passages,
            k=min(top_k, len(self._passages)),
        )
        # results[0] and scores[0] are arrays for the first (only) query
        return list(zip(results[0].tolist(), scores[0].tolist()))

    def search_with_ids(
        self, query: str, top_k: int = 100
    ) -> list[tuple[int, str, float]]:
        """Returns list of (passage_index, passage_text, score)."""
        if self._model is None:
            raise RuntimeError("Index not built or loaded.")
        tokenized = bm25s.tokenize([query], show_progress=False)
        results, scores = self._model.retrieve(
            tokenized,
            corpus=self._passages,
            k=min(top_k, len(self._passages)),
        )
        indices = [self._passages.index(p) for p in results[0].tolist()]
        return list(zip(indices, results[0].tolist(), scores[0].tolist()))

    def __len__(self) -> int:
        return len(self._passages)
