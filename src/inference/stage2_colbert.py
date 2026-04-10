import torch
from omegaconf import DictConfig
from pylate import models, rank
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ColBERTReranker:
    """
    Pipeline A re-ranker: ColBERT v2 via PyLate.

    Takes top-1000 candidates from Stage 1 and re-ranks to top-k (default 50).
    ColBERT uses late interaction (MaxSim) — faster than full cross-attention.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading ColBERT v2 from %s (device=%s)...", self.cfg.inference.colbert_model, device)
        self._model = models.ColBERT(
            model_name_or_path=self.cfg.inference.colbert_model,
            device=device,
        )
        logger.info("ColBERT v2 loaded.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank candidate passages.

        Args:
            query:      original query string (NOT the HyDE expansion)
            candidates: list of {"passage": str, "score": float} from Stage 1
            top_k:      number of results to return (defaults to cfg.inference.colbert_top_k)

        Returns:
            list of {"passage": str, "score": float} sorted by ColBERT score
        """
        self._load()
        top_k = top_k or self.cfg.inference.colbert_top_k
        passage_texts = [c["passage"] for c in candidates]

        query_embeddings = self._model.encode(
            [query],
            is_query=True,
            show_progress_bar=False,
        )
        doc_embeddings = self._model.encode(
            passage_texts,
            is_query=False,
            show_progress_bar=False,
        )

        doc_ids = [list(range(len(passage_texts)))]
        results = rank.rerank(
            documents_ids=doc_ids,
            queries_embeddings=query_embeddings,
            documents_embeddings=[doc_embeddings],
        )
        # results[0]: list of {"id": int, "score": float}, sorted desc
        top_results = results[0][:top_k]
        return [
            {"passage": passage_texts[r["id"]], "score": r["score"]}
            for r in top_results
        ]
