from omegaconf import DictConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ColBERTReranker:
    """
    Pipeline A re-ranker: ColBERT v2 via Ragatouille.

    Takes top-1000 candidates from Stage 1 and re-ranks to top-k (default 50).
    ColBERT uses late interaction (MaxSim) — faster than full cross-attention.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from ragatouille import RAGPretrainedModel
        logger.info("Loading ColBERT v2 from %s...", self.cfg.inference.colbert_model)
        self._model = RAGPretrainedModel.from_pretrained(self.cfg.inference.colbert_model)
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

        reranked = self._model.rerank(
            query=query,
            documents=passage_texts,
            k=top_k,
        )
        return [{"passage": r["content"], "score": r["score"]} for r in reranked]
