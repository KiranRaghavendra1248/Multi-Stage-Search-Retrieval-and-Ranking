from omegaconf import DictConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Pipeline B re-ranker: Cross-Encoder MiniLM.

    Takes top-1000 candidates from Stage 1 and re-ranks to top-k (default 10).
    Cross-encoder jointly encodes (query, passage) pairs — highest precision,
    higher latency than ColBERT.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder %s...", self.cfg.model.cross_encoder)
        self._model = CrossEncoder(self.cfg.model.cross_encoder)
        logger.info("CrossEncoder loaded.")

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Re-rank candidate passages.

        Args:
            query:      original query string
            candidates: list of {"passage": str, "score": float} from Stage 1
            top_k:      number of results to return (defaults to cfg.inference.crossencoder_top_k)

        Returns:
            list of {"passage": str, "score": float} sorted by cross-encoder score
        """
        self._load()
        top_k = top_k or self.cfg.inference.crossencoder_top_k
        passage_texts = [c["passage"] for c in candidates]

        pairs = [(query, p) for p in passage_texts]
        scores = self._model.predict(
            pairs,
            batch_size=self.cfg.inference.crossencoder_batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(passage_texts, scores.tolist()),
            key=lambda x: -x[1],
        )[:top_k]

        return [{"passage": text, "score": score} for text, score in ranked]
