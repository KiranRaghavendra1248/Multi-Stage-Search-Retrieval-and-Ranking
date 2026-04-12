import torch
from omegaconf import DictConfig
from sentence_transformers import CrossEncoder
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
        logger.info("Loading CrossEncoder %s...", self.cfg.model.cross_encoder)
        self._model = CrossEncoder(
            self.cfg.model.cross_encoder,
            max_length=512,
        )
        if torch.cuda.is_available():
            self._model.model.half()  # fp16 for ~2x throughput on GPU
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

    def rerank_batch(
        self,
        queries: list[str],
        candidates_batch: list[list[dict]],
        top_k: int | None = None,
    ) -> list[list[dict]]:
        """
        Re-rank a batch of queries with one predict() call.

        Flattens N queries × 1000 candidates = N×1000 pairs into a single list,
        then calls predict() once.  sentence-transformers internally processes
        mini-batches of crossencoder_batch_size (8) pairs per forward pass, so
        peak VRAM stays low regardless of the total pair count.

        Args:
            queries:          list of original query strings.
            candidates_batch: list of candidate lists, one per query.
            top_k:            results to return per query.

        Returns:
            list of lists of {"passage": str, "score": float}, one per query.
        """
        self._load()
        top_k = top_k or self.cfg.inference.crossencoder_top_k

        all_pairs: list[tuple[str, str]] = []
        all_passages: list[list[str]] = []
        sizes: list[int] = []
        for query, candidates in zip(queries, candidates_batch):
            passages = [c["passage"] for c in candidates]
            all_passages.append(passages)
            all_pairs.extend([(query, p) for p in passages])
            sizes.append(len(passages))

        # One predict() call across all pairs; sentence-transformers mini-batches
        # internally at crossencoder_batch_size (8 pairs per forward pass)
        all_scores = self._model.predict(
            all_pairs,
            batch_size=self.cfg.inference.crossencoder_batch_size,
            show_progress_bar=False,
        )

        results: list[list[dict]] = []
        offset = 0
        for passages, size in zip(all_passages, sizes):
            scores = all_scores[offset : offset + size]
            ranked = sorted(zip(passages, scores.tolist()), key=lambda x: -x[1])[:top_k]
            results.append([{"passage": t, "score": s} for t, s in ranked])
            offset += size
        return results
