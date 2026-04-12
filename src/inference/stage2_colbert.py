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
        if torch.cuda.is_available():
            self._model.half()  # fp16 for ~2x throughput on GPU
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

        encode_batch = self.cfg.inference.colbert_encode_batch_size
        query_embeddings = self._model.encode(
            [query],
            is_query=True,
            batch_size=encode_batch,
            show_progress_bar=False,
        )
        doc_embeddings = self._model.encode(
            passage_texts,
            is_query=False,
            batch_size=encode_batch,
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

    def rerank_batch(
        self,
        queries: list[str],
        candidates_batch: list[list[dict]],
        top_k: int | None = None,
    ) -> list[list[dict]]:
        """
        Re-rank a batch of queries.

        Query vecs are encoded together in one forward pass — queries are typically
        short (a few dozen tokens), so encoding a batch of 32 together is fast and
        uses little VRAM.  Doc encoding is sequential per-query (1000 docs at a time,
        processed in mini-batches of colbert_encode_batch_size) to keep peak VRAM
        low instead of loading all doc embeddings for all queries at once.

        Args:
            queries:          list of original query strings.
            candidates_batch: list of candidate lists, one per query.
            top_k:            results to return per query.

        Returns:
            list of lists of {"passage": str, "score": float}, one per query.
        """
        self._load()
        top_k = top_k or self.cfg.inference.colbert_top_k
        encode_batch = self.cfg.inference.colbert_encode_batch_size

        # Encode all queries in one forward pass (queries are short, VRAM cost is low)
        query_embeddings = self._model.encode(
            queries,
            is_query=True,
            batch_size=encode_batch,
            show_progress_bar=False,
        )

        results = []
        for query_emb, candidates in zip(query_embeddings, candidates_batch):
            passage_texts = [c["passage"] for c in candidates]
            # Per-query doc encode: up to 1000 docs, in mini-batches of encode_batch
            doc_embeddings = self._model.encode(
                passage_texts,
                is_query=False,
                batch_size=encode_batch,
                show_progress_bar=False,
            )
            doc_ids = [list(range(len(passage_texts)))]
            reranked = rank.rerank(
                documents_ids=doc_ids,
                queries_embeddings=[query_emb],
                documents_embeddings=[doc_embeddings],
            )
            top = reranked[0][:top_k]
            results.append([
                {"passage": passage_texts[r["id"]], "score": r["score"]}
                for r in top
            ])
        return results
