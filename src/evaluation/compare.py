import json
import pickle
import re
import time
import torch
from dataclasses import dataclass, asdict
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.beir_loader import load_beir_dev_eval
from src.evaluation.metrics import mrr_at_k, recall_at_k, ndcg_at_k
from src.training.bi_encoder import BiEncoder
from src.inference.stage1_dense import DenseRetriever
from src.indexing.faiss_index import load_faiss_index
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class VariantResult:
    name: str
    mrr_at_10: float
    recall_at_100: float
    ndcg_at_10: float
    avg_latency_ms: float


def run_comparison(cfg: DictConfig) -> list[VariantResult]:
    """
    Run all 7 pipeline variants on the MS MARCO dev set and return results.

    Variants:
        1. BM25 only
        2. Pre-trained MS MARCO bi-encoder (no fine-tuning) — upper bound benchmark
        3. Our fine-tuned bi-encoder only (Stage 1)
        4. Pipeline A: Fine-tuned bi-encoder + ColBERT (no query rewriting)
        5. Pipeline B: Fine-tuned bi-encoder + Cross-Encoder (no query rewriting)
        6. Pipeline A + query rewriting
        7. Pipeline B + query rewriting
    """
    logger.info("Loading BeIR dev queries and gold passages...")
    queries, gold_passages = load_beir_dev_eval(cfg)
    logger.info("Evaluating on %d dev queries.", len(queries))

    results = []

    # --- Lazy load components ---
    from src.indexing.bm25_index import BM25Index
    from src.inference.query_processor import process_query
    from src.inference.hyde import generate_hypothetical_doc
    from src.inference.stage1_dense import DenseRetriever
    from src.inference.stage2_colbert import ColBERTReranker
    from src.inference.stage2_crossencoder import CrossEncoderReranker

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    def _save_variant_result(result: VariantResult) -> None:
        """Save a single variant result to results/<slug>.json immediately after it finishes."""
        slug = re.sub(r"[^a-z0-9]+", "_", result.name.lower()).strip("_")
        out = results_dir / f"{slug}.json"
        with open(out, "w") as f:
            json.dump(asdict(result), f, indent=2)
        logger.info("[Result] %s — MRR@10: %.4f, NDCG@10: %.4f, Recall@100: %.4f, Latency: %.1f ms → saved to %s",
                    result.name, result.mrr_at_10, result.ndcg_at_10, result.recall_at_100, result.avg_latency_ms, out)

    # --- 1. BM25 baseline ---
    logger.info("Variant 1: BM25 baseline")
    bm25 = BM25Index.load(cfg.paths.bm25_index_dir)
    ranked_lists, latencies = [], []
    for q in tqdm(queries, desc="V1 BM25 baseline", unit="query", leave=True):
        t0 = time.perf_counter()
        hits = bm25.search(q, top_k=100)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked_lists.append([h[0] for h in hits])
    v1 = VariantResult(
        name="BM25 baseline",
        mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
        recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
        ndcg_at_10=ndcg_at_k(ranked_lists, gold_passages, k=10),
        avg_latency_ms=sum(latencies) / len(latencies),
    )
    _save_variant_result(v1)
    results.append(v1)

    # --- 2. e5-large-unsupervised (no fine-tuning) ---
    # Benchmarks what e5-large-unsupervised achieves straight from HuggingFace, without our training.
    # Same architecture and prefixes as V3 — isolates the contribution of our fine-tuning.
    # Key question: does our dense-teacher hard-negative training beat the off-the-shelf model?
    # FAISS index built in Phase 4 over the same 8.8M passage corpus.
    logger.info("Variant 2: e5-large-unsupervised (no fine-tuning)")
    pretrained_model = BiEncoder(
        cfg.model.pretrained_msmarco_biencoder,
        query_prefix=cfg.model.get("bi_encoder_query_prefix", ""),
        passage_prefix=cfg.model.get("bi_encoder_passage_prefix", ""),
    )
    pretrained_faiss_path = Path(cfg.paths.faiss_index_pretrained_path)
    passage_store_path = Path(cfg.paths.passage_store_path)

    logger.info("Loading pretrained FAISS index from %s", pretrained_faiss_path)
    pretrained_index = load_faiss_index(str(pretrained_faiss_path), cfg)
    with open(passage_store_path, "rb") as f:
        all_passages = pickle.load(f)

    pretrained_retriever = DenseRetriever(model=pretrained_model, cfg=cfg)
    pretrained_retriever._index = pretrained_index
    pretrained_retriever._passages = all_passages

    ranked_lists, latencies = [], []
    for query, gold in tqdm(zip(queries, gold_passages), total=len(queries), desc="V2 e5-large-unsupervised (no FT)", unit="query", leave=True):
        t0 = time.perf_counter()
        stage1 = pretrained_retriever.retrieve(query, top_k=1000)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked_lists.append([r["passage"] for r in stage1])
    v2 = VariantResult(
        name="e5-large-unsupervised (no fine-tuning)",
        mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
        recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
        ndcg_at_10=ndcg_at_k(ranked_lists, gold_passages, k=10),
        avg_latency_ms=sum(latencies) / len(latencies),
    )
    _save_variant_result(v2)
    results.append(v2)

    # --- Load our fine-tuned dense retriever (shared by variants 3–7) ---
    retriever = DenseRetriever.from_config(cfg)
    retriever.load()
    colbert = ColBERTReranker(cfg)
    cross_enc = CrossEncoderReranker(cfg)

    def _log_vram(label: str) -> None:
        """Log current and peak GPU memory after each variant."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 ** 3
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            logger.info("[VRAM] %s — allocated: %.2f GB, peak: %.2f GB", label, alloc, peak)

    def _run_variant(name: str, use_rewriting: bool, reranker=None) -> VariantResult:
        """
        Evaluate one pipeline variant using batched GPU processing.

        Outer loop: batches of eval_batch_size (32) queries.
        - FAISS retrieval: all 32 queries encoded and searched at once.
        - Reranking: rerank_batch() handles the fan-out internally.
        - HyDE/process_query: stays per-query (sequential LLM calls).
        Latency is measured over the GPU-heavy portion only (retrieve + rerank).
        """
        ranked_lists: list[list[str]] = []
        latencies: list[float] = []
        _cfg = cfg.copy()
        _cfg.inference.query_rewriting = use_rewriting
        batch_size: int = cfg.inference.eval_batch_size

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for start in tqdm(range(0, len(queries), batch_size), desc=name, unit="batch", leave=True):
            batch_q = queries[start : start + batch_size]
            batch_gold = gold_passages[start : start + batch_size]  # noqa: F841 — kept for symmetry

            # HyDE and query rewriting are LLM/CPU calls — stay per-query
            processed_qs = [process_query(q, _cfg) for q in batch_q]
            hyde_docs = [generate_hypothetical_doc(q, _cfg) for q in processed_qs]

            # GPU-heavy: batch FAISS retrieval
            t0 = time.perf_counter()
            stage1_batch = retriever.retrieve_batch(hyde_docs, top_k=1000)

            if reranker is None:
                final_batch = [s[:10] for s in stage1_batch]
            else:
                final_batch = reranker.rerank_batch(batch_q, stage1_batch)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_query_ms = elapsed_ms / len(batch_q)
            latencies.extend([per_query_ms] * len(batch_q))
            for final in final_batch:
                ranked_lists.append([r["passage"] for r in final])

        _log_vram(name)
        result = VariantResult(
            name=name,
            mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
            recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
            ndcg_at_10=ndcg_at_k(ranked_lists, gold_passages, k=10),
            avg_latency_ms=sum(latencies) / len(latencies),
        )
        _save_variant_result(result)
        return result

    results.append(_run_variant("Fine-tuned bi-encoder only", use_rewriting=False, reranker=None))
    results.append(_run_variant("Pipeline A: ColBERT", use_rewriting=False, reranker=colbert))
    results.append(_run_variant("Pipeline B: Cross-Encoder", use_rewriting=False, reranker=cross_enc))
    results.append(_run_variant("Pipeline A + Query Rewriting", use_rewriting=True, reranker=colbert))
    results.append(_run_variant("Pipeline B + Query Rewriting", use_rewriting=True, reranker=cross_enc))

    # --- RRF variants: BM25 + dense fusion only (no reranking) ---
    for rrf_name, rrf_retriever in [
        ("RRF: BM25 + Fine-tuned", retriever),
        ("RRF: BM25 + Pretrained", pretrained_retriever),
    ]:
        logger.info("Variant: %s", rrf_name)
        ranked_lists, latencies = [], []
        batch_size = cfg.inference.eval_batch_size

        for start in tqdm(range(0, len(queries), batch_size), desc=rrf_name, unit="batch", leave=True):
            batch_q = queries[start : start + batch_size]

            t0 = time.perf_counter()
            bm25_batch = [bm25.search(q, top_k=1000) for q in batch_q]
            dense_batch = rrf_retriever.retrieve_batch(batch_q, top_k=1000)
            fused_batch = [
                _rrf_fuse(
                    [h[0] for h in bm25_hits],
                    [r["passage"] for r in dense_hits],
                    top_k=1000,
                )
                for bm25_hits, dense_hits in zip(bm25_batch, dense_batch)
            ]
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_query_ms = elapsed_ms / len(batch_q)
            latencies.extend([per_query_ms] * len(batch_q))
            for fused in fused_batch:
                ranked_lists.append([r["passage"] for r in fused])

        result = VariantResult(
            name=rrf_name,
            mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
            recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
            ndcg_at_10=ndcg_at_k(ranked_lists, gold_passages, k=10),
            avg_latency_ms=sum(latencies) / len(latencies),
        )
        _save_variant_result(result)
        results.append(result)

    # --- RRF variants: BM25 + dense fusion → ColBERT(100) → Cross-Encoder(10) ---
    for rrf_name, rrf_retriever in [
        ("RRF: BM25 + Fine-tuned → ColBERT → Cross-Encoder", retriever),
        ("RRF: BM25 + Pretrained → ColBERT → Cross-Encoder", pretrained_retriever),
    ]:
        logger.info("Variant: %s", rrf_name)
        ranked_lists, latencies = [], []
        batch_size = cfg.inference.eval_batch_size

        for start in tqdm(range(0, len(queries), batch_size), desc=rrf_name, unit="batch", leave=True):
            batch_q = queries[start : start + batch_size]

            t0 = time.perf_counter()

            # BM25 top-1000 per query
            bm25_batch = [bm25.search(q, top_k=1000) for q in batch_q]

            # Dense top-1000 per query
            dense_batch = rrf_retriever.retrieve_batch(batch_q, top_k=1000)

            # RRF fusion → top-1000 candidates per query
            fused_batch = [
                _rrf_fuse(
                    [h[0] for h in bm25_hits],
                    [r["passage"] for r in dense_hits],
                    top_k=1000,
                )
                for bm25_hits, dense_hits in zip(bm25_batch, dense_batch)
            ]

            # ColBERT: 1000 → 100
            colbert_batch = colbert.rerank_batch(batch_q, fused_batch, top_k=100)

            # Cross-Encoder: 100 → 10
            final_batch = cross_enc.rerank_batch(batch_q, colbert_batch, top_k=10)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_query_ms = elapsed_ms / len(batch_q)
            latencies.extend([per_query_ms] * len(batch_q))
            for final in final_batch:
                ranked_lists.append([r["passage"] for r in final])

        result = VariantResult(
            name=rrf_name,
            mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
            recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
            ndcg_at_10=ndcg_at_k(ranked_lists, gold_passages, k=10),
            avg_latency_ms=sum(latencies) / len(latencies),
        )
        _save_variant_result(result)
        results.append(result)

    return results


def _rrf_fuse(
    bm25_passages: list[str],
    dense_passages: list[str],
    top_k: int = 1000,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion of BM25 and dense ranked lists.

    score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_dense(d))
    Passages missing from a list are assigned rank = len(list) + 1.

    Returns:
        list of {"passage": str, "score": float} sorted by RRF score descending
    """
    bm25_rank  = {p: i + 1 for i, p in enumerate(bm25_passages)}
    dense_rank = {p: i + 1 for i, p in enumerate(dense_passages)}

    bm25_default  = len(bm25_passages) + 1
    dense_default = len(dense_passages) + 1

    # dict.fromkeys preserves insertion order and deduplicates
    all_passages = dict.fromkeys(bm25_passages + dense_passages)
    scored = [
        {
            "passage": p,
            "score": (1.0 / (k + bm25_rank.get(p, bm25_default)))
                   + (1.0 / (k + dense_rank.get(p, dense_default))),
        }
        for p in all_passages
    ]
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def print_comparison_table(results: list[VariantResult]) -> str:
    header = f"{'Variant':<40} {'MRR@10':>8} {'NDCG@10':>9} {'Recall@100':>12} {'Latency(ms)':>14}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r.name:<40} {r.mrr_at_10:>8.4f} {r.ndcg_at_10:>9.4f} {r.recall_at_100:>12.4f} {r.avg_latency_ms:>14.1f}"
        )
    table = "\n".join(rows)
    print(table)
    return table
