import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from omegaconf import DictConfig
from tqdm import tqdm

from src.data.ms_marco_loader import load_msmarco_stream, iter_msmarco_stream
from src.evaluation.metrics import mrr_at_k, recall_at_k
from src.training.bi_encoder import BiEncoder
from src.inference.stage1_dense import DenseRetriever
from src.indexing.faiss_index import build_faiss_index, save_faiss_index, load_faiss_index
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class VariantResult:
    name: str
    mrr_at_10: float
    recall_at_100: float
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
    logger.info("Loading dev set for comparison...")
    dev_records = load_msmarco_stream(cfg, split=cfg.data.split_dev)
    dev_records = [r for r in dev_records if r["positive_passage"]]
    queries = [r["query"] for r in dev_records]
    gold_passages = [r["positive_passage"] for r in dev_records]
    logger.info("Evaluating on %d dev queries with positives.", len(queries))

    results = []

    # --- Lazy load components ---
    from src.indexing.bm25_index import BM25Index
    from src.inference.query_processor import process_query
    from src.inference.hyde import generate_hypothetical_doc
    from src.inference.stage1_dense import DenseRetriever
    from src.inference.stage2_colbert import ColBERTReranker
    from src.inference.stage2_crossencoder import CrossEncoderReranker

    # --- 1. BM25 baseline ---
    logger.info("Variant 1: BM25 baseline")
    bm25 = BM25Index.load(cfg.paths.bm25_index_dir)
    ranked_lists, latencies = [], []
    for q in tqdm(queries, desc="BM25"):
        t0 = time.perf_counter()
        hits = bm25.search(q, top_k=100)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked_lists.append([h[0] for h in hits])
    results.append(VariantResult(
        name="BM25 baseline",
        mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
        recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
        avg_latency_ms=sum(latencies) / len(latencies),
    ))

    # --- 2. Pre-trained MS MARCO bi-encoder (no fine-tuning) ---
    # Benchmarks what the pre-trained model achieves without our training.
    # Key question: does our hard-negative fine-tuning beat this?
    # Uses the same full corpus as variants 3-7 for a fair comparison.
    logger.info("Variant 2: Pre-trained MS MARCO bi-encoder (no fine-tuning)")
    pretrained_model = BiEncoder(cfg.model.pretrained_msmarco_biencoder)
    pretrained_faiss_path = Path(cfg.paths.faiss_index_pretrained_path)
    passage_store_path = Path(cfg.paths.passage_store_path)

    if pretrained_faiss_path.exists() and passage_store_path.exists():
        logger.info("Loading pre-built pretrained FAISS index from %s", pretrained_faiss_path)
        pretrained_index = load_faiss_index(str(pretrained_faiss_path), cfg)
        with open(passage_store_path, "rb") as f:
            all_passages = pickle.load(f)
    else:
        logger.info("Building pretrained FAISS index over full MS MARCO corpus...")
        all_passages = []
        for rec in iter_msmarco_stream(cfg, split=cfg.data.split_train):
            all_passages.extend(rec["passages"].get("passage_text", []))
        import numpy as np
        embs = pretrained_model.encode(all_passages, batch_size=512)
        embs = embs.astype(np.float32)
        pretrained_index = build_faiss_index(embs, cfg)
        save_faiss_index(pretrained_index, str(pretrained_faiss_path))
        logger.info("Pretrained FAISS index saved to %s", pretrained_faiss_path)

    pretrained_retriever = DenseRetriever(model=pretrained_model, cfg=cfg)
    pretrained_retriever._index = pretrained_index
    pretrained_retriever._passages = all_passages

    ranked_lists, latencies = [], []
    for query, gold in tqdm(zip(queries, gold_passages), total=len(queries), desc="Pretrained bi-enc"):
        t0 = time.perf_counter()
        stage1 = pretrained_retriever.retrieve(query, top_k=1000)
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked_lists.append([r["passage"] for r in stage1[:10]])
    results.append(VariantResult(
        name="Pre-trained MS MARCO bi-encoder",
        mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
        recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
        avg_latency_ms=sum(latencies) / len(latencies),
    ))

    # --- Load our fine-tuned dense retriever (shared by variants 3–7) ---
    retriever = DenseRetriever.from_config(cfg)
    retriever.load()
    colbert = ColBERTReranker(cfg)
    cross_enc = CrossEncoderReranker(cfg)

    def _run_variant(name: str, use_rewriting: bool, reranker=None) -> VariantResult:
        ranked_lists, latencies = [], []
        _cfg = cfg.copy()
        _cfg.inference.query_rewriting = use_rewriting

        for query, gold in tqdm(zip(queries, gold_passages), total=len(queries), desc=name):
            t0 = time.perf_counter()

            processed_q = process_query(query, _cfg)
            hyde_doc = generate_hypothetical_doc(processed_q, _cfg)
            stage1 = retriever.retrieve(hyde_doc, top_k=1000)

            if reranker is None:
                final = stage1[:10]
            else:
                final = reranker.rerank(query, stage1)

            latencies.append((time.perf_counter() - t0) * 1000)
            ranked_lists.append([r["passage"] for r in final])

        return VariantResult(
            name=name,
            mrr_at_10=mrr_at_k(ranked_lists, gold_passages, k=10),
            recall_at_100=recall_at_k(ranked_lists, gold_passages, k=100),
            avg_latency_ms=sum(latencies) / len(latencies),
        )

    results.append(_run_variant("Fine-tuned bi-encoder only", use_rewriting=False, reranker=None))
    results.append(_run_variant("Pipeline A: ColBERT", use_rewriting=False, reranker=colbert))
    results.append(_run_variant("Pipeline B: Cross-Encoder", use_rewriting=False, reranker=cross_enc))
    results.append(_run_variant("Pipeline A + Query Rewriting", use_rewriting=True, reranker=colbert))
    results.append(_run_variant("Pipeline B + Query Rewriting", use_rewriting=True, reranker=cross_enc))

    return results


def print_comparison_table(results: list[VariantResult]) -> str:
    header = f"{'Variant':<40} {'MRR@10':>8} {'Recall@100':>12} {'Latency(ms)':>14}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r.name:<40} {r.mrr_at_10:>8.4f} {r.recall_at_100:>12.4f} {r.avg_latency_ms:>14.1f}"
        )
    table = "\n".join(rows)
    print(table)
    return table
