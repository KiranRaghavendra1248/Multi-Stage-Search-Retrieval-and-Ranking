import numpy as np
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
from omegaconf import DictConfig

from src.data.ms_marco_loader import load_msmarco_stream
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def evaluate_recall(
    model: nn.Module,
    cfg: DictConfig,
    device: str = "cpu",
    k: int = 100,
    max_dev_queries: int = 1000,
) -> float:
    """
    Compute Recall@k on the MS MARCO dev set.

    Strategy:
    1. Load dev records (capped at max_dev_queries for speed during training).
    2. Collect all unique passage texts → encode → build FAISS flat index.
    3. For each dev query, retrieve top-k and check if gold passage is in results.

    Returns Recall@k (fraction of queries where gold is in top-k).
    """
    _model = model.module if isinstance(model, nn.DataParallel) else model
    _model.eval()

    logger.info("Loading dev set (cap=%d)...", max_dev_queries)
    dev_cfg = cfg.copy()
    dev_cfg.data.sample_cap = max_dev_queries
    dev_records = load_msmarco_stream(dev_cfg, split=cfg.data.split_dev)

    # Collect all passages and queries
    all_passages: list[str] = []
    passage_set: set[str] = set()
    queries: list[str] = []
    gold_passages: list[str] = []

    for rec in dev_records:
        if rec["positive_passage"] is None:
            continue
        queries.append(rec["query"])
        gold_passages.append(rec["positive_passage"])
        for pt in rec["passages"].get("passage_text", []):
            if pt not in passage_set:
                all_passages.append(pt)
                passage_set.add(pt)
        if rec["positive_passage"] not in passage_set:
            all_passages.append(rec["positive_passage"])
            passage_set.add(rec["positive_passage"])

    if not queries:
        logger.warning("No dev queries with positives found.")
        return 0.0

    logger.info("Encoding %d passages for validation index...", len(all_passages))
    passage_embs = _model.encode(all_passages, batch_size=256, device=device).astype(np.float32)

    # Build flat FAISS index (always flat for validation — no approximation)
    dim = passage_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(passage_embs)

    logger.info("Encoding %d dev queries...", len(queries))
    query_embs = _model.encode(queries, batch_size=256, device=device).astype(np.float32)

    # Retrieve top-k
    _, top_k_indices = index.search(query_embs, k)

    # Compute Recall@k
    hits = 0
    for i, gold in enumerate(gold_passages):
        retrieved = [all_passages[idx] for idx in top_k_indices[i] if idx >= 0]
        if gold in retrieved:
            hits += 1

    recall = hits / len(queries)
    logger.info("Recall@%d = %.4f (%d/%d)", k, recall, hits, len(queries))
    return recall
