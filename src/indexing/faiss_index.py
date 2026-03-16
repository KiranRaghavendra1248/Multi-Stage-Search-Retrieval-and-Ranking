import numpy as np
import faiss
from omegaconf import DictConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_faiss_index(embeddings: np.ndarray, cfg: DictConfig) -> faiss.Index:
    """
    Build a FAISS index from a matrix of L2-normalized embeddings.

    Local:  IndexFlatIP  — exact, no training needed, fine for ≤100k vectors
    Remote: IndexIVFFlat — approximate, fast for 3.2M vectors

    Embeddings must be float32 and L2-normalized (inner product == cosine similarity).
    """
    dim = embeddings.shape[1]
    index_type = cfg.faiss.index_type

    if index_type == "Flat":
        index = faiss.IndexFlatIP(dim)
        logger.info("Building FAISS IndexFlatIP (dim=%d)", dim)
        index.add(embeddings)

    elif index_type == "IVFFlat":
        nlist = cfg.faiss.nlist
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        logger.info("Training FAISS IndexIVFFlat (nlist=%d, dim=%d)...", nlist, dim)
        # Train on a sample if corpus is very large
        train_size = min(len(embeddings), 500_000)
        index.train(embeddings[:train_size])
        logger.info("Adding %d vectors to FAISS index...", len(embeddings))
        index.add(embeddings)
        index.nprobe = cfg.faiss.nprobe

    else:
        raise ValueError(f"Unknown faiss.index_type: {index_type}")

    if cfg.faiss.use_gpu:
        logger.info("Moving FAISS index to all GPUs...")
        index = faiss.index_cpu_to_all_gpus(index)

    logger.info("FAISS index ready. Total vectors: %d", index.ntotal)
    return index


def save_faiss_index(index: faiss.Index, path: str) -> None:
    # GPU indexes must be converted back to CPU before saving
    if hasattr(index, "index"):  # GpuIndex wrapper
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, path)
    logger.info("FAISS index saved to %s", path)


def load_faiss_index(path: str, cfg: DictConfig) -> faiss.Index:
    index = faiss.read_index(path)
    if cfg.faiss.index_type == "IVFFlat":
        index.nprobe = cfg.faiss.nprobe
    if cfg.faiss.use_gpu:
        index = faiss.index_cpu_to_all_gpus(index)
    logger.info("FAISS index loaded from %s (ntotal=%d)", path, index.ntotal)
    return index


def search_faiss(
    index: faiss.Index,
    query_embeddings: np.ndarray,
    top_k: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (indices [N, top_k], scores [N, top_k]).
    query_embeddings: float32, shape [N, dim], L2-normalized.
    """
    scores, indices = index.search(query_embeddings, top_k)
    return indices, scores
