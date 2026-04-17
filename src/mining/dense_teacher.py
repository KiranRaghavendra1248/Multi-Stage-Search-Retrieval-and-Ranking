"""
Dense teacher models for hard negative mining.

Two backends:

  TensorRTDenseTeacher — for encoder models (e.g. intfloat/e5-large-unsupervised)
      Loads via transformers, optionally compiled with torch_tensorrt for speed.
      Falls back to plain PyTorch if torch_tensorrt is unavailable.

  VLLMDenseTeacher — for large autoregressive/decoder embedding models
      (e.g. intfloat/e5-mistral-7b-instruct, 7B+)
      Calls a vLLM HTTP server running with --task embedding.
      Start with: make start-vllm-teacher
      We manually prepend the task instruction prefix to queries before encoding.
      vLLM handles last-token pooling internally.
"""
from __future__ import annotations

import json
import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.indexing.faiss_index import build_faiss_index, search_faiss
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class DenseTeacher(ABC):
    """Common interface for dense teacher models used in hard negative mining."""

    @abstractmethod
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings [N, dim]."""

    @abstractmethod
    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings [N, dim]."""

    def build_index(self, passages: list[str]) -> None:
        """Encode all passages and build a FAISS index for fast retrieval."""
        logger.info("Building dense teacher FAISS index over %d passages...", len(passages))
        self._passages = passages
        # Encode in chunks to avoid OOM on large corpora
        embs = self.encode_passages(passages)
        embs = embs.astype(np.float32)

        # Minimal cfg-like object for build_faiss_index.
        # use_gpu=False: 8.8M × 1024-dim fp32 = ~36GB — never fits on GPU.
        # Mining FAISS stays on CPU; only the encoder model uses GPU.
        class _Cfg:
            class faiss:
                index_type = "IVFFlat"
                nlist = 4096
                nprobe = 64
                use_gpu = False

        self._index = build_faiss_index(embs, _Cfg())
        logger.info("Dense teacher FAISS index built.")

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return top_k (passage_text, score) tuples for a single query."""
        q_emb = self.encode_queries([query]).astype(np.float32)
        indices, scores = search_faiss(self._index, q_emb, top_k=top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self._passages[idx], float(score)))
        return results

    def score_pair(self, query: str, passage: str) -> float:
        """Compute cosine similarity between a query and a passage (dot product of L2-normed vecs)."""
        q_emb = self.encode_queries([query]).astype(np.float32)   # [1, D]
        p_emb = self.encode_passages([passage]).astype(np.float32)  # [1, D]
        return float(np.dot(q_emb[0], p_emb[0]))


# ---------------------------------------------------------------------------
# TensorRTDenseTeacher — encoder models (e5-large-unsupervised, etc.)
# ---------------------------------------------------------------------------

class TensorRTDenseTeacher(DenseTeacher):
    """
    Dense teacher for encoder/autoencoder models (e.g. intfloat/e5-large-unsupervised).

    Uses torch_tensorrt for fast batched encoding when available.
    Falls back to plain PyTorch (identical results, slower) if not installed.

    Prefixes are prepended manually before tokenization — required by e5-family:
        query_prefix   = "query: "
        passage_prefix = "passage: "
    """

    def __init__(
        self,
        model_name: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 512,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading TensorRTDenseTeacher: %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        self._model.eval().to(self._device)

        # Compile with TensorRT for faster batched encoding
        try:
            import torch_tensorrt  # noqa: F401
            self._model = torch.compile(self._model, backend="tensorrt")
            logger.info("TensorRT compilation succeeded for %s", model_name)
        except Exception as e:
            logger.warning("torch_tensorrt not available (%s) — using plain PyTorch.", e)

        self._passages: list[str] = []
        self._index = None

    @staticmethod
    def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_emb / sum_mask

    def _encode(self, texts: list[str], prefix: str, desc: str) -> np.ndarray:
        prefixed = [prefix + t for t in texts]
        all_embs = []
        total = (len(prefixed) + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            for i in tqdm(range(0, len(prefixed), self.batch_size), total=total, desc=desc, unit="batch", leave=True):
                batch = prefixed[i : i + self.batch_size]
                enc = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt"
                )
                enc = {k: v.to(self._device) for k, v in enc.items()}
                out = self._model(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
                normed = F.normalize(pooled.float(), p=2, dim=-1)
                all_embs.append(normed.cpu().numpy())
        return np.vstack(all_embs)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix, "Teacher: encoding queries")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix, "Teacher: encoding passages")


# ---------------------------------------------------------------------------
# VLLMDenseTeacher — large decoder embedding models (e5-mistral-7b-instruct)
# ---------------------------------------------------------------------------

_E5_MISTRAL_TASK = "Given a web search query, retrieve relevant passages that answer the query"


class VLLMDenseTeacher(DenseTeacher):
    """
    Dense teacher for large autoregressive embedding models served via vLLM.

    The vLLM server must be running with --task embedding before Phase 2:
        make start-vllm-teacher

    Quantization on the server side controls VRAM:
        INT8 (~7-8 GB):  --quantization bitsandbytes --load-format bitsandbytes
        INT4 (~4-5 GB):  --quantization awq (requires AWQ-quantized model weights)

    e5-mistral-7b-instruct requires a task instruction prefix on queries.
    The passage_prefix is empty (model card: no prefix for documents).

    vLLM handles last-token pooling internally — we do not implement it here.
    Embeddings returned by the API are already L2-normalized.
    """

    def __init__(
        self,
        model_name: str,
        server_url: str,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 64,
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.server_url = server_url.rstrip("/") + "/v1/embeddings"
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self.timeout = timeout
        self._passages: list[str] = []
        self._index = None
        logger.info("VLLMDenseTeacher initialized. Server: %s  Model: %s", self.server_url, model_name)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """POST a single batch to the vLLM /v1/embeddings endpoint."""
        payload = {"model": self.model_name, "input": texts}
        resp = requests.post(self.server_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()["data"]
        # data is a list of {"index": i, "embedding": [...], "object": "embedding"}
        # Sort by index to maintain order
        data_sorted = sorted(data, key=lambda x: x["index"])
        embs = np.array([d["embedding"] for d in data_sorted], dtype=np.float32)
        # L2-normalize (vLLM may or may not normalize — do it here to be safe)
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-9)
        return embs / norms

    def _encode(self, texts: list[str], prefix: str, desc: str) -> np.ndarray:
        prefixed = [prefix + t for t in texts]
        all_embs = []
        total = (len(prefixed) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(prefixed), self.batch_size), total=total, desc=desc, unit="batch", leave=True):
            batch = prefixed[i : i + self.batch_size]
            embs = self._encode_batch(batch)
            all_embs.append(embs)
        return np.vstack(all_embs)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.query_prefix, "Teacher: encoding queries (vLLM)")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.passage_prefix, "Teacher: encoding passages (vLLM)")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_dense_teacher(cfg) -> DenseTeacher:
    """
    Instantiate the correct DenseTeacher subclass from config.

    cfg.mining.teacher values:
        "intfloat/e5-large-unsupervised"   → TensorRTDenseTeacher
        "intfloat/e5-mistral-7b-instruct"  → VLLMDenseTeacher
    """
    teacher_name: str = cfg.mining.teacher
    query_prefix: str = cfg.mining.teacher_query_prefix
    passage_prefix: str = cfg.mining.teacher_passage_prefix
    batch_size: int = cfg.mining.teacher_batch_size

    if "mistral" in teacher_name.lower() or "7b" in teacher_name.lower():
        server_url: str = cfg.model.get("teacher_embeddings_server", "http://localhost:8001")
        logger.info("Using VLLMDenseTeacher for %s (server: %s)", teacher_name, server_url)
        return VLLMDenseTeacher(
            model_name=teacher_name,
            server_url=server_url,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
        )
    else:
        logger.info("Using TensorRTDenseTeacher for %s", teacher_name)
        return TensorRTDenseTeacher(
            model_name=teacher_name,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            batch_size=batch_size,
        )
