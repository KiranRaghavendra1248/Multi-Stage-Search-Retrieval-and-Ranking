import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiEncoder(nn.Module):
    """
    Two-tower bi-encoder built on top of BERT/DistilBERT.

    Encodes text with mean pooling over token embeddings, then L2-normalizes
    so that inner product == cosine similarity (required by FAISS IndexFlatIP).

    query_prefix / passage_prefix are prepended before tokenization.
    Required by e5-family models (e.g. "query: " / "passage: ").
    Leave empty ("") for models that don't use prefixes (e.g. MiniLM).
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        super().__init__()
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _mean_pool(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pool token embeddings, masking out padding tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_emb = (token_embeddings * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_emb / sum_mask

    def forward(self, enc: BatchEncoding) -> torch.Tensor:
        """
        Args:
            enc: tokenizer output with input_ids, attention_mask on the correct device
        Returns:
            embeddings: [B, hidden_dim], L2-normalized float32
        """
        out = self.encoder(**enc)
        pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"])
        return nn.functional.normalize(pooled, p=2, dim=-1)

    @torch.no_grad()
    def _encode_with_prefix(
        self,
        texts: list[str],
        prefix: str,
        batch_size: int,
        device: torch.device,
        desc: str,
    ) -> np.ndarray:
        """Shared encode loop — prepends prefix before tokenization."""
        self.eval()
        self.to(device)
        all_embs = []
        prefixed = [prefix + t for t in texts]
        total_batches = (len(prefixed) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(prefixed), batch_size), total=total_batches, desc=desc, unit="batch", leave=True):
            batch = prefixed[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            embs = self.forward(enc)
            all_embs.append(embs.cpu().float().numpy())
        return np.vstack(all_embs)

    @torch.no_grad()
    def encode_queries(self, texts: list[str], batch_size: int = 64, device=None) -> np.ndarray:
        """Encode queries with query_prefix. Returns [N, dim] float32 L2-normalized."""
        if device is None:
            device = _get_device()
        return self._encode_with_prefix(texts, self.query_prefix, batch_size, torch.device(device), "Encoding queries")

    @torch.no_grad()
    def encode_passages(self, texts: list[str], batch_size: int = 64, device=None) -> np.ndarray:
        """Encode passages with passage_prefix. Returns [N, dim] float32 L2-normalized."""
        if device is None:
            device = _get_device()
        return self._encode_with_prefix(texts, self.passage_prefix, batch_size, torch.device(device), "Encoding passages")

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64, device=None) -> np.ndarray:
        """
        Backwards-compatible encode — applies passage_prefix.
        Used by legacy callers (phase4, pretrained retriever in compare.py).
        New code should prefer encode_queries() / encode_passages() explicitly.
        """
        return self.encode_passages(texts, batch_size=batch_size, device=device)

    def save(self, path: str) -> None:
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("BiEncoder saved to %s", path)

    @classmethod
    def load(cls, path: str, query_prefix: str = "", passage_prefix: str = "") -> "BiEncoder":
        instance = cls(model_name=path, query_prefix=query_prefix, passage_prefix=passage_prefix)
        logger.info("BiEncoder loaded from %s", path)
        return instance
