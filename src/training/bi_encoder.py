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
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.model_name = model_name
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
    def encode(self, texts: list[str], batch_size: int = 64, device=None) -> np.ndarray:
        """
        Encode a list of texts and return a numpy float32 array [N, dim].
        Used at inference time (FAISS indexing, retrieval).
        Auto-detects GPU if device is not specified.
        """
        if device is None:
            device = _get_device()
        device = torch.device(device)
        self.eval()
        self.to(device)
        all_embs = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Encoding", unit="batch", leave=True):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            embs = self.forward(enc)
            all_embs.append(embs.cpu().float().numpy())
        return np.vstack(all_embs)

    def save(self, path: str) -> None:
        self.encoder.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("BiEncoder saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "BiEncoder":
        instance = cls(model_name=path)
        logger.info("BiEncoder loaded from %s", path)
        return instance
