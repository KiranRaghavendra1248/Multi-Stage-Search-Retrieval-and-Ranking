import json
import random
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BatchEncoding
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TripletDataset(Dataset):
    """
    PyTorch Dataset over hard_negatives.jsonl.

    Each item returns raw strings (tokenization happens in collate_fn).
    Samples exactly 1 hard negative per item (configurable).

    JSONL format per line:
        {"query": str, "positive": str, "negatives": [str, ...]}
    """

    def __init__(self, jsonl_path: str, k_hard_negatives: int = 1):
        self.k = k_hard_negatives
        self.data: list[dict] = []
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Triplets file not found: {jsonl_path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("negatives"):
                    self.data.append(record)
        logger.info("TripletDataset loaded %d records from %s", len(self.data), path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        negatives = item["negatives"]
        # Sample k without replacement; if fewer available, take all
        k = min(self.k, len(negatives))
        sampled_negs = random.sample(negatives, k)
        return {
            "query": item["query"],
            "positive": item["positive"],
            "hard_negatives": sampled_negs,  # list of k strings
        }


def build_collate_fn(tokenizer_name: str, max_length: int = 128):
    """
    Returns a collate_fn that tokenizes a batch of TripletDataset items.

    Batch tensors:
        query_enc:   BatchEncoding [B, seq_len]
        pos_enc:     BatchEncoding [B, seq_len]
        hn_enc:      BatchEncoding [B*K, seq_len]  (flattened hard negatives)
        batch_size:  int B
        k:           int K (hard negatives per query)
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def collate_fn(batch: list[dict]) -> dict:
        B = len(batch)
        K = len(batch[0]["hard_negatives"])

        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]
        hard_negs_flat = [hn for item in batch for hn in item["hard_negatives"]]

        query_enc: BatchEncoding = tokenizer(
            queries, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        pos_enc: BatchEncoding = tokenizer(
            positives, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )
        hn_enc: BatchEncoding = tokenizer(
            hard_negs_flat, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt"
        )

        return {
            "query_enc": query_enc,
            "pos_enc": pos_enc,
            "hn_enc": hn_enc,
            "B": B,
            "K": K,
        }

    return collate_fn
