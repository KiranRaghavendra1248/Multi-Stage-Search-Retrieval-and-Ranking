import itertools
from typing import Iterator
from datasets import load_dataset
from omegaconf import DictConfig
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _extract_positive(passages: dict) -> str | None:
    """Return the text of the passage with is_selected == 1, or None."""
    texts = passages.get("passage_text", [])
    flags = passages.get("is_selected", [])
    for text, flag in zip(texts, flags):
        if flag == 1:
            return text
    return None


def load_msmarco_stream(
    cfg: DictConfig,
    split: str = "train",
) -> list[dict]:
    """
    Stream MS MARCO and return a list of records.

    Each record:
        {
            "query_id":         str,
            "query":            str,
            "passages":         dict,           # raw passages dict from dataset
            "positive_passage": str | None,     # text of is_selected==1 passage
        }

    Respects cfg.data.sample_cap (None = full dataset).
    """
    sample_cap: int | None = cfg.data.get("sample_cap", None) if split == "train" else None
    logger.info(
        "Loading MS MARCO (%s / %s), sample_cap=%s",
        cfg.data.dataset_config,
        split,
        sample_cap,
    )

    ds = load_dataset(
        cfg.data.dataset,
        cfg.data.dataset_config,
        split=split,
        streaming=True,
    )

    stream: Iterator = iter(ds)
    if sample_cap is not None:
        stream = itertools.islice(stream, sample_cap)

    records = []
    for row in stream:
        records.append(
            {
                "query_id": str(row["query_id"]),
                "query": row["query"],
                "passages": row["passages"],
                "positive_passage": _extract_positive(row["passages"]),
            }
        )

    logger.info("Loaded %d records from MS MARCO (%s)", len(records), split)
    return records


def iter_msmarco_stream(
    cfg: DictConfig,
    split: str = "train",
) -> Iterator[dict]:
    """
    Streaming variant — yields records one at a time without buffering.
    Use for Phase 2 (full 3.2M corpus) where memory matters.
    sample_cap is only applied to the train split; dev/validation is always fully streamed.
    """
    sample_cap: int | None = cfg.data.get("sample_cap", None) if split == "train" else None
    ds = load_dataset(
        cfg.data.dataset,
        cfg.data.dataset_config,
        split=split,
        streaming=True,
    )
    stream: Iterator = iter(ds)
    if sample_cap is not None:
        stream = itertools.islice(stream, sample_cap)

    for row in stream:
        yield {
            "query_id": str(row["query_id"]),
            "query": row["query"],
            "passages": row["passages"],
            "positive_passage": _extract_positive(row["passages"]),
        }
