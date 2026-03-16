import json
import os
from pathlib import Path
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TripletWriter:
    """
    Append-mode JSONL writer for hard negative triplets.

    Each line:
        {"query": str, "positive": str, "negatives": [str, ...]}

    Supports crash-safe resume:
        - Always appends; never overwrites.
        - Flushes every `flush_every` lines.
        - load_seen_query_ids() reads existing file to skip already-processed queries.
    """

    def __init__(self, path: str, flush_every: int = 100):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._count = 0
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, query: str, positive: str, negatives: list[str]) -> None:
        record = {"query": query, "positive": positive, "negatives": negatives}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._count += 1
        if self._count % self._flush_every == 0:
            self._fh.flush()
            logger.debug("Flushed %d triplets to %s", self._count, self.path)

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()
        logger.info("TripletWriter closed. Total written this session: %d", self._count)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @staticmethod
    def load_seen_query_ids(path: str) -> set[str]:
        """
        Read existing JSONL and return the set of query strings already written.
        Call this before starting the mining loop to enable resume.
        """
        seen = set()
        p = Path(path)
        if not p.exists():
            return seen
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    seen.add(record["query"])
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info("Loaded %d seen queries from %s (resume mode)", len(seen), p)
        return seen
