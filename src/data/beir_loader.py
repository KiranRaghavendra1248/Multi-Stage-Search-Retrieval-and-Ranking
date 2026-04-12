from typing import Iterator
from datasets import load_dataset
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def iter_beir_corpus(cfg) -> Iterator[dict]:
    """
    Stream all 8.8M passages from BeIR/msmarco corpus.

    Yields:
        {"id": str, "text": str}
    """
    logger.info("Streaming BeIR/msmarco corpus (~8.8M passages)...")
    ds = load_dataset(cfg.data.beir_corpus, "corpus", split="corpus", streaming=True)
    for row in ds:
        yield {"id": row["_id"], "text": row["text"]}


def load_beir_dev_eval(cfg) -> tuple[list[str], list[str]]:
    """
    Load dev queries and gold passage texts for evaluation using BeIR qrels.

    Process:
        1. Load validation qrels (BeIR/msmarco-qrels) — 7,440 query→corpus-id pairs
        2. Load all queries (BeIR/msmarco queries config) — query-id → query text
        3. Stream corpus to collect gold passage texts for the ~7,440 relevant corpus-ids
        4. Return parallel (queries, gold_passages) lists

    Returns:
        queries:       list of query strings
        gold_passages: list of gold passage texts, one per query
    """
    logger.info("Loading BeIR validation qrels from %s...", cfg.data.beir_qrels)
    qrels_ds = load_dataset(cfg.data.beir_qrels, split="validation")
    qid_to_cid = {str(r["query-id"]): str(r["corpus-id"]) for r in qrels_ds}
    logger.info("Loaded %d validation qrels.", len(qid_to_cid))

    logger.info("Loading BeIR queries...")
    queries_ds = load_dataset(cfg.data.beir_corpus, "queries", split="queries")
    qid_to_text = {r["_id"]: r["text"] for r in queries_ds}
    logger.info("Loaded %d queries.", len(qid_to_text))

    # Stream corpus to collect only the gold passage texts we need
    needed_cids = set(qid_to_cid.values())
    logger.info("Streaming corpus to collect %d gold passage texts...", len(needed_cids))
    cid_to_text: dict[str, str] = {}
    corpus_ds = load_dataset(cfg.data.beir_corpus, "corpus", split="corpus", streaming=True)
    for row in corpus_ds:
        if row["_id"] in needed_cids:
            cid_to_text[row["_id"]] = row["text"]
        if len(cid_to_text) == len(needed_cids):
            break
    logger.info("Collected %d gold passage texts.", len(cid_to_text))

    queries: list[str] = []
    gold_passages: list[str] = []
    for qid, cid in qid_to_cid.items():
        if qid in qid_to_text and cid in cid_to_text:
            queries.append(qid_to_text[qid])
            gold_passages.append(cid_to_text[cid])

    logger.info("Dev eval set: %d query-gold pairs.", len(queries))
    return queries, gold_passages
