import tempfile
import os
import pytest
from src.indexing.bm25_index import BM25Index


PASSAGES = [
    "The quick brown fox jumps over the lazy dog.",
    "Inflation is caused by excess money supply in the economy.",
    "Python is a popular programming language for data science.",
    "The Federal Reserve controls interest rates to manage inflation.",
    "Machine learning models require large amounts of training data.",
]


def test_build_and_search():
    idx = BM25Index()
    idx.build(PASSAGES)
    results = idx.search("inflation money supply", top_k=3)
    assert len(results) == 3
    texts = [r[0] for r in results]
    assert any("inflation" in t.lower() for t in texts)


def test_search_returns_scores():
    idx = BM25Index()
    idx.build(PASSAGES)
    results = idx.search("python programming", top_k=2)
    assert all(isinstance(score, float) for _, score in results)
    # Top result should be the Python passage
    assert "python" in results[0][0].lower()


def test_save_and_load(tmp_path):
    idx = BM25Index()
    idx.build(PASSAGES)
    save_dir = str(tmp_path / "bm25_test")
    idx.save(save_dir)

    loaded = BM25Index.load(save_dir)
    assert len(loaded) == len(PASSAGES)

    results = loaded.search("inflation", top_k=2)
    assert len(results) == 2
    assert any("inflation" in r[0].lower() for r in results)


def test_top_k_capped_at_corpus_size():
    idx = BM25Index()
    idx.build(PASSAGES)
    results = idx.search("the", top_k=100)
    assert len(results) == len(PASSAGES)


def test_len():
    idx = BM25Index()
    idx.build(PASSAGES)
    assert len(idx) == len(PASSAGES)
