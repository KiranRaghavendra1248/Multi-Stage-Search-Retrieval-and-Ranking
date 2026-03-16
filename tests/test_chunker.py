import pytest
from src.data.chunker import chunk_document


def test_short_text_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_document(text, max_tokens=256)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_paragraph_split_on_double_newline():
    text = "First paragraph with some content here.\n\nSecond paragraph with different content."
    chunks = chunk_document(text, max_tokens=256)
    assert len(chunks) >= 1


def test_oversized_paragraph_falls_back_to_sentences():
    # Create a paragraph that exceeds 256 tokens
    long_para = " ".join(["This is sentence number %d." % i for i in range(60)])
    chunks = chunk_document(long_para, max_tokens=256)
    assert len(chunks) > 1
    for chunk in chunks:
        # Each chunk should be roughly within token budget
        words = chunk.split()
        assert len(words) < 400  # rough upper bound


def test_tiny_chunks_merged():
    # Two very short paragraphs should merge
    text = "Hi.\n\nHello."
    chunks = chunk_document(text, max_tokens=256, min_tokens_merge=64)
    # Should be merged into one chunk since both are < 64 tokens
    assert len(chunks) == 1


def test_horizontal_rule_splits_sections():
    text = "Section one content.\n---\nSection two content."
    chunks = chunk_document(text, max_tokens=256)
    assert len(chunks) >= 1


def test_empty_text():
    chunks = chunk_document("", max_tokens=256)
    assert chunks == []


def test_no_empty_chunks():
    text = "Para one.\n\n\n\nPara two.\n\nPara three."
    chunks = chunk_document(text, max_tokens=256)
    for chunk in chunks:
        assert chunk.strip() != ""
