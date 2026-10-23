import pytest
from app.chunking import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("   ") == []


def test_short_text_returns_one_chunk():
    assert split_text("CloudRAG is a document assistant.") == ["CloudRAG is a document assistant."]


def test_long_text_is_split_with_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    assert all(chunks)
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=10, overlap=10)
