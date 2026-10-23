import chromadb
from app.config import CHROMA_COLLECTION, CHROMA_PATH
from app.ollama_client import embed_texts

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)


def index_chunks(chunks: list[dict]) -> int:
    """Replace the current collection contents with the supplied chunks."""
    existing = _collection.get()
    if existing.get("ids"):
        _collection.delete(ids=existing["ids"])
    if not chunks:
        return 0

    texts = [item["text"] for item in chunks]
    embeddings = embed_texts(texts)
    _collection.add(
        ids=[item["id"] for item in chunks],
        documents=texts,
        metadatas=[{"source": item["source"], "chunk_index": item["chunk_index"]} for item in chunks],
        embeddings=embeddings,
    )
    return len(chunks)


def search_chunks(question: str, top_k: int = 4) -> list[dict]:
    query_embedding = embed_texts([question])[0]
    result = _collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(_collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {
            "text": document,
            "source": metadata.get("source", "unknown"),
            "chunk_index": metadata.get("chunk_index", -1),
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def collection_count() -> int:
    return _collection.count()
