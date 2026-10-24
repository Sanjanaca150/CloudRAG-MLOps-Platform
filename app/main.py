from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from app.chunking import split_text
from app.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, TOP_K
from app.ollama_client import OllamaError, generate_answer
from app.vector_store import collection_count, index_chunks, search_chunks


app = FastAPI(
    title="CloudRAG MLOps Platform",
    description=(
        "Local-first RAG API with Ollama embeddings, "
        "ChromaDB retrieval, source citations, and a browser chat interface."
    ),
    version="0.2.1",
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
CHAT_PAGE = STATIC_DIR / "index.html"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=TOP_K, ge=1, le=8)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if len(value) < 3:
            raise ValueError("Question must contain at least 3 non-space characters.")

        if not value:
            raise ValueError("Question cannot be blank.")

        return value


class SourceCitation(BaseModel):
    source: str
    chunk_index: int
    distance: float


class AskResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    retrieved_chunks: int


@app.get("/", include_in_schema=False)
def chat_page():
    """Serve the browser chat interface."""
    if not CHAT_PAGE.is_file():
        raise HTTPException(
            status_code=500,
            detail="Chat interface is unavailable.",
        )

    return FileResponse(CHAT_PAGE, media_type="text/html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_chunks": collection_count(),
    }


@app.post("/ingest")
def ingest():
    if not DATA_DIR.is_dir():
        raise HTTPException(
            status_code=500,
            detail="Knowledge directory is unavailable.",
        )

    files = sorted(DATA_DIR.glob("*.txt"))

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No .txt documents found in data/knowledge.",
        )

    chunks = []

    for path in files:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not read knowledge document: {path.name}",
            ) from exc

        for index, chunk in enumerate(
            split_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
        ):
            chunks.append(
                {
                    "id": f"{path.name}::{index}",
                    "text": chunk,
                    "source": path.name,
                    "chunk_index": index,
                }
            )

    try:
        count = index_chunks(chunks)
    except OllamaError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return {
        "status": "indexed",
        "documents": len(files),
        "chunks": count,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    if collection_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No indexed chunks. Call POST /ingest first.",
        )

    try:
        matches = search_chunks(request.question, request.top_k)

        if not matches:
            raise HTTPException(
                status_code=404,
                detail="No relevant document chunks were found.",
            )

        context = "\n\n".join(
            (
                f"[Source: {item['source']} | "
                f"Chunk: {item['chunk_index']}]\n{item['text']}"
            )
            for item in matches
        )

        answer = generate_answer(request.question, context)

    except OllamaError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    citations = [
        SourceCitation(
            source=item["source"],
            chunk_index=item["chunk_index"],
            distance=round(float(item["distance"]), 4),
        )
        for item in matches
    ]

    return AskResponse(
        answer=answer,
        citations=citations,
        retrieved_chunks=len(matches),
    )
