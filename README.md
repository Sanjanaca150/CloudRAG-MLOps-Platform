# CloudRAG MLOps Platform

A local-first Retrieval-Augmented Generation (RAG) API built with FastAPI, Ollama, and ChromaDB. This is Phase 1 of a larger portfolio project; MLOps evaluation, Docker, CI/CD, and AWS deployment will be added in later phases.

## Features

- Ingest `.txt` documents from `data/knowledge/`
- Split documents into overlapping chunks
- Create local embeddings using Ollama's `nomic-embed-text`
- Store vectors persistently in ChromaDB
- Answer questions using `llama3.2:3b` and retrieved context
- Return source filenames and chunk citations
- Provide `/health`, `/ingest`, and `/ask` endpoints
- Automated unit and API tests

## Prerequisites

- Windows 10/11
- Python 3.12
- Ollama installed and running
- Models downloaded: `llama3.2:3b` and `nomic-embed-text`

## Run locally (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
ollama pull llama3.2:3b
ollama pull nomic-embed-text
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`, call `POST /ingest`, then call `POST /ask`.

Example request:

```json
{
  "question": "How many annual paid leave days do full-time employees receive?",
  "top_k": 4
}
```

## Tests

```powershell
python -m pytest -q
```

Tests do not require a running Ollama server; live ingestion and generation do.

## Important limitations

This phase is a local development prototype, not production-secure. It has no authentication, tenant isolation, malware scanning, or production deployment. Do not expose it publicly or index sensitive documents. Source citations indicate retrieved chunks; they do not independently prove every generated claim.
