# CloudRAG MLOps Platform

A local-first Retrieval-Augmented Generation (RAG) application that answers questions from knowledge documents using **FastAPI, Ollama, ChromaDB, MLflow, and Docker**. It includes a browser-based chat interface, source citations, automated tests, and GitHub Actions CI.

The project demonstrates an end-to-end local RAG and MLOps workflow without requiring paid cloud hosting.

## Features

- **Document ingestion:** Load `.txt` knowledge documents from `data/knowledge/`.
- **Text chunking:** Split documents into overlapping chunks for retrieval.
- **Local embeddings:** Generate embeddings using Ollama's `nomic-embed-text`.
- **Vector storage:** Store and retrieve document embeddings with persistent ChromaDB storage.
- **LLM question answering:** Use `llama3.2:3b` to generate answers grounded in retrieved context.
- **Source citations:** Return source filenames and chunk metadata with answers.
- **Browser chat UI:** Ask questions, index documents, view answers, and inspect citations.
- **FastAPI endpoints:** Health checks, document ingestion, and question answering.
- **MLOps evaluation:** Evaluate retrieval quality and track experiment results with MLflow.
- **Automated tests:** Run API and chunking tests with pytest.
- **Docker:** Package and run the API in a container.
- **CI/CD checks:** Run automated tests through GitHub Actions.

## Architecture

```text
Knowledge Documents (.txt)
          |
          v
   Text Chunking
          |
          v
 Ollama Embeddings
  nomic-embed-text
          |
          v
 Persistent ChromaDB
          |
          v
 User Question ---> Retrieve Relevant Chunks
                              |
                              v
                       Ollama LLM
                        llama3.2:3b
                              |
                              v
                     Answer + Citations
                              |
                              v
                    Browser UI / REST API
```

MLflow is used separately to track evaluation experiments and metrics. Docker packages the API, while Ollama runs on the host machine during local Docker deployment.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| API | FastAPI, Uvicorn |
| LLM inference | Ollama, Llama 3.2 |
| Embeddings | Ollama, `nomic-embed-text` |
| Vector database | ChromaDB |
| Evaluation tracking | MLflow |
| Testing | pytest, HTTPX |
| Frontend | HTML, CSS, JavaScript |
| Containerization | Docker |
| CI | GitHub Actions |

## Project Structure

```text
CloudRAG-MLOps-Platform/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── chunking.py
│   ├── ollama_client.py
│   ├── vector_store.py
│   ├── evaluation.py
│   └── static/
│       └── index.html
├── data/
│   └── knowledge/
├── evaluation/
│   └── eval_dataset.json
├── scripts/
│   └── run_evaluation.py
├── tests/
│   ├── test_api.py
│   └── test_chunking.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── requirements.txt
├── requirements-phase2.txt
└── README.md
```

## Prerequisites

- Windows 10/11
- Python 3.12 for the local Python environment
- Docker Desktop with its Linux container engine running
- Ollama installed and running
- The following Ollama models:

```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Option 1: Run locally with Python

Open PowerShell in the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure Ollama is running and both required models are available. Start the API:

```powershell
uvicorn app.main:app --reload
