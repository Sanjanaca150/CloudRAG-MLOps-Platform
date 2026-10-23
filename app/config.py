import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "knowledge"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "data" / "chroma"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "cloudrag_documents")
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
