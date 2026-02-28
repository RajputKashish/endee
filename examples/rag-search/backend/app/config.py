"""Configuration for the RAG Search application."""

import os
from typing import Optional


def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# Endee Vector Database
ENDEE_BASE_URL: str = _get_env("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN: Optional[str] = _get_env("ENDEE_AUTH_TOKEN", "") or None
INDEX_NAME: str = _get_env("ENDEE_INDEX_NAME", "documents")

# Embedding model (SentenceTransformers)
EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(_get_env("EMBEDDING_DIM", "384"))

# Index configuration
SPACE_TYPE: str = _get_env("ENDEE_SPACE_TYPE", "cosine")

# API
API_HOST: str = _get_env("API_HOST", "0.0.0.0")
API_PORT: int = int(_get_env("API_PORT", "8000"))
