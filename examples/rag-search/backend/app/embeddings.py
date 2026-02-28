"""Embedding model loading and vectorization helpers."""

from typing import Optional

from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_DIM, EMBEDDING_MODEL

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Load the embedding model (lazy singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single text string into a vector."""
    model = _get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple text strings into vectors (batched)."""
    model = _get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return [e.tolist() for e in embeddings]


def get_embedding_dim() -> int:
    """Return the expected embedding dimension (from config)."""
    return EMBEDDING_DIM
