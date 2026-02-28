"""Endee Vector Database client wrapper."""

from typing import Any, Optional

from endee import Endee, Precision

from .config import (
    ENDEE_AUTH_TOKEN,
    ENDEE_BASE_URL,
    EMBEDDING_DIM,
    INDEX_NAME,
    SPACE_TYPE,
)


def _get_client() -> Endee:
    """Create and configure the Endee client."""
    client = Endee(token=ENDEE_AUTH_TOKEN) if ENDEE_AUTH_TOKEN else Endee()
    client.set_base_url(ENDEE_BASE_URL)
    return client


def ensure_index_exists() -> None:
    """Create the index if it does not already exist."""
    client = _get_client()
    try:
        client.get_index(name=INDEX_NAME)
    except Exception:
        client.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            space_type=SPACE_TYPE,
            precision=Precision.INT8,
        )


def upsert_documents(
    vectors_with_metadata: list[dict[str, Any]],
) -> None:
    """Upsert document vectors into the Endee index.

    Each item should have: id, vector, meta (optional), filter (optional).
    """
    client = _get_client()
    index = client.get_index(name=INDEX_NAME)
    index.upsert(vectors_with_metadata)


def _normalize_filter(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert simple dict to Endee filter format: [{"key": {"$eq": "value"}}]."""
    return [{k: {"$eq": v}} for k, v in filters.items()]


def query_similar(
    vector: list[float],
    top_k: int = 5,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Query the index for similar vectors.

    Returns list of results with id, similarity, meta, etc.
    """
    client = _get_client()
    index = client.get_index(name=INDEX_NAME)
    kwargs: dict[str, Any] = {"vector": vector, "top_k": top_k}
    if filters:
        kwargs["filter"] = _normalize_filter(filters)
    return index.query(**kwargs)
