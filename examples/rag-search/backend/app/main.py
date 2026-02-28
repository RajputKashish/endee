"""FastAPI application for RAG semantic search."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import API_HOST, API_PORT
from .embeddings import embed_text, embed_texts
from .endee_client import ensure_index_exists, query_similar, upsert_documents


# --- Pydantic models ---


class DocumentInput(BaseModel):
    """Single document for ingestion."""

    id: str = Field(..., description="Unique document identifier")
    text: str = Field(..., description="Document text content")
    meta: Optional[dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")


class IngestRequest(BaseModel):
    """Request body for document ingestion."""

    documents: list[DocumentInput] = Field(..., description="List of documents to ingest")


class QueryRequest(BaseModel):
    """Request body for semantic search query."""

    query_text: str = Field(..., description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    filters: Optional[dict[str, Any]] = Field(default=None, description="Optional metadata filters")


# --- Lifespan: initialize model and index on startup ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embedding model and Endee index on startup."""
    # Pre-load embedding model
    embed_text("warmup")
    # Ensure Endee index exists
    ensure_index_exists()
    yield
    # Cleanup (if any) on shutdown
    pass


app = FastAPI(
    title="RAG Semantic Search",
    description="Document Q&A and semantic search powered by Endee Vector Database",
    lifespan=lifespan,
)

# Path to frontend (rag-search/frontend relative to backend/app)
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


# --- Endpoints ---


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the search UI."""
    index_path = _FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>RAG Search API</h1><p>Use /health, /ingest, /query</p>", status_code=200)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "rag-search"}


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict[str, Any]:
    """Ingest documents into the vector index."""
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    texts = [d.text for d in request.documents]
    vectors = embed_texts(texts)

    vectors_with_metadata = []
    for doc, vector in zip(request.documents, vectors):
        meta = dict(doc.meta or {})
        # Add snippet for display (first 200 chars)
        snippet = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
        meta["snippet"] = snippet
        vectors_with_metadata.append({
            "id": doc.id,
            "vector": vector,
            "meta": meta,
        })

    upsert_documents(vectors_with_metadata)
    return {"status": "ok", "ingested": len(request.documents)}


@app.post("/query")
async def query(request: QueryRequest) -> dict[str, Any]:
    """Query the vector index for similar documents."""
    query_vector = embed_text(request.query_text)
    results = query_similar(
        vector=query_vector,
        top_k=request.top_k,
        filters=request.filters,
    )
    return {"results": results}


# --- Run with uvicorn ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
