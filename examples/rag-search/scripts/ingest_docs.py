#!/usr/bin/env python3
"""Ingest documents from data/docs into the RAG Search index."""

import sys
from pathlib import Path

# Add backend to path so we can import app modules
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

import requests

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"
API_BASE = "http://localhost:8000"


def load_documents() -> list[dict]:
    """Load all documents from data/docs."""
    documents = []
    if not DOCS_DIR.exists():
        print(f"Docs directory not found: {DOCS_DIR}")
        return documents

    for path in sorted(DOCS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in (".md", ".txt", ".rst"):
            text = path.read_text(encoding="utf-8", errors="replace")
            doc_id = path.stem
            title = path.stem.replace("-", " ").replace("_", " ").title()
            snippet = (text[:200] + "...") if len(text) > 200 else text
            documents.append({
                "id": doc_id,
                "text": text,
                "meta": {"title": title, "source": path.name, "snippet": snippet},
            })
    return documents


def main() -> int:
    docs = load_documents()
    if not docs:
        print("No documents found to ingest.")
        return 1

    payload = {
        "documents": [
            {"id": d["id"], "text": d["text"], "meta": d["meta"]}
            for d in docs
        ]
    }

    try:
        resp = requests.post(f"{API_BASE}/ingest", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        print(f"Ingested {data.get('ingested', len(docs))} documents.")
        return 0
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Is the backend running on port 8000?")
        return 1
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(e.response.text)
        return 1


if __name__ == "__main__":
    sys.exit(main())
