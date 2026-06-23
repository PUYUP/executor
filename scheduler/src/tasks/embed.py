"""
tasks/embed.py
──────────────────────────
Tier-3 tasks (queue: embed)
─────────────────────────────────────────────────────────────────────────
Flow (continued from process.py):
  chunk_document → generate_embeddings → store_chunks
─────────────────────────────────────────────────────────────────────────
Design notes:
  • Batching     – chunks are grouped into batches of `embedding_batch_size`
                   before hitting the API to minimise round-trips.
  • Provider     – "openai"  → text-embedding-3-small (default)
                   "local"   → sentence-transformers (BAAI/bge-small-en)
  • Rate limit   – the `rate_limit` annotation (20/m) in celery_config
                   keeps us under the OpenAI embeddings quota.
  • store_chunks – intentionally thin: just attaches the vectors to each
                   chunk dict and hands off to your database layer.
                   Swap in your pgvector / Qdrant / Weaviate writer here.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import structlog

from main import app
from utils.embedder import get_embedder
from config.settings import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 of 7 — generate_embeddings
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.embed.generate_embeddings",
    bind=True,
    max_retries=10,
    default_retry_delay=30,
    queue="embed",
    # rate_limit set globally in celery_config.py  →  "20/m"
    time_limit=600,
    soft_time_limit=540,
    ignore_result=False,
)
def generate_embeddings(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates embedding vectors for every chunk in `metadata["chunks"]`.

    Chunks are batched to stay within API token limits.
    Each chunk is enriched with an `embedding` key (list[float]).

    Returns the full metadata dict with embedded chunks.
    """
    arxiv_id = metadata["arxiv_id"]
    chunks   = metadata.get("chunks", [])

    if not chunks:
        log.warning("generate_embeddings.no_chunks", arxiv_id=arxiv_id)
        return metadata

    log.info(
        "generate_embeddings.start",
        arxiv_id=arxiv_id,
        chunks=len(chunks),
        provider=settings.embedding_provider,
    )

    embedder = get_embedder()
    batch_size = settings.embedding_batch_size

    try:
        embedded_chunks: List[Dict[str, Any]] = []

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [c["text"] for c in batch]

            t0 = time.perf_counter()
            vectors = embedder.embed_batch(texts)
            elapsed = time.perf_counter() - t0

            log.debug(
                "generate_embeddings.batch_done",
                arxiv_id=arxiv_id,
                batch_start=batch_start,
                batch_size=len(batch),
                elapsed_s=round(elapsed, 2),
            )

            for chunk, vector in zip(batch, vectors):
                chunk_with_vec = chunk.copy()
                chunk_with_vec["embedding"]        = vector
                chunk_with_vec["embedding_model"]  = embedder.model_name
                chunk_with_vec["embedding_dim"]    = len(vector)
                embedded_chunks.append(chunk_with_vec)

    except Exception as exc:
        log.error(
            "generate_embeddings.failed",
            arxiv_id=arxiv_id,
            error=str(exc),
            attempt=self.request.retries,
        )
        # Exponential back-off: 30s, 60s, 120s …
        raise self.retry(exc=exc, countdown=30 * 2 ** self.request.retries)

    log.info(
        "generate_embeddings.done",
        arxiv_id=arxiv_id,
        embedded=len(embedded_chunks),
        dim=embedded_chunks[0]["embedding_dim"] if embedded_chunks else 0,
    )

    metadata["chunks"] = embedded_chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 7 of 7 — store_chunks
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.embed.store_chunks",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    queue="embed",
    ignore_result=False,
)
def store_chunks(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Persists embedded chunks to your vector store.

    ┌──────────────────────────────────────────────────────────┐
    │  INTEGRATION POINT — replace the body of _write_chunks   │
    │  with your actual DB writer (pgvector, Qdrant, Pinecone, │
    │  Weaviate, Chroma …).                                     │
    └──────────────────────────────────────────────────────────┘

    Each chunk arriving here is guaranteed to have:
      chunk_id, arxiv_id, title, section, text, embedding,
      embedding_model, embedding_dim, page_start, page_end,
      authors, categories, published, doi, token_count
    """
    arxiv_id = metadata["arxiv_id"]
    chunks   = metadata.get("chunks", [])

    # Drop chunks that somehow lost their embedding
    valid_chunks = [c for c in chunks if c.get("embedding")]
    if not valid_chunks:
        log.warning("store_chunks.no_valid_chunks", arxiv_id=arxiv_id)
        return {"arxiv_id": arxiv_id, "stored": 0, "status": "skipped"}

    log.info("store_chunks.start", arxiv_id=arxiv_id, count=len(valid_chunks))

    try:
        stored_count = _write_chunks(valid_chunks)
    except Exception as exc:
        log.error("store_chunks.failed", arxiv_id=arxiv_id, error=str(exc))
        raise self.retry(exc=exc)

    # Mark this paper as fully processed so scrape_topic won't re-queue it
    from utils.dedup import mark_as_processed
    mark_as_processed(arxiv_id)

    # Clean up the local PDF to reclaim disk space
    _cleanup_pdf(metadata.get("local_pdf_path"))

    log.info("store_chunks.done", arxiv_id=arxiv_id, stored=stored_count)

    return {
        "arxiv_id":   arxiv_id,
        "stored":     stored_count,
        "status":     "ok",
        "title":      metadata.get("title", ""),
        "categories": metadata.get("categories", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Integration stub — swap this out for your real vector store
# ─────────────────────────────────────────────────────────────────────────────

def _write_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Write `chunks` to your vector database.

    Each chunk dict has this shape:
    {
        "chunk_id":        "2401.12345_2_0",
        "arxiv_id":        "2401.12345",
        "title":           "Attention Is All You Need",
        "section":         "Methods",
        "text":            "We propose a new...",
        "embedding":       [0.012, -0.034, ...],   # list[float]
        "embedding_model": "text-embedding-3-small",
        "embedding_dim":   1536,
        "page_start":      3,
        "page_end":        5,
        "authors":         ["Vaswani, A.", ...],
        "categories":      ["cs.CL", "cs.LG"],
        "published":       "2017-06-12T00:00:00+00:00",
        "doi":             "...",
        "token_count":     420,
    }

    ── Example: Qdrant ─────────────────────────────────────────────────
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    client = QdrantClient(host="qdrant", port=6333)
    points = [
        PointStruct(
            id=abs(hash(c["chunk_id"])) % (2**63),
            vector=c["embedding"],
            payload={k: v for k, v in c.items() if k != "embedding"},
        )
        for c in chunks
    ]
    client.upsert(collection_name="arxiv_chunks", points=points)

    ── Example: pgvector ───────────────────────────────────────────────
    conn.executemany(
        "INSERT INTO chunks (chunk_id, arxiv_id, section, text, embedding, metadata)
         VALUES (%s, %s, %s, %s, %s::vector, %s)
         ON CONFLICT (chunk_id) DO NOTHING",
        [(c["chunk_id"], c["arxiv_id"], c["section"],
          c["text"], c["embedding"], json.dumps({...}))
         for c in chunks]
    )
    """
    # ── Default: log-only (replace with real writer) ──────────────────
    log.info(
        "_write_chunks.stub",
        count=len(chunks),
        sample_chunk_id=chunks[0]["chunk_id"] if chunks else None,
        note="Replace _write_chunks() with your vector store writer",
    )
    return len(chunks)


def _cleanup_pdf(pdf_path: str | None) -> None:
    """Remove the local PDF after successful embedding to free disk space."""
    if not pdf_path:
        return
    from pathlib import Path
    p = Path(pdf_path)
    if p.exists():
        p.unlink()
        log.debug("store_chunks.pdf_deleted", path=str(p))