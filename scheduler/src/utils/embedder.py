"""
utils/embedder.py
─────────────────────────────
Provider-agnostic embedding interface.

Usage:
    embedder = get_embedder()
    vectors  = embedder.embed_batch(["text one", "text two"])

Supported providers (set EMBEDDING_PROVIDER in .env):
  • "openai"  → OpenAI text-embedding-3-small (default, 1536-dim)
  • "local"   → sentence-transformers BAAI/bge-small-en-v1.5 (384-dim)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List

import structlog

from config.settings import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    model_name: str = ""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return a list of embedding vectors, one per input text."""
        ...

    def embed_one(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI provider
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self):
        from openai import OpenAI
        self._client    = OpenAI(api_key=settings.openai_api_key)
        self.model_name = settings.openai_embedding_model

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # OpenAI embedding endpoint strips newlines poorly — do it here
        cleaned = [t.replace("\n", " ") for t in texts]

        t0 = time.perf_counter()
        response = self._client.embeddings.create(
            input=cleaned,
            model=self.model_name,
        )
        elapsed = time.perf_counter() - t0

        vectors = [item.embedding for item in response.data]

        log.debug(
            "openai_embedder.batch",
            count=len(texts),
            model=self.model_name,
            elapsed_s=round(elapsed, 3),
            total_tokens=response.usage.total_tokens,
        )
        return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Local / offline provider (sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model_name = settings.local_embedding_model
        log.info("local_embedder.loading", model=self.model_name)
        self._model = SentenceTransformer(self.model_name)
        log.info("local_embedder.ready", model=self.model_name)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        t0 = time.perf_counter()
        # normalize_embeddings=True → cosine similarity == dot product
        vectors = self._model.encode(
            texts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        elapsed = time.perf_counter() - t0
        log.debug(
            "local_embedder.batch",
            count=len(texts),
            model=self.model_name,
            elapsed_s=round(elapsed, 3),
        )
        return vectors


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    """
    Returns a cached embedder instance (one per worker process).
    Avoids re-loading the model on every task call.
    """
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set it in .env or switch EMBEDDING_PROVIDER=local"
            )
        log.info("embedder.init", provider="openai", model=settings.openai_embedding_model)
        return OpenAIEmbedder()

    elif provider == "local":
        log.info("embedder.init", provider="local", model=settings.local_embedding_model)
        return LocalEmbedder()

    else:
        raise ValueError(
            f"Unknown embedding provider '{provider}'. "
            "Choose 'openai' or 'local'."
        )