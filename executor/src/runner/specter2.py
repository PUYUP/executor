"""SPECTER2 embedding executor.

Wraps the ``allenai/specter2`` model (via the ``adapters`` library) to
produce 768-dimensional dense embeddings from scientific text chunks.

Design goals
------------
* Lazy model loading — importing this module never triggers a GPU/disk hit.
* Async-safe — ``encode`` runs in a thread-pool executor so it does not
  block the event loop.
* Batched — inputs are processed in configurable mini-batches.
* Context-manager lifecycle mirrors :class:`~executor.grobid.GrobidExecutor`.

Adapters available
------------------
* ``allenai/specter2``               – document proximity / similarity  ← default
* ``allenai/specter2_adhoc_query``   – query-document retrieval
* ``allenai/specter2_classification`` – classification tasks
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

from executor.runner.parser import ChunkDict

logger = logging.getLogger(__name__)

# SPECTER2 always produces 768-dimensional embeddings
EMBEDDING_DIM = 768

# HuggingFace identifiers
_BASE_MODEL = "allenai/specter2_base"
_ADAPTER_PROXIMITY = "allenai/specter2"
_ADAPTER_QUERY = "allenai/specter2_adhoc_query"
_ADAPTER_CLASSIFICATION = "allenai/specter2_classification"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Specter2Config:
    """Tunable parameters for the SPECTER2 executor."""

    base_model: str = _BASE_MODEL
    """HuggingFace model ID for the SPECTER2 base."""

    adapter: str = _ADAPTER_PROXIMITY
    """HuggingFace adapter ID to load on top of the base model."""

    device: str = "cpu"
    """PyTorch device string: ``'cpu'``, ``'cuda'``, ``'cuda:0'``, ``'mps'``."""

    batch_size: int = 16
    """Number of texts to encode in one forward pass."""

    max_length: int = 512
    """Maximum subword token length (SPECTER2 hard limit)."""

    normalize_embeddings: bool = True
    """L2-normalize embeddings before returning (recommended for cosine search)."""

    thread_workers: int = 1
    """Thread-pool size for offloading blocking model calls off the event loop."""

    title_sep: str = "[SEP]"
    """Separator injected between title and body when building SPECTER2 input.
    SPECTER2 was trained with title + [SEP] + abstract/body concatenation."""

    @classmethod
    def from_env(cls) -> "Specter2Config":
        """Build config from environment variables (all optional, use SPECTER2_ prefix).

        Example::

            SPECTER2_DEVICE=cuda SPECTER2_BATCH_SIZE=32 python ...
        """
        import os

        return cls(
            base_model=os.getenv("SPECTER2_BASE_MODEL", _BASE_MODEL),
            adapter=os.getenv("SPECTER2_ADAPTER", _ADAPTER_PROXIMITY),
            device=os.getenv("SPECTER2_DEVICE", "cpu"),
            batch_size=int(os.getenv("SPECTER2_BATCH_SIZE", "16")),
            max_length=int(os.getenv("SPECTER2_MAX_LENGTH", "512")),
            normalize_embeddings=os.getenv(
                "SPECTER2_NORMALIZE", "true"
            ).lower() != "false",
            thread_workers=int(os.getenv("SPECTER2_THREAD_WORKERS", "1")),
        )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EmbeddedChunk:
    """A parsed text chunk enriched with its SPECTER2 embedding."""

    section: str
    chunk: int
    content: str
    embedding: list[float]

    # Provenance — mirrors document_chunks DB columns
    embedding_model: str = ""
    embedding_adapter: str = ""
    token_count: int | None = None
    embedding_normalized: bool = True

    @property
    def dim(self) -> int:
        return len(self.embedding)

    @property
    def was_truncated(self) -> bool:
        """True if the content was hard-truncated by the tokenizer."""
        return self.token_count is not None and self.token_count >= 512

    def to_dict(self) -> dict[str, object]:
        return {
            "section": self.section,
            "chunk": self.chunk,
            "content": self.content,
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "embedding_adapter": self.embedding_adapter,
            "token_count": self.token_count,
            "embedding_normalized": self.embedding_normalized,
        }

    @classmethod
    def from_chunk(
        cls,
        chunk: ChunkDict,
        embedding: list[float],
        *,
        embedding_model: str = "",
        embedding_adapter: str = "",
        token_count: int | None = None,
        embedding_normalized: bool = True,
    ) -> "EmbeddedChunk":
        return cls(
            section=chunk["section"],
            chunk=chunk["chunk"],
            content=chunk["content"],
            embedding=embedding,
            embedding_model=embedding_model,
            embedding_adapter=embedding_adapter,
            token_count=token_count,
            embedding_normalized=embedding_normalized,
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Specter2Executor:
    """Async SPECTER2 embedding executor.

    Usage::

        async with Specter2Executor() as specter:
            embedded = await specter.encode_chunks(chunks)

    Or with custom config::

        cfg = Specter2Config(device="cuda", batch_size=32)
        async with Specter2Executor(cfg) as specter:
            vectors = await specter.encode(["Text one", "Text two"])
    """

    def __init__(self, config: Specter2Config | None = None) -> None:
        self._cfg = config or Specter2Config()
        # Typed as Any: AutoTokenizer / AutoAdapterModel come from untyped
        # third-party libraries (transformers / adapters) that lack complete
        # stubs. Any is the correct annotation here — not a lazy shortcut.
        self._tokenizer: Any = None
        self._model: Any = None
        self._executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Load the model and tokenizer into memory (blocking, called once)."""
        if self._model is not None:
            return

        logger.info(
            "Loading SPECTER2 base=%s adapter=%s device=%s …",
            self._cfg.base_model,
            self._cfg.adapter,
            self._cfg.device,
        )
        loop = asyncio.get_running_loop()
        self._executor = ThreadPoolExecutor(
            max_workers=self._cfg.thread_workers,
            thread_name_prefix="specter2",
        )
        # Model loading is blocking — run in thread pool
        await loop.run_in_executor(self._executor, self._load_model)
        logger.info("SPECTER2 ready (dim=%d, device=%s)", EMBEDDING_DIM, self._cfg.device)

    async def close(self) -> None:
        """Release model memory and shut down the thread pool."""
        self._model = None
        self._tokenizer = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        logger.info("Specter2Executor closed")

    async def __aenter__(self) -> "Specter2Executor":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode a list of raw text strings into SPECTER2 embeddings.

        Args:
            texts: Plain text strings (already formatted if needed).

        Returns:
            List of 768-dimensional float vectors, one per input text.
        """
        embeddings, _ = await self._encode_with_meta(texts)
        return embeddings

    async def _encode_with_meta(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[int]]:
        """Encode texts and also return per-item token counts."""
        self._require_ready()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(self._encode_sync, texts),
        )

    async def encode_chunks(
        self,
        chunks: list[ChunkDict],
        *,
        title: str | None = None,
    ) -> list[EmbeddedChunk]:
        """Encode a list of parsed :class:`~executor.parser.ChunkDict` objects.

        SPECTER2 was trained on ``title [SEP] abstract`` input.  When *title*
        is provided it is prepended to every chunk's content with the
        configured separator, matching the training distribution.

        Args:
            chunks: Output of :func:`~executor.parser.parse_tei_to_chunks`.
            title:  Optional paper title to prepend to each chunk.

        Returns:
            :class:`EmbeddedChunk` list in the same order as *chunks*.
        """
        self._require_ready()

        texts = [
            f"{title} {self._cfg.title_sep} {c['content']}" if title else c["content"]
            for c in chunks
        ]
        embeddings, token_counts = await self._encode_with_meta(texts)
        return [
            EmbeddedChunk.from_chunk(
                chunk,
                emb,
                embedding_model=self._cfg.base_model,
                embedding_adapter=self._cfg.adapter,
                token_count=tc,
                embedding_normalized=self._cfg.normalize_embeddings,
            )
            for chunk, emb, tc in zip(chunks, embeddings, token_counts)
        ]

    # ------------------------------------------------------------------
    # Synchronous internals (run in thread pool)
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Blocking model load — must be called from a thread, not the event loop."""
        import torch
        from adapters import AutoAdapterModel  # type: ignore[import-untyped]
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        cfg = self._cfg
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
        model = AutoAdapterModel.from_pretrained(cfg.base_model)
        model.load_adapter(cfg.adapter, source="hf", load_as="specter2", set_active=True)
        model.eval()
        model.to(torch.device(cfg.device))

        self._tokenizer = tokenizer
        self._model = model

    def _encode_sync(self, texts: list[str]) -> tuple[list[list[float]], list[int]]:
        """Blocking batch encode — must be called from a thread.

        Returns:
            Tuple of (embeddings, token_counts). token_count == max_length
            indicates the input was truncated.
        """
        import torch

        cfg = self._cfg
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None  # guarded by _require_ready

        all_embeddings: list[list[float]] = []
        all_token_counts: list[int] = []

        for i in range(0, len(texts), cfg.batch_size):
            batch = texts[i : i + cfg.batch_size]

            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            )

            # Record actual (pre-padding) sequence lengths per item
            attention_mask = inputs["attention_mask"]  # (batch, seq_len)
            token_counts: list[int] = attention_mask.sum(dim=1).tolist()
            all_token_counts.extend(token_counts)

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            # CLS-token representation (SPECTER2 convention)
            embeddings: torch.Tensor = outputs.last_hidden_state[:, 0, :]

            if cfg.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().tolist())
            logger.debug(
                "Encoded batch %d-%d / %d (max_tokens=%d)",
                i + 1,
                min(i + cfg.batch_size, len(texts)),
                len(texts),
                max(token_counts),
            )

        return all_embeddings, all_token_counts

    def _require_ready(self) -> None:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(
                "Specter2Executor is not started. "
                "Use `async with Specter2Executor() as s:` or call `await s.start()` first."
            )
