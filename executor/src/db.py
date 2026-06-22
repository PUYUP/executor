"""PostgreSQL + pgvector database connection and schema management.

Provides:
* :class:`DatabaseConfig`  — connection parameters (env-var driven).
* :class:`DatabasePool`    — async connection pool lifecycle manager.
* :func:`ensure_schema`    — idempotent DDL (extension + tables + indexes).

Only the config and plumbing live here.  Query helpers belong in their
own domain modules (e.g. ``executor/repository.py``).

Prerequisites
-------------
* PostgreSQL is already running (Docker or bare-metal).
* The ``vector`` extension must be available in the server image
  (``pgvector/pgvector`` images ship it by default).

Environment variables (all optional — shown with defaults)
----------------------------------------------------------
.. code-block:: bash

    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=curiosift
    DB_USER=postgres
    DB_PASSWORD=postgres
    DB_MIN_POOL=2
    DB_MAX_POOL=10
    DB_COMMAND_TIMEOUT=30
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatabaseConfig:
    """Connection parameters for the PostgreSQL / pgvector database."""

    host: str = "localhost"
    port: int = 5432
    database: str = "curiosift"
    user: str = "postgres"
    password: str = "postgres"

    # Pool sizing
    min_size: int = 2
    max_size: int = 10

    # Per-statement timeout in seconds (0 = no limit)
    command_timeout: float = 30.0

    # Passed verbatim to asyncpg as server_settings
    server_settings: dict[str, str] = field(
        default_factory=lambda: {"application_name": "curiosift-executor"}
    )

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Build config entirely from environment variables.

        Example::

            DB_HOST=db DB_PASSWORD=secret python main.py
        """
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "curiosift"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            min_size=int(os.getenv("DB_MIN_POOL", "2")),
            max_size=int(os.getenv("DB_MAX_POOL", "10")),
            command_timeout=float(os.getenv("DB_COMMAND_TIMEOUT", "30")),
        )

    @property
    def dsn(self) -> str:
        """Return a ``postgresql://`` DSN string (password redacted for logging)."""
        return (
            f"postgresql://{self.user}:***@{self.host}:{self.port}/{self.database}"
        )

    def _dsn_with_password(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


# ---------------------------------------------------------------------------
# DDL — schema definitions
# ---------------------------------------------------------------------------

_DDL = """\
-- Enable pgvector extension (idempotent)
CREATE EXTENSION IF NOT EXISTS vector;

-- -----------------------------------------------------------------------
-- papers: rich bibliographic metadata
--
-- Design decisions
-- ----------------
-- * paper_id   : caller-controlled stable key (e.g. SHA-256 of DOI / arXiv ID).
-- * doi / arxiv_id / pmid / pmcid / s2_id : well-known external identifiers,
--   nullable because many sources only expose a subset.
-- * authors    : JSONB array — flexible enough to store name, affiliation,
--   ORCID and email per author without a separate join table.
--   Shape: [{"name": "…", "first": "…", "last": "…",
--            "affiliation": "…", "orcid": "…", "email": "…"}, …]
-- * affiliations : JSONB array of institution objects extracted by GROBID.
--   Shape: [{"name": "…", "department": "…", "country": "…"}, …]
-- * references_count / citations_count : bibliometric signals, populated
--   asynchronously from Semantic Scholar / OpenAlex.
-- * processing_status : lifecycle state machine —
--   pending → indexing → done | failed
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS papers (
    -- Primary key --------------------------------------------------------
    id                  BIGSERIAL       PRIMARY KEY,

    -- Stable internal identifier (required) ------------------------------
    paper_id            TEXT            NOT NULL UNIQUE,

    -- External identifiers (all optional, unique when set) ---------------
    doi                 TEXT            UNIQUE,          -- e.g. 10.1145/3442188.3445922
    arxiv_id            TEXT            UNIQUE,          -- e.g. 2301.07041
    pmid                TEXT            UNIQUE,          -- PubMed ID
    pmcid               TEXT            UNIQUE,          -- PubMed Central ID
    s2_id               TEXT            UNIQUE,          -- Semantic Scholar corpus ID

    -- Core bibliographic -------------------------------------------------
    title               TEXT            NOT NULL,
    abstract            TEXT,
    year                SMALLINT,                        -- publication year (fast range filter)
    date_published      DATE,                            -- full date when available

    -- People & institutions (structured JSONB) ---------------------------
    authors             JSONB           NOT NULL DEFAULT '[]'::jsonb,
    -- [{name, first, last, affiliation, orcid, email}, …]

    affiliations        JSONB           NOT NULL DEFAULT '[]'::jsonb,
    -- [{name, department, country}, …]

    -- Venue / publication ------------------------------------------------
    venue               TEXT,                            -- journal or conference name
    venue_type          TEXT,                            -- 'journal' | 'conference' | 'preprint' | 'book'
    publisher           TEXT,
    volume              TEXT,
    issue               TEXT,
    pages               TEXT,                            -- e.g. "1–14" or "1432"

    -- Classification & discovery -----------------------------------------
    keywords            TEXT[],                          -- author-supplied keywords
    fields_of_study     TEXT[],                          -- e.g. {'Computer Science','NLP'}
    language            TEXT            NOT NULL DEFAULT 'en',

    -- Access & availability ----------------------------------------------
    pdf_url             TEXT,
    open_access         BOOLEAN,
    license             TEXT,                            -- SPDX or free-text

    -- Bibliometric signals (populated asynchronously) --------------------
    references_count    INTEGER,
    citations_count     INTEGER,

    -- Provenance ---------------------------------------------------------
    source              TEXT,                            -- 'arxiv' | 'pubmed' | 'upload' | …
    source_metadata     JSONB           NOT NULL DEFAULT '{}'::jsonb,
    -- arbitrary source-specific raw fields for forward-compatibility

    -- Processing lifecycle -----------------------------------------------
    grobid_version      TEXT,
    processing_status   TEXT            NOT NULL DEFAULT 'pending',
    -- 'pending' → 'indexing' → 'done' | 'failed'
    error_message       TEXT,

    -- Timestamps ---------------------------------------------------------
    created_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Constraints --------------------------------------------------------
    CONSTRAINT papers_status_check
        CHECK (processing_status IN ('pending', 'indexing', 'done', 'failed')),
    CONSTRAINT papers_venue_type_check
        CHECK (venue_type IN ('journal', 'conference', 'preprint', 'book', 'thesis', 'report', NULL))
);

-- Indexes on papers ------------------------------------------------------

-- Exact identifier lookups (covered by UNIQUE but explicit for clarity)
CREATE INDEX IF NOT EXISTS idx_papers_doi        ON papers (doi)       WHERE doi IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id   ON papers (arxiv_id)  WHERE arxiv_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_pmid       ON papers (pmid)      WHERE pmid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_s2_id      ON papers (s2_id)     WHERE s2_id IS NOT NULL;

-- Date / year range filtering
CREATE INDEX IF NOT EXISTS idx_papers_year        ON papers (year)          WHERE year IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_papers_date_pub    ON papers (date_published) WHERE date_published IS NOT NULL;

-- Processing queue
CREATE INDEX IF NOT EXISTS idx_papers_status     ON papers (processing_status);

-- Full-text search on title (pg_trgm trigram index for ILIKE / similarity)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_papers_title_trgm
    ON papers USING gin (title gin_trgm_ops);

-- GIN index for JSONB authors array (supports @> containment queries)
CREATE INDEX IF NOT EXISTS idx_papers_authors_gin
    ON papers USING gin (authors);

-- GIN index for keyword and fields-of-study arrays
CREATE INDEX IF NOT EXISTS idx_papers_keywords_gin
    ON papers USING gin (keywords);
CREATE INDEX IF NOT EXISTS idx_papers_fos_gin
    ON papers USING gin (fields_of_study);

-- -----------------------------------------------------------------------
-- document_chunks: SPECTER2-embedded text chunks
--
-- Production design decisions
-- ---------------------------
-- * section_order : 0-based global reading order of sections within the
--   paper. Lets callers reconstruct document structure without re-parsing.
--
-- * content_hash  : SHA-256 of the raw content string, stored as a
--   generated column. Enables skip-on-no-change re-indexing:
--     SELECT id FROM document_chunks
--     WHERE paper_id = $1 AND content_hash = $2 AND embedding IS NOT NULL;
--
-- * word_count    : stored computed; avoids recomputing at query time and
--   supports filtering short/long chunks without loading content.
--
-- * embedding_model / embedding_adapter : provenance columns. When the
--   model is upgraded, stale chunks are found instantly:
--     SELECT * FROM document_chunks
--     WHERE embedding_model != 'allenai/specter2_base';
--
-- * token_count   : actual tokens fed to the encoder. A value equal to
--   max_length (default 512) means the chunk was hard-truncated and may
--   have lost trailing content. Useful for quality auditing.
--
-- * embedding_normalized : explicit flag; TRUE means dot-product distance
--   equals cosine distance, which halves query cost in pgvector.
--
-- * HNSW vs IVFFlat : IVFFlat requires a separate VACUUM+ANALYZE training
--   step, delivers poor recall on small tables, and needs lists retuned as
--   the table grows. HNSW (available since pgvector 0.5 / Postgres 15) is
--   trained incrementally, works from the first row, and gives consistently
--   better recall at the same ef_search budget.
--   m=16 (connections per graph node) and ef_construction=64 (build-time
--   candidate list) are conservative, high-quality defaults.
--
-- * Partial indexes   : all expensive indexes are filtered WHERE
--   embedding IS NOT NULL so they never bloat on un-embedded rows.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id                      BIGSERIAL       PRIMARY KEY,
    paper_id                TEXT            NOT NULL
                                REFERENCES papers(paper_id) ON DELETE CASCADE,

    -- Document position -----------------------------------------------
    section                 TEXT            NOT NULL,
    section_order           INTEGER         NOT NULL,
    -- 0-based global position of this section across the paper
    chunk                   INTEGER         NOT NULL,
    -- 0-based index of this chunk within the section

    -- Content ---------------------------------------------------------
    chunk_type              TEXT            NOT NULL DEFAULT 'body',
    -- 'abstract' | 'body' | 'conclusion' | 'caption' | 'equation' | 'other'
    content                 TEXT            NOT NULL,

    word_count              INTEGER         NOT NULL
                                GENERATED ALWAYS AS (
                                    array_length(
                                        string_to_array(
                                            trim(regexp_replace(content, '\\s+', ' ', 'g')),
                                            ' '
                                        ),
                                        1
                                    )
                                ) STORED,
    -- Recomputed from content on every write; never stale.

    content_hash            TEXT            NOT NULL
                                GENERATED ALWAYS AS (
                                    encode(sha256(content::bytea), 'hex')
                                ) STORED,
    -- SHA-256 hex digest of the raw content string.

    -- Embedding -------------------------------------------------------
    embedding               vector(768),
    -- NULL until the embedding worker completes this chunk.

    embedding_model         TEXT,
    -- HuggingFace base model ID, e.g. 'allenai/specter2_base'

    embedding_adapter       TEXT,
    -- Active adapter, e.g. 'allenai/specter2' or 'allenai/specter2_adhoc_query'

    embedding_normalized    BOOLEAN         NOT NULL DEFAULT TRUE,
    -- TRUE ⟹ L2-normalised; cosine sim == dot product (faster at query time)

    token_count             INTEGER,
    -- Tokens consumed by the tokenizer. token_count = 512 ⟹ truncation occurred.

    -- Timestamps ------------------------------------------------------
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Constraints -----------------------------------------------------
    UNIQUE (paper_id, section, chunk),

    CONSTRAINT chunks_type_check CHECK (
        chunk_type IN ('abstract', 'body', 'conclusion', 'caption', 'equation', 'other')
    ),
    CONSTRAINT chunks_section_order_nonneg  CHECK (section_order >= 0),
    CONSTRAINT chunks_chunk_nonneg          CHECK (chunk >= 0),
    CONSTRAINT chunks_token_count_positive  CHECK (token_count IS NULL OR token_count > 0),
    CONSTRAINT chunks_word_count_positive   CHECK (word_count > 0)
);

-- Indexes on document_chunks ------------------------------------------

-- HNSW vector index for cosine ANN search (partial: embedded rows only).
-- No pre-training required. Consistent high recall from first insert.
--   m=16              → connections per graph node (memory ↔ recall)
--   ef_construction=64 → build-time candidate list (quality ↔ speed)
-- At query time tune SET hnsw.ef_search = 100; for higher recall.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

-- Reading-order retrieval: reconstruct a paper's full chunk sequence.
CREATE INDEX IF NOT EXISTS idx_chunks_paper_order
    ON document_chunks (paper_id, section_order, chunk);

-- Embedding queue: find chunks still waiting for a vector.
CREATE INDEX IF NOT EXISTS idx_chunks_unembedded
    ON document_chunks (paper_id, created_at)
    WHERE embedding IS NULL;

-- Model staleness scan: find chunks embedded with an old model/adapter.
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_model
    ON document_chunks (embedding_model, embedding_adapter)
    WHERE embedding IS NOT NULL;

-- Deduplication: skip re-embedding when content is unchanged.
CREATE INDEX IF NOT EXISTS idx_chunks_content_hash
    ON document_chunks (content_hash)
    WHERE embedding IS NOT NULL;

-- Chunk-type filtering (e.g. abstract-only retrieval).
CREATE INDEX IF NOT EXISTS idx_chunks_type
    ON document_chunks (chunk_type, paper_id);
"""


# ---------------------------------------------------------------------------
# Pool manager
# ---------------------------------------------------------------------------


class DatabasePool:
    """Async PostgreSQL connection pool with pgvector support.

    Usage::

        async with DatabasePool() as db:
            await db.ensure_schema()
            async with db.pool.acquire() as conn:
                rows = await conn.fetch("SELECT 1")

    Or manual lifecycle::

        db = DatabasePool(DatabaseConfig.from_env())
        await db.start()
        ...
        await db.close()
    """

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self._cfg = config or DatabaseConfig.from_env()
        self._pool: "asyncpg.Pool | None" = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the connection pool and register the pgvector codec."""
        if self._pool is not None:
            return

        import asyncpg  # type: ignore[import-untyped]
        from pgvector.asyncpg import register_vector  # type: ignore[import-untyped]

        cfg = self._cfg

        async def _init_conn(conn: asyncpg.Connection) -> None:
            """Called for every new connection: register the vector type."""
            await register_vector(conn)

        logger.info("Connecting to %s …", cfg.dsn)
        self._pool = await asyncpg.create_pool(
            dsn=cfg._dsn_with_password(),
            min_size=cfg.min_size,
            max_size=cfg.max_size,
            command_timeout=cfg.command_timeout,
            server_settings=cfg.server_settings,
            init=_init_conn,
        )
        logger.info(
            "Database pool ready (min=%d max=%d)", cfg.min_size, cfg.max_size
        )

    async def close(self) -> None:
        """Gracefully drain and close all connections."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            logger.info("Database pool closed")

    async def __aenter__(self) -> "DatabasePool":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def pool(self) -> "asyncpg.Pool":
        """The underlying asyncpg pool (raises if not started)."""
        if self._pool is None:
            raise RuntimeError(
                "DatabasePool is not started. "
                "Use `async with DatabasePool() as db:` or call `await db.start()` first."
            )
        return self._pool

    async def ensure_schema(self) -> None:
        """Run DDL idempotently to create extension, tables, and indexes.

        Safe to call on every startup — all statements use ``IF NOT EXISTS``.
        """
        async with self.pool.acquire() as conn:
            await conn.execute(_DDL)
        logger.info("Database schema verified / created")

    async def ping(self) -> bool:
        """Return True if the database is reachable."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Database ping failed: %s", exc)
            return False
