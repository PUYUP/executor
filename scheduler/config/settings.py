"""
config/settings.py
───────────────────
Centralised settings — loaded from environment / .env file.
"""

from __future__ import annotations

from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Redis ──────────────────────────────────────────────────
    redis_host: str     = "localhost"  # from container: redis
    redis_port: int     = 6379
    redis_db: int       = 0
    redis_password: str = ""

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # ── ArXiv Scraping ─────────────────────────────────────────
    arxiv_topics: List[str] = Field(
        default=[
            "cs.AI",
            "cs.CL",   # Computation and Language (NLP)
            "cs.LG",   # Machine Learning
            "cs.IR",   # Information Retrieval
            "stat.ML",
        ]
    )
    arxiv_base_url: str     = "http://export.arxiv.org/api/query"
    max_results_per_topic: int = 50
    scrape_interval_seconds: float = 21_600.0    # 6 hours
    download_timeout_seconds: int  = 120

    # ── PDF Processing ─────────────────────────────────────────
    pdf_download_dir: str   = "/Volumes/SSD1/Private/Curio"  # docker use: /app/downloads
    pdf_max_size_mb: int    = 50

    # ── Chunking ───────────────────────────────────────────────
    chunk_size_tokens: int      = 512
    chunk_overlap_tokens: int   = 64
    min_chunk_chars: int        = 100   # discard tiny chunks

    # ── Embeddings ─────────────────────────────────────────────
    embedding_provider: str     = "openai"       # "openai" | "local"
    openai_api_key: str         = ""
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int   = 100             # texts per API call
    local_embedding_model: str  = "BAAI/bge-small-en-v1.5"

    # ── Logging ────────────────────────────────────────────────
    log_level: str          = "INFO"
    log_format: str         = "json"             # "json" | "console"

    @field_validator("arxiv_topics", mode="before")
    @classmethod
    def parse_topics(cls, v):
        if isinstance(v, str):
            return [t.strip() for t in v.split(",")]
        return v


settings = Settings()