"""
celery_app/utils/paper_schema.py
──────────────────────────────────
Pydantic models for validated data flowing between tasks.
All models use model_dump() to produce plain dicts for Celery serialisation.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PaperMetadata(BaseModel):
    """
    Validated metadata for a single ArXiv paper.
    Produced by scrape_paper_metadata, passed to every downstream task.
    """
    arxiv_id:         str
    title:            str
    abstract:         str
    authors:          List[str]          = Field(default_factory=list)
    categories:       List[str]          = Field(default_factory=list)
    primary_category: str                = ""
    published:        str                = ""   # ISO 8601
    updated:          str                = ""   # ISO 8601
    pdf_url:          str                = ""
    doi:              str                = ""
    journal_ref:      str                = ""

    # Populated by downstream tasks
    local_pdf_path:   Optional[str]      = None
    skip_reason:      Optional[str]      = None
    sections:         List[dict]         = Field(default_factory=list)
    full_text:        str                = ""
    page_count:       int                = 0
    chunks:           List[dict]         = Field(default_factory=list)

    @field_validator("title", "abstract", mode="before")
    @classmethod
    def normalise_whitespace(cls, v: str) -> str:
        return " ".join(v.split()) if isinstance(v, str) else v

    @field_validator("arxiv_id", mode="before")
    @classmethod
    def strip_version(cls, v: str) -> str:
        """Normalise '2401.12345v2' → '2401.12345'."""
        if isinstance(v, str) and "v" in v:
            return v.rsplit("v", 1)[0]
        return v


class ChunkSchema(BaseModel):
    """
    Single RAG chunk — what ultimately lands in the vector store.
    Embedded inside PaperMetadata.chunks list.
    """
    chunk_id:        str
    arxiv_id:        str
    title:           str
    section:         str
    text:            str
    page_start:      int               = 0
    page_end:        int               = 0
    authors:         List[str]         = Field(default_factory=list)
    categories:      List[str]         = Field(default_factory=list)
    published:       str               = ""
    doi:             str               = ""
    token_count:     int               = 0
    # Populated after embedding
    embedding:       Optional[List[float]] = None
    embedding_model: str               = ""
    embedding_dim:   int               = 0
