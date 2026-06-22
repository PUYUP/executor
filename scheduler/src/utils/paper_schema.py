from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    published: Optional[str] = None
    updated: Optional[str] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    primary_category: Optional[str] = None
    local_pdf_path: Optional[str] = None
    skip_reason: Optional[str] = None
    download_status: str = "pending"
    parse_status: str = "pending"
    chunk_status: str = "pending"
    embedding_status: str = "pending"