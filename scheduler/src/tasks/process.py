"""
tasks/process.py
────────────────────────────
Tier-2 tasks (queue: process)
─────────────────────────────────────────────────────────────────────────
Flow (continued from scrape.py):
  download_pdf  →  parse_pdf  →  clean_text  →  chunk_document
                                                      └─► generate_embeddings
─────────────────────────────────────────────────────────────────────────
Design notes:
  • parse_pdf   – layout-aware extraction via PyMuPDF; sections detected
                  by font-size heuristics.
  • clean_text  – remove boilerplate (headers/footers, ref numbers),
                  normalise whitespace.
  • chunk_document – recursive token-aware splitting with metadata
                     inheritance per chunk (paper id, section, page range).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf as fitz
import structlog
from celery import signature
from langchain_text_splitters import RecursiveCharacterTextSplitter

from main import app
from utils.text_cleaner import clean_academic_text
from config.settings import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 4 of 5 — parse_pdf
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="tasks.process.parse_pdf",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    queue="process",
    time_limit=300,
    soft_time_limit=240,
    ignore_result=False,
)
def parse_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts structured text from a downloaded PDF.

    Returns metadata enriched with:
      - `sections`: list of {title, text, page_start, page_end}
      - `full_text`: concatenated plain text (fallback)
      - `page_count`: int
    """
    arxiv_id     = metadata["arxiv_id"]
    pdf_path_str = metadata.get("local_pdf_path")

    # Skip papers that were flagged during download
    if not pdf_path_str or metadata.get("skip_reason"):
        log.warning(
            "parse_pdf.skip",
            arxiv_id=arxiv_id,
            reason=metadata.get("skip_reason", "no pdf path"),
        )
        return metadata

    pdf_path = Path(pdf_path_str)
    if not pdf_path.exists():
        log.error("parse_pdf.file_missing", arxiv_id=arxiv_id, path=str(pdf_path))
        raise self.retry(exc=FileNotFoundError(str(pdf_path)))

    log.info("parse_pdf.start", arxiv_id=arxiv_id)

    try:
        sections, page_count = _extract_sections(pdf_path)
    except FileNotFoundError as exc:
        log.error("parse_pdf.corrupt_pdf", arxiv_id=arxiv_id, error=str(exc))
        metadata["skip_reason"] = "corrupt_pdf"
        return metadata
    except Exception as exc:
        log.error("parse_pdf.failed", arxiv_id=arxiv_id, error=str(exc))
        raise self.retry(exc=exc)

    # Prepend abstract as its own section so it's always embedded
    abstract_section = {
        "title": "Abstract",
        "text": metadata.get("abstract", ""),
        "page_start": 0,
        "page_end": 0,
    }
    all_sections = [abstract_section] + sections

    full_text = "\n\n".join(s["text"] for s in all_sections if s["text"])

    log.info(
        "parse_pdf.done",
        arxiv_id=arxiv_id,
        page_count=page_count,
        sections=len(sections),
        chars=len(full_text),
    )

    metadata["sections"]   = all_sections
    metadata["full_text"]  = full_text
    metadata["page_count"] = page_count

    # Chain to clean_text
    (
        clean_text.s(metadata).set(queue="process")
        | chunk_document.s().set(queue="process")
        | signature(
            "tasks.embed.generate_embeddings",
            queue="embed",
            immutable=False,
        )
    ).apply_async()

    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 4b — clean_text
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="tasks.process.clean_text",
    bind=True,
    max_retries=2,
    queue="process",
    ignore_result=False,
)
def clean_text(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans raw extracted text in every section:
      - strip running headers/footers (page numbers, journal name)
      - normalise unicode & whitespace
      - remove reference section (not useful for RAG body search)
      - remove figure/table captions inline markers (Figure 1, Table 2…)
    """
    arxiv_id = metadata["arxiv_id"]
    log.info("clean_text.start", arxiv_id=arxiv_id)

    cleaned_sections = []
    for section in metadata.get("sections", []):
        cleaned = section.copy()
        cleaned["text"] = clean_academic_text(section["text"])
        if len(cleaned["text"]) >= settings.min_chunk_chars:
            cleaned_sections.append(cleaned)

    # Also clean the full_text field
    metadata["sections"]  = cleaned_sections
    metadata["full_text"] = clean_academic_text(metadata.get("full_text", ""))

    log.info(
        "clean_text.done",
        arxiv_id=arxiv_id,
        sections_kept=len(cleaned_sections),
    )
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Task 5 of 5 — chunk_document
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="tasks.process.chunk_document",
    bind=True,
    max_retries=2,
    queue="process",
    ignore_result=False,
)
def chunk_document(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Splits each section into RAG-ready chunks.

    Each chunk is a dict:
    {
        "chunk_id":      "<arxiv_id>_<section_idx>_<chunk_idx>",
        "arxiv_id":      str,
        "title":         str,         # paper title
        "section":       str,         # section heading
        "text":          str,         # chunk body
        "page_start":    int,
        "page_end":      int,
        "authors":       list[str],
        "categories":    list[str],
        "published":     str,         # ISO date
        "token_count":   int,         # approximate
    }

    Returns metadata with `chunks` key added.
    """
    arxiv_id = metadata["arxiv_id"]
    log.info("chunk_document.start", arxiv_id=arxiv_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_tokens * 4,     # ~4 chars per token
        chunk_overlap=settings.chunk_overlap_tokens * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks: List[Dict[str, Any]] = []

    for sec_idx, section in enumerate(metadata.get("sections", [])):
        raw_text = section.get("text", "")
        if len(raw_text) < settings.min_chunk_chars:
            continue

        split_texts = splitter.split_text(raw_text)

        for chunk_idx, chunk_text in enumerate(split_texts):
            if len(chunk_text) < settings.min_chunk_chars:
                continue

            chunk: Dict[str, Any] = {
                "chunk_id":    f"{arxiv_id}_{sec_idx}_{chunk_idx}",
                "arxiv_id":    arxiv_id,
                "title":       metadata.get("title", ""),
                "section":     section.get("title", "Body"),
                "text":        chunk_text,
                "page_start":  section.get("page_start", 0),
                "page_end":    section.get("page_end", 0),
                "authors":     metadata.get("authors", []),
                "categories":  metadata.get("categories", []),
                "published":   metadata.get("published", ""),
                "doi":         metadata.get("doi", ""),
                # Approximate token count (chars / 4)
                "token_count": len(chunk_text) // 4,
            }
            chunks.append(chunk)

    log.info(
        "chunk_document.done",
        arxiv_id=arxiv_id,
        chunks=len(chunks),
        avg_tokens=int(sum(c["token_count"] for c in chunks) / max(len(chunks), 1)),
    )

    metadata["chunks"] = chunks
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# PDF Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sections(
    pdf_path: Path,
) -> tuple[List[Dict[str, Any]], int]:
    """
    Extracts text grouped by section using PyMuPDF.

    Heuristic: a line is treated as a section heading when its font
    size is ≥ 1.2× the median body font size on that page.
    """
    doc  = fitz.open(str(pdf_path))
    page_count = len(doc)

    # Collect all text blocks with font info
    raw_blocks: List[Dict] = []
    for page_num in range(page_count):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block["type"] != 0:          # skip image blocks
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    raw_blocks.append({
                        "text":      span["text"],
                        "size":      span["size"],
                        "font":      span["font"],
                        "page":      page_num,
                        "is_bold":   "Bold" in span["font"] or "bold" in span["font"],
                    })

    if not raw_blocks:
        return [], page_count

    # Compute median font size (proxy for body text size)
    sizes = sorted(b["size"] for b in raw_blocks if b["text"].strip())
    median_size = sizes[len(sizes) // 2] if sizes else 12.0

    # Group into sections
    sections: List[Dict[str, Any]] = []
    current_section: Dict[str, Any] = {
        "title": "Introduction",
        "text":  "",
        "page_start": 0,
        "page_end": 0,
    }

    for block in raw_blocks:
        text = block["text"].strip()
        if not text:
            continue

        is_heading = (
            block["size"] >= median_size * 1.2
            and len(text) < 120          # headings are short
            and not text[-1] == ","      # not mid-sentence
        )

        if is_heading and current_section["text"]:
            current_section["page_end"] = block["page"]
            sections.append(current_section)
            current_section = {
                "title":      text,
                "text":       "",
                "page_start": block["page"],
                "page_end":   block["page"],
            }
        else:
            current_section["text"] += text + " "
            current_section["page_end"] = block["page"]

    if current_section["text"]:
        sections.append(current_section)

    doc.close()
    return sections, page_count