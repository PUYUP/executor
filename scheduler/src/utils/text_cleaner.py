"""
celery_app/utils/text_cleaner.py
──────────────────────────────────
Cleans raw text extracted from ArXiv PDFs before chunking.

Order of operations:
  1. Unicode normalisation
  2. Remove running headers / footers (page numbers, journal names)
  3. Collapse excessive whitespace
  4. Remove reference list (not useful for RAG body retrieval)
  5. Strip figure / table caption boilerplate markers
  6. Strip LaTeX artefacts that survive PDF extraction
"""

from __future__ import annotations

import re
import unicodedata


# ── Patterns ──────────────────────────────────────────────────────────────────

# Bare page numbers (lines that are only a digit or "Page N")
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:Page\s*)?\d{1,4}\s*$", re.MULTILINE)

# arXiv preprint headers that appear on each page
_ARXIV_HEADER_RE = re.compile(
    r"arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w.]+\]\s+\d{1,2}\s+\w+\s+\d{4}",
    re.IGNORECASE,
)

# "Figure N:", "Fig. N:", "Table N:", "Algorithm N:" captions
_CAPTION_RE = re.compile(
    r"(Figure|Fig\.|Table|Algorithm|Listing)\s+\d+[.:]",
    re.IGNORECASE,
)

# Inline citation markers: [1], [1,2], [1-3], (Smith et al., 2020)
_CITATION_INLINE_RE = re.compile(
    r"\[\d[\d,\s–\-]*\]"
    r"|(\([A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}[a-z]?(?:;\s*[A-Z][a-z]+(?:\s+et\s+al\.?)?,\s*\d{4}[a-z]?)*\))"
)

# LaTeX math remnants that leak through PyMuPDF: \mathbf, \mathrm, etc.
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+")

# Multiple spaces / tabs compressed to single space
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

# Three or more newlines → two (paragraph break)
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Hyphenated line breaks: "pro-\ncess" → "process"
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# References / Bibliography section header — everything after this is dropped
_REFERENCES_RE = re.compile(
    r"\n\s*(?:References|Bibliography|Works\s+Cited)\s*\n",
    re.IGNORECASE,
)

# Acknowledgements section — strip it (not useful for RAG)
_ACK_RE = re.compile(
    r"\n\s*Acknowledge?ments?\s*\n.*",
    re.IGNORECASE | re.DOTALL,
)


def clean_academic_text(text: str) -> str:
    """
    Full cleaning pipeline for a text block extracted from an ArXiv PDF.
    Returns cleaned string.
    """
    if not text:
        return ""

    # 1. Unicode normalisation (NFKC: ligatures, fancy quotes → ASCII)
    text = unicodedata.normalize("NFKC", text)

    # 2. Fix hyphenated line-breaks
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)

    # 3. Drop references / acknowledgements sections
    ref_match = _REFERENCES_RE.search(text)
    if ref_match:
        text = text[: ref_match.start()]

    ack_match = _ACK_RE.search(text)
    if ack_match:
        text = text[: ack_match.start()]

    # 4. Remove arXiv running headers
    text = _ARXIV_HEADER_RE.sub("", text)

    # 5. Remove bare page numbers
    text = _PAGE_NUMBER_RE.sub("", text)

    # 6. Strip inline citation markers (optionally keep for provenance)
    text = _CITATION_INLINE_RE.sub("", text)

    # 7. Strip figure / table caption tokens
    text = _CAPTION_RE.sub("", text)

    # 8. Remove surviving LaTeX commands
    text = _LATEX_CMD_RE.sub("", text)

    # 9. Normalise whitespace
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 10. Strip leading / trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text  = "\n".join(line for line in lines if line)

    return text.strip()


def truncate_to_tokens(text: str, max_tokens: int, chars_per_token: int = 4) -> str:
    """Quick token-count approximation truncation (no tokeniser dependency)."""
    max_chars = max_tokens * chars_per_token
    if len(text) <= max_chars:
        return text
    # Truncate at last sentence boundary within limit
    truncated = text[:max_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.8:
        return truncated[: last_period + 1]
    return truncated