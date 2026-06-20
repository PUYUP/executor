"""TEI XML → SPECTER2-ready chunk parser.

Converts GROBID's TEI XML output into a flat list of text chunks suitable
for embedding with SPECTER2 (or any other sentence/document encoder).

Output schema (one item per chunk):

    {
        "section": "Introduction",   # normalised section heading
        "chunk":   0,                # 0-based index within the section
        "content": "Full text …"     # clean, whitespace-normalised text
    }

Design notes
------------
* Chunking is paragraph-aware: adjacent short paragraphs are merged until
  the chunk reaches `max_words`.  Paragraphs that exceed `max_words` are
  split on sentence boundaries.
* SPECTER2 uses a 512-subword-token window.  400 words is a conservative
  safe limit that avoids truncation for typical academic prose.
* All XML is parsed with the stdlib `xml.etree.ElementTree`; no heavy NLP
  dependencies are required.
* Inline TEI elements (``<ref>``, ``<formula>``, ``<figure>``, …) are
  reduced to their text content so no markup leaks into the output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterator, TypedDict
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# TEI namespace
# ---------------------------------------------------------------------------

_TEI_NS = "http://www.tei-c.org/ns/1.0"
_NS = {"tei": _TEI_NS}


def _tag(local: str) -> str:
    """Return a Clark-notation tag, e.g. ``{http://…}body``."""
    return f"{{{_TEI_NS}}}{local}"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class ChunkDict(TypedDict):
    """JSON-serialisable dict shape for a single SPECTER2 chunk."""

    section: str
    chunk: int
    content: str


@dataclass
class Chunk:
    """A single text chunk ready for SPECTER2 encoding."""

    section: str
    chunk: int
    content: str

    def to_dict(self) -> ChunkDict:
        return ChunkDict(section=self.section, chunk=self.chunk, content=self.content)


@dataclass
class ParseConfig:
    """Tunable parameters for the parser."""

    max_words: int = 400
    """Maximum words per chunk (SPECTER2 safe limit ≈ 400 words / 512 tokens)."""

    min_words: int = 10
    """Chunks shorter than this are dropped (noise / lone headings)."""

    include_abstract: bool = True
    """Whether to include the abstract as a dedicated section."""

    include_title: bool = True
    """Whether to prepend title text to the abstract chunk."""

    normalise_section_names: bool = True
    """Lowercase & strip numeric prefixes from section headings."""

    skip_sections: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "acknowledgements",
                "acknowledgments",
                "funding",
                "conflict of interest",
                "competing interests",
                "author contributions",
                "supplementary",
                "appendix",
            }
        )
    )
    """Section headings (normalised) to exclude from the output."""


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def _iter_text(element: ET.Element) -> Iterator[str]:
    """Yield all text content from *element*, depth-first, skipping tags."""
    if element.text:
        yield element.text
    for child in element:
        # Skip figure / table captions — they tend to be noisy
        if child.tag in (_tag("figure"), _tag("note")):
            pass
        else:
            yield from _iter_text(child)
        if child.tail:
            yield child.tail


def _element_text(element: ET.Element) -> str:
    """Return clean, whitespace-normalised text for *element*."""
    raw = " ".join(_iter_text(element))
    # Normalise unicode (e.g. soft hyphens, non-breaking spaces)
    raw = unicodedata.normalize("NFKC", raw)
    # Collapse runs of whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


# ---------------------------------------------------------------------------
# Chunking helpers
# ---------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _split_by_sentences(text: str, max_words: int) -> list[str]:
    """Split *text* into pieces of at most *max_words*, on sentence boundaries."""
    sentences = _SENTENCE_END.split(text)
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0

    for sent in sentences:
        w = _word_count(sent)
        if current_words + w > max_words and current:
            pieces.append(" ".join(current))
            current = [sent]
            current_words = w
        else:
            current.append(sent)
            current_words += w

    if current:
        pieces.append(" ".join(current))
    return [p for p in pieces if p.strip()]


def _paragraphs_to_chunks(
    paragraphs: list[str],
    max_words: int,
    min_words: int,
) -> list[str]:
    """Merge short paragraphs into chunks; split long ones on sentences."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_words = 0

    for para in paragraphs:
        para_words = _word_count(para)

        if para_words > max_words:
            # Flush current buffer first
            if buffer:
                chunks.append(" ".join(buffer))
                buffer, buffer_words = [], 0
            # Split the oversized paragraph by sentences
            chunks.extend(_split_by_sentences(para, max_words))
            continue

        if buffer_words + para_words > max_words and buffer:
            chunks.append(" ".join(buffer))
            buffer, buffer_words = [para], para_words
        else:
            buffer.append(para)
            buffer_words += para_words

    if buffer:
        chunks.append(" ".join(buffer))

    return [c for c in chunks if _word_count(c) >= min_words]


# ---------------------------------------------------------------------------
# Section name normalisation
# ---------------------------------------------------------------------------

_LEADING_NUM = re.compile(r"^[\d.]+\s*")


def _normalise_section(name: str) -> str:
    name = _LEADING_NUM.sub("", name).strip()
    return name.lower()


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


class TEIParser:
    """Parse GROBID TEI XML into SPECTER2-ready :class:`Chunk` objects.

    Usage::

        parser = TEIParser()
        chunks = parser.parse(tei_xml_string)
        json_ready = [c.to_dict() for c in chunks]
    """

    def __init__(self, config: ParseConfig | None = None) -> None:
        self._cfg = config or ParseConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, tei_xml: str) -> list[Chunk]:
        """Parse a GROBID TEI XML string into a list of :class:`Chunk`.

        Args:
            tei_xml: Raw TEI XML returned by :class:`~executor.grobid.GrobidExecutor`.

        Returns:
            Ordered list of chunks across all sections.
        """
        root = ET.fromstring(tei_xml)
        chunks: list[Chunk] = []

        if self._cfg.include_abstract:
            chunks.extend(self._parse_abstract(root))

        chunks.extend(self._parse_body(root))
        return chunks

    # ------------------------------------------------------------------
    # Section parsers
    # ------------------------------------------------------------------

    def _parse_abstract(self, root: ET.Element) -> list[Chunk]:
        cfg = self._cfg
        abstract_el = root.find(".//tei:profileDesc/tei:abstract", _NS)
        if abstract_el is None:
            return []

        paragraphs = [
            _element_text(p)
            for p in abstract_el.iter(_tag("p"))
            if _element_text(p)
        ]

        if cfg.include_title:
            title_el = root.find(".//tei:titleStmt/tei:title", _NS)
            if title_el is not None:
                title_text = _element_text(title_el)
                if title_text:
                    paragraphs.insert(0, title_text)

        raw_chunks = _paragraphs_to_chunks(paragraphs, cfg.max_words, cfg.min_words)
        return [
            Chunk(section="abstract", chunk=i, content=text)
            for i, text in enumerate(raw_chunks)
        ]

    def _parse_body(self, root: ET.Element) -> list[Chunk]:
        cfg = self._cfg
        body = root.find(".//tei:text/tei:body", _NS)
        if body is None:
            return []

        all_chunks: list[Chunk] = []

        for div in body.findall(_tag("div")):
            section_name, chunks = self._parse_div(div)
            if chunks:
                all_chunks.extend(chunks)

        return all_chunks

    def _parse_div(
        self, div: ET.Element, parent_section: str | None = None
    ) -> tuple[str, list[Chunk]]:
        """Recursively parse a ``<div>`` element and its nested ``<div>`` children."""
        cfg = self._cfg

        # Resolve section heading
        head_el = div.find(_tag("head"))
        raw_heading = _element_text(head_el) if head_el is not None else ""
        if not raw_heading and parent_section:
            raw_heading = parent_section
        elif not raw_heading:
            raw_heading = "body"

        section_label = raw_heading
        normalised = _normalise_section(raw_heading)

        if normalised in cfg.skip_sections:
            return section_label, []

        # Collect paragraphs belonging directly to this div (not nested divs)
        paragraphs: list[str] = []
        for child in div:
            if child.tag == _tag("p"):
                text = _element_text(child)
                if text:
                    paragraphs.append(text)
            # Skip nested divs here — handled recursively below

        raw_chunks = _paragraphs_to_chunks(paragraphs, cfg.max_words, cfg.min_words)
        chunks: list[Chunk] = [
            Chunk(section=section_label, chunk=i, content=text)
            for i, text in enumerate(raw_chunks)
        ]

        # Recurse into nested divs
        for nested in div.findall(_tag("div")):
            _, nested_chunks = self._parse_div(nested, parent_section=section_label)
            chunks.extend(nested_chunks)

        return section_label, chunks


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def parse_tei_to_chunks(
    tei_xml: str,
    config: ParseConfig | None = None,
) -> list[ChunkDict]:
    """Parse GROBID TEI XML and return a JSON-serialisable list of chunk dicts.

    Args:
        tei_xml: Raw TEI XML string from GROBID.
        config:  Optional :class:`ParseConfig` to override defaults.

    Returns:
        A list of dicts with keys ``"section"``, ``"chunk"``, ``"content"``.

    Example::

        from executor.parser import parse_tei_to_chunks

        chunks = parse_tei_to_chunks(result.tei_xml)
        # [{"section": "abstract", "chunk": 0, "content": "…"}, …]
    """
    return [c.to_dict() for c in TEIParser(config).parse(tei_xml)]
