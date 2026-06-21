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
* Coverage: abstract, body (incl. nested divs), author list, and back
  matter (references / bibliography, appendix, acknowledgements, …) are
  all emitted as their own sections unless explicitly skipped.
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


@dataclass(frozen=True, slots=True)
class Chunk:
    """A single, immutable text chunk ready for SPECTER2 encoding."""

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

    include_authors: bool = True
    """Whether to emit an "authors" section listing author names/affiliations."""

    include_references: bool = True
    """Whether to emit a "references" section from the bibliography."""

    include_back_matter: bool = True
    """Whether to walk other <back> divs (appendix, acks, ...) like body divs."""

    normalise_section_names: bool = True
    """Lowercase & strip numeric prefixes from section headings."""

    skip_sections: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                # Left empty by default so "all sections" really means all.
                # Add headings here (normalised, lowercase) to exclude them,
                # e.g. {"acknowledgements", "funding"}.
            }
        )
    )
    """Section headings (normalised) to exclude from the output.

    Note: this does NOT apply to the dedicated "authors" / "references"
    sections, which are gated by their own include_* flags instead.
    """


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
# Author helpers
# ---------------------------------------------------------------------------


def _format_author(author_el: ET.Element) -> str:
    """Render a single TEI <author> element as 'Name (Affiliation)' text."""
    pers_name = author_el.find(_tag("persName"))
    if pers_name is not None:
        name = _element_text(pers_name)
    else:
        name = _element_text(author_el)

    affiliations = []
    for aff in author_el.findall(_tag("affiliation")):
        org_names = [
            _element_text(o) for o in aff.findall(_tag("orgName")) if _element_text(o)
        ]
        if org_names:
            affiliations.append(", ".join(org_names))
        else:
            text = _element_text(aff)
            if text:
                affiliations.append(text)

    email_el = author_el.find(_tag("email"))
    email = _element_text(email_el) if email_el is not None else ""

    parts = [name] if name else []
    if affiliations:
        parts.append(f"({'; '.join(affiliations)})")
    if email:
        parts.append(f"<{email}>")

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------


def _format_bibl_struct(bibl: ET.Element) -> str:
    """Render a <biblStruct> bibliography entry as a single readable string."""
    analytic = bibl.find(_tag("analytic"))
    monogr = bibl.find(_tag("monogr"))

    title_el = None
    if analytic is not None:
        title_el = analytic.find(_tag("title"))
    if title_el is None and monogr is not None:
        title_el = monogr.find(_tag("title"))
    title = _element_text(title_el) if title_el is not None else ""

    authors: list[str] = []
    author_source = analytic if analytic is not None else monogr
    if author_source is not None:
        for author_el in author_source.findall(_tag("author")):
            pers_name = author_el.find(_tag("persName"))
            name = _element_text(pers_name) if pers_name is not None else _element_text(author_el)
            if name:
                authors.append(name)

    date = ""
    if monogr is not None:
        imprint = monogr.find(_tag("imprint"))
        if imprint is not None:
            date_el = imprint.find(_tag("date"))
            if date_el is not None:
                date = date_el.get("when") or _element_text(date_el)

    venue = ""
    if monogr is not None:
        venue_title = monogr.find(_tag("title"))
        if venue_title is not None and venue_title is not title_el:
            venue = _element_text(venue_title)

    pieces = []
    if authors:
        pieces.append(", ".join(authors))
    if title:
        pieces.append(title)
    if venue:
        pieces.append(venue)
    if date:
        pieces.append(f"({date})")

    return ". ".join(p for p in pieces if p).strip()


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
            Ordered list of chunks across all sections (abstract, authors,
            body, references, and remaining back matter).
        """
        root = ET.fromstring(tei_xml)
        chunks: list[Chunk] = []

        if self._cfg.include_authors:
            chunks.extend(self._parse_authors(root))

        if self._cfg.include_title:
            chunks.extend(self._parse_title(root))

        if self._cfg.include_abstract:
            chunks.extend(self._parse_abstract(root))

        chunks.extend(self._parse_body(root))

        if self._cfg.include_references:
            chunks.extend(self._parse_references(root))

        if self._cfg.include_back_matter:
            chunks.extend(self._parse_back_matter(root))

        return chunks

    # ------------------------------------------------------------------
    # Section parsers
    # ------------------------------------------------------------------

    def _parse_authors(self, root: ET.Element) -> list[Chunk]:
        cfg = self._cfg
        author_els = root.findall(
            ".//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author", _NS
        )
        if not author_els:
            # Some document types (theses, books, monographs) carry the
            # author under monogr instead of analytic.
            author_els = root.findall(
                ".//tei:sourceDesc/tei:biblStruct/tei:monogr/tei:author", _NS
            )
        if not author_els:
            return []

        lines = [_format_author(a) for a in author_els]
        lines = [line for line in lines if line]
        if not lines:
            return []

        raw_chunks = _paragraphs_to_chunks(lines, cfg.max_words, min_words=1)
        return [
            Chunk(section="authors", chunk=i, content=text)
            for i, text in enumerate(raw_chunks)
        ]

    def _parse_title(self, root: ET.Element) -> list[Chunk]:
        """Extract the paper title as its own section.

        Deliberately independent of whether an abstract is present —
        GROBID sometimes fails to isolate an abstract, and previously the
        title was only emitted as a side-effect of abstract parsing,
        silently disappearing whenever profileDesc/abstract was absent.
        """
        title_el = root.find(".//tei:titleStmt/tei:title", _NS)
        if title_el is None:
            return []
        title_text = _element_text(title_el)
        if not title_text:
            return []
        return [Chunk(section="title", chunk=0, content=title_text)]

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
        if not paragraphs:
            return []

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

    def _parse_references(self, root: ET.Element) -> list[Chunk]:
        cfg = self._cfg
        list_bibl = root.find(".//tei:text/tei:back//tei:listBibl", _NS)
        if list_bibl is None:
            return []

        entries = [
            _format_bibl_struct(b) for b in list_bibl.findall(_tag("biblStruct"))
        ]
        entries = [e for e in entries if e]
        if not entries:
            return []

        # Each reference is short and self-contained; merge several per
        # chunk (up to max_words) but don't drop short ones via min_words.
        raw_chunks = _paragraphs_to_chunks(entries, cfg.max_words, min_words=1)
        return [
            Chunk(section="references", chunk=i, content=text)
            for i, text in enumerate(raw_chunks)
        ]

    def _parse_back_matter(self, root: ET.Element) -> list[Chunk]:
        """Parse remaining <back> divs (appendix, acknowledgements, etc.),
        skipping the bibliography div already handled by _parse_references.
        """
        cfg = self._cfg
        back = root.find(".//tei:text/tei:back", _NS)
        if back is None:
            return []

        all_chunks: list[Chunk] = []
        for div in back.findall(_tag("div")):
            if div.find(_tag("listBibl")) is not None or div.get("type") == "references":
                continue
            _, chunks = self._parse_div(div)
            if chunks:
                all_chunks.extend(chunks)

        return all_chunks


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
        # [{"section": "authors", "chunk": 0, "content": "…"}, …]
    """
    return [c.to_dict() for c in TEIParser(config).parse(tei_xml)]