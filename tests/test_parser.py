"""Tests for executor.parser."""

from executor.parser import ParseConfig, TEIParser, parse_tei_to_chunks

# ---------------------------------------------------------------------------
# Minimal TEI XML fixture (mirrors real GROBID output structure)
# ---------------------------------------------------------------------------

_TEI = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt>
        <title level="a" type="main">Attention Is All You Need</title>
      </titleStmt>
    </fileDesc>
    <profileDesc>
      <abstract>
        <div xmlns="http://www.tei-c.org/ns/1.0">
          <p>The dominant sequence transduction models are based on complex recurrent or
          convolutional neural networks. We propose a new simple network architecture,
          the Transformer, based solely on attention mechanisms.</p>
        </div>
      </abstract>
    </profileDesc>
  </teiHeader>
  <text>
    <body>
      <div xmlns="http://www.tei-c.org/ns/1.0">
        <head>1 Introduction</head>
        <p>Recurrent neural networks, long short-term memory and gated recurrent neural
        networks in particular, have been firmly established as state of the art approaches
        in sequence modelling and transduction problems.</p>
        <p>Attention mechanisms allow modelling of dependencies without regard to their
        distance in the input or output sequences.</p>
      </div>
      <div xmlns="http://www.tei-c.org/ns/1.0">
        <head>2 Background</head>
        <p>The goal of reducing sequential computation also forms the foundation of the
        Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural
        networks as basic building block.</p>
      </div>
      <div xmlns="http://www.tei-c.org/ns/1.0">
        <head>Acknowledgements</head>
        <p>We thank the anonymous reviewers for their feedback.</p>
      </div>
    </body>
  </text>
</TEI>"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_list_of_dicts() -> None:
    chunks = parse_tei_to_chunks(_TEI)
    assert isinstance(chunks, list)
    assert all(isinstance(c, dict) for c in chunks)


def test_required_keys_present() -> None:
    chunks = parse_tei_to_chunks(_TEI)
    for chunk in chunks:
        assert "section" in chunk
        assert "chunk" in chunk
        assert "content" in chunk


def test_chunk_index_is_sequential_per_section() -> None:
    chunks = parse_tei_to_chunks(_TEI)
    from collections import defaultdict

    seen: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        seen[c["section"]].append(c["chunk"])  # type: ignore[arg-type]
    for section, indices in seen.items():
        assert indices == list(range(len(indices))), (
            f"Non-sequential chunk indices in section '{section}': {indices}"
        )


def test_abstract_section_present() -> None:
    chunks = parse_tei_to_chunks(_TEI)
    sections = {c["section"] for c in chunks}
    assert "abstract" in sections


def test_acknowledgements_skipped_by_default() -> None:
    chunks = parse_tei_to_chunks(_TEI)
    sections = {c["section"] for c in chunks}
    assert "Acknowledgements" not in sections


def test_body_sections_present() -> None:
    chunks = parse_tei_to_chunks(_TEI)
    sections = {c["section"] for c in chunks}
    assert "1 Introduction" in sections
    assert "2 Background" in sections


def test_content_is_non_empty_string() -> None:
    chunks = parse_tei_to_chunks(_TEI)
    for c in chunks:
        assert isinstance(c["content"], str)
        assert len(c["content"].strip()) > 0


def test_max_words_respected() -> None:
    cfg = ParseConfig(max_words=20, min_words=1)
    chunks = parse_tei_to_chunks(_TEI, config=cfg)
    for c in chunks:
        word_count = len(c["content"].split())
        # Allow a small overshoot at sentence boundaries
        assert word_count <= 40, f"Chunk too long ({word_count} words): {c['content'][:80]}"


def test_include_title_in_abstract() -> None:
    cfg = ParseConfig(include_title=True)
    chunks = parse_tei_to_chunks(_TEI, config=cfg)
    abstract_chunks = [c for c in chunks if c["section"] == "abstract"]
    assert any("Attention Is All You Need" in c["content"] for c in abstract_chunks)


def test_exclude_abstract() -> None:
    cfg = ParseConfig(include_abstract=False)
    chunks = parse_tei_to_chunks(_TEI, config=cfg)
    assert all(c["section"] != "abstract" for c in chunks)


def test_tei_parser_class() -> None:
    parser = TEIParser()
    result = parser.parse(_TEI)
    assert result
    assert all(hasattr(c, "section") and hasattr(c, "chunk") and hasattr(c, "content") for c in result)
