"""Example: fuse GROBID + Docling results for a single PDF.

Run:
    python example_usage.py paper.pdf
"""

from __future__ import annotations

from executor.runner.parser import TEIParser
from executor.runner.parser import Chunk

import asyncio
import logging
import sys
from pathlib import Path

from executor.runner.grobid_exec import GrobidExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def process(pdf_path: Path) -> list[Chunk]:
    async with GrobidExecutor() as grobid:
        await grobid.wait_until_ready()
        grobid_result = await grobid.process_fulltext(pdf_path)

    if not grobid_result.ok:
        raise RuntimeError(f"GROBID failed: {grobid_result.error}")

    if grobid_result.tei_xml is None:
        raise RuntimeError("GROBID returned OK but no TEI XML")

    parser = TEIParser()
    chunks = parser.parse(grobid_result.tei_xml)
    return chunks


def _print_report(chunks: list[Chunk]) -> None:
    print("=" * 60)
    for chunk in chunks:
        print(f"Section: {chunk.section}")
        print(f"Chunk: {chunk.chunk}")
        print(f"Content: {chunk.content}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path/to/paper.pdf>")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    fused = asyncio.run(process(pdf))
    _print_report(fused)
