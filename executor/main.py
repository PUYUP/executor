"""Example: fuse GROBID + Docling results for a single PDF.

Run:
    python example_usage.py paper.pdf
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from executor.docling_exec import DoclingExecutor
from executor.grobid_exec import GrobidExecutor
from executor.mapper import DocumentMapper, FusedDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


async def process(pdf_path: Path) -> FusedDocument:
    mapper = DocumentMapper(iou_threshold=0.25, caption_threshold=0.2)

    # Run both executors concurrently — they're independent
    async with DoclingExecutor() as docling:
        async with GrobidExecutor() as grobid:
            await grobid.wait_until_ready()
            grobid_task = asyncio.create_task(grobid.process_fulltext(pdf_path))
            docling_task = asyncio.create_task(docling.process_pdf(pdf_path))
            grobid_result, docling_result = await asyncio.gather(grobid_task, docling_task)

    if not grobid_result.ok:
        raise RuntimeError(f"GROBID failed: {grobid_result.error}")
    if not docling_result.ok:
        raise RuntimeError(f"Docling failed: {docling_result.error}")

    if grobid_result.tei_xml is None:
        raise RuntimeError("GROBID returned OK but no TEI XML")

    fused = mapper.map(grobid_result.tei_xml, docling_result)
    return fused


def _print_report(doc: FusedDocument) -> None:
    g = doc.grobid

    print("=" * 60)
    print(f"TITLE   : {g.title}")
    print(f"AUTHORS : {', '.join(a.full_name for a in g.authors)}")
    print(f"ABSTRACT: {(g.abstract or '')[:120]}…")
    print(f"SECTIONS: {len(g.sections)}")
    print(f"REFS    : {len(g.references)}")
    print()
    print(f"ASSETS  : {len(doc.assets)} total")
    print(f"  figures : {len(doc.figures)}")
    print(f"  tables  : {len(doc.tables)}")
    print(f"  formulas: {len(doc.formulas)}")
    print(f"  matched : {len(doc.matched_assets)}")
    print(f"  unmatched: {len(doc.unmatched_assets)}")
    print()

    for i, fa in enumerate(doc.assets, 1):
        a = fa.asset
        status = f"✓ conf={fa.match_confidence:.2f}" if fa.grobid_ref else "✗ unmatched"
        label = fa.label or "-"
        caption = (fa.caption or "")[:60]
        print(
            f"  [{i}] {a.asset_type:<8} p{a.page_number}  {status:<22}"
            f"  label={label!r:<14} caption={caption!r}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py <path/to/paper.pdf>")
        sys.exit(1)

    pdf = Path(sys.argv[1])
    fused = asyncio.run(process(pdf))
    _print_report(fused)
