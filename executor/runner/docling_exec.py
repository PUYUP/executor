"""Docling executor for visual asset extraction.

Extracts high-fidelity images of figures, tables, and formulas from PDF
documents, complementing the GROBID text pipeline.

Design goals
------------
* Visual focus — Configured specifically to extract images (pictures, tables)
  and their associated captions/metadata.
* Async-safe — Docling relies on heavily CPU-bound pipelines. Execution is
  offloaded to a thread pool to avoid blocking the event loop.
* Typed outputs — Yields clean dataclasses containing the PIL Image and
  contextual metadata.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

from PIL import Image

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, FormatOption, PdfFormatOption

logger = logging.getLogger(__name__)

__all__ = [
    "AssetType",
    "DoclingConfig",
    "DoclingExecutor",
    "DoclingResult",
    "ExtractedAsset",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DoclingConfig:
    """Tunable parameters for the Docling executor."""

    extract_pictures: bool = True
    """Extract images/figures from the PDF."""

    extract_tables: bool = True
    """Extract table images from the PDF."""

    images_scale: float = 2.0
    """Resolution multiplier for extracted images (higher = better quality)."""

    thread_workers: int = 1
    """Thread-pool size for CPU-bound PDF processing.

    Note: Docling itself is not thread-safe across multiple concurrent
    conversions with the same converter instance, so >1 is only useful
    if you construct multiple DoclingExecutor instances in parallel.
    The default of 1 avoids contention while still freeing the event loop.
    """


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class AssetType(StrEnum):
    """Types of visual assets extracted by Docling."""

    PICTURE = "picture"
    TABLE = "table"
    FORMULA = "formula"
    OTHER = "other"


@dataclass
class ExtractedAsset:
    """A visual asset extracted from a document."""

    asset_type: AssetType
    """Type of the asset (picture, table, formula)."""

    image: Image.Image | None
    """The cropped PIL Image of the asset (if image extraction was enabled)."""

    caption: str | None = None
    """Caption or textual description associated with the asset."""

    text: str | None = None
    """For tables and formulas, the extracted text/markdown representation."""

    page_number: int | None = None
    """1-based page number where the asset was found."""

    bbox: tuple[float, float, float, float] | None = None
    """Bounding box (x0, y0, x1, y1) in the page's coordinate space."""

    @property
    def has_image(self) -> bool:
        return self.image is not None


@dataclass
class DoclingResult:
    """Holds the outcome of a single Docling processing run."""

    document: DoclingDocument | None = None
    """The full parsed DoclingDocument object."""

    assets: list[ExtractedAsset] = field(default_factory=list)
    """List of all extracted visual assets."""

    error: str | None = None
    """Error message if processing failed, None on success."""

    elapsed_ms: float = 0.0
    """Wall-clock time for the conversion in milliseconds."""

    @property
    def ok(self) -> bool:
        """True when conversion succeeded and a document is available."""
        return self.document is not None and self.error is None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class DoclingExecutor:
    """Async Docling executor for extracting visual assets.

    Usage::

        async with DoclingExecutor() as docling:
            result = await docling.process_pdf(Path("paper.pdf"))
            if result.ok:
                for asset in result.assets:
                    if asset.has_image:
                        asset.image.save(f"{asset.asset_type}_{asset.page_number}.png")

    The executor is *not* safe for concurrent calls to :meth:`process_pdf`
    from multiple coroutines because Docling's converter is not thread-safe.
    Process one PDF at a time, or use separate executor instances.
    """

    def __init__(self, config: DoclingConfig | None = None) -> None:
        self._cfg = config or DoclingConfig()
        self._converter: DocumentConverter | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Initialise the Docling converter and thread pool.

        Idempotent — safe to call multiple times; subsequent calls are no-ops.
        """
        if self._started:
            return

        logger.info("Initialising Docling DocumentConverter…")

        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self._cfg.images_scale

        # Attribute names changed across Docling releases; guard each one.
        if hasattr(pipeline_options, "generate_picture_images"):
            pipeline_options.generate_picture_images = self._cfg.extract_pictures
        if hasattr(pipeline_options, "generate_table_images"):
            pipeline_options.generate_table_images = self._cfg.extract_tables

        format_options: dict[InputFormat, FormatOption] = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }

        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options=format_options,
        )

        self._executor = ThreadPoolExecutor(
            max_workers=self._cfg.thread_workers,
            thread_name_prefix="docling",
        )

        self._started = True
        logger.info("Docling ready (scale=%.1f, pictures=%s, tables=%s)",
                    self._cfg.images_scale,
                    self._cfg.extract_pictures,
                    self._cfg.extract_tables)

    async def close(self) -> None:
        """Shut down the thread pool and release the converter."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._converter = None
        self._started = False
        logger.info("DoclingExecutor closed")

    async def __aenter__(self) -> "DoclingExecutor":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_pdf(self, pdf: Path | IO[bytes]) -> DoclingResult:
        """Parse a PDF and extract visual assets (images, tables, formulas).

        Parameters
        ----------
        pdf:
            Either a :class:`pathlib.Path` to a PDF file on disk, or an
            **open binary stream** (``IO[bytes]``).  Streams are read into
            memory and written to a temporary file so that Docling — which
            requires a file path — can process them.

        Returns
        -------
        DoclingResult
            Always returns a result object; check :attr:`DoclingResult.ok`
            before using :attr:`DoclingResult.document` or
            :attr:`DoclingResult.assets`.
        """
        self._require_ready()
        loop = asyncio.get_running_loop()

        # Narrow to Path | bytes before entering the thread pool so that the
        # type checker (and run_in_executor's signature) sees only those two
        # branches — IO[bytes] is consumed here, in the coroutine, avoiding
        # any I/O inside the thread pool.
        pdf_arg: Path | bytes = pdf if isinstance(pdf, Path) else pdf.read()

        t0 = time.perf_counter()
        try:
            result_doc: DoclingDocument = await loop.run_in_executor(
                self._executor,
                self._process_sync,
                pdf_arg,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assets = self._extract_assets(result_doc)
            logger.info(
                "Docling processed PDF in %.0f ms — %d asset(s) found.",
                elapsed_ms,
                len(assets),
            )
            return DoclingResult(
                document=result_doc,
                assets=assets,
                elapsed_ms=elapsed_ms,
            )

        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error("Docling processing failed after %.0f ms: %s", elapsed_ms, exc, exc_info=True)
            return DoclingResult(
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_sync(self, pdf: Path | bytes) -> DoclingDocument:
        """Blocking Docling conversion.  Runs inside the thread pool."""
        assert self._converter is not None  # guaranteed by _require_ready

        if isinstance(pdf, Path):
            conv_res = self._converter.convert(str(pdf))
        else:
            # Docling requires a file path, so write bytes to a NamedTemporaryFile.
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf)
                tmp_path = Path(tmp.name)
            try:
                conv_res = self._converter.convert(str(tmp_path))
            finally:
                tmp_path.unlink(missing_ok=True)

        return conv_res.document

    def _extract_assets(self, doc: DoclingDocument) -> list[ExtractedAsset]:
        """Convert Docling document elements into :class:`ExtractedAsset` objects."""
        assets: list[ExtractedAsset] = []

        for element in doc.iterate_items():
            # iterate_items() may yield (item, level) tuples or bare items
            # depending on the Docling version — unpack defensively.
            item: Any
            if isinstance(element, tuple):
                item = element[0]
            else:
                item = element

            item_class = item.__class__.__name__

            if item_class == "PictureItem":
                assets.append(self._build_asset(item, doc, AssetType.PICTURE))

            elif item_class == "TableItem":
                asset = self._build_asset(item, doc, AssetType.TABLE)
                # export_to_markdown() requires the parent document in Docling 2.x.
                if hasattr(item, "export_to_markdown"):
                    try:
                        asset.text = item.export_to_markdown(doc)
                    except TypeError:
                        # Older builds accept no arguments.
                        try:
                            asset.text = item.export_to_markdown()
                        except Exception:
                            pass
                    except Exception:
                        pass
                assets.append(asset)

            elif item_class == "EquationItem":
                asset = self._build_asset(item, doc, AssetType.FORMULA)
                # Try multiple known attribute / method names across versions.
                for attr in ("orig", "text", "latex"):
                    if hasattr(item, attr):
                        value = getattr(item, attr)
                        if value:
                            asset.text = str(value)
                            break
                if asset.text is None and hasattr(item, "export_to_document_tokens"):
                    try:
                        asset.text = " ".join(item.export_to_document_tokens(doc))
                    except Exception:
                        pass
                assets.append(asset)

        return assets

    def _build_asset(
        self,
        item: Any,
        doc: DoclingDocument,
        asset_type: AssetType,
    ) -> ExtractedAsset:
        """Construct an :class:`ExtractedAsset` from a Docling document element."""

        # --- Image -------------------------------------------------------
        image: Image.Image | None = None
        if hasattr(item, "get_image"):
            try:
                # Docling 2.x: get_image(doc) requires the document context.
                image = item.get_image(doc)
            except TypeError:
                # Older API accepted no arguments.
                try:
                    image = item.get_image()
                except Exception:
                    pass
            except Exception:
                pass

        # --- Caption -----------------------------------------------------
        caption: str | None = None
        # Prefer the high-level helper introduced in Docling 2.x.
        # caption_text() has no stub, so its return type is `object`; we
        # convert it to str ourselves rather than letting the checker infer
        # an unsafe assignment.
        if hasattr(item, "caption_text") and callable(item.caption_text):
            try:
                raw = item.caption_text(doc)
                caption = str(raw) if raw else None
            except Exception:
                pass
        # Fallback: iterate raw caption references.
        if caption is None and hasattr(item, "captions") and item.captions:
            parts: list[str] = []
            for ref in item.captions:
                # captions may be RefItem objects; resolve them via the doc.
                resolved: object = None
                if hasattr(ref, "resolve") and callable(ref.resolve):
                    try:
                        resolved = ref.resolve(doc)
                    except Exception:
                        pass
                if resolved is not None and hasattr(resolved, "text"):
                    parts.append(str(getattr(resolved, "text")))
                elif hasattr(ref, "text") and ref.text:
                    parts.append(str(ref.text))
            caption = " ".join(parts) or None

        # --- Provenance (page + bbox) ------------------------------------
        page_number: int | None = None
        bbox: tuple[float, float, float, float] | None = None

        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0]
            if hasattr(prov, "page_no"):
                page_number = int(prov.page_no)
            if hasattr(prov, "bbox"):
                b = prov.bbox
                # Docling uses l/t/r/b (left, top, right, bottom).
                if all(hasattr(b, attr) for attr in ("l", "t", "r", "b")):
                    bbox = (float(b.l), float(b.t), float(b.r), float(b.b))
                elif hasattr(b, "as_tuple") and callable(b.as_tuple):
                    try:
                        bbox = tuple(float(v) for v in b.as_tuple())  # type: ignore[assignment]
                    except Exception:
                        pass

        return ExtractedAsset(
            asset_type=asset_type,
            image=image,
            caption=caption,
            page_number=page_number,
            bbox=bbox,
        )

    def _require_ready(self) -> None:
        if not self._started or self._converter is None:
            raise RuntimeError(
                "DoclingExecutor is not started. "
                "Use `async with DoclingExecutor() as d:` or call `await d.start()` first."
            )
