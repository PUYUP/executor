"""Production-grade async GROBID executor.

Connects to a running GROBID instance at http://localhost:8070 (configurable)
and exposes all major REST endpoints with:
- Async-first design (httpx.AsyncClient)
- Transparent retries with exponential back-off (tenacity)
- Connection pooling & keep-alive
- Per-request and global timeout control
- Structured logging via the standard library
- Typed API surface (dataclasses + enums)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import IO

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & enums
# ---------------------------------------------------------------------------

GROBID_DEFAULT_URL = "http://localhost:8070"

_RETRYABLE = (
    httpx.ConnectError,
    httpx.TimeoutException,
    httpx.RemoteProtocolError,
)


class GrobidService(StrEnum):
    """GROBID REST endpoint names (path suffix after /api/)."""

    FULLTEXT = "processFulltextDocument"
    HEADER = "processHeaderDocument"
    REFERENCES = "processReferences"
    CITATION_LIST = "processCitationList"
    CITATION_PATENT_ST36 = "processCitationPatentST36"
    CITATION_PATENT_TXT = "processCitationPatentTXT"
    AFFILIATION = "processAffiliations"
    DATE = "processDates"
    NAMES_HEADER = "processNamesHeader"
    NAMES_CITATIONS = "processNamesCitations"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GrobidConfig:
    """Executor configuration with safe, production-ready defaults."""

    base_url: str = GROBID_DEFAULT_URL
    """Base URL of the running GROBID instance."""

    timeout: float = 60.0
    """Per-request timeout in seconds (generous for large PDFs)."""

    connect_timeout: float = 5.0
    """TCP connect timeout in seconds."""

    max_connections: int = 10
    """Max simultaneous HTTP connections in the pool."""

    max_keepalive_connections: int = 5
    """Max idle keep-alive connections."""

    retry_attempts: int = 3
    """How many times to retry on transient failures (0 = no retries)."""

    retry_min_wait: float = 1.0
    """Minimum wait between retries (seconds)."""

    retry_max_wait: float = 10.0
    """Maximum wait between retries (seconds)."""

    generate_ids: bool = True
    """Ask GROBID to generate element IDs in TEI output."""

    consolidate_header: int = 0
    """Header consolidation level (0=off, 1=crossref, 2=crossref+pubmed)."""

    consolidate_citations: int = 0
    """Citation consolidation level (0=off, 1=crossref, 2=crossref+pubmed)."""

    segment_sentences: int = 0
    """Sentence segmentation (0=off, 1=on)."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GrobidResult:
    """Holds the outcome of a single GROBID call."""

    service: GrobidService
    status_code: int
    tei_xml: str | None = None
    error: str | None = None
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """True when GROBID returned HTTP 200 with TEI output."""
        return self.status_code == 200 and self.tei_xml is not None

    def __repr__(self) -> str:
        state = "ok" if self.ok else f"err={self.error or self.status_code}"
        return f"<GrobidResult service={self.service} {state} elapsed={self.elapsed_ms:.0f}ms>"


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class GrobidExecutor:
    """Async GROBID executor.

    Usage (async context manager — preferred)::

        async with GrobidExecutor() as grobid:
            await grobid.wait_until_ready()
            result = await grobid.process_fulltext(Path("paper.pdf"))
            print(result.tei_xml)

    Or manage the lifecycle manually::

        grobid = GrobidExecutor()
        await grobid.start()
        ...
        await grobid.close()
    """

    def __init__(self, config: GrobidConfig | None = None) -> None:
        self._cfg = config or GrobidConfig()
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Create the underlying HTTP client and connection pool."""
        if self._client is not None:
            return
        limits = httpx.Limits(
            max_connections=self._cfg.max_connections,
            max_keepalive_connections=self._cfg.max_keepalive_connections,
        )
        timeout = httpx.Timeout(
            timeout=self._cfg.timeout,
            connect=self._cfg.connect_timeout,
        )
        self._client = httpx.AsyncClient(
            base_url=self._cfg.base_url,
            limits=limits,
            timeout=timeout,
            headers={"Accept": "application/xml"},
        )
        logger.info("GrobidExecutor started → %s", self._cfg.base_url)

    async def close(self) -> None:
        """Drain in-flight requests and close the connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("GrobidExecutor closed")

    async def __aenter__(self) -> "GrobidExecutor":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def is_alive(self) -> bool:
        """Return True if GROBID responds to the liveness probe."""
        client = self._require_client()
        try:
            # /api/isalive returns plain text "true" — must NOT send Accept: application/xml
            resp = await client.get(
                "/api/isalive",
                headers={"Accept": "text/plain"},
            )
            if resp.status_code == 200:
                # Verify the body too — GROBID returns the string "true"
                return resp.text.strip().lower() == "true"
            logger.debug("GROBID liveness probe returned HTTP %d", resp.status_code)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("GROBID liveness probe failed: %s", exc)
            return False

    async def wait_until_ready(
        self,
        *,
        poll_interval: float = 2.0,
        timeout: float = 60.0,
    ) -> None:
        """Block until GROBID is alive or *timeout* seconds elapse.

        Raises:
            TimeoutError: if GROBID is not ready within *timeout* seconds.
        """
        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            if await self.is_alive():
                logger.info("GROBID ready (attempt %d)", attempt)
                return
            remaining = deadline - time.monotonic()
            wait = min(poll_interval, remaining)
            if wait <= 0:
                break
            logger.debug("GROBID not ready yet, retrying in %.1fs…", wait)
            await asyncio.sleep(wait)
        raise TimeoutError(
            f"GROBID at {self._cfg.base_url} did not become ready within {timeout}s"
        )

    # ------------------------------------------------------------------
    # High-level processing helpers
    # ------------------------------------------------------------------

    async def process_fulltext(
        self,
        pdf: Path | IO[bytes] | bytes,
        *,
        consolidate_header: int | None = None,
        consolidate_citations: int | None = None,
        generate_ids: bool | None = None,
        segment_sentences: int | None = None,
    ) -> GrobidResult:
        """Parse a full scientific PDF → TEI XML."""
        return await self._process_pdf(
            GrobidService.FULLTEXT,
            pdf,
            consolidate_header=consolidate_header,
            consolidate_citations=consolidate_citations,
            generate_ids=generate_ids,
            segment_sentences=segment_sentences,
        )

    async def process_header(
        self,
        pdf: Path | IO[bytes] | bytes,
        *,
        consolidate_header: int | None = None,
    ) -> GrobidResult:
        """Extract header metadata only from a PDF."""
        return await self._process_pdf(
            GrobidService.HEADER,
            pdf,
            consolidate_header=consolidate_header,
        )

    async def process_references(
        self,
        pdf: Path | IO[bytes] | bytes,
        *,
        consolidate_citations: int | None = None,
    ) -> GrobidResult:
        """Extract bibliographic references from a PDF."""
        return await self._process_pdf(
            GrobidService.REFERENCES,
            pdf,
            consolidate_citations=consolidate_citations,
        )

    async def process_citation_list(
        self,
        citations: str,
        *,
        consolidate_citations: int | None = None,
    ) -> GrobidResult:
        """Parse a raw citation string / list (text, not PDF).

        Args:
            citations: Raw citation text (one per line recommended).
        """
        client = self._require_client()
        cfg = self._cfg
        data: dict[str, str | int] = {"citations": citations}
        if consolidate_citations is not None:
            data["consolidateCitations"] = consolidate_citations
        else:
            data["consolidateCitations"] = cfg.consolidate_citations

        return await self._call(
            GrobidService.CITATION_LIST,
            client=client,
            data=data,
        )

    async def process_affiliation(self, affiliation: str) -> GrobidResult:
        """Parse a raw affiliation string."""
        client = self._require_client()
        return await self._call(
            GrobidService.AFFILIATION,
            client=client,
            data={"affiliations": affiliation},
        )

    async def process_date(self, date: str) -> GrobidResult:
        """Parse a raw date string."""
        client = self._require_client()
        return await self._call(
            GrobidService.DATE,
            client=client,
            data={"date": date},
        )

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    async def process_fulltext_batch(
        self,
        pdfs: list[Path],
        *,
        concurrency: int = 4,
    ) -> list[GrobidResult]:
        """Process multiple PDFs concurrently.

        Args:
            pdfs: Paths to PDF files.
            concurrency: Max simultaneous GROBID calls.

        Returns:
            Results in the same order as *pdfs*.
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(pdf: Path) -> GrobidResult:
            async with semaphore:
                return await self.process_fulltext(pdf)

        return await asyncio.gather(*(_bounded(p) for p in pdfs))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_pdf(
        self,
        service: GrobidService,
        pdf: Path | IO[bytes] | bytes,
        *,
        consolidate_header: int | None = None,
        consolidate_citations: int | None = None,
        generate_ids: bool | None = None,
        segment_sentences: int | None = None,
    ) -> GrobidResult:
        client = self._require_client()
        cfg = self._cfg

        # Resolve PDF bytes / file object
        if isinstance(pdf, Path):
            pdf_bytes = pdf.read_bytes()
            filename = pdf.name
        elif isinstance(pdf, bytes):
            pdf_bytes = pdf
            filename = "document.pdf"
        else:
            pdf_bytes = pdf.read()
            filename = getattr(pdf, "name", "document.pdf")

        # Build multipart form
        data: dict[str, str | int] = {
            "consolidateHeader": (
                consolidate_header
                if consolidate_header is not None
                else cfg.consolidate_header
            ),
            "consolidateCitations": (
                consolidate_citations
                if consolidate_citations is not None
                else cfg.consolidate_citations
            ),
            "generateIDs": int(
                generate_ids if generate_ids is not None else cfg.generate_ids
            ),
            "segmentSentences": (
                segment_sentences
                if segment_sentences is not None
                else cfg.segment_sentences
            ),
        }
        files = {"input": (filename, pdf_bytes, "application/pdf")}

        return await self._call(service, client=client, data=data, files=files)

    async def _call(
        self,
        service: GrobidService,
        *,
        client: httpx.AsyncClient,
        data: dict[str, str | int],
        files: dict[str, tuple[str, bytes, str]] | None = None,
    ) -> GrobidResult:
        url = f"/api/{service}"
        cfg = self._cfg

        # Wrap in tenacity retry loop
        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type(_RETRYABLE),
                stop=stop_after_attempt(max(1, cfg.retry_attempts)),
                wait=wait_exponential(
                    min=cfg.retry_min_wait,
                    max=cfg.retry_max_wait,
                ),
                before_sleep=before_sleep_log(logger, logging.WARNING),
                reraise=True,
            ):
                with attempt:
                    t0 = time.perf_counter()
                    resp = await client.post(
                        url,
                        data=data,
                        files=files,
                        headers={"Accept": "application/xml"},  # ✅ scoped to XML endpoints only
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000

        except RetryError as exc:
            logger.error("All retry attempts exhausted for %s: %s", service, exc)
            return GrobidResult(
                service=service,
                status_code=0,
                error=str(exc),
                elapsed_ms=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error calling %s: %s", service, exc)
            return GrobidResult(
                service=service,
                status_code=0,
                error=str(exc),
                elapsed_ms=0.0,
            )

        # Handle response
        if resp.status_code == 200:
            tei = resp.text
            logger.debug(
                "%s → 200 OK (%.0f ms, %d bytes)", service, elapsed_ms, len(tei)
            )
            return GrobidResult(
                service=service,
                status_code=200,
                tei_xml=tei,
                elapsed_ms=elapsed_ms,
            )

        if resp.status_code == 204:
            # GROBID returns 204 when it cannot process the document
            logger.warning("%s → 204 No Content (%.0f ms)", service, elapsed_ms)
            return GrobidResult(
                service=service,
                status_code=204,
                error="GROBID could not process the document (204 No Content)",
                elapsed_ms=elapsed_ms,
            )

        # Any other non-200 status
        logger.error(
            "%s → HTTP %d (%.0f ms): %s",
            service,
            resp.status_code,
            elapsed_ms,
            resp.text[:200],
        )
        return GrobidResult(
            service=service,
            status_code=resp.status_code,
            error=resp.text[:500],
            elapsed_ms=elapsed_ms,
        )

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "GrobidExecutor is not started. "
                "Use `async with GrobidExecutor() as g:` or call `await g.start()` first."
            )
        return self._client


# ---------------------------------------------------------------------------
# CLI entry point  (python -m executor.grobid <pdf_path>)
# ---------------------------------------------------------------------------


async def _cli_main(argv: list[str]) -> None:
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not argv:
        print("Usage: python -m executor.grobid <pdf_path> [service]", file=sys.stderr)
        print(f"Services: {', '.join(s.value for s in GrobidService)}", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(argv[0])
    service_name = argv[1] if len(argv) > 1 else GrobidService.FULLTEXT

    try:
        service = GrobidService(service_name)
    except ValueError:
        print(f"Unknown service '{service_name}'.", file=sys.stderr)
        sys.exit(1)

    async with GrobidExecutor() as grobid:
        logger.info("Waiting for GROBID to be ready…")
        await grobid.wait_until_ready(timeout=30.0)

        logger.info("Processing %s with %s…", pdf_path, service)
        if service == GrobidService.FULLTEXT:
            result = await grobid.process_fulltext(pdf_path)
        elif service == GrobidService.HEADER:
            result = await grobid.process_header(pdf_path)
        elif service == GrobidService.REFERENCES:
            result = await grobid.process_references(pdf_path)
        else:
            print(f"Service '{service}' requires non-PDF input; use the API directly.")
            sys.exit(1)

    if result.ok:
        print(result.tei_xml)
    else:
        print(f"Error: {result.error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import sys

    asyncio.run(_cli_main(sys.argv[1:]))
