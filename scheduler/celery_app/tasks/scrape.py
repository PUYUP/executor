"""
celery_app/tasks/scrape.py
───────────────────────────
Tier-1 tasks (queue: scrape)
─────────────────────────────────────────────────────────────────────────
Flow:
  scrape_topic(topic)
      └─► [group] scrape_paper_metadata(arxiv_id) × N
                      └─► download_pdf(arxiv_id, pdf_url)
                              └─► parse_pdf (chain → process queue)
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from dataclasses import asdict

import arxiv
import httpx
import structlog
from celery import group, signature
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from celery_app.main import app
from celery_app.utils.dedup import is_already_processed, mark_as_queued
from celery_app.utils.paper_schema import PaperMetadata
from config.settings import settings

log = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 of 5 — scrape_topic
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.scrape.scrape_topic",
    bind=True,
    max_retries=3,
    default_retry_delay=300,
    queue="scrape",
    ignore_result=False,
)
def scrape_topic(
    self,
    topic: str,
    max_results: int = 50,
    sort_by: str = "submittedDate",
) -> Dict[str, Any]:
    """
    Entry point for a topic ingestion run.

    Queries ArXiv API for `topic`, filters already-processed papers,
    then fans out a scrape_paper_metadata task per new paper.
    Returns a summary dict.
    """
    log.info("scrape_topic.start", topic=topic, max_results=max_results)

    try:
        results = _query_arxiv(topic, max_results, sort_by)
    except Exception as exc:
        log.error("scrape_topic.query_failed", topic=topic, error=str(exc))
        raise self.retry(exc=exc)

    new_ids: List[str] = []
    skipped = 0

    for result in results:
        arxiv_id = result.entry_id.split("/")[-1]
        if is_already_processed(arxiv_id):
            skipped += 1
            continue
        mark_as_queued(arxiv_id)
        new_ids.append(arxiv_id)

    log.info(
        "scrape_topic.dispatching",
        topic=topic,
        new=len(new_ids),
        skipped=skipped,
    )

    if new_ids:
        # Fan-out: one scrape_paper_metadata task per new arxiv_id
        job = group(
            scrape_paper_metadata.s(arxiv_id).set(queue="scrape")
            for arxiv_id in new_ids
        )
        job.apply_async()

    return {"topic": topic, "new": len(new_ids), "skipped": skipped}


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 of 5 — scrape_paper_metadata
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.scrape.scrape_paper_metadata",
    bind=True,
    max_retries=5,
    default_retry_delay=60,
    queue="scrape",
    ignore_result=False,
)
def scrape_paper_metadata(self, arxiv_id: str) -> Dict[str, Any]:
    """
    Fetches full metadata for a single paper then triggers PDF download.

    Returns serialised PaperMetadata dict (passed downstream via chain).
    """
    log.info("scrape_paper_metadata.start", arxiv_id=arxiv_id)

    try:
        paper = _fetch_single_paper(arxiv_id)
    except Exception as exc:
        log.warning("scrape_paper_metadata.fetch_failed", arxiv_id=arxiv_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))

    metadata = PaperMetadata(
        arxiv_id=arxiv_id,
        title=paper.title.strip().replace("\n", " "),
        abstract=paper.summary.strip().replace("\n", " "),
        authors=[a.name for a in paper.authors],
        categories=paper.categories,
        published=paper.published.isoformat(),
        updated=paper.updated.isoformat(),
        pdf_url=paper.pdf_url,
        doi=paper.doi or "",
        journal_ref=paper.journal_ref or "",
        # primary_category=paper.primary_category.term if paper.primary_category else "",
    )

    log.info(
        "scrape_paper_metadata.done",
        arxiv_id=arxiv_id,
        title=metadata.title[:60],
    )

    metadata_dict = metadata.model_dump(exclude_none=True)

    # Chain: download_pdf → parse_pdf (process queue)
    (
        download_pdf.s(metadata_dict).set(queue="scrape")
        | signature(
            "celery_app.tasks.process.parse_pdf",
            queue="process",
            immutable=False,
        )
    ).apply_async()

    return metadata_dict


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 of 5 — download_pdf
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.scrape.download_pdf",
    bind=True,
    max_retries=5,
    default_retry_delay=120,
    queue="scrape",
    time_limit=settings.download_timeout_seconds + 30,
    soft_time_limit=settings.download_timeout_seconds,
    ignore_result=False,
)
def download_pdf(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Downloads the PDF for a paper to the local filesystem.

    Accepts the metadata dict from scrape_paper_metadata.
    Returns the same dict enriched with `local_pdf_path`.
    """
    arxiv_id = metadata["arxiv_id"]
    pdf_url  = metadata["pdf_url"]

    log.info("download_pdf.start", arxiv_id=arxiv_id, url=pdf_url)

    dest_path = Path(settings.pdf_download_dir) / f"{arxiv_id}.pdf"

    if dest_path.exists():
        log.info("download_pdf.cache_hit", arxiv_id=arxiv_id)
        metadata["local_pdf_path"] = str(dest_path)
        return metadata

    try:
        _download_file(pdf_url, dest_path)
    except Exception as exc:
        log.error("download_pdf.failed", arxiv_id=arxiv_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60 * 2 ** self.request.retries)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.pdf_max_size_mb:
        dest_path.unlink(missing_ok=True)
        log.warning(
            "download_pdf.too_large",
            arxiv_id=arxiv_id,
            size_mb=round(size_mb, 1),
        )
        # Don't retry — just skip this paper
        metadata["local_pdf_path"] = None
        metadata["skip_reason"] = f"PDF too large ({size_mb:.1f} MB)"
        return metadata

    log.info(
        "download_pdf.done",
        arxiv_id=arxiv_id,
        size_mb=round(size_mb, 2),
    )
    metadata["local_pdf_path"] = str(dest_path)
    return metadata


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _query_arxiv(topic: str, max_results: int, sort_by: str):
    sort_criterion = {
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
        "relevance":     arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
    }.get(sort_by, arxiv.SortCriterion.SubmittedDate)

    client = arxiv.Client(
        page_size=min(max_results, 100),
        delay_seconds=3,          # respect ArXiv rate limit
        num_retries=5,
    )
    search = arxiv.Search(
        query=f"cat:{topic}",
        max_results=max_results,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending,
    )
    return list(client.results(search))


def _fetch_single_paper(arxiv_id: str):
    client = arxiv.Client(num_retries=3, delay_seconds=1)
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))
    if not results:
        raise ValueError(f"No paper found for id={arxiv_id}")
    return results[0]


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream(
        "GET",
        url,
        timeout=settings.download_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": "arxiv-rag-scraper/1.0 (research purposes)"},
    ) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=8192):
                f.write(chunk)
