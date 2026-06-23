"""
tasks/maintenance.py
────────────────────────────────
Housekeeping tasks (queue: default)
─────────────────────────────────────────────────────────────────────────
• retry_dead_letters  – re-queues tasks from dead-letter queues (DLQ)
• purge_old_pdfs      – removes PDFs older than N days from disk
• pipeline_health     – emits a structured health-check event
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, cast

import redis
import structlog

from main import app
from config.settings import settings

log = structlog.get_logger(__name__)

# Redis client (shared across tasks in this module)
_redis: redis.Redis | None = None


def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[assignment]
    return _redis


# ─────────────────────────────────────────────────────────────────────────────
# retry_dead_letters
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.maintenance.retry_dead_letters",
    bind=True,
    max_retries=1,
    queue="default",
    ignore_result=False,
)
def retry_dead_letters(self, dlq_name: str, max_requeue: int = 50) -> Dict[str, Any]:
    """
    Drains up to `max_requeue` messages from a dead-letter queue
    and re-publishes them to their original destination queue.

    DLQ naming convention:  dlx.<original_queue>
    Re-queue destination:   <original_queue>
    """
    original_queue = dlq_name.replace("dlx.", "", 1)
    r = get_redis()

    log.info(
        "retry_dead_letters.start",
        dlq=dlq_name,
        destination=original_queue,
    )

    requeued = 0
    errors   = 0

    for _ in range(max_requeue):
        # RPOPLPUSH: atomic move from DLQ head → destination tail
        msg = r.rpoplpush(dlq_name, original_queue)
        if msg is None:
            break   # DLQ is empty
        try:
            requeued += 1
            log.debug(
                "retry_dead_letters.requeued",
                dlq=dlq_name,
                destination=original_queue,
            )
        except Exception as exc:
            errors += 1
            log.error(
                "retry_dead_letters.error",
                dlq=dlq_name,
                error=str(exc),
            )

    log.info(
        "retry_dead_letters.done",
        dlq=dlq_name,
        requeued=requeued,
        errors=errors,
    )
    return {"dlq": dlq_name, "requeued": requeued, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# purge_old_pdfs
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.maintenance.purge_old_pdfs",
    bind=True,
    queue="default",
    ignore_result=False,
)
def purge_old_pdfs(self, max_age_days: int = 7) -> Dict[str, Any]:
    """
    Removes PDF files from the download directory that are older than
    `max_age_days` days.  Useful safety net in case store_chunks failed
    to clean up after itself.
    """
    download_dir = Path(settings.pdf_download_dir)
    if not download_dir.exists():
        return {"deleted": 0, "freed_mb": 0.0}

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    deleted = 0
    freed_bytes = 0

    for pdf in download_dir.glob("*.pdf"):
        mtime = datetime.fromtimestamp(pdf.stat().st_mtime, tz=timezone.utc)
        if mtime < cutoff:
            size = pdf.stat().st_size
            pdf.unlink()
            freed_bytes += size
            deleted += 1
            log.debug("purge_old_pdfs.deleted", file=pdf.name, age_days=max_age_days)

    freed_mb = round(freed_bytes / (1024 * 1024), 2)
    log.info("purge_old_pdfs.done", deleted=deleted, freed_mb=freed_mb)
    return {"deleted": deleted, "freed_mb": freed_mb}


# ─────────────────────────────────────────────────────────────────────────────
# pipeline_health
# ─────────────────────────────────────────────────────────────────────────────

@app.task(
    name="celery_app.tasks.maintenance.pipeline_health",
    bind=True,
    queue="default",
    ignore_result=False,
)
def pipeline_health(self) -> Dict[str, Any]:
    """
    Emits a structured health snapshot:
      - Redis connectivity + queue depths
      - Dedup store size
      - PDF disk usage
    Scheduled every 15 minutes by Beat for monitoring dashboards.
    """
    r = get_redis()
    report: Dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "redis": "ok",
        "queues": {},
        "dedup_keys": 0,
        "pdf_dir_mb": 0.0,
    }

    # Queue depths
    queues = ["scrape", "process", "embed", "dlx.scrape", "dlx.process", "dlx.embed"]
    for q in queues:
        try:
            report["queues"][q] = cast(int, r.llen(q))
        except Exception:
            report["queues"][q] = -1

    # Dedup key count
    try:
        report["dedup_keys"] = cast(int, r.scard("arxiv:processed")) + cast(int, r.scard("arxiv:queued"))
    except Exception:
        pass

    # PDF disk usage
    download_dir = Path(settings.pdf_download_dir)
    if download_dir.exists():
        total = sum(f.stat().st_size for f in download_dir.glob("*.pdf"))
        report["pdf_dir_mb"] = round(total / (1024 * 1024), 2)
        report["pdf_count"]  = sum(1 for _ in download_dir.glob("*.pdf"))

    log.info("pipeline_health", **report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Add maintenance tasks to Beat schedule (called from main.py)
# ─────────────────────────────────────────────────────────────────────────────
# These are registered here so they can be imported in main._configure_beat_schedule
MAINTENANCE_BEAT_SCHEDULE = {
    "purge-old-pdfs-daily": {
        "task": "celery_app.tasks.maintenance.purge_old_pdfs",
        "schedule": 86_400,       # once per day
        "kwargs": {"max_age_days": 7},
        "options": {"queue": "default"},
    },
    "pipeline-health-check": {
        "task": "celery_app.tasks.maintenance.pipeline_health",
        "schedule": 900,          # every 15 minutes
        "options": {"queue": "default"},
    },
}