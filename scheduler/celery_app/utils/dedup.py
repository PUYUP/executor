"""
celery_app/utils/dedup.py
──────────────────────────
Redis-backed deduplication for ArXiv paper IDs.

Two Redis sets are used:
  arxiv:queued    – paper is in flight (downloaded / being processed)
  arxiv:processed – paper has been fully stored in the vector DB

A paper is skipped on the next Beat run if it exists in either set.
TTL on "queued" prevents stuck papers from blocking re-ingestion forever.
"""

from __future__ import annotations

from typing import cast

import redis
import structlog

from config.settings import settings

log = structlog.get_logger(__name__)

_QUEUED_TTL_SECONDS = 172_800   # 48 h


def _get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)  # type: ignore[return-value]


def is_already_processed(arxiv_id: str) -> bool:
    """Return True if the paper is queued OR fully processed."""
    r = _get_redis()
    return (
        cast(bool, r.sismember("arxiv:processed", arxiv_id))
        or cast(bool, r.sismember("arxiv:queued", arxiv_id))
    )


def mark_as_queued(arxiv_id: str) -> None:
    """Mark paper as in-flight.  Expires after 48 h if processing stalls."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.sadd("arxiv:queued", arxiv_id)
    pipe.setex(f"arxiv:queued:{arxiv_id}", _QUEUED_TTL_SECONDS, "1")
    pipe.execute()
    log.debug("dedup.queued", arxiv_id=arxiv_id)


def mark_as_processed(arxiv_id: str) -> None:
    """
    Promote paper from queued → processed.
    Called by store_chunks after successful write.
    """
    r = _get_redis()
    pipe = r.pipeline()
    pipe.srem("arxiv:queued", arxiv_id)
    pipe.delete(f"arxiv:queued:{arxiv_id}")
    pipe.sadd("arxiv:processed", arxiv_id)
    pipe.execute()
    log.debug("dedup.processed", arxiv_id=arxiv_id)


def reset_paper(arxiv_id: str) -> None:
    """Force re-ingestion of a specific paper (removes from both sets)."""
    r = _get_redis()
    pipe = r.pipeline()
    pipe.srem("arxiv:processed", arxiv_id)
    pipe.srem("arxiv:queued", arxiv_id)
    pipe.delete(f"arxiv:queued:{arxiv_id}")
    pipe.execute()
    log.info("dedup.reset", arxiv_id=arxiv_id)


def count_processed() -> int:
    return cast(int, _get_redis().scard("arxiv:processed"))


def count_queued() -> int:
    return cast(int, _get_redis().scard("arxiv:queued"))
