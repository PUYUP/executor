# config/logging.py
import logging
import sys
import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structured logging for Celery + app runtime.
    """

    # ── 1. Standard library logging ─────────────────────────────
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )

    # Optional: reduce noise from Celery internal logs
    logging.getLogger("celery").setLevel(logging.WARNING)
    logging.getLogger("kombu").setLevel(logging.WARNING)

    # ── 2. Structlog configuration ──────────────────────────────
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,

            # final output format (console-friendly)
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.INFO
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
