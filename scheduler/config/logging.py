"""
config/logging.py
──────────────────
Configures structlog for JSON (production) or pretty console (dev) output.
Called once from the Celery setup_logging signal in main.py.
"""

from __future__ import annotations

import logging
import sys

import structlog

from config.settings import settings


def configure_logging() -> None:
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        # Production: newline-delimited JSON — easy to ingest into Loki / CloudWatch
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: colourised human-readable output
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "arxiv", "urllib3", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
