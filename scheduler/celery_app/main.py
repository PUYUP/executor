"""
main.py
──────────────────
Celery application factory.
Import this module everywhere: `celery -A celery_app.main ...`
"""

from celery import Celery
from celery.signals import setup_logging, worker_ready, task_failure
from kombu import Exchange, Queue

from config.settings import settings
from config.logging import configure_logging


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_celery_app() -> Celery:
    app = Celery("arxiv_rag")
    app.config_from_object("config.celery_config")
    # Explicit imports are more reliable than autodiscover across Docker
    # bind-mount layouts — every task module must be listed here.
    app.conf.include = [
        "celery_app.tasks.scrape",
        "celery_app.tasks.process",
        "celery_app.tasks.embed",
        "celery_app.tasks.maintenance",
    ]
    _configure_queues(app)
    _configure_beat_schedule(app)
    return app


# ─── Queue Topology ───────────────────────────────────────────────────────────
# Three dedicated queues, each with a matching dead-letter queue.
#
#  scrape  → discover papers, download PDFs
#  process → parse PDF, clean text, chunk
#  embed   → generate embeddings, store vectors
#
# Tasks chain:  scrape_topic → scrape_paper_metadata → download_pdf
#                           → parse_pdf → chunk_document
#                           → generate_embeddings → store_chunks

def _configure_queues(app: Celery) -> None:
    default_exchange = Exchange("default", type="direct")
    scrape_exchange  = Exchange("scrape",  type="direct")
    process_exchange = Exchange("process", type="direct")
    embed_exchange   = Exchange("embed",   type="direct")
    dlx_exchange     = Exchange("dlx",     type="direct")   # dead-letter

    app.conf.task_queues = (
        Queue("default", default_exchange, routing_key="default"),
        # ── Tier 1: I/O bound ──
        Queue("scrape",  scrape_exchange,  routing_key="scrape",
              queue_arguments={"x-dead-letter-exchange": "dlx",
                               "x-dead-letter-routing-key": "dlx.scrape",
                               "x-message-ttl": 3_600_000}),         # 1h TTL
        # ── Tier 2: CPU bound ──
        Queue("process", process_exchange, routing_key="process",
              queue_arguments={"x-dead-letter-exchange": "dlx",
                               "x-dead-letter-routing-key": "dlx.process",
                               "x-message-ttl": 7_200_000}),         # 2h TTL
        # ── Tier 3: API rate-limited ──
        Queue("embed",   embed_exchange,   routing_key="embed",
              queue_arguments={"x-dead-letter-exchange": "dlx",
                               "x-dead-letter-routing-key": "dlx.embed",
                               "x-message-ttl": 14_400_000}),        # 4h TTL
        # ── Dead-letter sinks ──
        Queue("dlx.scrape",   dlx_exchange, routing_key="dlx.scrape"),
        Queue("dlx.process",  dlx_exchange, routing_key="dlx.process"),
        Queue("dlx.embed",    dlx_exchange, routing_key="dlx.embed"),
    )

    app.conf.task_default_queue    = "default"
    app.conf.task_default_exchange = "default"
    app.conf.task_default_routing_key = "default"


# ─── Periodic Beat Schedule ───────────────────────────────────────────────────

def _configure_beat_schedule(app: Celery) -> None:
    """
    Periodic ingestion of ArXiv topics defined in settings.
    Each topic triggers a full scrape → process → embed pipeline.
    """
    app.conf.beat_schedule = {
        # ── Main ingestion: every 6 hours per topic ──
        **{
            f"scrape-{topic.replace(' ', '-')}-periodic": {
                "task": "celery_app.tasks.scrape.scrape_topic",
                "schedule": settings.scrape_interval_seconds,
                "args": [topic],
                "kwargs": {"max_results": settings.max_results_per_topic},
                "options": {"queue": "scrape"},
            }
            for topic in settings.arxiv_topics
        },
        # ── Retry dead-letter queue items every hour ──
        "retry-failed-scrape": {
            "task": "celery_app.tasks.maintenance.retry_dead_letters",
            "schedule": 3600,
            "args": ["dlx.scrape"],
            "options": {"queue": "default"},
        },
        "retry-failed-process": {
            "task": "celery_app.tasks.maintenance.retry_dead_letters",
            "schedule": 3600,
            "args": ["dlx.process"],
            "options": {"queue": "default"},
        },
    }

    app.conf.beat_scheduler = "celery.beat.PersistentScheduler"


# ─── Signals ──────────────────────────────────────────────────────────────────

@setup_logging.connect
def on_setup_logging(**kwargs):
    configure_logging()


@worker_ready.connect
def on_worker_ready(sender, **kwargs):
    import structlog
    log = structlog.get_logger()
    log.info("worker_ready", hostname=sender.hostname)


@task_failure.connect
def on_task_failure(task_id, exception, traceback, sender, **kwargs):
    import structlog
    log = structlog.get_logger()
    log.error(
        "task_failed",
        task_id=task_id,
        task_name=sender.name,
        error=str(exception),
    )


# ─── Singleton ────────────────────────────────────────────────────────────────

app = create_celery_app()