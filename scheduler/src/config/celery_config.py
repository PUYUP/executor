"""
config/celery_config.py
────────────────────────
All Celery settings in one place.
Loaded via: app.config_from_object("config.celery_config")
"""

from config.settings import settings

# ─── Broker & Backend ─────────────────────────────────────────────────────────
broker_url                  = settings.redis_url
result_backend              = settings.redis_url
broker_connection_retry_on_startup = True

# ─── Serialization ────────────────────────────────────────────────────────────
task_serializer             = "json"
result_serializer           = "json"
accept_content              = ["json"]
timezone                    = "UTC"
enable_utc                  = True

# ─── Result Storage ───────────────────────────────────────────────────────────
result_expires              = 86_400          # 24 h
result_compression          = "gzip"
result_extended             = True            # store task name, args, kwargs

# ─── Reliability ──────────────────────────────────────────────────────────────
task_acks_late              = True            # ack AFTER task completes
task_reject_on_worker_lost  = True            # re-queue if worker dies mid-task
task_track_started          = True
worker_prefetch_multiplier  = 1               # one task at a time per worker slot
                                              # (overridden per worker CLI flag)

# ─── Retry Defaults (each task can override) ──────────────────────────────────
task_annotations = {
    "tasks.scrape.*": {
        "max_retries": 5,
        "default_retry_delay": 60,
    },
    "tasks.process.*": {
        "max_retries": 3,
        "default_retry_delay": 120,
    },
    "tasks.embed.*": {
        "max_retries": 10,
        "default_retry_delay": 30,
        "rate_limit": "20/m",        # stay under embedding API rate limit
    },
}

# ─── Task Routing ─────────────────────────────────────────────────────────────
task_routes = {
    # ── Scrape tier ──
    "tasks.scrape.scrape_topic":          {"queue": "scrape"},
    "tasks.scrape.scrape_paper_metadata": {"queue": "scrape"},
    "tasks.scrape.download_pdf":          {"queue": "scrape"},

    # ── Process tier ──
    "tasks.process.parse_pdf":            {"queue": "process"},
    "tasks.process.clean_text":           {"queue": "process"},
    "tasks.process.chunk_document":       {"queue": "process"},

    # ── Embed tier ──
    "tasks.embed.generate_embeddings":    {"queue": "embed"},
    "tasks.embed.store_chunks":           {"queue": "embed"},

    # ── Maintenance ──
    "tasks.maintenance.*":                {"queue": "default"},
}

# ─── Worker ───────────────────────────────────────────────────────────────────
worker_send_task_events     = True            # enables Flower real-time updates
task_send_sent_event        = True
worker_hijack_root_logger   = False           # use our structlog config

# ─── Chord / Group behaviour ──────────────────────────────────────────────────
# When a chord header fails, don't run the callback
chord_propagates_errors     = True

# ─── Broker Transport Options ─────────────────────────────────────────────────
broker_transport_options = {
    "visibility_timeout": 7200,      # 2 h — must be ≥ longest task
    "retry_policy": {
        "max_retries": 3,
    },
}