# ArXiv RAG Scraping Pipeline

Celery + Redis pipeline that ingests ArXiv papers end-to-end:
**discover → download → parse → chunk → embed → store**.

---

## Architecture

```
                         ┌─────────────────────────────────────────────────┐
                         │              Docker Compose                      │
                         │                                                  │
  ┌──────────┐  every   │  ┌──────────┐  ┌──────────────────────────────┐ │
  │   Beat   │──6 hrs──►│  │  SCRAPE  │  │  Queue Topology              │ │
  │Scheduler │          │  │ workers  │  │                              │ │
  └──────────┘          │  │ (×2)     │  │  scrape ──DLX──► dlx.scrape │ │
                         │  └────┬─────┘  │  process──DLX──►dlx.process │ │
                         │       │ chain  │  embed ──DLX──► dlx.embed   │ │
                         │  ┌────▼─────┐  └──────────────────────────────┘ │
                         │  │ PROCESS  │                                  │
                         │  │ workers  │   Beat also runs every hour:    │
                         │  │ (×2)     │   • retry_dead_letters (×3)     │
                         │  └────┬─────┘   • pipeline_health (×15min)   │
                         │       │ chain   • purge_old_pdfs (×daily)     │
                         │  ┌────▼─────┐                                  │
                         │  │  EMBED   │                                  │
                         │  │ workers  │                                  │
                         │  │ (×1)     │                                  │
                         │  └────┬─────┘                                  │
                         │       │                                        │
                         │  ┌────▼──────────┐  ┌────────┐               │
                         │  │ Vector Store  │  │ Flower │:5555           │
                         │  │ (your DB)     │  │Monitor │               │
                         │  └───────────────┘  └────────┘               │
                         └─────────────────────────────────────────────────┘
```

## Task Chain (7 tasks across 3 queues)

```
[SCRAPE queue]                [PROCESS queue]              [EMBED queue]
      │                              │                           │
scrape_topic                         │                           │
  │  Queries ArXiv API               │                           │
  │  Filters dedup                   │                           │
  └─► scrape_paper_metadata          │                           │
        │  Fetches full metadata     │                           │
        └─► download_pdf             │                           │
              │  HTTP stream PDF     │                           │
              └────────────────► parse_pdf                       │
                                   │  PyMuPDF section extract    │
                                   └─► clean_text                │
                                         │  Strip boilerplate    │
                                         └─► chunk_document      │
                                               │  512-tok chunks  │
                                               └─────────────► generate_embeddings
                                                                 │  Batch embed API
                                                                 └─► store_chunks
                                                                       │  Write vectors
                                                                       └─► mark_as_processed
```

## Quick Start

```bash
# 1. Copy env template
cp .env.example .env
# Edit .env — set OPENAI_API_KEY (or switch to EMBEDDING_PROVIDER=local)

# 2. Start everything
docker compose up -d

# 3. Watch tasks in Flower
open http://localhost:5555/flower

# 4. Manually trigger a scrape
docker compose exec worker-scrape \
    python scripts/trigger_scrape.py --topics cs.CL --max-results 20

# 5. Re-ingest a specific paper
docker compose exec worker-scrape \
    python scripts/trigger_scrape.py --arxiv-id 2401.12345
```

## Scaling

```bash
# Add more scrape workers when queue depth is high
docker compose up -d --scale worker-scrape=4

# Add more process workers for PDF-heavy workloads
docker compose up -d --scale worker-process=4

# Embed workers stay at 1–2 to respect API rate limits
docker compose up -d --scale worker-embed=2
```

## File Layout

```
arxiv-rag/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
│
├── config/
│   ├── settings.py          # Pydantic settings (reads .env)
│   ├── celery_config.py     # All Celery knobs
│   └── logging.py           # structlog JSON / console
│
├── celery_app/
│   ├── main.py              # App factory, queue topology, Beat schedule
│   ├── tasks/
│   │   ├── scrape.py        # Tasks 1–3  (scrape queue)
│   │   ├── process.py       # Tasks 4–5  (process queue)
│   │   ├── embed.py         # Tasks 6–7  (embed queue)
│   │   └── maintenance.py   # DLQ retry, health check, pdf purge
│   └── utils/
│       ├── dedup.py         # Redis-backed paper deduplication
│       ├── embedder.py      # OpenAI / local embedding abstraction
│       ├── paper_schema.py  # Pydantic models for inter-task data
│       └── text_cleaner.py  # Academic PDF text normalisation
│
└── scripts/
    └── trigger_scrape.py    # CLI to manually kick off ingestion
```

## Connecting Your Vector Store

Open `celery_app/tasks/embed.py` and replace the `_write_chunks()` stub
with your writer. Each chunk arriving there is a plain dict with:

| Field | Type | Description |
|---|---|---|
| `chunk_id` | str | `<arxiv_id>_<section_idx>_<chunk_idx>` |
| `arxiv_id` | str | e.g. `2401.12345` |
| `title` | str | Paper title |
| `section` | str | Section heading |
| `text` | str | Chunk body (≈512 tokens) |
| `embedding` | list[float] | Dense vector |
| `embedding_model` | str | e.g. `text-embedding-3-small` |
| `embedding_dim` | int | 1536 (OpenAI) / 384 (local) |
| `page_start` | int | PDF page number |
| `authors` | list[str] | Author names |
| `categories` | list[str] | ArXiv category codes |
| `published` | str | ISO 8601 date |
| `token_count` | int | Approximate token count |