FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc g++ libgl1 libglib2.0-0 libgomp1 \
        poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*


# ─── FINAL IMAGE ─────────────────────────────
FROM base AS final

WORKDIR /app

# Copy project
COPY . .

# Install dependencies (INI YANG HILANG SEBELUMNYA)
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

# Create user
RUN groupadd -r celery && useradd -r -g celery celery

RUN mkdir -p /app/downloads /app/logs \
    && chown -R celery:celery /app

USER celery

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD celery -A src.main inspect ping -d "celery@$$HOSTNAME" --timeout 5 || exit 1

CMD celery -A src.main worker \
    --loglevel=info \
    --concurrency=4 \
    --queues=default,scrape,process,embed