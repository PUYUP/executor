#!/usr/bin/env python
"""
scripts/trigger_scrape.py
──────────────────────────
Manually trigger a scrape run for one or more topics without waiting for Beat.

Usage:
  # Scrape default topics from settings
  python scripts/trigger_scrape.py

  # Scrape specific topics
  python scripts/trigger_scrape.py --topics cs.AI cs.CL --max-results 100

  # Re-ingest a specific paper (bypasses dedup)
  python scripts/trigger_scrape.py --arxiv-id 2401.12345
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks.scrape import scrape_topic, scrape_paper_metadata
from utils.dedup import reset_paper
from config.settings import settings


def main():
    parser = argparse.ArgumentParser(description="Trigger ArXiv scrape tasks")
    parser.add_argument(
        "--topics", nargs="*",
        help="ArXiv category codes (default: settings.arxiv_topics)"
    )
    parser.add_argument(
        "--max-results", type=int, default=settings.max_results_per_topic,
        help="Max papers per topic"
    )
    parser.add_argument(
        "--arxiv-id", type=str,
        help="Re-ingest a single paper by ArXiv ID"
    )
    args = parser.parse_args()

    if args.arxiv_id:
        print(f"Resetting and re-queuing paper: {args.arxiv_id}")
        reset_paper(args.arxiv_id)
        result = scrape_paper_metadata.apply_async(args=[args.arxiv_id], queue="scrape")
        print(f"Task ID: {result.id}")
        return

    topics = args.topics or settings.arxiv_topics
    for topic in topics:
        print(f"Triggering scrape for topic: {topic} (max={args.max_results})")
        result = scrape_topic.apply_async(
            args=[topic],
            kwargs={"max_results": args.max_results},
            queue="scrape",
        )
        print(f"  Task ID: {result.id}")


if __name__ == "__main__":
    main()
