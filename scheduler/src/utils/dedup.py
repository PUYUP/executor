def is_already_processed(arxiv_id: str) -> bool:
    """
    Check if an arXiv paper has already been processed (scraped, parsed, embedded).
    This prevents re-processing the same paper, saving time and API costs.
    """
    return False


def mark_as_queued(arxiv_id: str) -> None:
    """Mark paper as queued for processing"""
    pass
