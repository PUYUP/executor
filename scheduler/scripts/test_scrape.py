import json
import os
from datetime import datetime
from celery_app.tasks.scrape import scrape_paper_metadata, download_pdf
from celery_app.tasks.process import parse_pdf, clean_text, chunk_document


def save_chunks_to_json(chunks, arxiv_id: str, output_dir: str = "output"):
    """Simpan chunks ke file JSON."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/chunks_{arxiv_id}_{timestamp}.json"

    output = {
        "arxiv_id": arxiv_id,
        "timestamp": timestamp,
        "total_chunks": len(chunks) if isinstance(chunks, list) else None,
        "chunks": chunks,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Chunks saved to: {filename}")
    return filename


def main():
    arxiv_id = "2606.20564"

    result = scrape_paper_metadata(arxiv_id=arxiv_id)
    download_result = download_pdf(result)
    parsing = parse_pdf(download_result)
    cleans = clean_text(parsing)
    chunks = chunk_document(cleans)

    print(chunks)

    # Simpan ke JSON
    saved_path = save_chunks_to_json(chunks, arxiv_id=arxiv_id)
    print(f"File tersimpan di: {saved_path}")


if __name__ == "__main__":
    main()