import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

import arxiv


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import load_config
from core.schemas import DocumentMetadata


SORT_BY_MAP = {
    "relevance": arxiv.SortCriterion.Relevance,
    "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
    "submitted_date": arxiv.SortCriterion.SubmittedDate,
}

SORT_ORDER_MAP = {
    "ascending": arxiv.SortOrder.Ascending,
    "descending": arxiv.SortOrder.Descending,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download arXiv PDFs and save metadata manifests."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def build_search(config_query: str, max_results: int, sort_by: str, sort_order: str) -> arxiv.Search:
    return arxiv.Search(
        query=config_query,
        max_results=max_results,
        sort_by=SORT_BY_MAP.get(sort_by, arxiv.SortCriterion.SubmittedDate),
        sort_order=SORT_ORDER_MAP.get(sort_order, arxiv.SortOrder.Descending),
    )


def ensure_directories(paths: List[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def author_names(authors: List[Any]) -> List[str]:
    return [author.name for author in authors]


def save_metadata(metadata_dir: Path, document: DocumentMetadata) -> None:
    output_path = metadata_dir / f"{Path(document.source_file).stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    app_config = load_config(args.config)

    raw_dir = app_config.paths.raw_papers
    metadata_dir = app_config.paths.metadata
    ensure_directories([raw_dir, metadata_dir])

    client = arxiv.Client()
    search = build_search(
        config_query=app_config.ingestion.query,
        max_results=app_config.ingestion.max_results,
        sort_by=app_config.ingestion.sort_by,
        sort_order=app_config.ingestion.sort_order,
    )

    for paper in client.results(search):
        downloaded_path = Path(paper.download_pdf(dirpath=str(raw_dir)))
        document = DocumentMetadata(
            document_id=downloaded_path.stem,
            title=paper.title.strip(),
            source_file=downloaded_path.name,
            source_path=str(downloaded_path),
            source_type=app_config.ingestion.source,
            authors=author_names(paper.authors),
            abstract=(paper.summary or "").strip() or None,
            published_at=paper.published.isoformat() if paper.published else None,
            categories=list(paper.categories or []),
            primary_category=paper.primary_category,
            entry_id=paper.entry_id,
            pdf_url=paper.pdf_url,
            doi=paper.doi,
            journal_ref=paper.journal_ref,
            comment=paper.comment,
        )
        save_metadata(metadata_dir, document)
        print(f"Downloaded {downloaded_path.name}")


if __name__ == "__main__":
    main()
