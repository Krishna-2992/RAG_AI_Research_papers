import argparse
import json
import re
from pathlib import Path

import arxiv
from urllib.request import urlretrieve

SORT_BY_MAP = {
    "relevance": arxiv.SortCriterion.Relevance,
    "last_updated_date": arxiv.SortCriterion.LastUpdatedDate,
    "submitted_date": arxiv.SortCriterion.SubmittedDate,
}

SORT_ORDER_MAP = {
    "ascending": arxiv.SortOrder.Ascending,
    "descending": arxiv.SortOrder.Descending,
}


def sanitize_filename(name: str) -> str:
    """Remove invalid filename characters."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    return name[:180]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        required=True,
        help="arXiv search query",
    )

    parser.add_argument(
        "--output",
        default="papers",
        help="Download directory",
    )

    parser.add_argument(
        "--metadata",
        default="metadata",
        help="Metadata directory",
    )

    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--sort-by",
        default="submitted_date",
        choices=SORT_BY_MAP.keys(),
    )

    parser.add_argument(
        "--sort-order",
        default="descending",
        choices=SORT_ORDER_MAP.keys(),
    )

    return parser.parse_args()


def save_metadata(metadata_dir: Path, paper, pdf_name):
    metadata = {
        "document_id": Path(pdf_name).stem,
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "abstract": paper.summary,
        "published": paper.published.isoformat()
        if paper.published
        else None,
        "updated": paper.updated.isoformat()
        if paper.updated
        else None,
        "categories": paper.categories,
        "primary_category": paper.primary_category,
        "entry_id": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "journal_ref": paper.journal_ref,
        "comment": paper.comment,
    }

    metadata_file = metadata_dir / f"{Path(pdf_name).stem}.json"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    output_dir = Path(args.output)
    metadata_dir = Path(args.metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(
        page_size=100,
        delay_seconds=3,
        num_retries=5,
    )

    search = arxiv.Search(
        query=args.query,
        max_results=args.max_results,
        sort_by=SORT_BY_MAP[args.sort_by],
        sort_order=SORT_ORDER_MAP[args.sort_order],
    )

    papers = list(client.results(search))

    print(f"\nFound {len(papers)} papers\n")

    for i, paper in enumerate(papers, start=1):
        try:
            filename = sanitize_filename(paper.title) + ".pdf"
            pdf_path = output_dir / filename

            if pdf_path.exists():
                print(f"[{i}/{len(papers)}] Already exists: {filename}")
                continue

            urlretrieve(
                paper.pdf_url,
                str(pdf_path),
            )

            save_metadata(metadata_dir, paper, filename)

            print(f"[{i}/{len(papers)}] Downloaded: {filename}")

        except Exception as e:
            print(f"Failed: {paper.title}")
            print(e)

    print("\nDone.")


if __name__ == "__main__":
    main()