import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_config
from schemas import EmbeddedDocument, EmbeddingInfo, EmbeddingRecord, ChunkedDocument


def load_environment(env_file: Path) -> None:
    if env_file.exists():
        load_dotenv(env_file)


def build_output_path(input_path: Path, output_dir: Path) -> Path:
    return output_dir / input_path.name


def batch_items(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def read_chunk_file(input_path: Path) -> Dict[str, Any]:
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embedding_file(output_path: Path, payload: Dict[str, Any]) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def create_embeddings_for_file(
    client: OpenAI,
    input_path: Path,
    output_dir: Path,
    model: str,
    batch_size: int,
    provider: str,
) -> None:
    document = ChunkedDocument.model_validate(read_chunk_file(input_path))
    chunks = document.chunks
    if not chunks:
        print(f"Skipping {input_path.name}: no chunks found.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(input_path, output_dir)

    embeddings_payload = EmbeddedDocument(
        document=document.document,
        embedding=EmbeddingInfo(
            provider=provider,
            model=model,
            batch_size=batch_size,
        ),
        chunk_count=len(chunks),
        embeddings=[],
    )

    for batch in batch_items(chunks, batch_size):
        texts = [chunk.text for chunk in batch]
        response = client.embeddings.create(model=model, input=texts)
        for chunk, embedding_item in zip(batch, response.data):
            embeddings_payload.embeddings.append(
                EmbeddingRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    section_path=chunk.section_path,
                    token_count=chunk.token_count,
                    text=chunk.text,
                    vector=embedding_item.embedding,
                )
            )

    save_embedding_file(output_path, embeddings_payload.model_dump(mode="json"))
    print(f"Saved embeddings for {input_path.name} → {output_path}")


def list_input_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.json") if p.is_file()])


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path, default=None)
    partial_args, _ = bootstrap.parse_known_args()
    app_config = load_config(partial_args.config)
    parser = argparse.ArgumentParser(
        description="Create OpenAI embeddings from chunked paper JSON files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=partial_args.config,
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=app_config.paths.chunked_papers,
        help="Directory containing chunked paper JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=app_config.paths.embeddings,
        help="Directory where embedding JSON files will be written.",
    )
    parser.add_argument(
        "--model",
        default=app_config.embedding.model,
        help="OpenAI embedding model to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=app_config.embedding.batch_size,
        help="Number of chunks to send per embedding request.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=app_config.embedding.env_file,
        help="Path to .env file containing OPENAI_API_KEY.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_config(args.config)
    load_environment(args.env_file)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to the .env file or export it to the environment."
        )

    client = OpenAI(api_key=api_key)

    input_files = list_input_files(args.input_dir)
    if not input_files:
        raise FileNotFoundError(f"No JSON files found in {args.input_dir}")

    for input_path in input_files:
        create_embeddings_for_file(
            client=client,
            input_path=input_path,
            output_dir=args.output_dir,
            model=args.model,
            batch_size=args.batch_size,
            provider=app_config.embedding.provider,
        )


if __name__ == "__main__":
    main()
