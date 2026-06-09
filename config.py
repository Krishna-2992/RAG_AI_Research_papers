from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "advanced-rag"
    version: str = "1.0"


class PathsConfig(BaseModel):
    raw_papers: Path = Path("data/raw_papers")
    parsed_papers: Path = Path("data/parsed_papers")
    chunked_papers: Path = Path("data/chunked_papers")
    embeddings: Path = Path("data/embeddings")
    metadata: Path = Path("data/metadata")


class IngestionConfig(BaseModel):
    source: str = "arxiv"
    query: str = 'ti:"transformer" OR abs:"transformer" AND (cat:cs.LG OR cat:cs.CL)'
    max_results: int = 10
    sort_by: str = "submitted_date"
    sort_order: str = "descending"


class ParsingConfig(BaseModel):
    parser_version: str = "3.0"
    source_type: str = "arxiv"


class ChunkingConfig(BaseModel):
    strategy: str = "section_based"
    min_tokens: int = 120
    max_tokens: int = 300
    overlap_tokens: int = 35


class EmbeddingConfig(BaseModel):
    provider: str = "openai"
    model: str = "text-embedding-3-large"
    batch_size: int = 32
    env_file: Path = Path(".env")


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)


def resolve_config_path(config_path: Optional[Path] = None) -> Path:
    if config_path is not None:
        return config_path.resolve()
    return (Path(__file__).resolve().parent / "config.yaml").resolve()


@lru_cache(maxsize=4)
def load_config(config_path: Optional[Path] = None) -> AppConfig:
    resolved = resolve_config_path(config_path)
    with open(resolved, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return AppConfig.model_validate(payload)
