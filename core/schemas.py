from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    document_id: str
    title: str
    source_file: str
    source_path: Optional[str] = None
    source_type: str = "arxiv"
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    published_at: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    primary_category: Optional[str] = None
    entry_id: Optional[str] = None
    pdf_url: Optional[str] = None
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    comment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParserInfo(BaseModel):
    version: str
    body_font_size: float


class ParsedSection(BaseModel):
    section_id: str
    document_id: str
    section_title: str
    normalized_section_title: str
    raw_heading: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    paragraphs: List[str] = Field(default_factory=list)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    document: DocumentMetadata
    parser: ParserInfo
    sections: List[ParsedSection] = Field(default_factory=list)


class ChunkingInfo(BaseModel):
    strategy: str
    min_tokens: int
    max_tokens: int
    overlap_tokens: int


class ChunkRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    section_ids: List[str] = Field(default_factory=list)
    section_path: List[str] = Field(default_factory=list)
    primary_section: str = "Unknown"
    token_count: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    document: DocumentMetadata
    chunking: ChunkingInfo
    chunks: List[ChunkRecord] = Field(default_factory=list)


class EmbeddingInfo(BaseModel):
    provider: str
    model: str
    batch_size: int


class EmbeddingRecord(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    section_path: List[str] = Field(default_factory=list)
    token_count: int
    text: str
    vector: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddedDocument(BaseModel):
    document: DocumentMetadata
    embedding: EmbeddingInfo
    chunk_count: int
    embeddings: List[EmbeddingRecord] = Field(default_factory=list)

