# Advanced RAG Project Roadmap

This project is a research-paper RAG pipeline built around arXiv PDFs. The current repo already covers ingestion, parsing, chunking, and embeddings. What remains is the retrieval, answer generation, serving, and evaluation layers that turn the pipeline into a complete advanced RAG system.

## Current State

- `ingestion/` downloads PDFs and writes paper metadata.
- `ingestion/parse_pdfs.py` converts PDFs into structured documents.
- `chunking/` turns parsed sections into RAG-ready chunks.
- `embedding/` creates embeddings for those chunks.
- `core/` holds shared config and Pydantic schemas.

## Target End State

The final system should let you:

- ingest papers reproducibly
- parse them into clean, validated structures
- chunk them with metadata and provenance
- embed them into a vector database
- retrieve relevant chunks for a query
- rerank retrieved results
- generate grounded answers with citations
- evaluate retrieval and generation quality

## Build Order

### Phase 1: Foundation

Goal: make the pipeline stable and reproducible.

- Keep `core/config.py` and `core/schemas.py` as the shared contract.
- Use `config.yaml` as the single source of truth for paths and settings.
- Standardize output objects:
  - `DocumentMetadata`
  - `ParsedDocument`
  - `ChunkedDocument`
  - `EmbeddedDocument`
- Add logging and basic run scripts.
- Keep generated outputs separated in `data/raw_papers`, `data/parsed_papers`, `data/chunked_papers`, and `data/embeddings`.

### Phase 2: Vector Store

Goal: make embeddings searchable.

- Create `vector_store/`.
- Add Qdrant collection creation.
- Add embedding upsert logic.
- Store chunk metadata alongside vectors.
- Support reindexing from existing `data/embeddings` files.
- Add delete/rebuild utilities for a clean index.

### Phase 3: Retrieval

Goal: answer questions by finding the right context.

- Create `retrieval/`.
- Implement semantic search over Qdrant.
- Add metadata filters for document, section, date, or source.
- Return ranked chunks with scores and provenance.
- Add optional context expansion from neighboring chunks.
- Add query normalization or rewriting later if needed.

### Phase 4: Reranking

Goal: improve retrieval precision.

- Add a reranking layer after top-k retrieval.
- Start with simple heuristics if needed.
- Later add a cross-encoder or LLM-based reranker.
- Keep both retrieval and reranking scores in the result object.

### Phase 5: Generation

Goal: produce grounded answers.

- Create `prompt_builder/` if needed for assembling prompts.
- Create `generation/` or `llm_generation/` for answer synthesis.
- Pass retrieved chunks into a structured prompt.
- Require citations to chunk IDs or paper references.
- Add a refusal path for weak or irrelevant retrieval.
- Add answer formatting for concise, source-backed responses.

### Phase 6: Serving

Goal: make the system usable.

- Add a CLI query interface first.
- Add a FastAPI backend next if you want an API.
- Optionally add a UI using Streamlit or Gradio.
- Expose endpoints for:
  - query
  - retrieval debug
  - document lookup
  - reindexing

### Phase 7: Evaluation

Goal: know if the system is actually getting better.

- Add retrieval evaluation tests.
- Add answer faithfulness checks.
- Add citation correctness checks.
- Add latency measurements.
- Add a small benchmark set of questions.
- Track regressions across changes.

### Phase 8: Polish

Goal: make the repo feel production-ready.

- Write a complete `README.md`.
- Add `.env.example`.
- Add tests for schema validation and pipeline invariants.
- Add command-line entry points.
- Add better logging and error handling.
- Remove old or noisy artifacts from `data/processed`.

## Recommended Folder Structure

```text
project/
  core/
    config.py
    schemas.py
  ingestion/
    pdfs_downloader.py
    parse_pdfs.py
  chunking/
    chunk_sections.py
  embedding/
    create_embeddings.py
  vector_store/
  retrieval/
  prompt_builder/
  generation/
  evaluation/
  data/
    raw_papers/
    metadata/
    parsed_papers/
    chunked_papers/
    embeddings/
```

## What Counts As Complete

The project is complete when all of these work end to end:

- download papers
- parse PDFs into validated structure
- chunk cleanly
- embed chunks
- store vectors in Qdrant
- retrieve relevant chunks for a query
- rerank results
- generate a cited answer
- evaluate quality with repeatable tests

## Suggested Next Milestone

The next implementation milestone should be:

1. build `vector_store/`
2. ingest the current embeddings into Qdrant
3. implement retrieval
4. expose a CLI query command

That will convert the project from a data-prep pipeline into a real RAG application.



Here’s the order I recommend we actually execute:

Fix parser/chunker quality issues.
Define unified schemas and config.
Build Qdrant indexing pipeline.
Build retrieval pipeline.
Build answer generation with citations.
Add CLI for end-to-end querying.
Add evaluation suite.
Add API/UI.
Add advanced upgrades like hybrid search, reranking, query rewriting, caching, conversation memory.