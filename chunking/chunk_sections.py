import argparse
import json
import os
import re
from typing import Any, Dict, List, Sequence


DEFAULT_INPUT_DIR = "data/parsed_papers"
DEFAULT_OUTPUT_DIR = "data/chunked_papers"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_section_name(section_name: str) -> str:
    section_name = normalize_whitespace(section_name)
    if not section_name:
        return "Unknown"

    lowered = section_name.lower()
    canonical_names = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "background": "Background",
        "related work": "Related Work",
        "method": "Method",
        "methods": "Method",
        "methodology": "Method",
        "approach": "Approach",
        "experiments": "Experiments",
        "results": "Results",
        "evaluation": "Evaluation",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "references": "References",
    }

    for key, value in canonical_names.items():
        if key in lowered:
            return value

    return section_name


def trim_non_content_tail(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return ""

    split_markers = [
        r"\bAcknowledg(e)?ments?\b",
        r"\bReferences\b",
        r"\bBibliography\b",
    ]
    for marker in split_markers:
        match = re.search(marker, text, flags=re.IGNORECASE)
        if match and match.start() > 0:
            return normalize_whitespace(text[:match.start()])

    return text


def is_noise_section(section_name: str, text: str) -> bool:
    combined = f"{section_name} {text}".strip().lower()

    if not combined:
        return True

    noise_patterns = [
        r"\bpreprint version\b",
        r"\bcorresponding author\b",
        r"\bemail addresses?\b",
        r"\bkeywords?\b",
        r"\bdoi:\b",
        r"\barxiv\b",
        r"\bsupplementary materials?\b",
        r"\bappendix\b",
    ]
    if any(re.search(pattern, combined) for pattern in noise_patterns):
        return True

    if re.fullmatch(r"[ivxlcdm]+", section_name.strip().lower()):
        return True

    if len(normalize_whitespace(text)) < 40 and len(normalize_whitespace(section_name)) > 50:
        return True

    return False


def is_reference_like(text: str) -> bool:
    citations = len(re.findall(r"\[\d+\]", text))
    return citations >= 8


def approximate_tokens(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def split_into_paragraphs(text: str) -> List[str]:
    text = text.replace("\r\n", "\n")
    raw_parts = re.split(r"\n\s*\n", text)

    if len(raw_parts) == 1:
        raw_parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    return [normalize_whitespace(part) for part in raw_parts if normalize_whitespace(part)]


def split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = approximate_tokens(paragraph)

        if current_parts and current_tokens + paragraph_tokens > max_tokens:
            chunks.append(" ".join(current_parts))
            overlap_text = last_tokens_text(" ".join(current_parts), overlap_tokens)
            current_parts = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_tokens = approximate_tokens(" ".join(current_parts))
            continue

        if paragraph_tokens > max_tokens:
            sentence_parts = re.split(r"(?<=[.!?])\s+", paragraph)
            for sentence in sentence_parts:
                sentence = normalize_whitespace(sentence)
                if not sentence:
                    continue

                sentence_tokens = approximate_tokens(sentence)
                if current_parts and current_tokens + sentence_tokens > max_tokens:
                    chunks.append(" ".join(current_parts))
                    overlap_text = last_tokens_text(" ".join(current_parts), overlap_tokens)
                    current_parts = [overlap_text, sentence] if overlap_text else [sentence]
                    current_tokens = approximate_tokens(" ".join(current_parts))
                else:
                    current_parts.append(sentence)
                    current_tokens += sentence_tokens
            continue

        current_parts.append(paragraph)
        current_tokens += paragraph_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return [normalize_whitespace(chunk) for chunk in chunks if normalize_whitespace(chunk)]


def last_tokens_text(text: str, token_count: int) -> str:
    if token_count <= 0:
        return ""

    tokens = re.findall(r"\S+", text)
    if not tokens:
        return ""

    return " ".join(tokens[-token_count:])


def clean_section_entries(sections: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []

    for section in sections:
        raw_name = normalize_whitespace(str(section.get("section", "")))
        paragraphs = section.get("paragraphs")
        if isinstance(paragraphs, list) and paragraphs:
            raw_text = trim_non_content_tail("\n\n".join(str(p) for p in paragraphs))
        else:
            raw_text = trim_non_content_tail(str(section.get("text", "")))

        if not raw_text:
            continue

        section_name = normalize_section_name(raw_name)
        if is_noise_section(section_name, raw_text):
            continue
        if section_name == "References" or is_reference_like(raw_text):
            continue

        cleaned.append({
            "section": section_name,
            "text": raw_text,
        })

    return cleaned


def build_section_based_chunks(
    sections: Sequence[Dict[str, str]],
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    buffer_parts: List[str] = []
    buffer_sections: List[str] = []
    buffer_tokens = 0

    def flush_buffer() -> None:
        nonlocal buffer_parts, buffer_sections, buffer_tokens

        text = normalize_whitespace(" ".join(buffer_parts))
        if text:
            chunks.append({
                "section_path": list(dict.fromkeys(buffer_sections)),
                "text": text,
                "token_count": approximate_tokens(text),
            })

        buffer_parts = []
        buffer_sections = []
        buffer_tokens = 0

    for section in sections:
        section_name = section["section"]
        section_text = normalize_whitespace(section["text"])
        if not section_text:
            continue

        section_tokens = approximate_tokens(section_text)

        if section_tokens > max_tokens:
            flush_buffer()
            for part in split_long_text(section_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
                chunks.append({
                    "section_path": [section_name],
                    "text": part,
                    "token_count": approximate_tokens(part),
                })
            continue

        should_flush = (
            buffer_parts and
            buffer_tokens >= min_tokens and
            buffer_tokens + section_tokens > max_tokens
        )
        if should_flush:
            flush_buffer()

        buffer_parts.append(section_text)
        buffer_sections.append(section_name)
        buffer_tokens += section_tokens

    flush_buffer()
    return chunks


def enrich_chunks(
    paper: Dict[str, Any],
    chunks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    paper_id = paper.get("paper_id", "")
    title = paper.get("title", "")
    total_chunks = len(chunks)

    enriched: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        enriched.append({
            "chunk_id": f"{paper_id}::chunk_{index:03d}",
            "paper_id": paper_id,
            "title": title,
            "chunk_index": index,
            "total_chunks": total_chunks,
            "section_path": chunk["section_path"],
            "primary_section": chunk["section_path"][0] if chunk["section_path"] else "Unknown",
            "token_count": chunk["token_count"],
            "text": chunk["text"],
        })

    return enriched


def merge_small_chunks(chunks: Sequence[Dict[str, Any]], min_tokens: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []

    for chunk in chunks:
        if (
            merged and
            chunk["token_count"] < min_tokens and
            merged[-1]["section_path"] == chunk["section_path"]
        ):
            merged[-1]["text"] = normalize_whitespace(
                f'{merged[-1]["text"]} {chunk["text"]}'
            )
            merged[-1]["token_count"] = approximate_tokens(merged[-1]["text"])
            continue

        merged.append(dict(chunk))

    return merged


def process_paper(
    input_path: str,
    output_path: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        paper = json.load(f)

    cleaned_sections = clean_section_entries(paper.get("sections", []))
    chunks = build_section_based_chunks(
        cleaned_sections,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    chunks = merge_small_chunks(chunks, min_tokens=min_tokens)
    enriched_chunks = enrich_chunks(paper, chunks)

    payload = {
        "paper_id": paper.get("paper_id", ""),
        "title": paper.get("title", ""),
        "source_file": os.path.basename(input_path),
        "chunking": {
            "strategy": "section_based",
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
        },
        "chunks": enriched_chunks,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_all_papers(
    input_dir: str,
    output_dir: str,
    min_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        print(f"Chunking {input_path}...")
        process_paper(
            input_path=input_path,
            output_path=output_path,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create section-aware chunks from parsed paper JSON files."
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-tokens", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--overlap-tokens", type=int, default=35)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_papers(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
