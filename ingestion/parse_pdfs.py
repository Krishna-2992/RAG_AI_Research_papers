import json
import os
import re
import statistics
import sys
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import load_config
from schemas import DocumentMetadata, ParsedDocument, ParsedSection, ParserInfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse downloaded PDFs into structured section JSON."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the YAML config file.",
    )
    return parser.parse_args()

CANONICAL_SECTION_MAP = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "preliminaries": "Preliminaries",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Approach",
    "architecture": "Architecture",
    "experimental setup": "Experimental Setup",
    "experiments": "Experiments",
    "results": "Results",
    "evaluation": "Evaluation",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "references": "References",
}

KNOWN_SECTION_KEYWORDS = sorted(CANONICAL_SECTION_MAP, key=len, reverse=True)


def normalize_space(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_inline_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_paragraph_text(text: str) -> str:
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_all_caps(text: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", text)
    return bool(letters) and letters.upper() == letters


def is_equation_like(text: str) -> bool:
    math_symbols = sum(1 for ch in text if ch in "=<>∈≈±∑∫×÷√{}[]^_")
    return math_symbols >= 2 or bool(re.search(r"\b[a-zA-Z]\s*=\s*[\w(]", text))


def is_caption_like(text: str) -> bool:
    lowered = text.lower()
    return bool(re.match(r"^(figure|fig\.|table|algorithm)\s*\d+", lowered))


def is_affiliation_like(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "university",
        "institute",
        "department",
        "laboratory",
        "school of",
        "college",
        "@",
        "email",
        "corresponding author",
    ]
    return any(marker in lowered for marker in markers)


def is_person_name_like(text: str) -> bool:
    words = clean_inline_text(text).split()
    if not 2 <= len(words) <= 5:
        return False

    allowed_particles = {"de", "da", "del", "van", "von", "di", "du"}
    valid_words = 0
    for word in words:
        plain = re.sub(r"[^A-Za-z`'-]", "", word)
        if not plain:
            return False
        if plain.lower() in allowed_particles:
            valid_words += 1
            continue
        if plain[0].isupper() and plain[1:].islower():
            valid_words += 1
            continue
        return False

    return valid_words == len(words)


def suspicious_heading_text(text: str) -> bool:
    text = clean_inline_text(text)
    if not text:
        return True
    alpha_chars = re.findall(r"[A-Za-z]", text)
    if len(alpha_chars) < 4:
        return True
    if len(alpha_chars) / max(1, len(text)) < 0.45:
        return True
    if len(text) > 120:
        return True
    if text.endswith("."):
        return True
    if text.count(",") >= 2:
        return True
    if "http://" in text or "https://" in text:
        return True
    if is_affiliation_like(text):
        return True
    if is_equation_like(text):
        return True
    return False


def normalize_heading(raw_heading: str) -> str:
    text = clean_inline_text(raw_heading)
    text = re.sub(r"^(?:\d+(?:\.\d+)*|[IVXLCM]+)[\.\)]?\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" :-")
    lowered = text.lower()

    for key in KNOWN_SECTION_KEYWORDS:
        if lowered == key or lowered.startswith(f"{key} "):
            return CANONICAL_SECTION_MAP[key]

    return text or "Unknown"


def line_text_from_spans(spans: Sequence[Dict[str, Any]]) -> str:
    pieces = [span["text"] for span in spans if span.get("text", "").strip()]
    return clean_inline_text(" ".join(pieces))


def line_bbox(spans: Sequence[Dict[str, Any]], fallback_bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if fallback_bbox:
        return tuple(fallback_bbox)  # type: ignore[return-value]

    x0 = min(span["bbox"][0] for span in spans)
    y0 = min(span["bbox"][1] for span in spans)
    x1 = max(span["bbox"][2] for span in spans)
    y1 = max(span["bbox"][3] for span in spans)
    return (x0, y0, x1, y1)


def extract_lines(doc: fitz.Document) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []

    for page_index, page in enumerate(doc):
        page_dict = page.get_text("dict")
        page_width = float(page.rect.width)

        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                spans = [span for span in line["spans"] if span.get("text", "").strip()]
                if not spans:
                    continue

                text = line_text_from_spans(spans)
                if not text:
                    continue

                bbox = line_bbox(spans, line.get("bbox"))
                sizes = [float(span["size"]) for span in spans]
                bold = any(bool(span.get("flags", 0) & (1 << 4)) for span in spans)
                lines.append({
                    "page": page_index + 1,
                    "text": text,
                    "bbox": bbox,
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "width": bbox[2] - bbox[0],
                    "font_size": max(sizes),
                    "avg_font_size": sum(sizes) / len(sizes),
                    "is_bold": bold,
                    "page_width": page_width,
                })

    return lines


def estimate_body_font_size(lines: Sequence[Dict[str, Any]]) -> float:
    candidates = [
        line["avg_font_size"]
        for line in lines
        if 30 <= len(line["text"]) <= 300 and not is_caption_like(line["text"])
    ]
    if not candidates:
        candidates = [line["avg_font_size"] for line in lines]
    return statistics.median(candidates) if candidates else 10.0


def order_page_lines(page_lines: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not page_lines:
        return []

    page_width = page_lines[0]["page_width"]
    top_band = page_width * 0.72
    full_width = [line for line in page_lines if line["width"] >= top_band]
    normal = [line for line in page_lines if line not in full_width]

    left = [line for line in normal if line["x0"] < page_width * 0.5]
    right = [line for line in normal if line["x0"] >= page_width * 0.5]
    two_column = len(left) >= 8 and len(right) >= 8

    ordered: List[Dict[str, Any]] = []
    ordered.extend(sorted(full_width, key=lambda item: (item["y0"], item["x0"])))

    if two_column:
        ordered.extend(sorted(left, key=lambda item: (item["y0"], item["x0"])))
        ordered.extend(sorted(right, key=lambda item: (item["y0"], item["x0"])))
    else:
        ordered.extend(sorted(normal, key=lambda item: (item["y0"], item["x0"])))

    return ordered


def ordered_lines(doc: fitz.Document) -> List[Dict[str, Any]]:
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for line in extract_lines(doc):
        by_page.setdefault(line["page"], []).append(line)

    ordered: List[Dict[str, Any]] = []
    for page_number in sorted(by_page):
        ordered.extend(order_page_lines(by_page[page_number]))
    return ordered


def detect_title(lines: Sequence[Dict[str, Any]], body_font_size: float) -> str:
    first_page_lines = [line for line in lines if line["page"] == 1 and line["y0"] < 220]
    title_candidates = [
        line for line in first_page_lines
        if line["font_size"] >= body_font_size * 1.30
        and not is_affiliation_like(line["text"])
        and not is_caption_like(line["text"])
        and "arxiv:" not in line["text"].lower()
        and len(line["text"]) >= 8
    ]

    if not title_candidates:
        return ""

    title_candidates.sort(key=lambda item: (item["y0"], item["x0"]))
    title_lines = [title_candidates[0]["text"]]
    last_y = title_candidates[0]["y0"]

    for line in title_candidates[1:]:
        if line["y0"] - last_y <= 22:
            title_lines.append(line["text"])
            last_y = line["y0"]
        else:
            break

    return clean_inline_text(" ".join(title_lines))


def is_heading_candidate(text: str, line: Dict[str, Any], body_font_size: float, seen_section: bool) -> bool:
    text = clean_inline_text(text)
    if not text or suspicious_heading_text(text):
        return False
    if text[0].islower():
        return False

    numbered = bool(re.match(r"^(?:\d+(?:\.\d+)*|[IVXLCM]+)[\.\)]?\s+[A-Z]", text))
    known = any(
        text.lower() == keyword or text.lower().startswith(f"{keyword} ")
        for keyword in KNOWN_SECTION_KEYWORDS
    )
    big = line["font_size"] >= body_font_size * 1.12
    bold = line["is_bold"] and line["font_size"] >= body_font_size * 1.02
    caps = is_all_caps(text) and len(text.split()) <= 12

    if not (numbered or known or big or bold or caps):
        return False

    if line["page"] == 1 and line["y0"] < 260 and not known and not numbered:
        return False
    if line["width"] < line["page_width"] * 0.16 and not numbered and not known:
        return False
    if line["page"] <= 2 and is_person_name_like(text) and not known and not numbered:
        return False

    return True


def split_embedded_heading(text: str, body_font_size: float, line: Dict[str, Any], seen_section: bool) -> Tuple[Optional[str], str]:
    text = clean_inline_text(text)

    abstract_match = re.match(r"^(Abstract)\s*[:.\-]?\s+(.*)$", text, flags=re.IGNORECASE)
    if abstract_match:
        return abstract_match.group(1), abstract_match.group(2)

    numbered_match = re.match(
        r"^((?:\d+(?:\.\d+)*|[IVXLCM]+)[\.\)]?\s+[A-Z][A-Za-z0-9,/\-() ]{1,80}?)\s{2,}(.*)$",
        text,
    )
    if numbered_match and is_heading_candidate(numbered_match.group(1), line, body_font_size, seen_section):
        return numbered_match.group(1), numbered_match.group(2)

    return None, text


def push_section(
    sections: List[Dict[str, Any]],
    document_id: str,
    current_heading: Optional[str],
    current_raw_heading: Optional[str],
    current_page_start: Optional[int],
    paragraph_buffer: List[str],
    current_page_end: Optional[int],
) -> None:
    if not current_heading or not paragraph_buffer:
        return

    merged_text = merge_lines_to_paragraphs(paragraph_buffer)
    paragraphs = [clean_paragraph_text(paragraph) for paragraph in merged_text if clean_paragraph_text(paragraph)]
    if not paragraphs:
        return

    text = "\n\n".join(paragraphs)
    sections.append(
        ParsedSection(
            section_id=f"{document_id}::section_{len(sections):03d}",
            document_id=document_id,
            section_title=current_heading,
            normalized_section_title=current_heading,
            raw_heading=current_raw_heading or current_heading,
            page_start=current_page_start,
            page_end=current_page_end,
            paragraphs=paragraphs,
            text=text,
        ).model_dump(mode="json")
    )


def merge_lines_to_paragraphs(lines: Sequence[str]) -> List[str]:
    paragraphs: List[str] = []
    current = ""

    for raw_line in lines:
        line = clean_inline_text(raw_line)
        if not line or re.fullmatch(r"\d+", line):
            continue

        if line.startswith("•"):
            if current:
                paragraphs.append(current)
            current = line
            continue

        if not current:
            current = line
            continue

        if current.endswith("-"):
            current = current[:-1] + line.lstrip()
        else:
            current = f"{current} {line}"

    if current:
        paragraphs.append(current)

    return paragraphs


def load_document_metadata(metadata_dir: Path, pdf_path: Path, source_type: str) -> DocumentMetadata:
    metadata_path = metadata_dir / f"{pdf_path.stem}.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            return DocumentMetadata.model_validate(json.load(f))

    return DocumentMetadata(
        document_id=pdf_path.stem,
        title=pdf_path.stem,
        source_file=pdf_path.name,
        source_path=str(pdf_path),
        source_type=source_type,
    )


def extract_sections_from_pdf(path: Path, metadata_dir: Path, parser_version: str, source_type: str) -> ParsedDocument:
    doc = fitz.open(path)
    document = load_document_metadata(metadata_dir=metadata_dir, pdf_path=path, source_type=source_type)
    paper_id = document.document_id
    lines = ordered_lines(doc)
    body_font_size = estimate_body_font_size(lines)
    detected_title = detect_title(lines, body_font_size) or (doc.metadata.get("title") or "").strip() or paper_id
    document.title = document.title or detected_title
    if document.title == paper_id:
        document.title = detected_title

    sections: List[Dict[str, Any]] = []
    current_heading: Optional[str] = None
    current_raw_heading: Optional[str] = None
    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None
    paragraph_buffer: List[str] = []
    seen_real_section = False

    def finalize_current_section() -> None:
        nonlocal paragraph_buffer
        push_section(
            sections=sections,
            document_id=paper_id,
            current_heading=current_heading,
            current_raw_heading=current_raw_heading,
            current_page_start=current_page_start,
            paragraph_buffer=paragraph_buffer,
            current_page_end=current_page_end,
        )
        paragraph_buffer = []

    for line in lines:
        text = clean_inline_text(line["text"])
        if not text:
            continue
        if is_caption_like(text):
            continue
        if "arxiv:" in text.lower():
            continue
        if re.fullmatch(r"\d+", text):
            continue

        embedded_heading, remainder = split_embedded_heading(text, body_font_size, line, seen_real_section)
        if embedded_heading:
            if current_heading and paragraph_buffer:
                finalize_current_section()
            current_raw_heading = embedded_heading
            current_heading = normalize_heading(embedded_heading)
            current_page_start = line["page"]
            current_page_end = line["page"]
            paragraph_buffer = []
            seen_real_section = True
            if remainder:
                paragraph_buffer.append(remainder)
            continue

        if is_heading_candidate(text, line, body_font_size, seen_real_section):
            normalized = normalize_heading(text)

            if normalized == "References":
                if current_heading and paragraph_buffer:
                    finalize_current_section()
                break

            if current_heading and paragraph_buffer:
                finalize_current_section()

            current_raw_heading = text
            current_heading = normalized
            current_page_start = line["page"]
            current_page_end = line["page"]
            paragraph_buffer = []
            seen_real_section = True
            continue

        if not seen_real_section:
            continue

        if current_heading is None:
            current_heading = "Unknown"
            current_raw_heading = "Unknown"
            current_page_start = line["page"]

        paragraph_buffer.append(text)
        current_page_end = line["page"]

    if current_heading and paragraph_buffer:
        finalize_current_section()

    return ParsedDocument(
        document=document,
        parser=ParserInfo(
            version=parser_version,
            body_font_size=round(body_font_size, 2),
        ),
        sections=[ParsedSection.model_validate(section) for section in sections],
    )


def process_all_pdfs(config_path: Optional[Path] = None) -> None:
    app_config = load_config(config_path)
    pdf_dir = app_config.paths.raw_papers
    out_dir = app_config.paths.parsed_papers
    metadata_dir = app_config.paths.metadata
    out_dir.mkdir(parents=True, exist_ok=True)

    for filename in sorted(os.listdir(pdf_dir)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = pdf_dir / filename
        print(f"Parsing {pdf_path}...")
        parsed = extract_sections_from_pdf(
            path=pdf_path,
            metadata_dir=metadata_dir,
            parser_version=app_config.parsing.parser_version,
            source_type=app_config.parsing.source_type,
        )

        out_path = out_dir / f"{pdf_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed.model_dump(mode="json"), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    args = parse_args()
    process_all_pdfs(args.config)
