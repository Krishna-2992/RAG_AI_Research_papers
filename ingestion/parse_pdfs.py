import os
import re
import json
import fitz  # PyMuPDF

PDF_DIR = "data/raw_papers"
OUT_DIR = "data/parsed_papers"
os.makedirs(OUT_DIR, exist_ok=True)

# Common section names for research papers (can be extended)
SECTION_KEYWORDS = [
    "abstract", "introduction", "background", "related work",
    "method", "methods", "methodology",
    "experiments", "results", "evaluation",
    "discussion", "conclusion", "conclusions", "references"
]

def looks_like_heading(span, avg_font_size):
    """Heuristic: bigger font, short text, maybe all caps."""
    text = span["text"].strip()
    if not text:
        return False

    size = span["size"]
    flags = span["flags"]  # bold / italic info

    is_big = size >= avg_font_size * 1.15
    is_short = len(text) < 80
    is_all_caps = (len(re.sub(r"[^A-Za-z]", "", text)) > 0
                   and text.upper() == text)

    is_bold = bool(flags & (1 << 4))  # PyMuPDF bold flag [web:32][web:37]

    return is_short and (is_big or is_all_caps or is_bold)

def normalize_heading(text):
    t = re.sub(r"\s+", " ", text).strip().lower()
    # Strip leading numbering like "1. Introduction"
    t = re.sub(r"^\d+(\.\d+)*\s*", "", t)
    return t

def map_to_known_section(raw_heading):
    h = normalize_heading(raw_heading)
    for kw in SECTION_KEYWORDS:
        if kw in h:
            # Normalize a few variants
            if "abstract" in kw:
                return "Abstract"
            if "introduction" in kw:
                return "Introduction"
            if "method" in kw:
                return "Method"
            if "experiment" in kw or "result" in kw or "evaluation" in kw:
                return "Experiments / Results"
            if "discussion" in kw:
                return "Discussion"
            if "conclusion" in kw:
                return "Conclusion"
            if "reference" in kw:
                return "References"
    # Fallback: title‑cased original (without numbers)
    clean = re.sub(r"^\d+(\.\d+)*\s*", "", raw_heading).strip()
    return clean or "Unknown"

def extract_sections_from_pdf(path):
    doc = fitz.open(path)
    paper_id = os.path.splitext(os.path.basename(path))[0]

    # Compute average font size over first few pages for thresholding
    font_sizes = []
    for page in doc[: min(3, len(doc))]:
        blocks = page.get_text("dict")["blocks"]  # layout info [web:32][web:46]
        for b in blocks:
            if "lines" not in b:
                continue
            for l in b["lines"]:
                for s in l["spans"]:
                    if s["text"].strip():
                        font_sizes.append(s["size"])
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 10.0

    sections = []
    current_section_name = "Unknown"
    current_text_chunks = []

    def push_section():
        nonlocal sections, current_section_name, current_text_chunks
        text = "\n".join(current_text_chunks).strip()
        if text:
            sections.append({
                "section": current_section_name,
                "text": text,
            })
        current_text_chunks = []

    # Iterate pages in order and accumulate text under headings
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for l in b["lines"]:
                # Merge spans in a line
                line_spans = l["spans"]
                line_text = " ".join(s["text"] for s in line_spans).strip()
                if not line_text:
                    continue

                # Decide if this line is a heading (using first span as proxy)
                if looks_like_heading(line_spans[0], avg_font_size):
                    # Start new section
                    push_section()
                    current_section_name = map_to_known_section(line_text)
                else:
                    current_text_chunks.append(line_text)

    # Push last section
    push_section()

    # Title: best guess from metadata or first non‑empty section
    title = (doc.metadata.get("title") or "").strip() or paper_id

    return {
        "paper_id": paper_id,
        "title": title,
        "sections": sections,
    }

def process_all_pdfs():
    for fname in os.listdir(PDF_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(PDF_DIR, fname)
        print(f"Parsing {pdf_path}...")
        paper_json = extract_sections_from_pdf(pdf_path)

        out_path = os.path.join(
            OUT_DIR, os.path.splitext(fname)[0] + ".json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(paper_json, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_all_pdfs()