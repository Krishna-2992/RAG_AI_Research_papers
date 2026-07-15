from pathlib import Path
import json

# -------------------------------------------------------
# Project paths
# -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PARSED_PAPERS = PROJECT_ROOT / "data" / "parsed_papers"
CHUNKED_OUTPUT = PROJECT_ROOT / "data" / "chunk_pdfs"

CHUNKED_OUTPUT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------
# Text Optimization & Normalization Utilities
# -------------------------------------------------------
def clean_text_encoding(text):
    """
    Cleans up bad Unicode character sets and maps them to standard terms
    to ensure high lexical matching success inside the SPLADE++ encoder.
    """
    if not text:
        return ""
        
    replacements = {
        "â—¦": " degrees ",
        "âˆ†": " delta ",
        "Î¸": " theta ",
        "Ï•": " phi ",
        "Â·": " - ",
        "â—": " * ",
        "âˆ—": " * ",
        "â— ": " * ",
        "Ã—": "x",
        "â‰¥": " >= ",
        "â‰¤": " <= ",
        "âˆ’": "-",
        "Ï„": " tau ",
        "Ïˆ": " psi ",
        "Ï†": " phi ",
    }
    
    for broken_char, clean_char in replacements.items():
        text = text.replace(broken_char, clean_char)
        
    return " ".join(text.split())

# -------------------------------------------------------
# Resilient Document Tree Traversal Method
# -------------------------------------------------------
def extract_elements_adaptively(data):
    """
    Scans the JSON layout tree adaptively to catch items regardless 
    of flat array serialization variations.
    """
    text_items = []
    table_items = []

    # Strategy 1: Look for flat collection dumps
    if "texts" in data and isinstance(data["texts"], list):
        text_items.extend(data["texts"])
    if "tables" in data and isinstance(data["tables"], list):
        table_items.extend(data["tables"])
        
    # Strategy 2: Deep search dictionary collections for layout components
    def traverse_dict(node):
        if isinstance(node, dict):
            # Check for generic item blocks matching structural layouts
            if "text" in node and ("label" in node or "prov" in node):
                text_items.append(node)
            elif "data" in node and "table_cells" in node.get("data", {}):
                table_items.append(node)
            else:
                for k, v in node.items():
                    traverse_dict(v)
        elif isinstance(node, list):
            for element in node:
                traverse_dict(element)

    # Trigger adaptive scanning if flat strategies come up empty
    if not text_items and not table_items:
        traverse_dict(data)

    return text_items, table_items

# -------------------------------------------------------
# Production Hierarchy Processing Engine
# -------------------------------------------------------
def chunk_parsed_json(json_data, paper_name):
    """Processes Docling JSON tree schema arrays directly into hybrid-ready RAG chunks."""
    chunks = []
    chunk_counter = 1

    text_items, table_items = extract_elements_adaptively(json_data)
    current_section = "Introduction/Root"
    
    # 1. Process standard layout rich text strings
    for item in text_items:
        raw_content = item.get("text", "").strip()
        label = item.get("label", "paragraph")
        
        if not raw_content:
            continue
            
        # Track our current location down the document tree
        if label in ["title", "section_header", "heading"]:
            current_section = clean_text_encoding(raw_content)
            
        cleaned_content = clean_text_encoding(raw_content)
        
        # QUALITY CONTROL FILTER: Skip layout extraction noise fragments.
        if len(cleaned_content.split()) < 4 and label not in ["title", "section_header", "heading"]:
            continue
            
        hybrid_search_text = f"Document: {paper_name} | Section: {current_section} | Content: {cleaned_content}"
        
        page_no = 1
        if item.get("prov") and len(item["prov"]) > 0:
            page_no = item["prov"][0].get("page_no", 1)
            
        metadata = {
            "chunk_id": f"{paper_name}_chunk_{chunk_counter}",
            "document_name": paper_name,
            "element_type": label,
            "hierarchy_path": current_section,
            "page_numbers": page_no,
            "is_table": 0
        }
        
        chunks.append({
            "vector_text": hybrid_search_text,
            "raw_text": cleaned_content,
            "metadata": metadata
        })
        chunk_counter += 1

    # 2. Process multi-column data structures (Tables)
    for t_idx, table in enumerate(table_items):
        grid_cells = table.get("data", {}).get("table_cells", [])
        if not grid_cells:
            continue
            
        table_rows = {}
        for cell in grid_cells:
            r_idx = cell.get("start_row_offset_idx", 0)
            c_text = cell.get("text", "").strip()
            
            if r_idx not in table_rows:
                table_rows[r_idx] = []
            if c_text:
                table_rows[r_idx].append(clean_text_encoding(c_text))
                
        table_markdown_sim = "\n".join([" | ".join(row) for row in table_rows.values() if row])
        
        if not table_markdown_sim.strip():
            continue
            
        hybrid_search_text = f"Document: {paper_name} | Section: Data Table {t_idx+1} | Table Content:\n{table_markdown_sim}"
        
        page_no = 1
        if table.get("prov") and len(table["prov"]) > 0:
            page_no = table["prov"][0].get("page_no", 1)

        metadata = {
            "chunk_id": f"{paper_name}_chunk_{chunk_counter}",
            "document_name": paper_name,
            "element_type": "table",
            "hierarchy_path": f"Table_{t_idx+1}",
            "page_numbers": page_no,
            "is_table": 1
        }
        
        chunks.append({
            "vector_text": hybrid_search_text,
            "raw_text": table_markdown_sim,
            "metadata": metadata
        })
        chunk_counter += 1
        
    return chunks

# -------------------------------------------------------
# Pipeline Execution Loop
# -------------------------------------------------------
if __name__ == "__main__":
    json_files = list(PARSED_PAPERS.glob("**/*.json"))
    print(f"Found {len(json_files)} structural JSON document(s) for advanced chunking.\n")

    for json_path in json_files:
        paper_name = json_path.stem
        print(f"Analyzing structural mapping nodes for: {paper_name}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                json_content = json.load(f)
                
            optimized_chunks = chunk_parsed_json(json_content, paper_name)
            
            output_file_path = CHUNKED_OUTPUT / f"{paper_name}_hybrid_chunks.json"
            
            with open(output_file_path, "w", encoding="utf-8") as out_f:
                json.dump(optimized_chunks, out_f, indent=2, ensure_ascii=False)
                
            print(f"-> Successfully written {len(optimized_chunks)} chunks to: {output_file_path.name}\n")
            
        except Exception as e:
            print(f"Error executing pipeline chunking routines on {json_path.name}: {e}")

    print("Hybrid RAG chunking extraction phase complete.")