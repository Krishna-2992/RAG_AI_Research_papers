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
# Production Hierarchy Processing Engine
# -------------------------------------------------------
def build_breadcrumb_string(path_elements):
    """Converts a structural JSON path list into a clear text context string."""
    clean_elements = []
    for elem in path_elements:
        # Strip out technical markers to get raw semantic text labels
        if "/" in elem:
            parts = elem.split("/")
            if len(parts) > 1 and not parts[-1].isdigit():
                clean_elements.append(parts[-1].strip())
        else:
            clean_elements.append(elem.strip())
    return " > ".join(clean_elements) if clean_elements else "Root"

def chunk_parsed_json(json_data, paper_name):
    """Processes Docling structural nodes into hybrid-ready chunks."""
    chunks = []
    
    # Safely query structural elements tree from the internal model
    elements = json_data.get("description", {}).get("elements", [])
    if not elements:
        # Fallback to secondary root array check depending on Docling version dump
        elements = json_data.get("elements", [])

    for idx, item in enumerate(elements):
        item_type = item.get("type", "paragraph")
        text_content = item.get("text", "").strip()
        path_list = item.get("path", [])
        
        if not text_content:
            continue
            
        # Build contextual prefix breadcrumbs for SPLADE lexical keyword enrichment
        breadcrumb = build_breadcrumb_string(path_list)
        
        # Structure the explicit content block for Dense & Sparse matrix calculations
        # Prepending the hierarchy context directly feeds terms to SPLADE++
        hybrid_search_text = f"Document: {paper_name} | Section: {breadcrumb} | Content: {text_content}"
        
        # Track structural metadata payload parameters targeted for Milvus schema fields
        metadata = {
            "chunk_id": f"{paper_name}_chunk_{idx+1}",
            "document_name": paper_name,
            "element_type": item_type,
            "hierarchy_path": breadcrumb,
            "page_numbers": item.get("prov", [{}])[0].get("page_no", 1) if item.get("prov") else 1,
            "is_table": 1 if item_type == "table" else 0
        }
        
        chunks.append({
            "vector_text": hybrid_search_text,
            "raw_text": text_content,
            "metadata": metadata
        })
        
    return chunks

# -------------------------------------------------------
# Pipeline Execution Loop
# -------------------------------------------------------
json_files = list(PARSED_PAPERS.glob("**/*.json"))
print(f"Found {len(json_files)} structural JSON document(s) for advanced chunking.\n")

for json_path in json_files:
    # Use parent directory name or filename stem to match your current tree
    paper_name = json_path.stem
    print(f"Analyzing structural mapping nodes for: {paper_name}")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_content = json.load(f)
            
        optimized_chunks = chunk_parsed_json(json_content, paper_name)
        
        # Direct writing targeting the requested 'chunk_pdfs' destination folder
        output_file_path = CHUNKED_OUTPUT / f"{paper_name}_hybrid_chunks.json"
        
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            json.dump(optimized_chunks, out_f, indent=2, ensure_ascii=False)
            
        print(f"-> Successfully written {len(optimized_chunks)} chunks to: {output_file_path.name}\n")
        
    except Exception as e:
        print(f"Error executing pipeline chunking routines on {json_path.name}: {e}")

print("Hybrid RAG chunking extraction phase complete.")