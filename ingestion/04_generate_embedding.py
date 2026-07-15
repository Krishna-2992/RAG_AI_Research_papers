from pathlib import Path
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM

# -------------------------------------------------------
# Project paths
# -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKED_INPUT = PROJECT_ROOT / "data" / "chunk_pdfs"
EMBEDDINGS_OUTPUT = PROJECT_ROOT / "data" / "embedded_pdfs"
EMBEDDINGS_OUTPUT.mkdir(parents=True, exist_ok=True)

# Select computing device
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------------------------------
# 1. Load Local Dense Model (EmbeddingGemma)
# -------------------------------------------------------
print("Loading EmbeddingGemma...")
DENSE_MODEL_ID = "google/gemma-2b" # Replace with your exact Gemma variant if needed
dense_tokenizer = AutoTokenizer.from_pretrained(DENSE_MODEL_ID)
dense_model = AutoModelForCausalLM.from_pretrained(DENSE_MODEL_ID, torch_dtype=torch.float16 if device != "cpu" else torch.float32).to(device)
dense_model.eval()

def get_dense_embedding(text):
    """Generates normalized dense vectors via Gemma hidden states."""
    inputs = dense_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = dense_model(**inputs, output_hidden_states=True)
        # Standard approach: Take the hidden states of the last token
        embeddings = outputs.hidden_states[-1][:, -1, :]
        # Normalize vector to unit length
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings.cpu().squeeze().numpy().tolist()

# -------------------------------------------------------
# 2. Load Local Sparse Model (SPLADE++)
# -------------------------------------------------------
print("Loading SPLADE++...")
SPARSE_MODEL_ID = "naver/splade-cocondenser-ensembledistil"
sparse_tokenizer = AutoTokenizer.from_pretrained(SPARSE_MODEL_ID)
sparse_model = AutoModelForMaskedLM.from_pretrained(SPARSE_MODEL_ID).to(device)
sparse_model.eval()

def get_sparse_embedding(text):
    """Generates a {token_id: weight} dictionary via SPLADE max-pooling."""
    inputs = sparse_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        outputs = sparse_model(**inputs)
        # SPLADE logic: log(1 + relu(logits)) max-pooled across the sequence dimension (dim=1)
        logits = outputs.logits
        relu_logged = torch.log(1 + torch.relu(logits))
        sparse_vector = torch.max(relu_logged, dim=1).values.cpu().squeeze().numpy()
    
    # Filter out absolute zero weights to keep it strictly sparse for Milvus
    non_zero_indices = np.where(sparse_vector > 0.0)[0]
    
    # Milvus sparse layout accepts a dict of stringified integers or integer keys
    return {int(idx): float(sparse_vector[idx]) for idx in non_zero_indices}

# -------------------------------------------------------
# 3. Execution Pipeline Loop
# -------------------------------------------------------
chunk_files = list(CHUNKED_INPUT.glob("*_hybrid_chunks.json"))
print(f"\nFound {len(chunk_files)} chunked document file(s) to process.\n")

for file_path in chunk_files:
    print(f"Embedding elements inside: {file_path.name}")
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    embedded_chunks = []
    
    for item in chunks:
        # Use vector_text which contains Title + Section Hierarchy + Chunk Content
        text_to_embed = item["vector_text"]
        
        try:
            dense_vec = get_dense_embedding(text_to_embed)
            sparse_vec = get_sparse_embedding(text_to_embed)
            
            # Combine raw data, vectors, and metadata into a unified dictionary
            payload = {
                "vector_text": item["vector_text"],
                "raw_text": item["raw_text"],
                "dense_vector": dense_vec,
                "sparse_vector": sparse_vec,
                "metadata": item["metadata"]
            }
            embedded_chunks.append(payload)
        except Exception as e:
            print(f"Skipped a chunk due to processing error: {e}")
            
    # Save output data structure package
    output_path = EMBEDDINGS_OUTPUT / f"{file_path.stem}_ready.json"
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(embedded_chunks, out_f, indent=2, ensure_ascii=False)
        
    print(f"-> Generated representations for {len(embedded_chunks)} chunks -> Saved to {output_path.name}\n")

print("Local Embedding extraction complete.")