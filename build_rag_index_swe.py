"""
Build a FAISS vector index from all Swedish documents in the markdown_output folder.
Chunks each document and embeds all chunks for retrieval.
Uses a multilingual embedding model for Swedish text.
"""
import os
import json
import numpy as np


def chunk_document(filepath, chunk_size=150, overlap=40):
    """Split a document into overlapping chunks by words. Prefixes each chunk with source filename."""
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    # Clean up: remove excessive whitespace, keep paragraph structure
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_text = " ".join(lines)
    words = full_text.split()

    # Skip very short documents (< 50 words of content)
    if len(words) < 50:
        return []

    source_name = os.path.splitext(os.path.basename(filepath))[0]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        start += chunk_size - overlap

    return chunks


def load_all_documents(folder_path):
    """Load and chunk all markdown files in the folder."""
    all_chunks = []
    all_sources = []

    md_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".md")])
    print(f"Found {len(md_files)} markdown files")

    for filename in md_files:
        filepath = os.path.join(folder_path, filename)
        chunks = chunk_document(filepath, chunk_size=150, overlap=40)
        source_name = os.path.splitext(filename)[0]
        for chunk in chunks:
            all_chunks.append(chunk)
            all_sources.append(source_name)

    return all_chunks, all_sources


def embed_chunks(chunks, model_name="BAAI/bge-m3"):
    """Embed chunks using a multilingual sentence-transformer on GPU."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading multilingual embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name, device="cuda")

    print(f"Embedding {len(chunks)} chunks on GPU...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True, normalize_embeddings=True, batch_size=64)
    return embeddings


def build_faiss_index(embeddings):
    """Build a FAISS index from embeddings."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since normalized)
    index.add(embeddings.astype(np.float32))
    return index


def main():
    DOCUMENT_FOLDER = "/home/henrikstolpe/Documents/The Future of Warfare/markdown_output"
    INDEX_DIR = "./rag_index_swe"
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load and chunk all documents
    print("Loading and chunking all Swedish documents...")
    chunks, sources = load_all_documents(DOCUMENT_FOLDER)
    print(f"Created {len(chunks)} chunks from {len(set(sources))} documents")

    # Embed with multilingual model
    embeddings = embed_chunks(chunks)

    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save everything
    import faiss
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(os.path.join(INDEX_DIR, "sources.json"), "w", encoding="utf-8") as f:
        json.dump(sources, f, ensure_ascii=False)

    print(f"\nSaved index ({len(chunks)} chunks) to {INDEX_DIR}/")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Documents indexed: {len(set(sources))}")

    # Quick test with a Swedish query
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    query = "Vad är drönarkrigföring?"
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, 3)
    print(f"\nTestfråga: '{query}'")
    print(f"Topp 3 chunks:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  {i+1}. (score={score:.3f}, källa={sources[idx]}) {chunks[idx][:100]}...")


if __name__ == "__main__":
    main()
