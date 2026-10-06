"""
Build a FAISS vector index from the EU policy document.
Chunks the document and embeds each chunk for retrieval.
"""
import os
import json
import numpy as np

def chunk_document(filepath, chunk_size=300, overlap=50):
    """Split document into overlapping chunks by words."""
    with open(filepath) as f:
        text = f.read()

    # Clean up: remove excessive whitespace, keep paragraph structure
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    full_text = " ".join(lines)
    words = full_text.split()

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def embed_chunks(chunks, model_name="BAAI/bge-small-en-v1.5"):
    """Embed chunks using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name, device="cpu")

    print(f"Embedding {len(chunks)} chunks...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    return embeddings


def build_faiss_index(embeddings):
    """Build a FAISS index from embeddings."""
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim since normalized)
    index.add(embeddings.astype(np.float32))
    return index


def main():
    DOCUMENT_PATH = "euaiact"
    INDEX_DIR = "./rag_index"
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Chunk the document
    print("Chunking document...")
    chunks = chunk_document(DOCUMENT_PATH, chunk_size=150, overlap=40)
    print(f"Created {len(chunks)} chunks")

    # Embed
    embeddings = embed_chunks(chunks)

    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    # Save everything
    import faiss
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    print(f"Saved index ({len(chunks)} chunks) to {INDEX_DIR}/")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Quick test
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    query = "What are the fines for prohibited AI practices?"
    q_emb = embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(q_emb, 3)
    print(f"\nTest query: '{query}'")
    print(f"Top 3 chunks:")
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        print(f"  {i+1}. (score={score:.3f}) {chunks[idx][:100]}...")


if __name__ == "__main__":
    main()
