"""
Tests for the shared RAG retrieval module.
Run: python test_rag_retrieval.py
"""
import os
import sys

# Only run if index exists
INDEX_DIR = "./rag_index_swe"
if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
    print("⚠️ RAG index not found — skipping retrieval tests (run build_rag_index_swe.py first)")
    sys.exit(0)

from rag_retrieval import RAGRetriever, get_retriever


def test_initialization():
    """Test that retriever loads without errors."""
    retriever = get_retriever()
    assert retriever is not None
    assert len(retriever.chunks) > 100
    assert len(retriever.sources) == len(retriever.chunks)
    assert retriever.doc_count == len(retriever.chunks)
    print("✅ test_initialization PASSED")


def test_retrieve_basic():
    """Test basic retrieval returns results."""
    retriever = get_retriever()
    results = retriever.retrieve("Vad är totalförsvar?")
    assert len(results) == 3
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)
    assert all("idx" in r for r in results)
    assert all("source" in r for r in results)
    print("✅ test_retrieve_basic PASSED")


def test_retrieve_relevance():
    """Test that retrieved chunks are actually relevant."""
    retriever = get_retriever()
    results = retriever.retrieve("Vilken roll har drönare i markstriden?")
    combined_text = " ".join(r["text"].lower() for r in results)
    # At least one chunk should mention drones/UAV
    assert "drönare" in combined_text or "obemannad" in combined_text or "uav" in combined_text, \
        f"No drone-related content found in results"
    print("✅ test_retrieve_relevance PASSED")


def test_retrieve_top_k():
    """Test different top_k values."""
    retriever = get_retriever()
    results_3 = retriever.retrieve("logistik", final_k=3)
    results_5 = retriever.retrieve("logistik", final_k=5)
    assert len(results_3) == 3
    assert len(results_5) == 5
    print("✅ test_retrieve_top_k PASSED")


def test_retrieve_scores_sorted():
    """Test that results are sorted by score (descending)."""
    retriever = get_retriever()
    results = retriever.retrieve("artilleri eldunderstöd")
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), f"Scores not sorted: {scores}"
    print("✅ test_retrieve_scores_sorted PASSED")


def test_expand_query():
    """Test Swedish query expansion."""
    retriever = get_retriever()
    
    expansions = retriever.expand_query("Vad är totalförsvar?")
    assert len(expansions) >= 2
    assert "Vad är totalförsvar?" in expansions
    
    expansions = retriever.expand_query("Hur organiseras CBRN?")
    assert len(expansions) >= 2
    assert any("sätt" in e.lower() for e in expansions)
    
    print("✅ test_expand_query PASSED")


def test_retrieve_multi_query():
    """Test multi-query retrieval for doctrine topics."""
    retriever = get_retriever()
    queries = [
        "fördröjningsstrid fältarbeten",
        "minering brosprängning",
        "strid bebyggelse urban",
    ]
    results = retriever.retrieve_multi_query(queries, final_k=5)
    assert len(results) >= 3  # At least 3 results (may be fewer if deduplication by source)
    assert len(results) <= 5
    assert all("text" in r for r in results)
    assert all("source" in r for r in results)
    print("✅ test_retrieve_multi_query PASSED")


def test_retrieve_multi_query_dedup():
    """Test that deduplication by source works."""
    retriever = get_retriever()
    queries = ["logistik underhåll", "logistik transport", "logistik försörjning"]
    results = retriever.retrieve_multi_query(queries, final_k=5, deduplicate_by_source=True)
    sources = [r["source"] for r in results]
    # All sources should be unique
    assert len(sources) == len(set(sources)), f"Duplicate sources found: {sources}"
    print("✅ test_retrieve_multi_query_dedup PASSED")


def test_singleton():
    """Test that get_retriever returns the same instance."""
    r1 = get_retriever()
    r2 = get_retriever()
    assert r1 is r2
    print("✅ test_singleton PASSED")


def test_bm25_number_normalization():
    """Test that numbers like '35 000 000' are normalized."""
    text = RAGRetriever._normalize_numbers("böter på 35 000 000 euro")
    assert "35000000" in text
    print("✅ test_bm25_number_normalization PASSED")


def test_tokenize_stopwords():
    """Test that Swedish stopwords are filtered."""
    tokens = RAGRetriever._tokenize_bm25("denna är ett test för att kontrollera")
    assert "denna" not in tokens
    assert "för" not in tokens
    assert "att" not in tokens
    assert "test" in tokens
    assert "kontrollera" in tokens
    print("✅ test_tokenize_stopwords PASSED")


if __name__ == "__main__":
    print("Running RAG retrieval tests...\n")

    test_initialization()
    test_retrieve_basic()
    test_retrieve_relevance()
    test_retrieve_top_k()
    test_retrieve_scores_sorted()
    test_expand_query()
    test_retrieve_multi_query()
    test_retrieve_multi_query_dedup()
    test_singleton()
    test_bm25_number_normalization()
    test_tokenize_stopwords()

    print(f"\n{'='*50}")
    print("ALL 11 TESTS PASSED ✅")
    print(f"{'='*50}")
