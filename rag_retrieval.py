"""
Shared RAG Retrieval Module.

Single implementation of hybrid search (vector + BM25 + cross-encoder reranking)
used by all components: Q&A, simulation, evaluation, and scenario parsing.

Usage:
    from rag_retrieval import RAGRetriever
    retriever = RAGRetriever()
    results = retriever.retrieve("Vad är totalförsvar?")
    # Returns: [{"text": "...", "score": 0.85, "idx": 123, "source": "..."}]
"""
import os
import json
import math
import re
import numpy as np
from collections import Counter


# Swedish stopwords for BM25 filtering
SWEDISH_STOPWORDS = {
    "och", "att", "det", "som", "för", "den", "med", "har", "inte", "till",
    "ett", "var", "från", "kan", "ska", "vid", "eller", "om", "av", "på",
    "är", "de", "en", "denna", "dessa", "sin", "sina", "sitt", "hans",
    "hennes", "deras", "vår", "våra", "era", "ert", "sig", "dem", "dig",
    "mig", "oss", "han", "hon", "vi", "ni", "jag", "du", "man", "där",
    "här", "när", "hur", "vad", "vilken", "vilket", "vilka", "alla",
    "andra", "efter", "genom", "hade", "har", "inte", "utan", "under",
    "över", "mellan", "mot", "redan", "sedan", "bara", "också", "även",
    "bör", "ska", "skulle", "kommer", "blev", "bli", "blir", "vara",
    "varit", "enligt", "inom", "samt", "såsom", "dock", "dels",
}


class RAGRetriever:
    """
    Hybrid RAG retriever combining vector search, BM25, and cross-encoder reranking.
    
    Loads the FAISS index, embedding model, and cross-encoder once.
    All retrieval operations go through this single class.
    """

    def __init__(self, index_dir: str = "./rag_index_swe", embed_model_name: str = "BAAI/bge-m3",
                 embed_device: str = "cpu"):
        """
        Initialize retriever.

        Args:
            index_dir: Path to the FAISS index directory (contains index.faiss, chunks.json, sources.json)
            embed_model_name: Sentence transformer model for embeddings
            embed_device: Device for embedding model ("cpu" recommended to leave GPU for LLM)
        """
        import faiss
        from sentence_transformers import SentenceTransformer, CrossEncoder

        print(f"[RAGRetriever] Loading index from {index_dir}...")
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))

        with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
            self.chunks = json.load(f)

        sources_path = os.path.join(index_dir, "sources.json")
        if os.path.exists(sources_path):
            with open(sources_path, encoding="utf-8") as f:
                self.sources = json.load(f)
        else:
            self.sources = ["unknown"] * len(self.chunks)

        print(f"[RAGRetriever] Loading embedding model: {embed_model_name}")
        self.embed_model = SentenceTransformer(embed_model_name, device=embed_device)

        print(f"[RAGRetriever] Loading cross-encoder reranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

        # Build BM25 index
        self._build_bm25()

        print(f"[RAGRetriever] Ready. {len(self.chunks)} chunks indexed.")

    def _build_bm25(self):
        """Pre-compute BM25 statistics."""
        self.corpus_tokens = [self._tokenize_bm25(chunk) for chunk in self.chunks]
        self.doc_count = len(self.chunks)
        self.doc_lengths = [len(tokens) for tokens in self.corpus_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 1

        self.df = Counter()
        for tokens in self.corpus_tokens:
            for term in set(tokens):
                self.df[term] += 1

        self.tf_per_doc = [Counter(tokens) for tokens in self.corpus_tokens]

    @staticmethod
    def _normalize_numbers(text: str) -> str:
        """Normalize spaced numbers for better BM25 matching."""
        text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
        text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
        return text

    @staticmethod
    def _tokenize_bm25(text: str) -> list[str]:
        """Whitespace + lowercase tokenization with Swedish stopword removal."""
        text = RAGRetriever._normalize_numbers(text)
        tokens = [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split() if len(w) > 2]
        return [t for t in tokens if t not in SWEDISH_STOPWORDS]

    def _bm25_score(self, query_tokens: list[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute BM25 score for a single document."""
        score = 0.0
        doc_len = self.doc_lengths[doc_idx]
        tf = self.tf_per_doc[doc_idx]
        for term in query_tokens:
            if term not in tf:
                continue
            term_freq = tf[term]
            doc_freq = self.df.get(term, 0)
            if doc_freq == 0:
                continue
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
            tf_norm = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_len / self.avg_doc_length))
            score += idf * tf_norm
        return score

    def expand_query(self, question: str) -> list[str]:
        """Generate Swedish query expansions for better retrieval coverage."""
        expansions = [question]
        q = question.lower()
        if q.startswith("vad är ") or q.startswith("vad innebär "):
            expansions.append(question.replace("Vad är ", "Definiera ").replace("Vad innebär ", "Förklara "))
            expansions.append(question.replace("Vad är ", "Förklara ").replace("Vad innebär ", "Definiera "))
        elif q.startswith("vilka är ") or q.startswith("vilka "):
            expansions.append(question.replace("Vilka är ", "Beskriv ").replace("Vilka ", "Beskriv "))
        elif q.startswith("hur "):
            expansions.append(question.replace("Hur ", "På vilket sätt "))
        elif q.startswith("är "):
            expansions.append(question.replace("Är ", "").rstrip("?") + " regler")
        elif "vad säger" in q and "om" in q:
            topic = question.split("om")[-1].strip().rstrip("?")
            expansions.append(f"regler angående {topic}")
            expansions.append(f"{topic} krav skyldigheter")
        else:
            expansions.append(question.rstrip("?") + " doktrin metod")
        return expansions[:3]

    def retrieve(self, query: str, top_k: int = 15, final_k: int = 3) -> list[dict]:
        """
        Hybrid retrieval with query expansion and cross-encoder reranking.

        Args:
            query: The search query (Swedish)
            top_k: Number of candidates for reranking
            final_k: Number of final results to return

        Returns:
            List of dicts with keys: text, score, idx, source
        """
        queries = self.expand_query(query)
        all_candidates = {}

        for q in queries:
            # Vector search
            q_emb = self.embed_model.encode([q], normalize_embeddings=True).astype(np.float32)
            vec_scores, vec_indices = self.index.search(q_emb, top_k)

            # BM25 search
            query_tokens = self._tokenize_bm25(q)
            bm25_all = [(i, self._bm25_score(query_tokens, i)) for i in range(self.doc_count)]
            bm25_all.sort(key=lambda x: x[1], reverse=True)
            bm25_top = bm25_all[:top_k]

            # Merge: 60% vector + 40% BM25
            vec_max = max(vec_scores[0]) if max(vec_scores[0]) > 0 else 1.0
            for score, idx in zip(vec_scores[0], vec_indices[0]):
                idx = int(idx)
                norm_score = float(score) / vec_max
                if idx not in all_candidates or norm_score > all_candidates[idx]:
                    all_candidates[idx] = norm_score

            bm25_max = bm25_top[0][1] if bm25_top[0][1] > 0 else 1.0
            for idx, score in bm25_top:
                norm_score = score / bm25_max
                combined = 0.6 * all_candidates.get(idx, 0.0) + 0.4 * norm_score
                if idx not in all_candidates or combined > all_candidates[idx]:
                    all_candidates[idx] = combined

        # Sort and take top candidates for reranking
        ranked = [{"text": self.chunks[idx], "score": score, "idx": idx, "source": self.sources[idx]}
                  for idx, score in all_candidates.items()]
        # Secondary key on idx makes tie-breaking deterministic
        ranked.sort(key=lambda x: (x["score"], -x["idx"]), reverse=True)

        # Cross-encoder reranking
        rerank_candidates = ranked[:top_k]
        if rerank_candidates:
            pairs = [[query, c["text"]] for c in rerank_candidates]
            rerank_scores = self.reranker.predict(pairs)
            for i, score in enumerate(rerank_scores):
                rerank_candidates[i]["score"] = float(score)
            # Secondary key on idx makes tie-breaking deterministic
            rerank_candidates.sort(key=lambda x: (x["score"], -x["idx"]), reverse=True)

        return rerank_candidates[:final_k]

    def retrieve_multi_query(self, queries: list[str], top_k_per_query: int = 8,
                             final_k: int = 10, deduplicate_by_source: bool = True) -> list[dict]:
        """
        Retrieve using multiple queries, merge results, optionally deduplicate by source.

        Useful for simulation doctrine retrieval where multiple topics are needed.

        Args:
            queries: List of search queries
            top_k_per_query: Candidates per query for vector search
            final_k: Total results to return
            deduplicate_by_source: If True, return max 1 chunk per source document

        Returns:
            List of dicts with keys: text, score, idx, source
        """
        all_candidates = {}

        for q in queries:
            q_emb = self.embed_model.encode([q], normalize_embeddings=True).astype(np.float32)
            scores, indices = self.index.search(q_emb, top_k_per_query)
            for score, idx in zip(scores[0], indices[0]):
                idx = int(idx)
                if idx not in all_candidates or float(score) > all_candidates[idx]:
                    all_candidates[idx] = float(score)

        # Sort candidates (secondary key on idx for deterministic tie-breaking)
        ranked = sorted(all_candidates.items(), key=lambda x: (x[1], -x[0]), reverse=True)[:final_k * 3]

        # Rerank against composite query
        composite_query = " ".join(queries[:3])
        pairs = [[composite_query, self.chunks[idx]] for idx, _ in ranked]
        rerank_scores = self.reranker.predict(pairs)

        # Secondary key on chunk idx makes tie-breaking deterministic
        reranked = sorted(zip(ranked, rerank_scores), key=lambda x: (x[1], -x[0][0]), reverse=True)

        # Build results, optionally deduplicating by source
        if deduplicate_by_source:
            seen_sources = set()
            results = []
            for (idx, _), score in reranked:
                source = self.sources[idx]
                if source not in seen_sources and len(results) < final_k:
                    seen_sources.add(source)
                    results.append({"text": self.chunks[idx], "score": float(score),
                                    "idx": idx, "source": source})
        else:
            results = [{"text": self.chunks[idx], "score": float(score), "idx": idx, "source": self.sources[idx]}
                       for (idx, _), score in reranked[:final_k]]

        return results


# ============================================================
# Module-level singleton for convenience
# ============================================================
_instance = None


def get_retriever(index_dir: str = "./rag_index_swe", embed_model: str = "BAAI/bge-m3") -> RAGRetriever:
    """Get or create a singleton RAGRetriever instance."""
    global _instance
    if _instance is None:
        _instance = RAGRetriever(index_dir=index_dir, embed_model_name=embed_model)
    return _instance
