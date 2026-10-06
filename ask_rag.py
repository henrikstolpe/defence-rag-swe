"""
RAG-augmented inference using the fine-tuned Llama 3.2 model.
Retrieves relevant context from the EU AI Act document before generating answers.
Uses hybrid search (BM25 + vector) and two-stage classify-then-answer.
"""
import os
import json
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./eupolicy_rag_llama_model"
INDEX_DIR = "./rag_index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 15  # Retrieve candidates for hybrid reranking
FINAL_K = 3  # Final number of chunks to pass to LLM

# ============================================================
# Load RAG components
# ============================================================
print("Loading RAG index...")
import faiss
from sentence_transformers import SentenceTransformer

index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
with open(os.path.join(INDEX_DIR, "chunks.json")) as f:
    chunks = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL)

# Load cross-encoder for reranking
print("Loading cross-encoder reranker...")
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

# Build BM25 index
print("Building BM25 index...")
import re
from collections import Counter

def normalize_numbers(text):
    """Normalize spaced numbers (e.g., '35 000 000' -> '35000000') for better BM25 matching."""
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    return text

def tokenize_bm25(text):
    """Whitespace + lowercase tokenization with number normalization for BM25."""
    text = normalize_numbers(text)
    return [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split() if len(w) > 2]

# Precompute document frequencies and term frequencies
corpus_tokens = [tokenize_bm25(chunk) for chunk in chunks]
doc_count = len(chunks)
doc_lengths = [len(tokens) for tokens in corpus_tokens]
avg_doc_length = sum(doc_lengths) / doc_count

# Document frequency for each term
df = Counter()
for tokens in corpus_tokens:
    for term in set(tokens):
        df[term] += 1

# Term frequency per document
tf_per_doc = []
for tokens in corpus_tokens:
    tf_per_doc.append(Counter(tokens))

def bm25_score(query_tokens, doc_idx, k1=1.5, b=0.75):
    """Compute BM25 score for a single document."""
    score = 0.0
    doc_len = doc_lengths[doc_idx]
    tf = tf_per_doc[doc_idx]
    for term in query_tokens:
        if term not in tf:
            continue
        term_freq = tf[term]
        doc_freq = df.get(term, 0)
        if doc_freq == 0:
            continue
        idf = math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        tf_norm = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_len / avg_doc_length))
        score += idf * tf_norm
    return score

print(f"Index loaded: {len(chunks)} chunks (vector + BM25)")

# ============================================================
# Load LLM
# ============================================================
print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
model.eval()
print("Ready. Ask anything about the EU AI Act (type 'quit' to exit).\n")

# ============================================================
# System prompt with explicit refusal rules
# ============================================================
SYSTEM_PROMPT = (
    "You are an EU AI Act expert. Answer the question based ONLY on the provided context. "
    "If the context does not contain enough information to answer, say so. "
    "If the question asks to compare with something not mentioned in the context, refuse. "
    "You have NO knowledge beyond what is in the context. If you find yourself about to state "
    "a fact not present in the context, stop and refuse instead. "
    "Be concise: 2-3 sentences maximum."
)

# ============================================================
# RAG pipeline
# ============================================================
def expand_query(question):
    """Generate query expansions for better retrieval coverage."""
    expansions = [question]
    # Simple rule-based expansions
    q = question.lower()
    if q.startswith("what is ") or q.startswith("what are "):
        expansions.append(question.replace("What is ", "Define ").replace("What are ", "Define "))
        expansions.append(question.replace("What is ", "Explain ").replace("What are ", "Explain "))
    elif q.startswith("how "):
        expansions.append(question.replace("How ", "In what way "))
    elif q.startswith("does ") or q.startswith("do "):
        expansions.append(question.replace("Does ", "").replace("Do ", "").rstrip("?"))
    elif q.startswith("are "):
        expansions.append(question.replace("Are ", "").rstrip("?") + " rules")
    elif "what does" in q and "say about" in q:
        topic = question.split("say about")[-1].strip().rstrip("?")
        expansions.append(f"rules regarding {topic}")
        expansions.append(f"{topic} requirements obligations")
    else:
        # Generic: add keywords
        expansions.append(question.rstrip("?") + " regulation requirements")
    return expansions[:3]


def retrieve(query, top_k=TOP_K, final_k=FINAL_K):
    """Hybrid retrieval with query expansion and cross-encoder reranking."""
    # Expand query for better coverage
    queries = expand_query(query)

    # Collect candidates from all query expansions
    all_candidates = {}

    for q in queries:
        # Vector search
        q_emb = embed_model.encode([q], normalize_embeddings=True).astype(np.float32)
        vec_scores, vec_indices = index.search(q_emb, top_k)

        # BM25 search
        query_tokens = tokenize_bm25(q)
        bm25_all = [(i, bm25_score(query_tokens, i)) for i in range(doc_count)]
        bm25_all.sort(key=lambda x: x[1], reverse=True)
        bm25_top = bm25_all[:top_k]

        # Merge into candidates
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

    # Sort by initial score and take top candidates for reranking
    ranked = [{"text": chunks[idx], "score": score, "idx": idx} for idx, score in all_candidates.items()]
    ranked.sort(key=lambda x: x["score"], reverse=True)

    # Cross-encoder reranking on top candidates
    rerank_candidates = ranked[:top_k]
    if rerank_candidates:
        pairs = [[query, c["text"]] for c in rerank_candidates]
        rerank_scores = reranker.predict(pairs)
        for i, score in enumerate(rerank_scores):
            rerank_candidates[i]["score"] = float(score)
        rerank_candidates.sort(key=lambda x: x["score"], reverse=True)

    return rerank_candidates[:final_k]


def classify_answerable(question, context_text):
    """Stage 1: Classify using LLM with calibrated prompt."""
    classify_prompt = (
        "You are a classification system. Given a context and a question, determine if the context "
        "contains information relevant to answering the question. Respond with ONLY 'YES' or 'NO'.\n\n"
        "Rules:\n"
        "- Answer NO if the question asks to COMPARE, CONTRAST, or find DIFFERENCES with another country's laws, another regulation, or another company's approach.\n"
        "- Answer NO if the question mentions a specific company (OpenAI, Google, Meta, etc.) that is not named in the context.\n"
        "- Answer NO if the question asks about real-world statistics, budgets, or counts not stated in the context.\n"
        "- Answer YES if the context discusses the topic the question asks about, even if it only partially covers it.\n"
        "- Answer YES if the question asks 'What does the AI Act say about X?' and the context mentions X."
    )
    messages = [
        {"role": "system", "content": classify_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}\n\nCan this question be answered from the context? (YES/NO)"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response


def generate(question, context):
    """Stage 2: Generate an answer given a question and retrieved context."""
    context_text = "\n\n".join([f"[Source {i+1}, relevance: {c['score']:.2f}]: {c['text']}" for i, c in enumerate(context)])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def ask(question):
    """Full RAG pipeline: retrieve, classify, then generate."""
    context = retrieve(question)
    context_text = "\n\n".join([f"[Source {i+1}, relevance: {c['score']:.2f}]: {c['text']}" for i, c in enumerate(context)])

    # Stage 1: Classify if answerable
    is_answerable = classify_answerable(question, context_text)

    if not is_answerable:
        return "The provided context does not contain enough information to answer this question.", context

    # Stage 2: Generate answer
    answer = generate(question, context)
    return answer, context


# ============================================================
# Interactive loop
# ============================================================
while True:
    try:
        question = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not question:
        continue
    if question.lower() in ("quit", "exit", "q"):
        print("Bye!")
        break

    answer, context = ask(question)
    print(f"AI: {answer}")
    print(f"    [Retrieved {len(context)} sources, top score: {context[0]['score']:.3f}]")
    print()
