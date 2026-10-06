"""Test a specific question against the RAG system."""
import os
import json
import math
import re
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = "./eupolicy_rag_llama_model_swe"
INDEX_DIR = "./rag_index_swe"
EMBED_MODEL = "BAAI/bge-m3"
BASE_MODEL_ID = "AI-Sweden-Models/Llama-3-8B-instruct"
TOP_K = 15
FINAL_K = 3

# Load RAG components
print("Loading RAG components...")
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
    chunks = json.load(f)
with open(os.path.join(INDEX_DIR, "sources.json"), encoding="utf-8") as f:
    sources = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

# BM25
SWEDISH_STOPWORDS = {
    "och", "att", "det", "som", "för", "den", "med", "har", "inte", "till",
    "ett", "var", "från", "kan", "ska", "vid", "eller", "om", "av", "på",
    "är", "de", "en", "denna", "dessa", "sin", "sina", "sitt", "sig",
    "bör", "ska", "skulle", "enligt", "inom", "samt", "såsom", "dock",
}

def normalize_numbers(text):
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    return text

def tokenize_bm25(text):
    text = normalize_numbers(text)
    tokens = [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split() if len(w) > 2]
    return [t for t in tokens if t not in SWEDISH_STOPWORDS]

corpus_tokens = [tokenize_bm25(chunk) for chunk in chunks]
doc_count = len(chunks)
doc_lengths = [len(tokens) for tokens in corpus_tokens]
avg_doc_length = sum(doc_lengths) / doc_count

df = Counter()
for tokens in corpus_tokens:
    for term in set(tokens):
        df[term] += 1
tf_per_doc = [Counter(tokens) for tokens in corpus_tokens]

def bm25_score(query_tokens, doc_idx, k1=1.5, b=0.75):
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

def expand_query(question):
    expansions = [question]
    q = question.lower()
    if q.startswith("vad är ") or q.startswith("vad innebär "):
        expansions.append(question.replace("Vad är ", "Definiera ").replace("Vad innebär ", "Förklara "))
    elif "vad säger" in q and "om" in q:
        topic = question.split("om")[-1].strip().rstrip("?")
        expansions.append(f"regler angående {topic}")
    else:
        expansions.append(question.rstrip("?") + " doktrin metod")
    return expansions[:3]

def retrieve(query, top_k=TOP_K, final_k=FINAL_K):
    queries = expand_query(query)
    all_candidates = {}
    for q in queries:
        q_emb = embed_model.encode([q], normalize_embeddings=True).astype(np.float32)
        vec_scores, vec_indices = index.search(q_emb, top_k)
        query_tokens = tokenize_bm25(q)
        bm25_all = [(i, bm25_score(query_tokens, i)) for i in range(doc_count)]
        bm25_all.sort(key=lambda x: x[1], reverse=True)
        bm25_top = bm25_all[:top_k]

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

    ranked = [{"text": chunks[idx], "score": score, "idx": idx, "source": sources[idx]} for idx, score in all_candidates.items()]
    ranked.sort(key=lambda x: x["score"], reverse=True)

    rerank_candidates = ranked[:top_k]
    if rerank_candidates:
        pairs = [[query, c["text"]] for c in rerank_candidates]
        rerank_scores = reranker.predict(pairs)
        for i, score in enumerate(rerank_scores):
            rerank_candidates[i]["score"] = float(score)
        rerank_candidates.sort(key=lambda x: x["score"], reverse=True)

    return rerank_candidates[:final_k]

# Load LLM
print("Loading LLM...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=quantization_config,
    device_map={"": 0}, torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

SYSTEM_PROMPT = (
    "Du är en expert på svenska försvarsfrågor och militär doktrin. "
    "Besvara frågan ENBART baserat på den tillhandahållna kontexten. "
    "Om kontexten inte innehåller tillräcklig information för att svara, säg det. "
    "Om frågan ber om jämförelse med något som inte nämns i kontexten, vägra. "
    "Du har INGEN kunskap utöver det som finns i kontexten. Om du är på väg att ange "
    "ett faktum som inte finns i kontexten, stanna och vägra istället. "
    "Ge utförliga och detaljerade svar. Använd punktlistor och struktur där det är lämpligt."
)

def classify_answerable(question, context_text):
    classify_prompt = (
        "Du är ett klassificeringssystem. Givet en kontext och en fråga, avgör om kontexten "
        "innehåller information relevant för att besvara frågan. Svara med ENBART 'YES' eller 'NO'.\n\n"
        "Regler:\n"
        "- Svara NO ENBART om frågan ber om att JÄMFÖRA med ett annat lands strategi som INTE nämns i kontexten.\n"
        "- Svara NO ENBART om frågan nämner ett specifikt företag som INTE nämns i kontexten.\n"
        "- Svara NO ENBART om frågan frågar om specifik statistik som INTE anges i kontexten.\n"
        "- Svara YES om kontexten diskuterar ämnet som frågan handlar om.\n"
        "- Svara YES om kontexten innehåller relevant information.\n"
        "- Vid tveksamhet, svara YES."
    )
    messages = [
        {"role": "system", "content": classify_prompt},
        {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}\n\nKan denna fråga besvaras utifrån kontexten? (YES/NO)"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response or "JA" in response

def generate(question, context_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=600, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# === ASK THE QUESTION ===
import sys

# Parse args
use_claude = "--claude" in sys.argv
args = [a for a in sys.argv[1:] if a != "--claude"]

if args:
    question = " ".join(args)
else:
    question = input("\nFråga: ").strip()

print(f"\n{'='*70}")
print(f"FRÅGA: {question}")
print(f"MODELL: {'Claude (Anthropic API)' if use_claude else 'AI-Sweden Llama 3 8B (lokal)'}")
print(f"{'='*70}")

# Retrieve
context = retrieve(question)
print(f"\nHämtade {len(context)} källor:")
for i, c in enumerate(context):
    print(f"  {i+1}. [score={c['score']:.3f}] Källa: {c['source']}")
    print(f"     {c['text'][:150]}...")

# Build context string
context_text = "\n\n".join([f"[Källa {i+1}, relevans: {c['score']:.2f}]: {c['text']}" for i, c in enumerate(context)])

# Classify (always local)
is_answerable = classify_answerable(question, context_text)
print(f"\nKlassificering: {'JA - kan besvaras' if is_answerable else 'NEJ - kan inte besvaras'}")

# Generate
if not is_answerable:
    answer = "Den tillhandahållna kontexten innehåller inte tillräcklig information för att besvara denna fråga."
elif use_claude:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        temperature=0.0,  # Deterministic sampling for reproducible answers
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"}
        ],
    )
    answer = message.content[0].text
else:
    answer = generate(question, context_text)

print(f"\nSVAR:\n{answer}")
print(f"\n{'='*70}")
