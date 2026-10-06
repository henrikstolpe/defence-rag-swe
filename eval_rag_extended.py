"""
Extended evaluation script — double the questions, all new.
Validates the system's accuracy with a fresh set of test cases.
"""
import os
import json
import math
import re
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./eupolicy_rag_llama_model"
INDEX_DIR = "./rag_index"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 15
FINAL_K = 3

# ============================================================
# Load components
# ============================================================
print("Loading RAG components...")
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

# BM25 setup
def normalize_numbers(text):
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    return text

def tokenize_bm25(text):
    text = normalize_numbers(text)
    return [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split() if len(w) > 2]

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

print("Loading classifier model (Llama 3.2 3B)...")
import gc

GENERATOR_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

clf_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
clf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
clf_model.eval()

gen_tokenizer = None
gen_model = None

def swap_to_generator():
    global clf_model, clf_tokenizer, gen_model, gen_tokenizer
    if gen_model is not None:
        return
    del clf_model
    clf_model = None
    gc.collect()
    torch.cuda.empty_cache()
    print("  Swapping to Mistral 7B-Instruct (4-bit)...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL,
        quantization_config=quantization_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )
    gen_model.eval()

print("Ready.\n")

# ============================================================
# Helper functions
# ============================================================
def expand_query(question):
    """Generate query expansions for better retrieval coverage."""
    expansions = [question]
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
        expansions.append(question.rstrip("?") + " regulation requirements")
    return expansions[:3]


def retrieve(query, top_k=TOP_K, final_k=FINAL_K):
    """Hybrid retrieval with query expansion and cross-encoder reranking."""
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

    ranked = [{"text": chunks[idx], "score": score, "idx": idx} for idx, score in all_candidates.items()]
    ranked.sort(key=lambda x: x["score"], reverse=True)

    rerank_candidates = ranked[:top_k]
    if rerank_candidates:
        pairs = [[query, c["text"]] for c in rerank_candidates]
        rerank_scores = reranker.predict(pairs)
        for i, score in enumerate(rerank_scores):
            rerank_candidates[i]["score"] = float(score)
        rerank_candidates.sort(key=lambda x: x["score"], reverse=True)

    return rerank_candidates[:final_k]


SYSTEM_PROMPT = (
    "You are an EU AI Act expert. Answer the question based ONLY on the provided context. "
    "If the context does not contain enough information to answer, say so. "
    "If the question asks to compare with something not mentioned in the context, refuse. "
    "You have NO knowledge beyond what is in the context. If you find yourself about to state "
    "a fact not present in the context, stop and refuse instead. "
    "Be concise: 2-3 sentences maximum."
)


def classify_answerable(question, context_text):
    """Stage 1: Classify using fine-tuned 3B LLM."""
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
    prompt = clf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = clf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(clf_model.device)
    with torch.no_grad():
        output = clf_model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=clf_tokenizer.pad_token_id)
    response = clf_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response


def generate(question, context_text):
    """Stage 2: Generate using Mistral 7B-Instruct."""
    swap_to_generator()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
    ]
    prompt = gen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(gen_model.device)
    with torch.no_grad():
        output = gen_model.generate(**inputs, max_new_tokens=150, do_sample=False, use_cache=True, pad_token_id=gen_tokenizer.pad_token_id)
    return gen_tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def generate_with_classification(question, context):
    context_text = "\n\n".join([f"[Source {i+1}, relevance: {c['score']:.2f}]: {c['text']}" for i, c in enumerate(context)])
    if not classify_answerable(question, context_text):
        return "The provided context does not contain enough information to answer this question."
    return generate(question, context_text)


def is_refusal(response):
    refusal_indicators = [
        "does not contain", "cannot find", "cannot be answered",
        "not enough information", "no information", "not mentioned",
        "no mention", "not provided", "not include", "don't have enough",
        "insufficient", "not available in", "not addressed", "not specified",
        "not directly", "cannot answer", "unable to answer",
    ]
    return any(ind in response.lower() for ind in refusal_indicators)


def find_chunk_containing(keywords):
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw.lower() in chunk.lower())
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk


def semantic_similarity(text_a, text_b):
    """Compute cosine similarity between two texts using the embedding model."""
    embeddings = embed_model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


# ============================================================
# TEST 1: Answerable accuracy (20 questions)
# ============================================================
answerable_tests = [
    {
        "question": "What is the definition of an AI system under the EU AI Act?",
        "context_keywords": ["infer", "machine-based", "autonomy", "predictions"],
        "expected_facts": ["infer", "machine"],
        "reference_answer": "An AI system is a machine-based system designed to operate with varying levels of autonomy, that can infer from inputs to generate outputs such as predictions, content, recommendations, or decisions.",
    },
    {
        "question": "What biometric practices are prohibited?",
        "context_keywords": ["biometric categorisation", "political opinions", "race", "sexual orientation"],
        "expected_facts": ["biometric", "prohibit"],
        "reference_answer": "Biometric categorisation systems that deduce or infer political opinions, trade union membership, religious beliefs, race, sex life or sexual orientation from biometric data are prohibited.",
    },
    {
        "question": "What is the penalty for non-compliance by operators or notified bodies?",
        "context_keywords": ["15 000 000", "3 %", "operators", "notified bodies"],
        "expected_facts": ["15", "3"],
        "reference_answer": "Non-compliance by operators or notified bodies is subject to administrative fines of up to 15 million euros or 3 percent of total worldwide annual turnover, whichever is higher.",
    },
    {
        "question": "What does the AI Act say about AI in employment and recruitment?",
        "context_keywords": ["employment", "recruitment", "workers", "high-risk"],
        "expected_facts": ["employment", "high-risk"],
        "reference_answer": "AI systems used in employment, recruitment, and workers management are classified as high-risk because they may have significant impact on career prospects and livelihoods and may perpetuate historical patterns of discrimination.",
    },
    {
        "question": "What is the role of conformity assessment?",
        "context_keywords": ["conformity assessment", "placing on the market", "notified bodies"],
        "expected_facts": ["conformity", "market"],
        "reference_answer": "Conformity assessment ensures high-risk AI systems meet the requirements of the Regulation before being placed on the market or put into service, carried out by the provider or by notified bodies.",
    },
    {
        "question": "What are the transparency obligations for AI systems that interact with people?",
        "context_keywords": ["interact", "natural persons", "notified", "transparency"],
        "expected_facts": ["notif", "interact"],
        "reference_answer": "Natural persons must be notified that they are interacting with an AI system, unless this is obvious from the circumstances and context of use.",
    },
    {
        "question": "What is the purpose of the EU database for high-risk AI systems?",
        "context_keywords": ["EU database", "register", "high-risk", "publicly accessible"],
        "expected_facts": ["database", "register"],
        "reference_answer": "The EU database requires providers of high-risk AI systems to register themselves and information about their systems, increasing transparency towards the public and facilitating oversight.",
    },
    {
        "question": "How does the AI Act address deep fakes?",
        "context_keywords": ["deep fakes", "artificially generated", "manipulated", "disclose"],
        "expected_facts": ["disclose", "artificial"],
        "reference_answer": "Deployers who use AI to generate or manipulate content that resembles existing persons or events (deep fakes) must clearly disclose that the content has been artificially created or manipulated.",
    },
    {
        "question": "What are the rules on AI in critical infrastructure?",
        "context_keywords": ["critical infrastructure", "safety component", "high-risk"],
        "expected_facts": ["critical", "infrastructure"],
        "reference_answer": "AI systems used as safety components in critical infrastructure management are classified as high-risk, including systems used in energy, transport, and water supply.",
    },
    {
        "question": "What is the scope of the AI Act regarding research and development?",
        "context_keywords": ["scientific research", "development", "excluded", "prior to"],
        "expected_facts": ["research", "exclud"],
        "reference_answer": "AI systems specifically developed and put into service for the sole purpose of scientific research and development are excluded from the scope of the Regulation.",
    },
    {
        "question": "What obligations do importers of AI systems have?",
        "context_keywords": ["importer", "conformity", "CE marking", "placing on the market"],
        "expected_facts": ["importer", "conformity"],
        "reference_answer": "Importers must ensure that high-risk AI systems bear the CE marking, have undergone conformity assessment, and are accompanied by required documentation before placing them on the market.",
    },
    {
        "question": "What is the role of national competent authorities?",
        "context_keywords": ["national competent authorities", "market surveillance", "supervision"],
        "expected_facts": ["national", "authorit"],
        "reference_answer": "National competent authorities are responsible for market surveillance and supervision of AI systems, ensuring compliance with the Regulation at the national level.",
    },
    {
        "question": "What does the AI Act say about AI literacy?",
        "context_keywords": ["AI literacy", "providers", "deployers", "informed decisions"],
        "expected_facts": ["literacy", "inform"],
        "reference_answer": "AI literacy should equip providers, deployers and affected persons with the necessary knowledge to make informed decisions regarding AI systems, understanding their application and limitations.",
    },
    {
        "question": "What are the data governance requirements for high-risk AI?",
        "context_keywords": ["data governance", "training data", "bias", "representative"],
        "expected_facts": ["data", "bias"],
        "reference_answer": "High-risk AI systems require appropriate data governance including training data that is relevant, sufficiently representative, and free of errors, with attention to mitigating possible biases.",
    },
    {
        "question": "What is the role of the European Artificial Intelligence Board?",
        "context_keywords": ["Board", "advise", "Commission", "consistent application"],
        "expected_facts": ["board", "commission"],
        "reference_answer": "The European Artificial Intelligence Board advises the Commission and contributes to the consistent application of the Regulation across Member States.",
    },
    {
        "question": "What are the rules on AI in migration and border control?",
        "context_keywords": ["migration", "asylum", "border control", "high-risk"],
        "expected_facts": ["migration", "high-risk"],
        "reference_answer": "AI systems used in migration, asylum and border control management are classified as high-risk, including systems used for risk assessment of persons entering the territory or applying for asylum.",
    },
    {
        "question": "What does the AI Act say about open-source AI models?",
        "context_keywords": ["open-source", "free", "licence", "general-purpose"],
        "expected_facts": ["open", "source"],
        "reference_answer": "Open-source general-purpose AI models have specific provisions, with some obligations not applying if the model is made available under a free and open-source licence.",
    },
    {
        "question": "What human oversight requirements exist for high-risk AI?",
        "context_keywords": ["human oversight", "understand", "interpret", "intervene"],
        "expected_facts": ["human", "oversight"],
        "reference_answer": "High-risk AI systems must be designed to allow effective human oversight, enabling humans to understand, interpret outputs, and intervene or interrupt the system when necessary.",
    },
    {
        "question": "What are the record-keeping requirements for high-risk AI systems?",
        "context_keywords": ["logs", "record", "automatic", "traceability"],
        "expected_facts": ["log", "record"],
        "reference_answer": "High-risk AI systems must technically allow for automatic recording of events by means of logs over the duration of their lifetime, enabling traceability and monitoring.",
    },
    {
        "question": "What does the AI Act say about AI used in the administration of justice?",
        "context_keywords": ["judicial", "justice", "high-risk", "law"],
        "expected_facts": ["judicial", "high-risk"],
        "reference_answer": "AI systems intended to assist judicial authorities in researching and interpreting facts and law are classified as high-risk, though the final decision-making must remain a human-driven activity.",
    },
]

# ============================================================
# TEST 2: Refusal accuracy (20 questions)
# ============================================================
refusal_tests = [
    {"question": "How many AI systems have been banned in the EU since the Act took effect?", "context_keywords": ["sandbox", "innovation"]},
    {"question": "What is the total cost of implementing the AI Act across all member states?", "context_keywords": ["biometric", "identification"]},
    {"question": "Which AI company was the first to receive CE marking under the AI Act?", "context_keywords": ["deployer", "natural or legal person"]},
    {"question": "How many people work in AI governance roles across the EU?", "context_keywords": ["conformity assessment", "notified bodies"]},
    {"question": "What is the EU's total investment in AI startups?", "context_keywords": ["transparency", "natural persons"]},
    {"question": "How does the AI Act compare to Australia's AI Ethics Principles?", "context_keywords": ["high-risk", "education", "vocational"]},
    {"question": "What are the differences between the AI Act and Israel's AI regulation?", "context_keywords": ["social scoring", "discriminatory"]},
    {"question": "How does the AI Act's approach differ from South Korea's AI Basic Act?", "context_keywords": ["sandbox", "testing", "development"]},
    {"question": "What are the similarities between the AI Act and Singapore's PDPA?", "context_keywords": ["biometric categorisation", "political opinions"]},
    {"question": "How does the AI Act compare to the IEEE standards for AI ethics?", "context_keywords": ["deployer", "natural or legal person"]},
    {"question": "What percentage of EU companies have completed AI Act compliance?", "context_keywords": ["CE marking", "conformity"]},
    {"question": "How many AI regulatory sandboxes have produced commercial products?", "context_keywords": ["emotion recognition", "workplace"]},
    {"question": "What is Anthropic's compliance strategy for the AI Act?", "context_keywords": ["general-purpose AI", "systemic risk"]},
    {"question": "How has the AI Act affected venture capital investment in EU AI?", "context_keywords": ["transparency", "natural persons"]},
    {"question": "What did the OECD say about the EU AI Act's effectiveness?", "context_keywords": ["sandbox", "innovation"]},
    {"question": "How does the AI Act interact with the EU Cyber Resilience Act in practice?", "context_keywords": ["high-risk", "education"]},
    {"question": "What is the average time to complete a conformity assessment?", "context_keywords": ["biometric", "identification"]},
    {"question": "How many complaints have been filed under the AI Act?", "context_keywords": ["deployer", "natural or legal person"]},
    {"question": "What is Apple's position on the AI Act's requirements for Siri?", "context_keywords": ["emotion recognition", "workplace"]},
    {"question": "How does the AI Act compare to Japan's Social Principles of Human-Centric AI?", "context_keywords": ["social scoring", "discriminatory"]},
]

# ============================================================
# TEST 3: False refusal rate (20 questions, full pipeline)
# ============================================================
false_refusal_tests = [
    "What does the AI Act say about social scoring?",
    "Are there rules about AI in education under the AI Act?",
    "What is a general-purpose AI model?",
    "What are the prohibited AI practices listed in the AI Act?",
    "What does the AI Act say about biometric identification?",
    "What is the conformity assessment process for high-risk AI?",
    "What are the obligations of AI system providers?",
    "What does the AI Act say about AI regulatory sandboxes?",
    "How does the AI Act define high-risk AI systems?",
    "What transparency requirements apply to AI systems?",
    "What are the rules on AI in law enforcement?",
    "What does the AI Act say about data quality for AI training?",
    "What is the role of market surveillance authorities?",
    "What are the penalties for non-compliance with the AI Act?",
    "What does the AI Act say about AI and fundamental rights?",
    "What are the requirements for technical documentation of AI systems?",
    "What does the AI Act say about post-market monitoring?",
    "How does the AI Act address AI systems in healthcare?",
    "What are the rules on AI in credit scoring?",
    "What does the AI Act say about codes of practice?",
]

# ============================================================
# TEST 4: Parametric leakage (10 questions)
# ============================================================
parametric_leakage_tests = [
    {"question": "What is Claude by Anthropic and is it regulated under the AI Act?", "context_keywords": ["sandbox", "innovation"]},
    {"question": "How did Sam Altman respond to the AI Act?", "context_keywords": ["biometric", "identification"]},
    {"question": "What is the carbon footprint of training GPT-4?", "context_keywords": ["deployer", "natural or legal person"]},
    {"question": "How many users does ChatGPT have in the EU?", "context_keywords": ["conformity assessment", "notified bodies"]},
    {"question": "What is Mistral AI's valuation and market position?", "context_keywords": ["transparency", "natural persons"]},
    {"question": "What did Elon Musk say about AI regulation in Europe?", "context_keywords": ["social scoring", "discriminatory"]},
    {"question": "How much did Google spend on AI safety research in 2024?", "context_keywords": ["sandbox", "testing"]},
    {"question": "What is the accuracy of facial recognition systems like Clearview AI?", "context_keywords": ["CE marking", "conformity"]},
    {"question": "How many parameters does Meta's Llama 3 have?", "context_keywords": ["emotion recognition", "workplace"]},
    {"question": "What is the EU's ranking in global AI competitiveness indices?", "context_keywords": ["high-risk", "education"]},
]

# ============================================================
# TEST 5: Retrieval quality (20 questions)
# ============================================================
retrieval_tests = [
    {"question": "What is the definition of a deployer?", "must_contain": ["deployer"]},
    {"question": "What are the prohibited AI practices?", "must_contain": ["prohibit"]},
    {"question": "What is a high-risk AI system?", "must_contain": ["high-risk"]},
    {"question": "What are AI regulatory sandboxes?", "must_contain": ["sandbox"]},
    {"question": "What are the rules on biometric identification?", "must_contain": ["biometric"]},
    {"question": "What is the role of the AI Office?", "must_contain": ["AI Office"]},
    {"question": "What are the transparency requirements?", "must_contain": ["transparency"]},
    {"question": "What are general-purpose AI models?", "must_contain": ["general-purpose"]},
    {"question": "What does the AI Act say about emotion recognition?", "must_contain": ["emotion"]},
    {"question": "What is the conformity assessment process?", "must_contain": ["conformity"]},
    {"question": "What are the data governance requirements?", "must_contain": ["data"]},
    {"question": "What is the role of notified bodies?", "must_contain": ["notified bod"]},
    {"question": "What does the AI Act say about social scoring?", "must_contain": ["social scoring"]},
    {"question": "What are the rules on AI in employment?", "must_contain": ["employment"]},
    {"question": "What does the AI Act say about deep fakes?", "must_contain": ["deep fake"]},
    {"question": "What is the EU database for AI systems?", "must_contain": ["database"]},
    {"question": "What are the human oversight requirements?", "must_contain": ["oversight"]},
    {"question": "What does the AI Act say about cybersecurity?", "must_contain": ["cybersecurity"]},
    {"question": "What are the rules on AI in migration?", "must_contain": ["migration"]},
    {"question": "What does the AI Act say about AI literacy?", "must_contain": ["literacy"]},
]

# Ground truth for precision/recall: tighter multi-keyword relevance criteria
retrieval_ground_truth = [
    {"question": "What is the definition of a deployer?", "relevant_keywords": ["deployer", "natural or legal person"]},
    {"question": "What are the prohibited AI practices?", "relevant_keywords": ["prohibit", "Article 5"]},
    {"question": "What is a high-risk AI system?", "relevant_keywords": ["high-risk", "safety component"]},
    {"question": "What are AI regulatory sandboxes?", "relevant_keywords": ["sandbox", "controlled"]},
    {"question": "What are the rules on biometric identification?", "relevant_keywords": ["biometric identification", "real-time"]},
    {"question": "What is the role of the AI Office?", "relevant_keywords": ["AI Office", "facilitate"]},
    {"question": "What are the transparency requirements?", "relevant_keywords": ["transparency", "deployer", "inform"]},
    {"question": "What are general-purpose AI models?", "relevant_keywords": ["general-purpose", "systemic risk"]},
    {"question": "What does the AI Act say about emotion recognition?", "relevant_keywords": ["emotion", "workplace"]},
    {"question": "What is the conformity assessment process?", "relevant_keywords": ["conformity assessment", "notified bod"]},
    {"question": "What are the data governance requirements?", "relevant_keywords": ["data governance", "bias"]},
    {"question": "What is the role of notified bodies?", "relevant_keywords": ["notified bod", "competence"]},
    {"question": "What does the AI Act say about social scoring?", "relevant_keywords": ["social scoring"]},
    {"question": "What are the rules on AI in employment?", "relevant_keywords": ["employment", "recruitment"]},
    {"question": "What does the AI Act say about deep fakes?", "relevant_keywords": ["deep fake"]},
    {"question": "What is the EU database for AI systems?", "relevant_keywords": ["database", "register"]},
    {"question": "What are the human oversight requirements?", "relevant_keywords": ["human oversight", "interpret"]},
    {"question": "What does the AI Act say about cybersecurity?", "relevant_keywords": ["cybersecurity", "protect"]},
    {"question": "What are the rules on AI in migration?", "relevant_keywords": ["migration", "asylum"]},
    {"question": "What does the AI Act say about AI literacy?", "relevant_keywords": ["literacy", "provider"]},
]


# ============================================================
# Run evaluation
# ============================================================
def run_eval():
    results = {
        "answerable": {"total": 0, "correct": 0, "details": []},
        "refusal": {"total": 0, "correct": 0, "details": []},
        "false_refusal": {"total": 0, "refused": 0, "details": []},
        "parametric_leakage": {"total": 0, "leaked": 0, "details": []},
        "retrieval": {"total": 0, "hit": 0, "details": []},
    }

    # =====================================================
    # PHASE 1: All classification (3B model loaded)
    # =====================================================
    print("=" * 60)
    print("PHASE 1: CLASSIFICATION (Llama 3.2 3B fine-tuned)")
    print("=" * 60)

    # Prepare all tasks that need classification
    tasks = []

    # Test 1: Answerable
    for test in answerable_tests:
        context = find_chunk_containing(test["context_keywords"])
        context_text = f"[Source 1, relevance: 0.75]: {context}"
        tasks.append({"type": "answerable", "question": test["question"], "context_text": context_text, "test": test})

    # Test 2: Refusal
    for test in refusal_tests:
        context = find_chunk_containing(test["context_keywords"])
        context_text = f"[Source 1, relevance: 0.50]: {context}"
        tasks.append({"type": "refusal", "question": test["question"], "context_text": context_text})

    # Test 3: False refusal (full pipeline)
    for question in false_refusal_tests:
        retrieved = retrieve(question)
        context_text = "\n\n".join([f"[Source {i+1}, relevance: {c['score']:.2f}]: {c['text']}" for i, c in enumerate(retrieved)])
        tasks.append({"type": "false_refusal", "question": question, "context_text": context_text, "retrieved": retrieved})

    # Test 4: Parametric leakage
    for test in parametric_leakage_tests:
        context = find_chunk_containing(test["context_keywords"])
        context_text = f"[Source 1, relevance: 0.50]: {context}"
        tasks.append({"type": "leakage", "question": test["question"], "context_text": context_text})

    # Classify all
    print(f"\n  Classifying {len(tasks)} questions...")
    for task in tasks:
        task["classified_yes"] = classify_answerable(task["question"], task["context_text"])
        label = "YES" if task["classified_yes"] else "NO"
        print(f"    [{label}] ({task['type']}) {task['question'][:60]}...")

    # =====================================================
    # PHASE 2: Generation (swap to Mistral 7B)
    # =====================================================
    print("\n" + "=" * 60)
    print("PHASE 2: GENERATION (Mistral 7B-Instruct, 4-bit)")
    print("=" * 60)
    swap_to_generator()

    # Process all tasks
    print("\n--- Test 1: Answerable Accuracy (20 questions) ---")
    semantic_scores = []
    for task in [t for t in tasks if t["type"] == "answerable"]:
        if not task["classified_yes"]:
            response = "The provided context does not contain enough information to answer this question."
        else:
            response = generate(task["question"], task["context_text"])
        response_lower = response.lower()
        test = task["test"]
        facts_found = [f for f in test["expected_facts"] if f.lower() in response_lower]
        keyword_pass = len(facts_found) >= len(test["expected_facts"]) // 2 + 1 and not is_refusal(response)

        # Semantic similarity with reference answer
        if "reference_answer" in test and not is_refusal(response):
            sim_score = semantic_similarity(response, test["reference_answer"])
        else:
            sim_score = 0.0
        semantic_scores.append(sim_score)

        # Pass if keyword match OR semantic similarity >= 0.7
        correct_semantic = sim_score >= 0.7 and not is_refusal(response)
        is_correct = keyword_pass or correct_semantic

        results["answerable"]["total"] += 1
        if is_correct:
            results["answerable"]["correct"] += 1
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] Q: {task['question']}")
        print(f"        A: {response[:100]}...")
        print(f"        Keywords: {len(facts_found)}/{len(test['expected_facts'])} | Semantic: {sim_score:.2f}")
        if not is_correct and is_refusal(response):
            print(f"        ** FALSE REFUSAL **")
        results["answerable"]["details"].append({"question": task["question"], "correct": is_correct, "semantic_score": sim_score})

    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    print(f"\n  Avg semantic similarity: {avg_semantic:.2f}")

    print("\n--- Test 2: Refusal Accuracy (20 questions) ---")
    for task in [t for t in tasks if t["type"] == "refusal"]:
        if not task["classified_yes"]:
            response = "The provided context does not contain enough information to answer this question."
        else:
            response = generate(task["question"], task["context_text"])
        refused = is_refusal(response)
        results["refusal"]["total"] += 1
        if refused:
            results["refusal"]["correct"] += 1
        status = "PASS" if refused else "FAIL"
        print(f"  [{status}] Q: {task['question']}")
        print(f"        A: {response[:100]}...")
        if not refused:
            print(f"        ** HALLUCINATION **")
        results["refusal"]["details"].append({"question": task["question"], "refused": refused})

    print("\n--- Test 3: False Refusal Rate (20 questions) ---")
    for task in [t for t in tasks if t["type"] == "false_refusal"]:
        if not task["classified_yes"]:
            response = "The provided context does not contain enough information to answer this question."
        else:
            response = generate(task["question"], task["context_text"])
        refused = is_refusal(response)
        results["false_refusal"]["total"] += 1
        if refused:
            results["false_refusal"]["refused"] += 1
        status = "FAIL" if refused else "PASS"
        print(f"  [{status}] Q: {task['question']}")
        print(f"        A: {response[:100]}...")
        if refused:
            print(f"        ** FALSE REFUSAL **")
        results["false_refusal"]["details"].append({"question": task["question"], "refused": refused})

    print("\n--- Test 4: Parametric Leakage (10 questions) ---")
    for task in [t for t in tasks if t["type"] == "leakage"]:
        if not task["classified_yes"]:
            response = "The provided context does not contain enough information to answer this question."
        else:
            response = generate(task["question"], task["context_text"])
        refused = is_refusal(response)
        leaked = not refused
        results["parametric_leakage"]["total"] += 1
        if leaked:
            results["parametric_leakage"]["leaked"] += 1
        status = "PASS" if refused else "FAIL"
        print(f"  [{status}] Q: {task['question']}")
        print(f"        A: {response[:100]}...")
        if leaked:
            print(f"        ** LEAKAGE **")
        results["parametric_leakage"]["details"].append({"question": task["question"], "leaked": leaked})

    # --- Test 5: Retrieval (no model needed) ---
    print("\n--- Test 5: Retrieval Quality (20 questions) ---")
    for test in retrieval_tests:
        retrieved = retrieve(test["question"])
        combined_text = " ".join([c["text"] for c in retrieved]).lower()
        hit = all(kw.lower() in combined_text for kw in test["must_contain"])
        results["retrieval"]["total"] += 1
        if hit:
            results["retrieval"]["hit"] += 1
        status = "PASS" if hit else "FAIL"
        scores_str = ", ".join(f"{c['score']:.3f}" for c in retrieved)
        print(f"  [{status}] Q: {test['question']}  Scores: [{scores_str}]")
        results["retrieval"]["details"].append({"question": test["question"], "hit": hit})

    # --- Test 6: Precision/Recall ---
    print("\n--- Test 6: Precision@k and Recall@k (20 questions) ---")
    results["precision"] = {"total": 0, "sum_precision": 0.0, "details": []}
    results["recall"] = {"total": 0, "sum_recall": 0.0, "details": []}
    for test in retrieval_ground_truth:
        relevant_indices = set()
        for i, chunk in enumerate(chunks):
            if all(kw.lower() in chunk.lower() for kw in test["relevant_keywords"]):
                relevant_indices.add(i)
        retrieved = retrieve(test["question"])
        retrieved_indices = set(c["idx"] for c in retrieved)
        relevant_retrieved = retrieved_indices & relevant_indices
        precision = len(relevant_retrieved) / len(retrieved_indices) if retrieved_indices else 0.0
        recall = len(relevant_retrieved) / len(relevant_indices) if relevant_indices else 1.0
        results["precision"]["total"] += 1
        results["precision"]["sum_precision"] += precision
        results["recall"]["total"] += 1
        results["recall"]["sum_recall"] += recall
        print(f"  P@{FINAL_K}={precision:.2f} R@{FINAL_K}={recall:.2f} | {test['question']}")
        results["precision"]["details"].append({"question": test["question"], "precision": precision})
        results["recall"]["details"].append({"question": test["question"], "recall": recall})

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXTENDED EVAL SUMMARY (Mistral 7B-Instruct generator)")
    print("=" * 60)
    print(f"\n  Answerable accuracy:    {results['answerable']['correct']}/{results['answerable']['total']} "
          f"({100*results['answerable']['correct']/results['answerable']['total']:.0f}%)")
    print(f"  Refusal accuracy:       {results['refusal']['correct']}/{results['refusal']['total']} "
          f"({100*results['refusal']['correct']/results['refusal']['total']:.0f}%)")
    print(f"  False refusal rate:     {results['false_refusal']['refused']}/{results['false_refusal']['total']} "
          f"({100*results['false_refusal']['refused']/results['false_refusal']['total']:.0f}%) — lower is better")
    print(f"  Parametric leakage:     {results['parametric_leakage']['leaked']}/{results['parametric_leakage']['total']} "
          f"({100*results['parametric_leakage']['leaked']/results['parametric_leakage']['total']:.0f}%) — lower is better")
    print(f"  Retrieval hit rate:     {results['retrieval']['hit']}/{results['retrieval']['total']} "
          f"({100*results['retrieval']['hit']/results['retrieval']['total']:.0f}%)")

    with open("eval_extended_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Detailed results saved to eval_extended_results.json")


if __name__ == "__main__":
    run_eval()
