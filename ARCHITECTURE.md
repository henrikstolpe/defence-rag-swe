# Architecture & Inner Workings

Detailed technical documentation of the RAG+PEFT system for EU AI Act Q&A.

---

## System Overview

```
euaiact (raw text, EU AI Act Regulation 2024/1689)
    │
    ├──► build_rag_index.py ──► rag_index/ (FAISS + 832 chunks)
    │                                │
    │                                ▼
    │                    generate_rag_traindata.py ──► rag_traindata.json (multi-chunk format)
    │                                                        │
    │                                                        ▼
    │                              EU_policy_train_rag_llama32.py ──► eupolicy_rag_llama_model/ (classifier)
    │                                                                        │
    └──────────────────────────────────────────────────────────────────────────┤
                                                                               ▼
                                                              ask_rag.py / eval_rag_extended.py (runtime)
                                                                    │
                                                    ┌───────────────┼───────────────┐
                                                    ▼               ▼               ▼
                                              Retrieval      Classification    Generation
                                           (BGE + BM25 +    (Llama 3.2 3B    (Mistral 7B
                                            cross-encoder)    fine-tuned)      Instruct)
```

---

## Runtime Pipeline (per query)

### Step 1: Query Expansion
```python
"What are the fines?" → ["What are the fines?", "Define the fines?", "fines regulation requirements"]
```
Simple rule-based rephrasings to catch terminology mismatches between user queries and document text.

### Step 2: Hybrid Retrieval
For each expanded query:
- **Vector search**: BGE embeds query → FAISS cosine similarity → top-15
- **BM25 search**: tokenize with number normalization → TF-IDF scoring → top-15
- **Merge**: union of candidates, weighted 60% vector + 40% BM25

### Step 3: Cross-Encoder Reranking
- Takes the merged candidate pool (~15-25 chunks)
- Scores each (query, chunk) pair with `ms-marco-MiniLM-L-12-v2`
- Returns top-3 by cross-encoder score
- Much more accurate than bi-encoder similarity (~200ms on CPU)

### Step 4: Classification (Llama 3.2 3B, fine-tuned)
- Receives: context (3 chunks with relevance scores) + question
- Outputs: YES or NO
- Rules encoded in prompt: refuse comparisons, refuse external entities, refuse unstated statistics
- If NO → return immediate refusal (no generation needed)

### Step 5: Generation (Mistral 7B-Instruct, 4-bit)
- Only reached if classifier said YES
- Model swap: unload 3B classifier, load 7B generator
- Strict system prompt: "answer ONLY from context, NO knowledge beyond context"
- Greedy decoding, max 150 tokens
- Returns grounded 2-3 sentence answer

---

## 1. `build_rag_index.py`

### Purpose
Converts the raw EU AI Act document into a searchable vector database.

### Parameters
```python
DOCUMENT_PATH = "euaiact"
chunk_size = 150      # words per chunk
overlap = 40          # words of overlap
embed_model = "BAAI/bge-small-en-v1.5"  # 384-dim, CPU
```

### Output
- `rag_index/index.faiss` — 832 vectors, 384 dimensions
- `rag_index/chunks.json` — 832 text chunks

### Why 150-word chunks
Legal text packs discrete facts into individual paragraphs. Smaller chunks give higher precision per chunk and better relevance scores from the cross-encoder.

---

## 2. `generate_rag_traindata.py`

### Purpose
Creates training data in the exact format used at inference time (multi-chunk with relevance scores).

### Training Example Format
```json
{
  "messages": [
    {"role": "system", "content": "You are an EU AI Act expert. Answer ONLY from context..."},
    {"role": "user", "content": "Context:\n[Source 1, relevance: 0.82]: ...\n\n[Source 2, relevance: 0.45]: ...\n\n[Source 3, relevance: 0.38]: ...\n\nQuestion: What are the fines?"},
    {"role": "assistant", "content": "Non-compliance is subject to fines of up to 35 million euros..."}
  ]
}
```

### Data Composition (720 examples)
- **180 answerable** (45 unique × 4 repeats): relevant primary chunk + 2 random chunks → factual answer
- **360 standard unanswerable** (30 questions × 3 chunks × 4 repeats): irrelevant chunks → refusal
- **180 comparison unanswerable** (15 questions × 3 variations × 4 repeats): relevant AI Act chunks but external comparison question → refusal

### Why Multi-Chunk Format
Previous training used single chunks. Inference passes 3 chunks with `[Source N, relevance: X.XX]:` prefixes. Training on the exact inference format eliminates distribution mismatch and improves extraction accuracy.

---

## 3. `EU_policy_train_rag_llama32.py`

### Purpose
Fine-tunes Llama 3.2 3B-Instruct with QLoRA to serve as the classifier (and fallback generator).

### Configuration
```python
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
quantization = NF4, 4-bit, double quantization
LoRA: r=16, alpha=16, dropout=0.05
targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
training: 3 epochs, batch=2, grad_accum=4, lr=2e-4, cosine schedule
```

### Post-Training Test
Includes a two-stage classify-then-answer test to verify both behaviors work after training.

---

## 4. `ask_rag.py`

### Purpose
Interactive Q&A application with the full pipeline.

### Architecture
- Uses single 3B model for both classification and generation (no model swap in interactive mode for lower latency)
- Full hybrid retrieval with query expansion and cross-encoder reranking
- Two-stage classify-then-answer

### System Prompt
```
You are an EU AI Act expert. Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer, say so.
If the question asks to compare with something not mentioned in the context, refuse.
You have NO knowledge beyond what is in the context.
Be concise: 2-3 sentences maximum.
```

---

## 5. `eval_rag_extended.py`

### Purpose
Comprehensive evaluation with model swapping for best accuracy.

### Architecture
- **Phase 1**: Load Llama 3.2 3B, classify all 70 questions (YES/NO)
- **Phase 2**: Swap to Mistral 7B-Instruct (4-bit), generate answers for YES-classified questions
- Single swap minimizes latency

### Tests (90 total questions)
1. **Answerable accuracy** (20 questions) — fact extraction from relevant context
2. **Refusal accuracy** (20 questions) — refuse comparisons, external entities, unstated stats
3. **False refusal rate** (20 questions) — full pipeline, should NOT refuse
4. **Parametric leakage** (10 questions) — should refuse despite pretraining knowledge
5. **Retrieval hit rate** (20 questions) — keyword presence in top-3
6. **Precision@3 / Recall@3** (20 questions) — retrieval quality metrics

### Latest Results
| Category | Score |
|----------|-------|
| Answerable accuracy | 95% (19/20) |
| Refusal accuracy | 95% (19/20) |
| False refusal rate | 0% (0/20) |
| Parametric leakage | 0% (0/10) |
| Retrieval hit rate | 95% (19/20) |
| Avg Precision@3 | 0.45 |
| Avg Recall@3 | 0.18 |

---

## Retrieval Components

### Embedding Model: `BAAI/bge-small-en-v1.5`
- 33M parameters, 384-dim output, CPU
- Trained specifically for retrieval (query-document matching)
- Outperforms MiniLM on MTEB retrieval benchmarks by ~10%

### BM25 (Custom Implementation)
- Whitespace tokenization with number normalization ("35 000 000" → "35000000")
- Standard BM25 parameters: k1=1.5, b=0.75
- Catches keyword-specific queries that embeddings miss (Article numbers, EUR amounts)

### Cross-Encoder: `ms-marco-MiniLM-L-12-v2`
- 12-layer cross-encoder, ~33M parameters, CPU
- Scores (query, chunk) pairs together — much more accurate than bi-encoder
- Reranks top-15 hybrid candidates → selects top-3
- Adds ~200ms latency per query

### Hybrid Scoring
```python
hybrid_score = 0.6 * normalized_vector_score + 0.4 * normalized_bm25_score
```

---

## Model Details

### Classifier: Llama 3.2 3B-Instruct (fine-tuned)
- 3B parameters, loaded in bfloat16 (~6GB) or 4-bit (~2GB)
- Fine-tuned with QLoRA on 720 multi-chunk examples
- Specialized for YES/NO classification on answerability
- ~97% token accuracy on training data

### Generator: Mistral 7B-Instruct v0.3
- 7B parameters, loaded in 4-bit NF4 (~4GB)
- NOT fine-tuned — uses instruction following out of the box
- Excellent at precise fact extraction from context
- Strict system prompt prevents hallucination

### Memory Layout (8GB VRAM, one model at a time)
```
Classifier loaded:
├── Llama 3.2 3B (bfloat16): ~6.0 GB
├── KV cache:                 ~0.5 GB
└── Free:                     ~1.1 GB

Generator loaded:
├── Mistral 7B (4-bit NF4):  ~4.0 GB
├── KV cache:                 ~1.0 GB
└── Free:                     ~2.6 GB
```

---

## Quantization Details

**Method**: BitsAndBytes NF4 (for Mistral 7B generator)
- Each weight stored as 4 bits (16 possible values)
- NF4 distribution optimized for normally-distributed neural network weights
- Double quantization: quantization constants themselves quantized to 8-bit
- Compute dtype: bfloat16 (dequantized on-the-fly during forward pass)
- Model size: ~14GB (FP16) → ~4GB (NF4)

---

## Training Details

### The Training Loop
1. **Forward pass**: model reads multi-chunk context + question, predicts next token
2. **Loss calculation**: cross-entropy on assistant tokens only
3. **Backward pass**: gradients flow through frozen base to LoRA adapters only
4. **Weight update**: AdamW with cosine LR schedule

### Why QLoRA Works for Classification
- Only ~15M trainable parameters (0.5% of model)
- Sufficient for learning YES/NO boundary and refusal patterns
- Training fits in 8GB VRAM with batch size 2 + gradient accumulation 4
- Converges in ~20 minutes on RTX 4060

### Typical Training Curve
```
Epoch 0.4: loss=2.42, accuracy=53%
Epoch 0.9: loss=1.53, accuracy=66%
Epoch 1.5: loss=0.56, accuracy=86%
Epoch 2.0: loss=0.35, accuracy=91%
Epoch 2.5: loss=0.22, accuracy=95%
Epoch 3.0: loss=0.16, accuracy=97%
```

---

## Evolution of the System

| Version | Answerable | Refusal | False Refusal | Key Change |
|---------|-----------|---------|---------------|------------|
| v1 (single chunk, MiniLM, no classifier) | 90% | 30% | 0% | Baseline |
| v2 (+ strict classifier prompt) | 90% | 100% | 20% | Two-stage classify |
| v3 (+ softened classifier) | 85% | 95% | 5% | Balanced classifier |
| v4 (+ hybrid search, cross-encoder) | 85% | 95% | 0% | Better retrieval |
| v5 (+ multi-chunk training, query expansion) | 85% | 95% | 5% | Train/inference match |
| **v6 (+ Mistral 7B generator)** | **95%** | **95%** | **0%** | Better generator |
   - Rules: refuse comparisons with external topics, refuse if entities not in context, "when in doubt, answer NO"
   - If NO → return immediate refusal without generating

   **Step C: Generate** (`generate` function):
   - Only reached if classifier said YES
   - Constructs prompt with strict system message + retrieved chunks with relevance scores + question
   - Generates up to 100 new tokens with greedy decoding
   - Returns the grounded answer

3. **Interactive loop**:
   - Reads user input
   - Calls `ask(question)`
   - Prints the answer and retrieval metadata (number of sources, top score)
   - Exits on "quit", "exit", "q", Ctrl+C, or EOF

### System Prompt (with explicit refusal rules)
```
You are an EU AI Act expert. Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer, say so.
If the question asks to compare with something not mentioned in the context, refuse.
You have NO knowledge beyond what is in the context. If you find yourself about to state
a fact not present in the context, stop and refuse instead.
Be concise: 2-3 sentences maximum.
```

### Classifier Prompt
```
You are a strict classification system. Respond with ONLY 'YES' or 'NO'.
Rules:
- Answer NO if the question asks to COMPARE, CONTRAST, or find DIFFERENCES with anything not in context.
- Answer NO if the question mentions a specific entity not named in the context.
- Answer NO if answering would require knowledge beyond what is written.
- Answer NO if the question asks about real-world outcomes not stated in the context.
- Answer YES only if every fact needed is explicitly written in the context.
- When in doubt, answer NO.
```

### Hybrid Search Details
```python
# Vector search: FAISS cosine similarity (semantic matching)
# BM25 search: term frequency-inverse document frequency (keyword matching)
# Combination: 0.6 * normalized_vector_score + 0.4 * normalized_bm25_score
# Retrieves top-5 from each, merges candidates, returns top-3
```

### Context Format (with relevance scores)
```
[Source 1, relevance: 0.72]: The EU AI Act prohibits AI systems that deploy subliminal...
[Source 2, relevance: 0.65]: Social scoring of natural persons by public or private...
[Source 3, relevance: 0.58]: The risk-based approach tailors rules to the intensity...
```

### Generation Parameters
```python
max_new_tokens=100    # enough for 2-3 sentences
do_sample=False       # greedy decoding — deterministic, no randomness
use_cache=True        # KV cache for faster autoregressive generation
pad_token_id=...      # prevents warning about missing pad token
```

### Retrieval Score Interpretation
- Score > 0.7: Strong match — chunk is highly relevant
- Score 0.5-0.7: Moderate match — chunk may contain relevant info
- Score < 0.5: Weak match — chunk is likely not relevant

---

## 5. `eval_rag.py`

### Purpose
Automated evaluation of the full system across five dimensions, using the same hybrid search and two-stage classification as the runtime.

### Tests

1. **Answerable accuracy** (10 questions):
   - Hand-picked relevant chunks (isolates generation from retrieval)
   - Checks if key facts appear in the response
   - Measures: can the model extract information correctly?

2. **Refusal accuracy** (10 questions, with two-stage classification):
   - Includes 5 comparison-style questions (the hardest case)
   - Context is about the AI Act but question asks about external topics
   - Uses the full classify-then-generate pipeline
   - Measures: does the classifier + model refuse when it should?

3. **False refusal rate** (10 questions, full hybrid RAG + two-stage):
   - Full pipeline: hybrid retrieval + classification + generation
   - Questions that ARE answerable from the document
   - Measures: does the classifier over-refuse? (lower is better)

4. **Parametric leakage** (5 questions, with two-stage classification):
   - Topics the model knows from pretraining (GPT-4, Gemini, etc.)
   - Context is irrelevant to the question
   - Measures: does the model answer from memory instead of refusing?

5. **Retrieval quality** (10 questions):
   - Uses hybrid search (vector + BM25)
   - Checks if top-3 results contain required keywords
   - Independent of the LLM — tests the retrieval pipeline quality

### Latest Results

| Category | Score | Notes |
|----------|-------|-------|
| Answerable accuracy | 90% | 9/10 correct fact extraction |
| Refusal accuracy | 100% | 10/10 refused correctly (including all comparisons) |
| False refusal rate | 20% | 2/10 over-refused (tradeoff for safety) |
| Parametric leakage | 0% | 0/5 leaked (complete suppression) |
| Retrieval hit rate | 90% | 9/10 found relevant chunks |

### Output
- Prints pass/fail for each test with the model's response
- Summary with percentages for each dimension
- Saves detailed results to `eval_results.json`

---

## 6. `euaiact` (Source Document)

Plain text file containing the EU AI Act (Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024). Published in the Official Journal on 12 July 2024.

The document covers:
- Definition of AI systems and key concepts (deployer, provider, biometric data)
- Prohibited AI practices (social scoring, subliminal manipulation, emotion recognition in workplaces)
- High-risk AI system classification and requirements
- Rules on biometric identification in public spaces
- General-purpose AI models and systemic risk
- AI regulatory sandboxes
- Transparency obligations
- Governance structure (AI Office, national authorities)
- Penalties and fines (up to €35M or 7% of turnover)
- Conformity assessment and CE marking

~6200 lines, approximately 90,000 words.

---

## 7. `rag_index/` (Artifacts)

### `index.faiss`
Binary file containing the FAISS vector index. Stores 832 normalized 384-dimensional float32 vectors. File size: ~1.3MB.

### `chunks.json`
JSON array of 832 text strings, each approximately 150 words. These are the retrievable units — when FAISS returns an index, the corresponding text is looked up here and passed to the LLM.

---

## 8. `rag_traindata.json` (Training Data)

JSON array of 720 training examples in OpenAI chat format. Each example has a `messages` array with system, user, and assistant roles.

- 180 examples teach the model to extract answers from relevant context
- 540 examples teach the model to refuse:
  - 360 with irrelevant context (standard unanswerable)
  - 180 with relevant AI Act context but comparison questions (adversarial)
- Examples are shuffled (no ordering bias)
- The 1:3 split is intentional — refusal is harder to learn than answering

---

## 9. `eupolicy_rag_llama_model/` (Model Weights)

Contains the full merged Llama 3.2 3B model with LoRA weights folded in:
- `model.safetensors` — the model weights (~6.4GB)
- `config.json` — model architecture configuration
- `tokenizer.json` — tokenizer vocabulary and rules
- `tokenizer_config.json` — tokenizer settings
- `generation_config.json` — default generation parameters
- `chat_template.jinja` — chat formatting template

This is a standalone model — no need for the base Llama model or PEFT library at inference time. Load directly with `AutoModelForCausalLM.from_pretrained()`.

---

## Memory Layout (RTX 4060 8GB)

```
Total VRAM: 7.62 GB
├── Llama 3.2 3B (4-bit quantized): ~2.0 GB
├── LoRA adapters (during training):  ~0.3 GB
├── Optimizer states (AdamW):         ~0.6 GB
├── Activations + gradients:          ~2.5 GB
├── KV cache (inference):             ~0.5 GB
└── Free / fragmentation:             ~1.7 GB
```

This is why Llama 3.2 3B works on 8GB while Gemma 4 E2B (7.5GB quantized) does not — there's no room left for training overhead.

---

## Embedding Model Details

**Model**: `BAAI/bge-small-en-v1.5`
- 33M parameters (small — runs on CPU in seconds)
- 384-dimensional output embeddings
- Trained specifically for retrieval tasks (query-document matching)
- Outperforms MiniLM on MTEB retrieval benchmarks by ~10%
- Normalization: L2-normalized outputs enable cosine similarity via dot product

**Why this model over MiniLM**: BGE is trained with contrastive learning on retrieval pairs, making it better at distinguishing relevant from irrelevant documents. MiniLM is trained for general sentence similarity which is a different (easier) task.

---

## Hybrid Search Details

The system combines two retrieval methods for better coverage:

**Vector Search (FAISS + BGE)**
- Semantic matching — understands meaning and paraphrases
- Good at: "What rules exist for facial recognition?" → finds chunks about "biometric identification"
- Weak at: exact terms, numbers, article references

**BM25 (Keyword Search)**
- Term frequency-inverse document frequency scoring
- Good at: "Article 5", "EUR 35 000 000", "Directive 2016/680"
- Weak at: paraphrases, synonyms, conceptual queries

**Combination**
```python
hybrid_score = 0.6 * normalized_vector_score + 0.4 * normalized_bm25_score
```
- Retrieve top-5 from each method (union of candidates)
- Normalize each method's scores to [0, 1]
- Weighted merge favoring semantic (0.6) over keyword (0.4)
- Return top-3 for the LLM

**BM25 Implementation**
- Simple whitespace tokenization with lowercasing and punctuation stripping
- Precomputed document frequencies and per-document term frequencies at startup
- Standard BM25 parameters: k1=1.5, b=0.75
- No external library needed — implemented in ~30 lines of Python

---

## Quantization Details

**Method**: BitsAndBytes NF4 (Normal Float 4-bit)
- Each weight stored as 4 bits (16 possible values)
- NF4 distribution optimized for normally-distributed neural network weights
- Double quantization: the quantization constants themselves are quantized to 8-bit
- Compute dtype: bfloat16 (dequantized on-the-fly during forward pass)

**Impact**:
- Model size: 6.4GB (FP16) → ~2.0GB (NF4)
- Quality loss: <1% on most benchmarks
- Speed overhead: ~10-20% slower than FP16 (dequantization cost)
