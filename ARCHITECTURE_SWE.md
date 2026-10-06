# Architecture & Inner Workings — Swedish Defence RAG+PEFT System

Detailed technical documentation of the RAG+PEFT system for Swedish military/defence document Q&A and tactical scenario simulation.

---

## System Overview

```
markdown_output/ (95 Swedish military/defence documents)
    │
    ├──► build_rag_index_swe.py ──► rag_index_swe/ (FAISS + ~17,950 chunks, GPU embedding)
    │                                      │
    │                                      ▼
    │                    generate_rag_traindata_swe.py ──► rag_traindata_swe.json (450 examples)
    │                                                            │
    │                                                            ▼
    │                         rag_peft.py ──► eupolicy_rag_llama_model_swe/ (LoRA adapter)
    │                                                                        │
    └────────────────────────────────────────────────────────────────────────┤
                                                                             ▼
                                              ┌──────────────────────────────────────────────┐
                                              │           Runtime Applications                │
                                              ├──────────────────────────────────────────────┤
                                              │  ask_rag_swe.py        (interactive Q&A)     │
                                              │  test_question.py      (single query)        │
                                              │  eval_rag_extended_swe.py (evaluation)       │
                                              │  scenario_simulator.py (tactical wargaming)  │
                                              └──────────────────────────────────────────────┘
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                    Retrieval      Classification    Generation
                                 (BGE-M3 + BM25    (AI-Sweden        (
                                  + cross-encoder)  Llama 3 8B)       Claude)
```

---

## Runtime Pipeline (per query)

### Step 1: Query Expansion (Swedish)
```python
"Vad är totalförsvar?"   → ["Vad är totalförsvar?", "Definiera totalförsvar", "Förklara totalförsvar"]
"Vilka vapensystem?"     → ["Vilka vapensystem?", "Beskriv vapensystem"]
"Hur organiseras CBRN?"  → ["Hur organiseras CBRN?", "På vilket sätt organiseras CBRN?"]
```
Swedish-aware rule-based rephrasings to catch terminology mismatches across the 95-document corpus.

### Step 2: Hybrid Retrieval
For each expanded query:
- **Vector search**: BGE-M3 embeds query (CPU) → FAISS cosine similarity → top-15
- **BM25 search**: Swedish tokenization with stopword removal + number normalization → TF-IDF scoring → top-15
- **Merge**: union of candidates, weighted 60% vector + 40% BM25

### Step 3: Cross-Encoder Reranking
- Takes the merged candidate pool (~15–25 chunks)
- Scores each (query, chunk) pair with `ms-marco-MiniLM-L-12-v2` (CPU)
- Returns top-3 by cross-encoder score

### Step 4: Classification (AI-Sweden Llama 3 8B, QLoRA fine-tuned)
- Receives: context (3 chunks with relevance scores) + question
- Outputs: JA or NEJ (Swedish YES/NO)
- Detection: checks for both "JA"/"Ja" and "YES" in response
- If NEJ → return immediate refusal (no generation step)

### Step 5: Generation (multi-backend)
- Only reached if classifier said JA

- **Claude API**: Anthropic Claude with context + anti-hallucination instructions


---

## Component Details

### 1. `build_rag_index_swe.py`

**Purpose:** Indexes all 95 Swedish military/defence markdown documents into a searchable vector database.

**Input:** `markdown_output/` folder containing documents on doctrine, drones, logistics, totalförsvar, NATO, CBRN, artillery, air defence, urban warfare, helicopters, mobility, procurement, strategic planning, FOI reports, and more.

**Parameters:**
```python
CORPUS_PATH = "markdown_output/"        # 95 markdown files
embed_model = "BAAI/bge-m3"            # 1024-dim, multilingual
embed_device = "cuda"                   # GPU for indexing speed
embed_batch_size = 64                   # GPU batch processing
```

**Output:**
- `rag_index_swe/index.faiss` — ~17,950 vectors, 1024 dimensions
- `rag_index_swe/chunks.json` — ~17,950 text chunks (Swedish)

**Design choice — GPU for indexing, CPU for queries:**
Embedding 17,950 chunks requires GPU acceleration (batch=64) to complete in reasonable time. At query time, only a single vector is embedded, so CPU is sufficient and leaves the GPU free for the LLM.

---

### 2. `generate_rag_traindata_swe.py`

**Purpose:** Creates Swedish defence-domain training data for the classifier/generator.

**Data Composition (450 examples):**
- **Answerable examples** (~300): relevant chunks from military documents → factual answer in Swedish
- **Refusal examples** (~150): irrelevant or insufficient context → refusal response
- **Ratio:** 2:1 answerable-to-refusal

**Training Example Format:**
```json
{
  "messages": [
    {"role": "system", "content": "Du är en expert på svenska försvarsfrågor..."},
    {"role": "user", "content": "Kontext:\n[Källa 1, relevans: 0.82]: ...\n\nFråga: Hur organiseras totalförsvaret?"},
    {"role": "assistant", "content": "Totalförsvaret organiseras genom..."}
  ]
}
```

**Why 2:1 ratio:** The 8B model learns refusal patterns very quickly due to its larger capacity. More answerable examples prevent the model from becoming overly conservative, achieving 0% false refusal rate.

---

### 3. `rag_peft.py`

**Purpose:** Fine-tunes AI-Sweden Llama 3 8B-Instruct with QLoRA for defence-domain classification and generation.

**Configuration:**
```python
MODEL_ID = "AI-Sweden-Models/Llama-3-8B-instruct"
quantization = NF4, 4-bit, double quantization
LoRA: r=16, alpha=16, dropout=0.05
targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
training: 2 epochs, batch=1, grad_accum=8, lr=1.5e-4, cosine schedule
gradient_checkpointing: True
```

**Save strategy — LoRA adapter only:**
The merged 8B model in fp16 would be ~16GB — too large to save and reload in 8GB VRAM. The LoRA adapter is saved separately (~30MB) and loaded on top of the 4-bit quantized base model at inference time.

**Why gradient checkpointing:** Required to fit 8B model training in 8GB VRAM. Trades compute time for memory by recomputing activations during backpropagation instead of storing them.

---

### 4. `ask_rag_swe.py`

**Purpose:** Interactive Swedish Q&A with the full retrieval pipeline. Local model only.

**Architecture:**
- Single AI-Sweden Llama 3 8B model for both classification and generation
- Full hybrid retrieval with Swedish query expansion and cross-encoder reranking
- Two-stage classify-then-answer (same model, different prompts)

---

### 5. `test_question.py`

**Purpose:** Single question interface with optional Claude API generation.

**Usage:**
```bash
python test_question.py "Vad innebär DGO?"           # local model generation
python test_question.py --claude "Beskriv NBC-skydd"  # Claude API generation
```

**Architecture:**
- Classification always performed by local fine-tuned 8B model
- Generation routed to local model or Claude API based on `--claude` flag

---

### 6. `scenario_simulator.py`

**Purpose:** Tactical scenario simulator for wargaming analysis, grounded in both injected ORBAT data and retrieved Swedish doctrine.

**Scenario injected at runtime:**
```
Setting: Swedish Amfibiebataljon (Amf 4, Göteborg) vs Russian VDV BTG
         airlifted to Landvetter airport

Swedish forces (~400 troops):
├── CB90 assault craft
├── RBS-17 anti-ship missiles
├── Carl Gustaf recoilless rifles
├── Kustjägare (coastal rangers)
├── 81mm mortars
└── Limited mines and explosives

Russian forces (VDV BTG, ~600 troops):
├── BMD-4M infantry fighting vehicles
├── Sprut-SD light tanks
├── 2S9 Nona-S self-propelled mortars
├── No heavy armour
└── No IMR-2 engineering vehicles

Terrain: Göteborg–Landvetter corridor
├── 20km distance
├── Urban + forest terrain
├── Mölndalsån bridges (key chokepoints)
└── Rv40 tunnel
```

**Engineering norms included:** Mine laying rates, bridge demolition timelines, obstacle clearing times for specific vehicle types.

**Anti-hallucination prompt:**
- Requires `[Scenario]` citations for ORBAT/terrain facts
- Requires `[Doktrin källa X]` citations for doctrine references
- Must refuse to fabricate numbers, capabilities, or unit compositions
- Cannot invent equipment not listed in the scenario data

**Generation backends:**
```bash
python scenario_simulator.py              # local 8B model
python scenario_simulator.py --claude     # Claude API (Anthropic)
```

**Output files:** `scenario_answer_claude.md`, `scenario_answer_local.md`, `scenario_answer_mistral.md`

---

### 7. `eval_rag_extended_swe.py`

**Purpose:** Comprehensive evaluation of the system across multiple accuracy dimensions.

**Test categories:**
1. **Answerable accuracy** (20 questions) — correct fact extraction from military documents
2. **Refusal accuracy** (20 questions) — refuses out-of-scope or unanswerable questions
3. **False refusal rate** (20 questions) — full pipeline, should NOT refuse valid questions
4. **Parametric leakage** (10 questions) — should refuse despite LLM pretraining knowledge
5. **Retrieval hit rate** (20 questions) — relevant content found in top-3 across 95 documents

---

## Retrieval Components

All retrieval is handled by the shared `rag_retrieval.py` module (`RAGRetriever` class). This single implementation is used by the eval script, web interface Q&A, and simulation doctrine retrieval.

### Embedding Model: `BAAI/bge-m3`
- ~568M parameters, 1024-dim output
- Multilingual: supports 100+ languages including Swedish
- **Indexing:** GPU (batch=64) for speed across 17,950 chunks
- **Queries:** CPU to leave GPU for LLM inference
- Handles Swedish compound words, military acronyms, and morphological variants

### BM25 (Custom Implementation with Swedish Stopwords)
- Whitespace tokenization with number normalization
- Swedish stopword filtering (~80 common function words removed)
- Standard BM25 parameters: k1=1.5, b=0.75
- Catches keyword-specific queries (unit designations, weapon system names, doctrine references)

### Cross-Encoder: `ms-marco-MiniLM-L-12-v2`
- 12-layer cross-encoder, ~33M parameters, CPU
- Scores (query, chunk) pairs together for precise relevance ranking
- Reranks top-15 hybrid candidates → selects top-3

### Hybrid Scoring
```python
hybrid_score = 0.6 * normalized_vector_score + 0.4 * normalized_bm25_score
```

---

## Model Architecture

### Classifier: AI-Sweden Llama 3 8B-Instruct (QLoRA fine-tuned)
- 8B parameters, loaded in 4-bit NF4 quantization (~4.5GB)
- Continued pre-training of Llama 3 8B on Swedish text by AI Sweden
- Fine-tuned with QLoRA on 450 Swedish defence-domain examples
- Determines answerability (JA/NEJ) from retrieved context

### Generator

| Backend | Latency | Quality | Cost |
|---------|---------|---------|------|
| Claude API | ~5s | Excellent | Per-token |

### Memory Layout (8GB VRAM — Inference)
```
├── AI-Sweden Llama 3 8B (4-bit NF4): ~4.5 GB
├── LoRA adapter:                      ~0.03 GB
├── KV cache:                          ~1.0 GB
└── Free (for embedding on CPU):       ~2.1 GB
```

### Memory Layout (8GB VRAM — Training)
```
├── AI-Sweden Llama 3 8B (4-bit NF4): ~4.5 GB
├── LoRA adapters (trainable):         ~0.03 GB
├── Optimizer states (AdamW):          ~0.06 GB
├── Activations (checkpointed):        ~2.0 GB
└── Free:                              ~1.0 GB
```

---

## Swedish-Specific Adaptations

### Query Expansion Patterns
```python
"Vad är X?"         → ["Vad är X?", "Definiera X", "Förklara X"]
"Vilka är X?"       → ["Vilka är X?", "Beskriv X"]
"Hur X?"            → ["Hur X?", "På vilket sätt X?"]
"Vad säger...om X?" → ["Vad säger...om X?", "regler angående X", "X krav skyldigheter"]
default             → [question, question + " doktrin krav"]
```

### Swedish Stopwords (filtered from BM25)
```python
{"och", "att", "det", "som", "för", "den", "med", "har", "inte", "till",
 "ett", "var", "från", "kan", "ska", "vid", "eller", "om", "av", "på",
 "är", "de", "en", "denna", "dessa", "bör", "ska", "skulle", "enligt", ...}
```

### JA/NEJ Classification Detection
```python
response = tokenizer.decode(...).strip().upper()
return "YES" in response or "JA" in response
```
The Swedish model naturally responds "Ja" or "Nej" even when prompted with "YES/NO". Both are handled.

### Bilingual Refusal Detection
```python
refusal_indicators = [
    # Swedish
    "innehåller inte", "kan inte hitta", "kan inte besvaras",
    "inte tillräcklig information", "ingen information",
    # English (model may occasionally respond in English)
    "does not contain", "cannot find", "cannot be answered",
    "not enough information",
]
```

---

## Anti-Hallucination Design

### Problem
Military/defence topics are high-stakes domains where fabricated information is dangerous. LLMs have extensive pretraining knowledge about military systems and may generate plausible but unsourced claims.

### Solution: Multi-Layer Grounding

1. **Classification gate** — only answer if retrieved context is relevant (JA/NEJ)
2. **Source-citation requirement** — generation prompts require explicit citations:
   - `[Scenario]` for injected ORBAT/terrain data
   - `[Doktrin källa X]` for retrieved Swedish doctrine
3. **Anti-parametric-leakage prompt** — explicit instruction to use ONLY provided context
4. **Refusal training** — model trained to refuse when context is insufficient (80% refusal accuracy)
5. **Parametric leakage monitoring** — evaluated explicitly (20% leakage rate, acceptable with citation requirements)

---

## Evaluation Results

| Category | Score | Description |
|----------|-------|-------------|
| Answerable accuracy | 95% (19/20) | Correct extraction from military context |
| Refusal accuracy | 80% (16/20) | Refuses out-of-scope questions |
| False refusal rate | 0% (0/20) | Never refuses valid questions |
| Parametric leakage | 20% (2/10) | Citation requirements mitigate leakage |
| Retrieval hit rate | 95% (19/20) | Finds relevant content across 95 documents |

---

## Training Details

### Why QLoRA Works for 8B on 8GB VRAM
- 4-bit quantization reduces model from ~16GB to ~4.5GB
- LoRA adds only ~30MB of trainable parameters (0.4% of model)
- Gradient checkpointing trades compute for memory
- batch_size=1 with gradient_accumulation=8 gives effective batch of 8
- Total VRAM usage during training: ~7GB

### Why 2 Epochs
The 8B model converges faster than smaller models due to superior Swedish pre-training. 2 epochs achieve strong performance without over-fitting on refusal patterns.

### Why Separate LoRA Adapter
Merging LoRA into the base model produces a full-precision 8B model (~16GB). This cannot be reloaded into 8GB VRAM even with re-quantization due to intermediate memory requirements during the loading process.

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Build Phase)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  95 markdown docs ──► build_rag_index_swe.py ──► FAISS index            │
│         (GPU embedding, batch=64)              (~17,950 chunks)         │
│                                                                         │
│  FAISS index ──► generate_rag_traindata_swe.py ──► 450 training pairs   │
│                                                                         │
│  Training data ──► rag_peft.py ──► LoRA adapter  │
│         (QLoRA, 2 epochs, gradient checkpointing)                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        ONLINE (Query Phase)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Query ──► expansion ──► hybrid search ──► cross-encoder rerank         │
│                                                     │                   │
│                                                top-3 chunks             │
│                                                     │                   │
│                                                     ▼                   │
│                                    Classifier (local 8B + LoRA)         │
│                                          │                              │
│                                   JA ────┼──── NEJ                      │
│                                   │             │                       │
│                                   ▼             ▼                       │
│                          Generator           Refusal                    │
│                    (Claude API)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                   SCENARIO SIMULATOR (Special Mode)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Injected ORBAT + Terrain + Engineering Norms                           │
│         +                                                               │
│  Retrieved doctrine (from RAG pipeline)                                 │
│         │                                                               │
│         ▼                                                               │
│  Generator (Claude API) with anti-hallucination prompt        │
│         │                                                               │
│         ▼                                                               │
│  Cited tactical analysis ──► scenario_answer_*.md                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```


---

## UND — Intelligence Module (`und.py`)

### Purpose
Maintains a structured intelligence picture of all entities (units, structures, locations) in the simulation. Supports runtime injection of new observations with source tracking and confidence levels.

### Data Model

```
UND
├── units: dict[str, Unit]
│   ├── AMF4_BAT (battalion)
│   │   ├── AMF4_KP1 (company)
│   │   ├── AMF4_KP2 (company)
│   │   ├── AMF4_KJ (platoon — Kustjägare)
│   │   ├── AMF4_RBS (platoon — RBS-17)
│   │   └── AMF4_PIONEER (platoon — minering)
│   └── VDV_BTG (battalion)
│       ├── VDV_MAIN (company — Rv40 axis)
│       └── VDV_FLANK (company — southern flank)
├── structures: dict[str, Structure]
│   ├── BRO1_RV40, BRO2_GOTEBORG, BRO3_KVARNBY
│   ├── TUNNEL_KALLEBACK
│   └── LANDVETTER_AP
└── locations: dict[str, StrategicLocation]
    ├── MF1_MOLNLYCKE, MF2_KALLERED, MF3_MOLNDALSAN (minefields)
    ├── PL_ALFA, PL_BRAVO, PL_CHARLIE (phase lines)
    └── OBJ_HAMN (objective)
```

### Intelligence Injection API

```python
intel.inject(
    entity_id="VDV_MAIN",       # What entity
    attribute="position",        # What was observed
    value=[57.665, 12.20],      # Observed value
    source="KJ spaning",        # Who reported
    confidence=0.9,             # 0.0-1.0
    sim_time="H+06",           # When (optional, defaults to current)
)
```

**Confidence rules:**
- `≥ 0.5`: Updates the entity's assessed state
- `< 0.5`: Stored in history but does NOT update state
- All reports stored regardless of confidence (full audit trail)

### Integration Points

| Consumer | How it uses UND |
|----------|----------------|
| Simulation engine | Reads positions/status each timestep, writes combat results |
| Scenario simulator | Initializes UND at start, queries for context |
| Web interface | Could display UND summary alongside RAG answers |
| Voice intel (future) | Parses speech → calls `und.inject()` |
| Map visualization | Reads unit positions and structure status |

### Future: Voice Intelligence Pipeline

Documented in `DESIGN_voice_intel.md`. Will chain:
1. Whisper STT (Swedish audio → text)
2. Claude/LLM parser (free text → structured entity updates)
3. UND injection (updates simulation state)
