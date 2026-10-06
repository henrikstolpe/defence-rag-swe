# RAG + PEFT: Grounded Question Answering — Swedish Defence Documents

A system that combines Retrieval-Augmented Generation (RAG) with Parameter-Efficient Fine-Tuning (PEFT) to answer questions about Swedish military and defence documents. Includes a tactical scenario simulator for wargaming analysis.

Uses **AI-Sweden Llama 3 8B-Instruct** (fine-tuned with QLoRA) as classifier to prevent parametric leakage. Generation is handled by **Claude** (Anthropic API).

---

## How It Works

```
User Question (Swedish)
     │
     ▼
┌─────────────────────────────────────┐
│   Query Expansion (3 rephrasings)   │
│   Swedish patterns: Vad/Vilka/Hur   │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│   Hybrid Search (Vector + BM25)     │
│   BAAI/bge-m3 (multilingual, 1024d) │
│   + BM25 with Swedish stopwords     │
│   TOP_K=15 candidates               │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│   Cross-Encoder Reranking           │
│   ms-marco-MiniLM-L-12-v2          │
│   Rerank → top 3 chunks            │
└─────────────────────────────────────┘
     │  top-3 chunks with relevance scores
     ▼
┌──────────────────────────────────────────┐
│  Stage 1: Classifier                     │
│  AI-Sweden Llama 3 8B (QLoRA fine-tuned) │
│  "Can this be answered from context?"    │
│  Responds JA/NEJ                         │
│  If NEJ → immediate refusal              │
└──────────────────────────────────────────┘
     │  JA
     ▼
┌──────────────────────────────────────────┐
│  Stage 2: Generator                      │
│  Claude API (Anthropic)                  │
│  Generates grounded answer in Swedish    │
│  from retrieved context                  │
└──────────────────────────────────────────┘
     │
     ▼
   Answer (grounded in source, in Swedish)
```

---

## Document Corpus

95 Swedish military/defence markdown documents (~17,950 chunks indexed) from `markdown_output/`, covering:

- **Doctrine** — DGO (Doktrin för Gemensamma Operationer)
- **Drones / UAV** — reconnaissance, strike, counter-UAS
- **Logistics** — military supply chains, sustainment
- **Totalförsvar** — total defence, civil-military integration
- **NATO operations** — interoperability, collective defence
- **CBRN** — chemical, biological, radiological, nuclear defence
- **Artillery** — fire support, indirect fire systems
- **Air defence** — GBAD, missile systems
- **Urban warfare** — MOUT, combined arms in built-up areas
- **Helicopters** — army aviation, assault transport
- **Military mobility** — strategic and tactical movement
- **Defence procurement** — FMV, acquisition processes
- **Strategic planning** — threat assessments, capability development
- **FOI reports** — Swedish Defence Research Agency analyses

---

## Files

### Core Pipeline

| File | Purpose |
|------|---------|
| `build_rag_index_swe.py` | Indexes all 95 markdown files from `markdown_output/`, GPU-accelerated embedding |
| `generate_rag_traindata_swe.py` | Creates 450 training examples (2:1 answerable-to-refusal) with Swedish defence questions |
| `rag_peft.py` | Fine-tunes with QLoRA — 2 epochs, batch=1, grad_accum=8, gradient_checkpointing. Saves LoRA adapter |
| `ask_rag_swe.py` | Interactive Q&A (local model only) |
| `eval_rag_extended_swe.py` | Extended evaluation — 20 questions per category |
| `test_question.py` | Single question interface, supports `--claude` flag for Claude API generation |
| `scenario_simulator.py` | CLI tactical scenario with injected ORBAT, supports `--claude` flag |
| `sim_generator.py` | **Simulation engine** — two-step doctrine retrieval + Claude timeline generation |
| `scenario_parser.py` | Parses scenario text into structured UND data via Claude API |
| `rag_retrieval.py` | **Shared RAG module** — single implementation of hybrid search (vector + BM25 + reranking) |
| `web_interface.py` | **Web UI** — Gradio interface with scenario editor and simulation launcher |
| `und.py` | **UND** — Intelligence tracking module for units, structures, and locations |
| `test_und.py` | Unit tests for the UND module (16 tests) |
| `test_rag_retrieval.py` | Unit tests for the shared RAG retrieval module (11 tests) |
| `eval_reproducibility.py` | A/B test for Claude output consistency at different temperatures |
| `eval_claude_leakage_swe.py` | Claude-path leakage/refusal eval (production pipeline: classifier + Claude) |
| `DESIGN_voice_intel.md` | Design document for future voice-to-intelligence pipeline |
| `DOC_training_metrics.md` | Detailed explanation of all fine-tuning output metrics |
| `ANALYSIS_output_reproducibility.md` | Analysis of Claude output reproducibility and temperature impact |
| `EVAL_classifier_impact.md` | Analysis of classifier impact on refusal and leakage rates |

### Data & Artifacts

| File/Directory | Purpose |
|----------------|---------|
| `markdown_output/` | Source corpus — 95 Swedish military/defence markdown documents |
| `rag_index_swe/` | FAISS index + chunked documents (~17,950 chunks) |
| `rag_traindata_swe.json` | Training examples in multi-chunk chat format (450 examples) |
| `eupolicy_rag_llama_model_swe/` | Fine-tuned LoRA adapter for AI-Sweden Llama 3 8B |
| `eval_extended_results_swe.json` | Detailed evaluation results |
| `scenario_answer_claude.md` | Saved scenario simulator output (Claude) |
| `scenario_answer_local.md` | Saved scenario simulator output (local model) |
| `simulation_log.md` | Event log from last simulation run |

---

## Setup & Run

### Prerequisites

- NVIDIA GPU with at least 8GB VRAM (for classifier model)
- Python 3.10+
- HuggingFace access to `AI-Sweden-Models/Llama-3-8B-instruct`
- Anthropic API key for Claude generation, set as the `ANTHROPIC_API_KEY` environment variable:
  ```bash
  export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
  ```

### Install Dependencies

```bash
pip install transformers accelerate datasets trl peft bitsandbytes protobuf sentencepiece sentence-transformers faiss-cpu anthropic gradio folium playwright
```

### Step 1: Build the RAG Index

```bash
python build_rag_index_swe.py
```
Indexes all 95 documents from `markdown_output/`. Uses GPU for embedding (batch=64).

### Step 2: Generate Training Data

```bash
python generate_rag_traindata_swe.py
```
Produces 450 Swedish defence Q&A training examples.

### Step 3: Fine-Tune the Model (~25 minutes on RTX 4060)

```bash
python rag_peft.py
```

### Step 4: Evaluate

```bash
python eval_rag_extended_swe.py
```

### Step 5: Interactive Q&A

```bash
python ask_rag_swe.py
```

### Step 6: Web Interface (recommended)

```bash
python web_interface.py
```
Opens at **http://localhost:7860** with:
- Editable scenario panel (left) — modify forces, terrain, objectives
- Question input (right) — ask any question in Swedish
- Claude generation with source citations
- Progress indicator — shows retrieval → classification → generation stages
- Scrollable answer with source citations

### Step 7: Single Question (CLI)

```bash
python test_question.py "Vad är totalförsvar?"
python test_question.py --claude "Vilka vapensystem har Amfibiebataljonen?"
```

### Step 8: Tactical Scenario Simulator (CLI)

```bash
python scenario_simulator.py              # local model
python scenario_simulator.py --claude     # Claude API
```

---

## Web Interface

The web interface (`web_interface.py`) provides the easiest way to interact with the system:

```bash
python web_interface.py           # local only: http://localhost:7860
python web_interface.py --share   # + public gradio.live link (simulation disabled)
```

**Features:**
- **Scenario editor** — editable textbox with the default Amf 4 vs VDV scenario. Modify freely.
- **Question input** — type any question in Swedish and press Enter or click "Fråga"
- **Claude generation** — all answers generated by Claude API at temperature=0.0 for reproducibility
- **Local classifier gate** — fine-tuned AI-Sweden Llama 3 8B decides JA/NEJ before Claude is called (prevents parametric leakage)
- **Progress indicator** — shows current stage (🔍 searching → 🧠 classifying → ✍️ generating → ✅ done)
- **Scrollable answer** — formatted markdown with source citations and classification result
- **Tactical map** — click "🗺️ Visa taktisk karta" to display an interactive OpenStreetMap with:
  - NATO APP-6 unit symbols (blue rectangles = friendly, red diamonds = hostile)
  - Bridge and tunnel demolition targets with explosive requirements
  - Recommended minefield positions with area-of-effect circles
  - Phase lines (PL ALFA, BRAVO, CHARLIE)
  - Enemy advance axes and CB90 waterborne manoeuvre routes
  - Click any symbol for detailed popup with ORBAT/equipment data
- **Tactical simulation** — click "▶️ Kör simulering (48h)" to launch a 48-hour time-stepped simulation:
  - Opens in a new browser tab at http://localhost:7861/simulation.html
  - Animated map showing VDV advance, mining, bridge demolitions, RBS-17 engagements, and urban combat
  - Play/pause, step forward/back, timeline slider, and speed controls (1×/2×/4×)
  - Event log with combat reports and engineering actions
  - VDV vehicle strength tracker (BMD-4M losses over time)
  - **Note:** simulation is disabled when running with `--share` (insufficient system RAM for the tunnel + simulation together)

**Share mode:**
Running with `--share` creates a public `https://xxxxx.gradio.live` URL that anyone can use for Q&A.
Simulation is automatically disabled in share mode to prevent OOM crashes (the Gradio tunnel plus
simulation generation together exceed available system RAM). Run without `--share` for local-only
access with simulation enabled.

**Dependencies:** Requires `gradio` and `folium` (`pip install gradio folium`)

---

## Scenario Simulator

The scenario simulator generates a **doctrine-grounded animated tactical simulation** directly from the scenario text.

### How it works:
1. You edit the scenario in the textbox (any forces, any location, any terrain)
2. Click **"▶️ Kör simulering"**
3. Claude identifies what Swedish doctrine is needed (Step 1 — ~3s)
4. RAG retrieves matching doctrine chunks from the 95-document corpus
5. Claude generates 24 time-frames with unit movements, combat, and engineering events grounded in doctrine (Step 2 — ~20s)
6. An animated map opens in a new tab with play/pause, timeline slider, and event log

### Doctrine grounding:
- **[SWE: DocumentName]** — events based on Swedish doctrine from the corpus
- **(NATO)** — fallback to open-source NATO doctrine when Swedish docs don't cover a topic
- Enemy force TTPs use open-source adversary doctrine

### Features:
- Fully dynamic — change the scenario text and re-run for a different simulation
- 24 frames (2-hour steps over 48 hours)
- Left sidebar with scrollable event history (current events highlighted)
- Play/pause, step forward/back, timeline slider, speed controls (1×/2×/4×)
- Event log saved to `simulation_log.md` after each run
- **Video export** — click "📹 Exportera video" to render simulation as MP4 with map + events overlay
- Works with ANY scenario — different countries, forces, terrain

---

## UND — Intelligence Module

The UND module (`und.py`) tracks all entities in the simulation and supports runtime intelligence injection.

```python
from und import UND
intel = UND()
intel.load_scenario()  # Populates with Amf 4 vs VDV BTG scenario

# Inject observations during simulation
intel.set_sim_time("H+06")
intel.inject("VDV_MAIN", "position", [57.665, 12.20], source="KJ spaning", confidence=0.9)
intel.inject("BRO1_RV40", "status", "destroyed", source="Pioneer rapport", confidence=1.0)
intel.inject("MF1_MOLNLYCKE", "cleared", True, source="UAV observation", confidence=0.8)

# Query current state
print(intel.get_situation_summary())
intel.save("und_state.json")
```

**Tracked entities:**
- 6 friendly units (Amfbat + subunits: KP1, KP2, KJ, RBS, Pioneer)
- 3 hostile units (VDV BTG + subunits: Main, Flank)
- 5 structures (3 bridges, 1 tunnel, 1 airfield)
- 7 strategic locations (3 minefields, 3 phase lines, 1 objective)

**Features:**
- Time-stamped intelligence reports with source and confidence level
- Low-confidence reports stored but don't update assessed state
- Full history for each entity (audit trail)
- JSON export for simulation engine consumption
- Designed for future integration with voice-to-text intelligence injection (see `DESIGN_voice_intel.md`)

**Run tests:**
- `python test_und.py` (16 tests)
- `python test_rag_retrieval.py` (11 tests)

### Scenario
- **Situation:** Swedish Amfibiebataljon (Amf 4, Göteborg) defending against a Russian VDV BTG airlifted to Landvetter airport
- **Swedish forces:** ~400 troops — CB90 assault craft, RBS-17, Carl Gustaf, Kustjägare, 81mm mortars, limited mines/explosives
- **Russian forces:** VDV BTG (~600 troops) — BMD-4M, Sprut-SD, 2S9 Nona-S, no heavy armour, no IMR-2
- **Terrain:** Göteborg–Landvetter corridor, 20km, urban+forest, Mölndalsån bridges, Rv40 tunnel
- **Engineering norms:** Detailed mine laying rates, bridge demolition data, obstacle clearing times

### Anti-Hallucination
The simulator requires citation of sources:
- `[Scenario]` — for facts from the injected ORBAT/terrain data
- `[Doktrin källa X]` — for facts from retrieved Swedish doctrine documents
- Refuses to fabricate numbers or capabilities not present in sources

### Output
Answers are saved to `scenario_answer_claude.md` or `scenario_answer_local.md` depending on the generation backend.

---

## Evaluation Results

Extended evaluation (20 questions per category):

| Category | Score | Notes |
|----------|-------|-------|
| Answerable accuracy | **95%** (19/20) | Correct fact extraction from Swedish military context |
| Refusal accuracy | **85%** (17/20) | Refuses unanswerable/out-of-scope questions |
| False refusal rate | **0%** (0/20) | No over-refusal |
| Parametric leakage | **0%** (0/10) | Claude production path with classifier gate, temperature=0.0 |
| Retrieval hit rate | **95%** (19/20) | Hybrid search finds relevant chunks across 95 documents |

### Claude-path evaluation (production pipeline)

The production path (local classifier gate → Claude generation at temperature=0.0) was
independently tested against refusal, false-refusal, and parametric leakage:

| Metric | Score |
|--------|-------|
| Refusal accuracy | **100%** (20/20) |
| False refusal rate | **0%** (0/20) |
| Parametric leakage | **0%** (0/10) |

### Reproducibility (temperature=0.0 vs 1.0)

Setting `temperature=0.0` on all Claude calls improved answer consistency without
affecting leakage or refusal metrics:

| Metric | temp=1.0 (old) | temp=0.0 (new) | Gain |
|--------|---------------|----------------|------|
| Character similarity | 0.378 | 0.499 | +32% |
| Token Jaccard | 0.363 | 0.426 | +17% |
| Length variability | 576 chars | 373 chars | -35% |

---

## Key Design Decisions

### Shared RAG Retrieval Module
All retrieval operations (Q&A, evaluation, simulation doctrine lookup) use the single `rag_retrieval.py` module. This eliminates code duplication and ensures consistent retrieval behavior across all components. Any retrieval improvement applies everywhere automatically.

### Process Cleanup on Startup
The web interface automatically kills any leftover instances of itself and frees ports 7860/7861 on startup. This prevents zombie processes from consuming memory and causing OOM kills.

### Swedish-Native Model
AI-Sweden Llama 3 8B-Instruct is continued pre-training of Llama 3 8B on Swedish text. It handles Swedish military terminology naturally without translation artifacts.

### Multi-Backend Generation
The local 8B model serves as classifier (JA/NEJ gate to prevent parametric leakage). All answer generation is handled by Claude (Anthropic API) for maximum quality.

### Large Corpus Handling (95 documents, ~17,950 chunks)
GPU-accelerated embedding (batch=64) for indexing speed. CPU embedding for queries to leave GPU free for the LLM during inference.

### 2:1 Answerable-to-Refusal Training Ratio
The 8B model learns refusal patterns quickly due to its larger capacity. A 2:1 ratio prevents over-refusal while maintaining strong refusal accuracy (0% false refusal rate).

### LoRA Adapter (not merged)
The 8B model is too large to merge and reload in 8GB VRAM. The LoRA adapter is saved separately (~30MB) and loaded on top of the 4-bit quantized base model at inference time.

### JA/NEJ Detection
The Swedish model responds "Ja"/"Nej" instead of "YES"/"NO". The classifier checks for both Swedish and English affirmative responses.

### Anti-Parametric-Leakage Prompt
Forces source citations (`[Scenario]` or `[Doktrin källa X]`) to prevent the model from generating plausible-sounding but unsourced military information.

### Swedish Stopword Filtering
BM25 tokenization filters common Swedish function words for better keyword matching on military/technical text.

### Swedish Query Expansion
Rule-based rephrasings catch terminology mismatches across the diverse 95-document corpus.

---

## Performance

| Metric | Value |
|--------|-------|
| Training time | ~25 minutes (RTX 4060 8GB) |
| Classifier model | AI-Sweden Llama 3 8B-Instruct, QLoRA 4-bit |
| Generator options | Claude API (Anthropic), temperature=0.0 |
| LoRA rank | 16 |
| Training examples | 450 (2:1 answerable-to-refusal) |
| Training epochs | 2 |
| Embedding model | BAAI/bge-m3 (1024-dim, multilingual) |
| Embedding device | GPU for indexing (batch=64), CPU for queries |
| Cross-encoder | ms-marco-MiniLM-L-12-v2 (CPU) |
| Document corpus | 95 markdown files, ~17,950 chunks |
| Retrieval | Hybrid (vector + BM25) → cross-encoder rerank → top 3 |
| VRAM usage | ~5GB inference (4-bit base + LoRA adapter) |

---

## Quick Start — Unpack and Run from Scratch

### 1. Clone the repo and unpack

```bash
git clone https://github.com/henrikstolpe/defence-rag-swe.git
cd defence-rag-swe

# Unpack the tarball (contains all scripts, training data, and documents)
tar -xzf rag_peft_project.tar.gz
```

This extracts:
- All Python scripts in the current directory (`./`)
- The document corpus in `home/henrikstolpe/Documents/The Future of Warfare/markdown_output/`

Move the documents to a convenient location:
```bash
mkdir -p markdown_output
mv home/henrikstolpe/Documents/The\ Future\ of\ Warfare/markdown_output/* markdown_output/
rm -rf home/
```

### 2. Update the document path in `build_rag_index_swe.py`

Open `build_rag_index_swe.py` and update the `DOCUMENT_FOLDER` path to point to your local `markdown_output/` folder:

```python
DOCUMENT_FOLDER = "markdown_output"  # or the full path to your documents
```

### 3. Install dependencies

```bash
pip install transformers accelerate datasets trl peft bitsandbytes \
    protobuf sentencepiece sentence-transformers faiss-cpu anthropic gradio folium
```

### 4. Set the Anthropic API key (REQUIRED for Claude generation)

**This is a mandatory manual step.** All answer generation (web interface, simulation,
Claude-path evals) calls the Claude API, which reads the key from the `ANTHROPIC_API_KEY`
environment variable. Nothing is hardcoded — if the variable is not set, those components
will fail.

Set it in your shell:

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

To make it permanent across terminal sessions, append the same line to your shell profile:

```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

Verify it is set (should print a non-zero length, not the key itself):

```bash
echo "${#ANTHROPIC_API_KEY}"   # e.g. 108
```

> The local classifier (JA/NEJ gate) runs fully offline and does **not** need the key.
> Only the Claude generation step requires it.

### 5. Authenticate with HuggingFace (REQUIRED for the base model)

The base model `AI-Sweden-Models/Llama-3-8B-instruct` is downloaded from HuggingFace on
first run. Log in once so the download is authorized:

```bash
pip install huggingface_hub
huggingface-cli login   # paste a token from https://huggingface.co/settings/tokens
```

### 6. Set up Kiro workspace

1. Open **Kiro IDE**
2. **File → Open Folder** → select the `defence-rag-swe/` directory
3. If you also want the markdown documents visible in the explorer, add them as a second workspace folder:
   **File → Add Folder to Workspace** → select `markdown_output/`
4. Save the workspace: **File → Save Workspace As** → `rag_peft_project.code-workspace`

### 7. Run the full pipeline

Run each step in the Kiro terminal (`Ctrl+` ` to open terminal):

```bash
# Step 1: Build the RAG index (indexes all 95 documents, ~12 min on GPU)
python build_rag_index_swe.py

# Step 2: Generate training data
python generate_rag_traindata_swe.py

# Step 3: Fine-tune the model (~25 min on RTX 4060 8GB)
python rag_peft.py

# Step 4: Evaluate the system
python eval_rag_extended_swe.py

# Step 5: Interactive Q&A (local classifier + Claude generation)
python ask_rag_swe.py
```

### 8. Use the scenario simulator

```bash
# Ask a tactical question using the local model
python scenario_simulator.py "Ta fram plan för fördröjande fältarbete"

# Use Claude API (requires ANTHROPIC_API_KEY env var — see step 4)
python scenario_simulator.py --claude "Beskriv fördröjningsstrid mot VDV BTG"
```

### 9. Single question mode (without scenario)

```bash
python test_question.py "Vad är totalförsvaret?"
python test_question.py --claude "Vilken roll har drönare i markstriden?"
```

### 10. Launch the web interface

```bash
python web_interface.py           # local only: http://localhost:7860
python web_interface.py --share   # + public gradio.live link (simulation disabled)
```

---

## Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA with 8GB+ VRAM (tested on RTX 4060) |
| Python | 3.10+ |
| HuggingFace | Access to `AI-Sweden-Models/Llama-3-8B-instruct` |
| Disk space | ~15 GB (model downloads + index) |
| RAM | 16 GB recommended |

### API Key (required for Claude generation)

The system reads the Anthropic API key from the environment variable `ANTHROPIC_API_KEY`.
Set it before running any Claude-based component (web interface, simulation, eval):

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

To make it persistent, add the line to your `~/.bashrc` or `~/.profile`.

---

## Troubleshooting

### NVIDIA driver issues
If `nvidia-smi` fails after a crash, rebuild DKMS modules:
```bash
sudo dkms build nvidia/<version> -k $(uname -r) --force
sudo dkms install nvidia/<version> -k $(uname -r)
sudo modprobe nvidia
```

### Secure Boot blocking GPU driver
If you see "Key was rejected by service", configure DKMS to sign with an enrolled MOK key:
```bash
# Create and enroll a signing key
sudo mkdir -p /var/lib/shim-signed/mok
sudo openssl req -new -x509 -newkey rsa:2048 -keyout /var/lib/shim-signed/mok/MOK.priv \
  -outform DER -out /var/lib/shim-signed/mok/MOK.der -nodes -days 36500 -subj "/CN=NVIDIA/"
sudo mokutil --import /var/lib/shim-signed/mok/MOK.der

# Tell DKMS to use this key
echo 'mok_signing_key=/var/lib/shim-signed/mok/MOK.priv' | sudo tee /etc/dkms/framework.conf.d/mok-signing.conf
echo 'mok_certificate=/var/lib/shim-signed/mok/MOK.der' | sudo tee -a /etc/dkms/framework.conf.d/mok-signing.conf

# Reboot, enroll MOK at blue screen, then rebuild
sudo dkms build nvidia/<version> -k $(uname -r) --force
sudo dkms install nvidia/<version> -k $(uname -r)
```

### Out of memory during training
Reduce context length in `rag_peft.py`:
```python
max_length=384  # reduce from 512
```

### Model says "Nej" to everything
Ensure the `classify_answerable` function checks for both Swedish and English:
```python
return "YES" in response or "JA" in response
```
