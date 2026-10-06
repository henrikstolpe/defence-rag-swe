# Maintainability Analysis — How to Make This Codebase Easier to Extend

## Current State

The project has grown organically from a RAG experiment to a multi-component system with:
- RAG pipeline (index, retrieval, classification, generation)
- Tactical scenario simulator (Claude-generated timelines)
- Web interface (Gradio + separate HTTP server)
- Intelligence module (UND)
- Video export
- Scenario parsing

**Total lines of code:** ~2,500+ across 15+ Python files
**Main problem:** `web_interface.py` is a single 1,500+ line file that contains the RAG infrastructure, LLM loading, all UI definitions, the simulation renderer, and HTTP server logic. Any change risks breaking unrelated functionality.

---

## Key Problems

### 1. `web_interface.py` is a monolith

This file does everything:
- Loads RAG components (FAISS, embeddings, BM25)
- Loads the LLM (transformers, PEFT)
- Defines the Q&A pipeline (classify, generate)
- Defines the Gradio UI layout
- Contains the simulation frame renderer
- Runs HTTP servers
- Handles video export subprocess

**Risk:** Changing the simulation renderer might break the Q&A pipeline. Adding a UI button might break the LLM loading. Every edit touches a 1,500-line file where context is easy to lose.

### 2. No separation between data and logic

The scenario ORBAT data is embedded in multiple places:
- `scenario_simulator.py` — has its own copy of the scenario
- `web_interface.py` — has DEFAULT_SCENARIO
- `und.py` — has `load_scenario()` with the same data
- They drift apart when one is updated but not the others

### 3. No tests for the web interface or simulation

The `test_und.py` tests the UND module, but:
- No tests for RAG retrieval quality
- No tests for the simulation generator
- No tests for the web interface logic
- No integration tests

### 4. Shared state via global variables

The RAG components (index, chunks, embed_model) are loaded at module import time in `web_interface.py`. The simulation generator has its own `_rag_cache` dict. This makes it hard to test components in isolation.

### 5. Duplicated RAG code

The retrieval pipeline (BM25 + vector + reranker) is implemented in:
- `ask_rag_swe.py`
- `eval_rag_extended_swe.py`
- `test_question.py`
- `scenario_simulator.py`
- `web_interface.py`
- `sim_generator.py`

Any bug fix must be applied in 6 places.

### 6. API keys hardcoded

The Anthropic API key is written directly in `scenario_simulator.py`, `sim_generator.py`, `test_question.py`, and `web_interface.py`. Changing it requires editing 4 files.

---

## Recommended Refactoring Steps

### Step 1: Extract RAG retrieval into a shared module

Create `rag_retrieval.py`:
```python
class RAGRetriever:
    def __init__(self, index_dir="./rag_index_swe", embed_model="BAAI/bge-m3"):
        # Load FAISS, chunks, sources, embed_model, reranker, BM25
        ...
    
    def retrieve(self, query: str, top_k=3) -> list[dict]:
        # Expand query, hybrid search, rerank, return top-k
        ...
```

All scripts import from this single module. Bug fixes apply once.

**Effort:** 2-3 hours
**Impact:** Eliminates 6× duplicated code

### Step 2: Extract LLM loading into a shared module

Create `llm.py`:
```python
class LocalLLM:
    def __init__(self, model_path, base_model_id):
        # Load base + LoRA adapter
        ...
    
    def classify(self, question, context) -> bool:
        ...
    
    def generate(self, question, context) -> str:
        ...

class ClaudeLLM:
    def __init__(self, api_key=None):
        # Use env var or passed key
        ...
    
    def generate(self, question, context, system_prompt) -> str:
        ...
```

**Effort:** 2 hours
**Impact:** LLM logic separated from UI; testable in isolation

### Step 3: Extract API keys to a config file

Create `config.py`:
```python
import os
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
BASE_MODEL_ID = "AI-Sweden-Models/Llama-3-8B-instruct"
MODEL_PATH = "./eupolicy_rag_llama_model_swe"
INDEX_DIR = "./rag_index_swe"
```

Or use a `.env` file with `python-dotenv`.

**Effort:** 30 minutes
**Impact:** Single place to manage configuration; no secrets in code

### Step 4: Extract scenario data into a JSON file

Create `scenarios/default_goteborg.json`:
```json
{
  "name": "Göteborg — Amf 4 vs VDV BTG",
  "friendly": {...},
  "hostile": {...},
  "terrain": {...},
  "structures": [...],
  "minefields": [...]
}
```

All scripts load from this file. UND's `load_scenario()` reads it. The web interface's DEFAULT_SCENARIO is generated from it.

**Effort:** 1-2 hours
**Impact:** Single source of truth for scenario data

### Step 5: Split `web_interface.py` into modules

```
web_interface.py          → 200 lines (Gradio UI layout + button handlers only)
web_qa.py                 → Q&A pipeline (ask_question function)
web_simulation.py         → Simulation rendering (show_simulation function)
web_video.py              → Video export logic
```

Each module exports a single function that the UI calls.

**Effort:** 3-4 hours
**Impact:** Each component can be modified independently

### Step 6: Add integration tests

Create `tests/`:
```
tests/
├── test_retrieval.py     — Does retrieval find correct chunks?
├── test_classifier.py    — Does YES/JA detection work?
├── test_simulation.py    — Does sim_generator produce valid frames?
├── test_und.py           — Already exists, move here
└── test_scenario_parser.py — Does parsing produce valid UND data?
```

**Effort:** 3-4 hours
**Impact:** Catch regressions before they hit the UI

### Step 7: Add type hints and docstrings

The current code has inconsistent typing. Adding proper type hints helps Kiro understand function signatures and catch errors:

```python
def retrieve(query: str, top_k: int = 15, final_k: int = 3) -> list[dict[str, any]]:
    """Retrieve top-k chunks using hybrid search + reranking.
    
    Returns list of dicts with keys: text, score, idx, source
    """
```

**Effort:** 2-3 hours
**Impact:** Better IDE support, clearer interfaces for Kiro

---

## Priority Order

| Step | Effort | Impact on bugs | Impact on extensibility |
|------|--------|---------------|------------------------|
| 1. Shared RAG module | 2-3h | High (eliminates 6× duplication) | High |
| 3. Config file | 30min | Medium (no more hardcoded keys) | Medium |
| 5. Split web_interface.py | 3-4h | High (isolated components) | Very High |
| 4. Scenario JSON file | 1-2h | Medium (single source of truth) | High |
| 2. Shared LLM module | 2h | Medium | High |
| 6. Integration tests | 3-4h | Very High (catches regressions) | Medium |
| 7. Type hints | 2-3h | Low | Medium |

**Recommended order:** 3 → 1 → 5 → 4 → 6 → 2 → 7

Total effort: ~15-20 hours of refactoring for a significantly more maintainable codebase.

---

## What This Enables for Kiro

After refactoring:
- **Adding a new feature** (e.g., voice intel) only touches the new module + a small button handler in the UI
- **Fixing a retrieval bug** means changing ONE file, not six
- **Changing the LLM** means editing `config.py` and possibly `llm.py`, nothing else
- **Adding a new scenario** means creating a JSON file, no code changes
- **Testing** catches issues before they become UI bugs
- **Context window** stays manageable — Kiro reads one 200-line file instead of a 1,500-line monolith

---

## What NOT to Change

- The overall architecture (RAG + classifier + generator + simulation) is sound
- The UND module is already well-structured and tested
- The sim_generator's two-step approach (identify doctrine → retrieve → generate) works well
- The Gradio interface choice is appropriate for the use case

The structure is good — it just needs to be decomposed into smaller, testable pieces.
