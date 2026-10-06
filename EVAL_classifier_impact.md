# Classifier Impact Analysis

## Purpose

Measured the real impact of the fine-tuned local classifier (JA/NEJ gate) on system accuracy by running the full evaluation with and without the classifier active.

## Test Setup

- **With classifier:** Normal operation — local fine-tuned AI-Sweden Llama 3 8B classifies each question before generation
- **Without classifier:** Bypass — all questions are passed directly to generation (classifier always returns YES)
- **Generator:** Local 8B model (same for both tests)
- **Retrieval:** Identical (shared RAG module)
- **Test date:** 2026-08-24

## Results

| Category | With Classifier | Without Classifier | Delta |
|----------|----------------|-------------------|-------|
| Answerable accuracy | 95% (19/20) | 95% (19/20) | 0 |
| **Refusal accuracy** | **80% (16/20)** | **15% (3/20)** | **-65pp** |
| False refusal rate | 0% (0/20) | 0% (0/20) | 0 |
| **Parametric leakage** | **20% (2/10)** | **80% (8/10)** | **+60pp** |
| Retrieval hit rate | 95% (19/20) | 95% (19/20) | 0 |

## Key Findings

1. **Refusal accuracy drops from 80% to 15% without the classifier** — the model answers 13 additional questions it should refuse (comparisons with other countries, questions about specific companies not in context, external statistics).

2. **Parametric leakage jumps from 20% to 80%** — without the YES/NO gate, the model freely uses its pretraining knowledge about weapon system specs, company strategies, and real-world statistics that aren't in the provided context.

3. **Answerable accuracy is unaffected** — the classifier doesn't block valid questions (0% false refusal rate in both cases).

4. **Retrieval is unaffected** — the classifier operates after retrieval, so search quality is identical.

## Conclusion

The fine-tuned classifier is critical infrastructure for this system. It prevents:
- **65 percentage points** of hallucinated answers to out-of-scope questions
- **60 percentage points** of parametric leakage (using pretraining knowledge instead of context)

Without it, the system would answer confidently about topics not covered by the document corpus, using Claude/LLM general knowledge — exactly what we want to prevent in a grounded military decision support system.

## Architecture Implication

The classifier must remain as a local fine-tuned model (not replaced by a prompt-only approach) because:
- It was specifically trained on 450 examples of Swedish defence Q&A with explicit refusal patterns
- It understands Swedish comparison questions ("Hur jämför sig...", "Vilka skillnader...")
- It responds in Swedish ("Ja"/"Nej") requiring bilingual detection
- A prompt-only classifier via Claude would add API latency and cost for every single question
