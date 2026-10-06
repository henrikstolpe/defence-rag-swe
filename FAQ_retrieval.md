# FAQ: How Does the System Find the Right Information?

## Q: I have 95 documents. How does the system know which one to look in?

It doesn't look at whole documents — it looks at **small pieces** (chunks) from all documents at once. Each document is split into ~150-word pieces. Your 95 documents become ~18,000 small pieces. When you ask a question, the system searches all 18,000 pieces simultaneously and picks the 3 best matches.

---

## Q: How does it find the matching pieces?

It uses **two different search methods** and combines them:

### Method 1: Meaning search (vector/semantic)

Think of it like this: every chunk and every question gets converted into a long list of numbers (a "fingerprint" of its meaning). The system then compares the fingerprint of your question against the fingerprints of all 18,000 chunks and finds the ones that are most similar in meaning.

This is powerful because it understands **synonyms and related concepts**:
- You ask "lufthot" → it finds chunks about "flygstridskrafter", "luftanfall", "flyganfall"
- You ask "sprängning av bro" → it finds chunks about "demolering", "broförstöring"

### Method 2: Keyword search (BM25)

This is more like a traditional search engine. It looks for the **exact words** you used in your question and finds chunks that contain those same words.

This catches things that meaning search misses:
- Specific designations: "BMD-4M", "Rv 40", "Artikel 5"
- Numbers: "35 000 000", "8.8 km"
- Abbreviations: "CBRN", "VDV", "GrK"

### Combining them

The system runs both searches, then merges the results with a formula:

```
Final score = 60% meaning match + 40% keyword match
```

This gives you the best of both worlds.

---

## Q: What is the "cross-encoder reranking" step?

After the combined search finds ~15 candidate chunks, a separate AI model reads **each chunk together with your question** and scores how well they actually match.

The difference: the first search compares fingerprints independently (fast but approximate). The reranker reads the question and chunk together as a pair (slower but much more accurate).

Think of it like:
1. **First search** = quickly flipping through a book's index to find likely pages
2. **Reranking** = actually reading those pages to confirm which ones answer your question

The reranker picks the **top 3** from the 15 candidates.

---

## Q: Why only 3 chunks? Why not give the AI all 15?

Three reasons:
1. **Quality over quantity** — more chunks means more noise. The AI gets confused by irrelevant text.
2. **Token limits** — the AI can only read so much text at once. 3 chunks × 150 words = ~450 words of context, which is enough for a focused answer.
3. **Speed** — more context = slower generation.

---

## Q: What is "query expansion" and why does it help?

Before searching, the system rephrases your question 2-3 different ways:

```
Your question: "Vad är totalförsvar?"
Expanded to:
  1. "Vad är totalförsvar?"
  2. "Definiera totalförsvar"
  3. "Förklara totalförsvar"
```

This helps because the documents might use different wording than you. If you ask "Hur fungerar luftvärn?" but the document says "Luftvärnets uppgift är att...", the rephrased version "luftvärn uppgift funktion" has a better chance of matching.

---

## Q: Can it happen that the wrong chunks are selected?

Yes. This is the most common failure mode. Examples:

- **Ambiguous terms**: "bat" could match "bataljon" or "batteri" — the system might pick the wrong one.
- **Rare topics**: If only 2 out of 18,000 chunks mention your topic, and those chunks have low keyword overlap with your question, they might rank below irrelevant chunks.
- **Multi-topic questions**: "Jämför artilleri och luftvärn" — the system might find chunks about artilleri but miss the luftvärn chunks, or vice versa.

The **retrieval hit rate is 95%** — meaning 19 out of 20 times, the right information is in the top 3 chunks.

---

## Q: Does the AI ever answer from memory instead of the chunks?

This is called **parametric leakage**. The AI model has been trained on vast amounts of text and "knows" things about military topics. Sometimes it generates an answer from its training data instead of from the retrieved chunks.

We mitigate this by:
1. **Training the model to refuse** when context doesn't contain the answer
2. **Requiring source citations** in the answer ([Scenario] or [Doktrin källa X])
3. **Testing for leakage** — our evaluation explicitly checks for this (20% leakage rate with the strict citation prompt)

---

## Q: What if my question is about something not in any document?

The classifier (the first AI stage) determines if the retrieved chunks can answer your question. If it decides **no** — because the chunks are about a different topic, or because the question asks for specific data not present — it returns a refusal:

> "Den tillhandahållna kontexten innehåller inte tillräcklig information för att besvara denna fråga."

This happens for:
- Questions about other countries' militaries (comparison questions)
- Specific statistics not in the documents (budget numbers, personnel counts)
- Questions about named entities not in the corpus (specific companies, individuals)

---

## Q: How is a "chunk" defined? Why 150 words?

Each document is split into overlapping pieces of 150 words with 40 words of overlap between consecutive chunks.

```
Document: [word1 word2 word3 ... word150] [word111 word112 ... word260] [word221 ...]
           ├──── Chunk 1 ────────────────┤
                              ├──── Chunk 2 ────────────────────────────┤
                                                         ├──── Chunk 3 ...
```

**Why 150 words:**
- Legal/military text packs facts densely — 150 words usually contains one complete idea
- Shorter = more precise matching (the chunk is about ONE thing, not three)
- Shorter = the cross-encoder can judge relevance more accurately

**Why 40-word overlap:**
- Ensures no fact falls "between" two chunks
- A sentence that spans the boundary appears in both chunks

---

## Q: How fast is the search?

For 18,000 chunks:
- Vector search (FAISS): ~5 ms
- BM25 keyword search: ~50 ms
- Cross-encoder reranking (15 pairs): ~200 ms
- **Total retrieval: ~250 ms**

The slow part is the AI generation (3-15 seconds depending on model), not the search.

---

## Q: Can I see which chunks were selected?

Yes. Both `test_question.py` and `scenario_simulator.py` print the retrieved sources:

```
Hämtade 3 källor:
  1. [score=4.088] Källa: Pibat
  2. [score=1.996] Källa: Pibat  
  3. [score=1.731] Källa: ARTAKTIK
```

The web interface also shows sources below the answer. This lets you verify that the system found the right documents.

---

## Q: What happens if I add new documents to the corpus?

You need to rebuild the index:

```bash
python build_rag_index_swe.py
```

This re-chunks and re-embeds all documents (~12 minutes on GPU). The new content will then be searchable immediately. No retraining of the AI model is needed — the retrieval works independently of the model.
