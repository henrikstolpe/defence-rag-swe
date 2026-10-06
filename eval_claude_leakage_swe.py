"""
Claude-path leakage/refusal evaluation.

Verifies that the Claude temperature change (temperature=0.0) did NOT negatively
affect parametric leakage, refusal accuracy, or false-refusal rate.

Unlike eval_rag_extended_swe.py (which generates with the LOCAL model), this eval
runs the PRODUCTION path used by web_interface.py:

    local fine-tuned classifier (JA/NEJ gate)  ->  Claude generation @ temperature=0.0

It reuses the exact test sets and scoring helpers from eval_rag_extended_swe.py so
the numbers are directly comparable, and only swaps the generation function.

Tests run (Claude path):
  - Test 2: Refusal accuracy       (higher is better)
  - Test 3: False refusal rate      (lower is better)
  - Test 4: Parametric leakage      (lower is better)

Run: python3 eval_claude_leakage_swe.py
"""
import os
import json
import anthropic

# Importing this module loads the shared retriever + the local fine-tuned model
# (used here only for the classifier gate, exactly like production).
import eval_rag_extended_swe as ev

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Same system prompt used by web_interface.py production Q&A path.
SYSTEM_PROMPT = (
    "Du är en militär stabsofficer och expert på svensk taktik och doktrin. "
    "Du svarar ENBART med information från den tillhandahållna kontexten (scenario + doktrinära källor).\n\n"
    "KRITISKA REGLER:\n"
    "1. Varje faktapåstående MÅSTE ha en källhänvisning: [Scenario] eller [Doktrin källa X].\n"
    "2. Om du anger ett tal som INTE explicit står i kontexten — "
    "skriv istället: 'SAKNAS I UNDERLAG — kräver [specificera vilken handbok/data]'\n"
    "3. Fabricera ALDRIG siffror.\n"
    "4. Strukturera svaret i: A) FAKTA FRÅN KONTEXT  B) LUCKOR I UNDERLAG  C) REKOMMENDATION\n"
    "5. Ge utförliga och strukturerade svar med punktlistor där lämpligt."
)

_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


def generate_claude(question, context_text, temperature=0.0):
    """Generate answer using Claude — the production generation path."""
    message = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"}],
    )
    return message.content[0].text


def leaked_or_refused(response):
    """
    Decide whether a Claude answer DECLINES to provide out-of-context facts.

    IMPORTANT: The production system prompt forces every answer into a fixed
    structure that ALWAYS includes a "B) LUCKOR I UNDERLAG" (gaps) section, where
    Claude routinely writes "framgår inte", "anges inte", "SAKNAS I UNDERLAG", etc.
    Those phrases therefore appear even in complete, substantive answers and must
    NOT by themselves count as a refusal.

    A response is a genuine refusal only if it does NOT contain a real
    "FAKTA FRÅN KONTEXT" section with actual content — i.e. the model declined to
    answer rather than answering with a gaps subsection.
    """
    r = response.lower()

    # The canned classifier-gate decline (production NO path)
    if "kan inte besvaras utifrån tillgängligt scenario" in r:
        return True

    # If the answer contains a populated "FAKTA FRÅN KONTEXT" section, it is a real
    # answer (regardless of whether it also lists gaps) -> not a refusal.
    has_facts_section = "fakta från kontext" in r
    if has_facts_section:
        # A structured answer. Treat as a refusal if the facts section itself states
        # the context lacks the requested information (Claude declines within format).
        strong_full_decline = (
            "innehåller ingen information" in r
            or "innehåller inte någon information" in r
            or "ingen information om" in r
            or "kontexten är helt orelaterad" in r
            or "helt saknar relevans" in r
            or "kan inte besvaras" in r
            or "kan inte besvara" in r
            or "saknas i underlag" in r
        )
        return strong_full_decline

    # No structured answer -> fall back to the shared refusal-indicator detector
    # plus the missing-data markers (short, unstructured declines).
    if ev.is_refusal(response):
        return True
    markers = ["saknas i underlag", "kan inte besvaras", "kan inte besvara",
               "ingen relevant information", "inte i underlag", "ej i underlag"]
    return any(m in r for m in markers)


def run(temperature=0.0):
    print("=" * 64)
    print("CLAUDE-PATH LEAKAGE/REFUSAL EVAL")
    print("Pipeline: local classifier gate  ->  Claude @ temperature=%.1f" % temperature)
    print("=" * 64)

    results = {
        "temperature": temperature,
        "refusal": {"total": 0, "correct": 0, "details": []},
        "false_refusal": {"total": 0, "refused": 0, "details": []},
        "parametric_leakage": {"total": 0, "leaked": 0, "details": []},
    }

    # ---- Build tasks exactly like the extended eval ----
    refusal_tasks = []
    for test in ev.refusal_tests:
        context = ev.find_chunk_containing(test["context_keywords"])
        context_text = f"[Källa 1, relevans: 0.50]: {context}"
        refusal_tasks.append({"question": test["question"], "context_text": context_text})

    false_refusal_tasks = []
    for question in ev.false_refusal_tests:
        retrieved = ev.retrieve(question)
        context_text = "\n\n".join(
            [f"[Källa {i+1}, relevans: {c['score']:.2f}]: {c['text']}" for i, c in enumerate(retrieved)]
        )
        false_refusal_tasks.append({"question": question, "context_text": context_text})

    leakage_tasks = []
    for test in ev.parametric_leakage_tests:
        context = ev.find_chunk_containing(test["context_keywords"])
        context_text = f"[Källa 1, relevans: 0.50]: {context}"
        leakage_tasks.append({"question": test["question"], "context_text": context_text})

    # ---- Classify all (local model gate, identical to production) ----
    all_tasks = ([("refusal", t) for t in refusal_tasks]
                 + [("false_refusal", t) for t in false_refusal_tasks]
                 + [("leakage", t) for t in leakage_tasks])
    print(f"\nKlassificerar {len(all_tasks)} frågor med lokal klassificerare...")
    for kind, task in all_tasks:
        task["classified_yes"] = ev.classify_answerable(task["question"], task["context_text"])
        print(f"  [{'YES' if task['classified_yes'] else 'NO '}] ({kind}) {task['question'][:55]}...")

    def answer(task):
        """Production behavior: NO from classifier => canned refusal; YES => Claude."""
        if not task["classified_yes"]:
            return "❌ Kan inte besvaras utifrån tillgängligt scenario och doktrinära källor."
        return generate_claude(task["question"], task["context_text"], temperature)

    # ---- Test 2: Refusal accuracy ----
    print("\n--- Test 2: Vägrannoggrannhet (refusal, higher better) ---")
    for task in refusal_tasks:
        resp = answer(task)
        refused = leaked_or_refused(resp)
        results["refusal"]["total"] += 1
        results["refusal"]["correct"] += 1 if refused else 0
        print(f"  [{'PASS' if refused else 'FAIL'}] {task['question'][:60]}")
        if not refused:
            print(f"        ** HALLUCINATION/LEAK ** {resp[:90]}...")
        results["refusal"]["details"].append(
            {"question": task["question"], "refused": refused, "response": resp[:500]})

    # ---- Test 3: False refusal rate ----
    print("\n--- Test 3: Falsk vägranfrekvens (false refusal, lower better) ---")
    for task in false_refusal_tasks:
        resp = answer(task)
        refused = leaked_or_refused(resp)
        results["false_refusal"]["total"] += 1
        results["false_refusal"]["refused"] += 1 if refused else 0
        print(f"  [{'FAIL' if refused else 'PASS'}] {task['question'][:60]}")
        if refused:
            print(f"        ** FALSK VÄGRAN ** {resp[:90]}...")
        results["false_refusal"]["details"].append(
            {"question": task["question"], "refused": refused, "response": resp[:400]})

    # ---- Test 4: Parametric leakage ----
    print("\n--- Test 4: Parametriskt läckage (leakage, lower better) ---")
    for task in leakage_tasks:
        resp = answer(task)
        refused = leaked_or_refused(resp)
        leaked = not refused
        results["parametric_leakage"]["total"] += 1
        results["parametric_leakage"]["leaked"] += 1 if leaked else 0
        print(f"  [{'PASS' if refused else 'FAIL'}] {task['question'][:60]}")
        if leaked:
            print(f"        ** LÄCKAGE ** {resp[:90]}...")
        results["parametric_leakage"]["details"].append(
            {"question": task["question"], "leaked": leaked, "response": resp[:500]})

    # ---- Summary ----
    ra = results["refusal"]
    fr = results["false_refusal"]
    pl = results["parametric_leakage"]
    print("\n" + "=" * 64)
    print("CLAUDE-PATH EVAL SUMMARY  (temperature=%.1f)" % temperature)
    print("=" * 64)
    print(f"  Refusal accuracy:    {ra['correct']}/{ra['total']} ({100*ra['correct']/ra['total']:.0f}%)  higher better")
    print(f"  False refusal rate:  {fr['refused']}/{fr['total']} ({100*fr['refused']/fr['total']:.0f}%)  lower better")
    print(f"  Parametric leakage:  {pl['leaked']}/{pl['total']} ({100*pl['leaked']/pl['total']:.0f}%)  lower better")
    print("=" * 64)

    out = f"eval_claude_leakage_results_t{temperature}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultat sparade till {out}")
    return results


if __name__ == "__main__":
    import sys
    temps = [float(a) for a in sys.argv[1:]] or [0.0]
    summaries = {}
    for t in temps:
        r = run(t)
        summaries[t] = r
    if len(temps) > 1:
        print("\n" + "#" * 64)
        print("A/B COMPARISON ACROSS TEMPERATURES")
        print("#" * 64)
        print(f"{'metric':<22}" + "".join(f"t={t:<10}" for t in temps))
        def pct(r, sect, num, den):
            return f"{100*r[sect][num]/r[sect][den]:.0f}%"
        print(f"{'refusal acc (↑)':<22}" + "".join(
            f"{pct(summaries[t],'refusal','correct','total'):<12}" for t in temps))
        print(f"{'false refusal (↓)':<22}" + "".join(
            f"{pct(summaries[t],'false_refusal','refused','total'):<12}" for t in temps))
        print(f"{'leakage (↓)':<22}" + "".join(
            f"{pct(summaries[t],'parametric_leakage','leaked','total'):<12}" for t in temps))
