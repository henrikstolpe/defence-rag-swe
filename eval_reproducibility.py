"""
Reproducibility evaluation for Claude generation.

Measures how consistent Claude's answers are when the SAME question + context
is asked multiple times. Compares the old behavior (temperature=1.0, the API
default) against the new behavior (temperature=0.0).

We isolate the GENERATION step: a fixed context and question are sent to Claude
N times at each temperature. We then measure pairwise similarity between the
answers. Higher similarity = more reproducible.

Similarity metrics (no external deps):
  - char_ratio: difflib SequenceMatcher ratio on raw text (0..1)
  - token_jaccard: Jaccard overlap of word sets (0..1)

Run: python3 eval_reproducibility.py
"""
import os
import json
import difflib
from itertools import combinations
import anthropic

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

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

# Fixed context so retrieval is not a variable — we isolate generation determinism.
FIXED_CONTEXT = """[Doktrin källa 1, Fördröjningsstrid]: Fördröjningsstrid syftar till att vinna tid
genom att tvinga motståndaren att gruppera och omgruppera upprepade gånger. Egna förband
undviker avgörande strid och rör sig mellan förberedda fördröjningslinjer. Fältarbeten som
minering och brosprängning används för att skapa hinder och förlänga motståndarens
framryckningstid.

[Doktrin källa 2, Minering]: Minering av en spärr tar normalt 1-2 timmar för en mineringspluton.
Röjning av en oförberedd spärr tar motståndaren betydligt längre tid utan specialmateriel."""

QUESTION = "Hur genomför egna förband fördröjningsstrid enligt doktrinen?"

N_RUNS = 4  # runs per temperature setting


def call_claude(temperature: float) -> str:
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Kontext:\n{FIXED_CONTEXT}\n\nFråga: {QUESTION}"}],
    )
    return message.content[0].text


def char_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def token_jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def evaluate(temperature: float) -> dict:
    label = f"temperature={temperature}"
    print(f"\n=== Running {N_RUNS} generations at {label} ===")
    answers = []
    for i in range(N_RUNS):
        print(f"  Run {i+1}/{N_RUNS}...")
        answers.append(call_claude(temperature))

    char_scores = []
    jacc_scores = []
    for (i, a), (j, b) in combinations(enumerate(answers), 2):
        cr = char_ratio(a, b)
        jc = token_jaccard(a, b)
        char_scores.append(cr)
        jacc_scores.append(jc)
        print(f"  pair ({i+1},{j+1}): char_ratio={cr:.3f}  token_jaccard={jc:.3f}")

    lengths = [len(a) for a in answers]
    result = {
        "temperature": temperature,
        "n_runs": N_RUNS,
        "mean_char_ratio": round(sum(char_scores) / len(char_scores), 4),
        "min_char_ratio": round(min(char_scores), 4),
        "mean_token_jaccard": round(sum(jacc_scores) / len(jacc_scores), 4),
        "min_token_jaccard": round(min(jacc_scores), 4),
        "answer_lengths": lengths,
        "length_spread": max(lengths) - min(lengths),
        "identical_count": sum(1 for a, b in combinations(answers, 2) if a == b),
        "total_pairs": len(char_scores),
    }
    return result


def main():
    print("Reproducibility evaluation: same question + context, repeated generations.")
    print(f"Question: {QUESTION}")

    old = evaluate(1.0)   # old default behavior
    new = evaluate(0.0)   # new behavior after the change

    print("\n\n============================================================")
    print("RESULTS SUMMARY")
    print("============================================================")
    for r in (old, new):
        print(f"\ntemperature = {r['temperature']}")
        print(f"  mean char similarity : {r['mean_char_ratio']:.3f}  (min {r['min_char_ratio']:.3f})")
        print(f"  mean token Jaccard   : {r['mean_token_jaccard']:.3f}  (min {r['min_token_jaccard']:.3f})")
        print(f"  answer length spread : {r['length_spread']} chars  {r['answer_lengths']}")
        print(f"  identical pairs      : {r['identical_count']}/{r['total_pairs']}")

    char_gain = new["mean_char_ratio"] - old["mean_char_ratio"]
    jacc_gain = new["mean_token_jaccard"] - old["mean_token_jaccard"]
    print("\n------------------------------------------------------------")
    print(f"IMPROVEMENT (new - old):")
    print(f"  char similarity  : {char_gain:+.3f}")
    print(f"  token Jaccard    : {jacc_gain:+.3f}")
    improved = char_gain > 0.01 or jacc_gain > 0.01
    print(f"  VERDICT          : {'IMPROVED ✅' if improved else 'NO IMPROVEMENT ❌'}")
    print("============================================================")

    with open("eval_reproducibility_results.json", "w", encoding="utf-8") as f:
        json.dump({"old": old, "new": new,
                   "char_gain": round(char_gain, 4),
                   "token_jaccard_gain": round(jacc_gain, 4),
                   "improved": improved}, f, indent=2, ensure_ascii=False)
    print("\nResults written to eval_reproducibility_results.json")


if __name__ == "__main__":
    main()
