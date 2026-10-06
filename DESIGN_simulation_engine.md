# Simulation Engine — How Events and Decisions Are Made

## Overview

The tactical simulation generates a 48-hour animated timeline by combining:
1. The user's scenario text (forces, terrain, objectives)
2. Retrieved Swedish military doctrine from the RAG corpus
3. Claude's military reasoning capabilities

This document explains exactly how each component contributes to the simulation output.

---

## The Two-Step Pipeline

### Step 1: Doctrine Identification (~3 seconds)

**What happens:**
Claude receives the scenario text and is asked: "What Swedish military doctrine topics are needed to simulate this?"

**Output:** A JSON array of 5-8 Swedish search queries, e.g.:
```json
[
  "fördröjningsstrid fördröjande fältarbeten",
  "brosprängning hinderläggning tidsåtgång",
  "strid bebyggelse urban terräng",
  "amfibieoperationer kustförsvar taktik",
  "minering pansarspärr fördröjningsoperation"
]
```

**Why this step exists:**
Generic retrieval (searching for "military doctrine") returns irrelevant chunks. By first identifying *what specific doctrine is needed*, the retrieval becomes precise. For example:
- A scenario about urban combat → retrieves "strid i bebyggelse" doctrine
- A scenario about delay operations → retrieves "fördröjningsstrid" and "fältarbeten" norms
- A scenario with amphibious forces → retrieves "amfibieoperationer" doctrine

### Step 2: Doctrine Retrieval (~2 seconds)

**What happens:**
Each of the 5-8 queries from Step 1 is used to search the 17,950-chunk FAISS index:
1. Vector search (BGE-M3 embedding) — finds semantically similar chunks
2. Cross-encoder reranking — scores each (query, chunk) pair for precise relevance
3. Deduplication — takes the best chunk per source document (avoids 10 chunks from one document)

**Output:** 10 doctrine chunks from 10 different source documents, each with source attribution.

**Example retrieved chunks:**
```
[Källa: Pibat, relevans: 4.23]: Fördröjande fältarbeten planläggs så att de kan sättas igång på stor bredd...
[Källa: ARTAKTIK, relevans: 3.87]: Lång förberedelsetid + betäckt terräng → strid över stort djup...
[Källa: ATP-3.2.1.1 EDC V1 E, relevans: 2.94]: Obstacles must be covered by fire and observation...
```

### Step 3: Simulation Generation (~20-30 seconds)

**What happens:**
Claude receives:
- The system prompt (simulation engine rules)
- The 10 doctrine chunks (with source names)
- The full scenario text

Claude then generates 24 JSON frames (one every 2 hours) describing all unit positions, events, and status changes.

---

## How Decisions Are Made

### Unit Movement

Claude determines movement based on:
1. **Doctrine norms** — e.g., "5-8 km per day with opposition" from retrieved fördröjningsstrid doctrine
2. **Terrain** — forests slow vehicles, roads enable faster movement, urban areas restrict manoeuvre
3. **Obstacles** — minefields halt advance (duration from doctrine), destroyed bridges force detours
4. **Opposition** — engagement by defending forces slows/stops advance

### Friendly Force Behavior

Governed by Swedish doctrine on fördröjningsstrid:
1. **Initial phase** — prepare obstacles (mines, bridge demolitions) per Pibat/fältarbetsreglemente
2. **Delay lines** — withdraw between fördröjningslinjer per DGO principles
3. **Engagement** — fire at maximum effective range per weapon system capabilities
4. **Repositioning** — move via covered routes (CB90 waterways, forest roads)

### Hostile Force Behavior

Based on Claude's knowledge of adversary TTPs (marked `(NATO)` in events):
1. **Advance along roads** — VDV/BTG doctrine favours road-bound movement
2. **Split into elements** — main body + flanking force (standard BTG employment)
3. **Obstacle clearing** — manual if no engineering vehicles; time depends on obstacle type
4. **Fire support** — artillery preparation before assault

### Bridge Demolitions

Timing based on:
- Doctrine: "Fördröjande fältarbeten planläggs... förberedande order" [SWE: Pibat]
- Scenario: preparation time from bridge specifications
- Tactical logic: demolished after friendly forces pass but before hostile arrival

### Mining

Placement and effect based on:
- Doctrine: mining norms from Pibat (tidsåtgång, antal minor per spärr)
- Scenario: available mines (FFV 028 count)
- Tactical logic: placed on main advance axes to canalize enemy

### Combat Engagements

Triggered when:
- Hostile units enter weapon range of friendly positions
- Range determined by weapon system: RBS-17 (8km), Carl Gustaf (300-700m), 81mm GrK (5.5km)
- Outcome: vehicle losses proportional to engagement duration and weapon effectiveness

---

## Doctrine Source Attribution

Every event in the simulation is tagged with its doctrine source:

| Tag | Meaning | Example |
|-----|---------|---------|
| `[SWE: Pibat]` | Decision based on retrieved Swedish Pionjärbataljon doctrine | Bridge demolition timing |
| `[SWE: ARTAKTIK]` | Decision based on Swedish artillery tactics | Indirect fire employment |
| `[SWE: DGO]` | Decision based on Doktrin för Gemensamma Operationer | Delay line withdrawals |
| `[SWE: ATP-3.2.1.1]` | Decision based on NATO/Swedish land tactics | Advance to contact procedures |
| `(NATO)` | Fallback — no Swedish doctrine available for this topic | Enemy force TTPs, foreign weapon specs |

### When NATO Fallback Is Used

The `(NATO)` tag appears when:
- The RAG corpus doesn't contain doctrine for the specific topic
- Enemy force behavior (VDV TTPs, Russian doctrine) — never in Swedish corpus
- Specific weapon system performance data not in the documents
- Engagement outcomes and casualty calculations

### Why This Matters

Users can audit the simulation and see:
- Which decisions are grounded in their own doctrine documents ✅
- Which decisions rely on Claude's general military knowledge ⚠️
- Where additional doctrine documents would improve accuracy

---

## Frame Structure

Each of the 24 frames contains:

```json
{
  "hour": 10,
  "events": [
    "H+10: VDV Main hits Minfält 1, 1 BMD-4M destroyed [SWE: Pibat]",
    "H+10: VDV halted for manual clearing — 2h delay [SWE: Pibat]"
  ],
  "units": [
    {
      "id": "VDV_MAIN",
      "name": "VDV Main",
      "side": "hostile",
      "position": [57.670, 12.10],
      "vehicles_remaining": 11,
      "activity": "clearing minefield"
    },
    {
      "id": "AMF_BAT",
      "name": "Amfbat",
      "side": "friendly",
      "position": [57.675, 12.00],
      "vehicles_remaining": 16,
      "activity": "defending PL BRAVO"
    }
  ],
  "structures": [
    {"id": "BRO1", "name": "Rv40 bro", "position": [57.678, 12.01], "status": "prepared"}
  ],
  "minefields": [
    {"id": "MF1", "name": "Mölnlycke", "position": [57.670, 12.10], "cleared": false}
  ]
}
```

---

## Limitations and Known Issues

### JSON Truncation
Claude's response sometimes exceeds 16,000 tokens and gets truncated mid-JSON. The system handles this by:
1. Attempting to parse the full response
2. If that fails, searching backwards for the last complete frame object
3. If salvage finds ≥3 frames, using those (partial simulation)
4. If all fails, retrying with a simpler 12-frame prompt

### Determinism
The simulation is **non-deterministic** — running the same scenario twice produces different timelines. Claude's temperature is 0 (greedy decoding) but the token predictions still vary slightly between API calls.

### Parametric Leakage
Despite source attribution rules, Claude may occasionally generate events based on training data without marking them as `(NATO)`. This is inherent to LLM-based simulation — the model can't perfectly distinguish "I know this from the provided context" vs "I know this from pre-training."

### Coordinate Accuracy
Claude estimates positions based on place names in the scenario. These are approximate — real-world road routes aren't followed precisely. For more accurate movement, the scenario should include explicit waypoint coordinates.

---

## Improving Simulation Quality

### Add More Doctrine Documents
The most impactful improvement: add more Swedish military doctrine to the `markdown_output/` corpus. Specifically:
- Fältarbetsreglemente (field works regulations) — mining/demolition norms
- Stridsteknik (combat technique) — engagement procedures
- Förbandsreglementen (unit regulations) — specific unit TTPs
- Underrättelsereglemente — reconnaissance/intelligence procedures

### Provide Detailed Scenarios
The more specific the scenario, the better the simulation:
- Include exact coordinates for all positions
- Specify weapon ranges and ammunition quantities
- Include timing constraints (when forces arrive, preparation time available)
- Describe terrain in detail (road types, forest density, urban areas)

### Engineering Norms in Scenario
Include time/resource norms directly in the scenario text:
```
Minläggningstid: 3-4 timmar per 100m fält (10 man, manuell)
Brosprängning Bro 1: 4-5 timmar, 200 kg sprängmedel, 4 man
Vägspärr typ B: 2-4 timmar, 6 man
```
This prevents Claude from needing to estimate these values.
