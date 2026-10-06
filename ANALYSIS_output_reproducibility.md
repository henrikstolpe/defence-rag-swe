# Analys: Repeterbarhet och förutsägbarhet i Claude-svar

**Fråga:** Hur kan slutresultatet från Claude göras mer repeterbart och förutsägbart
när samma fråga ställs två gånger? Krävs ytterligare prompt-instruktioner till
Claude, eller orsakas variationen tidigare i kedjan (RAG-delen)?

**Kort svar:** Variationen kommer från **två oberoende källor**. Den största och mest
direkta är att alla Claude-anrop kör med `temperature = 1.0` (default), eftersom
parametern aldrig sätts explicit. Den andra, mindre men verkliga källan finns i
RAG-delen. Att sänka `temperature` ger den största omedelbara förbättringen; RAG-delen
är redan till största delen deterministisk men har några svaga punkter.

---

## 1. Var uppstår variationen? Översikt av pipelinen

```
Fråga
  │
  ▼
[1] Query-expansion   ──► deterministisk (ren strängmanipulation)
  │
  ▼
[2] Embedding (BGE-M3, CPU) ──► i praktiken deterministisk
  │
  ▼
[3] FAISS vektorsökning ──► deterministisk (exakt index)
  │
  ▼
[4] BM25 ──► deterministisk
  │
  ▼
[5] Sammanslagning + sortering ──► deterministisk, men svag på lika värden (ties)
  │
  ▼
[6] Cross-encoder reranking ──► i praktiken deterministisk
  │
  ▼
[7] Klassificerare (lokal Llama) ──► deterministisk (do_sample=False)
  │
  ▼
[8] CLAUDE-GENERERING ──► ICKE-DETERMINISTISK (temperature=1.0)  ◄── HUVUDKÄLLA
  │
  ▼
Svar
```

Slutsatsen är att kontexten som skickas till Claude i praktiken är samma vid varje
körning av samma fråga. Det är själva genereringssteget (steg 8) som står för den
stora variationen.

---

## 2. Huvudkälla: Claude körs med temperature = 1.0

Alla anrop till Claude i kodbasen utelämnar `temperature`. Anthropic-API:t använder då
standardvärdet **1.0**, vilket är maximal sampling-slumpmässighet. Det innebär att även
med *identisk* kontext kan formuleringar, struktur, ordval och ibland vilka detaljer som
lyfts fram skilja sig åt mellan två körningar.

Berörda anrop:

| Fil | Funktion | max_tokens | temperature idag |
|-----|----------|-----------|------------------|
| `web_interface.py` | `generate()` (Q&A-svar) | 4096 | (saknas → 1.0) |
| `sim_generator.py` | doktrin-topics (steg 1) | 500 | (saknas → 1.0) |
| `sim_generator.py` | frame-generering (steg 2) | 16000 | (saknas → 1.0) |
| `sim_generator.py` | retry-generering | 10000 | (saknas → 1.0) |
| `scenario_parser.py` | scenariotolkning | 8192 | (saknas → 1.0) |
| `scenario_parser.py` | andra anropet | 4096 | (saknas → 1.0) |
| `test_question.py` | test-generering | 1500 | (saknas → 1.0) |

### Rekommendation (störst effekt, minst arbete)

Sätt `temperature` explicit på varje `messages.create`-anrop.

- **För faktabaserad Q&A** (`web_interface.py` → `generate()`): använd
  `temperature=0.0`. Uppgiften är att återge fakta ur kontexten med källhänvisning —
  där är kreativ variation en nackdel. Detta ger den mest repeterbara utdatan.
- **För scenariotolkning** (`scenario_parser.py`): `temperature=0.0`. Tolkning av ett
  scenario till strukturerad data (UND/JSON) bör vara så deterministisk som möjligt.
- **För simuleringsgenerering** (`sim_generator.py`): `temperature` runt `0.2–0.4`.
  Här vill man ha viss variation mellan simuleringar, men lägre temperatur gör
  JSON-utdatan mer stabil och minskar risken för de trunkerings-/formatfel som redan
  hanteras med salvage-logik. Om full repeterbarhet önskas även här: `0.0`.

Exempel (Q&A-vägen i `web_interface.py`):

```python
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    temperature=0.0,          # <-- lägg till denna rad
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"}],
)
```

**Viktigt förbehåll:** `temperature=0.0` gör svaren *mycket mer* stabila men garanterar
inte bit-för-bit-identiska svar. Stora språkmodeller som körs via API kan ge små
skillnader mellan körningar även vid temperatur 0 (beroende på batchning och
flyttalsordning på serversidan). Förväntan bör vara "hög samstämmighet i innehåll och
struktur", inte "exakt samma tecken varje gång".

---

## 3. Andra källan: variation i RAG-delen

RAG-kontexten är till största delen deterministisk, men det finns några punkter som kan
ge olika resultat mellan körningar eller mellan snarlika frågor.

### 3.1 Sortering vid lika poäng (ties) — reell men liten risk
I sammanslagningen (`retrieve()`) och i reranking-sorteringen sorteras kandidater på
poäng med `list.sort(...)`. Pythons sortering är stabil, men ordningen på element med
*exakt samma poäng* avgörs av den ursprungliga insättningsordningen. Den ordningen
kommer i sin tur från en `dict` (`all_candidates`), där insättningsordningen är stabil i
moderna Python-versioner. I praktiken är detta stabilt inom samma process, men det är en
svag punkt: om två chunkar hamnar precis på gränsen till `final_k` kan resultatet vara
känsligt.

**Åtgärd:** lägg till ett sekundärt, deterministiskt sorteringskriterium, t.ex. `idx`:

```python
ranked.sort(key=lambda x: (x["score"], -x["idx"]), reverse=True)
# och för reranking:
rerank_candidates.sort(key=lambda x: (x["score"], -x["idx"]), reverse=True)
```

Detta gör att chunkar med samma poäng alltid hamnar i samma ordning.

### 3.2 Embeddings och cross-encoder — i praktiken deterministiska
Både BGE-M3-embeddings och cross-encoder körs på CPU i eval-läge (`model.eval()`
implicit hos sentence-transformers). Ingen sampling sker. Samma indata ger samma
vektorer och samma reranking-poäng inom samma miljö. Skillnader kan bara uppstå vid byte
av hårdvara, bibliotekversion eller trådantal, inte mellan två körningar på samma maskin.

### 3.3 FAISS och BM25 — deterministiska
Vektorsökningen använder ett exakt index (ingen approximativ, slumpad sökning i den
konfigurationen) och BM25 är ren aritmetik. Inga slumpkällor.

### 3.4 Query-expansion — deterministisk
`expand_query()` är enbart strängmanipulation baserad på frågans inledning. Samma fråga
ger alltid samma expansioner.

**Slutsats om RAG:** RAG-delen bidrar mycket lite till variationen mellan två körningar
av *samma* fråga. Den enda reella (men liten) risken är tie-break-ordningen i 3.1. Det
är alltså inte här huvudproblemet ligger.

---

## 4. Ytterligare prompt-instruktioner till Claude

Prompt-instruktioner höjer *förutsägbarheten i struktur och innehåll* men tar **inte**
bort sampling-slumpen — det gör bara `temperature`. De två åtgärderna kompletterar
varandra:

- `temperature=0.0` → mindre slumpmässig formulering (låg varians).
- Skarpare prompt → mer förutsägbar *form* och *innehåll* (rätt saker sägs, i rätt
  ordning, i rätt format).

Systemprompten i `web_interface.py` är redan bra strukturerad (kräver källhänvisning,
fast struktur A/B/C, förbud mot fabricerade siffror). Möjliga förstärkningar för ännu
högre förutsägbarhet:

1. **Fast utdataformat.** Ange exakt rubriknivå och ordning, t.ex. "Använd alltid
   exakt dessa rubriker i denna ordning: `## A) FAKTA FRÅN KONTEXT`, `## B) LUCKOR I
   UNDERLAG`, `## C) REKOMMENDATION`." Det minskar strukturell variation.
2. **Deterministisk källhänvisning.** Instruera Claude att alltid referera källor i den
   ordning de presenteras i kontexten (`[Doktrin källa 1]`, `[Doktrin källa 2]`...), så
   att samma fakta får samma etikett varje gång.
3. **Begränsa spekulation.** Förtydliga att inga alternativa formuleringar eller
   omskrivningar av samma fakta ska läggas till — endast det som efterfrågas.
4. **För simulering:** kräv strikt JSON utan omgivande text och med ett fast antal
   frames och fasta fältnamn. Det minskar de trunkerings-/parsningsfel som idag hanteras
   av salvage-logiken och gör antalet frames förutsägbart.

---

## 5. Prioriterad åtgärdslista

| Prio | Åtgärd | Fil(er) | Effekt | Arbetsinsats |
|------|--------|---------|--------|--------------|
| 1 | Sätt `temperature=0.0` för Q&A-generering | `web_interface.py` | **Stor** – största källan till variation | Minimal (1 rad) |
| 2 | Sätt `temperature=0.0` för scenariotolkning | `scenario_parser.py` | Stor för strukturerad utdata | Minimal |
| 3 | Sätt låg `temperature` (0.0–0.3) för simulering | `sim_generator.py` | Medel – stabilare JSON, färre parsningsfel | Minimal |
| 4 | Deterministisk tie-break i sortering (`idx`) | `rag_retrieval.py` (+ ev. `web_interface.py`) | Liten – tar bort gränsfall | Låg |
| 5 | Skärp systemprompt (fast format, fast källordning) | `web_interface.py` | Medel – förutsägbar form | Låg–medel |

---

## 6. Sammanfattning

- **Orsaken till att samma fråga ger olika svar är i första hand Claude-genereringen, inte
  RAG-delen.** Alla Claude-anrop kör med den implicita standardtemperaturen 1.0.
- **Den enskilt viktigaste ändringen** är att sätta `temperature=0.0` (för Q&A och
  scenariotolkning) respektive ett lågt värde för simulering. Det är en minimal
  kodändring med stor effekt.
- **RAG-delen är redan i praktiken deterministisk.** Den enda reella förbättringen där är
  ett deterministiskt tie-break-kriterium vid sortering, vilket eliminerar sällsynta
  gränsfall.
- **Prompt-instruktioner** kompletterar men ersätter inte temperatursänkning: de gör
  *formen* och *innehållet* mer förutsägbara, medan temperaturen styr *sampling-slumpen*.
- **Realistisk förväntan:** hög samstämmighet i innehåll och struktur mellan körningar.
  Bit-exakt identiska svar kan inte garanteras via ett moln-API ens vid temperatur 0.

*Denna analys innehåller endast rekommendationer. Inga kodändringar har gjorts.*
