"""
Scenario Simulator for RAG-based military Q&A.

Injects a tactical scenario with:
- A Swedish mechanized battalion (based on open-source ORBAT)
- An opposing Russian Battalion Tactical Group (BTG)
- Realistic locations in Sweden within 40km of each other

The scenario context is prepended to the RAG-retrieved context,
allowing the model to answer questions that require both doctrinal
knowledge AND specific tactical situation awareness.

Usage:
    python3 scenario_simulator.py "din fråga"
    python3 scenario_simulator.py --claude "din fråga"
"""
import os
import json
import math
import re
import sys
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# SCENARIO DATA (open-source ORBAT and locations)
# ============================================================

SCENARIO = """
=== TAKTISKT SCENARIO ===

TIDPUNKT: D+1, 0400Z (gryning)
VÄDER: Regn, +8°C, sikt 1-3 km, lågt molntäcke 200m

MARKFÖRHÅLLANDEN: Blöt mark, lerjord/morän, begränsar terrängkörning för pansarfordon

--- EGNA FÖRBAND: Amfibiebataljon (Amfbat), 4:e Amfibieregementet (Amf 4) ---

Baserad på amfibiebataljon ur Amf 4, Göteborg. Kustförsvarsförband optimerat
för littoral strid i skärgård och kustområden, nu insatt i landförsvar av Göteborg.

Personalstyrka: ~400 soldater
Gruppering: Område GÖTEBORG CENTRUM / ERIKSBERG (N 57°42', E 11°55')

Organisation:
• Bataljonsstab och stabskompani (40 pers)
  - Ledningsbåt (Stridsbåt 90 / CB90)
  - Stabsfordon (terrängbilar)
  - Sambandssystem: DART + marin samband
  - Sjukvårdsgrupp

• 2 × Amfibiekompani (vardera ~100 pers):
  - 3 × Amfibieplutoner (skytteinfanteri, specialiserat för kust/urban)
  - Pluton: 30 man, beväpnade med Ak 5C, Ksp 58, Grg m/86 (Carl Gustaf)
  - Transportkapacitet: 4 × Stridsbåt 90 (CB90) per pluton
  - Beväpning CB90: 12,7mm tung ksp + valfritt RWS
  - Skyttesoldater: Lättinfanteri, ingen pansarskyddad transport på land
  - Specialförmåga: Strid i bebyggelse, kustförsvar, bordning

• 1 × Kustrobotbatteri:
  - 4 × Robot 17 (RBS-17 Hellfire, sjömålsrobot)
  - Räckvidd: 8 km (sjömål), kan användas mot landmål (fordon, befälsplatser)
  - Lavetter: Fordonsbaserade (terrängbil)
  - 16 × robotar tillgängliga

• 1 × Granatkastarpluton:
  - 4 × 81mm GrK (bärbara, ej fordonsbaserade)
  - Räckvidd: 5,5 km
  - Ammunition: 200 granater (spräng + rök + lys)
  - OBS: Lättare system än mekbat (81mm vs 120mm)

• 1 × Kustjägargrupp (underrättelse/spaning):
  - 20 man specialförband (Kustjägare, KJ)
  - Förmåga: Dold spaning, eldledning, sabotage, strid bakom fiendens linjer
  - Transportkapacitet: 2 × CB90 (snabbinsats via vatten)
  - Beväpning: Ak 5C, Ksp 90, AG 90 (12,7mm prickskyttegevär), minor
  - UAS: 2 × Black Hornet nano-UAV

• 1 × Minröjnings-/mineringspluton:
  - 20 man
  - Sjöminor: 20 × grundbottenmina (kan kontrolleras, läggas i farleder)
  - Landminor: 100 × FFV 028 (PV-mina) — begränsat förråd
  - Sprängmedel: 150 kg sprängdeg m/46
  - Sprängkapslar: 80 st
  - Sprängningsladdning bro (förtillverkad, 25 kg): 6 st
  - Sprängkvalificerad personal: 8 man (4 grupper om 2)

Transportmedel (sjö):
  - 16 × Stridsbåt 90 (CB90) — snabb amfibietransport, 35+ knop
  - 4 × Trossbåt (logistik, tyngre materiel)
  - Kapacitet CB90: halvpluton (18 man) eller 1 × RBS-17 lavett

Transportmedel (land):
  - ~15 × Terrängbil (ej pansrade)
  - 2 × Bandvagn 410 (stabstransport)
  - SAKNAR: Pansarfordon, stridsvagnar, CV90, bandgrävare

Tillgänglig ammunition (24h förråd):
  - 81mm granatkastare: 200 granater
  - RBS-17: 16 robotar
  - Carl Gustaf (Grg m/86): 60 skott (pansarvärnsgranater + HE)
  - Handvapen/kulspruteammunition: grundförråd
  - Sjöminor: 20 st (kontrollerbara)
  - SAKNAR: Tung pansarvärnsrobot (RBS 57), tung artillerield, stridsvagnseld

KRITISKA BEGRÄNSNINGAR (amfibiebataljon i landstrid):
  - INGET PANSAR — all personal exponerad, inga pansarfordon
  - INGEN TUNGT INDIREKT ELD — 81mm GrK (5,5 km) vs fiendens 120mm Nona (8,8 km)
  - BEGRÄNSAD PANSARVÄRNSFÖRMÅGA — Carl Gustaf (300-700m) + RBS-17 (8 km, ej optimerat för landstrid)
  - INGET LUFTVÄRN — helt beroende av högre luftförsvar
  - STRID I BEBYGGELSE — huvudförmåga, gynnas av Göteborgs stadslandskap
  - SJÖRÖRLIGHET — kan manövrera via Göta Älv och hamnar (CB90, 35 knop)
  - KUSTJÄGARE — asymmetrisk fördel: spaning/sabotage bakom fiendens linjer

Stridsvärde: FULLT (90%+ personal och materiel tillgänglig)
Moralläge: Högt (hemmaförsvar)
Uthållighet: 48-72 timmar

UPPGIFT: Genomför försvar av GÖTEBORG genom fördröjningsstrid i
fördröjningsområde MÖLNDAL-LANDVETTER för att vinna tid (min 48h)
och förhindra fiendens framryckning mot Göteborgs hamn och centrum.

TAKTISKA FÖRDELAR (amfbat vs VDV i stadsförsvar):
  - Urban terräng neutraliserar fiendens fordonsöverläge
  - CB90 ger manöverförmåga via Göta Älv/kanaler (flankering, snabb omgruppering)
  - Kustjägare kan operera bakom fiendens linjer (Landvetter-området)
  - RBS-17 kan bekämpa BMD-4M på 8 km avstånd (genomslår lätt pansar)
  - Carl Gustaf effektiv mot alla VDV-fordon (BMD-4M, Sprut-SD har tunt pansar)
  - Sjöminor kan blockera Göta Älvs mynning mot sjöburen förstärkning

--- MOTSTÅNDARFÖRBAND: Rysk luftlandsatt bataljonstridsgrupp (VDV BTG) ---

Baserad på typisk VDV (Vozdushno-Desantnye Voyska) bataljonstridsgrupp,
76:e Luftlandsättningsdivisionen. Lufttransporterad till Landvetter med Il-76.

Personalstyrka: ~600 soldater
Gruppering: Område LANDVETTER FLYGPLATS (N 57°40', E 12°17'), luftlandsatt
Framryckningsriktning: Väst mot Göteborg centrum/hamn (20 km)

Organisation:
• BTG-stab med ledningskompani
  - Ledningsfordon (BTR-D variant)
  - EW-grupp (elektronisk krigföring): R-330Zh Zhitel (begränsad variant)

• 2 × Luftlandsättningskompanier (BMD-4M):
  - 3 × Plutoner med vardera 3 × BMD-4M
  - Totalt: 18 × BMD-4M
  - Beväpning BMD-4M: 100mm kanon 2A70 / robotavfyrare, 30mm akan 2A72, 7,62mm PKT
  - Vikt: 13,6 ton (lufttransporterbar i Il-76)
  - Pansarskydd: Lätt (skyddar mot 12,7mm frontalt, splitter övriga sidor)
  - Infanteri: 5 per vagn (begränsat utrymme jmf BMP-3:s 7)

• 1 × Stridsvagnskompani — SAKNAS (ej lufttransporterbart)
  OBS: VDV BTG saknar tung pansarförmåga. Inga MBT (T-80/T-72) ingår.

• 1 × Pansarvärnskompani (förstärkning):
  - 6 × Sprut-SD (125mm luftlandsättbar stridsvagn/pansarvärnskanon)
  - Vikt: 18 ton (lufttransporterbar i Il-76, max 1-2 per flygplan)
  - Beväpning: 125mm kanon 2A75 (samma ammunition som T-72)
  - Pansarskydd: MYCKET LÄTT (aluminium, skyddar mot 7,62mm + splitter)

• 1 × Artilleribatteri (luftlandsättbart):
  - 6 × 2S9 Nona-S (120mm bomkastare/haubitser på BMD-chassi)
  - Vikt: 8 ton (lufttransporterbar)
  - Räckvidd: 8,8 km (konventionell), 12,8 km (raketassisterad)
  - OBS: Betydligt kortare räckvidd än markbaserad 152mm

• 1 × Luftvärnspluton:
  - 2 × BTR-ZD Skrezhet (ZU-23-2 på BTR-D chassi)
  - 4 × 9K338 Igla-S (MANPADS-grupper)
  - OBS: Saknar medelräckviddigt luftvärn (inget Tor-M1 — ej lufttransporterbart)

• 1 × Spaningspluton:
  - 2 × BTR-D (spaningsversion)
  - 2 × Orlan-10 UAV

• 1 × Ingenjörspluton (begränsad):
  - Saknar IMR-2 (för tung för lufttransport)
  - 1 × BTR-D med pionjärutrustning
  - Manuell hinderröjning — betydligt långsammare än mekaniserad
  - Hinderröjningstid (manuell): 1-2 timmar per vägspärr typ A-B (jmf IMR-2: 20-40 min)

Fordonspark totalt:
  - 18 × BMD-4M (luftlandsättbar IFV, 13,6 ton)
  - 6 × Sprut-SD (luftlandsättbar PV-kanon, 18 ton)
  - 6 × 2S9 Nona-S (luftlandsättbar 120mm, 8 ton)
  - 2 × BTR-ZD (luftvärn, 8 ton)
  - ~15 × BTR-D/GAZ-varianter (logistik/stöd)
  - 2 × Orlan-10 UAV
  - SAKNAR: MBT, tung artilleri (152mm), medelräckviddigt luftvärn, IMR-2

Bedömd ammunition (begränsad — flygburen logistik):
  - 125mm Sprut-SD: 120 skott (20 per vagn)
  - 100mm BMD-kanon: 600 skott
  - 120mm Nona-S: 300 granater (50 per pjäs)
  - MANPADS Igla-S: 16 robotar

KRITISKA SVAGHETER (luftlandsatt förband):
  - INGET TUNGT PANSAR — alla fordon sårbara för 40mm akan (CV90)
  - Begränsad artillerieldsräckvidd (8,8 km vs 25-30 km för markbaserad)
  - Begränsad ammunition — beroende av fortsatt lufttransport för påfyllning
  - Ingen tung hinderröjningsförmåga — manuell röjning tar 3-5× längre tid
  - Inget medelräckviddigt luftvärn — sårbar för helikopteranfall
  - Begränsad uthållighet utan påfyllning: 24-36 timmar
  - Låg skyddsnivå: BMD-4M genomslås av 40mm frontalt

Stridsvärde: FULLT men KVALITATIVT BEGRÄNSAT (100% av vad som landade, men lätt utrustning)
Moralläge: Högt (elitstyrka, VDV-tradition)
Bedömd uthållighet: 24-36 timmar utan flygburen påfyllning

FIENDENS TROLIGA HANDLANDE:
- Snabb framryckning längs Rv 40 mot Göteborg centrum — tid är kritisk
- Försök att säkra Göteborgs hamn som logistikpunkt (för sjöburen förstärkning)
- Begränsad artilleriförberedelse (120mm, kort räckvidd, lite ammunition)
- Flankering med infanteri via Härrydavägar
- Drönarsök med Orlan-10 före framryckning
- EW-störning mot svenska sambandsmedel
- Manuell hinderröjning (1-2 tim per spärr — stor fördröjningseffekt!)
- BTG framryckningshastighet (med motstånd): 3-8 km/dygn (nedsatt pga lätta fordon)
- BTG framryckningshastighet (utan motstånd, väg): 30-50 km/dygn
- BEROENDE AV: Fortsatt kontroll av Landvetter flygplats för förstärkning/påfyllning

--- TERRÄNG OCH GEOGRAFI ---

Område: Västra Götaland, mellan Landvetter flygplats och Göteborg centrum
Avstånd mellan styrkorna: ~20 km
Terrängtyp: Kuperad skogsterräng med tätbebyggelse
- Skogsområden (40%): barrskog/blandskog, sikt 50-150m, jordtyp: morän/berg
- Bebyggelse (35%): Mölndal, Härryda, Kållered, förorter — gynnar försvar
- Öppna fält (15%): jordbruksmark vid Härryda, jordtyp: lerjord
- Vatten (10%): Mölndalsån, småsjöar, våtmarker — begränsar terrängkörning

Huvudvägar i området:
- Rv 40 (Borås-Göteborg) — huvudframryckningsaxel från Landvetter, 4 körfält
- E6 (norr-söder genom Göteborg) — kan nås via Rv 40
- Rv 27 (Kungsbacka-Landvetter) — alternativ axel söder
- Härrydavägar och skogsvägar — flankeringsvägar, smala, kuperade

Nyckelpunkter (med avstånd från Göteborg centrum):
- LANDVETTER FLYGPLATS: Fiendens utgångsgruppering (20 km öst)
- MÖLNLYCKE: Korsning Rv 40/lokalvägar (14 km), möjlig fördröjningsposition
- KÅLLERED: Alternativ axel söder (12 km)
- MÖLNDALSÅN (vattendraget): Naturligt hinder (6-8 km från centrum), 3 broar
- MÖLNDAL: Yttre fördröjningsposition, tätbebyggelse (5 km)
- GÖTEBORG CENTRUM: Försvarsmål (0 km)

Infrastruktur:
- 3 broar över Mölndalsån (se brospecifikationer nedan)
- Rv 40-tunneln vid Kallebäck (900m, kan blockeras/sprängas)
- Järnväg Göteborg-Borås (kan blockeras)
- Spårvagnsnät i Mölndal/Göteborg (hinder för pansarfordon i stad)
- Tät bebyggelse med flervåningshus — gynnar försvar, begränsar pansarmanöver

--- BROSPECIFIKATIONER I OPERATIONSOMRÅDET ---

Bro 1 "RV40 MÖLNDALSÅN": Motorvägsbro, betong/stål, spann 25m, klass 70
  → Sprängmedelsbehov: 200 kg (1 × förtillverkad laddning 25kg + 175 kg sprängdeg)
  → Förberedelsetid: 4-5 timmar, 4 man

Bro 2 "GÖTEBORGSVÄGEN": Stadsbro, betong, spann 15m, klass 60
  → Sprängmedelsbehov: 100 kg (1 × förtillverkad laddning 25kg + 75 kg sprängdeg)
  → Förberedelsetid: 3 timmar, 4 man

Bro 3 "KVARNBYGATAN": Mindre bro, betong, spann 10m, klass 40
  → Sprängmedelsbehov: 75 kg sprängdeg
  → Förberedelsetid: 2 timmar, 2 man

RV40-TUNNELN KALLEBÄCK: Bergtunnel, 900m lång, kan blockeras
  → Sprängmedelsbehov (takras, blockering): 100 kg sprängdeg
  → Förberedelsetid: 3-4 timmar, 4 man
  → Effekt: Permanent blockering av Rv40 huvudaxel in mot Göteborg

TOTAL sprängmedelsbehov broar + tunnel: 475 kg (av 500 kg tillgängligt → 25 kg reserv)

Särskilda terrängfaktorer:
- Kuperad terräng begränsar BMP-3/T-80 terrängframkomlighet
- Blöt lerjord + regn: pansarfordon fastnar utanför hårdgjorda vägar
- Tätbebyggelse i Mölndal/Kållered: kanaliserar fienden till huvudvägar
- 12 identifierade flankeringsvägar (smala, branta, skogsklädda)

=== SLUT SCENARIO ===
"""

# ============================================================
# RAG Infrastructure (same as test_question.py)
# ============================================================
MODEL_PATH = "./eupolicy_rag_llama_model_swe"
INDEX_DIR = "./rag_index_swe"
EMBED_MODEL = "BAAI/bge-m3"
BASE_MODEL_ID = "AI-Sweden-Models/Llama-3-8B-instruct"
TOP_K = 15
FINAL_K = 3

print("Loading RAG components...")
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
    chunks = json.load(f)
with open(os.path.join(INDEX_DIR, "sources.json"), encoding="utf-8") as f:
    sources = json.load(f)

embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

# BM25
SWEDISH_STOPWORDS = {
    "och", "att", "det", "som", "för", "den", "med", "har", "inte", "till",
    "ett", "var", "från", "kan", "ska", "vid", "eller", "om", "av", "på",
    "är", "de", "en", "denna", "dessa", "sin", "sina", "sitt", "sig",
    "bör", "ska", "skulle", "enligt", "inom", "samt", "såsom", "dock",
}

def normalize_numbers(text):
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    text = re.sub(r'(\d)\s+(\d)', lambda m: m.group(1) + m.group(2), text)
    return text

def tokenize_bm25(text):
    text = normalize_numbers(text)
    tokens = [w.strip(".,;:!?()[]{}\"'") for w in text.lower().split() if len(w) > 2]
    return [t for t in tokens if t not in SWEDISH_STOPWORDS]

corpus_tokens = [tokenize_bm25(chunk) for chunk in chunks]
doc_count = len(chunks)
doc_lengths = [len(tokens) for tokens in corpus_tokens]
avg_doc_length = sum(doc_lengths) / doc_count

df = Counter()
for tokens in corpus_tokens:
    for term in set(tokens):
        df[term] += 1
tf_per_doc = [Counter(tokens) for tokens in corpus_tokens]

def bm25_score(query_tokens, doc_idx, k1=1.5, b=0.75):
    score = 0.0
    doc_len = doc_lengths[doc_idx]
    tf = tf_per_doc[doc_idx]
    for term in query_tokens:
        if term not in tf:
            continue
        term_freq = tf[term]
        doc_freq = df.get(term, 0)
        if doc_freq == 0:
            continue
        idf = math.log((doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        tf_norm = (term_freq * (k1 + 1)) / (term_freq + k1 * (1 - b + b * doc_len / avg_doc_length))
        score += idf * tf_norm
    return score

def expand_query(question):
    expansions = [question]
    q = question.lower()
    if q.startswith("vad är ") or q.startswith("vad innebär "):
        expansions.append(question.replace("Vad är ", "Definiera ").replace("Vad innebär ", "Förklara "))
    elif "vad säger" in q and "om" in q:
        topic = question.split("om")[-1].strip().rstrip("?")
        expansions.append(f"regler angående {topic}")
    else:
        expansions.append(question.rstrip("?") + " doktrin metod")
    return expansions[:3]

def retrieve(query, top_k=TOP_K, final_k=FINAL_K):
    queries = expand_query(query)
    all_candidates = {}
    for q in queries:
        q_emb = embed_model.encode([q], normalize_embeddings=True).astype(np.float32)
        vec_scores, vec_indices = index.search(q_emb, top_k)
        query_tokens = tokenize_bm25(q)
        bm25_all = [(i, bm25_score(query_tokens, i)) for i in range(doc_count)]
        bm25_all.sort(key=lambda x: x[1], reverse=True)
        bm25_top = bm25_all[:top_k]

        vec_max = max(vec_scores[0]) if max(vec_scores[0]) > 0 else 1.0
        for score, idx in zip(vec_scores[0], vec_indices[0]):
            idx = int(idx)
            norm_score = float(score) / vec_max
            if idx not in all_candidates or norm_score > all_candidates[idx]:
                all_candidates[idx] = norm_score

        bm25_max = bm25_top[0][1] if bm25_top[0][1] > 0 else 1.0
        for idx, score in bm25_top:
            norm_score = score / bm25_max
            combined = 0.6 * all_candidates.get(idx, 0.0) + 0.4 * norm_score
            if idx not in all_candidates or combined > all_candidates[idx]:
                all_candidates[idx] = combined

    ranked = [{"text": chunks[idx], "score": score, "idx": idx, "source": sources[idx]} for idx, score in all_candidates.items()]
    ranked.sort(key=lambda x: x["score"], reverse=True)

    rerank_candidates = ranked[:top_k]
    if rerank_candidates:
        pairs = [[query, c["text"]] for c in rerank_candidates]
        rerank_scores = reranker.predict(pairs)
        for i, score in enumerate(rerank_scores):
            rerank_candidates[i]["score"] = float(score)
        rerank_candidates.sort(key=lambda x: x["score"], reverse=True)

    return rerank_candidates[:final_k]

# Load LLM
print("Loading LLM...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, quantization_config=quantization_config,
    device_map={"": 0}, torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

SYSTEM_PROMPT = (
    "Du är en militär stabsofficer och expert på svensk taktik och doktrin. "
    "Du svarar ENBART med information från den tillhandahållna kontexten (scenario + doktrinära källor).\n\n"
    "KRITISKA REGLER:\n"
    "1. Varje faktapåstående MÅSTE ha en källhänvisning: [Scenario] eller [Doktrin källa X].\n"
    "2. Om du anger ett tal (tid, mängd, avstånd, vikt) som INTE explicit står i kontexten — "
    "skriv istället: 'SAKNAS I UNDERLAG — kräver [specificera vilken handbok/data som behövs]'\n"
    "3. Fabricera ALDRIG siffror. Det är farligare att ge felaktiga värden än att säga att data saknas.\n"
    "4. Strukturera svaret i tre delar:\n"
    "   A) FAKTA FRÅN KONTEXT — vad som kan besvaras med källhänvisning\n"
    "   B) LUCKOR I UNDERLAG — vilken information som saknas för fullständigt svar\n"
    "   C) REKOMMENDATION — vilka dokument/data som behöver tillföras\n"
    "5. Om scenariot innehåller specifika ingenjörsdata (tidsåtgång, mineringsnormer, sprängmedelsberäkningar), "
    "använd dessa. Hitta INTE PÅ egna värden.\n"
    "6. Ge utförliga och strukturerade svar med punktlistor där lämpligt."
)

def classify_answerable(question, context_text):
    classify_prompt = (
        "Du är ett klassificeringssystem. Givet en kontext och en fråga, avgör om kontexten "
        "innehåller information relevant för att besvara frågan. Svara med ENBART 'YES' eller 'NO'.\n\n"
        "Regler:\n"
        "- Svara YES om kontexten innehåller relevant taktisk eller doktrinär information.\n"
        "- Svara YES om ett scenario presenteras som ger underlag att besvara frågan.\n"
        "- Svara NO ENBART om frågan är helt orelaterad till kontexten.\n"
        "- Vid tveksamhet, svara YES."
    )
    messages = [
        {"role": "system", "content": classify_prompt},
        {"role": "user", "content": f"Kontext:\n{context_text[:2000]}\n\nFråga: {question}\n\nKan denna fråga besvaras utifrån kontexten? (YES/NO)"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response or "JA" in response

def generate_local(question, context_text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=800, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def generate_claude(question, context_text):
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"}
        ],
    )
    return message.content[0].text

def generate_mistral(question, context_text):
    from mistralai import Mistral
    client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))
    response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"},
        ],
        max_tokens=2000,
    )
    return response.choices[0].message.content

# ============================================================
# MAIN
# ============================================================
use_claude = "--claude" in sys.argv
use_mistral = "--mistral" in sys.argv
args = [a for a in sys.argv[1:] if a not in ("--claude", "--mistral")]

if args:
    question = " ".join(args)
else:
    question = input("\nFråga (scenario): ").strip()

if use_claude:
    model_label = "Claude (Anthropic API)"
elif use_mistral:
    model_label = "Mistral Large (Mistral API)"
else:
    model_label = "AI-Sweden Llama 3 8B (lokal)"

print(f"\n{'='*70}")
print(f"SCENARIO-BASERAD FRÅGA")
print(f"Modell: {model_label}")
print(f"{'='*70}")
print(f"\nFRÅGA: {question}")

# Retrieve doctrinal context
rag_context = retrieve(question)
print(f"\nDoktrinära källor ({len(rag_context)} hämtade):")
for i, c in enumerate(rag_context):
    print(f"  {i+1}. [{c['score']:.3f}] {c['source']}: {c['text'][:80]}...")

# Combine scenario + doctrinal context
doctrinal_text = "\n\n".join([f"[Doktrin källa {i+1}, {c['source']}]: {c['text']}" for i, c in enumerate(rag_context)])
full_context = f"{SCENARIO}\n\n--- DOKTRINÄRA KÄLLOR ---\n\n{doctrinal_text}"

# Classify
is_answerable = classify_answerable(question, full_context)
print(f"\nKlassificering: {'JA' if is_answerable else 'NEJ'}")

# Generate
if not is_answerable:
    answer = "Kan inte besvaras utifrån tillgängligt scenario och doktrinära källor."
elif use_claude:
    answer = generate_claude(question, full_context)
elif use_mistral:
    answer = generate_mistral(question, full_context)
else:
    answer = generate_local(question, full_context)

print(f"\n{'='*70}")
print(f"SVAR:\n")
print(answer)
print(f"\n{'='*70}")

# Save to file
if use_claude:
    output_file = "scenario_answer_claude.md"
elif use_mistral:
    output_file = "scenario_answer_mistral.md"
else:
    output_file = "scenario_answer_local.md"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"# Fråga\n\n{question}\n\n")
    f.write(f"# Modell\n\n{model_label}\n\n")
    f.write(f"# Doktrinära källor\n\n")
    for i, c in enumerate(rag_context):
        f.write(f"{i+1}. [{c['score']:.3f}] **{c['source']}**\n")
    f.write(f"\n# Svar\n\n{answer}\n")
print(f"\nSvar sparat till: {output_file}")
