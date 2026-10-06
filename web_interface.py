"""
Web interface for the RAG Defence Q&A system.
Provides a simple dialogue with:
- A scenario editor (editable textbox)
- A question input
- A scrollable answer window
- Model selector (local / Claude / Mistral)

Run: python web_interface.py
Then open: http://localhost:7860
"""
import os
import sys
import json
import math
import re
import subprocess
import signal

# True when launched with --share. Simulation preparation is disabled in this
# mode because the gradio tunnel + simulation generation together spike system
# RAM and can trigger an OOM kill of the web server.
SHARE_MODE = "--share" in sys.argv

# ============================================================
# Kill any old instances of this application on startup
# ============================================================
def cleanup_old_processes():
    """Kill any leftover web_interface.py processes and free ports."""
    my_pid = os.getpid()
    try:
        # Find other web_interface.py processes
        result = subprocess.run(["pgrep", "-f", "web_interface.py"], capture_output=True, text=True)
        for pid_str in result.stdout.strip().split("\n"):
            if pid_str and int(pid_str) != my_pid:
                try:
                    os.kill(int(pid_str), signal.SIGKILL)
                    print(f"[Cleanup] Killed old process {pid_str}")
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass

    # Free ports
    for port in [7860, 7861]:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)

cleanup_old_processes()
import numpy as np
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./eupolicy_rag_llama_model_swe"
INDEX_DIR = "./rag_index_swe"
EMBED_MODEL = "BAAI/bge-m3"
BASE_MODEL_ID = "AI-Sweden-Models/Llama-3-8B-instruct"
TOP_K = 15
FINAL_K = 3

# ============================================================
# Default scenario
# ============================================================
DEFAULT_SCENARIO = """=== TAKTISKT SCENARIO ===

TIDPUNKT: D+1, 0400Z (gryning)
VÄDER: Regn, +8°C, sikt 1-3 km, lågt molntäcke 200m
MARKFÖRHÅLLANDEN: Blöt mark, lerjord/morän, begränsar terrängkörning för pansarfordon

--- EGNA FÖRBAND: Amfibiebataljon (Amfbat), 4:e Amfibieregementet (Amf 4) ---

Personalstyrka: ~400 soldater
Gruppering: GÖTEBORG CENTRUM / ERIKSBERG

Organisation:
• 2 × Amfibiekompani (vardera ~100 pers, lättinfanteri)
• 1 × Kustrobotbatteri (4 × RBS-17, räckvidd 8 km)
• 1 × Granatkastarpluton (4 × 81mm GrK, räckvidd 5.5 km)
• 1 × Kustjägargrupp (20 man specialförband)
• 1 × Minröjnings-/mineringspluton (20 man)
• 16 × Stridsbåt 90 (CB90, 35+ knop)

Beväpning: Ak 5C, Ksp 58, Carl Gustaf (Grg m/86), RBS-17
Sprängmedel: 150 kg sprängdeg + 6 × förtillverkad broladdning (25 kg)
Minor: 100 × FFV 028 (PV-mina), 20 × sjöminor

SAKNAR: Pansar, stridsvagnar, tungt luftvärn, tung artilleri

--- MOTSTÅNDARFÖRBAND: Rysk VDV BTG (luftlandsatt) ---

Personalstyrka: ~600 soldater
Gruppering: LANDVETTER FLYGPLATS (20 km öst om Göteborg)

Organisation:
• 18 × BMD-4M (lätt IFV, 13.6 ton, sårbar för 40mm)
• 6 × Sprut-SD (125mm kanon, 18 ton, tunt pansar)
• 6 × 2S9 Nona-S (120mm, räckvidd 8.8 km)
• 2 × BTR-ZD (23mm luftvärn)
• Saknar: Tungt pansar, IMR-2, medelräckviddigt luftvärn

Hinderröjning: Manuell (1-2 tim per spärr, saknar IMR-2)
Uthållighet: 24-36 timmar utan flygburen påfyllning

--- TERRÄNG: Göteborg–Landvetter ---
Avstånd: 20 km
Kuperad skog (40%) + tätbebyggelse (35%) + öppna fält (15%) + vatten (10%)
Huvudaxel: Rv 40, Rv40-tunnel Kallebäck (900m, sprängbar)
3 broar över Mölndalsån (sprängbara)

=== SLUT SCENARIO ==="""

# ============================================================
# Load RAG components
# ============================================================
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

# ============================================================
# Load LLM
# ============================================================
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
print("LLM loaded.")

# ============================================================
# Generation functions
# ============================================================
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

def classify_answerable(question, context_text):
    classify_prompt = (
        "Du är ett klassificeringssystem. Avgör om kontexten innehåller information "
        "relevant för att besvara frågan. Svara med ENBART 'YES' eller 'NO'.\n\n"
        "Regler:\n"
        "- Svara YES om kontexten diskuterar ämnet.\n"
        "- Svara YES om ett scenario presenteras som ger underlag.\n"
        "- Svara NO ENBART om frågan är helt orelaterad.\n"
        "- Vid tveksamhet, svara YES."
    )
    messages = [
        {"role": "system", "content": classify_prompt},
        {"role": "user", "content": f"Kontext:\n{context_text[:2000]}\n\nFråga: {question}\n\n(YES/NO)"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response or "JA" in response

def generate(question, context_text):
    """Generate answer using Claude API."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=0.0,  # Deterministic sampling for reproducible Q&A answers
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"}],
    )
    return message.content[0].text

# ============================================================
# Main Q&A function
# ============================================================
import gradio as gr

def ask_question(question, scenario, progress=gr.Progress()):
    """Process a question with scenario context and RAG retrieval."""
    if not question.strip():
        return "Ange en fråga."

    progress(0.1, desc="🔍 Söker i dokumenten...")

    # Retrieve doctrinal context
    rag_context = retrieve(question)
    source_info = "\n".join([f"  {i+1}. [{c['score']:.2f}] {c['source']}" for i, c in enumerate(rag_context)])

    # Build full context
    doctrinal_text = "\n\n".join([f"[Doktrin källa {i+1}, {c['source']}]: {c['text']}" for i, c in enumerate(rag_context)])

    if scenario.strip():
        full_context = f"{scenario}\n\n--- DOKTRINÄRA KÄLLOR ---\n\n{doctrinal_text}"
    else:
        full_context = doctrinal_text

    progress(0.3, desc="🧠 Klassificerar frågan...")

    # Classify (local fine-tuned model)
    is_answerable = classify_answerable(question, full_context)

    if not is_answerable:
        answer = "❌ Kan inte besvaras utifrån tillgängligt scenario och doktrinära källor."
    else:
        progress(0.5, desc="✍️ Genererar svar (Claude)...")
        answer = generate(question, full_context)

    progress(0.9, desc="✅ Klar!")

    # Format output
    output = f"## Svar\n\n{answer}\n\n---\n\n"
    output += f"### Doktrinära källor hämtade\n{source_info}\n\n"
    output += f"### Klassificering: {'JA ✅' if is_answerable else 'NEJ ❌'}\n"
    output += f"### Modell: Claude (Anthropic)"

    return output

# ============================================================
# Tactical Map (dynamic — reads from UND)
# ============================================================
import folium
from und import UND, Side, UnitType, StructureType, StructureStatus, LocationType
from scenario_parser import parse_scenario_cached


def generate_map_from_und(und_instance):
    """Generate map dynamically from UND state."""
    # Determine map center from terrain or unit positions
    all_positions = [u.position for u in und_instance.units.values() if u.position]
    if all_positions:
        center_lat = sum(p[0] for p in all_positions) / len(all_positions)
        center_lon = sum(p[1] for p in all_positions) / len(all_positions)
    else:
        center_lat, center_lon = 57.68, 12.05

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # --- UNITS ---
    for unit in und_instance.units.values():
        if not unit.position:
            continue

        if unit.side == Side.FRIENDLY:
            echelon = "II" if unit.unit_type == UnitType.BATTALION else ("I" if unit.unit_type == UnitType.COMPANY else "•")
            short_name = unit.name[:6] if len(unit.name) > 6 else unit.name
            svg = f'''<svg width="50" height="38" xmlns="http://www.w3.org/2000/svg">
              <rect x="2" y="10" width="46" height="25" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/>
              <text x="25" y="8" text-anchor="middle" font-size="9" font-weight="bold" fill="#0055aa">{echelon}</text>
              <text x="25" y="27" text-anchor="middle" font-size="8" font-weight="bold" fill="#0055aa">{short_name}</text>
            </svg>'''
        else:
            echelon = "II" if unit.unit_type == UnitType.BATTALION else ("I" if unit.unit_type == UnitType.COMPANY else "•")
            short_name = unit.name[:6] if len(unit.name) > 6 else unit.name
            svg = f'''<svg width="50" height="45" xmlns="http://www.w3.org/2000/svg">
              <polygon points="25,2 48,22 25,42 2,22" fill="#ffcccc" stroke="#cc0000" stroke-width="2"/>
              <text x="25" y="15" text-anchor="middle" font-size="9" font-weight="bold" fill="#cc0000">{echelon}</text>
              <text x="25" y="28" text-anchor="middle" font-size="7" font-weight="bold" fill="#cc0000">{short_name}</text>
            </svg>'''

        # Build popup
        veh_str = ", ".join(f"{k}:{v}" for k, v in unit.vehicles.items()) if unit.vehicles else "—"
        popup_html = f"<b>{unit.name}</b><br>Sida: {unit.side.value}<br>Styrka: {unit.strength or '?'}<br>Fordon: {veh_str}<br>Aktivitet: {unit.activity or '?'}"

        icon = folium.DivIcon(html=svg, icon_size=(50, 45), icon_anchor=(25, 22))
        folium.Marker(unit.position, popup=popup_html, tooltip=unit.name, icon=icon).add_to(m)

    # --- STRUCTURES ---
    for struct in und_instance.structures.values():
        if not struct.position:
            continue

        if struct.status in (StructureStatus.DESTROYED, StructureStatus.BLOCKED):
            color = "#b71c1c"
            fill = "#ffcdd2"
            symbol = "✕"
        elif struct.status == StructureStatus.PREPARED:
            color = "#e65100"
            fill = "#fff3cd"
            symbol = "⚡"
        else:
            color = "#e65100"
            fill = "#fff3cd"
            symbol = "⌇" if struct.structure_type == StructureType.BRIDGE else "T"

        svg = f'<svg width="24" height="24"><rect x="2" y="2" width="20" height="20" fill="{fill}" stroke="{color}" stroke-width="2"/><text x="12" y="16" text-anchor="middle" font-size="12" fill="{color}">{symbol}</text></svg>'
        popup = f"<b>{struct.name}</b><br>Status: {struct.status.value}<br>Spräng: {struct.demolition_kg or '?'} kg<br>Tid: {struct.demolition_time_h or '?'} tim"
        icon = folium.DivIcon(html=svg, icon_size=(24, 24), icon_anchor=(12, 12))
        folium.Marker(struct.position, popup=popup, tooltip=struct.name, icon=icon).add_to(m)

    # --- LOCATIONS ---
    for loc in und_instance.locations.values():
        if not loc.position:
            continue

        if loc.location_type == LocationType.MINEFIELD:
            svg = '<svg width="22" height="22"><line x1="4" y1="4" x2="18" y2="18" stroke="#2e7d32" stroke-width="2"/><line x1="18" y1="4" x2="4" y2="18" stroke="#2e7d32" stroke-width="2"/></svg>'
            icon = folium.DivIcon(html=svg, icon_size=(22, 22), icon_anchor=(11, 11))
            pos = loc.position if isinstance(loc.position[0], (int, float)) else loc.position[0]
            label = f"{loc.name}" + (" (RÖJT)" if loc.cleared else f" ({loc.mine_count} minor)")
            folium.Marker(pos, tooltip=label, icon=icon).add_to(m)
            folium.Circle(pos, radius=200, color="#2e7d32", fill=True, fill_opacity=0.12, weight=2, dash_array="4,4").add_to(m)

        elif loc.location_type == LocationType.PHASE_LINE:
            if isinstance(loc.position[0], list):
                folium.PolyLine(loc.position, color="purple", weight=3, opacity=0.7, dash_array="5,10", tooltip=loc.name).add_to(m)

        elif loc.location_type == LocationType.OBJECTIVE:
            pos = loc.position if isinstance(loc.position[0], (int, float)) else loc.position[0]
            folium.Marker(pos, tooltip=f"🎯 {loc.name}",
                icon=folium.DivIcon(html='<div style="font-size:18px">🎯</div>', icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

    # Legend
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: white; padding: 12px; border-radius: 4px; border: 2px solid #333;
         font-size: 11px; line-height: 2.0; font-family: monospace;">
    <b>DYNAMISK KARTA</b><br>
    <svg width="18" height="13"><rect x="1" y="1" width="16" height="11" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/></svg> Egna<br>
    <svg width="18" height="13"><polygon points="9,1 17,7 9,12 1,7" fill="#ffcccc" stroke="#cc0000" stroke-width="2"/></svg> Fiende<br>
    <svg width="18" height="13"><rect x="1" y="1" width="16" height="11" fill="#fff3cd" stroke="#e65100" stroke-width="2"/></svg> Struktur<br>
    <svg width="18" height="13"><line x1="3" y1="3" x2="15" y2="11" stroke="#2e7d32" stroke-width="2"/><line x1="15" y1="3" x2="3" y2="11" stroke="#2e7d32" stroke-width="2"/></svg> Minfält<br>
    <span style="color:purple">▪▪▪</span> Fördröjningslinje
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()


def generate_dynamic_simulation(und_instance, parsed):
    """Generate a time-stepped simulation from UND state and movement norms."""
    norms = parsed.get("movement_norms", {})
    advance_rate = norms.get("hostile_advance_with_opposition_kmday", 7)  # km/day
    clearing_time = norms.get("hostile_obstacle_clearing_hours", 1.5)
    reposition_speed = norms.get("friendly_reposition_speed_kmh", 30)

    # Get hostile units and their route (straight line toward friendly center)
    hostile_units = und_instance.get_units_by_side(Side.HOSTILE)
    friendly_units = und_instance.get_units_by_side(Side.FRIENDLY)

    if not hostile_units or not friendly_units:
        return []

    # Compute target (average friendly position)
    friendly_positions = [u.position for u in friendly_units if u.position]
    if not friendly_positions:
        return []
    target = [sum(p[0] for p in friendly_positions) / len(friendly_positions),
              sum(p[1] for p in friendly_positions) / len(friendly_positions)]

    # Get obstacles (minefields and structures along the path)
    minefields = [loc for loc in und_instance.locations.values() if loc.location_type == LocationType.MINEFIELD and not loc.cleared]
    bridges = [s for s in und_instance.structures.values() if s.structure_type in (StructureType.BRIDGE, StructureType.TUNNEL) and s.status == StructureStatus.INTACT]

    # Simulation: 48 hours, 2-hour steps
    sim_hours = 48
    step_hours = 2
    frames_html = []
    events = []

    import math
    import copy

    # Deep copy UND state for simulation
    sim_units_hostile = []
    for u in hostile_units:
        sim_units_hostile.append({"id": u.id, "name": u.name, "position": list(u.position) if u.position else None,
                                   "strength": u.strength, "vehicles": dict(u.vehicles), "halted_until": 0})

    sim_units_friendly = []
    for u in friendly_units:
        sim_units_friendly.append({"id": u.id, "name": u.name, "position": list(u.position) if u.position else None,
                                    "strength": u.strength})

    sim_minefields = [{"id": mf.id, "name": mf.name, "position": mf.position, "cleared": False, "mine_count": mf.mine_count} for mf in minefields]
    sim_structures = [{"id": s.id, "name": s.name, "position": s.position, "status": s.status.value,
                       "demolition_time_h": s.demolition_time_h} for s in bridges]

    # Pre-compute advance per step
    advance_per_step_deg = (advance_rate / 24 * step_hours) / 111  # rough deg per step

    for step in range(sim_hours // step_hours):
        hour = step * step_hours

        # --- Advance hostile units ---
        for hu in sim_units_hostile:
            if not hu["position"] or hour < hu["halted_until"]:
                continue

            # Check if hitting a minefield (within 500m)
            hit_mine = False
            for mf in sim_minefields:
                if mf["cleared"]:
                    continue
                dist = math.sqrt((hu["position"][0] - mf["position"][0])**2 + (hu["position"][1] - mf["position"][1])**2) * 111
                if dist < 0.5:  # within 500m
                    hu["halted_until"] = hour + clearing_time * 2  # clearing takes time
                    mf["cleared"] = True
                    events.append(f"H+{hour:02d}: ⚠️ {hu['name']} träffar {mf['name']}! Röjning {clearing_time:.0f}h.")
                    hit_mine = True
                    break

            if not hit_mine:
                # Check if approaching a destroyed bridge (within 300m)
                hit_bridge = False
                for ss in sim_structures:
                    if ss["status"] != "destroyed":
                        continue
                    dist = math.sqrt((hu["position"][0] - ss["position"][0])**2 + (hu["position"][1] - ss["position"][1])**2) * 111
                    if dist < 0.3:
                        hu["halted_until"] = hour + 4  # 4 hours to find alternate crossing
                        events.append(f"H+{hour:02d}: 🚧 {hu['name']} blockerad vid {ss['name']} (förstörd). Söker alternativ.")
                        hit_bridge = True
                        break

                if not hit_bridge:
                    # Move toward target
                    dx = target[0] - hu["position"][0]
                    dy = target[1] - hu["position"][1]
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist > 0.001:
                        hu["position"][0] += (dx / dist) * advance_per_step_deg
                        hu["position"][1] += (dy / dist) * advance_per_step_deg

        # Bridge demolitions: friendly blows bridges between H+14 and H+22
        if 14 <= hour <= 22:
            for ss in sim_structures:
                if ss["status"] == "intact" and hour >= 14 + (sim_structures.index(ss) * 2):
                    ss["status"] = "destroyed"
                    events.append(f"H+{hour:02d}: 💥 {ss['name']} SPRÄNGD!")

        # Initial events
        if hour == 0:
            events.append(f"H+{hour:02d}: Scenario startat. Fienden grupperad.")
        elif hour == 2 and not any("framryckning" in e for e in events):
            events.append(f"H+{hour:02d}: Fienden påbörjar framryckning.")

        # Engagement check: friendly RBS/weapons vs hostile in range
        for fu in sim_units_friendly:
            if not fu["position"]:
                continue
            for hu in sim_units_hostile:
                if not hu["position"]:
                    continue
                dist_km = math.sqrt((fu["position"][0] - hu["position"][0])**2 + (fu["position"][1] - hu["position"][1])**2) * 111
                if 4 < dist_km < 8 and hour >= 16 and f"engagerar" not in " ".join(events[-3:]):
                    events.append(f"H+{hour:02d}: 🚀 {fu['name']} engagerar {hu['name']} på {dist_km:.1f} km!")
                    # Reduce hostile vehicles
                    for vtype in list(hu["vehicles"].keys()):
                        if hu["vehicles"][vtype] > 1:
                            hu["vehicles"][vtype] -= 1
                            break

        # --- Render frame ---
        all_positions = [u["position"] for u in sim_units_hostile + sim_units_friendly if u.get("position")]
        if all_positions:
            clat = sum(p[0] for p in all_positions) / len(all_positions)
            clon = sum(p[1] for p in all_positions) / len(all_positions)
        else:
            clat, clon = 57.68, 12.05

        fm = folium.Map(location=[clat, clon], zoom_start=12, tiles="OpenStreetMap")

        # Time display with vehicle count
        total_hostile_veh = sum(sum(hu["vehicles"].values()) for hu in sim_units_hostile)
        time_div = f'<div style="position:fixed;top:10px;right:10px;z-index:1000;background:#333;color:white;padding:8px 14px;border-radius:4px;font-family:monospace;font-size:15px;">⏱️ H+{hour:02d} | Fiende fordon: {total_hostile_veh}</div>'
        fm.get_root().html.add_child(folium.Element(time_div))

        # Event log
        recent = events[-4:] if events else ["Väntar..."]
        evt_div = f'<div style="position:fixed;bottom:10px;right:10px;z-index:1000;background:rgba(0,0,0,0.85);color:#0f0;padding:8px;border-radius:4px;font-size:10px;font-family:monospace;max-width:380px;line-height:1.5;"><b>LOGG</b><br>{"<br>".join(recent)}</div>'
        fm.get_root().html.add_child(folium.Element(evt_div))

        # Hostile units
        for hu in sim_units_hostile:
            if not hu["position"]:
                continue
            svg = f'<svg width="36" height="32"><polygon points="18,2 34,16 18,30 2,16" fill="#ffcccc" stroke="#cc0000" stroke-width="2"/><text x="18" y="19" text-anchor="middle" font-size="7" fill="#cc0000">{hu["id"][:4]}</text></svg>'
            folium.Marker(hu["position"], tooltip=hu["name"],
                icon=folium.DivIcon(html=svg, icon_size=(36, 32), icon_anchor=(18, 16))).add_to(fm)

        # Friendly units
        for fu in sim_units_friendly:
            if not fu["position"]:
                continue
            svg = f'<svg width="36" height="28"><rect x="2" y="4" width="32" height="20" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/><text x="18" y="18" text-anchor="middle" font-size="7" fill="#0055aa">{fu["id"][:4]}</text></svg>'
            folium.Marker(fu["position"], tooltip=fu["name"],
                icon=folium.DivIcon(html=svg, icon_size=(36, 28), icon_anchor=(18, 14))).add_to(fm)

        # Minefields
        for mf in sim_minefields:
            color = "#999" if mf["cleared"] else "#2e7d32"
            svg = f'<svg width="18" height="18"><line x1="3" y1="3" x2="15" y2="15" stroke="{color}" stroke-width="2"/><line x1="15" y1="3" x2="3" y2="15" stroke="{color}" stroke-width="2"/></svg>'
            folium.Marker(mf["position"], tooltip=mf["name"] + (" (RÖJT)" if mf["cleared"] else ""),
                icon=folium.DivIcon(html=svg, icon_size=(18, 18), icon_anchor=(9, 9))).add_to(fm)

        # Structures
        for ss in sim_structures:
            symbol = "✕" if ss["status"] == "destroyed" else "⌇"
            color = "#b71c1c" if ss["status"] == "destroyed" else "#e65100"
            svg = f'<svg width="18" height="18"><rect x="2" y="2" width="14" height="14" fill="white" stroke="{color}" stroke-width="2"/><text x="9" y="13" text-anchor="middle" font-size="10" fill="{color}">{symbol}</text></svg>'
            folium.Marker(ss["position"], tooltip=ss["name"] + f" ({ss['status']})",
                icon=folium.DivIcon(html=svg, icon_size=(18, 18), icon_anchor=(9, 9))).add_to(fm)

        # Phase lines
        for loc in und_instance.locations.values():
            if loc.location_type == LocationType.PHASE_LINE and isinstance(loc.position[0], list):
                folium.PolyLine(loc.position, color="purple", weight=2, opacity=0.5, dash_array="5,10", tooltip=loc.name).add_to(fm)

        frames_html.append(fm._repr_html_())

    return frames_html

def generate_map():
    """Generate an interactive tactical map with NATO APP-6 symbology."""
    # Center on the operational area between Göteborg and Landvetter
    m = folium.Map(location=[57.68, 12.05], zoom_start=12, tiles="OpenStreetMap")

    # --- NATO APP-6 style unit symbols as SVG ---
    # Friendly battalion (blue rectangle with X for battalion, anchor for amphibious)
    friendly_bn_svg = """
    <svg width="60" height="45" xmlns="http://www.w3.org/2000/svg">
      <rect x="2" y="12" width="56" height="30" fill="#cce5ff" stroke="#0055aa" stroke-width="3"/>
      <text x="30" y="8" text-anchor="middle" font-size="10" font-weight="bold" fill="#0055aa">II</text>
      <text x="30" y="32" text-anchor="middle" font-size="11" font-weight="bold" fill="#0055aa">AMF</text>
      <text x="30" y="43" text-anchor="middle" font-size="7" fill="#0055aa">Amf 4</text>
    </svg>"""

    # Hostile battalion (red diamond with X for battalion)
    hostile_bn_svg = """
    <svg width="60" height="55" xmlns="http://www.w3.org/2000/svg">
      <polygon points="30,2 58,27 30,52 2,27" fill="#ffcccc" stroke="#cc0000" stroke-width="3"/>
      <text x="30" y="17" text-anchor="middle" font-size="10" font-weight="bold" fill="#cc0000">II</text>
      <text x="30" y="31" text-anchor="middle" font-size="9" font-weight="bold" fill="#cc0000">VDV</text>
      <text x="30" y="42" text-anchor="middle" font-size="7" fill="#cc0000">BTG</text>
    </svg>"""

    # Friendly special forces (blue rectangle with diagonal)
    friendly_sf_svg = """
    <svg width="40" height="35" xmlns="http://www.w3.org/2000/svg">
      <rect x="2" y="8" width="36" height="24" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/>
      <line x1="2" y1="32" x2="38" y2="8" stroke="#0055aa" stroke-width="2"/>
      <text x="20" y="5" text-anchor="middle" font-size="8" font-weight="bold" fill="#0055aa">•</text>
      <text x="20" y="24" text-anchor="middle" font-size="7" font-weight="bold" fill="#0055aa">KJ</text>
    </svg>"""

    # Friendly anti-tank (blue rectangle with diagonal line through)
    friendly_at_svg = """
    <svg width="40" height="35" xmlns="http://www.w3.org/2000/svg">
      <rect x="2" y="8" width="36" height="24" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/>
      <text x="20" y="5" text-anchor="middle" font-size="8" font-weight="bold" fill="#0055aa">•</text>
      <text x="20" y="24" text-anchor="middle" font-size="8" font-weight="bold" fill="#0055aa">RBS17</text>
    </svg>"""

    # --- ENEMY FORCES (red diamonds — hostile) ---
    hostile_icon = folium.DivIcon(
        html=hostile_bn_svg,
        icon_size=(60, 55),
        icon_anchor=(30, 27),
    )
    folium.Marker(
        [57.6686, 12.2919],
        popup="<b>HOSTILE — VDV BTG</b><br>76:e Luftlandsättningsdivisionen<br>~600 pers<br>18× BMD-4M (13.6t IFV)<br>6× Sprut-SD (125mm)<br>6× 2S9 Nona-S (120mm, 8.8km)<br>2× BTR-ZD (23mm LV)<br><br><i>Luftlandsatt via Il-76</i><br>Uthållighet: 24-36h utan påfyllning<br>Saknar: MBT, IMR-2, Tor-M1",
        tooltip="HOSTILE: VDV BTG — Landvetter",
        icon=hostile_icon,
    ).add_to(m)

    # --- OWN FORCES (blue rectangles — friendly) ---
    friendly_icon = folium.DivIcon(
        html=friendly_bn_svg,
        icon_size=(60, 45),
        icon_anchor=(30, 22),
    )
    folium.Marker(
        [57.7000, 11.9200],
        popup="<b>FRIENDLY — Amfibiebataljon</b><br>4:e Amfibieregementet (Amf 4)<br>~400 pers<br>2× Amfibiekompani<br>16× CB90 (35+ knop)<br>4× RBS-17 (8km)<br>4× 81mm GrK (5.5km)<br>100× FFV 028 PV-mina<br>150 kg sprängdeg<br><br>SAKNAR: Pansar, LV, tung artilleri",
        tooltip="FRIENDLY: Amfbat — Göteborg",
        icon=friendly_icon,
    ).add_to(m)

    # Kustjägare
    kj_icon = folium.DivIcon(
        html=friendly_sf_svg,
        icon_size=(40, 35),
        icon_anchor=(20, 17),
    )
    folium.Marker(
        [57.6580, 12.25],
        popup="<b>FRIENDLY — Kustjägare (KJ)</b><br>202. Kustjägarkompaniet<br>20 man specialförband<br>Uppgift: Spaning/sabotage bakom fiendens linjer<br>2× Black Hornet nano-UAV<br>AG 90 (12.7mm prickskytte)",
        tooltip="FRIENDLY: Kustjägare — bakom fienden",
        icon=kj_icon,
    ).add_to(m)

    # RBS-17
    at_icon = folium.DivIcon(
        html=friendly_at_svg,
        icon_size=(40, 35),
        icon_anchor=(20, 17),
    )
    folium.Marker(
        [57.6750, 12.00],
        popup="<b>FRIENDLY — Kustrobotbatteri</b><br>4× RBS-17 (Hellfire-variant)<br>Räckvidd: 8 km<br>Penetrerar BMD-4M/Sprut-SD<br>16 robotar tillgängliga",
        tooltip="FRIENDLY: RBS-17 eldställning",
        icon=at_icon,
    ).add_to(m)

    # Enemy probable axes of advance (red arrows)
    folium.PolyLine(
        [[57.6686, 12.2919], [57.6650, 12.18], [57.6700, 12.07], [57.6950, 11.97]],
        color="red", weight=4, opacity=0.7, dash_array="10",
        tooltip="HOSTILE: Trolig huvudaxel (Rv 40)",
    ).add_to(m)

    folium.PolyLine(
        [[57.6686, 12.2919], [57.6400, 12.20], [57.6350, 12.10], [57.6550, 11.99]],
        color="darkred", weight=3, opacity=0.5, dash_array="10",
        tooltip="HOSTILE: Alternativaxel söder (Rv 27)",
    ).add_to(m)

    # --- BRIDGES (orange — demolition targets) ---
    bridge_svg = """
    <svg width="30" height="30" xmlns="http://www.w3.org/2000/svg">
      <rect x="3" y="3" width="24" height="24" fill="#fff3cd" stroke="#e65100" stroke-width="2"/>
      <text x="15" y="20" text-anchor="middle" font-size="14" fill="#e65100">⌇</text>
    </svg>"""
    bridge_icon = folium.DivIcon(html=bridge_svg, icon_size=(30, 30), icon_anchor=(15, 15))

    bridges = [
        {"pos": [57.6780, 12.01], "name": "Bro 1: Rv40 Mölndalsån", "info": "Motorvägsbro 25m, klass 70<br>Behov: 200 kg (25kg förtillv. + 175kg sprängdeg)<br>Tid: 4-5 tim, 4 man"},
        {"pos": [57.6720, 11.99], "name": "Bro 2: Göteborgsvägen", "info": "Stadsbro 15m, klass 60<br>Behov: 100 kg<br>Tid: 3 tim, 4 man"},
        {"pos": [57.6650, 11.97], "name": "Bro 3: Kvarnbygatan", "info": "Mindre bro 10m, klass 40<br>Behov: 75 kg<br>Tid: 2 tim, 2 man"},
    ]
    for bridge in bridges:
        folium.Marker(
            bridge["pos"],
            popup=f"<b>DEMOLITION — {bridge['name']}</b><br>{bridge['info']}<br><br><i>Status: SPRÄNGBEREDSKAP</i>",
            tooltip=f"🟠 {bridge['name']}",
            icon=bridge_icon,
        ).add_to(m)

    # Rv40 tunnel
    tunnel_svg = """
    <svg width="30" height="30" xmlns="http://www.w3.org/2000/svg">
      <rect x="3" y="3" width="24" height="24" fill="#fff3cd" stroke="#e65100" stroke-width="2"/>
      <text x="15" y="20" text-anchor="middle" font-size="12" fill="#e65100">T</text>
    </svg>"""
    tunnel_icon = folium.DivIcon(html=tunnel_svg, icon_size=(30, 30), icon_anchor=(15, 15))
    folium.Marker(
        [57.6850, 11.98],
        popup="<b>DEMOLITION — Rv40-tunneln Kallebäck</b><br>900m bergtunnel<br>Behov: 100 kg sprängdeg<br>Tid: 3-4 tim, 4 man<br><br><i>Effekt: PERMANENT BLOCKERING av Rv40</i>",
        tooltip="🟠 Tunnel Kallebäck",
        icon=tunnel_icon,
    ).add_to(m)

    # --- MINEFIELDS (green X pattern — NATO obstacle symbol) ---
    mine_svg = """
    <svg width="30" height="30" xmlns="http://www.w3.org/2000/svg">
      <line x1="5" y1="5" x2="25" y2="25" stroke="#2e7d32" stroke-width="3"/>
      <line x1="25" y1="5" x2="5" y2="25" stroke="#2e7d32" stroke-width="3"/>
      <line x1="5" y1="15" x2="25" y2="15" stroke="#2e7d32" stroke-width="2"/>
    </svg>"""
    mine_icon = folium.DivIcon(html=mine_svg, icon_size=(30, 30), icon_anchor=(15, 15))

    mine_positions = [
        {"pos": [57.6700, 12.10], "name": "Minfält 1: Rv40 Mölnlycke", "info": "PV-minfält (FFV 028)<br>~100 minor<br>Blockerar huvudaxeln<br>Fördröjningseffekt: 1-2 tim (manuell röjning)"},
        {"pos": [57.6400, 12.08], "name": "Minfält 2: Kållered syd", "info": "PV-minfält (FFV 028)<br>~50 minor<br>Hindrar flankering söder"},
        {"pos": [57.6800, 12.03], "name": "Minfält 3: Framför Mölndalsån", "info": "Kombinerat PV + trupp<br>Kanaliserar mot sprängda broar<br>Samverkar med RBS-17 eld"},
    ]
    for mf in mine_positions:
        folium.Marker(
            mf["pos"],
            popup=f"<b>OBSTACLE — {mf['name']}</b><br>{mf['info']}",
            tooltip=f"💣 {mf['name']}",
            icon=mine_icon,
        ).add_to(m)
        folium.Circle(
            mf["pos"], radius=250, color="#2e7d32", fill=True,
            fill_opacity=0.15, weight=2, dash_array="5,5",
            tooltip=mf["name"],
        ).add_to(m)

    # --- DELAY LINES (purple dashed — FEBA/PL) ---
    folium.PolyLine(
        [[57.6750, 12.12], [57.6600, 12.12], [57.6500, 12.10]],
        color="purple", weight=3, opacity=0.8, dash_array="5,10",
        tooltip="PL ALFA — Fördröjningslinje 1: MÖLNLYCKE (14 km)",
    ).add_to(m)

    folium.PolyLine(
        [[57.6850, 12.02], [57.6750, 12.00], [57.6600, 11.97]],
        color="purple", weight=3, opacity=0.8, dash_array="5,10",
        tooltip="PL BRAVO — Fördröjningslinje 2: MÖLNDALSÅN (6-8 km)",
    ).add_to(m)

    folium.PolyLine(
        [[57.6900, 11.97], [57.6800, 11.96], [57.6650, 11.95]],
        color="purple", weight=4, opacity=0.9, dash_array="5,10",
        tooltip="PL CHARLIE — Fördröjningslinje 3: MÖLNDAL (5 km) — SISTA LINJEN",
    ).add_to(m)

    # --- CB90 MANEUVER ROUTES (blue, waterborne) ---
    folium.PolyLine(
        [[57.7000, 11.92], [57.7100, 11.93], [57.7150, 11.95], [57.7050, 11.97]],
        color="#0055aa", weight=3, opacity=0.6,
        tooltip="FRIENDLY: CB90 manöverrutt via Göta Älv (flankering/omgruppering)",
    ).add_to(m)

    # --- LEGEND (NATO style) ---
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
         background: white; padding: 14px; border-radius: 4px; border: 2px solid #333;
         font-size: 11px; line-height: 2.0; font-family: monospace;">
    <b style="font-size:13px">TAKTISK ÖVERSIKT — NATO APP-6</b><br>
    <svg width="20" height="15"><rect x="1" y="1" width="18" height="13" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/></svg>
      FRIENDLY (bataljon/enhet)<br>
    <svg width="20" height="15"><polygon points="10,1 19,8 10,14 1,8" fill="#ffcccc" stroke="#cc0000" stroke-width="2"/></svg>
      HOSTILE (bataljon/enhet)<br>
    <svg width="20" height="15"><rect x="1" y="1" width="18" height="13" fill="#fff3cd" stroke="#e65100" stroke-width="2"/></svg>
      Sprängmål (bro/tunnel)<br>
    <svg width="20" height="15"><line x1="3" y1="3" x2="17" y2="13" stroke="#2e7d32" stroke-width="2"/><line x1="17" y1="3" x2="3" y2="13" stroke="#2e7d32" stroke-width="2"/></svg>
      Minfält/hinder<br>
    <span style="color:purple; font-weight:bold">▪ ▪ ▪</span> Fördröjningslinje (PL)<br>
    <span style="color:red">- - -</span> Fiendens framryckningsaxel<br>
    <span style="color:#0055aa">───</span> CB90 sjöväg
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m._repr_html_()

# ============================================================
# Tactical Simulation Engine
# ============================================================

def run_simulation():
    """
    Time-stepped tactical simulation: VDV advance vs Amf 4 delay.
    Based on scenario norms:
    - VDV advance rate with opposition: 5-8 km/day
    - Mine clearing (manual): 1-2 hours per field
    - Bridge demolition prep: 2-5 hours
    - VDV without obstacle clearance vehicle
    - Amf 4 CB90 repositioning: near-instant via water

    Timeline: 48 hours, 2-hour time steps = 24 frames
    """
    import time

    # Simulation state
    events = []
    sim_hours = 48
    step_hours = 2

    # --- UNIT POSITIONS (lat, lon) over time ---
    # VDV main body advances along Rv40: Landvetter → Göteborg
    vdv_route = [
        (57.6686, 12.2919),  # H+0: Landvetter
        (57.6686, 12.2919),  # H+2: Preparing (arty prep)
        (57.6680, 12.25),    # H+4: Start advance
        (57.6670, 12.22),    # H+6: Advancing
        (57.6680, 12.18),    # H+8: Approaching Mölnlycke
        (57.6700, 12.12),    # H+10: HIT MINEFIELD 1 — halted
        (57.6700, 12.12),    # H+12: Clearing mines (manual, 2h)
        (57.6700, 12.10),    # H+14: Mines cleared, resume
        (57.6710, 12.08),    # H+16: Advancing past PL ALFA
        (57.6720, 12.06),    # H+18: Continuing
        (57.6740, 12.04),    # H+20: Approaching Mölndalsån
        (57.6780, 12.03),    # H+22: HIT MINEFIELD 3 — halted
        (57.6780, 12.03),    # H+24: Clearing mines (2h)
        (57.6780, 12.01),    # H+26: Reach bridge 1 — DESTROYED
        (57.6780, 12.01),    # H+28: Attempting river crossing (no bridge)
        (57.6780, 12.01),    # H+30: Still crossing (improvised)
        (57.6790, 12.00),    # H+32: Crossed Mölndalsån
        (57.6800, 11.99),    # H+34: Approaching tunnel
        (57.6850, 11.98),    # H+36: Tunnel BLOCKED — rerouting
        (57.6850, 11.98),    # H+38: Finding alternate route
        (57.6830, 11.97),    # H+40: Rerouted via local roads
        (57.6850, 11.96),    # H+42: Reaching PL CHARLIE
        (57.6870, 11.96),    # H+44: Engaged by Amfbat in Mölndal
        (57.6870, 11.96),    # H+46: HALTED — combat in Mölndal
    ]

    # VDV flanking force (southern route)
    vdv_flank_route = [
        (57.6686, 12.2919),  # H+0
        (57.6686, 12.2919),  # H+2
        (57.6500, 12.25),    # H+4
        (57.6400, 12.20),    # H+6
        (57.6380, 12.15),    # H+8
        (57.6400, 12.10),    # H+10: HIT MINEFIELD 2
        (57.6400, 12.10),    # H+12: Clearing (2h)
        (57.6400, 12.08),    # H+14
        (57.6420, 12.05),    # H+16
        (57.6450, 12.02),    # H+18: Engaged by RBS-17 (8km range)
        (57.6450, 12.02),    # H+20: 2× BMD-4M destroyed, withdrawing
        (57.6430, 12.04),    # H+22: Pulling back
        (57.6430, 12.04),    # H+24: Regrouping
        (57.6450, 12.02),    # H+26: Second attempt
        (57.6480, 12.00),    # H+28
        (57.6500, 11.99),    # H+30
        (57.6520, 11.98),    # H+32: Approaching Mölndal south
        (57.6550, 11.97),    # H+34
        (57.6580, 11.96),    # H+36
        (57.6600, 11.96),    # H+38: Engaged in urban combat
        (57.6600, 11.96),    # H+40: HALTED in Kållered
        (57.6600, 11.96),    # H+42
        (57.6600, 11.96),    # H+44
        (57.6600, 11.96),    # H+46: Still held
    ]

    # Amf 4 main body (repositions via CB90)
    amf_route = [
        (57.7000, 11.92),    # H+0: Göteborg
        (57.7000, 11.92),    # H+2
        (57.6900, 11.94),    # H+4: Moving to forward positions
        (57.6850, 11.96),    # H+6: At PL CHARLIE
        (57.6850, 11.96),    # H+8
        (57.6750, 12.00),    # H+10: Forward at PL BRAVO (bridges)
        (57.6750, 12.00),    # H+12: Preparing bridge demolition
        (57.6750, 12.00),    # H+14
        (57.6750, 12.00),    # H+16
        (57.6750, 12.00),    # H+18
        (57.6750, 12.00),    # H+20: BLOW bridges (all 3)
        (57.6750, 12.00),    # H+22
        (57.6750, 12.00),    # H+24: BLOW tunnel
        (57.6800, 11.97),    # H+26: Withdraw to PL CHARLIE
        (57.6850, 11.96),    # H+28: Defensive position Mölndal
        (57.6850, 11.96),    # H+30
        (57.6850, 11.96),    # H+32
        (57.6850, 11.96),    # H+34
        (57.6850, 11.96),    # H+36
        (57.6850, 11.96),    # H+38
        (57.6850, 11.96),    # H+40: Defending Mölndal
        (57.6850, 11.96),    # H+42
        (57.6850, 11.96),    # H+44: VDV arrives — contact
        (57.6850, 11.96),    # H+46: HOLDING
    ]

    # Events timeline
    event_log = [
        (0, "H+00: VDV BTG grupperat vid Landvetter. Amfbat förbereder fördröjning."),
        (2, "H+02: VDV artilleriförberedelse (2S9 Nona-S, 120mm) mot Mölnlycke."),
        (4, "H+04: VDV påbörjar framryckning längs Rv40. Flankerande styrka söderut."),
        (6, "H+06: Kustjägare rapporterar fiendens rörelser. Amfbat framgrupperar."),
        (8, "H+08: VDV huvudstyrka närmar sig PL ALFA (Mölnlycke)."),
        (10, "H+10: ⚠️ VDV TRÄFFAR MINFÄLT 1 vid Mölnlycke! 1× BMD-4M förstörd. Halt."),
        (12, "H+12: VDV manuell minröjning (saknar IMR-2). Flanken träffar minfält 2."),
        (14, "H+14: Minor röjda. VDV återupptar framryckning. Förlorat 4 timmar."),
        (16, "H+16: Amf 4 pionjärer FÖRBEREDER BROSPRÄNGNING vid Mölndalsån."),
        (18, "H+18: 🚀 RBS-17 ENGAGERAR flankerande BMD-4M! 2 st förstörda på 6 km."),
        (20, "H+20: VDV flank drar sig tillbaka. Amfbat SPRÄNGER alla 3 broar! 💥"),
        (22, "H+22: VDV huvudstyrka träffar minfält 3. Ytterligare fördröjning."),
        (24, "H+24: 💥 TUNNEL KALLEBÄCK SPRÄNGD — permanent blockering av Rv40."),
        (26, "H+26: VDV når Mölndalsån — broarna förstörda. Tvingas improvisera."),
        (28, "H+28: VDV försöker övergå Mölndalsån utan bro. Amfbat eld med 81mm GrK."),
        (30, "H+30: VDV lyckas korsa ån med förluster. 3× BMD-4M kvar av 18."),
        (32, "H+32: VDV når blockerad tunnel. Måste ta omväg via stadsgatorna."),
        (34, "H+34: Amfbat omgrupperar till Mölndal via CB90 (10 min sjöväg)."),
        (36, "H+36: VDV försöker kringgå tunnel. Kanaliseras i tät bebyggelse."),
        (38, "H+38: Strid i Mölndal. Carl Gustaf effektiv mot BMD-4M i närstrid."),
        (40, "H+40: VDV flank HEJDAD i Kållered. Begränsad ammunition kvar."),
        (42, "H+42: VDV stridsvärde kraftigt reducerat. Begär flygburen förstärkning."),
        (44, "H+44: VDV huvudstyrka i kontakt med Amfbat vid PL CHARLIE."),
        (46, "H+46: ✅ 46 TIMMAR — VDV HEJDAD. Uppgift nära löst (48h mål)."),
    ]

    # VDV strength over time (vehicles remaining)
    vdv_strength = [
        18, 18, 18, 18, 18,   # H+0 to H+8
        17, 17, 17, 17, 15,   # H+10 to H+18 (mine + RBS-17 losses)
        15, 14, 14, 14, 12,   # H+20 to H+28 (crossing losses)
        10, 10, 10, 10, 8,    # H+30 to H+38 (urban combat)
        7, 6, 5, 5,           # H+40 to H+46
    ]

    # Generate frames
    frames_html = []
    for step in range(min(len(vdv_route), sim_hours // step_hours)):
        hour = step * step_hours

        m = folium.Map(location=[57.68, 12.05], zoom_start=12, tiles="OpenStreetMap")

        # Time indicator
        time_html = f"""
        <div style="position: fixed; top: 10px; right: 10px; z-index: 1000;
             background: #333; color: white; padding: 10px 16px; border-radius: 4px;
             font-size: 16px; font-family: monospace; font-weight: bold;">
        ⏱️ H+{hour:02d} ({hour//24}d {hour%24:02d}h) | VDV BMD: {vdv_strength[step]}/18
        </div>"""
        m.get_root().html.add_child(folium.Element(time_html))

        # Event log
        current_events = [e[1] for e in event_log if e[0] <= hour]
        last_events = current_events[-4:] if len(current_events) > 4 else current_events
        event_html = f"""
        <div style="position: fixed; bottom: 10px; right: 10px; z-index: 1000;
             background: rgba(0,0,0,0.85); color: #0f0; padding: 10px; border-radius: 4px;
             font-size: 10px; font-family: monospace; max-width: 400px; line-height: 1.6;">
        <b>HÄNDELSELOGG</b><br>{'<br>'.join(last_events)}
        </div>"""
        m.get_root().html.add_child(folium.Element(event_html))

        # VDV main body
        hostile_bn_svg_small = f"""
        <svg width="44" height="38" xmlns="http://www.w3.org/2000/svg">
          <polygon points="22,2 42,19 22,36 2,19" fill="#ffcccc" stroke="#cc0000" stroke-width="2"/>
          <text x="22" y="23" text-anchor="middle" font-size="8" font-weight="bold" fill="#cc0000">VDV</text>
        </svg>"""
        folium.Marker(
            list(vdv_route[step]),
            tooltip=f"HOSTILE: VDV huvudstyrka (H+{hour})",
            icon=folium.DivIcon(html=hostile_bn_svg_small, icon_size=(44, 38), icon_anchor=(22, 19)),
        ).add_to(m)

        # VDV flank
        if step < len(vdv_flank_route):
            hostile_flank_svg = f"""
            <svg width="34" height="30" xmlns="http://www.w3.org/2000/svg">
              <polygon points="17,2 32,15 17,28 2,15" fill="#ffcccc" stroke="#cc0000" stroke-width="2"/>
              <text x="17" y="18" text-anchor="middle" font-size="6" fill="#cc0000">FLK</text>
            </svg>"""
            folium.Marker(
                list(vdv_flank_route[step]),
                tooltip=f"HOSTILE: VDV flank (H+{hour})",
                icon=folium.DivIcon(html=hostile_flank_svg, icon_size=(34, 30), icon_anchor=(17, 15)),
            ).add_to(m)

        # Amf 4
        friendly_bn_svg_small = f"""
        <svg width="44" height="34" xmlns="http://www.w3.org/2000/svg">
          <rect x="2" y="6" width="40" height="24" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/>
          <text x="22" y="22" text-anchor="middle" font-size="8" font-weight="bold" fill="#0055aa">AMF</text>
        </svg>"""
        folium.Marker(
            list(amf_route[step]),
            tooltip=f"FRIENDLY: Amfbat (H+{hour})",
            icon=folium.DivIcon(html=friendly_bn_svg_small, icon_size=(44, 34), icon_anchor=(22, 17)),
        ).add_to(m)

        # Phase lines (always visible)
        folium.PolyLine([[57.6750, 12.12], [57.6600, 12.12], [57.6500, 12.10]],
            color="purple", weight=2, opacity=0.5, dash_array="5,10", tooltip="PL ALFA").add_to(m)
        folium.PolyLine([[57.6850, 12.02], [57.6750, 12.00], [57.6600, 11.97]],
            color="purple", weight=2, opacity=0.5, dash_array="5,10", tooltip="PL BRAVO").add_to(m)
        folium.PolyLine([[57.6900, 11.97], [57.6800, 11.96], [57.6650, 11.95]],
            color="purple", weight=3, opacity=0.7, dash_array="5,10", tooltip="PL CHARLIE").add_to(m)

        # Minefields (show as active until cleared)
        mine_svg = '<svg width="20" height="20"><line x1="3" y1="3" x2="17" y2="17" stroke="#2e7d32" stroke-width="2"/><line x1="17" y1="3" x2="3" y2="17" stroke="#2e7d32" stroke-width="2"/></svg>'
        mine_cleared_svg = '<svg width="20" height="20"><line x1="3" y1="3" x2="17" y2="17" stroke="#999" stroke-width="2"/><line x1="17" y1="3" x2="3" y2="17" stroke="#999" stroke-width="2"/><line x1="2" y1="10" x2="18" y2="10" stroke="red" stroke-width="2"/></svg>'

        # Minefield 1: cleared at H+14
        mf1_svg = mine_cleared_svg if hour >= 14 else mine_svg
        folium.Marker([57.6700, 12.10], tooltip="Minfält 1" + (" (RÖJT)" if hour >= 14 else ""),
            icon=folium.DivIcon(html=mf1_svg, icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

        # Minefield 2: cleared at H+14
        mf2_svg = mine_cleared_svg if hour >= 14 else mine_svg
        folium.Marker([57.6400, 12.08], tooltip="Minfält 2" + (" (RÖJT)" if hour >= 14 else ""),
            icon=folium.DivIcon(html=mf2_svg, icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

        # Minefield 3: cleared at H+24
        mf3_svg = mine_cleared_svg if hour >= 24 else mine_svg
        folium.Marker([57.6800, 12.03], tooltip="Minfält 3" + (" (RÖJT)" if hour >= 24 else ""),
            icon=folium.DivIcon(html=mf3_svg, icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

        # Bridges (show destroyed after H+20)
        bridge_svg_intact = '<svg width="20" height="20"><rect x="2" y="2" width="16" height="16" fill="#fff3cd" stroke="#e65100" stroke-width="2"/><text x="10" y="14" text-anchor="middle" font-size="10" fill="#e65100">⌇</text></svg>'
        bridge_svg_blown = '<svg width="20" height="20"><rect x="2" y="2" width="16" height="16" fill="#ffcdd2" stroke="#b71c1c" stroke-width="2"/><text x="10" y="14" text-anchor="middle" font-size="10" fill="#b71c1c">✕</text></svg>'

        for bpos in [[57.6780, 12.01], [57.6720, 11.99], [57.6650, 11.97]]:
            svg = bridge_svg_blown if hour >= 20 else bridge_svg_intact
            label = "FÖRSTÖRD" if hour >= 20 else "Intakt"
            folium.Marker(bpos, tooltip=f"Bro ({label})",
                icon=folium.DivIcon(html=svg, icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

        # Tunnel (blocked after H+24)
        if hour >= 24:
            folium.Marker([57.6850, 11.98], tooltip="Tunnel BLOCKERAD 💥",
                icon=folium.DivIcon(html='<svg width="20" height="20"><rect x="2" y="2" width="16" height="16" fill="#ffcdd2" stroke="#b71c1c" stroke-width="2"/><text x="10" y="14" text-anchor="middle" font-size="8" fill="#b71c1c">T✕</text></svg>',
                icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)
        else:
            folium.Marker([57.6850, 11.98], tooltip="Tunnel (intakt)",
                icon=folium.DivIcon(html='<svg width="20" height="20"><rect x="2" y="2" width="16" height="16" fill="#fff3cd" stroke="#e65100" stroke-width="2"/><text x="10" y="14" text-anchor="middle" font-size="8" fill="#e65100">T</text></svg>',
                icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

        # Combat markers
        if hour >= 18 and hour <= 22:
            folium.Marker([57.6450, 12.02], tooltip="💥 RBS-17 engagerar!",
                icon=folium.DivIcon(html='<div style="font-size:20px">💥</div>', icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

        if hour >= 38:
            folium.Marker([57.6850, 11.96], tooltip="⚔️ Strid i Mölndal",
                icon=folium.DivIcon(html='<div style="font-size:20px">⚔️</div>', icon_size=(20, 20), icon_anchor=(10, 10))).add_to(m)

        frames_html.append(m._repr_html_())

    return frames_html


def run_simulation_ui(progress=gr.Progress()):
    """Run simulation and save as standalone HTML file, then return link."""
    progress(0.1, desc="🎬 Genererar simuleringsramar...")
    frames = run_simulation()
    progress(0.9, desc="✅ Simulering klar!")

    # Build standalone HTML file with animation
    frames_js = json.dumps(frames)
    html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Taktisk Simulering — 48h</title></head>
<body style="margin:0; font-family:monospace; background:#1a1a1a;">
<div style="text-align:center; padding:10px; background:#333; color:white;">
    <button onclick="playPause()" id="play-btn" style="font-size:16px; padding:8px 16px; cursor:pointer; background:#4CAF50; color:white; border:none; border-radius:4px;">⏸️ Paus</button>
    <button onclick="stepBack()" style="font-size:16px; padding:8px 16px; cursor:pointer; border:none; border-radius:4px;">⏮️</button>
    <button onclick="stepForward()" style="font-size:16px; padding:8px 16px; cursor:pointer; border:none; border-radius:4px;">⏭️</button>
    <input type="range" id="sim-slider" min="0" max="{len(frames)-1}" value="0"
           style="width:300px; vertical-align:middle;" oninput="goToFrame(parseInt(this.value))">
    <span id="frame-label" style="margin-left:10px; color:#0f0; font-size:18px;">H+00</span>
    <span style="margin-left:20px; color:#aaa;">Hastighet:</span>
    <button onclick="setSpeed(2000)" style="padding:4px 8px; cursor:pointer;">1×</button>
    <button onclick="setSpeed(1000)" style="padding:4px 8px; cursor:pointer; background:#555; color:white; border:none;">2×</button>
    <button onclick="setSpeed(500)" style="padding:4px 8px; cursor:pointer;">4×</button>
</div>
<iframe id="sim-frame" style="width:100%; height:calc(100vh - 60px); border:none;"></iframe>
<script>
    var frames = {frames_js};
    var currentFrame = 0;
    var playing = true;
    var speed = 1500;
    var interval = null;

    function showFrame(idx) {{
        currentFrame = idx;
        var iframe = document.getElementById('sim-frame');
        iframe.srcdoc = frames[idx];
        document.getElementById('sim-slider').value = idx;
        document.getElementById('frame-label').textContent = 'H+' + String(idx * 2).padStart(2, '0');
    }}

    function playPause() {{
        playing = !playing;
        document.getElementById('play-btn').textContent = playing ? '⏸️ Paus' : '▶️ Spela';
        document.getElementById('play-btn').style.background = playing ? '#4CAF50' : '#2196F3';
        if (playing) startPlay();
        else clearInterval(interval);
    }}

    function stepForward() {{
        if (currentFrame < frames.length - 1) showFrame(currentFrame + 1);
    }}

    function stepBack() {{
        if (currentFrame > 0) showFrame(currentFrame - 1);
    }}

    function goToFrame(idx) {{
        showFrame(idx);
    }}

    function setSpeed(ms) {{
        speed = ms;
        if (playing) {{ clearInterval(interval); startPlay(); }}
    }}

    function startPlay() {{
        interval = setInterval(function() {{
            if (currentFrame < frames.length - 1) {{
                showFrame(currentFrame + 1);
            }} else {{
                playing = false;
                document.getElementById('play-btn').textContent = '▶️ Spela igen';
                document.getElementById('play-btn').style.background = '#2196F3';
                clearInterval(interval);
            }}
        }}, speed);
    }}

    showFrame(0);
    startPlay();
</script>
</body></html>"""

    # Save to file and serve via local HTTP
    sim_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation.html")
    with open(sim_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Open in browser
    import webbrowser
    webbrowser.open("http://localhost:7860/file=simulation.html")

    return f"✅ Simulering sparad.\n\nLokal: http://localhost:7860/file=simulation.html"

# ============================================================
# Gradio Web Interface
# ============================================================
print("Starting web interface...")

with gr.Blocks(title="Försvarsfrågor RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇸🇪 Försvarsfrågor — RAG + Scenario Simulator")
    gr.Markdown("Ställ frågor om svenska försvarsdokument. Redigera scenariot för taktisk analys.")

    with gr.Row():
        with gr.Column(scale=1):
            scenario_box = gr.Textbox(
                label="Scenario (redigerbart)",
                value=DEFAULT_SCENARIO,
                lines=20,
                max_lines=40,
            )
            sim_btn = gr.Button(
                "▶️ Kör simulering (48h)" if not SHARE_MODE else "▶️ Simulering (avstängd i delat läge)",
                variant="secondary",
                interactive=not SHARE_MODE,
            )
            video_btn = gr.Button("📹 Exportera video", variant="secondary", interactive=not SHARE_MODE)
            video_download = gr.File(label="Ladda ner video", visible=False)
            parse_status = gr.Markdown(value="", visible=False)

        with gr.Column(scale=1):
            question_box = gr.Textbox(
                label="Fråga",
                placeholder="Ställ din fråga här...",
                lines=3,
            )
            submit_btn = gr.Button("Fråga", variant="primary", size="lg")
            answer_box = gr.Markdown(
                label="Svar",
                value="*Väntar på fråga...*",
            )

    # Simulation section
    with gr.Row(visible=False) as sim_row:
        sim_status = gr.Markdown(label="Simulering")

    def show_simulation(scenario_text):
        """Generate simulation via Claude. Uses yield for immediate feedback."""
        import threading, http.server, socketserver, webbrowser
        from sim_generator import generate_simulation_timeline

        # Simulation is disabled when running shared (--share): the tunnel plus
        # simulation generation spike system RAM and can OOM-kill the server.
        if SHARE_MODE:
            yield gr.update(visible=True), (
                "⚠️ **Simulering är avstängd i delat läge (--share).**\n\n"
                "Simuleringen kräver mycket systemminne och körs lokalt på port 7861, "
                "vilket inte fungerar över den delade länken. Starta servern utan "
                "`--share` för att köra simuleringen lokalt."
            )
            return

        # Start HTTP server if not already running
        def serve():
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            server = socketserver.TCPServer(("", 7861), type("Q", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None}))
            server.allow_reuse_address = True
            server.serve_forever()
        if not hasattr(show_simulation, '_s'):
            threading.Thread(target=serve, daemon=True).start()
            show_simulation._s = True

        # Immediate feedback to user
        yield gr.update(visible=True), "⏳ **Genererar taktisk simulering...**\n\nClaude analyserar scenariot och hämtar doktrin.\nDetta tar ca 20-30 sekunder."

        # Open browser for local users
        webbrowser.open("http://localhost:7861/simulation.html")

        # Write a "loading" page immediately
        loading_html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Simulering laddas...</title>
<meta http-equiv="refresh" content="3">
</head>
<body style="margin:0; display:flex; align-items:center; justify-content:center; height:100vh; background:#1a1a1a; font-family:monospace;">
<div style="text-align:center; color:white;">
<h1 style="color:#4CAF50;">⏳ Genererar taktisk simulering...</h1>
<p style="color:#aaa; font-size:18px;">Claude analyserar scenariot och skapar 24 tidsramar.<br>Detta tar ca 15-20 sekunder.</p>
<div style="margin-top:30px;"><div style="width:60px;height:60px;border:6px solid #333;border-top:6px solid #4CAF50;border-radius:50%;animation:spin 1s linear infinite;margin:auto;"></div></div>
<p style="color:#666; margin-top:20px;">Sidan uppdateras automatiskt...</p>
</div>
<style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
</body></html>"""

        sim_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation.html")
        with open(sim_file, "w", encoding="utf-8") as f:
            f.write(loading_html)

        # Now generate the actual simulation (this takes 15-20s)
        frames_data = generate_simulation_timeline(scenario_text)

        if not frames_data:
            error_html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Fel</title></head>
<body style="margin:0; display:flex; align-items:center; justify-content:center; height:100vh; background:#1a1a1a; font-family:monospace;">
<div style="text-align:center; color:#ff5252;"><h1>❌ Simulering misslyckades</h1><p>Claude kunde inte generera en tidslinje. Försök igen.</p></div>
</body></html>"""
            with open(sim_file, "w", encoding="utf-8") as f:
                f.write(error_html)
            yield gr.update(visible=True), "❌ Simulering misslyckades."
            return

        # Render frames
        frames_html = []
        for frame in frames_data:
            hour = frame.get("hour", 0)
            events = frame.get("events", [])
            units = frame.get("units", [])
            structures = frame.get("structures", [])
            minefields_data = frame.get("minefields", [])

            all_pos = [u["position"] for u in units if u.get("position")]
            if all_pos:
                clat = sum(p[0] for p in all_pos) / len(all_pos)
                clon = sum(p[1] for p in all_pos) / len(all_pos)
            else:
                clat, clon = 57.68, 12.05

            fm = folium.Map(location=[clat, clon], zoom_start=12, tiles="OpenStreetMap")

            # Time + vehicle count
            hostile_veh = sum(u.get("vehicles_remaining", 0) for u in units if u.get("side") == "hostile")
            time_div = f'<div style="position:fixed;top:10px;right:10px;z-index:1000;background:#333;color:white;padding:8px 14px;border-radius:4px;font-family:monospace;font-size:15px;">⏱️ H+{hour:02d} | Fiende fordon: {hostile_veh}</div>'
            fm.get_root().html.add_child(folium.Element(time_div))

            # Units
            for u in units:
                if not u.get("position"):
                    continue
                pos = u["position"]
                name = u.get("name", u.get("id", "?"))
                side = u.get("side", "neutral")
                veh = u.get("vehicles_remaining", "?")
                activity = u.get("activity", "")

                if side == "friendly":
                    svg = f'<svg width="40" height="32"><rect x="2" y="6" width="36" height="22" fill="#cce5ff" stroke="#0055aa" stroke-width="2"/><text x="20" y="21" text-anchor="middle" font-size="7" font-weight="bold" fill="#0055aa">{name[:5]}</text></svg>'
                else:
                    svg = f'<svg width="40" height="36"><polygon points="20,2 38,18 20,34 2,18" fill="#ffcccc" stroke="#cc0000" stroke-width="2"/><text x="20" y="21" text-anchor="middle" font-size="7" font-weight="bold" fill="#cc0000">{name[:5]}</text></svg>'

                popup = f"<b>{name}</b><br>Sida: {side}<br>Fordon: {veh}<br>{activity}"
                icon = folium.DivIcon(html=svg, icon_size=(40, 36), icon_anchor=(20, 18))
                folium.Marker(pos, popup=popup, tooltip=f"{name} ({activity})", icon=icon).add_to(fm)

            # Structures
            for s in structures:
                if not s.get("position"):
                    continue
                status = s.get("status", "intact")
                symbol = "✕" if status == "destroyed" else ("⚡" if status == "prepared" else "⌇")
                color = "#b71c1c" if status == "destroyed" else "#e65100"
                svg = f'<svg width="20" height="20"><rect x="2" y="2" width="16" height="16" fill="white" stroke="{color}" stroke-width="2"/><text x="10" y="14" text-anchor="middle" font-size="10" fill="{color}">{symbol}</text></svg>'
                folium.Marker(s["position"], tooltip=f"{s.get('name', '?')} ({status})",
                    icon=folium.DivIcon(html=svg, icon_size=(20, 20), icon_anchor=(10, 10))).add_to(fm)

            # Minefields
            for mf in minefields_data:
                if not mf.get("position"):
                    continue
                cleared = mf.get("cleared", False)
                color = "#999" if cleared else "#2e7d32"
                svg = f'<svg width="18" height="18"><line x1="3" y1="3" x2="15" y2="15" stroke="{color}" stroke-width="2"/><line x1="15" y1="3" x2="3" y2="15" stroke="{color}" stroke-width="2"/></svg>'
                folium.Marker(mf["position"], tooltip=mf.get("name", "?") + (" (RÖJT)" if cleared else ""),
                    icon=folium.DivIcon(html=svg, icon_size=(18, 18), icon_anchor=(9, 9))).add_to(fm)

            frames_html.append(fm._repr_html_())
            # Also store full standalone HTML for video export
            frames_html_full = getattr(show_simulation, '_full_frames', [])
            frames_html_full.append(fm.get_root().render())
            show_simulation._full_frames = frames_html_full

        # Collect all events with their hour for the sidebar
        import json as json_mod
        all_events_by_hour = {}
        for frame in frames_data:
            h = frame.get("hour", 0)
            for evt in frame.get("events", []):
                all_events_by_hour.setdefault(h, []).append(evt)

        # Save frames to disk for video export (includes event sidebar)
        import json as json_save
        video_frames = []
        full_frames = getattr(show_simulation, '_full_frames', frames_html)
        for i, frame_html in enumerate(full_frames):
            hour = i * 2
            events_up_to_now = []
            for h in sorted(all_events_by_hour.keys()):
                if h <= hour:
                    for evt in all_events_by_hour[h]:
                        events_up_to_now.append(evt)

            last_5 = events_up_to_now[-8:] if len(events_up_to_now) > 8 else events_up_to_now
            events_html_str = "".join(f'<div style="margin:3px 0;color:{"#4CAF50" if "H+{0:02d}".format(hour) in e else "#aaa"};font-size:10px;">{e}</div>' for e in last_5)

            # For video: inject event overlay directly into the folium HTML
            sidebar_div = f"""<div style="position:fixed;top:10px;left:10px;width:260px;max-height:calc(100vh - 20px);background:rgba(0,0,0,0.88);color:#ccc;padding:10px;overflow-y:auto;border-radius:4px;border:1px solid #444;z-index:99999;font-family:monospace;">
<h3 style="color:#4CAF50;margin:0 0 8px 0;font-size:12px;">📋 HÄNDELSER</h3>
{events_html_str}
</div>
<div style="position:fixed;top:10px;right:10px;z-index:99999;background:#333;color:white;padding:8px 14px;border-radius:4px;font-family:monospace;font-size:16px;">⏱️ H+{hour:02d}</div>"""

            # Insert overlay into the folium HTML (before </body>)
            if "</body>" in frame_html:
                composite = frame_html.replace("</body>", f"{sidebar_div}</body>")
            else:
                composite = frame_html + sidebar_div

            video_frames.append(composite)

        # Reset full frames for next run
        show_simulation._full_frames = []

        frames_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_frames.json")
        with open(frames_file, "w", encoding="utf-8") as ff:
            json_save.dump(video_frames, ff)

        all_events_flat = []
        for h in sorted(all_events_by_hour.keys()):
            for evt in all_events_by_hour[h]:
                all_events_flat.append({"hour": h, "text": evt})

        events_js = json_mod.dumps(all_events_flat, ensure_ascii=False)
        frames_js = json_mod.dumps(frames_html)

        final_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Taktisk Simulering — {len(frames_data)} steg</title></head>
<body style="margin:0; font-family:monospace; background:#1a1a1a; overflow:hidden;">
<div style="text-align:center; padding:8px; background:#333; color:white;">
    <button onclick="playPause()" id="play-btn" style="font-size:14px; padding:6px 14px; cursor:pointer; background:#4CAF50; color:white; border:none; border-radius:4px;">⏸️ Paus</button>
    <button onclick="stepBack()" style="font-size:14px; padding:6px 14px; cursor:pointer; border:none; border-radius:4px;">⏮️</button>
    <button onclick="stepForward()" style="font-size:14px; padding:6px 14px; cursor:pointer; border:none; border-radius:4px;">⏭️</button>
    <input type="range" id="sim-slider" min="0" max="{len(frames_html)-1}" value="0" style="width:250px; vertical-align:middle;" oninput="goToFrame(parseInt(this.value))">
    <span id="frame-label" style="margin-left:8px; color:#0f0; font-size:16px;">H+00</span>
    <button onclick="setSpeed(3000)" style="padding:3px 6px; cursor:pointer; font-size:12px;">1×</button>
    <button onclick="setSpeed(1500)" style="padding:3px 6px; cursor:pointer; font-size:12px; background:#555; color:white; border:none;">2×</button>
    <button onclick="setSpeed(750)" style="padding:3px 6px; cursor:pointer; font-size:12px;">4×</button>
</div>
<div style="display:flex; height:calc(100vh - 50px);">
    <div id="event-sidebar" style="width:320px; background:#111; color:#ccc; overflow-y:auto; padding:10px; border-right:2px solid #333; font-size:11px; line-height:1.8;">
        <h3 style="color:#4CAF50; margin:0 0 10px 0; font-size:13px;">📋 HÄNDELSELOGG</h3>
        <div id="event-list"></div>
    </div>
    <iframe id="sim-frame" style="flex:1; border:none;"></iframe>
</div>
<script>
    var frames = {frames_js};
    var allEvents = {events_js};
    var currentFrame = 0; var playing = true; var speed = 2500; var interval = null;

    function renderEvents(currentHour) {{
        var html = '';
        for (var i = 0; i < allEvents.length; i++) {{
            var evt = allEvents[i];
            var isCurrent = (evt.hour == currentHour);
            var isPast = (evt.hour < currentHour);
            var isFuture = (evt.hour > currentHour);
            var style = '';
            if (isCurrent) style = 'background:#1b5e20; color:#4CAF50; padding:3px 6px; border-radius:3px; font-weight:bold;';
            else if (isPast) style = 'color:#888;';
            else style = 'color:#444;';
            html += '<div style="margin:2px 0; ' + style + '">' + evt.text + '</div>';
        }}
        document.getElementById('event-list').innerHTML = html;
        // Scroll to current event
        var items = document.getElementById('event-list').children;
        for (var i = 0; i < items.length; i++) {{
            if (items[i].style.fontWeight == 'bold') {{
                items[i].scrollIntoView({{ block: 'center', behavior: 'smooth' }});
                break;
            }}
        }}
    }}

    function showFrame(idx) {{
        currentFrame = idx;
        document.getElementById('sim-frame').srcdoc = frames[idx];
        document.getElementById('sim-slider').value = idx;
        var hour = idx * 2;
        document.getElementById('frame-label').textContent = 'H+' + String(hour).padStart(2, '0');
        renderEvents(hour);
    }}
    function playPause() {{ playing = !playing; document.getElementById('play-btn').textContent = playing ? '⏸️ Paus' : '▶️ Spela'; if (playing) startPlay(); else clearInterval(interval); }}
    function stepForward() {{ if (currentFrame < frames.length - 1) showFrame(currentFrame + 1); }}
    function stepBack() {{ if (currentFrame > 0) showFrame(currentFrame - 1); }}
    function goToFrame(idx) {{ showFrame(idx); }}
    function setSpeed(ms) {{ speed = ms; if (playing) {{ clearInterval(interval); startPlay(); }} }}
    function startPlay() {{ interval = setInterval(function() {{ if (currentFrame < frames.length - 1) showFrame(currentFrame + 1); else {{ playing = false; document.getElementById('play-btn').textContent = '▶️ Igen'; clearInterval(interval); }} }}, speed); }}
    showFrame(0); startPlay();
</script>
</body></html>"""

        with open(sim_file, "w", encoding="utf-8") as f:
            f.write(final_html)

        # Save complete event log
        all_events = []
        for frame in frames_data:
            for event in frame.get("events", []):
                all_events.append(event)

        log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_log.md")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("# Simuleringslogg\n\n")
            f.write(f"Genererad: {len(frames_data)} tidsramar (H+00 till H+{frames_data[-1].get('hour', '?')})\n\n")
            f.write("## Händelser\n\n")
            for event in all_events:
                f.write(f"- {event}\n")
            f.write(f"\n## Slutstatus\n\n")
            last_frame = frames_data[-1]
            for u in last_frame.get("units", []):
                f.write(f"- **{u.get('name', '?')}** ({u.get('side')}): fordon={u.get('vehicles_remaining', '?')}, aktivitet={u.get('activity', '?')}\n")
            for s in last_frame.get("structures", []):
                f.write(f"- **{s.get('name', '?')}**: {s.get('status', '?')}\n")

        yield gr.update(visible=True), f"✅ Simulering klar ({len(frames_data)} steg)\n\nSimulering öppnad: http://localhost:7861/simulation.html\n\nHändelselogg sparad till: `simulation_log.md`"

    submit_btn.click(
        fn=ask_question,
        inputs=[question_box, scenario_box],
        outputs=answer_box,
        show_progress="full",
    )
    question_box.submit(
        fn=ask_question,
        inputs=[question_box, scenario_box],
        outputs=answer_box,
        show_progress="full",
    )
    sim_btn.click(
        fn=show_simulation,
        inputs=[scenario_box],
        outputs=[sim_row, sim_status],
    )

    def export_video_subprocess():
        """Export video as a separate process, then offer for download."""
        frames_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_frames.json")
        if not os.path.exists(frames_file):
            return gr.update(visible=True), "❌ Kör en simulering först — inga frames sparade.", gr.update(visible=False)

        import subprocess
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simulation_video.mp4")
        script = f"""
import json
from video_export import export_simulation_video
with open('{frames_file}', encoding='utf-8') as f:
    frames = json.load(f)
export_simulation_video(frames, '{video_path}')
"""
        result = subprocess.run(
            ["python3", "-c", script],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0 and os.path.exists(video_path):
            size_mb = os.path.getsize(video_path) / (1024 * 1024)
            return (
                gr.update(visible=True),
                f"✅ Video exporterad ({size_mb:.1f} MB) — ladda ner nedan:",
                gr.update(visible=True, value=video_path),
            )
        return gr.update(visible=True), f"❌ Export misslyckades:\n```\n{result.stderr[:300]}\n```", gr.update(visible=False)

    video_btn.click(
        fn=export_video_subprocess,
        outputs=[sim_row, sim_status, video_download],
        show_progress="full",
    )

if SHARE_MODE:
    print("[WebInterface] Running in SHARE mode — simulation preparation is disabled.")
demo.launch(server_name="0.0.0.0", server_port=7860, share=SHARE_MODE,
            allowed_paths=[os.path.dirname(os.path.abspath(__file__))])
