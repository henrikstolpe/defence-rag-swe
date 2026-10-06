"""
Extended evaluation script for the Swedish RAG system.
Uses a single fine-tuned AI-Sweden Llama 3 8B model for both classification and generation.
No model swap needed. Uses shared RAG retrieval module.
"""
import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from rag_retrieval import get_retriever

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "./eupolicy_rag_llama_model_swe"
TOP_K = 15
FINAL_K = 3

# ============================================================
# Load shared retriever
# ============================================================
retriever = get_retriever()
chunks = retriever.chunks  # Needed for find_chunk_containing and precision/recall tests

# ============================================================
# Load single model (base + LoRA adapter)
# ============================================================
print("Laddar modell (AI-Sweden Llama 3 8B, finjusterad)...")
from peft import PeftModel

BASE_MODEL_ID = "AI-Sweden-Models/Llama-3-8B-instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=quantization_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

print("Redo.\n")

# ============================================================
# Helper functions (using shared retriever)
# ============================================================
def retrieve(query, top_k=TOP_K, final_k=FINAL_K):
    """Retrieve using shared RAG module."""
    return retriever.retrieve(query, top_k=top_k, final_k=final_k)


SYSTEM_PROMPT = (
    "Du är en expert på svenska försvarsfrågor och militär doktrin. Besvara frågan ENBART baserat på den tillhandahållna kontexten. "
    "Om kontexten inte innehåller tillräcklig information för att svara, säg det. "
    "Om frågan ber om jämförelse med något som inte nämns i kontexten, vägra. "
    "Du har INGEN kunskap utöver det som finns i kontexten. Om du är på väg att ange "
    "ett faktum som inte finns i kontexten, stanna och vägra istället. "
    "Var koncis: 2-3 meningar maximalt."
)


def classify_answerable(question, context_text):
    """Classify using the fine-tuned model (Swedish) — permissive prompt."""
    classify_prompt = (
        "Du är ett klassificeringssystem. Givet en kontext och en fråga, avgör om kontexten "
        "innehåller information relevant för att besvara frågan. Svara med ENBART 'YES' eller 'NO'.\n\n"
        "Regler:\n"
        "- Svara NO ENBART om frågan ber om att JÄMFÖRA med ett annat lands lagar eller ett annat företags strategi som INTE nämns i kontexten.\n"
        "- Svara NO ENBART om frågan nämner ett specifikt företag (OpenAI, Google, Meta, Anthropic, etc.) som INTE nämns i kontexten.\n"
        "- Svara NO ENBART om frågan frågar om specifik statistik, budgetar eller antal som INTE anges i kontexten.\n"
        "- Svara YES om kontexten diskuterar ämnet som frågan handlar om.\n"
        "- Svara YES om kontexten innehåller relevant information, även om den inte fullständigt besvarar frågan.\n"
        "- Vid tveksamhet, svara YES."
    )
    messages = [
        {"role": "system", "content": classify_prompt},
        {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}\n\nKan denna fråga besvaras utifrån kontexten? (YES/NO)"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().upper()
    return "YES" in response or "JA" in response


def generate(question, context_text):
    """Generate answer using the same fine-tuned model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Kontext:\n{context_text}\n\nFråga: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def is_refusal(response):
    refusal_indicators = [
        "innehåller inte", "kan inte hitta", "kan inte besvaras",
        "inte tillräcklig information", "ingen information", "inte nämns",
        "inte nämnt", "inte tillhandahållen", "inte inkludera", "har inte tillräckligt",
        "otillräcklig", "inte tillgänglig", "inte behandlas", "inte specificerad",
        "inte direkt", "kan inte besvara", "inte möjligt att besvara",
        # Also check English refusals (model might respond in English)
        "does not contain", "cannot find", "cannot be answered",
        "not enough information", "no information", "not mentioned",
    ]
    return any(ind in response.lower() for ind in refusal_indicators)


def find_chunk_containing(keywords):
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw.lower() in chunk.lower())
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk


def semantic_similarity(text_a, text_b):
    """Compute cosine similarity between two texts using the embedding model."""
    embeddings = retriever.embed_model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


# ============================================================
# TEST 1: Answerable accuracy (20 questions)
# ============================================================
answerable_tests = [
    {
        "question": "Vad är syftet med manöverkrigföring enligt svensk militär doktrin?",
        "context_keywords": ["manöverkrigföring", "initiativ", "tempo", "kraftsamling"],
        "expected_facts": ["manöver", "initiativ"],
        "reference_answer": "Manöverkrigföring syftar till att genom högt tempo, överraskning och kraftsamling bryta motståndarens vilja och förmåga att fortsätta striden genom att angripa dennes kritiska sårbarheter.",
    },
    {
        "question": "Vilka uppgifter har drönare i det svenska försvaret?",
        "context_keywords": ["drönare", "UAV", "spaning", "övervakning", "obemannad"],
        "expected_facts": ["drönare", "spanin"],
        "reference_answer": "Drönare (UAV) används för spaningsuppgifter, övervakning, målangivning och i vissa fall för bekämpning. De ger förband förbättrad lägesbild utan att utsätta personal för direkt fara.",
    },
    {
        "question": "Vad innebär totalförsvar i svensk kontext?",
        "context_keywords": ["totalförsvar", "militärt försvar", "civilt försvar", "samhälle"],
        "expected_facts": ["totalförsvar", "civilt"],
        "reference_answer": "Totalförsvaret omfattar det militära försvaret och det civila försvaret och innebär hela samhällets samlade förmåga att möta ett väpnat angrepp och hantera kriser i fredstid.",
    },
    {
        "question": "Hur organiseras logistik i en brigad?",
        "context_keywords": ["logistik", "brigad", "underhåll", "försörjning", "ammunition"],
        "expected_facts": ["logistik", "brigad"],
        "reference_answer": "Brigadens logistik organiseras genom underhållsförband som ansvarar för försörjning av ammunition, drivmedel, livsmedel och sjukvård samt reparation och transport av materiel.",
    },
    {
        "question": "Vilken roll har artilleriet i markstrid?",
        "context_keywords": ["artilleri", "eldunderstöd", "indirekt eld", "bekämpning"],
        "expected_facts": ["artilleri", "eld"],
        "reference_answer": "Artilleriet levererar indirekt eldunderstöd för att bekämpa mål på djupet, understödja manöverförband och genomföra moteldstrid mot fiendens artilleri.",
    },
    {
        "question": "Vad innebär CBRN-skydd för svenska förband?",
        "context_keywords": ["CBRN", "kemiska", "biologiska", "radiologiska", "nukleära"],
        "expected_facts": ["CBRN", "skydd"],
        "reference_answer": "CBRN-skydd innebär förmåga att detektera, skydda sig mot och sanera effekter av kemiska, biologiska, radiologiska och nukleära hot för att upprätthålla förbandets stridsförmåga.",
    },
    {
        "question": "Hur genomförs strid i bebyggelse enligt svensk doktrin?",
        "context_keywords": ["bebyggelse", "urban", "strid", "närstrid", "husröjning"],
        "expected_facts": ["bebyggelse", "strid"],
        "reference_answer": "Strid i bebyggelse kännetecknas av korta stridsavstånd, komplex terräng och högt behov av samordning mellan vapenslag, med fokus på husröjning, eldöverfall och snabba förbandsrörelser.",
    },
    {
        "question": "Vilken betydelse har luftvärn för markförband?",
        "context_keywords": ["luftvärn", "lufthot", "skydd", "robot", "flygplan"],
        "expected_facts": ["luftvärn", "skydd"],
        "reference_answer": "Luftvärnet skyddar markförband och kritisk infrastruktur mot lufthot som flygplan, helikoptrar, kryssningsrobotar och drönare genom att bekämpa dessa med robot- och kanonsystem.",
    },
    {
        "question": "Hur bidrar helikoptrar till markoperationer?",
        "context_keywords": ["helikopter", "transport", "understöd", "luftburen", "rörlighet"],
        "expected_facts": ["helikopter", "transport"],
        "reference_answer": "Helikoptrar bidrar till markoperationer genom taktisk transport av förband, understöd med eldkraft, spaning samt medicinsk evakuering, vilket ger ökad rörlighet och flexibilitet.",
    },
    {
        "question": "Vad innebär Sveriges medlemskap i NATO för försvarsplaneringen?",
        "context_keywords": ["NATO", "kollektivt försvar", "artikel 5", "interoperabilitet"],
        "expected_facts": ["NATO", "försvar"],
        "reference_answer": "Sveriges NATO-medlemskap innebär att försvarsplaneringen integreras i alliansens kollektiva försvar enligt artikel 5, med krav på interoperabilitet och förmåga att ta emot och ge militärt stöd.",
    },
    {
        "question": "Vilka principer styr militär försörjning i fält?",
        "context_keywords": ["försörjning", "logistik", "principer", "uthållighet", "fält"],
        "expected_facts": ["försörjning", "uthållighet"],
        "reference_answer": "Militär försörjning i fält styrs av principerna om uthållighet, flexibilitet och framförhållning, där målet är att förband ska kunna verka kontinuerligt utan avbrott i tillförsel.",
    },
    {
        "question": "Hur genomförs övergång av vattendrag?",
        "context_keywords": ["övergång", "vattendrag", "ingenjör", "bro", "fordon"],
        "expected_facts": ["övergång", "vattendrag"],
        "reference_answer": "Övergång av vattendrag genomförs med stöd av ingenjörförband som anlägger broar eller färjor, med krav på eldunderstöd, rökläggning och snabb passage för att minimera sårbarhet.",
    },
    {
        "question": "Vad är syftet med fördröjningsstrid?",
        "context_keywords": ["fördröjning", "tid", "utrymme", "motståndare", "strid"],
        "expected_facts": ["fördröjning", "tid"],
        "reference_answer": "Fördröjningsstrid syftar till att vinna tid genom att sakta ner motståndarens framryckning, slita på dennes förband och skapa förutsättningar för egna motåtgärder utan att själv bli avgörande bunden.",
    },
    {
        "question": "Vilken roll har underrättelsetjänst i operationsplanering?",
        "context_keywords": ["underrättelse", "planering", "hotbild", "beslutsunderlag"],
        "expected_facts": ["underrättelse", "beslut"],
        "reference_answer": "Underrättelsetjänsten förser beslutsfattare med analyserad information om motståndarens förmåga, avsikter och terräng, vilket utgör grunden för operationsplanering och handlingsalternativ.",
    },
    {
        "question": "Hur organiseras det svenska försvarets ledningssystem?",
        "context_keywords": ["ledning", "ledningssystem", "stab", "order", "befäl"],
        "expected_facts": ["ledning", "stab"],
        "reference_answer": "Det svenska försvarets ledningssystem bygger på uppdragstaktik med stabsfunktioner som stöder chefen i beslutsfattande, ordergivning och uppföljning av operationer på alla nivåer.",
    },
    {
        "question": "Vilka krav ställs på militär mobilitet?",
        "context_keywords": ["mobilitet", "rörlighet", "transport", "infrastruktur", "förflyttning"],
        "expected_facts": ["mobilitet", "transport"],
        "reference_answer": "Militär mobilitet kräver tillgång till transportinfrastruktur, snabb förflyttningsförmåga och planering för att förband ska kunna omgruppera och nå operationsområdet i tid.",
    },
    {
        "question": "Vad innebär försvarsupphandling enligt svenska regler?",
        "context_keywords": ["upphandling", "materiel", "försvarsmateriel", "anskaffning"],
        "expected_facts": ["upphandling", "materiel"],
        "reference_answer": "Försvarsupphandling reglerar hur Försvarsmakten anskaffar materiel och tjänster, med krav på säkerhet, sekretess och hänsyn till operativa behov utöver normala upphandlingsregler.",
    },
    {
        "question": "Hur genomförs anfallsstrid på brigadnivå?",
        "context_keywords": ["anfall", "brigad", "genombrott", "kraftsamling", "manöver"],
        "expected_facts": ["anfall", "brigad"],
        "reference_answer": "Anfallsstrid på brigadnivå genomförs genom kraftsamling mot motståndarens svaga punkter, med syfte att uppnå genombrott och exploatera framgång genom snabb manöver i djupet.",
    },
    {
        "question": "Vilken funktion har ingenjörförband i försvaret?",
        "context_keywords": ["ingenjör", "fältarbeten", "minering", "hinder", "röjning"],
        "expected_facts": ["ingenjör", "hinder"],
        "reference_answer": "Ingenjörförband genomför fältarbeten som minering, hinder, broläggning och röjning för att öka egna förbands rörlighet och begränsa motståndarens framryckning.",
    },
    {
        "question": "Vad innebär värdlandsstöd vid NATO-operationer?",
        "context_keywords": ["värdlandsstöd", "NATO", "mottagande", "logistik", "samverkan"],
        "expected_facts": ["värdlandsstöd", "NATO"],
        "reference_answer": "Värdlandsstöd innebär att Sverige som värdnation tillhandahåller logistik, infrastruktur och samordning för allierade styrkor som verkar på eller transiterar genom svenskt territorium.",
    },
]

# ============================================================
# TEST 2: Refusal accuracy (20 questions)
# ============================================================
refusal_tests = [
    {"question": "Hur många stridsfordon har Sverige levererat till Ukraina sedan 2022?", "context_keywords": ["manöver", "brigad"]},
    {"question": "Vad kostar JAS Gripen E per styck i 2024 års priser?", "context_keywords": ["luftvärn", "robot"]},
    {"question": "Vilka förband deltog i den senaste Aurora-övningen?", "context_keywords": ["logistik", "underhåll"]},
    {"question": "Hur många värnpliktiga rycker in per år i Sverige?", "context_keywords": ["totalförsvar", "civilt försvar"]},
    {"question": "Vad är Försvarsmaktens totala budget för 2025?", "context_keywords": ["artilleri", "eldunderstöd"]},
    {"question": "Hur jämför sig svensk artillerieldkraft med Finlands?", "context_keywords": ["artilleri", "indirekt eld"]},
    {"question": "Vilka skillnader finns mellan svensk och norsk brigadstruktur?", "context_keywords": ["brigad", "manöver"]},
    {"question": "Hur skiljer sig svensk CBRN-förmåga från Frankrikes?", "context_keywords": ["CBRN", "skydd"]},
    {"question": "Vilka likheter finns mellan svensk och brittisk marininfanteridoktrin?", "context_keywords": ["amfibie", "landstigningsoperation"]},
    {"question": "Hur jämför sig svenska luftvärnssystem med det amerikanska Patriot-systemet?", "context_keywords": ["luftvärn", "robot"]},
    {"question": "Hur många Archer-artillerisystem är operativa i dag?", "context_keywords": ["artilleri", "bekämpning"]},
    {"question": "Vilken leverantör vann senaste upphandlingen av drönarsystem?", "context_keywords": ["drönare", "UAV"]},
    {"question": "Vad är Saabs strategi för nästa generations luftvärnssystem?", "context_keywords": ["luftvärn", "skydd"]},
    {"question": "Hur har Rysslands invasion av Ukraina påverkat svensk militär rekrytering?", "context_keywords": ["totalförsvar", "beredskap"]},
    {"question": "Vad sa ÖB om Gotlands försvarsförmåga 2024?", "context_keywords": ["försvar", "territoriell"]},
    {"question": "Hur samverkar svensk underrättelsetjänst med CIA i praktiken?", "context_keywords": ["underrättelse", "hotbild"]},
    {"question": "Vad är den genomsnittliga leveranstiden för beställd försvarsmateriel?", "context_keywords": ["upphandling", "materiel"]},
    {"question": "Hur många helikoptrar har havererat under övningar senaste tio åren?", "context_keywords": ["helikopter", "transport"]},
    {"question": "Vad är BAE Systems ståndpunkt om svensk stridsfordonsutveckling?", "context_keywords": ["fordon", "pansarvärn"]},
    {"question": "Hur jämför sig Sveriges försvarsutgifter med Polens som andel av BNP?", "context_keywords": ["försvar", "planering"]},
]

# ============================================================
# TEST 3: False refusal rate (20 questions, full pipeline)
# ============================================================
false_refusal_tests = [
    "Vad säger doktrinen om manöverkrigföring?",
    "Finns det regler om drönare i svenska försvaret?",
    "Vad innebär begreppet totalförsvar?",
    "Vilka typer av logistikstöd beskrivs för brigaden?",
    "Vad säger dokumenten om artilleriets roll i eldunderstöd?",
    "Hur beskrivs CBRN-hotets påverkan på stridsmiljön?",
    "Vilka principer gäller för strid i bebyggelse?",
    "Vad säger dokumenten om luftvärnets organisation?",
    "Hur beskrivs helikopterns roll i taktiska operationer?",
    "Vilka krav ställs för NATO-interoperabilitet?",
    "Vad säger dokumenten om fördröjningsstrid?",
    "Hur beskrivs underrättelseinhämtning?",
    "Vilka krav finns på militär mobilitet?",
    "Vad säger dokumenten om ingenjörförbands uppgifter?",
    "Hur beskrivs värdlandsstöd?",
    "Vilka principer styr anfallsstrid?",
    "Vad säger dokumenten om försörjning av ammunition?",
    "Hur beskrivs övergång av vattendrag?",
    "Vilka uppgifter har mekaniserade förband?",
    "Vad säger dokumenten om ledning och samordning?",
]

# ============================================================
# TEST 4: Parametric leakage (10 questions)
# ============================================================
parametric_leakage_tests = [
    {"question": "Vad är räckvidden på det ryska S-400 luftvärnssystemet?", "context_keywords": ["luftvärn", "robot"]},
    {"question": "Hur många soldater har Ryssland i Kaliningrad?", "context_keywords": ["totalförsvar", "hotbild"]},
    {"question": "Vad kostar ett Javelin-robotsystem?", "context_keywords": ["pansarvärn", "robot"]},
    {"question": "Vilken toppfart har CV90 stridsfordon?", "context_keywords": ["mekaniserad", "fordon"]},
    {"question": "Hur många kärnvapen har Ryssland i sitt strategiska arsenallager?",  "context_keywords": ["CBRN", "nukleär"]},
    {"question": "Vad är den maximala eldgivningshastigheten för Archer-systemet?", "context_keywords": ["artilleri", "bekämpning"]},
    {"question": "Hur långt kan Tomahawk-kryssningsrobotar flyga?", "context_keywords": ["luftvärn", "skydd"]},
    {"question": "Vad är Leopard 2-stridsvagnens frontpansartjocklek?", "context_keywords": ["mekaniserad", "pansarvärn"]},
    {"question": "Hur stor är det ukrainska flygvapnets förlusttal sedan 2022?", "context_keywords": ["flygstridskrafter", "bekämpning"]},
    {"question": "Vad är Turkiets Bayraktar TB2-drönares maximala flygtid?", "context_keywords": ["drönare", "UAV"]},
]

# ============================================================
# TEST 5: Retrieval quality (20 questions)
# ============================================================
retrieval_tests = [
    {"question": "Vad innebär manöverkrigföring?", "must_contain": ["manöver"]},
    {"question": "Vilka uppgifter har drönare?", "must_contain": ["drönare"]},
    {"question": "Vad är totalförsvar?", "must_contain": ["totalförsvar"]},
    {"question": "Hur organiseras logistik i brigaden?", "must_contain": ["logistik"]},
    {"question": "Vilken roll har artilleriet?", "must_contain": ["artilleri"]},
    {"question": "Vad innebär CBRN-skydd?", "must_contain": ["CBRN"]},
    {"question": "Hur genomförs strid i bebyggelse?", "must_contain": ["bebyggelse"]},
    {"question": "Vilken funktion har luftvärnet?", "must_contain": ["luftvärn"]},
    {"question": "Hur används helikoptrar i markoperationer?", "must_contain": ["helikopter"]},
    {"question": "Vad innebär NATO-medlemskap för Sverige?", "must_contain": ["NATO"]},
    {"question": "Vilka principer styr fördröjningsstrid?", "must_contain": ["fördröjning"]},
    {"question": "Hur genomförs underrättelseinhämtning?", "must_contain": ["underrättelse"]},
    {"question": "Vilka krav ställs på militär mobilitet?", "must_contain": ["mobilitet"]},
    {"question": "Vad gör ingenjörförband?", "must_contain": ["ingenjör"]},
    {"question": "Hur genomförs anfallsstrid?", "must_contain": ["anfall"]},
    {"question": "Vad innebär värdlandsstöd?", "must_contain": ["värdlandsstöd"]},
    {"question": "Hur sker försörjning av ammunition?", "must_contain": ["ammunition"]},
    {"question": "Vilka krav finns på ledningssystem?", "must_contain": ["ledning"]},
    {"question": "Hur genomförs övergång av vattendrag?", "must_contain": ["vattendrag"]},
    {"question": "Vad innebär försvarsupphandling?", "must_contain": ["upphandling"]},
]

# Ground truth for precision/recall
retrieval_ground_truth = [
    {"question": "Vad innebär manöverkrigföring?", "relevant_keywords": ["manöver", "initiativ"]},
    {"question": "Vilka uppgifter har drönare?", "relevant_keywords": ["drönare", "spaning"]},
    {"question": "Vad är totalförsvar?", "relevant_keywords": ["totalförsvar", "civilt försvar"]},
    {"question": "Hur organiseras logistik i brigaden?", "relevant_keywords": ["logistik", "underhåll"]},
    {"question": "Vilken roll har artilleriet?", "relevant_keywords": ["artilleri", "indirekt eld"]},
    {"question": "Vad innebär CBRN-skydd?", "relevant_keywords": ["CBRN", "kemiska"]},
    {"question": "Hur genomförs strid i bebyggelse?", "relevant_keywords": ["bebyggelse", "närstrid"]},
    {"question": "Vilken funktion har luftvärnet?", "relevant_keywords": ["luftvärn", "lufthot"]},
    {"question": "Hur används helikoptrar i markoperationer?", "relevant_keywords": ["helikopter", "transport"]},
    {"question": "Vad innebär NATO-medlemskap för Sverige?", "relevant_keywords": ["NATO", "interoperabilitet"]},
    {"question": "Vilka principer styr fördröjningsstrid?", "relevant_keywords": ["fördröjning", "tid"]},
    {"question": "Hur genomförs underrättelseinhämtning?", "relevant_keywords": ["underrättelse", "inhämtning"]},
    {"question": "Vilka krav ställs på militär mobilitet?", "relevant_keywords": ["mobilitet", "förflyttning"]},
    {"question": "Vad gör ingenjörförband?", "relevant_keywords": ["ingenjör", "fältarbeten"]},
    {"question": "Hur genomförs anfallsstrid?", "relevant_keywords": ["anfall", "kraftsamling"]},
    {"question": "Vad innebär värdlandsstöd?", "relevant_keywords": ["värdlandsstöd", "allierade"]},
    {"question": "Hur sker försörjning av ammunition?", "relevant_keywords": ["ammunition", "försörjning"]},
    {"question": "Vilka krav finns på ledningssystem?", "relevant_keywords": ["ledning", "stab"]},
    {"question": "Hur genomförs övergång av vattendrag?", "relevant_keywords": ["vattendrag", "bro"]},
    {"question": "Vad innebär försvarsupphandling?", "relevant_keywords": ["upphandling", "materiel"]},
]


# ============================================================
# Run evaluation
# ============================================================
def run_eval():
    results = {
        "answerable": {"total": 0, "correct": 0, "details": []},
        "refusal": {"total": 0, "correct": 0, "details": []},
        "false_refusal": {"total": 0, "refused": 0, "details": []},
        "parametric_leakage": {"total": 0, "leaked": 0, "details": []},
        "retrieval": {"total": 0, "hit": 0, "details": []},
    }

    # =====================================================
    # All tasks use the same model — no swap needed
    # =====================================================
    print("=" * 60)
    print("UTVÄRDERING (AI-Sweden Llama 3 8B, finjusterad — en modell)")
    print("=" * 60)

    tasks = []

    # Test 1: Answerable
    for test in answerable_tests:
        context = find_chunk_containing(test["context_keywords"])
        context_text = f"[Källa 1, relevans: 0.75]: {context}"
        tasks.append({"type": "answerable", "question": test["question"], "context_text": context_text, "test": test})

    # Test 2: Refusal
    for test in refusal_tests:
        context = find_chunk_containing(test["context_keywords"])
        context_text = f"[Källa 1, relevans: 0.50]: {context}"
        tasks.append({"type": "refusal", "question": test["question"], "context_text": context_text})

    # Test 3: False refusal (full pipeline)
    for question in false_refusal_tests:
        retrieved = retrieve(question)
        context_text = "\n\n".join([f"[Källa {i+1}, relevans: {c['score']:.2f}]: {c['text']}" for i, c in enumerate(retrieved)])
        tasks.append({"type": "false_refusal", "question": question, "context_text": context_text, "retrieved": retrieved})

    # Test 4: Parametric leakage
    for test in parametric_leakage_tests:
        context = find_chunk_containing(test["context_keywords"])
        context_text = f"[Källa 1, relevans: 0.50]: {context}"
        tasks.append({"type": "leakage", "question": test["question"], "context_text": context_text})

    # Classify all
    print(f"\n  Klassificerar {len(tasks)} frågor...")
    for task in tasks:
        task["classified_yes"] = classify_answerable(task["question"], task["context_text"])
        label = "YES" if task["classified_yes"] else "NO"
        print(f"    [{label}] ({task['type']}) {task['question'][:60]}...")

    # Generate answers for YES-classified tasks
    print("\n--- Test 1: Besvaringsbarhet (20 frågor) ---")
    semantic_scores = []
    for task in [t for t in tasks if t["type"] == "answerable"]:
        if not task["classified_yes"]:
            response = "Den tillhandahållna kontexten innehåller inte tillräcklig information för att besvara denna fråga."
        else:
            response = generate(task["question"], task["context_text"])
        response_lower = response.lower()
        test = task["test"]
        facts_found = [f for f in test["expected_facts"] if f.lower() in response_lower]
        keyword_pass = len(facts_found) >= len(test["expected_facts"]) // 2 + 1 and not is_refusal(response)

        if "reference_answer" in test and not is_refusal(response):
            sim_score = semantic_similarity(response, test["reference_answer"])
        else:
            sim_score = 0.0
        semantic_scores.append(sim_score)

        correct_semantic = sim_score >= 0.7 and not is_refusal(response)
        is_correct = keyword_pass or correct_semantic

        results["answerable"]["total"] += 1
        if is_correct:
            results["answerable"]["correct"] += 1
        status = "PASS" if is_correct else "FAIL"
        print(f"  [{status}] F: {task['question']}")
        print(f"        S: {response[:100]}...")
        print(f"        Nyckelord: {len(facts_found)}/{len(test['expected_facts'])} | Semantisk: {sim_score:.2f}")
        if not is_correct and is_refusal(response):
            print(f"        ** FALSK VÄGRAN **")
        results["answerable"]["details"].append({"question": task["question"], "correct": is_correct, "semantic_score": sim_score})

    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0
    print(f"\n  Genomsnittlig semantisk likhet: {avg_semantic:.2f}")

    print("\n--- Test 2: Vägrannoggrannhet (20 frågor) ---")
    for task in [t for t in tasks if t["type"] == "refusal"]:
        if not task["classified_yes"]:
            response = "Den tillhandahållna kontexten innehåller inte tillräcklig information för att besvara denna fråga."
        else:
            response = generate(task["question"], task["context_text"])
        refused = is_refusal(response)
        results["refusal"]["total"] += 1
        if refused:
            results["refusal"]["correct"] += 1
        status = "PASS" if refused else "FAIL"
        print(f"  [{status}] F: {task['question']}")
        print(f"        S: {response[:100]}...")
        if not refused:
            print(f"        ** HALLUCINATION **")
        results["refusal"]["details"].append({"question": task["question"], "refused": refused})

    print("\n--- Test 3: Falsk vägranfrekvens (20 frågor) ---")
    for task in [t for t in tasks if t["type"] == "false_refusal"]:
        if not task["classified_yes"]:
            response = "Den tillhandahållna kontexten innehåller inte tillräcklig information för att besvara denna fråga."
        else:
            response = generate(task["question"], task["context_text"])
        refused = is_refusal(response)
        results["false_refusal"]["total"] += 1
        if refused:
            results["false_refusal"]["refused"] += 1
        status = "FAIL" if refused else "PASS"
        print(f"  [{status}] F: {task['question']}")
        print(f"        S: {response[:100]}...")
        if refused:
            print(f"        ** FALSK VÄGRAN **")
        results["false_refusal"]["details"].append({"question": task["question"], "refused": refused})

    print("\n--- Test 4: Parametriskt läckage (10 frågor) ---")
    for task in [t for t in tasks if t["type"] == "leakage"]:
        if not task["classified_yes"]:
            response = "Den tillhandahållna kontexten innehåller inte tillräcklig information för att besvara denna fråga."
        else:
            response = generate(task["question"], task["context_text"])
        refused = is_refusal(response)
        leaked = not refused
        results["parametric_leakage"]["total"] += 1
        if leaked:
            results["parametric_leakage"]["leaked"] += 1
        status = "PASS" if refused else "FAIL"
        print(f"  [{status}] F: {task['question']}")
        print(f"        S: {response[:100]}...")
        if leaked:
            print(f"        ** LÄCKAGE **")
        results["parametric_leakage"]["details"].append({"question": task["question"], "leaked": leaked})

    # --- Test 5: Retrieval (no model needed) ---
    print("\n--- Test 5: Hämtningskvalitet (20 frågor) ---")
    for test in retrieval_tests:
        retrieved = retrieve(test["question"])
        combined_text = " ".join([c["text"] for c in retrieved]).lower()
        hit = all(kw.lower() in combined_text for kw in test["must_contain"])
        results["retrieval"]["total"] += 1
        if hit:
            results["retrieval"]["hit"] += 1
        status = "PASS" if hit else "FAIL"
        scores_str = ", ".join(f"{c['score']:.3f}" for c in retrieved)
        print(f"  [{status}] F: {test['question']}  Poäng: [{scores_str}]")
        results["retrieval"]["details"].append({"question": test["question"], "hit": hit})

    # --- Test 6: Precision/Recall ---
    print("\n--- Test 6: Precision@k och Recall@k (20 frågor) ---")
    results["precision"] = {"total": 0, "sum_precision": 0.0, "details": []}
    results["recall"] = {"total": 0, "sum_recall": 0.0, "details": []}
    for test in retrieval_ground_truth:
        relevant_indices = set()
        for i, chunk in enumerate(chunks):
            if all(kw.lower() in chunk.lower() for kw in test["relevant_keywords"]):
                relevant_indices.add(i)
        retrieved = retrieve(test["question"])
        retrieved_indices = set(c["idx"] for c in retrieved)
        relevant_retrieved = retrieved_indices & relevant_indices
        precision = len(relevant_retrieved) / len(retrieved_indices) if retrieved_indices else 0.0
        recall = len(relevant_retrieved) / len(relevant_indices) if relevant_indices else 1.0
        results["precision"]["total"] += 1
        results["precision"]["sum_precision"] += precision
        results["recall"]["total"] += 1
        results["recall"]["sum_recall"] += recall
        print(f"  P@{FINAL_K}={precision:.2f} R@{FINAL_K}={recall:.2f} | {test['question']}")
        results["precision"]["details"].append({"question": test["question"], "precision": precision})
        results["recall"]["details"].append({"question": test["question"], "recall": recall})

    # --- Summary ---
    print("\n" + "=" * 60)
    print("EXTENDED EVAL SUMMARY")
    print("Model: AI-Sweden Llama 3 8B (fine-tuned, single model)")
    print("=" * 60)
    print(f"\n  Answerable accuracy:    {results['answerable']['correct']}/{results['answerable']['total']} "
          f"({100*results['answerable']['correct']/results['answerable']['total']:.0f}%)")
    print(f"  Refusal accuracy:       {results['refusal']['correct']}/{results['refusal']['total']} "
          f"({100*results['refusal']['correct']/results['refusal']['total']:.0f}%)")
    print(f"  False refusal rate:     {results['false_refusal']['refused']}/{results['false_refusal']['total']} "
          f"({100*results['false_refusal']['refused']/results['false_refusal']['total']:.0f}%) — lower is better")
    print(f"  Parametric leakage:     {results['parametric_leakage']['leaked']}/{results['parametric_leakage']['total']} "
          f"({100*results['parametric_leakage']['leaked']/results['parametric_leakage']['total']:.0f}%) — lower is better")
    print(f"  Retrieval hit rate:     {results['retrieval']['hit']}/{results['retrieval']['total']} "
          f"({100*results['retrieval']['hit']/results['retrieval']['total']:.0f}%)")

    avg_precision = results["precision"]["sum_precision"] / results["precision"]["total"] if results["precision"]["total"] > 0 else 0
    avg_recall = results["recall"]["sum_recall"] / results["recall"]["total"] if results["recall"]["total"] > 0 else 0
    print(f"  Avg Precision@{FINAL_K}:        {avg_precision:.2f}")
    print(f"  Avg Recall@{FINAL_K}:           {avg_recall:.2f}")

    with open("eval_extended_results_swe.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\n  Detaljerade resultat sparade till eval_extended_results_swe.json")


if __name__ == "__main__":
    run_eval()
