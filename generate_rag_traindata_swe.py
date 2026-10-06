"""
Generate RAG-aware training data for Swedish defence documents.
Creates two types of examples:
1. ANSWERABLE: Context contains the answer -> model extracts it
2. UNANSWERABLE: Context does NOT contain the answer -> model refuses

Tailored for Swedish military/defence document corpus.
"""
import json
import random

random.seed(42)

# Load the document chunks
with open("rag_index_swe/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

# ============================================================
# ANSWERABLE examples: question + relevant context -> answer
# ============================================================
answerable = [
    {
        "context_keywords": ["drönare", "obemannad", "UAS"],
        "questions": [
            "Vilken roll har drönare i moderna markoperationer?",
            "Hur används obemannade system i markstriden?",
            "Vad säger doktrinen om drönares användning?",
        ],
        "answer": "Drönare och obemannade system har en central roll i moderna markoperationer för spaning, övervakning och målbekämpning. De ger förbandschefer ökad situationsmedvetenhet och möjlighet att verka på djupet utan att utsätta personal för direkt fara."
    },
    {
        "context_keywords": ["logistik", "underhåll", "försörjning"],
        "questions": [
            "Hur organiseras logistik i Försvarsmakten?",
            "Vilka principer gäller för militär logistik?",
            "Vad innebär logistik vid insats?",
        ],
        "answer": "Militär logistik omfattar planering och genomförande av underhåll, transport, försörjning och sjukvård som stödjer stridskrafterna. Den organiseras för att säkerställa att förband har rätt resurser på rätt plats vid rätt tidpunkt."
    },
    {
        "context_keywords": ["totalförsvar", "civil", "militär", "samhälle"],
        "questions": [
            "Vad är totalförsvaret?",
            "Hur samverkar det civila och militära försvaret?",
            "Vilka komponenter ingår i totalförsvaret?",
        ],
        "answer": "Totalförsvaret omfattar det militära försvaret och det civila försvaret tillsammans. Det syftar till att samhällets samlade resurser kan användas för att möta hot och angrepp mot Sverige, inklusive samverkan mellan civila och militära aktörer."
    },
    {
        "context_keywords": ["gemensamma operationer", "ledning", "samordning"],
        "questions": [
            "Vad innebär gemensamma operationer?",
            "Hur leds gemensamma operationer i Sverige?",
            "Vilka principer styr gemensamma operationer?",
        ],
        "answer": "Gemensamma operationer innebär samordnade militära insatser med deltagande från flera försvarsgrenar under gemensam ledning. De syftar till att uppnå synergier genom att kombinera mark-, sjö- och flygstridskrafter samt specialförband."
    },
    {
        "context_keywords": ["NATO", "allians", "kollektiv", "försvar"],
        "questions": [
            "Hur påverkar NATO-medlemskapet Sveriges försvar?",
            "Vilken roll har Sverige i NATO?",
            "Hur integreras Sverige i NATO:s försvarsplanering?",
        ],
        "answer": "Sveriges NATO-medlemskap innebär integration i alliansens kollektiva försvarsplanering och kommandostruktur. Sverige bidrar med militär förmåga till det gemensamma försvaret och får i gengäld skydd genom artikel 5-garantin."
    },
    {
        "context_keywords": ["hög risk", "hot", "Ryssland", "säkerhetspolitisk"],
        "questions": [
            "Hur beskrivs det säkerhetspolitiska läget?",
            "Vilka hot identifieras i den säkerhetspolitiska bedömningen?",
            "Hur påverkar omvärldsläget svensk försvarsplanering?",
        ],
        "answer": "Det säkerhetspolitiska läget präglas av ökade spänningar och ett försämrat omvärldsläge. Rysslands aggression mot Ukraina har fundamentalt förändrat den europeiska säkerhetsordningen och ställer krav på ökad svensk försvarsförmåga."
    },
    {
        "context_keywords": ["CBRN", "kemisk", "biologisk", "radiologisk", "nukleär"],
        "questions": [
            "Vad innebär CBRN-skydd?",
            "Hur hanteras CBRN-hot i Försvarsmakten?",
            "Vilka principer gäller för skydd mot CBRN?",
        ],
        "answer": "CBRN-skydd avser åtgärder mot kemiska, biologiska, radiologiska och nukleära hot. Det omfattar detektion, varning, fysiskt skydd, dekontaminering och medicinsk behandling av personal som exponerats för CBRN-ämnen."
    },
    {
        "context_keywords": ["artilleri", "eldgivning", "indirekt", "eld"],
        "questions": [
            "Vilken roll har artilleriet i markstriden?",
            "Hur används indirekt eld i operationer?",
            "Vad är artilleriets uppgifter?",
        ],
        "answer": "Artilleriet levererar indirekt eld för att stödja markstridskrafterna genom att bekämpa mål på djupet, skydda egna förband och störa fiendens operationer. Det är en central komponent för markoperativ eldkraft."
    },
    {
        "context_keywords": ["luftvärn", "lufthot", "skydd", "luftrum"],
        "questions": [
            "Vilken roll har luftvärnet?",
            "Hur skyddas förband mot lufthot?",
            "Vad är luftvärnets uppgifter?",
        ],
        "answer": "Luftvärnet skyddar förband, objekt och områden mot luftburna hot inklusive flygplan, helikoptrar, kryssningsmissiler och obemannade farkoster. Det utgör en kritisk del av det samlade luftförsvaret."
    },
    {
        "context_keywords": ["ingenjör", "röjning", "hinder", "fältarbeten"],
        "questions": [
            "Vilken roll har ingenjörförbanden?",
            "Vad innebär fältarbeten i markstriden?",
            "Hur stödjer ingenjörer markoperationer?",
        ],
        "answer": "Ingenjörförband genomför fältarbeten som hindring, röjning, förstärkning och infrastrukturarbeten. De möjliggör egna förbands rörelse, försvårar fiendens framryckning och skapar skydd för stridskrafterna."
    },
    {
        "context_keywords": ["stridsekonomi", "resurser", "prioritering", "kraft"],
        "questions": [
            "Vad innebär stridsekonomi?",
            "Hur tillämpas stridsekonomi i operationer?",
            "Vilka principer styr resursanvändning i strid?",
        ],
        "answer": "Stridsekonomi innebär att uppnå maximalt resultat med tillgängliga resurser genom medveten prioritering och kraftsamling. Det handlar om att avgöra var, när och hur stridskrafter ska sättas in för störst effekt."
    },
    {
        "context_keywords": ["rörlighet", "transport", "förflyttning", "manöver"],
        "questions": [
            "Hur säkerställs militär rörlighet?",
            "Vad innebär rörlighet i försvarssammanhang?",
            "Vilka aspekter ingår i militär rörlighet?",
        ],
        "answer": "Militär rörlighet avser förmågan att förflytta förband och materiel snabbt och effektivt. Det omfattar infrastruktur, transporter, värdlandsstöd och administrativa förutsättningar för att möjliggöra operativ manöver."
    },
    {
        "context_keywords": ["försörjning", "beredskap", "materiel", "budget"],
        "questions": [
            "Hur planeras försvarets materielförsörjning?",
            "Vilka utmaningar finns i försvarets materielplanering?",
            "Hur prioriteras anskaffning av försvarsmateriel?",
        ],
        "answer": "Försvarets materielförsörjning planeras genom strategiska prioriteringar baserade på hotbild, operativa behov och tillgängliga ekonomiska ramar. Anskaffning styrs av identifierade förmågebehov och genomförs i samverkan med nationell och internationell försvarsindustri."
    },
    {
        "context_keywords": ["urban", "strid", "bebyggelse", "stad"],
        "questions": [
            "Hur genomförs strid i bebyggelse?",
            "Vilka utmaningar finns vid strid i urban miljö?",
            "Vad säger doktrinen om strid i bebyggelse?",
        ],
        "answer": "Strid i bebyggelse ställer särskilda krav på förband avseende närstridsförmåga, samordning och skydd av civila. Urban terräng begränsar överblick och rörlighet men ger också möjligheter till skydd och dold gruppering."
    },
    {
        "context_keywords": ["helikopter", "luftburen", "insats", "transport"],
        "questions": [
            "Hur används helikoptrar i markoperationer?",
            "Vilken roll har helikoptrar i moderna operationer?",
            "Vad säger doktrinen om helikopterinsatser?",
        ],
        "answer": "Helikoptrar används i markoperationer för transport, spaning, eldunderstöd och räddning. De ger taktisk rörlighet och möjlighet att snabbt omgruppera förband eller genomföra insatser i svårtillgänglig terräng."
    },
]

# ============================================================
# UNANSWERABLE examples: question + irrelevant context -> refusal
# ============================================================
unanswerable_questions = [
    "Hur mycket kostar en JAS 39 Gripen?",
    "Hur många soldater har Försvarsmakten totalt?",
    "Vad är försvarsbudgetens exakta storlek 2026?",
    "Vilka leverantörer har kontrakt med FMV just nu?",
    "Hur många drönare har Försvarsmakten beställt?",
    "Vad är personalomsättningen i Försvarsmakten?",
    "Hur många värnpliktiga kallas in varje år?",
    "Vilken lön har en brigadgeneral?",
    "Hur många ubåtar har den svenska flottan?",
    "Vad kostar det nya artillerisystemet Archer?",
    "Hur snabbt kan Sverige mobilisera alla reserver?",
    "Vilka specifika förband är placerade i Norrbotten?",
    "Hur lång är grundutbildningen för värnpliktiga?",
    "Vad är exakt personalstyrka i varje regemente?",
    "Hur många helikoptrar av typ Black Hawk har Sverige?",
    # Adversarial: topics with parametric knowledge
    "Hur jämför sig den svenska armén med den finska?",
    "Vad sa ÖB om Ukraina-kriget?",
    "Hur presterade Leopard 2 i Ukraina?",
    "Vad tycker NATO om Sveriges försvarsbudget?",
    "Hur fungerar det ryska S-400 systemet?",
    "Vad är skillnaden mellan F-35 och JAS 39?",
    "Hur påverkar Kinas militära upprustning Europa?",
    "Vilka länder har köpt svensk försvarsmateriel senaste året?",
    "Hur fungerar det israeliska Iron Dome-systemet?",
    "Vad kostar USA:s försvarsbudget?",
    "Hur effektivt var det ukrainska luftvärnet?",
    "Vilka är de ryska styrkorna vid Finlands gräns?",
    "Hur jämför sig svensk försvarsförmåga med Norges?",
    "Vad sa Volodymyr Zelenskyj om svenska vapen?",
    "Hur många HIMARS har skickats till Ukraina?",
]

# Comparison questions: context IS relevant but question asks about external comparison
comparison_unanswerable = [
    "Hur jämför sig svensk drönardoktrin med den ukrainska?",
    "Vilka skillnader finns mellan svensk och finsk totalförsvarsmodell?",
    "Hur skiljer sig NATO:s doktrin från svensk doktrin för gemensamma operationer?",
    "Hur jämför sig Sveriges försvarsbudget med Polens?",
    "Vilka skillnader finns mellan svensk och norsk logistikdoktrin?",
    "Hur jämför sig svenska artillerisystem med tyska?",
    "Hur skiljer sig den svenska planerings- och ledningsmetoden från NATO:s?",
    "Vilka likheter finns mellan svensk och brittisk försvarsstrategi?",
    "Hur jämför sig svenskt luftvärn med det finska?",
    "Hur skiljer sig svensk strid i bebyggelse-doktrin från USA:s?",
    "Vilka skillnader finns mellan svenska och danska specialförband?",
    "Hur jämför sig svensk markstridsfömåga med Frankrikes?",
    "Hur skiljer sig Sveriges CBRN-skydd från Tysklands?",
    "Vilka skillnader finns mellan svensk och estnisk värnpliktsmodell?",
    "Hur jämför sig svenska försvarsanslagen med NATO:s tvåprocentsmål i praktiken?",
]

refusal_templates = [
    "Den tillhandahållna kontexten innehåller inte specifik information för att besvara denna fråga.",
    "Baserat på den tillgängliga kontexten kan jag inte hitta information som besvarar denna fråga.",
    "Denna fråga kan inte besvaras utifrån den tillhandahållna kontexten eftersom den inte innehåller relevanta detaljer.",
    "Kontexten innehåller inte information om detta ämne.",
    "Jag har inte tillräckligt med information i den tillhandahållna kontexten för att besvara detta korrekt.",
]

# ============================================================
# Find relevant chunks
# ============================================================
def find_relevant_chunk(keywords):
    """Find the chunk that best matches the keywords."""
    best_chunk = None
    best_score = 0
    for chunk in chunks:
        score = sum(1 for kw in keywords if kw.lower() in chunk.lower())
        if score > best_score:
            best_score = score
            best_chunk = chunk
    return best_chunk


def find_irrelevant_chunk(exclude_keywords):
    """Find a chunk that does NOT match the keywords."""
    candidates = []
    for chunk in chunks:
        score = sum(1 for kw in exclude_keywords if kw.lower() in chunk.lower())
        if score == 0:
            candidates.append(chunk)
    if candidates:
        return random.choice(candidates)
    return random.choice(chunks)


# ============================================================
# Generate training examples
# ============================================================
training_examples = []

SYSTEM_PROMPT = (
    "Du är en expert på svenska försvarsfrågor och militär doktrin. "
    "Besvara frågan ENBART baserat på den tillhandahållna kontexten. "
    "Om kontexten inte innehåller tillräcklig information för att svara, säg det. "
    "Om frågan ber om jämförelse med något som inte nämns i kontexten, vägra. "
    "Du har INGEN kunskap utöver det som finns i kontexten. Om du är på väg att ange "
    "ett faktum som inte finns i kontexten, stanna och vägra istället. "
    "Var koncis: 2-3 meningar maximalt."
)


def build_multi_chunk_context(primary_chunk, n_extra=2):
    """Build a multi-chunk context matching inference format."""
    sources = []
    sources.append(f"[Källa 1, relevans: {random.uniform(0.70, 0.95):.2f}]: {primary_chunk}")
    used = {primary_chunk}
    for i in range(n_extra):
        extra = random.choice(chunks)
        while extra in used:
            extra = random.choice(chunks)
        used.add(extra)
        score = random.uniform(0.30, 0.60)
        sources.append(f"[Källa {i+2}, relevans: {score:.2f}]: {extra}")
    return "\n\n".join(sources)


def build_irrelevant_multi_chunk_context(n_chunks=3):
    """Build multi-chunk context where none are relevant."""
    sources = []
    used = set()
    for i in range(n_chunks):
        chunk = random.choice(chunks)
        while chunk in used:
            chunk = random.choice(chunks)
        used.add(chunk)
        score = random.uniform(0.30, 0.55)
        sources.append(f"[Källa {i+1}, relevans: {score:.2f}]: {chunk}")
    return "\n\n".join(sources)


# Answerable examples
for item in answerable:
    context = find_relevant_chunk(item["context_keywords"])
    if not context:
        continue
    for question in item["questions"]:
        multi_context = build_multi_chunk_context(context)
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Kontext:\n{multi_context}\n\nFråga: {question}"},
                {"role": "assistant", "content": item["answer"]},
            ]
        })

# Unanswerable examples
for question in unanswerable_questions:
    for _ in range(3):
        multi_context = build_irrelevant_multi_chunk_context()
        refusal = random.choice(refusal_templates)
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Kontext:\n{multi_context}\n\nFråga: {question}"},
                {"role": "assistant", "content": refusal},
            ]
        })

# Comparison unanswerable
for question in comparison_unanswerable:
    for _ in range(3):
        primary = random.choice(chunks)
        multi_context = build_multi_chunk_context(primary)
        refusal = random.choice(refusal_templates)
        training_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Kontext:\n{multi_context}\n\nFråga: {question}"},
                {"role": "assistant", "content": refusal},
            ]
        })

# Balanced repeats: answerable 4x, unanswerable 2x
random.shuffle(training_examples)
all_examples = []

answerable_examples = [ex for ex in training_examples if not any(
    ex["messages"][2]["content"].startswith(t) for t in [
        "Den tillhandahållna", "Baserat på", "Denna fråga", "Kontexten", "Jag har inte"
    ]
)]
unanswerable_examples = [ex for ex in training_examples if ex not in answerable_examples]

for _ in range(4):
    shuffled = list(answerable_examples)
    random.shuffle(shuffled)
    all_examples.extend(shuffled)
for _ in range(2):
    shuffled = list(unanswerable_examples)
    random.shuffle(shuffled)
    all_examples.extend(shuffled)

random.shuffle(all_examples)

# Save
with open("rag_traindata_swe.json", "w", encoding="utf-8") as f:
    json.dump(all_examples, f, indent=2, ensure_ascii=False)

n_answerable = len(answerable_examples)
n_unanswerable = len(unanswerable_examples)

print(f"Generated {len(all_examples)} total training examples")
print(f"  Answerable: {n_answerable} unique ({n_answerable*4} with repeats)")
print(f"  Unanswerable: {n_unanswerable} unique ({n_unanswerable*2} with repeats)")
print(f"Saved to rag_traindata_swe.json")
