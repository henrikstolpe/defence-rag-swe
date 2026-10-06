"""
Simulation Generator — Uses Claude to produce a realistic timestep-by-timestep
tactical simulation from a scenario description.

Retrieves relevant Swedish doctrine from the RAG index and includes it
in the simulation prompt so Claude follows actual Swedish military doctrine.

Returns structured frame data that the map renderer can animate.
"""
import os
import json
import math
import re
import numpy as np
from collections import Counter
import anthropic

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# ============================================================
# RAG retrieval for doctrine (lightweight — uses pre-built index)
# ============================================================
INDEX_DIR = "./rag_index_swe"
EMBED_MODEL = "BAAI/bge-m3"

_rag_cache = {}

def get_doctrine_chunks(scenario_text: str, n_chunks: int = 10) -> list[str]:
    """
    Two-step doctrine retrieval:
    1. Ask Claude to identify what doctrine topics are needed for this specific scenario
    2. Retrieve chunks matching those topics from the RAG index
    """
    global _rag_cache

    if "embed_model" not in _rag_cache:
        import faiss
        from sentence_transformers import SentenceTransformer, CrossEncoder

        _rag_cache["index"] = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
        with open(os.path.join(INDEX_DIR, "chunks.json"), encoding="utf-8") as f:
            _rag_cache["chunks"] = json.load(f)
        with open(os.path.join(INDEX_DIR, "sources.json"), encoding="utf-8") as f:
            _rag_cache["sources"] = json.load(f)
        _rag_cache["embed_model"] = SentenceTransformer(EMBED_MODEL, device="cpu")
        _rag_cache["reranker"] = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device="cpu")

    index = _rag_cache["index"]
    chunks = _rag_cache["chunks"]
    sources = _rag_cache["sources"]
    embed_model = _rag_cache["embed_model"]
    reranker = _rag_cache["reranker"]

    # STEP 1: Ask Claude what doctrine is needed for this scenario
    print("[SimGenerator] Step 1: Identifying required doctrine topics...")
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    topic_message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=0.0,  # Deterministic doctrine-topic selection
        system="You identify what Swedish military doctrine topics are needed to simulate a given scenario. Output ONLY a JSON array of 5-8 Swedish search queries that would find the most relevant doctrine for this specific tactical situation. Each query should be 3-6 words in Swedish military terminology.",
        messages=[
            {"role": "user", "content": f"What doctrine is needed to simulate this scenario?\n\n{scenario_text[:2000]}"}
        ],
    )

    # Parse the topic queries
    topic_response = topic_message.content[0].text.strip()
    if topic_response.startswith("```"):
        topic_response = re.sub(r"^```(?:json)?\n?", "", topic_response)
        topic_response = re.sub(r"\n?```$", "", topic_response)

    try:
        queries = json.loads(topic_response)
        if not isinstance(queries, list):
            queries = []
    except json.JSONDecodeError:
        queries = []

    # Fallback if parsing fails
    if not queries:
        queries = [
            "fördröjningsstrid fördröjande fältarbeten",
            "minering mineringsnormer tidsåtgång",
            "strid bebyggelse urban terräng",
            "framryckning anfall manöver",
        ]

    print(f"[SimGenerator] Doctrine queries: {queries}")

    # STEP 2: Retrieve chunks for each identified topic
    print("[SimGenerator] Step 2: Retrieving doctrine chunks...")
    all_candidates = {}
    for q in queries:
        q_emb = embed_model.encode([q], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_emb, 8)
        for score, idx in zip(scores[0], indices[0]):
            idx = int(idx)
            if idx not in all_candidates or float(score) > all_candidates[idx]:
                all_candidates[idx] = float(score)

    # Sort and take top candidates for reranking
    ranked = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:25]

    # Rerank against a composite query derived from the scenario
    rerank_query = " ".join(queries[:3])
    pairs = [[rerank_query, chunks[idx]] for idx, _ in ranked]
    rerank_scores = reranker.predict(pairs)

    # Sort by rerank score and return top N unique sources
    reranked = sorted(zip(ranked, rerank_scores), key=lambda x: x[1], reverse=True)

    # Deduplicate by source — take best chunk per source document
    seen_sources = set()
    result = []
    for (idx, _), score in reranked:
        source = sources[idx]
        if source not in seen_sources and len(result) < n_chunks:
            seen_sources.add(source)
            result.append(f"[Källa: {source}, relevans: {score:.2f}]: {chunks[idx]}")

    print(f"[SimGenerator] Retrieved {len(result)} doctrine chunks from {len(seen_sources)} unique sources")
    return result


SIM_PROMPT = """You are a military simulation engine that follows SWEDISH MILITARY DOCTRINE as primary source.

You are provided with:
1. A tactical scenario describing forces, terrain, and objectives
2. Excerpts from actual Swedish military doctrine documents (retrieved from corpus)

DOCTRINE PRIORITY:
1. FIRST: Use provided Swedish doctrine excerpts for tactical decisions. Reference as [SWE: document_name] where document_name is taken from the "Källa:" field of the doctrine excerpt.
2. FALLBACK: If the provided doctrine does NOT cover a specific topic (e.g., specific engagement norms, vehicle capabilities, foreign force TTPs), use open-source NATO doctrine (AJP/ATP series). Reference as [NATO — ej i svenskt underlag].
3. Mark ANY event where you fall back to NATO doctrine in the event log with "(NATO)" prefix.

Base tactical decisions on doctrine:
- How defending forces conduct fördröjningsstrid (delay combat)
- How and when fältarbeten (field works) are executed
- Minering (mining) norms and timing
- Fördröjningslinjer (delay lines) and withdrawal procedures
- Strid i bebyggelse (urban combat) principles
- Engagement distances and weapons employment
- Movement rates and obstacle effects

RULES:
- Generate exactly 24 frames (one every 2 hours, H+00 to H+46)
- Each frame has ALL unit positions, structure statuses, and events at that timestep
- Units must move realistically along roads/terrain toward objectives
- Hostile forces should split into main body + flanking elements
- Apply doctrine-correct timing for mine clearing, bridge demolition, and movement
- Friendly forces conduct fördröjningsstrid according to Swedish doctrine principles
- Include combat engagements when forces are within weapon range
- Vehicle losses accumulate from mines, anti-tank weapons, and combat
- Use actual coordinates that follow roads between the locations
- Keep unit names SHORT (max 12 chars)
- In events, indicate doctrine source: [SWE: DocumentName] or (NATO) for fallback

OUTPUT FORMAT (JSON array of 24 frame objects):
[
  {
    "hour": 0,
    "events": ["H+00: Event description [SWE: Pibat]", "H+00: (NATO) Fallback event"],
    "units": [
      {"id": "X", "name": "short name", "side": "friendly|hostile", "position": [lat, lon], "vehicles_remaining": N, "activity": "short desc"}
    ],
    "structures": [
      {"id": "X", "name": "short", "position": [lat, lon], "status": "intact|prepared|destroyed"}
    ],
    "minefields": [
      {"id": "X", "name": "short", "position": [lat, lon], "cleared": false}
    ]
  }
]

IMPORTANT:
- Keep ALL strings SHORT to minimize token usage
- Include 2-3 hostile sub-units and 2-3 friendly sub-units
- Events should reference which doctrine source was used
- Output ONLY the JSON array, no other text
"""


def generate_simulation_timeline(scenario_text: str) -> list:
    """
    Send scenario + retrieved Swedish doctrine to Claude for a doctrine-grounded simulation.

    Args:
        scenario_text: The scenario description

    Returns: list of frame dicts
    """
    # Retrieve relevant doctrine chunks
    print("[SimGenerator] Retrieving doctrine from RAG index...")
    doctrine_chunks = get_doctrine_chunks(scenario_text, n_chunks=10)
    doctrine_text = "\n\n".join(doctrine_chunks)
    print(f"[SimGenerator] Retrieved {len(doctrine_chunks)} doctrine chunks")

    # Build the full message with doctrine + scenario
    user_message = f"""SWEDISH DOCTRINE EXCERPTS (use these for tactical decisions):

{doctrine_text}

---

SCENARIO TO SIMULATE:

{scenario_text}

---

Generate the 24-frame simulation JSON based on the above doctrine and scenario."""

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16000,
        temperature=0.2,  # Low temperature: stable JSON, some tactical variety
        system=SIM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    response_text = message.content[0].text.strip()

    # Clean markdown fences
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
        response_text = re.sub(r"\n?```$", "", response_text)

    try:
        frames = json.loads(response_text)
        if isinstance(frames, list) and len(frames) > 0:
            return frames
    except json.JSONDecodeError as e:
        print(f"[SimGenerator] JSON error: {e}")
        print(f"[SimGenerator] Response length: {len(response_text)} chars")
        # Salvage partial JSON — work backwards to find valid array
        for i in range(len(response_text) - 1, 100, -1):
            if response_text[i] == '}':
                candidate = response_text[:i+1] + "]"
                try:
                    frames = json.loads(candidate)
                    if isinstance(frames, list) and len(frames) >= 3:
                        print(f"[SimGenerator] Salvaged {len(frames)} frames")
                        return frames
                except:
                    continue

    # Retry with simpler prompt if first attempt failed
    print("[SimGenerator] Retrying with simplified request...")
    try:
        retry_msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10000,
            temperature=0.2,  # Low temperature for stable JSON on retry
            system="Generate a JSON array of 12 simulation frames. Each: {\"hour\":N,\"events\":[\"str\"],\"units\":[{\"id\":\"x\",\"name\":\"x\",\"side\":\"friendly|hostile\",\"position\":[lat,lon],\"vehicles_remaining\":N,\"activity\":\"x\"}],\"structures\":[{\"id\":\"x\",\"name\":\"x\",\"position\":[lat,lon],\"status\":\"intact|destroyed\"}],\"minefields\":[{\"id\":\"x\",\"name\":\"x\",\"position\":[lat,lon],\"cleared\":false}]}. Keep ALL strings under 30 chars. No special characters. Output ONLY valid JSON array.",
            messages=[{"role": "user", "content": f"12 frames (every 2h, H+00 to H+22):\n{scenario_text[:1500]}"}],
        )
        retry_text = retry_msg.content[0].text.strip()
        if retry_text.startswith("```"):
            retry_text = re.sub(r"^```(?:json)?\n?", "", retry_text)
            retry_text = re.sub(r"\n?```$", "", retry_text)
        frames = json.loads(retry_text)
        if isinstance(frames, list) and len(frames) >= 3:
            print(f"[SimGenerator] Retry succeeded: {len(frames)} frames")
            return frames
    except Exception as e2:
        print(f"[SimGenerator] Retry failed: {e2}")

    return []


if __name__ == "__main__":
    test_scenario = """
    EGNA: Amfibiebataljon Amf 4, Göteborg (57.700, 11.920), 400 man, 16 CB90, RBS-17
    FIENDE: VDV BTG, Landvetter (57.669, 12.292), 600 man, 18 BMD-4M, 6 Sprut-SD
    BROAR: Rv40 Mölndalsån (57.678, 12.010), Göteborgsvägen (57.672, 11.990), Kvarnbygatan (57.665, 11.970)
    TUNNEL: Kallebäck (57.685, 11.980)
    MINFÄLT: Mölnlycke (57.670, 12.100), Kållered (57.640, 12.080), Mölndalsån (57.680, 12.030)
    """
    
    print("Generating simulation...")
    frames = generate_simulation_timeline(test_scenario)
    print(f"Got {len(frames)} frames")
    if frames:
        print(f"Frame 0: {json.dumps(frames[0], indent=2, ensure_ascii=False)[:500]}")
