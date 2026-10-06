"""
Scenario Parser — Extracts structured data from free-text scenario descriptions.
Uses Claude API to parse military scenario text into UND-compatible entities.

Usage:
    from scenario_parser import parse_scenario
    parsed = parse_scenario(scenario_text)
    und.load_from_parsed(parsed)
"""
import os
import json
import re
import anthropic


PARSE_PROMPT = """You are a military scenario parser. Extract ALL entities from the scenario text into a strict JSON format.

Rules:
- Extract every unit mentioned (friendly and hostile), including subunits
- Extract every structure (bridges, tunnels, airfields, roads)
- Extract every strategic location (minefields, phase lines, objectives, firing positions)
- For positions, use [latitude, longitude] in decimal degrees
- If no exact coordinates given but a place name is mentioned, estimate coordinates for the Swedish location
- Assign unique IDs using format: SIDE_UNITTYPE_NUMBER (e.g., FR_BAT_1, HO_KP_2, STR_BRO_1)
- parent_id links subunits to their parent (null for top-level units)
- For phase lines, position is an array of coordinate pairs [[lat,lon], [lat,lon], ...]

Output ONLY valid JSON, no markdown formatting, no explanation.

JSON Schema:
{
  "units": [
    {
      "id": "string",
      "name": "string",
      "side": "friendly" | "hostile",
      "type": "battalion" | "company" | "platoon" | "squad" | "team",
      "position": [lat, lon],
      "strength": number_or_null,
      "vehicles": {"vehicle_type": count} or {},
      "weapons": {"weapon_type": count} or {},
      "ammunition": {"ammo_type": count} or {},
      "morale": "high" | "medium" | "low" | null,
      "readiness": 0.0_to_1.0_or_null,
      "activity": "string_or_null",
      "heading": degrees_or_null,
      "speed_kmh": number_or_null,
      "parent_id": "string_or_null"
    }
  ],
  "structures": [
    {
      "id": "string",
      "name": "string",
      "type": "bridge" | "tunnel" | "airfield" | "port" | "road" | "railway",
      "position": [lat, lon],
      "status": "intact" | "prepared" | "damaged" | "destroyed" | "blocked",
      "span_m": number_or_null,
      "load_class": number_or_null,
      "demolition_kg": number_or_null,
      "demolition_time_h": number_or_null,
      "personnel_needed": number_or_null
    }
  ],
  "locations": [
    {
      "id": "string",
      "name": "string",
      "type": "minefield" | "phase_line" | "objective" | "chokepoint" | "assembly_area" | "firing_position" | "observation_post",
      "position": [lat, lon] or [[lat,lon], ...],
      "controlled_by": "friendly" | "hostile" | "neutral" | null,
      "mine_count": number_or_null,
      "mine_type": "string_or_null",
      "description": "string_or_null"
    }
  ],
  "terrain": {
    "center": [lat, lon],
    "zoom": 10_to_15,
    "distance_km": number,
    "description": "string"
  },
  "movement_norms": {
    "hostile_advance_with_opposition_kmday": number,
    "hostile_advance_without_opposition_kmday": number,
    "hostile_obstacle_clearing_hours": number,
    "friendly_reposition_speed_kmh": number
  }
}
"""

CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY")


def parse_scenario(scenario_text: str) -> dict:
    """
    Parse a free-text scenario into structured data using Claude.

    Args:
        scenario_text: The scenario description (Swedish military text)

    Returns:
        Parsed dict with units, structures, locations, terrain, movement_norms
    """
    if not scenario_text.strip():
        return {"units": [], "structures": [], "locations": [], "terrain": None, "movement_norms": None}

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        temperature=0.0,  # Deterministic parsing of scenario to structured data
        system=PARSE_PROMPT,
        messages=[
            {"role": "user", "content": f"Parse this scenario:\n\n{scenario_text}"}
        ],
    )

    response_text = message.content[0].text.strip()

    # Clean up response — remove markdown code fences if present
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
        response_text = re.sub(r"\n?```$", "", response_text)

    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"[ScenarioParser] JSON parse error: {e}")
        print(f"[ScenarioParser] Raw response:\n{response_text[:500]}")
        return {"units": [], "structures": [], "locations": [], "terrain": None, "movement_norms": None}

    # Validate and fill defaults
    parsed.setdefault("units", [])
    parsed.setdefault("structures", [])
    parsed.setdefault("locations", [])
    parsed.setdefault("terrain", {"center": [57.68, 12.05], "zoom": 12, "distance_km": 20, "description": ""})
    parsed.setdefault("movement_norms", {
        "hostile_advance_with_opposition_kmday": 7,
        "hostile_advance_without_opposition_kmday": 40,
        "hostile_obstacle_clearing_hours": 1.5,
        "friendly_reposition_speed_kmh": 30,
    })

    return parsed


def parse_scenario_cached(scenario_text: str, _cache={}) -> dict:
    """Parse scenario with simple caching (avoids re-parsing identical text)."""
    key = hash(scenario_text.strip())
    if key not in _cache:
        _cache[key] = parse_scenario(scenario_text)
    return _cache[key]


def clear_parse_cache():
    """Clear the parse cache to force re-parsing."""
    parse_scenario_cached.__defaults__[0].clear()


if __name__ == "__main__":
    # Test with a simple scenario
    test_scenario = """
    === TAKTISKT SCENARIO ===
    TIDPUNKT: D+1, 0600Z
    
    EGNA: Svensk mekaniserad bataljon, 500 man
    Gruppering: Västerås (N 59.62, E 16.53)
    14 × Strv 122, 36 × CV90
    
    FIENDE: Rysk BTG, 800 man
    Gruppering: Sala (N 59.92, E 16.80)
    10 × T-80BVM, 24 × BMP-3
    Framryckningsriktning: sydväst
    
    TERRÄNG: 35 km avstånd, blandad skog/öppet
    2 broar över Svartån (sprängbara)
    """

    print("Parsing scenario...")
    result = parse_scenario(test_scenario)
    print(json.dumps(result, indent=2, ensure_ascii=False))


MAP_PARSE_PROMPT = """Extract unit positions and key structures from this military scenario. Output ONLY a JSON object.
Keep it minimal — only id, name, side, position, and type for each entity.
For positions use [latitude, longitude] decimal degrees. Estimate from place names if needed.

{
  "units": [{"id": "...", "name": "...", "side": "friendly|hostile", "type": "battalion|company|platoon", "position": [lat,lon], "strength": N, "vehicles": {...}, "parent_id": null}],
  "structures": [{"id": "...", "name": "...", "type": "bridge|tunnel|airfield", "position": [lat,lon], "status": "intact", "demolition_kg": N}],
  "locations": [{"id": "...", "name": "...", "type": "minefield|phase_line|objective", "position": [lat,lon], "controlled_by": "friendly|hostile|null", "mine_count": N, "mine_type": "...", "description": "..."}]
}
"""


def parse_scenario_for_map(scenario_text: str) -> dict:
    """Lightweight parsing focused on extracting positions for map rendering."""
    if not scenario_text.strip():
        return {"units": [], "structures": [], "locations": []}

    # Check cache first
    key = hash(("map_" + scenario_text).strip())
    if key in parse_scenario_cached.__defaults__[0]:
        return parse_scenario_cached.__defaults__[0][key]

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=0.0,  # Deterministic position extraction for map rendering
        system=MAP_PARSE_PROMPT,
        messages=[
            {"role": "user", "content": scenario_text[:3000]}  # Limit input to avoid long responses
        ],
    )

    response_text = message.content[0].text.strip()
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\n?", "", response_text)
        response_text = re.sub(r"\n?```$", "", response_text)

    try:
        parsed = json.loads(response_text)
        parsed.setdefault("units", [])
        parsed.setdefault("structures", [])
        parsed.setdefault("locations", [])
        # Cache it
        parse_scenario_cached.__defaults__[0][key] = parsed
        return parsed
    except json.JSONDecodeError:
        return {"units": [], "structures": [], "locations": []}
