# Dynamic Scenario Parsing — Design Document

## Problem

Currently the map and simulation are hardcoded. When a user edits the scenario textbox (adds units, changes locations, modifies forces), the map and simulation don't reflect those changes. They always show the same Amf 4 vs VDV scenario regardless of what's written in the scenario editor.

## Goal

When the scenario text is updated, the map and simulation should automatically reflect the new content — new units appear on the map, positions update, the simulation timeline adapts to the new force balance.

## Architecture

```
Scenario Textbox (user-editable)
       │
       ▼
┌─────────────────────────────┐
│  Scenario Parser            │
│  (Claude API or regex)      │
│  Extracts structured data:  │
│  - Units (name, side, pos)  │
│  - Structures (bridges)     │
│  - Locations (minefields)   │
│  - Terrain features         │
│  - Engineering norms        │
└─────────────────────────────┘
       │  Structured data
       ▼
┌─────────────────────────────┐
│  UND.load_from_parsed()     │
│  Populates intelligence     │
│  state from parsed output   │
└─────────────────────────────┘
       │
       ├──────────────────────────────┐
       ▼                              ▼
┌──────────────┐         ┌─────────────────────┐
│  Map render  │         │  Simulation engine  │
│  Reads from  │         │  Reads initial from │
│  UND state   │         │  UND, advances by   │
│              │         │  doctrinal norms    │
└──────────────┘         └─────────────────────┘
```

## Component 1: Scenario Parser

### Option A: Claude API (recommended)

Send the scenario text to Claude with a structured output prompt:

```python
def parse_scenario(scenario_text: str) -> dict:
    """Use Claude to extract structured entities from free-text scenario."""
    prompt = """
    Extract all military entities from this scenario into JSON format:
    {
      "units": [
        {"id": "...", "name": "...", "side": "friendly|hostile", 
         "type": "battalion|company|platoon",
         "position": [lat, lon], "strength": N,
         "vehicles": {"type": count}, "parent_id": "..."|null}
      ],
      "structures": [
        {"id": "...", "name": "...", "type": "bridge|tunnel|airfield",
         "position": [lat, lon], "span_m": N, "demolition_kg": N}
      ],
      "locations": [
        {"id": "...", "name": "...", "type": "minefield|phase_line|objective",
         "position": [lat, lon] or [[lat,lon],...],
         "mine_count": N, "mine_type": "..."}
      ],
      "terrain": {
        "center": [lat, lon],
        "distance_km": N,
        "advance_rate_kmh": N,
        "obstacle_clearing_h": N
      }
    }
    """
    # Call Claude, parse JSON response
    ...
```

**Pros:** Handles any free-text format, extracts coordinates from place names, understands military context.
**Cons:** API cost per parse, ~3-5s latency, non-deterministic.

### Option B: Regex + Template (fallback)

Parse the scenario using regex patterns for known structures:

```python
# Example patterns
unit_pattern = r"•\s+(\d+)\s*×\s*(\w[\w\s]+)\s*\(([^)]+)\)"
position_pattern = r"Gruppering:\s*.*?\(N\s*([\d.]+)'?,\s*E\s*([\d.]+)'?\)"
bridge_pattern = r"Bro\s+\d+\s+\"([^\"]+)\".*?spann\s+(\d+)m.*?(\d+)\s*kg"
```

**Pros:** No API cost, deterministic, fast.
**Cons:** Brittle, breaks with format changes, can't resolve place names to coordinates.

### Recommended: Hybrid

1. Try regex first for well-structured sections (bridge specs, org tables)
2. Fall back to Claude for ambiguous content (place names → coordinates, implied positions)
3. Cache parsed results to avoid re-parsing on every map/sim click

## Component 2: UND Population from Parsed Data

Add a new method to UND:

```python
class UND:
    def load_from_parsed(self, parsed: dict):
        """Replace current state with parsed scenario data."""
        self.units.clear()
        self.structures.clear()
        self.locations.clear()
        
        for u in parsed["units"]:
            self.add_unit(Unit(
                id=u["id"], name=u["name"],
                side=Side(u["side"]),
                unit_type=UnitType(u["type"]),
                position=u["position"],
                strength=u.get("strength"),
                vehicles=u.get("vehicles", {}),
                parent_id=u.get("parent_id"),
            ))
        
        for s in parsed["structures"]:
            self.add_structure(Structure(...))
        
        for loc in parsed["locations"]:
            self.add_location(StrategicLocation(...))
```

## Component 3: Dynamic Map Generation

Replace hardcoded markers with UND queries:

```python
def generate_map(und: UND):
    m = folium.Map(location=und.get_center(), zoom_start=12)
    
    for unit in und.get_units_by_side(Side.FRIENDLY):
        icon = make_nato_icon(unit.side, unit.unit_type, unit.name)
        folium.Marker(unit.position, icon=icon, popup=format_popup(unit)).add_to(m)
    
    for unit in und.get_units_by_side(Side.HOSTILE):
        icon = make_nato_icon(unit.side, unit.unit_type, unit.name)
        folium.Marker(unit.position, icon=icon, popup=format_popup(unit)).add_to(m)
    
    for struct in und.structures.values():
        icon = make_structure_icon(struct)
        folium.Marker(struct.position, icon=icon, popup=format_popup(struct)).add_to(m)
    
    for loc in und.locations.values():
        if loc.location_type == LocationType.MINEFIELD:
            draw_minefield(m, loc)
        elif loc.location_type == LocationType.PHASE_LINE:
            draw_phase_line(m, loc)
    
    return m._repr_html_()
```

## Component 4: Dynamic Simulation

Instead of hardcoded position arrays, compute movement per timestep:

```python
def simulate_step(und: UND, step_hours: int = 2):
    """Advance all units by one time step based on doctrinal norms."""
    for unit in und.get_units_by_side(Side.HOSTILE):
        if unit.activity == "advancing":
            # Check for obstacles in path
            obstacle = find_next_obstacle(unit, und)
            if obstacle:
                unit.activity = f"clearing {obstacle.name}"
                unit.speed_kmh = 0
            else:
                # Advance at doctrinal rate
                new_pos = advance_along_route(unit.position, unit.heading, 
                                             unit.speed_kmh * step_hours)
                unit.position = new_pos
    
    # Check for engagements (range-based)
    for friendly in und.get_units_by_side(Side.FRIENDLY):
        for hostile in und.get_units_by_side(Side.HOSTILE):
            if in_weapon_range(friendly, hostile):
                resolve_engagement(friendly, hostile, und)
```

### Movement Rules (from scenario norms)
- Advance with opposition: 5-10 km/day
- Advance without opposition: 30-50 km/day
- Minefield clearing (manual, no IMR-2): 1-2 hours
- Minefield clearing (mechanical, IMR-2): 20-40 minutes
- Bridge crossing without bridge: 2-4 hours (improvised)
- Urban advance: 2-5 km/day

### Engagement Rules
- RBS-17 range: 8 km (kills BMD-4M/Sprut-SD)
- Carl Gustaf range: 300-700m (kills all VDV vehicles)
- 81mm GrK range: 5.5 km (suppression/attrition)
- BMD-4M 100mm range: 4 km
- 2S9 Nona range: 8.8 km

## Integration in Web Interface

```python
# When scenario changes or map/sim button is clicked:
def on_scenario_change(scenario_text):
    parsed = parse_scenario(scenario_text)
    und = UND()
    und.load_from_parsed(parsed)
    return und

def show_map(scenario_text):
    und = on_scenario_change(scenario_text)
    return generate_map(und)

def run_simulation(scenario_text):
    und = on_scenario_change(scenario_text)
    frames = simulate(und, hours=48, step=2)
    return render_frames(frames)
```

## Migration Steps

1. Add `UND.load_from_parsed()` method
2. Implement `parse_scenario()` using Claude API
3. Refactor `generate_map()` to read from UND instead of hardcoded data
4. Refactor simulation to use UND state + movement rules
5. Connect scenario textbox changes to re-parsing
6. Cache parsed state to avoid re-parsing on every button click
7. Add "🔄 Uppdatera scenario" button that triggers re-parse

## Estimated Effort

| Component | Complexity | Time |
|-----------|-----------|------|
| Scenario parser (Claude) | Medium | 2-3 hours |
| UND.load_from_parsed() | Low | 30 min |
| Dynamic map from UND | Medium | 2 hours |
| Dynamic simulation from UND | High | 4-6 hours |
| Integration + testing | Medium | 2 hours |
| **Total** | | **~12 hours** |
