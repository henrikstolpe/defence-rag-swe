"""
UND — Underrättelsemodul (Intelligence Module)

Tracks intelligence data for:
- Military units and subunits (own and enemy)
- Key structures (bridges, tunnels, infrastructure)
- Strategic locations (objectives, phase lines, chokepoints)

Supports:
- Initial scenario population
- Runtime intelligence injection (observations, reports)
- Time-stamped intelligence history
- Confidence levels and source tracking
- Querying current state at any simulation time

Usage:
    from und import UND
    intel = UND()
    intel.load_scenario()  # Populates from default scenario
    intel.inject("VDV_BTG", "position", [57.665, 12.20], source="KJ spaning", confidence=0.8)
    state = intel.get_unit("VDV_BTG")
"""
import json
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional


class Side(Enum):
    FRIENDLY = "friendly"
    HOSTILE = "hostile"
    NEUTRAL = "neutral"


class UnitType(Enum):
    BATTALION = "battalion"
    COMPANY = "company"
    PLATOON = "platoon"
    SQUAD = "squad"
    TEAM = "team"


class StructureType(Enum):
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    AIRFIELD = "airfield"
    PORT = "port"
    ROAD = "road"
    RAILWAY = "railway"
    BUILDING = "building"


class StructureStatus(Enum):
    INTACT = "intact"
    PREPARED = "prepared"  # Ready for demolition
    DAMAGED = "damaged"
    DESTROYED = "destroyed"
    BLOCKED = "blocked"


class LocationType(Enum):
    OBJECTIVE = "objective"
    PHASE_LINE = "phase_line"
    CHOKEPOINT = "chokepoint"
    ASSEMBLY_AREA = "assembly_area"
    SUPPLY_POINT = "supply_point"
    OBSERVATION_POST = "observation_post"
    MINEFIELD = "minefield"
    FIRING_POSITION = "firing_position"


@dataclass
class IntelReport:
    """A single intelligence observation/report."""
    timestamp: str  # Simulation time (e.g., "H+04")
    attribute: str  # What was observed (position, strength, activity, etc.)
    value: any  # The observed value
    source: str  # Who/what reported it (e.g., "KJ spaning", "Orlan-10", "SIGINT")
    confidence: float  # 0.0 to 1.0
    real_time: str = ""  # When the report was created (wall clock)

    def __post_init__(self):
        if not self.real_time:
            self.real_time = datetime.now().isoformat()


@dataclass
class Unit:
    """A military unit with tracked intelligence attributes."""
    id: str
    name: str
    side: Side
    unit_type: UnitType
    parent_id: Optional[str] = None  # Parent unit ID

    # Current assessed state
    position: Optional[list] = None  # [lat, lon]
    strength: Optional[int] = None  # Personnel count
    vehicles: dict = field(default_factory=dict)  # {"BMD-4M": 18, ...}
    weapons: dict = field(default_factory=dict)  # {"2S9 Nona-S": 6, ...}
    ammunition: dict = field(default_factory=dict)  # {"125mm": 120, ...}
    morale: Optional[str] = None  # "high", "medium", "low", "broken"
    readiness: Optional[float] = None  # 0.0 to 1.0
    activity: Optional[str] = None  # "advancing", "defending", "withdrawing", etc.
    heading: Optional[float] = None  # Degrees (direction of movement)
    speed_kmh: Optional[float] = None  # Current movement speed

    # Intelligence history
    reports: list = field(default_factory=list)  # List of IntelReport

    def to_dict(self):
        d = asdict(self)
        d["side"] = self.side.value
        d["unit_type"] = self.unit_type.value
        d["reports"] = [asdict(r) for r in self.reports]
        return d


@dataclass
class Structure:
    """A key structure (bridge, tunnel, etc.) with tracked state."""
    id: str
    name: str
    structure_type: StructureType
    position: list  # [lat, lon]
    status: StructureStatus = StructureStatus.INTACT

    # Engineering data
    span_m: Optional[float] = None
    load_class: Optional[int] = None
    demolition_kg: Optional[float] = None  # Explosives needed
    demolition_time_h: Optional[float] = None  # Prep time in hours
    personnel_needed: Optional[int] = None

    # Intel history
    reports: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["structure_type"] = self.structure_type.value
        d["status"] = self.status.value
        d["reports"] = [asdict(r) for r in self.reports]
        return d


@dataclass
class StrategicLocation:
    """A strategic location (objective, minefield, phase line, etc.)."""
    id: str
    name: str
    location_type: LocationType
    position: list  # [lat, lon] or [[lat, lon], ...] for lines
    controlled_by: Optional[Side] = None
    description: Optional[str] = None

    # For minefields
    mine_count: Optional[int] = None
    mine_type: Optional[str] = None
    cleared: bool = False

    # Intel history
    reports: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["location_type"] = self.location_type.value
        d["controlled_by"] = self.controlled_by.value if self.controlled_by else None
        d["reports"] = [asdict(r) for r in self.reports]
        return d


class UND:
    """
    Underrättelsemodul — Intelligence tracking system.

    Maintains assessed state of all units, structures, and locations.
    Supports injection of new intelligence at any time during simulation.
    """

    def __init__(self):
        self.units: dict[str, Unit] = {}
        self.structures: dict[str, Structure] = {}
        self.locations: dict[str, StrategicLocation] = {}
        self.sim_time: str = "H+00"
        self.global_log: list[IntelReport] = []

    # ================================================================
    # Scenario population
    # ================================================================
    def load_from_parsed(self, parsed: dict):
        """Populate UND from scenario parser output. Replaces current state."""
        self.units.clear()
        self.structures.clear()
        self.locations.clear()
        self.global_log.clear()
        self.sim_time = "H+00"

        for u in parsed.get("units", []):
            self.add_unit(Unit(
                id=u["id"],
                name=u["name"],
                side=Side(u["side"]),
                unit_type=UnitType(u["type"]),
                position=u.get("position"),
                strength=u.get("strength"),
                vehicles=u.get("vehicles", {}),
                weapons=u.get("weapons", {}),
                ammunition=u.get("ammunition", {}),
                morale=u.get("morale"),
                readiness=u.get("readiness"),
                activity=u.get("activity"),
                heading=u.get("heading"),
                speed_kmh=u.get("speed_kmh"),
                parent_id=u.get("parent_id"),
            ))

        for s in parsed.get("structures", []):
            self.add_structure(Structure(
                id=s["id"],
                name=s["name"],
                structure_type=StructureType(s["type"]),
                position=s["position"],
                status=StructureStatus(s.get("status", "intact")),
                span_m=s.get("span_m"),
                load_class=s.get("load_class"),
                demolition_kg=s.get("demolition_kg"),
                demolition_time_h=s.get("demolition_time_h"),
                personnel_needed=s.get("personnel_needed"),
            ))

        for loc in parsed.get("locations", []):
            controlled = Side(loc["controlled_by"]) if loc.get("controlled_by") else None
            self.add_location(StrategicLocation(
                id=loc["id"],
                name=loc["name"],
                location_type=LocationType(loc["type"]),
                position=loc["position"],
                controlled_by=controlled,
                mine_count=loc.get("mine_count"),
                mine_type=loc.get("mine_type"),
                description=loc.get("description"),
            ))

    def load_scenario(self):
        """Populate UND with the default Göteborg/Landvetter scenario."""

        # --- FRIENDLY UNITS ---
        self.add_unit(Unit(
            id="AMF4_BAT",
            name="Amfibiebataljon (Amf 4)",
            side=Side.FRIENDLY,
            unit_type=UnitType.BATTALION,
            position=[57.7000, 11.9200],
            strength=400,
            vehicles={"CB90": 16, "Terrängbil": 15, "Bv410": 2},
            weapons={"RBS-17": 4, "81mm GrK": 4, "Carl Gustaf": 12},
            ammunition={"RBS-17 robot": 16, "81mm granat": 200, "Carl Gustaf": 60},
            morale="high",
            readiness=0.9,
            activity="preparing defence",
        ))

        self.add_unit(Unit(
            id="AMF4_KP1",
            name="1. Amfibiekompaniet",
            side=Side.FRIENDLY,
            unit_type=UnitType.COMPANY,
            parent_id="AMF4_BAT",
            position=[57.6900, 11.94],
            strength=100,
            activity="forward positioning",
        ))

        self.add_unit(Unit(
            id="AMF4_KP2",
            name="2. Amfibiekompaniet",
            side=Side.FRIENDLY,
            unit_type=UnitType.COMPANY,
            parent_id="AMF4_BAT",
            position=[57.7000, 11.92],
            strength=100,
            activity="reserve",
        ))

        self.add_unit(Unit(
            id="AMF4_KJ",
            name="Kustjägargrupp",
            side=Side.FRIENDLY,
            unit_type=UnitType.PLATOON,
            parent_id="AMF4_BAT",
            position=[57.6580, 12.25],
            strength=20,
            activity="reconnaissance behind enemy lines",
        ))

        self.add_unit(Unit(
            id="AMF4_RBS",
            name="Kustrobotbatteri (RBS-17)",
            side=Side.FRIENDLY,
            unit_type=UnitType.PLATOON,
            parent_id="AMF4_BAT",
            position=[57.6750, 12.00],
            strength=16,
            weapons={"RBS-17": 4},
            ammunition={"RBS-17 robot": 16},
            activity="in firing position",
        ))

        self.add_unit(Unit(
            id="AMF4_PIONEER",
            name="Minröjnings-/mineringspluton",
            side=Side.FRIENDLY,
            unit_type=UnitType.PLATOON,
            parent_id="AMF4_BAT",
            position=[57.6780, 12.01],
            strength=20,
            ammunition={"FFV 028 PV-mina": 100, "Sprängdeg m/46 (kg)": 150, "Broladdning 25kg": 6},
            activity="preparing bridge demolition",
        ))

        # --- HOSTILE UNITS ---
        self.add_unit(Unit(
            id="VDV_BTG",
            name="VDV Bataljonstridsgrupp (76:e Div)",
            side=Side.HOSTILE,
            unit_type=UnitType.BATTALION,
            position=[57.6686, 12.2919],
            strength=600,
            vehicles={"BMD-4M": 18, "Sprut-SD": 6, "2S9 Nona-S": 6, "BTR-ZD": 2, "BTR-D": 15, "Orlan-10": 2},
            weapons={"125mm 2A75": 6, "100mm 2A70": 18, "30mm 2A72": 18, "120mm Nona": 6, "Igla-S": 4},
            ammunition={"125mm": 120, "100mm": 600, "120mm Nona": 300, "Igla-S": 16},
            morale="high",
            readiness=1.0,
            activity="preparing advance",
            heading=270,  # West toward Göteborg
            speed_kmh=0,
        ))

        self.add_unit(Unit(
            id="VDV_MAIN",
            name="VDV Huvudstyrka (Rv40-axeln)",
            side=Side.HOSTILE,
            unit_type=UnitType.COMPANY,
            parent_id="VDV_BTG",
            position=[57.6686, 12.2919],
            strength=350,
            vehicles={"BMD-4M": 12, "Sprut-SD": 4, "2S9 Nona-S": 4},
            activity="preparing advance along Rv40",
            heading=270,
        ))

        self.add_unit(Unit(
            id="VDV_FLANK",
            name="VDV Flankerande styrka (söder)",
            side=Side.HOSTILE,
            unit_type=UnitType.COMPANY,
            parent_id="VDV_BTG",
            position=[57.6686, 12.2919],
            strength=200,
            vehicles={"BMD-4M": 6, "Sprut-SD": 2, "2S9 Nona-S": 2},
            activity="preparing southern flanking",
            heading=240,
        ))

        # --- STRUCTURES ---
        self.add_structure(Structure(
            id="BRO1_RV40",
            name="Bro 1: Rv40 Mölndalsån",
            structure_type=StructureType.BRIDGE,
            position=[57.6780, 12.01],
            span_m=25,
            load_class=70,
            demolition_kg=200,
            demolition_time_h=4.5,
            personnel_needed=4,
        ))

        self.add_structure(Structure(
            id="BRO2_GOTEBORG",
            name="Bro 2: Göteborgsvägen",
            structure_type=StructureType.BRIDGE,
            position=[57.6720, 11.99],
            span_m=15,
            load_class=60,
            demolition_kg=100,
            demolition_time_h=3,
            personnel_needed=4,
        ))

        self.add_structure(Structure(
            id="BRO3_KVARNBY",
            name="Bro 3: Kvarnbygatan",
            structure_type=StructureType.BRIDGE,
            position=[57.6650, 11.97],
            span_m=10,
            load_class=40,
            demolition_kg=75,
            demolition_time_h=2,
            personnel_needed=2,
        ))

        self.add_structure(Structure(
            id="TUNNEL_KALLEBACK",
            name="Rv40-tunneln Kallebäck",
            structure_type=StructureType.TUNNEL,
            position=[57.6850, 11.98],
            span_m=900,
            demolition_kg=100,
            demolition_time_h=3.5,
            personnel_needed=4,
        ))

        self.add_structure(Structure(
            id="LANDVETTER_AP",
            name="Landvetter flygplats",
            structure_type=StructureType.AIRFIELD,
            position=[57.6686, 12.2919],
            status=StructureStatus.INTACT,
        ))

        # --- STRATEGIC LOCATIONS ---
        self.add_location(StrategicLocation(
            id="MF1_MOLNLYCKE",
            name="Minfält 1: Rv40 Mölnlycke",
            location_type=LocationType.MINEFIELD,
            position=[57.6700, 12.10],
            controlled_by=Side.FRIENDLY,
            mine_count=100,
            mine_type="FFV 028 PV-mina",
            description="Blockerar huvudaxeln Rv40",
        ))

        self.add_location(StrategicLocation(
            id="MF2_KALLERED",
            name="Minfält 2: Kållered syd",
            location_type=LocationType.MINEFIELD,
            position=[57.6400, 12.08],
            controlled_by=Side.FRIENDLY,
            mine_count=50,
            mine_type="FFV 028 PV-mina",
            description="Hindrar flanking söder",
        ))

        self.add_location(StrategicLocation(
            id="MF3_MOLNDALSAN",
            name="Minfält 3: Framför Mölndalsån",
            location_type=LocationType.MINEFIELD,
            position=[57.6800, 12.03],
            controlled_by=Side.FRIENDLY,
            mine_count=50,
            mine_type="FFV 028 + truppmina",
            description="Kanaliserar mot sprängda broar",
        ))

        self.add_location(StrategicLocation(
            id="PL_ALFA",
            name="PL ALFA — Mölnlycke",
            location_type=LocationType.PHASE_LINE,
            position=[[57.6750, 12.12], [57.6600, 12.12], [57.6500, 12.10]],
            description="Fördröjningslinje 1 (14 km)",
        ))

        self.add_location(StrategicLocation(
            id="PL_BRAVO",
            name="PL BRAVO — Mölndalsån",
            location_type=LocationType.PHASE_LINE,
            position=[[57.6850, 12.02], [57.6750, 12.00], [57.6600, 11.97]],
            description="Fördröjningslinje 2 (6-8 km)",
        ))

        self.add_location(StrategicLocation(
            id="PL_CHARLIE",
            name="PL CHARLIE — Mölndal",
            location_type=LocationType.PHASE_LINE,
            position=[[57.6900, 11.97], [57.6800, 11.96], [57.6650, 11.95]],
            description="Fördröjningslinje 3 — SISTA LINJEN (5 km)",
        ))

        self.add_location(StrategicLocation(
            id="OBJ_HAMN",
            name="Mål: Göteborgs hamn",
            location_type=LocationType.OBJECTIVE,
            position=[57.7100, 11.93],
            controlled_by=Side.FRIENDLY,
            description="VDV:s troliga mål — hamn för sjöburen förstärkning",
        ))

    # ================================================================
    # CRUD operations
    # ================================================================
    def add_unit(self, unit: Unit):
        self.units[unit.id] = unit

    def add_structure(self, structure: Structure):
        self.structures[structure.id] = structure

    def add_location(self, location: StrategicLocation):
        self.locations[location.id] = location

    def get_unit(self, unit_id: str) -> Optional[Unit]:
        return self.units.get(unit_id)

    def get_structure(self, structure_id: str) -> Optional[Structure]:
        return self.structures.get(structure_id)

    def get_location(self, location_id: str) -> Optional[StrategicLocation]:
        return self.locations.get(location_id)

    def get_subunits(self, parent_id: str) -> list[Unit]:
        return [u for u in self.units.values() if u.parent_id == parent_id]

    def get_units_by_side(self, side: Side) -> list[Unit]:
        return [u for u in self.units.values() if u.side == side]

    # ================================================================
    # Intelligence injection
    # ================================================================
    def inject(self, entity_id: str, attribute: str, value, source: str = "unknown",
               confidence: float = 0.5, sim_time: Optional[str] = None):
        """
        Inject new intelligence about a unit, structure, or location.

        Args:
            entity_id: ID of the entity (unit, structure, or location)
            attribute: What was observed (e.g., "position", "strength", "status", "activity")
            value: The observed value
            source: Intelligence source (e.g., "KJ spaning", "SIGINT", "UAV Raven")
            confidence: Confidence level 0.0-1.0
            sim_time: Simulation time (defaults to current sim_time)
        """
        time = sim_time or self.sim_time

        report = IntelReport(
            timestamp=time,
            attribute=attribute,
            value=value,
            source=source,
            confidence=confidence,
        )

        # Add to global log
        self.global_log.append(report)

        # Update the entity
        entity = self.units.get(entity_id) or self.structures.get(entity_id) or self.locations.get(entity_id)
        if entity is None:
            print(f"[UND] Warning: Unknown entity '{entity_id}'. Creating report without entity update.")
            return report

        # Add report to entity history
        entity.reports.append(report)

        # Update entity state if confidence is sufficient
        if confidence >= 0.5:
            if isinstance(entity, Structure) and attribute == "status":
                entity.status = StructureStatus(value) if isinstance(value, str) else value
            elif isinstance(entity, Unit) and attribute in ("vehicles", "weapons", "ammunition"):
                # For dict attributes, update individual keys
                if isinstance(value, dict):
                    getattr(entity, attribute).update(value)
            elif hasattr(entity, attribute):
                setattr(entity, attribute, value)

        return report

    def set_sim_time(self, sim_time: str):
        """Advance simulation time."""
        self.sim_time = sim_time

    # ================================================================
    # Querying
    # ================================================================
    def get_situation_summary(self) -> str:
        """Get a text summary of the current intelligence picture."""
        lines = [f"=== UNDERRÄTTELSELÄGE {self.sim_time} ===\n"]

        lines.append("--- EGNA FÖRBAND ---")
        for u in self.get_units_by_side(Side.FRIENDLY):
            pos = f"({u.position[0]:.4f}, {u.position[1]:.4f})" if u.position else "okänd"
            lines.append(f"  {u.name} [{u.id}]: pos={pos}, styrka={u.strength}, aktivitet={u.activity}")

        lines.append("\n--- FIENDEFÖRBAND ---")
        for u in self.get_units_by_side(Side.HOSTILE):
            pos = f"({u.position[0]:.4f}, {u.position[1]:.4f})" if u.position else "okänd"
            veh = ", ".join(f"{k}:{v}" for k, v in u.vehicles.items()) if u.vehicles else "-"
            lines.append(f"  {u.name} [{u.id}]: pos={pos}, styrka={u.strength}, fordon=[{veh}]")

        lines.append("\n--- STRUKTURER ---")
        for s in self.structures.values():
            lines.append(f"  {s.name} [{s.id}]: status={s.status.value}")

        lines.append("\n--- MINFÄLT ---")
        for loc in self.locations.values():
            if loc.location_type == LocationType.MINEFIELD:
                status = "RÖJT" if loc.cleared else f"{loc.mine_count} minor"
                lines.append(f"  {loc.name}: {status}")

        return "\n".join(lines)

    def get_intel_history(self, entity_id: str) -> list[IntelReport]:
        """Get all intelligence reports for an entity."""
        entity = self.units.get(entity_id) or self.structures.get(entity_id) or self.locations.get(entity_id)
        if entity:
            return entity.reports
        return []

    # ================================================================
    # Serialization
    # ================================================================
    def save(self, filepath: str = "und_state.json"):
        """Save current intelligence state to JSON."""
        state = {
            "sim_time": self.sim_time,
            "units": {k: v.to_dict() for k, v in self.units.items()},
            "structures": {k: v.to_dict() for k, v in self.structures.items()},
            "locations": {k: v.to_dict() for k, v in self.locations.items()},
            "global_log": [asdict(r) for r in self.global_log],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[UND] State saved to {filepath}")

    def export_for_simulation(self) -> dict:
        """Export current state in a format consumable by the simulation engine."""
        return {
            "sim_time": self.sim_time,
            "friendly_units": [u.to_dict() for u in self.get_units_by_side(Side.FRIENDLY)],
            "hostile_units": [u.to_dict() for u in self.get_units_by_side(Side.HOSTILE)],
            "structures": [s.to_dict() for s in self.structures.values()],
            "locations": [l.to_dict() for l in self.locations.values()],
        }


# ================================================================
# CLI interface for testing
# ================================================================
if __name__ == "__main__":
    intel = UND()
    intel.load_scenario()

    print(intel.get_situation_summary())
    print("\n")

    # Example: inject new intelligence
    print("--- Injecting new intelligence ---")
    intel.set_sim_time("H+04")
    intel.inject("VDV_MAIN", "position", [57.6680, 12.25], source="KJ spaning", confidence=0.9)
    intel.inject("VDV_MAIN", "activity", "advancing along Rv40", source="KJ spaning", confidence=0.9)
    intel.inject("VDV_MAIN", "speed_kmh", 15.0, source="KJ spaning", confidence=0.7)
    intel.inject("VDV_BTG", "strength", 580, source="SIGINT intercept", confidence=0.6)

    print("\n--- After injection ---")
    vdv = intel.get_unit("VDV_MAIN")
    print(f"VDV Main: position={vdv.position}, activity={vdv.activity}, speed={vdv.speed_kmh} km/h")
    print(f"Reports: {len(vdv.reports)}")

    # Save state
    intel.save()
    print("\nDone.")
