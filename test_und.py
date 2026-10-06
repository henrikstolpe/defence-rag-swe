"""
Tests for the UND (Underrättelsemodul) intelligence module.
Run: python test_und.py
"""
import json
import os
from und import UND, Unit, Structure, StrategicLocation, IntelReport
from und import Side, UnitType, StructureType, StructureStatus, LocationType


def test_load_scenario():
    """Test that scenario loads all expected entities."""
    intel = UND()
    intel.load_scenario()

    # Check unit counts
    friendly = intel.get_units_by_side(Side.FRIENDLY)
    hostile = intel.get_units_by_side(Side.HOSTILE)
    assert len(friendly) == 6, f"Expected 6 friendly units, got {len(friendly)}"
    assert len(hostile) == 3, f"Expected 3 hostile units, got {len(hostile)}"

    # Check structures
    assert len(intel.structures) == 5, f"Expected 5 structures, got {len(intel.structures)}"

    # Check locations
    assert len(intel.locations) == 7, f"Expected 7 locations, got {len(intel.locations)}"

    print("✅ test_load_scenario PASSED")


def test_get_unit():
    """Test unit retrieval by ID."""
    intel = UND()
    intel.load_scenario()

    amf = intel.get_unit("AMF4_BAT")
    assert amf is not None
    assert amf.name == "Amfibiebataljon (Amf 4)"
    assert amf.side == Side.FRIENDLY
    assert amf.strength == 400
    assert amf.position == [57.7000, 11.9200]
    assert "CB90" in amf.vehicles
    assert amf.vehicles["CB90"] == 16

    vdv = intel.get_unit("VDV_BTG")
    assert vdv is not None
    assert vdv.side == Side.HOSTILE
    assert vdv.strength == 600
    assert vdv.vehicles["BMD-4M"] == 18

    # Non-existent unit
    assert intel.get_unit("NONEXISTENT") is None

    print("✅ test_get_unit PASSED")


def test_get_subunits():
    """Test subunit retrieval."""
    intel = UND()
    intel.load_scenario()

    amf_subs = intel.get_subunits("AMF4_BAT")
    assert len(amf_subs) == 5, f"Expected 5 subunits of AMF4_BAT, got {len(amf_subs)}"

    sub_ids = {u.id for u in amf_subs}
    assert "AMF4_KP1" in sub_ids
    assert "AMF4_KP2" in sub_ids
    assert "AMF4_KJ" in sub_ids
    assert "AMF4_RBS" in sub_ids
    assert "AMF4_PIONEER" in sub_ids

    vdv_subs = intel.get_subunits("VDV_BTG")
    assert len(vdv_subs) == 2, f"Expected 2 subunits of VDV_BTG, got {len(vdv_subs)}"

    print("✅ test_get_subunits PASSED")


def test_get_structure():
    """Test structure retrieval."""
    intel = UND()
    intel.load_scenario()

    bro1 = intel.get_structure("BRO1_RV40")
    assert bro1 is not None
    assert bro1.name == "Bro 1: Rv40 Mölndalsån"
    assert bro1.structure_type == StructureType.BRIDGE
    assert bro1.status == StructureStatus.INTACT
    assert bro1.span_m == 25
    assert bro1.demolition_kg == 200
    assert bro1.demolition_time_h == 4.5

    tunnel = intel.get_structure("TUNNEL_KALLEBACK")
    assert tunnel is not None
    assert tunnel.structure_type == StructureType.TUNNEL
    assert tunnel.demolition_kg == 100

    print("✅ test_get_structure PASSED")


def test_get_location():
    """Test location retrieval."""
    intel = UND()
    intel.load_scenario()

    mf1 = intel.get_location("MF1_MOLNLYCKE")
    assert mf1 is not None
    assert mf1.location_type == LocationType.MINEFIELD
    assert mf1.mine_count == 100
    assert mf1.mine_type == "FFV 028 PV-mina"
    assert mf1.cleared == False

    pl_a = intel.get_location("PL_ALFA")
    assert pl_a is not None
    assert pl_a.location_type == LocationType.PHASE_LINE
    assert len(pl_a.position) == 3  # 3 coordinate pairs for the line

    print("✅ test_get_location PASSED")


def test_inject_position():
    """Test injecting a position update."""
    intel = UND()
    intel.load_scenario()

    # Original position
    vdv = intel.get_unit("VDV_MAIN")
    assert vdv.position == [57.6686, 12.2919]

    # Inject new position
    intel.set_sim_time("H+04")
    report = intel.inject("VDV_MAIN", "position", [57.668, 12.22], source="KJ spaning", confidence=0.9)

    # Check update
    vdv = intel.get_unit("VDV_MAIN")
    assert vdv.position == [57.668, 12.22], f"Expected updated position, got {vdv.position}"

    # Check report stored
    assert len(vdv.reports) == 1
    assert vdv.reports[0].source == "KJ spaning"
    assert vdv.reports[0].confidence == 0.9
    assert vdv.reports[0].timestamp == "H+04"

    print("✅ test_inject_position PASSED")


def test_inject_low_confidence():
    """Test that low-confidence reports don't update state."""
    intel = UND()
    intel.load_scenario()

    original_pos = intel.get_unit("VDV_MAIN").position.copy()

    # Inject with low confidence
    intel.inject("VDV_MAIN", "position", [57.5, 12.0], source="unconfirmed rumor", confidence=0.3)

    # Position should NOT change
    assert intel.get_unit("VDV_MAIN").position == original_pos

    # But report should still be stored
    assert len(intel.get_unit("VDV_MAIN").reports) == 1

    print("✅ test_inject_low_confidence PASSED")


def test_inject_structure_status():
    """Test injecting structure status changes."""
    intel = UND()
    intel.load_scenario()

    # Bridge starts intact
    assert intel.get_structure("BRO1_RV40").status == StructureStatus.INTACT

    # Prepare for demolition
    intel.set_sim_time("H+16")
    intel.inject("BRO1_RV40", "status", "prepared", source="Pioneer rapport", confidence=1.0)
    assert intel.get_structure("BRO1_RV40").status == StructureStatus.PREPARED

    # Destroy it
    intel.set_sim_time("H+20")
    intel.inject("BRO1_RV40", "status", "destroyed", source="Pioneer rapport", confidence=1.0)
    assert intel.get_structure("BRO1_RV40").status == StructureStatus.DESTROYED

    # Check history
    reports = intel.get_intel_history("BRO1_RV40")
    assert len(reports) == 2
    assert reports[0].timestamp == "H+16"
    assert reports[1].timestamp == "H+20"

    print("✅ test_inject_structure_status PASSED")


def test_inject_minefield_cleared():
    """Test marking a minefield as cleared."""
    intel = UND()
    intel.load_scenario()

    mf = intel.get_location("MF1_MOLNLYCKE")
    assert mf.cleared == False

    intel.set_sim_time("H+14")
    intel.inject("MF1_MOLNLYCKE", "cleared", True, source="Observed VDV clearing activity", confidence=0.8)

    assert intel.get_location("MF1_MOLNLYCKE").cleared == True

    print("✅ test_inject_minefield_cleared PASSED")


def test_inject_vehicle_losses():
    """Test updating vehicle counts after combat."""
    intel = UND()
    intel.load_scenario()

    vdv = intel.get_unit("VDV_MAIN")
    assert vdv.vehicles["BMD-4M"] == 12

    # Report 2 BMD destroyed
    intel.set_sim_time("H+18")
    intel.inject("VDV_MAIN", "vehicles", {"BMD-4M": 10}, source="RBS-17 BDA", confidence=0.9)

    assert intel.get_unit("VDV_MAIN").vehicles["BMD-4M"] == 10

    print("✅ test_inject_vehicle_losses PASSED")


def test_inject_unknown_entity():
    """Test injecting for a non-existent entity doesn't crash."""
    intel = UND()
    intel.load_scenario()

    # Should not raise, just warn
    report = intel.inject("NONEXISTENT_UNIT", "position", [57.0, 12.0], source="test", confidence=0.5)

    # Global log should still have it
    assert len(intel.global_log) == 1

    print("✅ test_inject_unknown_entity PASSED")


def test_multiple_injections_history():
    """Test that multiple injections build a proper history."""
    intel = UND()
    intel.load_scenario()

    intel.set_sim_time("H+02")
    intel.inject("VDV_BTG", "activity", "artillery preparation", source="SIGINT", confidence=0.8)

    intel.set_sim_time("H+04")
    intel.inject("VDV_BTG", "activity", "advancing", source="KJ spaning", confidence=0.9)

    intel.set_sim_time("H+10")
    intel.inject("VDV_BTG", "activity", "halted at minefield", source="UAV Raven", confidence=0.95)

    vdv = intel.get_unit("VDV_BTG")
    assert len(vdv.reports) == 3
    assert vdv.activity == "halted at minefield"  # Last high-confidence update
    assert vdv.reports[0].timestamp == "H+02"
    assert vdv.reports[2].timestamp == "H+10"

    print("✅ test_multiple_injections_history PASSED")


def test_situation_summary():
    """Test that situation summary generates valid text."""
    intel = UND()
    intel.load_scenario()

    summary = intel.get_situation_summary()
    assert "UNDERRÄTTELSELÄGE" in summary
    assert "EGNA FÖRBAND" in summary
    assert "FIENDEFÖRBAND" in summary
    assert "STRUKTURER" in summary
    assert "MINFÄLT" in summary
    assert "Amfibiebataljon" in summary
    assert "VDV" in summary
    assert "intact" in summary

    print("✅ test_situation_summary PASSED")


def test_save_and_file():
    """Test saving state to JSON file."""
    intel = UND()
    intel.load_scenario()
    intel.set_sim_time("H+06")
    intel.inject("VDV_MAIN", "position", [57.665, 12.20], source="test", confidence=0.8)

    test_file = "und_state_test.json"
    intel.save(test_file)

    assert os.path.exists(test_file)

    with open(test_file, encoding="utf-8") as f:
        data = json.load(f)

    assert data["sim_time"] == "H+06"
    assert "AMF4_BAT" in data["units"]
    assert "VDV_BTG" in data["units"]
    assert "BRO1_RV40" in data["structures"]
    assert "MF1_MOLNLYCKE" in data["locations"]
    assert len(data["global_log"]) == 1

    # Cleanup
    os.remove(test_file)

    print("✅ test_save_and_file PASSED")


def test_export_for_simulation():
    """Test export format for simulation engine."""
    intel = UND()
    intel.load_scenario()

    export = intel.export_for_simulation()
    assert "sim_time" in export
    assert "friendly_units" in export
    assert "hostile_units" in export
    assert "structures" in export
    assert "locations" in export
    assert len(export["friendly_units"]) == 6
    assert len(export["hostile_units"]) == 3
    assert len(export["structures"]) == 5

    # Check serialization doesn't crash
    json_str = json.dumps(export, ensure_ascii=False)
    assert len(json_str) > 1000

    print("✅ test_export_for_simulation PASSED")


def test_sim_time_tracking():
    """Test simulation time management."""
    intel = UND()
    intel.load_scenario()

    assert intel.sim_time == "H+00"

    intel.set_sim_time("H+04")
    assert intel.sim_time == "H+04"

    intel.inject("VDV_BTG", "activity", "advancing", source="test", confidence=0.9)
    assert intel.global_log[-1].timestamp == "H+04"

    intel.set_sim_time("H+10")
    intel.inject("VDV_BTG", "activity", "halted", source="test", confidence=0.9)
    assert intel.global_log[-1].timestamp == "H+10"

    print("✅ test_sim_time_tracking PASSED")


# ================================================================
# Run all tests
# ================================================================
if __name__ == "__main__":
    print("Running UND module tests...\n")

    test_load_scenario()
    test_get_unit()
    test_get_subunits()
    test_get_structure()
    test_get_location()
    test_inject_position()
    test_inject_low_confidence()
    test_inject_structure_status()
    test_inject_minefield_cleared()
    test_inject_vehicle_losses()
    test_inject_unknown_entity()
    test_multiple_injections_history()
    test_situation_summary()
    test_save_and_file()
    test_export_for_simulation()
    test_sim_time_tracking()

    print(f"\n{'='*50}")
    print("ALL 16 TESTS PASSED ✅")
    print(f"{'='*50}")
