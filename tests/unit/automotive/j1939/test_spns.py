"""Tests for J1939 SPN definitions."""

from __future__ import annotations

from oscura.automotive.j1939.spns import STANDARD_SPNS, get_standard_spns


class TestStandardSPNs:
    """Test standard SPN definitions."""

    def test_get_standard_spns(self):
        """Test retrieving standard SPN definitions."""
        spns = get_standard_spns()

        assert isinstance(spns, dict)
        assert len(spns) > 0

    def test_standard_spns_contains_eec1(self):
        """Test EEC1 (PGN 61444) SPNs present."""
        spns = get_standard_spns()

        assert 61444 in spns
        assert len(spns[61444]) > 0

    def test_engine_speed_spn(self):
        """Test Engine Speed SPN definition."""
        spns = get_standard_spns()

        engine_speed = next((s for s in spns[61444] if s.spn == 190), None)

        assert engine_speed is not None
        assert engine_speed.name == "Engine Speed"
        assert engine_speed.start_bit == 24
        assert engine_speed.bit_length == 16
        assert engine_speed.resolution == 0.125
        assert engine_speed.unit == "rpm"

    def test_eec2_spns(self):
        """Test EEC2 (PGN 61443) SPNs."""
        spns = get_standard_spns()

        assert 61443 in spns
        # Check for accelerator pedal position
        accel = next((s for s in spns[61443] if s.spn == 91), None)
        assert accel is not None
        assert accel.name == "Accelerator Pedal Position 1"

    def test_ccvs1_spns(self):
        """Test CCVS1 (PGN 65265) SPNs."""
        spns = get_standard_spns()

        assert 65265 in spns
        # Check for vehicle speed
        speed = next((s for s in spns[65265] if s.spn == 84), None)
        assert speed is not None
        assert speed.name == "Wheel-Based Vehicle Speed"
        assert speed.unit == "km/h"

    def test_dm1_spns(self):
        """Test DM1 (PGN 65226) SPNs."""
        spns = get_standard_spns()

        assert 65226 in spns
        # Check for MIL status
        mil = next((s for s in spns[65226] if s.spn == 1213), None)
        assert mil is not None
        assert mil.name == "Malfunction Indicator Lamp Status"

    def test_spn_has_required_fields(self):
        """Test all SPNs have required fields."""
        spns = get_standard_spns()

        for spn_list in spns.values():
            for spn in spn_list:
                assert hasattr(spn, "spn")
                assert hasattr(spn, "name")
                assert hasattr(spn, "start_bit")
                assert hasattr(spn, "bit_length")
                assert spn.spn > 0
                assert len(spn.name) > 0
                assert spn.start_bit >= 0
                assert spn.bit_length > 0

    def test_standard_spns_immutable(self):
        """Test get_standard_spns returns a copy."""
        spns1 = get_standard_spns()
        spns2 = get_standard_spns()

        # Modifying one should not affect the other
        spns1[99999] = []

        assert 99999 not in spns2
        assert 99999 not in STANDARD_SPNS
