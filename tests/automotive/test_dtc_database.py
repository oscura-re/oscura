"""Tests for DTC database functionality.

Tests cover lookup, search, categorization, and parsing of diagnostic trouble codes.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.dtc import DTCDatabase


class TestDTCLookup:
    """Test DTC lookup functionality."""

    def test_lookup_valid_powertrain_code(self) -> None:
        """Test looking up a valid powertrain DTC."""
        info = DTCDatabase.lookup("P0420")
        assert info is not None
        assert info.code == "P0420"
        assert info.category == "Powertrain"
        assert "Catalyst" in info.description
        assert info.severity == "Medium"
        assert info.system == "Emissions Control"
        assert len(info.possible_causes) > 0

    def test_lookup_chassis_code(self) -> None:
        """Test looking up a chassis code."""
        info = DTCDatabase.lookup("C0035")
        assert info is not None
        assert info.code == "C0035"
        assert info.category == "Chassis"
        assert "wheel speed" in info.description.lower()
        assert info.system == "ABS"
        assert len(info.possible_causes) > 0

    def test_lookup_body_code(self) -> None:
        """Test looking up a body code."""
        info = DTCDatabase.lookup("B0001")
        assert info is not None
        assert info.code == "B0001"
        assert info.category == "Body"
        assert "airbag" in info.description.lower()
        assert len(info.possible_causes) > 0

    def test_lookup_network_code(self) -> None:
        """Test looking up network/communication codes."""
        info = DTCDatabase.lookup("U0100")
        assert info is not None
        assert info.code == "U0100"
        assert info.category == "Network"
        assert "ECM" in info.description or "PCM" in info.description
        assert len(info.possible_causes) > 0

    def test_lookup_case_insensitive(self) -> None:
        """Test that lookup is case-insensitive."""
        # Test different case variations
        info_upper = DTCDatabase.lookup("P0420")
        info_lower = DTCDatabase.lookup("p0420")
        info_mixed = DTCDatabase.lookup("P0420")

        assert info_upper is not None
        assert info_lower is not None
        assert info_mixed is not None

        # All should return the same code
        assert info_upper.code == info_lower.code == info_mixed.code == "P0420"

    def test_lookup_invalid_code(self):
        """Test lookup with invalid or non-existent codes."""
        # Non-existent code
        result = DTCDatabase.lookup("P9999")
        assert result is None

        # Invalid format
        result = DTCDatabase.lookup("INVALID")
        assert result is None

    def test_search_by_keyword(self) -> None:
        """Test searching DTCs by keyword."""
        # Search in description
        results = DTCDatabase.search("oxygen")
        assert len(results) > 0
        # Verify at least some results mention oxygen or O2
        assert any(
            "oxygen" in dtc.description.lower() or "o2" in dtc.description.lower()
            for dtc in results
        )

        # Search in system
        results = DTCDatabase.search("ABS")
        assert len(results) > 0
        for dtc in results:
            assert "ABS" in dtc.system or "abs" in dtc.description.lower()

        # Search in possible causes
        results = DTCDatabase.search("wiring")
        assert len(results) > 0
        # Verify at least some results mention wiring in possible causes
        assert any(
            any("wiring" in cause.lower() for cause in dtc.possible_causes) for dtc in results
        )

    def test_search_case_insensitive(self) -> None:
        """Test that search is case-insensitive."""
        results_lower = DTCDatabase.search("oxygen sensor")
        results_upper = DTCDatabase.search("OXYGEN SENSOR")
        results_mixed = DTCDatabase.search("Oxygen Sensor")

        assert len(results_lower) > 0
        assert len(results_lower) == len(results_upper)
        assert len(results_lower) == len(results_mixed)

    def test_search_no_results(self) -> None:
        """Test search with no matches."""
        results = DTCDatabase.search("nonexistent_system_xyz")
        assert results == []

    def test_get_by_category_powertrain(self) -> None:
        """Test getting all powertrain codes."""
        powertrain = DTCDatabase.get_by_category("Powertrain")
        assert len(powertrain) >= 100, f"Expected 100+ powertrain codes, got {len(powertrain)}"

        # Verify all are powertrain codes
        for dtc in powertrain:
            assert dtc.category == "Powertrain"
            assert dtc.code.startswith("P")

    def test_get_by_category_chassis(self) -> None:
        """Test getting chassis codes."""
        chassis = DTCDatabase.get_by_category("Chassis")
        assert len(chassis) >= 40, f"Expected 40+ chassis codes, got {len(chassis)}"

        # Verify all are chassis codes
        for dtc in chassis:
            assert dtc.category == "Chassis"
            assert dtc.code.startswith("C")

    def test_get_by_category_body(self) -> None:
        """Test getting body codes."""
        body = DTCDatabase.get_by_category("Body")
        assert len(body) >= 40, f"Expected 40+ body codes, got {len(body)}"

        # All should be Body category
        for dtc in body:
            assert dtc.category == "Body"
            assert dtc.code.startswith("B")

    def test_get_by_category_network(self) -> None:
        """Test getting network/communication codes."""
        network = DTCDatabase.get_by_category("Network")

        assert len(network) >= 20, f"Expected at least 20 network codes, got {len(network)}"

        # All should be Network category
        for dtc in network:
            assert dtc.category == "Network"
            assert dtc.code.startswith("U")

    def test_get_by_category_case_insensitive(self) -> None:
        """Test category lookup is case-insensitive."""
        lower = DTCDatabase.get_by_category("powertrain")
        upper = DTCDatabase.get_by_category("POWERTRAIN")
        mixed = DTCDatabase.get_by_category("PoWeRtRaIn")

        assert len(lower) == len(upper) == len(mixed) > 0

    def test_get_by_system(self) -> None:
        """Test getting DTCs by system."""
        # Test ABS system
        abs_codes = DTCDatabase.get_by_system("ABS")
        assert len(abs_codes) > 0
        assert all(dtc.system == "ABS" for dtc in abs_codes)

        # Test Oxygen Sensors
        o2_codes = DTCDatabase.get_by_system("Oxygen Sensors")
        assert len(o2_codes) > 0
        assert all(dtc.system == "Oxygen Sensors" for dtc in o2_codes)

        # Test case insensitivity
        abs_lower = DTCDatabase.get_by_system("abs")
        abs_upper = DTCDatabase.get_by_system("ABS")
        assert len(abs_lower) == len(abs_upper)

    def test_parse_dtc_valid_codes(self):
        """Test parsing valid DTC codes."""
        # Generic powertrain code
        result = DTCDatabase.parse_dtc("P0420")
        assert result == ("Powertrain", "Generic", "420")

        # Manufacturer specific chassis code
        result = DTCDatabase.parse_dtc("C1234")
        assert result == ("Chassis", "Manufacturer", "234")

        # Generic body code
        result = DTCDatabase.parse_dtc("B0001")
        assert result == ("Body", "Generic", "001")

        # Generic network code
        result = DTCDatabase.parse_dtc("U0100")
        assert result == ("Network", "Generic", "100")

        # Case insensitivity
        result = DTCDatabase.parse_dtc("p0420")
        assert result == ("Powertrain", "Generic", "420")

    def test_parse_dtc_invalid_codes(self):
        """Test parsing invalid DTC codes."""
        # Wrong length
        assert DTCDatabase.parse_dtc("P042") is None
        assert DTCDatabase.parse_dtc("P04200") is None

        # Invalid category
        assert DTCDatabase.parse_dtc("X0420") is None

        # Non-numeric digits
        assert DTCDatabase.parse_dtc("P042A") is None
        assert DTCDatabase.parse_dtc("PA420") is None

        # Empty string
        assert DTCDatabase.parse_dtc("") is None

    def test_get_all_codes(self):
        """Test getting all codes from database."""
        all_codes = DTCDatabase.get_all_codes()

        # Should have 200+ codes
        assert len(all_codes) >= 200

        # Should be sorted
        assert all_codes == sorted(all_codes)

        # Should contain specific codes
        assert "P0420" in all_codes
        assert "C0035" in all_codes
        assert "B0001" in all_codes
        assert "U0100" in all_codes

    def test_get_stats(self):
        """Test database statistics."""
        stats = DTCDatabase.get_stats()

        # Should have all categories
        assert "Powertrain" in stats
        assert "Chassis" in stats
        assert "Body" in stats
        assert "Network" in stats
        assert "Total" in stats

        # Total should be sum of categories
        assert stats["Total"] == (
            stats["Powertrain"] + stats["Chassis"] + stats["Body"] + stats["Network"]
        )

        # Should have 200+ total codes
        assert stats["Total"] >= 200

        # Check approximate distribution (based on requirements)
        assert stats["Powertrain"] >= 100
        assert stats["Chassis"] >= 40
        assert stats["Body"] >= 40
        assert stats["Network"] >= 20

    def test_dtc_info_dataclass(self):
        """Test DTCInfo dataclass structure."""
        info = DTCDatabase.lookup("P0420")
        assert info is not None

        # Check all required fields exist
        assert hasattr(info, "code")
        assert hasattr(info, "description")
        assert hasattr(info, "category")
        assert hasattr(info, "severity")
        assert hasattr(info, "system")
        assert hasattr(info, "possible_causes")

        # Check field types
        assert isinstance(info.code, str)
        assert isinstance(info.description, str)
        assert isinstance(info.category, str)
        assert isinstance(info.severity, str)
        assert isinstance(info.system, str)
        assert isinstance(info.possible_causes, list)
        assert all(isinstance(cause, str) for cause in info.possible_causes)

    def test_severity_levels(self):
        """Test that all DTCs have valid severity levels."""
        valid_severities = {"Critical", "High", "Medium", "Low"}

        all_codes = DTCDatabase.get_all_codes()
        for code in all_codes:
            info = DTCDatabase.lookup(code)
            assert info.severity in valid_severities

    def test_categories(self):
        """Test that all DTCs have valid categories."""
        valid_categories = {"Powertrain", "Chassis", "Body", "Network"}

        all_codes = DTCDatabase.get_all_codes()
        for code in all_codes:
            info = DTCDatabase.lookup(code)
            assert info.category in valid_categories

    def test_code_format(self):
        """Test that all codes follow proper DTC format."""
        all_codes = DTCDatabase.get_all_codes()

        for code in all_codes:
            # Should be 5 characters
            assert len(code) == 5

            # First character should be P, C, B, or U
            assert code[0] in ("P", "C", "B", "U")

            # Remaining characters should be numeric
            assert code[1:].isdigit()

    def test_common_powertrain_codes(self):
        """Test lookup of common powertrain codes."""
        common_codes = [
            "P0300",  # Random misfire
            "P0420",  # Catalyst efficiency
            "P0171",  # System too lean
            "P0101",  # MAF circuit
            "P0335",  # Crank sensor
        ]

        for code in common_codes:
            info = DTCDatabase.lookup(code)
            assert info is not None
            assert info.category == "Powertrain"
            assert len(info.possible_causes) > 0

    def test_common_chassis_codes(self):
        """Test lookup of common chassis codes."""
        common_codes = [
            "C0035",  # Left front wheel speed
            "C0040",  # Right front wheel speed
            "C0060",  # ABS pump motor
            "C0200",  # Steering angle sensor
        ]

        for code in common_codes:
            info = DTCDatabase.lookup(code)
            assert info is not None
            assert info.category == "Chassis"
            assert len(info.possible_causes) > 0

    def test_common_body_codes(self):
        """Test lookup of common body codes."""
        common_codes = [
            "B0001",  # Driver airbag
            "B0002",  # Passenger airbag
            "B1000",  # BCM malfunction
            "B1300",  # TPMS malfunction
        ]

        for code in common_codes:
            info = DTCDatabase.lookup(code)
            assert info is not None
            assert info.category == "Body"
            assert len(info.possible_causes) > 0

    def test_common_network_codes(self):
        """Test lookup of common network codes."""
        common_codes = [
            "U0100",  # Lost comm with ECM
            "U0101",  # Lost comm with TCM
            "U0121",  # Lost comm with ABS
            "U0155",  # Lost comm with cluster
        ]

        for code in common_codes:
            info = DTCDatabase.lookup(code)
            assert info is not None
            assert info.category == "Network"
            assert len(info.possible_causes) > 0

    def test_possible_causes_not_empty(self):
        """Test that all DTCs have at least one possible cause."""
        all_codes = DTCDatabase.get_all_codes()

        for code in all_codes:
            info = DTCDatabase.lookup(code)
            assert len(info.possible_causes) > 0
            assert all(len(cause.strip()) > 0 for cause in info.possible_causes)
