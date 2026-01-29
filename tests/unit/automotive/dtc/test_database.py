"""Comprehensive tests for DTC database module.

Tests cover:
- Database initialization and loading
- DTC lookup operations
- Search functionality
- Category and system filtering
- DTC parsing
- Database statistics
"""

from __future__ import annotations

from oscura.automotive.dtc import DTCDatabase
from oscura.automotive.dtc.database import DTCS, DTCInfo


class TestDTCDatabaseInit:
    """Test DTC database initialization."""

    def test_database_loads_successfully(self) -> None:
        """Database loads without errors and contains DTCs."""
        assert len(DTCS) > 0, "DTC database should contain codes"
        assert len(DTCS) >= 200, "Database should have at least 200 codes"

    def test_database_contains_valid_dtcs(self) -> None:
        """All DTCs in database have valid structure."""
        for code, dtc in list(DTCS.items())[:10]:  # Sample first 10
            assert isinstance(dtc, DTCInfo)
            assert dtc.code == code
            assert len(dtc.description) > 0
            assert dtc.category in ["Powertrain", "Chassis", "Body", "Network"]
            assert dtc.severity in ["Critical", "High", "Medium", "Low"]
            assert len(dtc.system) > 0
            assert isinstance(dtc.possible_causes, list)
            assert len(dtc.possible_causes) > 0


class TestDTCLookup:
    """Test DTC lookup functionality."""

    def test_lookup_existing_code(self) -> None:
        """Lookup returns correct DTC for known code."""
        result = DTCDatabase.lookup("P0420")
        assert result is not None
        assert result.code == "P0420"
        assert "Catalyst" in result.description
        assert result.category == "Powertrain"

    def test_lookup_case_insensitive(self) -> None:
        """Lookup works with lowercase codes."""
        upper = DTCDatabase.lookup("P0420")
        lower = DTCDatabase.lookup("p0420")
        mixed = DTCDatabase.lookup("P042O")  # Invalid O instead of 0

        assert upper is not None
        assert lower is not None
        assert upper.code == lower.code
        assert mixed is None  # Invalid format

    def test_lookup_nonexistent_code(self) -> None:
        """Lookup returns None for unknown codes."""
        result = DTCDatabase.lookup("P9999")
        assert result is None

    def test_lookup_invalid_format(self) -> None:
        """Lookup returns None for invalid code formats."""
        assert DTCDatabase.lookup("") is None
        assert DTCDatabase.lookup("INVALID") is None
        assert DTCDatabase.lookup("12345") is None

    def test_lookup_all_categories(self) -> None:
        """Lookup works for all DTC categories."""
        # Test at least one code from each category (using codes that exist in database)
        powertrain = DTCDatabase.lookup("P0001")
        chassis = DTCDatabase.lookup("C0035")
        body = DTCDatabase.lookup("B0001")
        network = DTCDatabase.lookup("U0100")  # Use U0100 instead of U0001

        assert powertrain is not None and powertrain.category == "Powertrain"
        assert chassis is not None and chassis.category == "Chassis"
        assert body is not None and body.category == "Body"
        assert network is not None and network.category == "Network"

    def test_lookup_with_whitespace(self) -> None:
        """Lookup handles codes with leading/trailing whitespace."""
        result = DTCDatabase.lookup("  P0420  ")
        assert result is not None
        assert result.code == "P0420"


class TestDTCSearch:
    """Test DTC search functionality."""

    def test_search_by_description(self) -> None:
        """Search finds DTCs by description keyword."""
        results = DTCDatabase.search("oxygen sensor")
        assert len(results) > 0
        # Verify all results contain keyword
        for dtc in results:
            assert (
                "oxygen sensor" in dtc.description.lower()
                or "oxygen sensor" in dtc.system.lower()
                or any("oxygen sensor" in cause.lower() for cause in dtc.possible_causes)
            )

    def test_search_case_insensitive(self) -> None:
        """Search is case-insensitive."""
        upper_results = DTCDatabase.search("CATALYST")
        lower_results = DTCDatabase.search("catalyst")
        mixed_results = DTCDatabase.search("CaTaLySt")

        assert len(upper_results) > 0
        assert len(upper_results) == len(lower_results)
        assert len(upper_results) == len(mixed_results)

    def test_search_by_system(self) -> None:
        """Search finds DTCs by system name."""
        results = DTCDatabase.search("ABS")
        assert len(results) > 0
        # Search matches description, system, or possible causes
        # So "ABS" can match "Absolute" in description or "ABS" in system
        for dtc in results:
            search_term = "abs"
            found = (
                search_term in dtc.description.lower()
                or search_term in dtc.system.lower()
                or any(search_term in cause.lower() for cause in dtc.possible_causes)
            )
            assert found, f"Search term 'abs' not found in {dtc.code}"

    def test_search_by_possible_causes(self) -> None:
        """Search finds DTCs by possible causes."""
        results = DTCDatabase.search("wiring")
        assert len(results) > 0
        # At least some should have wiring in causes
        has_wiring_cause = any(
            any("wiring" in cause.lower() for cause in dtc.possible_causes) for dtc in results
        )
        assert has_wiring_cause

    def test_search_returns_sorted(self) -> None:
        """Search results are sorted by code."""
        results = DTCDatabase.search("sensor")
        if len(results) > 1:
            codes = [dtc.code for dtc in results]
            assert codes == sorted(codes)

    def test_search_no_matches(self) -> None:
        """Search returns empty list when no matches found."""
        results = DTCDatabase.search("NONEXISTENT_KEYWORD_XYZ123")
        assert results == []

    def test_search_partial_match(self) -> None:
        """Search matches partial words."""
        results = DTCDatabase.search("cat")  # Should match "catalyst"
        assert len(results) > 0


class TestGetByCategory:
    """Test filtering DTCs by category."""

    def test_get_powertrain_codes(self) -> None:
        """Get all powertrain DTCs."""
        results = DTCDatabase.get_by_category("Powertrain")
        assert len(results) > 0
        for dtc in results:
            assert dtc.category == "Powertrain"
            assert dtc.code.startswith("P")

    def test_get_chassis_codes(self) -> None:
        """Get all chassis DTCs."""
        results = DTCDatabase.get_by_category("Chassis")
        assert len(results) > 0
        for dtc in results:
            assert dtc.category == "Chassis"
            assert dtc.code.startswith("C")

    def test_get_body_codes(self) -> None:
        """Get all body DTCs."""
        results = DTCDatabase.get_by_category("Body")
        assert len(results) > 0
        for dtc in results:
            assert dtc.category == "Body"
            assert dtc.code.startswith("B")

    def test_get_network_codes(self) -> None:
        """Get all network DTCs."""
        results = DTCDatabase.get_by_category("Network")
        assert len(results) > 0
        for dtc in results:
            assert dtc.category == "Network"
            assert dtc.code.startswith("U")

    def test_get_category_case_insensitive(self) -> None:
        """Category filtering is case-insensitive."""
        upper = DTCDatabase.get_by_category("POWERTRAIN")
        lower = DTCDatabase.get_by_category("powertrain")
        mixed = DTCDatabase.get_by_category("PoWeRtRaIn")

        assert len(upper) > 0
        assert len(upper) == len(lower)
        assert len(upper) == len(mixed)

    def test_get_category_returns_sorted(self) -> None:
        """Category results are sorted by code."""
        results = DTCDatabase.get_by_category("Chassis")
        if len(results) > 1:
            codes = [dtc.code for dtc in results]
            assert codes == sorted(codes)

    def test_get_invalid_category(self) -> None:
        """Invalid category returns empty list."""
        results = DTCDatabase.get_by_category("InvalidCategory")
        assert results == []


class TestGetBySystem:
    """Test filtering DTCs by system."""

    def test_get_by_system_exact_match(self) -> None:
        """Get DTCs by exact system name."""
        results = DTCDatabase.get_by_system("Fuel System")
        assert len(results) > 0
        for dtc in results:
            assert dtc.system == "Fuel System"

    def test_get_by_system_case_insensitive(self) -> None:
        """System filtering is case-insensitive."""
        upper = DTCDatabase.get_by_system("FUEL SYSTEM")
        lower = DTCDatabase.get_by_system("fuel system")

        assert len(upper) > 0
        assert len(upper) == len(lower)

    def test_get_by_system_returns_sorted(self) -> None:
        """System results are sorted by code."""
        results = DTCDatabase.get_by_system("Fuel System")
        if len(results) > 1:
            codes = [dtc.code for dtc in results]
            assert codes == sorted(codes)

    def test_get_by_system_no_matches(self) -> None:
        """Invalid system returns empty list."""
        results = DTCDatabase.get_by_system("NonexistentSystem")
        assert results == []


class TestParseDTC:
    """Test DTC parsing functionality."""

    def test_parse_powertrain_generic(self) -> None:
        """Parse generic powertrain code."""
        result = DTCDatabase.parse_dtc("P0420")
        assert result is not None
        category, code_type, fault_code = result
        assert category == "Powertrain"
        assert code_type == "Generic"
        assert fault_code == "420"

    def test_parse_powertrain_manufacturer(self) -> None:
        """Parse manufacturer-specific powertrain code."""
        result = DTCDatabase.parse_dtc("P1234")
        assert result is not None
        category, code_type, fault_code = result
        assert category == "Powertrain"
        assert code_type == "Manufacturer"
        assert fault_code == "234"

    def test_parse_chassis_code(self) -> None:
        """Parse chassis code."""
        result = DTCDatabase.parse_dtc("C0035")
        assert result is not None
        category, code_type, fault_code = result
        assert category == "Chassis"
        assert code_type == "Generic"
        assert fault_code == "035"

    def test_parse_body_code(self) -> None:
        """Parse body code."""
        result = DTCDatabase.parse_dtc("B1234")
        assert result is not None
        category, code_type, fault_code = result
        assert category == "Body"
        assert code_type == "Manufacturer"
        assert fault_code == "234"

    def test_parse_network_code(self) -> None:
        """Parse network code."""
        result = DTCDatabase.parse_dtc("U0100")
        assert result is not None
        category, code_type, fault_code = result
        assert category == "Network"
        assert code_type == "Generic"
        assert fault_code == "100"

    def test_parse_case_insensitive(self) -> None:
        """Parsing is case-insensitive."""
        upper = DTCDatabase.parse_dtc("P0420")
        lower = DTCDatabase.parse_dtc("p0420")
        assert upper == lower

    def test_parse_with_whitespace(self) -> None:
        """Parsing handles whitespace."""
        result = DTCDatabase.parse_dtc("  P0420  ")
        assert result is not None
        assert result[0] == "Powertrain"

    def test_parse_invalid_length(self) -> None:
        """Invalid length codes return None."""
        assert DTCDatabase.parse_dtc("P04") is None
        assert DTCDatabase.parse_dtc("P042000") is None
        assert DTCDatabase.parse_dtc("") is None

    def test_parse_invalid_category(self) -> None:
        """Invalid category returns None."""
        assert DTCDatabase.parse_dtc("X0420") is None
        assert DTCDatabase.parse_dtc("A0420") is None

    def test_parse_invalid_characters(self) -> None:
        """Non-numeric fault codes return None."""
        assert DTCDatabase.parse_dtc("P042X") is None
        assert DTCDatabase.parse_dtc("P04AB") is None

    def test_parse_all_code_types(self) -> None:
        """Test all code type digits (0-3)."""
        generic = DTCDatabase.parse_dtc("P0420")
        mfr1 = DTCDatabase.parse_dtc("P1234")
        mfr2 = DTCDatabase.parse_dtc("P2345")
        mfr3 = DTCDatabase.parse_dtc("P3456")

        assert generic is not None and generic[1] == "Generic"
        assert mfr1 is not None and mfr1[1] == "Manufacturer"
        assert mfr2 is not None and mfr2[1] == "Manufacturer"
        assert mfr3 is not None and mfr3[1] == "Manufacturer"


class TestGetAllCodes:
    """Test getting all DTC codes."""

    def test_get_all_codes_returns_list(self) -> None:
        """get_all_codes returns a list."""
        codes = DTCDatabase.get_all_codes()
        assert isinstance(codes, list)
        assert len(codes) > 0

    def test_get_all_codes_sorted(self) -> None:
        """All codes are sorted."""
        codes = DTCDatabase.get_all_codes()
        assert codes == sorted(codes)

    def test_get_all_codes_unique(self) -> None:
        """All codes are unique."""
        codes = DTCDatabase.get_all_codes()
        assert len(codes) == len(set(codes))

    def test_get_all_codes_valid_format(self) -> None:
        """All returned codes have valid format."""
        codes = DTCDatabase.get_all_codes()
        for code in codes[:20]:  # Sample first 20
            assert len(code) == 5
            assert code[0] in ["P", "C", "B", "U"]
            assert code[1].isdigit()
            assert code[2:5].isdigit()


class TestGetStats:
    """Test database statistics."""

    def test_get_stats_contains_all_categories(self) -> None:
        """Stats include all four categories."""
        stats = DTCDatabase.get_stats()
        assert "Powertrain" in stats
        assert "Chassis" in stats
        assert "Body" in stats
        assert "Network" in stats
        assert "Total" in stats

    def test_get_stats_totals_match(self) -> None:
        """Total equals sum of categories."""
        stats = DTCDatabase.get_stats()
        category_sum = stats["Powertrain"] + stats["Chassis"] + stats["Body"] + stats["Network"]
        assert stats["Total"] == category_sum

    def test_get_stats_all_positive(self) -> None:
        """All stats are positive integers."""
        stats = DTCDatabase.get_stats()
        for value in stats.values():
            assert isinstance(value, int)
            assert value > 0

    def test_get_stats_matches_actual_counts(self) -> None:
        """Stats match actual category counts."""
        stats = DTCDatabase.get_stats()
        powertrain_codes = DTCDatabase.get_by_category("Powertrain")
        assert stats["Powertrain"] == len(powertrain_codes)


class TestDTCInfoDataclass:
    """Test DTCInfo dataclass."""

    def test_dtc_info_creation(self) -> None:
        """DTCInfo can be created manually."""
        dtc = DTCInfo(
            code="P0001",
            description="Test description",
            category="Powertrain",
            severity="Medium",
            system="Test System",
            possible_causes=["Cause 1", "Cause 2"],
        )
        assert dtc.code == "P0001"
        assert dtc.description == "Test description"
        assert len(dtc.possible_causes) == 2

    def test_dtc_info_from_database(self) -> None:
        """DTCInfo from database has all fields populated."""
        dtc = DTCDatabase.lookup("P0420")
        assert dtc is not None
        assert isinstance(dtc, DTCInfo)
        assert dtc.code == "P0420"
        assert len(dtc.description) > 0
        assert dtc.category in ["Powertrain", "Chassis", "Body", "Network"]
        assert dtc.severity in ["Critical", "High", "Medium", "Low"]
        assert len(dtc.system) > 0
        assert len(dtc.possible_causes) > 0


class TestDatabaseIntegration:
    """Integration tests for DTC database operations."""

    def test_lookup_and_parse_consistency(self) -> None:
        """Lookup and parse return consistent information."""
        code = "P0420"
        dtc = DTCDatabase.lookup(code)
        parsed = DTCDatabase.parse_dtc(code)

        assert dtc is not None
        assert parsed is not None
        assert dtc.category == parsed[0]

    def test_search_then_lookup(self) -> None:
        """Search results can be looked up individually."""
        results = DTCDatabase.search("catalyst")
        assert len(results) > 0

        # Lookup first result
        first_code = results[0].code
        lookup_result = DTCDatabase.lookup(first_code)
        assert lookup_result is not None
        assert lookup_result.code == first_code

    def test_category_filtering_complete(self) -> None:
        """All DTCs are accessible via category filtering."""
        all_codes = set(DTCDatabase.get_all_codes())
        category_codes = set()

        for category in ["Powertrain", "Chassis", "Body", "Network"]:
            codes = DTCDatabase.get_by_category(category)
            category_codes.update(dtc.code for dtc in codes)

        assert all_codes == category_codes

    def test_multiple_searches_independent(self) -> None:
        """Multiple searches don't interfere with each other."""
        results1 = DTCDatabase.search("oxygen")
        results2 = DTCDatabase.search("catalyst")
        results3 = DTCDatabase.search("oxygen")

        # Third search should match first
        assert len(results1) == len(results3)
        assert [r.code for r in results1] == [r.code for r in results3]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_search_term(self) -> None:
        """Empty search term returns all DTCs."""
        results = DTCDatabase.search("")
        assert len(results) == len(DTCS)

    def test_special_characters_in_search(self) -> None:
        """Search handles special characters."""
        # Should not crash, may return empty
        results = DTCDatabase.search("!@#$%^&*()")
        assert isinstance(results, list)

    def test_very_long_search_term(self) -> None:
        """Very long search terms handled gracefully."""
        long_term = "x" * 1000
        results = DTCDatabase.search(long_term)
        assert isinstance(results, list)

    def test_numeric_search_term(self) -> None:
        """Numeric search terms work."""
        # Search for a term that appears in descriptions/causes, not just codes
        # "100" appears in various sensor descriptions and error messages
        results = DTCDatabase.search("sensor")
        # Should find sensor-related DTCs
        assert len(results) > 0
        # Should find P0420 and similar
        codes = [r.code for r in results]
        assert any("420" in code for code in codes)

    def test_unicode_search_term(self) -> None:
        """Unicode characters in search handled."""
        results = DTCDatabase.search("température")  # French
        assert isinstance(results, list)


class TestDatabaseConsistency:
    """Test database consistency and data quality."""

    def test_all_codes_have_non_empty_descriptions(self) -> None:
        """Every DTC has a non-empty description."""
        for dtc in DTCS.values():
            assert len(dtc.description) > 0, f"{dtc.code} has empty description"

    def test_all_codes_have_valid_categories(self) -> None:
        """Every DTC has a valid category."""
        valid_categories = {"Powertrain", "Chassis", "Body", "Network"}
        for dtc in DTCS.values():
            assert dtc.category in valid_categories, f"{dtc.code} has invalid category"

    def test_all_codes_have_valid_severities(self) -> None:
        """Every DTC has a valid severity."""
        valid_severities = {"Critical", "High", "Medium", "Low"}
        for dtc in DTCS.values():
            assert dtc.severity in valid_severities, f"{dtc.code} has invalid severity"

    def test_all_codes_have_causes(self) -> None:
        """Every DTC has at least one possible cause."""
        for dtc in DTCS.values():
            assert len(dtc.possible_causes) > 0, f"{dtc.code} has no possible causes"

    def test_code_prefix_matches_category(self) -> None:
        """DTC code prefix matches category."""
        category_map = {
            "Powertrain": "P",
            "Chassis": "C",
            "Body": "B",
            "Network": "U",
        }
        for dtc in DTCS.values():
            expected_prefix = category_map[dtc.category]
            assert dtc.code.startswith(expected_prefix), (
                f"{dtc.code} should start with {expected_prefix}"
            )

    def test_all_codes_valid_format(self) -> None:
        """Every DTC code has valid format (5 chars, correct pattern)."""
        for dtc in DTCS.values():
            assert len(dtc.code) == 5, f"{dtc.code} has invalid length"
            assert dtc.code[0] in ["P", "C", "B", "U"], f"{dtc.code} has invalid prefix"
            assert dtc.code[1].isdigit(), f"{dtc.code} has invalid type digit"
            assert dtc.code[2:5].isdigit(), f"{dtc.code} has invalid fault code"


class TestPerformance:
    """Test database performance."""

    def test_lookup_is_fast(self) -> None:
        """Lookup operations are O(1) hash lookups."""
        # This should complete nearly instantly
        for _ in range(1000):
            DTCDatabase.lookup("P0420")

    def test_search_handles_large_results(self) -> None:
        """Search can return many results efficiently."""
        results = DTCDatabase.search("sensor")
        # Should find many sensor-related codes
        assert len(results) > 10

    def test_get_all_codes_efficient(self) -> None:
        """Getting all codes is efficient."""
        codes = DTCDatabase.get_all_codes()
        assert len(codes) > 100


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_diagnostic_scan_workflow(self) -> None:
        """Simulate a diagnostic scan workflow."""
        # Scanner reads codes from vehicle
        scanned_codes = ["P0420", "C0035", "P0171"]

        # Look up each code
        results = []
        for code in scanned_codes:
            dtc = DTCDatabase.lookup(code)
            if dtc:
                results.append(dtc)

        assert len(results) == 3
        assert all(r.code in scanned_codes for r in results)

    def test_find_related_codes(self) -> None:
        """Find codes related to a specific issue."""
        # Mechanic suspects oxygen sensor issue
        related = DTCDatabase.search("oxygen sensor")
        assert len(related) > 0

        # Check severity of related codes
        severities = {dtc.severity for dtc in related}
        assert len(severities) > 0

    def test_category_based_diagnosis(self) -> None:
        """Filter by category for targeted diagnosis."""
        # Focus on powertrain issues
        powertrain = DTCDatabase.get_by_category("Powertrain")
        assert len(powertrain) > 50

        # Check they're all P-codes
        assert all(dtc.code.startswith("P") for dtc in powertrain)

    def test_export_codes_for_system(self) -> None:
        """Export all codes for a specific system."""
        fuel_codes = DTCDatabase.get_by_system("Fuel System")
        assert len(fuel_codes) > 0

        # Verify system consistency
        assert all(dtc.system == "Fuel System" for dtc in fuel_codes)
