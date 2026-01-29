"""Tests for verbosity level control in reports.

Tests for:

This module comprehensively tests the VerbosityLevel enum, VerbosityController class,
and apply_verbosity_level function to ensure correct report filtering and configuration.
"""

from __future__ import annotations

import pytest

from oscura.reporting.content.verbosity import (
    VerbosityController,
    VerbosityLevel,
    apply_verbosity_level,
)
from oscura.reporting.core import Report, ReportConfig, Section

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestVerbosityLevel:
    """Test VerbosityLevel enum functionality."""

    def test_verbosity_levels_exist(self):
        """Test that all five verbosity levels are defined."""
        assert VerbosityLevel.EXECUTIVE.value == "executive"
        assert VerbosityLevel.SUMMARY.value == "summary"
        assert VerbosityLevel.STANDARD.value == "standard"
        assert VerbosityLevel.DETAILED.value == "detailed"
        assert VerbosityLevel.DEBUG.value == "debug"

    def test_verbosity_level_count(self):
        """Test that exactly 5 verbosity levels are defined."""
        assert len(VerbosityLevel) == 5

    def test_verbosity_level_from_string(self):
        """Test creating VerbosityLevel from string."""
        assert VerbosityLevel("executive") == VerbosityLevel.EXECUTIVE
        assert VerbosityLevel("summary") == VerbosityLevel.SUMMARY
        assert VerbosityLevel("standard") == VerbosityLevel.STANDARD
        assert VerbosityLevel("detailed") == VerbosityLevel.DETAILED
        assert VerbosityLevel("debug") == VerbosityLevel.DEBUG

    def test_verbosity_level_invalid_string(self):
        """Test that invalid string raises ValueError."""
        with pytest.raises(ValueError):
            VerbosityLevel("invalid")

    def test_verbosity_level_case_sensitive(self):
        """Test that verbosity level strings are case-sensitive."""
        with pytest.raises(ValueError):
            VerbosityLevel("EXECUTIVE")


@pytest.mark.unit
class TestVerbosityController:
    """Test VerbosityController class functionality."""

    def test_default_level(self):
        """Test that default verbosity level is STANDARD."""
        controller = VerbosityController()
        assert controller.level == VerbosityLevel.STANDARD

    def test_custom_level(self):
        """Test creating controller with custom level."""
        controller = VerbosityController(level=VerbosityLevel.EXECUTIVE)
        assert controller.level == VerbosityLevel.EXECUTIVE

        controller = VerbosityController(level=VerbosityLevel.DEBUG)
        assert controller.level == VerbosityLevel.DEBUG


@pytest.mark.unit
class TestVerbosityControllerSectionInclusion:
    """Test should_include_section method for all verbosity levels."""

    def test_executive_level_sections(self):
        """Test section inclusion for EXECUTIVE level (1 page, pass/fail + key findings)."""
        controller = VerbosityController(level=VerbosityLevel.EXECUTIVE)

        # Included sections
        assert controller.should_include_section("executive_summary") is True
        assert controller.should_include_section("key_findings") is True

        # Excluded sections
        assert controller.should_include_section("summary") is False
        assert controller.should_include_section("results") is False
        assert controller.should_include_section("methodology") is False
        assert controller.should_include_section("plots") is False
        assert controller.should_include_section("tables") is False
        assert controller.should_include_section("measurements") is False
        assert controller.should_include_section("raw_data") is False
        assert controller.should_include_section("logs") is False

    def test_summary_level_sections(self):
        """Test section inclusion for SUMMARY level (2-5 pages, results + brief context)."""
        controller = VerbosityController(level=VerbosityLevel.SUMMARY)

        # Included sections
        assert controller.should_include_section("summary") is True
        assert controller.should_include_section("results") is True
        assert controller.should_include_section("key_plots") is True

        # Excluded sections
        assert controller.should_include_section("executive_summary") is False
        assert controller.should_include_section("methodology") is False
        assert controller.should_include_section("plots") is False
        assert controller.should_include_section("tables") is False
        assert controller.should_include_section("measurements") is False
        assert controller.should_include_section("raw_data") is False
        assert controller.should_include_section("logs") is False

    def test_standard_level_sections(self):
        """Test section inclusion for STANDARD level (5-20 pages, full results + methodology)."""
        controller = VerbosityController(level=VerbosityLevel.STANDARD)

        # Included sections
        assert controller.should_include_section("summary") is True
        assert controller.should_include_section("results") is True
        assert controller.should_include_section("methodology") is True
        assert controller.should_include_section("plots") is True
        assert controller.should_include_section("tables") is True

        # Excluded sections
        assert controller.should_include_section("executive_summary") is False
        assert controller.should_include_section("key_findings") is False
        assert controller.should_include_section("measurements") is False
        assert controller.should_include_section("intermediate_results") is False
        assert controller.should_include_section("raw_data") is False
        assert controller.should_include_section("logs") is False
        assert controller.should_include_section("provenance") is False

    def test_detailed_level_sections(self):
        """Test section inclusion for DETAILED level (20-50 pages, all measurements)."""
        controller = VerbosityController(level=VerbosityLevel.DETAILED)

        # Included sections
        assert controller.should_include_section("summary") is True
        assert controller.should_include_section("results") is True
        assert controller.should_include_section("methodology") is True
        assert controller.should_include_section("plots") is True
        assert controller.should_include_section("tables") is True
        assert controller.should_include_section("measurements") is True
        assert controller.should_include_section("intermediate_results") is True

        # Excluded sections
        assert controller.should_include_section("executive_summary") is False
        assert controller.should_include_section("key_findings") is False
        assert controller.should_include_section("key_plots") is False
        assert controller.should_include_section("raw_data") is False
        assert controller.should_include_section("logs") is False
        assert controller.should_include_section("provenance") is False

    def test_debug_level_sections(self):
        """Test section inclusion for DEBUG level (50+ pages, raw data + traces)."""
        controller = VerbosityController(level=VerbosityLevel.DEBUG)

        # Included sections - DEBUG includes everything
        assert controller.should_include_section("summary") is True
        assert controller.should_include_section("results") is True
        assert controller.should_include_section("methodology") is True
        assert controller.should_include_section("plots") is True
        assert controller.should_include_section("tables") is True
        assert controller.should_include_section("measurements") is True
        assert controller.should_include_section("intermediate_results") is True
        assert controller.should_include_section("raw_data") is True
        assert controller.should_include_section("logs") is True
        assert controller.should_include_section("provenance") is True

        # Still excluded sections (executive-specific)
        assert controller.should_include_section("executive_summary") is False
        assert controller.should_include_section("key_findings") is False

    def test_unknown_section_excluded(self):
        """Test that unknown section names are excluded at all levels."""
        for level in VerbosityLevel:
            controller = VerbosityController(level=level)
            assert controller.should_include_section("unknown_section") is False
            assert controller.should_include_section("random_content") is False
            assert controller.should_include_section("") is False

    def test_section_name_case_sensitivity(self):
        """Test that section names are case-sensitive."""
        controller = VerbosityController(level=VerbosityLevel.STANDARD)

        # Lowercase should match
        assert controller.should_include_section("summary") is True

        # Different cases should not match
        assert controller.should_include_section("Summary") is False
        assert controller.should_include_section("SUMMARY") is False


@pytest.mark.unit
class TestVerbosityControllerMaxPages:
    """Test get_max_pages method for all verbosity levels."""

    def test_executive_max_pages(self):
        """Test max pages for EXECUTIVE level (1 page)."""
        controller = VerbosityController(level=VerbosityLevel.EXECUTIVE)
        assert controller.get_max_pages() == 1

    def test_summary_max_pages(self):
        """Test max pages for SUMMARY level (5 pages)."""
        controller = VerbosityController(level=VerbosityLevel.SUMMARY)
        assert controller.get_max_pages() == 5

    def test_standard_max_pages(self):
        """Test max pages for STANDARD level (20 pages)."""
        controller = VerbosityController(level=VerbosityLevel.STANDARD)
        assert controller.get_max_pages() == 20

    def test_detailed_max_pages(self):
        """Test max pages for DETAILED level (50 pages)."""
        controller = VerbosityController(level=VerbosityLevel.DETAILED)
        assert controller.get_max_pages() == 50

    def test_debug_max_pages(self):
        """Test max pages for DEBUG level (999 pages, effectively unlimited)."""
        controller = VerbosityController(level=VerbosityLevel.DEBUG)
        assert controller.get_max_pages() == 999

    def test_max_pages_progression(self):
        """Test that max pages increase with verbosity level."""
        levels = [
            VerbosityLevel.EXECUTIVE,
            VerbosityLevel.SUMMARY,
            VerbosityLevel.STANDARD,
            VerbosityLevel.DETAILED,
            VerbosityLevel.DEBUG,
        ]

        max_pages = [VerbosityController(level=level).get_max_pages() for level in levels]

        # Each level should have more pages than the previous
        for i in range(len(max_pages) - 1):
            assert max_pages[i] < max_pages[i + 1]


@pytest.mark.unit
class TestApplyVerbosityLevel:
    """Test apply_verbosity_level function."""

    def test_apply_with_enum_level(self):
        """Test applying verbosity level using VerbosityLevel enum."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("summary", "Summary content")
        report.add_section("results", "Results content")
        report.add_section("raw_data", "Raw data content")

        apply_verbosity_level(report, VerbosityLevel.STANDARD)

        # Check that config was updated
        assert report.config.verbosity == "standard"

        # Check that appropriate sections are included
        section_titles = [s.title for s in report.sections]
        assert "summary" in section_titles
        assert "results" in section_titles
        # raw_data should be filtered out for STANDARD level
        assert "raw_data" not in section_titles

    def test_apply_with_string_level_lowercase(self):
        """Test applying verbosity level using lowercase string."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("summary", "Summary content")
        report.add_section("methodology", "Methodology content")

        apply_verbosity_level(report, "summary")

        assert report.config.verbosity == "summary"
        section_titles = [s.title for s in report.sections]
        assert "summary" in section_titles
        # methodology is not in SUMMARY level
        assert "methodology" not in section_titles

    def test_apply_with_string_level_mixed_case(self):
        """Test that string level is case-insensitive (converted to lowercase)."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("summary", "Summary content")

        apply_verbosity_level(report, "Standard")

        assert report.config.verbosity == "standard"

    def test_apply_executive_level_filters_correctly(self):
        """Test that EXECUTIVE level only keeps executive sections."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("executive_summary", "Executive summary")
        report.add_section("key_findings", "Key findings")
        report.add_section("summary", "Summary")
        report.add_section("results", "Results")
        report.add_section("methodology", "Methodology")

        apply_verbosity_level(report, VerbosityLevel.EXECUTIVE)

        section_titles = [s.title for s in report.sections]
        assert len(section_titles) == 2
        assert "executive_summary" in section_titles
        assert "key_findings" in section_titles
        assert "summary" not in section_titles
        assert "results" not in section_titles
        assert "methodology" not in section_titles

    def test_apply_debug_level_keeps_all_sections(self):
        """Test that DEBUG level keeps all applicable sections."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("summary", "Summary")
        report.add_section("results", "Results")
        report.add_section("methodology", "Methodology")
        report.add_section("plots", "Plots")
        report.add_section("tables", "Tables")
        report.add_section("measurements", "Measurements")
        report.add_section("raw_data", "Raw data")
        report.add_section("logs", "Logs")
        report.add_section("provenance", "Provenance")

        apply_verbosity_level(report, VerbosityLevel.DEBUG)

        section_titles = [s.title for s in report.sections]
        assert len(section_titles) == 9
        assert "summary" in section_titles
        assert "results" in section_titles
        assert "raw_data" in section_titles
        assert "logs" in section_titles
        assert "provenance" in section_titles

    def test_apply_removes_unknown_sections(self):
        """Test that sections not in verbosity level are removed."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("summary", "Summary")
        report.add_section("unknown_section", "Unknown")
        report.add_section("random_content", "Random")

        apply_verbosity_level(report, VerbosityLevel.STANDARD)

        section_titles = [s.title for s in report.sections]
        assert "summary" in section_titles
        assert "unknown_section" not in section_titles
        assert "random_content" not in section_titles

    def test_apply_preserves_section_order(self):
        """Test that section order is preserved after filtering."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("summary", "Summary content")
        report.add_section("results", "Results content")
        report.add_section("plots", "Plots content")

        apply_verbosity_level(report, VerbosityLevel.STANDARD)

        section_titles = [s.title for s in report.sections]
        assert section_titles == ["summary", "results", "plots"]

    def test_apply_with_empty_sections(self):
        """Test applying verbosity level to report with no sections."""
        config = ReportConfig()
        report = Report(config=config)

        apply_verbosity_level(report, VerbosityLevel.STANDARD)

        assert report.config.verbosity == "standard"
        assert len(report.sections) == 0

    def test_apply_updates_config_verbosity(self):
        """Test that config.verbosity is updated to the string value."""
        config = ReportConfig(verbosity="debug")
        report = Report(config=config)

        apply_verbosity_level(report, VerbosityLevel.EXECUTIVE)

        assert report.config.verbosity == "executive"

    def test_apply_to_report_without_sections_attribute(self):
        """Test applying verbosity to report-like object without sections attribute."""

        # Create a minimal mock object with just config
        class MinimalReport:
            def __init__(self):
                self.config = ReportConfig()

        report = MinimalReport()
        apply_verbosity_level(report, VerbosityLevel.STANDARD)  # type: ignore[arg-type]

        # Should update config without error
        assert report.config.verbosity == "standard"

    def test_apply_to_report_without_config_attribute(self):
        """Test applying verbosity to report-like object without config attribute."""

        # Create a minimal mock object with just sections
        class MinimalReport:
            def __init__(self):
                self.sections = [Section(title="summary", content="content")]

        report = MinimalReport()
        apply_verbosity_level(report, VerbosityLevel.STANDARD)  # type: ignore[arg-type]

        # Should filter sections without error
        assert len(report.sections) == 1

    def test_multiple_verbosity_applications(self):
        """Test applying different verbosity levels sequentially."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("executive_summary", "Executive summary")
        report.add_section("summary", "Summary")
        report.add_section("results", "Results")
        report.add_section("methodology", "Methodology")
        report.add_section("raw_data", "Raw data")

        # Apply STANDARD first
        apply_verbosity_level(report, VerbosityLevel.STANDARD)
        assert len(report.sections) == 3  # summary, results, methodology
        assert report.config.verbosity == "standard"

        # Apply EXECUTIVE - should filter from already filtered sections
        apply_verbosity_level(report, VerbosityLevel.EXECUTIVE)
        assert len(report.sections) == 0  # None of the remaining sections match EXECUTIVE
        assert report.config.verbosity == "executive"

    def test_apply_with_case_insensitive_section_titles(self):
        """Test that section title matching is case-insensitive."""
        config = ReportConfig()
        report = Report(config=config)
        report.add_section("Summary", "Summary with capital S")
        report.add_section("summary", "Summary with lowercase s")
        report.add_section("SUMMARY", "Summary all caps")

        apply_verbosity_level(report, VerbosityLevel.SUMMARY)

        section_titles = [s.title for s in report.sections]
        # All three variants should match (case-insensitive)
        assert section_titles == ["Summary", "summary", "SUMMARY"]


@pytest.mark.unit
class TestVerbosityIntegration:
    """Integration tests for verbosity system."""

    def test_full_workflow_executive(self):
        """Test complete workflow for EXECUTIVE verbosity level."""
        # Create report with typical sections
        config = ReportConfig(title="Test Report", author="Test Author")
        report = Report(config=config)

        report.add_section("executive_summary", "Overall pass/fail: PASS")
        report.add_section("key_findings", "Critical issue found in signal X")
        report.add_section("summary", "Detailed summary...")
        report.add_section("results", "Measurement results...")
        report.add_section("methodology", "Test methodology...")

        # Apply executive verbosity
        apply_verbosity_level(report, VerbosityLevel.EXECUTIVE)

        # Verify only executive sections remain
        assert len(report.sections) == 2
        assert report.sections[0].title == "executive_summary"
        assert report.sections[1].title == "key_findings"
        assert report.config.verbosity == "executive"

        # Verify max pages
        controller = VerbosityController(level=VerbosityLevel.EXECUTIVE)
        assert controller.get_max_pages() == 1

    def test_full_workflow_debug(self):
        """Test complete workflow for DEBUG verbosity level."""
        config = ReportConfig()
        report = Report(config=config)

        # Add comprehensive sections
        sections_to_add = [
            "summary",
            "results",
            "methodology",
            "plots",
            "tables",
            "measurements",
            "intermediate_results",
            "raw_data",
            "logs",
            "provenance",
        ]

        for section_name in sections_to_add:
            report.add_section(section_name, f"{section_name} content")

        # Apply debug verbosity
        apply_verbosity_level(report, VerbosityLevel.DEBUG)

        # All sections should remain
        assert len(report.sections) == 10
        assert report.config.verbosity == "debug"

        # Verify max pages
        controller = VerbosityController(level=VerbosityLevel.DEBUG)
        assert controller.get_max_pages() == 999

    def test_controller_consistency_with_apply_function(self):
        """Test that VerbosityController and apply_verbosity_level are consistent."""
        for level in VerbosityLevel:
            controller = VerbosityController(level=level)

            # Create report with all possible sections
            config = ReportConfig()
            report = Report(config=config)

            all_sections = [
                "executive_summary",
                "key_findings",
                "summary",
                "results",
                "key_plots",
                "methodology",
                "plots",
                "tables",
                "measurements",
                "intermediate_results",
                "raw_data",
                "logs",
                "provenance",
            ]

            for section in all_sections:
                report.add_section(section, f"{section} content")

            # Apply verbosity level
            apply_verbosity_level(report, level)

            # Verify that controller.should_include_section matches the filtering
            for section in all_sections:
                included = section in [s.title for s in report.sections]
                assert included == controller.should_include_section(section), (
                    f"Mismatch for section '{section}' at level {level.value}: "
                    f"controller says {controller.should_include_section(section)}, "
                    f"but section is {'present' if included else 'absent'}"
                )
