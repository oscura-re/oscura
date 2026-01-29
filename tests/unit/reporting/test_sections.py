"""Tests for report section generation.

Tests for:
"""

from datetime import datetime

import pytest

from oscura.reporting.core import Section
from oscura.reporting.sections import (
    create_appendix_section,
    create_conclusions_section,
    create_executive_summary_section,
    create_measurement_results_section,
    create_methodology_section,
    create_plots_section,
    create_standard_report_sections,
    create_title_section,
    create_violations_section,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestTitleSection:
    """Test create_title_section functionality."""

    def test_basic_title_section(self):
        """Test creating basic title section with minimal arguments."""
        section = create_title_section("Test Report")

        assert isinstance(section, Section)
        assert section.title == "Test Report"
        assert section.level == 0
        assert section.visible is True
        assert "Date:" in section.content

    def test_title_with_author(self):
        """Test title section with author."""
        section = create_title_section("Test Report", author="John Doe")

        assert section.title == "Test Report"
        assert "Author: John Doe" in section.content
        assert "Date:" in section.content

    def test_title_with_subtitle(self):
        """Test title section with subtitle."""
        section = create_title_section("Test Report", subtitle="Signal Analysis Results")

        assert section.title == "Test Report"
        assert "Signal Analysis Results" in section.content

    def test_title_with_custom_date(self):
        """Test title section with custom date."""
        custom_date = datetime(2024, 1, 15, 14, 30)
        section = create_title_section("Test Report", date=custom_date)

        assert section.title == "Test Report"
        assert "Date: 2024-01-15 14:30" in section.content

    def test_title_with_all_parameters(self):
        """Test title section with all parameters."""
        custom_date = datetime(2024, 1, 15, 14, 30)
        section = create_title_section(
            "Test Report", author="Jane Smith", date=custom_date, subtitle="Comprehensive Analysis"
        )

        assert section.title == "Test Report"
        assert "Comprehensive Analysis" in section.content
        assert "Author: Jane Smith" in section.content
        assert "Date: 2024-01-15 14:30" in section.content

    def test_title_content_order(self):
        """Test that title section content is in correct order."""
        section = create_title_section(
            "Test Report", author="Test Author", subtitle="Test Subtitle"
        )

        lines = section.content.split("\n")
        assert lines[0] == "Test Subtitle"
        assert lines[1] == "Author: Test Author"
        assert "Date:" in lines[2]


@pytest.mark.unit
class TestExecutiveSummarySection:
    """Test create_executive_summary_section functionality."""

    def test_basic_executive_summary(self):
        """Test creating basic executive summary."""
        results = {
            "pass_count": 5,
            "total_count": 5,
        }
        section = create_executive_summary_section(results)

        assert isinstance(section, Section)
        assert section.title == "Executive Summary"
        assert section.level == 1
        assert section.visible is True
        assert "All 5 tests passed" in section.content

    def test_executive_summary_with_failures(self):
        """Test executive summary with test failures."""
        results = {
            "pass_count": 3,
            "total_count": 5,
        }
        section = create_executive_summary_section(results)

        assert "2 of 5 tests failed" in section.content
        assert "40% failure rate" in section.content

    def test_executive_summary_with_key_findings(self):
        """Test executive summary with key findings."""
        results = {
            "pass_count": 5,
            "total_count": 5,
        }
        findings = [
            "Finding 1",
            "Finding 2",
            "Finding 3",
        ]
        section = create_executive_summary_section(results, key_findings=findings)

        assert "**Key Findings:**" in section.content
        assert "- Finding 1" in section.content
        assert "- Finding 2" in section.content
        assert "- Finding 3" in section.content

    def test_executive_summary_limits_findings_to_five(self):
        """Test that key findings are limited to top 5."""
        results = {"pass_count": 5, "total_count": 5}
        findings = [f"Finding {i}" for i in range(1, 11)]  # 10 findings
        section = create_executive_summary_section(results, key_findings=findings)

        # Should only contain first 5
        for i in range(1, 6):
            assert f"- Finding {i}" in section.content
        # Should not contain 6-10
        for i in range(6, 11):
            assert f"- Finding {i}" not in section.content

    def test_executive_summary_margin_critical(self):
        """Test executive summary with critical margin (negative)."""
        results = {
            "pass_count": 3,
            "total_count": 5,
            "min_margin": -5.2,
        }
        section = create_executive_summary_section(results)

        assert "**Margin Analysis:**" in section.content
        assert "Critical: Minimum margin is -5.2%" in section.content
        assert "violation" in section.content

    def test_executive_summary_margin_warning(self):
        """Test executive summary with warning margin (< 10%)."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "min_margin": 7.5,
        }
        section = create_executive_summary_section(results)

        assert "Warning: Minimum margin is 7.5%" in section.content
        assert "below recommended 10%" in section.content

    def test_executive_summary_margin_acceptable(self):
        """Test executive summary with acceptable margin (10-20%)."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "min_margin": 15.0,
        }
        section = create_executive_summary_section(results)

        assert "Acceptable: Minimum margin is 15.0%" in section.content
        assert "below target 20%" in section.content

    def test_executive_summary_margin_good(self):
        """Test executive summary with good margin (>= 20%)."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "min_margin": 25.0,
        }
        section = create_executive_summary_section(results)

        assert "Good: Minimum margin is 25.0%" in section.content
        assert "exceeds target 20%" in section.content

    def test_executive_summary_detailed_with_recommendations(self):
        """Test detailed executive summary with recommendations."""
        results = {
            "pass_count": 3,
            "total_count": 5,
            "violations": [
                {"parameter": "voltage", "value": 5.5, "spec": 5.0},
                {"parameter": "current", "value": 2.1, "spec": 2.0},
            ],
        }
        section = create_executive_summary_section(results, length="detailed")

        assert "**Recommendations:**" in section.content
        assert "- Address voltage violation" in section.content
        assert "- Address current violation" in section.content

    def test_executive_summary_detailed_limits_recommendations(self):
        """Test that recommendations are limited to 3 in detailed mode."""
        results = {
            "pass_count": 0,
            "total_count": 5,
            "violations": [{"parameter": f"param{i}"} for i in range(10)],
        }
        section = create_executive_summary_section(results, length="detailed")

        # Should only show first 3
        assert "- Address param0 violation" in section.content
        assert "- Address param1 violation" in section.content
        assert "- Address param2 violation" in section.content
        # Should not show 4th onward
        assert "- Address param3 violation" not in section.content

    def test_executive_summary_short_length(self):
        """Test executive summary with short length doesn't include recommendations."""
        results = {
            "pass_count": 3,
            "total_count": 5,
            "violations": [
                {"parameter": "voltage", "value": 5.5, "spec": 5.0},
            ],
        }
        section = create_executive_summary_section(results, length="short")

        # Should not include recommendations for non-detailed length
        assert "**Recommendations:**" not in section.content

    def test_executive_summary_empty_results(self):
        """Test executive summary with minimal results."""
        results = {}
        section = create_executive_summary_section(results)

        assert isinstance(section, Section)
        assert section.title == "Executive Summary"
        # Should still create valid section even with empty results


@pytest.mark.unit
class TestMeasurementResultsSection:
    """Test create_measurement_results_section functionality."""

    def test_basic_measurement_results(self):
        """Test creating measurement results section."""
        measurements = {
            "rise_time": {
                "value": 2.3e-9,
                "spec": 5e-9,
                "unit": "s",
                "passed": True,
            },
            "fall_time": {
                "value": 1.8e-9,
                "spec": 5e-9,
                "unit": "s",
                "passed": True,
            },
        }
        section = create_measurement_results_section(measurements)

        assert isinstance(section, Section)
        assert section.title == "Measurement Results"
        assert section.level == 1
        assert section.visible is True
        assert isinstance(section.content, list)
        # Content should be a list with table as last item
        assert len(section.content) == 1

    def test_measurement_results_with_failures(self):
        """Test measurement results section with failed measurements."""
        measurements = {
            "voltage": {
                "value": 5.5,
                "spec": 5.0,
                "passed": False,
            },
            "current": {
                "value": 1.8,
                "spec": 2.0,
                "passed": True,
            },
        }
        section = create_measurement_results_section(measurements)

        # Should have interpretation text when there are failures
        assert len(section.content) == 2
        assert "1 measurement(s) failed" in section.content[0]

    def test_measurement_results_multiple_failures(self):
        """Test measurement results with multiple failures."""
        measurements = {
            "param1": {"value": 1.0, "passed": False},
            "param2": {"value": 2.0, "passed": False},
            "param3": {"value": 3.0, "passed": True},
        }
        section = create_measurement_results_section(measurements)

        assert "2 measurement(s) failed" in section.content[0]

    def test_measurement_results_empty_measurements(self):
        """Test measurement results with empty measurements dict."""
        measurements = {}
        section = create_measurement_results_section(measurements)

        assert isinstance(section, Section)
        assert section.title == "Measurement Results"


@pytest.mark.unit
class TestPlotsSection:
    """Test create_plots_section functionality."""

    def test_basic_plots_section(self):
        """Test creating basic plots section."""
        figures = [
            {"type": "figure", "caption": "Signal Plot"},
            {"type": "figure", "caption": "Spectrum Plot"},
        ]
        section = create_plots_section(figures)

        assert isinstance(section, Section)
        assert section.title == "Waveform Plots"
        assert section.level == 1
        assert section.visible is True
        assert isinstance(section.content, list)
        assert len(section.content) == 2

    def test_plots_section_single_figure(self):
        """Test plots section with single figure."""
        figures = [{"type": "figure", "caption": "Test Plot"}]
        section = create_plots_section(figures)

        assert len(section.content) == 1
        assert section.content[0]["caption"] == "Test Plot"

    def test_plots_section_empty_figures(self):
        """Test plots section with empty figures list."""
        figures = []
        section = create_plots_section(figures)

        assert isinstance(section, Section)
        assert len(section.content) == 0


@pytest.mark.unit
class TestMethodologySection:
    """Test create_methodology_section functionality."""

    def test_basic_methodology_section(self):
        """Test creating basic methodology section."""
        analysis_params = {
            "sample_rate": 1e6,
            "num_samples": 10000,
            "duration": 0.01,
        }
        section = create_methodology_section(analysis_params)

        assert isinstance(section, Section)
        assert section.title == "Methodology"
        assert section.level == 1
        assert section.visible is True
        assert section.collapsible is True
        assert "Sample rate: 1e+06 Hz" in section.content
        assert "Number of samples: 10,000" in section.content
        assert "Capture duration: 0.01 s" in section.content

    def test_methodology_with_methods_standard(self):
        """Test methodology section with analysis methods in standard verbosity."""
        analysis_params = {
            "sample_rate": 1e6,
            "methods": ["FFT", "Digital filtering", "Edge detection"],
        }
        section = create_methodology_section(analysis_params, verbosity="standard")

        assert "**Analysis Methods:**" in section.content
        assert "- FFT" in section.content
        assert "- Digital filtering" in section.content
        assert "- Edge detection" in section.content

    def test_methodology_with_empty_methods(self):
        """Test methodology section with empty methods list."""
        analysis_params = {
            "sample_rate": 1e6,
            "methods": [],
        }
        section = create_methodology_section(analysis_params, verbosity="standard")

        assert "**Analysis Methods:**" in section.content
        assert "- Standard signal analysis algorithms" in section.content

    def test_methodology_with_standards(self):
        """Test methodology section with standards compliance."""
        analysis_params = {
            "sample_rate": 1e6,
            "standards": ["IEEE 802.3", "USB 2.0 Specification"],
        }
        section = create_methodology_section(analysis_params)

        assert "**Standards:**" in section.content
        assert "- IEEE 802.3" in section.content
        assert "- USB 2.0 Specification" in section.content

    def test_methodology_detailed_verbosity(self):
        """Test methodology section with detailed verbosity."""
        analysis_params = {
            "sample_rate": 1e6,
            "num_samples": 10000,
            "window_type": "hann",
            "filter_order": 8,
            "threshold": 0.5,
        }
        section = create_methodology_section(analysis_params, verbosity="detailed")

        assert "**Detailed Parameters:**" in section.content
        assert "- window_type: hann" in section.content
        assert "- filter_order: 8" in section.content
        assert "- threshold: 0.5" in section.content
        # Should not include standard params in detailed section
        assert "- sample_rate" not in section.content

    def test_methodology_summary_verbosity(self):
        """Test methodology section with summary verbosity (no methods)."""
        analysis_params = {
            "sample_rate": 1e6,
            "methods": ["FFT", "Filtering"],
        }
        section = create_methodology_section(analysis_params, verbosity="summary")

        # Summary mode should not include analysis methods
        assert "**Analysis Methods:**" not in section.content

    def test_methodology_minimal_params(self):
        """Test methodology section with minimal parameters."""
        analysis_params = {}
        section = create_methodology_section(analysis_params)

        assert isinstance(section, Section)
        assert section.title == "Methodology"


@pytest.mark.unit
class TestConclusionsSection:
    """Test create_conclusions_section functionality."""

    def test_conclusions_all_passed(self):
        """Test conclusions section when all tests passed."""
        results = {
            "pass_count": 5,
            "total_count": 5,
        }
        section = create_conclusions_section(results)

        assert isinstance(section, Section)
        assert section.title == "Conclusions"
        assert section.level == 1
        assert section.visible is True
        assert "meets all specifications" in section.content
        assert "ready for deployment" in section.content

    def test_conclusions_with_failures(self):
        """Test conclusions section with test failures."""
        results = {
            "pass_count": 3,
            "total_count": 5,
        }
        section = create_conclusions_section(results)

        assert "2 specification violation(s)" in section.content
        assert "must be addressed before deployment" in section.content

    def test_conclusions_risk_high(self):
        """Test conclusions section with high risk (negative margin)."""
        results = {
            "pass_count": 3,
            "total_count": 5,
            "min_margin": -5.0,
        }
        section = create_conclusions_section(results)

        assert "**Risk Assessment:**" in section.content
        assert "HIGH RISK: Specification violations detected." in section.content

    def test_conclusions_risk_medium(self):
        """Test conclusions section with medium risk (< 10% margin)."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "min_margin": 8.5,
        }
        section = create_conclusions_section(results)

        assert "MEDIUM RISK: Insufficient design margin." in section.content

    def test_conclusions_risk_low(self):
        """Test conclusions section with low risk (10-20% margin)."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "min_margin": 15.0,
        }
        section = create_conclusions_section(results)

        assert "LOW RISK: Adequate margin but below target." in section.content

    def test_conclusions_risk_acceptable(self):
        """Test conclusions section with acceptable risk (>= 20% margin)."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "min_margin": 25.0,
        }
        section = create_conclusions_section(results)

        assert "ACCEPTABLE: Sufficient design margin." in section.content

    def test_conclusions_with_recommendations(self):
        """Test conclusions section with recommendations."""
        results = {
            "pass_count": 5,
            "total_count": 5,
        }
        recommendations = [
            "Increase sample rate for better accuracy",
            "Add noise filtering",
            "Verify results with additional test cases",
        ]
        section = create_conclusions_section(results, recommendations=recommendations)

        assert "**Recommendations:**" in section.content
        for rec in recommendations:
            assert f"- {rec}" in section.content

    def test_conclusions_minimal_results(self):
        """Test conclusions section with minimal results."""
        results = {}
        section = create_conclusions_section(results)

        assert isinstance(section, Section)
        assert section.title == "Conclusions"


@pytest.mark.unit
class TestAppendixSection:
    """Test create_appendix_section functionality."""

    def test_basic_appendix_section(self):
        """Test creating basic appendix section."""
        raw_data = {
            "source_file": "/path/to/data.csv",
            "timestamp": "2024-01-15 14:30:00",
            "tool_version": "Oscura 1.0.0",
        }
        section = create_appendix_section(raw_data)

        assert isinstance(section, Section)
        assert section.title == "Appendix"
        assert section.level == 1
        assert section.visible is True
        assert section.collapsible is True
        assert "**Data Provenance:**" in section.content
        assert "Source: /path/to/data.csv" in section.content
        assert "Timestamp: 2024-01-15 14:30:00" in section.content
        assert "Tool Version: Oscura 1.0.0" in section.content

    def test_appendix_without_provenance(self):
        """Test appendix section without provenance information."""
        raw_data = {"data": [1, 2, 3, 4, 5]}
        section = create_appendix_section(raw_data, include_provenance=False)

        assert "**Data Provenance:**" not in section.content
        assert "**Raw Data:**" in section.content

    def test_appendix_partial_provenance(self):
        """Test appendix section with partial provenance data."""
        raw_data = {
            "source_file": "/path/to/data.csv",
            # No timestamp or tool_version
        }
        section = create_appendix_section(raw_data)

        assert "Source: /path/to/data.csv" in section.content
        assert "Timestamp:" not in section.content
        assert "Tool Version:" not in section.content

    def test_appendix_empty_data(self):
        """Test appendix section with empty data."""
        raw_data = {}
        section = create_appendix_section(raw_data)

        assert isinstance(section, Section)
        assert section.title == "Appendix"


@pytest.mark.unit
class TestViolationsSection:
    """Test create_violations_section functionality."""

    def test_violations_section_with_violations(self):
        """Test violations section with violations present."""
        violations = [
            {
                "parameter": "voltage",
                "value": 5.5,
                "specification": "< 5.0V",
                "severity": "CRITICAL",
            },
            {
                "parameter": "current",
                "value": 2.1,
                "specification": "< 2.0A",
                "severity": "WARNING",
            },
        ]
        section = create_violations_section(violations)

        assert isinstance(section, Section)
        assert section.title == "Violations"
        assert section.level == 1
        assert section.visible is True
        assert "2 specification violation(s) detected" in section.content
        assert "- **voltage**: 5.5 (spec: < 5.0V) [CRITICAL]" in section.content
        assert "- **current**: 2.1 (spec: < 2.0A) [WARNING]" in section.content

    def test_violations_section_no_violations(self):
        """Test violations section when no violations exist."""
        violations = []
        section = create_violations_section(violations)

        assert isinstance(section, Section)
        assert section.title == "Violations"
        assert section.visible is False  # Hidden when no violations
        assert "No specification violations detected." in section.content

    def test_violations_section_default_values(self):
        """Test violations section with missing optional fields."""
        violations = [
            {},  # Completely empty violation
            {"parameter": "test_param"},  # Minimal violation
        ]
        section = create_violations_section(violations)

        assert "- **Unknown**: N/A (spec: N/A) [WARNING]" in section.content
        assert "- **test_param**: N/A (spec: N/A) [WARNING]" in section.content

    def test_violations_section_single_violation(self):
        """Test violations section with single violation."""
        violations = [
            {
                "parameter": "frequency",
                "value": 105.5,
                "specification": "100 +/- 5 MHz",
                "severity": "MINOR",
            },
        ]
        section = create_violations_section(violations)

        assert "1 specification violation(s) detected" in section.content
        assert "- **frequency**: 105.5 (spec: 100 +/- 5 MHz) [MINOR]" in section.content


@pytest.mark.unit
class TestStandardReportSections:
    """Test create_standard_report_sections functionality."""

    def test_standard_report_basic(self):
        """Test creating standard report sections with basic results."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "measurements": {
                "voltage": {"value": 4.8, "spec": 5.0, "passed": True},
            },
            "analysis_params": {
                "sample_rate": 1e6,
                "num_samples": 10000,
            },
        }
        sections = create_standard_report_sections(results)

        assert isinstance(sections, list)
        assert len(sections) > 0
        # Should have executive summary, measurements, and methodology
        titles = [s.title for s in sections]
        assert "Executive Summary" in titles
        assert "Measurement Results" in titles
        assert "Methodology" in titles

    def test_standard_report_with_violations(self):
        """Test standard report sections include violations when present."""
        results = {
            "pass_count": 3,
            "total_count": 5,
            "violations": [
                {"parameter": "voltage", "value": 5.5},
                {"parameter": "current", "value": 2.1},
            ],
            "measurements": {},
        }
        sections = create_standard_report_sections(results)

        titles = [s.title for s in sections]
        assert "Violations" in titles

    def test_standard_report_with_figures(self):
        """Test standard report sections include plots when figures present."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "figures": [
                {"type": "figure", "caption": "Test Plot"},
            ],
        }
        sections = create_standard_report_sections(results)

        titles = [s.title for s in sections]
        assert "Waveform Plots" in titles

    def test_standard_report_detailed_verbosity(self):
        """Test standard report sections with detailed verbosity."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "analysis_params": {
                "sample_rate": 1e6,
            },
            "recommendations": ["Recommendation 1", "Recommendation 2"],
        }
        sections = create_standard_report_sections(results, verbosity="detailed")

        titles = [s.title for s in sections]
        # Detailed should include appendix
        assert "Appendix" in titles
        # Should include conclusions with recommendations
        assert "Conclusions" in titles

    def test_standard_report_summary_verbosity(self):
        """Test standard report sections with summary verbosity (no methodology)."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "analysis_params": {
                "sample_rate": 1e6,
            },
        }
        sections = create_standard_report_sections(results, verbosity="summary")

        titles = [s.title for s in sections]
        # Summary should not include methodology or appendix
        assert "Methodology" not in titles
        assert "Appendix" not in titles

    def test_standard_report_debug_verbosity(self):
        """Test standard report sections with debug verbosity."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "analysis_params": {
                "sample_rate": 1e6,
            },
        }
        sections = create_standard_report_sections(results, verbosity="debug")

        titles = [s.title for s in sections]
        # Debug should include everything including appendix
        assert "Methodology" in titles
        assert "Appendix" in titles

    def test_standard_report_with_conclusions(self):
        """Test standard report sections include conclusions when data present."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "conclusions": "All tests passed successfully",
            "recommendations": ["Increase test coverage"],
        }
        sections = create_standard_report_sections(results)

        titles = [s.title for s in sections]
        assert "Conclusions" in titles

    def test_standard_report_minimal_results(self):
        """Test standard report sections with minimal results."""
        results = {}
        sections = create_standard_report_sections(results)

        # Should return empty list or minimal sections when no data
        assert isinstance(sections, list)

    def test_standard_report_no_summary_data(self):
        """Test standard report without summary or pass_count."""
        results = {
            "measurements": {"voltage": {"value": 5.0}},
        }
        sections = create_standard_report_sections(results)

        titles = [s.title for s in sections]
        # Should not include executive summary if no summary data
        assert "Executive Summary" not in titles
        # But should include measurements
        assert "Measurement Results" in titles

    def test_standard_report_all_sections(self):
        """Test standard report with all possible sections."""
        results = {
            "pass_count": 4,
            "total_count": 5,
            "min_margin": 15.0,
            "violations": [{"parameter": "test", "value": 1.0}],
            "measurements": {"voltage": {"value": 5.0, "passed": True}},
            "figures": [{"type": "figure", "caption": "Plot 1"}],
            "analysis_params": {
                "sample_rate": 1e6,
                "methods": ["FFT"],
            },
            "recommendations": ["Test recommendation"],
        }
        sections = create_standard_report_sections(results, verbosity="detailed")

        titles = [s.title for s in sections]
        expected_titles = [
            "Executive Summary",
            "Violations",
            "Measurement Results",
            "Waveform Plots",
            "Methodology",
            "Conclusions",
            "Appendix",
        ]
        for expected in expected_titles:
            assert expected in titles, f"Missing section: {expected}"

    def test_standard_report_sections_are_section_objects(self):
        """Test that all returned sections are Section objects."""
        results = {
            "pass_count": 5,
            "total_count": 5,
            "measurements": {"voltage": {"value": 5.0}},
        }
        sections = create_standard_report_sections(results)

        for section in sections:
            assert isinstance(section, Section)
            assert hasattr(section, "title")
            assert hasattr(section, "content")
            assert hasattr(section, "level")
            assert hasattr(section, "visible")
