"""Comprehensive unit tests for oscura.reporting.comparison module.

Tests comparison report generation, waveform comparison, and diff visualization.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.reporting.comparison import (
    _create_changes_section,
    _create_detailed_comparison_section,
    _create_violations_comparison_section,
    _generate_comparison_summary,
    compare_waveforms,
    generate_comparison_report,
)
from oscura.reporting.core import Report, Section

pytestmark = pytest.mark.unit


class TestGenerateComparisonReport:
    """Tests for generate_comparison_report function."""

    def test_basic_comparison_report(self):
        """Test creating a basic comparison report."""
        baseline = {
            "measurements": {"rise_time": {"value": 2.0e-9, "unit": "s"}},
            "pass_count": 10,
            "total_count": 10,
        }
        current = {
            "measurements": {"rise_time": {"value": 2.5e-9, "unit": "s"}},
            "pass_count": 9,
            "total_count": 10,
        }

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report), "Should return Report object"
        assert report.config.title == "Comparison Report", "Should have correct title"
        assert len(report.sections) >= 1, "Should have at least 1 section"
        # Verify sections have content
        for section in report.sections:
            assert section.content, f"Section {section.title} should have content"

    def test_comparison_report_custom_title(self):
        """Test comparison report with custom title."""
        baseline = {"measurements": {}}
        current = {"measurements": {}}

        report = generate_comparison_report(baseline, current, title="My Custom Comparison")

        assert report.config.title == "My Custom Comparison"

    def test_comparison_report_side_by_side_mode(self):
        """Test comparison report in side-by-side mode."""
        baseline = {"measurements": {"param": {"value": 1.0}}}
        current = {"measurements": {"param": {"value": 2.0}}}

        report = generate_comparison_report(baseline, current, mode="side_by_side")

        assert isinstance(report, Report), "Should return Report object"
        assert len(report.sections) >= 1, "Should have at least 1 section"
        # Verify report has content
        markdown = report.to_markdown()
        assert "param" in markdown, "Report should contain parameter name"

    def test_comparison_report_inline_mode(self):
        """Test comparison report in inline mode."""
        baseline = {"measurements": {"param": {"value": 1.0}}}
        current = {"measurements": {"param": {"value": 2.0}}}

        report = generate_comparison_report(baseline, current, mode="inline")

        assert isinstance(report, Report), "Should return Report object"
        assert len(report.sections) >= 1, "Should have at least 1 section"
        # Verify report has content
        markdown = report.to_markdown()
        assert "param" in markdown, "Report should contain parameter name"

    def test_comparison_report_with_violations(self):
        """Test comparison report when both have violations."""
        baseline = {
            "measurements": {},
            "violations": [{"parameter": "rise_time"}, {"parameter": "fall_time"}],
        }
        current = {
            "measurements": {},
            "violations": [{"parameter": "rise_time"}, {"parameter": "skew"}],
        }

        report = generate_comparison_report(baseline, current)

        # Should have violations comparison section
        section_titles = [s.title for s in report.sections]
        assert "Violations Comparison" in section_titles

    def test_comparison_report_only_changes(self):
        """Test comparison report showing only changes."""
        baseline = {
            "measurements": {
                "param1": {"value": 1.0},
                "param2": {"value": 2.0},
            }
        }
        current = {
            "measurements": {
                "param1": {"value": 1.0},  # Same
                "param2": {"value": 5.0},  # Changed
            }
        }

        report = generate_comparison_report(baseline, current, show_only_changes=True)

        assert isinstance(report, Report), "Should return Report object"
        assert len(report.sections) >= 1, "Should have at least 1 section"
        # Verify only changed parameters appear
        markdown = report.to_markdown()
        # param2 should be in report (changed), param1 may or may not be
        assert "param2" in markdown, "Changed parameter should be in report"

    def test_comparison_report_no_measurements(self):
        """Test comparison report with no measurements."""
        baseline = {}
        current = {}

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report), "Should return Report object"
        assert len(report.sections) >= 1, "Should have at least 1 section"
        # Verify report handles empty case
        markdown = report.to_markdown()
        assert isinstance(markdown, str), "Should produce markdown output"
        assert len(markdown) > 0, "Should have some content"

    def test_comparison_report_highlight_changes(self):
        """Test comparison report with highlight_changes flag."""
        baseline = {"measurements": {"param": {"value": 1.0}}}
        current = {"measurements": {"param": {"value": 2.0}}}

        report = generate_comparison_report(baseline, current, highlight_changes=True)

        assert isinstance(report, Report), "Should return Report object"
        assert len(report.sections) >= 1, "Should have at least 1 section"
        markdown = report.to_markdown()
        assert "param" in markdown, "Report should contain parameter name"

    def test_comparison_report_with_kwargs(self):
        """Test comparison report with additional kwargs."""
        baseline = {"measurements": {}}
        current = {"measurements": {}}

        report = generate_comparison_report(baseline, current, author="Test Author")

        assert report.config.author == "Test Author"


class TestGenerateComparisonSummary:
    """Tests for _generate_comparison_summary function."""

    def test_summary_with_no_changes(self):
        """Test summary when no measurements changed."""
        baseline = {"measurements": {"param": {"value": 1.0, "passed": True}}}
        current = {"measurements": {"param": {"value": 1.0, "passed": True}}}

        summary = _generate_comparison_summary(baseline, current)

        assert "1 parameter(s)" in summary
        assert "changed significantly" not in summary

    def test_summary_with_significant_changes(self):
        """Test summary when measurements changed >5%."""
        baseline = {"measurements": {"param": {"value": 100.0, "passed": True}}}
        current = {"measurements": {"param": {"value": 110.0, "passed": True}}}

        summary = _generate_comparison_summary(baseline, current)

        assert "changed significantly" in summary

    def test_summary_with_improved_params(self):
        """Test summary with improved parameters (fail->pass)."""
        baseline = {"measurements": {"param": {"value": 100.0, "passed": False}}}
        current = {"measurements": {"param": {"value": 80.0, "passed": True}}}

        summary = _generate_comparison_summary(baseline, current)

        assert "improved" in summary.lower()

    def test_summary_with_degraded_params(self):
        """Test summary with degraded parameters (pass->fail)."""
        baseline = {"measurements": {"param": {"value": 80.0, "passed": True}}}
        current = {"measurements": {"param": {"value": 120.0, "passed": False}}}

        summary = _generate_comparison_summary(baseline, current)

        assert "degraded" in summary.lower()

    def test_summary_with_pass_rate_comparison(self):
        """Test summary includes pass rate comparison."""
        baseline = {"pass_count": 8, "total_count": 10, "measurements": {}}
        current = {"pass_count": 9, "total_count": 10, "measurements": {}}

        summary = _generate_comparison_summary(baseline, current)

        assert "Pass rate:" in summary
        assert "80.0%" in summary
        assert "90.0%" in summary

    def test_summary_with_empty_measurements(self):
        """Test summary with empty measurements."""
        baseline = {"measurements": {}}
        current = {"measurements": {}}

        summary = _generate_comparison_summary(baseline, current)

        assert "0 parameter(s)" in summary

    def test_summary_mixed_parameters(self):
        """Test summary with parameters in only one set."""
        baseline = {"measurements": {"param_a": {"value": 1.0}}}
        current = {"measurements": {"param_b": {"value": 2.0}}}

        summary = _generate_comparison_summary(baseline, current)

        assert "2 parameter(s)" in summary

    def test_summary_none_values_handled(self):
        """Test summary handles None values."""
        baseline = {"measurements": {"param": {"value": None}}}
        current = {"measurements": {"param": {"value": 1.0}}}

        summary = _generate_comparison_summary(baseline, current)

        assert isinstance(summary, str)


class TestCreateChangesSection:
    """Tests for _create_changes_section function."""

    def test_changes_section_basic(self):
        """Test basic changes section creation."""
        baseline = {"measurements": {"param": {"value": 1.0, "unit": "V"}}}
        current = {"measurements": {"param": {"value": 2.0, "unit": "V"}}}

        section = _create_changes_section(baseline, current)

        assert isinstance(section, Section), "Should return Section object"
        assert section.title == "Measurement Changes", "Should have correct title"
        assert section.visible is True, "Section should be visible"
        assert section.content, "Section should have content"
        assert "param" in str(section.content), "Content should mention parameter name"

    def test_changes_section_show_only_changes_true(self):
        """Test changes section with show_only_changes=True."""
        baseline = {
            "measurements": {
                "param1": {"value": 1.0},
                "param2": {"value": 2.0},
            }
        }
        current = {
            "measurements": {
                "param1": {"value": 1.0},  # Same
                "param2": {"value": 5.0},  # Changed
            }
        }

        section = _create_changes_section(baseline, current, show_only_changes=True)

        assert isinstance(section, Section), "Should return Section object"
        # Content should be filtered
        assert section.content is not None, "Section should have content"
        # Should show changed parameter
        assert "param2" in str(section.content), "Should show changed parameter"

    def test_changes_section_empty_measurements(self):
        """Test changes section with no measurements."""
        baseline = {"measurements": {}}
        current = {"measurements": {}}

        section = _create_changes_section(baseline, current)

        assert isinstance(section, Section), "Should return Section object"
        assert section.content, "Section should have content even if empty"

    def test_changes_section_content_is_table(self):
        """Test that changes section contains a table."""
        baseline = {"measurements": {"param": {"value": 1.0}}}
        current = {"measurements": {"param": {"value": 2.0}}}

        section = _create_changes_section(baseline, current)

        assert isinstance(section.content, list), "Content should be a list (table)"
        assert len(section.content) > 0, "Content list should not be empty"
        # Verify table structure has content
        assert any(section.content), "Content should have non-empty items"


class TestCreateViolationsComparisonSection:
    """Tests for _create_violations_comparison_section function."""

    def test_violations_section_new_violations(self):
        """Test violations section with new violations."""
        baseline = {"violations": [{"parameter": "existing"}]}
        current = {"violations": [{"parameter": "existing"}, {"parameter": "new_one"}]}

        section = _create_violations_comparison_section(baseline, current)

        assert isinstance(section, Section)
        assert section.title == "Violations Comparison"
        assert "New Violations" in section.content
        assert "new_one" in section.content

    def test_violations_section_resolved_violations(self):
        """Test violations section with resolved violations."""
        baseline = {"violations": [{"parameter": "old"}, {"parameter": "resolved"}]}
        current = {"violations": [{"parameter": "old"}]}

        section = _create_violations_comparison_section(baseline, current)

        assert "Resolved Violations" in section.content
        assert "resolved" in section.content

    def test_violations_section_persistent_violations(self):
        """Test violations section with persistent violations."""
        baseline = {"violations": [{"parameter": "persistent"}]}
        current = {"violations": [{"parameter": "persistent"}]}

        section = _create_violations_comparison_section(baseline, current)

        assert "Persistent Violations" in section.content
        assert "persistent" in section.content

    def test_violations_section_no_violations(self):
        """Test violations section when no violations exist."""
        baseline = {"violations": []}
        current = {"violations": []}

        section = _create_violations_comparison_section(baseline, current)

        assert "No violations" in section.content

    def test_violations_section_sorted_output(self):
        """Test violations are sorted alphabetically."""
        baseline = {"violations": [{"parameter": "z_param"}, {"parameter": "a_param"}]}
        current = {"violations": []}

        section = _create_violations_comparison_section(baseline, current)

        # a_param should appear before z_param
        content = section.content
        assert content.index("a_param") < content.index("z_param")


class TestCreateDetailedComparisonSection:
    """Tests for _create_detailed_comparison_section function."""

    def test_detailed_section_basic(self):
        """Test basic detailed comparison section."""
        baseline = {"param": {"value": 1.0, "unit": "V"}}
        current = {"param": {"value": 2.0, "unit": "V"}}

        section = _create_detailed_comparison_section(baseline, current)

        assert isinstance(section, Section)
        assert section.title == "Detailed Comparison"
        assert section.collapsible is True

    def test_detailed_section_improved_status(self):
        """Test detailed section shows improved status."""
        baseline = {"param": {"value": 1.0, "passed": False}}
        current = {"param": {"value": 0.5, "passed": True}}

        section = _create_detailed_comparison_section(baseline, current)

        assert "IMPROVED" in section.content

    def test_detailed_section_degraded_status(self):
        """Test detailed section shows degraded status."""
        baseline = {"param": {"value": 0.5, "passed": True}}
        current = {"param": {"value": 1.0, "passed": False}}

        section = _create_detailed_comparison_section(baseline, current)

        assert "DEGRADED" in section.content

    def test_detailed_section_no_change_status(self):
        """Test detailed section with no status change."""
        baseline = {"param": {"value": 1.0, "passed": True}}
        current = {"param": {"value": 1.1, "passed": True}}

        section = _create_detailed_comparison_section(baseline, current)

        assert "IMPROVED" not in section.content
        assert "DEGRADED" not in section.content

    def test_detailed_section_empty_measurements(self):
        """Test detailed section with no measurements."""
        section = _create_detailed_comparison_section({}, {})

        assert "No measurements to compare" in section.content

    def test_detailed_section_percentage_change(self):
        """Test detailed section shows percentage change."""
        baseline = {"param": {"value": 100.0}}
        current = {"param": {"value": 150.0}}

        section = _create_detailed_comparison_section(baseline, current)

        assert "+50" in section.content or "50" in section.content

    def test_detailed_section_delta_formatting(self):
        """Test detailed section formats delta values."""
        baseline = {"param": {"value": 1.0e-9, "unit": "s"}}
        current = {"param": {"value": 2.0e-9, "unit": "s"}}

        section = _create_detailed_comparison_section(baseline, current)

        # Should include delta symbol
        assert "\u0394" in section.content or "Δ" in section.content

    def test_detailed_section_sorted_params(self):
        """Test detailed section sorts parameters alphabetically."""
        baseline = {"z_param": {"value": 1.0}, "a_param": {"value": 2.0}}
        current = {"z_param": {"value": 1.0}, "a_param": {"value": 2.0}}

        section = _create_detailed_comparison_section(baseline, current)

        # a_param should appear before z_param
        assert section.content.index("a_param") < section.content.index("z_param")

    def test_detailed_section_side_by_side_mode(self):
        """Test detailed section with side_by_side mode."""
        baseline = {"param": {"value": 1.0}}
        current = {"param": {"value": 2.0}}

        section = _create_detailed_comparison_section(baseline, current, mode="side_by_side")

        assert isinstance(section, Section)

    def test_detailed_section_handles_zero_baseline(self):
        """Test detailed section handles zero baseline gracefully."""
        baseline = {"param": {"value": 0.0}}
        current = {"param": {"value": 1.0}}

        section = _create_detailed_comparison_section(baseline, current)

        # Should not crash on division by zero
        assert isinstance(section.content, str)


class TestCompareWaveforms:
    """Tests for compare_waveforms function."""

    def test_compare_waveforms_identical(self):
        """Test comparing identical waveforms."""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        baseline = {"data": signal}
        current = {"data": signal.copy()}

        result = compare_waveforms(baseline, current)

        assert result["correlation"] == pytest.approx(1.0)
        assert result["rms_difference"] == pytest.approx(0.0)
        assert result["max_difference"] == pytest.approx(0.0)
        assert result["mean_difference"] == pytest.approx(0.0)

    def test_compare_waveforms_different(self):
        """Test comparing different waveforms."""
        baseline = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        current = {"data": np.array([2.0, 3.0, 4.0, 5.0, 6.0])}

        result = compare_waveforms(baseline, current)

        assert result["correlation"] is not None
        assert result["correlation"] > 0.9  # Should still be highly correlated
        assert result["mean_difference"] == pytest.approx(1.0)

    def test_compare_waveforms_inverted(self):
        """Test comparing inverted waveforms."""
        baseline = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        current = {"data": np.array([-1.0, -2.0, -3.0, -4.0, -5.0])}

        result = compare_waveforms(baseline, current)

        assert result["correlation"] == pytest.approx(-1.0)

    def test_compare_waveforms_different_lengths(self):
        """Test comparing waveforms of different lengths."""
        baseline = {"data": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        current = {"data": np.array([1.0, 2.0, 3.0])}

        result = compare_waveforms(baseline, current)

        # Should truncate to shorter length
        assert result["correlation"] is not None

    def test_compare_waveforms_missing_baseline_data(self):
        """Test comparing when baseline has no data."""
        baseline = {}
        current = {"data": np.array([1.0, 2.0, 3.0])}

        result = compare_waveforms(baseline, current)

        assert result["correlation"] is None
        assert result["rms_difference"] is None

    def test_compare_waveforms_missing_current_data(self):
        """Test comparing when current has no data."""
        baseline = {"data": np.array([1.0, 2.0, 3.0])}
        current = {}

        result = compare_waveforms(baseline, current)

        assert result["correlation"] is None
        assert result["rms_difference"] is None

    def test_compare_waveforms_both_missing_data(self):
        """Test comparing when both have no data."""
        baseline = {}
        current = {}

        result = compare_waveforms(baseline, current)

        assert result["correlation"] is None
        assert result["rms_difference"] is None
        assert result["max_difference"] is None
        assert result["mean_difference"] is None

    def test_compare_waveforms_rms_calculation(self):
        """Test RMS difference calculation."""
        baseline = {"data": np.array([1.0, 2.0, 3.0, 4.0])}
        current = {"data": np.array([2.0, 3.0, 4.0, 5.0])}

        result = compare_waveforms(baseline, current)

        assert result["rms_difference"] == pytest.approx(1.0)

    def test_compare_waveforms_max_difference(self):
        """Test max difference calculation."""
        baseline = {"data": np.array([1.0, 2.0, 1.0, 2.0])}
        current = {"data": np.array([2.0, 3.0, 6.0, 2.0])}

        result = compare_waveforms(baseline, current)

        assert result["max_difference"] == pytest.approx(5.0)

    def test_compare_waveforms_negative_difference(self):
        """Test with negative differences."""
        baseline = {"data": np.array([5.0, 6.0, 7.0])}
        current = {"data": np.array([2.0, 3.0, 4.0])}

        result = compare_waveforms(baseline, current)

        assert result["mean_difference"] == pytest.approx(-3.0)
        # max_difference is absolute
        assert result["max_difference"] == pytest.approx(3.0)

    def test_compare_waveforms_sine_vs_cosine(self):
        """Test comparing sine vs cosine waves (90 degree phase shift)."""
        t = np.linspace(0, 2 * np.pi, 100)
        baseline = {"data": np.sin(t)}
        current = {"data": np.cos(t)}

        result = compare_waveforms(baseline, current)

        # Sine and cosine are orthogonal, correlation should be ~0
        assert abs(result["correlation"]) < 0.1


class TestComparisonReportIntegration:
    """Integration tests for comparison report generation."""

    def test_full_comparison_workflow(self):
        """Test complete comparison workflow."""
        baseline = {
            "measurements": {
                "rise_time": {"value": 2.0e-9, "unit": "s", "passed": True},
                "fall_time": {"value": 1.8e-9, "unit": "s", "passed": True},
                "overshoot": {"value": 5.0, "unit": "%", "passed": True},
            },
            "violations": [],
            "pass_count": 3,
            "total_count": 3,
        }

        current = {
            "measurements": {
                "rise_time": {"value": 2.5e-9, "unit": "s", "passed": True},
                "fall_time": {"value": 2.0e-9, "unit": "s", "passed": True},
                "overshoot": {"value": 15.0, "unit": "%", "passed": False},
            },
            "violations": [{"parameter": "overshoot"}],
            "pass_count": 2,
            "total_count": 3,
        }

        report = generate_comparison_report(baseline, current)

        # Verify report structure
        assert isinstance(report, Report)
        section_titles = [s.title for s in report.sections]

        assert "Comparison Summary" in section_titles
        assert "Measurement Changes" in section_titles
        assert "Violations Comparison" in section_titles
        assert "Detailed Comparison" in section_titles

    def test_comparison_report_to_markdown(self):
        """Test comparison report can be converted to markdown."""
        baseline = {"measurements": {"param": {"value": 1.0}}}
        current = {"measurements": {"param": {"value": 2.0}}}

        report = generate_comparison_report(baseline, current)
        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert "Comparison Report" in markdown

    def test_comparison_report_to_html(self):
        """Test comparison report can be converted to HTML."""
        baseline = {"measurements": {"param": {"value": 1.0}}}
        current = {"measurements": {"param": {"value": 2.0}}}

        report = generate_comparison_report(baseline, current)
        html = report.to_html()

        assert isinstance(html, str)
        assert "<html>" in html
        assert "Comparison Report" in html

    def test_multiple_parameter_changes(self):
        """Test comparison with many parameter changes."""
        # Start from 1 to avoid division by zero in percentage calculation
        baseline = {"measurements": {f"param_{i}": {"value": float(i + 1)} for i in range(20)}}
        current = {"measurements": {f"param_{i}": {"value": float((i + 1) * 2)} for i in range(20)}}

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report)
        # Should handle many parameters
        assert len(report.sections) >= 1


class TestReportingComparisonEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_comparison_with_none_values(self):
        """Test comparison handles None values."""
        baseline = {"measurements": {"param": {"value": None}}}
        current = {"measurements": {"param": {"value": 1.0}}}

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report)

    def test_comparison_with_missing_units(self):
        """Test comparison handles missing units."""
        baseline = {"measurements": {"param": {"value": 1.0}}}
        current = {"measurements": {"param": {"value": 2.0}}}

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report)

    def test_comparison_empty_both(self):
        """Test comparison with both empty."""
        report = generate_comparison_report({}, {})

        assert isinstance(report, Report)

    def test_comparison_large_percentage_change(self):
        """Test comparison with very large percentage change."""
        baseline = {"measurements": {"param": {"value": 0.001}}}
        current = {"measurements": {"param": {"value": 1000.0}}}

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report)

    def test_comparison_very_small_values(self):
        """Test comparison with very small values."""
        baseline = {"measurements": {"param": {"value": 1e-15, "unit": "s"}}}
        current = {"measurements": {"param": {"value": 2e-15, "unit": "s"}}}

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report)

    def test_comparison_very_large_values(self):
        """Test comparison with very large values."""
        baseline = {"measurements": {"param": {"value": 1e15, "unit": "Hz"}}}
        current = {"measurements": {"param": {"value": 2e15, "unit": "Hz"}}}

        report = generate_comparison_report(baseline, current)

        assert isinstance(report, Report)
