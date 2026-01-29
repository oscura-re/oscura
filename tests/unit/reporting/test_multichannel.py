"""Comprehensive unit tests for oscura.reporting.multichannel module.

Tests multi-channel report generation, channel comparison, and aggregation.
"""

from __future__ import annotations

import pytest

from oscura.reporting.core import Report, Section
from oscura.reporting.multichannel import (
    _create_aggregate_statistics_section,
    _create_channel_comparison_section,
    _create_channel_section,
    _generate_multichannel_summary,
    create_channel_crosstalk_section,
    generate_multichannel_report,
)

pytestmark = pytest.mark.unit


class TestGenerateMultichannelReport:
    """Tests for generate_multichannel_report function."""

    def test_basic_multichannel_report(self):
        """Test creating a basic multi-channel report."""
        channel_results = {
            "CH1": {
                "measurements": {"rise_time": {"value": 2.0e-9, "unit": "s"}},
                "pass_count": 10,
                "total_count": 10,
            },
            "CH2": {
                "measurements": {"rise_time": {"value": 2.5e-9, "unit": "s"}},
                "pass_count": 9,
                "total_count": 10,
            },
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)
        assert report.config.title == "Multi-Channel Analysis Report"
        assert len(report.sections) >= 1

    def test_multichannel_report_custom_title(self):
        """Test multi-channel report with custom title."""
        channel_results = {"CH1": {"measurements": {}}}

        report = generate_multichannel_report(channel_results, title="Custom Channel Report")

        assert report.config.title == "Custom Channel Report"

    def test_multichannel_report_single_channel(self):
        """Test multi-channel report with only one channel."""
        channel_results = {
            "CH1": {
                "measurements": {"param": {"value": 1.0}},
                "pass_count": 5,
                "total_count": 5,
            }
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_multichannel_report_many_channels(self):
        """Test multi-channel report with many channels."""
        channel_results = {
            f"CH{i}": {
                "measurements": {"param": {"value": float(i)}},
                "pass_count": 10,
                "total_count": 10,
            }
            for i in range(10)
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_multichannel_report_with_comparison(self):
        """Test multi-channel report with channel comparison enabled."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": 2.0}}},
        }

        report = generate_multichannel_report(channel_results, compare_channels=True)

        section_titles = [s.title for s in report.sections]
        assert "Channel Comparison" in section_titles

    def test_multichannel_report_without_comparison(self):
        """Test multi-channel report without channel comparison."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": 2.0}}},
        }

        report = generate_multichannel_report(channel_results, compare_channels=False)

        section_titles = [s.title for s in report.sections]
        assert "Channel Comparison" not in section_titles

    def test_multichannel_report_with_aggregate_stats(self):
        """Test multi-channel report with aggregate statistics."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": 2.0}}},
        }

        report = generate_multichannel_report(channel_results, aggregate_statistics=True)

        section_titles = [s.title for s in report.sections]
        assert "Aggregate Statistics" in section_titles

    def test_multichannel_report_without_aggregate_stats(self):
        """Test multi-channel report without aggregate statistics."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
        }

        report = generate_multichannel_report(channel_results, aggregate_statistics=False)

        section_titles = [s.title for s in report.sections]
        assert "Aggregate Statistics" not in section_titles

    def test_multichannel_report_with_individual_sections(self):
        """Test multi-channel report with individual channel sections."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": 2.0}}},
        }

        report = generate_multichannel_report(channel_results, individual_sections=True)

        section_titles = [s.title for s in report.sections]
        assert "Channel: CH1" in section_titles
        assert "Channel: CH2" in section_titles

    def test_multichannel_report_without_individual_sections(self):
        """Test multi-channel report without individual channel sections."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
        }

        report = generate_multichannel_report(channel_results, individual_sections=False)

        section_titles = [s.title for s in report.sections]
        assert not any("Channel: CH1" in t for t in section_titles)

    def test_multichannel_report_empty_channels(self):
        """Test multi-channel report with empty channel results."""
        channel_results = {}

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_multichannel_report_with_kwargs(self):
        """Test multi-channel report with additional kwargs."""
        channel_results = {"CH1": {"measurements": {}}}

        report = generate_multichannel_report(channel_results, author="Test Author")

        assert report.config.author == "Test Author"


class TestGenerateMultichannelSummary:
    """Tests for _generate_multichannel_summary function."""

    def test_summary_single_channel(self):
        """Test summary with single channel."""
        channel_results = {"CH1": {"pass_count": 10, "total_count": 10}}

        summary = _generate_multichannel_summary(channel_results)

        assert "1 channel(s)" in summary
        assert "100.0% pass rate" in summary

    def test_summary_multiple_channels(self):
        """Test summary with multiple channels."""
        channel_results = {
            "CH1": {"pass_count": 10, "total_count": 10},
            "CH2": {"pass_count": 8, "total_count": 10},
            "CH3": {"pass_count": 10, "total_count": 10},
        }

        summary = _generate_multichannel_summary(channel_results)

        assert "3 channel(s)" in summary
        assert "28/30 tests passed" in summary

    def test_summary_all_passed(self):
        """Test summary when all channels passed all tests."""
        channel_results = {
            "CH1": {"pass_count": 5, "total_count": 5},
            "CH2": {"pass_count": 5, "total_count": 5},
        }

        summary = _generate_multichannel_summary(channel_results)

        assert "All channels passed" in summary

    def test_summary_with_failures(self):
        """Test summary with channel failures."""
        channel_results = {
            "CH1": {"pass_count": 5, "total_count": 10},
            "CH2": {"pass_count": 10, "total_count": 10},
        }

        summary = _generate_multichannel_summary(channel_results)

        assert "Channels with failures:" in summary
        assert "CH1" in summary

    def test_summary_no_tests(self):
        """Test summary when no tests were run."""
        channel_results = {
            "CH1": {"pass_count": 0, "total_count": 0},
        }

        summary = _generate_multichannel_summary(channel_results)

        assert "1 channel(s)" in summary

    def test_summary_empty_channels(self):
        """Test summary with empty channels."""
        channel_results = {}

        summary = _generate_multichannel_summary(channel_results)

        assert "0 channel(s)" in summary


class TestCreateAggregateStatisticsSection:
    """Tests for _create_aggregate_statistics_section function."""

    def test_aggregate_stats_basic(self):
        """Test basic aggregate statistics section."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0, "unit": "V"}}},
            "CH2": {"measurements": {"param": {"value": 3.0, "unit": "V"}}},
        }

        section = _create_aggregate_statistics_section(channel_results)

        assert isinstance(section, Section)
        assert section.title == "Aggregate Statistics"
        assert section.visible is True

    def test_aggregate_stats_min_mean_max(self):
        """Test aggregate statistics includes min, mean, max, std."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": 2.0}}},
            "CH3": {"measurements": {"param": {"value": 3.0}}},
        }

        section = _create_aggregate_statistics_section(channel_results)

        # Content should be a list containing a table
        assert isinstance(section.content, list)
        assert len(section.content) > 0
        table = section.content[0]
        assert "Min" in table["headers"]
        assert "Mean" in table["headers"]
        assert "Max" in table["headers"]
        assert "Std Dev" in table["headers"]

    def test_aggregate_stats_multiple_params(self):
        """Test aggregate statistics with multiple parameters."""
        channel_results = {
            "CH1": {
                "measurements": {
                    "rise_time": {"value": 1.0e-9, "unit": "s"},
                    "fall_time": {"value": 2.0e-9, "unit": "s"},
                }
            },
            "CH2": {
                "measurements": {
                    "rise_time": {"value": 1.5e-9, "unit": "s"},
                    "fall_time": {"value": 2.5e-9, "unit": "s"},
                }
            },
        }

        section = _create_aggregate_statistics_section(channel_results)

        table = section.content[0]
        params = [row[0] for row in table["data"]]
        assert "rise_time" in params
        assert "fall_time" in params

    def test_aggregate_stats_empty_channels(self):
        """Test aggregate statistics with empty channels."""
        channel_results = {}

        section = _create_aggregate_statistics_section(channel_results)

        assert isinstance(section, Section)
        table = section.content[0]
        assert table["data"] == []

    def test_aggregate_stats_missing_measurements(self):
        """Test aggregate statistics with missing measurements."""
        channel_results = {
            "CH1": {"measurements": {"param_a": {"value": 1.0}}},
            "CH2": {"measurements": {"param_b": {"value": 2.0}}},
        }

        section = _create_aggregate_statistics_section(channel_results)

        # Should include both parameters
        table = section.content[0]
        params = [row[0] for row in table["data"]]
        assert "param_a" in params
        assert "param_b" in params

    def test_aggregate_stats_none_values_skipped(self):
        """Test aggregate statistics skips None values."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": None}}},
        }

        section = _create_aggregate_statistics_section(channel_results)

        # Should still produce valid statistics
        assert isinstance(section, Section)


class TestCreateChannelComparisonSection:
    """Tests for _create_channel_comparison_section function."""

    def test_channel_comparison_basic(self):
        """Test basic channel comparison section."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0, "unit": "V"}}},
            "CH2": {"measurements": {"param": {"value": 2.0, "unit": "V"}}},
        }

        section = _create_channel_comparison_section(channel_results)

        assert isinstance(section, Section)
        assert section.title == "Channel Comparison"
        assert section.visible is True

    def test_channel_comparison_table_structure(self):
        """Test channel comparison table has correct structure."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": 2.0}}},
        }

        section = _create_channel_comparison_section(channel_results)

        table = section.content[0]
        assert "Parameter" in table["headers"]
        assert "CH1" in table["headers"]
        assert "CH2" in table["headers"]

    def test_channel_comparison_all_params(self):
        """Test channel comparison includes all parameters."""
        channel_results = {
            "CH1": {"measurements": {"param_a": {"value": 1.0}}},
            "CH2": {"measurements": {"param_b": {"value": 2.0}}},
        }

        section = _create_channel_comparison_section(channel_results)

        table = section.content[0]
        params = [row[0] for row in table["data"]]
        assert "param_a" in params
        assert "param_b" in params

    def test_channel_comparison_missing_value(self):
        """Test channel comparison handles missing values."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {}},  # No param
        }

        section = _create_channel_comparison_section(channel_results)

        table = section.content[0]
        # CH2 column should have "-" for missing value
        ch2_idx = table["headers"].index("CH2")
        assert table["data"][0][ch2_idx] == "-"

    def test_channel_comparison_with_units(self):
        """Test channel comparison formats with units."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0e-9, "unit": "s"}}},
            "CH2": {"measurements": {"param": {"value": 2.0e-9, "unit": "s"}}},
        }

        section = _create_channel_comparison_section(channel_results)

        table = section.content[0]
        # Values should include unit
        assert "s" in str(table["data"][0][1]) or "n" in str(table["data"][0][1])

    def test_channel_comparison_many_channels(self):
        """Test channel comparison with many channels."""
        channel_results = {
            f"CH{i}": {"measurements": {"param": {"value": float(i)}}} for i in range(8)
        }

        section = _create_channel_comparison_section(channel_results)

        table = section.content[0]
        # Should have Parameter + all channels in headers
        assert len(table["headers"]) == 9


class TestCreateChannelSection:
    """Tests for _create_channel_section function."""

    def test_channel_section_basic(self):
        """Test basic channel section creation."""
        results = {
            "measurements": {"param": {"value": 1.0}},
            "pass_count": 5,
            "total_count": 5,
        }

        section = _create_channel_section("CH1", results)

        assert isinstance(section, Section)
        assert section.title == "Channel: CH1"
        assert section.level == 2
        assert section.visible is True

    def test_channel_section_with_measurements(self):
        """Test channel section includes measurement table."""
        results = {
            "measurements": {
                "rise_time": {"value": 2.0e-9, "unit": "s"},
                "fall_time": {"value": 1.8e-9, "unit": "s"},
            },
            "pass_count": 2,
            "total_count": 2,
        }

        section = _create_channel_section("CH1", results)

        # Should have subsections with measurements
        assert len(section.subsections) > 0
        assert section.subsections[0].title == "Measurements"

    def test_channel_section_pass_rate(self):
        """Test channel section shows pass rate."""
        results = {
            "measurements": {},
            "pass_count": 8,
            "total_count": 10,
        }

        section = _create_channel_section("CH1", results)

        assert "8/10 tests passed" in section.content
        assert "80.0% pass rate" in section.content

    def test_channel_section_no_pass_count(self):
        """Test channel section handles missing pass_count."""
        results = {"measurements": {"param": {"value": 1.0}}}

        section = _create_channel_section("CH1", results)

        assert isinstance(section, Section)

    def test_channel_section_empty_measurements(self):
        """Test channel section with no measurements."""
        results = {"pass_count": 5, "total_count": 5}

        section = _create_channel_section("CH1", results)

        # Should still create section but no measurement subsection
        assert isinstance(section, Section)


class TestCreateChannelCrosstalkSection:
    """Tests for create_channel_crosstalk_section function."""

    def test_crosstalk_section_basic(self):
        """Test basic crosstalk section creation."""
        crosstalk_results = {
            "crosstalk_matrix": [
                [0, -30, -25],
                [-32, 0, -28],
                [-26, -29, 0],
            ],
            "channels": ["CH1", "CH2", "CH3"],
        }

        section = create_channel_crosstalk_section(crosstalk_results)

        assert isinstance(section, Section)
        assert section.title == "Channel Crosstalk Analysis"
        assert section.level == 2

    def test_crosstalk_section_table_structure(self):
        """Test crosstalk table has correct structure."""
        crosstalk_results = {
            "crosstalk_matrix": [
                [0, -30],
                [-30, 0],
            ],
            "channels": ["CH1", "CH2"],
        }

        section = create_channel_crosstalk_section(crosstalk_results)

        # Content should include table
        assert isinstance(section.content, list)
        table = section.content[1]  # First element is text description
        assert "Aggressor" in table["headers"][0]
        assert "CH1" in table["headers"]
        assert "CH2" in table["headers"]

    def test_crosstalk_section_diagonal_dashes(self):
        """Test crosstalk table shows dashes on diagonal."""
        crosstalk_results = {
            "crosstalk_matrix": [
                [0, -30],
                [-30, 0],
            ],
            "channels": ["CH1", "CH2"],
        }

        section = create_channel_crosstalk_section(crosstalk_results)

        table = section.content[1]
        # Diagonal should be "-"
        assert table["data"][0][1] == "-"  # CH1->CH1
        assert table["data"][1][2] == "-"  # CH2->CH2

    def test_crosstalk_section_db_units(self):
        """Test crosstalk values are formatted with dB units."""
        crosstalk_results = {
            "crosstalk_matrix": [
                [0, -30],
                [-30, 0],
            ],
            "channels": ["CH1", "CH2"],
        }

        section = create_channel_crosstalk_section(crosstalk_results)

        table = section.content[1]
        # Non-diagonal values should have dB
        assert "dB" in str(table["data"][0][2])  # CH1->CH2

    def test_crosstalk_section_no_matrix(self):
        """Test crosstalk section handles missing matrix."""
        crosstalk_results = {}

        section = create_channel_crosstalk_section(crosstalk_results)

        assert isinstance(section, Section)
        assert "No crosstalk analysis available" in section.content

    def test_crosstalk_section_large_matrix(self):
        """Test crosstalk section with large channel count."""
        n = 8
        crosstalk_results = {
            "crosstalk_matrix": [[-30 if i != j else 0 for j in range(n)] for i in range(n)],
            "channels": [f"CH{i}" for i in range(n)],
        }

        section = create_channel_crosstalk_section(crosstalk_results)

        table = section.content[1]
        # Should have n+1 headers (Parameter + n channels)
        assert len(table["headers"]) == n + 1
        # Should have n rows
        assert len(table["data"]) == n


class TestMultichannelReportIntegration:
    """Integration tests for multi-channel report generation."""

    def test_full_multichannel_workflow(self):
        """Test complete multi-channel workflow."""
        channel_results = {
            "CH1": {
                "measurements": {
                    "rise_time": {"value": 2.0e-9, "unit": "s", "passed": True},
                    "overshoot": {"value": 5.0, "unit": "%", "passed": True},
                },
                "pass_count": 2,
                "total_count": 2,
            },
            "CH2": {
                "measurements": {
                    "rise_time": {"value": 2.5e-9, "unit": "s", "passed": True},
                    "overshoot": {"value": 12.0, "unit": "%", "passed": False},
                },
                "pass_count": 1,
                "total_count": 2,
            },
            "CH3": {
                "measurements": {
                    "rise_time": {"value": 1.8e-9, "unit": "s", "passed": True},
                    "overshoot": {"value": 3.0, "unit": "%", "passed": True},
                },
                "pass_count": 2,
                "total_count": 2,
            },
        }

        report = generate_multichannel_report(
            channel_results,
            compare_channels=True,
            aggregate_statistics=True,
            individual_sections=True,
        )

        # Verify report structure
        assert isinstance(report, Report)
        section_titles = [s.title for s in report.sections]

        assert "Executive Summary" in section_titles
        assert "Aggregate Statistics" in section_titles
        assert "Channel Comparison" in section_titles
        assert "Channel: CH1" in section_titles
        assert "Channel: CH2" in section_titles
        assert "Channel: CH3" in section_titles

    def test_multichannel_report_to_markdown(self):
        """Test multi-channel report can be converted to markdown."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
        }

        report = generate_multichannel_report(channel_results)
        markdown = report.to_markdown()

        assert isinstance(markdown, str)
        assert "Multi-Channel Analysis Report" in markdown

    def test_multichannel_report_to_html(self):
        """Test multi-channel report can be converted to HTML."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
        }

        report = generate_multichannel_report(channel_results)
        html = report.to_html()

        assert isinstance(html, str)
        assert "<html>" in html
        assert "Multi-Channel Analysis Report" in html

    def test_multichannel_with_crosstalk(self):
        """Test multi-channel report with crosstalk analysis."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
            "CH2": {"measurements": {"param": {"value": 2.0}}},
        }

        report = generate_multichannel_report(channel_results)

        # Add crosstalk section manually
        crosstalk = {
            "crosstalk_matrix": [[0, -30], [-30, 0]],
            "channels": ["CH1", "CH2"],
        }
        crosstalk_section = create_channel_crosstalk_section(crosstalk)
        report.sections.append(crosstalk_section)

        section_titles = [s.title for s in report.sections]
        assert "Channel Crosstalk Analysis" in section_titles


class TestReportingMultichannelEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_channel_with_none_values(self):
        """Test handling of None values in measurements."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": None}}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_channel_with_missing_unit(self):
        """Test handling of missing units."""
        channel_results = {
            "CH1": {"measurements": {"param": {"value": 1.0}}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_channels_with_different_params(self):
        """Test channels with completely different parameters."""
        channel_results = {
            "CH1": {"measurements": {"param_a": {"value": 1.0}}},
            "CH2": {"measurements": {"param_b": {"value": 2.0}}},
            "CH3": {"measurements": {"param_c": {"value": 3.0}}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_very_large_values(self):
        """Test handling of very large values."""
        channel_results = {
            "CH1": {"measurements": {"freq": {"value": 1e12, "unit": "Hz"}}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_very_small_values(self):
        """Test handling of very small values."""
        channel_results = {
            "CH1": {"measurements": {"time": {"value": 1e-15, "unit": "s"}}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_negative_values(self):
        """Test handling of negative values."""
        channel_results = {
            "CH1": {"measurements": {"offset": {"value": -2.5, "unit": "V"}}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_channel_names_special_chars(self):
        """Test channel names with special characters."""
        channel_results = {
            "CH-1_A": {"measurements": {"param": {"value": 1.0}}},
            "CH/2/B": {"measurements": {"param": {"value": 2.0}}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)
        section_titles = [s.title for s in report.sections]
        assert any("CH-1_A" in t for t in section_titles)
        assert any("CH/2/B" in t for t in section_titles)

    def test_empty_measurements_dict(self):
        """Test channel with empty measurements dictionary."""
        channel_results = {
            "CH1": {"measurements": {}},
        }

        report = generate_multichannel_report(channel_results)

        assert isinstance(report, Report)

    def test_all_channels_failed(self):
        """Test all channels with failures."""
        channel_results = {
            "CH1": {"pass_count": 0, "total_count": 5},
            "CH2": {"pass_count": 0, "total_count": 5},
        }

        summary = _generate_multichannel_summary(channel_results)

        assert "CH1" in summary
        assert "CH2" in summary
        assert "Channels with failures:" in summary
