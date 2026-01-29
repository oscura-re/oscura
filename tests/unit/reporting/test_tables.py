"""Unit tests for the table generation module.

Tests utilities for creating and formatting measurement
summary tables with professional appearance.
"""

import numpy as np
import pytest

from oscura.reporting.tables import (
    create_comparison_table,
    create_measurement_table,
    create_statistics_table,
    format_batch_summary_table,
)

pytestmark = pytest.mark.unit


class TestCreateMeasurementTable:
    """Tests for the create_measurement_table function."""

    def test_basic_table_creation(self):
        """Test creating a basic measurement table."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
            "fall_time": {"value": 1.8e-9, "unit": "s"},
        }

        result = create_measurement_table(measurements)

        assert result["type"] == "table"
        assert "headers" in result
        assert "data" in result
        assert len(result["data"]) == 2

    def test_table_with_spec(self):
        """Test table with specification column."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "spec": 5e-9, "unit": "s"},
        }

        result = create_measurement_table(measurements, show_spec=True)

        assert "Specification" in result["headers"]

    def test_table_without_spec(self):
        """Test table without specification column."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
        }

        result = create_measurement_table(measurements, show_spec=False)

        assert "Specification" not in result["headers"]

    def test_table_with_margin(self):
        """Test table with margin column."""
        measurements = {
            "rise_time": {"value": 2.5e-9, "spec": 5e-9, "unit": "s"},
        }

        result = create_measurement_table(measurements, show_margin=True)

        assert "Margin" in result["headers"]
        # Check margin calculation (50% margin)
        data_row = result["data"][0]
        margin_idx = result["headers"].index("Margin")
        assert "%" in str(data_row[margin_idx])

    def test_table_with_status(self):
        """Test table with pass/fail status."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "passed": True},
            "fall_time": {"value": 10e-9, "passed": False},
        }

        result = create_measurement_table(measurements, show_status=True)

        assert "Status" in result["headers"]
        status_idx = result["headers"].index("Status")

        # Check PASS status
        assert "PASS" in str(result["data"][0][status_idx])
        # Check FAIL status
        assert "FAIL" in str(result["data"][1][status_idx])

    def test_table_na_for_missing_value(self):
        """Test N/A for missing values."""
        measurements = {
            "rise_time": {"unit": "s"},  # No value
        }

        result = create_measurement_table(measurements)

        data_row = result["data"][0]
        assert "N/A" in str(data_row)

    def test_table_sorting(self):
        """Test table sorting by column."""
        measurements = {
            "zebra": {"value": 1.0},
            "alpha": {"value": 2.0},
            "middle": {"value": 3.0},
        }

        result = create_measurement_table(measurements, sort_by="Parameter")

        # Should be sorted by Parameter (alphabetical)
        params = [row[0] for row in result["data"]]
        assert params == sorted(params)

    def test_table_spec_types(self):
        """Test different specification types."""
        measurements = {
            "max_spec": {"value": 3.0, "spec": 5.0, "spec_type": "max"},
            "min_spec": {"value": 10.0, "spec": 5.0, "spec_type": "min"},
        }

        result = create_measurement_table(measurements, show_spec=True)

        spec_idx = result["headers"].index("Specification")
        assert "<" in str(result["data"][0][spec_idx])  # max spec
        assert ">" in str(result["data"][1][spec_idx])  # min spec

    def test_format_markdown(self):
        """Test markdown format output."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
        }

        result = create_measurement_table(measurements, format="markdown")

        assert isinstance(result, str)
        assert "|" in result  # Markdown table delimiter
        assert "---" in result  # Header separator

    def test_format_html(self):
        """Test HTML format output."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
        }

        result = create_measurement_table(measurements, format="html")

        assert isinstance(result, str)
        assert "<table" in result
        assert "<th>" in result
        assert "<td>" in result

    def test_format_csv(self):
        """Test CSV format output."""
        measurements = {
            "rise_time": {"value": 2.3e-9, "unit": "s"},
        }

        result = create_measurement_table(measurements, format="csv")

        assert isinstance(result, str)
        assert "Parameter" in result
        assert "rise_time" in result

    def test_invalid_format_raises(self):
        """Test that invalid format raises ValueError."""
        measurements = {"test": {"value": 1.0}}

        with pytest.raises(ValueError, match="Unknown format"):
            create_measurement_table(measurements, format="invalid")

    def test_html_pass_fail_classes(self):
        """Test that HTML includes pass/fail CSS classes."""
        measurements = {
            "passed_test": {"value": 1.0, "passed": True},
            "failed_test": {"value": 2.0, "passed": False},
        }

        result = create_measurement_table(measurements, format="html", show_status=True)

        assert 'class="pass"' in result
        assert 'class="fail"' in result


class TestCreateComparisonTable:
    """Tests for the create_comparison_table function."""

    def test_basic_comparison(self):
        """Test basic comparison table creation."""
        baseline = {"rise_time": {"value": 2.0e-9, "unit": "s"}}
        current = {"rise_time": {"value": 2.5e-9, "unit": "s"}}

        result = create_comparison_table(baseline, current)

        assert result["type"] == "table"
        assert "Baseline" in result["headers"]
        assert "Current" in result["headers"]

    def test_comparison_with_delta(self):
        """Test comparison with delta column."""
        baseline = {"test": {"value": 10.0}}
        current = {"test": {"value": 15.0}}

        result = create_comparison_table(baseline, current, show_delta=True)

        assert "Delta" in result["headers"]

    def test_comparison_with_percent_change(self):
        """Test comparison with percent change column."""
        baseline = {"test": {"value": 100.0}}
        current = {"test": {"value": 150.0}}

        result = create_comparison_table(baseline, current, show_percent_change=True)

        assert "% Change" in result["headers"]
        # 50% increase
        pct_idx = result["headers"].index("% Change")
        assert "+50" in str(result["data"][0][pct_idx])

    def test_comparison_handles_missing_baseline(self):
        """Test handling of missing baseline values."""
        baseline = {}
        current = {"new_param": {"value": 1.0}}

        result = create_comparison_table(baseline, current)

        data_row = result["data"][0]
        assert "-" in str(data_row[1])  # Baseline column

    def test_comparison_handles_missing_current(self):
        """Test handling of missing current values."""
        baseline = {"old_param": {"value": 1.0}}
        current = {}

        result = create_comparison_table(baseline, current)

        data_row = result["data"][0]
        assert "-" in str(data_row[2])  # Current column

    def test_comparison_handles_zero_baseline(self):
        """Test handling of zero baseline value."""
        baseline = {"test": {"value": 0.0}}
        current = {"test": {"value": 10.0}}

        result = create_comparison_table(baseline, current, show_percent_change=True)

        pct_idx = result["headers"].index("% Change")
        assert "-" in str(result["data"][0][pct_idx])  # Can't calculate % change

    def test_comparison_format_markdown(self):
        """Test comparison table markdown format."""
        baseline = {"test": {"value": 1.0}}
        current = {"test": {"value": 2.0}}

        result = create_comparison_table(baseline, current, format="markdown")

        assert isinstance(result, str)
        assert "|" in result

    def test_comparison_format_html(self):
        """Test comparison table HTML format."""
        baseline = {"test": {"value": 1.0}}
        current = {"test": {"value": 2.0}}

        result = create_comparison_table(baseline, current, format="html")

        assert isinstance(result, str)
        assert "<table" in result

    def test_comparison_union_of_params(self):
        """Test that comparison includes all parameters from both sets."""
        baseline = {"param_a": {"value": 1.0}}
        current = {"param_b": {"value": 2.0}}

        result = create_comparison_table(baseline, current)

        params = [row[0] for row in result["data"]]
        assert "param_a" in params
        assert "param_b" in params


class TestCreateStatisticsTable:
    """Tests for the create_statistics_table function."""

    def test_basic_statistics(self):
        """Test basic statistics table creation."""
        data = {
            "signal_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }

        result = create_statistics_table(data)

        assert result["type"] == "table"
        assert "Mean" in result["headers"]
        assert "Std" in result["headers"]
        assert "Min" in result["headers"]
        assert "Max" in result["headers"]
        assert "Median" in result["headers"]

    def test_custom_statistics(self):
        """Test custom statistics selection."""
        data = {
            "signal": np.array([1.0, 2.0, 3.0]),
        }

        result = create_statistics_table(data, statistics=["mean", "max"])

        assert "Mean" in result["headers"]
        assert "Max" in result["headers"]
        assert "Min" not in result["headers"]

    def test_statistics_values(self):
        """Test that statistics are calculated correctly."""
        data = {
            "test": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        }

        result = create_statistics_table(data)

        data_row = result["data"][0]
        # Mean should be 3.0
        mean_idx = result["headers"].index("Mean")
        assert "3" in str(data_row[mean_idx])

    def test_statistics_unknown_stat(self):
        """Test handling of unknown statistic."""
        data = {
            "signal": np.array([1.0, 2.0, 3.0]),
        }

        result = create_statistics_table(data, statistics=["unknown_stat"])

        data_row = result["data"][0]
        assert "-" in str(data_row[1])

    def test_statistics_format_markdown(self):
        """Test statistics table markdown format."""
        data = {"signal": np.array([1.0, 2.0, 3.0])}

        result = create_statistics_table(data, format="markdown")

        assert isinstance(result, str)
        assert "|" in result

    def test_statistics_format_html(self):
        """Test statistics table HTML format."""
        data = {"signal": np.array([1.0, 2.0, 3.0])}

        result = create_statistics_table(data, format="html")

        assert isinstance(result, str)
        assert "<table" in result

    def test_multiple_signals(self):
        """Test table with multiple signals."""
        data = {
            "signal_a": np.array([1.0, 2.0, 3.0]),
            "signal_b": np.array([10.0, 20.0, 30.0]),
        }

        result = create_statistics_table(data)

        assert len(result["data"]) == 2


class TestFormatBatchSummaryTable:
    """Tests for the format_batch_summary_table function."""

    def test_basic_batch_summary(self):
        """Test basic batch summary table creation."""
        batch_results = [
            {"dut_id": "DUT-1", "total_count": 10, "pass_count": 9},
            {"dut_id": "DUT-2", "total_count": 10, "pass_count": 10},
        ]

        result = format_batch_summary_table(batch_results)

        assert result["type"] == "table"
        assert "DUT ID" in result["headers"]
        assert "Yield" in result["headers"]

    def test_batch_includes_summary_row(self):
        """Test that batch summary includes total row."""
        batch_results = [
            {"dut_id": "DUT-1", "total_count": 10, "pass_count": 8},
            {"dut_id": "DUT-2", "total_count": 10, "pass_count": 10},
        ]

        result = format_batch_summary_table(batch_results)

        # Should have 2 DUTs + 1 TOTAL row
        assert len(result["data"]) == 3
        total_row = result["data"][-1]
        assert "TOTAL" in str(total_row[0])

    def test_batch_yield_calculation(self):
        """Test yield percentage calculation."""
        batch_results = [
            {"dut_id": "DUT-1", "total_count": 10, "pass_count": 8},
        ]

        result = format_batch_summary_table(batch_results)

        yield_idx = result["headers"].index("Yield")
        data_row = result["data"][0]
        assert "80.0%" in str(data_row[yield_idx])

    def test_batch_default_dut_id(self):
        """Test default DUT ID when not provided."""
        batch_results = [
            {"total_count": 10, "pass_count": 10},  # No dut_id
        ]

        result = format_batch_summary_table(batch_results)

        data_row = result["data"][0]
        assert "DUT-1" in str(data_row[0])

    def test_batch_empty_results(self):
        """Test handling of empty batch results."""
        result = format_batch_summary_table([])

        assert result["type"] == "table"
        assert result["headers"] == []
        assert result["data"] == []

    def test_batch_zero_tests(self):
        """Test handling of zero total tests."""
        batch_results = [
            {"dut_id": "DUT-1", "total_count": 0, "pass_count": 0},
        ]

        result = format_batch_summary_table(batch_results)

        yield_idx = result["headers"].index("Yield")
        data_row = result["data"][0]
        assert "0.0%" in str(data_row[yield_idx])

    def test_batch_format_markdown(self):
        """Test batch summary markdown format."""
        batch_results = [{"dut_id": "DUT-1", "total_count": 10, "pass_count": 10}]

        result = format_batch_summary_table(batch_results, format="markdown")

        assert isinstance(result, str)
        assert "|" in result

    def test_batch_format_html(self):
        """Test batch summary HTML format."""
        batch_results = [{"dut_id": "DUT-1", "total_count": 10, "pass_count": 10}]

        result = format_batch_summary_table(batch_results, format="html")

        assert isinstance(result, str)
        assert "<table" in result

    def test_batch_overall_yield(self):
        """Test overall yield calculation in summary row."""
        batch_results = [
            {"dut_id": "DUT-1", "total_count": 10, "pass_count": 10},
            {"dut_id": "DUT-2", "total_count": 10, "pass_count": 5},
        ]

        result = format_batch_summary_table(batch_results)

        # Overall: 15 passed / 20 total = 75%
        total_row = result["data"][-1]
        yield_idx = result["headers"].index("Yield")
        assert "75.0%" in str(total_row[yield_idx])


class TestTableFormatConsistency:
    """Tests for format consistency across table functions."""

    def test_all_tables_support_dict_format(self):
        """Test that all table functions support dict format."""
        measurements = {"test": {"value": 1.0}}
        data = {"signal": np.array([1.0, 2.0, 3.0])}
        batch = [{"dut_id": "DUT-1", "total_count": 10, "pass_count": 10}]

        assert create_measurement_table(measurements, format="dict")["type"] == "table"
        assert create_comparison_table(measurements, measurements, format="dict")["type"] == "table"
        assert create_statistics_table(data, format="dict")["type"] == "table"
        assert format_batch_summary_table(batch, format="dict")["type"] == "table"

    def test_all_tables_support_markdown_format(self):
        """Test that all table functions support markdown format."""
        measurements = {"test": {"value": 1.0}}
        data = {"signal": np.array([1.0, 2.0, 3.0])}
        batch = [{"dut_id": "DUT-1", "total_count": 10, "pass_count": 10}]

        assert "|" in create_measurement_table(measurements, format="markdown")
        assert "|" in create_comparison_table(measurements, measurements, format="markdown")
        assert "|" in create_statistics_table(data, format="markdown")
        assert "|" in format_batch_summary_table(batch, format="markdown")

    def test_all_tables_support_html_format(self):
        """Test that all table functions support HTML format."""
        measurements = {"test": {"value": 1.0}}
        data = {"signal": np.array([1.0, 2.0, 3.0])}
        batch = [{"dut_id": "DUT-1", "total_count": 10, "pass_count": 10}]

        assert "<table" in create_measurement_table(measurements, format="html")
        assert "<table" in create_comparison_table(measurements, measurements, format="html")
        assert "<table" in create_statistics_table(data, format="html")
        assert "<table" in format_batch_summary_table(batch, format="html")
