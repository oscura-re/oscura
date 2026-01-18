"""Unit tests for batch result aggregation module.

Tests BATCH-002: Result Aggregation

This test suite covers:
- Statistical aggregation of batch results
- Outlier detection using z-score method
- Multiple output format support (dict, dataframe, csv, excel, html)
- Plot generation capabilities
- Edge cases (empty data, missing values, single value)
- HTML report generation
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from oscura.batch.aggregate import _generate_html_report, aggregate_results

pytestmark = pytest.mark.unit


class TestAggregateResults:
    """Test aggregate_results function.

    Tests: BATCH-002
    """

    def test_basic_aggregation(self) -> None:
        """Test basic aggregation with default settings."""
        # Create sample results
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm", "d.wfm"],
                "rise_time": [1.0, 2.0, 3.0, 4.0],
                "fall_time": [2.0, 3.0, 4.0, 5.0],
            }
        )

        aggregated = aggregate_results(results)

        # Check dict format is default
        assert isinstance(aggregated, dict)
        assert "rise_time" in aggregated
        assert "fall_time" in aggregated

        # Check rise_time statistics
        stats = aggregated["rise_time"]
        assert stats["count"] == 4
        assert stats["mean"] == 2.5
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert stats["median"] == 2.5
        assert stats["q25"] == 1.75
        assert stats["q75"] == 3.25

    def test_specific_metrics_selection(self) -> None:
        """Test aggregating only specified metrics."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric_a": [1.0, 2.0],
                "metric_b": [3.0, 4.0],
                "metric_c": [5.0, 6.0],
            }
        )

        aggregated = aggregate_results(results, metrics=["metric_a", "metric_c"])

        assert "metric_a" in aggregated
        assert "metric_c" in aggregated
        assert "metric_b" not in aggregated

    def test_outlier_detection(self) -> None:
        """Test outlier detection using z-score method."""
        # Create data with clear outliers
        # Using more extreme outliers to ensure detection with threshold=2.0
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm", "d.wfm", "e.wfm"],
                "value": [10.0, 10.0, 10.0, 200.0, -100.0],  # 200.0 and -100.0 are clear outliers
            }
        )

        aggregated = aggregate_results(results, outlier_threshold=2.0)

        stats = aggregated["value"]
        assert len(stats["outliers"]) == 2
        assert 200.0 in stats["outliers"]
        assert -100.0 in stats["outliers"]
        assert len(stats["outlier_files"]) == 2
        assert "d.wfm" in stats["outlier_files"]
        assert "e.wfm" in stats["outlier_files"]

    def test_outlier_threshold_adjustment(self) -> None:
        """Test that outlier threshold affects detection."""
        # Use data with a clear outlier
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm", "d.wfm", "e.wfm"],
                "value": [10.0, 10.0, 10.0, 10.0, 50.0],  # 50.0 is outlier
            }
        )

        # Strict threshold - should detect outlier
        agg_strict = aggregate_results(results, outlier_threshold=1.5)
        assert len(agg_strict["value"]["outliers"]) >= 1

        # Lenient threshold - should detect fewer or no outliers
        agg_lenient = aggregate_results(results, outlier_threshold=5.0)
        assert len(agg_lenient["value"]["outliers"]) <= len(agg_strict["value"]["outliers"])

    def test_outlier_without_file_column(self) -> None:
        """Test outlier detection when 'file' column is missing."""
        results = pd.DataFrame(
            {
                "value": [10.0, 10.0, 10.0, 200.0],  # 200.0 is clear outlier
            }
        )

        aggregated = aggregate_results(results, outlier_threshold=2.0)

        stats = aggregated["value"]
        assert len(stats["outliers"]) >= 1
        # Should contain indices instead of filenames
        assert isinstance(stats["outlier_files"], list)
        assert all(isinstance(x, int | np.integer) for x in stats["outlier_files"])

    def test_empty_dataframe(self) -> None:
        """Test handling of empty input DataFrame."""
        results = pd.DataFrame()

        aggregated = aggregate_results(results)

        assert aggregated == {}

    def test_empty_dataframe_dataframe_format(self) -> None:
        """Test empty DataFrame with dataframe output format."""
        results = pd.DataFrame()

        aggregated = aggregate_results(results, output_format="dataframe")

        assert isinstance(aggregated, pd.DataFrame)
        assert aggregated.empty

    def test_all_nan_column(self) -> None:
        """Test handling of column with all NaN values."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "good_metric": [1.0, 2.0],
                "bad_metric": [np.nan, np.nan],
            }
        )

        aggregated = aggregate_results(results)

        # good_metric should have valid stats
        assert aggregated["good_metric"]["count"] == 2
        assert aggregated["good_metric"]["mean"] == 1.5

        # bad_metric should have NaN stats
        assert aggregated["bad_metric"]["count"] == 0
        assert np.isnan(aggregated["bad_metric"]["mean"])
        assert np.isnan(aggregated["bad_metric"]["std"])
        assert aggregated["bad_metric"]["outliers"] == []

    def test_partial_nan_values(self) -> None:
        """Test handling of columns with some NaN values."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm", "d.wfm"],
                "metric": [1.0, np.nan, 3.0, 4.0],
            }
        )

        aggregated = aggregate_results(results)

        stats = aggregated["metric"]
        assert stats["count"] == 3  # Only non-NaN values
        assert stats["mean"] == pytest.approx((1.0 + 3.0 + 4.0) / 3)

    def test_single_value(self) -> None:
        """Test handling of metric with single value."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm"],
                "metric": [42.0],
            }
        )

        aggregated = aggregate_results(results)

        stats = aggregated["metric"]
        assert stats["count"] == 1
        assert stats["mean"] == 42.0
        # Single value has NaN std in pandas
        assert np.isnan(stats["std"]) or stats["std"] == 0.0
        assert stats["min"] == 42.0
        assert stats["max"] == 42.0
        assert stats["median"] == 42.0
        # No outliers with single value
        assert stats["outliers"] == []

    def test_two_values_no_std(self) -> None:
        """Test handling when std is 0 (all values identical)."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm"],
                "metric": [5.0, 5.0, 5.0],
            }
        )

        aggregated = aggregate_results(results)

        stats = aggregated["metric"]
        assert stats["std"] == 0.0
        # No outliers when std=0
        assert stats["outliers"] == []

    def test_no_numeric_columns_raises_error(self) -> None:
        """Test that error is raised when no numeric columns found."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "status": ["ok", "ok"],
            }
        )

        with pytest.raises(ValueError, match="No numeric metrics found"):
            aggregate_results(results)

    def test_auto_exclude_file_and_error_columns(self) -> None:
        """Test that 'file' and 'error' columns are auto-excluded."""
        results = pd.DataFrame(
            {
                "file": [1, 2, 3],  # Numeric but should be excluded
                "error": [0, 0, 1],  # Numeric but should be excluded
                "metric": [10.0, 20.0, 30.0],
            }
        )

        aggregated = aggregate_results(results)

        assert "file" not in aggregated
        assert "error" not in aggregated
        assert "metric" in aggregated

    def test_missing_metric_column(self) -> None:
        """Test requesting a metric that doesn't exist."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric_a": [1.0, 2.0],
            }
        )

        # Should not raise error, just skip missing metric
        aggregated = aggregate_results(results, metrics=["metric_a", "metric_b"])

        assert "metric_a" in aggregated
        assert "metric_b" not in aggregated

    def test_dataframe_output_format(self) -> None:
        """Test DataFrame output format."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm"],
                "metric_a": [1.0, 2.0, 3.0],
                "metric_b": [4.0, 5.0, 6.0],
            }
        )

        aggregated = aggregate_results(results, output_format="dataframe")

        assert isinstance(aggregated, pd.DataFrame)
        assert "metric_a" in aggregated.index
        assert "metric_b" in aggregated.index
        assert "mean" in aggregated.columns
        assert "std" in aggregated.columns
        # List columns should be dropped
        assert "outliers" not in aggregated.columns
        assert "outlier_files" not in aggregated.columns

    def test_csv_output_format(self) -> None:
        """Test CSV export functionality."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric": [1.0, 2.0],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            aggregated = aggregate_results(results, output_format="csv", output_file=output_path)

            # Should return DataFrame
            assert isinstance(aggregated, pd.DataFrame)

            # Check file was created
            assert Path(output_path).exists()

            # Verify CSV contents
            loaded = pd.read_csv(output_path, index_col=0)
            assert "metric" in loaded.index
            assert "mean" in loaded.columns

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_excel_output_format(self) -> None:
        """Test Excel export functionality."""
        pytest.importorskip("openpyxl")  # Excel export requires openpyxl

        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric": [1.0, 2.0],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xlsx", delete=False) as f:
            output_path = f.name

        try:
            aggregated = aggregate_results(results, output_format="excel", output_file=output_path)

            # Should return DataFrame
            assert isinstance(aggregated, pd.DataFrame)

            # Check file was created
            assert Path(output_path).exists()

            # Verify Excel contents
            loaded = pd.read_excel(output_path, index_col=0)
            assert "metric" in loaded.index

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_html_output_format(self) -> None:
        """Test HTML report generation."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm"],
                "metric_a": [1.0, 2.0, 100.0],  # 100.0 is outlier
                "metric_b": [5.0, 6.0, 7.0],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
            output_path = f.name

        try:
            aggregated = aggregate_results(
                results,
                output_format="html",
                output_file=output_path,
                outlier_threshold=2.0,
            )

            # Should return DataFrame
            assert isinstance(aggregated, pd.DataFrame)

            # Check file was created
            assert Path(output_path).exists()

            # Verify HTML contents
            html_content = Path(output_path).read_text()
            assert "<!DOCTYPE html>" in html_content
            assert "Batch Analysis Report" in html_content
            assert "Summary Statistics" in html_content
            assert "metric_a" in html_content
            assert "metric_b" in html_content

        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_csv_without_output_file_raises_error(self) -> None:
        """Test that CSV format requires output_file."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm"],
                "metric": [1.0],
            }
        )

        with pytest.raises(ValueError, match="csv format requires output_file"):
            aggregate_results(results, output_format="csv")

    def test_excel_without_output_file_raises_error(self) -> None:
        """Test that Excel format requires output_file."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm"],
                "metric": [1.0],
            }
        )

        with pytest.raises(ValueError, match="excel format requires output_file"):
            aggregate_results(results, output_format="excel")

    def test_html_without_output_file_raises_error(self) -> None:
        """Test that HTML format requires output_file."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm"],
                "metric": [1.0],
            }
        )

        with pytest.raises(ValueError, match="html format requires output_file"):
            aggregate_results(results, output_format="html")

    def test_invalid_output_format_raises_error(self) -> None:
        """Test that invalid output format raises error."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm"],
                "metric": [1.0],
            }
        )

        with pytest.raises(ValueError, match="Unknown output_format"):
            aggregate_results(results, output_format="invalid")

    def test_statistics_precision(self) -> None:
        """Test that statistics are computed with correct precision."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm", "d.wfm", "e.wfm"],
                "metric": [1.1, 2.2, 3.3, 4.4, 5.5],
            }
        )

        aggregated = aggregate_results(results)
        stats = aggregated["metric"]

        assert stats["mean"] == pytest.approx(3.3, rel=1e-6)
        assert stats["median"] == pytest.approx(3.3, rel=1e-6)
        # Pandas uses sample std (ddof=1) by default
        assert stats["std"] == pytest.approx(1.7392527130926085, rel=1e-4)
        assert stats["q25"] == pytest.approx(2.2, rel=1e-6)
        assert stats["q75"] == pytest.approx(4.4, rel=1e-6)

    def test_large_dataset_performance(self) -> None:
        """Test aggregation with large dataset."""
        # Create large dataset
        n_files = 10000
        results = pd.DataFrame(
            {
                "file": [f"file_{i}.wfm" for i in range(n_files)],
                "metric": np.random.normal(100, 15, n_files),
            }
        )

        # Should complete without performance issues
        aggregated = aggregate_results(results, outlier_threshold=3.0)

        assert aggregated["metric"]["count"] == n_files
        assert 90 < aggregated["metric"]["mean"] < 110  # Should be ~100
        assert len(aggregated["metric"]["outliers"]) > 0  # Should find some outliers


class TestGenerateHtmlReport:
    """Test _generate_html_report internal function.

    Tests: BATCH-002
    """

    def test_basic_html_structure(self) -> None:
        """Test basic HTML report structure."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric": [1.0, 2.0],
            }
        )

        aggregated: dict[str, dict[str, Any]] = {
            "metric": {
                "count": 2,
                "mean": 1.5,
                "std": 0.5,
                "min": 1.0,
                "max": 2.0,
                "median": 1.5,
                "q25": 1.25,
                "q75": 1.75,
                "outliers": [],
                "outlier_files": [],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "<title>Batch Analysis Report</title>" in html
        assert "Summary Statistics" in html
        assert "metric" in html

    def test_html_with_outliers(self) -> None:
        """Test HTML report includes outlier section."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm", "c.wfm"],
                "metric": [1.0, 2.0, 100.0],
            }
        )

        aggregated: dict[str, dict[str, Any]] = {
            "metric": {
                "count": 3,
                "mean": 34.33,
                "std": 46.82,
                "min": 1.0,
                "max": 100.0,
                "median": 2.0,
                "q25": 1.5,
                "q75": 51.0,
                "outliers": [100.0],
                "outlier_files": ["c.wfm"],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        assert "Outliers Detected" in html
        assert "c.wfm" in html
        assert "100" in html  # The outlier value

    def test_html_without_outliers(self) -> None:
        """Test HTML report when no outliers detected."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric": [1.0, 2.0],
            }
        )

        aggregated: dict[str, dict[str, Any]] = {
            "metric": {
                "count": 2,
                "mean": 1.5,
                "std": 0.5,
                "min": 1.0,
                "max": 2.0,
                "median": 1.5,
                "q25": 1.25,
                "q75": 1.75,
                "outliers": [],
                "outlier_files": [],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        # Should not have outlier section
        assert "Outliers Detected" not in html

    def test_html_multiple_metrics(self) -> None:
        """Test HTML report with multiple metrics."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric_a": [1.0, 2.0],
                "metric_b": [3.0, 4.0],
            }
        )

        aggregated: dict[str, dict[str, Any]] = {
            "metric_a": {
                "count": 2,
                "mean": 1.5,
                "std": 0.5,
                "min": 1.0,
                "max": 2.0,
                "median": 1.5,
                "q25": 1.25,
                "q75": 1.75,
                "outliers": [],
                "outlier_files": [],
            },
            "metric_b": {
                "count": 2,
                "mean": 3.5,
                "std": 0.5,
                "min": 3.0,
                "max": 4.0,
                "median": 3.5,
                "q25": 3.25,
                "q75": 3.75,
                "outliers": [],
                "outlier_files": [],
            },
        }

        html = _generate_html_report(results, aggregated, ["metric_a", "metric_b"])

        assert "metric_a" in html
        assert "metric_b" in html

    def test_html_numeric_formatting(self) -> None:
        """Test that numeric values are formatted correctly in HTML."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm"],
                "metric": [1.23456789],
            }
        )

        aggregated: dict[str, dict[str, Any]] = {
            "metric": {
                "count": 1,
                "mean": 1.23456789,
                "std": 0.0,
                "min": 1.23456789,
                "max": 1.23456789,
                "median": 1.23456789,
                "q25": 1.23456789,
                "q75": 1.23456789,
                "outliers": [],
                "outlier_files": [],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        # Should use 4 significant figures (.4g format)
        assert "1.235" in html or "1.234" in html

    def test_html_css_styling(self) -> None:
        """Test that HTML includes CSS styling."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm"],
                "metric": [1.0],
            }
        )

        aggregated: dict[str, dict[str, Any]] = {
            "metric": {
                "count": 1,
                "mean": 1.0,
                "std": 0.0,
                "min": 1.0,
                "max": 1.0,
                "median": 1.0,
                "q25": 1.0,
                "q75": 1.0,
                "outliers": [],
                "outlier_files": [],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        assert "<style>" in html
        assert "</style>" in html
        assert "font-family:" in html
        assert "border-collapse:" in html
        assert ".outlier" in html

    def test_html_outlier_row_styling(self) -> None:
        """Test that outlier rows have special CSS class."""
        results = pd.DataFrame(
            {
                "file": ["a.wfm", "b.wfm"],
                "metric": [1.0, 100.0],
            }
        )

        aggregated: dict[str, dict[str, Any]] = {
            "metric": {
                "count": 2,
                "mean": 50.5,
                "std": 49.5,
                "min": 1.0,
                "max": 100.0,
                "median": 50.5,
                "q25": 25.75,
                "q75": 75.25,
                "outliers": [100.0],
                "outlier_files": ["b.wfm"],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        # Outlier row should have 'outlier' class
        assert "class='outlier'" in html or 'class="outlier"' in html
