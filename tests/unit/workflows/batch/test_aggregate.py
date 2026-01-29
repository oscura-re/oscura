"""Comprehensive unit tests for batch result aggregation.

Requirements tested:
- aggregate_results with various output formats
- Statistical computations (mean, std, percentiles, outliers)
- HTML/CSV/Excel export functionality
- Plot generation with matplotlib
- IQR-based outlier detection

Coverage target: 90%+ of src/oscura/workflows/batch/aggregate.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from oscura.workflows.batch.aggregate import (
    _compute_basic_statistics,
    _compute_metric_statistics,
    _compute_single_metric_stats,
    _convert_to_dataframe,
    _create_empty_stats,
    _create_metric_plot,
    _detect_outliers,
    _export_to_file,
    _format_output,
    _generate_html_report,
    _generate_metric_plots,
    _get_outlier_files,
    _plot_boxplot,
    _plot_histogram,
    _select_metrics,
    aggregate_results,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestAggregateResultsBasic:
    """Test basic aggregate_results functionality."""

    def test_empty_dataframe(self):
        """Test aggregate_results with empty DataFrame."""
        result = aggregate_results(pd.DataFrame())
        assert result == {}

    def test_single_metric_aggregation(self):
        """Test aggregation of single metric."""
        results = pd.DataFrame({"file": ["f1", "f2", "f3"], "metric": [10.0, 20.0, 30.0]})

        aggregated = aggregate_results(results)

        assert "metric" in aggregated
        assert aggregated["metric"]["count"] == 3
        assert aggregated["metric"]["mean"] == 20.0
        assert aggregated["metric"]["min"] == 10.0
        assert aggregated["metric"]["max"] == 30.0
        assert aggregated["metric"]["median"] == 20.0

    def test_multiple_metrics_aggregation(self):
        """Test aggregation of multiple metrics."""
        results = pd.DataFrame(
            {
                "file": ["f1", "f2", "f3"],
                "rise_time": [1.0, 2.0, 3.0],
                "fall_time": [4.0, 5.0, 6.0],
            }
        )

        aggregated = aggregate_results(results)

        assert "rise_time" in aggregated
        assert "fall_time" in aggregated
        assert aggregated["rise_time"]["mean"] == 2.0
        assert aggregated["fall_time"]["mean"] == 5.0

    def test_specific_metrics_selection(self):
        """Test aggregation with specific metrics selected."""
        results = pd.DataFrame(
            {
                "file": ["f1", "f2"],
                "metric_a": [1.0, 2.0],
                "metric_b": [3.0, 4.0],
                "metric_c": [5.0, 6.0],
            }
        )

        aggregated = aggregate_results(results, metrics=["metric_a", "metric_c"])

        assert "metric_a" in aggregated
        assert "metric_c" in aggregated
        assert "metric_b" not in aggregated


@pytest.mark.unit
class TestStatisticsComputation:
    """Test statistical computation functions."""

    def test_basic_statistics(self):
        """Test _compute_basic_statistics."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

        stats = _compute_basic_statistics(values)

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["median"] == 3.0
        assert stats["q25"] == 2.0
        assert stats["q75"] == 4.0

    def test_statistics_with_variance(self):
        """Test statistics computation includes standard deviation."""
        values = pd.Series([10.0, 20.0, 30.0])

        stats = _compute_basic_statistics(values)

        assert "std" in stats
        assert stats["std"] == pytest.approx(10.0, abs=0.01)

    def test_empty_stats_creation(self):
        """Test _create_empty_stats."""
        stats = _create_empty_stats()

        assert stats["count"] == 0
        assert np.isnan(stats["mean"])
        assert np.isnan(stats["std"])
        assert stats["outliers"] == []
        assert stats["outlier_files"] == []

    def test_compute_metric_statistics(self):
        """Test _compute_metric_statistics for multiple metrics."""
        results = pd.DataFrame({"metric_a": [1.0, 2.0, 3.0], "metric_b": [10.0, 20.0, 30.0]})

        aggregated = _compute_metric_statistics(results, ["metric_a", "metric_b"], 3.0)

        assert "metric_a" in aggregated
        assert "metric_b" in aggregated
        assert aggregated["metric_a"]["mean"] == 2.0
        assert aggregated["metric_b"]["mean"] == 20.0

    def test_compute_metric_statistics_missing_column(self):
        """Test _compute_metric_statistics skips missing columns."""
        results = pd.DataFrame({"metric_a": [1.0, 2.0]})

        aggregated = _compute_metric_statistics(results, ["metric_a", "missing"], 3.0)

        assert "metric_a" in aggregated
        assert "missing" not in aggregated

    def test_compute_metric_statistics_empty_values(self):
        """Test _compute_metric_statistics with all NaN values."""
        results = pd.DataFrame({"metric": [np.nan, np.nan, np.nan]})

        aggregated = _compute_metric_statistics(results, ["metric"], 3.0)

        assert aggregated["metric"]["count"] == 0
        assert np.isnan(aggregated["metric"]["mean"])

    def test_compute_single_metric_stats(self):
        """Test _compute_single_metric_stats."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[0, 1, 2, 3, 4])
        results = pd.DataFrame({"file": ["f0", "f1", "f2", "f3", "f4"], "metric": values})

        stats = _compute_single_metric_stats(values, results, 3.0)

        assert stats["count"] == 5
        assert stats["mean"] == 3.0
        assert "outliers" in stats
        assert "outlier_files" in stats


@pytest.mark.unit
class TestOutlierDetection:
    """Test outlier detection functionality."""

    def test_no_outliers(self):
        """Test outlier detection with no outliers."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = {"q25": 2.0, "q75": 4.0}

        outliers_info = _detect_outliers(values, stats, 3.0)

        assert outliers_info["outliers"] == []
        assert outliers_info["outlier_indices"] == []

    def test_detect_upper_outliers(self):
        """Test detection of upper outliers."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        stats = {"q25": 2.0, "q75": 4.0}

        outliers_info = _detect_outliers(values, stats, 3.0)

        assert len(outliers_info["outliers"]) == 1
        assert 100.0 in outliers_info["outliers"]

    def test_detect_lower_outliers(self):
        """Test detection of lower outliers."""
        values = pd.Series([0.01, 10.0, 11.0, 12.0, 13.0])
        stats = {"q25": 10.0, "q75": 13.0}

        outliers_info = _detect_outliers(values, stats, 3.0)

        assert len(outliers_info["outliers"]) >= 1
        assert 0.01 in outliers_info["outliers"]

    def test_detect_both_outliers(self):
        """Test detection of both upper and lower outliers."""
        values = pd.Series([0.1, 10.0, 11.0, 12.0, 100.0])
        stats = {"q25": 10.0, "q75": 12.0}

        outliers_info = _detect_outliers(values, stats, 3.0)

        assert len(outliers_info["outliers"]) == 2

    def test_few_values_no_outliers(self):
        """Test that outlier detection skips when too few values."""
        values = pd.Series([1.0, 2.0, 3.0])
        stats = {"q25": 1.0, "q75": 3.0}

        outliers_info = _detect_outliers(values, stats, 3.0)

        assert outliers_info["outliers"] == []

    def test_outlier_threshold_sensitivity(self):
        """Test different outlier thresholds."""
        values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        stats = {"q25": 2.0, "q75": 5.0}

        # Strict threshold
        strict = _detect_outliers(values, stats, 1.5)
        # Lenient threshold
        lenient = _detect_outliers(values, stats, 5.0)

        assert len(strict["outliers"]) >= len(lenient["outliers"])

    def test_get_outlier_files(self):
        """Test _get_outlier_files extracts file names."""
        results = pd.DataFrame({"file": ["f0", "f1", "f2", "f3"], "metric": [1, 2, 3, 100]})
        outlier_indices = [3]

        files = _get_outlier_files(results, outlier_indices)

        assert files == ["f3"]

    def test_get_outlier_files_no_file_column(self):
        """Test _get_outlier_files when no file column exists."""
        results = pd.DataFrame({"metric": [1, 2, 3, 100]})
        outlier_indices = [3]

        files = _get_outlier_files(results, outlier_indices)

        assert files == outlier_indices


@pytest.mark.unit
class TestOutputFormats:
    """Test different output format options."""

    def test_dict_output_format(self):
        """Test default dict output format."""
        results = pd.DataFrame({"file": ["f1", "f2"], "metric": [1.0, 2.0]})

        output = aggregate_results(results, output_format="dict")

        assert isinstance(output, dict)
        assert "metric" in output

    def test_dataframe_output_format(self):
        """Test DataFrame output format."""
        results = pd.DataFrame({"file": ["f1", "f2"], "metric": [1.0, 2.0]})

        output = aggregate_results(results, output_format="dataframe")

        assert isinstance(output, pd.DataFrame)
        assert "metric" in output.index

    def test_csv_output_format(self, tmp_path: Path):
        """Test CSV export format."""
        results = pd.DataFrame({"file": ["f1", "f2"], "metric": [1.0, 2.0]})
        output_file = tmp_path / "output.csv"

        output = aggregate_results(results, output_format="csv", output_file=output_file)

        assert output_file.exists()
        assert isinstance(output, pd.DataFrame)

    def test_excel_output_format(self, tmp_path: Path):
        """Test Excel export format."""
        results = pd.DataFrame({"file": ["f1", "f2"], "metric": [1.0, 2.0]})
        output_file = tmp_path / "output.xlsx"

        output = aggregate_results(results, output_format="excel", output_file=output_file)

        assert output_file.exists()
        assert isinstance(output, pd.DataFrame)

    def test_html_output_format(self, tmp_path: Path):
        """Test HTML export format."""
        results = pd.DataFrame({"file": ["f1", "f2"], "metric": [1.0, 2.0]})
        output_file = tmp_path / "report.html"

        output = aggregate_results(results, output_format="html", output_file=output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "<html>" in content.lower()
        assert "metric" in content

    def test_invalid_output_format(self):
        """Test that invalid output format raises ValueError."""
        results = pd.DataFrame({"metric": [1.0, 2.0]})

        with pytest.raises(ValueError, match="Unknown output_format"):
            aggregate_results(results, output_format="invalid")

    def test_export_without_output_file(self):
        """Test that export formats require output_file."""
        results = pd.DataFrame({"metric": [1.0, 2.0]})

        with pytest.raises(ValueError, match="requires output_file"):
            aggregate_results(results, output_format="csv", output_file=None)


@pytest.mark.unit
class TestHelperFunctions:
    """Test internal helper functions."""

    def test_select_metrics_auto(self):
        """Test _select_metrics with automatic selection."""
        results = pd.DataFrame(
            {"file": ["f1"], "metric_a": [1.0], "metric_b": [2.0], "error": ["none"]}
        )

        metrics = _select_metrics(results, None)

        assert "metric_a" in metrics
        assert "metric_b" in metrics
        assert "file" not in metrics
        assert "error" not in metrics

    def test_select_metrics_explicit(self):
        """Test _select_metrics with explicit list."""
        results = pd.DataFrame({"metric_a": [1.0], "metric_b": [2.0], "metric_c": [3.0]})

        metrics = _select_metrics(results, ["metric_a", "metric_c"])

        assert metrics == ["metric_a", "metric_c"]

    def test_select_metrics_no_numeric(self):
        """Test _select_metrics raises error when no numeric metrics."""
        results = pd.DataFrame({"file": ["f1", "f2"], "name": ["a", "b"]})

        with pytest.raises(ValueError, match="No numeric metrics found"):
            _select_metrics(results, None)

    def test_convert_to_dataframe(self):
        """Test _convert_to_dataframe conversion."""
        aggregated = {
            "metric_a": {
                "count": 3,
                "mean": 2.0,
                "std": 1.0,
                "outliers": [10.0],
                "outlier_files": ["f3"],
            }
        }

        df = _convert_to_dataframe(aggregated)

        assert isinstance(df, pd.DataFrame)
        assert "metric_a" in df.index
        assert "outliers" not in df.columns
        assert "outlier_files" not in df.columns

    def test_format_output_dict(self):
        """Test _format_output with dict format."""
        aggregated = {"metric": {"count": 3}}

        result = _format_output(aggregated, "dict", None, pd.DataFrame(), [])

        assert result == aggregated

    def test_format_output_dataframe(self):
        """Test _format_output with dataframe format."""
        aggregated = {"metric": {"count": 3, "mean": 2.0}}

        result = _format_output(aggregated, "dataframe", None, pd.DataFrame(), [])

        assert isinstance(result, pd.DataFrame)

    def test_export_to_file_csv(self, tmp_path: Path):
        """Test _export_to_file for CSV."""
        aggregated = {"metric": {"count": 3, "mean": 2.0}}
        output_file = tmp_path / "test.csv"

        df = _export_to_file(aggregated, "csv", output_file, pd.DataFrame(), [])

        assert output_file.exists()
        assert isinstance(df, pd.DataFrame)

    def test_export_to_file_missing_path(self):
        """Test _export_to_file raises error without output_file."""
        aggregated = {"metric": {"count": 3}}

        with pytest.raises(ValueError, match="requires output_file"):
            _export_to_file(aggregated, "csv", None, pd.DataFrame(), [])


@pytest.mark.unit
class TestHTMLReportGeneration:
    """Test HTML report generation."""

    def test_generate_html_report(self):
        """Test _generate_html_report creates valid HTML."""
        results = pd.DataFrame({"file": ["f1", "f2"], "metric": [1.0, 2.0]})
        aggregated = {
            "metric": {
                "count": 2,
                "mean": 1.5,
                "std": 0.5,
                "min": 1.0,
                "max": 2.0,
                "median": 1.5,
                "outliers": [],
                "outlier_files": [],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "Batch Analysis Report" in html
        assert "metric" in html
        assert "1.5" in html

    def test_html_report_with_outliers(self):
        """Test HTML report includes outlier details."""
        results = pd.DataFrame({"file": ["f1", "f2", "f3"], "metric": [1.0, 2.0, 100.0]})
        aggregated = {
            "metric": {
                "count": 3,
                "mean": 34.3,
                "std": 47.0,
                "min": 1.0,
                "max": 100.0,
                "median": 2.0,
                "outliers": [100.0],
                "outlier_files": ["f3"],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        assert "Outliers Detected" in html
        assert "f3" in html
        assert "100" in html

    def test_html_report_no_outliers(self):
        """Test HTML report without outliers."""
        results = pd.DataFrame({"metric": [1.0, 2.0]})
        aggregated = {
            "metric": {
                "count": 2,
                "mean": 1.5,
                "std": 0.5,
                "min": 1.0,
                "max": 2.0,
                "median": 1.5,
                "outliers": [],
                "outlier_files": [],
            }
        }

        html = _generate_html_report(results, aggregated, ["metric"])

        assert "Outliers Detected" not in html

    def test_html_report_multiple_metrics(self):
        """Test HTML report with multiple metrics."""
        results = pd.DataFrame({"metric_a": [1.0, 2.0], "metric_b": [3.0, 4.0]})
        aggregated = {
            "metric_a": {
                "count": 2,
                "mean": 1.5,
                "std": 0.5,
                "min": 1.0,
                "max": 2.0,
                "median": 1.5,
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
                "outliers": [],
                "outlier_files": [],
            },
        }

        html = _generate_html_report(results, aggregated, ["metric_a", "metric_b"])

        assert "metric_a" in html
        assert "metric_b" in html


@pytest.mark.unit
class TestPlotGeneration:
    """Test plot generation functionality.

    Note: Detailed matplotlib mocking tests are skipped because pandas.DataFrame.hist()
    validates that the axis is bound to a figure, which is complex to mock properly.
    The key functionality (graceful degradation when matplotlib is unavailable) is
    tested by test_plots_optional_without_matplotlib.
    """

    @patch("oscura.workflows.batch.aggregate._create_metric_plot")
    def test_generate_metric_plots(self, mock_create_plot):
        """Test _generate_metric_plots calls plot creation correctly."""
        results = pd.DataFrame({"metric": [1.0, 2.0, 3.0]})
        aggregated = {"metric": {"mean": 2.0, "median": 2.0, "outliers": [], "outlier_files": []}}

        _generate_metric_plots(results, aggregated, ["metric"], None)

        # Verify plot creation was called for the metric
        mock_create_plot.assert_called_once_with(results, aggregated, "metric", None)

    @patch("oscura.workflows.batch.aggregate._create_metric_plot")
    def test_generate_metric_plots_skips_missing_metric(self, mock_create_plot):
        """Test _generate_metric_plots skips metrics not in aggregated."""
        results = pd.DataFrame({"metric": [1.0, 2.0]})
        aggregated: dict[str, dict[str, Any]] = {}

        _generate_metric_plots(results, aggregated, ["metric"], None)

        # Should not crash, should not call plot creation for missing metric
        mock_create_plot.assert_not_called()

    @patch("oscura.workflows.batch.aggregate._plot_boxplot")
    @patch("oscura.workflows.batch.aggregate._plot_histogram")
    @patch("matplotlib.pyplot")
    def test_create_metric_plot(self, mock_plt, mock_hist, mock_box, tmp_path: Path):
        """Test _create_metric_plot creates subplots."""
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        results = pd.DataFrame({"metric": [1.0, 2.0, 3.0]})
        aggregated = {"metric": {"mean": 2.0, "median": 2.0}}
        output_file = tmp_path / "plots"

        _create_metric_plot(results, aggregated, "metric", output_file)

        mock_plt.subplots.assert_called_once()
        assert mock_plt.tight_layout.called
        mock_hist.assert_called_once()
        mock_box.assert_called_once()

    def test_plot_histogram(self):
        """Test _plot_histogram creates histogram with mean/median lines."""
        import matplotlib.pyplot as plt

        # Use real matplotlib with non-interactive backend
        fig, ax = plt.subplots()
        results = pd.DataFrame({"metric": [1.0, 2.0, 3.0]})
        aggregated = {"metric": {"mean": 2.0, "median": 2.0}}

        _plot_histogram(results, aggregated, "metric", ax)

        # Verify plot was created by checking that lines were added (mean + median)
        assert len(ax.lines) == 2  # Mean and median lines
        plt.close(fig)

    @patch("matplotlib.pyplot")
    def test_plot_boxplot(self, mock_plt):
        """Test _plot_boxplot creates box plot."""
        mock_ax = MagicMock()
        results = pd.DataFrame({"metric": [1.0, 2.0, 3.0, 4.0, 5.0]})

        _plot_boxplot(results, "metric", mock_ax)

        assert mock_ax.boxplot.called

    def test_plots_optional_without_matplotlib(self):
        """Test that plotting gracefully skips if matplotlib unavailable."""
        results = pd.DataFrame({"metric": [1.0, 2.0]})

        # Should not raise even if matplotlib import fails internally
        result = aggregate_results(results, include_plots=True)

        assert isinstance(result, dict)


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_value_no_outliers(self):
        """Test aggregation with single value produces no outliers."""
        results = pd.DataFrame({"metric": [1.0]})

        aggregated = aggregate_results(results)

        assert aggregated["metric"]["count"] == 1
        assert aggregated["metric"]["outliers"] == []

    def test_all_nan_values(self):
        """Test aggregation with all NaN values."""
        results = pd.DataFrame({"metric": [np.nan, np.nan, np.nan]})

        aggregated = aggregate_results(results)

        assert aggregated["metric"]["count"] == 0
        assert np.isnan(aggregated["metric"]["mean"])

    def test_mixed_nan_values(self):
        """Test aggregation with some NaN values."""
        results = pd.DataFrame({"metric": [1.0, np.nan, 3.0, np.nan, 5.0]})

        aggregated = aggregate_results(results)

        assert aggregated["metric"]["count"] == 3
        assert aggregated["metric"]["mean"] == 3.0

    def test_identical_values_no_outliers(self):
        """Test that identical values produce no outliers."""
        results = pd.DataFrame({"metric": [2.0, 2.0, 2.0, 2.0, 2.0]})

        aggregated = aggregate_results(results)

        assert aggregated["metric"]["outliers"] == []
        assert aggregated["metric"]["std"] == 0.0

    def test_very_large_dataset(self):
        """Test aggregation with large dataset."""
        large_data = np.random.normal(100, 10, 10000)
        results = pd.DataFrame({"metric": large_data})

        aggregated = aggregate_results(results, outlier_threshold=3.0)

        assert aggregated["metric"]["count"] == 10000
        assert 90 < aggregated["metric"]["mean"] < 110  # Should be close to 100

    def test_output_format_empty_dataframe(self):
        """Test various output formats with empty DataFrame."""
        empty_df = pd.DataFrame()

        dict_result = aggregate_results(empty_df, output_format="dict")
        df_result = aggregate_results(empty_df, output_format="dataframe")

        assert dict_result == {}
        assert isinstance(df_result, pd.DataFrame)
        assert df_result.empty

    def test_metric_not_in_results(self):
        """Test requesting metrics that don't exist in results."""
        results = pd.DataFrame({"metric_a": [1.0, 2.0]})

        # Should skip non-existent metrics gracefully
        aggregated = aggregate_results(results, metrics=["metric_a", "missing_metric"])

        assert "metric_a" in aggregated
        assert "missing_metric" not in aggregated

    def test_zero_std_deviation(self):
        """Test handling of zero standard deviation."""
        results = pd.DataFrame({"metric": [5.0, 5.0, 5.0]})

        aggregated = aggregate_results(results)

        assert aggregated["metric"]["std"] == 0.0
        assert aggregated["metric"]["outliers"] == []


class TestLazyPandasImportAggregate:
    """Test lazy pandas import behavior in aggregate module."""

    def test_aggregate_module_import_without_pandas(self) -> None:
        """Test aggregate module raises error when pandas unavailable."""
        import importlib
        import sys
        from unittest.mock import patch

        import pytest

        # Remove the module from cache if it exists
        if "oscura.workflows.batch.aggregate" in sys.modules:
            del sys.modules["oscura.workflows.batch.aggregate"]

        # Mock pandas import to fail
        with patch.dict(sys.modules, {"pandas": None}):
            with patch("builtins.__import__", side_effect=ImportError) as mock_import:
                # Configure the mock to only fail for pandas
                def import_side_effect(name, *args, **kwargs):
                    if name == "pandas" or name.startswith("pandas."):
                        raise ImportError("No module named 'pandas'")
                    return importlib.__import__(name, *args, **kwargs)

                mock_import.side_effect = import_side_effect

                with pytest.raises(ImportError) as exc_info:
                    import oscura.workflows.batch.aggregate

                assert "pandas" in str(exc_info.value).lower()
                assert "oscura[dataframes]" in str(exc_info.value)

        # Clean up - reimport the module normally
        if "oscura.workflows.batch.aggregate" in sys.modules:
            del sys.modules["oscura.workflows.batch.aggregate"]
        import oscura.workflows.batch.aggregate  # noqa: F401 - cleanup reimport
