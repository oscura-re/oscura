import pytest

"""Unit tests for the automated chart type selection module.

Tests intelligent chart type selection based on data
characteristics to optimize data visualization in reports.
"""

import numpy as np

from oscura.reporting.chart_selection import (
    auto_select_chart,
    get_axis_scaling,
    recommend_chart_with_reasoning,
)

pytestmark = pytest.mark.unit


class TestAutoSelectChart:
    """Tests for the auto_select_chart function."""

    def test_time_series_returns_line(self):
        """Test that time series data returns line chart."""
        result = auto_select_chart("time_series", (1000, 2))
        assert result == "line"

    def test_time_series_any_shape(self):
        """Test time series with various shapes."""
        assert auto_select_chart("time_series", (100,)) == "line"
        assert auto_select_chart("time_series", (10000, 2)) == "line"
        assert auto_select_chart("time_series", (50, 5)) == "line"

    def test_frequency_returns_spectrum(self):
        """Test that frequency data returns spectrum chart."""
        result = auto_select_chart("frequency", (512,))
        assert result == "spectrum"

    def test_distribution_returns_histogram(self):
        """Test that distribution data returns histogram."""
        result = auto_select_chart("distribution", (500,))
        assert result == "histogram"

    def test_categorical_returns_bar(self):
        """Test that categorical data returns bar chart."""
        result = auto_select_chart("categorical", (10,))
        assert result == "bar"

    def test_categorical_few_items_no_data(self):
        """Test categorical with few items but no data."""
        result = auto_select_chart("categorical", (4,))
        assert result == "bar"  # No data provided, defaults to bar

    def test_categorical_few_items_with_positive_data(self):
        """Test categorical with few positive items returns pie."""
        data = np.array([30.0, 25.0, 20.0, 15.0, 10.0])
        result = auto_select_chart("categorical", data.shape, data=data)
        assert result == "pie"

    def test_categorical_few_items_with_negative_data(self):
        """Test categorical with negative values returns bar."""
        data = np.array([30.0, -25.0, 20.0, 15.0])
        result = auto_select_chart("categorical", data.shape, data=data)
        assert result == "bar"

    def test_categorical_many_items_returns_bar(self):
        """Test categorical with many items returns bar."""
        data = np.array([10.0, 20.0, 15.0, 25.0, 30.0, 5.0, 12.0, 18.0])
        result = auto_select_chart("categorical", data.shape, data=data)
        assert result == "bar"

    def test_comparison_moderate_returns_scatter(self):
        """Test comparison with moderate data returns scatter."""
        result = auto_select_chart("comparison", (500, 2))
        assert result == "scatter"

    def test_comparison_large_returns_bar(self):
        """Test comparison with large data returns bar."""
        result = auto_select_chart("comparison", (50000, 2))
        assert result == "bar"

    def test_comparison_1d_returns_bar(self):
        """Test 1D comparison returns bar."""
        result = auto_select_chart("comparison", (100,))
        assert result == "bar"

    def test_correlation_returns_scatter(self):
        """Test that correlation data returns scatter."""
        result = auto_select_chart("correlation", (1000, 2))
        assert result == "scatter"

    def test_matrix_returns_heatmap(self):
        """Test that matrix data returns heatmap."""
        result = auto_select_chart("matrix", (100, 100))
        assert result == "heatmap"

    def test_parts_returns_pie(self):
        """Test that parts-to-whole data returns pie."""
        result = auto_select_chart("parts", (5,))
        assert result == "pie"

    def test_default_1d_small_returns_bar(self):
        """Test default behavior for small 1D data."""
        result = auto_select_chart("unknown_type", (15,))
        assert result == "bar"

    def test_default_1d_large_returns_histogram(self):
        """Test default behavior for large 1D data."""
        result = auto_select_chart("unknown_type", (500,))
        assert result == "histogram"

    def test_default_2d_small_returns_scatter(self):
        """Test default behavior for small 2D data."""
        result = auto_select_chart("unknown_type", (30, 2))
        assert result == "scatter"

    def test_default_2d_large_returns_heatmap(self):
        """Test default behavior for large 2D data."""
        result = auto_select_chart("unknown_type", (100, 100))
        assert result == "heatmap"

    def test_default_fallback_returns_line(self):
        """Test fallback for unhandled shapes."""
        result = auto_select_chart("unknown_type", ())
        assert result == "line"


class TestRecommendChartWithReasoning:
    """Tests for the recommend_chart_with_reasoning function."""

    def test_returns_dict_with_required_keys(self):
        """Test that result contains required keys."""
        result = recommend_chart_with_reasoning("time_series", (1000,))

        assert "chart_type" in result
        assert "reasoning" in result

    def test_time_series_reasoning(self):
        """Test reasoning for time series data."""
        result = recommend_chart_with_reasoning("time_series", (1000,))

        assert result["chart_type"] == "line"
        assert "line" in result["reasoning"].lower()

    def test_frequency_reasoning(self):
        """Test reasoning for frequency data."""
        result = recommend_chart_with_reasoning("frequency", (512,))

        assert result["chart_type"] == "spectrum"
        assert (
            "spectrum" in result["reasoning"].lower() or "frequency" in result["reasoning"].lower()
        )

    def test_distribution_reasoning(self):
        """Test reasoning for distribution data."""
        result = recommend_chart_with_reasoning("distribution", (500,))

        assert result["chart_type"] == "histogram"
        assert (
            "histogram" in result["reasoning"].lower()
            or "distribution" in result["reasoning"].lower()
        )

    def test_categorical_reasoning(self):
        """Test reasoning for categorical data."""
        result = recommend_chart_with_reasoning("categorical", (10,))

        assert result["chart_type"] == "bar"
        assert "bar" in result["reasoning"].lower() or "categorical" in result["reasoning"].lower()

    def test_correlation_reasoning(self):
        """Test reasoning for correlation data."""
        result = recommend_chart_with_reasoning("correlation", (100, 2))

        assert result["chart_type"] == "scatter"
        assert (
            "scatter" in result["reasoning"].lower() or "correlation" in result["reasoning"].lower()
        )

    def test_matrix_reasoning(self):
        """Test reasoning for matrix data."""
        result = recommend_chart_with_reasoning("matrix", (50, 50))

        assert result["chart_type"] == "heatmap"
        assert "heatmap" in result["reasoning"].lower() or "matrix" in result["reasoning"].lower()

    def test_parts_reasoning(self):
        """Test reasoning for parts-to-whole data."""
        result = recommend_chart_with_reasoning("parts", (4,))

        assert result["chart_type"] == "pie"
        assert "pie" in result["reasoning"].lower() or "part" in result["reasoning"].lower()

    def test_reasoning_with_data(self):
        """Test recommendation with actual data."""
        data = np.array([25.0, 35.0, 20.0, 20.0])
        result = recommend_chart_with_reasoning("categorical", data.shape, data=data)

        assert result["chart_type"] == "pie"


class TestGetAxisScaling:
    """Tests for the get_axis_scaling function."""

    def test_returns_dict_with_required_keys(self):
        """Test that result contains required keys."""
        result = get_axis_scaling("time_series")

        assert "x_scale" in result
        assert "y_scale" in result

    def test_default_linear_scaling(self):
        """Test default linear scaling."""
        result = get_axis_scaling("time_series")

        assert result["x_scale"] == "linear"
        assert result["y_scale"] == "linear"

    def test_frequency_log_scaling(self):
        """Test frequency data gets log scaling."""
        result = get_axis_scaling("frequency")

        assert result["x_scale"] == "log"
        assert result["y_scale"] == "log"

    def test_distribution_linear_scaling(self):
        """Test distribution data gets linear scaling."""
        result = get_axis_scaling("distribution")

        assert result["x_scale"] == "linear"
        assert result["y_scale"] == "linear"

    def test_large_range_data_log_y(self):
        """Test that large range data gets log y scale."""
        # Data spanning > 3 orders of magnitude
        data = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        result = get_axis_scaling("comparison", data=data)

        assert result["y_scale"] == "log"

    def test_small_range_data_linear_y(self):
        """Test that small range data stays linear."""
        # Data within 2 orders of magnitude
        data = np.array([10.0, 50.0, 100.0, 150.0])
        result = get_axis_scaling("comparison", data=data)

        assert result["y_scale"] == "linear"

    def test_data_with_zeros(self):
        """Test handling data with zeros."""
        data = np.array([0.0, 1.0, 10.0, 100.0])
        result = get_axis_scaling("comparison", data=data)

        # Should still work (zeros are filtered)
        assert result["y_scale"] in ["linear", "log"]

    def test_all_zero_data(self):
        """Test handling all zero data."""
        data = np.array([0.0, 0.0, 0.0])
        result = get_axis_scaling("comparison", data=data)

        assert result["y_scale"] == "linear"

    def test_negative_data(self):
        """Test handling negative data."""
        data = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        result = get_axis_scaling("comparison", data=data)

        # Negative data should stay linear (log not applicable)
        assert result["y_scale"] == "linear"

    def test_empty_data(self):
        """Test handling empty data array."""
        data = np.array([])
        result = get_axis_scaling("comparison", data=data)

        assert result["x_scale"] == "linear"
        assert result["y_scale"] == "linear"


class TestChartTypeConsistency:
    """Tests for consistency in chart type selection."""

    def test_all_chart_types_have_reasoning(self):
        """Test that all chart types have reasoning in recommend_chart_with_reasoning."""
        test_cases = [
            ("time_series", (100,)),
            ("frequency", (512,)),
            ("distribution", (500,)),
            ("categorical", (10,)),
            ("correlation", (100, 2)),
            ("matrix", (50, 50)),
            ("parts", (5,)),
        ]

        for data_type, shape in test_cases:
            result = recommend_chart_with_reasoning(data_type, shape)
            assert result["reasoning"], f"No reasoning for {data_type}"
            assert len(result["reasoning"]) > 10, f"Reasoning too short for {data_type}"

    def test_chart_selection_deterministic(self):
        """Test that chart selection is deterministic."""
        for _ in range(10):
            result1 = auto_select_chart("time_series", (1000,))
            result2 = auto_select_chart("time_series", (1000,))
            assert result1 == result2

    def test_valid_chart_types_returned(self):
        """Test that only valid chart types are returned."""
        valid_types = {"line", "scatter", "bar", "histogram", "heatmap", "pie", "spectrum"}

        test_cases = [
            ("time_series", (100,)),
            ("frequency", (512,)),
            ("distribution", (500,)),
            ("categorical", (10,)),
            ("comparison", (100, 2)),
            ("correlation", (100, 2)),
            ("matrix", (50, 50)),
            ("parts", (5,)),
            ("unknown", (1000,)),
            ("unknown", (100, 100)),
        ]

        for data_type, shape in test_cases:
            result = auto_select_chart(data_type, shape)
            assert result in valid_types, f"Invalid chart type '{result}' for {data_type}"


class TestReportingChartSelectionEdgeCases:
    """Edge case tests for chart selection."""

    def test_very_small_shape(self):
        """Test with very small data shapes."""
        assert auto_select_chart("distribution", (1,)) in {"bar", "histogram"}
        assert auto_select_chart("matrix", (1, 1)) in {"scatter", "heatmap"}

    def test_very_large_shape(self):
        """Test with very large data shapes."""
        result = auto_select_chart("time_series", (1_000_000,))
        assert result == "line"

    def test_3d_shape(self):
        """Test with 3D data shape."""
        result = auto_select_chart("matrix", (10, 10, 10))
        # Matrix type returns heatmap regardless of shape
        assert result == "heatmap"

    def test_single_element_pie_data(self):
        """Test pie chart selection with single element."""
        data = np.array([100.0])
        result = auto_select_chart("categorical", data.shape, data=data)
        assert result == "pie"

    def test_none_data(self):
        """Test with None data parameter."""
        result = auto_select_chart("categorical", (5,), data=None)
        assert result == "bar"
