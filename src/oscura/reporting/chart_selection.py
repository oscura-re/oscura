"""Automated chart type selection for Oscura reports.

This module provides intelligent chart type selection based on data
characteristics to optimize data visualization in reports.


Example:
    >>> from oscura.reporting import auto_select_chart
    >>> chart_type = auto_select_chart("time_series", (1000, 2))
    >>> print(chart_type)  # "line"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

ChartType = Literal["line", "scatter", "bar", "histogram", "heatmap", "pie", "spectrum"]


def _should_use_pie_chart(data_shape: tuple[int, ...], data: NDArray[np.float64] | None) -> bool:
    """Check if data is suitable for pie chart.

    Args:
        data_shape: Shape of data array
        data: Optional data array

    Returns:
        True if pie chart is appropriate
    """
    if not (len(data_shape) > 0 and data_shape[0] <= 6 and data is not None):
        return False

    all_non_negative = np.all(data >= 0)
    if not bool(all_non_negative):
        return False

    total = np.sum(data)
    return bool(total > 0 and np.allclose(data / total * 100, data / total * 100))


def _select_chart_by_shape(data_shape: tuple[int, ...]) -> ChartType:
    """Select chart type based on data shape.

    Args:
        data_shape: Shape of data array

    Returns:
        Appropriate chart type
    """
    if len(data_shape) == 1:
        return "bar" if data_shape[0] < 20 else "histogram"

    if len(data_shape) == 2:
        if data_shape[0] > 50 and data_shape[1] > 50:
            return "heatmap"
        return "scatter"

    return "line"


def auto_select_chart(
    data_type: str,
    data_shape: tuple[int, ...],
    *,
    data: NDArray[np.float64] | None = None,
) -> ChartType:
    """Automatically select appropriate chart type based on data characteristics.

    Args:
        data_type: Type of data - one of:
            - "time_series": Time-domain waveform data
            - "frequency": Frequency-domain spectral data
            - "distribution": Statistical distribution data
            - "comparison": Comparative measurements
            - "correlation": Correlation or scatter data
            - "categorical": Categorical comparison data
            - "matrix": 2D matrix data
            - "parts": Part-to-whole relationships
        data_shape: Shape of the data array (rows, [columns]).
        data: Optional actual data array for additional analysis.

    Returns:
        Recommended chart type: 'line', 'scatter', 'bar', 'histogram',
        'heatmap', 'pie', or 'spectrum'.

    Example:
        >>> # Time series data → line plot
        >>> auto_select_chart("time_series", (1000, 2))
        'line'

        >>> # Distribution data → histogram
        >>> auto_select_chart("distribution", (500,))
        'histogram'

        >>> # Categorical comparison → bar chart
        >>> auto_select_chart("categorical", (5,))
        'bar'

        >>> # 2D matrix → heatmap
        >>> auto_select_chart("matrix", (100, 100))
        'heatmap'

    References:
        REPORT-028: Automated Chart Type Selection
    """
    # Direct type mappings
    type_map: dict[str, ChartType] = {
        "time_series": "line",
        "frequency": "spectrum",
        "distribution": "histogram",
        "correlation": "scatter",
        "matrix": "heatmap",
        "parts": "pie",
    }

    if data_type in type_map:
        return type_map[data_type]

    # Categorical with possible pie chart
    if data_type == "categorical":
        if _should_use_pie_chart(data_shape, data):
            return "pie"
        return "bar"

    # Comparison with size-based selection
    if data_type == "comparison":
        if len(data_shape) >= 2 and data_shape[0] < 10000:
            return "scatter"
        return "bar"

    # Default based on shape
    return _select_chart_by_shape(data_shape)


def recommend_chart_with_reasoning(
    data_type: str,
    data_shape: tuple[int, ...],
    *,
    data: NDArray[np.float64] | None = None,
) -> dict[str, str | ChartType]:
    """Recommend chart type with reasoning explanation.

    Args:
        data_type: Type of data (see auto_select_chart).
        data_shape: Shape of the data array.
        data: Optional actual data array.

    Returns:
        Dictionary with 'chart_type' and 'reasoning' keys.

    Example:
        >>> result = recommend_chart_with_reasoning("time_series", (1000, 2))
        >>> print(result['chart_type'])  # "line"
        >>> print(result['reasoning'])  # "Time series data best shown with line plot"

    References:
        REPORT-028: Automated Chart Type Selection
    """
    chart_type = auto_select_chart(data_type, data_shape, data=data)

    # Generate reasoning
    reasoning_map = {
        "line": "Time series or sequential data best visualized with line plot",
        "scatter": "Point data or correlation best shown with scatter plot",
        "bar": "Categorical or discrete comparison best shown with bar chart",
        "histogram": "Distribution data best represented as histogram",
        "heatmap": "2D matrix data best visualized as heatmap",
        "pie": "Part-to-whole relationship best shown with pie chart",
        "spectrum": "Frequency domain data best shown with log-scale spectrum plot",
    }

    reasoning = reasoning_map.get(
        chart_type, f"Data characteristics suggest {chart_type} visualization"
    )

    return {
        "chart_type": chart_type,
        "reasoning": reasoning,
    }


def get_axis_scaling(
    data_type: str,
    data: NDArray[np.float64] | None = None,
) -> dict[str, str]:
    """Recommend axis scaling (linear vs log) based on data type.

    Args:
        data_type: Type of data.
        data: Optional actual data array for range analysis.

    Returns:
        Dictionary with 'x_scale' and 'y_scale' keys ('linear' or 'log').

    Example:
        >>> scaling = get_axis_scaling("frequency")
        >>> print(scaling)  # {'x_scale': 'log', 'y_scale': 'log'}

    References:
        REPORT-028: Automated Chart Type Selection
    """
    # Default linear scaling
    x_scale = "linear"
    y_scale = "linear"

    # Frequency data: both axes log
    if data_type == "frequency":
        x_scale = "log"
        y_scale = "log"

    # Check data range if provided
    if data is not None and len(data) > 0:
        # If data spans > 3 orders of magnitude, use log scale
        data_min = np.min(data[data > 0]) if np.any(data > 0) else 0
        data_max = np.max(data)
        if data_min > 0 and data_max / data_min > 1000:
            y_scale = "log"

    return {
        "x_scale": x_scale,
        "y_scale": y_scale,
    }


__all__ = [
    "ChartType",
    "auto_select_chart",
    "get_axis_scaling",
    "recommend_chart_with_reasoning",
]
