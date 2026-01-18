"""Histogram utilities with automatic bin optimization.

This module provides intelligent histogram bin calculation using
established statistical rules.


Example:
    >>> from oscura.visualization.histogram import calculate_optimal_bins
    >>> bins = calculate_optimal_bins(data, method="freedman-diaconis")

References:
    Sturges' rule (1926)
    Freedman-Diaconis rule (1981)
    Scott's rule (1979)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def calculate_optimal_bins(
    data: NDArray[np.float64],
    *,
    method: Literal["auto", "sturges", "freedman-diaconis", "scott"] = "auto",
    min_bins: int = 5,
    max_bins: int = 200,
) -> int:
    """Calculate optimal histogram bin count using statistical rules.

    : Automatically calculate optimal histogram bin count
    using Sturges, Freedman-Diaconis, or Scott's rule.

    Args:
        data: Input data array
        method: Binning method to use
            - "auto": Auto-select based on data characteristics
            - "sturges": Sturges' rule (good for normal distributions)
            - "freedman-diaconis": Freedman-Diaconis rule (robust to outliers)
            - "scott": Scott's rule (good for smooth distributions)
        min_bins: Minimum number of bins (default: 5)
        max_bins: Maximum number of bins (default: 200)

    Returns:
        Optimal number of bins (clamped to [min_bins, max_bins])

    Raises:
        ValueError: If data is empty or invalid

    Example:
        >>> data = np.random.randn(1000)
        >>> bins = calculate_optimal_bins(data, method="freedman-diaconis")
        >>> hist, edges = np.histogram(data, bins=bins)

        >>> # Auto-select method
        >>> bins = calculate_optimal_bins(data, method="auto")

    References:
        VIS-025: Histogram Bin Optimization
        Sturges (1926): k = ceil(log2(n) + 1)
        Freedman-Diaconis (1981): h = 2 * IQR * n^(-1/3)
        Scott (1979): h = 3.5 * std * n^(-1/3)
    """
    if len(data) == 0:
        raise ValueError("Data array cannot be empty")
    if min_bins < 1:
        raise ValueError("min_bins must be >= 1")
    if max_bins < min_bins:
        raise ValueError("max_bins must be >= min_bins")

    # Remove NaN values
    clean_data = data[~np.isnan(data)]

    if len(clean_data) < 2:
        return min_bins

    # Auto-select method based on data characteristics
    if method == "auto":
        method = _auto_select_method(clean_data)

    # Calculate bins using selected method
    if method == "sturges":
        bins = _sturges_bins(clean_data)
    elif method == "freedman-diaconis":
        bins = _freedman_diaconis_bins(clean_data)
    elif method == "scott":
        bins = _scott_bins(clean_data)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Clamp to valid range
    bins = max(min_bins, min(max_bins, bins))

    return bins


def calculate_bin_edges(
    data: NDArray[np.float64],
    n_bins: int,
) -> NDArray[np.float64]:
    """Calculate histogram bin edges for given bin count.

    Args:
        data: Input data array
        n_bins: Number of bins

    Returns:
        Array of bin edges (length n_bins + 1)

    Raises:
        ValueError: If data is empty or n_bins < 1.

    Example:
        >>> data = np.random.randn(1000)
        >>> n_bins = calculate_optimal_bins(data)
        >>> edges = calculate_bin_edges(data, n_bins)
    """
    if len(data) == 0:
        raise ValueError("Data array cannot be empty")
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    # Remove NaN values
    clean_data = data[~np.isnan(data)]

    if len(clean_data) == 0:
        return np.array([0.0, 1.0])

    # Calculate edges
    data_min = np.min(clean_data)
    data_max = np.max(clean_data)

    # Handle single-value data
    if data_min == data_max:
        return np.array([data_min - 0.5, data_max + 0.5])

    edges: NDArray[np.float64] = np.linspace(data_min, data_max, n_bins + 1)
    return edges


def _sturges_bins(data: NDArray[np.float64]) -> int:
    """Calculate bins using Sturges' rule.

    Sturges' rule: k = ceil(log2(n) + 1)

    Good for: Normal distributions, small to moderate sample sizes

    Args:
        data: Input data

    Returns:
        Number of bins
    """
    n = len(data)
    bins = int(np.ceil(np.log2(n) + 1))
    return bins


def _freedman_diaconis_bins(data: NDArray[np.float64]) -> int:
    """Calculate bins using Freedman-Diaconis rule.

    Freedman-Diaconis rule: h = 2 * IQR(x) / n^(1/3)
    where h is bin width and IQR is interquartile range

    Good for: Robust estimation, data with outliers

    Args:
        data: Input data

    Returns:
        Number of bins
    """
    n = len(data)

    # Calculate IQR
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25

    if iqr == 0:
        # Fall back to Sturges if IQR is zero
        return _sturges_bins(data)

    # Calculate bin width
    bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))

    # Calculate number of bins
    data_range = np.ptp(data)  # peak-to-peak (max - min)

    if bin_width == 0:
        return _sturges_bins(data)

    bins = int(np.ceil(data_range / bin_width))

    return max(1, bins)


def _scott_bins(data: NDArray[np.float64]) -> int:
    """Calculate bins using Scott's rule.

    Scott's rule: h = 3.5 * std(x) / n^(1/3)
    where h is bin width

    Good for: Smooth distributions, normally distributed data

    Args:
        data: Input data

    Returns:
        Number of bins
    """
    n = len(data)

    # Calculate standard deviation
    std = np.std(data)

    if std == 0:
        # Fall back to Sturges if std is zero
        return _sturges_bins(data)

    # Calculate bin width
    bin_width = 3.5 * std / (n ** (1.0 / 3.0))

    # Calculate number of bins
    data_range = np.ptp(data)

    if bin_width == 0:
        return _sturges_bins(data)

    bins = int(np.ceil(data_range / bin_width))

    return max(1, bins)


def _auto_select_method(
    data: NDArray[np.float64],
) -> Literal["sturges", "freedman-diaconis", "scott"]:
    """Auto-select binning method based on data characteristics.

    Selection criteria:
    - Use Sturges for small samples (n < 100)
    - Use Freedman-Diaconis for data with outliers (high skewness)
    - Use Scott for smooth, normal-like distributions

    Args:
        data: Input data

    Returns:
        Selected method name
    """
    n = len(data)

    # Small samples: use Sturges
    if n < 100:
        return "sturges"

    # Calculate skewness to detect outliers
    mean = np.mean(data)
    std = np.std(data)

    if std == 0:
        return "sturges"

    skewness = np.mean(((data - mean) / std) ** 3)

    # High skewness indicates outliers: use Freedman-Diaconis (robust)
    if abs(skewness) > 0.5:
        return "freedman-diaconis"

    # Normal-like distribution: use Scott
    return "scott"


__all__ = [
    "calculate_bin_edges",
    "calculate_optimal_bins",
]
