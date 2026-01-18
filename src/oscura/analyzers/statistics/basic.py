"""Basic statistical analysis functions.

This module provides basic statistical measures for signal analysis,
including mean, variance, percentiles, and moment statistics.


Example:
    >>> from oscura.analyzers.statistics.basic import basic_stats, percentiles
    >>> stats = basic_stats(trace)
    >>> print(f"Mean: {stats['mean']}, Std: {stats['std']}")
    >>> pct = percentiles(trace, [25, 50, 75])

References:
    IEEE 1241-2010 Statistical analysis methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def basic_stats(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    ddof: int = 0,
) -> dict[str, float]:
    """Compute basic statistical measures.

    Calculates mean, variance, standard deviation, min, max, and range.

    Args:
        trace: Input trace or numpy array.
        ddof: Delta degrees of freedom for variance (default 0).

    Returns:
        Dictionary with statistics:
            - mean: Arithmetic mean
            - variance: Sample variance
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - range: Max - min
            - count: Number of samples

    Example:
        >>> stats = basic_stats(trace)
        >>> print(f"Mean: {stats['mean']:.6f}")
        >>> print(f"Range: {stats['range']:.3f}")
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    return {
        "mean": float(np.mean(data)),
        "variance": float(np.var(data, ddof=ddof)),
        "std": float(np.std(data, ddof=ddof)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "range": float(np.max(data) - np.min(data)),
        "count": len(data),
    }


def percentiles(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    p: list[float] | None = None,
) -> dict[str, float]:
    """Compute percentiles and quartiles.

    Args:
        trace: Input trace or numpy array.
        p: List of percentile values (0-100). If None, computes standard
            quartiles [0, 25, 50, 75, 100].

    Returns:
        Dictionary mapping percentile names to values:
            - p0, p25, p50, p75, p100 for quartiles
            - p{n} for custom percentiles

    Example:
        >>> pct = percentiles(trace)
        >>> print(f"Median: {pct['p50']}")
        >>> print(f"IQR: {pct['p75'] - pct['p25']}")

        >>> custom = percentiles(trace, [1, 10, 90, 99])
        >>> print(f"1st percentile: {custom['p1']}")
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    if p is None:
        p = [0, 25, 50, 75, 100]

    values = np.percentile(data, p)

    result = {}
    for pct, val in zip(p, values, strict=False):
        key = f"p{int(pct)}" if pct == int(pct) else f"p{pct}"
        result[key] = float(val)

    return result


def quartiles(
    trace: WaveformTrace | NDArray[np.floating[Any]],
) -> dict[str, float]:
    """Compute quartiles and IQR.

    Convenience function for quartile analysis.

    Args:
        trace: Input trace or numpy array.

    Returns:
        Dictionary with quartile statistics:
            - q1: First quartile (25th percentile)
            - median: Median (50th percentile)
            - q3: Third quartile (75th percentile)
            - iqr: Interquartile range (Q3 - Q1)
            - lower_fence: Q1 - 1.5 * IQR
            - upper_fence: Q3 + 1.5 * IQR

    Example:
        >>> q = quartiles(trace)
        >>> print(f"IQR: {q['iqr']}")
    """
    pct = percentiles(trace, [25, 50, 75])

    q1 = pct["p25"]
    median = pct["p50"]
    q3 = pct["p75"]
    iqr = q3 - q1

    return {
        "q1": q1,
        "median": median,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": q1 - 1.5 * iqr,
        "upper_fence": q3 + 1.5 * iqr,
    }


def weighted_mean(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    weights: NDArray[np.floating[Any]] | None = None,
) -> float:
    """Compute weighted mean.

    Args:
        trace: Input trace or numpy array.
        weights: Weight array (same length as data). If None, equal weights.

    Returns:
        Weighted mean value.

    Example:
        >>> weights = np.linspace(0.5, 1.0, len(trace.data))
        >>> wm = weighted_mean(trace, weights)
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    if weights is None:
        return float(np.mean(data))

    return float(np.average(data, weights=weights))


def running_stats(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    window_size: int,
) -> dict[str, NDArray[np.float64]]:
    """Compute running (rolling) statistics.

    Args:
        trace: Input trace or numpy array.
        window_size: Rolling window size in samples.

    Returns:
        Dictionary with running statistics arrays:
            - mean: Running mean
            - std: Running standard deviation
            - min: Running minimum
            - max: Running maximum

    Example:
        >>> running = running_stats(trace, window_size=100)
        >>> plt.plot(running['mean'])
    """
    data = trace.data if isinstance(trace, WaveformTrace) else trace

    n = len(data)

    window_size = min(window_size, n)

    # Pre-allocate output arrays
    result_len = n - window_size + 1
    running_mean = np.zeros(result_len, dtype=np.float64)
    running_std = np.zeros(result_len, dtype=np.float64)
    running_min = np.zeros(result_len, dtype=np.float64)
    running_max = np.zeros(result_len, dtype=np.float64)

    # Compute statistics for each window position
    for i in range(result_len):
        window = data[i : i + window_size]
        running_mean[i] = np.mean(window)
        running_std[i] = np.std(window)
        running_min[i] = np.min(window)
        running_max[i] = np.max(window)

    return {
        "mean": running_mean,
        "std": running_std,
        "min": running_min,
        "max": running_max,
    }


def summary_stats(
    trace: WaveformTrace | NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Compute comprehensive statistical summary.

    Combines basic stats, percentiles, and additional measures.

    Args:
        trace: Input trace or numpy array.

    Returns:
        Dictionary with comprehensive statistics.

    Example:
        >>> summary = summary_stats(trace)
        >>> for key, value in summary.items():
        ...     print(f"{key}: {value}")
    """
    basic = basic_stats(trace)
    quart = quartiles(trace)

    data = trace.data if isinstance(trace, WaveformTrace) else trace

    # Add additional measures
    basic.update(quart)
    basic["median_abs_dev"] = float(np.median(np.abs(data - np.median(data))))
    basic["rms"] = float(np.sqrt(np.mean(data**2)))
    basic["peak_to_rms"] = basic["max"] / basic["rms"] if basic["rms"] > 0 else float("nan")

    return basic


__all__ = [
    "basic_stats",
    "percentiles",
    "quartiles",
    "running_stats",
    "summary_stats",
    "weighted_mean",
]
