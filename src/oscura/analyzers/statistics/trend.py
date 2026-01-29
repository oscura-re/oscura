"""Trend detection and analysis for signal data.

This module provides linear trend detection, drift analysis, and
detrending functions for identifying systematic changes in signals.


Example:
    >>> from oscura.analyzers.statistics.trend import (
    ...     detect_trend, detrend, moving_average
    ... )
    >>> result = detect_trend(trace)
    >>> print(f"Slope: {result['slope']:.2e} V/s")
    >>> detrended = detrend(trace)

References:
    Montgomery, D. C. (2012). Introduction to Statistical Quality Control
    NIST Engineering Statistics Handbook
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import stats

from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TrendResult:
    """Result of trend analysis.

    Attributes:
        slope: Trend slope (units per second).
        intercept: Trend intercept (at t=0).
        r_squared: Coefficient of determination.
        p_value: Statistical significance (p < 0.05 is significant).
        std_error: Standard error of slope estimate.
        is_significant: Whether trend is statistically significant.
        trend_line: Fitted trend values at each sample.
    """

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    std_error: float
    is_significant: bool
    trend_line: NDArray[np.float64]


def detect_trend(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    significance_level: float = 0.05,
    sample_rate: float | None = None,
) -> TrendResult:
    """Detect linear trend in signal data.

    Fits a linear regression and tests for statistical significance.
    Reports slope, R-squared, and whether drift is significant.

    Args:
        trace: Input trace or numpy array.
        significance_level: P-value threshold for significance (default 0.05).
        sample_rate: Sample rate in Hz (required for array input).

    Returns:
        TrendResult with trend analysis.

    Raises:
        ValueError: If trace is array and sample_rate is not provided.

    Example:
        >>> result = detect_trend(trace)
        >>> if result.is_significant:
        ...     print(f"Significant drift: {result.slope:.2e} V/s")
        ...     print(f"R-squared: {result.r_squared:.4f}")

    References:
        NIST Engineering Statistics Handbook Section 6.6
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data
        fs = trace.metadata.sample_rate
    else:
        data = trace
        if sample_rate is None:
            raise ValueError("sample_rate required when trace is array")
        fs = sample_rate

    n = len(data)

    if n < 3:
        return TrendResult(
            slope=np.nan,
            intercept=np.nan,
            r_squared=np.nan,
            p_value=np.nan,
            std_error=np.nan,
            is_significant=False,
            trend_line=np.full(n, np.nan, dtype=np.float64),
        )

    # Time axis in seconds
    t = np.arange(n) / fs

    # Linear regression
    result = stats.linregress(t, data)

    slope = float(result.slope)
    intercept = float(result.intercept)
    r_squared = float(result.rvalue**2)
    p_value = float(result.pvalue)
    std_error = float(result.stderr)
    is_significant = p_value < significance_level

    # Compute trend line
    trend_line = intercept + slope * t

    return TrendResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        p_value=p_value,
        std_error=std_error,
        is_significant=is_significant,
        trend_line=trend_line.astype(np.float64),
    )


def detrend(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    method: Literal["linear", "constant", "polynomial"] = "linear",
    order: int = 1,
    return_trend: bool = False,
    sample_rate: float | None = None,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove trend from signal data.

    Subtracts fitted trend to isolate fluctuations around baseline.

    Args:
        trace: Input trace or numpy array.
        method: Detrending method:
            - "constant": Remove mean (DC offset)
            - "linear": Remove linear trend (default)
            - "polynomial": Remove polynomial trend
        order: Polynomial order (for method="polynomial").
        return_trend: If True, also return the removed trend.
        sample_rate: Sample rate in Hz (required for array input, only for linear).

    Returns:
        Detrended data array.
        If return_trend=True, returns (detrended, trend).

    Raises:
        ValueError: If method is not recognized.

    Example:
        >>> detrended = detrend(trace, method="linear")
        >>> # Or get the trend too
        >>> detrended, trend = detrend(trace, return_trend=True)
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data.astype(np.float64)
        fs = trace.metadata.sample_rate
    else:
        data = np.array(trace, dtype=np.float64)
        fs = sample_rate if sample_rate else 1.0

    n = len(data)

    if method == "constant":
        trend = np.full(n, np.mean(data), dtype=np.float64)

    elif method == "linear":
        result = detect_trend(trace, sample_rate=fs)
        trend = result.trend_line

    elif method == "polynomial":
        t = np.arange(n)
        coeffs = np.polyfit(t, data, order)
        trend = np.polyval(coeffs, t)

    else:
        raise ValueError(f"Unknown method: {method}")

    detrended = data - trend

    if return_trend:
        return detrended, trend.astype(np.float64)
    return detrended


def moving_average(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    window_size: int,
    method: Literal["simple", "exponential", "weighted"] = "simple",
    alpha: float = 0.1,
) -> NDArray[np.float64]:
    """Compute moving average of signal.

    Smooths signal by averaging over sliding window.

    Args:
        trace: Input trace or numpy array.
        window_size: Size of averaging window in samples.
        method: Averaging method:
            - "simple": Simple moving average (default)
            - "exponential": Exponential moving average
            - "weighted": Linearly weighted moving average
        alpha: Smoothing factor for exponential method (0-1).

    Returns:
        Smoothed signal array (same length as input).

    Raises:
        ValueError: If method is not recognized.

    Example:
        >>> smoothed = moving_average(trace, window_size=10)
        >>> # Exponential smoothing
        >>> ema = moving_average(trace, window_size=10, method="exponential", alpha=0.2)
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data.astype(np.float64)
    else:
        data = np.array(trace, dtype=np.float64)

    n = len(data)

    window_size = min(window_size, n)

    if window_size < 1:
        return data.copy()

    if method == "simple":
        # Simple moving average using convolution
        kernel = np.ones(window_size) / window_size
        # Pad for same output length
        padded = np.pad(data, (window_size - 1, 0), mode="edge")
        result = np.convolve(padded, kernel, mode="valid")

    elif method == "exponential":
        # Exponential moving average
        result = np.zeros(n, dtype=np.float64)
        result[0] = data[0]
        for i in range(1, n):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]

    elif method == "weighted":
        # Linearly weighted moving average
        weights = np.arange(1, window_size + 1, dtype=np.float64)
        weights = weights / np.sum(weights)

        padded = np.pad(data, (window_size - 1, 0), mode="edge")
        result = np.convolve(padded, weights, mode="valid")

    else:
        raise ValueError(f"Unknown method: {method}")

    return result.astype(np.float64)


def detect_drift_segments(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    segment_size: int,
    threshold_slope: float | None = None,
    sample_rate: float | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    """Detect segments with significant drift.

    Divides signal into segments and identifies those with
    statistically significant linear trends.

    Args:
        trace: Input trace or numpy array.
        segment_size: Size of each segment in samples.
        threshold_slope: Minimum slope magnitude to flag (units/second).
            If None, uses statistical significance.
        sample_rate: Sample rate in Hz (required for array input).

    Returns:
        List of dictionaries describing drift segments:
            - start_sample: Start index of segment
            - end_sample: End index of segment
            - start_time: Start time in seconds
            - end_time: End time in seconds
            - slope: Trend slope
            - r_squared: Coefficient of determination

    Raises:
        ValueError: If trace is array and sample_rate is not provided.

    Example:
        >>> segments = detect_drift_segments(trace, segment_size=1000)
        >>> for seg in segments:
        ...     print(f"Drift at {seg['start_time']:.3f}s: {seg['slope']:.2e} V/s")
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data
        fs = trace.metadata.sample_rate
    else:
        data = trace
        if sample_rate is None:
            raise ValueError("sample_rate required when trace is array")
        fs = sample_rate

    n = len(data)
    drift_segments = []

    for start in range(0, n, segment_size):
        end = min(start + segment_size, n)

        if end - start < 10:  # Need minimum points for regression
            continue

        segment_data = data[start:end]
        segment_trace = segment_data  # Array

        result = detect_trend(segment_trace, sample_rate=fs)

        # Check if drift is significant
        is_drift = result.is_significant
        if threshold_slope is not None:
            is_drift = is_drift and abs(result.slope) >= threshold_slope

        if is_drift:
            drift_segments.append(
                {
                    "start_sample": start,
                    "end_sample": end,
                    "start_time": start / fs,
                    "end_time": end / fs,
                    "slope": result.slope,
                    "r_squared": result.r_squared,
                    "p_value": result.p_value,
                }
            )

    return drift_segments


def change_point_detection(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    min_segment_size: int = 10,
    penalty: float | None = None,
) -> list[int]:
    """Detect change points in signal level or trend.

    Identifies locations where the signal characteristics change
    significantly, using a simple CUSUM-based approach.

    Args:
        trace: Input trace or numpy array.
        min_segment_size: Minimum samples between change points.
        penalty: Penalty for adding change points (controls sensitivity).
            If None, auto-selected based on signal variance.

    Returns:
        List of sample indices where changes occur.

    Example:
        >>> change_points = change_point_detection(trace)
        >>> for cp in change_points:
        ...     print(f"Change at sample {cp}")
    """
    data = trace.data if isinstance(trace, WaveformTrace) else np.array(trace, dtype=np.float64)

    n = len(data)

    if n < 2 * min_segment_size:
        return []

    # Auto-select penalty if not provided
    if penalty is None:
        penalty = np.var(data) * 2

    # Simple binary segmentation using mean-shift cost
    change_points = []
    segments = [(0, n)]

    while segments:
        start, end = segments.pop(0)
        segment = data[start:end]
        seg_len = len(segment)

        if seg_len < 2 * min_segment_size:
            continue

        # Find best split point
        best_cost_reduction = -np.inf
        best_split = None

        for split in range(min_segment_size, seg_len - min_segment_size):
            left = segment[:split]
            right = segment[split:]

            # Cost = sum of squared deviations from segment mean
            cost_whole = np.sum((segment - np.mean(segment)) ** 2)
            cost_left = np.sum((left - np.mean(left)) ** 2)
            cost_right = np.sum((right - np.mean(right)) ** 2)

            cost_reduction = cost_whole - (cost_left + cost_right) - penalty

            if cost_reduction > best_cost_reduction:
                best_cost_reduction = cost_reduction
                best_split = split

        # If significant cost reduction, add change point
        if best_split is not None and best_cost_reduction > 0:
            cp = start + best_split
            change_points.append(cp)

            # Add new segments to process
            segments.append((start, cp))
            segments.append((cp, end))

    change_points.sort()
    return change_points


def piecewise_linear_fit(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    n_segments: int = 3,
    sample_rate: float | None = None,
) -> dict:  # type: ignore[type-arg]
    """Fit piecewise linear model to signal.

    Divides signal into segments and fits linear trends to each.

    Args:
        trace: Input trace or numpy array.
        n_segments: Number of segments to fit.
        sample_rate: Sample rate in Hz (required for array input).

    Returns:
        Dictionary with fit results:
            - breakpoints: Sample indices of segment boundaries
            - segments: List of (slope, intercept) for each segment
            - fitted: Full fitted signal
            - residuals: Fitting residuals

    Raises:
        ValueError: If trace is array and sample_rate is not provided.

    Example:
        >>> result = piecewise_linear_fit(trace, n_segments=4)
        >>> print(f"Breakpoints: {result['breakpoints']}")
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data
        fs = trace.metadata.sample_rate
    else:
        data = np.array(trace, dtype=np.float64)
        if sample_rate is None:
            raise ValueError("sample_rate required when trace is array")
        fs = sample_rate

    n = len(data)

    # Calculate segment boundaries
    segment_size = n // n_segments
    breakpoints = [i * segment_size for i in range(1, n_segments)]
    breakpoints = [0, *breakpoints, n]

    # Fit each segment
    segments = []
    fitted = np.zeros(n, dtype=np.float64)

    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]

        segment_data = data[start:end]
        t = np.arange(len(segment_data)) / fs

        if len(t) >= 2:
            slope, intercept = np.polyfit(t, segment_data, 1)
            fitted[start:end] = intercept + slope * t
            segments.append(
                {
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "start": start,
                    "end": end,
                }
            )

    residuals = data - fitted

    return {
        "breakpoints": breakpoints,
        "segments": segments,
        "fitted": fitted,
        "residuals": residuals,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }


__all__ = [
    "TrendResult",
    "change_point_detection",
    "detect_drift_segments",
    "detect_trend",
    "detrend",
    "moving_average",
    "piecewise_linear_fit",
]
