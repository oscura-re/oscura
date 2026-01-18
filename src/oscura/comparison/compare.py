"""Trace comparison functions for Oscura.

This module provides functions for comparing waveform traces including
difference calculation, correlation, and similarity scoring.


Example:
    >>> from oscura.comparison import compare_traces, similarity_score
    >>> result = compare_traces(trace1, trace2)
    >>> score = similarity_score(trace1, trace2)

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal as sp_signal
from scipy import stats

from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ComparisonResult:
    """Result of a trace comparison operation.

    Attributes:
        match: True if traces are considered matching.
        similarity: Similarity score (0.0 to 1.0).
        max_difference: Maximum absolute difference.
        rms_difference: RMS of the difference.
        correlation: Correlation coefficient.
        difference_trace: Difference waveform (optional).
        violations: Indices where difference exceeds threshold.
        statistics: Additional comparison statistics.
    """

    match: bool
    similarity: float
    max_difference: float
    rms_difference: float
    correlation: float
    difference_trace: WaveformTrace | None = None
    violations: NDArray[np.int64] | None = None
    statistics: dict | None = None  # type: ignore[type-arg]


def difference(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    normalize: bool = False,
    channel_name: str | None = None,
) -> WaveformTrace:
    """Compute difference between two traces.

    Calculates the element-wise difference (trace1 - trace2). Traces
    are aligned to the shorter length.

    Args:
        trace1: First trace.
        trace2: Second trace.
        normalize: Normalize difference to percentage of reference range.
        channel_name: Name for the result trace.

    Returns:
        WaveformTrace containing the difference.

    Raises:
        ValueError: If input traces contain NaN or Inf values.

    Example:
        >>> diff = difference(measured, reference)
        >>> max_error = np.max(np.abs(diff.data))
    """
    # Get data
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Check for NaN/Inf values
    if np.any(~np.isfinite(data1)) or np.any(~np.isfinite(data2)):
        raise ValueError("Input traces contain NaN or Inf values")

    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Compute difference
    diff = data1 - data2

    if normalize:
        # Normalize to percentage of reference range
        ref_range = np.ptp(data2)
        if ref_range > 0:
            diff = (diff / ref_range) * 100.0

    new_metadata = TraceMetadata(
        sample_rate=trace1.metadata.sample_rate,
        vertical_scale=None,
        vertical_offset=None,
        acquisition_time=trace1.metadata.acquisition_time,
        trigger_info=trace1.metadata.trigger_info,
        source_file=trace1.metadata.source_file,
        channel_name=channel_name or "difference",
    )

    return WaveformTrace(data=diff, metadata=new_metadata)


def correlation(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    mode: Literal["full", "same", "valid"] = "same",
    normalize: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute cross-correlation between two traces.

    Calculates the cross-correlation of two waveforms, useful for
    finding time delays and pattern matching.

    Args:
        trace1: First trace.
        trace2: Second trace.
        mode: Correlation mode:
            - "full": Full correlation (length N+M-1)
            - "same": Same length as longer input
            - "valid": Only overlapping region
        normalize: Normalize to correlation coefficient (-1 to 1).

    Returns:
        Tuple of (lags, correlation_values).

    Example:
        >>> lags, corr = correlation(trace1, trace2)
        >>> delay = lags[np.argmax(corr)]
    """
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    if normalize:
        # Normalize inputs
        data1 = (data1 - np.mean(data1)) / (np.std(data1) + 1e-10)
        data2 = (data2 - np.mean(data2)) / (np.std(data2) + 1e-10)

    # Compute cross-correlation
    corr = sp_signal.correlate(data1, data2, mode=mode)

    if normalize:
        # Normalize by length for correlation coefficient
        corr = corr / len(data1)

    # Compute lag axis in samples
    if mode == "full":
        lags = np.arange(-(len(data2) - 1), len(data1))
    elif mode == "same":
        lags = np.arange(-len(data1) // 2, len(data1) - len(data1) // 2)
    else:  # valid
        lags = np.arange(0, len(data1) - len(data2) + 1)

    return lags.astype(np.float64), corr


def similarity_score(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    method: Literal["correlation", "rms", "mse", "cosine"] = "correlation",
    normalize_amplitude: bool = True,
    normalize_offset: bool = True,
) -> float:
    """Compute similarity score between two traces.

    Returns a score from 0.0 (completely different) to 1.0 (identical).

    Args:
        trace1: First trace.
        trace2: Second trace.
        method: Similarity metric:
            - "correlation": Pearson correlation coefficient (default)
            - "rms": 1 - normalized RMS difference
            - "mse": 1 - normalized mean squared error
            - "cosine": Cosine similarity
        normalize_amplitude: Normalize amplitude before comparison.
        normalize_offset: Remove DC offset before comparison.

    Returns:
        Similarity score (0.0 to 1.0).

    Raises:
        ValueError: If input traces contain NaN or Inf values.

    Example:
        >>> score = similarity_score(measured, reference)
        >>> if score > 0.95:
        ...     print("Traces match")
    """
    # Get data
    data1 = trace1.data.astype(np.float64).copy()
    data2 = trace2.data.astype(np.float64).copy()

    # Check for NaN/Inf values
    if np.any(~np.isfinite(data1)) or np.any(~np.isfinite(data2)):
        raise ValueError("Input traces contain NaN or Inf values")

    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Normalize offset (remove DC)
    if normalize_offset:
        data1 = data1 - np.mean(data1)
        data2 = data2 - np.mean(data2)

    # Normalize amplitude
    if normalize_amplitude:
        std1 = np.std(data1)
        std2 = np.std(data2)
        if std1 > 0:
            data1 = data1 / std1
        if std2 > 0:
            data2 = data2 / std2

    if method == "correlation":
        # Pearson correlation coefficient
        # Handle constant inputs gracefully
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
            try:
                r, _ = stats.pearsonr(data1, data2)
                # Handle NaN result (constant traces after normalization)
                if np.isnan(r):
                    # If both traces are constant and identical, perfect match
                    if np.allclose(data1, data2, equal_nan=False):
                        r = 1.0
                    else:
                        r = 0.0
            except Exception:
                r = 0.0
        # Map from [-1, 1] to [0, 1]
        return float((r + 1) / 2)

    elif method == "rms":
        # RMS-based similarity
        rms_diff = np.sqrt(np.mean((data1 - data2) ** 2))
        rms_ref = np.sqrt(np.mean(data2**2)) + 1e-10
        return float(max(0, 1 - rms_diff / rms_ref))

    elif method == "mse":
        # MSE-based similarity
        mse = np.mean((data1 - data2) ** 2)
        var_ref = np.var(data2) + 1e-10
        return float(max(0, 1 - mse / var_ref))

    elif method == "cosine":
        # Cosine similarity
        dot = np.dot(data1, data2)
        norm1 = np.linalg.norm(data1) + 1e-10
        norm2 = np.linalg.norm(data2) + 1e-10
        cosine = dot / (norm1 * norm2)
        # Map from [-1, 1] to [0, 1]
        return float((cosine + 1) / 2)

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def compare_traces(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    tolerance: float | None = None,
    tolerance_pct: float | None = None,
    method: Literal["absolute", "relative", "statistical"] = "absolute",
    include_difference: bool = True,
) -> ComparisonResult:
    """Compare two traces and determine if they match.

    Comprehensive comparison of two waveforms including difference
    analysis, correlation, and match determination.

    Args:
        trace1: First trace (typically measured).
        trace2: Second trace (typically reference).
        tolerance: Absolute tolerance for matching.
        tolerance_pct: Percentage tolerance (0-100) relative to reference range.
        method: Comparison method:
            - "absolute": Compare absolute values
            - "relative": Compare relative to reference
            - "statistical": Use statistical tests
        include_difference: Include difference trace in result.

    Returns:
        ComparisonResult with match status and statistics.

    Raises:
        ValueError: If method is unknown.

    Example:
        >>> result = compare_traces(measured, golden, tolerance=0.01)
        >>> if result.match:
        ...     print(f"Match! Similarity: {result.similarity:.1%}")
    """
    # Get data
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Align lengths
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Compute difference
    diff = data1 - data2

    # Compute statistics
    max_diff = float(np.max(np.abs(diff)))
    rms_diff = float(np.sqrt(np.mean(diff**2)))

    # Compute correlation
    if len(data1) > 1:
        # Handle constant inputs (e.g., DC signals) gracefully
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
            try:
                corr, _ = stats.pearsonr(data1, data2)
            except Exception:
                # Fallback for any correlation computation issues
                corr = 0.0
    else:
        corr = 1.0 if data1[0] == data2[0] else 0.0

    # Compute similarity score
    sim_score = similarity_score(trace1, trace2)

    # Determine tolerance
    if tolerance is None and tolerance_pct is not None:
        ref_range = float(np.ptp(data2))
        tolerance = ref_range * tolerance_pct / 100.0
    elif tolerance is None:
        # Default: 1% of reference range
        ref_range = float(np.ptp(data2))
        tolerance = ref_range * 0.01

    # Find violations
    violations = np.where(np.abs(diff) > tolerance)[0]

    # Determine match
    if method == "absolute":
        match = max_diff <= tolerance
    elif method == "relative":
        ref_range = float(np.ptp(data2)) + 1e-10
        relative_max = max_diff / ref_range
        match = relative_max <= (tolerance_pct or 1.0) / 100.0
    elif method == "statistical":
        # Use t-test for statistical matching
        _, p_value = stats.ttest_rel(data1, data2)
        match = p_value > 0.05  # No significant difference
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create difference trace if requested
    diff_trace = None
    if include_difference:
        diff_trace = difference(trace1, trace2, channel_name="comparison_diff")

    # Compute additional statistics
    statistics = {
        "mean_difference": float(np.mean(diff)),
        "std_difference": float(np.std(diff)),
        "median_difference": float(np.median(diff)),
        "num_violations": len(violations),
        "violation_rate": len(violations) / min_len if min_len > 0 else 0,
        "p_value": float(stats.ttest_rel(data1, data2)[1]) if len(data1) > 1 else 1.0,
    }

    return ComparisonResult(
        match=match,
        similarity=sim_score,
        max_difference=max_diff,
        rms_difference=rms_diff,
        correlation=float(corr),
        difference_trace=diff_trace,
        violations=violations if len(violations) > 0 else None,
        statistics=statistics,
    )
