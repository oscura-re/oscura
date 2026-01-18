"""Intelligent trace comparison for auto-discovery.

This module provides automatic trace comparison with alignment, difference
detection, and plain-language explanations.


Example:
    >>> from oscura.discovery import compare_traces
    >>> diff = compare_traces(trace1, trace2)
    >>> for d in diff.differences:
    ...     print(f"{d.category}: {d.description}")

References:
    Oscura Auto-Discovery Specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import signal as sp_signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class Difference:
    """Individual difference between traces.

    Attributes:
        category: Difference category (timing, amplitude, pattern, transitions).
        timestamp_us: Timestamp in microseconds.
        description: Plain language explanation.
        severity: Severity level (CRITICAL, WARNING, INFO).
        impact_score: Impact score (0.0-1.0, higher = more severe).
        expected_value: Expected value from reference.
        actual_value: Actual value from measured trace.
        delta_value: Absolute difference.
        delta_percent: Percentage difference.
        confidence: Confidence in this difference detection.
    """

    category: str
    timestamp_us: float
    description: str
    severity: str
    impact_score: float
    expected_value: float | None = None
    actual_value: float | None = None
    delta_value: float | None = None
    delta_percent: float | None = None
    confidence: float = 1.0


@dataclass
class TraceDiff:
    """Result of intelligent trace comparison.

    Attributes:
        summary: High-level summary of comparison.
        alignment_method: Method used to align traces.
        similarity_score: Overall similarity (0.0-1.0).
        differences: List of detected differences, sorted by impact.
        visual_path: Path to generated visual comparison (if created).
        stats: Statistical comparison metrics.
    """

    summary: str
    alignment_method: str
    similarity_score: float
    differences: list[Difference] = field(default_factory=list)
    visual_path: str | None = None
    stats: dict | None = None  # type: ignore[type-arg]


def _align_time_based(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Align traces based on time (sync to t=0).

    Args:
        trace1: First trace.
        trace2: Second trace.

    Returns:
        Tuple of (data1, data2, offset_samples).
    """
    # Simply align to start (t=0)
    min_len = min(len(trace1.data), len(trace2.data))
    data1 = trace1.data[:min_len].astype(np.float64)
    data2 = trace2.data[:min_len].astype(np.float64)

    return data1, data2, 0


def _align_trigger_based(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    threshold_pct: float = 50.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Align traces based on trigger point (first edge).

    Args:
        trace1: First trace.
        trace2: Second trace.
        threshold_pct: Threshold percentage for edge detection.

    Returns:
        Tuple of (data1, data2, offset_samples).
    """
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Find first significant edge in each trace
    range1 = np.ptp(data1)
    range2 = np.ptp(data2)

    threshold1 = np.min(data1) + range1 * threshold_pct / 100.0
    threshold2 = np.min(data2) + range2 * threshold_pct / 100.0

    # Find first crossing
    idx1 = np.where(data1 > threshold1)[0]
    idx2 = np.where(data2 > threshold2)[0]

    offset1 = idx1[0] if len(idx1) > 0 else 0
    offset2 = idx2[0] if len(idx2) > 0 else 0

    # Align to earliest trigger
    if offset1 <= offset2:
        offset_samples = offset2 - offset1
        data1_aligned = data1[offset1:]
        data2_aligned = data2[offset2:]
    else:
        offset_samples = offset1 - offset2
        data1_aligned = data1[offset1:]
        data2_aligned = data2[offset2:]

    # Truncate to same length
    min_len = min(len(data1_aligned), len(data2_aligned))
    return data1_aligned[:min_len], data2_aligned[:min_len], offset_samples


def _align_pattern_based(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """Align traces using cross-correlation.

    Args:
        trace1: First trace.
        trace2: Second trace.

    Returns:
        Tuple of (data1, data2, offset_samples).
    """
    data1 = trace1.data.astype(np.float64)
    data2 = trace2.data.astype(np.float64)

    # Normalize for correlation
    data1_norm = (data1 - np.mean(data1)) / (np.std(data1) + 1e-10)
    data2_norm = (data2 - np.mean(data2)) / (np.std(data2) + 1e-10)

    # Cross-correlation
    correlation = sp_signal.correlate(data1_norm, data2_norm, mode="full")

    # Find peak
    peak_idx = np.argmax(np.abs(correlation))
    offset_samples = peak_idx - (len(data2) - 1)

    # Align based on offset
    if offset_samples >= 0:
        data1_aligned = data1[offset_samples:]
        data2_aligned = data2
    else:
        data1_aligned = data1
        data2_aligned = data2[-offset_samples:]

    # Truncate to same length
    min_len = min(len(data1_aligned), len(data2_aligned))
    return data1_aligned[:min_len], data2_aligned[:min_len], int(offset_samples)


def _detect_timing_differences(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    sample_rate: float,
) -> list[Difference]:
    """Detect timing differences between aligned traces.

    Args:
        data1: First trace data.
        data2: Second trace data.
        sample_rate: Sample rate in Hz.

    Returns:
        List of timing differences.
    """
    differences = []

    # Look for timing shifts in edges
    # Compute derivatives to find edges
    diff1 = np.diff(data1)
    diff2 = np.diff(data2)

    # Find significant edges (> 10% of range per sample)
    range1 = np.ptp(data1)
    range2 = np.ptp(data2)

    edge_threshold1 = range1 * 0.1
    edge_threshold2 = range2 * 0.1

    edges1 = np.where(np.abs(diff1) > edge_threshold1)[0]
    edges2 = np.where(np.abs(diff2) > edge_threshold2)[0]

    # Compare edge counts
    if abs(len(edges1) - len(edges2)) > 2:
        delta_edges = abs(len(edges1) - len(edges2))
        timestamp_us = 0.0

        differences.append(
            Difference(
                category="timing",
                timestamp_us=timestamp_us,
                description=f"Trace 1 has {len(edges1)} transitions while Trace 2 has {len(edges2)} transitions (difference: {delta_edges})",
                severity="WARNING" if delta_edges > 5 else "INFO",
                impact_score=min(1.0, delta_edges / 10.0),
                confidence=0.90,
            )
        )

    return differences


def _detect_amplitude_differences(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    sample_rate: float,
) -> list[Difference]:
    """Detect amplitude differences between aligned traces.

    Args:
        data1: First trace data.
        data2: Second trace data.
        sample_rate: Sample rate in Hz.

    Returns:
        List of amplitude differences.
    """
    differences = []  # type: ignore[var-annotated]

    # Compute amplitude difference
    amp_diff = np.abs(data1 - data2)
    ref_range = np.ptp(data2)

    if ref_range == 0:
        return differences

    # Find points with significant amplitude difference
    threshold = ref_range * 0.05  # 5% of swing

    significant_diffs = np.where(amp_diff > threshold)[0]

    if len(significant_diffs) > len(data1) * 0.1:  # More than 10% of samples
        max_diff_idx = np.argmax(amp_diff)
        max_diff = amp_diff[max_diff_idx]
        timestamp_us = max_diff_idx / sample_rate * 1e6

        delta_percent = (max_diff / ref_range) * 100.0

        severity = "CRITICAL" if delta_percent > 20 else "WARNING" if delta_percent > 5 else "INFO"

        differences.append(
            Difference(
                category="amplitude",
                timestamp_us=float(timestamp_us),
                description=f"Voltage differs by {max_diff:.3f}V ({delta_percent:.1f}% of signal swing)",
                severity=severity,
                impact_score=min(1.0, delta_percent / 20.0),
                expected_value=float(data2[max_diff_idx]),
                actual_value=float(data1[max_diff_idx]),
                delta_value=float(max_diff),
                delta_percent=delta_percent,
                confidence=0.95,
            )
        )

    return differences


def _detect_pattern_differences(
    data1: NDArray[np.float64],
    data2: NDArray[np.float64],
    sample_rate: float,
) -> list[Difference]:
    """Detect pattern differences between aligned traces.

    Args:
        data1: First trace data.
        data2: Second trace data.
        sample_rate: Sample rate in Hz.

    Returns:
        List of pattern differences.
    """
    differences = []  # type: ignore[var-annotated]

    # Compute correlation
    if len(data1) < 2:
        return differences

    data1_norm = (data1 - np.mean(data1)) / (np.std(data1) + 1e-10)
    data2_norm = (data2 - np.mean(data2)) / (np.std(data2) + 1e-10)

    correlation = np.corrcoef(data1_norm, data2_norm)[0, 1]

    if correlation < 0.95:
        severity = "CRITICAL" if correlation < 0.8 else "WARNING" if correlation < 0.95 else "INFO"

        differences.append(
            Difference(
                category="pattern",
                timestamp_us=0.0,
                description=f"Signal patterns differ (correlation: {correlation:.2f}, expected: >0.95)",
                severity=severity,
                impact_score=1.0 - correlation,
                confidence=0.88,
            )
        )

    return differences


def compare_traces(
    trace1: WaveformTrace,
    trace2: WaveformTrace,
    *,
    alignment: Literal["time", "trigger", "pattern", "auto"] = "auto",
    difference_types: list[str] | None = None,
    severity_threshold: str | None = None,
) -> TraceDiff:
    """Compare traces with intelligent alignment and difference detection.

    Automatically aligns traces and identifies timing, amplitude, pattern,
    and transition differences with plain-language explanations.

    Args:
        trace1: First trace (typically measured/actual).
        trace2: Second trace (typically reference/expected).
        alignment: Alignment method:
            - "time": Sync to t=0
            - "trigger": Sync to first edge (≥50% swing)
            - "pattern": Cross-correlation alignment
            - "auto": Try all methods, use best
        difference_types: Types to detect (default: all).
        severity_threshold: Only return differences at or above this level.

    Returns:
        TraceDiff with alignment method, differences, and summary.

    Example:
        >>> diff = compare_traces(measured, golden)
        >>> for d in diff.differences[:5]:
        ...     print(f"{d.severity}: {d.description}")

    References:
        DISC-004: Intelligent Trace Comparison
    """
    difference_types = difference_types or [
        "timing",
        "amplitude",
        "pattern",
        "transitions",
    ]

    # Try alignment methods
    if alignment == "auto":
        # Try all methods and pick best correlation
        methods = ["time", "trigger", "pattern"]
        best_corr = -1
        best_method = "time"
        best_aligned = None

        for method in methods:
            if method == "time":
                d1, d2, offset = _align_time_based(trace1, trace2)
            elif method == "trigger":
                d1, d2, offset = _align_trigger_based(trace1, trace2)
            else:  # pattern
                d1, d2, offset = _align_pattern_based(trace1, trace2)

            # Compute correlation
            if len(d1) > 1:
                d1_norm = (d1 - np.mean(d1)) / (np.std(d1) + 1e-10)
                d2_norm = (d2 - np.mean(d2)) / (np.std(d2) + 1e-10)
                corr = np.corrcoef(d1_norm, d2_norm)[0, 1]

                if corr > best_corr:
                    best_corr = corr
                    best_method = method
                    best_aligned = (d1, d2, offset)

        data1, data2, offset = best_aligned  # type: ignore[misc]
        alignment_method = f"{best_method}-based"
    else:
        # Use specified method
        if alignment == "time":
            data1, data2, offset = _align_time_based(trace1, trace2)
        elif alignment == "trigger":
            data1, data2, offset = _align_trigger_based(trace1, trace2)
        else:  # pattern
            data1, data2, offset = _align_pattern_based(trace1, trace2)

        alignment_method = f"{alignment}-based"

    sample_rate = trace1.metadata.sample_rate

    # Detect differences
    all_differences = []

    if "timing" in difference_types:
        all_differences.extend(_detect_timing_differences(data1, data2, sample_rate))

    if "amplitude" in difference_types:
        all_differences.extend(_detect_amplitude_differences(data1, data2, sample_rate))

    if "pattern" in difference_types:
        all_differences.extend(_detect_pattern_differences(data1, data2, sample_rate))

    # Sort by impact score (descending)
    all_differences.sort(key=lambda d: d.impact_score, reverse=True)

    # Filter by severity threshold
    if severity_threshold:
        severity_order = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}
        threshold_level = severity_order.get(severity_threshold, 0)

        filtered = [
            d for d in all_differences if severity_order.get(d.severity, 0) >= threshold_level
        ]
        all_differences = filtered

    # Compute similarity score
    if len(data1) > 1:
        data1_norm = (data1 - np.mean(data1)) / (np.std(data1) + 1e-10)
        data2_norm = (data2 - np.mean(data2)) / (np.std(data2) + 1e-10)
        correlation = np.corrcoef(data1_norm, data2_norm)[0, 1]
        similarity_score = float((correlation + 1) / 2)  # Map [-1,1] to [0,1]
    else:
        similarity_score = 1.0 if len(data1) == 0 or data1[0] == data2[0] else 0.0

    # Build summary
    if similarity_score > 0.95:
        summary = "Signals are very similar"
    elif similarity_score > 0.85:
        summary = "Signals are similar with minor differences"
    elif similarity_score > 0.70:
        summary = "Signals show moderate differences"
    else:
        summary = "Signals are significantly different"

    critical_count = sum(1 for d in all_differences if d.severity == "CRITICAL")
    warning_count = sum(1 for d in all_differences if d.severity == "WARNING")

    if critical_count > 0:
        summary += f" ({critical_count} critical issue(s))"
    elif warning_count > 0:
        summary += f" ({warning_count} warning(s))"

    # Statistics
    stats = {
        "correlation": float(correlation) if len(data1) > 1 else 1.0,
        "rms_error": float(np.sqrt(np.mean((data1 - data2) ** 2))),
        "max_deviation": float(np.max(np.abs(data1 - data2))),
        "max_deviation_time": float(np.argmax(np.abs(data1 - data2)) / sample_rate),
        "avg_timing_offset": float(offset / sample_rate * 1e9),  # ns
    }

    return TraceDiff(
        summary=summary,
        alignment_method=alignment_method,
        similarity_score=similarity_score,
        differences=all_differences,
        stats=stats,
    )


__all__ = [
    "Difference",
    "TraceDiff",
    "compare_traces",
]
