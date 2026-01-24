"""Anomaly detection in signal traces.

This module provides automated detection of glitches, timing violations,
and protocol errors with context extraction for debugging.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def find_anomalies(
    trace: NDArray[np.float64],
    anomaly_type: str = "glitch",
    *,
    threshold: float | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    sample_rate: float | None = None,
    context_samples: int = 100,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Find glitches, timing violations, or protocol errors in traces.

    Anomaly detection with context extraction.
    Integrates with QUAL-005 glitch detection for signal quality analysis.

    Args:
        trace: Input signal trace
        anomaly_type: Type of anomaly to detect:
            - 'glitch': Short-duration voltage spikes/dips
            - 'timing': Edge timing violations (requires sample_rate)
            - 'protocol': Protocol-level errors (requires decoded data)
        threshold: Detection threshold. Meaning depends on anomaly_type:
            - glitch: Voltage deviation from expected level
            - timing: Timing violation threshold in seconds
        min_width: Minimum anomaly width in seconds (requires sample_rate)
        max_width: Maximum anomaly width in seconds (requires sample_rate)
        sample_rate: Sample rate in Hz (required for timing analysis)
        context_samples: Number of samples to include before/after anomaly
            for context extraction (default: 100)
        **kwargs: Additional type-specific parameters

    Returns:
        List of anomaly dictionaries, each containing:
        - index: Sample index where anomaly occurs
        - type: Anomaly type
        - severity: Severity score (0-1, higher is worse)
        - duration: Duration in samples
        - amplitude: Amplitude deviation (for glitches)
        - context: ±context_samples around anomaly
        - description: Human-readable description

    Raises:
        ValueError: If invalid anomaly_type or missing required parameters

    Examples:
        >>> # Detect voltage glitches
        >>> trace = np.array([0, 0, 0, 0.8, 0, 0, 0])  # Spike at index 3
        >>> anomalies = find_anomalies(
        ...     trace,
        ...     anomaly_type='glitch',
        ...     threshold=0.5,
        ...     sample_rate=1e6
        ... )
        >>> print(f"Found {len(anomalies)} glitches")

        >>> # Detect timing violations
        >>> anomalies = find_anomalies(
        ...     trace,
        ...     anomaly_type='timing',
        ...     min_width=10e-9,  # 10 ns minimum
        ...     max_width=100e-9,  # 100 ns maximum
        ...     sample_rate=1e9
        ... )

    Notes:
        - Glitch detection uses derivative and threshold methods
        - Timing detection requires sample_rate for width calculations
        - Context extraction handles edge cases at trace boundaries
        - Integrates with QUAL-005 for comprehensive signal quality analysis

    References:
        SRCH-002: Anomaly Search
        QUAL-005: Glitch Detection
    """
    if trace.size == 0:
        return []

    valid_types = {"glitch", "timing", "protocol"}
    if anomaly_type not in valid_types:
        raise ValueError(f"Invalid anomaly_type '{anomaly_type}'. Must be one of: {valid_types}")

    anomalies: list[dict[str, Any]] = []

    if anomaly_type == "glitch":
        anomalies = _detect_glitches(
            trace,
            threshold=threshold,
            min_width=min_width,
            max_width=max_width,
            sample_rate=sample_rate,
            context_samples=context_samples,
        )

    elif anomaly_type == "timing":
        if sample_rate is None:
            raise ValueError("sample_rate required for timing anomaly detection")

        anomalies = _detect_timing_violations(
            trace,
            sample_rate=sample_rate,
            min_width=min_width,
            max_width=max_width,
            context_samples=context_samples,
        )

    elif anomaly_type == "protocol":
        # Protocol error detection would integrate with protocol decoders
        # For now, return empty list with note
        anomalies = []

    return anomalies


def _detect_glitches(
    trace: NDArray[np.float64],
    threshold: float | None,
    min_width: float | None,
    max_width: float | None,
    sample_rate: float | None,
    context_samples: int,
) -> list[dict[str, Any]]:
    """Detect voltage glitches using derivative method."""
    glitches: list[dict[str, Any]] = []

    # Auto-threshold if not provided
    threshold_value: float
    if threshold is None:
        # Use 3 sigma as default threshold
        threshold_value = float(3 * np.std(trace))
    else:
        threshold_value = threshold

    # Compute derivative to find rapid changes
    derivative = np.diff(trace)
    abs_derivative = np.abs(derivative)

    # Find points where derivative exceeds threshold
    glitch_candidates = np.where(abs_derivative > threshold_value)[0]

    if len(glitch_candidates) == 0:
        return glitches

    # Group consecutive points into glitch events
    glitch_groups = []
    current_group = [glitch_candidates[0]]

    for idx in glitch_candidates[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            glitch_groups.append(current_group)
            current_group = [idx]

    if current_group:
        glitch_groups.append(current_group)

    # Compute baseline once for all glitches (performance optimization)
    # For very large arrays (>1M samples), use percentile approximation
    if len(trace) > 1_000_000:
        # Fast approximation: 50th percentile with linear interpolation
        baseline = float(np.percentile(trace, 50, method="linear"))
    else:
        baseline = float(np.median(trace))

    # Filter by width if specified
    for group in glitch_groups:
        start_idx = group[0]
        end_idx = group[-1] + 1
        duration_samples = end_idx - start_idx

        # Check width constraints
        if sample_rate is not None:
            duration_seconds = duration_samples / sample_rate

            if min_width is not None and duration_seconds < min_width:
                continue
            if max_width is not None and duration_seconds > max_width:
                continue

        # Extract context
        ctx_start = max(0, start_idx - context_samples)
        ctx_end = min(len(trace), end_idx + context_samples)
        context = trace[ctx_start:ctx_end].copy()

        # Compute amplitude deviation (baseline computed once above)
        amplitude = np.max(np.abs(trace[start_idx:end_idx] - baseline))

        # Severity: normalized amplitude
        severity = min(1.0, amplitude / (threshold_value * 3))

        glitches.append(
            {
                "index": start_idx,
                "type": "glitch",
                "severity": float(severity),
                "duration": duration_samples,
                "amplitude": float(amplitude),
                "context": context,
                "description": f"Glitch at sample {start_idx}, amplitude {amplitude:.3g}",
            }
        )

    return glitches


def _detect_timing_violations(
    trace: NDArray[np.float64],
    sample_rate: float,
    min_width: float | None,
    max_width: float | None,
    context_samples: int,
) -> list[dict[str, Any]]:
    """Detect timing violations (pulse width violations)."""
    violations = []

    # Simple threshold for digital signal
    threshold = (np.max(trace) + np.min(trace)) / 2
    digital = (trace >= threshold).astype(int)

    # Find edges
    edges = np.diff(digital)
    rising_edges = np.where(edges == 1)[0]
    falling_edges = np.where(edges == -1)[0]

    # Measure pulse widths
    for rise in rising_edges:
        # Find next falling edge
        next_fall = falling_edges[falling_edges > rise]
        if len(next_fall) == 0:
            continue

        fall = next_fall[0]
        pulse_width_samples = fall - rise
        pulse_width_seconds = pulse_width_samples / sample_rate

        # Check violations
        violated = False
        violation_type = ""

        if min_width is not None and pulse_width_seconds < min_width:
            violated = True
            violation_type = "too_short"

        if max_width is not None and pulse_width_seconds > max_width:
            violated = True
            violation_type = "too_long"

        if violated:
            # Extract context
            ctx_start = max(0, rise - context_samples)
            ctx_end = min(len(trace), fall + context_samples)
            context = trace[ctx_start:ctx_end].copy()

            # Severity based on deviation
            if min_width is not None:
                deviation = abs(pulse_width_seconds - min_width) / min_width
            elif max_width is not None:
                deviation = abs(pulse_width_seconds - max_width) / max_width
            else:
                deviation = 0.0

            severity = min(1.0, deviation)

            violations.append(
                {
                    "index": rise,
                    "type": f"timing_{violation_type}",
                    "severity": float(severity),
                    "duration": pulse_width_samples,
                    "amplitude": float(pulse_width_seconds),
                    "context": context,
                    "description": (
                        f"Timing violation at sample {rise}: "
                        f"pulse width {pulse_width_seconds * 1e9:.1f} ns ({violation_type})"
                    ),
                }
            )

    return violations
