"""Automatic anomaly detection and highlighting.

This module detects unusual signal features (glitches, dropouts, noise
spikes, timing violations) to guide user attention.


Example:
    >>> from oscura.discovery import find_anomalies
    >>> anomalies = find_anomalies(trace)
    >>> for anom in anomalies:
    ...     print(f"{anom.timestamp_us:.2f}us: {anom.type} - {anom.description}")

References:
    IEEE 1057-2017: Digitizing Waveform Recorders
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.analyzers.statistics.basic import basic_stats
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray

AnomalyType = Literal[
    "glitch",
    "dropout",
    "noise_spike",
    "timing_violation",
    "ringing",
    "overshoot",
    "undershoot",
]

Severity = Literal["CRITICAL", "WARNING", "INFO"]


@dataclass
class Anomaly:
    """Detected signal anomaly.

    Represents an unusual or interesting signal feature with timing,
    classification, and plain-language explanation.

    Attributes:
        timestamp_us: Anomaly start time in microseconds.
        type: Type of anomaly detected.
        severity: Impact level (CRITICAL, WARNING, INFO).
        description: Plain-language explanation.
        duration_ns: Duration in nanoseconds.
        confidence: Detection confidence (0.0-1.0).
        metadata: Additional type-specific information.

    Example:
        >>> anomaly = Anomaly(
        ...     timestamp_us=45.23,
        ...     type="glitch",
        ...     severity="WARNING",
        ...     description="Brief 35ns pulse, likely noise spike",
        ...     duration_ns=35.0,
        ...     confidence=0.92
        ... )
    """

    timestamp_us: float
    type: AnomalyType
    severity: Severity
    description: str
    duration_ns: float = 0.0
    confidence: float = 1.0
    metadata: dict[str, float] = field(default_factory=dict)


def find_anomalies(
    trace: WaveformTrace | DigitalTrace,
    *,
    severity_filter: list[Severity] | None = None,
    min_confidence: float = 0.7,
    anomaly_types: list[AnomalyType] | None = None,
) -> list[Anomaly]:
    """Detect anomalies in signal automatically.

    Identifies glitches, dropouts, noise spikes, timing violations, ringing,
    and overshoot/undershoot without requiring user configuration.

    Args:
        trace: Input waveform or digital trace.
        severity_filter: Only return specified severity levels (default: all).
        min_confidence: Minimum confidence threshold (0.0-1.0).
        anomaly_types: Specific anomaly types to detect (default: all).

    Returns:
        List of detected Anomaly objects, sorted by timestamp.

    Raises:
        ValueError: If trace is empty or invalid.

    Example:
        >>> anomalies = find_anomalies(trace, severity_filter=['CRITICAL', 'WARNING'])
        >>> print(f"Found {len(anomalies)} critical/warning anomalies")
        >>> for anom in anomalies[:5]:
        ...     print(f"  {anom.timestamp_us:.2f}us: {anom.type} - {anom.description}")

    References:
        DISC-002: Anomaly Highlighting
    """
    # Validate input
    if len(trace) == 0:
        raise ValueError("Cannot detect anomalies in empty trace")

    # Get signal data
    if isinstance(trace, WaveformTrace):
        data = trace.data
        sample_rate = trace.metadata.sample_rate
    else:
        data = trace.data.astype(np.float64)
        sample_rate = trace.metadata.sample_rate

    # Compute basic statistics for reference
    stats = basic_stats(data)
    voltage_swing = stats["max"] - stats["min"]

    # Collect all anomalies
    all_anomalies: list[Anomaly] = []

    # Define which anomaly types to check
    if anomaly_types is None:
        check_types: list[AnomalyType] = [
            "glitch",
            "dropout",
            "noise_spike",
            "timing_violation",
            "ringing",
            "overshoot",
            "undershoot",
        ]
    else:
        check_types = anomaly_types

    # Detect each type
    if "glitch" in check_types:
        all_anomalies.extend(_detect_glitches(data, sample_rate, voltage_swing, stats))

    if "dropout" in check_types:
        all_anomalies.extend(_detect_dropouts(data, sample_rate, voltage_swing, stats))

    if "noise_spike" in check_types:
        all_anomalies.extend(_detect_noise_spikes(data, sample_rate, voltage_swing, stats))

    if "timing_violation" in check_types:
        all_anomalies.extend(_detect_timing_violations(data, sample_rate, stats))

    if "ringing" in check_types:
        all_anomalies.extend(_detect_ringing(data, sample_rate, voltage_swing, stats))

    if "overshoot" in check_types:
        all_anomalies.extend(_detect_overshoot(data, sample_rate, voltage_swing, stats))

    if "undershoot" in check_types:
        all_anomalies.extend(_detect_undershoot(data, sample_rate, voltage_swing, stats))

    # Filter by confidence
    all_anomalies = [a for a in all_anomalies if a.confidence >= min_confidence]

    # Filter by severity if requested
    if severity_filter is not None:
        all_anomalies = [a for a in all_anomalies if a.severity in severity_filter]

    # Sort by timestamp
    all_anomalies.sort(key=lambda a: a.timestamp_us)

    return all_anomalies


def _detect_glitches(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    stats: dict[str, float],
) -> list[Anomaly]:
    """Detect brief narrow pulses (glitches).

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.

    Returns:
        List of detected glitch anomalies.
    """
    anomalies: list[Anomaly] = []

    if voltage_swing == 0 or len(data) < 10:
        return anomalies

    # Threshold for glitch detection
    threshold = stats["mean"]
    glitch_threshold = voltage_swing * 0.3  # 30% of swing

    # Find samples far from mean
    deviations = np.abs(data - threshold)
    glitch_candidates = np.where(deviations > glitch_threshold)[0]

    if len(glitch_candidates) == 0:
        return anomalies

    # Group consecutive samples into glitches
    glitch_groups = []
    current_group = [glitch_candidates[0]]

    for idx in glitch_candidates[1:]:
        if idx == current_group[-1] + 1:
            current_group.append(idx)
        else:
            glitch_groups.append(current_group)
            current_group = [idx]

    glitch_groups.append(current_group)

    # Analyze each glitch
    for group in glitch_groups:
        duration_samples = len(group)
        duration_ns = (duration_samples / sample_rate) * 1e9

        # Only report glitches < 50ns
        if duration_ns < 50:
            timestamp_us = (group[0] / sample_rate) * 1e6
            magnitude = np.max(np.abs(data[group] - threshold))

            # Determine severity based on magnitude
            if magnitude > voltage_swing * 0.5:
                severity: Severity = "WARNING"
            else:
                severity = "INFO"

            description = f"Brief {duration_ns:.0f}ns pulse, likely noise spike"

            anomalies.append(
                Anomaly(
                    timestamp_us=timestamp_us,
                    type="glitch",
                    severity=severity,
                    description=description,
                    duration_ns=duration_ns,
                    confidence=0.85,
                    metadata={"magnitude": magnitude},
                )
            )

    return anomalies


def _detect_dropouts(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    stats: dict[str, float],
) -> list[Anomaly]:
    """Detect missing transitions or prolonged holds.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.

    Returns:
        List of detected dropout anomalies.
    """
    anomalies: list[Anomaly] = []

    if voltage_swing == 0 or len(data) < 100:
        return anomalies

    # Estimate expected period from transitions
    threshold = (stats["max"] + stats["min"]) / 2
    digital = data > threshold
    transitions = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(transitions) < 5:
        return anomalies

    # Calculate typical transition interval
    intervals = np.diff(transitions)
    expected_period = np.median(intervals)

    # Find unusually long intervals (>2x expected)
    for i, interval in enumerate(intervals):
        if interval > expected_period * 2.0:
            timestamp_us = (transitions[i] / sample_rate) * 1e6
            duration_ns = (interval / sample_rate) * 1e9
            multiplier = interval / expected_period

            description = f"Missing transition, signal held for {multiplier:.1f}x expected duration"

            # Severity based on how long the dropout is
            if multiplier > 5.0:
                severity: Severity = "CRITICAL"
            elif multiplier > 3.0:
                severity = "WARNING"
            else:
                severity = "INFO"

            anomalies.append(
                Anomaly(
                    timestamp_us=timestamp_us,
                    type="dropout",
                    severity=severity,
                    description=description,
                    duration_ns=duration_ns,
                    confidence=0.88,
                    metadata={"expected_period_ns": (expected_period / sample_rate) * 1e9},
                )
            )

    return anomalies


def _detect_noise_spikes(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    stats: dict[str, float],
) -> list[Anomaly]:
    """Detect noise spikes (>20% of signal swing).

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.

    Returns:
        List of detected noise spike anomalies.
    """
    anomalies: list[Anomaly] = []

    if voltage_swing == 0 or len(data) < 10:
        return anomalies

    # Use running window to detect local spikes
    window = 10
    spike_threshold = voltage_swing * 0.2

    for i in range(window, len(data) - window):
        local_mean = np.mean(data[i - window : i + window])
        deviation = abs(data[i] - local_mean)

        if deviation > spike_threshold:
            timestamp_us = (i / sample_rate) * 1e6
            percent = (deviation / voltage_swing) * 100

            description = f"Noise spike {percent:.0f}% of signal swing"

            # Severity based on spike magnitude
            if percent > 50:
                severity: Severity = "WARNING"
            else:
                severity = "INFO"

            anomalies.append(
                Anomaly(
                    timestamp_us=timestamp_us,
                    type="noise_spike",
                    severity=severity,
                    description=description,
                    duration_ns=(1 / sample_rate) * 1e9,
                    confidence=0.80,
                    metadata={"deviation_v": deviation},
                )
            )

            # Skip ahead to avoid duplicate detections
            i += window

    # Limit number of noise spikes reported
    return anomalies[:50]


def _detect_timing_violations(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    stats: dict[str, float],
) -> list[Anomaly]:
    """Detect timing violations (±5% of expected timing).

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        stats: Basic statistics.

    Returns:
        List of detected timing violation anomalies.
    """
    anomalies: list[Anomaly] = []

    if len(data) < 100:
        return anomalies

    # Find edges
    threshold = stats["mean"]
    digital = data > threshold
    transitions = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(transitions) < 10:
        return anomalies

    # Analyze timing consistency
    intervals = np.diff(transitions)
    expected_interval = np.median(intervals)
    tolerance = expected_interval * 0.05  # 5% tolerance

    # Find violations
    for i, interval in enumerate(intervals):
        deviation = abs(interval - expected_interval)

        if deviation > tolerance:
            timestamp_us = (transitions[i] / sample_rate) * 1e6
            percent_dev = (deviation / expected_interval) * 100

            description = f"Timing deviation {percent_dev:.1f}% from expected"

            # Severity based on deviation magnitude
            if percent_dev > 15:
                severity: Severity = "WARNING"
            else:
                severity = "INFO"

            anomalies.append(
                Anomaly(
                    timestamp_us=timestamp_us,
                    type="timing_violation",
                    severity=severity,
                    description=description,
                    duration_ns=(interval / sample_rate) * 1e9,
                    confidence=0.75,
                    metadata={"deviation_percent": percent_dev},
                )
            )

    # Limit violations reported
    return anomalies[:20]


def _detect_ringing(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    stats: dict[str, float],
) -> list[Anomaly]:
    """Detect ringing (≥3 oscillations).

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.

    Returns:
        List of detected ringing anomalies.
    """
    anomalies: list[Anomaly] = []

    if voltage_swing == 0 or len(data) < 50:
        return anomalies

    # Look for oscillations after transitions
    threshold = stats["mean"]
    digital = data > threshold
    transitions = np.where(np.diff(digital.astype(int)) != 0)[0]

    for trans_idx in transitions:
        # Check window after transition
        window_size = min(50, len(data) - trans_idx - 1)
        if window_size < 10:
            continue

        window = data[trans_idx + 1 : trans_idx + 1 + window_size]

        # Count zero crossings (oscillations)
        window_mean = np.mean(window)
        crossings = np.sum(np.diff(np.sign(window - window_mean)) != 0)

        # Ringing should have ≥3 oscillations
        if crossings >= 6:  # 6 crossings = 3 full oscillations
            timestamp_us = (trans_idx / sample_rate) * 1e6
            duration_ns = (window_size / sample_rate) * 1e9
            num_oscillations = crossings // 2

            description = f"Ringing with {num_oscillations} oscillations after edge"

            severity: Severity = "INFO"

            anomalies.append(
                Anomaly(
                    timestamp_us=timestamp_us,
                    type="ringing",
                    severity=severity,
                    description=description,
                    duration_ns=duration_ns,
                    confidence=0.70,
                    metadata={"oscillations": num_oscillations},
                )
            )

    return anomalies[:10]


def _detect_overshoot(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    stats: dict[str, float],
) -> list[Anomaly]:
    """Detect overshoot (>10% beyond high rail).

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.

    Returns:
        List of detected overshoot anomalies.
    """
    anomalies: list[Anomaly] = []

    if voltage_swing == 0:
        return anomalies

    # Define expected high rail (based on histogram peaks)
    high_rail = stats["max"] * 0.95  # Expected rail at 95th percentile
    overshoot_threshold = high_rail * 1.1  # 10% above rail

    # Find overshoot samples
    overshoots = np.where(data > overshoot_threshold)[0]

    if len(overshoots) == 0:
        return anomalies

    # Group consecutive samples
    groups = []
    current = [overshoots[0]]

    for idx in overshoots[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]

    groups.append(current)

    # Report each overshoot event
    for group in groups:
        timestamp_us = (group[0] / sample_rate) * 1e6
        peak_value = np.max(data[group])
        percent_over = ((peak_value - high_rail) / high_rail) * 100

        description = (
            f"Signal exceeded expected high level by {percent_over:.0f}% (peak: {peak_value:.2f}V)"
        )

        # Severity based on overshoot magnitude
        if percent_over > 20:
            severity: Severity = "WARNING"
        else:
            severity = "INFO"

        anomalies.append(
            Anomaly(
                timestamp_us=timestamp_us,
                type="overshoot",
                severity=severity,
                description=description,
                duration_ns=(len(group) / sample_rate) * 1e9,
                confidence=0.82,
                metadata={"peak_voltage": peak_value},
            )
        )

    return anomalies[:10]


def _detect_undershoot(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    stats: dict[str, float],
) -> list[Anomaly]:
    """Detect undershoot (>10% beyond low rail).

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage.
        stats: Basic statistics.

    Returns:
        List of detected undershoot anomalies.
    """
    anomalies: list[Anomaly] = []

    if voltage_swing == 0:
        return anomalies

    # Define expected low rail
    low_rail = stats["min"] * 1.05  # Expected rail at 5th percentile
    undershoot_threshold = low_rail * 0.9  # 10% below rail (more negative)

    # Find undershoot samples
    undershoots = np.where(data < undershoot_threshold)[0]

    if len(undershoots) == 0:
        return anomalies

    # Group consecutive samples
    groups = []
    current = [undershoots[0]]

    for idx in undershoots[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            groups.append(current)
            current = [idx]

    groups.append(current)

    # Report each undershoot event
    for group in groups:
        timestamp_us = (group[0] / sample_rate) * 1e6
        min_value = np.min(data[group])
        percent_under = ((low_rail - min_value) / abs(low_rail)) * 100 if low_rail != 0 else 0

        description = (
            f"Signal fell below expected low level by {percent_under:.0f}% (min: {min_value:.2f}V)"
        )

        # Severity based on undershoot magnitude
        if percent_under > 20:
            severity: Severity = "WARNING"
        else:
            severity = "INFO"

        anomalies.append(
            Anomaly(
                timestamp_us=timestamp_us,
                type="undershoot",
                severity=severity,
                description=description,
                duration_ns=(len(group) / sample_rate) * 1e9,
                confidence=0.82,
                metadata={"min_voltage": min_value},
            )
        )

    return anomalies[:10]


__all__ = [
    "Anomaly",
    "AnomalyType",
    "Severity",
    "find_anomalies",
]
