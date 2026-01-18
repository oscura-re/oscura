"""Fuzzy matching for timing and pattern analysis.

This module provides fuzzy matching capabilities for tolerating
timing variations and pattern deviations in real-world signals.


Example:
    >>> from oscura.exploratory.fuzzy import fuzzy_timing_match
    >>> result = fuzzy_timing_match(edges, expected_period=1e-6, tolerance=0.1)
    >>> print(f"Match confidence: {result.confidence:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FuzzyTimingResult:
    """Result of fuzzy timing match.

    Attributes:
        match: True if timing matches within tolerance.
        confidence: Match confidence (0.0 to 1.0).
        period: Detected period.
        deviation: Deviation from expected period.
        jitter_rms: RMS timing jitter.
        outlier_count: Number of timing outliers.
        outlier_indices: Indices of outlier edges.
    """

    match: bool
    confidence: float
    period: float
    deviation: float
    jitter_rms: float
    outlier_count: int
    outlier_indices: list[int]


def fuzzy_timing_match(
    trace_or_edges: WaveformTrace | NDArray[np.float64],
    *,
    expected_period: float | None = None,
    tolerance: float = 0.1,
    sample_rate: float | None = None,
) -> FuzzyTimingResult:
    """Match timing with fuzzy tolerance.

    Allows timing variations while still detecting protocol patterns.
    Useful for signals with jitter or clock drift.

    Args:
        trace_or_edges: WaveformTrace or array of edge times.
        expected_period: Expected period in seconds.
        tolerance: Tolerance as fraction (0.1 = 10%).
        sample_rate: Sample rate (required if trace provided).

    Returns:
        FuzzyTimingResult with match information.

    Raises:
        ValueError: If sample_rate is invalid when WaveformTrace provided.

    Example:
        >>> result = fuzzy_timing_match(trace, expected_period=1e-6, tolerance=0.1)
        >>> print(f"Period match: {result.match}")
        >>> print(f"Actual period: {result.period:.3e} s")

    References:
        FUZZY-001: Fuzzy Timing Tolerance
    """
    # Extract edges if WaveformTrace provided
    if isinstance(trace_or_edges, WaveformTrace):
        data = trace_or_edges.data
        sample_rate = sample_rate or trace_or_edges.metadata.sample_rate

        if sample_rate is None or sample_rate <= 0:
            raise ValueError("Valid sample_rate required for WaveformTrace")

        # Threshold and find edges
        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        threshold = (v_min + v_max) / 2
        digital = data > threshold

        # Find all transitions (rising and falling edges)
        edge_samples = np.where(np.abs(np.diff(digital.astype(int))) > 0)[0]
        edges = edge_samples / sample_rate
    else:
        edges = trace_or_edges

    if len(edges) < 2:
        return FuzzyTimingResult(
            match=False,
            confidence=0.0,
            period=0.0,
            deviation=1.0,
            jitter_rms=0.0,
            outlier_count=0,
            outlier_indices=[],
        )

    # Calculate inter-edge intervals
    intervals = np.diff(edges)

    # Detect period (use median for robustness)
    detected_period = np.median(intervals)

    # Use expected period if provided, otherwise use detected
    if expected_period is None:
        expected_period = detected_period

    # Calculate deviation
    deviation = abs(detected_period - expected_period) / expected_period

    # Determine match
    match = bool(deviation <= tolerance)

    # Calculate jitter
    normalized_intervals = intervals / detected_period
    jitter_rms = np.std(normalized_intervals - 1.0) * detected_period

    # Find outliers
    outlier_threshold = expected_period * tolerance * 3  # 3x tolerance
    deviations = np.abs(intervals - expected_period)
    outlier_mask = deviations > outlier_threshold
    outlier_count = int(np.sum(outlier_mask))
    outlier_indices = list(np.where(outlier_mask)[0])

    # Calculate confidence
    # Higher confidence for lower deviation and fewer outliers
    confidence = max(0.0, 1.0 - deviation / tolerance)
    confidence *= max(0.0, 1.0 - outlier_count / max(len(intervals), 1))
    confidence = min(1.0, confidence)

    return FuzzyTimingResult(
        match=match,
        confidence=confidence,
        period=detected_period,
        deviation=deviation,
        jitter_rms=jitter_rms,
        outlier_count=outlier_count,
        outlier_indices=outlier_indices,
    )


@dataclass
class FuzzyPatternResult:
    """Result of fuzzy pattern match.

    Attributes:
        matches: List of match locations with scores.
        best_match_score: Score of best match.
        total_matches: Total number of matches found.
        pattern_variations: Common pattern variations found.
    """

    matches: list[dict[str, Any]]
    best_match_score: float
    total_matches: int
    pattern_variations: list[tuple[tuple[int, ...], int]]


def fuzzy_pattern_match(
    trace: WaveformTrace,
    pattern: list[int] | tuple[int, ...],
    *,
    max_errors: int = 1,
    error_weight: float = 0.5,
) -> FuzzyPatternResult:
    """Match pattern with allowed bit errors.

    Finds pattern occurrences allowing for bit errors, useful for
    noisy signals or partial matches.

    Args:
        trace: Signal trace to search.
        pattern: Bit pattern to find (list of 0s and 1s).
        max_errors: Maximum allowed bit errors.
        error_weight: Weight reduction per error.

    Returns:
        FuzzyPatternResult with match locations.

    Example:
        >>> result = fuzzy_pattern_match(trace, [0, 1, 0, 1, 0, 1], max_errors=1)
        >>> print(f"Found {result.total_matches} matches")
        >>> for match in result.matches[:5]:
        ...     print(f"  Position {match['position']}: score {match['score']:.2f}")

    References:
        FUZZY-002: Fuzzy Pattern Matching
    """
    pattern = tuple(pattern)
    pattern_len = len(pattern)

    # Handle empty pattern
    if pattern_len == 0:
        return FuzzyPatternResult(
            matches=[],
            best_match_score=0.0,
            total_matches=0,
            pattern_variations=[],
        )

    # Convert trace to digital
    data = trace.data
    v_min = np.percentile(data, 5)
    v_max = np.percentile(data, 95)
    threshold = (v_min + v_max) / 2
    digital = (data > threshold).astype(int)

    # Find edges and sample at bit centers
    edges = np.where(np.diff(digital) != 0)[0]

    if len(edges) < 2:
        return FuzzyPatternResult(
            matches=[],
            best_match_score=0.0,
            total_matches=0,
            pattern_variations=[],
        )

    # Estimate bit period from edges
    # Note: This algorithm works best with signals that have frequent transitions.
    # For patterns with long runs of identical bits, edge density may be insufficient
    # for accurate bit recovery. Best suited for alternating patterns or noisy signals.
    gaps = np.diff(edges)

    # Use minimum gap as bit period estimate
    # The smallest gap between edges is typically one bit period
    estimated_bit_period = float(np.min(gaps))

    # Sample bits at regular intervals
    bits_list = []
    sample_pos = edges[0] + estimated_bit_period / 2

    while sample_pos < len(digital):
        idx = int(sample_pos)
        if idx < len(digital):
            bits_list.append(digital[idx])
        sample_pos += estimated_bit_period

    bits = np.array(bits_list)

    # Search for pattern with fuzzy matching
    matches = []
    variations: dict[tuple[int, ...], int] = {}

    for i in range(len(bits) - pattern_len + 1):
        window = tuple(bits[i : i + pattern_len])

        # Count errors
        errors = sum(1 for a, b in zip(window, pattern, strict=False) if a != b)

        if errors <= max_errors:
            score = 1.0 - errors * error_weight
            matches.append(
                {
                    "position": i,
                    "sample_position": int(edges[0] + i * estimated_bit_period),
                    "errors": errors,
                    "score": score,
                    "actual_pattern": window,
                }
            )

            # Track variations
            if window != pattern:
                variations[window] = variations.get(window, 0) + 1

    # Sort matches by score
    matches.sort(key=lambda x: x["score"], reverse=True)  # type: ignore[arg-type, return-value]

    best_score = matches[0]["score"] if matches else 0.0

    # Sort variations by frequency
    variation_list = sorted(variations.items(), key=lambda x: x[1], reverse=True)

    return FuzzyPatternResult(
        matches=matches,
        best_match_score=best_score,  # type: ignore[arg-type]
        total_matches=len(matches),
        pattern_variations=variation_list[:10],
    )


@dataclass
class FuzzyProtocolResult:
    """Result of fuzzy protocol detection.

    Attributes:
        detected_protocol: Most likely protocol.
        confidence: Detection confidence.
        alternatives: Alternative protocol candidates.
        timing_score: Score based on timing match.
        pattern_score: Score based on pattern match.
        recommendations: Suggestions for improving detection.
    """

    detected_protocol: str
    confidence: float
    alternatives: list[tuple[str, float]]
    timing_score: float
    pattern_score: float
    recommendations: list[str]


# Protocol signatures for fuzzy matching
PROTOCOL_SIGNATURES = {
    "UART": {
        "start_bit": 0,
        "stop_bits": 1,
        "frame_size": [8, 9, 10, 11],  # With start/stop
        "typical_rates": [9600, 19200, 38400, 57600, 115200],
    },
    "I2C": {
        "start_pattern": [1, 0],  # SDA falls while SCL high
        "stop_pattern": [0, 1],  # SDA rises while SCL high
        "ack_bit": 0,
        "typical_rates": [100e3, 400e3, 1e6, 3.4e6],
    },
    "SPI": {
        "idle_clock": [0, 1],  # CPOL options
        "clock_phase": [0, 1],  # CPHA options
        "frame_size": [8, 16],
        "typical_rates": [1e6, 5e6, 10e6, 20e6, 40e6],
    },
    "CAN": {
        "start_of_frame": 0,
        "frame_patterns": ["standard", "extended"],
        "typical_rates": [125e3, 250e3, 500e3, 1e6],
    },
}


def fuzzy_protocol_detect(
    trace: WaveformTrace,
    *,
    candidates: list[str] | None = None,
    timing_tolerance: float = 0.15,
    pattern_tolerance: int = 2,
) -> FuzzyProtocolResult:
    """Detect protocol with fuzzy matching.

    Uses timing tolerance and pattern flexibility to identify
    protocols even with non-ideal signals.

    Args:
        trace: Signal trace to analyze.
        candidates: List of protocols to consider (None = all).
        timing_tolerance: Timing tolerance as fraction.
        pattern_tolerance: Maximum pattern bit errors.

    Returns:
        FuzzyProtocolResult with detection results.

    Example:
        >>> result = fuzzy_protocol_detect(trace)
        >>> print(f"Detected: {result.detected_protocol}")
        >>> print(f"Confidence: {result.confidence:.1%}")

    References:
        FUZZY-003: Fuzzy Protocol Detection
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    if candidates is None:
        candidates = list(PROTOCOL_SIGNATURES.keys())

    # Analyze signal characteristics
    v_min = np.percentile(data, 5)
    v_max = np.percentile(data, 95)
    threshold = (v_min + v_max) / 2
    digital = data > threshold

    edges = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(edges) < 4:
        return FuzzyProtocolResult(
            detected_protocol="Unknown",
            confidence=0.0,
            alternatives=[],
            timing_score=0.0,
            pattern_score=0.0,
            recommendations=["Insufficient edges for protocol detection"],
        )

    # Estimate bit rate
    intervals = np.diff(edges)
    median_interval = np.median(intervals)
    estimated_bitrate = sample_rate / median_interval

    # Score each protocol
    scores: dict[str, dict[str, float]] = {}

    for protocol in candidates:
        if protocol not in PROTOCOL_SIGNATURES:
            continue

        sig = PROTOCOL_SIGNATURES[protocol]
        timing_score = 0.0
        pattern_score = 0.0

        # Check timing against typical rates
        if "typical_rates" in sig:
            rates = sig["typical_rates"]
            if hasattr(rates, "__iter__"):
                for rate in rates:
                    if isinstance(rate, int | float):
                        ratio = estimated_bitrate / rate
                        if (1 - timing_tolerance) <= ratio <= (1 + timing_tolerance):
                            timing_score = max(timing_score, 1 - abs(1 - ratio) / timing_tolerance)

        # Check patterns
        if "start_pattern" in sig:
            # Sample first few bits
            bits = []
            pos = edges[0] + median_interval / 2
            for _ in range(4):
                if pos < len(digital):
                    bits.append(int(digital[int(pos)]))
                pos += median_interval

            expected = sig["start_pattern"]
            if len(bits) >= len(expected):  # type: ignore[arg-type]
                errors = sum(
                    1  # type: ignore[misc]
                    for a, b in zip(bits[: len(expected)], expected, strict=False)  # type: ignore[call-overload, arg-type]
                    if a != b  # type: ignore[misc, call-overload, arg-type]
                )
                if errors <= pattern_tolerance:
                    pattern_score = 1 - errors * 0.3

        # Check frame size
        if "frame_size" in sig:
            # Estimate frame size from inter-frame gaps
            gap_threshold = median_interval * 2
            long_gaps = intervals[intervals > gap_threshold]
            if len(long_gaps) > 0:
                frame_samples = np.median(long_gaps)
                frame_bits = frame_samples / median_interval
                for valid_size in sig["frame_size"]:  # type: ignore[attr-defined]
                    if abs(frame_bits - valid_size) < 1.5:
                        pattern_score = max(pattern_score, 0.7)

        scores[protocol] = {
            "timing": timing_score,
            "pattern": pattern_score,
            "total": timing_score * 0.5 + pattern_score * 0.5,
        }

    # Find best match
    if not scores:
        return FuzzyProtocolResult(
            detected_protocol="Unknown",
            confidence=0.0,
            alternatives=[],
            timing_score=0.0,
            pattern_score=0.0,
            recommendations=["No matching protocols found"],
        )

    sorted_protocols = sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True)
    best_protocol, best_scores = sorted_protocols[0]

    confidence = best_scores["total"]
    alternatives = [(p, s["total"]) for p, s in sorted_protocols[1:4] if s["total"] > 0.2]

    # Generate recommendations
    recommendations = []

    if confidence < 0.5:
        recommendations.append("Low confidence - verify with protocol-specific decoder")

    if best_scores["timing"] > best_scores["pattern"]:
        recommendations.append("Timing matched better than patterns - check signal quality")

    if best_scores["pattern"] > best_scores["timing"]:
        recommendations.append("Patterns matched but timing off - check clock accuracy")

    if not alternatives:
        recommendations.append("No alternative protocols detected")

    return FuzzyProtocolResult(
        detected_protocol=best_protocol,
        confidence=confidence,
        alternatives=alternatives,
        timing_score=best_scores["timing"],
        pattern_score=best_scores["pattern"],
        recommendations=recommendations,
    )


__all__ = [
    "PROTOCOL_SIGNATURES",
    "FuzzyPatternResult",
    "FuzzyProtocolResult",
    "FuzzyTimingResult",
    "fuzzy_pattern_match",
    "fuzzy_protocol_detect",
    "fuzzy_timing_match",
]
