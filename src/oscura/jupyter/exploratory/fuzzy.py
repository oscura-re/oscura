"""Fuzzy matching for timing and pattern analysis.

This module provides fuzzy matching capabilities for tolerating
timing variations and pattern deviations in real-world signals.


Example:
    >>> from oscura.jupyter.exploratory.fuzzy import fuzzy_timing_match
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
    edges = _extract_edges_from_trace(trace_or_edges, sample_rate)

    if len(edges) < 2:
        return _create_empty_timing_result()

    intervals = np.diff(edges)
    detected_period = float(np.median(intervals))
    expected_period = expected_period if expected_period is not None else detected_period

    deviation, match = _calculate_timing_deviation(detected_period, expected_period, tolerance)
    jitter_rms = _calculate_timing_jitter(intervals, detected_period)
    outlier_count, outlier_indices = _find_timing_outliers(intervals, expected_period, tolerance)
    confidence = _calculate_timing_confidence(deviation, tolerance, outlier_count, len(intervals))

    return FuzzyTimingResult(
        match=match,
        confidence=confidence,
        period=detected_period,
        deviation=deviation,
        jitter_rms=jitter_rms,
        outlier_count=outlier_count,
        outlier_indices=outlier_indices,
    )


def _extract_edges_from_trace(
    trace_or_edges: WaveformTrace | NDArray[np.float64], sample_rate: float | None
) -> NDArray[np.float64]:
    """Extract edge times from trace or pass through edge array."""
    if isinstance(trace_or_edges, WaveformTrace):
        data = trace_or_edges.data
        sample_rate = sample_rate or trace_or_edges.metadata.sample_rate

        if sample_rate is None or sample_rate <= 0:
            raise ValueError("Valid sample_rate required for WaveformTrace")

        v_min = np.percentile(data, 5)
        v_max = np.percentile(data, 95)
        threshold = (v_min + v_max) / 2
        digital = data > threshold

        edge_samples = np.where(np.abs(np.diff(digital.astype(int))) > 0)[0]
        return edge_samples / sample_rate

    return trace_or_edges


def _create_empty_timing_result() -> FuzzyTimingResult:
    """Create empty timing result for insufficient edges."""
    return FuzzyTimingResult(
        match=False,
        confidence=0.0,
        period=0.0,
        deviation=1.0,
        jitter_rms=0.0,
        outlier_count=0,
        outlier_indices=[],
    )


def _calculate_timing_deviation(
    detected: float, expected: float, tolerance: float
) -> tuple[float, bool]:
    """Calculate period deviation and match status."""
    deviation = abs(detected - expected) / expected
    return deviation, bool(deviation <= tolerance)


def _calculate_timing_jitter(intervals: NDArray[np.float64], detected_period: float) -> float:
    """Calculate RMS timing jitter."""
    normalized_intervals = intervals / detected_period
    return float(np.std(normalized_intervals - 1.0) * detected_period)


def _find_timing_outliers(
    intervals: NDArray[np.float64], expected_period: float, tolerance: float
) -> tuple[int, list[int]]:
    """Find outlier intervals exceeding tolerance threshold."""
    outlier_threshold = expected_period * tolerance * 3
    deviations = np.abs(intervals - expected_period)
    outlier_mask = deviations > outlier_threshold
    outlier_count = int(np.sum(outlier_mask))
    outlier_indices = list(np.where(outlier_mask)[0])
    return outlier_count, outlier_indices


def _calculate_timing_confidence(
    deviation: float, tolerance: float, outlier_count: int, total_intervals: int
) -> float:
    """Calculate timing match confidence score."""
    confidence = max(0.0, 1.0 - deviation / tolerance)
    confidence *= max(0.0, 1.0 - outlier_count / max(total_intervals, 1))
    return min(1.0, confidence)


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


def _convert_trace_to_digital_bits(
    trace: WaveformTrace,
) -> tuple[NDArray[np.int_], NDArray[np.intp], float]:
    """Convert analog trace to digital bit sequence.

    Args:
        trace: Input waveform trace.

    Returns:
        Tuple of (digital_signal, edges, estimated_bit_period).
    """
    data = trace.data
    v_min = np.percentile(data, 5)
    v_max = np.percentile(data, 95)
    threshold = (v_min + v_max) / 2
    digital = (data > threshold).astype(int)
    edges = np.where(np.diff(digital) != 0)[0]

    if len(edges) < 2:
        return digital, edges, 0.0

    gaps = np.diff(edges)
    estimated_bit_period = float(np.min(gaps))
    return digital, edges, estimated_bit_period


def _sample_bits_from_digital(
    digital: NDArray[np.int_], edges: NDArray[np.intp], estimated_bit_period: float
) -> NDArray[np.int_]:
    """Sample bits from digital signal at estimated bit period.

    Args:
        digital: Digital signal array.
        edges: Edge indices.
        estimated_bit_period: Estimated bit period in samples.

    Returns:
        Array of sampled bits.
    """
    bits_list = []
    sample_pos = edges[0] + estimated_bit_period / 2

    while sample_pos < len(digital):
        idx = int(sample_pos)
        if idx < len(digital):
            bits_list.append(digital[idx])
        sample_pos += estimated_bit_period

    return np.array(bits_list)


def _search_pattern_with_errors(
    bits: NDArray[np.int_],
    pattern: tuple[int, ...],
    max_errors: int,
    error_weight: float,
    edges: NDArray[np.intp],
    estimated_bit_period: float,
) -> tuple[list[dict[str, Any]], dict[tuple[int, ...], int]]:
    """Search for pattern matches allowing errors.

    Args:
        bits: Sampled bit sequence.
        pattern: Pattern to search for.
        max_errors: Maximum allowed errors.
        error_weight: Weight reduction per error.
        edges: Edge indices for position calculation.
        estimated_bit_period: Bit period for position calculation.

    Returns:
        Tuple of (matches, variations).
    """
    pattern_len = len(pattern)
    matches = []
    variations: dict[tuple[int, ...], int] = {}

    for i in range(len(bits) - pattern_len + 1):
        window = tuple(bits[i : i + pattern_len])
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

            if window != pattern:
                variations[window] = variations.get(window, 0) + 1

    return matches, variations


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

    # Setup: handle empty pattern
    if pattern_len == 0:
        return FuzzyPatternResult(
            matches=[], best_match_score=0.0, total_matches=0, pattern_variations=[]
        )

    # Processing: convert to digital and sample bits
    digital, edges, estimated_bit_period = _convert_trace_to_digital_bits(trace)

    if len(edges) < 2:
        return FuzzyPatternResult(
            matches=[], best_match_score=0.0, total_matches=0, pattern_variations=[]
        )

    bits = _sample_bits_from_digital(digital, edges, estimated_bit_period)

    # Result building: search for pattern matches
    matches, variations = _search_pattern_with_errors(
        bits, pattern, max_errors, error_weight, edges, estimated_bit_period
    )

    matches.sort(key=lambda x: x["score"], reverse=True)
    best_score: float = float(matches[0]["score"]) if matches else 0.0
    variation_list = sorted(variations.items(), key=lambda x: x[1], reverse=True)

    return FuzzyPatternResult(
        matches=matches,
        best_match_score=best_score,
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
    digital, edges, sample_rate = _prepare_fuzzy_detection_data(trace)

    if len(edges) < 4:
        return _create_unknown_result("Insufficient edges for protocol detection")

    estimated_bitrate = _estimate_bitrate(edges, sample_rate)
    candidates_list = candidates if candidates else list(PROTOCOL_SIGNATURES.keys())
    scores = _score_all_protocols(
        digital,
        edges,
        estimated_bitrate,
        sample_rate,
        candidates_list,
        timing_tolerance,
        pattern_tolerance,
    )

    if not scores:
        return _create_unknown_result("No matching protocols found")

    return _build_detection_result(scores)


def _prepare_fuzzy_detection_data(
    trace: WaveformTrace,
) -> tuple[NDArray[np.bool_], NDArray[np.int_], float]:
    """Prepare signal data for fuzzy protocol detection.

    Args:
        trace: Input waveform trace.

    Returns:
        Tuple of (digital_signal, edge_indices, sample_rate).
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate
    v_min = np.percentile(data, 5)
    v_max = np.percentile(data, 95)
    threshold = (v_min + v_max) / 2
    digital = data > threshold
    edges = np.where(np.diff(digital.astype(int)) != 0)[0]
    return digital, edges, sample_rate


def _estimate_bitrate(edges: NDArray[np.int_], sample_rate: float) -> float:
    """Estimate signal bitrate from edge spacing.

    Args:
        edges: Array of edge sample indices.
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated bitrate in bps.
    """
    intervals = np.diff(edges)
    median_interval = np.median(intervals)
    return float(sample_rate / median_interval)


def _score_all_protocols(
    digital: NDArray[np.bool_],
    edges: NDArray[np.int_],
    estimated_bitrate: float,
    sample_rate: float,
    candidates: list[str],
    timing_tolerance: float,
    pattern_tolerance: int,
) -> dict[str, dict[str, float]]:
    """Score each candidate protocol against signal characteristics.

    Args:
        digital: Digital signal array.
        edges: Edge sample indices.
        estimated_bitrate: Estimated bitrate in bps.
        sample_rate: Sample rate in Hz.
        candidates: List of protocol names to test.
        timing_tolerance: Timing tolerance as fraction.
        pattern_tolerance: Maximum pattern bit errors.

    Returns:
        Dict mapping protocol names to score dicts with timing/pattern/total scores.
    """
    scores: dict[str, dict[str, float]] = {}
    intervals = np.diff(edges)
    median_interval = np.median(intervals)

    for protocol in candidates:
        if protocol not in PROTOCOL_SIGNATURES:
            continue

        sig = PROTOCOL_SIGNATURES[protocol]
        timing_score = _score_protocol_timing(sig, estimated_bitrate, timing_tolerance)
        pattern_score = _score_protocol_patterns(
            sig, digital, edges, intervals, median_interval, pattern_tolerance
        )

        scores[protocol] = {
            "timing": timing_score,
            "pattern": pattern_score,
            "total": timing_score * 0.5 + pattern_score * 0.5,
        }

    return scores


def _score_protocol_timing(
    sig: dict[str, Any],
    estimated_bitrate: float,
    timing_tolerance: float,
) -> float:
    """Score protocol timing match.

    Args:
        sig: Protocol signature dictionary.
        estimated_bitrate: Estimated bitrate in bps.
        timing_tolerance: Timing tolerance as fraction.

    Returns:
        Timing score (0.0 to 1.0).
    """
    timing_score = 0.0

    if "typical_rates" in sig:
        rates = sig["typical_rates"]
        if hasattr(rates, "__iter__"):
            for rate in rates:
                if isinstance(rate, int | float):
                    ratio = estimated_bitrate / rate
                    if (1 - timing_tolerance) <= ratio <= (1 + timing_tolerance):
                        timing_score = max(timing_score, 1 - abs(1 - ratio) / timing_tolerance)

    return timing_score


def _score_protocol_patterns(
    sig: dict[str, Any],
    digital: NDArray[np.bool_],
    edges: NDArray[np.int_],
    intervals: NDArray[np.float64],
    median_interval: float,
    pattern_tolerance: int,
) -> float:
    """Score protocol pattern match.

    Args:
        sig: Protocol signature dictionary.
        digital: Digital signal array.
        edges: Edge sample indices.
        intervals: Inter-edge intervals.
        median_interval: Median interval between edges.
        pattern_tolerance: Maximum pattern bit errors.

    Returns:
        Pattern score (0.0 to 1.0).
    """
    pattern_score = 0.0

    if "start_pattern" in sig:
        pattern_score = max(
            pattern_score,
            _check_start_pattern(sig, digital, edges, median_interval, pattern_tolerance),
        )

    if "frame_size" in sig:
        pattern_score = max(pattern_score, _check_frame_size(sig, intervals, median_interval))

    return pattern_score


def _check_start_pattern(
    sig: dict[str, Any],
    digital: NDArray[np.bool_],
    edges: NDArray[np.int_],
    median_interval: float,
    pattern_tolerance: int,
) -> float:
    """Check start pattern match.

    Args:
        sig: Protocol signature dictionary.
        digital: Digital signal array.
        edges: Edge sample indices.
        median_interval: Median interval between edges.
        pattern_tolerance: Maximum pattern bit errors.

    Returns:
        Start pattern score (0.0 to 1.0).
    """
    bits = []
    pos = edges[0] + median_interval / 2
    for _ in range(4):
        if pos < len(digital):
            bits.append(int(digital[int(pos)]))
        pos += median_interval

    expected = sig["start_pattern"]
    if len(bits) >= len(expected):
        errors = sum(1 for a, b in zip(bits[: len(expected)], expected, strict=False) if a != b)
        if errors <= pattern_tolerance:
            return 1 - errors * 0.3

    return 0.0


def _check_frame_size(
    sig: dict[str, Any],
    intervals: NDArray[np.float64],
    median_interval: float,
) -> float:
    """Check frame size match.

    Args:
        sig: Protocol signature dictionary.
        intervals: Inter-edge intervals.
        median_interval: Median interval between edges.

    Returns:
        Frame size score (0.0 to 1.0).
    """
    gap_threshold = median_interval * 2
    long_gaps = intervals[intervals > gap_threshold]
    if len(long_gaps) > 0:
        frame_samples = np.median(long_gaps)
        frame_bits = frame_samples / median_interval
        for valid_size in sig["frame_size"]:
            if abs(frame_bits - valid_size) < 1.5:
                return 0.7
    return 0.0


def _create_unknown_result(recommendation: str) -> FuzzyProtocolResult:
    """Create result for unknown protocol detection.

    Args:
        recommendation: Recommendation message explaining why unknown.

    Returns:
        FuzzyProtocolResult indicating unknown protocol.
    """
    return FuzzyProtocolResult(
        detected_protocol="Unknown",
        confidence=0.0,
        alternatives=[],
        timing_score=0.0,
        pattern_score=0.0,
        recommendations=[recommendation],
    )


def _build_detection_result(scores: dict[str, dict[str, float]]) -> FuzzyProtocolResult:
    """Build final detection result from scores.

    Args:
        scores: Dict mapping protocol names to score dicts.

    Returns:
        FuzzyProtocolResult with best match and alternatives.
    """
    sorted_protocols = sorted(scores.items(), key=lambda x: x[1]["total"], reverse=True)
    best_protocol, best_scores = sorted_protocols[0]
    confidence = best_scores["total"]
    alternatives = [(p, s["total"]) for p, s in sorted_protocols[1:4] if s["total"] > 0.2]
    recommendations = _generate_recommendations(best_scores, confidence, alternatives)

    return FuzzyProtocolResult(
        detected_protocol=best_protocol,
        confidence=confidence,
        alternatives=alternatives,
        timing_score=best_scores["timing"],
        pattern_score=best_scores["pattern"],
        recommendations=recommendations,
    )


def _generate_recommendations(
    best_scores: dict[str, float],
    confidence: float,
    alternatives: list[tuple[str, float]],
) -> list[str]:
    """Generate recommendations based on detection results.

    Args:
        best_scores: Score dict for best matching protocol.
        confidence: Overall confidence score.
        alternatives: List of alternative protocol candidates.

    Returns:
        List of recommendation strings.
    """
    recommendations = []

    if confidence < 0.5:
        recommendations.append("Low confidence - verify with protocol-specific decoder")
    if best_scores["timing"] > best_scores["pattern"]:
        recommendations.append("Timing matched better than patterns - check signal quality")
    if best_scores["pattern"] > best_scores["timing"]:
        recommendations.append("Patterns matched but timing off - check clock accuracy")
    if not alternatives:
        recommendations.append("No alternative protocols detected")

    return recommendations


__all__ = [
    "PROTOCOL_SIGNATURES",
    "FuzzyPatternResult",
    "FuzzyProtocolResult",
    "FuzzyTimingResult",
    "fuzzy_pattern_match",
    "fuzzy_protocol_detect",
    "fuzzy_timing_match",
]
