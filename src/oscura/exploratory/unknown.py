"""Unknown signal analysis and reverse engineering.

This module provides tools for analyzing signals from unknown systems
and protocols, including binary field detection and pattern analysis.

- UNKNOWN-001: Binary Field Detection
- UNKNOWN-002: Protocol Auto-Detection with Fuzzy Matching
- UNKNOWN-003: Unknown Signal Characterization
- UNKNOWN-004: Pattern Frequency Analysis
- UNKNOWN-005: Reverse Engineering Workflow

Example:
    >>> from oscura.exploratory.unknown import characterize_unknown_signal
    >>> result = characterize_unknown_signal(trace)
    >>> print(f"Signal type: {result.signal_type}")
    >>> print(f"Suggested protocols: {result.suggested_protocols}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class BinaryFieldResult:
    """Result of binary field detection.

    Attributes:
        fields: List of detected fields with positions.
        field_count: Total number of fields detected.
        bit_rate: Estimated bit rate in bps.
        encoding: Detected encoding type.
        confidence: Detection confidence.
    """

    fields: list[dict[str, Any]]
    field_count: int
    bit_rate: float | None
    encoding: str
    confidence: float


def detect_binary_fields(
    trace: WaveformTrace,
    *,
    min_field_bits: int = 4,
    max_gap_ratio: float = 2.0,
) -> BinaryFieldResult:
    """Detect binary fields in unknown signal per UNKNOWN-001.

    Analyzes signal for structured binary data patterns including
    start/stop markers, length fields, and data payloads.

    Args:
        trace: Signal trace to analyze.
        min_field_bits: Minimum bits to consider a field.
        max_gap_ratio: Maximum gap ratio for field boundaries.

    Returns:
        BinaryFieldResult with detected fields.

    Example:
        >>> result = detect_binary_fields(trace)
        >>> for field in result.fields:
        ...     print(f"Field at {field['start_sample']}: {field['length']} bits")

    References:
        UNKNOWN-001: Binary Field Detection
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Threshold for digital conversion
    v_min = np.percentile(data, 5)
    v_max = np.percentile(data, 95)
    threshold = (v_min + v_max) / 2

    # Convert to digital
    digital = (data > threshold).astype(int)

    # Find edges
    edges = np.where(np.diff(digital) != 0)[0]

    if len(edges) < 2:
        return BinaryFieldResult(
            fields=[],
            field_count=0,
            bit_rate=None,
            encoding="unknown",
            confidence=0.0,
        )

    # Estimate bit period from edge spacing
    edge_gaps = np.diff(edges)
    median_gap = np.median(edge_gaps)
    bit_period = median_gap

    # Group edges into fields
    fields = []
    current_field_start = edges[0]
    current_field_edges = [edges[0]]

    for i in range(1, len(edges)):
        gap = edges[i] - edges[i - 1]

        if gap > max_gap_ratio * bit_period:
            # End current field
            if len(current_field_edges) >= min_field_bits:
                current_field_edges[-1] - current_field_edges[0]
                n_bits = len(current_field_edges) - 1

                # Extract bit pattern
                bits = []
                for j in range(len(current_field_edges) - 1):
                    start = current_field_edges[j]
                    end = current_field_edges[j + 1]
                    mid = (start + end) // 2
                    bits.append(digital[mid])

                fields.append(
                    {
                        "start_sample": int(current_field_start),
                        "end_sample": int(current_field_edges[-1]),
                        "length": n_bits,
                        "bits": bits,
                        "timestamp": current_field_start / sample_rate,
                    }
                )

            # Start new field
            current_field_start = edges[i]
            current_field_edges = [edges[i]]
        else:
            current_field_edges.append(edges[i])

    # Handle last field
    if len(current_field_edges) >= min_field_bits:
        current_field_edges[-1] - current_field_edges[0]
        n_bits = len(current_field_edges) - 1
        bits = []
        for j in range(len(current_field_edges) - 1):
            start = current_field_edges[j]
            end = current_field_edges[j + 1]
            mid = (start + end) // 2
            bits.append(digital[mid])

        fields.append(
            {
                "start_sample": int(current_field_start),
                "end_sample": int(current_field_edges[-1]),
                "length": n_bits,
                "bits": bits,
                "timestamp": current_field_start / sample_rate,
            }
        )

    # Estimate bit rate
    bit_rate = sample_rate / bit_period if bit_period > 0 else None

    # Detect encoding
    encoding = _detect_encoding(digital, edges, bit_period)

    # Calculate confidence
    confidence = min(1.0, len(fields) / 10.0) * 0.8
    if bit_rate is not None:
        confidence += 0.2

    return BinaryFieldResult(
        fields=fields,
        field_count=len(fields),
        bit_rate=bit_rate,
        encoding=encoding,
        confidence=confidence,
    )


def _detect_encoding(
    digital: NDArray[np.int_],
    edges: NDArray[np.int_],
    bit_period: float,
) -> str:
    """Detect signal encoding type.

    Args:
        digital: Digital signal.
        edges: Edge positions.
        bit_period: Estimated bit period.

    Returns:
        Encoding type name.
    """
    if len(edges) < 4:
        return "unknown"

    # Analyze edge spacing patterns
    gaps = np.diff(edges)

    # Check for Manchester (edges every half bit)
    if np.std(gaps) < bit_period * 0.3:
        return "manchester"

    # Check for NRZ (edges at bit boundaries)
    normalized_gaps = gaps / bit_period
    integer_gaps = np.round(normalized_gaps)
    residuals = np.abs(normalized_gaps - integer_gaps)

    if np.mean(residuals) < 0.2:
        return "nrz"

    # Check for NRZI
    if np.mean(normalized_gaps > 0.8) > 0.7:
        return "nrzi"

    return "unknown"


@dataclass
class UnknownSignalCharacterization:
    """Comprehensive characterization of unknown signal.

    Attributes:
        signal_type: 'digital', 'analog', or 'mixed'.
        is_periodic: True if signal is periodic.
        fundamental_frequency: Fundamental frequency if periodic.
        dc_offset: DC offset voltage.
        amplitude: Signal amplitude.
        rise_time: Estimated rise time.
        fall_time: Estimated fall time.
        suggested_protocols: List of possible protocols.
        noise_floor: Estimated noise floor.
        snr_db: Signal-to-noise ratio in dB.
        features: Dictionary of extracted features.
    """

    signal_type: Literal["digital", "analog", "mixed"]
    is_periodic: bool
    fundamental_frequency: float | None
    dc_offset: float
    amplitude: float
    rise_time: float | None
    fall_time: float | None
    suggested_protocols: list[tuple[str, float]]
    noise_floor: float
    snr_db: float
    features: dict[str, Any] = field(default_factory=dict)


def characterize_unknown_signal(
    trace: WaveformTrace,
) -> UnknownSignalCharacterization:
    """Comprehensive characterization of unknown signal per UNKNOWN-003.

    Analyzes signal characteristics to determine type, periodicity,
    and suggest possible protocols.

    Args:
        trace: Signal trace to characterize.

    Returns:
        UnknownSignalCharacterization with all extracted features.

    Example:
        >>> result = characterize_unknown_signal(trace)
        >>> print(f"Signal type: {result.signal_type}")
        >>> print(f"Periodic: {result.is_periodic}")
        >>> for protocol, confidence in result.suggested_protocols:
        ...     print(f"  {protocol}: {confidence:.1%}")

    References:
        UNKNOWN-003: Unknown Signal Characterization
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Handle edge case of very short traces
    if len(data) < 2:
        return UnknownSignalCharacterization(
            signal_type="analog",
            is_periodic=False,
            fundamental_frequency=None,
            dc_offset=float(data[0]) if len(data) > 0 else 0.0,
            amplitude=0.0,
            rise_time=None,
            fall_time=None,
            suggested_protocols=[],
            noise_floor=0.0,
            snr_db=float("inf"),
            features={},
        )

    # Basic statistics
    v_min = np.min(data)
    v_max = np.max(data)
    v_mean = np.mean(data)
    v_std = np.std(data)

    dc_offset = v_mean
    amplitude = (v_max - v_min) / 2

    # Determine signal type
    # Digital signals have bimodal distribution
    hist, bin_edges = np.histogram(data, bins=50)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in histogram
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 0.1 * np.max(hist):
            peaks.append((centers[i], hist[i]))

    if len(peaks) >= 4:
        # Many peaks suggest analog signal (e.g., sine wave with noisy histogram)
        signal_type: Literal["digital", "analog", "mixed"] = "analog"
    elif len(peaks) == 2 or len(peaks) == 3:
        # Two peaks suggest digital (bimodal), but check if they're well-separated
        peak_positions = [p[0] for p in peaks]
        # Normalize peak positions to 0-1 range
        normalized_peaks = [(p - v_min) / (v_max - v_min) for p in peak_positions]

        # If peaks are well-separated (one < 0.4, one > 0.6), likely digital
        has_low_peak = any(p < 0.4 for p in normalized_peaks)
        has_high_peak = any(p > 0.6 for p in normalized_peaks)

        if has_low_peak and has_high_peak:
            signal_type = "digital"
        else:
            # Peaks not well separated, likely analog
            signal_type = "analog"
    elif len(peaks) == 1:
        # Check for modulated signal
        signal_type = "mixed" if v_std > 0.2 * amplitude else "analog"
    else:
        signal_type = "analog"

    # Check periodicity via FFT
    from scipy import signal as sp_signal

    n = len(data)
    # Need at least 4 samples for meaningful FFT analysis
    if n >= 4:
        f, psd = sp_signal.welch(data, fs=sample_rate, nperseg=min(4096, n))

        # Find dominant frequency (excluding DC)
        psd_no_dc = psd.copy()
        psd_no_dc[0] = 0

        if len(psd_no_dc) > 0 and np.any(psd_no_dc > 0):
            peak_idx = np.argmax(psd_no_dc)
            mean_psd = np.mean(psd_no_dc[psd_no_dc > 0]) if np.any(psd_no_dc > 0) else 0
            fundamental_frequency = f[peak_idx] if psd_no_dc[peak_idx] > 10 * mean_psd else None
        else:
            fundamental_frequency = None

        is_periodic = fundamental_frequency is not None
    else:
        fundamental_frequency = None
        is_periodic = False

    # Estimate noise floor
    if n >= 4:
        noise_floor = np.median(np.sort(psd)[: len(psd) // 4]) if len(psd) > 0 else 0.0
        signal_power = np.max(psd) - noise_floor if len(psd) > 0 else 0.0
    else:
        noise_floor = 0.0
        signal_power = 0.0
    snr_db = 10 * np.log10(signal_power / noise_floor) if noise_floor > 0 else 0

    # Estimate rise/fall times for digital signals
    rise_time = None
    fall_time = None

    if signal_type == "digital":
        threshold_low = v_min + 0.1 * (v_max - v_min)
        threshold_high = v_min + 0.9 * (v_max - v_min)

        # Find rising edges
        rising_times = []
        falling_times = []

        for i in range(1, len(data) - 1):
            if data[i - 1] < threshold_low and data[i + 1] > threshold_high:
                # Rising edge
                rising_times.append(1 / sample_rate)
            elif data[i - 1] > threshold_high and data[i + 1] < threshold_low:
                # Falling edge
                falling_times.append(1 / sample_rate)

        if rising_times:
            rise_time = float(np.median(rising_times))
        if falling_times:
            fall_time = float(np.median(falling_times))

    # Suggest protocols
    suggested_protocols = _suggest_protocols(signal_type, fundamental_frequency, sample_rate, data)

    # Collect features
    features = {
        "v_min": v_min,
        "v_max": v_max,
        "v_mean": v_mean,
        "v_std": v_std,
        "crest_factor": v_max / np.sqrt(np.mean(data**2)) if np.mean(data**2) > 0 else 0,
        "n_peaks": len(peaks),
        "peak_positions": [p[0] for p in peaks],
    }

    return UnknownSignalCharacterization(
        signal_type=signal_type,
        is_periodic=is_periodic,
        fundamental_frequency=fundamental_frequency,
        dc_offset=dc_offset,
        amplitude=amplitude,
        rise_time=rise_time,
        fall_time=fall_time,
        suggested_protocols=suggested_protocols,
        noise_floor=noise_floor,
        snr_db=snr_db,
        features=features,
    )


def _suggest_protocols(
    signal_type: str,
    frequency: float | None,
    sample_rate: float,
    data: NDArray[np.float64],
) -> list[tuple[str, float]]:
    """Suggest possible protocols based on signal characteristics.

    Args:
        signal_type: Signal type (digital/analog/mixed).
        frequency: Fundamental frequency.
        sample_rate: Sample rate.
        data: Signal data.

    Returns:
        List of (protocol_name, confidence) tuples.
    """
    suggestions = []  # type: ignore[var-annotated]

    if signal_type != "digital":
        return suggestions

    # Estimate bit rate from signal
    v_min = np.percentile(data, 5)
    v_max = np.percentile(data, 95)
    threshold = (v_min + v_max) / 2
    digital = data > threshold
    edges = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(edges) < 2:
        return suggestions

    median_gap = np.median(np.diff(edges))
    estimated_bitrate = sample_rate / median_gap

    # Check common baud rates for UART
    uart_rates = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
    for rate in uart_rates:
        ratio = estimated_bitrate / rate
        if 0.9 <= ratio <= 1.1:
            suggestions.append(("UART", 0.7 + 0.3 * (1 - abs(1 - ratio))))
            break

    # Check for I2C (two-wire, specific timing)
    if 50e3 <= estimated_bitrate <= 400e3:
        suggestions.append(("I2C", 0.5))
    elif 400e3 < estimated_bitrate <= 3.4e6:
        suggestions.append(("I2C Fast Mode", 0.5))

    # Check for SPI (higher speeds)
    if estimated_bitrate >= 1e6:
        suggestions.append(("SPI", 0.4))

    # Check for CAN
    can_rates = [125e3, 250e3, 500e3, 1e6]
    for rate in can_rates:  # type: ignore[assignment]
        if 0.9 <= estimated_bitrate / rate <= 1.1:
            suggestions.append(("CAN", 0.6))
            break

    # Sort by confidence
    suggestions.sort(key=lambda x: x[1], reverse=True)

    return suggestions


@dataclass
class PatternFrequencyResult:
    """Result of pattern frequency analysis.

    Attributes:
        patterns: Dictionary of pattern to count.
        most_common: List of (pattern, count) for most common patterns.
        entropy: Shannon entropy of pattern distribution.
        repetition_rate: Rate of pattern repetition.
    """

    patterns: dict[tuple[int, ...], int]
    most_common: list[tuple[tuple[int, ...], int]]
    entropy: float
    repetition_rate: float


def analyze_pattern_frequency(
    trace: WaveformTrace,
    *,
    pattern_length: int = 8,
    min_occurrences: int = 2,
) -> PatternFrequencyResult:
    """Analyze frequency of bit patterns per UNKNOWN-004.

    Identifies recurring patterns that may indicate protocol structure
    or data framing.

    Args:
        trace: Signal trace to analyze.
        pattern_length: Length of patterns to search for.
        min_occurrences: Minimum occurrences to report.

    Returns:
        PatternFrequencyResult with pattern statistics.

    Example:
        >>> result = analyze_pattern_frequency(trace, pattern_length=8)
        >>> for pattern, count in result.most_common[:5]:
        ...     print(f"Pattern {pattern}: {count} occurrences")

    References:
        UNKNOWN-004: Pattern Frequency Analysis
    """
    data = trace.data

    # Convert to digital
    v_min = np.percentile(data, 5)
    v_max = np.percentile(data, 95)
    threshold = (v_min + v_max) / 2
    digital = (data > threshold).astype(int)

    # Find bit boundaries from edges
    edges = np.where(np.diff(digital) != 0)[0]

    if len(edges) < 2:
        return PatternFrequencyResult(
            patterns={},
            most_common=[],
            entropy=0.0,
            repetition_rate=0.0,
        )

    # Estimate bit period
    median_gap = np.median(np.diff(edges))

    # Sample at bit centers
    bits = []
    sample_pos = edges[0] + median_gap / 2

    while sample_pos < len(digital):
        idx = int(sample_pos)
        if idx < len(digital):
            bits.append(digital[idx])
        sample_pos += median_gap

    # Count patterns
    patterns: dict[tuple[int, ...], int] = {}

    for i in range(len(bits) - pattern_length + 1):
        pattern = tuple(bits[i : i + pattern_length])
        patterns[pattern] = patterns.get(pattern, 0) + 1

    # Filter by minimum occurrences
    patterns = {p: c for p, c in patterns.items() if c >= min_occurrences}

    # Find most common
    most_common = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:20]

    # Calculate entropy
    total = sum(patterns.values())
    if total > 0:
        probs = np.array(list(patterns.values())) / total
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
    else:
        entropy = 0.0

    # Repetition rate
    repetition_rate = 1 - len(patterns) / total if total > 0 else 0.0

    return PatternFrequencyResult(
        patterns=patterns,
        most_common=most_common,
        entropy=entropy,
        repetition_rate=repetition_rate,
    )


@dataclass
class ReverseEngineeringResult:
    """Result of reverse engineering workflow.

    Attributes:
        signal_char: Signal characterization.
        binary_fields: Detected binary fields.
        pattern_analysis: Pattern frequency analysis.
        protocol_hypothesis: Most likely protocol.
        confidence: Overall confidence.
        recommendations: List of next steps.
    """

    signal_char: UnknownSignalCharacterization
    binary_fields: BinaryFieldResult
    pattern_analysis: PatternFrequencyResult
    protocol_hypothesis: str
    confidence: float
    recommendations: list[str]


def reverse_engineer_protocol(
    trace: WaveformTrace,
) -> ReverseEngineeringResult:
    """Comprehensive reverse engineering workflow per UNKNOWN-005.

    Combines all unknown signal analysis techniques to build
    a hypothesis about the protocol in use.

    Args:
        trace: Signal trace to reverse engineer.

    Returns:
        ReverseEngineeringResult with comprehensive analysis.

    Example:
        >>> result = reverse_engineer_protocol(trace)
        >>> print(f"Protocol hypothesis: {result.protocol_hypothesis}")
        >>> print(f"Confidence: {result.confidence:.1%}")
        >>> for rec in result.recommendations:
        ...     print(f"- {rec}")

    References:
        UNKNOWN-005: Reverse Engineering Workflow
    """
    # Run all analysis steps
    signal_char = characterize_unknown_signal(trace)
    binary_fields = detect_binary_fields(trace)
    pattern_analysis = analyze_pattern_frequency(trace)

    # Build hypothesis
    protocol_hypothesis = "Unknown"
    confidence = 0.0

    if signal_char.suggested_protocols:
        protocol_hypothesis = signal_char.suggested_protocols[0][0]
        confidence = signal_char.suggested_protocols[0][1]

    # Generate recommendations
    recommendations = []

    if signal_char.signal_type != "digital":
        recommendations.append("Signal appears analog - check if correct probe/channel")

    if binary_fields.field_count == 0:
        recommendations.append("No binary fields detected - try adjusting threshold")

    if binary_fields.encoding == "manchester":
        recommendations.append("Manchester encoding detected - common in Ethernet, 1-Wire")

    if pattern_analysis.repetition_rate > 0.5:
        recommendations.append(
            "High pattern repetition - likely periodic protocol (e.g., I2C polling)"
        )

    if signal_char.snr_db < 10:
        recommendations.append("Low SNR - consider using averaging or filtering")

    if not signal_char.suggested_protocols:
        recommendations.append("No protocol match - try capturing with different settings")

    if binary_fields.bit_rate is not None:
        recommendations.append(f"Estimated bit rate: {binary_fields.bit_rate:.0f} bps")

    return ReverseEngineeringResult(
        signal_char=signal_char,
        binary_fields=binary_fields,
        pattern_analysis=pattern_analysis,
        protocol_hypothesis=protocol_hypothesis,
        confidence=confidence,
        recommendations=recommendations,
    )


__all__ = [
    "BinaryFieldResult",
    "PatternFrequencyResult",
    "ReverseEngineeringResult",
    "UnknownSignalCharacterization",
    "analyze_pattern_frequency",
    "characterize_unknown_signal",
    "detect_binary_fields",
    "reverse_engineer_protocol",
]
