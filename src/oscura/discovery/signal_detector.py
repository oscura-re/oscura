"""Automatic signal characterization and type detection.

This module provides intelligent signal type detection, extracting
characteristics without requiring user expertise.


Example:
    >>> from oscura.discovery import characterize_signal
    >>> result = characterize_signal(trace)
    >>> print(f"{result.signal_type}: {result.confidence:.2f}")
    UART: 0.94

References:
    IEEE 181-2011: Transitional Waveform Definitions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.analyzers.statistics.basic import basic_stats
from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray

SignalType = Literal["digital", "analog", "pwm", "uart", "spi", "i2c", "unknown"]


@dataclass
class SignalCharacterization:
    """Result of automatic signal characterization.

    Contains detected signal type, confidence score, and extracted parameters.

    Attributes:
        signal_type: Detected signal type.
        confidence: Confidence score (0.0-1.0).
        voltage_low: Low voltage level in volts.
        voltage_high: High voltage level in volts.
        frequency_hz: Dominant frequency in Hz.
        parameters: Additional signal-specific parameters.
        quality_metrics: Signal quality measurements.
        alternatives: Alternative signal type suggestions.

    Example:
        >>> result = characterize_signal(trace)
        >>> if result.confidence >= 0.8:
        ...     print(f"High confidence: {result.signal_type}")
    """

    signal_type: SignalType
    confidence: float
    voltage_low: float
    voltage_high: float
    frequency_hz: float
    parameters: dict[str, Any] = field(default_factory=dict)
    quality_metrics: dict[str, float] = field(default_factory=dict)
    alternatives: list[tuple[SignalType, float]] = field(default_factory=list)


def characterize_signal(
    trace: WaveformTrace | DigitalTrace,
    *,
    confidence_threshold: float = 0.6,
    include_alternatives: bool = False,
    min_alternatives: int = 3,
) -> SignalCharacterization:
    """Automatically characterize signal type and properties.

    Analyzes waveform to detect signal type (digital, analog, PWM, UART, SPI, I2C)
    and extract key parameters without requiring manual configuration.

    Args:
        trace: Input waveform or digital trace.
        confidence_threshold: Minimum confidence for primary detection (0.0-1.0).
        include_alternatives: Whether to include alternative suggestions.
        min_alternatives: Minimum number of alternatives when confidence is low.

    Returns:
        SignalCharacterization with detected type and parameters.

    Raises:
        ValueError: If trace is empty or invalid.

    Example:
        >>> result = characterize_signal(trace, confidence_threshold=0.8)
        >>> print(f"Signal: {result.signal_type}")
        >>> print(f"Confidence: {result.confidence:.2f}")
        >>> print(f"Voltage: {result.voltage_low:.2f}V to {result.voltage_high:.2f}V")
        Signal: UART
        Confidence: 0.94
        Voltage: 0.02V to 3.28V

    References:
        DISC-001: Automatic Signal Characterization
    """
    # Validate and extract trace data
    data, sample_rate, is_analog = _validate_and_extract_trace(trace)

    # Compute basic statistics and voltage levels
    stats = basic_stats(data)
    voltage_low, voltage_high, voltage_swing = _compute_voltage_levels(data)

    # Run all signal type detectors
    candidates = _detect_all_signal_types(data, sample_rate, voltage_swing, is_analog)

    # Select best signal type with refinement logic
    best_type, best_confidence = _select_best_signal_type(candidates)

    # Estimate dominant frequency
    frequency_hz = _estimate_frequency(data, sample_rate)

    # Extract type-specific parameters and quality metrics
    parameters = _extract_parameters(best_type, data, sample_rate, voltage_low, voltage_high)
    quality_metrics = _compute_quality_metrics(
        data, stats, sample_rate, voltage_low, voltage_high, candidates["digital"]
    )

    # Prepare alternatives if requested
    alternatives = _build_alternatives(
        candidates,
        best_type,
        best_confidence,
        include_alternatives,
        confidence_threshold,
        min_alternatives,
    )

    return SignalCharacterization(
        signal_type=best_type,
        confidence=round(best_confidence, 2),
        voltage_low=voltage_low,
        voltage_high=voltage_high,
        frequency_hz=frequency_hz,
        parameters=parameters,
        quality_metrics=quality_metrics,
        alternatives=alternatives,
    )


def _validate_and_extract_trace(
    trace: WaveformTrace | DigitalTrace,
) -> tuple[NDArray[np.floating[Any]], float, bool]:
    """Validate trace and extract data, sample rate, and analog flag.

    Args:
        trace: Input waveform or digital trace.

    Returns:
        Tuple of (data, sample_rate, is_analog).

    Raises:
        ValueError: If trace is empty.
    """
    if len(trace) == 0:
        raise ValueError("Cannot characterize empty trace")

    if isinstance(trace, WaveformTrace):
        return trace.data, trace.metadata.sample_rate, True
    else:
        return trace.data.astype(np.float64), trace.metadata.sample_rate, False


def _compute_voltage_levels(
    data: NDArray[np.floating[Any]],
) -> tuple[float, float, float]:
    """Compute voltage levels using robust percentiles.

    Args:
        data: Signal data array.

    Returns:
        Tuple of (voltage_low, voltage_high, voltage_swing).
    """
    voltage_low = float(np.percentile(data, 5))
    voltage_high = float(np.percentile(data, 95))
    voltage_swing = voltage_high - voltage_low
    return voltage_low, voltage_high, voltage_swing


def _detect_all_signal_types(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
    is_analog: bool,
) -> dict[SignalType, float]:
    """Run all signal type detectors and return confidence scores.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage swing.
        is_analog: Whether input is from analog trace.

    Returns:
        Dictionary mapping signal types to confidence scores.
    """
    return {
        "digital": _detect_digital(data, voltage_swing),
        "analog": _detect_analog(data, voltage_swing, is_analog),
        "pwm": _detect_pwm(data, sample_rate, voltage_swing),
        "uart": _detect_uart(data, sample_rate, voltage_swing),
        "spi": _detect_spi(data, sample_rate, voltage_swing),
        "i2c": _detect_i2c(data, sample_rate, voltage_swing),
    }


def _select_best_signal_type(
    candidates: dict[SignalType, float],
) -> tuple[SignalType, float]:
    """Select best signal type with refinement logic.

    Args:
        candidates: Dictionary of signal types and confidence scores.

    Returns:
        Tuple of (best_type, best_confidence).
    """
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    best_type, best_confidence = sorted_candidates[0]

    # If confidence is too low, mark as unknown
    if best_confidence < 0.5:
        return "unknown", best_confidence

    # Refine analog classification if digital characteristics present
    if best_type == "analog":
        return _refine_analog_classification(candidates)

    return best_type, best_confidence


def _refine_analog_classification(
    candidates: dict[SignalType, float],
) -> tuple[SignalType, float]:
    """Refine analog classification when digital characteristics are present.

    Args:
        candidates: Dictionary of signal types and confidence scores.

    Returns:
        Tuple of (refined_type, confidence).
    """
    digital_confidence = candidates.get("digital", 0)
    analog_confidence = candidates.get("analog", 0)
    protocol_confidence = max(
        candidates.get("uart", 0),
        candidates.get("spi", 0),
        candidates.get("pwm", 0),
    )

    # If digital or protocol detectors have some confidence, don't call it purely analog
    if digital_confidence > 0.3 or protocol_confidence > 0.2:
        if protocol_confidence > 0.3:
            return "unknown", protocol_confidence
        elif digital_confidence > 0.4:
            return "digital", digital_confidence
        else:
            return "unknown", max(digital_confidence, protocol_confidence, analog_confidence)

    return "analog", analog_confidence


def _compute_quality_metrics(
    data: NDArray[np.floating[Any]],
    stats: dict[str, float],
    sample_rate: float,
    voltage_low: float,
    voltage_high: float,
    digital_confidence: float,
) -> dict[str, float]:
    """Compute signal quality metrics.

    Args:
        data: Signal data array.
        stats: Basic statistics dictionary.
        sample_rate: Sample rate in Hz.
        voltage_low: Low voltage level.
        voltage_high: High voltage level.
        digital_confidence: Confidence that signal is digital.

    Returns:
        Dictionary of quality metrics.
    """
    noise_level = _estimate_noise_level(data, voltage_low, voltage_high, digital_confidence)
    return {
        "snr_db": _estimate_snr(data, stats),
        "jitter_ns": _estimate_jitter(data, sample_rate) * 1e9,
        "noise_level": noise_level,
    }


def _build_alternatives(
    candidates: dict[SignalType, float],
    best_type: SignalType,
    best_confidence: float,
    include_alternatives: bool,
    confidence_threshold: float,
    min_alternatives: int,
) -> list[tuple[SignalType, float]]:
    """Build list of alternative signal type suggestions.

    Args:
        candidates: All candidate types with confidence scores.
        best_type: Best detected signal type.
        best_confidence: Confidence of best type.
        include_alternatives: Whether to include alternatives.
        confidence_threshold: Threshold for including alternatives.
        min_alternatives: Minimum number of alternatives.

    Returns:
        List of (signal_type, confidence) tuples.
    """
    alternatives: list[tuple[SignalType, float]] = []

    if not (include_alternatives or best_confidence < confidence_threshold):
        return alternatives

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    # Include top alternatives (excluding the winner)
    for sig_type, conf in sorted_candidates[1:]:
        if len(alternatives) >= min_alternatives:
            break
        if conf >= 0.3:  # Only include reasonable alternatives
            alternatives.append((sig_type, conf))

    return alternatives


def _estimate_noise_level(
    data: NDArray[np.floating[Any]],
    voltage_low: float,
    voltage_high: float,
    digital_confidence: float,
) -> float:
    """Estimate noise level in signal.

    For digital signals, measures deviation from ideal logic levels.
    For analog signals, uses normalized std.

    Args:
        data: Signal data array.
        voltage_low: Low voltage level.
        voltage_high: High voltage level.
        digital_confidence: Confidence that signal is digital.

    Returns:
        Noise level as fraction of voltage swing (0.0-1.0).
    """
    voltage_swing = voltage_high - voltage_low
    if voltage_swing == 0:
        return 0.0

    # For digital signals, estimate noise from deviation around logic levels
    if digital_confidence >= 0.5:
        threshold = (voltage_high + voltage_low) / 2
        low_samples = data[data < threshold]
        high_samples = data[data >= threshold]

        noise_estimates = []
        if len(low_samples) > 0:
            # Deviation from the low level
            low_level = np.min(data)
            low_noise = np.std(low_samples - low_level)
            noise_estimates.append(low_noise)
        if len(high_samples) > 0:
            # Deviation from the high level
            high_level = np.max(data)
            high_noise = np.std(high_samples - high_level)
            noise_estimates.append(high_noise)

        if noise_estimates:
            avg_noise = np.mean(noise_estimates)
            return float(avg_noise / voltage_swing)

    # For analog signals, use std as fraction of range
    # But cap it at 0.5 to indicate high variability, not noise
    std_noise = float(np.std(data) / voltage_swing)
    return min(0.5, std_noise)


def _detect_digital(data: NDArray[np.floating[Any]], voltage_swing: float) -> float:
    """Detect digital signal characteristics.

    Args:
        data: Signal data array.
        voltage_swing: Peak-to-peak voltage swing.

    Returns:
        Confidence score (0.0-1.0).
    """
    if voltage_swing == 0:
        return 0.0

    # Check for bimodal distribution (two distinct levels)
    hist, bin_edges = np.histogram(data, bins=50)

    # Normalize histogram
    hist = hist / np.sum(hist)

    # Find peaks in histogram (should have 2 for digital)
    peak_threshold = np.max(hist) * 0.3
    peaks = np.where(hist > peak_threshold)[0]

    if len(peaks) < 2:
        return 0.3  # Low confidence

    # Check if peaks are well separated
    peak_separation = (bin_edges[peaks[-1]] - bin_edges[peaks[0]]) / voltage_swing

    # Digital signals spend most time at rails
    edge_bins = hist[:5].sum() + hist[-5:].sum()

    # Combine factors
    bimodal_score = min(1.0, len(peaks) / 2.0)  # Closer to 2 peaks is better
    separation_score = min(1.0, peak_separation)
    rail_score = min(1.0, edge_bins * 2)  # More time at rails is better

    confidence = bimodal_score * 0.4 + separation_score * 0.3 + rail_score * 0.3
    return min(0.95, confidence)  # type: ignore[no-any-return]


def _detect_analog(data: NDArray[np.floating[Any]], voltage_swing: float, is_analog: bool) -> float:
    """Detect analog signal characteristics.

    Args:
        data: Signal data array.
        voltage_swing: Peak-to-peak voltage swing.
        is_analog: Whether input is from analog trace.

    Returns:
        Confidence score (0.0-1.0).
    """
    if voltage_swing == 0:
        return 0.0

    # Check if signal has strong digital characteristics first
    # If it does, this is NOT analog - reduce confidence significantly
    digital_confidence = _detect_digital(data, voltage_swing)
    if digital_confidence >= 0.6:
        # Strong digital signal - very low analog confidence
        return max(0.0, 0.4 - digital_confidence * 0.3)

    # Analog signals have continuous distribution
    hist, _ = np.histogram(data, bins=50)
    hist = hist / np.sum(hist)

    # Check for uniform or Gaussian-like distribution
    uniform_score = 1.0 - np.std(hist)

    # Check for smooth transitions (not many abrupt changes)
    diff = np.diff(data)
    smooth_score = 1.0 - min(1.0, np.mean(np.abs(diff)) / voltage_swing)

    # Analog traces get boost
    source_score = 1.0 if is_analog else 0.5  # Reduced from 0.7

    confidence = uniform_score * 0.4 + smooth_score * 0.3 + source_score * 0.3

    # Further reduce if there's any digital characteristics
    if digital_confidence > 0.3:
        confidence *= 1.0 - digital_confidence * 0.5

    return min(0.9, confidence)  # type: ignore[no-any-return]


def _detect_pwm(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
) -> float:
    """Detect PWM signal characteristics.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage swing.

    Returns:
        Confidence score (0.0-1.0).
    """
    if voltage_swing == 0 or len(data) < 100:
        return 0.0

    # PWM should have digital levels
    digital_score = _detect_digital(data, voltage_swing)

    if digital_score < 0.5:
        return 0.0

    # Threshold signal
    threshold = (np.max(data) + np.min(data)) / 2
    digital = data > threshold

    # Find transitions
    transitions = np.diff(digital.astype(int))
    rising = np.where(transitions > 0)[0]
    falling = np.where(transitions < 0)[0]

    if len(rising) < 3 or len(falling) < 3:
        return 0.0

    # Check for periodic transitions
    rising_periods = np.diff(rising)
    period_std = np.std(rising_periods) if len(rising_periods) > 0 else 0
    mean_period = np.mean(rising_periods) if len(rising_periods) > 0 else 1

    periodicity_score = 1.0 - min(1.0, period_std / (mean_period + 1e-10))

    # PWM should have varying duty cycle
    duty_cycles = []
    for i in range(min(len(rising), len(falling))):
        if i < len(falling) and falling[i] > rising[i]:
            duty = (falling[i] - rising[i]) / (mean_period + 1e-10)
            duty_cycles.append(duty)

    duty_variation = np.std(duty_cycles) if len(duty_cycles) > 1 else 0
    variation_score = min(1.0, duty_variation * 5)  # Some variation expected

    confidence = digital_score * 0.3 + periodicity_score * 0.5 + variation_score * 0.2

    # Boost if strong periodicity with variation (classic PWM signature)
    if periodicity_score > 0.7 and variation_score > 0.3:
        confidence = min(0.94, confidence * 1.1)

    return min(0.94, confidence)


def _detect_uart(
    data: NDArray[np.floating[Any]], sample_rate: float, voltage_swing: float
) -> float:
    """Detect UART signal characteristics.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage swing.

    Returns:
        Confidence score (0.0-1.0).
    """
    if voltage_swing == 0 or len(data) < 200:
        return 0.0

    # UART should be digital
    digital_score = _detect_digital(data, voltage_swing)
    if digital_score < 0.7:  # Strict threshold for UART
        return 0.0

    # Check for bimodal (two-level) distribution
    # UART should have primarily two voltage levels, not continuous values
    hist, _ = np.histogram(data, bins=50)
    hist = hist / np.sum(hist)

    # Count significant histogram bins (>5% of samples)
    significant_bins = np.sum(hist > 0.05)

    # UART should have at most 2-4 significant bins (low and high with some noise)
    # Sine wave will have many bins
    if significant_bins > 6:
        return 0.0

    # Threshold signal
    threshold = (np.max(data) + np.min(data)) / 2
    digital = data > threshold

    # Find edges
    transitions = np.diff(digital.astype(int))
    edges = np.where(np.abs(transitions) > 0)[0]

    if len(edges) < 10:
        return 0.0

    # UART has consistent bit timing
    edge_intervals = np.diff(edges)

    # Look for common baud rates
    common_bauds = [9600, 19200, 38400, 57600, 115200]
    baud_scores = []

    for baud in common_bauds:
        bit_period_samples = sample_rate / baud
        # Count edges that align with this baud rate (stricter alignment)
        aligned = np.sum(np.abs(edge_intervals % bit_period_samples) < bit_period_samples * 0.15)
        baud_scores.append(aligned / len(edge_intervals))

    timing_score = max(baud_scores) if baud_scores else 0.0

    # UART requires strong timing alignment
    if timing_score < 0.4:
        return 0.0

    # UART idles high typically
    idle_score = np.mean(digital[-100:])

    confidence = digital_score * 0.3 + timing_score * 0.6 + idle_score * 0.1

    # Boost confidence if timing alignment is strong
    if timing_score > 0.7:
        confidence = min(0.96, confidence * 1.1)

    return min(0.96, confidence)  # type: ignore[no-any-return]


def _detect_spi(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
) -> float:
    """Detect SPI signal characteristics.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage swing.

    Returns:
        Confidence score (0.0-1.0).
    """
    if voltage_swing == 0 or len(data) < 200:
        return 0.0

    # SPI should be digital
    digital_score = _detect_digital(data, voltage_swing)
    if digital_score < 0.6:
        return 0.0

    # Threshold signal
    threshold = (np.max(data) + np.min(data)) / 2
    digital = data > threshold

    # Find edges
    transitions = np.diff(digital.astype(int))
    edges = np.where(np.abs(transitions) > 0)[0]

    if len(edges) < 20:
        return 0.0

    # SPI typically has bursts of regular clock transitions
    edge_intervals = np.diff(edges)

    # Check for consistent clock period
    median_interval = np.median(edge_intervals)
    interval_std = np.std(edge_intervals)
    consistency_score = 1.0 - min(1.0, interval_std / (median_interval + 1e-10))

    # SPI has many transitions (clock toggling)
    transition_density = len(edges) / len(data)
    density_score = min(1.0, transition_density * 20)

    confidence = digital_score * 0.3 + consistency_score * 0.5 + density_score * 0.2

    # Boost confidence if consistency is very high (strong clock signal)
    if consistency_score > 0.8 and density_score > 0.5:
        confidence = min(0.95, confidence * 1.15)

    return min(0.95, confidence)  # type: ignore[no-any-return]


def _detect_i2c(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_swing: float,
) -> float:
    """Detect I2C signal characteristics.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_swing: Peak-to-peak voltage swing.

    Returns:
        Confidence score (0.0-1.0).
    """
    # I2C detection requires both SDA and SCL, single channel is limited
    # This is a placeholder that gives low confidence
    digital_score = _detect_digital(data, voltage_swing)
    return min(0.6, digital_score * 0.5)


def _estimate_frequency(data: NDArray[np.floating[Any]], sample_rate: float) -> float:
    """Estimate dominant frequency in signal.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.

    Returns:
        Dominant frequency in Hz.
    """
    if len(data) < 10:
        return 0.0

    # Simple zero-crossing based frequency estimate
    mean_val = np.mean(data)
    crossings = np.where(np.diff(np.sign(data - mean_val)) != 0)[0]

    if len(crossings) < 2:
        return 0.0

    # Average period between crossings (half periods)
    avg_half_period = np.mean(np.diff(crossings))
    period_samples = avg_half_period * 2

    frequency = sample_rate / period_samples if period_samples > 0 else 0.0
    return frequency


def _estimate_snr(data: NDArray[np.floating[Any]], stats: dict[str, float]) -> float:
    """Estimate signal-to-noise ratio.

    Args:
        data: Signal data array.
        stats: Basic statistics dictionary.

    Returns:
        Estimated SNR in dB.
    """
    signal_power = stats["mean"] ** 2
    noise_power = stats["variance"]

    if noise_power == 0:
        return 100.0  # Very high SNR

    snr = signal_power / noise_power
    snr_db = 10 * np.log10(snr) if snr > 0 else 0.0

    return max(0.0, min(100.0, snr_db))


def _estimate_jitter(data: NDArray[np.floating[Any]], sample_rate: float) -> float:
    """Estimate timing jitter.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated jitter in seconds.
    """
    # Simple edge-to-edge jitter estimate
    threshold = (np.max(data) + np.min(data)) / 2
    digital = data > threshold
    edges = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(edges) < 3:
        return 0.0

    edge_intervals = np.diff(edges)
    jitter_samples = np.std(edge_intervals)
    jitter_seconds = jitter_samples / sample_rate

    return jitter_seconds  # type: ignore[no-any-return]


def _extract_parameters(
    signal_type: SignalType,
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    voltage_low: float,
    voltage_high: float,
) -> dict[str, Any]:
    """Extract signal-specific parameters.

    Args:
        signal_type: Detected signal type.
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        voltage_low: Low voltage level.
        voltage_high: High voltage level.

    Returns:
        Dictionary of parameters specific to signal type.
    """
    params: dict[str, Any] = {}

    if signal_type in ("digital", "uart", "spi", "i2c"):
        # Add logic level parameters
        logic_family = _guess_logic_family(voltage_low, voltage_high)
        if logic_family != "Unknown":
            params["logic_family"] = logic_family

    if signal_type == "pwm":
        # Calculate duty cycle
        threshold = (voltage_high + voltage_low) / 2
        digital = data > threshold
        duty_cycle = np.mean(digital)
        params["duty_cycle"] = round(duty_cycle, 3)

    if signal_type == "uart":
        # Estimate baud rate
        params["estimated_baud"] = _estimate_baud_rate(data, sample_rate)

    return params


def _guess_logic_family(voltage_low: float, voltage_high: float) -> str:
    """Guess logic family from voltage levels.

    Args:
        voltage_low: Low voltage level in volts.
        voltage_high: High voltage level in volts.

    Returns:
        Logic family name.
    """
    voltage_swing = voltage_high - voltage_low

    # Match to closest standard voltage level
    # This handles noise better than fixed ranges
    standard_levels = [
        (1.8, "1.8V LVCMOS"),
        (3.3, "3.3V LVCMOS"),
        (5.0, "5V TTL/CMOS"),
    ]

    # Find closest match
    closest_diff = float("inf")
    second_closest_diff = float("inf")
    closest_family = "Unknown"
    closest_level = 0.0

    for level, family in standard_levels:
        diff = abs(voltage_swing - level)
        if diff < closest_diff:
            second_closest_diff = closest_diff
            closest_diff = diff
            closest_family = family
            closest_level = level
        elif diff < second_closest_diff:
            second_closest_diff = diff

    # Only return a match if:
    # 1. Closest match is within 50% tolerance
    # 2. AND it's significantly closer than second-best (not ambiguous)
    if closest_diff == float("inf") or closest_diff > closest_level * 0.5:
        return "Unknown"

    # Check if ambiguous (second closest is also pretty close)
    # If second-best is within 20% more distance, it's too ambiguous
    if second_closest_diff < closest_diff * 1.2:
        return "Unknown"  # Too ambiguous

    return closest_family


def _estimate_baud_rate(data: NDArray[np.floating[Any]], sample_rate: float) -> int:
    """Estimate UART baud rate.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated baud rate in bps.
    """
    # Find bit period from edge intervals
    threshold = (np.max(data) + np.min(data)) / 2
    digital = data > threshold
    edges = np.where(np.diff(digital.astype(int)) != 0)[0]

    if len(edges) < 10:
        return 9600  # Default fallback

    edge_intervals = np.diff(edges)
    # Use median to be robust to outliers
    median_interval = np.median(edge_intervals)
    estimated_baud = int(sample_rate / median_interval)

    # Snap to common baud rates
    common_bauds = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]
    closest_baud = min(common_bauds, key=lambda x: abs(x - estimated_baud))

    return closest_baud


__all__ = [
    "SignalCharacterization",
    "SignalType",
    "characterize_signal",
]
