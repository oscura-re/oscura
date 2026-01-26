"""Signal classification and measurement intelligence for Oscura.

This module provides intelligent signal type detection, quality assessment,
and measurement suitability checking to help users understand why they might
get NaN results and which measurements are appropriate for their signals.


Example:
    >>> import oscura as osc
    >>> trace = osc.load('signal.wfm')
    >>> classification = osc.classify_signal(trace)
    >>> print(f"Signal type: {classification['type']}")
    >>> print(f"Characteristics: {classification['characteristics']}")
    >>> quality = osc.assess_signal_quality(trace)
    >>> print(f"SNR: {quality['snr']:.1f} dB")
    >>> suggestions = osc.suggest_measurements(trace)
    >>> print(f"Recommended measurements: {suggestions}")

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
    IEEE 1057-2017: Standard for Digitizing Waveform Recorders
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from oscura.core.types import WaveformTrace
    from oscura.reporting.config import AnalysisDomain


# =============================================================================
# Helper Functions for classify_signal
# =============================================================================


def _extract_signal_data(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    sample_rate: float,
) -> tuple[NDArray[np.floating[Any]], float]:
    """Extract signal data and sample rate from trace or ndarray.

    Args:
        trace: Input waveform trace or numpy array.
        sample_rate: Sample rate in Hz (only used if trace is ndarray).

    Returns:
        Tuple of (data array, sample rate).
    """
    if isinstance(trace, np.ndarray):
        return trace, sample_rate
    return trace.data, trace.metadata.sample_rate


def _create_insufficient_data_result() -> dict[str, Any]:
    """Create result dict for insufficient data case.

    Returns:
        Classification result dict with unknown type.
    """
    return {
        "type": "unknown",
        "signal_type": "unknown",
        "is_digital": False,
        "is_periodic": False,
        "characteristics": ["insufficient_data"],
        "dc_component": False,
        "frequency_estimate": None,
        "dominant_frequency": None,
        "snr_db": None,
        "confidence": 0.0,
        "noise_level": 0.0,
        "levels": None,
    }


def _compute_signal_statistics(data: NDArray[np.floating[Any]]) -> dict[str, float]:
    """Compute basic signal statistics.

    Args:
        data: Signal data array.

    Returns:
        Dict with mean, std, min, max, amplitude statistics.
    """
    mean_val = float(np.mean(data))
    std_val = float(np.std(data))
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    amplitude = max_val - min_val

    return {
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "amplitude": amplitude,
    }


def _is_dc_signal(stats: dict[str, float]) -> bool:
    """Check if signal is DC (very low variation).

    Args:
        stats: Signal statistics from _compute_signal_statistics.

    Returns:
        True if signal is DC.
    """
    # Use coefficient of variation (CV) for DC detection
    cv = stats["std"] / (abs(stats["mean"]) + stats["amplitude"] / 2 + 1e-12)
    return stats["amplitude"] < 1e-9 or cv < 0.005


def _create_dc_result(stats: dict[str, float]) -> dict[str, Any]:
    """Create result dict for DC signal.

    Args:
        stats: Signal statistics.

    Returns:
        Classification result dict for DC signal.
    """
    return {
        "type": "dc",
        "signal_type": "dc",
        "is_digital": False,
        "is_periodic": False,
        "characteristics": ["constant"],
        "dc_component": True,
        "frequency_estimate": None,
        "dominant_frequency": None,
        "snr_db": None,
        "confidence": 0.95,
        "noise_level": stats["std"],
        "levels": None,
    }


def _add_noise_characteristics(
    characteristics: list[str],
    noise_level: float,
    amplitude: float,
) -> None:
    """Add noise-related characteristics to list.

    Args:
        characteristics: List to append characteristics to (modified in-place).
        noise_level: Estimated noise level.
        amplitude: Signal amplitude.
    """
    noise_ratio = noise_level / (amplitude + 1e-12)

    if noise_ratio < 0.05:
        characteristics.append("clean")
    elif noise_ratio < 0.15:
        characteristics.append("low_noise")
    elif noise_ratio < 0.30:
        characteristics.append("moderate_noise")
    else:
        characteristics.append("noisy")


def _classify_periodicity(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    threshold: float,
    is_digital: bool,
    digital_levels: dict[str, float] | None,
    n: int,
) -> tuple[bool, float | None, float]:
    """Classify signal periodicity using multiple detection methods.

    Args:
        data: Signal data array.
        sample_rate: Sample rate in Hz.
        threshold: Periodicity threshold for autocorrelation.
        is_digital: Whether signal is digital.
        digital_levels: Digital levels dict (if digital).
        n: Data length.

    Returns:
        Tuple of (is_periodic, period_estimate, periodicity_score).
    """
    # Try autocorrelation first
    is_periodic, period_estimate, periodicity_score = _detect_periodicity(
        data, sample_rate, threshold
    )

    # For digital signals, try edge-based detection
    if not is_periodic and is_digital:
        edge_periodic, edge_period, edge_confidence = _detect_edge_periodicity(
            data, sample_rate, digital_levels
        )
        if edge_periodic:
            return edge_periodic, edge_period, edge_confidence

    # Try FFT-based detection for undersampled signals
    if n >= 64:
        fft_periodic, fft_period, fft_confidence = _detect_periodicity_fft(data, sample_rate)
        if fft_periodic:
            # Compare FFT and autocorrelation results
            if is_periodic and period_estimate is not None:
                # Reconcile conflicting frequency estimates
                period_estimate, periodicity_score = _reconcile_period_estimates(
                    period_estimate, fft_period, fft_confidence
                )
            else:
                # Only FFT detected periodicity
                is_periodic = fft_periodic
                period_estimate = fft_period
                periodicity_score = fft_confidence

    return is_periodic, period_estimate, periodicity_score


def _reconcile_period_estimates(
    auto_period: float,
    fft_period: float | None,
    fft_confidence: float,
) -> tuple[float | None, float]:
    """Reconcile autocorrelation and FFT period estimates.

    Args:
        auto_period: Period from autocorrelation.
        fft_period: Period from FFT.
        fft_confidence: Confidence from FFT.

    Returns:
        Tuple of (reconciled period, confidence).
    """
    if fft_period is None or fft_period <= 0:
        return auto_period, fft_confidence

    # Calculate frequency ratio
    auto_freq = 1.0 / auto_period if auto_period > 0 else 0
    fft_freq = 1.0 / fft_period if fft_period > 0 else 0
    freq_ratio = max(auto_freq, fft_freq) / (min(auto_freq, fft_freq) + 1e-12)

    # If frequencies differ >20%, prefer higher frequency
    if freq_ratio > 1.2 and fft_freq > auto_freq:
        return fft_period, fft_confidence

    return auto_period, fft_confidence


def _add_transient_characteristics(
    characteristics: list[str],
    data: NDArray[np.floating[Any]],
    stats: dict[str, float],
    digital_levels: dict[str, float] | None,
) -> None:
    """Add pulsed/transient characteristics to list.

    Args:
        characteristics: List to append characteristics to (modified in-place).
        data: Signal data array.
        stats: Signal statistics.
        digital_levels: Digital levels dict (if digital).
    """
    edge_count = _count_edges(data, digital_levels)
    samples_per_edge = len(data) / max(edge_count, 1)

    if edge_count > 2 and samples_per_edge > 100:
        characteristics.append("pulsed")
    elif edge_count < 3 and stats["amplitude"] > stats["std"] * 2:
        characteristics.append("transient")


def _detect_mixed_signal(
    data: NDArray[np.floating[Any]],
    digital_levels: dict[str, float],
) -> bool:
    """Check if signal is mixed (digital transitions + analog variation).

    Args:
        data: Signal data array.
        digital_levels: Digital levels dict with "low" and "high" keys.

    Returns:
        True if signal appears mixed.
    """
    threshold = (digital_levels["low"] + digital_levels["high"]) / 2
    low_region = data[data < threshold]
    high_region = data[data >= threshold]

    if len(low_region) == 0 or len(high_region) == 0:
        return False

    low_std = np.std(low_region)
    high_std = np.std(high_region)
    level_separation = digital_levels["high"] - digital_levels["low"]

    # Type narrowing: numpy comparison returns np.bool_
    result: bool = bool(low_std > level_separation * 0.1 or high_std > level_separation * 0.1)
    return result


def _compute_snr(amplitude: float, noise_level: float) -> float | None:
    """Compute signal-to-noise ratio.

    Args:
        amplitude: Signal amplitude.
        noise_level: Noise level.

    Returns:
        SNR in dB or None if not calculable.
    """
    if amplitude <= noise_level * 10:
        return None

    signal_power = amplitude**2 / 8  # Approximate for most waveforms
    noise_power = noise_level**2

    if noise_power <= 1e-20:
        return None

    # Type narrowing: numpy operations return numpy types
    snr_db: float = float(10 * np.log10(signal_power / noise_power))
    return snr_db


def _create_classification_result(
    signal_type: str,
    is_digital: bool,
    is_periodic: bool,
    characteristics: list[str],
    dc_component: bool,
    frequency_estimate: float | None,
    snr_db: float | None,
    confidence: float,
    noise_level: float,
    digital_levels: dict[str, float] | None,
) -> dict[str, Any]:
    """Create classification result dictionary.

    Args:
        signal_type: Signal type string.
        is_digital: Whether signal is digital.
        is_periodic: Whether signal is periodic.
        characteristics: List of characteristic strings.
        dc_component: Whether DC component is present.
        frequency_estimate: Estimated frequency in Hz.
        snr_db: SNR in dB.
        confidence: Classification confidence.
        noise_level: Noise level.
        digital_levels: Digital levels dict (if digital).

    Returns:
        Classification result dictionary.
    """
    return {
        "type": signal_type,
        "signal_type": signal_type,
        "is_digital": is_digital,
        "is_periodic": is_periodic,
        "characteristics": characteristics,
        "dc_component": dc_component,
        "frequency_estimate": frequency_estimate,
        "dominant_frequency": frequency_estimate,
        "snr_db": float(snr_db) if snr_db is not None else None,
        "confidence": float(confidence),
        "noise_level": float(noise_level),
        "levels": digital_levels if is_digital else None,
    }


# =============================================================================
# Public API Functions
# =============================================================================


def classify_signal(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    sample_rate: float = 1.0,
    *,
    digital_threshold_ratio: float = 0.8,
    dc_threshold_percent: float = 90.0,
    periodicity_threshold: float = 0.7,
) -> dict[str, Any]:
    """Classify signal type and characteristics.

    Automatically detects whether a signal is digital, analog, or mixed,
    identifies key characteristics like periodicity and noise.

    Args:
        trace: Input waveform trace or numpy array to classify.
        sample_rate: Sample rate in Hz (only used if trace is ndarray).
        digital_threshold_ratio: Ratio for digital detection (0-1).
        dc_threshold_percent: Percentage of DC for DC classification.
        periodicity_threshold: Correlation threshold for periodic (0-1).

    Returns:
        Dictionary with signal_type, is_digital, is_periodic, characteristics,
        frequency_estimate, snr_db, confidence, noise_level, levels.

    Example:
        >>> info = osc.classify_signal(trace)
        >>> print(f"Type: {info['signal_type']}")

    References:
        IEEE 181-2011: Digital waveform characterization
    """
    data, trace_sample_rate = _extract_signal_data(trace, sample_rate)
    n = len(data)

    if n < 10:
        return _create_insufficient_data_result()

    stats = _compute_signal_statistics(data)

    if _is_dc_signal(stats):
        return _create_dc_result(stats)

    is_digital, digital_levels, confidence = _detect_digital_signal(data, digital_threshold_ratio)
    signal_type = "digital" if is_digital else "analog"

    characteristics = _build_characteristics(data, stats, is_digital, digital_levels)

    is_periodic, frequency_estimate, periodicity_score = _analyze_periodicity(
        data,
        trace_sample_rate,
        periodicity_threshold,
        is_digital,
        digital_levels,
        n,
        characteristics,
    )
    confidence = max(confidence, periodicity_score) if is_periodic else confidence

    dc_component = abs(stats["mean"]) > (stats["amplitude"] * dc_threshold_percent / 100.0)
    _add_transient_characteristics(
        characteristics, data, stats, digital_levels if is_digital else None
    )

    if is_digital and digital_levels and _detect_mixed_signal(data, digital_levels):
        signal_type = "mixed"
        characteristics.append("analog_variation")

    noise_level = _estimate_noise_level(data)
    snr_db = _compute_snr(stats["amplitude"], noise_level)

    return _create_classification_result(
        signal_type,
        is_digital,
        is_periodic,
        characteristics,
        dc_component,
        frequency_estimate,
        snr_db,
        confidence,
        noise_level,
        digital_levels,
    )


def _build_characteristics(
    data: NDArray[np.floating[Any]],
    stats: dict[str, float],
    is_digital: bool,
    digital_levels: dict[str, float] | None,
) -> list[str]:
    """Build initial characteristics list.

    Args:
        data: Signal data.
        stats: Signal statistics.
        is_digital: Whether signal is digital.
        digital_levels: Digital levels if applicable.

    Returns:
        List of characteristic strings.
    """
    characteristics = []
    if is_digital:
        characteristics.append("digital_levels")

    noise_level = _estimate_noise_level(data)
    _add_noise_characteristics(characteristics, noise_level, stats["amplitude"])
    return characteristics


def _analyze_periodicity(
    data: NDArray[np.floating[Any]],
    trace_sample_rate: float,
    periodicity_threshold: float,
    is_digital: bool,
    digital_levels: dict[str, float] | None,
    n: int,
    characteristics: list[str],
) -> tuple[bool, float | None, float]:
    """Analyze signal periodicity.

    Args:
        data: Signal data.
        trace_sample_rate: Sample rate.
        periodicity_threshold: Detection threshold.
        is_digital: Whether signal is digital.
        digital_levels: Digital levels if applicable.
        n: Data length.
        characteristics: Characteristics list to update.

    Returns:
        Tuple of (is_periodic, frequency_estimate, periodicity_score).
    """
    is_periodic, period_estimate, periodicity_score = _classify_periodicity(
        data, trace_sample_rate, periodicity_threshold, is_digital, digital_levels, n
    )

    if is_periodic:
        characteristics.append("periodic")
        frequency_estimate = (
            1.0 / period_estimate if period_estimate and period_estimate > 0 else None
        )
    else:
        characteristics.append("aperiodic")
        frequency_estimate = None

    return is_periodic, frequency_estimate, periodicity_score


def assess_signal_quality(
    trace: WaveformTrace,
) -> dict[str, Any]:
    """Assess signal quality metrics.

    Analyzes signal quality including SNR, noise level, clipping, saturation,
    and other quality indicators that affect measurement accuracy.

    Args:
        trace: Input waveform trace to assess.

    Returns:
        Dictionary containing:
        - snr: Signal-to-noise ratio in dB (or None if not applicable)
        - noise_level: RMS noise level in signal units
        - clipping: True if signal shows clipping
        - saturation: True if signal appears saturated
        - warnings: List of quality warning strings
        - dynamic_range: Signal dynamic range in dB
        - crest_factor: Peak-to-RMS ratio

    Example:
        >>> trace = osc.load('noisy_sine.wfm')
        >>> quality = osc.assess_signal_quality(trace)
        >>> print(f"SNR: {quality['snr']:.1f} dB")
        SNR: 42.3 dB
        >>> if quality['warnings']:
        ...     print(f"Warnings: {quality['warnings']}")

    References:
        IEEE 1057-2017: ADC quality metrics
    """
    data = trace.data
    n = len(data)
    warnings: list[str] = []

    if n < 10:
        warnings.append("Insufficient data for quality assessment")
        return _create_empty_quality_result(warnings)

    stats = _calculate_basic_stats(data)
    clipping, clipping_warnings = _detect_clipping(data, n, stats)
    warnings.extend(clipping_warnings)

    saturation, saturation_warning = _detect_saturation(data, trace, n)
    if saturation_warning:
        warnings.append(saturation_warning)

    noise_level = _estimate_noise_level(data)
    snr = _calculate_snr(stats, noise_level)
    dynamic_range = _calculate_dynamic_range(stats)
    crest_factor = _calculate_crest_factor(stats)

    quantization_warning = _check_quantization(data, n, stats)
    if quantization_warning:
        warnings.append(quantization_warning)

    sample_rate_warnings = _check_quality_sample_rate(trace)
    warnings.extend(sample_rate_warnings)

    return {
        "snr": float(snr) if snr is not None else None,
        "noise_level": float(noise_level),
        "clipping": bool(clipping),
        "saturation": bool(saturation),
        "warnings": warnings,
        "dynamic_range": float(dynamic_range) if dynamic_range is not None else None,
        "crest_factor": float(crest_factor) if crest_factor is not None else None,
    }


def _create_empty_quality_result(warnings: list[str]) -> dict[str, Any]:
    """Create quality result dict for insufficient data case.

    Args:
        warnings: List of warning messages.

    Returns:
        Quality result dictionary with null values.
    """
    return {
        "snr": None,
        "noise_level": 0.0,
        "clipping": False,
        "saturation": False,
        "warnings": warnings,
        "dynamic_range": None,
        "crest_factor": None,
    }


def _calculate_basic_stats(data: NDArray[np.floating[Any]]) -> dict[str, float]:
    """Calculate basic signal statistics.

    Args:
        data: Signal data array.

    Returns:
        Dict with min, max, mean, rms, amplitude values.
    """
    return {
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "rms": float(np.sqrt(np.mean(data**2))),
        "amplitude": float(np.max(data) - np.min(data)),
    }


def _detect_clipping(
    data: NDArray[np.floating[Any]], n: int, stats: dict[str, float]
) -> tuple[bool, list[str]]:
    """Detect signal clipping at extremes.

    Args:
        data: Signal data array.
        n: Number of samples.
        stats: Basic statistics dict.

    Returns:
        Tuple of (clipping_detected, warning_messages).
    """
    warnings: list[str] = []
    amplitude = stats["amplitude"]

    if amplitude <= 1e-9:
        return False, warnings

    tolerance = amplitude * 0.01
    at_min = data <= (stats["min"] + tolerance)
    at_max = data >= (stats["max"] - tolerance)
    min_run_length = max(int(n * 0.15), 100)

    max_min_run, max_max_run = _find_max_consecutive_runs(at_min, at_max, n)

    clipping = False
    if max_min_run >= min_run_length:
        clipping = True
        warnings.append(f"Signal clipping detected at minimum ({max_min_run} consecutive samples)")
    if max_max_run >= min_run_length:
        clipping = True
        warnings.append(f"Signal clipping detected at maximum ({max_max_run} consecutive samples)")

    return clipping, warnings


def _find_max_consecutive_runs(
    at_min: NDArray[np.bool_], at_max: NDArray[np.bool_], n: int
) -> tuple[int, int]:
    """Find maximum consecutive run lengths at min and max extremes.

    Args:
        at_min: Boolean array indicating samples at minimum.
        at_max: Boolean array indicating samples at maximum.
        n: Number of samples.

    Returns:
        Tuple of (max_min_run, max_max_run).
    """
    max_min_run = 0
    max_max_run = 0
    current_min_run = 0
    current_max_run = 0

    for i in range(n):
        if at_min[i]:
            current_min_run += 1
            max_min_run = max(max_min_run, current_min_run)
        else:
            current_min_run = 0

        if at_max[i]:
            current_max_run += 1
            max_max_run = max(max_max_run, current_max_run)
        else:
            current_max_run = 0

    return max_min_run, max_max_run


def _detect_saturation(
    data: NDArray[np.floating[Any]], trace: WaveformTrace, n: int
) -> tuple[bool, str | None]:
    """Detect signal saturation (stuck at one level).

    Args:
        data: Signal data array.
        trace: Waveform trace for classification.
        n: Number of samples.

    Returns:
        Tuple of (saturation_detected, warning_message).
    """
    unique_values = len(np.unique(data))
    classification = classify_signal(trace)

    if classification["type"] == "digital":
        if unique_values < 2:
            return True, f"Signal saturation detected (only {unique_values} unique value)"
    else:
        if unique_values < max(10, n // 1000):
            return True, f"Signal saturation detected (only {unique_values} unique values)"

    return False, None


def _calculate_snr(stats: dict[str, float], noise_level: float) -> float | None:
    """Calculate signal-to-noise ratio.

    Args:
        stats: Basic statistics dict.
        noise_level: Estimated noise level.

    Returns:
        SNR in dB or None if not calculable.
    """
    amplitude = stats["amplitude"]

    if amplitude <= noise_level * 10:
        return None

    signal_power = amplitude**2 / 8
    noise_power = noise_level**2

    if noise_power > 1e-20:
        return float(10 * np.log10(signal_power / noise_power))

    return float("inf")


def _calculate_dynamic_range(stats: dict[str, float]) -> float | None:
    """Calculate signal dynamic range.

    Args:
        stats: Basic statistics dict.

    Returns:
        Dynamic range in dB or None.
    """
    min_val = stats["min"]
    max_val = stats["max"]

    if min_val != 0 and max_val != 0 and max_val > 1e-20:
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = max_val / (abs(min_val) + 1e-20)
            if ratio > 0 and np.isfinite(ratio):
                return float(20 * np.log10(ratio))

    return None


def _calculate_crest_factor(stats: dict[str, float]) -> float | None:
    """Calculate crest factor (peak-to-RMS ratio).

    Args:
        stats: Basic statistics dict.

    Returns:
        Crest factor or None.
    """
    rms_val = stats["rms"]
    if rms_val > 1e-12:
        return float(max(abs(stats["max"]), abs(stats["min"])) / rms_val)
    return None


def _check_quantization(
    data: NDArray[np.floating[Any]], n: int, stats: dict[str, float]
) -> str | None:
    """Check for quantization issues.

    Args:
        data: Signal data array.
        n: Number of samples.
        stats: Basic statistics dict.

    Returns:
        Warning message or None.
    """
    if n <= 100:
        return None

    sorted_data = np.sort(data)
    diffs = np.diff(sorted_data)
    diffs = diffs[diffs > 1e-15]

    if len(diffs) > 10:
        min_step = float(np.min(diffs))
        amplitude = stats["amplitude"]
        if amplitude / min_step < 256:
            return f"Low resolution detected ({int(amplitude / min_step)} levels), may affect measurement accuracy"

    return None


def _check_quality_sample_rate(trace: WaveformTrace) -> list[str]:
    """Check if sample rate is adequate for signal frequency in quality assessment.

    Args:
        trace: Waveform trace with metadata.

    Returns:
        List of warning messages.
    """
    warnings: list[str] = []
    classification = classify_signal(trace)

    if classification["frequency_estimate"] is None:
        return warnings

    freq = classification["frequency_estimate"]
    sample_rate = trace.metadata.sample_rate
    nyquist_rate = 2 * freq

    if sample_rate < nyquist_rate * 5:
        warnings.append(
            f"Sample rate ({sample_rate:.3e} Hz) may be insufficient for "
            f"signal frequency ({freq:.3e} Hz). Recommend at least 10x oversampling"
        )

    samples_per_period = sample_rate / freq
    if samples_per_period < 10 and "sample rate" not in "".join(warnings).lower():
        warnings.append(
            f"Very low oversampling detected ({samples_per_period:.1f} samples per period). "
            "Signal may be undersampled or frequency detection may be inaccurate. "
            "Recommend at least 10 samples per period"
        )

    return warnings


def _get_measurement_categories() -> dict[str, list[str]]:
    """Get categorized list of measurement types.

    Returns:
        Dictionary mapping category names to measurement lists.
    """
    return {
        "frequency": ["frequency", "period"],
        "edge": ["rise_time", "fall_time"],
        "amplitude": ["amplitude", "overshoot", "undershoot", "preshoot"],
        "duty": ["duty_cycle", "pulse_width"],
        "statistical": ["mean", "rms"],
        "spectral": ["thd", "snr", "sinad", "enob", "sfdr", "fft", "psd"],
    }


def _check_dc_signal_compatibility(
    signal_type: str,
    measurement_name: str,
    categories: dict[str, list[str]],
    state: dict[str, Any],
) -> None:
    """Check if measurement is compatible with DC signals.

    Args:
        signal_type: Type of signal (e.g., "dc", "digital").
        measurement_name: Name of the measurement to check.
        categories: Dict mapping category names to measurement lists.
        state: Mutable dict with suitable, warnings, suggestions, expected_result.
    """
    if signal_type != "dc":
        return

    if measurement_name in categories["frequency"]:
        state["suitable"] = False
        state["warnings"].append(f"{measurement_name} measurement not suitable for DC signal")
        state["suggestions"].append("Use 'mean' or 'rms' measurements for DC signals")
        state["expected_result"] = "nan"
    elif measurement_name in categories["edge"]:
        state["suitable"] = False
        state["warnings"].append(f"{measurement_name} requires signal transitions")
        state["suggestions"].append("Signal appears to be DC with no edges")
        state["expected_result"] = "nan"
    elif measurement_name in categories["duty"]:
        state["suitable"] = False
        state["warnings"].append(f"{measurement_name} requires periodic signal")
        state["expected_result"] = "nan"


def _check_aperiodic_signal_compatibility(
    characteristics: list[str],
    measurement_name: str,
    categories: dict[str, list[str]],
    state: dict[str, Any],
) -> None:
    """Check if measurement is compatible with aperiodic signals.

    Args:
        characteristics: List of signal characteristics.
        measurement_name: Name of the measurement to check.
        categories: Dict mapping category names to measurement lists.
        state: Mutable dict with suitable, warnings, suggestions, expected_result, confidence.
    """
    if "aperiodic" not in characteristics:
        return

    periodic_measurements = categories["frequency"] + categories["duty"]
    if measurement_name in periodic_measurements:
        state["suitable"] = False
        state["confidence"] = 0.6
        state["warnings"].append(f"{measurement_name} requires periodic signal")
        state["suggestions"].append("Signal does not appear periodic")
        state["expected_result"] = "nan"
    elif measurement_name in categories["spectral"]:
        state["warnings"].append(
            "Spectral measurements on aperiodic signals may not show clear peaks"
        )
        state["suggestions"].append("Consider time-domain or statistical analysis")
        state["expected_result"] = "unreliable"


def _check_digital_signal_compatibility(
    signal_type: str,
    measurement_name: str,
    categories: dict[str, list[str]],
    state: dict[str, Any],
) -> None:
    """Check if measurement is compatible with digital signals.

    Args:
        signal_type: Type of signal.
        measurement_name: Name of the measurement to check.
        categories: Dict mapping category names to measurement lists.
        state: Mutable dict with warnings, suggestions, expected_result, confidence.
    """
    if signal_type != "digital":
        return

    if measurement_name in categories["amplitude"] and measurement_name != "amplitude":
        state["warnings"].append(
            f"{measurement_name} designed for analog signals with overshoot/ringing"
        )
        state["suggestions"].append("Digital signals may show zero overshoot/undershoot")
        state["expected_result"] = "unreliable"
        state["confidence"] = 0.5


def _check_edge_count_requirements(
    trace: WaveformTrace,
    measurement_name: str,
    categories: dict[str, list[str]],
    classification: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Check if signal has sufficient edges for edge-based measurements.

    Args:
        trace: Input waveform trace.
        measurement_name: Name of the measurement to check.
        categories: Dict mapping category names to measurement lists.
        classification: Signal classification info.
        state: Mutable dict with suitable, warnings, suggestions, expected_result.
    """
    edge_based = categories["edge"] + categories["duty"]
    if measurement_name not in edge_based:
        return

    edge_count = _count_edges(trace.data, classification.get("levels"))
    if edge_count < 2:
        state["suitable"] = False
        state["warnings"].append(f"{measurement_name} requires at least 2 signal edges")
        state["suggestions"].append(f"Signal has only {edge_count} detected edge(s)")
        state["expected_result"] = "nan"


def _check_quality_impacts(
    quality: dict[str, Any],
    measurement_name: str,
    categories: dict[str, list[str]],
    state: dict[str, Any],
) -> None:
    """Check how signal quality issues affect measurement suitability.

    Args:
        quality: Signal quality assessment.
        measurement_name: Name of the measurement to check.
        categories: Dict mapping category names to measurement lists.
        state: Mutable dict with warnings, expected_result, confidence.
    """
    affected_by_clipping = categories["edge"] + categories["amplitude"]

    if quality["clipping"] and measurement_name in affected_by_clipping:
        state["warnings"].append("Signal clipping detected, may affect measurement accuracy")
        if state["expected_result"] != "nan":
            state["expected_result"] = "unreliable"
        state["confidence"] = min(state["confidence"], 0.6)

    if quality["saturation"]:
        state["warnings"].append("Signal saturation detected, measurements may be unreliable")
        if state["expected_result"] != "nan":
            state["expected_result"] = "unreliable"
        state["confidence"] = min(state["confidence"], 0.5)

    if quality["snr"] is not None and quality["snr"] < 20:
        if measurement_name in categories["edge"]:
            state["warnings"].append(
                f"Low SNR ({quality['snr']:.1f} dB) may affect edge timing measurements"
            )
            state["suggestions"].append("Consider filtering signal to improve SNR")
            state["confidence"] = min(state["confidence"], 0.7)


def _check_sample_rate_adequacy(
    trace: WaveformTrace,
    measurement_name: str,
    categories: dict[str, list[str]],
    classification: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Check if sample rate is adequate for timing measurements.

    Args:
        trace: Input waveform trace.
        measurement_name: Name of the measurement to check.
        categories: Dict mapping category names to measurement lists.
        classification: Signal classification info.
        state: Mutable dict with warnings, suggestions, expected_result, confidence.
    """
    timing_measurements = categories["edge"] + categories["frequency"]
    if measurement_name not in timing_measurements:
        return

    if classification["frequency_estimate"] is None:
        return

    nyquist_rate = 2 * classification["frequency_estimate"]
    if trace.metadata.sample_rate < nyquist_rate * 5:
        state["warnings"].append("Sample rate may be too low for accurate timing measurements")
        state["suggestions"].append(
            f"Recommend sample rate > {nyquist_rate * 10:.3e} Hz (10x signal frequency)"
        )
        state["expected_result"] = "unreliable"
        state["confidence"] = min(state["confidence"], 0.6)


def _check_data_length_adequacy(
    trace: WaveformTrace,
    measurement_name: str,
    categories: dict[str, list[str]],
    classification: dict[str, Any],
    state: dict[str, Any],
) -> None:
    """Check if signal length is adequate for the measurement.

    Args:
        trace: Input waveform trace.
        measurement_name: Name of the measurement to check.
        categories: Dict mapping category names to measurement lists.
        classification: Signal classification info.
        state: Mutable dict with warnings, suggestions, expected_result, confidence.
    """
    n = len(trace.data)

    # Check spectral measurements
    if measurement_name in categories["spectral"]:
        if n < 256:
            state["warnings"].append(
                f"Signal length ({n} samples) may be too short for spectral analysis"
            )
            state["suggestions"].append(
                "Recommend at least 1024 samples for FFT-based measurements"
            )
            state["expected_result"] = "unreliable"
            state["confidence"] = min(state["confidence"], 0.5)

    # Check frequency measurements
    if measurement_name in categories["frequency"]:
        if classification["frequency_estimate"] is not None:
            min_samples = trace.metadata.sample_rate / classification["frequency_estimate"]
            if n < min_samples * 0.5:
                state["warnings"].append(
                    f"Signal length ({n} samples) captures < 0.5 periods, "
                    "frequency measurement may fail"
                )
                state["suggestions"].append(
                    "Capture at least 2 periods for reliable frequency measurement"
                )
                state["expected_result"] = "unreliable"
                state["confidence"] = min(state["confidence"], 0.5)
            elif n < min_samples * 2:
                state["suggestions"].append("Capture at least 10 periods for best accuracy")
                state["confidence"] = min(state["confidence"], 0.75)


def check_measurement_suitability(
    trace: WaveformTrace,
    measurement_name: str,
) -> dict[str, Any]:
    """Check if a measurement is suitable for this signal.

    Analyzes signal characteristics to determine if a specific measurement
    will produce valid results, and provides warnings and suggestions.

    Args:
        trace: Input waveform trace.
        measurement_name: Name of measurement to check (e.g., "frequency", "rise_time").

    Returns:
        Dictionary containing:
        - suitable: True if measurement is appropriate for this signal
        - confidence: Confidence in suitability assessment (0.0-1.0)
        - warnings: List of warning strings
        - suggestions: List of suggestion strings
        - expected_result: "valid", "nan", or "unreliable"

    Example:
        >>> trace = osc.load('dc_signal.wfm')
        >>> check = osc.check_measurement_suitability(trace, "frequency")
        >>> if not check['suitable']:
        ...     print(f"Warning: {check['warnings']}")
        Warning: ['Frequency measurement not suitable for DC signal']

    References:
        IEEE 181-2011: Measurement applicability
    """
    classification = classify_signal(trace)
    quality = assess_signal_quality(trace)

    state: dict[str, Any] = {
        "suitable": True,
        "confidence": 0.8,
        "expected_result": "valid",
        "warnings": [],
        "suggestions": [],
    }

    categories = _get_measurement_categories()
    signal_type = classification["type"]
    characteristics = classification["characteristics"]

    # Run all compatibility checks
    _check_dc_signal_compatibility(signal_type, measurement_name, categories, state)
    _check_aperiodic_signal_compatibility(characteristics, measurement_name, categories, state)
    _check_digital_signal_compatibility(signal_type, measurement_name, categories, state)
    _check_edge_count_requirements(trace, measurement_name, categories, classification, state)
    _check_quality_impacts(quality, measurement_name, categories, state)
    _check_sample_rate_adequacy(trace, measurement_name, categories, classification, state)
    _check_data_length_adequacy(trace, measurement_name, categories, classification, state)

    # Extract confidence (guaranteed to be float from initialization)
    confidence_value = float(state["confidence"])
    return {
        "suitable": state["suitable"],
        "confidence": confidence_value,
        "warnings": state["warnings"],
        "suggestions": state["suggestions"],
        "expected_result": state["expected_result"],
    }


def suggest_measurements(
    trace: WaveformTrace,
    *,
    max_suggestions: int = 10,
) -> list[dict[str, Any]]:
    """Suggest appropriate measurements for a signal.

    Analyzes signal characteristics and recommends the most suitable
    measurements, ranked by relevance and reliability.

    Args:
        trace: Input waveform trace.
        max_suggestions: Maximum number of suggestions to return.

    Returns:
        List of dictionaries, each containing:
        - name: Measurement name
        - category: Measurement category (e.g., "timing", "amplitude", "spectral")
        - priority: Priority ranking (1=highest)
        - rationale: Why this measurement is recommended
        - confidence: Confidence in recommendation (0.0-1.0)

    Example:
        >>> trace = osc.load('square_wave.wfm')
        >>> suggestions = osc.suggest_measurements(trace)
        >>> for s in suggestions[:3]:
        ...     print(f"{s['name']}: {s['rationale']}")
        frequency: Periodic digital signal detected
        duty_cycle: Suitable for pulse analysis
        rise_time: Digital edges detected

    References:
        Best practices for waveform analysis
    """
    classification = classify_signal(trace)
    quality = assess_signal_quality(trace)

    signal_type = classification["type"]
    characteristics = classification["characteristics"]

    suggestions: list[dict[str, Any]] = []

    # Core statistical measurements (always applicable)
    _add_statistical_suggestions(suggestions)

    # Early return for DC signals
    if signal_type == "dc":
        _add_dc_signal_suggestion(suggestions)
        return sorted(suggestions, key=lambda x: cast("int", x["priority"]))[:max_suggestions]

    # Add suggestions based on signal characteristics
    _add_amplitude_suggestion(suggestions, signal_type)

    if "periodic" in characteristics:
        _add_periodic_suggestions(suggestions, classification)

    if signal_type in ("digital", "mixed"):
        _add_digital_signal_suggestions(
            suggestions, trace, classification, quality, characteristics
        )

    if signal_type in ("analog", "mixed"):
        _add_analog_signal_suggestions(suggestions, quality)

    if "periodic" in characteristics and "clean" in characteristics:
        _add_spectral_suggestions(suggestions, trace)

    # Sort by priority and limit
    suggestions = sorted(suggestions, key=lambda x: cast("int", x["priority"]))
    return suggestions[:max_suggestions]


def _add_statistical_suggestions(suggestions: list[dict[str, Any]]) -> None:
    """Add core statistical measurement suggestions."""
    suggestions.append(
        {
            "name": "mean",
            "category": "statistical",
            "priority": 1,
            "rationale": "Basic DC level measurement, always applicable",
            "confidence": 1.0,
        }
    )

    suggestions.append(
        {
            "name": "rms",
            "category": "statistical",
            "priority": 2,
            "rationale": "RMS voltage measurement, useful for all signal types",
            "confidence": 1.0,
        }
    )


def _add_dc_signal_suggestion(suggestions: list[dict[str, Any]]) -> None:
    """Add suggestion for DC signal noise measurement."""
    suggestions.append(
        {
            "name": "amplitude",
            "category": "amplitude",
            "priority": 3,
            "rationale": "Measure noise/variation level in DC signal",
            "confidence": 0.9,
        }
    )


def _add_amplitude_suggestion(suggestions: list[dict[str, Any]], signal_type: str) -> None:
    """Add general amplitude measurement suggestion."""
    suggestions.append(
        {
            "name": "amplitude",
            "category": "amplitude",
            "priority": 3,
            "rationale": f"Peak-to-peak amplitude for {signal_type} signal",
            "confidence": 0.95,
        }
    )


def _add_periodic_suggestions(
    suggestions: list[dict[str, Any]], classification: dict[str, Any]
) -> None:
    """Add frequency/period suggestions for periodic signals."""
    suggestions.append(
        {
            "name": "frequency",
            "category": "timing",
            "priority": 4,
            "rationale": "Periodic signal detected, frequency measurement applicable",
            "confidence": classification["confidence"],
        }
    )

    suggestions.append(
        {
            "name": "period",
            "category": "timing",
            "priority": 5,
            "rationale": "Period measurement for periodic signal",
            "confidence": classification["confidence"],
        }
    )


def _add_digital_signal_suggestions(
    suggestions: list[dict[str, Any]],
    trace: WaveformTrace,
    classification: dict[str, Any],
    quality: dict[str, Any],
    characteristics: list[str],
) -> None:
    """Add edge timing and pulse measurement suggestions for digital signals."""
    edge_count = _count_edges(trace.data, classification.get("levels"))

    if edge_count >= 2:
        snr_conf = 0.9 if quality["snr"] and quality["snr"] > 20 else 0.7

        suggestions.append(
            {
                "name": "rise_time",
                "category": "timing",
                "priority": 6,
                "rationale": f"Digital edges detected ({edge_count} edges)",
                "confidence": snr_conf,
            }
        )

        suggestions.append(
            {
                "name": "fall_time",
                "category": "timing",
                "priority": 7,
                "rationale": f"Digital edges detected ({edge_count} edges)",
                "confidence": snr_conf,
            }
        )

    if "periodic" in characteristics and edge_count >= 2:
        duty_conf = 0.85 if edge_count >= 4 else 0.75

        suggestions.append(
            {
                "name": "duty_cycle",
                "category": "timing",
                "priority": 8,
                "rationale": "Periodic pulse train detected",
                "confidence": duty_conf,
            }
        )

        suggestions.append(
            {
                "name": "pulse_width",
                "category": "timing",
                "priority": 9,
                "rationale": "Pulse measurements suitable for periodic digital signal",
                "confidence": duty_conf,
            }
        )


def _add_analog_signal_suggestions(
    suggestions: list[dict[str, Any]], quality: dict[str, Any]
) -> None:
    """Add overshoot/undershoot suggestions for analog signals."""
    if not quality["clipping"]:
        suggestions.append(
            {
                "name": "overshoot",
                "category": "amplitude",
                "priority": 10,
                "rationale": "Analog signal, overshoot measurement applicable",
                "confidence": 0.8,
            }
        )

        suggestions.append(
            {
                "name": "undershoot",
                "category": "amplitude",
                "priority": 11,
                "rationale": "Analog signal, undershoot measurement applicable",
                "confidence": 0.8,
            }
        )


def _add_spectral_suggestions(suggestions: list[dict[str, Any]], trace: WaveformTrace) -> None:
    """Add spectral analysis suggestions for clean periodic signals."""
    if len(trace.data) >= 256:
        suggestions.append(
            {
                "name": "thd",
                "category": "spectral",
                "priority": 12,
                "rationale": "Clean periodic signal suitable for harmonic analysis",
                "confidence": 0.85,
            }
        )

        suggestions.append(
            {
                "name": "snr",
                "category": "spectral",
                "priority": 13,
                "rationale": "Spectral SNR measurement for signal quality",
                "confidence": 0.8,
            }
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _detect_digital_signal(
    data: NDArray[np.floating[Any]],
    threshold_ratio: float,
) -> tuple[bool, dict[str, float] | None, float]:
    """Detect if signal is digital based on bimodal distribution.

    Args:
        data: Signal data array.
        threshold_ratio: Ratio of samples at two levels to consider digital.

    Returns:
        Tuple of (is_digital, levels_dict, confidence).
    """
    # Use histogram to find peaks
    # Use more bins for better resolution on digital signals
    n_bins = min(100, len(np.unique(data)))
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks (local maxima or significant bins)
    peaks = []

    # Special case: if only 2 bins (perfect digital signal), both are peaks
    if len(hist) == 2:
        for i in range(len(hist)):
            if hist[i] > len(data) * 0.01:
                peaks.append((i, hist[i], bin_centers[i]))
    else:
        # Find local maxima in histogram
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                # Lower threshold for peak detection
                if hist[i] > len(data) * 0.01:  # At least 1% of samples
                    peaks.append((i, hist[i], bin_centers[i]))

    # If we have exactly 2 dominant peaks, likely digital
    if len(peaks) >= 2:
        # Sort by count
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)

        # Take top 2 peaks
        peak1, peak2 = peaks[0], peaks[1]

        # Check if these two peaks account for most samples
        total_in_peaks = peak1[1] + peak2[1]
        ratio = total_in_peaks / len(data)

        # Also check that peaks are well separated
        peak_separation = abs(peak1[2] - peak2[2])
        data_range = np.ptp(data)

        # Peaks should be separated by at least 30% of data range
        if ratio >= threshold_ratio and peak_separation > data_range * 0.3:
            low_level = min(peak1[2], peak2[2])
            high_level = max(peak1[2], peak2[2])

            confidence = min(0.95, ratio)

            return True, {"low": float(low_level), "high": float(high_level)}, confidence

    return False, None, 0.0


def _estimate_noise_level(data: NDArray[np.floating[Any]]) -> float:
    """Estimate noise level using median absolute deviation.

    Args:
        data: Signal data array.

    Returns:
        Estimated RMS noise level.
    """
    if len(data) < 10:
        return 0.0

    # Use differencing to remove slow variations
    diffs = np.diff(data)

    # MAD (Median Absolute Deviation) is robust to outliers
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff))

    # Convert MAD to RMS noise estimate
    # For Gaussian noise: sigma ≈ 1.4826 * MAD
    # Divide by sqrt(2) because diff amplifies noise by sqrt(2)
    noise_estimate = (1.4826 * mad) / np.sqrt(2)

    return float(noise_estimate)


def _detect_periodicity(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    threshold: float,
) -> tuple[bool, float | None, float]:
    """Detect if signal is periodic using autocorrelation.

    Args:
        data: Signal data array.
        sample_rate: Sampling rate in Hz.
        threshold: Correlation threshold for periodic detection.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    n = len(data)

    if n < 20:
        return False, None, 0.0

    # Remove DC for autocorrelation
    data_ac = data - np.mean(data)

    # Check if there's any variation
    if np.std(data_ac) < 1e-12:
        return False, None, 0.0

    # Compute autocorrelation for lags up to n-10 to detect signals with ~1 period
    # This allows finding periodicity even when we have just 1 period of data
    # Keep at least 10 samples of overlap for correlation
    max_lag = min(n - 10, 20000)  # Limit for performance

    autocorr = np.correlate(data_ac, data_ac, mode="full")
    autocorr = autocorr[n - 1 : n - 1 + max_lag]

    # Normalize
    if abs(autocorr[0]) > 1e-12:
        autocorr = autocorr / autocorr[0]
    else:
        return False, None, 0.0

    # Find peaks in autocorrelation (exclude lag=0 and very small lags)
    # Start searching from lag > n/100 to avoid noise
    min_lag = max(3, n // 100)
    peaks = []

    for i in range(min_lag, len(autocorr) - 2):
        # Use stronger peak detection
        if (
            autocorr[i] > autocorr[i - 1]
            and autocorr[i] > autocorr[i + 1]
            and autocorr[i] > autocorr[i - 2]
            and autocorr[i] > autocorr[i + 2]
        ):
            if autocorr[i] > threshold:
                peaks.append((i, autocorr[i]))

    if peaks:
        # Take first significant peak as period
        period_samples = peaks[0][0]
        confidence = float(peaks[0][1])

        period_seconds = period_samples / sample_rate

        return True, period_seconds, confidence

    return False, None, 0.0


def _count_edges(
    data: NDArray[np.floating[Any]],
    levels: dict[str, float] | None,
) -> int:
    """Count number of edges in signal.

    Args:
        data: Signal data array.
        levels: Optional digital levels dict with "low" and "high" keys.

    Returns:
        Number of edges detected.
    """
    if len(data) < 3:
        return 0

    if levels is not None:
        # Use provided levels
        threshold = (levels["low"] + levels["high"]) / 2
    else:
        # Use median as threshold
        threshold = float(np.median(data))

    # Find crossings
    above = data > threshold
    crossings = np.diff(above.astype(int))

    # Count non-zero crossings (both rising and falling)
    edge_count = np.sum(np.abs(crossings))

    return int(edge_count)


def _detect_periodicity_fft(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
) -> tuple[bool, float | None, float]:
    """Detect periodicity using FFT (frequency domain analysis).

    This method works well for signals with few periods where autocorrelation
    may fail. It finds the dominant frequency component in the signal.

    Args:
        data: Signal data array.
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    n = len(data)

    if n < 64:
        return False, None, 0.0

    # Remove DC component
    data_ac = data - np.mean(data)

    # Check if there's any variation
    if np.std(data_ac) < 1e-12:
        return False, None, 0.0

    # Compute FFT
    fft = np.fft.rfft(data_ac)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)

    # Compute power spectrum
    power = np.abs(fft) ** 2

    # Skip DC component (index 0)
    if len(power) < 3:
        return False, None, 0.0

    power = power[1:]
    freqs = freqs[1:]

    # Find peak in power spectrum
    peak_idx = np.argmax(power)
    peak_power = power[peak_idx]
    peak_freq = freqs[peak_idx]

    # Check if peak is significant compared to total power
    total_power = np.sum(power)
    if total_power < 1e-20:
        return False, None, 0.0

    power_ratio = peak_power / total_power

    # For periodic signals, the dominant frequency should have significant power
    # Require at least 10% of total power in the peak
    if power_ratio < 0.1:
        return False, None, 0.0

    # Check that frequency is reasonable (not too low or too high)
    nyquist = sample_rate / 2
    if peak_freq < sample_rate / n or peak_freq > nyquist * 0.9:
        return False, None, 0.0

    # Estimate period
    period_seconds = 1.0 / peak_freq

    # Confidence based on how dominant the peak is
    # High power ratio -> high confidence
    confidence = min(0.95, 0.5 + power_ratio)

    return True, period_seconds, float(confidence)


def _detect_edge_periodicity(
    data: NDArray[np.floating[Any]],
    sample_rate: float,
    levels: dict[str, float] | None,
) -> tuple[bool, float | None, float]:
    """Detect periodicity in digital signals by analyzing edge spacing.

    This method works well for signals with few periods where autocorrelation
    may fail. It detects regular patterns in edge timing.

    Args:
        data: Signal data array.
        sample_rate: Sampling rate in Hz.
        levels: Digital levels dict with "low" and "high" keys.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    if len(data) < 10 or levels is None:
        return False, None, 0.0

    intervals = _extract_edge_intervals(data, levels)
    if intervals is None or len(intervals) < 1:
        return False, None, 0.0

    mean_interval_raw = np.mean(intervals)
    mean_interval: float = float(mean_interval_raw)
    if mean_interval < 1:
        return False, None, 0.0

    return _analyze_interval_pattern(intervals, mean_interval, sample_rate, len(data))


def _extract_edge_intervals(
    data: NDArray[np.floating[Any]], levels: dict[str, float]
) -> NDArray[np.intp] | None:
    """Extract intervals between edges.

    Args:
        data: Signal data array.
        levels: Digital levels dict.

    Returns:
        Array of edge intervals or None if insufficient edges.
    """
    threshold = (levels["low"] + levels["high"]) / 2
    above = data > threshold
    crossings = np.diff(above.astype(int))
    edge_positions = np.where(crossings != 0)[0]

    if len(edge_positions) < 2:
        return None

    return np.diff(edge_positions)


def _analyze_interval_pattern(
    intervals: NDArray[np.intp], mean_interval: float, sample_rate: float, n_samples: int
) -> tuple[bool, float | None, float]:
    """Analyze interval pattern to detect periodicity.

    Args:
        intervals: Edge intervals array.
        mean_interval: Mean interval value.
        sample_rate: Sampling rate in Hz.
        n_samples: Total number of samples.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    std_interval = np.std(intervals)
    cv = std_interval / mean_interval

    # Special case: single interval
    if len(intervals) == 1:
        period_samples = 2 * intervals[0]
        period_seconds = period_samples / sample_rate
        return True, period_seconds, 0.7

    # High variation - check for alternating pattern
    if cv > 0.3:
        return _check_alternating_pattern(intervals, sample_rate)

    # Regular intervals - estimate period
    cv_float: float = float(cv)
    return _estimate_regular_period(mean_interval, cv_float, sample_rate, n_samples)


def _check_alternating_pattern(
    intervals: NDArray[np.intp], sample_rate: float
) -> tuple[bool, float | None, float]:
    """Check if intervals follow alternating pattern (square wave).

    Args:
        intervals: Edge intervals array.
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    if len(intervals) >= 4:
        odd_intervals = intervals[::2]
        even_intervals = intervals[1::2]

        odd_cv = np.std(odd_intervals) / (np.mean(odd_intervals) + 1e-12)
        even_cv = np.std(even_intervals) / (np.mean(even_intervals) + 1e-12)

        if odd_cv < 0.2 and even_cv < 0.2:
            period_samples = np.mean(odd_intervals) + np.mean(even_intervals)
            period_seconds = period_samples / sample_rate
            confidence = 1.0 - max(odd_cv, even_cv)
            return True, period_seconds, float(confidence)

    elif len(intervals) == 2:
        period_samples = intervals[0] + intervals[1]
        period_seconds = period_samples / sample_rate
        return True, period_seconds, 0.75

    return False, None, 0.0


def _estimate_regular_period(
    mean_interval: float, cv: float, sample_rate: float, n_samples: int
) -> tuple[bool, float | None, float]:
    """Estimate period from regular intervals.

    Args:
        mean_interval: Mean interval between edges.
        cv: Coefficient of variation.
        sample_rate: Sampling rate in Hz.
        n_samples: Total number of samples.

    Returns:
        Tuple of (is_periodic, period_seconds, confidence).
    """
    period_samples = 2 * mean_interval
    num_periods = n_samples / period_samples

    if num_periods >= 0.5:
        period_seconds = period_samples / sample_rate
        confidence = 1.0 - min(cv / 0.3, 0.5)
        return True, period_seconds, float(confidence)

    return False, None, 0.0


@dataclass
class AnalysisRecommendation:
    """Recommendation for an analysis to run.

    Attributes:
        domain: Analysis domain to run.
        priority: Priority ranking (1=highest).
        confidence: Expected confidence if run (0.0-1.0).
        reasoning: Human-readable explanation.
        estimated_runtime_ms: Estimated runtime in milliseconds.
        prerequisites_met: Whether all prerequisites are satisfied.
    """

    domain: AnalysisDomain
    priority: int  # 1=highest priority
    confidence: float  # Expected confidence if run
    reasoning: str
    estimated_runtime_ms: int = 100
    prerequisites_met: bool = True


def _add_foundational_recommendations(
    recommendations: list[AnalysisRecommendation],
    exclude: set[AnalysisDomain],
) -> None:
    """Add foundational analysis recommendations (waveform, statistics).

    Args:
        recommendations: List to append recommendations to
        exclude: Domains to exclude
    """
    from oscura.reporting.config import AnalysisDomain

    if AnalysisDomain.WAVEFORM not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.WAVEFORM,
                priority=1,
                confidence=0.95,
                reasoning="Basic waveform measurements are always applicable",
                estimated_runtime_ms=50,
            )
        )

    if AnalysisDomain.STATISTICS not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.STATISTICS,
                priority=1,
                confidence=0.95,
                reasoning="Statistical analysis provides foundational metrics",
                estimated_runtime_ms=30,
            )
        )


def _add_spectral_recommendation(
    recommendations: list[AnalysisRecommendation],
    exclude: set[AnalysisDomain],
    is_periodic: bool,
) -> None:
    """Add spectral analysis recommendation.

    Args:
        recommendations: List to append recommendations to
        exclude: Domains to exclude
        is_periodic: Whether signal is periodic
    """
    from oscura.reporting.config import AnalysisDomain

    if AnalysisDomain.SPECTRAL not in exclude:
        spectral_conf = 0.85 if is_periodic else 0.70
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.SPECTRAL,
                priority=2 if is_periodic else 3,
                confidence=spectral_conf,
                reasoning="Spectral analysis reveals frequency content"
                + (" - signal appears periodic" if is_periodic else ""),
                estimated_runtime_ms=100,
            )
        )


def _add_digital_recommendations(
    recommendations: list[AnalysisRecommendation],
    exclude: set[AnalysisDomain],
    dominant_freq: float | None,
) -> None:
    """Add digital signal analysis recommendations.

    Args:
        recommendations: List to append recommendations to
        exclude: Domains to exclude
        dominant_freq: Dominant frequency in Hz
    """
    from oscura.reporting.config import AnalysisDomain

    if AnalysisDomain.DIGITAL not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.DIGITAL,
                priority=1,
                confidence=0.90,
                reasoning="Digital signal detected - edge and timing analysis recommended",
                estimated_runtime_ms=80,
            )
        )

    if AnalysisDomain.TIMING not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.TIMING,
                priority=2,
                confidence=0.85,
                reasoning="Timing analysis valuable for digital signals",
                estimated_runtime_ms=60,
            )
        )

    if AnalysisDomain.PROTOCOLS not in exclude and dominant_freq:
        # Check if frequency matches common baud rates
        common_bauds = [9600, 19200, 38400, 57600, 115200]
        if any(abs(dominant_freq * 2 - b) / b < 0.1 for b in common_bauds):
            recommendations.append(
                AnalysisRecommendation(
                    domain=AnalysisDomain.PROTOCOLS,
                    priority=3,
                    confidence=0.70,
                    reasoning=f"Frequency {dominant_freq:.0f} Hz suggests serial protocol",
                    estimated_runtime_ms=150,
                )
            )


def _add_periodic_recommendations(
    recommendations: list[AnalysisRecommendation],
    exclude: set[AnalysisDomain],
    is_digital: bool,
) -> None:
    """Add periodic signal analysis recommendations.

    Args:
        recommendations: List to append recommendations to
        exclude: Domains to exclude
        is_digital: Whether signal is digital
    """
    from oscura.reporting.config import AnalysisDomain

    if AnalysisDomain.JITTER not in exclude and is_digital:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.JITTER,
                priority=3,
                confidence=0.80,
                reasoning="Periodic digital signal - jitter analysis applicable",
                estimated_runtime_ms=120,
            )
        )

    if AnalysisDomain.EYE not in exclude and is_digital:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.EYE,
                priority=3,
                confidence=0.75,
                reasoning="Eye diagram analysis for signal integrity assessment",
                estimated_runtime_ms=200,
            )
        )


def _add_pattern_and_entropy_recommendations(
    recommendations: list[AnalysisRecommendation],
    exclude: set[AnalysisDomain],
    data_length: int,
    is_periodic: bool,
) -> None:
    """Add pattern and entropy analysis recommendations.

    Args:
        recommendations: List to append recommendations to
        exclude: Domains to exclude
        data_length: Length of signal data
        is_periodic: Whether signal is periodic
    """
    from oscura.reporting.config import AnalysisDomain

    # Pattern analysis - good for complex signals
    if AnalysisDomain.PATTERNS not in exclude and data_length > 1000:
        pattern_conf = 0.70 if is_periodic else 0.50
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.PATTERNS,
                priority=4,
                confidence=pattern_conf,
                reasoning="Pattern analysis can reveal repeating structures",
                estimated_runtime_ms=500,
            )
        )

    # Entropy analysis - useful for random/encrypted data
    if AnalysisDomain.ENTROPY not in exclude:
        recommendations.append(
            AnalysisRecommendation(
                domain=AnalysisDomain.ENTROPY,
                priority=5,
                confidence=0.80,
                reasoning="Entropy analysis characterizes randomness and complexity",
                estimated_runtime_ms=100,
            )
        )


def _filter_by_confidence(
    recommendations: list[AnalysisRecommendation],
    confidence_target: float,
) -> list[AnalysisRecommendation]:
    """Filter recommendations by confidence threshold.

    Args:
        recommendations: List of recommendations
        confidence_target: Minimum confidence threshold

    Returns:
        Filtered recommendations
    """
    return [r for r in recommendations if r.confidence >= confidence_target]


def _filter_by_time_budget(
    recommendations: list[AnalysisRecommendation],
    time_budget_seconds: float,
) -> list[AnalysisRecommendation]:
    """Filter recommendations by time budget.

    Args:
        recommendations: List of recommendations
        time_budget_seconds: Time budget in seconds

    Returns:
        Filtered recommendations within budget
    """
    budget_ms = time_budget_seconds * 1000
    cumulative = 0
    filtered = []

    # Sort by priority and confidence for selection
    for rec in sorted(recommendations, key=lambda x: (x.priority, -x.confidence)):
        if cumulative + rec.estimated_runtime_ms <= budget_ms:
            filtered.append(rec)
            cumulative += rec.estimated_runtime_ms

    return filtered


def recommend_analyses(
    data: NDArray[np.floating[Any]],
    sample_rate: float = 1.0,
    *,
    time_budget_seconds: float | None = None,
    confidence_target: float = 0.7,
    exclude_domains: list[AnalysisDomain] | None = None,
) -> list[AnalysisRecommendation]:
    """Recommend which analyses to run based on signal characteristics.

    Uses signal classification, quality metrics, and heuristics to
    recommend the most valuable analyses for a given signal.

    Args:
        data: Input signal data.
        sample_rate: Sample rate in Hz.
        time_budget_seconds: Optional time budget (prioritizes faster analyses).
        confidence_target: Minimum expected confidence threshold.
        exclude_domains: Domains to exclude from recommendations.

    Returns:
        List of AnalysisRecommendation sorted by priority.

    Example:
        >>> import numpy as np
        >>> import oscura as osc
        >>> # Generate test signal
        >>> t = np.linspace(0, 1, 10000)
        >>> signal = np.sin(2 * np.pi * 100 * t)
        >>> recommendations = osc.recommend_analyses(signal, sample_rate=10000)
        >>> for rec in recommendations[:3]:
        ...     print(f"{rec.domain.value}: {rec.reasoning}")
        waveform: Basic waveform measurements are always applicable
        statistics: Statistical analysis provides foundational metrics
        spectral: Spectral analysis reveals frequency content - signal appears periodic
    """
    recommendations: list[AnalysisRecommendation] = []
    exclude = set(exclude_domains or [])

    # Extract signal features via classification
    classification = classify_signal(data, sample_rate)
    is_digital = classification.get("is_digital", False)
    is_periodic = classification.get("is_periodic", False)
    dominant_freq = classification.get("dominant_frequency")

    # Build recommendations based on signal characteristics
    _add_foundational_recommendations(recommendations, exclude)
    _add_spectral_recommendation(recommendations, exclude, is_periodic)

    if is_digital:
        _add_digital_recommendations(recommendations, exclude, dominant_freq)

    if is_periodic:
        _add_periodic_recommendations(recommendations, exclude, is_digital)

    _add_pattern_and_entropy_recommendations(recommendations, exclude, len(data), is_periodic)

    # Apply filtering and ranking
    recommendations = _filter_by_confidence(recommendations, confidence_target)

    if time_budget_seconds is not None:
        recommendations = _filter_by_time_budget(recommendations, time_budget_seconds)

    # Final ranking by priority, then confidence
    recommendations.sort(key=lambda x: (x.priority, -x.confidence))

    return recommendations


def get_optimal_domain_order(
    recommendations: list[AnalysisRecommendation],
) -> list[AnalysisDomain]:
    """Get optimal order for running analyses.

    Considers dependencies and priorities to determine best order.

    Args:
        recommendations: List of analysis recommendations.

    Returns:
        Ordered list of domains to analyze.

    Example:
        >>> import numpy as np
        >>> import oscura as osc
        >>> # Generate test signal
        >>> t = np.linspace(0, 1, 10000)
        >>> signal = np.sin(2 * np.pi * 100 * t)
        >>> recommendations = osc.recommend_analyses(signal, sample_rate=10000)
        >>> order = osc.get_optimal_domain_order(recommendations)
        >>> print([d.value for d in order])
        ['waveform', 'statistics', 'spectral', 'patterns', 'entropy']
    """
    # Avoid circular import
    from oscura.reporting.config import AnalysisDomain

    # Define dependencies
    dependencies = {
        AnalysisDomain.JITTER: [AnalysisDomain.TIMING],
        AnalysisDomain.EYE: [AnalysisDomain.DIGITAL],
        AnalysisDomain.PROTOCOLS: [AnalysisDomain.DIGITAL],
        AnalysisDomain.INFERENCE: [AnalysisDomain.PATTERNS],
    }

    # Build order respecting dependencies
    ordered = []
    remaining = {r.domain for r in recommendations}

    while remaining:
        # Find domains with satisfied dependencies
        ready = []
        for domain in remaining:
            deps = dependencies.get(domain, [])
            if all(d not in remaining or d in ordered for d in deps):
                ready.append(domain)

        if not ready:
            # No ready domains - just add remaining (circular deps)
            ready = list(remaining)

        # Add highest priority ready domain
        for rec in sorted(recommendations, key=lambda x: (x.priority, -x.confidence)):
            if rec.domain in ready:
                ordered.append(rec.domain)
                remaining.discard(rec.domain)
                break

    return ordered


__all__ = [
    "AnalysisRecommendation",
    "assess_signal_quality",
    "check_measurement_suitability",
    "classify_signal",
    "get_optimal_domain_order",
    "recommend_analyses",
    "suggest_measurements",
]
