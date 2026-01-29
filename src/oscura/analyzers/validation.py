"""Signal validation and suitability checking for measurements.

This module provides helper functions to determine whether a signal is suitable
for specific measurements before attempting them. This helps avoid NaN results
and provides better user feedback.

Example:
    >>> from oscura.analyzers.validation import is_suitable_for_frequency, get_valid_measurements
    >>> suitable, reason = is_suitable_for_frequency(trace)
    >>> if suitable:
    ...     freq = frequency(trace)
    >>> valid_measurements = get_valid_measurements(trace)
    >>> print(f"Applicable measurements: {', '.join(valid_measurements)}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


def is_suitable_for_frequency_measurement(trace: WaveformTrace) -> tuple[bool, str]:
    """Check if trace is suitable for frequency measurement.

    Args:
        trace: Input waveform trace.

    Returns:
        Tuple of (is_suitable, reason). If not suitable, reason explains why.

    Example:
        >>> suitable, reason = is_suitable_for_frequency_measurement(trace)
        >>> if suitable:
        ...     freq = frequency(trace)
        ... else:
        ...     print(f"Cannot measure frequency: {reason}")
    """
    from oscura.analyzers.waveform.measurements import _find_edges

    data = trace.data
    n = len(data)

    # Check minimum samples
    if n < 3:
        return False, f"Insufficient samples ({n} < 3)"

    # Check for variation (DC signal)
    if np.std(data) < 1e-12:
        return False, "Signal has no variation (DC or constant)"

    # Check for edges
    rising_edges = _find_edges(trace, "rising")
    if len(rising_edges) < 2:
        return (
            False,
            f"Insufficient edges for periodic measurement (found {len(rising_edges)} rising edges, need at least 2)",
        )

    # Check period consistency (is it periodic?)
    if len(rising_edges) >= 3:
        edge_times = rising_edges * trace.metadata.time_base
        periods = np.diff(edge_times)
        period_cv = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else float("inf")

        if period_cv > 0.2:
            return (
                False,
                f"Signal is not periodic (period variation: {period_cv * 100:.1f}% > 20%)",
            )

    return True, "Signal is suitable for frequency measurement"


def is_suitable_for_duty_cycle_measurement(trace: WaveformTrace) -> tuple[bool, str]:
    """Check if trace is suitable for duty cycle measurement.

    Args:
        trace: Input waveform trace.

    Returns:
        Tuple of (is_suitable, reason).

    Example:
        >>> suitable, reason = is_suitable_for_duty_cycle_measurement(trace)
        >>> if suitable:
        ...     dc = duty_cycle(trace)
    """
    from oscura.analyzers.waveform.measurements import _find_edges

    # Check if suitable for frequency first (duty cycle needs periodic signal)
    freq_suitable, freq_reason = is_suitable_for_frequency_measurement(trace)
    if not freq_suitable:
        return False, freq_reason

    # Need both rising and falling edges
    rising = _find_edges(trace, "rising")
    falling = _find_edges(trace, "falling")

    if len(rising) == 0:
        return False, "No rising edges detected"

    if len(falling) == 0:
        return False, "No falling edges detected"

    return True, "Signal is suitable for duty cycle measurement"


def is_suitable_for_rise_time_measurement(trace: WaveformTrace) -> tuple[bool, str]:
    """Check if trace is suitable for rise time measurement.

    Args:
        trace: Input waveform trace.

    Returns:
        Tuple of (is_suitable, reason).

    Example:
        >>> suitable, reason = is_suitable_for_rise_time_measurement(trace)
        >>> if suitable:
        ...     rt = rise_time(trace)
    """
    from oscura.analyzers.waveform.measurements import _find_edges, _find_levels

    data = trace.data
    n = len(data)

    if n < 3:
        return False, f"Insufficient samples ({n} < 3)"

    # Check for amplitude
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return False, "Signal has no amplitude (flat or inverted)"

    # Check for rising edges
    rising_edges = _find_edges(trace, "rising")

    if len(rising_edges) == 0:
        return False, "No rising edges detected"

    # Check sample rate vs transition time
    # Find a rising transition and count samples across it
    sample_rate = trace.metadata.sample_rate
    low_ref = low + 0.1 * amplitude
    high_ref = low + 0.9 * amplitude

    # Find first rising transition
    crossings = np.where((data[:-1] < low_ref) & (data[1:] >= low_ref))[0]

    if len(crossings) > 0:
        idx = crossings[0]
        # Count samples from 10% to 90%
        remaining = data[idx:]
        above_high = remaining >= high_ref

        if np.any(above_high):
            end_offset = np.argmax(above_high)
            samples_in_transition = end_offset

            if samples_in_transition < 2:
                est_rise_time = samples_in_transition / sample_rate
                recommended_rate = 10 / est_rise_time
                return (
                    False,
                    f"Insufficient sample rate for transition (< 2 samples). "
                    f"Recommend sample rate > {recommended_rate:.3e} Hz",
                )

    return True, "Signal is suitable for rise time measurement"


def is_suitable_for_fall_time_measurement(trace: WaveformTrace) -> tuple[bool, str]:
    """Check if trace is suitable for fall time measurement.

    Args:
        trace: Input waveform trace.

    Returns:
        Tuple of (is_suitable, reason).
    """
    from oscura.analyzers.waveform.measurements import _find_edges, _find_levels

    data = trace.data
    n = len(data)

    if n < 3:
        return False, f"Insufficient samples ({n} < 3)"

    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return False, "Signal has no amplitude (flat or inverted)"

    # Check for falling edges
    falling_edges = _find_edges(trace, "falling")

    if len(falling_edges) == 0:
        return False, "No falling edges detected"

    return True, "Signal is suitable for fall time measurement"


def is_suitable_for_jitter_measurement(trace: WaveformTrace) -> tuple[bool, str]:
    """Check if trace is suitable for jitter measurement.

    Args:
        trace: Input waveform trace.

    Returns:
        Tuple of (is_suitable, reason).

    Example:
        >>> suitable, reason = is_suitable_for_jitter_measurement(trace)
        >>> if suitable:
        ...     jitter = rms_jitter(trace)
    """
    from oscura.analyzers.waveform.measurements import _find_edges

    # Jitter needs periodic signal
    freq_suitable, freq_reason = is_suitable_for_frequency_measurement(trace)
    if not freq_suitable:
        return False, freq_reason

    # Need at least 3 edges (2 periods minimum)
    edges = _find_edges(trace, "rising")

    if len(edges) < 3:
        return (
            False,
            f"Insufficient edges for jitter measurement (found {len(edges)}, need at least 3)",
        )

    return True, "Signal is suitable for jitter measurement"


def get_valid_measurements(trace: WaveformTrace) -> list[str]:
    """Get list of measurements that are suitable for this trace.

    Analyzes the signal characteristics and returns the names of all
    measurement functions that should return valid (non-NaN) results.

    Args:
        trace: Input waveform trace.

    Returns:
        List of measurement function names (without parentheses).

    Example:
        >>> valid = get_valid_measurements(trace)
        >>> print(f"Applicable measurements: {', '.join(valid)}")
        >>> # Then apply only valid measurements
        >>> for meas_name in valid:
        ...     func = getattr(tk, meas_name)
        ...     result = func(trace)
    """
    valid: list[str] = []

    _add_basic_measurements(valid, trace)
    _add_edge_measurements(valid, trace)
    _add_frequency_measurements(valid, trace)
    _add_pulse_measurements(valid, trace)
    _add_overshoot_measurements(valid, trace)
    _add_slew_rate_measurement(valid)

    return valid


def _add_basic_measurements(valid: list[str], trace: WaveformTrace) -> None:
    """Add basic measurements that almost always work.

    Args:
        valid: List to append to.
        trace: Input trace.
    """
    if len(trace.data) > 0:
        valid.extend(["mean", "rms"])
    if len(trace.data) >= 2:
        valid.append("amplitude")


def _add_edge_measurements(valid: list[str], trace: WaveformTrace) -> None:
    """Add edge-based measurements.

    Args:
        valid: List to append to.
        trace: Input trace.
    """
    if is_suitable_for_rise_time_measurement(trace)[0]:
        valid.append("rise_time")
    if is_suitable_for_fall_time_measurement(trace)[0]:
        valid.append("fall_time")


def _add_frequency_measurements(valid: list[str], trace: WaveformTrace) -> None:
    """Add frequency/period measurements.

    Args:
        valid: List to append to.
        trace: Input trace.
    """
    if is_suitable_for_frequency_measurement(trace)[0]:
        valid.extend(["frequency", "period"])
    if is_suitable_for_duty_cycle_measurement(trace)[0]:
        valid.append("duty_cycle")
    if is_suitable_for_jitter_measurement(trace)[0]:
        valid.extend(["rms_jitter", "peak_to_peak_jitter"])


def _add_pulse_measurements(valid: list[str], trace: WaveformTrace) -> None:
    """Add pulse width measurements.

    Args:
        valid: List to append to.
        trace: Input trace.
    """
    from oscura.analyzers.waveform.measurements import _find_edges

    rising = _find_edges(trace, "rising")
    falling = _find_edges(trace, "falling")

    if len(rising) > 0 and len(falling) > 0:
        valid.append("pulse_width")


def _add_overshoot_measurements(valid: list[str], trace: WaveformTrace) -> None:
    """Add overshoot/undershoot measurements.

    Args:
        valid: List to append to.
        trace: Input trace.
    """
    from oscura.analyzers.waveform.measurements import _find_levels

    if len(trace.data) >= 3:
        low, high = _find_levels(trace.data)
        if high - low > 0:
            valid.extend(["overshoot", "undershoot", "preshoot"])


def _add_slew_rate_measurement(valid: list[str]) -> None:
    """Add slew rate if edge measurements available.

    Args:
        valid: List to check and append to.
    """
    if "rise_time" in valid or "fall_time" in valid:
        valid.append("slew_rate")


def analyze_signal_characteristics(trace: WaveformTrace) -> dict[str, bool | int | str | list[str]]:
    """Perform comprehensive signal characteristic analysis.

    Determines signal type, edge counts, periodicity, and recommends
    applicable measurements.

    Args:
        trace: Input waveform trace.

    Returns:
        Dictionary containing signal characteristics including sufficient_samples,
        has_amplitude, has_variation, edge counts, periodicity, signal_type.

    Example:
        >>> chars = analyze_signal_characteristics(trace)
        >>> if chars['is_periodic']:
        ...     print("Signal is periodic")
    """
    from oscura.analyzers.waveform.measurements import _find_edges

    data = trace.data
    characteristics = _init_characteristics(data)

    if not characteristics["has_variation"]:
        characteristics["signal_type"] = "dc"
        characteristics["recommended_measurements"] = ["mean", "rms"]
        return characteristics

    rising_edges = _find_edges(trace, "rising")
    falling_edges = _find_edges(trace, "falling")
    _update_edge_counts(characteristics, rising_edges, falling_edges)
    _check_periodicity(characteristics, rising_edges)
    _classify_signal_type(characteristics, data, len(data))

    characteristics["recommended_measurements"] = get_valid_measurements(trace)

    return characteristics


def _init_characteristics(data: NDArray[np.float64]) -> dict[str, bool | int | str | list[str]]:
    """Initialize characteristics dictionary with basic signal properties.

    Args:
        data: Signal data array.

    Returns:
        Dictionary with initial characteristics.
    """
    n = len(data)
    std = np.std(data)
    amplitude = np.max(data) - np.min(data)

    characteristics: dict[str, int | str | list[str]] = {
        "sufficient_samples": int(n >= 16),
        "has_amplitude": int(amplitude > 1e-12),
        "has_variation": int(std > 1e-12),
        "has_edges": int(False),
        "is_periodic": int(False),
        "edge_count": 0,
        "rising_edge_count": 0,
        "falling_edge_count": 0,
        "signal_type": "unknown",
        "recommended_measurements": [],
    }
    return characteristics


def _update_edge_counts(
    characteristics: dict[str, bool | int | str | list[str]],
    rising_edges: NDArray[np.float64],
    falling_edges: NDArray[np.float64],
) -> None:
    """Update edge count statistics in characteristics dictionary.

    Args:
        characteristics: Dictionary to update.
        rising_edges: Array of rising edge indices.
        falling_edges: Array of falling edge indices.
    """
    rising_edge_count = len(rising_edges)
    falling_edge_count = len(falling_edges)
    edge_count = rising_edge_count + falling_edge_count

    characteristics["rising_edge_count"] = rising_edge_count
    characteristics["falling_edge_count"] = falling_edge_count
    characteristics["edge_count"] = edge_count
    characteristics["has_edges"] = edge_count > 0


def _check_periodicity(
    characteristics: dict[str, bool | int | str | list[str]], rising_edges: NDArray[np.float64]
) -> None:
    """Check if signal is periodic based on edge spacing.

    Args:
        characteristics: Dictionary to update.
        rising_edges: Array of rising edge indices.
    """
    if len(rising_edges) >= 3:
        periods = np.diff(rising_edges)
        period_cv = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else float("inf")

        if period_cv < 0.2:  # Less than 20% variation
            characteristics["is_periodic"] = True


def _classify_signal_type(
    characteristics: dict[str, bool | int | str | list[str]], data: NDArray[np.float64], n: int
) -> None:
    """Classify signal type based on characteristics.

    Args:
        characteristics: Dictionary to update.
        data: Signal data array.
        n: Number of samples.
    """
    if not characteristics["has_edges"]:
        characteristics["signal_type"] = _classify_no_edge_signal(data, n)
    elif characteristics["is_periodic"]:
        characteristics["signal_type"] = "periodic_digital"
    else:
        characteristics["signal_type"] = "aperiodic_digital"


def _classify_no_edge_signal(data: NDArray[np.float64], n: int) -> str:
    """Classify signal without edges (analog or noise).

    Args:
        data: Signal data array.
        n: Number of samples.

    Returns:
        Signal type classification string.
    """
    if n >= 16:
        fft_result = np.abs(np.fft.rfft(data - np.mean(data)))
        peak_power = np.max(fft_result[1:]) if len(fft_result) > 1 else 0
        avg_power = np.mean(fft_result[1:]) if len(fft_result) > 1 else 0

        if peak_power > 10 * avg_power:
            return "periodic_analog"
        else:
            return "noise"
    return "unknown"


def get_measurement_requirements(measurement_name: str) -> dict[str, str | int | list[str]]:
    """Get requirements for a specific measurement.

    Args:
        measurement_name: Name of the measurement function.

    Returns:
        Dictionary containing:
            - description: str - what the measurement computes
            - min_samples: int - minimum data points needed
            - required_signal_types: list[str] - suitable signal types
            - required_features: list[str] - required signal features
            - common_nan_causes: list[str] - common reasons for NaN

    Example:
        >>> reqs = get_measurement_requirements('frequency')
        >>> print(f"Minimum samples: {reqs['min_samples']}")
        >>> print(f"Required features: {', '.join(reqs['required_features'])}")
    """
    requirements = _get_all_measurement_requirements()
    default = _get_default_measurement_requirements()
    return requirements.get(measurement_name, default)


def _get_all_measurement_requirements() -> dict[str, dict[str, str | int | list[str]]]:
    """Get complete measurement requirements dictionary."""
    timing_reqs = _get_timing_measurement_requirements()
    amplitude_reqs = _get_amplitude_measurement_requirements()
    jitter_reqs = _get_jitter_measurement_requirements()
    statistical_reqs = _get_statistical_measurement_requirements()

    return {**timing_reqs, **amplitude_reqs, **jitter_reqs, **statistical_reqs}


def _get_timing_measurement_requirements() -> dict[str, dict[str, str | int | list[str]]]:
    """Get requirements for timing-related measurements."""
    return {
        "frequency": {
            "description": "Measures the repetition rate of a periodic signal",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "periodic_analog"],
            "required_features": ["edges", "periodic"],
            "common_nan_causes": [
                "DC signal (no transitions)",
                "Aperiodic signal (< 2 edges)",
                "Highly variable period (> 20% variation)",
            ],
        },
        "period": {
            "description": "Measures time between consecutive edges",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "periodic_analog"],
            "required_features": ["edges", "periodic"],
            "common_nan_causes": [
                "DC signal",
                "Fewer than 2 edges detected",
                "Aperiodic signal",
            ],
        },
        "duty_cycle": {
            "description": "Measures ratio of high time to period",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital"],
            "required_features": ["rising_edges", "falling_edges", "periodic"],
            "common_nan_causes": [
                "Non-periodic signal",
                "Missing rising or falling edges",
                "DC signal",
            ],
        },
        "rise_time": {
            "description": "Measures time for rising edge transition",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "aperiodic_digital", "periodic_analog"],
            "required_features": ["rising_edges", "amplitude"],
            "common_nan_causes": [
                "No rising edges",
                "Insufficient sample rate",
                "DC signal",
            ],
        },
        "fall_time": {
            "description": "Measures time for falling edge transition",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "aperiodic_digital", "periodic_analog"],
            "required_features": ["falling_edges", "amplitude"],
            "common_nan_causes": [
                "No falling edges",
                "Insufficient sample rate",
                "DC signal",
            ],
        },
        "pulse_width": {
            "description": "Measures duration of high or low pulse",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "aperiodic_digital"],
            "required_features": ["rising_edges", "falling_edges"],
            "common_nan_causes": [
                "Missing edge pairs",
                "DC signal",
                "Incomplete pulses",
            ],
        },
        "slew_rate": {
            "description": "Measures dV/dt during transitions",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "aperiodic_digital"],
            "required_features": ["edges", "amplitude"],
            "common_nan_causes": ["No edges", "No amplitude", "DC signal"],
        },
    }


def _get_amplitude_measurement_requirements() -> dict[str, dict[str, str | int | list[str]]]:
    """Get requirements for amplitude-related measurements."""
    return {
        "amplitude": {
            "description": "Measures peak-to-peak voltage",
            "min_samples": 2,
            "required_signal_types": ["all"],
            "required_features": [],
            "common_nan_causes": ["Fewer than 2 samples"],
        },
        "overshoot": {
            "description": "Measures overshoot above high level",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "aperiodic_digital"],
            "required_features": ["amplitude"],
            "common_nan_causes": ["No amplitude", "DC signal"],
        },
        "undershoot": {
            "description": "Measures undershoot below low level",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital", "aperiodic_digital"],
            "required_features": ["amplitude"],
            "common_nan_causes": ["No amplitude", "DC signal"],
        },
    }


def _get_jitter_measurement_requirements() -> dict[str, dict[str, str | int | list[str]]]:
    """Get requirements for jitter measurements."""
    return {
        "rms_jitter": {
            "description": "Measures timing uncertainty (RMS)",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital"],
            "required_features": ["edges", "periodic"],
            "common_nan_causes": [
                "Fewer than 3 edges",
                "Non-periodic signal",
                "DC signal",
            ],
        },
        "peak_to_peak_jitter": {
            "description": "Measures peak-to-peak timing variation",
            "min_samples": 3,
            "required_signal_types": ["periodic_digital"],
            "required_features": ["edges", "periodic"],
            "common_nan_causes": [
                "Fewer than 3 edges",
                "Non-periodic signal",
                "DC signal",
            ],
        },
    }


def _get_statistical_measurement_requirements() -> dict[str, dict[str, str | int | list[str]]]:
    """Get requirements for statistical measurements."""
    return {
        "mean": {
            "description": "Calculates DC level (average voltage)",
            "min_samples": 1,
            "required_signal_types": ["all"],
            "required_features": [],
            "common_nan_causes": ["No data"],
        },
        "rms": {
            "description": "Calculates root-mean-square voltage",
            "min_samples": 1,
            "required_signal_types": ["all"],
            "required_features": [],
            "common_nan_causes": ["No data"],
        },
    }


def _get_default_measurement_requirements() -> dict[str, str | int | list[str]]:
    """Get default requirements for undocumented measurements."""
    return {
        "description": "Measurement not documented",
        "min_samples": 1,
        "required_signal_types": ["unknown"],
        "required_features": [],
        "common_nan_causes": ["Check measurement documentation"],
    }


__all__ = [
    "analyze_signal_characteristics",
    "get_measurement_requirements",
    "get_valid_measurements",
    "is_suitable_for_duty_cycle_measurement",
    "is_suitable_for_fall_time_measurement",
    "is_suitable_for_frequency_measurement",
    "is_suitable_for_jitter_measurement",
    "is_suitable_for_rise_time_measurement",
]
