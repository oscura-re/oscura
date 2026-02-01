"""Waveform timing and amplitude measurements.

This module provides IEEE 181-2011 and IEEE 1057-2017 compliant
waveform measurements including rise/fall time, period, frequency,
amplitude, and RMS.


Example:
    >>> from oscura.analyzers.waveform.measurements import rise_time, measure
    >>> t_rise = rise_time(trace)
    >>> results = measure(trace, parameters=["rise_time", "frequency"])

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
    IEEE 1057-2017: Standard for Digitizing Waveform Recorders
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
from numpy import floating as np_floating

from oscura.core.measurement_result import make_inapplicable, make_measurement

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import MeasurementResult, WaveformTrace


# Measurement metadata: unit information for all waveform measurements
MEASUREMENT_METADATA: dict[str, dict[str, str]] = {
    # Time-domain measurements
    "rise_time": {"unit": "s", "description": "Rise time (10%-90%)"},
    "fall_time": {"unit": "s", "description": "Fall time (90%-10%)"},
    "period": {"unit": "s", "description": "Signal period"},
    "pulse_width": {"unit": "s", "description": "Pulse width"},
    "jitter": {"unit": "s", "description": "Period jitter"},
    # Frequency measurements
    "frequency": {"unit": "Hz", "description": "Signal frequency"},
    "clock_frequency": {"unit": "Hz", "description": "Clock frequency"},
    "dominant_freq": {"unit": "Hz", "description": "Dominant frequency"},
    # Voltage measurements
    "amplitude": {"unit": "V", "description": "Peak-to-peak amplitude"},
    "mean": {"unit": "V", "description": "Mean voltage"},
    "rms": {"unit": "V", "description": "RMS voltage"},
    "threshold": {"unit": "V", "description": "Logic threshold"},
    "min": {"unit": "V", "description": "Minimum voltage"},
    "max": {"unit": "V", "description": "Maximum voltage"},
    "std": {"unit": "V", "description": "Standard deviation"},
    "median": {"unit": "V", "description": "Median voltage"},
    # Ratio measurements (0-1, displayed as percentage)
    "duty_cycle": {"unit": "ratio", "description": "Duty cycle"},
    # Percentage measurements (already 0-100)
    "overshoot": {"unit": "%", "description": "Overshoot percentage"},
    "undershoot": {"unit": "%", "description": "Undershoot percentage"},
    "thd": {"unit": "%", "description": "Total harmonic distortion"},
    # Decibel measurements
    "snr": {"unit": "dB", "description": "Signal-to-noise ratio"},
    "sinad": {"unit": "dB", "description": "SINAD"},
    "sfdr": {"unit": "dB", "description": "Spurious-free dynamic range"},
    # Dimensionless measurements
    "enob": {"unit": "", "description": "Effective number of bits"},
    "rising_edges": {"unit": "", "description": "Rising edge count"},
    "falling_edges": {"unit": "", "description": "Falling edge count"},
    "outliers": {"unit": "", "description": "Outlier count"},
    # Statistical measurements (squared units)
    "variance": {"unit": "V²", "description": "Variance"},
}


def rise_time(
    trace: WaveformTrace,
    *,
    ref_levels: tuple[float, float] = (0.1, 0.9),
) -> MeasurementResult:
    """Measure rise time between reference levels.

    Computes the time for a signal to transition from the lower
    reference level to the upper reference level, per IEEE 181-2011.

    Args:
        trace: Input waveform trace.
        ref_levels: Reference levels as fractions (0.0 to 1.0).
            Default (0.1, 0.9) for 10%-90% rise time.

    Returns:
        MeasurementResult with rise time in seconds, or inapplicable if no rising edge.

    Example:
        >>> result = rise_time(trace)
        >>> if result["applicable"]:
        ...     print(f"Rise time: {result['display']}")

    References:
        IEEE 181-2011 Section 5.2
    """
    if len(trace.data) < 3:
        return make_inapplicable("s", "Insufficient data (need ≥3 samples)")

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0 or np.isnan(amplitude):
        return make_inapplicable("s", "Constant signal (no transitions)")

    # Calculate reference voltages
    low_ref = low + ref_levels[0] * amplitude
    high_ref = low + ref_levels[1] * amplitude

    # Find rising edge: where signal crosses from below low_ref to above high_ref
    sample_period = 1.0 / trace.metadata.sample_rate

    # Find first crossing of low reference (going up)
    below_low = data < low_ref
    above_low = data >= low_ref

    # Find transitions from below to above low_ref
    transitions = np.where(below_low[:-1] & above_low[1:])[0]

    if len(transitions) == 0:
        return make_inapplicable("s", "No rising edges detected")

    best_rise_time: float | np_floating[Any] = np.nan

    for start_idx in transitions:
        # Find where signal crosses high reference
        remaining = data[start_idx:]
        above_high = remaining >= high_ref

        if not np.any(above_high):
            continue

        end_offset = np.argmax(above_high)
        end_idx = start_idx + end_offset

        # Ensure monotonic rise (no dips)
        segment = data[start_idx : end_idx + 1]
        if len(segment) < 2:
            continue

        # Interpolate for sub-sample accuracy
        t_low = _interpolate_crossing_time(data, start_idx, low_ref, sample_period, rising=True)
        t_high = _interpolate_crossing_time(data, end_idx - 1, high_ref, sample_period, rising=True)

        if t_low is not None and t_high is not None:
            rt = t_high - t_low
            if rt > 0 and (np.isnan(best_rise_time) or rt < best_rise_time):
                best_rise_time = rt

    if np.isnan(best_rise_time):
        return make_inapplicable("s", "No valid rising edge found")

    return make_measurement(float(best_rise_time), "s")


def fall_time(
    trace: WaveformTrace,
    *,
    ref_levels: tuple[float, float] = (0.9, 0.1),
) -> MeasurementResult:
    """Measure fall time between reference levels.

    Computes the time for a signal to transition from the upper
    reference level to the lower reference level, per IEEE 181-2011.

    Args:
        trace: Input waveform trace.
        ref_levels: Reference levels as fractions (0.0 to 1.0).
            Default (0.9, 0.1) for 90%-10% fall time.

    Returns:
        MeasurementResult with fall time in seconds, or inapplicable if no falling edge.

    Example:
        >>> result = fall_time(trace)
        >>> if result["applicable"]:
        ...     print(f"Fall time: {result['display']}")

    References:
        IEEE 181-2011 Section 5.2
    """
    if len(trace.data) < 3:
        return make_inapplicable("s", "Insufficient data (need ≥3 samples)")

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0 or np.isnan(amplitude):
        return make_inapplicable("s", "Constant signal (no transitions)")

    # Calculate reference voltages (note: ref_levels[0] is the higher one for fall)
    high_ref = low + ref_levels[0] * amplitude
    low_ref = low + ref_levels[1] * amplitude

    sample_period = 1.0 / trace.metadata.sample_rate

    # Find where signal is above high reference
    above_high = data >= high_ref
    below_high = data < high_ref

    # Find transitions from above to below high_ref
    transitions = np.where(above_high[:-1] & below_high[1:])[0]

    if len(transitions) == 0:
        return make_inapplicable("s", "No falling edges detected")

    best_fall_time: float | np_floating[Any] = np.nan

    for start_idx in transitions:
        # Find where signal crosses low reference
        remaining = data[start_idx:]
        below_low = remaining <= low_ref

        if not np.any(below_low):
            continue

        end_offset = np.argmax(below_low)
        end_idx = start_idx + end_offset

        segment = data[start_idx : end_idx + 1]
        if len(segment) < 2:
            continue

        # Interpolate for sub-sample accuracy
        t_high = _interpolate_crossing_time(data, start_idx, high_ref, sample_period, rising=False)
        t_low = _interpolate_crossing_time(data, end_idx - 1, low_ref, sample_period, rising=False)

        if t_high is not None and t_low is not None:
            ft = t_low - t_high
            if ft > 0 and (np.isnan(best_fall_time) or ft < best_fall_time):
                best_fall_time = ft

    if np.isnan(best_fall_time):
        return make_inapplicable("s", "No valid falling edge found")

    return make_measurement(float(best_fall_time), "s")


@overload
def period(
    trace: WaveformTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
    return_all: Literal[False] = False,
) -> MeasurementResult: ...


@overload
def period(
    trace: WaveformTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
    return_all: Literal[True],
) -> NDArray[np.float64]: ...


def period(
    trace: WaveformTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
    return_all: bool = False,
) -> MeasurementResult | NDArray[np.float64]:
    """Measure signal period between consecutive edges.

    Computes the time between consecutive rising or falling edges.

    Args:
        trace: Input waveform trace.
        edge_type: Type of edges to use ("rising" or "falling").
        return_all: If True, return array of all periods. If False, return MeasurementResult.

    Returns:
        MeasurementResult with period in seconds (mean), or array of periods if return_all=True.

    Example:
        >>> result = period(trace)
        >>> if result["applicable"]:
        ...     print(f"Period: {result['display']}")

    References:
        IEEE 181-2011 Section 5.3
    """
    edges = _find_edges(trace, edge_type)

    if len(edges) < 2:
        if return_all:
            return np.array([], dtype=np.float64)
        return make_inapplicable("s", f"Insufficient {edge_type} edges (need ≥2)")

    periods = np.diff(edges)

    if return_all:
        return periods
    return make_measurement(float(np.mean(periods)), "s")


def frequency(
    trace: WaveformTrace,
    *,
    method: Literal["edge", "fft"] = "edge",
) -> MeasurementResult:
    """Measure signal frequency.

    Computes frequency either from edge-to-edge period or using FFT.
    The "edge" method automatically falls back to FFT if edge detection fails
    (e.g., for sine or triangle waves without clear rising edges).

    Args:
        trace: Input waveform trace.
        method: Measurement method:
            - "edge": 1/period from edge timing with automatic FFT fallback (default)
            - "fft": Peak of FFT magnitude spectrum (always use FFT)

    Returns:
        MeasurementResult with frequency in Hz, or inapplicable if measurement not possible.

    Raises:
        ValueError: If method is not one of the supported types.

    Example:
        >>> result = frequency(trace)
        >>> if result["applicable"]:
        ...     print(f"Frequency: {result['display']}")

        >>> # Force FFT method for smooth waveforms
        >>> result = frequency(trace, method="fft")

    References:
        IEEE 181-2011 Section 5.3
    """
    if method == "edge":
        # Try edge detection first (faster and more accurate for square waves)
        T = period(trace, edge_type="rising", return_all=False)

        # Fall back to FFT if edge detection fails
        if not T["applicable"] or not T["value"] or T["value"] <= 0:
            # Try FFT fallback for smooth waveforms (sine, triangle)
            return _frequency_fft(trace)

        return make_measurement(1.0 / T["value"], "Hz")

    elif method == "fft":
        return _frequency_fft(trace)

    else:
        raise ValueError(f"Unknown method: {method}")


def _frequency_fft(trace: WaveformTrace) -> MeasurementResult:
    """Compute frequency using FFT peak detection.

    Internal helper function for FFT-based frequency measurement.

    Args:
        trace: Input waveform trace.

    Returns:
        MeasurementResult with frequency in Hz, or inapplicable if measurement not possible.
    """
    if len(trace.data) < 16:
        return make_inapplicable("Hz", "Insufficient data for FFT (need ≥16 samples)")

    # Remove DC offset before FFT
    data = trace.data - np.mean(trace.data)

    # Check if signal is essentially constant (DC only)
    if np.std(data) < 1e-10:
        return make_inapplicable("Hz", "Constant signal (DC only)")

    n = len(data)
    fft_mag = np.abs(np.fft.rfft(data))

    # Find peak (skip DC component at index 0)
    peak_idx = np.argmax(fft_mag[1:]) + 1

    # Verify peak is significant (SNR check)
    # If the peak is not at least 3x the mean, it's likely noise
    if fft_mag[peak_idx] < 3.0 * np.mean(fft_mag[1:]):
        return make_inapplicable("Hz", "No dominant frequency (noisy signal)")

    # Calculate frequency from peak index
    freq_resolution = trace.metadata.sample_rate / n
    return make_measurement(float(peak_idx * freq_resolution), "Hz")


def duty_cycle(
    trace: WaveformTrace,
    *,
    percentage: bool = False,
) -> MeasurementResult:
    """Measure duty cycle.

    Computes duty cycle as the ratio of positive pulse width to period.
    Uses robust algorithm that handles extreme duty cycles (1%-99%) and
    incomplete waveforms (fewer than 2 complete cycles visible).

    Falls back to time-domain calculation when edge-based measurement fails.

    Args:
        trace: Input waveform trace.
        percentage: Ignored (always returns ratio, display format shows %).

    Returns:
        MeasurementResult with duty cycle as ratio (0-1), or inapplicable if not possible.

    Example:
        >>> result = duty_cycle(trace)
        >>> if result["applicable"]:
        ...     print(f"Duty cycle: {result['display']}")  # Shows as percentage

    References:
        IEEE 181-2011 Section 5.4
    """
    # Strategy: Use multiple methods depending on what edges are available
    # Method 1 (best): period + pulse width (needs 2+ rising edges, 1+ falling edge)
    # Method 2 (fallback): time-based calculation from data samples

    pw_pos = pulse_width(trace, polarity="positive", return_all=False)
    T = period(trace, edge_type="rising", return_all=False)

    # Method 1: Standard period-based calculation
    if isinstance(pw_pos, dict) and pw_pos.get("applicable") and pw_pos.get("value"):
        T_value = T.get("value") if isinstance(T, dict) else None
        if isinstance(T, dict) and T.get("applicable") and T_value and T_value > 0:
            pw_value = pw_pos["value"]
            if pw_value:
                dc = pw_value / T_value
                return make_measurement(dc, "ratio")

    # Method 2: Fallback for incomplete waveforms - time-domain calculation
    # Calculate fraction of time signal spends above midpoint threshold
    data = trace.data
    if len(data) < 3:
        return make_inapplicable("ratio", "Insufficient data (need ≥3 samples)")

    # Convert boolean data to float if needed
    if data.dtype == bool:
        data = data.astype(np.float64)

    low, high = _find_levels(data)
    amplitude = high - low

    # Check for invalid levels
    if amplitude <= 0 or np.isnan(amplitude):
        return make_inapplicable("ratio", "Constant signal (no transitions)")

    # Calculate threshold at 50% of amplitude
    mid = low + 0.5 * amplitude

    # Count samples above threshold
    above_threshold = data >= mid
    samples_high = np.sum(above_threshold)
    total_samples = len(data)

    if total_samples == 0:
        return make_inapplicable("ratio", "No data available")

    dc = float(samples_high) / total_samples
    return make_measurement(dc, "ratio")


@overload
def pulse_width(
    trace: WaveformTrace,
    *,
    polarity: Literal["positive", "negative"] = "positive",
    ref_level: float = 0.5,
    return_all: Literal[False] = False,
) -> MeasurementResult: ...


@overload
def pulse_width(
    trace: WaveformTrace,
    *,
    polarity: Literal["positive", "negative"] = "positive",
    ref_level: float = 0.5,
    return_all: Literal[True],
) -> NDArray[np.float64]: ...


def pulse_width(
    trace: WaveformTrace,
    *,
    polarity: Literal["positive", "negative"] = "positive",
    ref_level: float = 0.5,
    return_all: bool = False,
) -> MeasurementResult | NDArray[np.float64]:
    """Measure pulse width.

    Computes positive or negative pulse width at the specified reference level.

    Args:
        trace: Input waveform trace.
        polarity: "positive" for high pulses, "negative" for low pulses.
        ref_level: Reference level as fraction (0.0 to 1.0). Default 0.5 (50%).
        return_all: If True, return array of all widths. If False, return MeasurementResult.

    Returns:
        MeasurementResult with pulse width in seconds (mean), or array if return_all=True.

    Example:
        >>> result = pulse_width(trace, polarity="positive")
        >>> if result["applicable"]:
        ...     print(f"Pulse width: {result['display']}")

    References:
        IEEE 181-2011 Section 5.4
    """
    rising_edges = _find_edges(trace, "rising", ref_level)
    falling_edges = _find_edges(trace, "falling", ref_level)

    if len(rising_edges) == 0 or len(falling_edges) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        edge_type = (
            "rising and falling"
            if len(rising_edges) == 0 and len(falling_edges) == 0
            else ("rising" if len(rising_edges) == 0 else "falling")
        )
        return make_inapplicable("s", f"No {edge_type} edges found")

    widths: list[float] = []

    if polarity == "positive":
        # Rising to falling
        for r in rising_edges:
            # Find next falling edge after this rising edge
            next_falling = falling_edges[falling_edges > r]
            if len(next_falling) > 0:
                widths.append(next_falling[0] - r)
    else:
        # Falling to rising
        for f in falling_edges:
            # Find next rising edge after this falling edge
            next_rising = rising_edges[rising_edges > f]
            if len(next_rising) > 0:
                widths.append(next_rising[0] - f)

    if len(widths) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return make_inapplicable("s", f"No {polarity} pulses found")

    widths_arr = np.array(widths, dtype=np.float64)

    if return_all:
        return widths_arr
    return make_measurement(float(np.mean(widths_arr)), "s")


def overshoot(trace: WaveformTrace) -> MeasurementResult:
    """Measure overshoot percentage.

    Computes overshoot as (max - high) / amplitude * 100%.

    Args:
        trace: Input waveform trace.

    Returns:
        MeasurementResult with overshoot as percentage, or inapplicable if not applicable.

    Example:
        >>> result = overshoot(trace)
        >>> if result["applicable"]:
        ...     print(f"Overshoot: {result['display']}")

    References:
        IEEE 181-2011 Section 5.5
    """
    if len(trace.data) < 3:
        return make_inapplicable("%", "Insufficient data (need ≥3 samples)")

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0 or np.isnan(amplitude):
        return make_inapplicable("%", "Constant signal (no amplitude)")

    max_val = np.max(data)

    if max_val <= high:
        return make_measurement(0.0, "%")

    return make_measurement(float((max_val - high) / amplitude * 100), "%")


def undershoot(trace: WaveformTrace) -> MeasurementResult:
    """Measure undershoot percentage.

    Computes undershoot as (low - min) / amplitude * 100%.

    Args:
        trace: Input waveform trace.

    Returns:
        MeasurementResult with undershoot as percentage, or inapplicable if not applicable.

    Example:
        >>> result = undershoot(trace)
        >>> if result["applicable"]:
        ...     print(f"Undershoot: {result['display']}")

    References:
        IEEE 181-2011 Section 5.5
    """
    if len(trace.data) < 3:
        return make_inapplicable("%", "Insufficient data (need ≥3 samples)")

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0 or np.isnan(amplitude):
        return make_inapplicable("%", "Constant signal (no amplitude)")

    min_val = np.min(data)

    if min_val >= low:
        return make_measurement(0.0, "%")

    return make_measurement(float((low - min_val) / amplitude * 100), "%")


def preshoot(
    trace: WaveformTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
) -> MeasurementResult:
    """Measure preshoot percentage.

    Computes preshoot before transitions as percentage of amplitude.

    Args:
        trace: Input waveform trace.
        edge_type: Type of edge to analyze ("rising" or "falling").

    Returns:
        MeasurementResult with preshoot as percentage, or inapplicable if not applicable.

    Example:
        >>> result = preshoot(trace)
        >>> if result["applicable"]:
        ...     print(f"Preshoot: {result['display']}")

    References:
        IEEE 181-2011 Section 5.5
    """
    if len(trace.data) < 10:
        return make_inapplicable("%", "Insufficient data (need ≥10 samples)")

    # Convert memoryview to ndarray if needed
    data = np.asarray(trace.data)
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0 or np.isnan(amplitude):
        return make_inapplicable("%", "Constant signal (no amplitude)")

    # Find edge crossings at 50%
    mid = (low + high) / 2

    if edge_type == "rising":
        # Look for minimum before rising edge that goes below low level
        crossings = np.where((data[:-1] < mid) & (data[1:] >= mid))[0]
        if len(crossings) == 0:
            return make_inapplicable("%", "No rising edges found")

        max_preshoot = 0.0
        for idx in crossings:
            # Look at samples before crossing
            pre_samples = max(0, idx - 10)
            pre_region = data[pre_samples:idx]
            if len(pre_region) > 0:
                min_pre = np.min(pre_region)
                if min_pre < low:
                    preshoot_val = (low - min_pre) / amplitude * 100
                    max_preshoot = max(max_preshoot, preshoot_val)

        return make_measurement(max_preshoot, "%")

    else:  # falling
        crossings = np.where((data[:-1] >= mid) & (data[1:] < mid))[0]
        if len(crossings) == 0:
            return make_inapplicable("%", "No falling edges found")

        max_preshoot = 0.0
        for idx in crossings:
            pre_samples = max(0, idx - 10)
            pre_region = data[pre_samples:idx]
            if len(pre_region) > 0:
                max_pre = np.max(pre_region)
                if max_pre > high:
                    preshoot_val = (max_pre - high) / amplitude * 100
                    max_preshoot = max(max_preshoot, preshoot_val)

        return make_measurement(max_preshoot, "%")


def amplitude(trace: WaveformTrace) -> MeasurementResult:
    """Measure peak-to-peak amplitude.

    Computes Vpp as the difference between histogram-based high and low levels.

    Args:
        trace: Input waveform trace.

    Returns:
        MeasurementResult with amplitude in volts (or input units).

    Example:
        >>> result = amplitude(trace)
        >>> if result["applicable"]:
        ...     print(f"Amplitude: {result['display']}")

    References:
        IEEE 1057-2017 Section 4.2
    """
    if len(trace.data) < 2:
        return make_inapplicable("V", "Insufficient data (need ≥2 samples)")

    low, high = _find_levels(trace.data)
    amp = high - low

    if np.isnan(amp):
        return make_inapplicable("V", "Cannot determine amplitude")

    return make_measurement(amp, "V")


def rms(
    trace: WaveformTrace,
    *,
    ac_coupled: bool = False,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> MeasurementResult:
    """Compute RMS voltage.

    Calculates root-mean-square voltage of the waveform.

    Args:
        trace: Input waveform trace.
        ac_coupled: If True, remove DC offset before computing RMS.
        nan_policy: How to handle NaN values:
            - "propagate": Return inapplicable if any NaN present (default)
            - "omit": Ignore NaN values in calculation
            - "raise": Raise ValueError if any NaN present

    Returns:
        MeasurementResult with RMS voltage in volts (or input units).

    Raises:
        ValueError: If nan_policy="raise" and data contains NaN.

    Example:
        >>> result = rms(trace)
        >>> if result["applicable"]:
        ...     print(f"RMS: {result['display']}")

        >>> # Handle traces with NaN values
        >>> result = rms(trace, nan_policy="omit")

    References:
        IEEE 1057-2017 Section 4.3
    """
    if len(trace.data) == 0:
        return make_inapplicable("V", "Empty trace")

    # Convert memoryview to ndarray if needed
    data = np.asarray(trace.data)

    # Handle NaN based on policy
    if nan_policy == "raise":
        if np.any(np.isnan(data)):
            raise ValueError("Input data contains NaN values")
    elif nan_policy == "omit":
        # Use nanmean and nansum for NaN-safe calculation
        if ac_coupled:
            data = data - np.nanmean(data)
        return make_measurement(float(np.sqrt(np.nanmean(data**2))), "V")
    else:  # propagate
        if np.any(np.isnan(data)):
            return make_inapplicable("V", "Data contains NaN values")

    if ac_coupled:
        data = data - np.mean(data)

    return make_measurement(float(np.sqrt(np.mean(data**2))), "V")


def mean(
    trace: WaveformTrace,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> MeasurementResult:
    """Compute mean (DC) voltage.

    Calculates arithmetic mean of the waveform.

    Args:
        trace: Input waveform trace.
        nan_policy: How to handle NaN values:
            - "propagate": Return inapplicable if any NaN present (default)
            - "omit": Ignore NaN values in calculation
            - "raise": Raise ValueError if any NaN present

    Returns:
        MeasurementResult with mean voltage in volts (or input units).

    Raises:
        ValueError: If nan_policy="raise" and data contains NaN.

    Example:
        >>> result = mean(trace)
        >>> if result["applicable"]:
        ...     print(f"DC: {result['display']}")

        >>> # Handle traces with NaN values
        >>> result = mean(trace, nan_policy="omit")

    References:
        IEEE 1057-2017 Section 4.3
    """
    if len(trace.data) == 0:
        return make_inapplicable("V", "Empty trace")

    # Convert memoryview to ndarray if needed
    data = np.asarray(trace.data)

    # Handle NaN based on policy
    if nan_policy == "raise":
        if np.any(np.isnan(data)):
            raise ValueError("Input data contains NaN values")
        return make_measurement(float(np.mean(data)), "V")
    elif nan_policy == "omit":
        return make_measurement(float(np.nanmean(data)), "V")
    else:  # propagate
        if np.any(np.isnan(data)):
            return make_inapplicable("V", "Data contains NaN values")
        return make_measurement(float(np.mean(data)), "V")


def measure(
    trace: WaveformTrace,
    *,
    parameters: list[str] | None = None,
    include_units: bool = True,
) -> dict[str, Any]:
    """Compute multiple waveform measurements.

    Unified function for computing all or selected waveform measurements.
    Returns MeasurementResult format with applicability tracking.

    Args:
        trace: Input waveform trace.
        parameters: List of measurement names to compute. If None, compute all.
            Valid names: rise_time, fall_time, period, frequency, duty_cycle,
            amplitude, rms, mean, overshoot, undershoot, preshoot
        include_units: If True, include units in output (returns MeasurementResult).
            If False, return raw values (with NaN for inapplicable).

    Returns:
        Dictionary mapping measurement names to MeasurementResults (if include_units=True)
        or raw values (if include_units=False).

    Example:
        >>> results = measure(trace)
        >>> if results['rise_time']['applicable']:
        ...     print(f"Rise time: {results['rise_time']['display']}")

        >>> # Get specific measurements
        >>> results = measure(trace, parameters=["frequency", "amplitude"])

        >>> # Get raw values (legacy compatibility)
        >>> results = measure(trace, include_units=False)
        >>> freq = results["frequency"]  # float or np.nan

    References:
        IEEE 181-2011, IEEE 1057-2017
    """
    all_measurements = {
        "rise_time": rise_time,
        "fall_time": fall_time,
        "period": lambda t: period(t, return_all=False),
        "frequency": frequency,
        "duty_cycle": duty_cycle,
        "pulse_width_pos": lambda t: pulse_width(t, polarity="positive", return_all=False),
        "pulse_width_neg": lambda t: pulse_width(t, polarity="negative", return_all=False),
        "amplitude": amplitude,
        "rms": rms,
        "mean": mean,
        "overshoot": overshoot,
        "undershoot": undershoot,
        "preshoot": preshoot,
    }

    if parameters is None:
        selected = all_measurements
    else:
        selected = {k: v for k, v in all_measurements.items() if k in parameters}

    results: dict[str, Any] = {}

    for name, func in selected.items():
        try:
            measurement_result = func(trace)  # type: ignore[operator]

            if include_units:
                # Return full MeasurementResult
                results[name] = measurement_result
            else:
                # Legacy mode: extract raw value (NaN if inapplicable)
                if measurement_result["applicable"]:
                    results[name] = measurement_result["value"]
                else:
                    results[name] = np.nan

        except Exception:
            # On error, create inapplicable result
            if include_units:
                results[name] = make_inapplicable("", "Measurement failed")
            else:
                results[name] = np.nan

    return results


# =============================================================================
# Helper Functions
# =============================================================================


def _find_levels(data: NDArray[np_floating[Any]]) -> tuple[float, float]:
    """Find low and high levels using histogram method.

    Robust algorithm that handles extreme duty cycles (1%-99%) by using
    adaptive percentile-based level detection when histogram method fails.

    Args:
        data: Waveform data array.

    Returns:
        Tuple of (low_level, high_level).
    """
    # Convert boolean data to float if needed (for digital signals)
    if data.dtype == np.bool_:
        data = data.astype(np.float64)

    # Check for all-NaN data
    if np.all(np.isnan(data)):
        return float(np.nan), float(np.nan)

    # Use percentiles for robust level detection
    # For extreme duty cycles, use wider percentile range
    p01, p05, p10, p50, p90, p95, p99 = np.percentile(data, [1, 5, 10, 50, 90, 95, 99])

    # Check for constant or near-constant signal
    data_range = p99 - p01
    if data_range < 1e-10 or np.isnan(data_range):  # Essentially constant or NaN
        return float(p50), float(p50)

    # Refine using histogram peaks
    hist, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in lower and upper halves
    mid_idx = len(hist) // 2
    low_idx = np.argmax(hist[:mid_idx])
    high_idx = mid_idx + np.argmax(hist[mid_idx:])

    low = bin_centers[low_idx]
    high = bin_centers[high_idx]

    # Sanity check - if histogram method failed, use adaptive percentiles
    if high <= low:
        # For extreme duty cycles, use min/max with small outlier rejection
        # p01 and p99 remove top/bottom 1% outliers (noise, ringing)
        return float(p01), float(p99)

    return float(low), float(high)


def _find_edges(
    trace: WaveformTrace,
    edge_type: Literal["rising", "falling"],
    ref_level: float = 0.5,
) -> NDArray[np.float64]:
    """Find edge timestamps in a waveform.

    Args:
        trace: Input waveform.
        edge_type: Type of edges to find.
        ref_level: Reference level as fraction (0.0 to 1.0). Default 0.5 (50%).

    Returns:
        Array of edge timestamps in seconds.
    """
    data = trace.data
    sample_period = 1.0 / trace.metadata.sample_rate

    if len(data) < 3:
        return np.array([], dtype=np.float64)

    # Convert boolean data to float for arithmetic (NumPy 2.0+ compatibility)
    if data.dtype == bool:
        data = data.astype(np.float64)

    low, high = _find_levels(data)
    # Use ref_level parameter to compute threshold
    mid = low + ref_level * (high - low)

    if edge_type == "rising":
        crossings = np.where((data[:-1] < mid) & (data[1:] >= mid))[0]
    else:
        crossings = np.where((data[:-1] >= mid) & (data[1:] < mid))[0]

    # Convert to timestamps with interpolation
    timestamps = np.zeros(len(crossings), dtype=np.float64)

    for i, idx in enumerate(crossings):
        base_time = idx * sample_period

        # Linear interpolation
        if idx < len(data) - 1:
            v1, v2 = data[idx], data[idx + 1]
            if abs(v2 - v1) > 1e-12:
                t_offset = (mid - v1) / (v2 - v1) * sample_period
                t_offset = max(0, min(sample_period, t_offset))
                timestamps[i] = base_time + t_offset
            else:
                timestamps[i] = base_time + sample_period / 2
        else:
            timestamps[i] = base_time

    return timestamps


def _interpolate_crossing_time(
    data: NDArray[np_floating[Any]],
    idx: int,
    threshold: float,
    sample_period: float,
    rising: bool,
) -> float | None:
    """Interpolate threshold crossing time.

    Args:
        data: Waveform data.
        idx: Sample index near crossing.
        threshold: Threshold level.
        sample_period: Time between samples.
        rising: True for rising edge, False for falling.

    Returns:
        Time of crossing in seconds, or None if not found.
    """
    if idx < 0 or idx >= len(data) - 1:
        return None

    v1, v2 = data[idx], data[idx + 1]

    # Check direction
    if rising and not (v1 < threshold <= v2):
        # Search nearby
        for offset in range(-2, 3):
            check_idx = idx + offset
            if 0 <= check_idx < len(data) - 1:
                v1, v2 = data[check_idx], data[check_idx + 1]
                if v1 < threshold <= v2:
                    idx = check_idx
                    break
        else:
            return None

    if not rising and not (v1 >= threshold > v2):
        for offset in range(-2, 3):
            check_idx = idx + offset
            if 0 <= check_idx < len(data) - 1:
                v1, v2 = data[check_idx], data[check_idx + 1]
                if v1 >= threshold > v2:
                    idx = check_idx
                    break
        else:
            return None

    v1, v2 = data[idx], data[idx + 1]
    dv = v2 - v1

    if abs(dv) < 1e-12:
        t_offset = sample_period / 2
    else:
        t_offset = (threshold - v1) / dv * sample_period
        t_offset = max(0, min(sample_period, t_offset))

    return idx * sample_period + t_offset


__all__ = [
    "amplitude",
    "duty_cycle",
    "fall_time",
    "frequency",
    "mean",
    "measure",
    "overshoot",
    "period",
    "preshoot",
    "pulse_width",
    "rise_time",
    "rms",
    "undershoot",
]
