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

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


def rise_time(
    trace: WaveformTrace,
    *,
    ref_levels: tuple[float, float] = (0.1, 0.9),
) -> float | np_floating[Any]:
    """Measure rise time between reference levels.

    Computes the time for a signal to transition from the lower
    reference level to the upper reference level, per IEEE 181-2011.

    Args:
        trace: Input waveform trace.
        ref_levels: Reference levels as fractions (0.0 to 1.0).
            Default (0.1, 0.9) for 10%-90% rise time.

    Returns:
        Rise time in seconds, or np.nan if no valid rising edge found.

    Example:
        >>> t_rise = rise_time(trace)
        >>> print(f"Rise time: {t_rise * 1e9:.2f} ns")

    References:
        IEEE 181-2011 Section 5.2
    """
    if len(trace.data) < 3:
        return np.nan

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return np.nan

    # Calculate reference voltages
    low_ref = low + ref_levels[0] * amplitude
    high_ref = low + ref_levels[1] * amplitude

    # Find rising edge: where signal crosses from below low_ref to above high_ref
    sample_period = trace.metadata.time_base

    # Find first crossing of low reference (going up)
    below_low = data < low_ref
    above_low = data >= low_ref

    # Find transitions from below to above low_ref
    transitions = np.where(below_low[:-1] & above_low[1:])[0]

    if len(transitions) == 0:
        return np.nan

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

    return best_rise_time


def fall_time(
    trace: WaveformTrace,
    *,
    ref_levels: tuple[float, float] = (0.9, 0.1),
) -> float | np_floating[Any]:
    """Measure fall time between reference levels.

    Computes the time for a signal to transition from the upper
    reference level to the lower reference level, per IEEE 181-2011.

    Args:
        trace: Input waveform trace.
        ref_levels: Reference levels as fractions (0.0 to 1.0).
            Default (0.9, 0.1) for 90%-10% fall time.

    Returns:
        Fall time in seconds, or np.nan if no valid falling edge found.

    Example:
        >>> t_fall = fall_time(trace)
        >>> print(f"Fall time: {t_fall * 1e9:.2f} ns")

    References:
        IEEE 181-2011 Section 5.2
    """
    if len(trace.data) < 3:
        return np.nan

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return np.nan

    # Calculate reference voltages (note: ref_levels[0] is the higher one for fall)
    high_ref = low + ref_levels[0] * amplitude
    low_ref = low + ref_levels[1] * amplitude

    sample_period = trace.metadata.time_base

    # Find where signal is above high reference
    above_high = data >= high_ref
    below_high = data < high_ref

    # Find transitions from above to below high_ref
    transitions = np.where(above_high[:-1] & below_high[1:])[0]

    if len(transitions) == 0:
        return np.nan

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

    return best_fall_time


@overload
def period(
    trace: WaveformTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
    return_all: Literal[False] = False,
) -> float | np_floating[Any]: ...


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
) -> float | np_floating[Any] | NDArray[np.float64]:
    """Measure signal period between consecutive edges.

    Computes the time between consecutive rising or falling edges.

    Args:
        trace: Input waveform trace.
        edge_type: Type of edges to use ("rising" or "falling").
        return_all: If True, return array of all periods. If False, return mean.

    Returns:
        Period in seconds (mean if return_all=False), or array of periods.

    Example:
        >>> T = period(trace)
        >>> print(f"Period: {T * 1e6:.2f} us")

    References:
        IEEE 181-2011 Section 5.3
    """
    edges = _find_edges(trace, edge_type)

    if len(edges) < 2:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

    periods = np.diff(edges)

    if return_all:
        return periods
    return float(np.mean(periods))


def frequency(
    trace: WaveformTrace,
    *,
    method: Literal["edge", "fft"] = "edge",
) -> float | np_floating[Any]:
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
        Frequency in Hz, or np.nan if measurement not possible.

    Raises:
        ValueError: If method is not one of the supported types.

    Example:
        >>> f = frequency(trace)
        >>> print(f"Frequency: {f / 1e6:.3f} MHz")

        >>> # Force FFT method for smooth waveforms
        >>> f = frequency(trace, method="fft")

    References:
        IEEE 181-2011 Section 5.3
    """
    if method == "edge":
        # Try edge detection first (faster and more accurate for square waves)
        T = period(trace, edge_type="rising", return_all=False)

        # Fall back to FFT if edge detection fails
        if np.isnan(T) or T <= 0:
            # Try FFT fallback for smooth waveforms (sine, triangle)
            return _frequency_fft(trace)

        return 1.0 / T

    elif method == "fft":
        return _frequency_fft(trace)

    else:
        raise ValueError(f"Unknown method: {method}")


def _frequency_fft(trace: WaveformTrace) -> float | np_floating[Any]:
    """Compute frequency using FFT peak detection.

    Internal helper function for FFT-based frequency measurement.

    Args:
        trace: Input waveform trace.

    Returns:
        Frequency in Hz, or np.nan if measurement not possible.
    """
    if len(trace.data) < 16:
        return np.nan

    # Remove DC offset before FFT
    data = trace.data - np.mean(trace.data)

    # Check if signal is essentially constant (DC only)
    if np.std(data) < 1e-10:
        return np.nan

    n = len(data)
    fft_mag = np.abs(np.fft.rfft(data))

    # Find peak (skip DC component at index 0)
    peak_idx = np.argmax(fft_mag[1:]) + 1

    # Verify peak is significant (SNR check)
    # If the peak is not at least 3x the mean, it's likely noise
    if fft_mag[peak_idx] < 3.0 * np.mean(fft_mag[1:]):
        return np.nan

    # Calculate frequency from peak index
    freq_resolution = trace.metadata.sample_rate / n
    return float(peak_idx * freq_resolution)


def duty_cycle(
    trace: WaveformTrace,
    *,
    percentage: bool = False,
) -> float | np_floating[Any]:
    """Measure duty cycle.

    Computes duty cycle as the ratio of positive pulse width to period.
    Uses robust algorithm that handles extreme duty cycles (1%-99%) and
    incomplete waveforms (fewer than 2 complete cycles visible).

    Falls back to time-domain calculation when edge-based measurement fails.

    Args:
        trace: Input waveform trace.
        percentage: If True, return as percentage (0-100). If False, return ratio (0-1).

    Returns:
        Duty cycle as ratio or percentage, or np.nan if measurement not possible.

    Example:
        >>> dc = duty_cycle(trace, percentage=True)
        >>> print(f"Duty cycle: {dc:.1f}%")

    References:
        IEEE 181-2011 Section 5.4
    """
    # Strategy: Use multiple methods depending on what edges are available
    # Method 1 (best): period + pulse width (needs 2+ rising edges, 1+ falling edge)
    # Method 2 (fallback): time-based calculation from data samples

    pw_pos = pulse_width(trace, polarity="positive", return_all=False)
    T = period(trace, edge_type="rising", return_all=False)

    # Method 1: Standard period-based calculation
    if not np.isnan(pw_pos) and not np.isnan(T) and T > 0:
        dc = pw_pos / T
        if percentage:
            return dc * 100
        return dc

    # Method 2: Fallback for incomplete waveforms - time-domain calculation
    # Calculate fraction of time signal spends above midpoint threshold
    data = trace.data
    if len(data) < 3:
        return np.nan

    # Convert boolean data to float if needed
    if data.dtype == bool:
        data = data.astype(np.float64)

    low, high = _find_levels(data)
    amplitude = high - low

    # Check for invalid levels
    if amplitude <= 0 or np.isnan(amplitude):
        return np.nan

    # Calculate threshold at 50% of amplitude
    mid = low + 0.5 * amplitude

    # Count samples above threshold
    above_threshold = data >= mid
    samples_high = np.sum(above_threshold)
    total_samples = len(data)

    if total_samples == 0:
        return np.nan

    dc = float(samples_high) / total_samples

    if percentage:
        return dc * 100
    return dc


@overload
def pulse_width(
    trace: WaveformTrace,
    *,
    polarity: Literal["positive", "negative"] = "positive",
    ref_level: float = 0.5,
    return_all: Literal[False] = False,
) -> float | np_floating[Any]: ...


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
) -> float | np_floating[Any] | NDArray[np.float64]:
    """Measure pulse width.

    Computes positive or negative pulse width at the specified reference level.

    Args:
        trace: Input waveform trace.
        polarity: "positive" for high pulses, "negative" for low pulses.
        ref_level: Reference level as fraction (0.0 to 1.0). Default 0.5 (50%).
        return_all: If True, return array of all widths. If False, return mean.

    Returns:
        Pulse width in seconds.

    Example:
        >>> pw = pulse_width(trace, polarity="positive")
        >>> print(f"Pulse width: {pw * 1e6:.2f} us")

    References:
        IEEE 181-2011 Section 5.4
    """
    rising_edges = _find_edges(trace, "rising", ref_level)
    falling_edges = _find_edges(trace, "falling", ref_level)

    if len(rising_edges) == 0 or len(falling_edges) == 0:
        if return_all:
            return np.array([], dtype=np.float64)
        return np.nan

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
        return np.nan

    widths_arr = np.array(widths, dtype=np.float64)

    if return_all:
        return widths_arr
    return float(np.mean(widths_arr))


def overshoot(trace: WaveformTrace) -> float | np_floating[Any]:
    """Measure overshoot percentage.

    Computes overshoot as (max - high) / amplitude * 100%.

    Args:
        trace: Input waveform trace.

    Returns:
        Overshoot as percentage, or np.nan if not applicable.

    Example:
        >>> os = overshoot(trace)
        >>> print(f"Overshoot: {os:.1f}%")

    References:
        IEEE 181-2011 Section 5.5
    """
    if len(trace.data) < 3:
        return np.nan

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return np.nan

    max_val = np.max(data)

    if max_val <= high:
        return 0.0

    return float((max_val - high) / amplitude * 100)


def undershoot(trace: WaveformTrace) -> float | np_floating[Any]:
    """Measure undershoot percentage.

    Computes undershoot as (low - min) / amplitude * 100%.

    Args:
        trace: Input waveform trace.

    Returns:
        Undershoot as percentage, or np.nan if not applicable.

    Example:
        >>> us = undershoot(trace)
        >>> print(f"Undershoot: {us:.1f}%")

    References:
        IEEE 181-2011 Section 5.5
    """
    if len(trace.data) < 3:
        return np.nan

    data = trace.data
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return np.nan

    min_val = np.min(data)

    if min_val >= low:
        return 0.0

    return float((low - min_val) / amplitude * 100)


def preshoot(
    trace: WaveformTrace,
    *,
    edge_type: Literal["rising", "falling"] = "rising",
) -> float | np_floating[Any]:
    """Measure preshoot percentage.

    Computes preshoot before transitions as percentage of amplitude.

    Args:
        trace: Input waveform trace.
        edge_type: Type of edge to analyze ("rising" or "falling").

    Returns:
        Preshoot as percentage, or np.nan if not applicable.

    Example:
        >>> ps = preshoot(trace)
        >>> print(f"Preshoot: {ps:.1f}%")

    References:
        IEEE 181-2011 Section 5.5
    """
    if len(trace.data) < 10:
        return np.nan

    # Convert memoryview to ndarray if needed
    data = np.asarray(trace.data)
    low, high = _find_levels(data)
    amplitude = high - low

    if amplitude <= 0:
        return np.nan

    # Find edge crossings at 50%
    mid = (low + high) / 2

    if edge_type == "rising":
        # Look for minimum before rising edge that goes below low level
        crossings = np.where((data[:-1] < mid) & (data[1:] >= mid))[0]
        if len(crossings) == 0:
            return np.nan

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

        return max_preshoot

    else:  # falling
        crossings = np.where((data[:-1] >= mid) & (data[1:] < mid))[0]
        if len(crossings) == 0:
            return np.nan

        max_preshoot = 0.0
        for idx in crossings:
            pre_samples = max(0, idx - 10)
            pre_region = data[pre_samples:idx]
            if len(pre_region) > 0:
                max_pre = np.max(pre_region)
                if max_pre > high:
                    preshoot_val = (max_pre - high) / amplitude * 100
                    max_preshoot = max(max_preshoot, preshoot_val)

        return max_preshoot


def amplitude(trace: WaveformTrace) -> float | np_floating[Any]:
    """Measure peak-to-peak amplitude.

    Computes Vpp as the difference between histogram-based high and low levels.

    Args:
        trace: Input waveform trace.

    Returns:
        Amplitude in volts (or input units).

    Example:
        >>> vpp = amplitude(trace)
        >>> print(f"Amplitude: {vpp:.3f} V")

    References:
        IEEE 1057-2017 Section 4.2
    """
    if len(trace.data) < 2:
        return np.nan

    low, high = _find_levels(trace.data)
    return high - low


def rms(
    trace: WaveformTrace,
    *,
    ac_coupled: bool = False,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> float | np_floating[Any]:
    """Compute RMS voltage.

    Calculates root-mean-square voltage of the waveform.

    Args:
        trace: Input waveform trace.
        ac_coupled: If True, remove DC offset before computing RMS.
        nan_policy: How to handle NaN values:
            - "propagate": Return NaN if any NaN present (default, NumPy behavior)
            - "omit": Ignore NaN values in calculation
            - "raise": Raise ValueError if any NaN present

    Returns:
        RMS voltage in volts (or input units).

    Raises:
        ValueError: If nan_policy="raise" and data contains NaN.

    Example:
        >>> v_rms = rms(trace)
        >>> print(f"RMS: {v_rms:.3f} V")

        >>> # Handle traces with NaN values
        >>> v_rms = rms(trace, nan_policy="omit")


    References:
        IEEE 1057-2017 Section 4.3
    """
    if len(trace.data) == 0:
        return np.nan

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
        return float(np.sqrt(np.nanmean(data**2)))
    # else propagate - default NumPy behavior

    if ac_coupled:
        data = data - np.mean(data)

    return float(np.sqrt(np.mean(data**2)))


def mean(
    trace: WaveformTrace,
    *,
    nan_policy: Literal["propagate", "omit", "raise"] = "propagate",
) -> float | np_floating[Any]:
    """Compute mean (DC) voltage.

    Calculates arithmetic mean of the waveform.

    Args:
        trace: Input waveform trace.
        nan_policy: How to handle NaN values:
            - "propagate": Return NaN if any NaN present (default, NumPy behavior)
            - "omit": Ignore NaN values in calculation
            - "raise": Raise ValueError if any NaN present

    Returns:
        Mean voltage in volts (or input units).

    Raises:
        ValueError: If nan_policy="raise" and data contains NaN.

    Example:
        >>> v_dc = mean(trace)
        >>> print(f"DC: {v_dc:.3f} V")

        >>> # Handle traces with NaN values
        >>> v_dc = mean(trace, nan_policy="omit")


    References:
        IEEE 1057-2017 Section 4.3
    """
    if len(trace.data) == 0:
        return np.nan

    # Convert memoryview to ndarray if needed
    data = np.asarray(trace.data)

    # Handle NaN based on policy
    if nan_policy == "raise":
        if np.any(np.isnan(data)):
            raise ValueError("Input data contains NaN values")
        return float(np.mean(data))
    elif nan_policy == "omit":
        return float(np.nanmean(data))
    else:  # propagate
        return float(np.mean(data))


def measure(
    trace: WaveformTrace,
    *,
    parameters: list[str] | None = None,
    include_units: bool = True,
) -> dict[str, Any]:
    """Compute multiple waveform measurements.

    Unified function for computing all or selected waveform measurements.

    Args:
        trace: Input waveform trace.
        parameters: List of measurement names to compute. If None, compute all.
            Valid names: rise_time, fall_time, period, frequency, duty_cycle,
            amplitude, rms, mean, overshoot, undershoot, preshoot
        include_units: If True, include units in output.

    Returns:
        Dictionary mapping measurement names to values (and units if requested).

    Example:
        >>> results = measure(trace)
        >>> print(f"Rise time: {results['rise_time']['value']} {results['rise_time']['unit']}")

        >>> results = measure(trace, parameters=["frequency", "amplitude"])

    References:
        IEEE 181-2011, IEEE 1057-2017
    """
    all_measurements = {
        "rise_time": (rise_time, "s"),
        "fall_time": (fall_time, "s"),
        "period": (lambda t: period(t, return_all=False), "s"),
        "frequency": (frequency, "Hz"),
        "duty_cycle": (lambda t: duty_cycle(t, percentage=True), "%"),
        "pulse_width_pos": (
            lambda t: pulse_width(t, polarity="positive", return_all=False),
            "s",
        ),
        "pulse_width_neg": (
            lambda t: pulse_width(t, polarity="negative", return_all=False),
            "s",
        ),
        "amplitude": (amplitude, "V"),
        "rms": (rms, "V"),
        "mean": (mean, "V"),
        "overshoot": (overshoot, "%"),
        "undershoot": (undershoot, "%"),
        "preshoot": (preshoot, "%"),
    }

    if parameters is None:
        selected = all_measurements
    else:
        selected = {k: v for k, v in all_measurements.items() if k in parameters}

    results: dict[str, Any] = {}

    for name, (func, unit) in selected.items():
        try:
            value = func(trace)  # type: ignore[operator]
        except Exception:
            value = np.nan

        if include_units:
            results[name] = {"value": value, "unit": unit}
        else:
            results[name] = value

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
    sample_period = trace.metadata.time_base

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
