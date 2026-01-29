"""Basic power analysis for Oscura.

Provides fundamental power calculations including instantaneous power,
average power, RMS power, peak power, and energy.


Example:
    >>> from oscura.analyzers.power.basic import instantaneous_power, power_statistics
    >>> power_trace = instantaneous_power(voltage_trace, current_trace)
    >>> stats = power_statistics(power_trace)
    >>> print(f"Average: {stats['average']:.2f} W, Peak: {stats['peak']:.2f} W")
"""

from typing import Any

import numpy as np
from scipy import interpolate

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace


def instantaneous_power(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    interpolate_if_needed: bool = True,
) -> WaveformTrace:
    """Calculate instantaneous power from voltage and current traces.

    P(t) = V(t) * I(t)

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.
        interpolate_if_needed: If True, interpolate if sample rates differ.

    Returns:
        Power waveform trace (in Watts if inputs are V and A).

    Raises:
        AnalysisError: If sample rates mismatch and interpolation disabled.

    Example:
        >>> power = instantaneous_power(v_trace, i_trace)
        >>> print(f"Peak power: {np.max(power.data):.2f} W")

    References:
        IEC 61000-4-7 (power measurements)
    """
    v_data = voltage.data
    i_data = current.data
    v_rate = voltage.metadata.sample_rate
    i_rate = current.metadata.sample_rate

    # Handle different sample rates
    if v_rate != i_rate:
        if not interpolate_if_needed:
            raise AnalysisError(
                f"Sample rate mismatch: voltage={v_rate}, current={i_rate}. "
                "Set interpolate_if_needed=True to interpolate."
            )

        # Use higher sample rate and interpolate the other
        if v_rate > i_rate:
            # Interpolate current to voltage sample rate
            t_current = np.arange(len(i_data)) / i_rate
            t_voltage = np.arange(len(v_data)) / v_rate
            interp = interpolate.interp1d(
                t_current, i_data, kind="linear", fill_value="extrapolate"
            )
            i_data = interp(t_voltage)
            sample_rate = v_rate
        else:
            # Interpolate voltage to current sample rate
            t_voltage = np.arange(len(v_data)) / v_rate
            t_current = np.arange(len(i_data)) / i_rate
            interp = interpolate.interp1d(
                t_voltage, v_data, kind="linear", fill_value="extrapolate"
            )
            v_data = interp(t_current)
            sample_rate = i_rate
    else:
        sample_rate = v_rate

    # Handle different lengths
    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    # Calculate instantaneous power
    power_data = v_data * i_data

    return WaveformTrace(
        data=power_data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


def average_power(
    power: WaveformTrace | None = None,
    *,
    voltage: WaveformTrace | None = None,
    current: WaveformTrace | None = None,
) -> float:
    """Calculate average (mean) power.

    P_avg = (1/T) * integral(P(t) dt)

    Args:
        power: Power trace (if already calculated).
        voltage: Voltage trace (alternative to power).
        current: Current trace (alternative to power).

    Returns:
        Average power in Watts.

    Raises:
        AnalysisError: If neither power nor both voltage and current provided.

    Example:
        >>> p_avg = average_power(power_trace)
        >>> # or
        >>> p_avg = average_power(voltage=v, current=i)
    """
    if power is None:
        if voltage is None or current is None:
            raise AnalysisError("Either power trace or both voltage and current traces required")
        power = instantaneous_power(voltage, current)

    return float(np.mean(power.data))


def rms_power(
    power: WaveformTrace | None = None,
    *,
    voltage: WaveformTrace | None = None,
    current: WaveformTrace | None = None,
) -> float:
    """Calculate RMS power.

    P_rms = sqrt(mean(P(t)^2))

    Args:
        power: Power trace.
        voltage: Voltage trace (alternative).
        current: Current trace (alternative).

    Returns:
        RMS power in Watts.

    Raises:
        AnalysisError: If neither power nor both voltage and current provided.

    Example:
        >>> p_rms = rms_power(power_trace)
    """
    if power is None:
        if voltage is None or current is None:
            raise AnalysisError("Either power trace or both voltage and current traces required")
        power = instantaneous_power(voltage, current)

    return float(np.sqrt(np.mean(power.data**2)))


def peak_power(
    power: WaveformTrace | None = None,
    *,
    voltage: WaveformTrace | None = None,
    current: WaveformTrace | None = None,
    absolute: bool = True,
) -> float:
    """Calculate peak power.

    Args:
        power: Power trace.
        voltage: Voltage trace (alternative).
        current: Current trace (alternative).
        absolute: If True, return absolute maximum. If False, maximum value.

    Returns:
        Peak power in Watts.

    Raises:
        AnalysisError: If neither power nor both voltage and current provided.

    Example:
        >>> p_peak = peak_power(power_trace)
    """
    if power is None:
        if voltage is None or current is None:
            raise AnalysisError("Either power trace or both voltage and current traces required")
        power = instantaneous_power(voltage, current)

    if absolute:
        return float(np.max(np.abs(power.data)))
    return float(np.max(power.data))


def energy(
    power: WaveformTrace | None = None,
    *,
    voltage: WaveformTrace | None = None,
    current: WaveformTrace | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> float:
    """Calculate energy (integral of power over time).

    E = integral(P(t) dt)

    Args:
        power: Power trace.
        voltage: Voltage trace (alternative).
        current: Current trace (alternative).
        start_time: Start time for integration (seconds).
        end_time: End time for integration (seconds).

    Returns:
        Energy in Joules.

    Raises:
        AnalysisError: If neither power nor both voltage and current provided.

    Example:
        >>> e = energy(power_trace)
        >>> print(f"Total energy: {e*1e3:.2f} mJ")
    """
    if power is None:
        if voltage is None or current is None:
            raise AnalysisError("Either power trace or both voltage and current traces required")
        power = instantaneous_power(voltage, current)

    data = power.data
    sample_period = power.metadata.time_base

    # Apply time limits
    if start_time is not None or end_time is not None:
        time_vector = np.arange(len(data)) * sample_period
        if start_time is not None:
            mask = time_vector >= start_time
        else:
            mask = np.ones(len(data), dtype=bool)
        if end_time is not None:
            mask &= time_vector <= end_time
        data = data[mask]

    # Integrate using trapezoidal rule (scipy is stable across versions)
    from scipy.integrate import trapezoid

    return float(trapezoid(data, dx=sample_period))


def power_statistics(
    power: WaveformTrace | None = None,
    *,
    voltage: WaveformTrace | None = None,
    current: WaveformTrace | None = None,
) -> dict[str, float]:
    """Calculate comprehensive power statistics.

    Args:
        power: Power trace.
        voltage: Voltage trace (alternative).
        current: Current trace (alternative).

    Returns:
        Dictionary with:
        - average: Mean power
        - rms: RMS power
        - peak: Peak power (absolute)
        - peak_positive: Maximum positive power
        - peak_negative: Maximum negative power (regeneration)
        - energy: Total energy
        - min: Minimum power value
        - std: Standard deviation

    Raises:
        AnalysisError: If neither power nor both voltage and current provided.

    Example:
        >>> stats = power_statistics(voltage=v, current=i)
        >>> print(f"Average: {stats['average']:.2f} W")
        >>> print(f"Peak: {stats['peak']:.2f} W")
        >>> print(f"Energy: {stats['energy']*1e3:.2f} mJ")
    """
    if power is None:
        if voltage is None or current is None:
            raise AnalysisError("Either power trace or both voltage and current traces required")
        power = instantaneous_power(voltage, current)

    data = power.data
    sample_period = power.metadata.time_base

    # Use scipy trapezoid for stable API across NumPy versions
    from scipy.integrate import trapezoid

    return {
        "average": float(np.mean(data)),
        "rms": float(np.sqrt(np.mean(data**2))),
        "peak": float(np.max(np.abs(data))),
        "peak_positive": float(np.max(data)),
        "peak_negative": float(np.min(data)),
        "energy": float(trapezoid(data, dx=sample_period)),
        "min": float(np.min(data)),
        "std": float(np.std(data)),
    }


def power_profile(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    window_size: int | None = None,
) -> dict[str, Any]:
    """Generate power profile with rolling statistics.

    Args:
        voltage: Voltage trace.
        current: Current trace.
        window_size: Window size for rolling calculations. If None, auto-select.

    Returns:
        Dictionary with:
        - power_trace: Instantaneous power trace
        - rolling_avg: Rolling average power
        - rolling_peak: Rolling peak power
        - cumulative_energy: Cumulative energy over time
        - statistics: Overall power statistics
    """
    power = instantaneous_power(voltage, current)
    data = power.data
    sample_period = power.metadata.time_base

    if window_size is None:
        # Auto-select: ~1% of total samples or 100, whichever is larger
        window_size = max(100, len(data) // 100)

    # Ensure window size is odd for centered window
    if window_size % 2 == 0:
        window_size += 1

    # Rolling average
    kernel = np.ones(window_size) / window_size
    rolling_avg = np.convolve(data, kernel, mode="same")

    # Rolling peak (using scipy.ndimage.maximum_filter would be faster but numpy-only)
    from scipy.ndimage import maximum_filter1d

    rolling_peak = maximum_filter1d(np.abs(data), size=window_size)

    # Cumulative energy
    cumulative_energy = np.cumsum(data) * sample_period

    return {
        "power_trace": power,
        "rolling_avg": WaveformTrace(
            data=rolling_avg.astype(np.float64),
            metadata=power.metadata,
        ),
        "rolling_peak": WaveformTrace(
            data=rolling_peak.astype(np.float64),
            metadata=power.metadata,
        ),
        "cumulative_energy": WaveformTrace(
            data=cumulative_energy.astype(np.float64),
            metadata=power.metadata,
        ),
        "statistics": power_statistics(power),
    }


__all__ = [
    "average_power",
    "energy",
    "instantaneous_power",
    "peak_power",
    "power_profile",
    "power_statistics",
    "rms_power",
]
