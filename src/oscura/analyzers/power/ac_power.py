"""AC power analysis for Oscura.

Provides AC power calculations including reactive power, apparent power,
power factor, and harmonic analysis.


Example:
    >>> from oscura.analyzers.power.ac_power import power_factor, reactive_power
    >>> pf = power_factor(voltage_trace, current_trace)
    >>> q = reactive_power(voltage_trace, current_trace)
    >>> print(f"Power factor: {pf:.3f}, Reactive power: {q:.2f} VAR")
"""

import numpy as np

from oscura.analyzers.power.basic import average_power
from oscura.core.types import WaveformTrace


def phase_angle(
    voltage: WaveformTrace,
    current: WaveformTrace,
) -> float:
    """Calculate phase angle between voltage and current.

    Uses cross-correlation to determine the phase shift.

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.

    Returns:
        Phase angle in radians (positive = current lags voltage).

    Example:
        >>> phi = phase_angle(v_trace, i_trace)
        >>> print(f"Phase angle: {np.degrees(phi):.1f} degrees")
    """
    v_data = voltage.data
    i_data = current.data

    # Ensure same length
    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    # Remove DC offset
    v_ac = v_data - np.mean(v_data)
    i_ac = i_data - np.mean(i_data)

    # Cross-correlation to find phase shift
    # correlate(v, i) tells us how much to shift i to align with v
    # A positive lag means i needs to be shifted right (i lags v)
    correlation = np.correlate(v_ac, i_ac, mode="full")
    lags = np.arange(-len(i_ac) + 1, len(v_ac))

    # Find peak correlation
    peak_idx = np.argmax(np.abs(correlation))
    lag_samples = lags[peak_idx]

    # Convert lag to phase angle
    # Estimate frequency using zero crossings
    v_crossings = np.where(np.diff(np.signbit(v_ac)))[0]
    if len(v_crossings) >= 2:
        period_samples = 2 * np.mean(np.diff(v_crossings))
        # Negative lag because correlation gives us how much to shift i to align with v
        # If lag is negative, i is ahead of v (capacitive), phase should be negative
        # If lag is positive, i is behind v (inductive), phase should be positive
        phase = -2 * np.pi * lag_samples / period_samples
    else:
        # Fallback: use FFT to find fundamental frequency
        fft_v = np.fft.rfft(v_ac)
        freq_idx = np.argmax(np.abs(fft_v[1:])) + 1
        period_samples = len(v_ac) / freq_idx
        phase = -2 * np.pi * lag_samples / period_samples

    return float(phase)


def reactive_power(
    voltage: WaveformTrace,
    current: WaveformTrace,
    *,
    frequency: float | None = None,
) -> float:
    """Calculate reactive power (Q) in VAR.

    Q = V_rms * I_rms * sin(phi)

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.
        frequency: Fundamental frequency in Hz. If None, auto-detect.

    Returns:
        Reactive power in VAR (positive = inductive, negative = capacitive).

    Example:
        >>> q = reactive_power(v_trace, i_trace)
        >>> print(f"Reactive power: {q:.2f} VAR")

    References:
        IEEE 1459-2010 (Power quality definitions)
    """
    v_data = voltage.data
    i_data = current.data

    # Ensure same length
    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    # Calculate RMS values
    v_rms = float(np.sqrt(np.mean(v_data**2)))
    i_rms = float(np.sqrt(np.mean(i_data**2)))

    # Calculate phase angle
    phi = phase_angle(voltage, current)

    return v_rms * i_rms * np.sin(phi)  # type: ignore[no-any-return]


def apparent_power(
    voltage: WaveformTrace,
    current: WaveformTrace,
) -> float:
    """Calculate apparent power (S) in VA.

    S = V_rms * I_rms

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.

    Returns:
        Apparent power in VA.

    Example:
        >>> s = apparent_power(v_trace, i_trace)
        >>> print(f"Apparent power: {s:.2f} VA")

    References:
        IEEE 1459-2010 (Power quality definitions)
    """
    v_data = voltage.data
    i_data = current.data

    # Ensure same length
    min_len = min(len(v_data), len(i_data))
    v_data = v_data[:min_len]
    i_data = i_data[:min_len]

    v_rms = float(np.sqrt(np.mean(v_data**2)))
    i_rms = float(np.sqrt(np.mean(i_data**2)))

    return v_rms * i_rms


def power_factor(
    voltage: WaveformTrace,
    current: WaveformTrace,
) -> float:
    """Calculate power factor (PF = P / S).

    For sinusoidal waveforms, PF = cos(phi).
    For non-sinusoidal waveforms, includes distortion effects.

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.

    Returns:
        Power factor (0 to 1, can be negative for regeneration).

    Example:
        >>> pf = power_factor(v_trace, i_trace)
        >>> print(f"Power factor: {pf:.3f}")

    References:
        IEEE 1459-2010
    """
    p = average_power(voltage=voltage, current=current)
    s = apparent_power(voltage, current)

    if s == 0:
        return 0.0

    return p / s


def displacement_power_factor(
    voltage: WaveformTrace,
    current: WaveformTrace,
) -> float:
    """Calculate displacement power factor (DPF).

    DPF = cos(phi) where phi is the phase angle between fundamental
    components of voltage and current.

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.

    Returns:
        Displacement power factor.

    Example:
        >>> dpf = displacement_power_factor(v_trace, i_trace)
    """
    # Extract fundamental components
    v_fund = _extract_fundamental(voltage)
    i_fund = _extract_fundamental(current)

    # Calculate phase angle of fundamentals
    phi = phase_angle(v_fund, i_fund)

    return float(np.cos(phi))


def distortion_power_factor(
    voltage: WaveformTrace,
    current: WaveformTrace,
) -> float:
    """Calculate distortion power factor.

    Distortion PF = True PF / Displacement PF
    = sqrt(1 / (1 + THD_i^2))  (for sinusoidal voltage)

    Args:
        voltage: Voltage waveform trace.
        current: Current waveform trace.

    Returns:
        Distortion power factor.

    Example:
        >>> dist_pf = distortion_power_factor(v_trace, i_trace)
    """
    pf = power_factor(voltage, current)
    dpf = displacement_power_factor(voltage, current)

    if dpf == 0:
        return 0.0

    return pf / dpf


def total_harmonic_distortion_power(
    trace: WaveformTrace,
    fundamental_freq: float | None = None,
    max_harmonic: int = 50,
) -> float:
    """Calculate Total Harmonic Distortion for power analysis.

    THD = sqrt(sum(V_h^2 for h=2..N)) / V_1

    Args:
        trace: Voltage or current waveform.
        fundamental_freq: Fundamental frequency. If None, auto-detect.
        max_harmonic: Maximum harmonic order to include.

    Returns:
        THD as a ratio (not percentage).

    Example:
        >>> thd = total_harmonic_distortion_power(current_trace)
        >>> print(f"Current THD: {thd*100:.1f}%")
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # FFT
    n = len(data)
    fft_result = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)

    # Find fundamental
    if fundamental_freq is None:
        # Auto-detect: find largest peak above DC
        magnitudes = np.abs(fft_result[1:])
        fund_idx = int(np.argmax(magnitudes)) + 1
        fundamental_freq = freqs[fund_idx]
    else:
        fund_idx = int(np.round(fundamental_freq * n / sample_rate))

    # Get fundamental magnitude
    v1 = np.abs(fft_result[fund_idx])
    if v1 == 0:
        return 0.0

    # Sum harmonic magnitudes
    harmonic_sum_sq = 0.0
    for h in range(2, max_harmonic + 1):
        h_idx = h * fund_idx
        if h_idx < len(fft_result):
            harmonic_sum_sq += np.abs(fft_result[h_idx]) ** 2

    return float(np.sqrt(harmonic_sum_sq) / v1)


def _extract_fundamental(trace: WaveformTrace) -> WaveformTrace:
    """Extract fundamental component of a waveform."""
    data = trace.data
    n = len(data)

    # FFT
    fft_result = np.fft.rfft(data)

    # Find fundamental peak
    magnitudes = np.abs(fft_result[1:])
    fund_idx = np.argmax(magnitudes) + 1

    # Zero out all but fundamental
    filtered_fft = np.zeros_like(fft_result)
    filtered_fft[fund_idx] = fft_result[fund_idx]

    # Inverse FFT
    fundamental = np.fft.irfft(filtered_fft, n=n)

    return WaveformTrace(
        data=fundamental.astype(np.float64),
        metadata=trace.metadata,
    )


def three_phase_power(
    v_a: WaveformTrace,
    v_b: WaveformTrace,
    v_c: WaveformTrace,
    i_a: WaveformTrace,
    i_b: WaveformTrace,
    i_c: WaveformTrace,
) -> dict[str, float]:
    """Calculate three-phase power quantities.

    Args:
        v_a: Phase A voltage trace.
        v_b: Phase B voltage trace.
        v_c: Phase C voltage trace.
        i_a: Phase A current trace.
        i_b: Phase B current trace.
        i_c: Phase C current trace.

    Returns:
        Dictionary with:
        - total_active: Total active power (W)
        - total_reactive: Total reactive power (VAR)
        - total_apparent: Total apparent power (VA)
        - power_factor: Three-phase power factor
        - phase_a_power: Phase A active power
        - phase_b_power: Phase B active power
        - phase_c_power: Phase C active power
    """
    # Calculate per-phase powers
    p_a = average_power(voltage=v_a, current=i_a)
    p_b = average_power(voltage=v_b, current=i_b)
    p_c = average_power(voltage=v_c, current=i_c)

    q_a = reactive_power(v_a, i_a)
    q_b = reactive_power(v_b, i_b)
    q_c = reactive_power(v_c, i_c)

    total_p = p_a + p_b + p_c
    total_q = q_a + q_b + q_c
    total_s = np.sqrt(total_p**2 + total_q**2)

    pf = total_p / total_s if total_s != 0 else 0.0

    return {
        "total_active": total_p,
        "total_reactive": total_q,
        "total_apparent": total_s,
        "power_factor": pf,
        "phase_a_power": p_a,
        "phase_b_power": p_b,
        "phase_c_power": p_c,
    }


__all__ = [
    "apparent_power",
    "displacement_power_factor",
    "distortion_power_factor",
    "phase_angle",
    "power_factor",
    "reactive_power",
    "three_phase_power",
    "total_harmonic_distortion_power",
]
