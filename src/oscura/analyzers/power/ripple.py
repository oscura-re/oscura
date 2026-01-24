"""Ripple measurement for Oscura.

Provides AC ripple analysis for DC power supply outputs.


Example:
    >>> from oscura.analyzers.power.ripple import ripple, ripple_statistics
    >>> r_pp, r_rms = ripple(dc_output_trace)
    >>> print(f"Ripple: {r_pp*1e3:.2f} mV pp, {r_rms*1e3:.2f} mV rms")
"""

import numpy as np
from scipy import signal

from oscura.core.exceptions import AnalysisError
from oscura.core.types import WaveformTrace


def ripple(
    trace: WaveformTrace,
    *,
    dc_coupling: bool = False,
) -> tuple[float, float]:
    """Measure AC ripple on a DC signal.

    Isolates the AC component from the DC offset and measures
    peak-to-peak and RMS ripple.

    Args:
        trace: DC voltage/current waveform with AC ripple.
        dc_coupling: If True, include DC component in measurement.
                    If False (default), remove DC for pure AC ripple.

    Returns:
        Tuple of (ripple_pp, ripple_rms) in signal units.

    Example:
        >>> vpp, vrms = ripple(output_voltage)
        >>> print(f"Ripple: {vpp*1e3:.2f} mV pp, {vrms*1e3:.2f} mV rms")

    References:
        IEC 61000-4-7 (power quality)
    """
    data = trace.data

    if dc_coupling:
        ac_component = data
    else:
        # Remove DC (mean)
        ac_component = data - np.mean(data)

    ripple_pp = float(np.max(ac_component) - np.min(ac_component))
    ripple_rms = float(np.sqrt(np.mean(ac_component**2)))

    return ripple_pp, ripple_rms


def ripple_percentage(
    trace: WaveformTrace,
) -> tuple[float, float]:
    """Measure ripple as percentage of DC level.

    Args:
        trace: DC voltage/current waveform with AC ripple.

    Returns:
        Tuple of (ripple_pp_percent, ripple_rms_percent).

    Example:
        >>> pp_pct, rms_pct = ripple_percentage(output_voltage)
        >>> print(f"Ripple: {pp_pct:.2f}% pp, {rms_pct:.2f}% rms")
    """
    dc_level = float(np.mean(trace.data))

    if dc_level == 0:
        return np.nan, np.nan

    r_pp, r_rms = ripple(trace)

    return (r_pp / dc_level * 100, r_rms / dc_level * 100)


def ripple_frequency(
    trace: WaveformTrace,
    *,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
) -> float:
    """Find dominant ripple frequency.

    Args:
        trace: DC voltage waveform with AC ripple.
        min_frequency: Minimum frequency to consider (Hz).
        max_frequency: Maximum frequency to consider (Hz).

    Returns:
        Dominant ripple frequency in Hz.

    Example:
        >>> f_ripple = ripple_frequency(output_voltage)
        >>> print(f"Ripple frequency: {f_ripple/1e3:.2f} kHz")
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Remove DC
    ac_data = data - np.mean(data)

    # FFT
    n = len(ac_data)
    fft_result = np.abs(np.fft.rfft(ac_data))
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)

    # Apply frequency limits
    freq_mask = np.ones(len(freqs), dtype=bool)
    if min_frequency is not None:
        freq_mask &= freqs >= min_frequency
    if max_frequency is not None:
        freq_mask &= freqs <= max_frequency

    # Exclude DC
    freq_mask[0] = False

    if not np.any(freq_mask):
        return 0.0

    # Find peak
    masked_fft = fft_result.copy()
    masked_fft[~freq_mask] = 0
    peak_idx = np.argmax(masked_fft)

    return float(freqs[peak_idx])


def ripple_harmonics(
    trace: WaveformTrace,
    fundamental_freq: float | None = None,
    n_harmonics: int = 10,
) -> dict[int, float]:
    """Analyze ripple harmonics.

    Args:
        trace: DC voltage waveform with AC ripple.
        fundamental_freq: Fundamental ripple frequency. If None, auto-detect.
        n_harmonics: Number of harmonics to analyze.

    Returns:
        Dictionary mapping harmonic number to amplitude.

    Example:
        >>> harmonics = ripple_harmonics(output_voltage)
        >>> for h, amp in harmonics.items():
        ...     print(f"H{h}: {amp*1e3:.2f} mV")
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Remove DC
    ac_data = data - np.mean(data)

    # Find fundamental if not provided
    if fundamental_freq is None:
        fundamental_freq = ripple_frequency(trace)

    if fundamental_freq <= 0:
        return {}

    # FFT
    n = len(ac_data)
    fft_result = np.abs(np.fft.rfft(ac_data)) * 2 / n  # Scale for amplitude
    freqs = np.fft.rfftfreq(n, 1 / sample_rate)

    harmonics = {}
    for h in range(1, n_harmonics + 1):
        target_freq = h * fundamental_freq
        # Find closest bin
        idx = np.argmin(np.abs(freqs - target_freq))
        if idx < len(fft_result):
            harmonics[h] = float(fft_result[idx])

    return harmonics


def ripple_statistics(
    trace: WaveformTrace,
) -> dict[str, float]:
    """Calculate comprehensive ripple statistics.

    Args:
        trace: DC voltage waveform with AC ripple.

    Returns:
        Dictionary with:
        - dc_level: DC (mean) level
        - ripple_pp: Peak-to-peak ripple
        - ripple_rms: RMS ripple
        - ripple_pp_percent: Peak-to-peak as % of DC
        - ripple_rms_percent: RMS as % of DC
        - ripple_frequency: Dominant ripple frequency
        - crest_factor: Ripple peak / ripple RMS

    Example:
        >>> stats = ripple_statistics(output_voltage)
        >>> print(f"DC: {stats['dc_level']:.2f} V")
        >>> print(f"Ripple: {stats['ripple_pp']*1e3:.2f} mV pp")
    """
    data = trace.data
    dc_level = float(np.mean(data))
    ac_data = data - dc_level

    r_pp = float(np.max(ac_data) - np.min(ac_data))
    r_rms = float(np.sqrt(np.mean(ac_data**2)))
    r_peak = float(np.max(np.abs(ac_data)))

    crest_factor = r_peak / r_rms if r_rms > 0 else 0.0

    if dc_level != 0:
        r_pp_pct = r_pp / dc_level * 100
        r_rms_pct = r_rms / dc_level * 100
    else:
        r_pp_pct = np.nan
        r_rms_pct = np.nan

    return {
        "dc_level": dc_level,
        "ripple_pp": r_pp,
        "ripple_rms": r_rms,
        "ripple_pp_percent": r_pp_pct,
        "ripple_rms_percent": r_rms_pct,
        "ripple_frequency": ripple_frequency(trace),
        "crest_factor": crest_factor,
    }


def extract_ripple(
    trace: WaveformTrace,
    *,
    high_pass_freq: float | None = None,
    filter_order: int = 4,
) -> WaveformTrace:
    """Extract AC ripple component from DC signal.

    Args:
        trace: DC voltage waveform with AC ripple.
        high_pass_freq: High-pass filter cutoff. If None, uses simple DC removal.
        filter_order: Order of the Butterworth high-pass filter (default: 4).
            Higher orders give sharper cutoff but more phase distortion.

    Returns:
        Waveform trace containing only the AC ripple component.

    Raises:
        AnalysisError: If high_pass_freq exceeds Nyquist frequency

    Example:
        >>> ac_ripple = extract_ripple(output_voltage)
        >>> # Now analyze or plot just the ripple
        >>> # With custom filter order for sharper cutoff:
        >>> ac_ripple = extract_ripple(output_voltage, high_pass_freq=10, filter_order=6)
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    if high_pass_freq is not None:
        # Use high-pass filter
        nyquist = sample_rate / 2
        if high_pass_freq >= nyquist:
            raise AnalysisError(
                f"High-pass frequency {high_pass_freq} Hz must be less than "
                f"Nyquist frequency {nyquist} Hz"
            )

        sos = signal.butter(filter_order, high_pass_freq / nyquist, btype="high", output="sos")
        ac_data = signal.sosfiltfilt(sos, data)
    else:
        # Simple DC removal
        ac_data = data - np.mean(data)

    return WaveformTrace(
        data=ac_data.astype(np.float64),
        metadata=trace.metadata,
    )


def ripple_envelope(
    trace: WaveformTrace,
    *,
    method: str = "hilbert",
    peak_window_size: int | None = None,
) -> WaveformTrace:
    """Extract ripple envelope (for amplitude modulation analysis).

    Args:
        trace: DC voltage waveform with AC ripple.
        method: Envelope detection method ("hilbert" or "peak").
        peak_window_size: Window size for peak envelope detection (samples).
            If None, defaults to a size that covers approximately one ripple period.
            Only used when method="peak".

    Returns:
        Waveform trace containing the ripple envelope.

    Raises:
        AnalysisError: If unknown envelope method specified

    Example:
        >>> envelope = ripple_envelope(output_voltage)
        >>> # Analyze envelope for beat frequencies, etc.
        >>> # With custom peak window size:
        >>> envelope = ripple_envelope(output_voltage, method="peak", peak_window_size=200)
    """
    # First extract AC component
    ac_trace = extract_ripple(trace)
    ac_data = ac_trace.data

    if method == "hilbert":
        analytic_signal = signal.hilbert(ac_data)
        envelope = np.abs(analytic_signal)
    elif method == "peak":
        # Simple peak detection
        from scipy.ndimage import maximum_filter1d

        # Determine window size
        if peak_window_size is None:
            # Default: scale to signal frequency if possible
            # Try to detect ripple frequency and use ~1 period
            ripple_freq = ripple_frequency(trace)
            sample_rate = trace.metadata.sample_rate
            if ripple_freq > 0:
                # Use approximately one period of the ripple
                peak_window_size = max(10, int(sample_rate / ripple_freq))
            else:
                # Fallback: use 1% of signal length, min 10, max 1000
                peak_window_size = max(10, min(1000, len(ac_data) // 100))

        envelope = maximum_filter1d(np.abs(ac_data), size=peak_window_size)
    else:
        raise AnalysisError(f"Unknown envelope method: {method}")

    return WaveformTrace(
        data=envelope.astype(np.float64),
        metadata=trace.metadata,
    )


__all__ = [
    "extract_ripple",
    "ripple",
    "ripple_envelope",
    "ripple_frequency",
    "ripple_harmonics",
    "ripple_percentage",
    "ripple_statistics",
]
