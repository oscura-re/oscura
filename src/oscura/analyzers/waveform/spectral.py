"""Spectral analysis functions for waveform data.

This module provides FFT, PSD, and spectral quality metrics
per IEEE 1241-2010 for ADC characterization.


Example:
    >>> from oscura.analyzers.waveform.spectral import fft, thd, snr
    >>> freq, magnitude = fft(trace)
    >>> thd_db = thd(trace)
    >>> snr_db = snr(trace)

References:
    IEEE 1241-2010: Standard for Terminology and Test Methods for
    Analog-to-Digital Converters
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy import signal as sp_signal

from oscura.core.exceptions import AnalysisError, InsufficientDataError
from oscura.utils.windowing import get_window

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace

# Global FFT cache statistics
_fft_cache_stats = {"hits": 0, "misses": 0, "size": 128}


def get_fft_cache_stats() -> dict[str, int]:
    """Get FFT cache statistics.

    Returns:
        Dictionary with cache hits, misses, and configured size.

    Example:
        >>> stats = get_fft_cache_stats()
        >>> print(f"Cache hit rate: {stats['hits'] / (stats['hits'] + stats['misses']):.1%}")
    """
    return _fft_cache_stats.copy()


def clear_fft_cache() -> None:
    """Clear the FFT result cache.

    Useful for freeing memory or forcing recomputation.

    Example:
        >>> clear_fft_cache()  # Clear cached FFT results
    """
    _compute_fft_cached.cache_clear()
    _fft_cache_stats["hits"] = 0
    _fft_cache_stats["misses"] = 0


def configure_fft_cache(size: int) -> None:
    """Configure FFT cache size.

    Args:
        size: Maximum number of FFT results to cache (default 128).

    Example:
        >>> configure_fft_cache(256)  # Increase cache size for better hit rate
    """
    global _compute_fft_cached
    _fft_cache_stats["size"] = size
    # Recreate cache with new size
    _compute_fft_cached = lru_cache(maxsize=size)(_compute_fft_impl)
    _fft_cache_stats["hits"] = 0
    _fft_cache_stats["misses"] = 0


def _compute_fft_impl(
    data_bytes: bytes,
    n: int,
    window: str,
    nfft: int,
    detrend_method: str,
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Internal cached FFT implementation.

    Args:
        data_bytes: Hash-friendly bytes representation of data.
        n: Number of samples.
        window: Window function name.
        nfft: FFT length.
        detrend_method: Detrend method.
        sample_rate: Sample rate in Hz.

    Returns:
        (freq, magnitude_db, phase) tuple.
    """
    # Reconstruct data from bytes
    data = np.frombuffer(data_bytes, dtype=np.float64)

    # Apply window
    w = get_window(window, n)
    data_windowed = data * w

    # Compute FFT
    spectrum = np.fft.rfft(data_windowed, n=nfft)

    # Frequency axis
    freq = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

    # Magnitude in dB (normalized by window coherent gain)
    window_gain = np.sum(w) / n
    magnitude = np.abs(spectrum) / (n * window_gain)
    # Avoid log(0)
    magnitude = np.maximum(magnitude, 1e-20)
    magnitude_db = 20 * np.log10(magnitude)

    # Phase
    phase = np.angle(spectrum)

    return freq, magnitude_db, phase


# Create cached version with default size
_compute_fft_cached = lru_cache(maxsize=128)(_compute_fft_impl)


def fft(
    trace: WaveformTrace,
    *,
    window: str = "hann",
    nfft: int | None = None,
    detrend: Literal["none", "mean", "linear"] = "mean",
    return_phase: bool = False,
    use_cache: bool = True,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """Compute windowed FFT with optional zero-padding and caching.

    Computes the single-sided magnitude spectrum in dB with optional
    phase output. Uses configurable windowing and zero-padding.
    Results are cached for repeated analysis of the same data.

    Args:
        trace: Input waveform trace.
        window: Window function name (default "hann").
        nfft: FFT length. If None, uses next power of 2.
        detrend: Detrend method before FFT:
            - "none": No detrending
            - "mean": Remove DC offset (default)
            - "linear": Remove linear trend
        return_phase: If True, also return phase in radians.
        use_cache: If True, cache FFT results for reuse (default True).

    Returns:
        If return_phase=False:
            (frequencies, magnitude_db) - Frequency axis and magnitude in dB
        If return_phase=True:
            (frequencies, magnitude_db, phase_rad) - Plus phase in radians

    Raises:
        InsufficientDataError: If trace has fewer than 2 samples.

    Example:
        >>> freq, mag = fft(trace)
        >>> plt.semilogx(freq, mag)
        >>> plt.xlabel("Frequency (Hz)")
        >>> plt.ylabel("Magnitude (dB)")
        >>> # Check cache performance
        >>> stats = get_fft_cache_stats()
        >>> print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")

    References:
        IEEE 1241-2010 Section 4.1.1
    """
    data = trace.data
    n = len(data)

    if n < 2:
        raise InsufficientDataError(
            "FFT requires at least 2 samples",
            required=2,
            available=n,
            analysis_type="fft",
        )

    data_processed = _apply_detrend(data, detrend)
    nfft_computed = _compute_nfft(n, nfft)
    sample_rate = trace.metadata.sample_rate

    if use_cache:
        return _fft_cached_path(
            data_processed, n, window, nfft_computed, detrend, sample_rate, return_phase
        )
    else:
        return _fft_direct_path(data_processed, n, window, nfft_computed, sample_rate, return_phase)


def _apply_detrend(
    data: NDArray[np.float64], detrend: Literal["none", "mean", "linear"]
) -> NDArray[np.float64]:
    """Apply detrending to data.

    Args:
        data: Input data.
        detrend: Detrend method.

    Returns:
        Detrended data.
    """
    if detrend == "mean":
        detrended: NDArray[np.float64] = data - np.mean(data)
        return detrended
    elif detrend == "linear":
        linear_detrend: NDArray[np.float64] = np.asarray(
            sp_signal.detrend(data, type="linear"), dtype=np.float64
        )
        return linear_detrend
    else:
        return data


def _compute_nfft(n: int, nfft: int | None) -> int:
    """Compute FFT length.

    Args:
        n: Data length.
        nfft: Requested FFT length or None.

    Returns:
        Computed FFT length (power of 2 or max of nfft and n).
    """
    return int(2 ** np.ceil(np.log2(n))) if nfft is None else max(nfft, n)


def _fft_cached_path(
    data_processed: NDArray[np.float64],
    n: int,
    window: str,
    nfft_computed: int,
    detrend: str,
    sample_rate: float,
    return_phase: bool,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """Execute cached FFT computation path.

    Args:
        data_processed: Preprocessed data.
        n: Data length.
        window: Window function name.
        nfft_computed: FFT length.
        detrend: Detrend method string.
        sample_rate: Sample rate.
        return_phase: Whether to return phase.

    Returns:
        FFT results (with or without phase).
    """
    data_bytes = data_processed.tobytes()
    freq, magnitude_db, phase = _compute_fft_cached(
        data_bytes, n, window, nfft_computed, detrend, sample_rate
    )
    _fft_cache_stats["hits"] += 1

    if return_phase:
        return freq, magnitude_db, phase
    else:
        return freq, magnitude_db


def _fft_direct_path(
    data_processed: NDArray[np.float64],
    n: int,
    window: str,
    nfft_computed: int,
    sample_rate: float,
    return_phase: bool,
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
):
    """Execute non-cached FFT computation path.

    Args:
        data_processed: Preprocessed data.
        n: Data length.
        window: Window function name.
        nfft_computed: FFT length.
        sample_rate: Sample rate.
        return_phase: Whether to return phase.

    Returns:
        FFT results (with or without phase).
    """
    _fft_cache_stats["misses"] += 1

    w = get_window(window, n)
    data_windowed = data_processed * w
    spectrum = np.fft.rfft(data_windowed, n=nfft_computed)
    freq = np.fft.rfftfreq(nfft_computed, d=1.0 / sample_rate)

    # Magnitude in dB
    window_gain = np.sum(w) / n
    magnitude = np.abs(spectrum) / (n * window_gain)
    magnitude = np.maximum(magnitude, 1e-20)
    magnitude_db = 20 * np.log10(magnitude)

    if return_phase:
        phase = np.angle(spectrum)
        return freq, magnitude_db, phase
    else:
        return freq, magnitude_db


def psd(
    trace: WaveformTrace,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Power Spectral Density using Welch's method.

    Uses overlapped segment averaging for reduced variance PSD estimation.

    Args:
        trace: Input waveform trace.
        window: Window function name (default "hann").
        nperseg: Segment length. If None, uses n // 8 or 256, whichever larger.
        noverlap: Overlap between segments. If None, uses nperseg // 2.
        nfft: FFT length per segment. If None, uses nperseg.
        scaling: Output scaling:
            - "density": Power spectral density (V^2/Hz)
            - "spectrum": Power spectrum (V^2)

    Returns:
        (frequencies, psd) - Frequency axis and PSD in dB/Hz.

    Raises:
        InsufficientDataError: If trace has insufficient data.

    Example:
        >>> freq, psd_db = psd(trace)
        >>> plt.semilogx(freq, psd_db)
        >>> plt.ylabel("PSD (dB/Hz)")

    References:
        Welch, P. D. (1967). "The use of fast Fourier transform for the
        estimation of power spectra"
    """
    data = trace.data
    n = len(data)

    if n < 16:
        raise InsufficientDataError(
            "PSD requires at least 16 samples",
            required=16,
            available=n,
            analysis_type="psd",
        )

    sample_rate = trace.metadata.sample_rate

    # Default segment length
    if nperseg is None:
        nperseg = max(256, n // 8)
        nperseg = min(nperseg, n)

    # Default overlap (50% for Hann window)
    if noverlap is None:
        noverlap = nperseg // 2

    freq, psd_linear = sp_signal.welch(
        data,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling=scaling,
        detrend="constant",
    )

    # Convert to dB
    psd_linear = np.maximum(psd_linear, 1e-20)
    psd_db = 10 * np.log10(psd_linear)

    return freq, psd_db


def periodogram(
    trace: WaveformTrace,
    *,
    window: str = "hann",
    nfft: int | None = None,
    scaling: Literal["density", "spectrum"] = "density",
    detrend: Literal["constant", "linear", False] = "constant",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute classical periodogram PSD estimate.

    Single-segment PSD estimation using scaled FFT magnitude squared.

    Args:
        trace: Input waveform trace.
        window: Window function name (default "hann").
        nfft: FFT length. If None, uses data length.
        scaling: Output scaling ("density" or "spectrum").
        detrend: Detrending method.

    Returns:
        (frequencies, psd) - Frequency axis and PSD.

    Example:
        >>> freq, psd = periodogram(trace)

    References:
        IEEE 1241-2010 Section 4.1.2
    """
    sample_rate = trace.metadata.sample_rate

    freq, psd_linear = sp_signal.periodogram(
        trace.data,
        fs=sample_rate,
        window=window,
        nfft=nfft,
        scaling=scaling,
        detrend=detrend,
    )

    # Convert to dB
    psd_linear = np.maximum(psd_linear, 1e-20)
    psd_db = 10 * np.log10(psd_linear)

    return freq, psd_db


def bartlett_psd(
    trace: WaveformTrace,
    *,
    n_segments: int = 8,
    window: str = "rectangular",
    nfft: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Bartlett's method PSD estimate.

    Averages periodograms of non-overlapping segments for
    reduced variance at cost of frequency resolution.

    Args:
        trace: Input waveform trace.
        n_segments: Number of non-overlapping segments.
        window: Window function per segment.
        nfft: FFT length per segment.

    Returns:
        (frequencies, psd_db) - Frequency axis and PSD in dB.

    Raises:
        AnalysisError: If no segments were processed (empty trace).
        InsufficientDataError: If trace has fewer than 16*n_segments samples.

    Example:
        >>> freq, psd = bartlett_psd(trace, n_segments=8)
    """
    data = trace.data
    n = len(data)
    sample_rate = trace.metadata.sample_rate

    segment_length = n // n_segments

    if segment_length < 16:
        raise InsufficientDataError(
            f"Bartlett requires at least {16 * n_segments} samples for {n_segments} segments",
            required=16 * n_segments,
            available=n,
            analysis_type="bartlett_psd",
        )

    if nfft is None:
        nfft = segment_length

    # Accumulate periodograms
    psd_sum = None
    w = get_window(window, segment_length)
    window_power = np.sum(w**2)

    for i in range(n_segments):
        segment = data[i * segment_length : (i + 1) * segment_length]
        segment_windowed = segment * w

        spectrum = np.fft.rfft(segment_windowed, n=nfft)
        # Power spectrum (V^2)
        psd_segment = (np.abs(spectrum) ** 2) / (sample_rate * window_power)

        if psd_sum is None:
            psd_sum = psd_segment
        else:
            psd_sum += psd_segment

    if psd_sum is None:
        raise AnalysisError("No segments were processed - input trace may be empty")
    psd_avg = psd_sum / n_segments

    # Frequency axis
    freq = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

    # Convert to dB
    psd_avg = np.maximum(psd_avg, 1e-20)
    psd_db = 10 * np.log10(psd_avg)

    return freq, psd_db


def spectrogram(
    trace: WaveformTrace,
    *,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Short-Time Fourier Transform spectrogram.

    Time-frequency representation for analyzing non-stationary signals.

    Args:
        trace: Input waveform trace.
        window: Window function name.
        nperseg: Segment length. If None, auto-selected.
        noverlap: Overlap between segments. If None, uses nperseg - 1.
        nfft: FFT length per segment.

    Returns:
        (times, frequencies, magnitude_db) - Time axis, frequency axis,
        and magnitude in dB as 2D array.

    Example:
        >>> t, f, Sxx = spectrogram(trace)
        >>> plt.pcolormesh(t, f, Sxx, shading='auto')
        >>> plt.ylabel('Frequency (Hz)')
        >>> plt.xlabel('Time (s)')
    """
    data = trace.data
    n = len(data)
    sample_rate = trace.metadata.sample_rate

    if nperseg is None:
        nperseg = min(256, n // 4)
        nperseg = max(nperseg, 16)

    if noverlap is None:
        noverlap = nperseg - nperseg // 8

    freq, times, Sxx = sp_signal.spectrogram(
        data,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling="spectrum",
    )

    # Convert to dB
    Sxx = np.maximum(Sxx, 1e-20)
    Sxx_db = 10 * np.log10(Sxx)

    return times, freq, Sxx_db


def _find_fundamental(
    freq: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    *,
    min_freq: float = 0.0,
) -> tuple[int, float, float]:
    """Find fundamental frequency in spectrum.

    Args:
        freq: Frequency axis.
        magnitude: Magnitude spectrum (linear, not dB).
        min_freq: Minimum frequency to consider.

    Returns:
        (index, frequency, magnitude) of fundamental.
    """
    # Skip DC and frequencies below min_freq
    valid_mask = freq > max(min_freq, freq[1])
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return 0, 0.0, 0.0

    # Find peak in valid region
    local_peak_idx = np.argmax(magnitude[valid_indices])
    peak_idx = valid_indices[local_peak_idx]

    return int(peak_idx), float(freq[peak_idx]), float(magnitude[peak_idx])


def _find_harmonic_indices(
    freq: NDArray[np.float64],
    fundamental_freq: float,
    n_harmonics: int,
) -> list[int]:
    """Find indices of harmonic frequencies.

    Args:
        freq: Frequency axis.
        fundamental_freq: Fundamental frequency.
        n_harmonics: Number of harmonics to find.

    Returns:
        List of indices for harmonics 2, 3, ..., n_harmonics+1.
    """
    indices = []

    for h in range(2, n_harmonics + 2):
        target_freq = h * fundamental_freq
        if target_freq > freq[-1]:
            break

        # Find closest bin
        idx = np.argmin(np.abs(freq - target_freq))
        indices.append(int(idx))

    return indices


def thd(
    trace: WaveformTrace,
    *,
    n_harmonics: int = 10,
    window: str = "hann",
    nfft: int | None = None,
    return_db: bool = True,
) -> float:
    """Compute Total Harmonic Distortion.

    THD is the ratio of harmonic power to fundamental power.

    Args:
        trace: Input waveform trace.
        n_harmonics: Number of harmonics to include (default 10).
        window: Window function for FFT.
        nfft: FFT length. If None, uses data length (no zero-padding) to
            preserve coherent sampling per IEEE 1241-2010.
        return_db: If True, return in dB. If False, return percentage.

    Returns:
        THD in dB or percentage.

    Example:
        >>> thd_db = thd(trace)
        >>> thd_pct = thd(trace, return_db=False)
        >>> print(f"THD: {thd_db:.1f} dB ({thd_pct:.2f}%)")

    References:
        IEEE 1241-2010 Section 4.1.4.2
    """
    # Use data length as NFFT to avoid zero-padding that breaks coherence
    if nfft is None:
        nfft = len(trace.data)

    result = fft(trace, window=window, nfft=nfft, detrend="mean")
    freq, mag_db = result[0], result[1]

    # Convert to linear
    magnitude = 10 ** (mag_db / 20)

    # Find fundamental
    _fund_idx, fund_freq, fund_mag = _find_fundamental(freq, magnitude)

    if fund_mag == 0 or fund_freq == 0:
        return np.nan

    # Find harmonics
    harmonic_indices = _find_harmonic_indices(freq, fund_freq, n_harmonics)

    if len(harmonic_indices) == 0:
        return 0.0 if not return_db else -np.inf

    # Sum harmonic power
    harmonic_power = sum(magnitude[i] ** 2 for i in harmonic_indices)

    # THD ratio
    thd_ratio = np.sqrt(harmonic_power) / fund_mag

    if return_db:
        if thd_ratio <= 0:
            return -np.inf
        return float(20 * np.log10(thd_ratio))
    else:
        return float(thd_ratio * 100)


def snr(
    trace: WaveformTrace,
    *,
    n_harmonics: int = 10,
    window: str = "hann",
    nfft: int | None = None,
) -> float:
    """Compute Signal-to-Noise Ratio.

    SNR is the ratio of signal power to noise power, excluding harmonics.

    Args:
        trace: Input waveform trace.
        n_harmonics: Number of harmonics to exclude from noise.
        window: Window function for FFT.
        nfft: FFT length. If None, uses data length (no zero-padding) to
            preserve coherent sampling per IEEE 1241-2010.

    Returns:
        SNR in dB.

    Example:
        >>> snr_db = snr(trace)
        >>> print(f"SNR: {snr_db:.1f} dB")

    References:
        IEEE 1241-2010 Section 4.1.4.1
    """
    if nfft is None:
        nfft = len(trace.data)

    freq, magnitude = _compute_magnitude_spectrum(trace, window, nfft)
    fund_idx, fund_freq, fund_mag = _find_fundamental(freq, magnitude)

    if fund_mag == 0 or fund_freq == 0:
        return np.nan

    harmonic_indices = _find_harmonic_indices(freq, fund_freq, n_harmonics)
    exclude_indices = _build_exclusion_set(fund_idx, harmonic_indices, len(magnitude))

    signal_power = _compute_signal_power(magnitude, fund_idx)
    noise_power = _compute_noise_power(magnitude, exclude_indices)

    if noise_power <= 0:
        return np.inf

    return float(10 * np.log10(signal_power / noise_power))


def _compute_magnitude_spectrum(
    trace: WaveformTrace, window: str, nfft: int
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """Compute magnitude spectrum from FFT.

    Args:
        trace: Input waveform.
        window: Window function name.
        nfft: FFT length.

    Returns:
        Tuple of (frequency array, magnitude array).
    """
    result = fft(trace, window=window, nfft=nfft, detrend="mean")
    freq, mag_db = result[0], result[1]
    magnitude = 10 ** (mag_db / 20)
    return freq, magnitude


def _build_exclusion_set(fund_idx: int, harmonic_indices: list[int], n_bins: int) -> set[int]:
    """Build set of frequency bins to exclude from noise.

    Args:
        fund_idx: Fundamental frequency bin index.
        harmonic_indices: Harmonic bin indices.
        n_bins: Total number of bins.

    Returns:
        Set of bin indices to exclude.
    """
    exclude_indices = {0}  # DC

    # Exclude fundamental +/- 3 bins
    for offset in range(-3, 4):
        idx = fund_idx + offset
        if 0 <= idx < n_bins:
            exclude_indices.add(idx)

    # Exclude harmonics +/- 3 bins
    for h_idx in harmonic_indices:
        for offset in range(-3, 4):
            idx = h_idx + offset
            if 0 <= idx < n_bins:
                exclude_indices.add(idx)

    return exclude_indices


def _compute_signal_power(magnitude: NDArray[np.floating[Any]], fund_idx: int) -> float:
    """Compute signal power from fundamental.

    Args:
        magnitude: Magnitude spectrum.
        fund_idx: Fundamental bin index.

    Returns:
        Signal power (3-bin sum around fundamental).
    """
    signal_power = 0.0
    for offset in range(-1, 2):
        idx = fund_idx + offset
        if 0 <= idx < len(magnitude):
            signal_power += magnitude[idx] ** 2
    return signal_power


def _compute_noise_power(magnitude: NDArray[np.floating[Any]], exclude_indices: set[int]) -> float:
    """Compute noise power from non-excluded bins.

    Args:
        magnitude: Magnitude spectrum.
        exclude_indices: Bins to exclude.

    Returns:
        Noise power.
    """
    noise_power = 0.0
    for i in range(len(magnitude)):
        if i not in exclude_indices:
            noise_power += magnitude[i] ** 2
    return noise_power


def sinad(
    trace: WaveformTrace,
    *,
    window: str = "hann",
    nfft: int | None = None,
) -> float:
    """Compute Signal-to-Noise and Distortion ratio.

    SINAD is the ratio of signal power to noise plus distortion power.

    Args:
        trace: Input waveform trace.
        window: Window function for FFT.
        nfft: FFT length. If None, uses data length (no zero-padding) to
            preserve coherent sampling per IEEE 1241-2010.

    Returns:
        SINAD in dB.

    Example:
        >>> sinad_db = sinad(trace)
        >>> print(f"SINAD: {sinad_db:.1f} dB")

    References:
        IEEE 1241-2010 Section 4.1.4.3
    """
    # Use data length as NFFT to avoid zero-padding that breaks coherence
    if nfft is None:
        nfft = len(trace.data)

    result = fft(trace, window=window, nfft=nfft, detrend="mean")
    freq, mag_db = result[0], result[1]
    magnitude = 10 ** (mag_db / 20)

    # Find fundamental
    fund_idx, _fund_freq, fund_mag = _find_fundamental(freq, magnitude)

    if fund_mag == 0:
        return np.nan

    # Signal power: use 3-bin window around fundamental to capture spectral leakage
    signal_power = 0.0
    for offset in range(-1, 2):
        idx = fund_idx + offset
        if 0 <= idx < len(magnitude):
            signal_power += magnitude[idx] ** 2

    # Total power (exclude DC)
    total_power = np.sum(magnitude[1:] ** 2)

    # Noise + distortion power = everything except signal
    nad_power = total_power - signal_power

    if nad_power <= 0:
        return np.inf

    sinad_ratio = signal_power / nad_power
    return float(10 * np.log10(sinad_ratio))


def enob(
    trace: WaveformTrace,
    *,
    window: str = "hann",
    nfft: int | None = None,
) -> float:
    """Compute Effective Number of Bits from SINAD.

    ENOB = (SINAD - 1.76) / 6.02

    Args:
        trace: Input waveform trace.
        window: Window function for FFT.
        nfft: FFT length.

    Returns:
        ENOB in bits, or np.nan if SINAD is invalid.

    Example:
        >>> bits = enob(trace)
        >>> print(f"ENOB: {bits:.2f} bits")

    References:
        IEEE 1241-2010 Section 4.1.4.4
    """
    sinad_db = sinad(trace, window=window, nfft=nfft)

    if np.isnan(sinad_db) or sinad_db <= 0:
        return np.nan

    return float((sinad_db - 1.76) / 6.02)


def sfdr(
    trace: WaveformTrace,
    *,
    window: str = "hann",
    nfft: int | None = None,
) -> float:
    """Compute Spurious-Free Dynamic Range.

    SFDR is the ratio of fundamental to largest spurious component.

    Args:
        trace: Input waveform trace.
        window: Window function for FFT.
        nfft: FFT length. If None, uses data length (no zero-padding) to
            preserve coherent sampling per IEEE 1241-2010.

    Returns:
        SFDR in dBc (dB relative to carrier/fundamental).

    Example:
        >>> sfdr_db = sfdr(trace)
        >>> print(f"SFDR: {sfdr_db:.1f} dBc")

    References:
        IEEE 1241-2010 Section 4.1.4.5
    """
    # Use data length as NFFT to avoid zero-padding that breaks coherence
    if nfft is None:
        nfft = len(trace.data)

    result = fft(trace, window=window, nfft=nfft, detrend="mean")
    freq, mag_db = result[0], result[1]
    magnitude = 10 ** (mag_db / 20)

    # Find fundamental
    fund_idx, _fund_freq, fund_mag = _find_fundamental(freq, magnitude)

    if fund_mag == 0:
        return np.nan

    # Create mask for spurs (exclude fundamental and DC)
    spur_mask = np.ones(len(magnitude), dtype=bool)
    spur_mask[0] = False  # DC
    spur_mask[fund_idx] = False

    # Exclude more bins adjacent to fundamental to account for spectral leakage
    # For Hann window, typical main lobe width is ~4 bins
    for offset in range(-5, 6):
        if offset == 0:
            continue
        idx = fund_idx + offset
        if 0 <= idx < len(magnitude):
            spur_mask[idx] = False

    # Find largest spur
    spur_magnitudes = magnitude[spur_mask]
    if len(spur_magnitudes) == 0:
        return np.inf

    max_spur = np.max(spur_magnitudes)

    if max_spur <= 0:
        return np.inf

    sfdr_ratio = fund_mag / max_spur
    return float(20 * np.log10(sfdr_ratio))


def hilbert_transform(
    trace: WaveformTrace,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Hilbert transform for envelope and instantaneous frequency.

    Computes the analytic signal to extract envelope (instantaneous
    amplitude), instantaneous phase, and instantaneous frequency.

    Args:
        trace: Input waveform trace.

    Returns:
        (envelope, phase, inst_freq) - Instantaneous amplitude,
        phase (radians), and frequency (Hz).

    Example:
        >>> envelope, phase, inst_freq = hilbert_transform(trace)
        >>> plt.plot(trace.time_vector, envelope)

    References:
        Oppenheim, A. V. & Schafer, R. W. (2009). Discrete-Time
        Signal Processing, 3rd ed.
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    # Compute analytic signal
    analytic = sp_signal.hilbert(data)

    # Instantaneous amplitude (envelope)
    envelope = np.abs(analytic)

    # Instantaneous phase
    phase = np.unwrap(np.angle(analytic))

    # Instantaneous frequency (derivative of phase / 2pi)
    inst_freq = np.zeros_like(phase)
    inst_freq[1:] = np.diff(phase) * sample_rate / (2 * np.pi)
    inst_freq[0] = inst_freq[1]  # Extrapolate first sample

    return envelope, phase, inst_freq


def cwt(
    trace: WaveformTrace,
    *,
    wavelet: Literal["morlet", "mexh", "ricker"] = "morlet",
    scales: NDArray[np.float64] | None = None,
    n_scales: int = 64,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Continuous Wavelet Transform for time-frequency analysis.

    Uses CWT to analyze non-stationary signals with multi-resolution
    time-frequency representation.

    Args:
        trace: Input waveform trace.
        wavelet: Wavelet type:
            - "morlet": Morlet wavelet (complex, good for frequency localization)
            - "mexh": Mexican hat wavelet (real, good for feature detection)
            - "ricker": Ricker wavelet (real, synonym for Mexican hat)
        scales: Array of scales to use. If None, auto-generated logarithmically.
        n_scales: Number of scales if auto-generated (default 64).

    Returns:
        (scales, frequencies, coefficients) where coefficients is 2D array
        of shape (n_scales, n_samples).

    Raises:
        InsufficientDataError: If trace has fewer than 8 samples.
        ValueError: If wavelet type is not recognized.

    Example:
        >>> scales, freqs, coef = cwt(trace, wavelet="morlet")
        >>> plt.pcolormesh(trace.time_vector, freqs, np.abs(coef))
        >>> plt.ylabel("Frequency (Hz)")

    References:
        Mallat, S. (2009). A Wavelet Tour of Signal Processing, 3rd ed.
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    if len(data) < 8:
        raise InsufficientDataError(
            "CWT requires at least 8 samples",
            required=8,
            available=len(data),
            analysis_type="cwt",
        )

    # Auto-generate scales if not provided
    if scales is None:
        # Logarithmically spaced scales from 1 to n/8
        scales = np.logspace(0, np.log10(len(data) / 8), n_scales)

    # Select wavelet function
    if wavelet == "morlet":
        # Morlet wavelet (complex): good for frequency analysis
        widths = scales
        coefficients = sp_signal.cwt(data, sp_signal.morlet2, widths)
    elif wavelet in ("mexh", "ricker"):
        # Mexican hat wavelet (Ricker): real, good for edge detection
        widths = scales
        coefficients = sp_signal.cwt(data, sp_signal.ricker, widths)
    else:
        raise ValueError(f"Unknown wavelet: {wavelet}. Choose from: morlet, mexh, ricker")

    # Convert scales to frequencies
    # For Morlet: f = fc / (scale * dt) where fc = center frequency
    # For simplicity, use approximate relation: f = 1 / (scale * dt)
    dt = 1.0 / sample_rate
    frequencies = 1.0 / (scales * dt)

    return scales, frequencies, coefficients


def dwt(
    trace: WaveformTrace,
    *,
    wavelet: str = "db4",
    level: int | None = None,
    mode: str = "symmetric",
) -> dict[str, NDArray[np.float64]]:
    """Compute Discrete Wavelet Transform for multi-level decomposition.

    Decomposes signal into approximation (low-frequency) and detail
    (high-frequency) coefficients at multiple levels.

    Args:
        trace: Input waveform trace.
        wavelet: Wavelet family:
            - "dbN": Daubechies wavelets (e.g., "db1", "db4", "db8")
            - "symN": Symlet wavelets (e.g., "sym2", "sym8")
            - "coifN": Coiflet wavelets (e.g., "coif1", "coif5")
        level: Decomposition level (auto-computed if None).
        mode: Signal extension mode ("symmetric", "periodic", "zero", etc.).

    Returns:
        Dictionary with keys:
            - "cA": Final approximation coefficients
            - "cD1", "cD2", ...: Detail coefficients at each level

    Raises:
        AnalysisError: If DWT decomposition fails.
        ImportError: If PyWavelets library is not installed.
        InsufficientDataError: If trace has fewer than 4 samples.

    Example:
        >>> coeffs = dwt(trace, wavelet="db4", level=3)
        >>> print(f"Approximation: {len(coeffs['cA'])} coefficients")
        >>> print(f"Detail levels: {[k for k in coeffs if k.startswith('cD')]}")

    Note:
        Requires pywt library. Install with: pip install PyWavelets

    References:
        Daubechies, I. (1992). Ten Lectures on Wavelets
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("DWT requires PyWavelets library. Install with: pip install PyWavelets")

    data = trace.data

    if len(data) < 4:
        raise InsufficientDataError(
            "DWT requires at least 4 samples",
            required=4,
            available=len(data),
            analysis_type="dwt",
        )

    # Auto-select decomposition level
    if level is None:
        level = pywt.dwt_max_level(len(data), wavelet)
        # Limit to reasonable levels
        level = min(level, 8)

    try:
        # Perform multi-level DWT
        coeffs = pywt.wavedec(data, wavelet, mode=mode, level=level)
    except ValueError as e:
        raise AnalysisError(f"DWT decomposition failed: {e}", analysis_type="dwt")

    # Package into dictionary
    result = {"cA": coeffs[0]}  # Approximation coefficients

    for i, detail in enumerate(coeffs[1:], start=1):
        result[f"cD{i}"] = detail

    return result


def idwt(
    coeffs: dict[str, NDArray[np.float64]],
    *,
    wavelet: str = "db4",
    mode: str = "symmetric",
) -> NDArray[np.float64]:
    """Reconstruct signal from DWT coefficients.

    Performs inverse DWT to reconstruct the original signal from
    approximation and detail coefficients.

    Args:
        coeffs: Dictionary of DWT coefficients from dwt().
        wavelet: Wavelet family (must match original decomposition).
        mode: Signal extension mode (must match original decomposition).

    Returns:
        Reconstructed signal array.

    Raises:
        AnalysisError: If IDWT reconstruction fails.
        ImportError: If PyWavelets library is not installed.

    Example:
        >>> coeffs = dwt(trace, wavelet="db4")
        >>> # Modify coefficients (e.g., denoise)
        >>> coeffs["cD1"] *= 0  # Remove finest detail
        >>> reconstructed = idwt(coeffs, wavelet="db4")
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("IDWT requires PyWavelets library. Install with: pip install PyWavelets")

    # Reconstruct coefficient list
    cA = coeffs["cA"]

    # Get detail levels in order
    detail_keys = sorted(
        [k for k in coeffs if k.startswith("cD")],
        key=lambda x: int(x[2:]),
    )
    details = [coeffs[k] for k in detail_keys]

    # Combine into pywt format
    coeff_list = [cA, *details]

    try:
        reconstructed = pywt.waverec(coeff_list, wavelet, mode=mode)
    except ValueError as e:
        raise AnalysisError(f"IDWT reconstruction failed: {e}", analysis_type="idwt")

    return np.asarray(reconstructed, dtype=np.float64)


def mfcc(
    trace: WaveformTrace,
    *,
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int | None = None,
    n_mels: int = 40,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> NDArray[np.float64]:
    """Compute Mel-Frequency Cepstral Coefficients for audio analysis.

    MFCCs are widely used in speech and audio processing for feature
    extraction and pattern recognition.

    Args:
        trace: Input waveform trace.
        n_mfcc: Number of MFCC coefficients to return (default 13).
        n_fft: FFT window size (default 512).
        hop_length: Number of samples between frames. If None, uses n_fft // 4.
        n_mels: Number of Mel filterbank channels (default 40).
        fmin: Minimum frequency for Mel filterbank (Hz).
        fmax: Maximum frequency for Mel filterbank (default: sample_rate/2).

    Returns:
        2D array of shape (n_mfcc, n_frames) with MFCC time series.

    Raises:
        InsufficientDataError: If trace has fewer than n_fft samples.

    Example:
        >>> mfcc_features = mfcc(audio_trace, n_mfcc=13)
        >>> print(f"MFCCs: {mfcc_features.shape[0]} coefficients, {mfcc_features.shape[1]} frames")

    Note:
        This is a custom implementation. For production use, consider librosa.

    References:
        Davis, S. & Mermelstein, P. (1980). "Comparison of parametric
        representations for monosyllabic word recognition"
    """
    data = trace.data
    sample_rate = trace.metadata.sample_rate

    if len(data) < n_fft:
        raise InsufficientDataError(
            f"MFCC requires at least {n_fft} samples",
            required=n_fft,
            available=len(data),
            analysis_type="mfcc",
        )

    if hop_length is None:
        hop_length = n_fft // 4

    if fmax is None:
        fmax = sample_rate / 2

    # Compute STFT (Short-Time Fourier Transform)
    # Use scipy's spectrogram as a proxy
    _f, _t, Sxx = sp_signal.spectrogram(
        data,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        scaling="spectrum",
    )

    # Power spectrum magnitude
    power_spec = np.abs(Sxx)

    # Create Mel filterbank
    mel_filters = _mel_filterbank(n_mels, n_fft, sample_rate, fmin, fmax)

    # Apply Mel filterbank to power spectrum
    mel_spec = mel_filters @ power_spec

    # Convert to log scale (dB)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel_spec = 10 * np.log10(mel_spec)

    # Compute DCT (Discrete Cosine Transform) to get cepstral coefficients
    # Use scipy.fft.dct
    from scipy.fft import dct

    mfcc_features = dct(log_mel_spec, axis=0, type=2, norm="ortho")[:n_mfcc, :]

    return np.asarray(mfcc_features, dtype=np.float64)


def _mel_filterbank(
    n_filters: int,
    n_fft: int,
    sample_rate: float,
    fmin: float,
    fmax: float,
) -> NDArray[np.float64]:
    """Create Mel-scale filterbank matrix.

    Args:
        n_filters: Number of Mel filters.
        n_fft: FFT size.
        sample_rate: Sampling rate in Hz.
        fmin: Minimum frequency (Hz).
        fmax: Maximum frequency (Hz).

    Returns:
        Filterbank matrix of shape (n_filters, n_fft // 2 + 1).
    """

    def hz_to_mel(hz: float) -> float:
        """Convert Hz to Mel scale."""
        return 2595 * np.log10(1 + hz / 700)  # type: ignore[no-any-return]

    def mel_to_hz(mel: float) -> float:
        """Convert Mel scale to Hz."""
        return 700 * (10 ** (mel / 2595) - 1)

    # Convert frequency range to Mel scale
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)

    # Create n_filters + 2 equally spaced points in Mel scale
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)

    # Convert back to Hz
    hz_points = np.array([mel_to_hz(float(m)) for m in mel_points])

    # Convert Hz to FFT bin indices
    n_freqs = n_fft // 2 + 1
    freq_bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create filterbank
    filterbank = np.zeros((n_filters, n_freqs))

    for i in range(n_filters):
        left = freq_bins[i]
        center = freq_bins[i + 1]
        right = freq_bins[i + 2]

        # Rising slope
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)

        # Falling slope
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)

    return filterbank


# ==========================================================================
# MEM-004, MEM-005, MEM-006: Chunked Processing for Large Signals
# ==========================================================================


def spectrogram_chunked(
    trace: WaveformTrace,
    *,
    chunk_size: int = 100_000_000,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    overlap_factor: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute spectrogram for very large signals using chunked processing.

    Processes signal in chunks with overlap to handle files larger than RAM.
    Stitches STFT results from overlapping chunks to create continuous spectrogram.

    Args:
        trace: Input waveform trace.
        chunk_size: Maximum samples per chunk (default 100M).
        window: Window function name.
        nperseg: Segment length for STFT. If None, auto-selected.
        noverlap: Overlap between STFT segments.
        nfft: FFT length per segment.
        overlap_factor: Overlap factor between chunks (default 2.0).

    Returns:
        (times, frequencies, magnitude_db) as 2D spectrogram.

    Example:
        >>> t, f, Sxx = spectrogram_chunked(trace, chunk_size=50_000_000)
    """
    data = trace.data
    n = len(data)
    sample_rate = trace.metadata.sample_rate

    nperseg, noverlap = _set_spectrogram_defaults(nperseg, noverlap, n)
    chunk_overlap = int(overlap_factor * nperseg)

    if n <= chunk_size:
        return spectrogram(trace, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

    chunks_stft, chunks_times, freq = _process_spectrogram_chunks(
        data, n, chunk_size, chunk_overlap, sample_rate, window, nperseg, noverlap, nfft
    )

    Sxx = np.concatenate(chunks_stft, axis=1)
    times = np.concatenate(chunks_times)

    Sxx = np.maximum(Sxx, 1e-20)
    Sxx_db: NDArray[np.float64] = np.asarray(10 * np.log10(Sxx), dtype=np.float64)

    return times, freq, Sxx_db


def _set_spectrogram_defaults(nperseg: int | None, noverlap: int | None, n: int) -> tuple[int, int]:
    """Set default spectrogram parameters.

    Args:
        nperseg: Segment length or None.
        noverlap: Overlap or None.
        n: Data length.

    Returns:
        Tuple of (nperseg, noverlap).
    """
    if nperseg is None:
        nperseg = min(256, n // 4)
        nperseg = max(nperseg, 16)
    if noverlap is None:
        noverlap = nperseg - nperseg // 8
    return nperseg, noverlap


def _process_spectrogram_chunks(
    data: NDArray[np.float64],
    n: int,
    chunk_size: int,
    chunk_overlap: int,
    sample_rate: float,
    window: str,
    nperseg: int,
    noverlap: int,
    nfft: int | None,
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]], NDArray[np.float64]]:
    """Process all spectrogram chunks.

    Args:
        data: Signal data.
        n: Data length.
        chunk_size: Chunk size in samples.
        chunk_overlap: Overlap between chunks.
        sample_rate: Sampling rate.
        window: Window function.
        nperseg: Segment length.
        noverlap: Segment overlap.
        nfft: FFT length.

    Returns:
        Tuple of (chunks_stft, chunks_times, freq).
    """
    chunks_stft = []
    chunks_times = []
    chunk_start = 0
    freq: NDArray[np.float64] | None = None

    while chunk_start < n:
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_data = _extract_spectrogram_chunk(data, chunk_start, chunk_end, chunk_overlap, n)

        freq_local, times_chunk, Sxx_chunk = sp_signal.spectrogram(
            chunk_data,
            fs=sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            scaling="spectrum",
        )

        if freq is None:
            freq = freq_local

        times_adjusted = _adjust_chunk_times(
            times_chunk, chunk_data, data, chunk_start, chunk_end, chunk_overlap, sample_rate
        )
        Sxx_trimmed, times_trimmed = _trim_chunk_overlap(
            Sxx_chunk, times_adjusted, chunk_start, chunk_end, n, sample_rate
        )

        chunks_stft.append(Sxx_trimmed)
        chunks_times.append(times_trimmed)
        chunk_start += chunk_size

    if freq is None:
        raise ValueError("No chunks processed - data length too small")

    return chunks_stft, chunks_times, freq


def _extract_spectrogram_chunk(
    data: NDArray[np.float64],
    chunk_start: int,
    chunk_end: int,
    chunk_overlap: int,
    n: int,
) -> NDArray[np.float64]:
    """Extract chunk data with overlap.

    Args:
        data: Full data array.
        chunk_start: Chunk start index.
        chunk_end: Chunk end index.
        chunk_overlap: Overlap size.
        n: Total data length.

    Returns:
        Chunk data array.
    """
    chunk_data_start = chunk_start - chunk_overlap if chunk_start > 0 else 0
    chunk_data_end = chunk_end + chunk_overlap if chunk_end < n else n
    return data[chunk_data_start:chunk_data_end]


def _adjust_chunk_times(
    times_chunk: NDArray[np.float64],
    chunk_data: NDArray[np.float64],
    data: NDArray[np.float64],
    chunk_start: int,
    chunk_end: int,
    chunk_overlap: int,
    sample_rate: float,
) -> NDArray[np.float64]:
    """Adjust chunk times for global position.

    Args:
        times_chunk: Local chunk times.
        chunk_data: Chunk data array.
        data: Full data array.
        chunk_start: Chunk start index.
        chunk_end: Chunk end index.
        chunk_overlap: Overlap size.
        sample_rate: Sampling rate.

    Returns:
        Adjusted time array.
    """
    chunk_data_start = chunk_start - chunk_overlap if chunk_start > 0 else 0
    time_offset = chunk_data_start / sample_rate
    return times_chunk + time_offset


def _trim_chunk_overlap(
    Sxx_chunk: NDArray[np.float64],
    times_adjusted: NDArray[np.float64],
    chunk_start: int,
    chunk_end: int,
    n: int,
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Trim overlap regions from chunk.

    Args:
        Sxx_chunk: Chunk spectrogram.
        times_adjusted: Adjusted times.
        chunk_start: Chunk start index.
        chunk_end: Chunk end index.
        n: Total data length.
        sample_rate: Sampling rate.

    Returns:
        Tuple of (trimmed spectrogram, trimmed times).
    """
    if chunk_start > 0 and chunk_end < n:
        # Middle chunk: trim both sides
        valid_time_start = chunk_start / sample_rate
        valid_time_end = chunk_end / sample_rate
        valid_mask = (times_adjusted >= valid_time_start) & (times_adjusted < valid_time_end)
    elif chunk_start > 0:
        # Last chunk: trim left overlap
        valid_time_start = chunk_start / sample_rate
        valid_mask = times_adjusted >= valid_time_start
    elif chunk_end < n:
        # First chunk: trim right overlap
        valid_time_end = chunk_end / sample_rate
        valid_mask = times_adjusted < valid_time_end
    else:
        # Single chunk
        return Sxx_chunk, times_adjusted

    return Sxx_chunk[:, valid_mask], times_adjusted[valid_mask]


def psd_chunked(
    trace: WaveformTrace,
    *,
    chunk_size: int = 100_000_000,
    window: str = "hann",
    nperseg: int | None = None,
    noverlap: int | None = None,
    nfft: int | None = None,
    scaling: Literal["density", "spectrum"] = "density",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Welch PSD for very large signals using chunked processing.


    Processes signal in chunks with proper overlap handling to compute
    Power Spectral Density for files larger than available RAM. The result
    is equivalent to computing Welch PSD on the entire signal but with
    bounded memory usage.

    Args:
        trace: Input waveform trace.
        chunk_size: Maximum samples per chunk (default 100M).
        window: Window function name.
        nperseg: Segment length for Welch. If None, auto-selected.
        noverlap: Overlap between Welch segments. If None, uses nperseg // 2.
        nfft: FFT length per segment.
        scaling: Output scaling ("density" or "spectrum").

    Returns:
        (frequencies, psd_db) - Frequency axis and PSD in dB.

    Example:
        >>> # Process 10 GB file in 50M sample chunks
        >>> freq, psd = psd_chunked(trace, chunk_size=50_000_000, nperseg=4096)
        >>> print(f"Frequency resolution: {freq[1] - freq[0]:.3f} Hz")

    Note:
        Memory usage is bounded by chunk_size, not file size.
        The result may differ slightly from standard psd() due to
        chunk boundary handling, but variance is typically reduced
        due to increased averaging.

    References:
        Welch, P. D. (1967). "The use of fast Fourier transform for the
        estimation of power spectra"
    """
    data = trace.data
    n = len(data)
    sample_rate = trace.metadata.sample_rate

    # Set default parameters
    nperseg, noverlap, nfft = _set_psd_defaults(nperseg, noverlap, nfft, n, chunk_size)

    # If data fits in one chunk, use standard PSD
    if n <= chunk_size:
        return psd(
            trace, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling=scaling
        )

    # Process chunks and accumulate
    psd_sum, total_segments, freq = _process_psd_chunks(
        data, sample_rate, chunk_size, nperseg, noverlap, nfft, window, scaling, n
    )

    # Fallback if processing failed
    if psd_sum is None or total_segments == 0 or freq is None:
        return psd(
            trace, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling=scaling
        )

    # Average and convert to dB
    psd_avg = psd_sum / total_segments
    psd_avg = np.maximum(psd_avg, 1e-20)
    psd_db = 10 * np.log10(psd_avg)

    return freq, psd_db


def _set_psd_defaults(
    nperseg: int | None,
    noverlap: int | None,
    nfft: int | None,
    n: int,
    chunk_size: int,
) -> tuple[int, int, int]:
    """Set default PSD parameters."""
    if nperseg is None:
        nperseg = max(256, min(n // 8, chunk_size // 8))
        nperseg = min(nperseg, n)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg
    return nperseg, noverlap, nfft


def _process_psd_chunks(
    data: NDArray[np.float64],
    sample_rate: float,
    chunk_size: int,
    nperseg: int,
    noverlap: int,
    nfft: int,
    window: str,
    scaling: str,
    n: int,
) -> tuple[NDArray[np.float64] | None, int, NDArray[np.float64] | None]:
    """Process chunks and accumulate PSD estimates."""
    chunk_overlap = nperseg
    psd_sum: NDArray[np.float64] | None = None
    total_segments = 0
    freq: NDArray[np.float64] | None = None
    chunk_start = 0

    while chunk_start < n:
        chunk_data = _extract_chunk_with_overlap(data, chunk_start, chunk_size, chunk_overlap, n)
        if len(chunk_data) < nperseg:
            break

        f, psd_linear = sp_signal.welch(
            chunk_data,
            fs=sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            scaling=scaling,
            detrend="constant",
        )

        num_segments = max(1, (len(chunk_data) - noverlap) // (nperseg - noverlap))

        if psd_sum is None:
            psd_sum = psd_linear * num_segments
            freq = f
        else:
            psd_sum += psd_linear * num_segments

        total_segments += num_segments
        chunk_start += chunk_size

    return psd_sum, total_segments, freq


def _extract_chunk_with_overlap(
    data: NDArray[np.float64],
    chunk_start: int,
    chunk_size: int,
    chunk_overlap: int,
    n: int,
) -> NDArray[np.float64]:
    """Extract chunk with overlap on both sides."""
    chunk_data_start = max(0, chunk_start - chunk_overlap)
    chunk_end = min(chunk_start + chunk_size, n)
    chunk_data_end = min(chunk_end + chunk_overlap, n)
    return data[chunk_data_start:chunk_data_end]


def fft_chunked(
    trace: WaveformTrace,
    *,
    segment_size: int = 1_000_000,
    overlap_pct: float = 50.0,
    window: str = "hann",
    nfft: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute FFT for very long signals using segmented processing.

    Divides signal into overlapping segments, computes FFT for each,
    and averages the magnitude spectra to reduce variance.

    Args:
        trace: Input waveform trace.
        segment_size: Size of each segment in samples.
        overlap_pct: Percentage overlap between segments (0-100).
        window: Window function name.
        nfft: FFT length. If None, uses segment_size.

    Returns:
        (frequencies, magnitude_db) - Averaged magnitude spectrum in dB.

    Raises:
        AnalysisError: If no segments were processed (empty trace).

    Example:
        >>> freq, mag = fft_chunked(trace, segment_size=1_000_000)

    References:
        Welch's method for spectral estimation
    """
    data = trace.data
    n = len(data)
    sample_rate = trace.metadata.sample_rate

    if n < segment_size:
        result = fft(trace, window=window, nfft=nfft)
        return result[0], result[1]

    overlap_samples = int(segment_size * overlap_pct / 100.0)
    hop = segment_size - overlap_samples
    num_segments = max(1, (n - overlap_samples) // hop)

    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(segment_size)))

    freq, magnitude_sum = _accumulate_fft_segments(
        data, n, num_segments, hop, segment_size, window, nfft, sample_rate
    )

    magnitude_avg = magnitude_sum / num_segments
    magnitude_avg = np.maximum(magnitude_avg, 1e-20)
    magnitude_db = 20 * np.log10(magnitude_avg)

    return freq, magnitude_db


def _accumulate_fft_segments(
    data: NDArray[np.float64],
    n: int,
    num_segments: int,
    hop: int,
    segment_size: int,
    window: str,
    nfft: int,
    sample_rate: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Accumulate FFT magnitudes from all segments.

    Args:
        data: Signal data.
        n: Data length.
        num_segments: Number of segments.
        hop: Hop size between segments.
        segment_size: Size of each segment.
        window: Window function name.
        nfft: FFT length.
        sample_rate: Sampling rate.

    Returns:
        Tuple of (freq, magnitude_sum).

    Raises:
        AnalysisError: If no segments processed.
    """
    freq: NDArray[np.float64] | None = None
    magnitude_sum: NDArray[np.float64] | None = None
    w = get_window(window, segment_size)
    window_gain = np.sum(w) / segment_size

    for i in range(num_segments):
        segment = _extract_fft_segment(data, i, hop, segment_size, n)
        segment = segment - np.mean(segment)
        segment_windowed = segment * w
        spectrum = np.fft.rfft(segment_windowed, n=nfft)
        magnitude = np.abs(spectrum) / (segment_size * window_gain)

        if magnitude_sum is None:
            magnitude_sum = magnitude
            freq = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
        else:
            magnitude_sum += magnitude

    if magnitude_sum is None or freq is None:
        raise AnalysisError("No segments were processed - input trace may be empty")

    return freq, magnitude_sum


def _extract_fft_segment(
    data: NDArray[np.float64],
    segment_idx: int,
    hop: int,
    segment_size: int,
    n: int,
) -> NDArray[np.float64]:
    """Extract segment for FFT processing.

    Args:
        data: Full data array.
        segment_idx: Segment index.
        hop: Hop size.
        segment_size: Segment size.
        n: Total data length.

    Returns:
        Segment data (padded if needed).
    """
    start = segment_idx * hop
    end = min(start + segment_size, n)

    if end - start < segment_size:
        segment = np.zeros(segment_size)
        segment[: end - start] = data[start:end]
    else:
        segment = data[start:end]

    return segment


__all__ = [
    "bartlett_psd",
    "clear_fft_cache",
    "configure_fft_cache",
    "cwt",
    "dwt",
    "enob",
    "fft",
    "fft_chunked",
    "get_fft_cache_stats",
    "hilbert_transform",
    "idwt",
    "mfcc",
    "periodogram",
    "psd",
    "psd_chunked",
    "sfdr",
    "sinad",
    "snr",
    "spectrogram",
    "spectrogram_chunked",
    "thd",
]
