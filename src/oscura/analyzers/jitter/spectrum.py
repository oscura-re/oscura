"""Jitter spectrum analysis.

This module provides FFT-based analysis of TIE data to identify
periodic jitter sources and their frequencies.


Example:
    >>> from oscura.analyzers.jitter.spectrum import jitter_spectrum
    >>> result = jitter_spectrum(tie_data, sample_rate=1e9)
    >>> print(f"Dominant frequency: {result.dominant_frequency / 1e3:.1f} kHz")

References:
    IEEE 2414-2020: Standard for Jitter and Phase Noise
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class JitterSpectrumResult:
    """Result of jitter spectrum analysis.

    Attributes:
        frequencies: Frequency array in Hz.
        magnitude: Magnitude spectrum in seconds.
        magnitude_db: Magnitude spectrum in dB (relative to 1 ps).
        dominant_frequency: Frequency of largest component in Hz.
        dominant_magnitude: Magnitude at dominant frequency in seconds.
        noise_floor: Estimated noise floor in seconds.
        peaks: List of (frequency, magnitude) tuples for detected peaks.
    """

    frequencies: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    magnitude_db: NDArray[np.float64]
    dominant_frequency: float | None
    dominant_magnitude: float | None
    noise_floor: float
    peaks: list[tuple[float, float]]


def _create_empty_jitter_spectrum() -> JitterSpectrumResult:
    """Create empty jitter spectrum result for insufficient data.

    Returns:
        Empty JitterSpectrumResult.
    """
    return JitterSpectrumResult(
        frequencies=np.array([]),
        magnitude=np.array([]),
        magnitude_db=np.array([]),
        dominant_frequency=None,
        dominant_magnitude=None,
        noise_floor=0.0,
        peaks=[],
    )


def _preprocess_tie_data(
    valid_data: NDArray[np.float64], window: str, detrend: bool
) -> tuple[NDArray[np.float64], float, int]:
    """Preprocess TIE data with detrending and windowing.

    Args:
        valid_data: Valid TIE data (NaNs removed).
        window: Window function name.
        detrend: Whether to detrend data.

    Returns:
        Tuple of (windowed_data, window_factor, n_samples).
    """
    n = len(valid_data)

    if detrend:
        x = np.arange(n)
        slope, intercept = np.polyfit(x, valid_data, 1)
        data_detrended = valid_data - (slope * x + intercept)
    else:
        data_detrended = valid_data - np.mean(valid_data)

    win = {"hann": np.hanning(n), "hamming": np.hamming(n), "blackman": np.blackman(n)}.get(
        window, np.ones(n)
    )

    window_factor = np.sqrt(np.mean(win**2))
    return data_detrended * win, float(window_factor), n


def _compute_jitter_fft(
    data_windowed: NDArray[np.float64], n: int, window_factor: float, sample_rate: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute FFT of windowed jitter data.

    Args:
        data_windowed: Windowed TIE data.
        n: Original sample count.
        window_factor: Window power factor.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (frequencies, magnitude, magnitude_db).
    """
    nfft = int(2 ** np.ceil(np.log2(n)))
    spectrum = np.fft.rfft(data_windowed, n=nfft)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    magnitude = np.abs(spectrum) * 2 / n / window_factor
    magnitude_db = 20 * np.log10(magnitude / 1e-12 + 1e-20)
    return frequencies, magnitude, magnitude_db


def jitter_spectrum(
    tie_data: NDArray[np.float64],
    sample_rate: float,
    *,
    window: str = "hann",
    detrend: bool = True,
    n_peaks: int = 10,
) -> JitterSpectrumResult:
    """Compute FFT of TIE data to identify jitter frequency components.

    Identifies periodic jitter sources by frequency, useful for
    debugging EMI-induced jitter and power supply noise.

    Args:
        tie_data: Time Interval Error data in seconds.
        sample_rate: Sample rate of TIE data (edges per second).
        window: Window function ("hann", "hamming", "blackman", "none").
        detrend: Remove linear trend from data before FFT.
        n_peaks: Number of peaks to identify.

    Returns:
        JitterSpectrumResult with frequency analysis.

    Example:
        >>> result = jitter_spectrum(tie_data, sample_rate=1e9)
        >>> for freq, mag in result.peaks:
        ...     print(f"{freq/1e3:.1f} kHz: {mag*1e12:.2f} ps")

    References:
        IEEE 2414-2020 Section 6.8
    """
    # Setup: validate and prepare data
    valid_data = tie_data[~np.isnan(tie_data)]
    if len(valid_data) < 16:
        return _create_empty_jitter_spectrum()

    # Processing: apply preprocessing and FFT
    data_windowed, window_factor, n = _preprocess_tie_data(valid_data, window, detrend)
    frequencies, magnitude, magnitude_db = _compute_jitter_fft(
        data_windowed, n, window_factor, sample_rate
    )

    # Formatting: identify peaks and dominant frequency
    noise_floor = float(np.median(magnitude))
    peaks = identify_periodic_components(
        frequencies, magnitude, n_peaks=n_peaks, min_height=noise_floor * 3
    )
    dominant_frequency, dominant_magnitude = (peaks[0][0], peaks[0][1]) if peaks else (None, None)

    return JitterSpectrumResult(
        frequencies=frequencies,
        magnitude=magnitude,
        magnitude_db=magnitude_db,
        dominant_frequency=dominant_frequency,
        dominant_magnitude=dominant_magnitude,
        noise_floor=noise_floor,
        peaks=peaks,
    )


def identify_periodic_components(
    frequencies: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    *,
    n_peaks: int = 10,
    min_height: float | None = None,
    min_distance: int = 3,
) -> list[tuple[float, float]]:
    """Identify periodic components from jitter spectrum.

    Finds peaks in the magnitude spectrum that indicate periodic
    jitter sources.

    Args:
        frequencies: Frequency array in Hz.
        magnitude: Magnitude spectrum.
        n_peaks: Maximum number of peaks to return.
        min_height: Minimum peak height (default: 3x median).
        min_distance: Minimum distance between peaks in bins.

    Returns:
        List of (frequency, magnitude) tuples, sorted by magnitude.
    """
    if len(magnitude) < 3:
        return []

    min_height_value: float
    min_height_value = float(np.median(magnitude) * 3) if min_height is None else min_height

    # Find peaks
    peak_indices, _properties = find_peaks(
        magnitude,
        height=min_height_value,
        distance=min_distance,
    )

    if len(peak_indices) == 0:
        return []

    # Get peak heights
    peak_heights = magnitude[peak_indices]

    # Sort by magnitude (descending)
    sorted_order = np.argsort(peak_heights)[::-1]
    sorted_indices = peak_indices[sorted_order][:n_peaks]

    # Build result list
    peaks = [(float(frequencies[idx]), float(magnitude[idx])) for idx in sorted_indices]

    return peaks


__all__ = [
    "JitterSpectrumResult",
    "identify_periodic_components",
    "jitter_spectrum",
]
