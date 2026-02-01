"""Signal classification and waveform type detection.

This module provides automatic signal classification to detect periodicity,
waveform shape, and signal characteristics. Classification results are used
to determine which measurements are applicable to a given signal.

Example:
    >>> from oscura.analyzers.signal_classification import classify_signal
    >>> from oscura.core.types import WaveformTrace, TraceMetadata
    >>> import numpy as np
    >>> # Create test signal
    >>> t = np.linspace(0, 0.01, 1000)
    >>> data = np.sin(2 * np.pi * 1000 * t)
    >>> meta = TraceMetadata(sample_rate=1e5, units="V")
    >>> trace = WaveformTrace(data=data, metadata=meta)
    >>> result = classify_signal(trace)
    >>> print(f"Waveform: {result['waveform']['waveform_type']}")
    Waveform: sine

References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
    Signal Processing Fundamentals (autocorrelation methods)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


def detect_periodicity(signal: NDArray[np.floating[Any]], sample_rate: float) -> dict[str, Any]:
    """Detect periodic vs aperiodic signals using autocorrelation.

    Uses normalized autocorrelation to identify periodic patterns in the signal.
    A strong peak at non-zero lag indicates periodicity, while absence of peaks
    suggests aperiodic signals (impulse, DC, random noise).

    Args:
        signal: Input signal array.
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary containing:
            - periodicity: "periodic", "aperiodic", or "quasi-periodic"
            - confidence: Float between 0.0 and 1.0
            - period: Period in seconds (None if aperiodic)

    Example:
        >>> # Periodic signal
        >>> signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, 1000))
        >>> result = detect_periodicity(signal, 100000)
        >>> print(f"Type: {result['periodicity']}, Period: {result['period']}")
        Type: periodic, Period: 0.001

        >>> # Aperiodic signal (impulse)
        >>> impulse = np.zeros(1000)
        >>> impulse[500] = 1.0
        >>> result = detect_periodicity(impulse, 100000)
        >>> print(f"Type: {result['periodicity']}")
        Type: aperiodic
    """
    if len(signal) < 16:
        return {
            "periodicity": "aperiodic",
            "confidence": 0.0,
            "period": None,
        }

    # Remove DC offset for better autocorrelation
    signal_centered = signal - np.mean(signal)

    # Check if signal is essentially constant (DC only)
    if np.std(signal_centered) < 1e-10:
        return {
            "periodicity": "aperiodic",
            "confidence": 1.0,
            "period": None,
        }

    # Compute normalized autocorrelation
    # Use only first half of signal to avoid edge effects
    max_lag = min(len(signal_centered) // 2, 1000)
    autocorr = np.correlate(signal_centered, signal_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]  # Take only positive lags
    autocorr = autocorr[:max_lag]

    # Normalize by zero-lag value
    if autocorr[0] > 1e-10:
        autocorr = autocorr / autocorr[0]
    else:
        return {
            "periodicity": "aperiodic",
            "confidence": 1.0,
            "period": None,
        }

    # Find peaks in autocorrelation (skip first 2 samples to avoid DC)
    min_period_samples = 3
    search_region = autocorr[min_period_samples:]

    if len(search_region) < 3:
        return {
            "periodicity": "aperiodic",
            "confidence": 0.5,
            "period": None,
        }

    # Find first significant peak
    peak_idx = -1
    peak_value = 0.0

    # Look for local maxima
    for i in range(1, len(search_region) - 1):
        if (
            search_region[i] > search_region[i - 1]
            and search_region[i] > search_region[i + 1]
            and search_region[i] > peak_value
        ):
            peak_value = search_region[i]
            peak_idx = i + min_period_samples

    # Classify based on peak strength
    # Strong peak (>0.5) = periodic
    # Moderate peak (0.3-0.5) = quasi-periodic
    # Weak/no peak (<0.3) = aperiodic
    if peak_value > 0.5:
        period_samples = peak_idx
        period_seconds = period_samples / sample_rate
        return {
            "periodicity": "periodic",
            "confidence": float(peak_value),
            "period": float(period_seconds),
        }
    elif peak_value > 0.3:
        period_samples = peak_idx
        period_seconds = period_samples / sample_rate
        return {
            "periodicity": "quasi-periodic",
            "confidence": float(peak_value),
            "period": float(period_seconds),
        }
    else:
        return {
            "periodicity": "aperiodic",
            "confidence": float(1.0 - peak_value),
            "period": None,
        }


def classify_waveform(signal: NDArray[np.floating[Any]], sample_rate: float) -> dict[str, Any]:
    """Classify waveform shape using spectral and time-domain analysis.

    Analyzes harmonic content, duty cycle, edge sharpness, and DC component
    to determine the waveform type.

    Args:
        signal: Input signal array.
        sample_rate: Sample rate in Hz.

    Returns:
        Dictionary containing:
            - waveform_type: One of "sine", "square", "triangle", "sawtooth",
                           "pwm", "impulse", "dc", "noise", "unknown"
            - confidence: Float between 0.0 and 1.0
            - characteristics: Dict with additional features

    Example:
        >>> # Square wave
        >>> signal = np.sign(np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, 1000)))
        >>> result = classify_waveform(signal, 100000)
        >>> print(f"Type: {result['waveform_type']}")
        Type: square
    """
    if len(signal) < 16:
        return {
            "waveform_type": "unknown",
            "confidence": 0.0,
            "characteristics": {},
        }

    # Remove DC offset
    signal_centered = signal - np.mean(signal)
    dc_component = float(np.mean(signal))

    # Check for DC signal (no variation)
    signal_std = np.std(signal_centered)
    if signal_std < 1e-10:
        return {
            "waveform_type": "dc",
            "confidence": 1.0,
            "characteristics": {
                "dc_level": float(dc_component),
                "noise_level": 0.0,
            },
        }

    # Compute FFT for harmonic analysis
    n = len(signal_centered)
    fft_mag = np.abs(np.fft.rfft(signal_centered))
    fft_freqs = np.fft.rfftfreq(n, 1 / sample_rate)

    # Skip DC component
    fft_mag = fft_mag[1:]
    fft_freqs = fft_freqs[1:]

    if len(fft_mag) < 3:
        return {
            "waveform_type": "unknown",
            "confidence": 0.0,
            "characteristics": {},
        }

    # Find fundamental frequency (strongest component)
    fundamental_idx = np.argmax(fft_mag)
    fundamental_freq = fft_freqs[fundamental_idx]
    fundamental_mag = fft_mag[fundamental_idx]

    # Check if signal is essentially noise (no dominant frequency)
    mean_mag = np.mean(fft_mag)
    if fundamental_mag < 3.0 * mean_mag:
        return {
            "waveform_type": "noise",
            "confidence": 0.8,
            "characteristics": {
                "snr_estimate": float(fundamental_mag / mean_mag) if mean_mag > 0 else 0.0,
            },
        }

    # Analyze harmonics (multiples of fundamental)
    harmonics = []
    for harmonic_num in range(2, 6):  # Check 2nd through 5th harmonic
        harmonic_freq = fundamental_freq * harmonic_num
        # Find closest frequency bin
        freq_idx = np.argmin(np.abs(fft_freqs - harmonic_freq))
        if freq_idx < len(fft_mag):
            harmonic_mag = fft_mag[freq_idx]
            # Normalize by fundamental
            harmonic_ratio = harmonic_mag / fundamental_mag if fundamental_mag > 0 else 0.0
            harmonics.append(float(harmonic_ratio))

    # Calculate total harmonic distortion (THD)
    harmonic_power = sum(h**2 for h in harmonics)
    thd = np.sqrt(harmonic_power) if harmonics else 0.0

    # Classify based on harmonic content
    characteristics: dict[str, Any] = {
        "fundamental_freq": float(fundamental_freq),
        "thd": float(thd),
        "harmonics": harmonics,
    }

    # Early check for impulse: Very brief pulse (check time domain first)
    # For impulse detection, we need to check if signal is mostly near zero
    # with only brief excursions. Use absolute signal, not centered.
    abs_max = np.max(np.abs(signal))
    # Count samples that are significant (>10% of peak)
    threshold = 0.1 * abs_max if abs_max > 0 else signal_std
    significant_samples = np.sum(np.abs(signal) > threshold)
    pulse_width_ratio = significant_samples / len(signal) if len(signal) > 0 else 0

    # True impulse: very few significant samples AND mostly zero baseline
    signal_range = np.max(signal) - np.min(signal)
    near_zero_count = np.sum(np.abs(signal - np.min(signal)) < 0.1 * signal_range)
    near_zero_ratio = near_zero_count / len(signal) if len(signal) > 0 else 0

    if pulse_width_ratio < 0.15 and near_zero_ratio > 0.7:
        return {
            "waveform_type": "impulse",
            "confidence": 0.85,
            "characteristics": {
                **characteristics,
                "pulse_width_ratio": float(pulse_width_ratio),
            },
        }

    # Sine wave: Low THD (<0.1), single dominant frequency
    if thd < 0.1:
        return {
            "waveform_type": "sine",
            "confidence": 0.9,
            "characteristics": characteristics,
        }

    # Early check for PWM: estimate duty cycle first
    duty_cycle = _estimate_duty_cycle(signal)

    # PWM: Extreme duty cycle (far from 50%) with digital-like behavior
    if duty_cycle < 0.35 or duty_cycle > 0.65:  # Duty cycle far from 50%
        # Additional check: signal should be mostly at two levels (digital-like)
        unique_count = len(np.unique(signal))
        if unique_count <= 10:  # Few discrete levels = digital PWM
            return {
                "waveform_type": "pwm",
                "confidence": 0.8,
                "characteristics": {**characteristics, "duty_cycle": float(duty_cycle)},
            }

    # Square wave: Strong odd harmonics (1, 3, 5, 7...)
    # Theoretical ratios: 1/3, 1/5, 1/7, 1/9
    if len(harmonics) >= 2:
        # Check for strong 3rd harmonic and weak 2nd harmonic
        if harmonics[1] > 0.2 and harmonics[0] < 0.15:  # 3rd strong, 2nd weak
            # Additional check: duty cycle near 50%
            if 0.4 < duty_cycle < 0.6:
                return {
                    "waveform_type": "square",
                    "confidence": 0.85,
                    "characteristics": {**characteristics, "duty_cycle": float(duty_cycle)},
                }
            else:
                # Square wave with duty cycle != 50% = PWM
                return {
                    "waveform_type": "pwm",
                    "confidence": 0.8,
                    "characteristics": {**characteristics, "duty_cycle": float(duty_cycle)},
                }

    # Triangle wave: Strong odd harmonics with rapid decay (1/9, 1/25, ...)
    if len(harmonics) >= 2:
        # Check for rapidly decaying odd harmonics
        if (
            harmonics[1] > 0.05 and harmonics[1] < 0.15  # 3rd harmonic weaker than square
        ):
            return {
                "waveform_type": "triangle",
                "confidence": 0.75,
                "characteristics": characteristics,
            }

    # Sawtooth: Both odd and even harmonics (1/2, 1/3, 1/4, ...)
    if len(harmonics) >= 2:
        if harmonics[0] > 0.3 and harmonics[1] > 0.15:  # Both 2nd and 3rd strong
            return {
                "waveform_type": "sawtooth",
                "confidence": 0.75,
                "characteristics": characteristics,
            }

    # Unknown waveform type
    return {
        "waveform_type": "unknown",
        "confidence": 0.5,
        "characteristics": characteristics,
    }


def classify_signal(trace: WaveformTrace) -> dict[str, Any]:
    """Perform complete signal classification.

    Combines periodicity detection, waveform classification, and signal
    quality assessment to provide comprehensive signal characterization.

    Args:
        trace: Input waveform trace.

    Returns:
        Dictionary containing:
            - domain: "analog", "digital", or "mixed"
            - periodicity: Dict from detect_periodicity()
            - waveform: Dict from classify_waveform()
            - signal_quality: Dict with SNR, clipping detection, etc.

    Example:
        >>> from oscura.core.types import WaveformTrace, TraceMetadata
        >>> import numpy as np
        >>> # Sine wave example
        >>> t = np.linspace(0, 0.01, 1000)
        >>> data = 3.3 * np.sin(2 * np.pi * 1000 * t)
        >>> meta = TraceMetadata(sample_rate=1e5, units="V")
        >>> trace = WaveformTrace(data=data, metadata=meta)
        >>> result = classify_signal(trace)
        >>> print(f"Domain: {result['domain']}")
        >>> print(f"Type: {result['waveform']['waveform_type']}")
        >>> print(f"Periodic: {result['periodicity']['periodicity']}")
        Domain: analog
        Type: sine
        Periodic: periodic
    """
    signal = trace.data
    sample_rate = trace.metadata.sample_rate

    # Detect domain (analog vs digital vs mixed)
    domain = _detect_domain(signal)

    # Detect periodicity
    periodicity = detect_periodicity(signal, sample_rate)

    # Classify waveform shape
    waveform = classify_waveform(signal, sample_rate)

    # Assess signal quality
    signal_quality = _assess_signal_quality(signal)

    return {
        "domain": domain,
        "periodicity": periodicity,
        "waveform": waveform,
        "signal_quality": signal_quality,
    }


# =============================================================================
# Helper Functions
# =============================================================================


def _detect_domain(signal: NDArray[np.floating[Any]]) -> str:
    """Detect if signal is analog, digital, or mixed.

    Args:
        signal: Input signal array.

    Returns:
        "analog", "digital", or "mixed"
    """
    if len(signal) < 3:
        return "analog"

    # Count unique values
    unique_values = np.unique(signal)

    # If only 2 unique values, likely digital
    if len(unique_values) <= 2:
        return "digital"

    # Check if signal spends most time at discrete levels (digital-like)
    # Build histogram and check if most samples are at a few discrete levels
    hist, _ = np.histogram(signal, bins=50)
    # If >80% of samples are in top 3 bins, likely digital or mixed
    sorted_bins = np.sort(hist)[::-1]
    top_3_ratio = np.sum(sorted_bins[:3]) / len(signal) if len(signal) > 0 else 0

    if top_3_ratio > 0.8:
        # Check if there are transitions (if yes, digital; if no, DC)
        if len(unique_values) <= 5:
            return "digital"
        else:
            return "mixed"

    return "analog"


def _assess_signal_quality(signal: NDArray[np.floating[Any]]) -> dict[str, Any]:
    """Assess signal quality (SNR, clipping, etc.).

    Args:
        signal: Input signal array.

    Returns:
        Dictionary with quality metrics.
    """
    if len(signal) < 3:
        return {
            "snr": 0.0,
            "clipping_detected": False,
            "noise_level": 0.0,
        }

    # Estimate SNR (simplified)
    signal_power = float(np.mean(signal**2))
    noise_estimate = float(np.std(signal) * 0.1)  # Rough estimate
    snr = 10 * np.log10(signal_power / noise_estimate**2) if noise_estimate > 0 else 100.0

    # Detect clipping (signal at rail)
    signal_min = float(np.min(signal))
    signal_max = float(np.max(signal))
    signal_range = signal_max - signal_min

    # Check if many samples are at min or max
    threshold = signal_range * 0.01  # Within 1% of rail
    at_min = np.sum(signal <= (signal_min + threshold))
    at_max = np.sum(signal >= (signal_max - threshold))
    clipping_ratio = (at_min + at_max) / len(signal)

    clipping_detected = clipping_ratio > 0.1  # >10% of samples at rail

    return {
        "snr": float(snr),
        "clipping_detected": bool(clipping_detected),
        "noise_level": float(noise_estimate),
        "dynamic_range": float(signal_range),
    }


def _estimate_duty_cycle(signal: NDArray[np.floating[Any]]) -> float:
    """Estimate duty cycle from signal.

    Args:
        signal: Input signal array.

    Returns:
        Duty cycle as ratio (0.0 to 1.0).
    """
    if len(signal) < 3:
        return 0.5

    # Find signal levels using histogram
    hist, bin_edges = np.histogram(signal, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in lower and upper halves
    mid_idx = len(hist) // 2
    low_idx = np.argmax(hist[:mid_idx]) if mid_idx > 0 else 0
    high_idx = mid_idx + np.argmax(hist[mid_idx:])

    low = bin_centers[low_idx]
    high = bin_centers[high_idx]

    # Calculate threshold at midpoint
    mid = (low + high) / 2

    # Count samples above threshold
    above_threshold = np.sum(signal >= mid)
    duty_cycle = above_threshold / len(signal) if len(signal) > 0 else 0.5

    return float(duty_cycle)


__all__ = [
    "classify_signal",
    "classify_waveform",
    "detect_periodicity",
]
