"""Periodic pattern detection using multiple algorithms.

This module implements robust periodic pattern detection for digital signals
and binary data using autocorrelation, FFT spectral analysis, and suffix array
techniques.


Author: Oscura Development Team
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class PeriodResult:
    """Result of period detection.

    Attributes:
        period_samples: Period in number of samples
        period_seconds: Period in seconds (if sample_rate provided)
        frequency_hz: Fundamental frequency in Hz
        confidence: Detection confidence (0-1)
        method: Detection method used
        harmonics: List of detected harmonic frequencies (optional)
    """

    period_samples: float
    period_seconds: float
    frequency_hz: float
    confidence: float
    method: str
    harmonics: list[float] | None = field(default=None)

    # Alias for compatibility with tests
    @property
    def period(self) -> float:
        """Alias for period_samples for test compatibility."""
        return self.period_samples

    def __post_init__(self) -> None:
        """Validate period result values."""
        if self.period_samples <= 0:
            raise ValueError("period_samples must be positive")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("confidence must be in range [0, 1]")


def detect_period(
    trace: NDArray[np.float64],
    sample_rate: float = 1.0,
    method: Literal["auto", "autocorr", "fft", "suffix"] = "auto",
    min_period: int = 2,
    max_period: int | None = None,
) -> PeriodResult | None:
    """Detect dominant period in signal using best available method.

    : Periodic Pattern Detection

    This function automatically selects the most appropriate algorithm based
    on signal characteristics or uses the specified method.

    Args:
        trace: Input signal array (1D)
        sample_rate: Sampling rate in Hz (default: 1.0)
        method: Detection method ('auto', 'autocorr', 'fft', 'suffix')
        min_period: Minimum period to detect in samples
        max_period: Maximum period to detect in samples (None = len(trace)//2)

    Returns:
        PeriodResult with detected period information, or None if no period found

    Raises:
        ValueError: If trace is empty, min_period invalid, or parameters inconsistent

    Examples:
        >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        >>> result = detect_period(signal, sample_rate=1000.0)
        >>> print(f"Period: {result.period_seconds:.3f}s, Freq: {result.frequency_hz:.1f}Hz")
    """
    # Input validation
    if trace.size == 0:
        raise ValueError("trace cannot be empty")
    if min_period < 2:
        raise ValueError("min_period must be at least 2")
    if max_period is not None and max_period < min_period:
        raise ValueError("max_period must be >= min_period")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    # Ensure 1D array
    trace = np.asarray(trace).flatten()

    # Set default max_period
    if max_period is None:
        max_period = len(trace) // 2
    max_period = min(max_period, len(trace) // 2)

    # Auto-select method based on signal characteristics
    if method == "auto":
        # Use FFT for longer signals (more efficient)
        # Use autocorr for shorter signals or binary data
        if len(trace) > 10000:
            method = "fft"
        elif np.all(np.isin(trace, [0, 1])):
            method = "autocorr"  # Better for binary signals
        else:
            method = "fft"

    # Dispatch to appropriate method
    if method == "autocorr":
        results = detect_periods_autocorr(trace, sample_rate, max_period, min_correlation=0.5)
        return results[0] if results else None

    elif method == "fft":
        min_freq = sample_rate / max_period if max_period else None
        max_freq = sample_rate / min_period if min_period > 0 else None
        results = detect_periods_fft(trace, sample_rate, min_freq, max_freq, num_peaks=5)
        return results[0] if results else None

    elif method == "suffix":
        # Suffix array method for exact repeats
        period_samples = _detect_period_suffix(trace, min_period, max_period)
        if period_samples is None:
            return None
        return PeriodResult(
            period_samples=float(period_samples),
            period_seconds=period_samples / sample_rate,
            frequency_hz=sample_rate / period_samples,
            confidence=0.9,  # High confidence for exact matches
            method="suffix",
        )
    else:
        raise ValueError(f"Unknown method: {method}")


def detect_periods_fft(
    trace: NDArray[np.float64],
    sample_rate: float = 1.0,
    min_freq: float | None = None,
    max_freq: float | None = None,
    num_peaks: int = 5,
) -> list[PeriodResult]:
    """Detect periods using FFT spectral analysis.

    Uses power spectral density to identify dominant frequencies and their
    harmonics. More efficient than autocorrelation for long signals.

    Args:
        trace: Input signal array (1D)
        sample_rate: Sampling rate in Hz
        min_freq: Minimum frequency to consider (Hz)
        max_freq: Maximum frequency to consider (Hz)
        num_peaks: Maximum number of peaks to return

    Returns:
        List of PeriodResult sorted by confidence (strongest first)

    Examples:
        >>> signal = np.sin(2*np.pi*10*np.linspace(0, 1, 1000))
        >>> periods = detect_periods_fft(signal, sample_rate=1000.0, num_peaks=3)
    """
    trace = np.asarray(trace).flatten()
    if trace.size == 0:
        return []

    power, freqs = _compute_power_spectrum(trace, sample_rate)
    peak_indices = _find_valid_peaks(power, freqs, min_freq, max_freq, num_peaks)

    if len(peak_indices) == 0:
        return []

    return _build_period_results(peak_indices, freqs, power, sample_rate)


def _compute_power_spectrum(
    trace: NDArray[np.float64], sample_rate: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute power spectrum from FFT.

    Args:
        trace: Input signal array.
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (power spectrum, frequencies).
    """
    trace_centered = trace - np.mean(trace)
    n = len(trace_centered)

    # Apply Hanning window to reduce spectral leakage (improves noise robustness)
    window = np.hanning(n)
    trace_windowed = trace_centered * window

    fft_result = np.fft.rfft(trace_windowed)
    power = np.asarray(np.abs(fft_result) ** 2, dtype=np.float64)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    return power, freqs


def _find_valid_peaks(
    power: NDArray[np.float64],
    freqs: NDArray[np.float64],
    min_freq: float | None,
    max_freq: float | None,
    num_peaks: int,
) -> NDArray[np.intp]:
    """Find valid spectral peaks within frequency range.

    Args:
        power: Power spectrum array.
        freqs: Frequency array.
        min_freq: Minimum frequency.
        max_freq: Maximum frequency.
        num_peaks: Maximum peaks to return.

    Returns:
        Array of peak indices sorted by power.
    """
    valid_mask = np.ones(len(freqs), dtype=bool)
    if min_freq is not None:
        valid_mask &= freqs >= min_freq
    if max_freq is not None:
        valid_mask &= freqs <= max_freq
    valid_mask[0] = False  # Exclude DC

    peak_indices = _find_spectral_peaks(power, min_distance=1)
    peak_indices = peak_indices[valid_mask[peak_indices]]

    if len(peak_indices) == 0:
        empty_result: NDArray[np.signedinteger[Any]] = peak_indices
        return empty_result

    peak_powers = power[peak_indices]
    sorted_indices = np.argsort(peak_powers)[::-1][:num_peaks]
    result: NDArray[np.signedinteger[Any]] = peak_indices[sorted_indices]
    return result


def _build_period_results(
    peak_indices: NDArray[np.intp],
    freqs: NDArray[np.float64],
    power: NDArray[np.float64],
    sample_rate: float,
) -> list[PeriodResult]:
    """Build PeriodResult objects from spectral peaks.

    Args:
        peak_indices: Array of peak indices.
        freqs: Frequency array.
        power: Power spectrum array.
        sample_rate: Sampling rate in Hz.

    Returns:
        List of PeriodResult objects.
    """
    results = []
    max_power = np.max(power[peak_indices]) if len(peak_indices) > 0 else 1.0

    for idx in peak_indices:
        freq = freqs[idx]
        if freq == 0:
            continue

        harmonics = _detect_harmonics(freq, freqs, power, power[idx])

        results.append(
            PeriodResult(
                period_samples=sample_rate / freq,
                period_seconds=1.0 / freq,
                frequency_hz=freq,
                confidence=min(float(power[idx] / max_power), 1.0),
                method="fft",
                harmonics=harmonics if harmonics else None,
            )
        )

    return results


def _detect_harmonics(
    freq: float, freqs: NDArray[np.float64], power: NDArray[np.float64], base_power: float
) -> list[float]:
    """Detect harmonic frequencies.

    Args:
        freq: Base frequency.
        freqs: Frequency array.
        power: Power spectrum.
        base_power: Power at base frequency.

    Returns:
        List of harmonic frequencies.
    """
    harmonics = []
    for mult in range(2, 6):
        harmonic_freq = freq * mult
        if harmonic_freq < freqs[-1]:
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if power[harmonic_idx] > 0.1 * base_power:
                harmonics.append(harmonic_freq)
    return harmonics


def detect_periods_autocorr(
    trace: NDArray[np.float64],
    sample_rate: float = 1.0,
    max_period: int | None = None,
    min_correlation: float = 0.5,
) -> list[PeriodResult]:
    """Detect periods using autocorrelation.

    : Periodic Pattern Detection via autocorrelation

    Computes normalized autocorrelation and finds peaks corresponding to
    periodic patterns. More robust to noise than simple pattern matching.

    Args:
        trace: Input signal array (1D)
        sample_rate: Sampling rate in Hz
        max_period: Maximum period to search (samples)
        min_correlation: Minimum correlation threshold (0-1)

    Returns:
        List of PeriodResult sorted by confidence

    Examples:
        >>> signal = np.tile([1, 0, 1, 0], 100)
        >>> periods = detect_periods_autocorr(signal, min_correlation=0.7)
    """
    trace = np.asarray(trace).flatten()

    if trace.size == 0:
        return []

    # Set default max_period
    if max_period is None:
        max_period = len(trace) // 2
    max_period = min(max_period, len(trace) // 2)

    # Compute normalized autocorrelation using FFT (efficient)
    trace_centered = trace - np.mean(trace)

    # Compute via FFT convolution
    n = len(trace_centered)
    fft_trace = np.fft.fft(trace_centered, n=2 * n)
    autocorr = np.fft.ifft(fft_trace * np.conj(fft_trace)).real[:n]

    # Normalize
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    # Limit search range
    autocorr = autocorr[: max_period + 1]

    # Find peaks (skip lag 0)
    peaks = _find_spectral_peaks(autocorr[1:], min_distance=1) + 1

    # Filter by minimum correlation
    peaks = peaks[autocorr[peaks] >= min_correlation]

    if len(peaks) == 0:
        return []

    # Sort by correlation value
    peak_corrs = autocorr[peaks]
    sorted_indices = np.argsort(peak_corrs)[::-1]
    peaks = peaks[sorted_indices]

    # Build results
    results = []
    for lag in peaks[:5]:  # Top 5 peaks
        period_samples = float(lag)
        period_seconds = period_samples / sample_rate
        frequency_hz = sample_rate / period_samples
        confidence = float(autocorr[lag])

        results.append(
            PeriodResult(
                period_samples=period_samples,
                period_seconds=period_seconds,
                frequency_hz=frequency_hz,
                confidence=confidence,
                method="autocorr",
            )
        )

    return results


def validate_period(
    trace: NDArray[np.float64], period: float, tolerance: float = 0.01
) -> tuple[bool, float]:
    """Validate detected period against signal.

    : Period validation

    Verifies that the signal actually repeats at the given period by measuring
    correlation between shifted copies of the signal.

    Args:
        trace: Input signal array
        period: Period to validate (in samples)
        tolerance: Allowed fractional deviation in period

    Returns:
        Tuple of (is_valid, actual_confidence)
        - is_valid: True if period is confirmed
        - actual_confidence: Measured correlation strength (0-1)

    Examples:
        >>> signal = np.tile([1, 2, 3, 4], 50)
        >>> is_valid, conf = validate_period(signal, period=4.0)
        >>> assert is_valid and conf > 0.95
    """
    trace = np.asarray(trace).flatten()

    if trace.size == 0:
        return False, 0.0

    if period < 1 or period >= len(trace):
        return False, 0.0

    # Convert to integer lag for nearest-neighbor validation
    lag = int(round(period))

    # Check if lag is within tolerance
    if abs(period - lag) > period * tolerance:
        # Use interpolation for sub-sample periods
        lag_low = int(np.floor(period))
        lag_high = int(np.ceil(period))
        alpha = period - lag_low

        if lag_high >= len(trace):
            lag = lag_low
        else:
            # Weighted average of both lags
            corr_low = _compute_lag_correlation(trace, lag_low)
            corr_high = _compute_lag_correlation(trace, lag_high)
            confidence = (1 - alpha) * corr_low + alpha * corr_high

            is_valid = confidence >= 0.5
            return is_valid, float(confidence)

    # Simple case: integer lag
    confidence = _compute_lag_correlation(trace, lag)
    is_valid = confidence >= 0.5

    return is_valid, float(confidence)


def _compute_lag_correlation(trace: NDArray[np.float64], lag: int) -> float:
    """Compute normalized correlation at specific lag.

    Args:
        trace: Input signal
        lag: Lag in samples

    Returns:
        Normalized correlation coefficient (0-1)
    """
    if lag <= 0 or lag >= len(trace):
        return 0.0

    # Center the signal
    trace_centered = trace - np.mean(trace)

    # Compute correlation
    n_overlap = len(trace) - lag
    part1 = trace_centered[:n_overlap]
    part2 = trace_centered[lag : lag + n_overlap]

    # Pearson correlation
    std1 = np.std(part1)
    std2 = np.std(part2)

    if std1 == 0 or std2 == 0:
        return 0.0

    correlation = np.mean(part1 * part2) / (std1 * std2)

    # Clamp to [0, 1] range
    return float(np.clip(correlation, 0, 1))


def _find_spectral_peaks(data: NDArray[np.float64], min_distance: int = 1) -> NDArray[np.intp]:
    """Find peaks in 1D array.

    Simple peak detection: point is peak if higher than neighbors.
    Includes noise threshold to filter out spurious peaks.
    Handles boundary conditions at edges of the array.

    Args:
        data: 1D array
        min_distance: Minimum distance between peaks

    Returns:
        Array of peak indices
    """
    if len(data) < 2:
        return np.array([], dtype=np.intp)

    # Calculate noise threshold (5% of max value to filter noise peaks)
    threshold = 0.05 * np.max(data)

    # Find local maxima above threshold
    peaks_list: list[int] = []

    # Check interior points (only elements with both neighbors)
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] > threshold:
            peaks_list.append(i)

    # Check boundary elements (for Nyquist frequency peaks)
    # Only consider boundary as peak if it's a TRUE local maximum (not just monotonic increase)
    # Require it to be significantly higher than neighbor (at least 2x threshold)
    if len(data) >= 3:
        # Check first element
        if data[0] > data[1] and data[0] > max(threshold * 2, data[1] * 1.5):
            peaks_list.insert(0, 0)
        # Check last element (important for Nyquist frequency in FFT)
        if data[-1] > data[-2] and data[-1] > max(threshold * 2, data[-2] * 1.5):
            peaks_list.append(len(data) - 1)

    peaks: NDArray[np.intp] = np.array(peaks_list, dtype=np.intp)

    # Apply minimum distance constraint
    if len(peaks) > 0 and min_distance > 1:
        # Keep highest peaks when too close
        filtered_peaks_list: list[int] = []
        last_peak = -min_distance

        # Sort by height
        sorted_indices = np.argsort(data[peaks])[::-1]

        for idx in sorted_indices:
            peak_pos = int(peaks[idx])
            if peak_pos - last_peak >= min_distance:
                filtered_peaks_list.append(peak_pos)
                last_peak = peak_pos

        peaks = np.array(np.sort(np.array(filtered_peaks_list, dtype=np.intp)), dtype=np.intp)

    return peaks


def _detect_period_suffix(
    trace: NDArray[np.float64], min_period: int, max_period: int
) -> int | None:
    """Detect period using suffix array (for exact repeats).

    : Suffix array-based period detection

    This method finds the longest exact repeating substring, which corresponds
    to the period for perfectly periodic signals.

    Args:
        trace: Input signal (will be converted to bytes)
        min_period: Minimum period length
        max_period: Maximum period length

    Returns:
        Period in samples, or None if not found
    """
    # Convert to byte sequence for suffix array
    if trace.dtype == np.bool_ or np.all(np.isin(trace, [0, 1])):
        # Binary signal
        trace_bytes = np.packbits(trace.astype(np.uint8))
    else:
        # Use raw bytes
        trace_bytes = trace.astype(np.uint8)

    n = len(trace_bytes)

    # Simple period detection: check for repeating patterns
    for period in range(min_period, min(max_period + 1, n // 2)):
        # Check if trace repeats with this period
        num_repeats = n // period
        if num_repeats < 2:
            continue

        # Compare first period with subsequent periods
        pattern = trace_bytes[:period]
        matches = 0

        for i in range(1, num_repeats):
            segment = trace_bytes[i * period : (i + 1) * period]
            if len(segment) == period and np.array_equal(pattern, segment):
                matches += 1

        # If most repetitions match, consider it valid
        if matches >= num_repeats * 0.8 - 1:
            return period

    return None


class PeriodicPatternDetector:
    """Object-oriented wrapper for periodic pattern detection.

    Provides a class-based interface for period detection operations,
    wrapping the functional API for consistency with test expectations.



    Example:
        >>> detector = PeriodicPatternDetector()
        >>> result = detector.detect_period(signal)
        >>> print(f"Period: {result.period} samples, confidence: {result.confidence}")
    """

    def __init__(
        self,
        method: Literal["auto", "autocorr", "fft", "autocorrelation"] = "auto",
        sample_rate: float = 1.0,
        min_period: int = 2,
        max_period: int | None = None,
    ):
        """Initialize periodic pattern detector.

        Args:
            method: Detection method ('auto', 'autocorr', 'fft', 'autocorrelation').
            sample_rate: Sample rate in Hz.
            min_period: Minimum period in samples.
            max_period: Maximum period in samples.
        """
        # Map 'autocorrelation' to 'autocorr' for test compatibility
        if method == "autocorrelation":
            method = "autocorr"
        self.method = method
        self.sample_rate = sample_rate
        self.min_period = min_period
        self.max_period = max_period

    def detect_period(self, trace: NDArray[np.float64]) -> PeriodResult:
        """Detect the dominant period in the signal.

        Args:
            trace: Input signal array (1D, boolean or numeric).

        Returns:
            PeriodResult with detected period information.

        Raises:
            ValueError: If trace is empty or too short.

        Example:
            >>> detector = PeriodicPatternDetector(method="autocorr")
            >>> result = detector.detect_period(np.tile([1, 0], 100))
            >>> result.period == 2
            True
        """
        trace = np.asarray(trace).flatten()

        # Validate input
        if trace.size == 0:
            raise ValueError("trace cannot be empty")
        if trace.size < 3:
            raise ValueError("trace must have at least 3 elements for period detection")

        result = detect_period(
            trace,
            sample_rate=self.sample_rate,
            method=self.method,
            min_period=self.min_period,
            max_period=self.max_period,
        )

        if result is None:
            # Return a low-confidence result if no period found
            return PeriodResult(
                period_samples=1.0,
                period_seconds=1.0 / self.sample_rate,
                frequency_hz=self.sample_rate,
                confidence=0.0,
                method=self.method,
            )

        return result

    def detect_multiple_periods(
        self, trace: NDArray[np.float64], num_periods: int = 5
    ) -> list[PeriodResult]:
        """Detect multiple periods in the signal.

        Args:
            trace: Input signal array.
            num_periods: Maximum number of periods to return.

        Returns:
            List of PeriodResult sorted by confidence.
        """
        trace = np.asarray(trace).flatten()

        if self.method in ["fft", "auto"]:
            min_freq = self.sample_rate / self.max_period if self.max_period else None
            max_freq = self.sample_rate / self.min_period if self.min_period > 0 else None
            return detect_periods_fft(trace, self.sample_rate, min_freq, max_freq, num_periods)
        else:
            return detect_periods_autocorr(
                trace, self.sample_rate, self.max_period, min_correlation=0.3
            )[:num_periods]

    def validate(self, trace: NDArray[np.float64], period: float, tolerance: float = 0.01) -> bool:
        """Validate a detected period.

        Args:
            trace: Input signal array.
            period: Period to validate.
            tolerance: Tolerance for period matching.

        Returns:
            True if period is valid.
        """
        is_valid, _ = validate_period(trace, period, tolerance)
        return is_valid


__all__ = [
    "PeriodResult",
    "PeriodicPatternDetector",
    "detect_period",
    "detect_periods_autocorr",
    "detect_periods_fft",
    "validate_period",
]
