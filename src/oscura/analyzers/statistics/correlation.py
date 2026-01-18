"""Correlation analysis for signal data.

This module provides autocorrelation, cross-correlation, and related
analysis functions for identifying signal relationships and periodicities.


Example:
    >>> from oscura.analyzers.statistics.correlation import (
    ...     autocorrelation, cross_correlation, correlate_chunked
    ... )
    >>> acf = autocorrelation(trace, max_lag=1000)
    >>> xcorr, lag, coef = cross_correlation(trace1, trace2)
    >>> # Memory-efficient correlation for large signals
    >>> result = correlate_chunked(large_signal1, large_signal2)

References:
    Oppenheim, A. V. & Schafer, R. W. (2009). Discrete-Time Signal Processing
    IEEE 1241-2010: Standard for Terminology and Test Methods for ADCs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CrossCorrelationResult:
    """Result of cross-correlation analysis.

    Attributes:
        correlation: Full correlation array.
        lags: Lag values in samples.
        lag_times: Lag values in seconds.
        peak_lag: Lag at maximum correlation (samples).
        peak_lag_time: Lag at maximum correlation (seconds).
        peak_coefficient: Maximum correlation coefficient.
        sample_rate: Sample rate used for time conversion.
    """

    correlation: NDArray[np.float64]
    lags: NDArray[np.intp]
    lag_times: NDArray[np.float64]
    peak_lag: int
    peak_lag_time: float
    peak_coefficient: float
    sample_rate: float


def autocorrelation(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    max_lag: int | None = None,
    normalized: bool = True,
    sample_rate: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute autocorrelation of a signal.

    Measures self-similarity of a signal at different time lags.
    Useful for detecting periodicities and characteristic time scales.

    Args:
        trace: Input trace or numpy array.
        max_lag: Maximum lag to compute (samples). If None, uses n // 2.
        normalized: If True, normalize to correlation coefficients [-1, 1].
        sample_rate: Sample rate in Hz (for time axis). Required if trace is array.

    Returns:
        Tuple of (lags_time, autocorrelation):
            - lags_time: Time values for each lag in seconds
            - autocorrelation: Normalized autocorrelation values

    Raises:
        ValueError: If sample_rate is not provided when trace is array.

    Example:
        >>> lag_times, acf = autocorrelation(trace, max_lag=1000)
        >>> # Find first zero crossing for decorrelation time
        >>> zero_idx = np.where(acf[1:] < 0)[0][0]
        >>> decorr_time = lag_times[zero_idx]

    References:
        Box, G. E. P. & Jenkins, G. M. (1976). Time Series Analysis
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data
        fs = trace.metadata.sample_rate
    else:
        data = trace
        if sample_rate is None:
            raise ValueError("sample_rate required when trace is array")
        fs = sample_rate

    n = len(data)

    if max_lag is None:
        max_lag = n // 2

    max_lag = min(max_lag, n - 1)

    # Remove mean for proper correlation
    data_centered = data - np.mean(data)

    # Compute autocorrelation via FFT (faster for large n)
    if n > 256:
        # Zero-pad for full correlation
        nfft = int(2 ** np.ceil(np.log2(2 * n)))
        fft_data = np.fft.rfft(data_centered, n=nfft)
        acf_full = np.fft.irfft(fft_data * np.conj(fft_data), n=nfft)
        acf = acf_full[: max_lag + 1]
    else:
        # Direct computation for small n
        acf = np.correlate(data_centered, data_centered, mode="full")
        acf = acf[n - 1 : n + max_lag]

    # Normalize
    if normalized and acf[0] > 0:
        acf = acf / acf[0]

    # Time axis
    lags = np.arange(max_lag + 1)
    lag_times = lags / fs

    return lag_times, acf.astype(np.float64)


def cross_correlation(
    trace1: WaveformTrace | NDArray[np.floating[Any]],
    trace2: WaveformTrace | NDArray[np.floating[Any]],
    *,
    max_lag: int | None = None,
    normalized: bool = True,
    sample_rate: float | None = None,
) -> CrossCorrelationResult:
    """Compute cross-correlation between two signals.

    Measures similarity between signals at different time lags.
    Useful for finding time delays, alignments, and relationships.

    Args:
        trace1: First input trace or numpy array (reference).
        trace2: Second input trace or numpy array.
        max_lag: Maximum lag to compute (samples). If None, uses min(n1, n2) // 2.
        normalized: If True, normalize to correlation coefficients [-1, 1].
        sample_rate: Sample rate in Hz. Required if traces are arrays.

    Returns:
        CrossCorrelationResult with correlation data and optimal lag.

    Raises:
        ValueError: If sample_rate is not provided when traces are arrays.

    Example:
        >>> result = cross_correlation(trace1, trace2)
        >>> print(f"Optimal lag: {result.peak_lag_time * 1e6:.1f} us")
        >>> print(f"Correlation: {result.peak_coefficient:.3f}")

    References:
        Oppenheim, A. V. & Schafer, R. W. (2009). Discrete-Time Signal Processing
    """
    if isinstance(trace1, WaveformTrace):
        data1 = trace1.data
        fs = trace1.metadata.sample_rate
    else:
        data1 = trace1
        if sample_rate is None:
            raise ValueError("sample_rate required when traces are arrays")
        fs = sample_rate

    if isinstance(trace2, WaveformTrace):
        data2 = trace2.data
        # Use trace2 sample rate if available and trace1 wasn't a WaveformTrace
        if not isinstance(trace1, WaveformTrace):
            fs = trace2.metadata.sample_rate
    else:
        data2 = trace2

    n1, n2 = len(data1), len(data2)

    if max_lag is None:
        max_lag = min(n1, n2) // 2

    # Center the data
    data1_centered = data1 - np.mean(data1)
    data2_centered = data2 - np.mean(data2)

    # Full cross-correlation
    # Note: np.correlate(a, b) computes sum(a[n+k] * conj(b[k]))
    # For cross-correlation where we want to detect b delayed relative to a,
    # we need correlate(b, a) so positive lag means b is delayed
    xcorr_full = np.correlate(data2_centered, data1_centered, mode="full")

    # Extract relevant portion around zero lag
    # Full correlation has length n1 + n2 - 1, with zero lag at index n1 - 1
    # (since we swapped the order above)
    zero_lag_idx = n1 - 1
    start_idx = max(0, zero_lag_idx - max_lag)
    end_idx = min(len(xcorr_full), zero_lag_idx + max_lag + 1)
    xcorr = xcorr_full[start_idx:end_idx]

    # Create lag array
    lags = np.arange(start_idx - zero_lag_idx, end_idx - zero_lag_idx)

    # Normalize
    if normalized:
        norm1 = np.sqrt(np.sum(data1_centered**2))
        norm2 = np.sqrt(np.sum(data2_centered**2))
        if norm1 > 0 and norm2 > 0:
            xcorr = xcorr / (norm1 * norm2)

    # Find peak
    peak_local_idx = np.argmax(np.abs(xcorr))
    peak_lag = int(lags[peak_local_idx])
    peak_coefficient = float(xcorr[peak_local_idx])

    # Time values
    lag_times = lags / fs
    peak_lag_time = peak_lag / fs

    return CrossCorrelationResult(
        correlation=xcorr.astype(np.float64),
        lags=lags,
        lag_times=lag_times.astype(np.float64),
        peak_lag=peak_lag,
        peak_lag_time=peak_lag_time,
        peak_coefficient=peak_coefficient,
        sample_rate=fs,
    )


def correlation_coefficient(
    trace1: WaveformTrace | NDArray[np.floating[Any]],
    trace2: WaveformTrace | NDArray[np.floating[Any]],
) -> float:
    """Compute Pearson correlation coefficient between two signals.

    Simple measure of linear relationship between signals at zero lag.

    Args:
        trace1: First input trace or numpy array.
        trace2: Second input trace or numpy array.

    Returns:
        Correlation coefficient in range [-1, 1].

    Example:
        >>> r = correlation_coefficient(trace1, trace2)
        >>> print(f"Correlation: {r:.3f}")
    """
    data1 = trace1.data if isinstance(trace1, WaveformTrace) else trace1

    data2 = trace2.data if isinstance(trace2, WaveformTrace) else trace2

    # Ensure same length
    n = min(len(data1), len(data2))
    data1 = data1[:n]
    data2 = data2[:n]

    # Compute correlation
    return float(np.corrcoef(data1, data2)[0, 1])


def find_periodicity(
    trace: WaveformTrace | NDArray[np.floating[Any]],
    *,
    min_period_samples: int = 2,
    max_period_samples: int | None = None,
    sample_rate: float | None = None,
) -> dict[str, float | int | list[dict[str, int | float]]]:
    """Find dominant periodicity in signal using autocorrelation.

    Detects the primary periodic component by finding the first
    significant peak in the autocorrelation function.

    Args:
        trace: Input trace or numpy array.
        min_period_samples: Minimum period to consider (samples).
        max_period_samples: Maximum period to consider (samples).
        sample_rate: Sample rate in Hz (required for array input).

    Returns:
        Dictionary with periodicity analysis:
            - period_samples: Period in samples
            - period_time: Period in seconds
            - frequency: Frequency in Hz
            - strength: Autocorrelation at period (0-1)
            - harmonics: List of detected harmonics

    Raises:
        ValueError: If sample_rate is not provided when trace is array.

    Example:
        >>> result = find_periodicity(trace)
        >>> print(f"Period: {result['period_time']*1e6:.2f} us")
        >>> print(f"Frequency: {result['frequency']/1e3:.1f} kHz")
    """
    if isinstance(trace, WaveformTrace):
        data = trace.data
        fs = trace.metadata.sample_rate
    else:
        data = trace
        if sample_rate is None:
            raise ValueError("sample_rate required when trace is array")
        fs = sample_rate

    n = len(data)

    if max_period_samples is None:
        max_period_samples = n // 2

    # Compute autocorrelation
    _lag_times, acf = autocorrelation(
        trace,
        max_lag=max_period_samples,
        sample_rate=sample_rate if sample_rate else fs,
    )

    # Find peaks in autocorrelation (after lag 0)
    # Look for local maxima
    acf_search = acf[min_period_samples:]

    if len(acf_search) < 3:
        return {
            "period_samples": np.nan,
            "period_time": np.nan,
            "frequency": np.nan,
            "strength": np.nan,
            "harmonics": [],
        }

    # Find local maxima
    local_max = (acf_search[1:-1] > acf_search[:-2]) & (acf_search[1:-1] > acf_search[2:])
    max_indices = np.where(local_max)[0] + 1  # +1 for offset from [1:-1]

    if len(max_indices) == 0:
        # No local maxima found, use global max
        primary_idx = int(np.argmax(acf_search)) + min_period_samples
        strength = float(acf[primary_idx])
    else:
        # Find strongest peak
        peak_values = acf_search[max_indices]
        best_peak_idx = int(np.argmax(peak_values))
        primary_idx = int(max_indices[best_peak_idx]) + min_period_samples
        strength = float(acf[primary_idx])

    period_samples = int(primary_idx)
    period_time = period_samples / fs
    frequency = 1.0 / period_time if period_time > 0 else np.nan

    # Find harmonics (peaks at multiples of period)
    harmonics: list[dict[str, int | float]] = []
    for h in range(2, 6):  # Check up to 5th harmonic
        harmonic_lag = h * period_samples
        if harmonic_lag < len(acf):
            # Look for peak near expected harmonic
            search_range = max(1, period_samples // 4)
            start = int(max(0, harmonic_lag - search_range))
            end = int(min(len(acf), harmonic_lag + search_range))
            local_max_idx = int(start + int(np.argmax(acf[start:end])))
            harmonic_strength = float(acf[local_max_idx])

            if harmonic_strength > 0.3:  # Threshold for significant harmonic
                harmonics.append(
                    {
                        "harmonic": h,
                        "lag_samples": local_max_idx,
                        "strength": harmonic_strength,
                    }
                )

    return {
        "period_samples": period_samples,
        "period_time": float(period_time),
        "frequency": float(frequency),
        "strength": strength,
        "harmonics": harmonics,
    }


def coherence(
    trace1: WaveformTrace | NDArray[np.floating[Any]],
    trace2: WaveformTrace | NDArray[np.floating[Any]],
    *,
    nperseg: int | None = None,
    sample_rate: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute magnitude-squared coherence between two signals.

    Measures frequency-domain correlation between signals.
    Coherence of 1 indicates perfect linear relationship at that frequency.

    Args:
        trace1: First input trace or numpy array.
        trace2: Second input trace or numpy array.
        nperseg: Segment length for estimation. If None, auto-selected.
        sample_rate: Sample rate in Hz (required for array input).

    Returns:
        Tuple of (frequencies, coherence):
            - frequencies: Frequency values in Hz
            - coherence: Magnitude-squared coherence [0, 1]

    Raises:
        ValueError: If sample_rate is not provided when traces are arrays.

    Example:
        >>> freq, coh = coherence(trace1, trace2)
        >>> # Find frequencies with high coherence
        >>> high_coh_freqs = freq[coh > 0.8]
    """
    from scipy import signal as sp_signal

    if isinstance(trace1, WaveformTrace):
        data1 = trace1.data
        fs = trace1.metadata.sample_rate
    else:
        data1 = trace1
        if sample_rate is None:
            raise ValueError("sample_rate required when traces are arrays")
        fs = sample_rate

    data2 = trace2.data if isinstance(trace2, WaveformTrace) else trace2

    # Ensure same length
    n = min(len(data1), len(data2))
    data1 = data1[:n]
    data2 = data2[:n]

    if nperseg is None:
        nperseg = min(256, n // 4)
        nperseg = max(nperseg, 16)

    freq, coh = sp_signal.coherence(data1, data2, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)

    return freq, coh.astype(np.float64)


def correlate_chunked(
    signal1: NDArray[np.floating[Any]],
    signal2: NDArray[np.floating[Any]],
    *,
    mode: str = "same",
    chunk_size: int | None = None,
) -> NDArray[np.float64]:
    """Memory-efficient cross-correlation using overlap-save FFT method.

    Computes cross-correlation for large signals that don't fit in memory
    by processing in chunks using the overlap-save method with FFT.

    Args:
        signal1: First input signal array.
        signal2: Second input signal array (kernel/template).
        mode: Correlation mode - 'same', 'valid', or 'full' (default 'same').
        chunk_size: Size of chunks for processing. If None, auto-selected.

    Returns:
        Cross-correlation result with same semantics as numpy.correlate.

    Raises:
        ValueError: If signals are empty or mode is invalid.

    Example:
        >>> import numpy as np
        >>> # Large signals
        >>> signal1 = np.random.randn(100_000_000)
        >>> signal2 = np.random.randn(10_000)
        >>> # Memory-efficient correlation
        >>> result = correlate_chunked(signal1, signal2, mode='same')
        >>> print(f"Result shape: {result.shape}")

    Notes:
        Uses overlap-save FFT-based convolution which is memory-efficient
        and faster than direct correlation for large signals.

    References:
        MEM-008: Chunked Correlation
        Oppenheim & Schafer (2009): Discrete-Time Signal Processing, Ch 8
    """
    if len(signal1) == 0 or len(signal2) == 0:
        raise ValueError("Input signals cannot be empty")

    if mode not in ("same", "valid", "full"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'same', 'valid', or 'full'")

    n1 = len(signal1)
    n2 = len(signal2)

    # Determine chunk size
    if chunk_size is None:
        # Auto-select: aim for ~100MB chunks
        bytes_per_sample = 8  # float64
        target_bytes = 100 * 1024 * 1024
        chunk_size = min(target_bytes // bytes_per_sample, n1)
        # Round to power of 2 for FFT efficiency
        chunk_size = 2 ** int(np.log2(chunk_size))

    # Ensure chunk_size is larger than filter length for overlap-save
    # Otherwise overlap-save doesn't make sense
    min_chunk_size = max(2 * n2, 64)

    # For small signals or when chunk_size is too small, use direct method
    if n1 <= min_chunk_size or n2 >= n1 or chunk_size < min_chunk_size:
        mode_literal = cast("Literal['same', 'valid', 'full']", mode)
        result = np.correlate(signal1, signal2, mode=mode_literal)
        return result.astype(np.float64)

    # For correlation, we need to flip signal2
    signal2_flipped = signal2[::-1].copy()

    # Overlap-save parameters
    # L = chunk size, M = filter length
    L = max(chunk_size, min_chunk_size)
    M = n2
    overlap = M - 1

    # Ensure step size is positive (L must be > overlap)
    step_size = L - overlap
    if step_size <= 0:
        # Fall back to direct method if chunk is too small
        mode_literal = cast("Literal['same', 'valid', 'full']", mode)
        result = np.correlate(signal1, signal2, mode=mode_literal)
        return result.astype(np.float64)

    # FFT size (power of 2, >= L + M - 1)
    nfft = int(2 ** np.ceil(np.log2(L + M - 1)))

    # Pre-compute FFT of flipped signal2 (kernel)
    kernel_fft = np.fft.fft(signal2_flipped, n=nfft)

    # Output length based on mode
    if mode == "full":
        output_len = n1 + n2 - 1
    elif mode == "same":
        output_len = n1
    else:  # valid
        output_len = max(0, n1 - n2 + 1)

    # Initialize output
    output = np.zeros(output_len, dtype=np.float64)

    # Process chunks with overlap-save
    pos = 0  # Position in signal1
    max_iterations = (n1 // step_size) + 2  # Safety limit
    iteration = 0

    while pos < n1 and iteration < max_iterations:
        iteration += 1

        # Extract chunk with overlap from previous chunk
        if pos == 0:
            # First chunk: no overlap needed
            chunk_start = 0
            chunk = signal1[0 : min(L, n1)]
        else:
            # Subsequent chunks: include overlap
            chunk_start = max(0, pos - overlap)
            chunk_end = min(chunk_start + L, n1)
            chunk = signal1[chunk_start:chunk_end]

        # Zero-pad chunk to FFT size
        chunk_padded = np.zeros(nfft, dtype=np.float64)
        chunk_padded[: len(chunk)] = chunk

        # Perform FFT-based convolution
        chunk_fft = np.fft.fft(chunk_padded)
        conv_fft = chunk_fft * kernel_fft
        conv_result = np.fft.ifft(conv_fft).real

        # Extract valid portion (discard transient at start)
        if pos == 0:
            # First chunk
            valid_start = 0
            valid_end = min(L, len(conv_result))
        else:
            # Subsequent chunks: discard overlap region
            valid_start = overlap
            valid_end = min(len(chunk), len(conv_result))

        valid_output = conv_result[valid_start:valid_end]

        # Determine output range based on mode
        if mode == "full":
            # Full convolution includes all overlap
            out_start = pos
            out_end = min(out_start + len(valid_output), output_len)
        elif mode == "same":
            # Same mode: center-aligned
            offset = (M - 1) // 2
            out_start = max(0, pos - offset)
            out_end = min(out_start + len(valid_output), output_len)
            # Adjust valid_output if we're at boundaries
            if pos == 0 and offset > 0:
                valid_output = valid_output[offset:]
        else:  # valid
            # Valid mode: only where signals fully overlap
            offset = M - 1
            if pos < offset:
                # Skip this chunk, not in valid region yet
                pos += step_size
                continue
            out_start = pos - offset
            out_end = min(out_start + len(valid_output), output_len)

        # Copy to output
        copy_len = min(len(valid_output), out_end - out_start)
        if copy_len > 0:
            output[out_start : out_start + copy_len] = valid_output[:copy_len]

        # Move to next chunk with guaranteed progress
        pos += step_size

    return output


__all__ = [
    "CrossCorrelationResult",
    "autocorrelation",
    "coherence",
    "correlate_chunked",
    "correlation_coefficient",
    "cross_correlation",
    "find_periodicity",
]
