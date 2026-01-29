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

from oscura.core.numba_backend import njit, prange
from oscura.core.types import WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


@njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]  # Numba JIT decorator
def _autocorr_direct_numba(
    data: np.ndarray,  # type: ignore[type-arg]
    max_lag: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """Compute autocorrelation using direct method with Numba JIT compilation.

    Alternative implementation using Numba for autocorrelation computation.
    Uses parallel execution across lags for potential speedup on multi-core systems.

    **Note**: In practice, numpy.correlate is faster for most cases due to highly
    optimized BLAS/LAPACK routines. This function is provided for educational
    purposes and specific use cases where custom computation logic is needed.

    Args:
        data: Mean-centered input signal data (1D array).
        max_lag: Maximum lag to compute (inclusive).

    Returns:
        Autocorrelation values from lag 0 to max_lag (unnormalized).

    Performance characteristics:
        - First call: ~100-200ms compilation overhead (cached for subsequent calls)
        - Typical performance: ~2ms for n=100, ~10ms for n=256 (compiled)
        - NumPy's correlate: ~0.02ms for n=100 (100x faster due to BLAS)
        - Parallel execution: Benefits from multi-core for large max_lag

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(128) - np.mean(np.random.randn(128))
        >>> acf = _autocorr_direct_numba(data, max_lag=64)
        >>> print(acf.shape)  # (65,)

    Notes:
        - Input data should be mean-centered before calling this function
        - Result is NOT normalized; caller must normalize if needed
        - NumPy's correlate is recommended for production use (faster)
        - Thread safety: Numba releases GIL, safe for parallel execution

    References:
        Box, G. E. P. & Jenkins, G. M. (1976). Time Series Analysis
    """
    n = len(data)
    acf = np.zeros(max_lag + 1, dtype=np.float64)

    # Compute autocorrelation for each lag in parallel
    for lag in prange(max_lag + 1):
        sum_val = 0.0
        for i in range(n - lag):
            sum_val += data[i] * data[i + lag]
        acf[lag] = sum_val

    return acf


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

    This function automatically selects the optimal computation method:
    - Small signals (n < 256): Direct method using numpy.correlate (optimized BLAS)
    - Large signals (n >= 256): FFT-based method (O(n log n) complexity)

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

    Performance:
        - Small signals (n<256): ~0.02-0.05ms using numpy.correlate
        - Large signals (n>=256): ~0.1-1ms using FFT method
        - Both methods use highly optimized numerical libraries

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

    # Compute autocorrelation using optimal method based on signal size:
    # - n >= 256: FFT-based method (O(n log n) complexity, fastest for large signals)
    # - n < 256: Direct method using numpy.correlate (highly optimized BLAS)
    # Note: Numba implementation available (_autocorr_direct_numba) but numpy.correlate
    #       is faster due to optimized BLAS/LAPACK routines
    if n >= 256:
        # Zero-pad for full correlation
        nfft = int(2 ** np.ceil(np.log2(2 * n)))
        fft_data = np.fft.rfft(data_centered, n=nfft)
        acf_full = np.fft.irfft(fft_data * np.conj(fft_data), n=nfft)
        acf = acf_full[: max_lag + 1]
    else:
        # Direct computation for small signals (numpy.correlate uses optimized BLAS)
        acf = np.correlate(data_centered, data_centered, mode="full")
        acf = acf[n - 1 : n + max_lag]

    # Normalize
    if normalized and acf[0] > 0:
        acf = acf / acf[0]

    # Time axis
    lags = np.arange(max_lag + 1)
    lag_times = lags / fs

    return lag_times, acf.astype(np.float64)


def _extract_correlation_data(
    trace1: WaveformTrace | NDArray[np.floating[Any]],
    trace2: WaveformTrace | NDArray[np.floating[Any]],
    sample_rate: float | None,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], float]:
    """Extract data arrays and sample rate from traces.

    Args:
        trace1: First input trace or array
        trace2: Second input trace or array
        sample_rate: Sample rate if traces are arrays

    Returns:
        Tuple of (data1, data2, sample_rate)

    Raises:
        ValueError: If sample_rate needed but not provided
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
        if not isinstance(trace1, WaveformTrace):
            fs = trace2.metadata.sample_rate
    else:
        data2 = trace2

    return data1, data2, fs


def _compute_normalized_xcorr(
    data1_centered: NDArray[np.floating[Any]],
    data2_centered: NDArray[np.floating[Any]],
    xcorr: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Normalize cross-correlation to [-1, 1].

    Args:
        data1_centered: First centered data array
        data2_centered: Second centered data array
        xcorr: Raw cross-correlation

    Returns:
        Normalized cross-correlation
    """
    norm1 = np.sqrt(np.sum(data1_centered**2))
    norm2 = np.sqrt(np.sum(data2_centered**2))
    if norm1 > 0 and norm2 > 0:
        # Division returns proper NDArray type
        if TYPE_CHECKING:
            from typing import cast

            return cast("NDArray[np.floating[Any]]", xcorr / (norm1 * norm2))
        else:
            return xcorr / (norm1 * norm2)  # type: ignore[return-value]
    return xcorr


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
    data1, data2, fs = _extract_correlation_data(trace1, trace2, sample_rate)
    n1 = len(data1)
    max_lag = min(len(data1), len(data2)) // 2 if max_lag is None else max_lag

    # Center and compute correlation
    data1_centered = data1 - np.mean(data1)
    data2_centered = data2 - np.mean(data2)
    xcorr_full = np.correlate(data2_centered, data1_centered, mode="full")

    # Extract relevant portion
    zero_lag_idx = n1 - 1
    start_idx = max(0, zero_lag_idx - max_lag)
    end_idx = min(len(xcorr_full), zero_lag_idx + max_lag + 1)
    xcorr = xcorr_full[start_idx:end_idx]
    lags = np.arange(start_idx - zero_lag_idx, end_idx - zero_lag_idx)

    # Normalize if requested
    if normalized:
        xcorr = _compute_normalized_xcorr(data1_centered, data2_centered, xcorr)

    # Find peak
    peak_local_idx = np.argmax(np.abs(xcorr))
    peak_lag = int(lags[peak_local_idx])
    peak_coefficient = float(xcorr[peak_local_idx])

    return CrossCorrelationResult(
        correlation=xcorr.astype(np.float64),
        lags=lags,
        lag_times=(lags / fs).astype(np.float64),
        peak_lag=peak_lag,
        peak_lag_time=peak_lag / fs,
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


def _extract_periodicity_data(
    trace: WaveformTrace | NDArray[np.floating[Any]], sample_rate: float | None
) -> tuple[NDArray[np.floating[Any]], float]:
    """Extract data and sample rate from trace.

    Args:
        trace: Input trace or array.
        sample_rate: Sample rate if array.

    Returns:
        Tuple of (data, sample_rate).

    Raises:
        ValueError: If sample_rate not provided for array.
    """
    if isinstance(trace, WaveformTrace):
        return trace.data, trace.metadata.sample_rate
    else:
        if sample_rate is None:
            raise ValueError("sample_rate required when trace is array")
        return trace, sample_rate


def _find_primary_peak(acf: NDArray[np.float64], min_period_samples: int) -> tuple[int, float]:
    """Find primary peak in autocorrelation function.

    Args:
        acf: Autocorrelation function.
        min_period_samples: Minimum period to consider.

    Returns:
        Tuple of (period_samples, strength).
    """
    acf_search = acf[min_period_samples:]

    if len(acf_search) < 3:
        return -1, np.nan

    local_max = (acf_search[1:-1] > acf_search[:-2]) & (acf_search[1:-1] > acf_search[2:])
    max_indices = np.where(local_max)[0] + 1

    if len(max_indices) == 0:
        primary_idx = int(np.argmax(acf_search)) + min_period_samples
    else:
        peak_values = acf_search[max_indices]
        best_peak_idx = int(np.argmax(peak_values))
        primary_idx = int(max_indices[best_peak_idx]) + min_period_samples

    return primary_idx, float(acf[primary_idx])


def _find_harmonics(acf: NDArray[np.float64], period_samples: int) -> list[dict[str, int | float]]:
    """Find harmonic peaks at multiples of primary period.

    Args:
        acf: Autocorrelation function.
        period_samples: Primary period in samples.

    Returns:
        List of harmonic dictionaries.
    """
    harmonics: list[dict[str, int | float]] = []

    for h in range(2, 6):
        harmonic_lag = h * period_samples
        if harmonic_lag >= len(acf):
            break

        search_range = max(1, period_samples // 4)
        start = int(max(0, harmonic_lag - search_range))
        end = int(min(len(acf), harmonic_lag + search_range))
        local_max_idx = int(start + int(np.argmax(acf[start:end])))
        harmonic_strength = float(acf[local_max_idx])

        if harmonic_strength > 0.3:
            harmonics.append(
                {
                    "harmonic": h,
                    "lag_samples": local_max_idx,
                    "strength": harmonic_strength,
                }
            )

    return harmonics


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
    # Setup: extract data and compute autocorrelation
    data, fs = _extract_periodicity_data(trace, sample_rate)
    n = len(data)
    max_period_samples = max_period_samples if max_period_samples is not None else n // 2

    _lag_times, acf = autocorrelation(
        trace, max_lag=max_period_samples, sample_rate=sample_rate if sample_rate else fs
    )

    # Processing: find primary peak and harmonics
    period_samples, strength = _find_primary_peak(acf, min_period_samples)

    if period_samples < 0:
        return {
            "period_samples": np.nan,
            "period_time": np.nan,
            "frequency": np.nan,
            "strength": np.nan,
            "harmonics": [],
        }

    period_time = period_samples / fs
    frequency = 1.0 / period_time if period_time > 0 else np.nan
    harmonics = _find_harmonics(acf, period_samples)

    # Result building: construct result dictionary
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
    _validate_chunked_inputs(signal1, signal2, mode)

    n1, n2 = len(signal1), len(signal2)
    chunk_size_final = _determine_chunk_size(chunk_size, n1, n2)

    # Use direct correlation for small signals
    if _should_use_direct_method(n1, n2, chunk_size_final):
        return _direct_correlate(signal1, signal2, mode)

    # Setup overlap-save parameters
    params = _setup_overlap_save_params(chunk_size_final, n2)
    if params is None:
        return _direct_correlate(signal1, signal2, mode)

    # Prepare kernel FFT and output buffer
    signal2_flipped = signal2[::-1].copy()
    kernel_fft = np.fft.fft(signal2_flipped, n=params.nfft)
    output = np.zeros(_get_output_length(n1, n2, mode), dtype=np.float64)

    # Process signal in chunks
    _process_chunks_overlap_save(signal1, kernel_fft, output, params, mode)

    return output


def _validate_chunked_inputs(
    signal1: NDArray[np.floating[Any]], signal2: NDArray[np.floating[Any]], mode: str
) -> None:
    """Validate inputs for chunked correlation."""
    if len(signal1) == 0 or len(signal2) == 0:
        raise ValueError("Input signals cannot be empty")
    if mode not in ("same", "valid", "full"):
        raise ValueError(f"Invalid mode: {mode}. Must be 'same', 'valid', or 'full'")


def _determine_chunk_size(chunk_size: int | None, n1: int, n2: int) -> int:
    """Determine optimal chunk size for processing."""
    if chunk_size is not None:
        return chunk_size

    # Auto-select: aim for ~100MB chunks (float64 = 8 bytes)
    target_bytes = 100 * 1024 * 1024
    auto_chunk = min(target_bytes // 8, n1)
    log2_val: np.floating[Any] = np.log2(auto_chunk)
    chunk_power_of_2: int = int(2 ** int(log2_val))  # Round to power of 2
    return chunk_power_of_2


def _should_use_direct_method(n1: int, n2: int, chunk_size: int) -> bool:
    """Check if direct correlation should be used instead of chunked."""
    min_chunk_size = max(2 * n2, 64)
    return n1 <= min_chunk_size or n2 >= n1 or chunk_size < min_chunk_size


def _direct_correlate(
    signal1: NDArray[np.floating[Any]], signal2: NDArray[np.floating[Any]], mode: str
) -> NDArray[np.float64]:
    """Perform direct correlation using numpy."""
    mode_literal = cast("Literal['same', 'valid', 'full']", mode)
    result = np.correlate(signal1, signal2, mode=mode_literal)
    return result.astype(np.float64)


@dataclass
class _OverlapSaveParams:
    """Parameters for overlap-save algorithm."""

    chunk_size: int
    filter_len: int
    overlap: int
    step_size: int
    nfft: int


def _setup_overlap_save_params(chunk_size: int, n2: int) -> _OverlapSaveParams | None:
    """Setup overlap-save algorithm parameters."""
    min_chunk_size = max(2 * n2, 64)
    L = max(chunk_size, min_chunk_size)
    M = n2
    overlap = M - 1
    step_size = L - overlap

    if step_size <= 0:
        return None

    nfft = int(2 ** np.ceil(np.log2(L + M - 1)))
    return _OverlapSaveParams(L, M, overlap, step_size, nfft)


def _get_output_length(n1: int, n2: int, mode: str) -> int:
    """Calculate output length based on correlation mode."""
    if mode == "full":
        return n1 + n2 - 1
    elif mode == "same":
        return n1
    else:  # valid
        return max(0, n1 - n2 + 1)


def _process_chunks_overlap_save(
    signal1: NDArray[np.floating[Any]],
    kernel_fft: NDArray[np.complexfloating[Any, Any]],
    output: NDArray[np.float64],
    params: _OverlapSaveParams,
    mode: str,
) -> None:
    """Process signal in chunks using overlap-save method."""
    n1 = len(signal1)
    pos = 0
    max_iterations = (n1 // params.step_size) + 2

    for _iteration in range(max_iterations):
        if pos >= n1:
            break

        # Extract and process chunk
        chunk = _extract_chunk(signal1, pos, params, n1)
        conv_result = _convolve_chunk_fft(chunk, kernel_fft, params.nfft)
        valid_output = _extract_valid_portion(conv_result, pos, params)

        # Write to output buffer
        _write_chunk_output(output, valid_output, pos, params, mode)

        pos += params.step_size


def _extract_chunk(
    signal1: NDArray[np.floating[Any]], pos: int, params: _OverlapSaveParams, n1: int
) -> NDArray[np.floating[Any]]:
    """Extract chunk with appropriate overlap."""
    if pos == 0:
        return signal1[0 : min(params.chunk_size, n1)]

    chunk_start = max(0, pos - params.overlap)
    chunk_end = min(chunk_start + params.chunk_size, n1)
    return signal1[chunk_start:chunk_end]


def _convolve_chunk_fft(
    chunk: NDArray[np.floating[Any]],
    kernel_fft: NDArray[np.complexfloating[Any, Any]],
    nfft: int,
) -> NDArray[np.floating[Any]]:
    """Perform FFT-based convolution on chunk."""
    chunk_padded = np.zeros(nfft, dtype=np.float64)
    chunk_padded[: len(chunk)] = chunk
    chunk_fft = np.fft.fft(chunk_padded)
    conv_fft = chunk_fft * kernel_fft
    return np.fft.ifft(conv_fft).real


def _extract_valid_portion(
    conv_result: NDArray[np.floating[Any]], pos: int, params: _OverlapSaveParams
) -> NDArray[np.floating[Any]]:
    """Extract valid portion from convolution result."""
    if pos == 0:
        valid_start = 0
        valid_end = min(params.chunk_size, len(conv_result))
    else:
        valid_start = params.overlap
        valid_end = min(params.chunk_size, len(conv_result))

    return conv_result[valid_start:valid_end]


def _write_chunk_output(
    output: NDArray[np.float64],
    valid_output: NDArray[np.floating[Any]],
    pos: int,
    params: _OverlapSaveParams,
    mode: str,
) -> None:
    """Write chunk output to final buffer based on mode."""
    output_len = len(output)
    offset = (params.filter_len - 1) // 2 if mode == "same" else params.filter_len - 1

    if mode == "full":
        out_start = pos
    elif mode == "same":
        out_start = max(0, pos - offset)
        if pos == 0 and offset > 0:
            valid_output = valid_output[offset:]
    else:  # valid
        if pos < offset:
            return
        out_start = pos - offset

    out_end = min(out_start + len(valid_output), output_len)
    copy_len = min(len(valid_output), out_end - out_start)

    if copy_len > 0:
        output[out_start : out_start + copy_len] = valid_output[:copy_len]


__all__ = [
    "CrossCorrelationResult",
    "autocorrelation",
    "coherence",
    "correlate_chunked",
    "correlation_coefficient",
    "cross_correlation",
    "find_periodicity",
]
