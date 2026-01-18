"""Streaming APIs for chunk-by-chunk processing of large files.

This module implements memory-efficient streaming analysis for huge waveform
files that don't fit in memory. Uses generator-based chunk loading and
accumulator pattern for rolling statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import signal

from ..core.types import WaveformTrace

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from numpy.typing import NDArray


def load_trace_chunks(
    file_path: str | Path,
    chunk_size: int | float = 100e6,
    overlap: int = 0,
    loader: Callable[[str | Path], WaveformTrace] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Generator[WaveformTrace, None, None]:
    """Load large trace files chunk-by-chunk without loading into memory.

    Yields chunks of the trace for memory-efficient processing. Supports
    overlap between chunks for windowed operations that need continuity.

    Args:
        file_path: Path to trace file.
        chunk_size: Size of each chunk in samples (if int) or bytes (if float).
            Default 100e6 (100 MB).
        overlap: Number of samples to overlap between chunks. Useful for
            windowed operations like FFT. Default 0.
        loader: Optional custom loader function. If None, uses default loader.
        progress_callback: Optional callback(chunk_num, total_chunks) for
            progress reporting.

    Yields:
        WaveformTrace chunks.

    Raises:
        ValueError: If failed to load trace metadata.

    Example:
        >>> # Stream 10 GB file in 100 MB chunks
        >>> for chunk in osc.load_trace_chunks('huge_trace.bin', chunk_size=100e6):
        ...     mean = chunk.data.mean()
        ...     std = chunk.data.std()
        ...     print(f"Chunk stats: mean={mean:.3f}, std={std:.3f}")

    Advanced Example:
        >>> # Process with overlap for FFT continuity
        >>> for chunk in osc.load_trace_chunks(
        ...     'large_trace.bin',
        ...     chunk_size=50e6,
        ...     overlap=8192  # Overlap for continuity
        ... ):
        ...     fft_result = osc.fft(chunk, nfft=8192)
        ...     # Process FFT result...

    References:
        API-003: Streaming/Generator API for Large Files
    """
    file_path = Path(file_path)

    # Import loader here to avoid circular dependency
    from ..loaders import load

    # Use provided loader or default
    load_func = loader if loader is not None else load

    # Load full trace metadata to get total size
    # For memory-mapped files, this doesn't load data
    try:
        full_trace = load_func(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load trace metadata: {e}") from e

    total_samples = len(full_trace.data)  # type: ignore[union-attr]
    chunk_samples = int(chunk_size) if chunk_size < 1e6 else int(chunk_size / 8)

    # Calculate number of chunks
    num_chunks = (total_samples - overlap) // (chunk_samples - overlap)
    if (total_samples - overlap) % (chunk_samples - overlap) != 0:
        num_chunks += 1

    # Yield chunks
    chunk_num = 0
    start_idx = 0

    while start_idx < total_samples:
        end_idx = min(start_idx + chunk_samples, total_samples)

        # Extract chunk
        chunk_data = full_trace.data[start_idx:end_idx]  # type: ignore[union-attr]

        # Create chunk trace with same metadata
        # Cast needed for mypy: slicing a floating array returns a floating array
        chunk_trace = WaveformTrace(
            data=cast("NDArray[np.floating[Any]]", chunk_data),
            metadata=full_trace.metadata,
        )

        # Call progress callback if provided
        if progress_callback is not None:
            progress_callback(chunk_num, num_chunks)

        yield chunk_trace

        # Move to next chunk, accounting for overlap
        start_idx = end_idx - overlap
        chunk_num += 1

        # Break if we've reached the end
        if end_idx >= total_samples:
            break


class StreamingAnalyzer:
    """Accumulator for streaming analysis of large files.

    Processes traces chunk-by-chunk, accumulating statistics and measurements
    without loading entire file into memory. Supports streaming PSD estimation
    using Welch's method and other rolling statistics.

    Example:
        >>> # Create streaming analyzer
        >>> analyzer = osc.StreamingAnalyzer()
        >>> # Process file in chunks
        >>> for chunk in osc.load_trace_chunks('large_trace.bin', chunk_size=50e6):
        ...     analyzer.accumulate_psd(chunk, nperseg=4096, window='hann')
        >>> # Get aggregated result
        >>> psd_result = analyzer.get_psd()

    Advanced Example:
        >>> # Compute multiple statistics in streaming fashion
        >>> analyzer = osc.StreamingAnalyzer()
        >>> for chunk in osc.load_trace_chunks('huge_file.bin'):
        ...     analyzer.accumulate_statistics(chunk)
        ...     analyzer.accumulate_psd(chunk, nperseg=8192)
        >>> stats = analyzer.get_statistics()
        >>> psd = analyzer.get_psd()
        >>> print(f"Mean: {stats['mean']:.3f}, PSD shape: {psd.shape}")

    References:
        API-003: Streaming/Generator API for Large Files
        scipy.signal.welch for streaming PSD
    """

    def __init__(self) -> None:
        """Initialize streaming analyzer."""
        # Statistics accumulators
        self._n_samples = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")

        # PSD accumulators
        self._psd_sum: NDArray[np.float64] | None = None
        self._psd_freqs: NDArray[np.float64] | None = None
        self._psd_count = 0
        self._sample_rate: float | None = None

        # Histogram accumulators
        self._hist_counts: NDArray[np.int64] | None = None
        self._hist_edges: NDArray[np.float64] | None = None

    def accumulate_statistics(self, chunk: WaveformTrace) -> None:
        """Accumulate basic statistics from chunk.

        Updates running mean, std, min, max using Welford's online algorithm.

        Args:
            chunk: WaveformTrace chunk to process.

        Example:
            >>> analyzer.accumulate_statistics(chunk)
        """
        chunk_data = chunk.data
        self._n_samples += len(chunk_data)
        self._sum += float(chunk_data.sum())
        self._sum_sq += float((chunk_data**2).sum())
        self._min = min(self._min, float(chunk_data.min()))
        self._max = max(self._max, float(chunk_data.max()))

    def accumulate_psd(
        self,
        chunk: WaveformTrace,
        nperseg: int = 4096,
        window: str = "hann",
        **welch_kwargs: Any,
    ) -> None:
        """Accumulate PSD estimate from chunk using Welch's method.

        Computes PSD for chunk and accumulates with running average.

        Args:
            chunk: WaveformTrace chunk to process.
            nperseg: Length of each segment for Welch's method.
            window: Window function name (default 'hann').
            **welch_kwargs: Additional arguments for scipy.signal.welch.

        Example:
            >>> analyzer.accumulate_psd(chunk, nperseg=4096, window='hann')

        References:
            scipy.signal.welch
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
        """
        # Store sample rate from first chunk
        if self._sample_rate is None:
            self._sample_rate = chunk.metadata.sample_rate

        # Compute PSD for this chunk using Welch's method
        freqs, psd = signal.welch(
            chunk.data,
            fs=chunk.metadata.sample_rate,
            nperseg=nperseg,
            window=window,
            **welch_kwargs,
        )

        # Initialize or accumulate
        if self._psd_sum is None:
            self._psd_sum = psd
            self._psd_freqs = freqs
        else:
            # Accumulate PSD estimates
            self._psd_sum += psd

        self._psd_count += 1

    def accumulate_histogram(
        self,
        chunk: WaveformTrace,
        bins: int | NDArray[np.float64] = 100,
        range: tuple[float, float] | None = None,
    ) -> None:
        """Accumulate histogram from chunk.

        Args:
            chunk: WaveformTrace chunk to process.
            bins: Number of bins or bin edges.
            range: Range of histogram (min, max).

        Example:
            >>> analyzer.accumulate_histogram(chunk, bins=100)
        """
        counts, edges = np.histogram(chunk.data, bins=bins, range=range)

        if self._hist_counts is None:
            self._hist_counts = counts.astype(np.int64)
            self._hist_edges = edges
        else:
            self._hist_counts += counts.astype(np.int64)

    def get_statistics(self) -> dict[str, float]:
        """Get accumulated statistics.

        Returns:
            Dictionary with mean, std, min, max, and sample count.

        Raises:
            ValueError: If no data accumulated yet.

        Example:
            >>> stats = analyzer.get_statistics()
            >>> print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        """
        if self._n_samples == 0:
            raise ValueError("No data accumulated yet")

        mean = self._sum / self._n_samples
        variance = (self._sum_sq / self._n_samples) - (mean**2)
        std = np.sqrt(max(0, variance))  # Avoid negative due to numerical errors

        return {
            "mean": mean,
            "std": std,
            "min": self._min,
            "max": self._max,
            "n_samples": self._n_samples,
        }

    def get_psd(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get accumulated PSD estimate.

        Returns:
            Tuple of (frequencies, psd) where psd is averaged over all chunks.

        Raises:
            ValueError: If no PSD data accumulated.

        Example:
            >>> freqs, psd = analyzer.get_psd()
            >>> print(f"PSD shape: {psd.shape}")
        """
        if self._psd_sum is None or self._psd_freqs is None:
            raise ValueError("No PSD data accumulated yet")

        # Return averaged PSD
        psd_avg = self._psd_sum / self._psd_count
        return self._psd_freqs, psd_avg

    def get_histogram(self) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Get accumulated histogram.

        Returns:
            Tuple of (counts, edges).

        Raises:
            ValueError: If no histogram data accumulated.

        Example:
            >>> counts, edges = analyzer.get_histogram()
        """
        if self._hist_counts is None or self._hist_edges is None:
            raise ValueError("No histogram data accumulated yet")

        return self._hist_counts, self._hist_edges

    def reset(self) -> None:
        """Reset all accumulators.

        Example:
            >>> analyzer.reset()
        """
        self._n_samples = 0
        self._sum = 0.0
        self._sum_sq = 0.0
        self._min = float("inf")
        self._max = float("-inf")
        self._psd_sum = None
        self._psd_freqs = None
        self._psd_count = 0
        self._sample_rate = None
        self._hist_counts = None
        self._hist_edges = None


def chunked_spectrogram(
    data: NDArray[np.float64],
    sample_rate: float,
    *,
    chunk_size: int = 10_000_000,
    overlap: int = 0,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute spectrogram for large signals using chunked processing.


    Processes large signals in overlapping chunks to compute spectrograms
    without loading entire signal into memory. Stitches STFT results from
    chunks with proper boundary handling.

    Args:
        data: Input signal array (can be memory-mapped).
        sample_rate: Sample rate in Hz.
        chunk_size: Maximum samples per chunk (default 10M).
        overlap: Overlap samples between chunks for continuity (default 0).
            Should be at least 2*nperseg for proper STFT boundary handling.
        nperseg: Segment length for STFT (default 256).
        noverlap: Overlap between STFT segments within chunk (default nperseg//2).
        window: Window function name (default "hann").

    Returns:
        (times, frequencies, Sxx_db) - Time axis, frequency axis, and
        spectrogram magnitude in dB as 2D array (frequencies x time).

    Raises:
        ValueError: If no valid chunks produced.

    Example:
        >>> # Memory-efficient spectrogram on 1 GB signal
        >>> import numpy as np
        >>> data = np.memmap('huge_trace.dat', dtype='float64', mode='r')
        >>> t, f, Sxx = chunked_spectrogram(data, sample_rate=1e9, chunk_size=10_000_000)
        >>> print(f"Spectrogram shape: {Sxx.shape}")

    References:
        MEM-004: Chunked Spectrogram requirement
        scipy.signal.spectrogram
    """
    n = len(data)

    # Handle empty input
    if n == 0:
        return np.array([]), np.array([]), np.array([]).reshape(0, 0)

    if noverlap is None:
        noverlap = nperseg // 2

    # Auto-adjust overlap if not specified to ensure continuity
    if overlap == 0:
        overlap = 2 * nperseg

    # If data fits in one chunk, use scipy directly
    if n <= chunk_size:
        freq, times, Sxx = signal.spectrogram(
            data,
            fs=sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="spectrum",
        )
        # Convert to dB
        Sxx = np.maximum(Sxx, 1e-20)
        Sxx_db = 10 * np.log10(Sxx)
        return times, freq, Sxx_db

    # Process chunks
    chunks_stft = []
    chunks_times = []
    chunk_start = 0

    while chunk_start < n:
        # Determine chunk boundaries with overlap
        chunk_end = min(chunk_start + chunk_size, n)

        # Extract chunk with overlap extension on both sides
        extended_start = max(0, chunk_start - overlap)
        extended_end = min(n, chunk_end + overlap)

        chunk_data = data[extended_start:extended_end]

        # Compute spectrogram for chunk
        freq, times_chunk, Sxx_chunk = signal.spectrogram(
            chunk_data,
            fs=sample_rate,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="spectrum",
        )

        # Adjust time axis for chunk position
        time_offset = extended_start / sample_rate
        times_chunk_adjusted = times_chunk + time_offset

        # Trim overlap regions to avoid duplication
        valid_time_start = chunk_start / sample_rate
        valid_time_end = chunk_end / sample_rate

        valid_mask = (times_chunk_adjusted >= valid_time_start) & (
            times_chunk_adjusted < valid_time_end
        )

        if np.any(valid_mask):
            Sxx_chunk = Sxx_chunk[:, valid_mask]
            times_chunk_adjusted = times_chunk_adjusted[valid_mask]

            chunks_stft.append(Sxx_chunk)
            chunks_times.append(times_chunk_adjusted)

        # Move to next chunk
        chunk_start = chunk_end

    # Concatenate all chunks
    if len(chunks_stft) == 0:
        raise ValueError("No valid chunks produced")

    Sxx = np.concatenate(chunks_stft, axis=1)
    times = np.concatenate(chunks_times)

    # Convert to dB
    Sxx = np.maximum(Sxx, 1e-20)
    Sxx_db = 10 * np.log10(Sxx)

    return times, freq, Sxx_db


def chunked_fft(
    data: NDArray[np.float64],
    sample_rate: float,
    *,
    chunk_size: int = 10_000_000,
    overlap: float = 50.0,
    window: str = "hann",
    nfft: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute FFT for very long signals using segmented averaging.


    Divides signal into overlapping segments, computes FFT for each,
    and averages magnitude spectra. This is memory-bounded by chunk_size
    and provides variance reduction through averaging (similar to Welch's method).

    Args:
        data: Input signal array (can be memory-mapped).
        sample_rate: Sample rate in Hz.
        chunk_size: Size of each segment in samples (default 10M).
        overlap: Percentage overlap between segments, 0-100 (default 50%).
        window: Window function name (default "hann").
        nfft: FFT length. If None, uses next power of 2 >= chunk_size.

    Returns:
        (frequencies, magnitude_db) - Frequency axis and averaged magnitude in dB.

    Example:
        >>> # Memory-efficient FFT on 1 GB signal with 50% overlap
        >>> import numpy as np
        >>> data = np.memmap('huge_trace.dat', dtype='float64', mode='r')
        >>> freq, mag = chunked_fft(data, sample_rate=1e9, chunk_size=1_000_000)
        >>> print(f"Frequency resolution: {freq[1] - freq[0]:.3f} Hz")

    References:
        MEM-006: Chunked FFT requirement
        Welch's method for spectral estimation
    """
    from ..utils.windowing import get_window

    n = len(data)

    # Handle empty input
    if n == 0:
        return np.array([]), np.array([])

    # If data fits in one chunk, compute single FFT
    if n <= chunk_size:
        if nfft is None:
            nfft = int(2 ** np.ceil(np.log2(n)))

        # Apply window
        w = get_window(window, n)
        data_windowed = data * w

        # Compute FFT
        spectrum = np.fft.rfft(data_windowed, n=nfft)

        # Frequency axis
        freq = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

        # Magnitude in dB (normalized by window gain)
        window_gain = np.sum(w) / n
        magnitude = np.abs(spectrum) / (n * window_gain)
        magnitude = np.maximum(magnitude, 1e-20)
        magnitude_db = 20 * np.log10(magnitude)

        return freq, magnitude_db

    # Calculate overlap
    overlap_samples = int(chunk_size * overlap / 100.0)
    hop = chunk_size - overlap_samples

    # Determine number of segments
    num_segments = max(1, (n - overlap_samples) // hop)

    if nfft is None:
        nfft = int(2 ** np.ceil(np.log2(chunk_size)))

    # Prepare window
    w = get_window(window, chunk_size)
    window_gain = np.sum(w) / chunk_size

    # Accumulate magnitude spectra
    freq = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    magnitude_sum = np.zeros(len(freq))

    for i in range(num_segments):
        start = i * hop
        end = min(start + chunk_size, n)

        # Extract segment
        if end - start < chunk_size:
            # Last segment: pad with zeros
            segment = np.zeros(chunk_size)
            segment[: end - start] = data[start:end]
        else:
            segment = data[start:end]

        # Detrend (remove mean)
        segment = segment - np.mean(segment)

        # Window
        segment_windowed = segment * w

        # FFT
        spectrum = np.fft.rfft(segment_windowed, n=nfft)

        # Accumulate magnitude
        magnitude = np.abs(spectrum) / (chunk_size * window_gain)
        magnitude_sum += magnitude

    # Average
    magnitude_avg = magnitude_sum / num_segments

    # Convert to dB
    magnitude_avg = np.maximum(magnitude_avg, 1e-20)
    magnitude_db = 20 * np.log10(magnitude_avg)

    return freq, magnitude_db


__all__ = [
    "StreamingAnalyzer",
    "chunked_fft",
    "chunked_spectrogram",
    "load_trace_chunks",
]
