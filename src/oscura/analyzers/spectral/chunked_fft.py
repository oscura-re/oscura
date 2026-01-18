"""Chunked FFT computation for very long signals.

This module implements FFT computation for signals larger than memory
using overlapping segments with result aggregation.


Example:
    >>> from oscura.analyzers.spectral.chunked_fft import fft_chunked
    >>> freqs, spectrum = fft_chunked('huge_signal.bin', segment_size=1e6, overlap_pct=50)
    >>> print(f"FFT shape: {spectrum.shape}")

References:
    scipy.fft for FFT computation
    Welch's method for spectral averaging
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import fft, signal

from oscura.core.memoize import memoize_analysis

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from numpy.typing import NDArray


@memoize_analysis(maxsize=16)
def fft_chunked(
    file_path: str | Path,
    segment_size: int | float,
    overlap_pct: float = 50.0,
    *,
    window: str | NDArray[np.float64] = "hann",
    nfft: int | None = None,
    detrend: str | bool = False,
    scaling: str = "density",
    average_method: str = "mean",
    sample_rate: float = 1.0,
    dtype: str = "float32",
    preserve_phase: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | NDArray[np.complex128]]:
    """Compute FFT for very long signals using overlapping segments.


    Processes signal in segments with overlap, computes FFT per segment,
    and aggregates using specified method. Handles window edge effects.

    Args:
        file_path: Path to signal file (binary format).
        segment_size: Segment size in samples.
        overlap_pct: Overlap percentage between segments (0-100).
        window: Window function name or array.
        nfft: FFT length (default: segment_size, zero-padded if larger).
        detrend: Detrend type ('constant', 'linear', False).
        scaling: Scaling mode ('density' or 'spectrum').
        average_method: Aggregation method ('mean', 'median', 'max').
        sample_rate: Sample rate in Hz (for frequency axis).
        dtype: Data type of input file ('float32' or 'float64').
        preserve_phase: If True, preserve phase information (complex output).

    Returns:
        Tuple of (frequencies, spectrum) where:
        - frequencies: Frequency bins in Hz.
        - spectrum: Averaged FFT magnitude (or complex if preserve_phase=True).

    Raises:
        ValueError: If overlap_pct not in [0, 100] or file cannot be read.

    Example:
        >>> # Process 10 GB file with 1M sample segments, 50% overlap
        >>> freqs, spectrum = fft_chunked(
        ...     'huge_signal.bin',
        ...     segment_size=1e6,
        ...     overlap_pct=50,
        ...     window='hann',
        ...     sample_rate=1e9,
        ...     dtype='float32'
        ... )
        >>> print(f"Spectrum shape: {spectrum.shape}")

    References:
        MEM-006: Chunked FFT for Very Long Signals
    """
    if not 0 <= overlap_pct < 100:
        raise ValueError(
            f"overlap_pct must be in [0, 100), got {overlap_pct}. Note: 100% overlap would create an infinite loop."
        )

    segment_size = int(segment_size)
    if nfft is None:
        nfft = segment_size

    # Calculate overlap in samples
    noverlap = int(segment_size * overlap_pct / 100)

    # Determine dtype
    np_dtype = np.float32 if dtype == "float32" else np.float64
    bytes_per_sample = 4 if dtype == "float32" else 8

    # Open file and get total size
    file_path = Path(file_path)
    file_size_bytes = file_path.stat().st_size
    total_samples = file_size_bytes // bytes_per_sample

    # Generate window
    if isinstance(window, str):
        window_arr = signal.get_window(window, segment_size)
    else:
        window_arr = np.asarray(window)

    # Initialize accumulators
    fft_accum: list[NDArray[np.float64] | NDArray[np.complex128]] = []

    # Process segments
    for segment in _generate_segments(file_path, total_samples, segment_size, noverlap, np_dtype):
        # Apply detrending
        if detrend:
            segment = signal.detrend(segment, type=detrend)

        # Apply window
        windowed = segment * window_arr[: len(segment)]

        # Zero-pad if needed
        if len(windowed) < nfft:
            windowed = np.pad(windowed, (0, nfft - len(windowed)), mode="constant")

        # Compute FFT
        fft_result = fft.rfft(windowed, n=nfft)

        # Store result (magnitude or complex)
        if preserve_phase:
            fft_accum.append(fft_result)
        else:
            fft_accum.append(np.abs(fft_result))

    # Aggregate results
    if len(fft_accum) == 0:
        raise ValueError(f"No segments processed from {file_path}")

    if average_method == "mean":
        spectrum = np.mean(fft_accum, axis=0)
    elif average_method == "median":
        spectrum = np.median(fft_accum, axis=0)
    elif average_method == "max":
        spectrum = np.max(fft_accum, axis=0)
    else:
        raise ValueError(
            f"Unknown average_method: {average_method}. Use 'mean', 'median', or 'max'."
        )

    # Apply scaling
    if scaling == "density" and not preserve_phase:
        # Convert to PSD-like scaling
        spectrum = spectrum**2 / (sample_rate * np.sum(window_arr**2))
    elif scaling == "spectrum" and not preserve_phase:
        # RMS scaling
        spectrum = spectrum / len(window_arr)

    # Frequency axis
    frequencies = fft.rfftfreq(nfft, d=1 / sample_rate)

    return frequencies, spectrum


def _generate_segments(
    file_path: Path,
    total_samples: int,
    segment_size: int,
    noverlap: int,
    dtype: type,
) -> Iterator[NDArray[np.float64]]:
    """Generate overlapping segments from file.

    Args:
        file_path: Path to binary file.
        total_samples: Total number of samples in file.
        segment_size: Samples per segment.
        noverlap: Overlap samples between segments.
        dtype: NumPy dtype for data.

    Yields:
        Segment arrays.
    """
    hop = segment_size - noverlap
    offset = 0

    with open(file_path, "rb") as f:
        while offset < total_samples:
            # Read segment
            f.seek(offset * dtype().itemsize)
            segment_data: NDArray[np.float64] = np.fromfile(f, dtype=dtype, count=segment_size)

            if len(segment_data) == 0:
                break

            yield segment_data

            offset += hop


def welch_psd_chunked(
    file_path: str | Path,
    segment_size: int | float = 256,
    overlap_pct: float = 50.0,
    *,
    window: str | NDArray[np.float64] = "hann",
    nfft: int | None = None,
    detrend: str | bool = "constant",
    scaling: str = "density",
    sample_rate: float = 1.0,
    dtype: str = "float32",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute Welch PSD estimate for very long signals.

    Similar to fft_chunked but specifically implements Welch's method
    for power spectral density estimation.


    Args:
        file_path: Path to signal file.
        segment_size: Segment size for Welch's method.
        overlap_pct: Overlap percentage (typically 50%).
        window: Window function.
        nfft: FFT length.
        detrend: Detrend type.
        scaling: Scaling mode ('density' or 'spectrum').
        sample_rate: Sample rate in Hz.
        dtype: Data type of input file.

    Returns:
        Tuple of (frequencies, psd).

    Example:
        >>> freqs, psd = welch_psd_chunked('signal.bin', segment_size=1024, sample_rate=1e6)
        >>> print(f"PSD shape: {psd.shape}")

    References:
        MEM-005: Chunked Welch PSD
        Welch, P.D. (1967). "The use of fast Fourier transform for the
        estimation of power spectra"
    """
    freqs, spectrum = fft_chunked(
        file_path,
        segment_size=segment_size,
        overlap_pct=overlap_pct,
        window=window,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
        average_method="mean",
        sample_rate=sample_rate,
        dtype=dtype,
        preserve_phase=False,
    )
    # preserve_phase=False guarantees float64 output, not complex128
    return freqs, spectrum  # type: ignore[return-value]


def fft_chunked_parallel(
    file_path: str | Path,
    segment_size: int | float,
    overlap_pct: float = 50.0,
    *,
    n_workers: int = 4,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute chunked FFT with parallel processing.

    Similar to fft_chunked but uses multiple workers for parallel
    segment processing. Useful for very large files on multi-core systems.

    Args:
        file_path: Path to signal file.
        segment_size: Segment size in samples.
        overlap_pct: Overlap percentage.
        n_workers: Number of parallel workers.
        **kwargs: Additional arguments passed to fft_chunked.

    Returns:
        Tuple of (frequencies, spectrum).

    Note:
        FUTURE ENHANCEMENT: Parallel processing with multiprocessing/joblib.
        Currently uses serial processing (n_workers parameter is reserved
        for future implementation). The serial fallback provides correct
        results; parallelization is an optimization opportunity.

    Example:
        >>> freqs, spectrum = fft_chunked_parallel(
        ...     'signal.bin',
        ...     segment_size=1e6,
        ...     overlap_pct=50,
        ...     n_workers=8
        ... )
    """
    # Future: Implement parallel processing with multiprocessing or joblib
    # For now, fall back to serial processing
    freqs, spectrum = fft_chunked(file_path, segment_size, overlap_pct, **kwargs)
    # kwargs may contain preserve_phase, handle both float64 and complex128
    return freqs, spectrum  # type: ignore[return-value]


def streaming_fft(
    file_path: str | Path,
    segment_size: int | float,
    overlap_pct: float = 50.0,
    *,
    window: str | NDArray[np.float64] = "hann",
    nfft: int | None = None,
    detrend: str | bool = False,
    sample_rate: float = 1.0,
    dtype: str = "float32",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """Stream FFT computation yielding frequency bins as computed.

    Implements streaming/generator API for memory-efficient FFT computation
    on very large files. Yields frequency bins as they are computed, allowing
    downstream processing before all segments are complete.


    Args:
        file_path: Path to signal file (binary format).
        segment_size: Segment size in samples.
        overlap_pct: Overlap percentage between segments (0-100).
        window: Window function name or array.
        nfft: FFT length (default: segment_size).
        detrend: Detrend type ('constant', 'linear', False).
        sample_rate: Sample rate in Hz (for frequency axis).
        dtype: Data type of input file ('float32' or 'float64').
        progress_callback: Optional callback(current, total) to report progress.

    Yields:
        Tuple of (frequencies, fft_magnitude) for each segment.

    Raises:
        ValueError: If overlap_pct not in valid range.

    Example:
        >>> # Stream FFT results as computed
        >>> def on_progress(current, total):
        ...     print(f"Progress: {current}/{total} ({current/total*100:.1f}%)")
        >>>
        >>> for frequencies, magnitude in streaming_fft(
        ...     'huge_signal.bin',
        ...     segment_size=1e6,
        ...     overlap_pct=50,
        ...     progress_callback=on_progress
        ... ):
        ...     # Process each segment immediately
        ...     peak_freq = frequencies[magnitude.argmax()]
        ...     print(f"Peak frequency: {peak_freq:.2e} Hz")

    References:
        API-003: Streaming/Generator API for Large Files
    """
    if not 0 <= overlap_pct < 100:
        raise ValueError(
            f"overlap_pct must be in [0, 100), got {overlap_pct}. Note: 100% overlap would create an infinite loop."
        )

    segment_size = int(segment_size)
    if nfft is None:
        nfft = segment_size

    # Calculate overlap in samples
    noverlap = int(segment_size * overlap_pct / 100)

    # Determine dtype
    np_dtype = np.float32 if dtype == "float32" else np.float64
    bytes_per_sample = 4 if dtype == "float32" else 8

    # Open file and get total size
    file_path = Path(file_path)
    file_size_bytes = file_path.stat().st_size
    total_samples = file_size_bytes // bytes_per_sample

    # Calculate total segments for progress reporting
    hop = segment_size - noverlap
    total_segments = max(1, (total_samples - segment_size) // hop + 1)

    # Generate window
    if isinstance(window, str):
        window_arr = signal.get_window(window, segment_size)
    else:
        window_arr = np.asarray(window)

    # Frequency axis (computed once)
    frequencies = fft.rfftfreq(nfft, d=1 / sample_rate)

    # Process and yield segments
    segment_count = 0
    for segment in _generate_segments(file_path, total_samples, segment_size, noverlap, np_dtype):
        # Apply detrending
        if detrend:
            segment = signal.detrend(segment, type=detrend)

        # Apply window
        windowed = segment * window_arr[: len(segment)]

        # Zero-pad if needed
        if len(windowed) < nfft:
            windowed = np.pad(windowed, (0, nfft - len(windowed)), mode="constant")

        # Compute FFT
        fft_result = fft.rfft(windowed, n=nfft)
        magnitude = np.abs(fft_result)

        # Yield result immediately
        yield frequencies, magnitude

        # Update progress
        segment_count += 1  # noqa: SIM113
        if progress_callback is not None:
            progress_callback(segment_count, total_segments)


class StreamingAnalyzer:
    """Accumulator for streaming analysis across chunks.

    Enables processing of huge files chunk-by-chunk with accumulation
    of statistics, PSD estimates, and other aggregated measurements.


    Attributes:
        chunk_count: Number of chunks processed.
        accumulated_psd: Accumulated PSD estimate (if accumulate_psd called).
        accumulated_stats: Dictionary of accumulated statistics.

    Example:
        >>> analyzer = StreamingAnalyzer()
        >>> for chunk in load_trace_chunks('large.bin', chunk_size=50e6):
        ...     analyzer.accumulate_psd(chunk, nperseg=4096)
        ...     analyzer.accumulate_stats(chunk)
        >>> psd = analyzer.get_psd()
        >>> stats = analyzer.get_stats()

    References:
        API-003: Streaming/Generator API for Large Files
    """

    def __init__(self) -> None:
        """Initialize streaming analyzer."""
        self.chunk_count: int = 0
        self._psd_accumulator: list[NDArray[Any]] = []
        self._psd_frequencies: NDArray[Any] | None = None
        self._psd_config: dict[str, Any] = {}
        self._stats_accumulator: dict[str, list[float]] = {
            "mean": [],
            "std": [],
            "min": [],
            "max": [],
        }

    def accumulate_psd(
        self,
        chunk: NDArray[Any],
        nperseg: int = 256,
        window: str = "hann",
        sample_rate: float = 1.0,
    ) -> None:
        """Accumulate PSD estimate from chunk using Welch's method.

        Args:
            chunk: Data chunk to process.
            nperseg: Length of each segment for Welch's method.
            window: Window function name.
            sample_rate: Sample rate in Hz.

        Example:
            >>> analyzer.accumulate_psd(chunk, nperseg=4096, window='hann')
        """
        # Compute Welch PSD for this chunk
        freqs, psd = signal.welch(chunk, fs=sample_rate, nperseg=nperseg, window=window)

        # Store frequencies on first call
        if self._psd_frequencies is None:
            self._psd_frequencies = freqs
            self._psd_config = {
                "nperseg": nperseg,
                "window": window,
                "sample_rate": sample_rate,
            }

        # Accumulate PSD
        self._psd_accumulator.append(psd)
        self.chunk_count += 1

    def accumulate_stats(self, chunk: NDArray[np.float64]) -> None:
        """Accumulate basic statistics from chunk.

        Args:
            chunk: Data chunk to process.

        Example:
            >>> analyzer.accumulate_stats(chunk)
        """
        self._stats_accumulator["mean"].append(float(np.mean(chunk)))
        self._stats_accumulator["std"].append(float(np.std(chunk)))
        self._stats_accumulator["min"].append(float(np.min(chunk)))
        self._stats_accumulator["max"].append(float(np.max(chunk)))

    def get_psd(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Get aggregated PSD estimate.

        Returns:
            Tuple of (frequencies, psd) with averaged PSD across chunks.

        Raises:
            ValueError: If no PSD data accumulated.

        Example:
            >>> freqs, psd = analyzer.get_psd()
        """
        if not self._psd_accumulator:
            raise ValueError("No PSD data accumulated. Call accumulate_psd() first.")

        if self._psd_frequencies is None:
            raise ValueError("PSD frequencies not initialized. Call accumulate_psd() first.")

        # Average PSDs across all chunks
        avg_psd = np.mean(self._psd_accumulator, axis=0)
        return self._psd_frequencies, avg_psd

    def get_stats(self) -> dict[str, float]:
        """Get aggregated statistics.

        Returns:
            Dictionary with overall mean, std, min, max.

        Example:
            >>> stats = analyzer.get_stats()
            >>> print(f"Overall mean: {stats['mean']:.3f}")
        """
        if not self._stats_accumulator["mean"]:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        return {
            "mean": float(np.mean(self._stats_accumulator["mean"])),
            "std": float(np.mean(self._stats_accumulator["std"])),
            "min": float(np.min(self._stats_accumulator["min"])),
            "max": float(np.max(self._stats_accumulator["max"])),
        }

    def reset(self) -> None:
        """Reset all accumulated data.

        Example:
            >>> analyzer.reset()
        """
        self.chunk_count = 0
        self._psd_accumulator.clear()
        self._psd_frequencies = None
        self._psd_config.clear()
        for key in self._stats_accumulator:
            self._stats_accumulator[key].clear()


__all__ = [
    "StreamingAnalyzer",
    "fft_chunked",
    "fft_chunked_parallel",
    "streaming_fft",
    "welch_psd_chunked",
]
