"""Chunked spectrogram computation for memory-constrained processing.

This module implements memory-bounded spectrogram computation that processes
files in chunks with proper boundary handling for continuity.


Example:
    >>> from oscura.analyzers.spectral.chunked import spectrogram_chunked
    >>> spec = spectrogram_chunked('large_file.bin', chunk_size=100e6, nperseg=4096)
    >>> print(f"Spectrogram shape: {spec.shape}")

References:
    scipy.signal.stft for STFT computation
    IEEE 1057-2017 for spectral analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from oscura.core.memoize import memoize_analysis

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


@memoize_analysis(maxsize=8)
def spectrogram_chunked(
    file_path: str | Path,
    chunk_size: int | float,
    nperseg: int = 256,
    noverlap: int | None = None,
    *,
    nfft: int | None = None,
    window: str | tuple | NDArray = "hann",  # type: ignore[type-arg]
    detrend: str | bool = False,
    return_onesided: bool = True,
    scaling: str = "density",
    mode: str = "psd",
    overlap_factor: float = 2.0,
    dtype: str = "float32",
    **kwargs: Any,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Compute spectrogram for large files using chunked processing.


    Processes file in chunks with `overlap_factor * nperseg` overlap on
    boundaries to ensure continuity. Computes scipy.signal.stft per chunk
    and stitches results.

    Args:
        file_path: Path to signal file (binary format).
        chunk_size: Chunk size in samples.
        nperseg: Length of each segment for STFT.
        noverlap: Number of points to overlap (default: nperseg // 2).
        nfft: FFT length (default: nperseg).
        window: Window function name or array.
        detrend: Detrend type ('constant', 'linear', False).
        return_onesided: Return one-sided spectrum for real input.
        scaling: Scaling mode ('density' or 'spectrum').
        mode: Output mode ('psd', 'magnitude', 'angle', 'phase', 'complex').
        overlap_factor: Factor for chunk boundary overlap (default: 2.0 = 2*nperseg).
        dtype: Data type of input file ('float32' or 'float64').
        **kwargs: Additional arguments passed to scipy.signal.stft.

    Returns:
        Tuple of (frequencies, times, Sxx) where:
        - frequencies: Array of frequency bins (Hz, requires sample_rate in kwargs).
        - times: Array of time segments (seconds).
        - Sxx: Spectrogram array (frequencies x time segments).

    Raises:
        ValueError: If chunk_size < nperseg or file cannot be read.

    Example:
        >>> # Process 10 GB file in 100M sample chunks
        >>> f, t, Sxx = spectrogram_chunked(
        ...     'huge_trace.bin',
        ...     chunk_size=100e6,
        ...     nperseg=4096,
        ...     noverlap=2048,
        ...     sample_rate=1e9,  # 1 GSa/s
        ...     dtype='float32'
        ... )
        >>> print(f"Spectrogram shape: {Sxx.shape}")

    References:
        MEM-004: Chunked Spectrogram requirement
    """
    chunk_size = int(chunk_size)
    if noverlap is None:
        noverlap = nperseg // 2
    if nfft is None:
        nfft = nperseg

    if chunk_size < nperseg:
        raise ValueError(f"chunk_size ({chunk_size}) must be >= nperseg ({nperseg})")

    # Determine dtype
    np_dtype = np.float32 if dtype == "float32" else np.float64
    bytes_per_sample = 4 if dtype == "float32" else 8

    # Calculate boundary overlap (default: 2*nperseg for continuity)
    boundary_overlap = int(overlap_factor * nperseg)

    # Open file and get total size
    file_path = Path(file_path)
    file_size_bytes = file_path.stat().st_size
    total_samples = file_size_bytes // bytes_per_sample

    # Prepare chunk generator
    chunks = _generate_chunks(file_path, total_samples, chunk_size, boundary_overlap, np_dtype)

    # Process first chunk to initialize arrays
    first_chunk = next(chunks)
    f, t_chunk, Sxx_chunk = signal.spectrogram(
        first_chunk,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        window=window,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        mode=mode,
        **kwargs,
    )

    # Initialize result arrays
    Sxx_list = [Sxx_chunk]
    time_offset = 0.0
    times_list = [t_chunk + time_offset]

    # Get sample rate for time calculation
    fs = kwargs.get("sample_rate", kwargs.get("fs", 1.0))
    time_offset += len(first_chunk) / fs

    # Process remaining chunks
    for chunk_data in chunks:
        _, t_chunk, Sxx_chunk = signal.spectrogram(
            chunk_data,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            window=window,
            detrend=detrend,
            return_onesided=return_onesided,
            scaling=scaling,
            mode=mode,
            **kwargs,
        )

        Sxx_list.append(Sxx_chunk)
        times_list.append(t_chunk + time_offset)
        time_offset += (len(chunk_data) - boundary_overlap) / fs

    # Concatenate all chunks
    Sxx = np.concatenate(Sxx_list, axis=1)
    times = np.concatenate(times_list)

    return f, times, Sxx


def _generate_chunks(
    file_path: Path,
    total_samples: int,
    chunk_size: int,
    boundary_overlap: int,
    dtype: type,
) -> Iterator[NDArray[Any]]:
    """Generate overlapping chunks from file.

    Args:
        file_path: Path to binary file.
        total_samples: Total number of samples in file.
        chunk_size: Samples per chunk.
        boundary_overlap: Overlap samples between chunks.
        dtype: NumPy dtype for data.

    Yields:
        Chunk arrays with boundary overlap.
    """
    offset = 0

    with open(file_path, "rb") as f:
        while offset < total_samples:
            # Calculate chunk boundaries
            chunk_start = max(0, offset - boundary_overlap)
            chunk_end = min(total_samples, offset + chunk_size)
            chunk_len = chunk_end - chunk_start

            # Seek to start position
            f.seek(chunk_start * dtype().itemsize)

            # Read chunk
            chunk_data: NDArray[np.float64] = np.fromfile(f, dtype=dtype, count=chunk_len)

            if len(chunk_data) == 0:
                break

            yield chunk_data

            # Advance offset (accounting for overlap)
            offset += chunk_size


def spectrogram_chunked_generator(
    file_path: str | Path,
    chunk_size: int | float,
    nperseg: int = 256,
    noverlap: int | None = None,
    **kwargs: Any,
) -> Iterator[tuple[NDArray[Any], NDArray[Any], NDArray[Any]]]:
    """Generator version that yields spectrogram chunks.


    Yields chunks of (frequencies, times, Sxx) for streaming processing.
    Useful when full spectrogram doesn't fit in memory.

    Args:
        file_path: Path to signal file.
        chunk_size: Chunk size in samples.
        nperseg: Segment length for STFT.
        noverlap: Overlap samples (default: nperseg // 2).
        **kwargs: Additional arguments for spectrogram.

    Yields:
        Tuples of (frequencies, times, Sxx) for each chunk.

    Example:
        >>> for f, t, Sxx_chunk in spectrogram_chunked_generator('file.bin', chunk_size=50e6):
        ...     # Process or save each chunk separately
        ...     print(f"Processing chunk with {Sxx_chunk.shape[1]} time segments")
    """
    chunk_size = int(chunk_size)
    if noverlap is None:
        noverlap = nperseg // 2

    # Determine dtype
    dtype = kwargs.get("dtype", "float32")
    np_dtype = np.float32 if dtype == "float32" else np.float64
    bytes_per_sample = 4 if dtype == "float32" else 8

    boundary_overlap = int(kwargs.get("overlap_factor", 2.0) * nperseg)

    # Open file and get total size
    file_path = Path(file_path)
    file_size_bytes = file_path.stat().st_size
    total_samples = file_size_bytes // bytes_per_sample

    # Generate and process chunks
    chunks = _generate_chunks(file_path, total_samples, chunk_size, boundary_overlap, np_dtype)

    for chunk_data in chunks:
        f, t, Sxx_chunk = signal.spectrogram(
            chunk_data,
            nperseg=nperseg,
            noverlap=noverlap,
            **{k: v for k, v in kwargs.items() if k != "dtype"},
        )
        yield f, t, Sxx_chunk


__all__ = [
    "spectrogram_chunked",
    "spectrogram_chunked_generator",
]
