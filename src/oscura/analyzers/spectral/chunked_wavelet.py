"""Chunked wavelet transform for memory-bounded processing.

This module implements memory-bounded wavelet transforms (CWT and DWT)
with segment processing and boundary handling.


Example:
    >>> from oscura.analyzers.spectral.chunked_wavelet import cwt_chunked, dwt_chunked
    >>> coeffs = cwt_chunked('large_signal.bin', scales=[1, 2, 4, 8], wavelet='morl')
    >>> print(f"CWT coefficients shape: {coeffs.shape}")

References:
    pywt (PyWavelets) for wavelet transforms
    Mallat, S. (1999). "A Wavelet Tour of Signal Processing"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from numpy.typing import NDArray


def cwt_chunked(
    file_path: str | Path,
    scales: Sequence[float],
    wavelet: str = "morl",
    *,
    chunk_size: int | float = 1e6,
    overlap_factor: float = 2.0,
    sample_rate: float = 1.0,
    dtype: str = "float32",
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Compute continuous wavelet transform for large files.


    Processes signal in chunks with overlap to handle boundaries,
    computes CWT per chunk, and stitches results.

    Args:
        file_path: Path to signal file (binary format).
        scales: Scales for CWT (wavelet dilations).
        wavelet: Wavelet name ('morl', 'mexh', 'cmor', etc.).
        chunk_size: Chunk size in samples.
        overlap_factor: Overlap factor for boundaries (e.g., 2.0 = 2x max scale).
        sample_rate: Sample rate in Hz.
        dtype: Data type of input file ('float32' or 'float64').

    Returns:
        Tuple of (coefficients, frequencies) where:
        - coefficients: CWT coefficients (scales x time).
        - frequencies: Corresponding frequencies for each scale.

    Raises:
        ImportError: If pywt (PyWavelets) is not installed.
        ValueError: If file cannot be read or scales invalid.

    Example:
        >>> scales = np.arange(1, 128)
        >>> coeffs, freqs = cwt_chunked(
        ...     'signal.bin',
        ...     scales=scales,
        ...     wavelet='morl',
        ...     chunk_size=1e6,
        ...     sample_rate=1e6
        ... )
        >>> print(f"CWT shape: {coeffs.shape}")

    References:
        MEM-007: Chunked Wavelet Transform
    """
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "pywt (PyWavelets) is required for wavelet transforms. "
            "Install with: pip install PyWavelets"
        ) from e

    chunk_size = int(chunk_size)
    scales: NDArray[np.float64] = np.asarray(scales)  # type: ignore[no-redef]

    # Calculate boundary overlap (proportional to max scale)
    max_scale = np.max(scales)
    boundary_overlap = int(overlap_factor * max_scale)

    # Determine dtype
    np_dtype = np.float32 if dtype == "float32" else np.float64
    bytes_per_sample = 4 if dtype == "float32" else 8

    # Open file and get total size
    file_path = Path(file_path)
    file_size_bytes = file_path.stat().st_size
    total_samples = file_size_bytes // bytes_per_sample

    # Process chunks
    coeffs_list: list[NDArray[Any]] = []
    chunks = _generate_chunks(file_path, total_samples, chunk_size, boundary_overlap, np_dtype)

    for chunk_data in chunks:
        # Compute CWT for this chunk
        coeffs_chunk, freqs = pywt.cwt(
            chunk_data,
            scales,
            wavelet,
            sampling_period=1 / sample_rate,
        )

        # Remove boundary overlap regions (except first/last chunk)
        if len(coeffs_list) > 0:
            # Remove left boundary
            trim_left = boundary_overlap
            coeffs_chunk = coeffs_chunk[:, trim_left:]

        coeffs_list.append(coeffs_chunk)

    # Concatenate all chunks
    if len(coeffs_list) == 0:
        raise ValueError(f"No chunks processed from {file_path}")

    coefficients = np.concatenate(coeffs_list, axis=1)

    return coefficients, freqs


def _setup_dwt_parameters(
    file_path: Path, wavelet: str, level: int | None, dtype: str
) -> tuple[type, int, int, int, int]:
    """Setup DWT parameters and calculate boundaries."""
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "pywt (PyWavelets) is required for wavelet transforms. "
            "Install with: pip install PyWavelets"
        ) from e

    np_dtype = np.float32 if dtype == "float32" else np.float64
    bytes_per_sample = 4 if dtype == "float32" else 8

    wavelet_obj = pywt.Wavelet(wavelet)
    filter_len = wavelet_obj.dec_len
    boundary_overlap = filter_len * (2 ** (level or 1))

    file_size_bytes = file_path.stat().st_size
    total_samples = file_size_bytes // bytes_per_sample

    return np_dtype, bytes_per_sample, boundary_overlap, total_samples, filter_len


def _process_dwt_chunks(
    file_path: Path,
    total_samples: int,
    chunk_size: int,
    boundary_overlap: int,
    np_dtype: type,
    wavelet: str,
    mode: str,
    level: int | None,
) -> list[list[NDArray[Any]]]:
    """Process file chunks and compute DWT for each."""
    try:
        import pywt
    except ImportError as e:
        raise ImportError("pywt required") from e

    coeffs_list: list[list[NDArray[Any]]] = []
    chunks = _generate_chunks(file_path, total_samples, chunk_size, boundary_overlap, np_dtype)

    for chunk_data in chunks:
        coeffs_chunk = pywt.wavedec(chunk_data, wavelet, mode=mode, level=level)
        coeffs_list.append(coeffs_chunk)

    return coeffs_list


def _merge_dwt_level_coeffs(
    coeffs_list: list[list[NDArray[Any]]],
    level_idx: int,
    level_overlap: int,
) -> NDArray[Any]:
    """Merge coefficients for a single decomposition level."""
    merged_level_coeffs = []

    for chunk_idx, chunk_coeffs in enumerate(coeffs_list):
        level_coeffs = chunk_coeffs[level_idx]

        if chunk_idx == 0:
            # First chunk - keep all except right overlap
            if level_overlap > 0 and len(level_coeffs) > level_overlap:
                merged_level_coeffs.append(level_coeffs[:-level_overlap])
            else:
                merged_level_coeffs.append(level_coeffs)
        elif chunk_idx == len(coeffs_list) - 1:
            # Last chunk - trim left overlap, keep rest
            if level_overlap > 0 and len(level_coeffs) > level_overlap:
                merged_level_coeffs.append(level_coeffs[level_overlap:])
            else:
                center_start = max(0, level_overlap // 2)
                merged_level_coeffs.append(level_coeffs[center_start:])
        else:
            # Middle chunks - trim both sides
            if level_overlap > 0 and len(level_coeffs) > 2 * level_overlap:
                merged_level_coeffs.append(level_coeffs[level_overlap:-level_overlap])
            else:
                center_start = max(0, level_overlap // 2)
                center_end = max(center_start + 1, len(level_coeffs) - level_overlap // 2)
                merged_level_coeffs.append(level_coeffs[center_start:center_end])

    if merged_level_coeffs:
        return np.concatenate(merged_level_coeffs)
    else:
        level_coeffs_list = [chunk_coeffs[level_idx] for chunk_coeffs in coeffs_list]
        return np.concatenate(level_coeffs_list)


def dwt_chunked(
    file_path: str | Path,
    wavelet: str = "db4",
    level: int | None = None,
    *,
    chunk_size: int | float = 1e6,
    mode: str = "symmetric",
    dtype: str = "float32",
) -> list[NDArray[Any]]:
    """Compute discrete wavelet transform for large files.


    Processes signal in chunks and computes multilevel DWT.
    Handles boundaries using specified extension mode.

    Args:
        file_path: Path to signal file (binary format).
        wavelet: Wavelet name ('db4', 'haar', 'sym5', etc.).
        level: Decomposition level (None = maximum level).
        chunk_size: Chunk size in samples.
        mode: Signal extension mode for boundaries.
        dtype: Data type of input file ('float32' or 'float64').

    Returns:
        List of coefficient arrays [cA_n, cD_n, ..., cD_1] where:
        - cA_n: Approximation coefficients at level n.
        - cD_i: Detail coefficients at level i.

    Raises:
        ImportError: If pywt is not installed.
        ValueError: If file cannot be read.

    Example:
        >>> coeffs = dwt_chunked(
        ...     'signal.bin',
        ...     wavelet='db4',
        ...     level=5,
        ...     chunk_size=1e6
        ... )
        >>> print(f"Approximation shape: {coeffs[0].shape}")

    References:
        MEM-007: Chunked Wavelet Transform
        Daubechies, I. (1992). "Ten Lectures on Wavelets"
    """
    chunk_size = int(chunk_size)
    file_path = Path(file_path)

    np_dtype, bytes_per_sample, boundary_overlap, total_samples, filter_len = _setup_dwt_parameters(
        file_path, wavelet, level, dtype
    )

    coeffs_list = _process_dwt_chunks(
        file_path, total_samples, chunk_size, boundary_overlap, np_dtype, wavelet, mode, level
    )

    if len(coeffs_list) == 0:
        raise ValueError(f"No chunks processed from {file_path}")

    num_levels = len(coeffs_list[0])
    merged_coeffs = []

    for level_idx in range(num_levels):
        downsample_factor = 2**level_idx
        level_overlap = boundary_overlap // downsample_factor

        if len(coeffs_list) == 1:
            merged_coeffs.append(coeffs_list[0][level_idx])
        else:
            merged_coeffs.append(_merge_dwt_level_coeffs(coeffs_list, level_idx, level_overlap))

    return merged_coeffs


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

            # Seek and read
            f.seek(chunk_start * dtype().itemsize)
            chunk_data: NDArray[np.float64] = np.fromfile(f, dtype=dtype, count=chunk_len)

            if len(chunk_data) == 0:
                break

            yield chunk_data

            # Advance offset
            offset += chunk_size


def cwt_chunked_generator(
    file_path: str | Path,
    scales: Sequence[float],
    wavelet: str = "morl",
    *,
    chunk_size: int | float = 1e6,
    **kwargs: Any,
) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]:
    """Generator version that yields CWT chunks.

    Yields CWT coefficients for each chunk, useful for streaming processing.

    Args:
        file_path: Path to signal file.
        scales: Scales for CWT.
        wavelet: Wavelet name.
        chunk_size: Chunk size in samples.
        **kwargs: Additional arguments.

    Yields:
        Tuples of (coefficients, frequencies) for each chunk.

    Raises:
        ImportError: If pywt (PyWavelets) is not installed.

    Example:
        >>> for coeffs_chunk, freqs in cwt_chunked_generator('file.bin', scales=[1, 2, 4]):
        ...     # Process each chunk separately
        ...     print(f"Chunk shape: {coeffs_chunk.shape}")
    """
    try:
        import pywt
    except ImportError as e:
        raise ImportError(
            "pywt (PyWavelets) is required for wavelet transforms. "
            "Install with: pip install PyWavelets"
        ) from e

    chunk_size = int(chunk_size)
    scales: NDArray[np.float64] = np.asarray(scales)  # type: ignore[no-redef]

    # Determine dtype
    dtype = kwargs.get("dtype", "float32")
    np_dtype = np.float32 if dtype == "float32" else np.float64
    bytes_per_sample = 4 if dtype == "float32" else 8

    # Open file and get total size
    file_path = Path(file_path)
    file_size_bytes = file_path.stat().st_size
    total_samples = file_size_bytes // bytes_per_sample

    # Calculate boundary overlap
    max_scale = np.max(scales)
    boundary_overlap = int(kwargs.get("overlap_factor", 2.0) * max_scale)

    # Process chunks
    sample_rate = kwargs.get("sample_rate", 1.0)
    chunks = _generate_chunks(file_path, total_samples, chunk_size, boundary_overlap, np_dtype)

    for chunk_data in chunks:
        coeffs_chunk, freqs = pywt.cwt(
            chunk_data,
            scales,
            wavelet,
            sampling_period=1 / sample_rate,
        )
        yield coeffs_chunk, freqs


__all__ = [
    "cwt_chunked",
    "cwt_chunked_generator",
    "dwt_chunked",
]
