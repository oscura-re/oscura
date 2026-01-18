"""Wavelet transform analysis for waveform data.

This module provides continuous and discrete wavelet transforms,
including chunked implementations for memory-efficient processing.


Example:
    >>> from oscura.analyzers.waveform.wavelets import cwt_chunked, dwt_chunked
    >>> # Process large file with chunked CWT
    >>> coeffs, scales = cwt_chunked('large_file.bin', scales=np.arange(1, 128))

References:
    Mallat, S. (2008). A Wavelet Tour of Signal Processing
    Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike

    from numpy.typing import NDArray

try:
    import pywt

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


def cwt_chunked(
    file_path: str | PathLike[str],
    scales: NDArray[np.floating[Any]],
    wavelet: str = "morl",
    *,
    chunk_size: int = 10_000_000,
    overlap_factor: float = 2.0,
    dtype: type = np.float64,
) -> Generator[tuple[NDArray[np.float64], NDArray[np.float64]], None, None]:
    """Compute continuous wavelet transform on large files using chunked processing.

    Processes file in overlapping chunks to handle files larger than memory.
    Uses overlap to ensure continuity at chunk boundaries.

    Args:
        file_path: Path to binary file containing signal data.
        scales: Array of scales for CWT computation.
        wavelet: Wavelet name (default 'morl' - Morlet).
        chunk_size: Number of samples per chunk (default 10M).
        overlap_factor: Overlap as multiple of max scale (default 2.0).
        dtype: Data type for file reading (default float64).

    Yields:
        Tuple of (coefficients, valid_scales) for each chunk:
            - coefficients: CWT coefficients [scales x time]
            - valid_scales: Scales used (same as input)

    Raises:
        ImportError: If PyWavelets is not available.
        FileNotFoundError: If file does not exist.

    Example:
        >>> import numpy as np
        >>> scales = np.arange(1, 128)
        >>> for coeffs, scales_out in cwt_chunked('signal.bin', scales):
        ...     # Process each chunk
        ...     print(f"Chunk shape: {coeffs.shape}")

    References:
        MEM-007: Chunked Wavelet Transform
        Torrence & Compo (1998): Practical guide to wavelet analysis
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets not available. Install with: pip install PyWavelets")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file size and calculate total samples
    file_size = path.stat().st_size
    bytes_per_sample = np.dtype(dtype).itemsize
    total_samples = file_size // bytes_per_sample

    # Calculate overlap based on maximum scale
    max_scale = int(np.max(scales))
    overlap = int(max_scale * overlap_factor)

    # Process file in chunks
    offset = 0
    with open(path, "rb") as f:
        while offset < total_samples:
            # Read chunk with overlap
            f.seek(offset * bytes_per_sample)
            chunk_samples = min(chunk_size, total_samples - offset)
            read_samples = min(chunk_samples + overlap, total_samples - offset)

            chunk_data: NDArray[np.float64] = np.fromfile(f, dtype=dtype, count=read_samples)

            if len(chunk_data) == 0:
                break

            # Compute CWT on chunk
            coefficients, _frequencies = pywt.cwt(chunk_data, scales, wavelet, sampling_period=1.0)

            # Extract valid portion (exclude overlap region for continuity)
            if offset + chunk_samples < total_samples:
                # Not the last chunk - exclude overlap from end
                valid_coeffs = coefficients[:, :chunk_samples]
            else:
                # Last chunk - include everything
                valid_coeffs = coefficients

            yield valid_coeffs.astype(np.float64), scales.astype(np.float64)

            # Move to next chunk
            offset += chunk_samples


def dwt_chunked(
    file_path: str | PathLike[str],
    wavelet: str = "db4",
    level: int | None = None,
    *,
    chunk_size: int = 10_000_000,
    mode: str = "symmetric",
    dtype: type = np.float64,
) -> Generator[dict[str, NDArray[np.float64]], None, None]:
    """Compute discrete wavelet transform on large files using chunked processing.

    Processes file in chunks with boundary handling for DWT reconstruction.

    Args:
        file_path: Path to binary file containing signal data.
        wavelet: Wavelet name (default 'db4' - Daubechies 4).
        level: Decomposition level. If None, uses maximum possible.
        chunk_size: Number of samples per chunk (default 10M).
        mode: Signal extension mode (default 'symmetric').
        dtype: Data type for file reading (default float64).

    Yields:
        Dictionary with DWT coefficients for each chunk:
            - 'cA{level}': Approximation coefficients at level
            - 'cD{level}', 'cD{level-1}', ..., 'cD1': Detail coefficients

    Raises:
        ImportError: If PyWavelets is not available.
        FileNotFoundError: If file does not exist.

    Example:
        >>> for coeffs in dwt_chunked('signal.bin', wavelet='db4', level=3):
        ...     # Process each chunk's wavelet decomposition
        ...     print(f"Approximation shape: {coeffs['cA3'].shape}")
        ...     print(f"Details: {[k for k in coeffs if k.startswith('cD')]}")

    References:
        MEM-007: Chunked Wavelet Transform
        Mallat (2008): Wavelet Tour of Signal Processing
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets not available. Install with: pip install PyWavelets")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get file size
    file_size = path.stat().st_size
    bytes_per_sample = np.dtype(dtype).itemsize
    total_samples = file_size // bytes_per_sample

    # Calculate filter length for boundary handling
    wavelet_obj = pywt.Wavelet(wavelet)
    filter_len = wavelet_obj.dec_len

    # Overlap needed for boundary continuity
    if level is None:
        level = pywt.dwt_max_level(chunk_size, filter_len)

    overlap = filter_len * (2**level)

    # Process file in chunks
    offset = 0
    with open(path, "rb") as f:
        while offset < total_samples:
            # Read chunk with overlap
            f.seek(offset * bytes_per_sample)
            chunk_samples = min(chunk_size, total_samples - offset)
            read_samples = min(chunk_samples + overlap, total_samples - offset)

            chunk_data: NDArray[np.float64] = np.fromfile(f, dtype=dtype, count=read_samples)

            if len(chunk_data) == 0:
                break

            # Compute DWT on chunk
            coeffs = pywt.wavedec(chunk_data, wavelet, mode=mode, level=level)

            # Build result dictionary
            result = {}
            result[f"cA{level}"] = coeffs[0].astype(np.float64)
            for i, detail in enumerate(coeffs[1:], 1):
                result[f"cD{level - i + 1}"] = detail.astype(np.float64)

            yield result

            # Move to next chunk
            offset += chunk_samples


def cwt(
    signal: NDArray[np.floating[Any]],
    scales: NDArray[np.floating[Any]],
    wavelet: str = "morl",
    *,
    sampling_period: float = 1.0,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Compute continuous wavelet transform.

    Wrapper around PyWavelets CWT for in-memory processing.

    Args:
        signal: Input signal array.
        scales: Array of scales for CWT computation.
        wavelet: Wavelet name (default 'morl' - Morlet).
        sampling_period: Sampling period (default 1.0).

    Returns:
        Tuple of (coefficients, frequencies):
            - coefficients: Complex CWT coefficients [scales x time]
            - frequencies: Corresponding frequencies for each scale

    Raises:
        ImportError: If PyWavelets is not available.

    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
        >>> scales = np.arange(1, 128)
        >>> coeffs, freqs = cwt(signal, scales)
        >>> print(f"CWT shape: {coeffs.shape}")
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets not available. Install with: pip install PyWavelets")

    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)

    return coefficients.astype(np.complex128), frequencies.astype(np.float64)


def dwt(
    signal: NDArray[np.floating[Any]],
    wavelet: str = "db4",
    level: int | None = None,
    *,
    mode: str = "symmetric",
) -> list[NDArray[np.float64]]:
    """Compute discrete wavelet transform.

    Wrapper around PyWavelets DWT for in-memory processing.

    Args:
        signal: Input signal array.
        wavelet: Wavelet name (default 'db4' - Daubechies 4).
        level: Decomposition level. If None, uses maximum possible.
        mode: Signal extension mode (default 'symmetric').

    Returns:
        List of wavelet coefficients [cA_n, cD_n, cD_n-1, ..., cD1]:
            - cA_n: Approximation coefficients at level n
            - cD_i: Detail coefficients at level i

    Raises:
        ImportError: If PyWavelets is not available.

    Example:
        >>> import numpy as np
        >>> signal = np.random.randn(1024)
        >>> coeffs = dwt(signal, wavelet='db4', level=3)
        >>> print(f"Approximation: {coeffs[0].shape}")
        >>> print(f"Details: {[c.shape for c in coeffs[1:]]}")
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets not available. Install with: pip install PyWavelets")

    coeffs = pywt.wavedec(signal, wavelet, mode=mode, level=level)

    return [c.astype(np.float64) for c in coeffs]


__all__ = ["cwt", "cwt_chunked", "dwt", "dwt_chunked"]
