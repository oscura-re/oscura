"""FFT analysis functions.

This module provides FFT-based spectral analysis including chunked processing
for large datasets.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

# Re-export file-based chunked FFT functions
from oscura.analyzers.spectral.chunked_fft import (
    fft_chunked as fft_chunked_file,
)
from oscura.analyzers.spectral.chunked_fft import (
    fft_chunked_parallel,
    streaming_fft,
    welch_psd_chunked,
)


def fft_chunked(
    data: NDArray[np.floating[Any]],
    chunk_size: int = 8192,
    *,
    window: str = "hann",
    overlap_pct: float = 50.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute FFT on large array using chunked processing.

    Processes array in overlapping segments to handle large datasets
    that may not fit in FFT at once.

    Args:
        data: Input signal array.
        chunk_size: Size of each processing chunk.
        window: Window function name.
        overlap_pct: Overlap percentage between chunks (0-100).

    Returns:
        Tuple of (frequencies, magnitudes) arrays.

    Example:
        >>> signal = np.sin(2 * np.pi * 1000 * np.linspace(0, 1, 1000000))
        >>> freqs, mags = fft_chunked(signal, chunk_size=4096)
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    # Calculate overlap
    hop_size = int(chunk_size * (1 - overlap_pct / 100))

    # Generate window
    from scipy import signal as sp_signal

    window_arr = sp_signal.get_window(window, chunk_size)

    # Process chunks
    chunks = []
    for i in range(0, len(data) - chunk_size + 1, hop_size):
        chunk = data[i : i + chunk_size]
        windowed = chunk * window_arr

        # Compute FFT
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)
        chunks.append(magnitude)

    # Average all chunks
    if chunks:
        avg_magnitude = np.mean(chunks, axis=0)
    else:
        # Handle case where signal shorter than chunk_size
        windowed = data * sp_signal.get_window(window, len(data))
        fft_result = np.fft.rfft(windowed)
        avg_magnitude = np.abs(fft_result)

    # Generate frequency array
    frequencies = np.fft.rfftfreq(chunk_size, d=1.0)

    return frequencies, avg_magnitude


__all__ = [
    "fft_chunked",
    "fft_chunked_file",
    "fft_chunked_parallel",
    "streaming_fft",
    "welch_psd_chunked",
]
