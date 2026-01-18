"""Binary file loader for raw signal data.

Loads raw binary files containing signal data with user-specified format.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike


def load_binary(
    path: str | PathLike[str],
    *,
    dtype: str | np.dtype[Any] = "float64",
    sample_rate: float = 1.0,
    channels: int = 1,
    channel: int = 0,
    offset: int = 0,
    count: int = -1,
) -> WaveformTrace:
    """Load raw binary file as waveform trace.

    Args:
        path: Path to the binary file.
        dtype: NumPy dtype for the data (default: float64).
        sample_rate: Sample rate in Hz.
        channels: Number of interleaved channels.
        channel: Channel index to load (0-based).
        offset: Number of samples to skip from start.
        count: Number of samples to read (-1 for all).

    Returns:
        WaveformTrace containing the loaded data.

    Example:
        >>> from oscura.loaders.binary import load_binary
        >>> trace = load_binary("signal.bin", dtype="int16", sample_rate=1e6)
    """
    path = Path(path)

    # Load raw data
    data = np.fromfile(path, dtype=dtype, count=count, offset=offset * np.dtype(dtype).itemsize)

    # Handle multi-channel data
    if channels > 1:
        # Reshape and select channel
        samples_per_channel = len(data) // channels
        data = data[: samples_per_channel * channels].reshape(-1, channels)
        data = data[:, channel]

    # Create metadata
    metadata = TraceMetadata(
        sample_rate=sample_rate,
        source_file=str(path),
        channel_name=f"Channel {channel}",
    )

    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


__all__ = ["load_binary"]
