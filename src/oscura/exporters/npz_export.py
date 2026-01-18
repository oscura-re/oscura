"""NumPy NPZ export functionality for Oscura.

This module provides export to NumPy's compressed archive format for
efficient storage and fast loading of trace data.


Example:
    >>> from oscura.exporters.npz_export import export_npz
    >>> export_npz(trace, "waveform.npz")
    >>> # Load later with numpy
    >>> import numpy as np
    >>> data = np.load("waveform.npz")
    >>> signal = data['signal']
    >>> sample_rate = float(data['sample_rate'])
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import DigitalTrace, WaveformTrace

if TYPE_CHECKING:
    from numpy.typing import NDArray


def export_npz(
    data: WaveformTrace | DigitalTrace | dict[str, Any] | NDArray[Any],
    path: str | Path,
    *,
    compressed: bool = True,
    include_metadata: bool = True,
    include_time: bool = False,
) -> None:
    """Export data to NumPy NPZ archive format.

    Creates a NumPy .npz file containing the trace data and optional metadata.
    Files can be loaded with `numpy.load()` for fast array access.

    Args:
        data: Data to export. Can be:
            - WaveformTrace or DigitalTrace
            - Dictionary of arrays
            - NumPy array
        path: Output file path (should end with .npz).
        compressed: Use compression (default True). Results in smaller files
            but slightly slower save/load.
        include_metadata: Include metadata in the archive.
        include_time: Include precomputed time array (increases file size).

    Raises:
        TypeError: If data type is not supported.

    Example:
        >>> export_npz(trace, "waveform.npz")
        >>> # Load later
        >>> data = np.load("waveform.npz")
        >>> signal = data['signal']
        >>> sample_rate = float(data['sample_rate'])
        >>> time = np.arange(len(signal)) / sample_rate

    References:
        EXP-004
    """
    path = Path(path)

    # Ensure .npz extension
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")

    if isinstance(data, WaveformTrace | DigitalTrace):
        _export_trace(data, path, compressed, include_metadata, include_time)
    elif isinstance(data, dict):
        _export_dict(data, path, compressed)
    elif isinstance(data, np.ndarray):
        _export_array(data, path, compressed)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def _export_trace(
    trace: WaveformTrace | DigitalTrace,
    path: Path,
    compressed: bool,
    include_metadata: bool,
    include_time: bool,
) -> None:
    """Export trace to NPZ.

    Args:
        trace: Trace to export.
        path: Output file path.
        compressed: Use compression.
        include_metadata: Include metadata arrays.
        include_time: Include time array.
    """
    arrays: dict[str, Any] = {}

    # Main signal data
    arrays["signal"] = trace.data

    # Time array (optional - can be reconstructed from sample_rate)
    if include_time:
        arrays["time"] = trace.time_vector

    # Metadata as scalars
    if include_metadata:
        meta = trace.metadata
        arrays["sample_rate"] = np.array(meta.sample_rate)
        arrays["samples"] = np.array(len(trace.data))

        if hasattr(meta, "channel"):
            arrays["channel"] = np.array(str(meta.channel or ""), dtype="U64")

        if hasattr(meta, "source_file") and meta.source_file:
            arrays["source_file"] = np.array(str(meta.source_file), dtype="U256")

        if hasattr(meta, "capture_time") and meta.capture_time:
            arrays["capture_time"] = np.array(meta.capture_time.isoformat(), dtype="U64")

        if hasattr(meta, "units") and meta.units:
            arrays["units"] = np.array(str(meta.units), dtype="U16")

        # Add trace type marker
        if isinstance(trace, DigitalTrace):
            arrays["trace_type"] = np.array("digital", dtype="U16")
        else:
            arrays["trace_type"] = np.array("waveform", dtype="U16")

    # Save
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _export_dict(
    data: dict[str, Any],
    path: Path,
    compressed: bool,
) -> None:
    """Export dictionary of arrays to NPZ.

    Args:
        data: Dictionary to export.
        path: Output file path.
        compressed: Use compression.
    """
    # Convert values to arrays
    arrays = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arrays[key] = value
        elif isinstance(value, list | tuple | int | float):
            arrays[key] = np.array(value)
        elif isinstance(value, str):
            arrays[key] = np.array(value, dtype="U256")
        else:
            # Try to convert, skip on failure
            with contextlib.suppress(TypeError, ValueError):
                arrays[key] = np.array(value)

    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _export_array(
    data: NDArray[Any],
    path: Path,
    compressed: bool,
) -> None:
    """Export single array to NPZ.

    Args:
        data: Array to export.
        path: Output file path.
        compressed: Use compression.
    """
    if compressed:
        np.savez_compressed(path, data=data)
    else:
        np.savez(path, data=data)


def load_npz(path: str | Path) -> dict[str, NDArray[Any]]:
    """Load NPZ file and return dictionary of arrays.

    Convenience wrapper around numpy.load() that returns a regular dict.

    Args:
        path: Path to NPZ file.

    Returns:
        Dictionary mapping array names to numpy arrays.

    Example:
        >>> data = load_npz("waveform.npz")
        >>> signal = data['signal']
        >>> sample_rate = float(data['sample_rate'])

    References:
        EXP-004
    """
    path = Path(path)

    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


__all__ = [
    "export_npz",
    "load_npz",
]
