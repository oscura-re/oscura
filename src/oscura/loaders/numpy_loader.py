"""NumPy NPZ file loader.

This module provides loading of waveform data from NumPy .npz archive files.


Example:
    >>> from oscura.loaders.numpy_loader import load_npz
    >>> trace = load_npz("waveform.npz")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike


# Common array names for waveform data
DATA_ARRAY_NAMES = ["data", "waveform", "signal", "samples", "y", "voltage"]

# Common metadata keys
SAMPLE_RATE_KEYS = ["sample_rate", "samplerate", "fs", "sampling_rate", "rate"]
VERTICAL_SCALE_KEYS = ["vertical_scale", "v_scale", "scale", "volts_per_div"]
VERTICAL_OFFSET_KEYS = ["vertical_offset", "v_offset", "offset"]


def load_npz(
    path: str | PathLike[str],
    *,
    channel: str | int | None = None,
    sample_rate: float | None = None,
    mmap: bool = False,
) -> WaveformTrace:
    """Load waveform data from a NumPy NPZ archive.


    Extracts waveform array and metadata from an NPZ file. The function
    looks for common array names like 'data', 'waveform', 'signal', etc.

    Args:
        path: Path to the .npz file.
        channel: Specific array name or index to load. If None, auto-detects.
        sample_rate: Override sample rate (if not found in file metadata).
        mmap: If True, use memory mapping to avoid loading entire file into RAM.
            Data stays on disk until accessed. Useful for very large files.

    Returns:
        WaveformTrace containing the waveform data and metadata.

    Raises:
        LoaderError: If the file cannot be loaded.
        FormatError: If no valid waveform data is found.

    Example:
        >>> trace = load_npz("waveform.npz")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")

        >>> # Load specific channel from multi-channel file
        >>> trace = load_npz("multi.npz", channel="ch1")

        >>> # Memory-map large file to avoid loading all into RAM
        >>> trace = load_npz("large.npz", mmap=True)
        >>> # Access only what you need: trace.data[1000:2000]

    Security Warning:
        NPZ files may contain pickled Python objects. Only load NPZ files from
        trusted sources. Loading malicious NPZ files could execute arbitrary
        code. For untrusted data, prefer formats like plain NumPy arrays (.npy),
        CSV, or HDF5.
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    try:
        npz = np.load(path, allow_pickle=True, mmap_mode="r" if mmap else None)
    except Exception as e:
        raise LoaderError(
            "Failed to load NPZ file",
            file_path=str(path),
            details=str(e),
        ) from e

    try:
        # Find waveform data array
        data_array = _find_data_array(npz, channel)

        if data_array is None:
            available = list(npz.keys())
            raise FormatError(
                "No waveform data found in NPZ file",
                file_path=str(path),
                expected=f"Array named: {', '.join(DATA_ARRAY_NAMES)}",
                got=f"Arrays: {', '.join(available)}",
            )

        # Convert to float64 (keep mmap if enabled)
        if mmap and isinstance(data_array, np.memmap):
            # Keep as memmap, just ensure float64 dtype
            if data_array.dtype != np.float64:
                # For memmap, we need to copy to convert dtype
                try:
                    data = data_array.astype(np.float64)
                except (ValueError, TypeError) as e:
                    raise FormatError(
                        "Data array is not numeric",
                        file_path=str(path),
                        expected="Numeric dtype (int, float)",
                        got=f"{data_array.dtype}",
                    ) from e
            else:
                data = data_array
        else:
            try:
                data = data_array.astype(np.float64)
            except (ValueError, TypeError) as e:
                raise FormatError(
                    "Data array is not numeric",
                    file_path=str(path),
                    expected="Numeric dtype (int, float)",
                    got=f"{data_array.dtype}",
                ) from e

        # Extract metadata
        detected_sample_rate = _find_metadata_value(npz, SAMPLE_RATE_KEYS)
        detected_vertical_scale = _find_metadata_value(npz, VERTICAL_SCALE_KEYS)
        detected_vertical_offset = _find_metadata_value(npz, VERTICAL_OFFSET_KEYS)

        # Use provided sample_rate if specified
        if sample_rate is not None:
            detected_sample_rate = sample_rate
        elif detected_sample_rate is None:
            detected_sample_rate = 1e6  # Default to 1 MSa/s

        # Build metadata
        metadata = TraceMetadata(
            sample_rate=float(detected_sample_rate),
            vertical_scale=float(detected_vertical_scale)
            if detected_vertical_scale is not None
            else None,
            vertical_offset=float(detected_vertical_offset)
            if detected_vertical_offset is not None
            else None,
            source_file=str(path),
            channel_name=_get_channel_name(npz, channel),
        )

        return WaveformTrace(data=data, metadata=metadata)

    finally:
        npz.close()


def _find_data_array(
    npz: np.lib.npyio.NpzFile,
    channel: str | int | None,
) -> NDArray[np.float64] | None:
    """Find the waveform data array in the NPZ file.

    Args:
        npz: Loaded NPZ file.
        channel: Specific channel name or index.

    Returns:
        Waveform data array or None if not found.
    """
    keys = list(npz.keys())

    # If channel specified by name
    if isinstance(channel, str):
        if channel in keys:
            return npz[channel]
        # Try case-insensitive match
        channel_lower = channel.lower()
        for key in keys:
            if key.lower() == channel_lower:
                return npz[key]
        return None

    # If channel specified by index
    if isinstance(channel, int):
        # Find numeric-suffixed arrays (ch1, ch2, etc.)
        channel_arrays = [k for k in keys if _is_channel_array(k)]
        if channel_arrays and channel < len(channel_arrays):
            return npz[sorted(channel_arrays)[channel]]
        # Or use nth array
        data_arrays = [k for k in keys if _is_data_array(k)]
        if data_arrays and channel < len(data_arrays):
            return npz[data_arrays[channel]]
        return None

    # Auto-detect: look for common data array names
    for name in DATA_ARRAY_NAMES:
        if name in keys:
            return npz[name]
        # Try case-insensitive match
        name_lower = name.lower()
        for key in keys:
            if key.lower() == name_lower:
                return npz[key]

    # Fall back to first 1D or 2D array
    for key in keys:
        arr = npz[key]
        if isinstance(arr, np.ndarray) and arr.ndim in (1, 2):
            # Skip metadata scalars
            if arr.size > 10:  # Arbitrary threshold
                return arr.ravel() if arr.ndim == 2 else arr

    return None


def _is_channel_array(name: str) -> bool:
    """Check if array name looks like a channel (ch1, channel1, etc.)."""
    name_lower = name.lower()
    return (
        name_lower.startswith("ch")
        or name_lower.startswith("channel")
        or name_lower.startswith("analog")
    )


def _is_data_array(name: str) -> bool:
    """Check if array name looks like waveform data."""
    name_lower = name.lower()
    return any(data_name in name_lower for data_name in DATA_ARRAY_NAMES)


def _find_metadata_value(
    npz: np.lib.npyio.NpzFile,
    key_names: list[str],
) -> float | None:
    """Find a metadata value by trying multiple key names.

    Args:
        npz: Loaded NPZ file.
        key_names: List of possible key names to try.

    Returns:
        Metadata value or None if not found.
    """
    keys = list(npz.keys())

    for name in key_names:
        # Exact match
        if name in keys:
            value = npz[name]
            if np.isscalar(value):
                return float(value)  # type: ignore[arg-type]
            elif isinstance(value, np.ndarray) and value.size == 1:
                return float(value.item())  # type: ignore[arg-type]

        # Case-insensitive match
        name_lower = name.lower()
        for key in keys:
            if key.lower() == name_lower:
                value = npz[key]
                if np.isscalar(value):
                    return float(value)  # type: ignore[arg-type]
                elif isinstance(value, np.ndarray) and value.size == 1:
                    return float(value.item())  # type: ignore[arg-type]

    # Check for metadata dict
    if "metadata" in keys:
        metadata = npz["metadata"]
        if isinstance(metadata, np.ndarray):
            try:
                meta_dict = metadata.item()
                if isinstance(meta_dict, dict):
                    for name in key_names:
                        if name in meta_dict:
                            return float(meta_dict[name])
            except (ValueError, TypeError):
                pass

    return None


def _get_channel_name(
    npz: np.lib.npyio.NpzFile,
    channel: str | int | None,
) -> str:
    """Get a channel name for the loaded data.

    Args:
        npz: Loaded NPZ file.
        channel: Channel specification.

    Returns:
        Channel name string.
    """
    if isinstance(channel, str):
        return channel
    elif isinstance(channel, int):
        return f"CH{channel + 1}"

    # Try to find channel name in metadata
    keys = list(npz.keys())
    if "channel_name" in keys:
        value = npz["channel_name"]
        # NPZ values are always ndarrays
        return str(value.item())

    return "CH1"


def list_arrays(path: str | PathLike[str]) -> list[str]:
    """List all arrays in an NPZ file.

    Args:
        path: Path to the NPZ file.

    Returns:
        List of array names.

    Raises:
        LoaderError: If file not found or cannot be read.

    Example:
        >>> arrays = list_arrays("multi.npz")
        >>> print(arrays)
        ['ch1', 'ch2', 'sample_rate']
    """
    path = Path(path)
    if not path.exists():
        raise LoaderError("File not found", file_path=str(path))

    try:
        with np.load(path, allow_pickle=True) as npz:
            return list(npz.keys())
    except Exception as e:
        raise LoaderError(
            "Failed to read NPZ file",
            file_path=str(path),
            details=str(e),
        ) from e


def load_raw_binary(
    path: str | PathLike[str],
    *,
    dtype: str = "float32",
    sample_rate: float = 1e6,
    mmap: bool = False,
    offset: int = 0,
    count: int = -1,
) -> WaveformTrace:
    """Load waveform data from a raw binary file.


    Loads raw binary waveform data with optional memory mapping for
    files larger than available RAM.

    Args:
        path: Path to the raw binary file.
        dtype: Data type of samples (float32, float64, int16, etc.).
        sample_rate: Sample rate in Hz.
        mmap: If True, use memory mapping to avoid loading entire file.
        offset: Number of elements to skip at start of file.
        count: Number of elements to read (-1 = all).

    Returns:
        WaveformTrace containing the waveform data and metadata.

    Raises:
        LoaderError: If the file cannot be loaded.

    Example:
        >>> # Load entire file into memory
        >>> trace = load_raw_binary("signal.bin", dtype="float32", sample_rate=1e9)

        >>> # Memory-map large file
        >>> trace = load_raw_binary("large.bin", dtype="float32", sample_rate=1e9, mmap=True)
        >>> # Access subset: trace.data[1000:2000]

        >>> # Load only portion of file
        >>> trace = load_raw_binary("signal.bin", dtype="int16", offset=1000, count=10000)
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError("File not found", file_path=str(path))

    try:
        data: NDArray[np.float64] | np.memmap[Any, np.dtype[Any]]
        if mmap:
            # Memory-mapped array (stays on disk)
            data = np.memmap(
                path,
                dtype=dtype,
                mode="r",
                offset=offset * np.dtype(dtype).itemsize,
                shape=(count,) if count > 0 else None,
            )
            # Convert to float64 if needed (may copy)
            if data.dtype != np.float64:
                # For large files, user should slice before converting
                # data = data.astype(np.float64)  # This would load entire file!
                # Instead, keep original dtype and convert in WaveformTrace
                pass
        else:
            # Load into memory
            data_raw = np.fromfile(path, dtype=dtype, count=count, offset=offset)
            # Convert to float64
            data = data_raw.astype(np.float64)

        metadata = TraceMetadata(
            sample_rate=sample_rate,
            source_file=str(path),
            channel_name="RAW",
        )

        return WaveformTrace(data=data, metadata=metadata)

    except Exception as e:
        raise LoaderError(
            "Failed to load raw binary file",
            file_path=str(path),
            details=str(e),
        ) from e


__all__ = ["list_arrays", "load_npz", "load_raw_binary"]
