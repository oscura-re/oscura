"""Sigrok session file (.sr) loader.

This module provides loading of sigrok session files containing
logic analyzer captures. Sigrok sessions are ZIP archives containing
metadata and binary signal data.


Example:
    >>> from oscura.loaders.sigrok import load_sigrok
    >>> trace = load_sigrok("capture.sr")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
    >>> print(f"Channels: {len(trace.data)}")
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import DigitalTrace, TraceMetadata

if TYPE_CHECKING:
    from os import PathLike


def load_sigrok(
    path: str | PathLike[str],
    *,
    channel: str | int | None = None,
) -> DigitalTrace:
    """Load a sigrok session file (.sr).

    Sigrok session files are ZIP archives containing:
    - metadata: JSON file with capture settings
    - logic-1-*: Binary files with sample data

    Args:
        path: Path to the sigrok .sr session file.
        channel: Optional channel name or index to load. If None,
            loads the first channel or merges all channels.

    Returns:
        DigitalTrace containing the digital signal data and metadata.

    Raises:
        LoaderError: If the file cannot be loaded.
        FormatError: If the file is not a valid sigrok session.

    Example:
        >>> trace = load_sigrok("capture.sr")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
        >>> print(f"Duration: {trace.duration:.6f} seconds")

    References:
        sigrok session file format specification
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    if not zipfile.is_zipfile(path):
        raise FormatError(
            "File is not a valid sigrok session (not a ZIP archive)",
            file_path=str(path),
            expected="ZIP archive",
        )

    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Parse metadata
            metadata_dict = _parse_metadata(zf, path)

            # Get sample rate from metadata
            sample_rate = metadata_dict.get("samplerate", 1_000_000)

            # Get channel information
            channels = metadata_dict.get("channels", [])
            total_channels = metadata_dict.get("total probes", len(channels))

            # Find and read logic data files
            logic_files = [name for name in zf.namelist() if name.startswith("logic-1")]

            if not logic_files:
                raise FormatError(
                    "No logic data found in sigrok session",
                    file_path=str(path),
                    expected="logic-1-* data files",
                )

            # Read and combine logic data
            data = _read_logic_data(zf, logic_files, total_channels)

            # Select specific channel if requested
            if channel is not None:
                if isinstance(channel, int):
                    if channel < 0 or channel >= data.shape[0]:
                        raise LoaderError(
                            f"Channel index {channel} out of range",
                            file_path=str(path),
                            details=f"Available channels: 0-{data.shape[0] - 1}",
                        )
                    channel_data = data[channel]
                    channel_name = channels[channel] if channel < len(channels) else f"D{channel}"
                elif isinstance(channel, str):
                    if channel in channels:
                        idx = channels.index(channel)
                        channel_data = data[idx]
                        channel_name = channel
                    else:
                        raise LoaderError(
                            f"Channel '{channel}' not found",
                            file_path=str(path),
                            details=f"Available channels: {channels}",
                        )
                else:
                    channel_data = data[0]  # type: ignore[unreachable]
                    channel_name = channels[0] if channels else "D0"
            else:
                # Default to first channel
                channel_data = data[0] if data.ndim > 1 else data
                channel_name = channels[0] if channels else "D0"

            # Compute edges
            edges = _compute_edges(channel_data, sample_rate)

            # Build metadata
            trace_metadata = TraceMetadata(
                sample_rate=float(sample_rate),
                source_file=str(path),
                channel_name=channel_name,
                trigger_info=metadata_dict.get("trigger", None),
            )

            return DigitalTrace(
                data=channel_data,
                metadata=trace_metadata,
                edges=edges,
            )

    except zipfile.BadZipFile as e:
        raise FormatError(
            "Corrupted sigrok session file",
            file_path=str(path),
            expected="Valid ZIP archive",
        ) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load sigrok session",
            file_path=str(path),
            details=str(e),
            fix_hint="Ensure the file is a valid sigrok session (.sr) file.",
        ) from e


def _parse_metadata(zf: zipfile.ZipFile, path: Path) -> dict[str, Any]:
    """Parse sigrok session metadata.

    Args:
        zf: Open ZipFile object.
        path: Path to the session file (for error messages).

    Returns:
        Dictionary of metadata values.
    """
    metadata: dict[str, Any] = {}

    # Try to read metadata file (JSON format in newer versions)
    if "metadata" in zf.namelist():
        try:
            with zf.open("metadata") as f:
                content = f.read().decode("utf-8")
                # Parse key=value format (sigrok classic format)
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Try to convert numeric values
                        try:
                            if "." in value:
                                metadata[key] = float(value)
                            else:
                                metadata[key] = int(value)
                        except ValueError:
                            metadata[key] = value
        except Exception:
            pass  # Use defaults if metadata parsing fails

    # Extract channel names from probe entries
    channels: list[str] = []
    for key, value in metadata.items():
        if key.startswith("probe"):
            try:
                idx = int(key.replace("probe", ""))
                while len(channels) <= idx:
                    channels.append(f"D{len(channels)}")
                channels[idx] = value
            except ValueError:
                pass

    if channels:
        metadata["channels"] = channels

    return metadata


def _read_logic_data(
    zf: zipfile.ZipFile,
    logic_files: list[str],
    total_channels: int,
) -> NDArray[np.bool_]:
    """Read and decode logic data from sigrok session.

    Args:
        zf: Open ZipFile object.
        logic_files: List of logic data file names.
        total_channels: Total number of digital channels.

    Returns:
        Boolean array of shape (channels, samples).
    """
    # Sort logic files to ensure correct order
    logic_files = sorted(logic_files)

    # Determine bytes per sample based on channel count
    bytes_per_sample = (total_channels + 7) // 8

    # Read all logic data
    all_data = []
    for logic_file in logic_files:
        with zf.open(logic_file) as f:
            raw_data = f.read()
            all_data.append(raw_data)

    # Combine data
    combined = b"".join(all_data)

    # Convert to numpy array
    if bytes_per_sample == 1:
        raw = np.frombuffer(combined, dtype=np.uint8)
    elif bytes_per_sample == 2:
        raw = np.frombuffer(combined, dtype=np.uint16)
    elif bytes_per_sample <= 4:
        # Pad to 4 bytes and read as uint32
        padded = combined + b"\x00" * (len(combined) % 4)
        raw = np.frombuffer(padded, dtype=np.uint32)
    else:
        # Handle larger sample widths
        raw = np.frombuffer(combined, dtype=np.uint8)

    # Extract individual channel bits
    n_samples = len(raw)
    channels_data = np.zeros((total_channels, n_samples), dtype=np.bool_)

    for ch in range(total_channels):
        if bytes_per_sample <= 4:
            channels_data[ch] = (raw >> ch) & 1
        else:
            # For larger widths, calculate byte and bit position
            byte_idx = ch // 8
            bit_idx = ch % 8
            byte_data = raw[byte_idx::bytes_per_sample]
            channels_data[ch, : len(byte_data)] = (byte_data >> bit_idx) & 1

    return channels_data


def _compute_edges(
    data: NDArray[np.bool_],
    sample_rate: float,
) -> list[tuple[float, bool]]:
    """Compute edge timestamps from digital data.

    Args:
        data: Boolean array of digital samples.
        sample_rate: Sample rate in Hz.

    Returns:
        List of (timestamp, is_rising) tuples.
    """
    edges: list[tuple[float, bool]] = []

    if len(data) < 2:
        return edges

    # Find transitions
    diff = np.diff(data.astype(np.int8))
    rising_indices = np.where(diff == 1)[0]
    falling_indices = np.where(diff == -1)[0]

    time_per_sample = 1.0 / sample_rate

    # Add rising edges
    for idx in rising_indices:
        timestamp = (idx + 1) * time_per_sample
        edges.append((timestamp, True))

    # Add falling edges
    for idx in falling_indices:
        timestamp = (idx + 1) * time_per_sample
        edges.append((timestamp, False))

    # Sort by timestamp
    edges.sort(key=lambda x: x[0])

    return edges


__all__ = ["load_sigrok"]
