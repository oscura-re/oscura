"""NI TDMS (Technical Data Management Streaming) file loader.

This module provides loading of NI LabVIEW TDMS files using the
npTDMS library when available.


Example:
    >>> from oscura.loaders.tdms import load_tdms
    >>> trace = load_tdms("measurement.tdms")
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

# Try to import npTDMS for TDMS support
try:
    from nptdms import TdmsFile

    NPTDMS_AVAILABLE = True
except ImportError:
    NPTDMS_AVAILABLE = False


def load_tdms(
    path: str | PathLike[str],
    *,
    channel: str | int | None = None,
    group: str | None = None,
) -> WaveformTrace:
    """Load an NI TDMS file.

    TDMS files contain hierarchical data with groups and channels.
    Each channel can have associated properties including sample rate.

    Args:
        path: Path to the TDMS file.
        channel: Channel name or index to load. If None, loads the
            first channel found.
        group: Group name to select from. If None, uses the first group.

    Returns:
        WaveformTrace containing the channel data and metadata.

    Raises:
        Exception: If the file cannot be read or parsed.
        LoaderError: If the file cannot be loaded or npTDMS is not installed.

    Example:
        >>> trace = load_tdms("measurement.tdms", group="Voltage", channel="CH1")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
        >>> print(f"Duration: {trace.duration:.6f} seconds")

    References:
        NI TDMS File Format: https://www.ni.com/en-us/support/documentation/
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    if not NPTDMS_AVAILABLE:
        raise LoaderError(
            "npTDMS library required for TDMS files",
            file_path=str(path),
            fix_hint="Install npTDMS: pip install npTDMS",
        )

    try:
        return _load_with_nptdms(path, channel=channel, group=group)
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load TDMS file",
            file_path=str(path),
            details=str(e),
            fix_hint="Ensure the file is a valid NI TDMS format.",
        ) from e


def _load_with_nptdms(
    path: Path,
    *,
    channel: str | int | None = None,
    group: str | None = None,
) -> WaveformTrace:
    """Load TDMS using npTDMS library.

    Args:
        path: Path to the TDMS file.
        channel: Channel name or index.
        group: Group name to select.

    Returns:
        WaveformTrace with channel data and metadata.

    Raises:
        FormatError: If file is not valid TDMS format or has no data.
        LoaderError: If channel or group not found.
    """
    tdms_file = _parse_tdms_file(path)
    target_group = _select_tdms_group(tdms_file, group, path)
    target_channel = _select_tdms_channel(target_group, channel, path)
    data = _extract_channel_data(target_channel, path)
    metadata = _build_tdms_metadata(target_channel, target_group, tdms_file, path)

    return WaveformTrace(data=data, metadata=metadata)


def _parse_tdms_file(path: Path) -> Any:
    """Parse TDMS file and validate structure."""
    try:
        tdms_file = TdmsFile.read(str(path))
    except Exception as e:
        raise FormatError(
            "Failed to parse TDMS file",
            file_path=str(path),
            expected="Valid NI TDMS format",
        ) from e

    groups = list(tdms_file.groups())
    if not groups:
        raise FormatError(
            "No groups found in TDMS file",
            file_path=str(path),
        )

    return tdms_file


def _select_tdms_group(tdms_file: Any, group: str | None, path: Path) -> Any:
    """Select target group from TDMS file."""
    groups = list(tdms_file.groups())

    if group is None:
        return groups[0]

    for g in groups:
        if g.name == group:
            return g

    available_groups = [g.name for g in groups]
    raise LoaderError(
        f"Group '{group}' not found",
        file_path=str(path),
        details=f"Available groups: {available_groups}",
    )


def _select_tdms_channel(target_group: Any, channel: str | int | None, path: Path) -> Any:
    """Select target channel from TDMS group."""
    channels = list(target_group.channels())

    if not channels:
        raise FormatError(
            f"No channels found in group '{target_group.name}'",
            file_path=str(path),
        )

    if channel is None:
        return channels[0]

    if isinstance(channel, int):
        return _select_channel_by_index(channels, channel, path)
    elif isinstance(channel, str):
        return _select_channel_by_name(channels, channel, path)
    else:
        return channels[0]  # type: ignore[unreachable]


def _select_channel_by_index(channels: list[Any], channel: int, path: Path) -> Any:
    """Select channel by index."""
    if channel < 0 or channel >= len(channels):
        raise LoaderError(
            f"Channel index {channel} out of range",
            file_path=str(path),
            details=f"Available channels: 0-{len(channels) - 1}",
        )
    return channels[channel]


def _select_channel_by_name(channels: list[Any], channel: str, path: Path) -> Any:
    """Select channel by name."""
    for ch in channels:
        if ch.name == channel:
            return ch

    available_channels = [ch.name for ch in channels]
    raise LoaderError(
        f"Channel '{channel}' not found",
        file_path=str(path),
        details=f"Available channels: {available_channels}",
    )


def _extract_channel_data(target_channel: Any, path: Path) -> NDArray[np.float64]:
    """Extract and validate channel data."""
    data = target_channel.data
    if data is None or len(data) == 0:
        raise FormatError(
            f"Channel '{target_channel.name}' has no data",
            file_path=str(path),
        )

    return np.asarray(data, dtype=np.float64)


def _build_tdms_metadata(
    target_channel: Any,
    target_group: Any,
    tdms_file: Any,
    path: Path,
) -> TraceMetadata:
    """Build metadata from TDMS channel properties."""
    sample_rate = _get_sample_rate(target_channel, target_group, tdms_file)
    vertical_scale = target_channel.properties.get("NI_Scale[0]_Linear_Slope")
    vertical_offset = target_channel.properties.get("NI_Scale[0]_Linear_Y_Intercept")

    return TraceMetadata(
        sample_rate=sample_rate,
        vertical_scale=float(vertical_scale) if vertical_scale is not None else None,
        vertical_offset=float(vertical_offset) if vertical_offset is not None else None,
        source_file=str(path),
        channel_name=target_channel.name,
        trigger_info=_extract_tdms_properties(target_channel),
    )


def _get_sample_rate(
    channel: Any,
    group: Any,
    tdms_file: Any,
) -> float:
    """Extract sample rate from TDMS channel properties.

    Checks multiple common property names used by different NI software.

    Args:
        channel: TDMS channel object.
        group: TDMS group object.
        tdms_file: TDMS file object.

    Returns:
        Sample rate in Hz.
    """
    # Common property names for sample rate
    sample_rate_keys = [
        "wf_samples",  # DAQmx
        "wf_increment",  # Waveform dt (inverse of sample rate)
        "NI_RF_IQ_Rate",  # RF signal analyzer
        "SamplingFrequency",  # SignalExpress
        "dt",  # Delta time
        "Fs",  # Sample rate
        "SampleRate",
        "sample_rate",
    ]

    # Check channel properties
    for key in sample_rate_keys:
        value = channel.properties.get(key)
        if value is not None:
            if key in ("wf_increment", "dt"):
                # These are time intervals, invert for sample rate
                if value > 0:
                    return 1.0 / float(value)
            else:
                return float(value)

    # Check group properties
    for key in sample_rate_keys:
        value = group.properties.get(key)
        if value is not None:
            if key in ("wf_increment", "dt"):
                if value > 0:
                    return 1.0 / float(value)
            else:
                return float(value)

    # Check file properties
    for key in sample_rate_keys:
        value = tdms_file.properties.get(key)
        if value is not None:
            if key in ("wf_increment", "dt"):
                if value > 0:
                    return 1.0 / float(value)
            else:
                return float(value)

    # Default sample rate if not found
    return 1.0e6  # 1 MHz default


def _extract_tdms_properties(channel: Any) -> dict[str, Any] | None:
    """Extract relevant properties from TDMS channel.

    Args:
        channel: TDMS channel object.

    Returns:
        Dictionary of properties, or None if no useful properties found.
    """
    props: dict[str, Any] = {}

    # Common useful properties
    useful_keys = [
        "unit_string",
        "NI_ChannelName",
        "wf_start_time",
        "wf_start_offset",
        "description",
        "NI_Scale[0]_Linear_Slope",
        "NI_Scale[0]_Linear_Y_Intercept",
    ]

    for key in useful_keys:
        value = channel.properties.get(key)
        if value is not None:
            props[key] = value

    return props if props else None


def list_tdms_channels(
    path: str | PathLike[str],
) -> dict[str, list[str]]:
    """List all groups and channels in a TDMS file.

    Args:
        path: Path to the TDMS file.

    Returns:
        Dictionary mapping group names to lists of channel names.

    Raises:
        LoaderError: If the file cannot be loaded.

    Example:
        >>> channels = list_tdms_channels("measurement.tdms")
        >>> for group, chans in channels.items():
        ...     print(f"Group '{group}': {chans}")
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    if not NPTDMS_AVAILABLE:
        raise LoaderError(
            "npTDMS library required for TDMS files",
            file_path=str(path),
            fix_hint="Install npTDMS: pip install npTDMS",
        )

    try:
        tdms_file = TdmsFile.read(str(path))
        result: dict[str, list[str]] = {}

        for group in tdms_file.groups():
            channel_names = [ch.name for ch in group.channels()]
            result[group.name] = channel_names

        return result

    except Exception as e:
        raise LoaderError(
            "Failed to read TDMS file",
            file_path=str(path),
            details=str(e),
        ) from e


__all__ = ["list_tdms_channels", "load_tdms"]
