"""Tektronix Session File (.tss) Loader.

This module provides loading functionality for Tektronix session files (.tss),
which are ZIP archives containing multiple waveform captures, instrument
configuration, measurements, and annotations.

A .tss session file typically contains:
- Multiple .wfm waveform files (one per channel/capture)
- session.json: Instrument configuration and setup
- measurements.json: Stored measurement results (optional)
- annotations.json: User annotations and markers (optional)

Example:
    >>> import oscura as osc
    >>> trace = osc.load("oscilloscope_session.tss")
    >>> print(f"Channel: {trace.metadata.channel_name}")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")

    >>> # Load specific channel
    >>> trace = osc.load("session.tss", channel="CH2")

    >>> # Load all channels
    >>> channels = osc.load_all_channels("session.tss")
    >>> for name, trace in channels.items():
    ...     print(f"{name}: {len(trace.data)} samples")

References:
    Tektronix Programming Manual for Session Files
    TekScope PC Analysis Software Documentation
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from os import PathLike
from pathlib import Path
from typing import Any

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import DigitalTrace, IQTrace, WaveformTrace


def load_tss(
    path: str | PathLike[str],
    *,
    channel: str | int | None = None,
) -> WaveformTrace | DigitalTrace | IQTrace:
    """Load a Tektronix session file (.tss).

    Tektronix session files are ZIP archives containing multiple waveform
    captures along with instrument configuration and analysis results.

    Args:
        path: Path to the Tektronix .tss session file.
        channel: Optional channel name or index to load. If None,
            loads the first waveform found (alphabetically).
            String names are case-insensitive (e.g., "ch1", "CH1").
            Integer index is 0-based.

    Returns:
        WaveformTrace, DigitalTrace, or IQTrace containing the channel data.

    Raises:
        LoaderError: If the file cannot be loaded or doesn't exist.
        FormatError: If the file is not a valid Tektronix session.

    Example:
        >>> # Load first channel (default)
        >>> trace = load_tss("session.tss")

        >>> # Load specific channel by name
        >>> trace = load_tss("session.tss", channel="CH2")

        >>> # Load by index
        >>> trace = load_tss("session.tss", channel=1)  # Second channel
    """
    path = Path(path)

    # Validate file
    _validate_tss_file(path)

    # Load all waveforms and select requested channel
    waveforms = _load_all_waveforms(path)

    if not waveforms:
        raise FormatError(
            "No waveforms found in session file",
            file_path=str(path),
            expected="At least one .wfm file in the archive",
            got="Empty session or no waveform files",
        )

    # Select channel
    trace, channel_name = _select_channel(waveforms, channel, path)

    # Enrich metadata with session information
    try:
        with zipfile.ZipFile(path, "r") as zf:
            session_metadata = _parse_session_metadata(zf, path)
            trace = _enrich_metadata_from_session(trace, session_metadata, str(path))
    except Exception:
        # Session metadata is optional, continue with waveform metadata
        pass

    return trace


def load_all_channels_tss(
    path: Path,
) -> dict[str, WaveformTrace | DigitalTrace | IQTrace]:
    """Load all channels from a Tektronix session file.

    Args:
        path: Path to the .tss session file.

    Returns:
        Dictionary mapping channel names to traces.
        Channel names are derived from .wfm filenames (e.g., "ch1", "ch2").

    Raises:
        LoaderError: If the file cannot be loaded.
        FormatError: If no waveforms found in session.

    Example:
        >>> channels = load_all_channels_tss(Path("session.tss"))
        >>> for name, trace in channels.items():
        ...     print(f"{name}: {trace.metadata.sample_rate} Hz")
    """
    # Validate file
    _validate_tss_file(path)

    # Load all waveforms
    waveforms = _load_all_waveforms(path)

    if not waveforms:
        raise FormatError(
            "No waveforms found in session file",
            file_path=str(path),
            expected="At least one .wfm file",
            got="Empty archive",
        )

    # Enrich all traces with session metadata
    try:
        with zipfile.ZipFile(path, "r") as zf:
            session_metadata = _parse_session_metadata(zf, path)
            for name in waveforms:
                waveforms[name] = _enrich_metadata_from_session(
                    waveforms[name], session_metadata, str(path)
                )
    except Exception:
        # Session metadata is optional
        pass

    return waveforms


def _validate_tss_file(path: Path) -> None:
    """Validate that file exists and is a valid ZIP archive.

    Args:
        path: Path to validate.

    Raises:
        LoaderError: If file doesn't exist or can't be read.
        FormatError: If file is not a ZIP archive.
    """
    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
            fix_hint="Ensure the file path is correct and the file exists.",
        )

    if not zipfile.is_zipfile(path):
        raise FormatError(
            "Not a valid ZIP archive",
            file_path=str(path),
            expected="Tektronix session file (.tss) is a ZIP archive",
            got="File is not a ZIP file",
        )

    try:
        with zipfile.ZipFile(path, "r") as zf:
            # Test archive integrity
            bad_file = zf.testzip()
            if bad_file is not None:
                raise FormatError(
                    f"Corrupted ZIP archive: {bad_file}",
                    file_path=str(path),
                    expected="Valid ZIP archive",
                    got="Corrupted file detected",
                )
    except zipfile.BadZipFile as e:
        raise FormatError(
            "Corrupted or invalid ZIP archive",
            file_path=str(path),
            details=str(e),
        ) from e


def _parse_session_metadata(zf: zipfile.ZipFile, path: Path) -> dict[str, Any]:
    """Parse session.json metadata from the archive.

    Args:
        zf: Open ZipFile object.
        path: Path to session file (for error messages).

    Returns:
        Dictionary containing session metadata.
        Returns empty dict if session.json not found (non-fatal).
    """
    # Look for session metadata files
    metadata_files = [
        "session.json",
        "Session.json",
        "metadata.json",
        "Metadata.json",
    ]

    for metadata_file in metadata_files:
        try:
            with zf.open(metadata_file) as f:
                data: dict[str, Any] = json.load(f)
                return data
        except KeyError:
            continue
        except json.JSONDecodeError as e:
            # Non-fatal: log warning and continue
            import warnings

            warnings.warn(
                f"Failed to parse {metadata_file} in {path.name}: {e}",
                stacklevel=2,
            )
            return {}

    # Session metadata is optional
    return {}


def _extract_waveform_list(zf: zipfile.ZipFile) -> list[str]:
    """Extract list of .wfm files in the session.

    Args:
        zf: Open ZipFile object.

    Returns:
        List of .wfm file paths within the archive.
        Excludes macOS metadata files (__MACOSX).
    """
    return [
        name
        for name in zf.namelist()
        if name.lower().endswith(".wfm") and not name.startswith("__MACOSX")
    ]


def _load_wfm_from_archive(
    zf: zipfile.ZipFile,
    wfm_name: str,
    path: Path,
) -> WaveformTrace | DigitalTrace | IQTrace:
    """Extract and load a .wfm file from the archive.

    Args:
        zf: Open ZipFile object.
        wfm_name: Name of .wfm file within archive.
        path: Path to session file (for error messages).

    Returns:
        Loaded trace from the waveform file.

    Raises:
        LoaderError: If waveform cannot be loaded.
    """
    try:
        # Extract waveform to temporary file
        # (load_tektronix_wfm expects file path, not bytes)
        wfm_bytes = zf.read(wfm_name)

        with tempfile.NamedTemporaryFile(suffix=".wfm", delete=True) as tmp:
            tmp.write(wfm_bytes)
            tmp.flush()

            # Use existing Tektronix loader
            from oscura.loaders.tektronix import load_tektronix_wfm

            trace = load_tektronix_wfm(tmp.name)

            return trace

    except Exception as e:
        raise LoaderError(
            f"Failed to load waveform from session: {wfm_name}",
            file_path=str(path),
            details=str(e),
            fix_hint="Waveform file may be corrupted or incompatible.",
        ) from e


def _load_all_waveforms(path: Path) -> dict[str, WaveformTrace | DigitalTrace | IQTrace]:
    """Load all waveforms from the session file.

    Args:
        path: Path to .tss session file.

    Returns:
        Dictionary mapping channel names to traces.
    """
    waveforms: dict[str, WaveformTrace | DigitalTrace | IQTrace] = {}

    with zipfile.ZipFile(path, "r") as zf:
        wfm_files = _extract_waveform_list(zf)

        for wfm_name in sorted(wfm_files):  # Sort for consistent ordering
            # Derive channel name from filename
            channel_name = _derive_channel_name(wfm_name)

            # Load waveform
            trace = _load_wfm_from_archive(zf, wfm_name, path)

            # Store with normalized channel name
            waveforms[channel_name] = trace

    return waveforms


def _derive_channel_name(wfm_filename: str) -> str:
    """Derive channel name from .wfm filename.

    Args:
        wfm_filename: Filename like "CH1.wfm", "CH2_Voltage.wfm", etc.

    Returns:
        Normalized channel name (lowercase, e.g., "ch1", "ch2", "d0").

    Examples:
        >>> _derive_channel_name("CH1.wfm")
        'ch1'
        >>> _derive_channel_name("subdir/CH2_Voltage.wfm")
        'ch2'
        >>> _derive_channel_name("D0.wfm")
        'd0'
        >>> _derive_channel_name("MATH1.wfm")
        'math1'
    """
    # Get base filename without path
    basename = Path(wfm_filename).stem  # Remove extension

    # Remove path components if nested
    basename = basename.split("/")[-1].split("\\")[-1]

    # Extract channel identifier (first part before underscore)
    channel_id = basename.split("_")[0]

    # Normalize to lowercase
    return channel_id.lower()


def _select_channel(
    waveforms: dict[str, WaveformTrace | DigitalTrace | IQTrace],
    channel: str | int | None,
    path: Path,
) -> tuple[WaveformTrace | DigitalTrace | IQTrace, str]:
    """Select specific channel from waveforms dictionary.

    Args:
        waveforms: Dictionary of channel name to trace.
        channel: Channel selector (name, index, or None for first).
        path: Path to session file (for error messages).

    Returns:
        Tuple of (selected_trace, channel_name).

    Raises:
        LoaderError: If channel not found or index out of range.
    """
    if channel is None:
        # Default: first channel (alphabetically sorted)
        channel_name = sorted(waveforms.keys())[0]
        return waveforms[channel_name], channel_name

    if isinstance(channel, int):
        # Select by index
        channel_names = sorted(waveforms.keys())
        if channel < 0 or channel >= len(channel_names):
            raise LoaderError(
                f"Channel index {channel} out of range",
                file_path=str(path),
                fix_hint=f"Available channels: {', '.join(channel_names)} (indices 0-{len(channel_names) - 1})",
            )
        channel_name = channel_names[channel]
        return waveforms[channel_name], channel_name

    # Select by name (case-insensitive)
    channel_lower = channel.lower()
    for name, trace in waveforms.items():
        if name.lower() == channel_lower:
            return trace, name

    # Channel not found
    available = ", ".join(sorted(waveforms.keys()))
    raise LoaderError(
        f"Channel '{channel}' not found in session",
        file_path=str(path),
        fix_hint=f"Available channels: {available}",
    )


def _enrich_metadata_from_session(
    trace: WaveformTrace | DigitalTrace | IQTrace,
    session_metadata: dict[str, Any],
    source_file: str,
) -> WaveformTrace | DigitalTrace | IQTrace:
    """Enrich waveform metadata with session-level information.

    Args:
        trace: Original trace from .wfm file.
        session_metadata: Session metadata from session.json.
        source_file: Path to .tss file (for source_file metadata).

    Returns:
        Trace with enriched metadata.
    """
    # Create new metadata with session information
    from dataclasses import replace

    metadata = trace.metadata

    # Update source file to point to .tss instead of temp .wfm
    metadata = replace(metadata, source_file=source_file)

    # Add trigger info from session if available
    if "trigger" in session_metadata and metadata.trigger_info is None:
        metadata = replace(metadata, trigger_info=session_metadata["trigger"])

    # Return trace with updated metadata
    if isinstance(trace, WaveformTrace):
        return WaveformTrace(data=trace.data, metadata=metadata)
    if isinstance(trace, DigitalTrace):
        return DigitalTrace(data=trace.data, metadata=metadata, edges=trace.edges)
    # IQTrace
    return IQTrace(
        i_data=trace.i_data,
        q_data=trace.q_data,
        metadata=metadata,
    )


__all__ = [
    "load_all_channels_tss",
    "load_tss",
]
