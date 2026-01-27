"""Tektronix WFM file loader.

This module provides loading of Tektronix oscilloscope .wfm files
using the tm_data_types library when available, with fallback to
basic binary parsing.

Supports both analog and digital waveforms from Tektronix oscilloscopes
including mixed-signal instruments.


Example:
    >>> from oscura.loaders.tektronix import load_tektronix_wfm
    >>> trace = load_tektronix_wfm("TEK00001.wfm")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")

    >>> # Load digital waveform
    >>> digital_trace = load_tektronix_wfm("digital_capture.wfm")
    >>> print(f"Digital trace: {len(digital_trace.data)} samples")
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
from numpy.typing import NDArray

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import DigitalTrace, IQTrace, TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike

# Logger for debug output
logger = logging.getLogger(__name__)

# Try to import tm_data_types for full Tektronix support
try:
    import tm_data_types  # type: ignore[import-untyped]  # Optional third-party library

    TM_DATA_TYPES_AVAILABLE = True
except ImportError:
    TM_DATA_TYPES_AVAILABLE = False

# Type alias for return type
TektronixTrace = Union[WaveformTrace, DigitalTrace, IQTrace]

# Minimum file size for valid WFM files
MIN_WFM_FILE_SIZE = 512


def load_tektronix_wfm(
    path: str | PathLike[str],
    *,
    channel: int = 0,
) -> TektronixTrace:
    """Load a Tektronix oscilloscope WFM file.

    Extracts waveform data and metadata from Tektronix .wfm files.
    Uses the tm_data_types library when available for full support,
    otherwise falls back to basic binary parsing.

    Supports both analog and digital waveforms from mixed-signal
    oscilloscopes (channels 5-8 are typically digital on MSO scopes).

    Args:
        path: Path to the Tektronix .wfm file.
        channel: Channel index for multi-channel files (default: 0).

    Returns:
        WaveformTrace for analog waveforms or DigitalTrace for digital waveforms.

    Raises:
        LoaderError: If the file cannot be loaded.
        FormatError: If the file is not a valid Tektronix WFM file.

    Example:
        >>> trace = load_tektronix_wfm("TEK00001.wfm")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
        >>> print(f"Channel: {trace.metadata.channel_name}")

        >>> # Check trace type
        >>> if isinstance(trace, DigitalTrace):
        ...     print("Digital waveform loaded")
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    # File size validation
    file_size = path.stat().st_size
    if file_size < MIN_WFM_FILE_SIZE:
        raise FormatError(
            f"File too small ({file_size} bytes), may be empty or corrupted",
            file_path=str(path),
            expected=f"At least {MIN_WFM_FILE_SIZE} bytes",
            got=f"{file_size} bytes",
        )

    logger.debug("Loading Tektronix WFM file: %s (%d bytes)", path, file_size)

    if TM_DATA_TYPES_AVAILABLE:
        return _load_with_tm_data_types(path, channel=channel)
    else:
        return _load_basic(path, channel=channel)


def _load_with_tm_data_types(
    path: Path,
    *,
    channel: int = 0,
) -> TektronixTrace:
    """Load Tektronix WFM using tm_data_types library.

    Handles multiple waveform formats:
    - Multi-channel container with analog_waveforms
    - Direct AnalogWaveform with y_axis_values
    - Legacy format with y_data
    - DigitalWaveform with y_axis_byte_values

    Args:
        path: Path to the WFM file.
        channel: Channel index.

    Returns:
        WaveformTrace for analog data or DigitalTrace for digital data.

    Raises:
        FormatError: If the file format is not recognized or invalid.
        LoaderError: If the file cannot be loaded.
    """
    try:
        # Use tm_data_types to read the file
        wfm = tm_data_types.read_file(str(path))

        # Log object information for debugging
        wfm_type = type(wfm).__name__
        available_attrs = [attr for attr in dir(wfm) if not attr.startswith("_")]
        logger.debug("WFM object type: %s", wfm_type)
        logger.debug("WFM attributes: %s", available_attrs[:20])  # First 20 attrs

        # Check for digital waveforms attribute
        if hasattr(wfm, "digital_waveforms"):
            logger.debug("Digital waveforms found: %d", len(wfm.digital_waveforms))

        # Dispatch to appropriate loader based on waveform format
        return _dispatch_waveform_loader(wfm, wfm_type, available_attrs, path, channel)

    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to load Tektronix WFM file",
            file_path=str(path),
            details=str(e),
            fix_hint="Ensure the file is a valid Tektronix WFM format.",
        ) from e


def _dispatch_waveform_loader(
    wfm: Any,
    wfm_type: str,
    available_attrs: list[str],
    path: Path,
    channel: int,
) -> TektronixTrace:
    """Dispatch to appropriate waveform loader based on format.

    Args:
        wfm: Waveform object from tm_data_types.
        wfm_type: Type name of waveform object.
        available_attrs: List of available attributes on waveform object.
        path: Path to WFM file.
        channel: Channel index.

    Returns:
        Loaded trace (WaveformTrace, DigitalTrace, or IQTrace).

    Raises:
        FormatError: If no recognized waveform format found.
    """
    # Path 1: Multi-channel container format (wrapped analog)
    if hasattr(wfm, "analog_waveforms") and len(wfm.analog_waveforms) > channel:
        logger.debug("Loading from analog_waveforms[%d]", channel)
        return _load_analog_waveforms_container(wfm.analog_waveforms[channel], path, channel)

    # Path 2: Direct AnalogWaveform format (tm_data_types 0.3.0+)
    if hasattr(wfm, "y_axis_values") and wfm_type == "AnalogWaveform":
        logger.debug("Loading direct AnalogWaveform with y_axis_values")
        return _load_analog_waveform_direct(wfm, path, channel)

    # Path 3: DigitalWaveform format
    if wfm_type == "DigitalWaveform" or hasattr(wfm, "y_axis_byte_values"):
        logger.debug("Loading DigitalWaveform with y_axis_byte_values")
        return _load_digital_waveform(wfm, path, channel)

    # Path 4: Legacy single channel format with y_data
    if hasattr(wfm, "y_data"):
        logger.debug("Loading legacy format with y_data")
        return _load_legacy_y_data(wfm, path)

    # Path 5: Check for wrapped digital waveforms
    if hasattr(wfm, "digital_waveforms") and len(wfm.digital_waveforms) > channel:
        logger.debug("Loading from digital_waveforms[%d]", channel)
        return _load_digital_waveform(wfm.digital_waveforms[channel], path, channel)

    # Path 6: IQWaveform format (I/Q data)
    if wfm_type == "IQWaveform" or (
        hasattr(wfm, "i_axis_values") and hasattr(wfm, "q_axis_values")
    ):
        logger.debug("Loading IQWaveform with i_axis_values and q_axis_values")
        return _load_iq_waveform(wfm, path)

    # No recognized format - provide detailed error
    raise FormatError(
        f"No waveform data found. Object type: {wfm_type}. "
        f"Available attributes: {', '.join(available_attrs[:15])}",
        file_path=str(path),
        expected="Tektronix analog or digital waveform data",
        fix_hint=(
            "This file may use an unsupported Tektronix format variant. "
            "Check that tm_data_types is up to date: pip install -U tm_data_types"
        ),
    )


def _load_analog_waveforms_container(
    waveform: Any,
    path: Path,
    channel: int,
) -> WaveformTrace:
    """Load analog waveform from multi-channel container format.

    Args:
        waveform: Analog waveform object from container.
        path: Path to WFM file.
        channel: Channel index.

    Returns:
        WaveformTrace with extracted data.
    """
    data = np.array(waveform.y_data, dtype=np.float64)
    sample_rate = 1.0 / waveform.x_increment if waveform.x_increment > 0 else 1e6
    vertical_scale = getattr(waveform, "y_scale", None)
    vertical_offset = getattr(waveform, "y_offset", None)
    channel_name = getattr(waveform, "name", f"CH{channel + 1}")

    # Use original wfm for trigger info (need to get it from parent)
    return _build_waveform_trace(
        data=data,
        sample_rate=sample_rate,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        channel_name=channel_name,
        path=path,
        wfm=waveform,
    )


def _load_analog_waveform_direct(
    wfm: Any,
    path: Path,
    channel: int,
) -> WaveformTrace:
    """Load direct AnalogWaveform format (tm_data_types 0.3.0+).

    Args:
        wfm: AnalogWaveform object.
        path: Path to WFM file.
        channel: Channel index.

    Returns:
        WaveformTrace with extracted data.
    """
    # Extract raw integer values and reconstruct voltage values
    y_raw = np.array(wfm.y_axis_values, dtype=np.float64)
    y_spacing = float(wfm.y_axis_spacing) if wfm.y_axis_spacing else 1.0
    y_offset = float(wfm.y_axis_offset) if wfm.y_axis_offset else 0.0
    data = y_raw * y_spacing + y_offset

    x_spacing = float(wfm.x_axis_spacing) if wfm.x_axis_spacing else 1e-6
    sample_rate = 1.0 / x_spacing if x_spacing > 0 else 1e6
    vertical_offset = y_offset
    channel_name = (
        wfm.source_name if hasattr(wfm, "source_name") and wfm.source_name else f"CH{channel + 1}"
    )

    return _build_waveform_trace(
        data=data,
        sample_rate=sample_rate,
        vertical_scale=None,
        vertical_offset=vertical_offset,
        channel_name=channel_name,
        path=path,
        wfm=wfm,
    )


def _load_legacy_y_data(
    wfm: Any,
    path: Path,
) -> WaveformTrace:
    """Load legacy single channel format with y_data.

    Args:
        wfm: Legacy waveform object with y_data.
        path: Path to WFM file.

    Returns:
        WaveformTrace with extracted data.
    """
    data = np.array(wfm.y_data, dtype=np.float64)
    x_increment = getattr(wfm, "x_increment", 1e-6)
    sample_rate = 1.0 / x_increment if x_increment > 0 else 1e6
    vertical_scale = getattr(wfm, "y_scale", None)
    vertical_offset = getattr(wfm, "y_offset", None)
    channel_name = getattr(wfm, "name", "CH1")

    return _build_waveform_trace(
        data=data,
        sample_rate=sample_rate,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        channel_name=channel_name,
        path=path,
        wfm=wfm,
    )


def _build_waveform_trace(
    data: NDArray[np.float64],
    sample_rate: float,
    vertical_scale: float | None,
    vertical_offset: float | None,
    channel_name: str,
    path: Path,
    wfm: Any,
) -> WaveformTrace:
    """Build a WaveformTrace from extracted data.

    Args:
        data: Waveform sample data.
        sample_rate: Sample rate in Hz.
        vertical_scale: Vertical scale in volts/div.
        vertical_offset: Vertical offset in volts.
        channel_name: Channel name.
        path: Source file path.
        wfm: Original waveform object for trigger info extraction.

    Returns:
        Constructed WaveformTrace.
    """
    # Extract acquisition time if available
    acquisition_time = None
    if hasattr(wfm, "date_time"):
        with contextlib.suppress(ValueError, AttributeError):
            acquisition_time = wfm.date_time

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        acquisition_time=acquisition_time,
        source_file=str(path),
        channel_name=channel_name,
        trigger_info=_extract_trigger_info(wfm),
    )

    return WaveformTrace(data=data, metadata=metadata)


def _load_digital_waveform(
    wfm: Any,
    path: Path,
    channel: int = 0,
) -> DigitalTrace:
    """Load a digital waveform from tm_data_types object.

    Handles DigitalWaveform objects with y_axis_byte_values attribute,
    commonly used for digital/logic analyzer captures on mixed-signal
    oscilloscopes.

    Args:
        wfm: DigitalWaveform object from tm_data_types.
        path: Source file path.
        channel: Channel index.

    Returns:
        DigitalTrace with boolean sample data.

    Raises:
        FormatError: If DigitalWaveform has no recognized data attribute.
    """
    logger.debug("Extracting digital waveform data")

    # Extract digital sample data
    data = _extract_digital_samples(wfm, path)

    # Extract timing information
    sample_rate = _extract_sample_rate(wfm)

    # Extract channel name
    channel_name = _extract_channel_name(wfm, channel)

    # Build metadata
    metadata = TraceMetadata(
        sample_rate=sample_rate,
        source_file=str(path),
        channel_name=channel_name,
    )

    # Extract edge information if available
    edges = _extract_edges(wfm)

    return DigitalTrace(data=data, metadata=metadata, edges=edges)


def _extract_digital_samples(wfm: Any, path: Path) -> NDArray[np.bool_]:
    """Extract digital sample data from waveform object."""
    # Try y_axis_byte_values (most common)
    if hasattr(wfm, "y_axis_byte_values"):
        raw_bytes = wfm.y_axis_byte_values
        byte_array = np.frombuffer(bytes(raw_bytes), dtype=np.uint8)
        data = byte_array.astype(np.bool_)
        logger.debug("Loaded %d digital samples from y_axis_byte_values", len(data))
        return data

    # Try samples attribute
    if hasattr(wfm, "samples"):
        data = np.array(wfm.samples, dtype=np.bool_)
        logger.debug("Loaded %d digital samples from samples", len(data))
        return data

    # Try alternative data attributes
    for attr in ["data", "digital_data", "logic_data"]:
        if hasattr(wfm, attr):
            data = np.array(getattr(wfm, attr), dtype=np.bool_)
            logger.debug("Loaded %d digital samples from %s", len(data), attr)
            return data

    # No recognized attribute found
    raise FormatError(
        "DigitalWaveform has no recognized data attribute",
        file_path=str(path),
        expected="y_axis_byte_values, samples, or data attribute",
    )


def _extract_sample_rate(wfm: Any) -> float:
    """Extract sample rate from waveform timing attributes."""
    x_spacing = 1e-6  # Default 1 microsecond per sample

    if hasattr(wfm, "x_axis_spacing") and wfm.x_axis_spacing:
        x_spacing = float(wfm.x_axis_spacing)
    elif hasattr(wfm, "horizontal_spacing") and wfm.horizontal_spacing:
        x_spacing = float(wfm.horizontal_spacing)

    return 1.0 / x_spacing if x_spacing > 0 else 1e6


def _extract_channel_name(wfm: Any, channel: int) -> str:
    """Extract channel name from waveform object."""
    # Try source_name first
    if hasattr(wfm, "source_name") and wfm.source_name:
        name: str = str(wfm.source_name)
        return name

    # Try name attribute
    if hasattr(wfm, "name") and wfm.name:
        name_str: str = str(wfm.name)
        return name_str

    # Default: digital channels labeled D1, D2, etc.
    return f"D{channel + 1}"


def _extract_edges(wfm: Any) -> list[tuple[float, bool]] | None:
    """Extract edge timing information if available."""
    if not hasattr(wfm, "edges"):
        return None

    try:
        return [(float(ts), bool(is_rising)) for ts, is_rising in wfm.edges]
    except (TypeError, ValueError):
        return None


def _load_iq_waveform(
    wfm: Any,
    path: Path,
) -> IQTrace:
    """Load I/Q waveform data from tm_data_types IQWaveform object.

    Handles IQWaveform objects with i_axis_values and q_axis_values,
    commonly used for RF and software-defined radio captures.

    Args:
        wfm: IQWaveform object from tm_data_types.
        path: Source file path.

    Returns:
        IQTrace with I and Q component data.
    """
    logger.debug("Extracting I/Q waveform data")

    # Extract I/Q data
    i_data = np.array(wfm.i_axis_values, dtype=np.float64)
    q_data = np.array(wfm.q_axis_values, dtype=np.float64)

    logger.debug("Loaded %d I/Q samples", len(i_data))

    # Apply scaling if available
    if hasattr(wfm, "iq_axis_spacing") and wfm.iq_axis_spacing:
        iq_spacing = float(wfm.iq_axis_spacing)
        i_data = i_data * iq_spacing
        q_data = q_data * iq_spacing
    if hasattr(wfm, "iq_axis_offset") and wfm.iq_axis_offset:
        iq_offset = float(wfm.iq_axis_offset)
        i_data = i_data + iq_offset
        q_data = q_data + iq_offset

    # Extract timing information
    x_spacing = 1e-6  # Default 1 microsecond per sample
    if hasattr(wfm, "x_axis_spacing") and wfm.x_axis_spacing:
        x_spacing = float(wfm.x_axis_spacing)

    sample_rate = 1.0 / x_spacing if x_spacing > 0 else 1e6

    # Extract channel name
    channel_name = "IQ1"
    if hasattr(wfm, "source_name") and wfm.source_name:
        channel_name = wfm.source_name

    # Build metadata
    metadata = TraceMetadata(
        sample_rate=sample_rate,
        source_file=str(path),
        channel_name=channel_name,
    )

    return IQTrace(i_data=i_data, q_data=q_data, metadata=metadata)


def _load_basic(
    path: Path,
    *,
    channel: int = 0,
) -> WaveformTrace:
    """Basic Tektronix WFM loader without tm_data_types.

    This is a simplified loader that reads the basic waveform data
    from Tektronix WFM files, including support for WFM#003 format.
    For full feature support, install tm_data_types.

    Args:
        path: Path to the WFM file.
        channel: Channel index (ignored in basic mode).

    Returns:
        WaveformTrace with basic metadata.

    Raises:
        FormatError: If the file format is invalid or cannot be parsed.
        LoaderError: If the file cannot be read.
    """
    try:
        with open(path, "rb") as f:
            # Read full file for format detection
            file_data = f.read()

            if len(file_data) < MIN_WFM_FILE_SIZE:
                raise FormatError(
                    "File too small to be a valid Tektronix WFM",
                    file_path=str(path),
                    expected=f"At least {MIN_WFM_FILE_SIZE} bytes",
                    got=f"{len(file_data)} bytes",
                )

            # Detect WFM format version
            if file_data[2:10] == b":WFM#003":
                return _parse_wfm003(file_data, path, channel)
            else:
                # Legacy WFM format (older versions)
                return _parse_wfm_legacy(file_data, path, channel)

    except OSError as e:
        raise LoaderError(
            "Failed to read Tektronix WFM file",
            file_path=str(path),
            details=str(e),
        ) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to parse Tektronix WFM file",
            file_path=str(path),
            details=str(e),
            fix_hint="Install tm_data_types for full Tektronix support: pip install tm_data_types",
        ) from e


def _parse_wfm003(
    file_data: bytes,
    path: Path,
    channel: int = 0,
) -> WaveformTrace:
    """Parse Tektronix WFM#003 format files.

    WFM#003 is a binary format used by Tektronix oscilloscopes.
    The file structure consists of:
    - Static file header (first ~80 bytes)
    - Main waveform header (~838 bytes total)
    - Waveform data (int16 samples)
    - Optional metadata footer (tekmeta!)

    Args:
        file_data: Raw file bytes.
        path: Path to file (for error messages).
        channel: Channel index.

    Returns:
        WaveformTrace with extracted data and metadata.

    Raises:
        FormatError: If the file signature is invalid or no waveform data found.
    """

    _validate_wfm003_signature(file_data, path)
    header_size = 838
    waveform_bytes = _extract_waveform_data(file_data, header_size, path)
    data = np.frombuffer(waveform_bytes, dtype=np.int16).astype(np.float64)

    # Extract metadata from header
    sample_rate = _extract_sample_interval(file_data, header_size)
    vertical_scale, vertical_offset = _extract_vertical_params(file_data, header_size)
    channel_name = f"CH{channel + 1}"

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        source_file=str(path),
        channel_name=channel_name,
    )

    return WaveformTrace(data=data, metadata=metadata)


def _validate_wfm003_signature(file_data: bytes, path: Path) -> None:
    """Validate WFM#003 file signature."""
    signature = file_data[2:10]
    if signature != b":WFM#003":
        raise FormatError(
            "Invalid WFM#003 signature",
            file_path=str(path),
            expected=":WFM#003",
            got=signature.decode("latin-1", errors="replace"),
        )


def _extract_waveform_data(file_data: bytes, header_size: int, path: Path) -> bytes:
    """Extract waveform data region from file."""
    footer_start = len(file_data)
    if b"tekmeta!" in file_data:
        footer_start = file_data.find(b"tekmeta!")

    waveform_bytes = file_data[header_size:footer_start]

    if len(waveform_bytes) < 2:
        raise FormatError(
            "No waveform data found in WFM#003 file",
            file_path=str(path),
        )

    # Ensure even number of bytes for int16
    if len(waveform_bytes) % 2 != 0:
        waveform_bytes = waveform_bytes[:-1]

    return waveform_bytes


def _extract_sample_interval(file_data: bytes, header_size: int) -> float:
    """Extract sample rate from header doubles."""
    import struct

    # Default 1 MSa/s
    sample_rate = 1e6

    try:
        # Search for reasonable sample interval values (doubles in header)
        for offset in range(16, min(header_size - 8, 200), 8):
            val = struct.unpack("<d", file_data[offset : offset + 8])[0]
            # Sample intervals are typically 1e-12 to 1e-3 (1ps to 1ms)
            if 1e-12 < val < 1e-3:
                sample_rate = 1.0 / val
                break
    except (struct.error, ZeroDivisionError):
        pass

    return sample_rate


def _extract_vertical_params(
    file_data: bytes, header_size: int
) -> tuple[float | None, float | None]:
    """Extract vertical scale and offset from header."""
    import struct

    vertical_scale = None
    vertical_offset = None

    try:
        # Vertical scale is often in a specific range
        for offset in range(16, min(header_size - 8, 400), 8):
            val = struct.unpack("<d", file_data[offset : offset + 8])[0]
            # Vertical scale is typically 1e-9 to 1e3 (nV to kV range)
            if 1e-9 < abs(val) < 1e3 and vertical_scale is None:
                vertical_scale = abs(val)
                # Offset might be nearby
                next_val = struct.unpack("<d", file_data[offset + 8 : offset + 16])[0]
                if abs(next_val) < 1e6:
                    vertical_offset = next_val
                break
    except struct.error:
        pass

    return vertical_scale, vertical_offset


def _parse_wfm_legacy(
    file_data: bytes,
    path: Path,
    channel: int = 0,
) -> WaveformTrace:
    """Parse legacy Tektronix WFM formats (pre-WFM#003).

    Args:
        file_data: Raw file bytes.
        path: Path to file (for error messages).
        channel: Channel index.

    Returns:
        WaveformTrace with extracted data and metadata.

    Raises:
        FormatError: If no waveform data is found in the file.
    """
    import struct

    # Default values
    sample_rate = 1e6  # Default 1 MSa/s
    vertical_scale = None
    vertical_offset = None

    # Try to find sample interval in header (little-endian double at offset ~40)
    try:
        # Sample interval is typically at offset 40 in many WFM versions
        sample_interval_bytes = file_data[40:48]
        if len(sample_interval_bytes) == 8:
            sample_interval = struct.unpack("<d", sample_interval_bytes)[0]
            if 0 < sample_interval < 1:  # Sanity check
                sample_rate = 1.0 / sample_interval
    except (struct.error, ZeroDivisionError):
        pass

    # Read waveform data - assume rest of file is float32 samples after 512-byte header
    header_size = 512
    data_size = len(file_data) - header_size

    if data_size <= 0:
        raise FormatError(
            "No waveform data in file",
            file_path=str(path),
        )

    raw_data = file_data[header_size:]

    # Try to interpret as float32 or int16
    try:
        # Try float32 first (common in Tektronix files)
        data = np.frombuffer(raw_data, dtype=np.float32).astype(np.float64)
    except ValueError:
        # Fall back to int16
        data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        data = data / 32768.0  # Normalize to -1 to 1

    # Build metadata
    metadata = TraceMetadata(
        sample_rate=sample_rate,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        source_file=str(path),
        channel_name=f"CH{channel + 1}",
    )

    return WaveformTrace(data=data, metadata=metadata)


def _extract_trigger_info(wfm: Any) -> dict[str, Any] | None:
    """Extract trigger information from Tektronix waveform object.

    Args:
        wfm: Tektronix waveform object from tm_data_types.

    Returns:
        Dictionary of trigger settings or None.
    """
    trigger_info: dict[str, Any] = {}

    if hasattr(wfm, "trigger_level"):
        trigger_info["level"] = wfm.trigger_level
    if hasattr(wfm, "trigger_slope"):
        trigger_info["slope"] = wfm.trigger_slope
    if hasattr(wfm, "trigger_position"):
        trigger_info["position"] = wfm.trigger_position

    return trigger_info if trigger_info else None


__all__ = ["TektronixTrace", "load_tektronix_wfm"]
