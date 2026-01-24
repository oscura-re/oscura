"""Oscura data loaders for various file formats.

This module provides a unified load() function that auto-detects file formats
and delegates to the appropriate loader.


Example:
    >>> import oscura as osc
    >>> trace = osc.load("capture.wfm")
    >>> print(f"Loaded {len(trace.data)} samples")

    >>> # Load all channels from multi-channel file
    >>> channels = osc.load_all_channels("multi_channel.wfm")
    >>> for name, trace in channels.items():
    ...     print(f"{name}: {len(trace.data)} samples")
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from oscura.core.exceptions import LoaderError, UnsupportedFormatError
from oscura.core.types import DigitalTrace, IQTrace, WaveformTrace

# Loader registry for cleaner dispatch
_LOADER_REGISTRY: dict[str, tuple[str, str]] = {
    "tektronix": ("oscura.loaders.tektronix", "load_tektronix_wfm"),
    "tek": ("oscura.loaders.tektronix", "load_tektronix_wfm"),
    "rigol": ("oscura.loaders.rigol", "load_rigol_wfm"),
    "numpy": ("oscura.loaders.numpy_loader", "load_npz"),
    "csv": ("oscura.loaders.csv_loader", "load_csv"),
    "hdf5": ("oscura.loaders.hdf5_loader", "load_hdf5"),
    "sigrok": ("oscura.loaders.sigrok", "load_sigrok"),
    "vcd": ("oscura.loaders.vcd", "load_vcd"),
    "pcap": ("oscura.loaders.pcap", "load_pcap"),
    "wav": ("oscura.loaders.wav", "load_wav"),
    "tdms": ("oscura.loaders.tdms", "load_tdms"),
    "touchstone": ("oscura.loaders.touchstone", "load_touchstone"),
    "chipwhisperer": ("oscura.loaders.chipwhisperer", "load_chipwhisperer"),
}


def _dispatch_loader(
    loader_name: str, path: Path, **kwargs: Any
) -> WaveformTrace | DigitalTrace | IQTrace:
    """Dispatch to registered loader.

    Args:
        loader_name: Name of loader to use.
        path: Path to file.
        **kwargs: Additional arguments for loader.

    Returns:
        Loaded data.

    Raises:
        UnsupportedFormatError: If loader not registered.
    """
    if loader_name not in _LOADER_REGISTRY:
        raise UnsupportedFormatError(
            loader_name,
            list(_LOADER_REGISTRY.keys()),
            file_path=str(path),
        )

    module_path, func_name = _LOADER_REGISTRY[loader_name]

    # Dynamically import the module
    import importlib
    import inspect

    module = importlib.import_module(module_path)
    loader_func = getattr(module, func_name)

    # Filter kwargs to only include parameters the function accepts
    sig = inspect.signature(loader_func)
    valid_kwargs = {}
    for key, value in kwargs.items():
        if key in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ):
            valid_kwargs[key] = value

    # Call loader with appropriate arguments
    result = loader_func(path, **valid_kwargs)
    return cast("WaveformTrace | DigitalTrace | IQTrace", result)


# Import alias modules for DSL compatibility
from oscura.loaders import (
    binary,
    csv,
    hdf5,
)

# Import configurable binary loading functionality
from oscura.loaders.configurable import (
    BitfieldDef,
    BitfieldExtractor,
    ConfigurablePacketLoader,
    DeviceConfig,
    DeviceInfo,
    DeviceMapper,
    HeaderFieldDef,
    PacketFormatConfig,
    ParsedPacket,
    SampleFormatDef,
    detect_source_type,
    extract_channels,
    load_binary_packets,
    load_packets_streaming,
)
from oscura.loaders.lazy import LazyWaveformTrace, load_trace_lazy
from oscura.loaders.preprocessing import (
    IdleRegion,
    IdleStatistics,
    IdleStats,
    detect_idle_regions,
    get_idle_statistics,
    trim_idle,
)
from oscura.loaders.validation import (
    PacketValidator,
    SequenceGap,
    SequenceValidation,
    ValidationResult,
    ValidationStats,
)

if TYPE_CHECKING:
    from os import PathLike

    from oscura.core.types import Trace

# Logger for debug output
logger = logging.getLogger(__name__)

# Supported format extensions mapped to loader names
SUPPORTED_FORMATS: dict[str, str] = {
    ".wfm": "auto_wfm",  # Auto-detect Tektronix vs Rigol
    ".npz": "numpy",
    ".csv": "csv",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".sr": "sigrok",
    ".pcap": "pcap",
    ".pcapng": "pcap",
    ".wav": "wav",
    ".vcd": "vcd",
    ".tdms": "tdms",
    # Touchstone S-parameter formats
    ".s1p": "touchstone",
    ".s2p": "touchstone",
    ".s3p": "touchstone",
    ".s4p": "touchstone",
    ".s5p": "touchstone",
    ".s6p": "touchstone",
    ".s7p": "touchstone",
    ".s8p": "touchstone",
}

# File size warning threshold for lazy loading suggestion (100 MB)
LARGE_FILE_WARNING_THRESHOLD = 100 * 1024 * 1024


def load(
    path: str | PathLike[str],
    *,
    format: str | None = None,
    channel: str | int | None = None,
    lazy: bool = False,
    **kwargs: Any,
) -> Trace:
    """Load trace data from file with automatic format detection.

    This is the primary entry point for loading oscilloscope and logic
    analyzer data. The file format is auto-detected from the extension
    unless explicitly specified.

    Supports both analog waveforms (WaveformTrace) and digital waveforms
    (DigitalTrace) from mixed-signal oscilloscopes.

    Args:
        path: Path to the file to load.
        format: Optional format override (e.g., "tektronix", "rigol", "csv").
            If not specified, format is auto-detected from file extension.
        channel: Optional channel name or index for multi-channel files.
        lazy: If True, use lazy loading for huge files (see load_lazy).
        **kwargs: Additional arguments passed to the specific loader.

    Returns:
        WaveformTrace or DigitalTrace depending on the file content.

    Raises:
        UnsupportedFormatError: If the file format is not recognized.
        FileNotFoundError: If the file does not exist.

    Example:
        >>> import oscura as osc
        >>> trace = osc.load("oscilloscope_capture.wfm")
        >>> print(f"Loaded {len(trace.data)} samples at {trace.metadata.sample_rate} Hz")

        >>> # Force specific loader
        >>> trace = osc.load("data.bin", format="tektronix")

        >>> # Check if digital trace
        >>> if isinstance(trace, DigitalTrace):
        ...     print("Loaded digital waveform")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Check file size and warn for large files
    file_size = path.stat().st_size
    if file_size > LARGE_FILE_WARNING_THRESHOLD and not lazy:
        warnings.warn(
            f"File is large ({file_size / 1024 / 1024:.1f} MB). "
            "Consider using lazy=True for better memory efficiency.",
            stacklevel=2,
        )

    # Handle lazy loading request
    if lazy:
        return load_lazy(path, **kwargs)  # type: ignore[return-value]

    # Determine format
    if format is not None:
        loader_name = format.lower()
    else:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                ext,
                list(SUPPORTED_FORMATS.keys()),
                file_path=str(path),
            )
        loader_name = SUPPORTED_FORMATS[ext]

    # Dispatch to appropriate loader
    if loader_name == "auto_wfm":
        return _load_wfm_auto(path, channel=channel, **kwargs)
    else:
        # Use registry-based dispatch for all other loaders
        return _dispatch_loader(loader_name, path, channel=channel, **kwargs)


def _load_wfm_auto(
    path: Path,
    *,
    channel: str | int | None = None,
    **kwargs: Any,
) -> Trace:
    """Auto-detect WFM format (Tektronix vs Rigol) and load.

    Distinguishes between Tektronix and Rigol WFM formats by examining
    the file's magic bytes.

    Args:
        path: Path to the .wfm file.
        channel: Optional channel for multi-channel files.
        **kwargs: Additional arguments for the loader.

    Returns:
        WaveformTrace or DigitalTrace from the detected loader.

    Raises:
        LoaderError: If the WFM format cannot be determined.
    """
    # Read first bytes to detect format
    try:
        with open(path, "rb") as f:
            magic = f.read(32)
    except OSError as e:
        raise LoaderError(
            "Failed to read file for format detection",
            file_path=str(path),
            details=str(e),
        ) from e

    # Tektronix WFM files typically start with specific patterns
    # Rigol files have different magic bytes
    # This is a simplified detection - real implementation would be more robust

    # Check for Rigol signature (often starts with certain patterns)
    if magic[:4] in (b"\x00\x00\x01\x00", b"RIGOL"):
        from oscura.loaders.rigol import load_rigol_wfm

        return load_rigol_wfm(path, **kwargs)

    # Default to Tektronix
    from oscura.loaders.tektronix import load_tektronix_wfm

    return load_tektronix_wfm(path, **kwargs)


def load_all_channels(
    path: str | PathLike[str],
    *,
    format: str | None = None,
) -> dict[str, WaveformTrace | DigitalTrace]:
    """Load all channels from a multi-channel waveform file.

    Reads the file once and extracts all available channels (both analog
    and digital). This is more efficient than loading each channel
    separately when you need multiple channels.

    Args:
        path: Path to the multi-channel waveform file.
        format: Optional format override (e.g., "tektronix", "rigol").

    Returns:
        Dictionary mapping channel names to traces.
        Analog channels are named "ch1", "ch2", etc.
        Digital channels are named "d1", "d2", etc.

    Raises:
        UnsupportedFormatError: If the file format is not recognized.
        FileNotFoundError: If the file does not exist.

    Example:
        >>> import oscura as osc
        >>> channels = osc.load_all_channels("multi_channel.wfm")
        >>> for name, trace in channels.items():
        ...     print(f"{name}: {len(trace.data)} samples")
        ch1: 10000 samples
        ch2: 10000 samples
        d1: 10000 samples

        >>> # Access specific channel
        >>> analog_ch1 = channels["ch1"]
        >>> digital_d1 = channels["d1"]
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Determine format
    if format is not None:
        loader_name = format.lower()
    else:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                ext,
                list(SUPPORTED_FORMATS.keys()),
                file_path=str(path),
            )
        loader_name = SUPPORTED_FORMATS[ext]

    # Currently only supports Tektronix WFM for multi-channel loading
    if loader_name in ("auto_wfm", "tektronix", "tek"):
        return _load_all_channels_tektronix(path)
    else:
        # For other formats, try loading as single channel
        trace = load(path, format=format)
        channel_name = getattr(trace.metadata, "channel_name", None) or "ch1"
        return {channel_name: trace}  # type: ignore[dict-item]


def _load_all_channels_tektronix(
    path: Path,
) -> dict[str, WaveformTrace | DigitalTrace]:
    """Load all channels from a Tektronix WFM file.

    Args:
        path: Path to the Tektronix .wfm file.

    Returns:
        Dictionary mapping channel names to traces.

    Raises:
        LoaderError: If the file cannot be read or parsed.
    """
    try:
        import tm_data_types  # type: ignore[import-not-found, import-untyped]
    except ImportError:
        # Fall back to single channel loading
        trace = load(path, format="tektronix")
        channel_name = getattr(trace.metadata, "channel_name", None) or "ch1"
        return {channel_name: trace}  # type: ignore[dict-item]

    try:
        wfm = tm_data_types.read_file(str(path))
    except Exception as e:
        raise LoaderError(
            "Failed to read Tektronix WFM file",
            file_path=str(path),
            details=str(e),
        ) from e

    channels: dict[str, WaveformTrace | DigitalTrace] = {}

    # Extract analog waveforms
    if hasattr(wfm, "analog_waveforms") and wfm.analog_waveforms:
        import numpy as np

        from oscura.loaders.tektronix import _build_waveform_trace

        for i, awfm in enumerate(wfm.analog_waveforms):
            try:
                data = np.array(awfm.y_data, dtype=np.float64)
                x_increment = getattr(awfm, "x_increment", 1e-6)
                sample_rate = 1.0 / x_increment if x_increment > 0 else 1e6
                vertical_scale = getattr(awfm, "y_scale", None)
                vertical_offset = getattr(awfm, "y_offset", None)
                channel_name = getattr(awfm, "name", f"CH{i + 1}")

                trace = _build_waveform_trace(
                    data=data,
                    sample_rate=sample_rate,
                    vertical_scale=vertical_scale,
                    vertical_offset=vertical_offset,
                    channel_name=channel_name,
                    path=path,
                    wfm=awfm,
                )
                channels[f"ch{i + 1}"] = trace
            except Exception as e:
                logger.warning("Failed to extract analog channel %d: %s", i + 1, e)

    # Extract digital waveforms
    if hasattr(wfm, "digital_waveforms") and wfm.digital_waveforms:
        from oscura.loaders.tektronix import _load_digital_waveform

        for i, dwfm in enumerate(wfm.digital_waveforms):
            try:
                trace = _load_digital_waveform(dwfm, path, i)
                channels[f"d{i + 1}"] = trace
            except Exception as e:
                logger.warning("Failed to extract digital channel %d: %s", i + 1, e)

    # Handle direct waveform formats (single file = single channel)
    if not channels:
        wfm_type = type(wfm).__name__

        if wfm_type == "DigitalWaveform" or hasattr(wfm, "y_axis_byte_values"):
            from oscura.loaders.tektronix import _load_digital_waveform

            trace = _load_digital_waveform(wfm, path, 0)
            channel_name = trace.metadata.channel_name or "d1"
            channels[channel_name.lower()] = trace

        elif hasattr(wfm, "y_axis_values") or hasattr(wfm, "y_data"):
            # Direct analog waveform
            trace = load(path, format="tektronix")
            channel_name = trace.metadata.channel_name or "ch1"
            channels[channel_name.lower()] = trace  # type: ignore[assignment]

    if not channels:
        raise LoaderError(
            "No channels found in file",
            file_path=str(path),
            fix_hint="File may be empty or use an unsupported format variant.",
        )

    return channels


def get_supported_formats() -> list[str]:
    """Get list of supported file formats.

    Returns:
        List of supported file extensions.

    Example:
        >>> from oscura.loaders import get_supported_formats
        >>> print(get_supported_formats())
        ['.wfm', '.npz', '.csv', '.h5', ...]
    """
    return list(SUPPORTED_FORMATS.keys())


def load_lazy(path: str | PathLike[str], **kwargs: Any) -> LazyWaveformTrace | WaveformTrace:
    """Load trace with lazy loading for huge files.

    Convenience wrapper for lazy loading. See load_trace_lazy for details.

    Args:
        path: Path to the file.
        **kwargs: Additional arguments (sample_rate, lazy=True, etc.).

    Returns:
        LazyWaveformTrace or WaveformTrace.

    Example:
        >>> trace = osc.loaders.load_lazy("huge_trace.npy", sample_rate=1e9)
        >>> print(f"Length: {trace.length}")  # Metadata available immediately

    References:
        API-017: Lazy Loading for Huge Files
    """
    from oscura.loaders.lazy import load_trace_lazy

    return load_trace_lazy(path, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "LARGE_FILE_WARNING_THRESHOLD",
    "SUPPORTED_FORMATS",
    # Configurable binary loading
    "BitfieldDef",
    "BitfieldExtractor",
    "ConfigurablePacketLoader",
    "DeviceConfig",
    "DeviceInfo",
    "DeviceMapper",
    "DigitalTrace",
    "HeaderFieldDef",
    "IdleRegion",
    "IdleStatistics",
    "IdleStats",
    "LazyWaveformTrace",
    "PacketFormatConfig",
    "PacketValidator",
    "ParsedPacket",
    "SampleFormatDef",
    "SequenceGap",
    "SequenceValidation",
    "ValidationResult",
    "ValidationStats",
    "WaveformTrace",
    "binary",
    "csv",
    "detect_idle_regions",
    "detect_source_type",
    "extract_channels",
    "get_idle_statistics",
    "get_supported_formats",
    "hdf5",
    "load",
    "load_all_channels",
    "load_binary_packets",
    "load_lazy",
    "load_packets_streaming",
    "load_trace_lazy",
    "trim_idle",
]
