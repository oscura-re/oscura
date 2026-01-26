"""Rigol WFM file loader.

This module provides loading of Rigol oscilloscope .wfm files
using the RigolWFM library when available.


Example:
    >>> from oscura.loaders.rigol import load_rigol_wfm
    >>> trace = load_rigol_wfm("DS1054Z_001.wfm")
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

# Try to import RigolWFM for full Rigol support
try:
    import RigolWFM.wfm as rigol_wfm  # type: ignore[import-untyped]  # Optional third-party library

    RIGOL_WFM_AVAILABLE = True
except ImportError:
    RIGOL_WFM_AVAILABLE = False


def load_rigol_wfm(
    path: str | PathLike[str],
    *,
    channel: int = 0,
) -> WaveformTrace:
    """Load a Rigol oscilloscope WFM file.

    Extracts waveform data and metadata from Rigol .wfm files.
    Uses the RigolWFM library when available for full support.

    Args:
        path: Path to the Rigol .wfm file.
        channel: Channel index for multi-channel files (default: 0).

    Returns:
        WaveformTrace containing the waveform data and metadata.

    Raises:
        LoaderError: If the file cannot be loaded or does not exist.

    Example:
        >>> trace = load_rigol_wfm("DS1054Z_001.wfm")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
        >>> print(f"Vertical scale: {trace.metadata.vertical_scale} V/div")
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    # Try RigolWFM first for full metadata, fall back to basic loader
    if RIGOL_WFM_AVAILABLE:
        try:
            return _load_with_rigolwfm(path, channel=channel)
        except Exception:
            # RigolWFM failed (likely synthetic/malformed file)
            # Force garbage collection to close any leaked file handles
            import gc

            gc.collect()
            # Try basic loader as fallback
            return _load_basic(path, channel=channel)
    else:
        return _load_basic(path, channel=channel)


def _load_with_rigolwfm(
    path: Path,
    *,
    channel: int = 0,
) -> WaveformTrace:
    """Load Rigol WFM using RigolWFM library.

    Args:
        path: Path to the WFM file.
        channel: Channel index.

    Returns:
        WaveformTrace with full metadata.

    Raises:
        FormatError: If no waveform data is found in the file.
        LoaderError: If the file cannot be loaded.
    """
    try:
        # Auto-detect model and load waveform
        wfm = _load_rigol_with_model_detection(path)

        # Extract channel data
        data, sample_rate, vertical_scale, vertical_offset, channel_name = (
            _extract_rigol_channel_data(wfm, channel, str(path))
        )

        # Build metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=vertical_offset,
            source_file=str(path),
            channel_name=channel_name,
            trigger_info=_extract_trigger_info(wfm),
        )

        return WaveformTrace(data=data, metadata=metadata)

    except Exception as e:
        if isinstance(e, FormatError):
            raise
        raise LoaderError(
            "Failed to load Rigol WFM file with RigolWFM library",
            file_path=str(path),
            details=str(e),
            fix_hint="File may be malformed or incompatible with RigolWFM library.",
        ) from e


def _detect_rigol_model_from_filename(path: Path) -> str | None:
    """Auto-detect Rigol oscilloscope model from filename.

    Args:
        path: Path to WFM file.

    Returns:
        Model string ("Z" or "E") or None if detection fails.
    """
    filename_upper = path.name.upper()

    # Check for Rigol model indicators in filename
    if "DS1" not in filename_upper and "MSO1" not in filename_upper and "DHO" not in filename_upper:
        return None

    # Z-series (DS1000Z, MSO1000Z, DHO series)
    if "Z" in filename_upper or "MSO" in filename_upper or "DHO" in filename_upper:
        return "Z"

    # E-series (DS1000E)
    if "E" in filename_upper:
        return "E"

    return None


def _load_rigol_with_model_detection(path: Path) -> Any:
    """Load Rigol WFM file with automatic model detection.

    Args:
        path: Path to WFM file.

    Returns:
        Loaded waveform object from RigolWFM.

    Raises:
        RuntimeError: If loading fails with all models.
    """
    model = _detect_rigol_model_from_filename(path)

    # Try detected model first, then fallback to both models
    models_to_try = [model] if model else ["Z", "E"]

    last_error = None
    for try_model in models_to_try:
        try:
            return rigol_wfm.Wfm.from_file(str(path), model=try_model)
        except Exception as e:
            last_error = e
            continue

    # None of the models worked
    raise last_error if last_error else RuntimeError("Failed to load WFM file")


def _extract_rigol_channel_data(
    wfm: Any,
    channel: int,
    file_path: str,
) -> tuple[NDArray[np.float64], float, float | None, float | None, str]:
    """Extract channel data from Rigol waveform object.

    Args:
        wfm: Rigol waveform object from RigolWFM.
        channel: Channel index.
        file_path: File path for error messages.

    Returns:
        Tuple of (data, sample_rate, vertical_scale, vertical_offset, channel_name).

    Raises:
        FormatError: If no waveform data found.
    """
    # Multi-channel format
    if hasattr(wfm, "channels") and len(wfm.channels) > channel:
        ch = wfm.channels[channel]
        data = np.array(ch.volts, dtype=np.float64)
        sample_rate = wfm.sample_rate if hasattr(wfm, "sample_rate") else 1e6
        vertical_scale = ch.volts_per_div if hasattr(ch, "volts_per_div") else None
        vertical_offset = ch.volt_offset if hasattr(ch, "volt_offset") else None
        channel_name = f"CH{channel + 1}"
        return data, sample_rate, vertical_scale, vertical_offset, channel_name

    # Single channel format
    if hasattr(wfm, "volts"):
        data = np.array(wfm.volts, dtype=np.float64)
        sample_rate = wfm.sample_rate if hasattr(wfm, "sample_rate") else 1e6
        vertical_scale = wfm.volts_per_div if hasattr(wfm, "volts_per_div") else None
        vertical_offset = wfm.volt_offset if hasattr(wfm, "volt_offset") else None
        channel_name = "CH1"
        return data, sample_rate, vertical_scale, vertical_offset, channel_name

    # No recognized format
    raise FormatError(
        "No waveform data found in Rigol file",
        file_path=file_path,
        expected="Rigol channel data",
    )


def _load_basic(
    path: Path,
    *,
    channel: int = 0,
) -> WaveformTrace:
    """Basic Rigol WFM loader without RigolWFM library.

    This is a simplified loader that reads basic waveform data
    from Rigol WFM files. For full feature support, install RigolWFM.

    Args:
        path: Path to the WFM file.
        channel: Channel index (ignored in basic mode).

    Returns:
        WaveformTrace with basic metadata.

    Raises:
        FormatError: If the file is too small or has no waveform data.
        LoaderError: If the file cannot be read or parsed.
    """
    try:
        with open(path, "rb", buffering=65536) as f:
            # Read header
            header = f.read(256)

            # Basic validation
            if len(header) < 256:
                raise FormatError(
                    "File too small to be a valid Rigol WFM",
                    file_path=str(path),
                    expected="At least 256 bytes header",
                    got=f"{len(header)} bytes",
                )

            # Default values
            sample_rate = 1e6  # Default 1 MSa/s
            vertical_scale = None
            vertical_offset = None

            # Read waveform data
            f.seek(0, 2)
            file_size = f.tell()
            data_size = file_size - 256

            if data_size <= 0:
                raise FormatError(
                    "No waveform data in file",
                    file_path=str(path),
                )

            f.seek(256)
            raw_data = f.read(data_size)

            # Rigol typically uses int16 or int8 for samples
            try:
                # Try int16 first (common in Rigol files)
                data = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
                data = data / 32768.0  # Normalize to -1 to 1
            except ValueError:
                # Fall back to int8
                data = np.frombuffer(raw_data, dtype=np.int8).astype(np.float64)
                data = data / 128.0  # Normalize to -1 to 1

        # Build metadata
        metadata = TraceMetadata(
            sample_rate=sample_rate,
            vertical_scale=vertical_scale,
            vertical_offset=vertical_offset,
            source_file=str(path),
            channel_name=f"CH{channel + 1}",
        )

        return WaveformTrace(data=data, metadata=metadata)

    except OSError as e:
        raise LoaderError(
            "Failed to read Rigol WFM file",
            file_path=str(path),
            details=str(e),
        ) from e
    except Exception as e:
        if isinstance(e, LoaderError | FormatError):
            raise
        raise LoaderError(
            "Failed to parse Rigol WFM file",
            file_path=str(path),
            details=str(e),
            fix_hint="Install RigolWFM for full Rigol support: pip install RigolWFM",
        ) from e


def _extract_trigger_info(wfm: Any) -> dict[str, Any] | None:
    """Extract trigger information from Rigol waveform object.

    Args:
        wfm: Rigol waveform object from RigolWFM.

    Returns:
        Dictionary of trigger settings or None.
    """
    trigger_info: dict[str, Any] = {}

    if hasattr(wfm, "trigger_level"):
        trigger_info["level"] = wfm.trigger_level
    if hasattr(wfm, "trigger_mode"):
        trigger_info["mode"] = wfm.trigger_mode
    if hasattr(wfm, "trigger_source"):
        trigger_info["source"] = wfm.trigger_source

    return trigger_info if trigger_info else None


__all__ = ["load_rigol_wfm"]
