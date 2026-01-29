"""WAV audio file loader.

This module provides loading of WAV audio files using scipy.io.wavfile.
WAV files are useful for audio signal analysis and can contain
oscilloscope data recorded as audio.


Example:
    >>> from oscura.loaders.wav import load_wav
    >>> trace = load_wav("recording.wav")
    >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike


def _extract_multichannel_data(
    data: NDArray[np.floating[Any]],
    channel: int | str | None,
    file_path: str,
) -> tuple[NDArray[np.floating[Any]], str]:
    """Extract audio data and channel name from multichannel WAV data.

    Args:
        data: Multi-channel audio data array (samples x channels).
        channel: Channel selector (int, str, or None).
        file_path: Path to file (for error messages).

    Returns:
        Tuple of (audio_data, channel_name).

    Raises:
        LoaderError: If channel selection is invalid.
    """
    n_channels = data.shape[1]
    channel_names = ["left", "right"] if n_channels == 2 else [f"ch{i}" for i in range(n_channels)]

    # Default to first channel
    if channel is None:
        return data[:, 0], channel_names[0]

    # Handle integer channel index
    if isinstance(channel, int):
        if channel < 0 or channel >= n_channels:
            raise LoaderError(
                f"Channel index {channel} out of range",
                file_path=file_path,
                details=f"Available channels: 0-{n_channels - 1}",
            )
        channel_name = channel_names[channel] if channel < len(channel_names) else f"ch{channel}"
        return data[:, channel], channel_name

    # Handle string channel selector
    if isinstance(channel, str):
        return _extract_multichannel_by_name(data, channel, n_channels, file_path)

    # Unreachable code (all cases covered above)
    raise AssertionError("Unexpected channel type")


def _extract_multichannel_by_name(
    data: NDArray[np.floating[Any]],
    channel: str,
    n_channels: int,
    file_path: str,
) -> tuple[NDArray[np.floating[Any]], str]:
    """Extract audio data by channel name string.

    Args:
        data: Multi-channel audio data array.
        channel: Channel name string.
        n_channels: Number of available channels.
        file_path: Path to file (for error messages).

    Returns:
        Tuple of (audio_data, channel_name).

    Raises:
        LoaderError: If channel name is invalid.
    """
    channel_lower = channel.lower()

    # Left channel
    if channel_lower in ("left", "l", "0"):
        return data[:, 0], "left"

    # Right channel
    if channel_lower in ("right", "r", "1") and n_channels >= 2:
        return data[:, 1], "right"

    # Mono mix (average of all channels)
    if channel_lower in ("mono", "mix", "avg"):
        return np.mean(data, axis=1).astype(np.float64), "mono"

    # Invalid channel name
    raise LoaderError(
        f"Invalid channel specifier: '{channel}'",
        file_path=file_path,
        details="Use 'left', 'right', 'mono', or channel index",
    )


def _extract_mono_data(
    data: NDArray[np.floating[Any]],
    channel: int | str | None,
    file_path: str,
) -> tuple[NDArray[np.floating[Any]], str]:
    """Extract audio data from mono WAV data.

    Args:
        data: Mono audio data array.
        channel: Channel selector (int, str, or None).
        file_path: Path to file (for error messages).

    Returns:
        Tuple of (audio_data, channel_name).

    Raises:
        LoaderError: If non-zero channel index requested for mono file.
    """
    if channel is not None and isinstance(channel, int) and channel != 0:
        raise LoaderError(
            f"Channel index {channel} out of range",
            file_path=file_path,
            details="File is mono (only channel 0 available)",
        )
    return data, "mono"


def _normalize_audio_data(
    audio_data: NDArray[np.float64],
    original_dtype: np.dtype[Any],
    normalize: bool,
) -> NDArray[np.float64]:
    """Normalize audio data based on original dtype.

    Args:
        audio_data: Audio data as float64.
        original_dtype: Original data type from WAV file.
        normalize: Whether to normalize to [-1, 1] range.

    Returns:
        Normalized audio data.
    """
    if not normalize:
        return audio_data

    # Normalize based on original dtype
    if original_dtype == np.int16:
        return audio_data / 32768.0
    elif original_dtype == np.int32:
        return audio_data / 2147483648.0
    elif original_dtype == np.uint8:
        return (audio_data - 128.0) / 128.0
    elif original_dtype in (np.float32, np.float64):
        # Already in float format, clip to [-1, 1] range
        max_val = float(np.max(np.abs(audio_data)))
        if max_val > 1.0:
            return audio_data / max_val
        return audio_data
    else:
        # Unknown dtype, return as-is
        return audio_data


def load_wav(
    path: str | PathLike[str],
    *,
    channel: int | str | None = None,
    normalize: bool = True,
) -> WaveformTrace:
    """Load a WAV audio file.

    Extracts audio samples and sample rate from WAV files. Supports
    mono and stereo files, with automatic normalization to [-1, 1] range.

    Args:
        path: Path to the WAV file.
        channel: Channel to load for stereo files. Can be:
            - 0 or "left": Left channel
            - 1 or "right": Right channel
            - "mono" or "mix": Average of both channels
            - None: First channel (left for stereo)
        normalize: If True, normalize samples to [-1, 1] range.
            Default is True.

    Returns:
        WaveformTrace containing the audio data and metadata.

    Raises:
        LoaderError: If the file cannot be loaded.
        FormatError: If the file is not a valid WAV file.

    Example:
        >>> trace = load_wav("recording.wav")
        >>> print(f"Sample rate: {trace.metadata.sample_rate} Hz")
        >>> print(f"Duration: {trace.duration:.2f} seconds")

        >>> # Load right channel of stereo file
        >>> trace = load_wav("stereo.wav", channel="right")

    References:
        WAV file format: https://en.wikipedia.org/wiki/WAV
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    try:
        sample_rate, data = wavfile.read(str(path))
    except ValueError as e:
        raise FormatError(
            "Invalid WAV file format",
            file_path=str(path),
            expected="Valid WAV audio file",
        ) from e
    except Exception as e:
        raise LoaderError(
            "Failed to read WAV file",
            file_path=str(path),
            details=str(e),
        ) from e

    # Handle stereo/multichannel files
    if data.ndim == 2:
        audio_data, channel_name = _extract_multichannel_data(data, channel, str(path))
    else:
        audio_data, channel_name = _extract_mono_data(data, channel, str(path))

    # Convert to float64 and normalize if requested
    audio_data = _normalize_audio_data(audio_data.astype(np.float64), data.dtype, normalize)

    # Build metadata
    metadata = TraceMetadata(
        sample_rate=float(sample_rate),
        source_file=str(path),
        channel_name=channel_name,
        trigger_info={
            "original_dtype": str(data.dtype),
            "n_channels": data.shape[1] if data.ndim == 2 else 1,
            "normalized": normalize,
        },
    )

    return WaveformTrace(data=audio_data, metadata=metadata)


def get_wav_info(
    path: str | PathLike[str],
) -> dict:  # type: ignore[type-arg]
    """Get WAV file information without loading all data.

    Args:
        path: Path to the WAV file.

    Returns:
        Dictionary with file information:
        - sample_rate: Sample rate in Hz
        - n_channels: Number of channels
        - n_samples: Number of samples per channel
        - duration: Duration in seconds
        - dtype: Sample data type

    Raises:
        LoaderError: If the file cannot be read.

    Example:
        >>> info = get_wav_info("recording.wav")
        >>> print(f"Duration: {info['duration']:.2f}s")
        >>> print(f"Channels: {info['n_channels']}")
    """
    path = Path(path)

    if not path.exists():
        raise LoaderError(
            "File not found",
            file_path=str(path),
        )

    try:
        sample_rate, data = wavfile.read(str(path))

        n_samples = data.shape[0]
        n_channels = data.shape[1] if data.ndim == 2 else 1
        duration = n_samples / sample_rate

        return {
            "sample_rate": sample_rate,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "duration": duration,
            "dtype": str(data.dtype),
        }

    except Exception as e:
        raise LoaderError(
            "Failed to read WAV file info",
            file_path=str(path),
            details=str(e),
        ) from e


__all__ = ["get_wav_info", "load_wav"]
