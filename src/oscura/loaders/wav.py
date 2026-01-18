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
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile

from oscura.core.exceptions import FormatError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from os import PathLike


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
        n_channels = data.shape[1]
        channel_names = (
            ["left", "right"] if n_channels == 2 else [f"ch{i}" for i in range(n_channels)]
        )

        if channel is None:
            # Default to first channel
            audio_data = data[:, 0]
            channel_name = channel_names[0]
        elif isinstance(channel, int):
            if channel < 0 or channel >= n_channels:
                raise LoaderError(
                    f"Channel index {channel} out of range",
                    file_path=str(path),
                    details=f"Available channels: 0-{n_channels - 1}",
                )
            audio_data = data[:, channel]
            channel_name = (
                channel_names[channel] if channel < len(channel_names) else f"ch{channel}"
            )
        elif isinstance(channel, str):
            channel_lower = channel.lower()
            if channel_lower in ("left", "l", "0"):
                audio_data = data[:, 0]
                channel_name = "left"
            elif channel_lower in ("right", "r", "1") and n_channels >= 2:
                audio_data = data[:, 1]
                channel_name = "right"
            elif channel_lower in ("mono", "mix", "avg"):
                # Average all channels
                audio_data = np.mean(data, axis=1)
                channel_name = "mono"
            else:
                raise LoaderError(
                    f"Invalid channel specifier: '{channel}'",
                    file_path=str(path),
                    details="Use 'left', 'right', 'mono', or channel index",
                )
        else:
            audio_data = data[:, 0]  # type: ignore[unreachable]
            channel_name = channel_names[0]
    else:
        # Mono file
        if channel is not None and isinstance(channel, int) and channel != 0:
            raise LoaderError(
                f"Channel index {channel} out of range",
                file_path=str(path),
                details="File is mono (only channel 0 available)",
            )
        audio_data = data
        channel_name = "mono"

    # Convert to float64
    audio_data = audio_data.astype(np.float64)

    # Normalize based on original dtype
    if normalize:
        if data.dtype == np.int16:
            audio_data = audio_data / 32768.0
        elif data.dtype == np.int32:
            audio_data = audio_data / 2147483648.0
        elif data.dtype == np.uint8:
            audio_data = (audio_data - 128.0) / 128.0
        elif data.dtype in (np.float32, np.float64):
            # Already in float format, typically [-1, 1]
            # Clip to ensure range
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

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
