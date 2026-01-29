"""Base classes for signal generation.

This module provides abstract base classes for signal generators,
ensuring consistent interfaces across all signal types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SignalMetadata:
    """Metadata for generated signals.

    Attributes:
        sample_rate: Sample rate in Hz
        duration: Signal duration in seconds
        channel_names: List of channel names
        description: Human-readable description
        generator: Name of generator that created this signal
        parameters: Dictionary of generation parameters
    """

    sample_rate: float
    duration: float
    channel_names: list[str] = field(default_factory=lambda: ["ch1"])
    description: str = ""
    generator: str = "SignalBuilder"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedSignal:
    """Container for generated signal data.

    Attributes:
        data: Dictionary mapping channel names to signal arrays
        metadata: Signal metadata
        time: Time array (optional, computed on demand)
    """

    data: dict[str, np.ndarray]
    metadata: SignalMetadata
    _time: np.ndarray | None = field(default=None, repr=False)

    @property
    def time(self) -> np.ndarray:
        """Get time array, computing if necessary."""
        if self._time is None:
            n_samples = len(next(iter(self.data.values())))
            self._time = np.arange(n_samples) / self.metadata.sample_rate
        return self._time

    @property
    def num_channels(self) -> int:
        """Number of channels in signal."""
        return len(self.data)

    @property
    def num_samples(self) -> int:
        """Number of samples per channel."""
        return len(next(iter(self.data.values())))

    def get_channel(self, name: str) -> np.ndarray:
        """Get signal data for a specific channel.

        Args:
            name: Channel name

        Returns:
            Signal array for the channel

        Raises:
            KeyError: If channel name not found
        """
        if name not in self.data:
            raise KeyError(f"Channel '{name}' not found. Available: {list(self.data.keys())}")
        return self.data[name]

    def save_npz(self, path: Path | str) -> None:
        """Save signal to NPZ format.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "sample_rate": self.metadata.sample_rate,
            "duration": self.metadata.duration,
            "channel_names": np.array(self.metadata.channel_names),
            "description": self.metadata.description,
            "generator": self.metadata.generator,
        }

        # Add channel data
        for name, data in self.data.items():
            save_dict[name] = data

        # Add parameters as JSON-serializable
        for key, value in self.metadata.parameters.items():
            if isinstance(value, (int, float, str, bool)):
                save_dict[f"param_{key}"] = value

        np.savez_compressed(path, **save_dict)

    @classmethod
    def load_npz(cls, path: Path | str) -> GeneratedSignal:
        """Load signal from NPZ format.

        Args:
            path: Input file path

        Returns:
            GeneratedSignal instance
        """
        path = Path(path)
        loaded = np.load(path, allow_pickle=True)

        sample_rate = float(loaded["sample_rate"])
        duration = float(loaded["duration"])
        channel_names = list(loaded.get("channel_names", ["ch1"]))
        description = str(loaded.get("description", ""))
        generator = str(loaded.get("generator", "unknown"))

        # Extract channel data
        data = {}
        for name in channel_names:
            if name in loaded:
                data[name] = loaded[name]

        # Extract parameters
        parameters = {}
        for key in loaded.files:
            if key.startswith("param_"):
                param_name = key[6:]  # Remove "param_" prefix
                parameters[param_name] = (
                    loaded[key].item() if loaded[key].ndim == 0 else loaded[key]
                )

        metadata = SignalMetadata(
            sample_rate=sample_rate,
            duration=duration,
            channel_names=channel_names,
            description=description,
            generator=generator,
            parameters=parameters,
        )

        return cls(data=data, metadata=metadata)


class BaseSignalGenerator(ABC):
    """Abstract base class for signal generators.

    Subclasses implement specific signal generation logic while
    inheriting common functionality like time base creation and
    noise addition.
    """

    def __init__(self, sample_rate: float = 1e6, duration: float = 0.01):
        """Initialize generator.

        Args:
            sample_rate: Sample rate in Hz (default 1 MHz)
            duration: Signal duration in seconds (default 10 ms)
        """
        self._sample_rate = sample_rate
        self._duration = duration

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return self._duration

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return int(self._sample_rate * self._duration)

    @property
    def time(self) -> np.ndarray:
        """Time array."""
        return np.arange(self.num_samples) / self._sample_rate

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate the signal.

        Returns:
            Generated signal as numpy array
        """

    def add_noise(self, signal: np.ndarray, snr_db: float) -> np.ndarray:
        """Add Gaussian noise to signal at specified SNR.

        Args:
            signal: Input signal
            snr_db: Target SNR in dB

        Returns:
            Signal with added noise
        """
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise

    def add_jitter(
        self,
        signal: np.ndarray,
        rj_rms: float = 0.0,
        dj_pp: float = 0.0,
    ) -> np.ndarray:
        """Add jitter to digital signal.

        Args:
            signal: Input digital signal
            rj_rms: Random jitter RMS in seconds
            dj_pp: Deterministic jitter peak-to-peak in seconds

        Returns:
            Signal with jitter applied
        """
        if rj_rms == 0 and dj_pp == 0:
            return signal

        # Find edges
        edges = np.where(np.diff(signal > 0.5))[0]
        if len(edges) == 0:
            return signal

        # Apply jitter by interpolating signal with shifted time base
        t_original = np.arange(len(signal)) / self._sample_rate
        t_jittered = t_original.copy()

        for edge_idx in edges:
            # Add random jitter
            jitter = np.random.randn() * rj_rms if rj_rms > 0 else 0
            # Add deterministic jitter (sinusoidal pattern)
            jitter += (dj_pp / 2) * np.sin(2 * np.pi * edge_idx / len(edges)) if dj_pp > 0 else 0

            # Apply jitter around edge
            edge_region = slice(max(0, edge_idx - 5), min(len(signal), edge_idx + 6))
            t_jittered[edge_region] += jitter

        return np.interp(t_original, t_jittered, signal)
