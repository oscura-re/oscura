"""Signal builder utilities for demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SignalBuilder:
    """Fluent builder for creating complex synthetic signals.

    Provides a fluent interface for building test signals by chaining
    multiple signal components together. Supports sine waves, harmonics,
    noise, and various waveform types.

    Attributes:
        sample_rate: Sample rate in Hz
        duration: Signal duration in seconds

    Example:
        >>> builder = SignalBuilder(sample_rate=1e6, duration=0.01)
        >>> trace = (builder
        ...     .add_sine(frequency=1000, amplitude=1.0)
        ...     .add_harmonics(fundamental=1000, thd_percent=5.0)
        ...     .add_noise(snr_db=60)
        ...     .build())
        >>> print(len(trace.data))
        10000
    """

    sample_rate: float
    duration: float
    _components: list[dict[str, Any]] = field(default_factory=list, init=False)
    _num_samples: int = field(init=False)
    _time: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        """Initialize internal state after dataclass initialization."""
        self._num_samples = int(self.duration * self.sample_rate)
        self._time = np.arange(self._num_samples) / self.sample_rate

    def add_sine(
        self, frequency: float, amplitude: float = 1.0, phase: float = 0.0
    ) -> SignalBuilder:
        """Add sine wave component.

        Args:
            frequency: Frequency in Hz
            amplitude: Peak amplitude
            phase: Phase offset in radians

        Returns:
            Self for method chaining

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6, duration=0.001)
            >>> builder.add_sine(frequency=1000, amplitude=2.0)
            SignalBuilder(...)
        """
        self._components.append(
            {"type": "sine", "frequency": frequency, "amplitude": amplitude, "phase": phase}
        )
        return self

    def add_square(
        self, frequency: float, amplitude: float = 1.0, duty_cycle: float = 0.5
    ) -> SignalBuilder:
        """Add square wave component.

        Args:
            frequency: Frequency in Hz
            amplitude: Peak amplitude
            duty_cycle: Duty cycle (0.0 to 1.0)

        Returns:
            Self for method chaining

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6, duration=0.001)
            >>> builder.add_square(frequency=1000, duty_cycle=0.3)
            SignalBuilder(...)
        """
        self._components.append(
            {
                "type": "square",
                "frequency": frequency,
                "amplitude": amplitude,
                "duty_cycle": duty_cycle,
            }
        )
        return self

    def add_harmonics(self, fundamental: float, thd_percent: float = 5.0) -> SignalBuilder:
        """Add harmonic distortion to signal.

        Adds harmonics (2nd through 5th) based on specified THD percentage.
        Harmonic amplitudes decrease with order.

        Args:
            fundamental: Fundamental frequency in Hz
            thd_percent: Total harmonic distortion as percentage (0-100)

        Returns:
            Self for method chaining

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6, duration=0.001)
            >>> builder.add_sine(frequency=1000).add_harmonics(fundamental=1000, thd_percent=10.0)
            SignalBuilder(...)
        """
        self._components.append(
            {"type": "harmonics", "fundamental": fundamental, "thd_percent": thd_percent}
        )
        return self

    def add_pulse(
        self,
        frequency: float,
        amplitude: float = 1.0,
        width: float | None = None,
        rise_time: float = 1e-9,
    ) -> SignalBuilder:
        """Add pulse train component.

        Args:
            frequency: Pulse repetition frequency in Hz
            amplitude: Pulse amplitude
            width: Pulse width in seconds (defaults to 50% duty cycle)
            rise_time: Rise/fall time in seconds

        Returns:
            Self for method chaining

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6, duration=0.001)
            >>> builder.add_pulse(frequency=1000, amplitude=5.0, width=100e-6)
            SignalBuilder(...)
        """
        period = 1.0 / frequency
        pulse_width = width if width is not None else period * 0.5
        self._components.append(
            {
                "type": "pulse",
                "frequency": frequency,
                "amplitude": amplitude,
                "width": pulse_width,
                "rise_time": rise_time,
            }
        )
        return self

    def add_noise(self, snr_db: float = 40.0) -> SignalBuilder:
        """Add white Gaussian noise.

        Args:
            snr_db: Signal-to-noise ratio in dB

        Returns:
            Self for method chaining

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6, duration=0.001)
            >>> builder.add_sine(frequency=1000).add_noise(snr_db=60)
            SignalBuilder(...)
        """
        self._components.append({"type": "noise", "snr_db": snr_db})
        return self

    def add_dc_offset(self, offset: float) -> SignalBuilder:
        """Add DC offset to signal.

        Args:
            offset: DC offset value

        Returns:
            Self for method chaining

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6, duration=0.001)
            >>> builder.add_sine(frequency=1000).add_dc_offset(2.5)
            SignalBuilder(...)
        """
        self._components.append({"type": "dc_offset", "offset": offset})
        return self

    def build(self) -> dict[str, NDArray[np.float64]]:
        """Build signal from accumulated components.

        Generates the final signal by combining all added components.
        Signal is normalized if noise is present.

        Returns:
            Dictionary with 'ch1' key containing signal array

        Example:
            >>> builder = SignalBuilder(sample_rate=1e6, duration=0.001)
            >>> result = builder.add_sine(frequency=1000).build()
            >>> 'ch1' in result
            True
        """
        signal = np.zeros(self._num_samples)

        # First pass: add all non-noise components
        for component in self._components:
            if component["type"] == "sine":
                signal += component["amplitude"] * np.sin(
                    2 * np.pi * component["frequency"] * self._time + component["phase"]
                )
            elif component["type"] == "square":
                phase = (self._time * component["frequency"]) % 1.0
                square_wave = np.where(
                    phase < component["duty_cycle"], component["amplitude"], -component["amplitude"]
                )
                signal += square_wave
            elif component["type"] == "harmonics":
                # Add 2nd through 5th harmonics with decreasing amplitudes
                fund = component["fundamental"]
                thd = component["thd_percent"] / 100.0
                # Simple model: each harmonic gets 1/n of the THD energy
                for n in range(2, 6):
                    harmonic_amp = thd / (n**1.5)  # Decreasing amplitude
                    signal += harmonic_amp * np.sin(2 * np.pi * n * fund * self._time)
            elif component["type"] == "pulse":
                period = 1.0 / component["frequency"]
                pulse_start_times = np.arange(0, self.duration, period)
                for start_time in pulse_start_times:
                    start_idx = int(start_time * self.sample_rate)
                    end_idx = int((start_time + component["width"]) * self.sample_rate)
                    rise_samples = max(1, int(component["rise_time"] * self.sample_rate))

                    if end_idx >= self._num_samples:
                        break

                    # Rising edge
                    if start_idx + rise_samples < self._num_samples:
                        signal[start_idx : start_idx + rise_samples] = component[
                            "amplitude"
                        ] * np.linspace(0, 1, rise_samples)

                    # Flat top
                    signal[start_idx + rise_samples : end_idx - rise_samples] = component[
                        "amplitude"
                    ]

                    # Falling edge
                    if end_idx < self._num_samples:
                        signal[end_idx - rise_samples : end_idx] = component[
                            "amplitude"
                        ] * np.linspace(1, 0, rise_samples)
            elif component["type"] == "dc_offset":
                signal += component["offset"]

        # Second pass: add noise (must be after signal is generated)
        for component in self._components:
            if component["type"] == "noise":
                signal_power = np.mean(signal**2)
                if signal_power > 0:
                    noise_power = signal_power / (10 ** (component["snr_db"] / 10))
                    noise = np.random.normal(0, np.sqrt(noise_power), self._num_samples)
                    signal += noise

        return {"ch1": signal}


__all__ = ["SignalBuilder"]
