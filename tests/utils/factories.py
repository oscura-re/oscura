"""Test data factories for Oscura tests.

This module provides factory classes for generating test data with
sensible defaults and easy customization.
"""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class SignalFactory:
    """Factory for creating test signals with configurable parameters.

    Attributes:
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        seed: Random seed for reproducibility.
    """

    sample_rate: float = 1e6
    duration: float = 0.001
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    @property
    def num_samples(self) -> int:
        """Calculate number of samples based on sample rate and duration."""
        return int(self.sample_rate * self.duration)

    def sine(
        self,
        frequency: float = 1e3,
        amplitude: float = 1.0,
        phase: float = 0.0,
        dc_offset: float = 0.0,
    ) -> NDArray[np.float64]:
        """Generate a sine wave signal.

        Args:
            frequency: Frequency in Hz.
            amplitude: Peak amplitude.
            phase: Phase offset in radians.
            dc_offset: DC offset.

        Returns:
            NumPy array containing the sine wave.
        """
        t = np.arange(self.num_samples) / self.sample_rate
        return amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset

    def square(
        self,
        frequency: float = 1e3,
        duty_cycle: float = 0.5,
        low: float = 0.0,
        high: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate a square wave signal.

        Args:
            frequency: Frequency in Hz.
            duty_cycle: Duty cycle (0.0 to 1.0).
            low: Low level value.
            high: High level value.

        Returns:
            NumPy array containing the square wave.
        """
        t = np.arange(self.num_samples) / self.sample_rate
        period = 1.0 / frequency
        phase = (t % period) / period
        return np.where(phase < duty_cycle, high, low)

    def noise(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        distribution: Literal["gaussian", "uniform"] = "gaussian",
    ) -> NDArray[np.float64]:
        """Generate noise signal.

        Args:
            mean: Mean value.
            std: Standard deviation (for Gaussian) or half-range (for uniform).
            distribution: Type of noise distribution.

        Returns:
            NumPy array containing the noise signal.
        """
        if distribution == "gaussian":
            return self._rng.normal(mean, std, self.num_samples)
        else:  # uniform
            return self._rng.uniform(mean - std, mean + std, self.num_samples)

    def pulse(
        self,
        pulse_width: float = 1e-6,
        pulse_delay: float = 0.0,
        amplitude: float = 1.0,
        baseline: float = 0.0,
    ) -> NDArray[np.float64]:
        """Generate a single pulse signal.

        Args:
            pulse_width: Width of the pulse in seconds.
            pulse_delay: Delay before the pulse starts in seconds.
            amplitude: Pulse amplitude.
            baseline: Baseline value.

        Returns:
            NumPy array containing the pulse signal.
        """
        signal = np.full(self.num_samples, baseline, dtype=np.float64)
        start_idx = int(pulse_delay * self.sample_rate)
        end_idx = start_idx + int(pulse_width * self.sample_rate)
        if start_idx < self.num_samples:
            signal[start_idx : min(end_idx, self.num_samples)] = amplitude
        return signal

    def chirp(
        self,
        f_start: float = 100.0,
        f_end: float = 10000.0,
        amplitude: float = 1.0,
    ) -> NDArray[np.float64]:
        """Generate a linear chirp (frequency sweep) signal.

        Args:
            f_start: Starting frequency in Hz.
            f_end: Ending frequency in Hz.
            amplitude: Signal amplitude.

        Returns:
            NumPy array containing the chirp signal.
        """
        t = np.arange(self.num_samples) / self.sample_rate
        duration = self.num_samples / self.sample_rate
        # Linear chirp: f(t) = f_start + (f_end - f_start) * t / duration
        k = (f_end - f_start) / duration
        phase = 2 * np.pi * (f_start * t + 0.5 * k * t**2)
        return amplitude * np.sin(phase)

    def with_noise(
        self,
        signal: NDArray[np.float64],
        snr_db: float = 20.0,
    ) -> NDArray[np.float64]:
        """Add Gaussian noise to a signal to achieve target SNR.

        Args:
            signal: Input signal.
            snr_db: Target signal-to-noise ratio in dB.

        Returns:
            Signal with added noise.
        """
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self._rng.normal(0, np.sqrt(noise_power), len(signal))
        return signal + noise

    def digital_pattern(
        self,
        pattern: list[int] | str,
        bit_rate: float = 1e6,
        low: float = 0.0,
        high: float = 3.3,
    ) -> NDArray[np.float64]:
        """Generate a digital signal from a bit pattern.

        Args:
            pattern: Bit pattern as list of 0/1 or binary string.
            bit_rate: Bit rate in bits per second.
            low: Voltage level for logic 0.
            high: Voltage level for logic 1.

        Returns:
            NumPy array containing the digital signal.
        """
        if isinstance(pattern, str):
            pattern = [int(b) for b in pattern]

        samples_per_bit = int(self.sample_rate / bit_rate)
        total_samples = len(pattern) * samples_per_bit

        # Repeat each bit value for samples_per_bit samples
        signal = np.repeat(pattern, samples_per_bit).astype(np.float64)
        signal = np.where(signal > 0, high, low)

        # Pad or truncate to requested duration
        if len(signal) < self.num_samples:
            signal = np.pad(signal, (0, self.num_samples - len(signal)), constant_values=low)
        else:
            signal = signal[: self.num_samples]

        return signal


@dataclass
class PacketFactory:
    """Factory for creating test packets with configurable formats.

    Attributes:
        sync_pattern: Synchronization pattern bytes.
        header_size: Size of the packet header in bytes.
        payload_size: Size of the payload in bytes.
        include_checksum: Whether to include a checksum.
        seed: Random seed for reproducibility.
    """

    sync_pattern: bytes = b"\xaa\x55"
    header_size: int = 8
    payload_size: int = 64
    include_checksum: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._sequence = 0

    @property
    def packet_size(self) -> int:
        """Calculate total packet size."""
        checksum_size = 2 if self.include_checksum else 0
        return self.header_size + self.payload_size + checksum_size

    def create(
        self,
        payload: bytes | None = None,
        sequence_num: int | None = None,
        corrupt: bool = False,
    ) -> bytes:
        """Create a single packet.

        Args:
            payload: Custom payload bytes (or auto-generate if None).
            sequence_num: Sequence number (or auto-increment if None).
            corrupt: If True, corrupt the packet (flip sync pattern).

        Returns:
            Complete packet as bytes.
        """
        import struct

        packet = bytearray()

        # Sync pattern
        packet.extend(self.sync_pattern)

        # Sequence number (2 bytes)
        seq = sequence_num if sequence_num is not None else self._sequence
        self._sequence = (self._sequence + 1) % 65536
        packet.extend(struct.pack("<H", seq))

        # Length field (2 bytes)
        packet.extend(struct.pack("<H", self.payload_size))

        # Pad header
        while len(packet) < self.header_size:
            packet.append(0x00)

        # Payload
        if payload is not None:
            packet.extend(payload[: self.payload_size].ljust(self.payload_size, b"\x00"))
        else:
            packet.extend(bytes(self._rng.integers(0, 256, self.payload_size, dtype=np.uint8)))

        # Checksum
        if self.include_checksum:
            checksum = self._calculate_crc16(packet)
            packet.extend(struct.pack("<H", checksum))

        # Corruption
        if corrupt:
            packet[0] ^= 0xFF

        return bytes(packet)

    def create_batch(
        self,
        count: int,
        corrupt_rate: float = 0.0,
    ) -> bytes:
        """Create multiple packets as a continuous byte stream.

        Args:
            count: Number of packets to create.
            corrupt_rate: Fraction of packets to corrupt (0.0 to 1.0).

        Returns:
            Concatenated packet bytes.
        """
        packets = bytearray()
        for _ in range(count):
            corrupt = self._rng.random() < corrupt_rate
            packets.extend(self.create(corrupt=corrupt))
        return bytes(packets)

    def create_with_gaps(
        self,
        count: int,
        gap_size_range: tuple[int, int] = (10, 100),
        gap_fill: int = 0x00,
    ) -> bytes:
        """Create packets with random gaps between them.

        Args:
            count: Number of packets to create.
            gap_size_range: Min and max gap size in bytes.
            gap_fill: Byte value to use for gap fill.

        Returns:
            Packets with gaps as bytes.
        """
        data = bytearray()
        for i in range(count):
            if i > 0:
                gap_size = self._rng.integers(gap_size_range[0], gap_size_range[1])
                data.extend(bytes([gap_fill] * gap_size))
            data.extend(self.create())
        return bytes(data)

    def _calculate_crc16(self, data: bytes | bytearray) -> int:
        """Calculate CRC-16 checksum (Modbus variant)."""
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF


@dataclass
class WaveformFactory:
    """Factory for creating complete waveform data structures.

    This factory creates waveform objects compatible with Oscura's
    internal representation.

    Attributes:
        sample_rate: Sample rate in Hz.
        num_samples: Number of samples.
        num_channels: Number of channels.
        seed: Random seed for reproducibility.
    """

    sample_rate: float = 1e6
    num_samples: int = 10000
    num_channels: int = 1
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def create(
        self,
        pattern: Literal["sine", "square", "noise", "pulse"] = "sine",
        frequency: float = 1e3,
        amplitude: float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a waveform data structure.

        Args:
            pattern: Type of waveform to generate.
            frequency: Frequency for periodic waveforms.
            amplitude: Signal amplitude.
            **kwargs: Additional parameters passed to the signal generator.

        Returns:
            Dictionary with waveform data and metadata.
        """
        signal_factory = SignalFactory(
            sample_rate=self.sample_rate,
            duration=self.num_samples / self.sample_rate,
            seed=self.seed,
        )

        channels = []
        for ch in range(self.num_channels):
            if pattern == "sine":
                data = signal_factory.sine(
                    frequency=frequency,
                    amplitude=amplitude,
                    phase=ch * np.pi / 4,  # Offset phase per channel
                )
            elif pattern == "square":
                data = signal_factory.square(
                    frequency=frequency,
                    high=amplitude,
                )
            elif pattern == "noise":
                data = signal_factory.noise(std=amplitude)
            elif pattern == "pulse":
                data = signal_factory.pulse(amplitude=amplitude, **kwargs)
            else:
                data = signal_factory.noise(std=amplitude)

            channels.append(
                {
                    "name": f"CH{ch + 1}",
                    "data": data,
                    "scale": 1.0,
                    "offset": 0.0,
                    "unit": "V",
                }
            )

        time_array = np.arange(self.num_samples) / self.sample_rate

        return {
            "time": time_array,
            "channels": channels,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "num_channels": self.num_channels,
            "metadata": {
                "pattern": pattern,
                "frequency": frequency,
                "amplitude": amplitude,
            },
        }

    def create_multichannel(
        self,
        channel_configs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create a multi-channel waveform with different configurations per channel.

        Args:
            channel_configs: List of configuration dicts for each channel.
                Each dict can have: pattern, frequency, amplitude, phase.

        Returns:
            Dictionary with waveform data and metadata.
        """
        signal_factory = SignalFactory(
            sample_rate=self.sample_rate,
            duration=self.num_samples / self.sample_rate,
            seed=self.seed,
        )

        channels = []
        for i, config in enumerate(channel_configs):
            pattern = config.get("pattern", "sine")
            frequency = config.get("frequency", 1e3)
            amplitude = config.get("amplitude", 1.0)
            phase = config.get("phase", 0.0)

            if pattern == "sine":
                data = signal_factory.sine(frequency=frequency, amplitude=amplitude, phase=phase)
            elif pattern == "square":
                data = signal_factory.square(frequency=frequency, high=amplitude)
            elif pattern == "noise":
                data = signal_factory.noise(std=amplitude)
            else:
                data = signal_factory.sine(frequency=frequency, amplitude=amplitude)

            channels.append(
                {
                    "name": config.get("name", f"CH{i + 1}"),
                    "data": data,
                    "scale": config.get("scale", 1.0),
                    "offset": config.get("offset", 0.0),
                    "unit": config.get("unit", "V"),
                }
            )

        time_array = np.arange(self.num_samples) / self.sample_rate

        return {
            "time": time_array,
            "channels": channels,
            "sample_rate": self.sample_rate,
            "num_samples": self.num_samples,
            "num_channels": len(channel_configs),
            "metadata": {"channel_configs": channel_configs},
        }


# Convenience aliases for common factory configurations
def create_test_signal(
    pattern: str = "sine",
    sample_rate: float = 1e6,
    duration: float = 0.001,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Quick helper to create a test signal.

    Args:
        pattern: Signal pattern (sine, square, noise, pulse, chirp).
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        **kwargs: Additional arguments for the pattern generator.

    Returns:
        NumPy array with the signal.
    """
    factory = SignalFactory(sample_rate=sample_rate, duration=duration)

    if pattern == "sine":
        return factory.sine(**kwargs)
    elif pattern == "square":
        return factory.square(**kwargs)
    elif pattern == "noise":
        return factory.noise(**kwargs)
    elif pattern == "pulse":
        return factory.pulse(**kwargs)
    elif pattern == "chirp":
        return factory.chirp(**kwargs)
    else:
        return factory.sine(**kwargs)


def create_test_packets(
    count: int = 10,
    packet_size: int = 72,
    **kwargs: Any,
) -> bytes:
    """Quick helper to create test packets.

    Args:
        count: Number of packets.
        packet_size: Total packet size in bytes.
        **kwargs: Additional arguments for PacketFactory.

    Returns:
        Byte stream of packets.
    """
    # Calculate payload size from total packet size
    sync_size = len(kwargs.get("sync_pattern", b"\xaa\x55"))
    header_size = kwargs.get("header_size", 8)
    checksum_size = 2 if kwargs.get("include_checksum", True) else 0
    payload_size = packet_size - header_size - checksum_size

    factory = PacketFactory(payload_size=payload_size, **kwargs)
    return factory.create_batch(count)
