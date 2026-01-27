"""Synthetic test data generation with known ground truth.

Provides utilities for generating synthetic test data with known properties
for validation and testing purposes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from oscura.core.types import TraceMetadata, WaveformTrace


@dataclass
class SyntheticPacketConfig:
    """Configuration for synthetic packet generation."""

    packet_size: int = 1024
    header_size: int = 16
    sync_pattern: bytes = b"\xaa\x55"
    include_sequence: bool = True
    include_timestamp: bool = True
    include_checksum: bool = True
    checksum_algorithm: str = "crc16"
    noise_level: float = 0.0  # 0-1, fraction of corrupted packets


@dataclass
class SyntheticSignalConfig:
    """Configuration for synthetic digital signal."""

    pattern_type: Literal["square", "uart", "spi", "i2c", "random"] = "square"
    sample_rate: float = 100e6
    duration_samples: int = 10000
    frequency: float = 1e6  # For clock/square wave
    noise_snr_db: float = 40  # Signal-to-noise ratio


@dataclass
class SyntheticMessageConfig:
    """Configuration for synthetic protocol messages."""

    message_size: int = 64
    num_fields: int = 5
    include_header: bool = True
    include_length: bool = True
    include_checksum: bool = True
    variation: float = 0.1  # Fraction of variable bytes


@dataclass
class GroundTruth:
    """Ground truth data for validation."""

    field_boundaries: list[int] = field(default_factory=list)
    field_types: list[str] = field(default_factory=list)
    sequence_numbers: list[int] = field(default_factory=list)
    pattern_period: int | None = None
    cluster_labels: list[int] = field(default_factory=list)
    checksum_offsets: list[int] = field(default_factory=list)
    decoded_bytes: list[int] = field(default_factory=list)
    edge_positions: list[int] = field(default_factory=list)
    frequency_hz: float | None = None


class SyntheticDataGenerator:
    """Generate synthetic test data with known ground truth."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility.

        Args:
            seed: Random seed for reproducible generation.
        """
        self.rng = np.random.default_rng(seed)

    def generate_packets(
        self, config: SyntheticPacketConfig, count: int = 100
    ) -> tuple[bytes, GroundTruth]:
        """Generate synthetic binary packets.

        Args:
            config: Packet generation configuration.
            count: Number of packets to generate.

        Returns:
            Tuple of (binary_data, ground_truth).
        """
        packets = bytearray()
        ground_truth = GroundTruth()

        for i in range(count):
            packet = bytearray()

            # Sync pattern
            packet.extend(config.sync_pattern)

            # Sequence number (2 bytes)
            if config.include_sequence:
                packet.extend(struct.pack("<H", i))
                ground_truth.sequence_numbers.append(i)

            # Timestamp (4 bytes)
            if config.include_timestamp:
                timestamp = i * 1000  # Incrementing by 1000 microseconds
                packet.extend(struct.pack("<I", timestamp))

            # Padding to header size
            while len(packet) < config.header_size:
                packet.append(0x00)

            # Sample data
            sample_data_size = config.packet_size - config.header_size
            if config.include_checksum:
                sample_data_size -= 2

            # Counter pattern for samples (easier to validate)
            for j in range(sample_data_size // 2):
                packet.extend(struct.pack("<H", j))

            # Checksum (CRC-16)
            if config.include_checksum:
                checksum = self._calculate_crc16(packet)
                checksum_offset = len(packets) + len(packet)
                ground_truth.checksum_offsets.append(checksum_offset)
                packet.extend(struct.pack("<H", checksum))

            packets.extend(packet)

        # Apply noise (corrupt random packets)
        if config.noise_level > 0:
            packets = bytearray(
                self.corrupt_packets(bytes(packets), config.packet_size, config.noise_level)
            )

        return bytes(packets), ground_truth

    def generate_digital_signal(
        self, config: SyntheticSignalConfig
    ) -> tuple[NDArray[np.float64], GroundTruth]:
        """Generate synthetic digital signal.

        Args:
            config: Signal generation configuration.

        Returns:
            Tuple of (signal_array, ground_truth).
        """
        ground_truth = GroundTruth()
        signal: NDArray[np.float64]

        if config.pattern_type == "square":
            # Generate square wave
            period_samples = int(config.sample_rate / config.frequency)
            ground_truth.pattern_period = period_samples
            ground_truth.frequency_hz = config.frequency

            t = np.arange(config.duration_samples)
            signal = (np.sin(2 * np.pi * config.frequency * t / config.sample_rate) > 0).astype(
                np.float64
            )

            # Track edge positions
            edges = np.where(np.diff(signal) != 0)[0] + 1
            ground_truth.edge_positions = edges.tolist()

        elif config.pattern_type == "uart":
            # Generate UART signal (8N1)
            signal, uart_truth = self._generate_uart_signal(config)
            ground_truth.decoded_bytes = uart_truth["bytes"]
            ground_truth.edge_positions = uart_truth["edges"]

        elif config.pattern_type == "random":
            # Random digital signal
            signal = self.rng.choice([0.0, 1.0], size=config.duration_samples)

        else:
            # Default to simple pattern
            pattern = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=np.float64)
            signal = np.tile(pattern, config.duration_samples // len(pattern) + 1)[
                : config.duration_samples
            ]
            ground_truth.pattern_period = len(pattern)

        # Scale to 3.3V logic levels
        signal = signal * 3.3

        # Add noise
        if config.noise_snr_db < np.inf:
            noisy_signal = self.add_noise(signal, config.noise_snr_db)
            assert isinstance(noisy_signal, np.ndarray), (
                "add_noise should return ndarray for ndarray input"
            )
            signal = noisy_signal

        return signal, ground_truth

    def generate_protocol_messages(
        self, config: SyntheticMessageConfig, count: int = 100
    ) -> tuple[list[bytes], GroundTruth]:
        """Generate synthetic protocol messages.

        Args:
            config: Message generation configuration.
            count: Number of messages to generate.

        Returns:
            Tuple of (message_list, ground_truth).
        """
        messages = []
        ground_truth = GroundTruth()

        # Define field structure
        field_boundaries = [0]
        field_types = []

        current_offset = 0

        if config.include_header:
            # 2-byte sync pattern
            field_boundaries.append(current_offset + 2)
            field_types.append("constant")
            current_offset += 2

        if config.include_length:
            # 2-byte length field
            field_boundaries.append(current_offset + 2)
            field_types.append("length")
            current_offset += 2

        # Sequence number (2 bytes)
        field_boundaries.append(current_offset + 2)
        field_types.append("sequence")
        current_offset += 2

        # Timestamp (4 bytes)
        field_boundaries.append(current_offset + 4)
        field_types.append("timestamp")
        current_offset += 4

        # Payload (variable size)
        payload_size = config.message_size - current_offset
        if config.include_checksum:
            payload_size -= 2

        field_boundaries.append(current_offset + payload_size)
        field_types.append("data")
        current_offset += payload_size

        if config.include_checksum:
            field_boundaries.append(current_offset + 2)
            field_types.append("checksum")

        ground_truth.field_boundaries = field_boundaries
        ground_truth.field_types = field_types

        # Generate messages
        for i in range(count):
            message = bytearray()

            if config.include_header:
                message.extend(b"\xaa\x55")

            if config.include_length:
                message.extend(struct.pack("<H", config.message_size))

            # Sequence number
            message.extend(struct.pack("<H", i))
            ground_truth.sequence_numbers.append(i)

            # Timestamp
            timestamp = i * 100  # Incrementing
            message.extend(struct.pack("<I", timestamp))

            # Payload (partially random, partially constant based on variation)
            for _ in range(payload_size):
                if self.rng.random() < config.variation:
                    message.append(self.rng.integers(0, 256))
                else:
                    message.append(0x42)  # Constant byte

            # Checksum
            if config.include_checksum:
                checksum = self._calculate_crc16(message)
                message.extend(struct.pack("<H", checksum))

            messages.append(bytes(message))

        return messages, ground_truth

    def add_noise(
        self, data: bytes | NDArray[np.float64], snr_db: float = 20
    ) -> bytes | NDArray[np.float64]:
        """Add noise to data.

        Args:
            data: Input data (bytes or numpy array).
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Noisy data (same type as input).
        """
        if isinstance(data, bytes):
            # For bytes, add random bit flips
            data_array = np.frombuffer(data, dtype=np.uint8)
            noise_rate = 10 ** (-snr_db / 10)
            mask = self.rng.random(len(data_array)) < noise_rate
            noisy = data_array.copy()
            noisy[mask] ^= self.rng.integers(1, 256, size=int(np.sum(mask)), dtype=np.uint8)
            return noisy.tobytes()
        else:
            # For arrays, add Gaussian noise
            signal_power = np.mean(data**2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = self.rng.normal(0, np.sqrt(noise_power), len(data))
            return data + noise

    def corrupt_packets(
        self, packets: bytes, packet_size: int, corruption_rate: float = 0.01
    ) -> bytes:
        """Corrupt random packets for testing error handling.

        Args:
            packets: Binary packet data.
            packet_size: Size of each packet in bytes.
            corruption_rate: Fraction of packets to corrupt (0-1).

        Returns:
            Corrupted packet data.
        """
        packets_array = bytearray(packets)
        num_packets = len(packets) // packet_size

        for i in range(num_packets):
            if self.rng.random() < corruption_rate:
                # Corrupt sync marker
                offset = i * packet_size
                packets_array[offset] ^= 0xFF

        return bytes(packets_array)

    def _calculate_crc16(self, data: bytes | bytearray) -> int:
        """Calculate CRC-16 checksum.

        Args:
            data: Input data.

        Returns:
            CRC-16 checksum value.
        """
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    def _generate_uart_signal(
        self, config: SyntheticSignalConfig
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Generate UART signal encoding a test message.

        Args:
            config: Signal configuration.

        Returns:
            Tuple of (signal, metadata_dict).
        """
        # UART parameters
        baud_rate = 9600
        samples_per_bit = int(config.sample_rate / baud_rate)

        # Test message
        message = b"Hello, World!"

        # Encode with start/stop bits
        bits = []
        edges = []

        for byte_val in message:
            # Start bit (0)
            bits.extend([0] * samples_per_bit)
            edges.append(len(bits))

            # Data bits (LSB first)
            for i in range(8):
                bit = (byte_val >> i) & 1
                bits.extend([bit] * samples_per_bit)
                if i > 0 and bits[-1] != bits[-samples_per_bit - 1]:
                    edges.append(len(bits) - samples_per_bit)

            # Stop bit (1)
            bits.extend([1] * samples_per_bit)
            if bits[-1] != bits[-samples_per_bit - 1]:
                edges.append(len(bits) - samples_per_bit)

        # Pad to duration
        signal = np.array(bits[: config.duration_samples], dtype=np.float64)
        if len(signal) < config.duration_samples:
            padding = np.ones(config.duration_samples - len(signal), dtype=np.float64)
            signal = np.concatenate([signal, padding])

        metadata = {"bytes": list(message), "edges": edges[: len(signal)]}

        return signal, metadata


# =============================================================================
# Convenience functions for generating WaveformTrace objects
# =============================================================================


def generate_sine_wave(
    frequency: float = 1e6,
    amplitude: float = 1.0,
    sample_rate: float = 100e6,
    duration: float = 10e-6,
    offset: float = 0.0,
    phase: float = 0.0,
    noise_level: float = 0.0,
) -> WaveformTrace:
    """Generate a sine wave WaveformTrace.

    Args:
        frequency: Signal frequency in Hz.
        amplitude: Peak amplitude (will produce 2*amplitude peak-to-peak).
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        offset: DC offset.
        phase: Initial phase in radians.
        noise_level: RMS noise level to add (0 for clean signal).

    Returns:
        WaveformTrace containing the sine wave.

    Example:
        >>> from oscura.validation.testing import generate_sine_wave
        >>> trace = generate_sine_wave(frequency=1e6, amplitude=1.0)
        >>> print(f"Samples: {len(trace.data)}")
    """
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset

    if noise_level > 0:
        rng = np.random.default_rng(42)
        data = data + rng.normal(0, noise_level, num_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def generate_square_wave(
    frequency: float = 1e6,
    duty_cycle: float = 0.5,
    sample_rate: float = 100e6,
    duration: float = 10e-6,
    low: float = 0.0,
    high: float = 1.0,
    noise_level: float = 0.0,
) -> WaveformTrace:
    """Generate a square wave WaveformTrace.

    Args:
        frequency: Signal frequency in Hz.
        duty_cycle: Duty cycle (0.0 to 1.0).
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        low: Low voltage level.
        high: High voltage level.
        noise_level: RMS noise level to add.

    Returns:
        WaveformTrace containing the square wave.

    Example:
        >>> from oscura.validation.testing import generate_square_wave
        >>> trace = generate_square_wave(frequency=500e3, duty_cycle=0.3)
    """
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    period = 1.0 / frequency

    # Create square wave using modulo operation
    phase = (t % period) / period
    data = np.where(phase < duty_cycle, high, low).astype(np.float64)

    if noise_level > 0:
        rng = np.random.default_rng(42)
        data = data + rng.normal(0, noise_level, num_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def generate_dc(
    level: float = 1.0,
    sample_rate: float = 100e6,
    duration: float = 10e-6,
    noise_level: float = 0.0,
) -> WaveformTrace:
    """Generate a DC (constant) signal WaveformTrace.

    Args:
        level: DC voltage level.
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        noise_level: RMS noise level to add.

    Returns:
        WaveformTrace containing the DC signal.

    Example:
        >>> from oscura.validation.testing import generate_dc
        >>> trace = generate_dc(level=1.5, duration=10e-6)
    """
    num_samples = int(sample_rate * duration)
    data = np.full(num_samples, level, dtype=np.float64)

    if noise_level > 0:
        rng = np.random.default_rng(42)
        data = data + rng.normal(0, noise_level, num_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def generate_multi_tone(
    frequencies: list[float],
    amplitudes: list[float] | None = None,
    phases: list[float] | None = None,
    sample_rate: float = 100e6,
    duration: float = 100e-6,
    noise_level: float = 0.0,
) -> WaveformTrace:
    """Generate a multi-tone (sum of sine waves) WaveformTrace.

    Args:
        frequencies: List of frequencies in Hz.
        amplitudes: List of amplitudes for each tone. If None, all 1.0.
        phases: List of phases in radians. If None, all 0.0.
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        noise_level: RMS noise level to add.

    Returns:
        WaveformTrace containing the multi-tone signal.

    Raises:
        ValueError: If frequencies, amplitudes, and phases have different lengths.

    Example:
        >>> from oscura.validation.testing import generate_multi_tone
        >>> trace = generate_multi_tone(
        ...     frequencies=[1e6, 2.5e6, 4e6],
        ...     amplitudes=[1.0, 0.5, 0.25]
        ... )
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)
    if phases is None:
        phases = [0.0] * len(frequencies)

    if len(frequencies) != len(amplitudes) or len(frequencies) != len(phases):
        raise ValueError("frequencies, amplitudes, and phases must have same length")

    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = np.zeros(num_samples, dtype=np.float64)

    for freq, amp, phase in zip(frequencies, amplitudes, phases, strict=True):
        data += amp * np.sin(2 * np.pi * freq * t + phase)

    if noise_level > 0:
        rng = np.random.default_rng(42)
        data = data + rng.normal(0, noise_level, num_samples)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def generate_pulse(
    width: float = 1e-6,
    rise_time: float = 10e-9,
    fall_time: float = 10e-9,
    sample_rate: float = 1e9,
    duration: float = 10e-6,
    low: float = 0.0,
    high: float = 1.0,
    pulse_position: float = 0.5,
    overshoot: float = 0.0,
) -> WaveformTrace:
    """Generate a pulse WaveformTrace with configurable rise/fall times.

    Args:
        width: Pulse width in seconds.
        rise_time: Rise time (10%-90%) in seconds.
        fall_time: Fall time (90%-10%) in seconds.
        sample_rate: Sample rate in Hz.
        duration: Duration in seconds.
        low: Low voltage level.
        high: High voltage level.
        pulse_position: Position of pulse center as fraction of duration.
        overshoot: Overshoot as fraction of amplitude (0 for none).

    Returns:
        WaveformTrace containing the pulse.

    Example:
        >>> from oscura.validation.testing import generate_pulse
        >>> trace = generate_pulse(width=1e-6, rise_time=10e-9)
    """
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = np.full(num_samples, low, dtype=np.float64)

    # Pulse timing
    center = duration * pulse_position
    start = center - width / 2
    end = center + width / 2

    amplitude = high - low

    for i, time in enumerate(t):
        if time < start:
            data[i] = low
        elif time < start + rise_time:
            # Rising edge (exponential approach)
            progress = (time - start) / rise_time
            data[i] = low + amplitude * progress
            if overshoot > 0 and progress > 0.9:
                data[i] += amplitude * overshoot * np.sin(np.pi * (progress - 0.9) / 0.1)
        elif time < end:
            data[i] = high
        elif time < end + fall_time:
            # Falling edge
            progress = (time - end) / fall_time
            data[i] = high - amplitude * progress
        else:
            data[i] = low

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


# =============================================================================
# Legacy convenience functions (kept for backward compatibility)
# =============================================================================


def generate_packets(count: int = 100, **kwargs: Any) -> tuple[bytes, GroundTruth]:
    """Generate synthetic packets with defaults.

    Args:
        count: Number of packets to generate.
        **kwargs: Additional configuration parameters.

    Returns:
        Tuple of (binary_data, ground_truth).
    """
    config = SyntheticPacketConfig(**kwargs)
    generator = SyntheticDataGenerator()
    return generator.generate_packets(config, count)


def generate_digital_signal(
    pattern: str = "square", **kwargs: Any
) -> tuple[NDArray[np.float64], GroundTruth]:
    """Generate synthetic signal with defaults.

    Args:
        pattern: Pattern type ('square', 'uart', 'random', etc.).
        **kwargs: Additional configuration parameters.

    Returns:
        Tuple of (signal_array, ground_truth).
    """
    # Determine pattern type
    valid_patterns = ["square", "uart", "spi", "i2c", "random"]
    pattern_type = pattern if pattern in valid_patterns else "square"

    # Filter out pattern_type from kwargs to avoid duplicate argument error
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "pattern_type"}

    config = SyntheticSignalConfig(
        pattern_type=cast("Literal['square', 'uart', 'spi', 'i2c', 'random']", pattern_type),
        **filtered_kwargs,
    )
    generator = SyntheticDataGenerator()
    return generator.generate_digital_signal(config)


def generate_protocol_messages(count: int = 100, **kwargs: Any) -> tuple[list[bytes], GroundTruth]:
    """Generate synthetic messages with defaults.

    Args:
        count: Number of messages to generate.
        **kwargs: Additional configuration parameters.

    Returns:
        Tuple of (message_list, ground_truth).
    """
    config = SyntheticMessageConfig(**kwargs)
    generator = SyntheticDataGenerator()
    return generator.generate_protocol_messages(config, count)


def generate_test_dataset(
    output_dir: str,
    num_packets: int = 1000,
    num_signals: int = 10,
    num_messages: int = 500,
) -> dict[str, Any]:
    """Generate complete test dataset with ground truth.

    Args:
        output_dir: Directory to save test data.
        num_packets: Number of packets to generate.
        num_signals: Number of signals to generate.
        num_messages: Number of messages to generate.

    Returns:
        Dictionary with dataset metadata and file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = SyntheticDataGenerator()
    metadata: dict[str, Any] = {
        "dataset_type": "synthetic_test_data",
        "generated_files": [],
    }

    # Generate packets
    packet_config = SyntheticPacketConfig()
    packets, packet_truth = generator.generate_packets(packet_config, num_packets)
    packet_file = output_path / "test_packets.bin"
    packet_file.write_bytes(packets)
    metadata["generated_files"].append(
        {
            "path": str(packet_file),
            "type": "packets",
            "count": num_packets,
            "ground_truth": {
                "sequence_numbers": packet_truth.sequence_numbers[:10],  # First 10
                "checksum_offsets": packet_truth.checksum_offsets[:10],
            },
        }
    )

    # Generate signals
    for i in range(num_signals):
        signal_config = SyntheticSignalConfig(
            pattern_type="square" if i % 2 == 0 else "uart",
            frequency=1e6 * (i + 1),
        )
        signal, signal_truth = generator.generate_digital_signal(signal_config)
        signal_file = output_path / f"test_signal_{i:03d}.npy"
        np.save(signal_file, signal)
        metadata["generated_files"].append(
            {
                "path": str(signal_file),
                "type": "signal",
                "pattern": signal_config.pattern_type,
                "ground_truth": {
                    "frequency_hz": signal_truth.frequency_hz,
                    "period_samples": signal_truth.pattern_period,
                },
            }
        )

    # Generate protocol messages
    message_config = SyntheticMessageConfig()
    messages, message_truth = generator.generate_protocol_messages(message_config, num_messages)
    messages_file = output_path / "test_messages.bin"
    with messages_file.open("wb") as f:
        for msg in messages:
            f.write(msg)
    metadata["generated_files"].append(
        {
            "path": str(messages_file),
            "type": "messages",
            "count": num_messages,
            "ground_truth": {
                "field_boundaries": message_truth.field_boundaries,
                "field_types": message_truth.field_types,
                "message_size": message_config.message_size,
            },
        }
    )

    # Save metadata
    import json

    metadata_file = output_path / "dataset_metadata.json"
    with metadata_file.open("w") as f:
        json.dump(metadata, f, indent=2)

    metadata["metadata_file"] = str(metadata_file)

    return metadata
