"""Core data models for CAN bus analysis.

This module defines the fundamental data structures used throughout the
automotive CAN analysis subsystem.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = [
    "ByteAnalysis",
    "CANMessage",
    "CANMessageList",
    "ChecksumInfo",
    "CounterPattern",
    "DecodedSignal",
    "MessageAnalysis",
    "SignalDefinition",
]


@dataclass
class CANMessage:
    """A single CAN message.

    Attributes:
        arbitration_id: CAN arbitration ID (11-bit or 29-bit).
        timestamp: Message timestamp in seconds.
        data: Message data bytes (0-8 bytes for CAN 2.0, up to 64 for CAN-FD).
        is_extended: True for 29-bit extended ID.
        is_fd: True for CAN-FD frame.
        channel: CAN bus channel number (if available).
    """

    arbitration_id: int
    timestamp: float
    data: bytes
    is_extended: bool = False
    is_fd: bool = False
    channel: int = 0

    def __post_init__(self) -> None:
        """Validate message data."""
        if not isinstance(self.data, bytes):
            object.__setattr__(self, "data", bytes(self.data))  # type: ignore[unreachable]

    @property
    def dlc(self) -> int:
        """Data length code."""
        return len(self.data)

    def __repr__(self) -> str:
        """Human-readable representation."""
        id_str = (
            f"0x{self.arbitration_id:03X}"
            if not self.is_extended
            else f"0x{self.arbitration_id:08X}"
        )
        data_str = self.data.hex().upper()
        return f"CANMessage({id_str}, t={self.timestamp:.6f}s, data={data_str})"


@dataclass
class CANMessageList:
    """A collection of CAN messages.

    This class provides convenient operations on collections of CAN messages.

    Attributes:
        messages: List of CAN messages.
    """

    messages: list[CANMessage] = field(default_factory=list)

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self.messages)

    def __iter__(self) -> Iterator[CANMessage]:
        """Iterate over messages."""
        return iter(self.messages)

    def __getitem__(self, index: int | slice) -> CANMessage | list[CANMessage]:
        """Get message by index."""
        return self.messages[index]

    def append(self, message: CANMessage) -> None:
        """Add a message to the list."""
        self.messages.append(message)

    def filter_by_id(self, arbitration_id: int) -> CANMessageList:
        """Filter messages by arbitration ID.

        Args:
            arbitration_id: CAN ID to filter for.

        Returns:
            New CANMessageList containing only messages with the specified ID.
        """
        filtered = [msg for msg in self.messages if msg.arbitration_id == arbitration_id]
        return CANMessageList(messages=filtered)

    def unique_ids(self) -> set[int]:
        """Get set of unique arbitration IDs in this collection.

        Returns:
            Set of unique CAN IDs.
        """
        return {msg.arbitration_id for msg in self.messages}

    def time_range(self) -> tuple[float, float]:
        """Get time range of messages.

        Returns:
            Tuple of (first_timestamp, last_timestamp).
        """
        if not self.messages:
            return (0.0, 0.0)
        timestamps = [msg.timestamp for msg in self.messages]
        return (min(timestamps), max(timestamps))


@dataclass
class SignalDefinition:
    """Definition of a signal within a CAN message.

    Attributes:
        name: Signal name.
        start_bit: Starting bit position (0-63).
        length: Signal length in bits.
        byte_order: Byte order ('big_endian' or 'little_endian').
        value_type: Value type ('unsigned', 'signed', 'float').
        scale: Scaling factor (raw_value * scale).
        offset: Offset (scaled_value + offset).
        unit: Physical unit (e.g., 'rpm', 'km/h', '°C').
        min_value: Minimum valid value.
        max_value: Maximum valid value.
        comment: Description or notes.
    """

    name: str
    start_bit: int
    length: int
    byte_order: Literal["big_endian", "little_endian"] = "big_endian"
    value_type: Literal["unsigned", "signed", "float"] = "unsigned"
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""
    min_value: float | None = None
    max_value: float | None = None
    comment: str = ""

    @property
    def start_byte(self) -> int:
        """Get starting byte position."""
        return self.start_bit // 8

    def extract_raw(self, data: bytes) -> int:
        """Extract raw value from message data.

        Args:
            data: Message data bytes.

        Returns:
            Raw integer value.
        """
        # Convert bytes to bit array
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

        # Extract signal bits
        if self.byte_order == "big_endian":
            signal_bits = bits[self.start_bit : self.start_bit + self.length]
        else:
            # Little endian: reverse byte order
            byte_start = self.start_bit // 8
            bit_offset = self.start_bit % 8
            num_bytes = (self.length + bit_offset + 7) // 8
            bytes_range = data[byte_start : byte_start + num_bytes]
            reversed_bytes = bytes(reversed(bytes_range))
            bits = np.unpackbits(np.frombuffer(reversed_bytes, dtype=np.uint8))
            signal_bits = bits[bit_offset : bit_offset + self.length]

        # Convert bits to integer
        raw_value = int("".join(str(b) for b in signal_bits), 2)

        # Handle signed values
        if self.value_type == "signed" and raw_value >= (1 << (self.length - 1)):
            raw_value -= 1 << self.length

        return raw_value

    def decode(self, data: bytes) -> float:
        """Decode signal value from message data.

        Args:
            data: Message data bytes.

        Returns:
            Decoded physical value.
        """
        raw = self.extract_raw(data)
        return raw * self.scale + self.offset


@dataclass
class DecodedSignal:
    """A decoded signal value.

    Attributes:
        name: Signal name.
        value: Decoded physical value.
        unit: Physical unit.
        timestamp: Message timestamp.
        raw_value: Raw integer value before scaling.
        definition: Signal definition used for decoding.
    """

    name: str
    value: float
    unit: str
    timestamp: float
    raw_value: int | None = None
    definition: SignalDefinition | None = None

    def __repr__(self) -> str:
        """Human-readable representation."""
        if self.unit:
            return f"{self.name}: {self.value:.2f} {self.unit} @ {self.timestamp:.6f}s"
        return f"{self.name}: {self.value:.2f} @ {self.timestamp:.6f}s"


@dataclass
class ByteAnalysis:
    """Analysis results for a single byte position.

    Attributes:
        position: Byte position (0-7 for CAN 2.0).
        entropy: Shannon entropy (0.0 = constant, higher = more variable).
        min_value: Minimum observed value.
        max_value: Maximum observed value.
        mean: Mean value.
        std: Standard deviation.
        is_constant: True if byte never changes.
        unique_values: Number of unique values observed.
        most_common_value: Most frequently occurring value.
        change_rate: Fraction of messages where value changes from previous.
    """

    position: int
    entropy: float
    min_value: int
    max_value: int
    mean: float
    std: float
    is_constant: bool
    unique_values: int
    most_common_value: int
    change_rate: float


@dataclass
class CounterPattern:
    """Detected counter or sequence pattern.

    Attributes:
        byte_position: Byte position of counter.
        bit_range: Tuple of (start_bit, length) if sub-byte counter.
        increment: Typical increment value (usually 1).
        wraps_at: Value where counter wraps to 0.
        confidence: Confidence score (0.0-1.0).
        pattern_type: Type of pattern ('counter', 'sequence', 'toggle').
    """

    byte_position: int
    bit_range: tuple[int, int] | None = None
    increment: int = 1
    wraps_at: int = 255
    confidence: float = 0.0
    pattern_type: Literal["counter", "sequence", "toggle"] = "counter"


@dataclass
class ChecksumInfo:
    """Detected checksum or CRC information.

    Attributes:
        byte_position: Byte position of checksum.
        algorithm: Detected algorithm (e.g., 'CRC-8-SAE-J1850', 'XOR', 'SUM').
        polynomial: CRC polynomial (if CRC).
        covered_bytes: Byte positions covered by checksum.
        confidence: Confidence score (0.0-1.0).
        validation_rate: Fraction of messages with valid checksum.
    """

    byte_position: int
    algorithm: str
    polynomial: int | None = None
    covered_bytes: list[int] = field(default_factory=list)
    confidence: float = 0.0
    validation_rate: float = 0.0


@dataclass
class MessageAnalysis:
    """Complete analysis of a CAN message ID.

    Attributes:
        arbitration_id: CAN arbitration ID.
        message_count: Number of messages analyzed.
        frequency_hz: Average message frequency in Hz.
        period_ms: Average period in milliseconds.
        period_jitter_ms: Period jitter (std dev) in milliseconds.
        byte_analyses: Per-byte analysis results.
        detected_counters: Detected counter patterns.
        detected_checksum: Detected checksum information.
        suggested_signals: Suggested signal boundaries.
        correlations: Correlations with other message IDs.
    """

    arbitration_id: int
    message_count: int
    frequency_hz: float
    period_ms: float
    period_jitter_ms: float
    byte_analyses: list[ByteAnalysis]
    detected_counters: list[CounterPattern] = field(default_factory=list)
    detected_checksum: ChecksumInfo | None = None
    suggested_signals: list[dict[str, Any]] = field(default_factory=list)
    correlations: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string.
        """
        lines = [
            f"=== Message 0x{self.arbitration_id:03X} Analysis ===",
            f"Count: {self.message_count} messages",
            f"Frequency: {self.frequency_hz:.1f} Hz ({self.period_ms:.1f} ms period, jitter: {self.period_jitter_ms:.2f} ms)",
            "",
            "Byte Analysis:",
        ]

        for ba in self.byte_analyses:
            if ba.is_constant:
                lines.append(f"  Byte {ba.position}: CONSTANT (0x{ba.most_common_value:02X})")
            else:
                lines.append(
                    f"  Byte {ba.position}: entropy={ba.entropy:.2f}, "
                    f"range=[0x{ba.min_value:02X}-0x{ba.max_value:02X}], "
                    f"change_rate={ba.change_rate:.2f}"
                )

        if self.detected_counters:
            lines.append("")
            lines.append("Detected Counters:")
            for counter in self.detected_counters:
                lines.append(
                    f"  Byte {counter.byte_position}: {counter.pattern_type} "
                    f"(increment={counter.increment}, wraps at {counter.wraps_at}, "
                    f"confidence={counter.confidence:.2f})"
                )

        if self.detected_checksum:
            lines.append("")
            lines.append("Detected Checksum:")
            cs = self.detected_checksum
            lines.append(
                f"  Byte {cs.byte_position}: {cs.algorithm} "
                f"(validation_rate={cs.validation_rate:.2f}, confidence={cs.confidence:.2f})"
            )

        if self.suggested_signals:
            lines.append("")
            lines.append("Suggested Signal Boundaries:")
            for sig in self.suggested_signals:
                lines.append(f"  {sig}")

        return "\n".join(lines)
