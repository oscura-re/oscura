"""Protocol decoder base class (sigrok-inspired).

This module provides the base class for protocol decoders,
following a sigrok-inspired API for consistency with the
open-source protocol decoding ecosystem.


Example:
    >>> from oscura.analyzers.protocols.base import ProtocolDecoder
    >>> class UARTDecoder(ProtocolDecoder):
    ...     id = "uart"
    ...     name = "UART"
    ...     channels = [{"id": "rx", "name": "RX", "desc": "Receive data"}]
    ...     def decode(self, trace):
    ...         # Implementation
    ...         pass

References:
    sigrok Protocol Decoder API: https://sigrok.org/wiki/Protocol_decoder_API
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from oscura.core.types import DigitalTrace, ProtocolPacket

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class AnnotationLevel(IntEnum):
    """Annotation hierarchy levels.

    Protocol decoders use multiple annotation levels for different
    levels of detail, from raw bits to high-level interpretations.
    """

    BITS = 0  # Raw bit values
    BYTES = 1  # Byte values
    WORDS = 2  # Words/frames
    FIELDS = 3  # Named fields
    PACKETS = 4  # Complete packets
    MESSAGES = 5  # High-level messages


@dataclass
class Annotation:
    """Protocol annotation at a specific time range.

    Attributes:
        start_time: Start time in seconds.
        end_time: End time in seconds.
        level: Annotation level (bits, bytes, packets, etc.).
        text: Human-readable annotation text.
        data: Raw data associated with annotation.
        metadata: Additional annotation metadata.
    """

    start_time: float
    end_time: float
    level: AnnotationLevel
    text: str
    data: bytes | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelDef:
    """Channel definition for protocol decoder.

    Attributes:
        id: Channel identifier (e.g., "tx", "rx", "clk").
        name: Human-readable name.
        desc: Description of channel purpose.
        required: Whether channel is required.
    """

    id: str
    name: str
    desc: str = ""
    required: bool = True


@dataclass
class OptionDef:
    """Option definition for protocol decoder.

    Attributes:
        id: Option identifier.
        name: Human-readable name.
        desc: Description.
        default: Default value.
        values: List of valid values (if enumerated).
    """

    id: str
    name: str
    desc: str = ""
    default: Any = None
    values: list[Any] | None = None


class DecoderState:
    """Base class for decoder state machines.

    Protocol decoders can subclass this to track their internal state
    during frame/packet decoding.
    """

    def __init__(self) -> None:
        """Initialize decoder state."""
        self.reset()

    def reset(self) -> None:
        """Reset state to initial values."""


class ProtocolDecoder(ABC):
    """Base class for protocol decoders.

    Provides sigrok-inspired API for implementing protocol decoders
    that convert digital traces to decoded protocol packets.

    Class Attributes:
        api_version: Protocol decoder API version.
        id: Unique decoder identifier.
        name: Human-readable decoder name.
        longname: Full name with description.
        desc: Short description.
        license: License identifier.
        inputs: Required input types (e.g., ["logic"]).
        outputs: Output types produced.
        channels: Required channel definitions.
        optional_channels: Optional channel definitions.
        options: Configurable options.
        annotations: Annotation type definitions.

    Example:
        >>> class SPIDecoder(ProtocolDecoder):
        ...     id = "spi"
        ...     name = "SPI"
        ...     channels = [
        ...         ChannelDef("clk", "CLK", "Clock"),
        ...         ChannelDef("mosi", "MOSI", "Master Out Slave In"),
        ...         ChannelDef("miso", "MISO", "Master In Slave Out"),
        ...     ]
        ...     optional_channels = [
        ...         ChannelDef("cs", "CS#", "Chip Select", required=False),
        ...     ]
        ...     options = [
        ...         OptionDef("cpol", "Clock Polarity", default=0, values=[0, 1]),
        ...         OptionDef("cpha", "Clock Phase", default=0, values=[0, 1]),
        ...     ]
    """

    # API version
    api_version: int = 3

    # Decoder identification
    id: str = "unknown"
    name: str = "Unknown"
    longname: str = ""
    desc: str = ""
    license: str = "MIT"

    # Input/output types
    inputs: list[str] = ["logic"]
    outputs: list[str] = ["packets"]

    # Channel definitions
    channels: list[ChannelDef] = []
    optional_channels: list[ChannelDef] = []

    # Options
    options: list[OptionDef] = []

    # Annotation definitions (override in subclass)
    annotations: list[tuple[str, str]] = []

    def __init__(self, **options: Any) -> None:
        """Initialize decoder with options.

        Args:
            **options: Decoder-specific options.

        Raises:
            ValueError: If unknown option is provided
        """
        self._options: dict[str, Any] = {}
        self._annotations: list[Annotation] = []
        self._packets: list[ProtocolPacket] = []
        self._state = DecoderState()

        # Set default options
        for opt in self.options:
            self._options[opt.id] = opt.default

        # Override with provided options
        for key, value in options.items():
            if any(opt.id == key for opt in self.options):
                self._options[key] = value
            else:
                raise ValueError(f"Unknown option: {key}")

    def get_option(self, name: str) -> Any:
        """Get option value.

        Args:
            name: Option name.

        Returns:
            Option value.
        """
        return self._options.get(name)

    def set_option(self, name: str, value: Any) -> None:
        """Set option value.

        Args:
            name: Option name.
            value: New value.
        """
        self._options[name] = value

    def reset(self) -> None:
        """Reset decoder state.

        Clears all accumulated annotations and packets, and resets
        the internal state machine to initial state.
        """
        self._annotations.clear()
        self._packets.clear()
        self._state.reset()

    def put_annotation(
        self,
        start_time: float,
        end_time: float,
        level: AnnotationLevel,
        text: str,
        data: bytes | None = None,
        **metadata: Any,
    ) -> None:
        """Add an annotation.

        Args:
            start_time: Start time in seconds.
            end_time: End time in seconds.
            level: Annotation level.
            text: Annotation text.
            data: Associated binary data.
            **metadata: Additional metadata.
        """
        self._annotations.append(
            Annotation(
                start_time=start_time,
                end_time=end_time,
                level=level,
                text=text,
                data=data,
                metadata=metadata,
            )
        )

    def put_packet(
        self,
        timestamp: float,
        data: bytes,
        annotations: dict[str, Any] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        """Add a decoded packet.

        Args:
            timestamp: Packet start time.
            data: Decoded data bytes.
            annotations: Packet annotations.
            errors: Detected errors.
        """
        self._packets.append(
            ProtocolPacket(
                timestamp=timestamp,
                protocol=self.id,
                data=data,
                annotations=annotations or {},
                errors=errors or [],
            )
        )

    @abstractmethod
    def decode(
        self,
        trace: DigitalTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode a digital trace.

        This is the main entry point for decoding. Implementations should
        yield ProtocolPacket objects as they are decoded.

        Args:
            trace: Primary input trace.
            **channels: Additional channel data by name.

        Yields:
            Decoded protocol packets.

        Example:
            >>> decoder = UARTDecoder(baudrate=115200)
            >>> for packet in decoder.decode(trace):
            ...     print(f"Data: {packet.data.hex()}")
        """

    def get_annotations(
        self,
        *,
        level: AnnotationLevel | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[Annotation]:
        """Get accumulated annotations.

        Args:
            level: Filter by annotation level.
            start_time: Filter by start time (inclusive).
            end_time: Filter by end time (inclusive).

        Returns:
            List of matching annotations.
        """
        result = self._annotations

        if level is not None:
            result = [a for a in result if a.level == level]

        if start_time is not None:
            result = [a for a in result if a.end_time >= start_time]

        if end_time is not None:
            result = [a for a in result if a.start_time <= end_time]

        return result

    def get_packets(self) -> list[ProtocolPacket]:
        """Get all decoded packets.

        Returns:
            List of decoded packets.
        """
        return list(self._packets)

    @classmethod
    def get_channel_ids(cls, include_optional: bool = False) -> list[str]:
        """Get list of channel IDs.

        Args:
            include_optional: Include optional channels.

        Returns:
            List of channel ID strings.
        """
        ids = [ch.id for ch in cls.channels]
        if include_optional:
            ids.extend(ch.id for ch in cls.optional_channels)
        return ids

    @classmethod
    def get_option_ids(cls) -> list[str]:
        """Get list of option IDs.

        Returns:
            List of option ID strings.
        """
        return [opt.id for opt in cls.options]


class SyncDecoder(ProtocolDecoder):
    """Base class for synchronous protocol decoders.

    Synchronous protocols use a clock signal for timing. This base class
    provides helpers for clock edge detection and data sampling.
    """

    def sample_on_edge(
        self,
        clock: NDArray[np.bool_],
        data: NDArray[np.bool_],
        edge: str = "rising",
    ) -> NDArray[np.bool_]:
        """Sample data on clock edges.

        Args:
            clock: Clock signal.
            data: Data signal.
            edge: "rising" or "falling".

        Returns:
            Data values at clock edges.
        """
        if edge == "rising":
            edges = np.where(~clock[:-1] & clock[1:])[0]
        else:
            edges = np.where(clock[:-1] & ~clock[1:])[0]

        # Sample data after edge (shifted by 1)
        sample_indices = edges + 1
        sample_indices = sample_indices[sample_indices < len(data)]

        result: NDArray[np.bool_] = data[sample_indices]
        return result


class AsyncDecoder(ProtocolDecoder):
    """Base class for asynchronous protocol decoders.

    Asynchronous protocols (like UART) use timing-based sampling without
    a separate clock signal. This base class provides helpers for
    bit-timing and symbol detection.
    """

    def __init__(self, baudrate: int = 9600, **options: Any) -> None:
        """Initialize async decoder.

        Args:
            baudrate: Bit rate in bps.
            **options: Additional options.
        """
        super().__init__(**options)
        self._baudrate = baudrate

    @property
    def baudrate(self) -> int:
        """Get baud rate."""
        return self._baudrate

    @baudrate.setter
    def baudrate(self, value: int) -> None:
        """Set baud rate."""
        self._baudrate = value

    def bit_time(self, sample_rate: float) -> float:
        """Get bit time in samples.

        Args:
            sample_rate: Sample rate in Hz.

        Returns:
            Number of samples per bit.
        """
        return sample_rate / self._baudrate

    def find_start_bit(
        self,
        data: NDArray[np.bool_],
        start_idx: int = 0,
        idle_high: bool = True,
    ) -> int | None:
        """Find start bit transition.

        Args:
            data: Digital signal.
            start_idx: Start search index.
            idle_high: True if idle is high (standard UART).

        Returns:
            Index of start bit, or None if not found.
        """
        search_region = data[start_idx:]

        if idle_high:
            # Look for falling edge (high to low)
            transitions = np.where(search_region[:-1] & ~search_region[1:])[0]
        else:
            # Look for rising edge (low to high)
            transitions = np.where(~search_region[:-1] & search_region[1:])[0]

        if len(transitions) == 0:
            return None

        return int(start_idx + transitions[0])


__all__ = [
    "Annotation",
    "AnnotationLevel",
    "AsyncDecoder",
    "ChannelDef",
    "DecoderState",
    "OptionDef",
    "ProtocolDecoder",
    "SyncDecoder",
]
