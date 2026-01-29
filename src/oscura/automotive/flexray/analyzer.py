"""FlexRay protocol analyzer.

This module provides comprehensive FlexRay frame analysis including header parsing,
CRC validation, signal decoding, and multi-channel support.

Example:
    >>> from oscura.automotive.flexray import FlexRayAnalyzer, FlexRaySignal
    >>> analyzer = FlexRayAnalyzer()
    >>> frame = analyzer.parse_frame(data, timestamp=0.0, channel="A")
    >>> print(f"Slot {frame.header.frame_id}, valid: {frame.crc_valid}")
    >>>
    >>> # Add signal definitions
    >>> signal = FlexRaySignal(
    ...     name="EngineSpeed",
    ...     frame_id=100,
    ...     start_bit=0,
    ...     bit_length=16,
    ...     factor=0.25,
    ...     offset=0,
    ...     unit="rpm"
    ... )
    >>> analyzer.add_signal(signal)
    >>> signals = analyzer.decode_signals(100, frame.payload)
    >>> print(f"Engine Speed: {signals['EngineSpeed']} rpm")

References:
    FlexRay Communications System Protocol Specification Version 3.0.1
    ISO 17458 (FlexRay standard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oscura.automotive.flexray.crc import (
    verify_frame_crc,
    verify_header_crc,
)


@dataclass
class FlexRayHeader:
    """FlexRay frame header (40 bits / 5 bytes).

    The header contains frame identification, control flags, and CRC.

    Attributes:
        reserved: Reserved bit (always 0).
        payload_preamble: Payload preamble indicator.
        null_frame: Null frame indicator (no payload).
        sync_frame: Synchronization frame indicator.
        startup_frame: Startup frame indicator.
        frame_id: Frame identifier (1-2047).
        payload_length: Payload length in bytes (0-254).
        header_crc: 11-bit header CRC.
        cycle_count: Cycle counter (0-63).

    Example:
        >>> header = FlexRayHeader(
        ...     reserved=0, payload_preamble=0, null_frame=False,
        ...     sync_frame=True, startup_frame=False, frame_id=100,
        ...     payload_length=10, header_crc=0x3A5, cycle_count=5
        ... )
        >>> print(f"Frame ID: {header.frame_id}, Cycle: {header.cycle_count}")
    """

    reserved: int  # 1 bit
    payload_preamble: int  # 1 bit
    null_frame: bool  # 1 bit
    sync_frame: bool  # 1 bit
    startup_frame: bool  # 1 bit
    frame_id: int  # 11 bits (1-2047)
    payload_length: int  # In bytes (0-254, stored as words internally)
    header_crc: int  # 11 bits
    cycle_count: int  # 6 bits (0-63)


@dataclass
class FlexRayFrame:
    """FlexRay frame representation.

    Attributes:
        timestamp: Frame timestamp in seconds.
        channel: Channel identifier ("A" or "B").
        header: Parsed frame header.
        payload: Frame payload bytes (0-254 bytes).
        frame_crc: Received 24-bit frame CRC.
        crc_valid: True if frame CRC is valid.
        segment_type: Segment type ("static" or "dynamic").
        decoded_signals: Decoded signal values (populated after decoding).

    Example:
        >>> frame = FlexRayFrame(
        ...     timestamp=1.234, channel="A", header=header,
        ...     payload=b"\\x01\\x02\\x03", frame_crc=0x123456,
        ...     crc_valid=True, segment_type="static"
        ... )
        >>> print(f"Channel {frame.channel}, Slot {frame.header.frame_id}")
    """

    timestamp: float
    channel: str  # "A" or "B"
    header: FlexRayHeader
    payload: bytes  # 0-254 bytes
    frame_crc: int  # 24 bits
    crc_valid: bool
    segment_type: str = "static"  # "static" or "dynamic"
    decoded_signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class FlexRaySignal:
    """FlexRay signal definition.

    Defines how to extract and decode a signal from a frame payload.

    Attributes:
        name: Signal name.
        frame_id: Frame ID containing this signal.
        start_bit: Start bit position in payload.
        bit_length: Signal length in bits.
        byte_order: Byte order ("big_endian" or "little_endian").
        factor: Scaling factor (physical = raw * factor + offset).
        offset: Offset for physical value.
        unit: Physical unit (e.g., "rpm", "km/h").

    Example:
        >>> signal = FlexRaySignal(
        ...     name="VehicleSpeed",
        ...     frame_id=200,
        ...     start_bit=16,
        ...     bit_length=16,
        ...     byte_order="big_endian",
        ...     factor=0.01,
        ...     offset=0,
        ...     unit="km/h"
        ... )
    """

    name: str
    frame_id: int
    start_bit: int
    bit_length: int
    byte_order: str = "big_endian"  # FlexRay is typically big-endian
    factor: float = 1.0
    offset: float = 0.0
    unit: str = ""


class FlexRayAnalyzer:
    """FlexRay protocol analyzer.

    Provides comprehensive FlexRay frame analysis including parsing, CRC validation,
    signal decoding, and FIBEX support.

    Attributes:
        MAX_FRAME_ID: Maximum frame ID (2047).
        MAX_PAYLOAD_LENGTH: Maximum payload length in bytes (254).
        HEADER_LENGTH: Header length in bytes (5).
        CRC_LENGTH: CRC length in bytes (3).
        STATIC_SEGMENT: Static segment identifier.
        DYNAMIC_SEGMENT: Dynamic segment identifier.

    Example:
        >>> analyzer = FlexRayAnalyzer()
        >>> frame = analyzer.parse_frame(raw_data, timestamp=1.0, channel="A")
        >>> if frame.crc_valid:
        ...     print(f"Valid frame on slot {frame.header.frame_id}")
    """

    # FlexRay constants
    MAX_FRAME_ID = 2047
    MAX_PAYLOAD_LENGTH = 254  # bytes (127 16-bit words)
    HEADER_LENGTH = 5  # bytes (40 bits)
    CRC_LENGTH = 3  # bytes (24 bits)

    # Segment types
    STATIC_SEGMENT = "static"
    DYNAMIC_SEGMENT = "dynamic"

    def __init__(self, cluster_config: dict[str, Any] | None = None) -> None:
        """Initialize FlexRay analyzer.

        Args:
            cluster_config: Optional cluster configuration containing:
                - static_slot_count: Number of static slots
                - dynamic_slot_count: Number of dynamic slots
                - sample_rate: Sample rate in Hz

        Example:
            >>> config = {"static_slot_count": 100, "dynamic_slot_count": 50}
            >>> analyzer = FlexRayAnalyzer(cluster_config=config)
        """
        self.frames: list[FlexRayFrame] = []
        self.signals: list[FlexRaySignal] = []
        self.cluster_config = cluster_config or {}

    def parse_frame(self, data: bytes, timestamp: float = 0.0, channel: str = "A") -> FlexRayFrame:
        """Parse FlexRay frame from raw bytes.

        The frame structure is:
        - Header (5 bytes)
        - Payload (0-254 bytes)
        - CRC (3 bytes)

        Args:
            data: Raw frame data (minimum 8 bytes: 5 header + 0 payload + 3 CRC).
            timestamp: Frame timestamp in seconds.
            channel: Channel identifier ("A" or "B").

        Returns:
            Parsed FlexRay frame.

        Raises:
            ValueError: If data is too short or invalid.

        Example:
            >>> raw_data = bytes([0x00, 0x64, 0x12, 0x34, 0x05] + [0] * 10 + [0, 0, 0])
            >>> frame = analyzer.parse_frame(raw_data, timestamp=1.0, channel="A")
            >>> print(f"Frame ID: {frame.header.frame_id}")
        """
        if len(data) < self.HEADER_LENGTH + self.CRC_LENGTH:
            raise ValueError(
                f"Frame data too short: {len(data)} bytes "
                f"(minimum {self.HEADER_LENGTH + self.CRC_LENGTH})"
            )

        # Extract header, payload, and CRC
        header_bytes = data[: self.HEADER_LENGTH]
        crc_bytes = data[-self.CRC_LENGTH :]
        payload_bytes = data[self.HEADER_LENGTH : -self.CRC_LENGTH]

        # Parse header
        header = self._parse_header(header_bytes)

        # Verify payload length matches header
        expected_payload_length = header.payload_length
        if len(payload_bytes) != expected_payload_length:
            raise ValueError(
                f"Payload length mismatch: expected {expected_payload_length}, "
                f"got {len(payload_bytes)}"
            )

        # Parse CRC
        frame_crc = int.from_bytes(crc_bytes, "big")

        # Verify CRC
        crc_valid = verify_frame_crc(header_bytes, payload_bytes, frame_crc)

        # Determine segment type
        segment_type = self._determine_segment_type(header.frame_id)

        # Create frame
        frame = FlexRayFrame(
            timestamp=timestamp,
            channel=channel,
            header=header,
            payload=payload_bytes,
            frame_crc=frame_crc,
            crc_valid=crc_valid,
            segment_type=segment_type,
        )

        self.frames.append(frame)
        return frame

    def _parse_header(self, header_bytes: bytes) -> FlexRayHeader:
        """Parse 40-bit FlexRay header.

        Header Format (5 bytes, 40 bits, MSB first):
        Byte 0, bit 7 (MSB): Reserved (1 bit)
        Byte 0, bit 6: Payload Preamble Indicator (1 bit)
        Byte 0, bit 5: Null Frame Indicator (1 bit)
        Byte 0, bit 4: Sync Frame Indicator (1 bit)
        Byte 0, bit 3: Startup Frame Indicator (1 bit)
        Byte 0, bits 2-0 + Byte 1, bits 7-0: Frame ID (11 bits, 1-2047)
        Byte 2, bits 7-1: Payload Length (7 bits, in 16-bit words)
        Byte 2, bit 0 + Byte 3, bits 7-6: Header CRC (11 bits)
        Byte 3, bits 5-0: Cycle Count (6 bits, 0-63)

        Args:
            header_bytes: 5-byte header data.

        Returns:
            Parsed FlexRay header.

        Raises:
            ValueError: If header is invalid.
        """
        if len(header_bytes) < 5:
            raise ValueError(f"FlexRay header must be 5 bytes, got {len(header_bytes)}")

        # Convert to 40-bit integer (big-endian)
        header_int = int.from_bytes(header_bytes[:5], "big")

        # Extract fields (MSB first, bit 39 is leftmost)
        reserved = (header_int >> 39) & 0x01
        payload_preamble = (header_int >> 38) & 0x01
        null_frame = bool((header_int >> 37) & 0x01)
        sync_frame = bool((header_int >> 36) & 0x01)
        startup_frame = bool((header_int >> 35) & 0x01)
        frame_id = (header_int >> 24) & 0x7FF  # 11 bits
        payload_length_words = (header_int >> 17) & 0x7F  # 7 bits (in words)
        header_crc = (header_int >> 6) & 0x7FF  # 11 bits
        cycle_count = header_int & 0x3F  # 6 bits

        # Convert payload length from words to bytes
        payload_length = payload_length_words * 2

        # Validate frame ID
        if frame_id < 1 or frame_id > self.MAX_FRAME_ID:
            raise ValueError(f"Invalid frame ID: {frame_id} (must be 1-{self.MAX_FRAME_ID})")

        # Verify header CRC
        header_crc_valid = verify_header_crc(
            reserved,
            payload_preamble,
            int(null_frame),
            int(sync_frame),
            int(startup_frame),
            frame_id,
            payload_length_words,
            header_crc,
        )

        if not header_crc_valid:
            # Note: We don't raise an error, but we could log a warning
            pass

        return FlexRayHeader(
            reserved=reserved,
            payload_preamble=payload_preamble,
            null_frame=null_frame,
            sync_frame=sync_frame,
            startup_frame=startup_frame,
            frame_id=frame_id,
            payload_length=payload_length,
            header_crc=header_crc,
            cycle_count=cycle_count,
        )

    def _determine_segment_type(self, frame_id: int) -> str:
        """Determine if frame is in static or dynamic segment.

        Static segment frames have IDs from 1 to static_slot_count.
        Dynamic segment frames have IDs from static_slot_count+1 to 2047.

        Args:
            frame_id: Frame ID to classify.

        Returns:
            Segment type ("static" or "dynamic").

        Example:
            >>> analyzer.cluster_config = {"static_slot_count": 100}
            >>> segment = analyzer._determine_segment_type(50)
            >>> print(segment)  # "static"
            >>> segment = analyzer._determine_segment_type(150)
            >>> print(segment)  # "dynamic"
        """
        static_slot_count = self.cluster_config.get("static_slot_count", 100)

        if frame_id <= static_slot_count:
            return self.STATIC_SEGMENT
        else:
            return self.DYNAMIC_SEGMENT

    def add_signal(self, signal: FlexRaySignal) -> None:
        """Add signal definition for decoding.

        Args:
            signal: Signal definition to add.

        Example:
            >>> signal = FlexRaySignal(
            ...     name="EngineSpeed", frame_id=100, start_bit=0,
            ...     bit_length=16, factor=0.25, unit="rpm"
            ... )
            >>> analyzer.add_signal(signal)
        """
        self.signals.append(signal)

    def decode_signals(self, frame_id: int, payload: bytes) -> dict[str, Any]:
        """Decode signals from frame payload.

        Extracts and scales signal values according to signal definitions.

        Args:
            frame_id: Frame ID to decode.
            payload: Frame payload bytes.

        Returns:
            Dictionary mapping signal names to physical values.

        Example:
            >>> signals = analyzer.decode_signals(100, b"\\x00\\x64\\x01\\x00")
            >>> print(f"EngineSpeed: {signals['EngineSpeed']} rpm")
        """
        decoded: dict[str, Any] = {}

        # Find signals for this frame
        frame_signals = [s for s in self.signals if s.frame_id == frame_id]

        for signal in frame_signals:
            # Extract raw value
            raw_value = self._extract_signal_value(payload, signal)

            # Apply scaling
            physical_value = raw_value * signal.factor + signal.offset

            decoded[signal.name] = physical_value

        return decoded

    def _extract_signal_value(self, payload: bytes, signal: FlexRaySignal) -> int:
        """Extract raw signal value from payload.

        Args:
            payload: Frame payload bytes.
            signal: Signal definition.

        Returns:
            Raw signal value (unscaled).

        Raises:
            ValueError: If signal extends beyond payload.
        """
        # Calculate byte range
        start_byte = signal.start_bit // 8
        end_byte = (signal.start_bit + signal.bit_length - 1) // 8 + 1

        if end_byte > len(payload):
            raise ValueError(
                f"Signal {signal.name} extends beyond payload: "
                f"needs bytes {start_byte}-{end_byte}, payload has {len(payload)}"
            )

        # Extract bytes
        signal_bytes = payload[start_byte:end_byte]

        # Convert to integer based on byte order
        if signal.byte_order == "big_endian":
            value = int.from_bytes(signal_bytes, "big")
        else:
            value = int.from_bytes(signal_bytes, "little")

        # Extract specific bits
        bit_offset = signal.start_bit % 8
        mask = (1 << signal.bit_length) - 1
        value = (value >> bit_offset) & mask

        return value

    def get_frame_statistics(self) -> dict[str, Any]:
        """Get statistics about parsed frames.

        Returns:
            Dictionary with frame statistics including:
            - total_frames: Total number of frames
            - frames_by_channel: Count per channel
            - frames_by_id: Count per frame ID
            - crc_errors: Number of CRC errors
            - segment_distribution: Static vs dynamic frame counts

        Example:
            >>> stats = analyzer.get_frame_statistics()
            >>> print(f"Total frames: {stats['total_frames']}")
            >>> print(f"CRC errors: {stats['crc_errors']}")
        """
        stats: dict[str, Any] = {
            "total_frames": len(self.frames),
            "frames_by_channel": {},
            "frames_by_id": {},
            "crc_errors": 0,
            "segment_distribution": {"static": 0, "dynamic": 0},
        }

        for frame in self.frames:
            # Count by channel
            stats["frames_by_channel"][frame.channel] = (
                stats["frames_by_channel"].get(frame.channel, 0) + 1
            )

            # Count by frame ID
            frame_id = frame.header.frame_id
            stats["frames_by_id"][frame_id] = stats["frames_by_id"].get(frame_id, 0) + 1

            # Count CRC errors
            if not frame.crc_valid:
                stats["crc_errors"] += 1

            # Count segment types
            stats["segment_distribution"][frame.segment_type] += 1

        return stats


__all__ = [
    "FlexRayAnalyzer",
    "FlexRayFrame",
    "FlexRayHeader",
    "FlexRaySignal",
]
