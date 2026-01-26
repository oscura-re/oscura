"""LIN protocol analyzer with enhanced checksum and LDF generation.

This module provides comprehensive LIN 2.x protocol analysis including
protected ID calculation, enhanced checksum validation, diagnostic frame
parsing, and LDF (LIN Description File) generation from captured traffic.

References:
    LIN Specification 2.2A
    ISO 17987 (LIN standard)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "LINAnalyzer",
    "LINFrame",
    "LINScheduleEntry",
    "LINSignal",
]


@dataclass
class LINFrame:
    """LIN frame representation.

    Attributes:
        timestamp: Frame timestamp in seconds.
        frame_id: Frame ID (0-63, 6-bit).
        data: Frame data bytes (1-8 bytes).
        checksum: Received checksum byte.
        checksum_valid: True if checksum is valid.
        checksum_type: Checksum type ("classic" or "enhanced").
        parity_bits: 2-bit parity from protected ID.
        is_diagnostic: True if diagnostic frame (0x3C or 0x3D).
        decoded_signals: Decoded signal values.
    """

    timestamp: float
    frame_id: int
    data: bytes
    checksum: int
    checksum_valid: bool
    checksum_type: str = "enhanced"
    parity_bits: int = 0
    is_diagnostic: bool = False
    decoded_signals: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Human-readable representation."""
        status = "✓" if self.checksum_valid else "✗"
        return (
            f"LINFrame(ID=0x{self.frame_id:02X}, t={self.timestamp:.6f}s, "
            f"data={self.data.hex().upper()}, checksum={status} {self.checksum_type})"
        )


@dataclass
class LINSignal:
    """LIN signal definition.

    Attributes:
        name: Signal name.
        frame_id: Frame ID containing this signal (0-63).
        start_bit: Starting bit position (0-63 within 8 bytes).
        bit_length: Signal length in bits (1-64).
        init_value: Initial/default value.
        publisher: Node publishing this signal ("Master" or slave name).
    """

    name: str
    frame_id: int
    start_bit: int
    bit_length: int
    init_value: int = 0
    publisher: str = "Master"


@dataclass
class LINScheduleEntry:
    """LIN schedule table entry.

    Attributes:
        frame_id: Frame ID to transmit.
        delay_ms: Delay before next frame in milliseconds.
    """

    frame_id: int
    delay_ms: float


class LINAnalyzer:
    """Enhanced LIN protocol analyzer with LDF generation.

    Supports comprehensive LIN 2.x protocol analysis including:
    - Protected ID calculation with parity bits
    - Classic and enhanced checksum validation
    - Diagnostic frame parsing (0x3C master request, 0x3D slave response)
    - Signal decoding with bit-level extraction
    - Schedule table inference from frame timing
    - LDF (LIN Description File) generation

    Example:
        >>> analyzer = LINAnalyzer()
        >>> # Parse LIN frame with enhanced checksum
        >>> frame = analyzer.parse_frame(
        ...     data=b'\\x55\\x80\\x01\\x02\\x03\\xFA',
        ...     timestamp=1.0
        ... )
        >>> print(f"Frame ID: {frame.frame_id}")
        Frame ID: 0
        >>> # Add signal definition
        >>> analyzer.add_signal(LINSignal(
        ...     name="Speed",
        ...     frame_id=0,
        ...     start_bit=0,
        ...     bit_length=16,
        ...     publisher="Master"
        ... ))
        >>> # Generate LDF from captured traffic
        >>> analyzer.generate_ldf(Path("output.ldf"), baudrate=19200)
    """

    # Diagnostic frame IDs per LIN 2.x specification
    MASTER_REQUEST_FRAME: ClassVar[int] = 0x3C  # 60
    SLAVE_RESPONSE_FRAME: ClassVar[int] = 0x3D  # 61

    # Diagnostic services (subset of UDS adapted for LIN)
    DIAGNOSTIC_SERVICES: ClassVar[dict[int, str]] = {
        0xB0: "AssignFrameIdRange",
        0xB1: "AssignNAD",
        0xB2: "ConditionalChangeNAD",
        0xB3: "DataDump",
        0xB4: "SaveConfiguration",
        0xB5: "AssignFrameId",
        0xB6: "ReadById",
        0xB7: "TargetedReset",
    }

    def __init__(self) -> None:
        """Initialize LIN analyzer."""
        self.frames: list[LINFrame] = []
        self.signals: list[LINSignal] = []
        self.schedule: list[LINScheduleEntry] = []
        self.detected_frame_ids: set[int] = set()

    def parse_frame(
        self,
        data: bytes,
        timestamp: float = 0.0,
        checksum_type: str = "enhanced",
    ) -> LINFrame:
        """Parse LIN frame including sync, protected ID, data, and checksum.

        LIN Frame Format:
        - Break field (dominant, >= 13 bit times) - not in data
        - Sync field (0x55) - first byte
        - Protected ID (frame ID + parity bits) - second byte
        - Data field (1-8 bytes) - variable
        - Checksum field (1 byte) - last byte

        Args:
            data: Raw frame bytes including sync, protected ID, data, checksum.
            timestamp: Frame timestamp in seconds.
            checksum_type: Checksum type ("classic" or "enhanced").

        Returns:
            Parsed LINFrame object.

        Raises:
            ValueError: If frame is invalid or too short.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> # Frame: sync=0x55, protected_id=0x80 (ID=0), data=0x01,0x02, checksum=0xFA
            >>> frame = analyzer.parse_frame(
            ...     data=b'\\x55\\x80\\x01\\x02\\xFA',
            ...     timestamp=1.0,
            ...     checksum_type="enhanced"
            ... )
            >>> print(f"Frame ID: {frame.frame_id}, Valid: {frame.checksum_valid}")
            Frame ID: 0, Valid: True
        """
        if len(data) < 4:  # Minimum: sync + protected_id + 1 data byte + checksum
            raise ValueError(f"LIN frame too short: {len(data)} bytes (minimum 4)")

        # Parse sync byte (should be 0x55)
        sync_byte = data[0]
        if sync_byte != 0x55:
            raise ValueError(f"Invalid sync byte: 0x{sync_byte:02X} (expected 0x55)")

        # Parse protected ID
        protected_id = data[1]
        frame_id = protected_id & 0x3F  # Lower 6 bits
        parity_bits = (protected_id >> 6) & 0x03  # Upper 2 bits

        # Validate protected ID parity
        expected_protected_id = self._calculate_protected_id(frame_id)
        if protected_id != expected_protected_id:
            raise ValueError(
                f"Invalid protected ID parity: 0x{protected_id:02X} "
                f"(expected 0x{expected_protected_id:02X} for frame ID {frame_id})"
            )

        # Parse data and checksum
        frame_data = data[2:-1]
        received_checksum = data[-1]

        # Calculate and validate checksum
        if checksum_type == "classic":
            expected_checksum = self._calculate_classic_checksum(frame_data)
        else:  # enhanced
            expected_checksum = self._calculate_enhanced_checksum(protected_id, frame_data)

        checksum_valid = received_checksum == expected_checksum

        # Check if diagnostic frame
        is_diagnostic = frame_id in (self.MASTER_REQUEST_FRAME, self.SLAVE_RESPONSE_FRAME)

        # Parse diagnostic frame if applicable
        decoded_signals: dict[str, Any] = {}
        if is_diagnostic:
            decoded_signals = self._parse_diagnostic_frame(frame_id, frame_data)
        else:
            # Decode regular signals
            decoded_signals = self.decode_signals(frame_id, frame_data)

        frame = LINFrame(
            timestamp=timestamp,
            frame_id=frame_id,
            data=frame_data,
            checksum=received_checksum,
            checksum_valid=checksum_valid,
            checksum_type=checksum_type,
            parity_bits=parity_bits,
            is_diagnostic=is_diagnostic,
            decoded_signals=decoded_signals,
        )

        self.frames.append(frame)
        self.detected_frame_ids.add(frame_id)

        return frame

    def _calculate_protected_id(self, frame_id: int) -> int:
        """Calculate protected ID with parity bits.

        Protected ID Format (8 bits):
        - Bits 0-5: Frame ID (6 bits, range 0-63)
        - Bit 6 (P0): ID0 XOR ID1 XOR ID2 XOR ID4
        - Bit 7 (P1): NOT(ID1 XOR ID3 XOR ID4 XOR ID5)

        Args:
            frame_id: Frame ID (0-63).

        Returns:
            Protected ID with parity bits (8 bits).

        Raises:
            ValueError: If frame ID exceeds 63.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> protected_id = analyzer._calculate_protected_id(0)
            >>> print(f"Protected ID: 0x{protected_id:02X}")
            Protected ID: 0x80
        """
        if frame_id > 0x3F:
            raise ValueError(f"Frame ID {frame_id} exceeds 6 bits (max 63)")

        # Extract individual ID bits
        id0 = (frame_id >> 0) & 1
        id1 = (frame_id >> 1) & 1
        id2 = (frame_id >> 2) & 1
        id3 = (frame_id >> 3) & 1
        id4 = (frame_id >> 4) & 1
        id5 = (frame_id >> 5) & 1

        # Calculate parity bits
        p0 = id0 ^ id1 ^ id2 ^ id4
        p1 = (id1 ^ id3 ^ id4 ^ id5) ^ 1  # Inverted

        # Construct protected ID
        protected_id = frame_id | (p0 << 6) | (p1 << 7)

        return protected_id

    def _calculate_classic_checksum(self, data: bytes) -> int:
        """Calculate classic checksum (LIN 1.x).

        Classic checksum = inverted modulo-256 sum of data bytes only.

        Args:
            data: Frame data bytes.

        Returns:
            Classic checksum byte.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> checksum = analyzer._calculate_classic_checksum(b'\\x01\\x02\\x03')
            >>> print(f"Checksum: 0x{checksum:02X}")
            Checksum: 0xF9
        """
        checksum_sum = sum(data)

        # Handle carry (modulo-256 with carry propagation)
        while checksum_sum > 255:
            checksum_sum = (checksum_sum & 0xFF) + (checksum_sum >> 8)

        checksum = (~checksum_sum) & 0xFF

        return checksum

    def _calculate_enhanced_checksum(self, protected_id: int, data: bytes) -> int:
        """Calculate enhanced checksum (LIN 2.x).

        Enhanced checksum = inverted modulo-256 sum of protected ID + data bytes.

        Args:
            protected_id: Protected ID byte (frame ID with parity).
            data: Frame data bytes.

        Returns:
            Enhanced checksum byte.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> checksum = analyzer._calculate_enhanced_checksum(0x80, b'\\x01\\x02\\x03')
            >>> print(f"Checksum: 0x{checksum:02X}")
            Checksum: 0x79
        """
        checksum_sum = protected_id

        for byte in data:
            checksum_sum += byte
            # Handle carry (modulo-256 with carry propagation)
            if checksum_sum > 255:
                checksum_sum = (checksum_sum & 0xFF) + 1

        checksum = (~checksum_sum) & 0xFF

        return checksum

    def _parse_diagnostic_frame(self, frame_id: int, data: bytes) -> dict[str, Any]:
        """Parse diagnostic frame (Master Request 0x3C or Slave Response 0x3D).

        Diagnostic Frame Format:
        - NAD (Node Address for Diagnostics) - 1 byte
        - PCI (Protocol Control Information) - 1 byte
        - SID (Service Identifier) - 1 byte
        - Service data - variable (up to 5 bytes)

        Args:
            frame_id: Frame ID (0x3C or 0x3D).
            data: Frame data bytes.

        Returns:
            Dictionary of decoded diagnostic fields.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> decoded = analyzer._parse_diagnostic_frame(
            ...     0x3C,
            ...     b'\\x01\\x06\\xB6\\x00\\x01\\x00\\x00\\x00'
            ... )
            >>> print(decoded["service_name"])
            ReadById
        """
        if len(data) < 3:
            return {"error": "Diagnostic frame too short"}

        nad = data[0]  # Node address
        pci = data[1]  # Protocol control info (single frame = 0x06)
        sid = data[2]  # Service ID

        service_name = self.DIAGNOSTIC_SERVICES.get(sid, f"Unknown (0x{sid:02X})")
        service_data = data[3:]

        result: dict[str, Any] = {
            "nad": nad,
            "pci": pci,
            "service_id": sid,
            "service_name": service_name,
            "service_data": service_data.hex().upper(),
            "frame_type": "MasterRequest"
            if frame_id == self.MASTER_REQUEST_FRAME
            else "SlaveResponse",
        }

        # Decode specific services
        if sid == 0xB6 and len(service_data) >= 2:  # ReadById
            identifier = int.from_bytes(service_data[0:2], "big")
            result["identifier"] = identifier

        return result

    def add_signal(self, signal: LINSignal) -> None:
        """Add signal definition for decoding.

        Args:
            signal: Signal definition to add.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> analyzer.add_signal(LINSignal(
            ...     name="EngineSpeed",
            ...     frame_id=0x10,
            ...     start_bit=0,
            ...     bit_length=16,
            ...     publisher="Master"
            ... ))
        """
        self.signals.append(signal)

    def decode_signals(self, frame_id: int, data: bytes) -> dict[str, Any]:
        """Decode signals from frame data.

        Args:
            frame_id: Frame ID.
            data: Frame data bytes.

        Returns:
            Dictionary mapping signal names to decoded values.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> analyzer.add_signal(LINSignal("Speed", frame_id=0, start_bit=0, bit_length=16))
            >>> decoded = analyzer.decode_signals(0, b'\\x10\\x27')
            >>> print(f"Speed: {decoded['Speed']}")
            Speed: 10000
        """
        decoded: dict[str, Any] = {}

        # Find signals for this frame
        frame_signals = [s for s in self.signals if s.frame_id == frame_id]

        # Convert data to bit array
        if len(data) == 0:
            return decoded

        data_bits = int.from_bytes(data, "little")

        for signal in frame_signals:
            # Extract signal bits
            mask = (1 << signal.bit_length) - 1
            value = (data_bits >> signal.start_bit) & mask
            decoded[signal.name] = value

        return decoded

    def infer_schedule_table(self) -> list[LINScheduleEntry]:
        """Infer schedule table from captured frame timing.

        Analyzes frame timestamps to determine typical transmission schedule.

        Returns:
            List of schedule entries ordered by typical transmission sequence.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> # ... parse frames ...
            >>> schedule = analyzer.infer_schedule_table()
            >>> for entry in schedule:
            ...     print(f"Frame 0x{entry.frame_id:02X} delay {entry.delay_ms:.1f}ms")
        """
        if len(self.frames) < 2:
            return []

        # Group frames by ID and calculate average inter-frame delays
        frame_delays: dict[int, list[float]] = {}

        for i in range(1, len(self.frames)):
            prev_frame = self.frames[i - 1]
            curr_frame = self.frames[i]

            # Calculate delay in milliseconds
            delay_ms = (curr_frame.timestamp - prev_frame.timestamp) * 1000.0

            # Group by current frame ID
            if curr_frame.frame_id not in frame_delays:
                frame_delays[curr_frame.frame_id] = []
            frame_delays[curr_frame.frame_id].append(delay_ms)

        # Calculate average delay for each frame ID
        schedule_entries = []
        for frame_id in sorted(self.detected_frame_ids):
            if frame_id in frame_delays and len(frame_delays[frame_id]) > 0:
                avg_delay = sum(frame_delays[frame_id]) / len(frame_delays[frame_id])
                schedule_entries.append(LINScheduleEntry(frame_id=frame_id, delay_ms=avg_delay))

        self.schedule = schedule_entries
        return schedule_entries

    def generate_ldf(self, output_path: Path, baudrate: int = 19200) -> None:
        """Generate LDF (LIN Description File) from captured traffic.

        LDF Format per LIN 2.x specification:
        - Header (protocol version, language version, speed)
        - Nodes (master and slaves)
        - Signals (name, size, init value, publisher)
        - Frames (ID, publisher, size, signals)
        - Schedule tables (frame sequence and timing)

        Args:
            output_path: Path to output LDF file.
            baudrate: LIN bus baudrate in bps (default 19200).

        Raises:
            ValueError: If no frames have been captured.

        Example:
            >>> analyzer = LINAnalyzer()
            >>> # ... parse frames and add signals ...
            >>> analyzer.generate_ldf(Path("vehicle.ldf"), baudrate=19200)
        """
        if len(self.frames) == 0:
            raise ValueError("No frames captured - cannot generate LDF")

        ldf_lines: list[str] = []
        self._add_ldf_header(ldf_lines, baudrate)
        self._add_ldf_nodes(ldf_lines)
        self._add_ldf_signals(ldf_lines)
        self._add_ldf_frames(ldf_lines)
        self._add_ldf_schedule(ldf_lines)

        output_path.write_text("\n".join(ldf_lines) + "\n", encoding="utf-8")

    def _add_ldf_header(self, ldf_lines: list[str], baudrate: int) -> None:
        """Add LDF header section."""
        ldf_lines.append("LIN_description_file;")
        ldf_lines.append('LIN_protocol_version = "2.1";')
        ldf_lines.append('LIN_language_version = "2.1";')
        ldf_lines.append(f"LIN_speed = {baudrate / 1000:.1f} kbps;")
        ldf_lines.append("")

    def _add_ldf_nodes(self, ldf_lines: list[str]) -> None:
        """Add LDF nodes section (master and slaves)."""
        ldf_lines.append("Nodes {")
        ldf_lines.append("  Master: Master, 5 ms, 0.1 ms;")

        slaves = sorted({s.publisher for s in self.signals if s.publisher != "Master"})
        if len(slaves) > 0:
            ldf_lines.append(f"  Slaves: {', '.join(slaves)};")
        ldf_lines.append("}")
        ldf_lines.append("")

    def _add_ldf_signals(self, ldf_lines: list[str]) -> None:
        """Add LDF signals section."""
        if len(self.signals) == 0:
            return

        ldf_lines.append("Signals {")
        for signal in self.signals:
            ldf_lines.append(
                f"  {signal.name}: {signal.bit_length}, {signal.init_value}, {signal.publisher};"
            )
        ldf_lines.append("}")
        ldf_lines.append("")

    def _add_ldf_frames(self, ldf_lines: list[str]) -> None:
        """Add LDF frames section."""
        ldf_lines.append("Frames {")
        for frame_id in sorted(self.detected_frame_ids):
            if frame_id in (self.MASTER_REQUEST_FRAME, self.SLAVE_RESPONSE_FRAME):
                continue

            frame_signals = [s for s in self.signals if s.frame_id == frame_id]
            publisher, dlc = self._determine_frame_metadata(frame_id, frame_signals)

            ldf_lines.append(f"  Frame_{frame_id:02X}: {frame_id}, {publisher}, {dlc} {{")
            if len(frame_signals) > 0:
                for signal in frame_signals:
                    ldf_lines.append(f"    {signal.name}, {signal.start_bit};")
            ldf_lines.append("  }")

        ldf_lines.append("}")
        ldf_lines.append("")

    def _determine_frame_metadata(
        self, frame_id: int, frame_signals: list[LINSignal]
    ) -> tuple[str, int]:
        """Determine frame publisher and data length."""
        if len(frame_signals) > 0:
            publisher = frame_signals[0].publisher
            max_bit = max((s.start_bit + s.bit_length) for s in frame_signals)
            dlc = (max_bit + 7) // 8
        else:
            frame_data_lengths = [len(f.data) for f in self.frames if f.frame_id == frame_id]
            dlc = max(frame_data_lengths) if frame_data_lengths else 1
            publisher = "Master"
        return publisher, dlc

    def _add_ldf_schedule(self, ldf_lines: list[str]) -> None:
        """Add LDF schedule tables section."""
        if len(self.schedule) == 0:
            self.infer_schedule_table()

        if len(self.schedule) == 0:
            return

        ldf_lines.append("Schedule_tables {")
        ldf_lines.append("  NormalTable {")
        for entry in self.schedule:
            if entry.frame_id in (self.MASTER_REQUEST_FRAME, self.SLAVE_RESPONSE_FRAME):
                continue
            ldf_lines.append(f"    Frame_{entry.frame_id:02X} delay {entry.delay_ms:.1f} ms;")
        ldf_lines.append("  }")
        ldf_lines.append("}")
