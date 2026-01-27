"""J1939 protocol analyzer with transport protocol and SPN support.

This module provides comprehensive J1939 (SAE J1939) protocol analysis for heavy-duty
vehicles and industrial equipment, including:
- J1939 CAN identifier decoding (29-bit extended IDs)
- PGN (Parameter Group Number) extraction
- Priority and address parsing
- Transport protocol (TP.CM, TP.DT, BAM) multi-packet reassembly
- SPN (Suspect Parameter Number) decoding
- Message and topology export

Example:
    >>> from oscura.automotive.j1939.analyzer import J1939Analyzer
    >>> analyzer = J1939Analyzer()
    >>> msg = analyzer.parse_message(0x18FEF100, b'\\x00\\x01\\x02\\x03\\x04\\x05\\x06\\x07', 1.0)
    >>> print(msg.identifier.pgn)
    65265
    >>> print(msg.pgn_name)
    Cruise Control/Vehicle Speed

References:
    SAE J1939/21 - Data Link Layer
    SAE J1939/71 - Vehicle Application Layer
    SAE J1939/73 - Application Layer - Diagnostics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "J1939SPN",
    "J1939Analyzer",
    "J1939Identifier",
    "J1939Message",
]


@dataclass
class J1939Identifier:
    """J1939 CAN identifier breakdown.

    Attributes:
        priority: Message priority (0-7, lower is higher priority).
        reserved: Reserved bit (typically 0).
        data_page: Data page bit (0 or 1).
        pdu_format: PDU Format (8 bits).
        pdu_specific: PDU Specific (8 bits, destination or group extension).
        source_address: Source address (8 bits).
        pgn: Calculated Parameter Group Number.
    """

    priority: int
    reserved: int
    data_page: int
    pdu_format: int
    pdu_specific: int
    source_address: int
    pgn: int


@dataclass
class J1939Message:
    """J1939 message representation.

    Attributes:
        timestamp: Message timestamp in seconds.
        can_id: 29-bit extended CAN identifier.
        identifier: Decoded J1939 identifier components.
        data: Message data payload (up to 8 bytes for single frame).
        pgn_name: Human-readable PGN name (if known).
        decoded_spns: Decoded Suspect Parameter Numbers.
        is_transport_protocol: True if transport protocol message (TP.CM/TP.DT/BAM).
        transport_info: Transport protocol metadata.
    """

    timestamp: float
    can_id: int
    identifier: J1939Identifier
    data: bytes
    pgn_name: str | None = None
    decoded_spns: dict[str, Any] = field(default_factory=dict)
    is_transport_protocol: bool = False
    transport_info: dict[str, Any] | None = None


@dataclass
class J1939SPN:
    """Suspect Parameter Number definition.

    Attributes:
        spn: SPN number.
        name: Parameter name.
        start_bit: Bit position in data (0-based).
        bit_length: Number of bits.
        resolution: Scaling factor.
        offset: Offset to add after scaling.
        unit: Engineering unit.
        data_range: Valid data range (min, max).
    """

    spn: int
    name: str
    start_bit: int
    bit_length: int
    resolution: float = 1.0
    offset: float = 0.0
    unit: str = ""
    data_range: tuple[float, float] = (0.0, 0.0)


class J1939Analyzer:
    """J1939 protocol analyzer for heavy-duty vehicles.

    Supports comprehensive J1939 protocol analysis including:
    - 29-bit extended CAN ID decoding
    - PGN extraction and naming
    - Transport protocol (TP.CM, TP.DT, BAM) multi-packet reassembly
    - SPN decoding with user-defined parameters
    - Message history and export

    Example:
        >>> analyzer = J1939Analyzer()
        >>> # Parse a single-frame message
        >>> msg = analyzer.parse_message(0x0CF00400, b'\\xff\\xff\\xff\\xff\\xff\\xff\\xff\\xff')
        >>> print(f"PGN: {msg.identifier.pgn}, Priority: {msg.identifier.priority}")
        PGN: 61444, Priority: 3
        >>> # Add SPN definition
        >>> spn = J1939SPN(
        ...     spn=190,
        ...     name="Engine Speed",
        ...     start_bit=24,
        ...     bit_length=16,
        ...     resolution=0.125,
        ...     unit="rpm"
        ... )
        >>> analyzer.add_spn_definition(61444, spn)
    """

    # Well-known PGNs per SAE J1939/71
    PGNS: ClassVar[dict[int, str]] = {
        0: "Torque/Speed Control 1",
        59392: "Acknowledgment",
        59904: "Request",
        60160: "Transport Protocol - Connection Management (TP.CM)",
        60416: "Transport Protocol - Data Transfer (TP.DT)",
        60928: "Address Claimed",
        61440: "Electronic Retarder Controller 1",
        61441: "Electronic Brake Controller 1",
        61442: "Electronic Transmission Controller 1",
        61443: "Electronic Engine Controller 2",
        61444: "Electronic Engine Controller 1",
        65226: "Active Diagnostic Trouble Codes",
        65227: "Previously Active Diagnostic Trouble Codes",
        65265: "Cruise Control/Vehicle Speed",
    }

    # Transport Protocol Connection Management control bytes
    TP_CM_RTS: ClassVar[int] = 16  # Request To Send
    TP_CM_CTS: ClassVar[int] = 17  # Clear To Send
    TP_CM_EOM_ACK: ClassVar[int] = 19  # End of Message Acknowledgment
    TP_CM_BAM: ClassVar[int] = 32  # Broadcast Announce Message
    TP_CM_ABORT: ClassVar[int] = 255  # Connection Abort

    def __init__(self) -> None:
        """Initialize J1939 analyzer."""
        self.messages: list[J1939Message] = []
        # Transport sessions: (source_addr, dest_addr) -> session dict
        self.transport_sessions: dict[tuple[int, int], dict[str, Any]] = {}
        # SPN definitions: PGN -> [SPNs]
        self.spn_definitions: dict[int, list[J1939SPN]] = {}

    def parse_message(self, can_id: int, data: bytes, timestamp: float = 0.0) -> J1939Message:
        """Parse J1939 message from CAN frame.

        Args:
            can_id: 29-bit extended CAN identifier.
            data: Message data payload (up to 8 bytes).
            timestamp: Message timestamp in seconds.

        Returns:
            Parsed J1939 message.

        Raises:
            ValueError: If CAN ID is invalid or data is too long.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> msg = analyzer.parse_message(0x18FEF100, b'\\x00\\x01\\x02\\x03')
            >>> msg.identifier.pgn
            65265
        """
        if can_id > 0x1FFFFFFF:
            raise ValueError(f"Invalid 29-bit CAN ID: 0x{can_id:08X}")
        if len(data) > 8:
            raise ValueError(f"Data too long: {len(data)} bytes (max 8)")

        # Decode identifier
        identifier = self._decode_identifier(can_id)

        # Get PGN name
        pgn_name = self.PGNS.get(identifier.pgn)

        # Parse transport protocol
        transport_info = self._parse_transport_protocol(identifier.pgn, data)
        is_transport = transport_info is not None

        # Handle multi-packet transport data
        if is_transport and transport_info is not None:
            tp_type = transport_info.get("type")
            if tp_type == "TP.DT":
                # Data transfer packet - might complete a session
                self._handle_transport_data(
                    identifier.source_address,
                    identifier.pdu_specific
                    if self._is_pdu1_format(identifier.pdu_format)
                    else 0xFF,
                    data,
                    timestamp,
                )

        # Decode SPNs if not transport protocol
        decoded_spns: dict[str, Any] = {}
        if not is_transport:
            decoded_spns = self._decode_spns(identifier.pgn, data)

        # Create message
        msg = J1939Message(
            timestamp=timestamp,
            can_id=can_id,
            identifier=identifier,
            data=data,
            pgn_name=pgn_name,
            decoded_spns=decoded_spns,
            is_transport_protocol=is_transport,
            transport_info=transport_info,
        )

        self.messages.append(msg)
        return msg

    def _decode_identifier(self, can_id: int) -> J1939Identifier:
        """Decode 29-bit J1939 identifier into components.

        J1939 Identifier Format (bits 28-0):
        - Priority (bits 26-28): 3 bits
        - Reserved (bit 25): 1 bit
        - Data Page (bit 24): 1 bit
        - PDU Format (bits 16-23): 8 bits
        - PDU Specific (bits 8-15): 8 bits (destination or group extension)
        - Source Address (bits 0-7): 8 bits

        Args:
            can_id: 29-bit extended CAN identifier.

        Returns:
            Decoded identifier components.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> ident = analyzer._decode_identifier(0x18FEF100)
            >>> ident.priority
            6
            >>> ident.pgn
            65265
        """
        priority = (can_id >> 26) & 0x07
        reserved = (can_id >> 25) & 0x01
        data_page = (can_id >> 24) & 0x01
        pdu_format = (can_id >> 16) & 0xFF
        pdu_specific = (can_id >> 8) & 0xFF
        source_address = can_id & 0xFF

        # Calculate PGN
        pgn = self._calculate_pgn(pdu_format, pdu_specific, data_page)

        return J1939Identifier(
            priority=priority,
            reserved=reserved,
            data_page=data_page,
            pdu_format=pdu_format,
            pdu_specific=pdu_specific,
            source_address=source_address,
            pgn=pgn,
        )

    def _calculate_pgn(self, pdu_format: int, pdu_specific: int, data_page: int) -> int:
        """Calculate PGN (Parameter Group Number).

        PGN Calculation:
        - If PDU1 format (PDU Format < 240): PGN = DP | PF | 00
        - If PDU2 format (PDU Format >= 240): PGN = DP | PF | PS

        Where:
        - DP = Data Page (bit 24)
        - PF = PDU Format (bits 16-23)
        - PS = PDU Specific (bits 8-15)

        Args:
            pdu_format: PDU Format byte.
            pdu_specific: PDU Specific byte.
            data_page: Data Page bit.

        Returns:
            Calculated PGN.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> analyzer._calculate_pgn(0xFE, 0xF1, 0)
            65265
        """
        if self._is_pdu1_format(pdu_format):
            # PDU1: PDU Specific is destination address, set to 0 for PGN
            pgn = (data_page << 16) | (pdu_format << 8) | 0x00
        else:
            # PDU2: PDU Specific is group extension
            pgn = (data_page << 16) | (pdu_format << 8) | pdu_specific

        return pgn

    def _is_pdu1_format(self, pdu_format: int) -> bool:
        """Check if PDU1 format (destination-specific).

        PDU1 format if PDU Format < 240 (0xF0), meaning PDU Specific
        field contains destination address.

        Args:
            pdu_format: PDU Format byte.

        Returns:
            True if PDU1 format.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> analyzer._is_pdu1_format(0xEF)
            True
            >>> analyzer._is_pdu1_format(0xF0)
            False
        """
        return pdu_format < 240

    def _parse_transport_protocol(self, pgn: int, data: bytes) -> dict[str, Any] | None:
        """Parse transport protocol (TP.CM, TP.DT, BAM).

        TP.CM (PGN 60160) format:
        - Byte 0: Control byte (RTS=16, CTS=17, EOM_ACK=19, BAM=32, ABORT=255)
        - Byte 1-2: Total message size (little-endian)
        - Byte 3: Total packets
        - Byte 4: Max packets (CTS) or Reserved (RTS/BAM)
        - Byte 5-7: PGN of data (little-endian, 3 bytes)

        TP.DT (PGN 60416) format:
        - Byte 0: Sequence number (1-255)
        - Byte 1-7: Data (7 bytes per packet)

        Args:
            pgn: Parameter Group Number.
            data: Message data.

        Returns:
            Transport protocol metadata or None if not transport.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> # TP.CM RTS
            >>> info = analyzer._parse_transport_protocol(
            ...     60160,
            ...     b'\\x10\\x20\\x00\\x05\\xff\\xf0\\x04\\x00'
            ... )
            >>> info['control']
            'RTS'
            >>> info['total_size']
            32
        """
        if pgn == 60160:  # TP.CM
            if len(data) < 8:
                return None

            control = data[0]
            total_size = int.from_bytes(data[1:3], "little")
            total_packets = data[3]
            max_packets = data[4] if control == self.TP_CM_CTS else 0xFF
            data_pgn = int.from_bytes(data[5:8], "little")

            cm_types = {
                16: "RTS",
                17: "CTS",
                19: "EOM_ACK",
                32: "BAM",
                255: "ABORT",
            }

            return {
                "type": "TP.CM",
                "control": cm_types.get(control, f"Unknown ({control})"),
                "total_size": total_size,
                "total_packets": total_packets,
                "max_packets": max_packets if control == self.TP_CM_CTS else None,
                "data_pgn": data_pgn,
            }

        elif pgn == 60416:  # TP.DT
            if len(data) < 1:
                return None

            sequence = data[0]
            packet_data = data[1:]

            return {
                "type": "TP.DT",
                "sequence": sequence,
                "data": packet_data.hex(),
            }

        return None

    def _handle_transport_data(
        self, source_address: int, dest_address: int, data: bytes, timestamp: float
    ) -> bytes | None:
        """Handle multi-packet transport protocol data transfer.

        Reassembles TP.DT packets into complete messages based on TP.CM
        connection management.

        Args:
            source_address: Source address.
            dest_address: Destination address.
            data: TP.DT packet data.
            timestamp: Packet timestamp.

        Returns:
            Complete reassembled message if session complete, None otherwise.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> # Setup session first with TP.CM
            >>> analyzer.parse_message(
            ...     0x18ECF100,
            ...     b'\\x10\\x0e\\x00\\x02\\xff\\xf0\\x04\\x00',
            ...     1.0
            ... )
            <...>
            >>> # Send TP.DT packets
            >>> analyzer.parse_message(
            ...     0x18EBF100,
            ...     b'\\x01\\x00\\x01\\x02\\x03\\x04\\x05\\x06',
            ...     1.1
            ... )
            <...>
        """
        session_key = (source_address, dest_address)

        # Extract sequence number and packet data
        if len(data) < 1:
            return None

        sequence = data[0]
        packet_data = data[1:]

        # Check if session exists
        if session_key not in self.transport_sessions:
            # No active session - ignore
            return None

        session = self.transport_sessions[session_key]

        # Append packet data
        if "packets" not in session:
            session["packets"] = {}

        session["packets"][sequence] = packet_data
        session["last_timestamp"] = timestamp

        # Check if complete
        expected_packets = session.get("total_packets", 0)
        if len(session["packets"]) == expected_packets:
            # Reassemble in sequence order
            complete_data = b"".join(session["packets"][i] for i in range(1, expected_packets + 1))

            # Trim to actual size
            total_size = session.get("total_size", len(complete_data))
            complete_data = complete_data[:total_size]

            # Clean up session
            del self.transport_sessions[session_key]

            return complete_data

        return None

    def _decode_spns(self, pgn: int, data: bytes) -> dict[str, Any]:
        """Decode Suspect Parameter Numbers from PGN data.

        Args:
            pgn: Parameter Group Number.
            data: Message data.

        Returns:
            Dictionary of decoded SPN values.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> spn = J1939SPN(
            ...     spn=190,
            ...     name="Engine Speed",
            ...     start_bit=24,
            ...     bit_length=16,
            ...     resolution=0.125,
            ...     unit="rpm"
            ... )
            >>> analyzer.add_spn_definition(61444, spn)
            >>> # Data with engine speed value
            >>> decoded = analyzer._decode_spns(61444, b'\\x00\\x00\\x00\\x00\\x10\\x00\\x00\\x00')
            >>> decoded.get('Engine Speed')
            2.0
        """
        decoded: dict[str, Any] = {}

        if pgn not in self.spn_definitions:
            return decoded

        # Convert data to bit array
        data_bits = int.from_bytes(data, "little")

        for spn in self.spn_definitions[pgn]:
            # Extract bits
            mask = (1 << spn.bit_length) - 1
            raw_value = (data_bits >> spn.start_bit) & mask

            # Apply scaling
            scaled_value = (raw_value * spn.resolution) + spn.offset

            # Store with name
            decoded[spn.name] = scaled_value

        return decoded

    def add_spn_definition(self, pgn: int, spn: J1939SPN) -> None:
        """Add SPN definition for decoding.

        Args:
            pgn: Parameter Group Number.
            spn: SPN definition.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> spn = J1939SPN(
            ...     spn=190,
            ...     name="Engine Speed",
            ...     start_bit=24,
            ...     bit_length=16,
            ...     resolution=0.125,
            ...     offset=0.0,
            ...     unit="rpm",
            ...     data_range=(0.0, 8031.875)
            ... )
            >>> analyzer.add_spn_definition(61444, spn)
        """
        if pgn not in self.spn_definitions:
            self.spn_definitions[pgn] = []

        self.spn_definitions[pgn].append(spn)

    def export_messages(self, output_path: Path) -> None:
        """Export parsed messages as JSON.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = J1939Analyzer()
            >>> analyzer.parse_message(0x18FEF100, b'\\x00\\x01\\x02\\x03')
            <...>
            >>> analyzer.export_messages(Path("j1939_messages.json"))
        """
        messages_data = []
        for msg in self.messages:
            msg_dict = {
                "timestamp": msg.timestamp,
                "can_id": f"0x{msg.can_id:08X}",
                "pgn": msg.identifier.pgn,
                "pgn_name": msg.pgn_name,
                "priority": msg.identifier.priority,
                "source_address": msg.identifier.source_address,
                "destination_address": (
                    msg.identifier.pdu_specific
                    if self._is_pdu1_format(msg.identifier.pdu_format)
                    else 0xFF
                ),
                "data": msg.data.hex(),
                "decoded_spns": msg.decoded_spns,
                "is_transport_protocol": msg.is_transport_protocol,
                "transport_info": msg.transport_info,
            }
            messages_data.append(msg_dict)

        with output_path.open("w") as f:
            json.dump(
                {"messages": messages_data, "total_messages": len(messages_data)},
                f,
                indent=2,
            )
