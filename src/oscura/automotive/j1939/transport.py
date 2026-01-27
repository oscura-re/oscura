"""J1939 Transport Protocol (TP) implementation.

This module implements J1939 transport protocol for multi-packet message
transfer, including:
- TP.CM (Connection Management): RTS, CTS, EOM_ACK, BAM, ABORT
- TP.DT (Data Transfer): Sequence-based packet delivery
- Session management and reassembly

References:
    SAE J1939/21 - Data Link Layer (Transport Protocol)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "TransportProtocol",
    "TransportSession",
]


@dataclass
class TransportSession:
    """Active transport protocol session.

    Attributes:
        source_address: Source address initiating transfer.
        dest_address: Destination address (0xFF for broadcast).
        data_pgn: PGN of the data being transferred.
        total_size: Total message size in bytes.
        total_packets: Total number of packets.
        max_packets: Maximum packets per CTS (connection mode only).
        packets: Received packets (sequence -> data).
        started_at: Session start timestamp.
        last_timestamp: Last packet timestamp.
        is_broadcast: True for BAM (broadcast), False for RTS/CTS.
    """

    source_address: int
    dest_address: int
    data_pgn: int
    total_size: int
    total_packets: int
    max_packets: int = 0xFF
    packets: dict[int, bytes] = field(default_factory=dict)
    started_at: float = 0.0
    last_timestamp: float = 0.0
    is_broadcast: bool = False

    def is_complete(self) -> bool:
        """Check if all packets received.

        Returns:
            True if all packets received.

        Example:
            >>> session = TransportSession(0, 0xFF, 61444, 32, 5)
            >>> session.is_complete()
            False
            >>> for i in range(1, 6):
            ...     session.packets[i] = b'\\x00' * 7
            >>> session.is_complete()
            True
        """
        return len(self.packets) == self.total_packets

    def reassemble(self) -> bytes:
        """Reassemble complete message from packets.

        Returns:
            Reassembled message data.

        Raises:
            ValueError: If session is incomplete.

        Example:
            >>> session = TransportSession(0, 0xFF, 61444, 14, 2)
            >>> session.packets[1] = b'\\x00\\x01\\x02\\x03\\x04\\x05\\x06'
            >>> session.packets[2] = b'\\x07\\x08\\x09\\x0a\\x0b\\x0c\\x0d'
            >>> data = session.reassemble()
            >>> len(data)
            14
        """
        if not self.is_complete():
            raise ValueError(f"Session incomplete: {len(self.packets)}/{self.total_packets}")

        # Reassemble in sequence order
        complete_data = b"".join(self.packets[i] for i in range(1, self.total_packets + 1))

        # Trim to actual size (last packet may be padded)
        return complete_data[: self.total_size]


class TransportProtocol:
    """J1939 Transport Protocol handler.

    Manages multi-packet message transfer sessions for messages
    larger than 8 bytes using TP.CM and TP.DT.

    Example:
        >>> tp = TransportProtocol()
        >>> # Parse TP.CM RTS
        >>> cm_info = tp.parse_cm(b'\\x10\\x20\\x00\\x05\\xff\\xf0\\x04\\x00')
        >>> cm_info['control']
        'RTS'
        >>> cm_info['total_size']
        32
    """

    # PGN for transport protocol
    PGN_TP_CM = 60160  # Connection Management
    PGN_TP_DT = 60416  # Data Transfer

    # Connection Management control bytes
    CM_RTS = 16  # Request To Send
    CM_CTS = 17  # Clear To Send
    CM_EOM_ACK = 19  # End of Message Acknowledgment
    CM_BAM = 32  # Broadcast Announce Message
    CM_ABORT = 255  # Connection Abort

    def __init__(self) -> None:
        """Initialize transport protocol handler."""
        # Active sessions: (source_addr, dest_addr) -> session
        self.sessions: dict[tuple[int, int], TransportSession] = {}

    def parse_cm(self, data: bytes) -> dict[str, Any] | None:
        """Parse TP.CM (Connection Management) message.

        TP.CM format (8 bytes):
        - Byte 0: Control byte (RTS/CTS/EOM_ACK/BAM/ABORT)
        - Byte 1-2: Total message size (little-endian)
        - Byte 3: Total packets
        - Byte 4: Max packets (CTS) or Reserved (RTS/BAM)
        - Byte 5-7: PGN of data (little-endian, 3 bytes)

        Args:
            data: TP.CM message data (8 bytes).

        Returns:
            Parsed CM metadata or None if invalid.

        Example:
            >>> tp = TransportProtocol()
            >>> # RTS message
            >>> info = tp.parse_cm(b'\\x10\\x20\\x00\\x05\\xff\\xf0\\x04\\x00')
            >>> info['control']
            'RTS'
            >>> info['data_pgn']
            61680
        """
        if len(data) < 8:
            return None

        control = data[0]
        total_size = int.from_bytes(data[1:3], "little")
        total_packets = data[3]
        max_packets = data[4]
        data_pgn = int.from_bytes(data[5:8], "little")

        cm_types = {
            self.CM_RTS: "RTS",
            self.CM_CTS: "CTS",
            self.CM_EOM_ACK: "EOM_ACK",
            self.CM_BAM: "BAM",
            self.CM_ABORT: "ABORT",
        }

        return {
            "control": cm_types.get(control, f"Unknown ({control})"),
            "control_byte": control,
            "total_size": total_size,
            "total_packets": total_packets,
            "max_packets": max_packets,
            "data_pgn": data_pgn,
        }

    def parse_dt(self, data: bytes) -> dict[str, Any] | None:
        """Parse TP.DT (Data Transfer) message.

        TP.DT format:
        - Byte 0: Sequence number (1-255)
        - Byte 1-7: Data (7 bytes per packet)

        Args:
            data: TP.DT message data (8 bytes).

        Returns:
            Parsed DT metadata or None if invalid.

        Example:
            >>> tp = TransportProtocol()
            >>> info = tp.parse_dt(b'\\x01\\x00\\x01\\x02\\x03\\x04\\x05\\x06')
            >>> info['sequence']
            1
            >>> info['data']
            b'\\x00\\x01\\x02\\x03\\x04\\x05\\x06'
        """
        if len(data) < 1:
            return None

        sequence = data[0]
        packet_data = data[1:]

        return {
            "sequence": sequence,
            "data": packet_data,
        }

    def start_session(
        self,
        source_address: int,
        dest_address: int,
        cm_info: dict[str, Any],
        timestamp: float,
    ) -> TransportSession | None:
        """Start transport protocol session from TP.CM.

        Args:
            source_address: Source address.
            dest_address: Destination address.
            cm_info: Parsed TP.CM metadata.
            timestamp: Session start timestamp.

        Returns:
            Created session or None if invalid.

        Example:
            >>> tp = TransportProtocol()
            >>> cm_info = tp.parse_cm(b'\\x10\\x20\\x00\\x05\\xff\\xf0\\x04\\x00')
            >>> session = tp.start_session(0, 0xFF, cm_info, 1.0)
            >>> session.total_size
            32
        """
        control = cm_info.get("control_byte")
        if control not in (self.CM_RTS, self.CM_BAM):
            return None

        session = TransportSession(
            source_address=source_address,
            dest_address=dest_address,
            data_pgn=cm_info["data_pgn"],
            total_size=cm_info["total_size"],
            total_packets=cm_info["total_packets"],
            max_packets=cm_info.get("max_packets", 0xFF),
            started_at=timestamp,
            last_timestamp=timestamp,
            is_broadcast=(control == self.CM_BAM),
        )

        session_key = (source_address, dest_address)
        self.sessions[session_key] = session

        return session

    def add_packet(
        self,
        source_address: int,
        dest_address: int,
        dt_info: dict[str, Any],
        timestamp: float,
    ) -> bytes | None:
        """Add TP.DT packet to session.

        Args:
            source_address: Source address.
            dest_address: Destination address.
            dt_info: Parsed TP.DT metadata.
            timestamp: Packet timestamp.

        Returns:
            Complete reassembled message if session complete, None otherwise.

        Example:
            >>> tp = TransportProtocol()
            >>> cm = tp.parse_cm(b'\\x20\\x0e\\x00\\x02\\xff\\xf0\\x04\\x00')
            >>> tp.start_session(0, 0xFF, cm, 1.0)
            <...>
            >>> dt1 = tp.parse_dt(b'\\x01\\x00\\x01\\x02\\x03\\x04\\x05\\x06')
            >>> tp.add_packet(0, 0xFF, dt1, 1.1)
            >>> dt2 = tp.parse_dt(b'\\x02\\x07\\x08\\x09\\x0a\\x0b\\x0c\\x0d')
            >>> data = tp.add_packet(0, 0xFF, dt2, 1.2)
            >>> len(data)
            14
        """
        session_key = (source_address, dest_address)

        if session_key not in self.sessions:
            return None

        session = self.sessions[session_key]
        sequence = dt_info["sequence"]
        packet_data = dt_info["data"]

        # Add packet
        session.packets[sequence] = packet_data
        session.last_timestamp = timestamp

        # Check if complete
        if session.is_complete():
            complete_data = session.reassemble()
            del self.sessions[session_key]
            return complete_data

        return None
