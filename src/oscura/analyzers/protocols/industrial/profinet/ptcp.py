"""PTCP (Precision Transparent Clock Protocol) implementation for PROFINET.

PTCP provides time synchronization for PROFINET IRT (Isochronous Real-Time)
communication with sub-microsecond accuracy.

References:
    PROFINET Specification V2.4 - Section 4.7 PTCP
    IEC 61158-6-10:2014
"""

from __future__ import annotations

from typing import Any, ClassVar


class PTCPParser:
    """PTCP (Precision Transparent Clock Protocol) frame parser.

    PTCP provides precise time synchronization for PROFINET networks,
    supporting IRT communication with sub-microsecond accuracy.

    Attributes:
        TLV_TYPES: Mapping of PTCP TLV (Type-Length-Value) types.
    """

    # PTCP Frame Types (based on Frame ID range 0xFF20-0xFF8F)
    FRAME_TYPES: ClassVar[dict[int, str]] = {
        0xFF40: "RTSync PDU with Follow-Up",
        0xFF41: "RTSync PDU",
        0xFF42: "Follow-Up",
        0xFF43: "Delay Request",
        0xFF44: "Delay Response with Follow-Up",
        0xFF45: "Delay Response",
    }

    # PTCP TLV Types
    TLV_TYPES: ClassVar[dict[int, str]] = {
        0x00: "End",
        0x01: "Subdomain UUID",
        0x02: "Time",
        0x03: "Time Extension",
        0x04: "Master Source Address",
        0x05: "Port Parameter",
        0x06: "Delay Parameter",
        0x07: "Port Time",
        0x08: "Optional",
    }

    @staticmethod
    def parse_frame(frame_id: int, data: bytes) -> dict[str, Any]:
        """Parse PTCP frame.

        PTCP Frame Format:
        - Sequence ID (2 bytes)
        - Reserved (2 bytes)
        - TLV blocks (variable)

        Args:
            frame_id: PROFINET frame ID (0xFF20-0xFF8F range).
            data: Raw PTCP frame data.

        Returns:
            Parsed PTCP frame data.

        Raises:
            ValueError: If frame is too short or invalid.

        Example:
            >>> parser = PTCPParser()
            >>> result = parser.parse_frame(0xFF41, ptcp_data)
            >>> print(f"Type: {result['frame_type']}, Sequence: {result['sequence_id']}")
        """
        if len(data) < 4:
            raise ValueError(f"PTCP frame too short: {len(data)} bytes (minimum 4)")

        sequence_id = int.from_bytes(data[0:2], "big")
        reserved = int.from_bytes(data[2:4], "big")

        result: dict[str, Any] = {
            "frame_id": frame_id,
            "frame_type": PTCPParser.FRAME_TYPES.get(frame_id, f"Unknown (0x{frame_id:04X})"),
            "sequence_id": sequence_id,
            "reserved": reserved,
            "tlv_blocks": [],
        }

        # Parse TLV blocks
        offset = 4
        while offset < len(data):
            if offset + 2 > len(data):
                break

            tlv_type = data[offset]
            tlv_length = data[offset + 1]

            if tlv_type == 0x00:  # End marker
                break

            if offset + 2 + tlv_length > len(data):
                break

            tlv_data = data[offset + 2 : offset + 2 + tlv_length]
            tlv_block = PTCPParser._parse_tlv(tlv_type, tlv_data)
            result["tlv_blocks"].append(tlv_block)

            offset += 2 + tlv_length

        return result

    @staticmethod
    def _parse_tlv(tlv_type: int, data: bytes) -> dict[str, Any]:
        """Parse PTCP TLV (Type-Length-Value) block.

        Args:
            tlv_type: TLV type code.
            data: TLV data bytes.

        Returns:
            Parsed TLV block data.
        """
        block: dict[str, Any] = {
            "type": PTCPParser.TLV_TYPES.get(tlv_type, f"Unknown (0x{tlv_type:02X})"),
            "type_raw": tlv_type,
            "length": len(data),
        }

        # Parse based on TLV type
        if tlv_type == 0x01 and len(data) >= 16:
            PTCPParser._parse_subdomain_uuid(block, data)
        elif tlv_type == 0x02 and len(data) >= 10:
            PTCPParser._parse_time(block, data)
        elif tlv_type == 0x03 and len(data) >= 6:
            PTCPParser._parse_time_extension(block, data)
        elif tlv_type == 0x04 and len(data) >= 6:
            PTCPParser._parse_master_source_address(block, data)
        elif tlv_type == 0x05 and len(data) >= 14:
            PTCPParser._parse_port_parameter(block, data)
        elif tlv_type == 0x06 and len(data) >= 20:
            PTCPParser._parse_delay_parameter(block, data)
        elif tlv_type == 0x07 and len(data) >= 10:
            PTCPParser._parse_port_time(block, data)
        else:
            block["data_hex"] = data.hex()

        return block

    @staticmethod
    def _parse_subdomain_uuid(block: dict[str, Any], data: bytes) -> None:
        """Parse Subdomain UUID TLV."""
        block["subdomain_uuid"] = data[:16].hex()

    @staticmethod
    def _parse_time(block: dict[str, Any], data: bytes) -> None:
        """Parse Time TLV."""
        seconds = int.from_bytes(data[0:6], "big")
        nanoseconds = int.from_bytes(data[6:10], "big")
        block["seconds"] = seconds
        block["nanoseconds"] = nanoseconds
        block["timestamp"] = seconds + nanoseconds / 1e9

    @staticmethod
    def _parse_time_extension(block: dict[str, Any], data: bytes) -> None:
        """Parse Time Extension TLV."""
        epoch = int.from_bytes(data[0:2], "big")
        seconds_high = int.from_bytes(data[2:6], "big")
        block["epoch"] = epoch
        block["seconds_high"] = seconds_high

    @staticmethod
    def _parse_master_source_address(block: dict[str, Any], data: bytes) -> None:
        """Parse Master Source Address TLV."""
        block["mac_address"] = ":".join(f"{b:02x}" for b in data[:6])

    @staticmethod
    def _parse_port_parameter(block: dict[str, Any], data: bytes) -> None:
        """Parse Port Parameter TLV."""
        block["t2_port_rx_delay"] = int.from_bytes(data[0:4], "big")
        block["t3_port_tx_delay"] = int.from_bytes(data[4:8], "big")
        block["port_mac_address"] = ":".join(f"{b:02x}" for b in data[8:14])

    @staticmethod
    def _parse_delay_parameter(block: dict[str, Any], data: bytes) -> None:
        """Parse Delay Parameter TLV."""
        block["request_port_rx_delay"] = int.from_bytes(data[0:4], "big")
        block["request_port_tx_delay"] = int.from_bytes(data[4:8], "big")
        block["response_port_rx_delay"] = int.from_bytes(data[8:12], "big")
        block["response_port_tx_delay"] = int.from_bytes(data[12:16], "big")
        block["cable_delay"] = int.from_bytes(data[16:20], "big")

    @staticmethod
    def _parse_port_time(block: dict[str, Any], data: bytes) -> None:
        """Parse Port Time TLV."""
        seconds = int.from_bytes(data[0:6], "big")
        nanoseconds = int.from_bytes(data[6:10], "big")
        block["seconds"] = seconds
        block["nanoseconds"] = nanoseconds
        block["timestamp"] = seconds + nanoseconds / 1e9


__all__ = ["PTCPParser"]
