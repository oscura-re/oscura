"""Unit tests for PTCP (Precision Transparent Clock Protocol) parser."""

from __future__ import annotations

import pytest

from oscura.analyzers.protocols.industrial.profinet.ptcp import PTCPParser


class TestPTCPParser:
    """Tests for PTCPParser class."""

    def test_parse_basic_frame(self) -> None:
        """Test parsing basic PTCP frame."""
        frame_id = 0xFF41  # RTSync PDU
        data = bytes(
            [
                0x00,
                0x10,  # Sequence ID: 16
                0x00,
                0x00,  # Reserved
                0x00,  # End marker (no TLVs)
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert result["frame_id"] == 0xFF41
        assert result["frame_type"] == "RTSync PDU"
        assert result["sequence_id"] == 16
        assert result["reserved"] == 0
        assert len(result["tlv_blocks"]) == 0

    def test_parse_frame_with_time_tlv(self) -> None:
        """Test parsing PTCP frame with Time TLV."""
        frame_id = 0xFF41
        data = bytes(
            [
                0x00,
                0x01,  # Sequence ID: 1
                0x00,
                0x00,  # Reserved
                # Time TLV (type 0x02)
                0x02,  # Type: Time
                0x0A,  # Length: 10
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x01,  # Seconds: 1
                0x00,
                0x00,
                0x00,
                0x64,  # Nanoseconds: 100
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert len(result["tlv_blocks"]) == 1
        assert result["tlv_blocks"][0]["type"] == "Time"
        assert result["tlv_blocks"][0]["seconds"] == 1
        assert result["tlv_blocks"][0]["nanoseconds"] == 100
        assert result["tlv_blocks"][0]["timestamp"] == 1.0 + 100e-9

    def test_parse_frame_with_subdomain_uuid(self) -> None:
        """Test parsing PTCP frame with Subdomain UUID TLV."""
        frame_id = 0xFF40
        uuid = bytes(
            [
                0x01,
                0x02,
                0x03,
                0x04,
                0x05,
                0x06,
                0x07,
                0x08,
                0x09,
                0x0A,
                0x0B,
                0x0C,
                0x0D,
                0x0E,
                0x0F,
                0x10,
            ]
        )

        data = (
            bytes(
                [
                    0x00,
                    0x02,  # Sequence ID
                    0x00,
                    0x00,  # Reserved
                    0x01,  # Type: Subdomain UUID
                    0x10,  # Length: 16
                ]
            )
            + uuid
            + bytes([0x00])
        )  # End marker

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert len(result["tlv_blocks"]) == 1
        assert result["tlv_blocks"][0]["type"] == "Subdomain UUID"
        assert result["tlv_blocks"][0]["subdomain_uuid"] == uuid.hex()

    def test_parse_frame_with_master_source_address(self) -> None:
        """Test parsing PTCP frame with Master Source Address TLV."""
        frame_id = 0xFF41
        data = bytes(
            [
                0x00,
                0x03,  # Sequence ID
                0x00,
                0x00,  # Reserved
                0x04,  # Type: Master Source Address
                0x06,  # Length: 6
                0x00,
                0x11,
                0x22,
                0x33,
                0x44,
                0x55,  # MAC: 00:11:22:33:44:55
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert len(result["tlv_blocks"]) == 1
        assert result["tlv_blocks"][0]["type"] == "Master Source Address"
        assert result["tlv_blocks"][0]["mac_address"] == "00:11:22:33:44:55"

    def test_parse_frame_with_port_parameter(self) -> None:
        """Test parsing PTCP frame with Port Parameter TLV."""
        frame_id = 0xFF41
        data = bytes(
            [
                0x00,
                0x04,  # Sequence ID
                0x00,
                0x00,  # Reserved
                0x05,  # Type: Port Parameter
                0x0E,  # Length: 14
                0x00,
                0x00,
                0x00,
                0x10,  # T2 Port RX Delay: 16
                0x00,
                0x00,
                0x00,
                0x20,  # T3 Port TX Delay: 32
                0xAA,
                0xBB,
                0xCC,
                0xDD,
                0xEE,
                0xFF,  # Port MAC (6 bytes)
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert len(result["tlv_blocks"]) == 1
        assert result["tlv_blocks"][0]["type"] == "Port Parameter"
        assert result["tlv_blocks"][0]["t2_port_rx_delay"] == 16
        assert result["tlv_blocks"][0]["t3_port_tx_delay"] == 32
        assert result["tlv_blocks"][0]["port_mac_address"] == "aa:bb:cc:dd:ee:ff"

    def test_parse_frame_with_delay_parameter(self) -> None:
        """Test parsing PTCP frame with Delay Parameter TLV."""
        frame_id = 0xFF43  # Delay Request
        data = bytes(
            [
                0x00,
                0x05,  # Sequence ID
                0x00,
                0x00,  # Reserved
                0x06,  # Type: Delay Parameter
                0x14,  # Length: 20
                0x00,
                0x00,
                0x00,
                0x0A,  # Request Port RX Delay: 10
                0x00,
                0x00,
                0x00,
                0x0B,  # Request Port TX Delay: 11
                0x00,
                0x00,
                0x00,
                0x0C,  # Response Port RX Delay: 12
                0x00,
                0x00,
                0x00,
                0x0D,  # Response Port TX Delay: 13
                0x00,
                0x00,
                0x00,
                0x64,  # Cable Delay: 100
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert result["frame_type"] == "Delay Request"
        assert len(result["tlv_blocks"]) == 1
        assert result["tlv_blocks"][0]["type"] == "Delay Parameter"
        assert result["tlv_blocks"][0]["request_port_rx_delay"] == 10
        assert result["tlv_blocks"][0]["request_port_tx_delay"] == 11
        assert result["tlv_blocks"][0]["response_port_rx_delay"] == 12
        assert result["tlv_blocks"][0]["response_port_tx_delay"] == 13
        assert result["tlv_blocks"][0]["cable_delay"] == 100

    def test_parse_frame_with_time_extension(self) -> None:
        """Test parsing PTCP frame with Time Extension TLV."""
        frame_id = 0xFF42  # Follow-Up
        data = bytes(
            [
                0x00,
                0x06,  # Sequence ID
                0x00,
                0x00,  # Reserved
                0x03,  # Type: Time Extension
                0x06,  # Length: 6
                0x00,
                0x01,  # Epoch: 1
                0x00,
                0x00,
                0x10,
                0x00,  # Seconds High: 4096
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert len(result["tlv_blocks"]) == 1
        assert result["tlv_blocks"][0]["type"] == "Time Extension"
        assert result["tlv_blocks"][0]["epoch"] == 1
        assert result["tlv_blocks"][0]["seconds_high"] == 4096

    def test_parse_frame_with_port_time(self) -> None:
        """Test parsing PTCP frame with Port Time TLV."""
        frame_id = 0xFF45  # Delay Response
        data = bytes(
            [
                0x00,
                0x07,  # Sequence ID
                0x00,
                0x00,  # Reserved
                0x07,  # Type: Port Time
                0x0A,  # Length: 10
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x02,  # Seconds: 2
                0x00,
                0x00,
                0x01,
                0xF4,  # Nanoseconds: 500
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert result["frame_type"] == "Delay Response"
        assert len(result["tlv_blocks"]) == 1
        assert result["tlv_blocks"][0]["type"] == "Port Time"
        assert result["tlv_blocks"][0]["seconds"] == 2
        assert result["tlv_blocks"][0]["nanoseconds"] == 500
        assert result["tlv_blocks"][0]["timestamp"] == 2.0 + 500e-9

    def test_parse_frame_with_multiple_tlvs(self) -> None:
        """Test parsing PTCP frame with multiple TLV blocks."""
        frame_id = 0xFF41
        data = bytes(
            [
                0x00,
                0x08,  # Sequence ID
                0x00,
                0x00,  # Reserved
                # TLV 1: Master Source Address
                0x04,
                0x06,
                0x00,
                0x11,
                0x22,
                0x33,
                0x44,
                0x55,
                # TLV 2: Time
                0x02,
                0x0A,
                0x00,
                0x00,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x64,
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert len(result["tlv_blocks"]) == 2
        assert result["tlv_blocks"][0]["type"] == "Master Source Address"
        assert result["tlv_blocks"][1]["type"] == "Time"

    def test_parse_frame_with_unknown_tlv(self) -> None:
        """Test parsing PTCP frame with unknown TLV type."""
        frame_id = 0xFF41
        data = bytes(
            [
                0x00,
                0x09,  # Sequence ID
                0x00,
                0x00,  # Reserved
                0xFF,  # Unknown TLV type
                0x04,  # Length: 4
                0x01,
                0x02,
                0x03,
                0x04,  # Data
                0x00,  # End marker
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert len(result["tlv_blocks"]) == 1
        assert "Unknown" in result["tlv_blocks"][0]["type"]
        assert result["tlv_blocks"][0]["data_hex"] == "01020304"

    def test_parse_frame_too_short(self) -> None:
        """Test parsing PTCP frame that is too short."""
        parser = PTCPParser()

        with pytest.raises(ValueError, match="too short"):
            parser.parse_frame(0xFF41, bytes([0x00, 0x01]))

    def test_parse_frame_truncated_tlv(self) -> None:
        """Test parsing frame with truncated TLV."""
        frame_id = 0xFF41
        data = bytes(
            [
                0x00,
                0x0A,  # Sequence ID
                0x00,
                0x00,  # Reserved
                0x02,  # Type: Time
                0x0A,  # Length: 10
                0x00,
                0x01,  # Only 2 bytes of 10-byte TLV
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        # Should stop parsing when TLV is incomplete
        assert len(result["tlv_blocks"]) == 0

    def test_parse_unknown_frame_id(self) -> None:
        """Test parsing PTCP frame with unknown frame ID."""
        frame_id = 0xFF99  # Unknown in PTCP range
        data = bytes(
            [
                0x00,
                0x0B,
                0x00,
                0x00,
                0x00,
            ]
        )

        parser = PTCPParser()
        result = parser.parse_frame(frame_id, data)

        assert "Unknown" in result["frame_type"]
        assert result["frame_id"] == 0xFF99


class TestPTCPConstants:
    """Tests for PTCP constants and mappings."""

    def test_frame_types(self) -> None:
        """Test PTCP frame type mappings."""
        assert PTCPParser.FRAME_TYPES[0xFF40] == "RTSync PDU with Follow-Up"
        assert PTCPParser.FRAME_TYPES[0xFF41] == "RTSync PDU"
        assert PTCPParser.FRAME_TYPES[0xFF42] == "Follow-Up"
        assert PTCPParser.FRAME_TYPES[0xFF43] == "Delay Request"
        assert PTCPParser.FRAME_TYPES[0xFF44] == "Delay Response with Follow-Up"
        assert PTCPParser.FRAME_TYPES[0xFF45] == "Delay Response"

    def test_tlv_types(self) -> None:
        """Test PTCP TLV type mappings."""
        assert PTCPParser.TLV_TYPES[0x00] == "End"
        assert PTCPParser.TLV_TYPES[0x01] == "Subdomain UUID"
        assert PTCPParser.TLV_TYPES[0x02] == "Time"
        assert PTCPParser.TLV_TYPES[0x03] == "Time Extension"
        assert PTCPParser.TLV_TYPES[0x04] == "Master Source Address"
        assert PTCPParser.TLV_TYPES[0x05] == "Port Parameter"
        assert PTCPParser.TLV_TYPES[0x06] == "Delay Parameter"
        assert PTCPParser.TLV_TYPES[0x07] == "Port Time"
        assert PTCPParser.TLV_TYPES[0x08] == "Optional"
