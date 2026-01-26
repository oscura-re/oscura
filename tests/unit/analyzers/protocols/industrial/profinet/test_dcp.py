"""Unit tests for DCP (Discovery and Configuration Protocol) parser."""

from __future__ import annotations

import pytest

from oscura.analyzers.protocols.industrial.profinet.dcp import DCPParser


class TestDCPParser:
    """Tests for DCPParser class."""

    def test_parse_identify_request(self) -> None:
        """Test parsing DCP Identify request."""
        # DCP Identify Request frame
        # Service ID: 0x03 (Identify)
        # Service Type: 0x00 (Request)
        # XID: 0x12345678
        # Response Delay: 0x0000
        # DCPDataLength: 4
        # Option: 0xFF (All Selector), Suboption: 0xFF, Length: 0
        data = bytes(
            [
                0x03,  # Service ID: Identify
                0x00,  # Service Type: Request
                0x12,
                0x34,
                0x56,
                0x78,  # Transaction ID
                0x00,
                0x00,  # Response Delay
                0x00,
                0x04,  # DCPDataLength: 4
                0xFF,  # Option: All Selector
                0xFF,  # Suboption: All
                0x00,
                0x00,  # Block Length: 0
            ]
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert result["service"] == "Identify"
        assert result["service_id"] == 0x03
        assert result["service_type"] == "Request"
        assert result["transaction_id"] == 0x12345678
        assert result["response_delay"] == 0
        assert result["data_length"] == 4
        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["option"] == "All Selector"

    def test_parse_device_name_block(self) -> None:
        """Test parsing Name of Station block."""
        device_name = b"TestDevice123"

        # DCP Set request with Name of Station
        data = (
            bytes(
                [
                    0x02,  # Service ID: Set
                    0x00,  # Service Type: Request
                    0x00,
                    0x00,
                    0x00,
                    0x01,  # Transaction ID
                    0x00,
                    0x00,  # Response Delay
                    0x00,
                    0x11,  # DCPDataLength: 17 (4 + 13)
                    0x02,  # Option: Device Properties
                    0x02,  # Suboption: Name of Station
                    0x00,
                    0x0D,  # Block Length: 13
                ]
            )
            + device_name
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert result["service"] == "Set"
        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["option"] == "Device Properties"
        assert result["blocks"][0]["suboption_name"] == "Name of Station"
        assert result["blocks"][0]["device_name"] == "TestDevice123"

    def test_parse_device_id_block(self) -> None:
        """Test parsing Device ID block."""
        # DCP response with Device ID
        data = bytes(
            [
                0x03,  # Service ID: Identify
                0x01,  # Service Type: Response Success
                0x00,
                0x00,
                0x00,
                0x02,  # Transaction ID
                0x00,
                0x00,  # Response Delay
                0x00,
                0x08,  # DCPDataLength: 8 (4 + 4)
                0x02,  # Option: Device Properties
                0x03,  # Suboption: Device ID
                0x00,
                0x04,  # Block Length: 4
                0x00,
                0x2A,  # Vendor ID: 0x002A (Siemens)
                0x00,
                0x01,  # Device ID: 0x0001
            ]
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert result["service"] == "Identify"
        assert result["service_type"] == "Response Success"
        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["suboption_name"] == "Device ID"
        assert result["blocks"][0]["vendor_id"] == 0x002A
        assert result["blocks"][0]["device_id"] == 0x0001

    def test_parse_device_role_block(self) -> None:
        """Test parsing Device Role block."""
        data = bytes(
            [
                0x03,  # Service ID: Identify
                0x01,  # Service Type: Response Success
                0x00,
                0x00,
                0x00,
                0x03,  # Transaction ID
                0x00,
                0x00,  # Response Delay
                0x00,
                0x06,  # DCPDataLength: 6 (4 + 2)
                0x02,  # Option: Device Properties
                0x04,  # Suboption: Device Role
                0x00,
                0x02,  # Block Length: 2
                0x00,
                0x01,  # Device Role: 0x0001 (IO-Device)
            ]
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["suboption_name"] == "Device Role"
        assert result["blocks"][0]["device_role"] == 0x0001
        assert "IO-Device" in result["blocks"][0]["role_names"]

    def test_parse_ip_parameter_block(self) -> None:
        """Test parsing IP parameter block."""
        data = bytes(
            [
                0x02,  # Service ID: Set
                0x01,  # Service Type: Response Success
                0x00,
                0x00,
                0x00,
                0x04,  # Transaction ID
                0x00,
                0x00,  # Response Delay
                0x00,
                0x10,  # DCPDataLength: 16 (4 + 12)
                0x01,  # Option: IP
                0x02,  # Suboption: IP parameter
                0x00,
                0x0C,  # Block Length: 12
                192,
                168,
                1,
                100,  # IP Address: 192.168.1.100
                255,
                255,
                255,
                0,  # Subnet Mask: 255.255.255.0
                192,
                168,
                1,
                1,  # Gateway: 192.168.1.1
            ]
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["option"] == "IP"
        assert result["blocks"][0]["suboption_name"] == "IP parameter"
        assert result["blocks"][0]["ip_address"] == "192.168.1.100"
        assert result["blocks"][0]["subnet_mask"] == "255.255.255.0"
        assert result["blocks"][0]["gateway"] == "192.168.1.1"

    def test_parse_mac_address_block(self) -> None:
        """Test parsing MAC address block."""
        data = bytes(
            [
                0x01,  # Service ID: Get
                0x01,  # Service Type: Response Success
                0x00,
                0x00,
                0x00,
                0x05,  # Transaction ID
                0x00,
                0x00,  # Response Delay
                0x00,
                0x0A,  # DCPDataLength: 10 (4 + 6)
                0x01,  # Option: IP
                0x01,  # Suboption: MAC address
                0x00,
                0x06,  # Block Length: 6
                0x00,
                0x11,
                0x22,
                0x33,
                0x44,
                0x55,  # MAC: 00:11:22:33:44:55
            ]
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert len(result["blocks"]) == 1
        assert result["blocks"][0]["option"] == "IP"
        assert result["blocks"][0]["suboption_name"] == "MAC address"
        assert result["blocks"][0]["mac_address"] == "00:11:22:33:44:55"

    def test_parse_multiple_blocks(self) -> None:
        """Test parsing frame with multiple DCP blocks."""
        device_name = b"Device1"

        data = (
            bytes(
                [
                    0x03,  # Service ID: Identify
                    0x01,  # Service Type: Response Success
                    0x00,
                    0x00,
                    0x00,
                    0x06,  # Transaction ID
                    0x00,
                    0x00,  # Response Delay
                    0x00,
                    0x15,  # DCPDataLength: 21
                    # Block 1: Name of Station
                    0x02,  # Option: Device Properties
                    0x02,  # Suboption: Name of Station
                    0x00,
                    0x07,  # Block Length: 7
                ]
            )
            + device_name
            + bytes(
                [
                    0x00,  # Padding to 2-byte boundary
                    # Block 2: Device ID
                    0x02,  # Option: Device Properties
                    0x03,  # Suboption: Device ID
                    0x00,
                    0x04,  # Block Length: 4
                    0x00,
                    0x2A,  # Vendor ID
                    0x00,
                    0x01,  # Device ID
                ]
            )
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert len(result["blocks"]) == 2
        assert result["blocks"][0]["device_name"] == "Device1"
        assert result["blocks"][1]["vendor_id"] == 0x002A
        assert result["blocks"][1]["device_id"] == 0x0001

    def test_parse_frame_too_short(self) -> None:
        """Test parsing frame that is too short."""
        parser = DCPParser()

        with pytest.raises(ValueError, match="too short"):
            parser.parse_frame(bytes([0x01, 0x02, 0x03]))

    def test_parse_unknown_service(self) -> None:
        """Test parsing frame with unknown service ID."""
        data = bytes(
            [
                0xFF,  # Unknown Service ID
                0x00,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x00,  # Minimal valid frame
            ]
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        assert "Unknown" in result["service"]
        assert result["service_id"] == 0xFF

    def test_parse_block_alignment(self) -> None:
        """Test that block parsing respects 2-byte alignment."""
        # Block with odd length (requires padding)
        data = bytes(
            [
                0x01,
                0x00,
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x07,  # DCPDataLength: 7
                0x02,
                0x02,
                0x00,
                0x03,  # Block: length 3
                0x41,
                0x42,
                0x43,  # "ABC"
                # Padding byte would be here in real frame
            ]
        )

        parser = DCPParser()
        result = parser.parse_frame(data)

        # Should parse the block correctly
        assert len(result["blocks"]) == 1


class TestDCPConstants:
    """Tests for DCP constants and mappings."""

    def test_service_ids(self) -> None:
        """Test DCP service ID mappings."""
        assert DCPParser.SERVICE_IDS[0x01] == "Get"
        assert DCPParser.SERVICE_IDS[0x02] == "Set"
        assert DCPParser.SERVICE_IDS[0x03] == "Identify"
        assert DCPParser.SERVICE_IDS[0x04] == "Hello"

    def test_service_types(self) -> None:
        """Test DCP service type mappings."""
        assert DCPParser.SERVICE_TYPES[0x00] == "Request"
        assert DCPParser.SERVICE_TYPES[0x01] == "Response Success"
        assert DCPParser.SERVICE_TYPES[0x05] == "Response Not Supported"

    def test_options(self) -> None:
        """Test DCP option mappings."""
        assert DCPParser.OPTIONS[0x01] == "IP"
        assert DCPParser.OPTIONS[0x02] == "Device Properties"
        assert DCPParser.OPTIONS[0xFF] == "All Selector"

    def test_device_props_suboptions(self) -> None:
        """Test device properties suboption mappings."""
        assert DCPParser.DEVICE_PROPS_SUBOPTIONS[0x02] == "Name of Station"
        assert DCPParser.DEVICE_PROPS_SUBOPTIONS[0x03] == "Device ID"
        assert DCPParser.DEVICE_PROPS_SUBOPTIONS[0x04] == "Device Role"

    def test_ip_suboptions(self) -> None:
        """Test IP suboption mappings."""
        assert DCPParser.IP_SUBOPTIONS[0x01] == "MAC address"
        assert DCPParser.IP_SUBOPTIONS[0x02] == "IP parameter"
        assert DCPParser.IP_SUBOPTIONS[0x03] == "Full IP Suite"
