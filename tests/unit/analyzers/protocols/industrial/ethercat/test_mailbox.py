"""Tests for EtherCAT mailbox protocol parsers.

Test coverage:
- Mailbox header parsing
- CoE (CAN over EtherCAT) parsing
- FoE (File over EtherCAT) parsing
- SoE (Servo over EtherCAT) parsing
- EoE (Ethernet over EtherCAT) parsing
- SDO parsing
"""

import pytest

from oscura.analyzers.protocols.industrial.ethercat.mailbox import parse_mailbox


class TestMailboxParsing:
    """Test mailbox protocol parsing."""

    def test_parse_mailbox_too_short(self) -> None:
        """Test parsing mailbox data that is too short."""
        with pytest.raises(ValueError, match="Mailbox data too short"):
            parse_mailbox(b"\x00\x01")

    def test_parse_mailbox_header(self) -> None:
        """Test parsing mailbox header."""
        # Length=10, Address=1, Channel=0, Priority=0, Type=2 (CoE), Counter=0
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x02)  # Type=CoE (2), Counter=0

        result = parse_mailbox(bytes(data))

        assert result["length"] == 10
        assert result["address"] == 1
        assert result["channel"] == 0
        assert result["priority"] == 0
        assert result["protocol"] == "CoE"
        assert result["protocol_type"] == 0x02
        assert result["counter"] == 0

    def test_parse_coe_mailbox(self) -> None:
        """Test parsing CoE mailbox data."""
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x02)  # Type=CoE
        # CoE data
        data.extend((0x1000).to_bytes(2, "little"))  # Number (SDO index)
        data.append(0x01)  # Service = SDO Request

        result = parse_mailbox(bytes(data))

        assert result["protocol"] == "CoE"
        assert "coe" in result
        assert result["coe"]["number"] == 0x1000
        assert "SDO Request" in result["coe"]["service"]

    def test_parse_coe_sdo(self) -> None:
        """Test parsing CoE SDO data."""
        data = bytearray()
        data.extend((16).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x02)  # Type=CoE
        # CoE data
        data.extend((0x1000).to_bytes(2, "little"))  # Number
        data.append(0x01)  # Service = SDO Request
        # SDO data (8 bytes minimum)
        data.append(0x23)  # Command (Download Initiate)
        data.extend((0x1018).to_bytes(2, "little"))  # Index
        data.append(0x01)  # Subindex
        data.extend(b"\x00\x00\x00\x00")  # Data bytes

        result = parse_mailbox(bytes(data))

        assert "coe" in result
        assert "sdo" in result["coe"]
        sdo = result["coe"]["sdo"]
        assert sdo["index"] == 0x1018
        assert sdo["subindex"] == 0x01

    def test_parse_foe_mailbox(self) -> None:
        """Test parsing FoE mailbox data."""
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x03)  # Type=FoE
        # FoE data
        data.append(0x01)  # OpCode = Read Request
        data.append(0x00)  # Reserved
        data.extend((0x00000001).to_bytes(4, "little"))  # Packet number

        result = parse_mailbox(bytes(data))

        assert result["protocol"] == "FoE"
        assert "foe" in result
        assert "Read Request" in result["foe"]["opcode"]
        assert result["foe"]["packet_number"] == 1

    def test_parse_soe_mailbox(self) -> None:
        """Test parsing SoE mailbox data."""
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x04)  # Type=SoE
        # SoE data
        # OpCode=1 (Read Request), InComplete=0, Error=0, DriveNo=0
        data.append(0x01)  # Header
        data.append(0x00)  # Reserved
        data.extend((0x0010).to_bytes(2, "little"))  # IDN

        result = parse_mailbox(bytes(data))

        assert result["protocol"] == "SoE"
        assert "soe" in result
        assert "Read Request" in result["soe"]["opcode"]
        assert result["soe"]["idn"] == 0x0010
        assert not result["soe"]["incomplete"]
        assert not result["soe"]["error"]
        assert result["soe"]["drive_number"] == 0

    def test_parse_eoe_mailbox(self) -> None:
        """Test parsing EoE mailbox data."""
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x05)  # Type=EoE
        # EoE data
        # Type=0 (Fragment), Port=0
        data.append(0x00)  # Type/Port
        # LastFragment=1, TimeAppend=0, TimeRequest=0
        data.append(0x01)  # Flags
        # FragmentNumber=0, FrameOffset=0
        data.extend((0x0000).to_bytes(2, "little"))

        result = parse_mailbox(bytes(data))

        assert result["protocol"] == "EoE"
        assert "eoe" in result
        assert "Fragment" in result["eoe"]["type"]
        assert result["eoe"]["last_fragment"]
        assert not result["eoe"]["time_append"]
        assert not result["eoe"]["time_request"]

    def test_parse_unknown_protocol(self) -> None:
        """Test parsing unknown protocol type."""
        data = bytearray()
        data.extend((6).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0xFF)  # Type=0x0F (lower 4 bits), Counter=0x0F (upper 4 bits)

        result = parse_mailbox(bytes(data))

        assert "Unknown" in result["protocol"]
        # Protocol type is lower 4 bits of 0xFF = 0x0F
        assert result["protocol_type"] == 0x0F

    def test_parse_error_protocol(self) -> None:
        """Test parsing error protocol (type 0)."""
        data = bytearray()
        data.extend((6).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x00)  # Type=ERR

        result = parse_mailbox(bytes(data))

        assert result["protocol"] == "ERR"

    def test_parse_mailbox_with_priority(self) -> None:
        """Test parsing mailbox with priority bits set."""
        data = bytearray()
        data.extend((6).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0xC0)  # Channel=0, Priority=3 (bits 6-7)
        data.append(0x02)  # Type=CoE

        result = parse_mailbox(bytes(data))

        assert result["priority"] == 3
        assert result["channel"] == 0

    def test_parse_mailbox_with_counter(self) -> None:
        """Test parsing mailbox with counter bits set."""
        data = bytearray()
        data.extend((6).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x52)  # Type=2 (CoE), Counter=5 (bits 4-6)

        result = parse_mailbox(bytes(data))

        assert result["protocol_type"] == 0x02
        assert result["counter"] == 5

    def test_parse_coe_minimal(self) -> None:
        """Test parsing minimal CoE data (header only)."""
        data = bytearray()
        data.extend((8).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x02)  # Type=CoE
        # Minimal CoE data (just number)
        data.extend((0x1000).to_bytes(2, "little"))  # Number

        result = parse_mailbox(bytes(data))

        assert "coe" in result
        assert result["coe"]["number"] == 0x1000

    def test_parse_foe_minimal(self) -> None:
        """Test parsing minimal FoE data."""
        data = bytearray()
        data.extend((7).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x03)  # Type=FoE
        # Minimal FoE data (opcode only)
        data.append(0x04)  # OpCode = Ack

        result = parse_mailbox(bytes(data))

        assert "foe" in result
        assert "Ack" in result["foe"]["opcode"]

    def test_parse_soe_with_flags(self) -> None:
        """Test parsing SoE data with flags set."""
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x04)  # Type=SoE
        # SoE data with InComplete and Error flags
        # OpCode=1, InComplete=1 (bit 3), Error=1 (bit 4), DriveNo=2
        data.append(0x59)  # 01011001 binary
        data.append(0x00)
        data.extend((0x0020).to_bytes(2, "little"))  # IDN

        result = parse_mailbox(bytes(data))

        assert "soe" in result
        assert result["soe"]["incomplete"]
        assert result["soe"]["error"]
        assert result["soe"]["drive_number"] == 2

    def test_parse_eoe_init_request(self) -> None:
        """Test parsing EoE Init Request."""
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x05)  # Type=EoE
        # EoE Init Request: Type=1, Port=0
        data.append(0x01)  # Type/Port
        data.append(0x00)  # Flags
        data.extend((0x0000).to_bytes(2, "little"))

        result = parse_mailbox(bytes(data))

        assert "eoe" in result
        assert "Init Request" in result["eoe"]["type"]

    def test_parse_eoe_with_time_flags(self) -> None:
        """Test parsing EoE data with time flags."""
        data = bytearray()
        data.extend((10).to_bytes(2, "little"))  # Length
        data.extend((1).to_bytes(2, "little"))  # Address
        data.append(0x00)  # Channel/Priority
        data.append(0x05)  # Type=EoE
        # EoE Fragment with time flags
        data.append(0x00)  # Type=Fragment, Port=0
        data.append(0x06)  # TimeAppend=1, TimeRequest=1
        data.extend((0x0000).to_bytes(2, "little"))

        result = parse_mailbox(bytes(data))

        assert "eoe" in result
        assert result["eoe"]["time_append"]
        assert result["eoe"]["time_request"]
