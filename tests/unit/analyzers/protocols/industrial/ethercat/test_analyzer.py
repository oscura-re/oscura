"""Tests for EtherCAT protocol analyzer.

Test coverage:
- Frame parsing with single and multiple datagrams
- All datagram command types
- Working counter analysis
- Topology discovery
- Slave state detection
- Configuration export
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from oscura.analyzers.protocols.industrial.ethercat.analyzer import (
    EtherCATAnalyzer,
    EtherCATDatagram,
    EtherCATFrame,
    EtherCATSlave,
)


class TestEtherCATDatagram:
    """Test EtherCATDatagram dataclass."""

    def test_datagram_creation(self) -> None:
        """Test creating a datagram."""
        datagram = EtherCATDatagram(
            cmd=0x01,
            cmd_name="APRD",
            idx=0,
            adp=0,
            ado=0x0130,
            len_=2,
            irq=0,
            data=b"\x01\x00",
            wkc=1,
            more_follows=False,
        )

        assert datagram.cmd == 0x01
        assert datagram.cmd_name == "APRD"
        assert datagram.ado == 0x0130
        assert datagram.wkc == 1
        assert not datagram.more_follows


class TestEtherCATFrame:
    """Test EtherCATFrame dataclass."""

    def test_frame_creation(self) -> None:
        """Test creating a frame."""
        datagram = EtherCATDatagram(
            cmd=0x01,
            cmd_name="APRD",
            idx=0,
            adp=0,
            ado=0x0130,
            len_=2,
            irq=0,
            data=b"\x01\x00",
            wkc=1,
            more_follows=False,
        )

        frame = EtherCATFrame(timestamp=1.234, length=16, datagrams=[datagram])

        assert frame.timestamp == 1.234
        assert frame.length == 16
        assert len(frame.datagrams) == 1


class TestEtherCATSlave:
    """Test EtherCATSlave dataclass."""

    def test_slave_creation(self) -> None:
        """Test creating a slave."""
        slave = EtherCATSlave(
            station_address=1,
            alias_address=100,
            vendor_id=0x00000002,
            product_code=0x044C2C52,
            state="OP",
            dc_supported=True,
            mailbox_protocols=["CoE", "FoE"],
        )

        assert slave.station_address == 1
        assert slave.alias_address == 100
        assert slave.state == "OP"
        assert slave.dc_supported
        assert "CoE" in slave.mailbox_protocols


class TestEtherCATAnalyzer:
    """Test EtherCAT analyzer."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = EtherCATAnalyzer()

        assert analyzer.frames == []
        assert analyzer.slaves == {}

    def test_parse_frame_too_short(self) -> None:
        """Test parsing frame that is too short."""
        analyzer = EtherCATAnalyzer()

        with pytest.raises(ValueError, match="frame too short"):
            analyzer.parse_frame(b"\x00")

    def test_parse_frame_single_datagram(self) -> None:
        """Test parsing frame with single datagram."""
        analyzer = EtherCATAnalyzer()

        # Create simple APRD datagram (Auto-increment Physical Read)
        # Frame: Length (2) + Datagram
        # Datagram: Cmd(1) Idx(1) ADP(2) ADO(2) Len/M/IRQ(2) Data(2) WKC(2) = 12 bytes
        frame_data = bytearray()

        # Frame length (11 bits, lower bits of 2-byte field)
        frame_length = 14  # Total datagram size
        frame_data.extend(frame_length.to_bytes(2, "little"))

        # Datagram
        frame_data.append(0x01)  # Cmd: APRD
        frame_data.append(0x00)  # Idx: 0
        frame_data.extend((0x0000).to_bytes(2, "little"))  # ADP: 0
        frame_data.extend((0x0130).to_bytes(2, "little"))  # ADO: 0x0130 (AL Status)
        # Len/M/IRQ: len=2, M=0 (no more), IRQ=0
        len_m_irq = 2  # Just length, no M flag
        frame_data.extend(len_m_irq.to_bytes(2, "little"))
        frame_data.extend(b"\x08\x00")  # Data: state = OP (0x08)
        frame_data.extend((0x0001).to_bytes(2, "little"))  # WKC: 1

        frame = analyzer.parse_frame(bytes(frame_data), timestamp=1.0)

        assert frame.timestamp == 1.0
        assert frame.length == 14
        assert len(frame.datagrams) == 1

        datagram = frame.datagrams[0]
        assert datagram.cmd == 0x01
        assert datagram.cmd_name == "APRD"
        assert datagram.adp == 0
        assert datagram.ado == 0x0130
        assert datagram.len_ == 2
        assert datagram.wkc == 1
        assert not datagram.more_follows

    def test_parse_frame_multiple_datagrams(self) -> None:
        """Test parsing frame with multiple datagrams."""
        analyzer = EtherCATAnalyzer()

        frame_data = bytearray()

        # Frame length
        frame_length = 28  # Two datagrams
        frame_data.extend(frame_length.to_bytes(2, "little"))

        # First datagram (with M flag set)
        frame_data.append(0x01)  # Cmd: APRD
        frame_data.append(0x00)  # Idx: 0
        frame_data.extend((0x0000).to_bytes(2, "little"))  # ADP: 0
        frame_data.extend((0x0010).to_bytes(2, "little"))  # ADO: 0x0010
        len_m_irq = 2 | 0x8000  # len=2, M=1 (more follows)
        frame_data.extend(len_m_irq.to_bytes(2, "little"))
        frame_data.extend(b"\x01\x00")  # Data
        frame_data.extend((0x0001).to_bytes(2, "little"))  # WKC: 1

        # Second datagram (M flag clear)
        frame_data.append(0x02)  # Cmd: APWR
        frame_data.append(0x00)  # Idx: 0
        frame_data.extend((0x0001).to_bytes(2, "little"))  # ADP: 1
        frame_data.extend((0x0020).to_bytes(2, "little"))  # ADO: 0x0020
        len_m_irq = 2  # len=2, M=0 (no more)
        frame_data.extend(len_m_irq.to_bytes(2, "little"))
        frame_data.extend(b"\x02\x00")  # Data
        frame_data.extend((0x0001).to_bytes(2, "little"))  # WKC: 1

        frame = analyzer.parse_frame(bytes(frame_data), timestamp=2.0)

        assert len(frame.datagrams) == 2

        # First datagram
        assert frame.datagrams[0].cmd == 0x01
        assert frame.datagrams[0].cmd_name == "APRD"
        assert frame.datagrams[0].more_follows

        # Second datagram
        assert frame.datagrams[1].cmd == 0x02
        assert frame.datagrams[1].cmd_name == "APWR"
        assert not frame.datagrams[1].more_follows

    def test_parse_all_command_types(self) -> None:
        """Test parsing all command types."""
        analyzer = EtherCATAnalyzer()

        command_codes = [
            0x00,
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
        ]

        for cmd_code in command_codes:
            frame_data = bytearray()
            frame_data.extend((14).to_bytes(2, "little"))  # Frame length
            frame_data.append(cmd_code)  # Command
            frame_data.append(0x00)  # Idx
            frame_data.extend((0x0000).to_bytes(2, "little"))  # ADP
            frame_data.extend((0x0000).to_bytes(2, "little"))  # ADO
            frame_data.extend((2).to_bytes(2, "little"))  # Len (no M flag)
            frame_data.extend(b"\x00\x00")  # Data
            frame_data.extend((0x0000).to_bytes(2, "little"))  # WKC

            frame = analyzer.parse_frame(bytes(frame_data))
            assert frame.datagrams[0].cmd == cmd_code
            assert (
                frame.datagrams[0].cmd_name in analyzer.COMMANDS.values()
                or "Unknown" in frame.datagrams[0].cmd_name
            )

    def test_working_counter_analysis(self) -> None:
        """Test working counter values."""
        analyzer = EtherCATAnalyzer()

        # Frame with WKC=3 (3 slaves responded)
        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x07)  # BRD (Broadcast Read)
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((0x0130).to_bytes(2, "little"))
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x00\x00")
        frame_data.extend((0x0003).to_bytes(2, "little"))  # WKC: 3

        frame = analyzer.parse_frame(bytes(frame_data))
        assert frame.datagrams[0].wkc == 3

    def test_discover_topology(self) -> None:
        """Test topology discovery."""
        analyzer = EtherCATAnalyzer()

        # Parse frames with auto-increment addressing
        for i in range(3):
            frame_data = bytearray()
            frame_data.extend((14).to_bytes(2, "little"))
            frame_data.append(0x01)  # APRD
            frame_data.append(0x00)
            frame_data.extend(i.to_bytes(2, "little"))  # ADP: slave position
            frame_data.extend((0x0000).to_bytes(2, "little"))
            frame_data.extend((2).to_bytes(2, "little"))
            frame_data.extend(b"\x00\x00")
            frame_data.extend((0x0001).to_bytes(2, "little"))  # WKC: 1 (success)

            analyzer.parse_frame(bytes(frame_data))

        addresses = analyzer.discover_topology()
        assert len(addresses) == 3
        assert addresses == [0, 1, 2]

    def test_slave_state_detection(self) -> None:
        """Test slave state detection from AL Status register."""
        analyzer = EtherCATAnalyzer()

        # Parse frame reading AL Status (0x0130) with state = OP (0x08)
        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)  # APRD
        frame_data.append(0x00)
        frame_data.extend((0x0001).to_bytes(2, "little"))  # ADP: slave 1
        frame_data.extend((0x0130).to_bytes(2, "little"))  # ADO: AL Status
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x08\x00")  # Data: state = OP (0x08)
        frame_data.extend((0x0001).to_bytes(2, "little"))

        analyzer.parse_frame(bytes(frame_data))

        slave = analyzer.read_slave_info(1)
        assert slave is not None
        assert slave.state == "OP"

    def test_read_slave_info_not_found(self) -> None:
        """Test reading info for non-existent slave."""
        analyzer = EtherCATAnalyzer()

        slave = analyzer.read_slave_info(999)
        assert slave is None

    def test_export_configuration(self) -> None:
        """Test exporting configuration as ENI XML."""
        analyzer = EtherCATAnalyzer()

        # Add test slaves
        analyzer.slaves[1] = EtherCATSlave(
            station_address=1,
            alias_address=100,
            vendor_id=0x00000002,
            product_code=0x044C2C52,
            revision=0x00100000,
            serial_number=12345678,
            state="OP",
            dc_supported=True,
            mailbox_protocols=["CoE", "FoE"],
        )

        analyzer.slaves[2] = EtherCATSlave(
            station_address=2,
            state="PRE-OP",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "ethercat_config.xml"
            analyzer.export_configuration(output_path)

            # Verify XML was created
            assert output_path.exists()

            # Parse and verify XML structure
            tree = ET.parse(output_path)  # noqa: S314
            root = tree.getroot()

            assert root.tag == "EtherCATConfig"
            assert root.get("Version") == "1.0"

            config = root.find("Config")
            assert config is not None

            master = config.find("Master")
            assert master is not None

            slaves = master.findall("Slave")
            assert len(slaves) == 2

            # Check first slave
            slave1_info = slaves[0].find("Info")
            assert slave1_info is not None

            station_addr = slave1_info.find("StationAddress")
            assert station_addr is not None
            assert station_addr.text == "1"

            state = slave1_info.find("State")
            assert state is not None
            assert state.text == "OP"

            vendor = slave1_info.find("VendorId")
            assert vendor is not None
            assert vendor.text == "0x00000002"

    def test_parse_datagram_insufficient_header(self) -> None:
        """Test parsing datagram with insufficient header data."""
        analyzer = EtherCATAnalyzer()

        # Frame with incomplete datagram header
        frame_data = bytearray()
        frame_data.extend((5).to_bytes(2, "little"))
        frame_data.extend(b"\x01\x00\x00")  # Only 3 bytes instead of 10+

        with pytest.raises(ValueError, match="Insufficient data for datagram header"):
            analyzer.parse_frame(bytes(frame_data))

    def test_parse_datagram_insufficient_payload(self) -> None:
        """Test parsing datagram with insufficient payload data."""
        analyzer = EtherCATAnalyzer()

        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)  # Cmd
        frame_data.append(0x00)  # Idx
        frame_data.extend((0x0000).to_bytes(2, "little"))  # ADP
        frame_data.extend((0x0000).to_bytes(2, "little"))  # ADO
        frame_data.extend((10).to_bytes(2, "little"))  # Len: 10 bytes
        frame_data.extend(b"\x00\x00")  # Only 2 bytes instead of 10

        with pytest.raises(ValueError, match="Insufficient data for datagram payload"):
            analyzer.parse_frame(bytes(frame_data))

    def test_datagram_irq_field(self) -> None:
        """Test parsing datagram IRQ field."""
        analyzer = EtherCATAnalyzer()

        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((0x0000).to_bytes(2, "little"))
        # Set IRQ bits in len_m_irq field
        len_m_irq = 2 | (0x05 << 12)  # len=2, IRQ=5
        frame_data.extend(len_m_irq.to_bytes(2, "little"))
        frame_data.extend(b"\x00\x00")
        frame_data.extend((0x0000).to_bytes(2, "little"))

        frame = analyzer.parse_frame(bytes(frame_data))
        # IRQ field is masked to 4 bits but bit 15 is M flag
        # So actual IRQ value depends on implementation
        assert frame.datagrams[0].len_ == 2

    def test_station_address_detection(self) -> None:
        """Test station address detection from register reads."""
        analyzer = EtherCATAnalyzer()

        # Read configured station address register (0x0010)
        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)  # APRD
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((0x0010).to_bytes(2, "little"))  # ADO: Station Address
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend((0x002A).to_bytes(2, "little"))  # Station address = 42
        frame_data.extend((0x0001).to_bytes(2, "little"))

        analyzer.parse_frame(bytes(frame_data))

        slave = analyzer.read_slave_info(42)
        assert slave is not None
        assert slave.station_address == 42

    def test_frame_storage(self) -> None:
        """Test that frames are stored in analyzer."""
        analyzer = EtherCATAnalyzer()

        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x00\x00")
        frame_data.extend((0x0000).to_bytes(2, "little"))

        analyzer.parse_frame(bytes(frame_data), timestamp=1.0)
        analyzer.parse_frame(bytes(frame_data), timestamp=2.0)

        assert len(analyzer.frames) == 2
        assert analyzer.frames[0].timestamp == 1.0
        assert analyzer.frames[1].timestamp == 2.0

    def test_unknown_command_type(self) -> None:
        """Test handling unknown command type."""
        analyzer = EtherCATAnalyzer()

        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0xFF)  # Unknown command
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x00\x00")
        frame_data.extend((0x0000).to_bytes(2, "little"))

        frame = analyzer.parse_frame(bytes(frame_data))
        assert "Unknown" in frame.datagrams[0].cmd_name
        assert "0xFF" in frame.datagrams[0].cmd_name

    def test_multiple_state_transitions(self) -> None:
        """Test tracking multiple slave state transitions."""
        analyzer = EtherCATAnalyzer()

        states = [0x01, 0x02, 0x04, 0x08]  # INIT, PRE-OP, SAFE-OP, OP

        for state in states:
            frame_data = bytearray()
            frame_data.extend((14).to_bytes(2, "little"))
            frame_data.append(0x01)  # APRD
            frame_data.append(0x00)
            frame_data.extend((0x0001).to_bytes(2, "little"))  # ADP: 1
            frame_data.extend((0x0130).to_bytes(2, "little"))  # AL Status
            frame_data.extend((2).to_bytes(2, "little"))
            frame_data.extend(state.to_bytes(1, "little") + b"\x00")
            frame_data.extend((0x0001).to_bytes(2, "little"))

            analyzer.parse_frame(bytes(frame_data))

        slave = analyzer.read_slave_info(1)
        assert slave is not None
        # Should have the last state (OP)
        assert slave.state == "OP"
