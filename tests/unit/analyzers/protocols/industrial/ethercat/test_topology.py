"""Tests for EtherCAT topology discovery.

Test coverage:
- Network type detection (line, ring, star)
- Slave ordering
- Open port detection
"""

from oscura.analyzers.protocols.industrial.ethercat.analyzer import EtherCATAnalyzer
from oscura.analyzers.protocols.industrial.ethercat.topology import (
    TopologyAnalyzer,
    TopologyInfo,
)


class TestTopologyInfo:
    """Test TopologyInfo dataclass."""

    def test_topology_info_creation(self) -> None:
        """Test creating topology info."""
        info = TopologyInfo(
            network_type="line",
            slave_count=3,
            slave_addresses=[0, 1, 2],
            open_ports=[0, 2],
        )

        assert info.network_type == "line"
        assert info.slave_count == 3
        assert len(info.slave_addresses) == 3
        assert len(info.open_ports) == 2


class TestTopologyAnalyzer:
    """Test topology analyzer."""

    def test_analyzer_initialization(self) -> None:
        """Test initializing topology analyzer."""
        ethercat_analyzer = EtherCATAnalyzer()
        topology = TopologyAnalyzer(ethercat_analyzer)

        assert topology.analyzer is ethercat_analyzer

    def test_detect_line_topology(self) -> None:
        """Test detecting line topology (2 open ports)."""
        analyzer = EtherCATAnalyzer()

        # Create frames reading port descriptor with open ports
        # Port descriptor at 0x0007 - simulate first slave (port 0 open)
        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)  # APRD
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))  # Slave 0
        frame_data.extend((0x0007).to_bytes(2, "little"))  # Port descriptor
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x00\x02")  # Port 0 open (0), Port 1 connected (2)
        frame_data.extend((0x0001).to_bytes(2, "little"))

        analyzer.parse_frame(bytes(frame_data))

        # Last slave (port 3 open)
        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)  # APRD
        frame_data.append(0x00)
        frame_data.extend((0x0002).to_bytes(2, "little"))  # Slave 2
        frame_data.extend((0x0007).to_bytes(2, "little"))  # Port descriptor
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x01\x02")  # Port open pattern
        frame_data.extend((0x0001).to_bytes(2, "little"))

        analyzer.parse_frame(bytes(frame_data))

        topology = TopologyAnalyzer(analyzer)
        network_type = topology.detect_network_type()

        assert network_type == "line"

    def test_detect_ring_topology(self) -> None:
        """Test detecting ring topology (no open ports)."""
        analyzer = EtherCATAnalyzer()

        # No frames with port descriptor indicating open ports
        # Ring topology has all ports connected
        topology = TopologyAnalyzer(analyzer)
        network_type = topology.detect_network_type()

        # With no port information, should detect as ring (no open ports)
        assert network_type == "ring"

    def test_detect_star_topology(self) -> None:
        """Test detecting star topology (>2 open ports)."""
        analyzer = EtherCATAnalyzer()

        # Create multiple slaves with open ports (star topology)
        for i in range(4):
            frame_data = bytearray()
            frame_data.extend((14).to_bytes(2, "little"))
            frame_data.append(0x01)  # APRD
            frame_data.append(0x00)
            frame_data.extend(i.to_bytes(2, "little"))
            frame_data.extend((0x0007).to_bytes(2, "little"))  # Port descriptor
            frame_data.extend((2).to_bytes(2, "little"))
            frame_data.extend(b"\x00\x02")  # Port pattern indicating open port
            frame_data.extend((0x0001).to_bytes(2, "little"))

            analyzer.parse_frame(bytes(frame_data))

        topology = TopologyAnalyzer(analyzer)
        network_type = topology.detect_network_type()

        assert network_type == "star"

    def test_analyze_complete_topology(self) -> None:
        """Test analyzing complete topology."""
        analyzer = EtherCATAnalyzer()

        # Parse frames to discover slaves
        for i in range(3):
            frame_data = bytearray()
            frame_data.extend((14).to_bytes(2, "little"))
            frame_data.append(0x01)  # APRD
            frame_data.append(0x00)
            frame_data.extend(i.to_bytes(2, "little"))
            frame_data.extend((0x0000).to_bytes(2, "little"))
            frame_data.extend((2).to_bytes(2, "little"))
            frame_data.extend(b"\x00\x00")
            frame_data.extend((0x0001).to_bytes(2, "little"))  # WKC > 0

            analyzer.parse_frame(bytes(frame_data))

        topology = TopologyAnalyzer(analyzer)
        info = topology.analyze()

        assert info.slave_count == 3
        assert len(info.slave_addresses) == 3
        assert info.slave_addresses == [0, 1, 2]
        assert info.network_type in ["line", "ring", "star", "unknown"]

    def test_get_slave_order(self) -> None:
        """Test getting slaves in physical order."""
        analyzer = EtherCATAnalyzer()

        # Parse frames in order
        for i in [2, 0, 1]:  # Parse out of order
            frame_data = bytearray()
            frame_data.extend((14).to_bytes(2, "little"))
            frame_data.append(0x01)  # APRD
            frame_data.append(0x00)
            frame_data.extend(i.to_bytes(2, "little"))
            frame_data.extend((0x0000).to_bytes(2, "little"))
            frame_data.extend((2).to_bytes(2, "little"))
            frame_data.extend(b"\x00\x00")
            frame_data.extend((0x0001).to_bytes(2, "little"))

            analyzer.parse_frame(bytes(frame_data))

        topology = TopologyAnalyzer(analyzer)
        order = topology.get_slave_order()

        # Should return sorted order
        assert order == [0, 1, 2]

    def test_empty_topology(self) -> None:
        """Test analyzing topology with no slaves."""
        analyzer = EtherCATAnalyzer()
        topology = TopologyAnalyzer(analyzer)

        info = topology.analyze()

        assert info.slave_count == 0
        assert info.slave_addresses == []
        assert info.network_type == "ring"  # No open ports detected

    def test_single_slave_topology(self) -> None:
        """Test topology with single slave."""
        analyzer = EtherCATAnalyzer()

        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x00\x00")
        frame_data.extend((0x0001).to_bytes(2, "little"))

        analyzer.parse_frame(bytes(frame_data))

        topology = TopologyAnalyzer(analyzer)
        info = topology.analyze()

        assert info.slave_count == 1
        assert info.slave_addresses == [0]

    def test_detect_unknown_topology(self) -> None:
        """Test detecting unknown topology (1 open port)."""
        analyzer = EtherCATAnalyzer()

        # Single open port (unusual configuration)
        frame_data = bytearray()
        frame_data.extend((14).to_bytes(2, "little"))
        frame_data.append(0x01)
        frame_data.append(0x00)
        frame_data.extend((0x0000).to_bytes(2, "little"))
        frame_data.extend((0x0007).to_bytes(2, "little"))  # Port descriptor
        frame_data.extend((2).to_bytes(2, "little"))
        frame_data.extend(b"\x01\x02")  # One open port
        frame_data.extend((0x0001).to_bytes(2, "little"))

        analyzer.parse_frame(bytes(frame_data))

        topology = TopologyAnalyzer(analyzer)
        network_type = topology.detect_network_type()

        assert network_type == "unknown"
