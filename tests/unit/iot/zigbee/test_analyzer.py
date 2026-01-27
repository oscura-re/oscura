"""Tests for Zigbee protocol analyzer.

Tests NWK layer parsing, APS layer decoding, ZCL frame parsing,
topology discovery, and device management.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscura.iot.zigbee import ZigbeeAnalyzer, ZigbeeDevice, ZigbeeFrame


class TestZigbeeFrame:
    """Test ZigbeeFrame dataclass."""

    def test_frame_creation(self) -> None:
        """Test basic frame creation."""
        frame = ZigbeeFrame(
            timestamp=1.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
        )

        assert frame.timestamp == 1.0
        assert frame.frame_type == "DATA"
        assert frame.source_address == 0x1234
        assert frame.dest_address == 0x0000
        assert frame.sequence_number == 0
        assert frame.payload == b""

    def test_frame_with_payload(self) -> None:
        """Test frame with payload data."""
        payload = bytes([0x01, 0x02, 0x03])
        frame = ZigbeeFrame(
            timestamp=2.0,
            frame_type="COMMAND",
            source_address=0x5678,
            dest_address=0x0000,
            payload=payload,
            sequence_number=42,
        )

        assert frame.payload == payload
        assert frame.sequence_number == 42

    def test_frame_with_ieee_addresses(self) -> None:
        """Test frame with IEEE addresses."""
        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
            source_ieee=0x0013A20040A12345,
            dest_ieee=0x0013A20040A00000,
        )

        assert frame.source_ieee == 0x0013A20040A12345
        assert frame.dest_ieee == 0x0013A20040A00000


class TestZigbeeDevice:
    """Test ZigbeeDevice dataclass."""

    def test_device_creation(self) -> None:
        """Test basic device creation."""
        device = ZigbeeDevice(short_address=0x1234)

        assert device.short_address == 0x1234
        assert device.ieee_address is None
        assert device.device_type == "unknown"
        assert device.clusters == []

    def test_device_with_clusters(self) -> None:
        """Test device with cluster support."""
        device = ZigbeeDevice(
            short_address=0x5678,
            ieee_address=0x0013A20040A12345,
            device_type="end_device",
            clusters=[0x0006, 0x0008, 0x0402],
        )

        assert device.clusters == [0x0006, 0x0008, 0x0402]
        assert device.device_type == "end_device"


class TestZigbeeAnalyzer:
    """Test ZigbeeAnalyzer main functionality."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = ZigbeeAnalyzer()

        assert len(analyzer.frames) == 0
        assert len(analyzer.devices) == 0
        assert len(analyzer.network_keys) == 0

    def test_add_frame(self) -> None:
        """Test adding frames to analyzer."""
        analyzer = ZigbeeAnalyzer()
        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
        )

        analyzer.add_frame(frame)

        assert len(analyzer.frames) == 1
        assert len(analyzer.devices) == 2  # Source and destination
        assert 0x1234 in analyzer.devices
        assert 0x0000 in analyzer.devices

    def test_add_frame_with_ieee(self) -> None:
        """Test adding frame updates IEEE addresses."""
        analyzer = ZigbeeAnalyzer()
        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
            source_ieee=0x0013A20040A12345,
        )

        analyzer.add_frame(frame)

        assert analyzer.devices[0x1234].ieee_address == 0x0013A20040A12345

    def test_add_network_key(self) -> None:
        """Test adding network key."""
        analyzer = ZigbeeAnalyzer()
        key = bytes([0x01] * 16)

        analyzer.add_network_key(key)

        assert len(analyzer.network_keys) == 1
        assert analyzer.network_keys[0] == key

    def test_add_network_key_invalid_length(self) -> None:
        """Test adding network key with invalid length."""
        analyzer = ZigbeeAnalyzer()
        key = bytes([0x01] * 8)  # Wrong length

        with pytest.raises(ValueError, match="must be 16 bytes"):
            analyzer.add_network_key(key)


class TestNWKLayerParsing:
    """Test NWK layer parsing."""

    def test_parse_basic_nwk_frame(self) -> None:
        """Test parsing basic NWK frame."""
        analyzer = ZigbeeAnalyzer()

        # Basic NWK frame: frame control (2), dest (2), src (2), radius (1), seq (1)
        data = bytes(
            [
                0x08,
                0x00,  # Frame control (data frame)
                0x00,
                0x00,  # Destination address (coordinator)
                0x34,
                0x12,  # Source address 0x1234
                0x1E,  # Radius (30)
                0x01,  # Sequence number
            ]
        )

        result = analyzer.parse_nwk_layer(data)

        assert result["frame_type"] == "DATA"
        assert result["dest_address"] == 0x0000
        assert result["source_address"] == 0x1234
        assert result["radius"] == 30
        assert result["sequence"] == 1
        assert result["security"] is False

    def test_parse_nwk_frame_with_ieee_addresses(self) -> None:
        """Test parsing NWK frame with IEEE addresses."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x08,
                0x18,  # Frame control (dest and src IEEE present)
                0x00,
                0x00,  # Destination address
                0x34,
                0x12,  # Source address
                0x1E,  # Radius
                0x01,  # Sequence
                # Destination IEEE (8 bytes, little-endian)
                0x00,
                0x00,
                0xA0,
                0x40,
                0x20,
                0xA2,
                0x13,
                0x00,
                # Source IEEE (8 bytes, little-endian)
                0x45,
                0x23,
                0xA1,
                0x40,
                0x20,
                0xA2,
                0x13,
                0x00,
            ]
        )

        result = analyzer.parse_nwk_layer(data)

        # 0x00, 0x00, 0xA0, 0x40, 0x20, 0xA2, 0x13, 0x00 in little-endian = 0x0013A22040A00000
        assert result["dest_ieee"] == 0x0013A22040A00000
        # 0x45, 0x23, 0xA1, 0x40, 0x20, 0xA2, 0x13, 0x00 in little-endian = 0x0013A22040A12345
        assert result["source_ieee"] == 0x0013A22040A12345

    def test_parse_nwk_frame_insufficient_data(self) -> None:
        """Test parsing NWK frame with insufficient data."""
        analyzer = ZigbeeAnalyzer()
        data = bytes([0x08, 0x00, 0x00])  # Too short

        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.parse_nwk_layer(data)

    def test_parse_nwk_frame_with_security(self) -> None:
        """Test parsing NWK frame with security header."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x08,
                0x02,  # Frame control (security enabled)
                0x00,
                0x00,  # Destination
                0x34,
                0x12,  # Source
                0x1E,  # Radius
                0x01,  # Sequence
                # Security header (minimum 5 bytes)
                0x05,  # Security control (level 5)
                0x01,
                0x00,
                0x00,
                0x00,  # Frame counter
            ]
        )

        result = analyzer.parse_nwk_layer(data)

        assert result["security"] is True
        assert result["security_data"] is not None
        assert result["security_data"]["security_level"] == 5


class TestAPSLayerParsing:
    """Test APS layer parsing."""

    def test_parse_basic_aps_frame(self) -> None:
        """Test parsing basic APS frame."""
        analyzer = ZigbeeAnalyzer()

        # APS frame with unicast delivery
        data = bytes(
            [
                0x00,  # Frame control (unicast, data frame)
                0x01,  # Destination endpoint
                0x06,
                0x00,  # Cluster ID (On/Off)
                0x04,
                0x01,  # Profile ID (HA - Home Automation)
                0x01,  # Source endpoint
                0x05,  # APS counter
            ]
        )

        result = analyzer.parse_aps_layer(data)

        assert result["frame_type"] == 0
        assert result["dest_endpoint"] == 0x01
        assert result["cluster_id"] == 0x0006
        assert result["cluster_name"] == "On/Off"
        assert result["profile_id"] == 0x0104
        assert result["source_endpoint"] == 0x01
        assert result["aps_counter"] == 0x05

    def test_parse_aps_frame_with_group(self) -> None:
        """Test parsing APS frame with group addressing."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x04,  # Frame control (group delivery)
                0x01,  # Destination endpoint
                0x01,
                0x00,  # Group address
                0x06,
                0x00,  # Cluster ID
                0x04,
                0x01,  # Profile ID
                0x01,  # Source endpoint
                0x05,  # APS counter
            ]
        )

        result = analyzer.parse_aps_layer(data)

        assert result["delivery_mode"] == 1
        assert result["group_address"] == 0x0001

    def test_parse_aps_frame_insufficient_data(self) -> None:
        """Test parsing APS frame with insufficient data."""
        analyzer = ZigbeeAnalyzer()
        data = bytes([0x00])  # Too short

        with pytest.raises(ValueError, match="Insufficient data"):
            analyzer.parse_aps_layer(data)


class TestZCLParsing:
    """Test ZCL frame parsing."""

    def test_parse_onoff_on_command(self) -> None:
        """Test parsing On/Off cluster On command."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x01,  # Frame control (cluster-specific, client to server)
                0x00,  # Transaction sequence
                0x01,  # Command ID (On)
            ]
        )

        result = analyzer.parse_zcl_frame(0x0006, data)

        assert result["cluster_id"] == 0x0006
        assert result["cluster_name"] == "On/Off"
        assert result["command_name"] == "On"
        assert result["frame_type"] == "cluster_specific"

    def test_parse_onoff_off_command(self) -> None:
        """Test parsing On/Off cluster Off command."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x01,  # Frame control
                0x00,  # Transaction sequence
                0x00,  # Command ID (Off)
            ]
        )

        result = analyzer.parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Off"

    def test_parse_onoff_toggle_command(self) -> None:
        """Test parsing On/Off cluster Toggle command."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x01,  # Frame control
                0x00,  # Transaction sequence
                0x02,  # Command ID (Toggle)
            ]
        )

        result = analyzer.parse_zcl_frame(0x0006, data)

        assert result["command_name"] == "Toggle"

    def test_parse_level_control_command(self) -> None:
        """Test parsing Level Control cluster command."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x01,  # Frame control
                0x00,  # Transaction sequence
                0x00,  # Command ID (Move to Level)
                0x80,  # Level (128)
                0x0A,
                0x00,  # Transition time (10)
            ]
        )

        result = analyzer.parse_zcl_frame(0x0008, data)

        assert result["cluster_id"] == 0x0008
        assert result["cluster_name"] == "Level Control"
        assert result["command_name"] == "Move to Level"
        assert result["details"]["level"] == 0x80
        assert result["details"]["transition_time"] == 10

    def test_parse_global_read_attributes(self) -> None:
        """Test parsing global Read Attributes command."""
        analyzer = ZigbeeAnalyzer()

        data = bytes(
            [
                0x00,  # Frame control (global)
                0x01,  # Transaction sequence
                0x00,  # Command ID (Read Attributes)
                0x00,
                0x00,  # Attribute ID 0x0000
                0x05,
                0x00,  # Attribute ID 0x0005
            ]
        )

        result = analyzer.parse_zcl_frame(0x0006, data)

        assert result["frame_type"] == "global"
        assert result["command_name"] == "Read Attributes"
        assert result["details"]["attribute_ids"] == [0x0000, 0x0005]

    def test_parse_zcl_insufficient_data(self) -> None:
        """Test parsing ZCL frame with insufficient data."""
        analyzer = ZigbeeAnalyzer()
        data = bytes([0x01])  # Too short

        result = analyzer.parse_zcl_frame(0x0006, data)

        assert "error" in result
        assert result["cluster_id"] == 0x0006


class TestTopologyDiscovery:
    """Test network topology discovery."""

    def test_discover_basic_topology(self) -> None:
        """Test discovering basic network topology."""
        analyzer = ZigbeeAnalyzer()

        # Add frames from end devices to coordinator
        frames = [
            ZigbeeFrame(
                timestamp=0.0,
                frame_type="DATA",
                source_address=0x1234,
                dest_address=0x0000,
            ),
            ZigbeeFrame(
                timestamp=1.0,
                frame_type="DATA",
                source_address=0x5678,
                dest_address=0x0000,
            ),
        ]

        for frame in frames:
            analyzer.add_frame(frame)

        topology = analyzer.discover_topology()

        assert 0x0000 in topology
        assert 0x1234 in topology[0x0000]
        assert 0x5678 in topology[0x0000]

    def test_discover_topology_identifies_coordinator(self) -> None:
        """Test topology discovery identifies coordinator."""
        analyzer = ZigbeeAnalyzer()

        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
        )
        analyzer.add_frame(frame)
        analyzer.discover_topology()

        assert analyzer.devices[0x0000].device_type == "coordinator"

    def test_discover_topology_identifies_end_device(self) -> None:
        """Test topology discovery identifies end device."""
        analyzer = ZigbeeAnalyzer()

        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
        )
        analyzer.add_frame(frame)
        topology = analyzer.discover_topology()

        assert analyzer.devices[0x1234].device_type == "end_device"
        assert analyzer.devices[0x1234].parent_address == 0x0000

    def test_discover_empty_topology(self) -> None:
        """Test discovering topology with no frames."""
        analyzer = ZigbeeAnalyzer()
        topology = analyzer.discover_topology()

        assert topology == {}


class TestTopologyExport:
    """Test topology export functionality."""

    def test_export_topology_json(self, tmp_path: Path) -> None:
        """Test exporting topology to JSON."""
        analyzer = ZigbeeAnalyzer()

        # Add some frames
        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
            source_ieee=0x0013A20040A12345,
        )
        analyzer.add_frame(frame)
        analyzer.discover_topology()

        output_path = tmp_path / "topology.json"
        analyzer.export_topology(output_path)

        assert output_path.exists()

        # Verify JSON content
        with output_path.open() as f:
            data = json.load(f)

        assert "devices" in data
        assert "topology" in data
        assert "0x0000" in data["topology"]

    def test_export_topology_dot(self, tmp_path: Path) -> None:
        """Test exporting topology to GraphViz DOT."""
        analyzer = ZigbeeAnalyzer()

        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
        )
        analyzer.add_frame(frame)
        analyzer.discover_topology()

        output_path = tmp_path / "topology.json"
        analyzer.export_topology(output_path)

        dot_path = tmp_path / "topology.dot"
        assert dot_path.exists()

        # Verify DOT content
        content = dot_path.read_text()
        assert "digraph ZigbeeNetwork" in content
        assert "0x0000" in content
        assert "0x1234" in content

    def test_export_topology_with_clusters(self, tmp_path: Path) -> None:
        """Test exporting topology with cluster information."""
        analyzer = ZigbeeAnalyzer()

        frame = ZigbeeFrame(
            timestamp=0.0,
            frame_type="DATA",
            source_address=0x1234,
            dest_address=0x0000,
        )
        analyzer.add_frame(frame)

        # Add cluster info
        analyzer.devices[0x1234].clusters = [0x0006, 0x0008]
        analyzer.discover_topology()

        output_path = tmp_path / "topology.json"
        analyzer.export_topology(output_path)

        with output_path.open() as f:
            data = json.load(f)

        device_clusters = data["devices"]["4660"]["clusters"]  # 0x1234 = 4660
        assert len(device_clusters) == 2
        assert device_clusters[0]["name"] == "On/Off"
        assert device_clusters[1]["name"] == "Level Control"


class TestIntegrationScenarios:
    """Test complete integration scenarios."""

    def test_complete_onoff_transaction(self) -> None:
        """Test complete On/Off command transaction."""
        analyzer = ZigbeeAnalyzer()

        # Simulate NWK frame
        nwk_data = bytes(
            [
                0x08,
                0x00,  # Frame control
                0x00,
                0x00,  # Dest address
                0x34,
                0x12,  # Source address
                0x1E,
                0x01,  # Radius, sequence
                # APS layer
                0x00,  # Frame control
                0x01,  # Dest endpoint
                0x06,
                0x00,  # Cluster ID (On/Off)
                0x04,
                0x01,  # Profile ID
                0x01,  # Source endpoint
                0x05,  # APS counter
                # ZCL layer
                0x01,  # Frame control
                0x00,  # Transaction sequence
                0x01,  # Command ID (On)
            ]
        )

        nwk = analyzer.parse_nwk_layer(nwk_data)
        aps = analyzer.parse_aps_layer(nwk["payload"])
        zcl = analyzer.parse_zcl_frame(aps["cluster_id"], aps["payload"])

        assert nwk["frame_type"] == "DATA"
        assert aps["cluster_name"] == "On/Off"
        assert zcl["command_name"] == "On"

    def test_multiple_devices_network(self) -> None:
        """Test network with multiple devices."""
        analyzer = ZigbeeAnalyzer()

        # Add frames from multiple devices
        devices = [0x1234, 0x5678, 0xABCD]
        for i, addr in enumerate(devices):
            frame = ZigbeeFrame(
                timestamp=float(i),
                frame_type="DATA",
                source_address=addr,
                dest_address=0x0000,
            )
            analyzer.add_frame(frame)

        assert len(analyzer.devices) == 4  # 3 devices + coordinator
        assert len(analyzer.frames) == 3

        topology = analyzer.discover_topology()
        assert len(topology[0x0000]) == 3
