"""Unit tests for PROFINET protocol analyzer.

Tests PROFINET frame parsing, device discovery, RT/IRT decoding, and topology export.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from oscura.analyzers.protocols.industrial.profinet import (
    ProfinetAnalyzer,
    ProfinetDevice,
    ProfinetFrame,
)


class TestProfinetAnalyzer:
    """Tests for ProfinetAnalyzer class."""

    def test_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = ProfinetAnalyzer()
        assert analyzer.frames == []
        assert analyzer.devices == {}

    def test_classify_frame_id(self) -> None:
        """Test frame ID classification."""
        analyzer = ProfinetAnalyzer()

        # RT_CLASS_1 (cyclic real-time data)
        assert analyzer._classify_frame_id(0x8000) == "RT_CLASS_1"
        assert analyzer._classify_frame_id(0x8100) == "RT_CLASS_1"
        assert analyzer._classify_frame_id(0xBFFF) == "RT_CLASS_1"

        # RT_CLASS_UDP
        assert analyzer._classify_frame_id(0xC000) == "RT_CLASS_UDP"
        assert analyzer._classify_frame_id(0xFBFF) == "RT_CLASS_UDP"

        # RT_CLASS_2 (IRT)
        assert analyzer._classify_frame_id(0xFC00) == "RT_CLASS_2"
        assert analyzer._classify_frame_id(0xFCFF) == "RT_CLASS_2"

        # RT_CLASS_3 (IRT with fragmentation)
        assert analyzer._classify_frame_id(0xFD00) == "RT_CLASS_3"
        assert analyzer._classify_frame_id(0xFDFF) == "RT_CLASS_3"

        # PTCP
        assert analyzer._classify_frame_id(0xFF20) == "PTCP"
        assert analyzer._classify_frame_id(0xFF41) == "PTCP"
        assert analyzer._classify_frame_id(0xFF8F) == "PTCP"

        # Reserved
        assert analyzer._classify_frame_id(0x0000) == "Reserved"
        assert analyzer._classify_frame_id(0x7FFF) == "Reserved"

    def test_parse_rt_frame(self) -> None:
        """Test Real-Time frame parsing."""
        analyzer = ProfinetAnalyzer()

        # RT frame with cycle counter, data status, and I/O data
        frame_id = 0x8000
        rt_data = bytes(
            [
                0x12,
                0x34,  # Cycle counter = 0x1234 (4660)
                0xEC,  # Data status = 0xEC (PRIMARY, REDUNDANCY, RUN, VALID)
                0x01,
                0x02,
                0x03,
                0x04,  # I/O data
            ]
        )

        result = analyzer._parse_rt_frame(frame_id, rt_data)

        assert result["frame_id"] == 0x8000
        assert result["cycle_counter"] == 0x1234
        assert result["data_status"] == 0xEC
        assert result["data_status_flags"]["primary"] is True
        assert result["data_status_flags"]["redundancy"] is True
        assert result["data_status_flags"]["data_valid"] is True
        assert result["data_status_flags"]["provider_state"] == "RUN"
        assert result["io_data"] == "01020304"
        assert result["io_data_length"] == 4

    def test_parse_rt_frame_stop_state(self) -> None:
        """Test RT frame with STOP state."""
        analyzer = ProfinetAnalyzer()

        frame_id = 0x8100
        rt_data = bytes(
            [
                0x00,
                0x01,  # Cycle counter
                0x20,  # Data status: VALID, STOP
                0xFF,  # I/O data
            ]
        )

        result = analyzer._parse_rt_frame(frame_id, rt_data)

        assert result["data_status_flags"]["provider_state"] == "STOP"
        assert result["data_status_flags"]["data_valid"] is True
        assert result["data_status_flags"]["primary"] is False

    def test_parse_rt_frame_too_short(self) -> None:
        """Test RT frame that is too short."""
        analyzer = ProfinetAnalyzer()

        result = analyzer._parse_rt_frame(0x8000, bytes([0x12, 0x34]))

        assert "error" in result
        assert "too short" in result["error"].lower()

    def test_parse_frame_basic_ethernet(self) -> None:
        """Test parsing complete Ethernet frame."""
        analyzer = ProfinetAnalyzer()

        # Construct Ethernet frame: Dest MAC + Source MAC + EtherType + PROFINET data
        dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x01])
        source_mac = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
        ethertype = bytes([0x88, 0x92])  # PROFINET EtherType

        # PROFINET RT frame
        frame_id = bytes([0x80, 0x00])  # RT_CLASS_1
        cycle_counter = bytes([0x00, 0x10])
        data_status = bytes([0x35])  # VALID + RUN
        io_data = bytes([0xAA, 0xBB, 0xCC, 0xDD])

        ethernet_frame = (
            dest_mac + source_mac + ethertype + frame_id + cycle_counter + data_status + io_data
        )

        frame = analyzer.parse_frame(ethernet_frame, timestamp=1.5)

        assert frame.timestamp == 1.5
        assert frame.frame_type == "RT_CLASS_1"
        assert frame.frame_id == 0x8000
        assert frame.source_mac == "00:11:22:33:44:55"
        assert frame.dest_mac == "01:0e:cf:00:00:01"
        assert frame.cycle_counter == 0x0010
        assert frame.data_status == 0x35
        assert len(analyzer.frames) == 1

    def test_parse_frame_vlan_tagged(self) -> None:
        """Test parsing VLAN-tagged Ethernet frame."""
        analyzer = ProfinetAnalyzer()

        dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x01])
        source_mac = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
        vlan_tag = bytes([0x81, 0x00])  # VLAN EtherType
        vlan_id = bytes([0x00, 0x64])  # VLAN ID = 100
        ethertype = bytes([0x88, 0x92])  # PROFINET EtherType

        frame_id = bytes([0x80, 0x00])
        rt_data = bytes([0x00, 0x01, 0x35, 0xAA])

        ethernet_frame = dest_mac + source_mac + vlan_tag + vlan_id + ethertype + frame_id + rt_data

        frame = analyzer.parse_frame(ethernet_frame, timestamp=2.0)

        assert frame.frame_type == "RT_CLASS_1"
        assert frame.frame_id == 0x8000

    def test_parse_frame_invalid_ethertype(self) -> None:
        """Test parsing frame with invalid EtherType."""
        analyzer = ProfinetAnalyzer()

        # Wrong EtherType
        ethernet_frame = bytes([0x00] * 12 + [0x08, 0x00] + [0x00] * 10)

        with pytest.raises(ValueError, match="Not a PROFINET frame"):
            analyzer.parse_frame(ethernet_frame)

    def test_parse_frame_too_short(self) -> None:
        """Test parsing frame that is too short."""
        analyzer = ProfinetAnalyzer()

        with pytest.raises(ValueError, match="too short"):
            analyzer.parse_frame(bytes([0x00] * 10))

    def test_discover_devices_empty(self) -> None:
        """Test device discovery with no frames."""
        analyzer = ProfinetAnalyzer()
        devices = analyzer.discover_devices()
        assert devices == []

    def test_export_topology(self) -> None:
        """Test topology export."""
        analyzer = ProfinetAnalyzer()

        # Add a mock device
        device = ProfinetDevice(
            mac_address="00:11:22:33:44:55",
            device_name="TestDevice",
            device_type="IO-Device",
            vendor_id=0x002A,
            device_id=0x0001,
            station_type="IO-Device",
            ip_address="192.168.1.100",
            subnet_mask="255.255.255.0",
            gateway="192.168.1.1",
        )
        analyzer.devices["00:11:22:33:44:55"] = device

        # Add mock frame
        dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x01])
        source_mac = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
        ethertype = bytes([0x88, 0x92])
        frame_id = bytes([0x80, 0x00])
        rt_data = bytes([0x00, 0x01, 0x35, 0xAA])

        ethernet_frame = dest_mac + source_mac + ethertype + frame_id + rt_data
        analyzer.parse_frame(ethernet_frame, timestamp=1.0)

        # Export topology
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "topology.json"
            analyzer.export_topology(output_path)

            assert output_path.exists()

            # Read and verify
            import json

            with output_path.open() as f:
                topology = json.load(f)

            assert topology["network_type"] == "PROFINET IO"
            assert len(topology["devices"]) == 1
            assert topology["devices"][0]["mac_address"] == "00:11:22:33:44:55"
            assert topology["devices"][0]["device_name"] == "TestDevice"
            assert topology["frame_count"] == 1
            assert "RT_CLASS_1" in topology["frame_types"]

    def test_get_frame_type_statistics(self) -> None:
        """Test frame type statistics."""
        analyzer = ProfinetAnalyzer()

        # Parse multiple frames with different types
        def create_rt_frame(frame_id_val: int) -> bytes:
            dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x01])
            source_mac = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
            ethertype = bytes([0x88, 0x92])
            frame_id = frame_id_val.to_bytes(2, "big")
            rt_data = bytes([0x00, 0x01, 0x35, 0xAA])
            return dest_mac + source_mac + ethertype + frame_id + rt_data

        analyzer.parse_frame(create_rt_frame(0x8000), timestamp=1.0)
        analyzer.parse_frame(create_rt_frame(0x8000), timestamp=1.1)
        analyzer.parse_frame(create_rt_frame(0xFC00), timestamp=1.2)

        stats = analyzer._get_frame_type_statistics()

        assert stats["RT_CLASS_1"] == 2
        assert stats["RT_CLASS_2"] == 1


class TestProfinetFrame:
    """Tests for ProfinetFrame dataclass."""

    def test_frame_creation(self) -> None:
        """Test frame creation."""
        frame = ProfinetFrame(
            timestamp=1.5,
            frame_type="RT_CLASS_1",
            frame_id=0x8000,
            source_mac="00:11:22:33:44:55",
            dest_mac="01:0e:cf:00:00:01",
            cycle_counter=100,
            data_status=0x35,
            payload=b"\xaa\xbb\xcc",
            decoded={"test": "data"},
        )

        assert frame.timestamp == 1.5
        assert frame.frame_type == "RT_CLASS_1"
        assert frame.frame_id == 0x8000
        assert frame.cycle_counter == 100
        assert frame.data_status == 0x35
        assert frame.payload == b"\xaa\xbb\xcc"
        assert frame.decoded["test"] == "data"


class TestProfinetDevice:
    """Tests for ProfinetDevice dataclass."""

    def test_device_creation(self) -> None:
        """Test device creation."""
        device = ProfinetDevice(
            mac_address="00:11:22:33:44:55",
            device_name="MyDevice",
            device_type="IO-Controller",
            vendor_id=0x002A,
            device_id=0x0001,
            station_type="IO-Controller",
            ip_address="192.168.1.100",
            subnet_mask="255.255.255.0",
            gateway="192.168.1.1",
        )

        assert device.mac_address == "00:11:22:33:44:55"
        assert device.device_name == "MyDevice"
        assert device.vendor_id == 0x002A
        assert device.device_id == 0x0001
        assert device.station_type == "IO-Controller"
        assert device.ip_address == "192.168.1.100"
        assert device.modules == []

    def test_device_default_values(self) -> None:
        """Test device with default values."""
        device = ProfinetDevice(mac_address="00:11:22:33:44:55")

        assert device.device_name is None
        assert device.device_type is None
        assert device.vendor_id is None
        assert device.station_type == "DEVICE"
        assert device.modules == []


class TestProfinetDCPIntegration:
    """Tests for DCP frame parsing integration."""

    def test_parse_dcp_frame(self) -> None:
        """Test parsing DCP frame through analyzer."""
        analyzer = ProfinetAnalyzer()

        # DCP Identify request (Frame ID 0xFEFC)
        dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x00])
        source_mac = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
        ethertype = bytes([0x88, 0x92])
        frame_id = bytes([0xFE, 0xFC])
        dcp_data = bytes(
            [
                0x03,  # Service ID: Identify
                0x00,  # Service Type: Request
                0x00,
                0x00,
                0x00,
                0x01,  # Transaction ID
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

        ethernet_frame = dest_mac + source_mac + ethertype + frame_id + dcp_data

        frame = analyzer.parse_frame(ethernet_frame, timestamp=1.0)

        assert frame.frame_type == "DCP"
        assert frame.frame_id == 0xFEFC
        assert "service" in frame.decoded
        assert frame.decoded["service"] == "Identify"

    def test_update_device_info_from_dcp(self) -> None:
        """Test device information extraction from DCP response."""
        analyzer = ProfinetAnalyzer()

        # DCP response with device info
        dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x00])
        source_mac = bytes([0x00, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE])
        ethertype = bytes([0x88, 0x92])
        frame_id = bytes([0xFE, 0xFD])
        device_name = b"MyDevice"

        dcp_data = (
            bytes(
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
                    0x14,  # DCPDataLength: 20
                    # Block 1: Name of Station
                    0x02,  # Option: Device Properties
                    0x02,  # Suboption: Name of Station
                    0x00,
                    0x08,  # Block Length: 8
                ]
            )
            + device_name
            + bytes(
                [
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

        ethernet_frame = dest_mac + source_mac + ethertype + frame_id + dcp_data

        frame = analyzer.parse_frame(ethernet_frame, timestamp=1.0)

        assert frame.frame_type == "DCP"
        devices = analyzer.discover_devices()
        assert len(devices) == 1
        assert devices[0].device_name == "MyDevice"
        assert devices[0].vendor_id == 0x002A
        assert devices[0].device_id == 0x0001


class TestProfinetIntegration:
    """Integration tests for PROFINET analyzer."""

    def test_full_workflow(self) -> None:
        """Test complete workflow: parse frames, discover devices, export."""
        analyzer = ProfinetAnalyzer()

        # Create multiple RT frames from different devices
        def create_frame(source_mac_bytes: bytes, frame_id_val: int) -> bytes:
            dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x01])
            ethertype = bytes([0x88, 0x92])
            frame_id = frame_id_val.to_bytes(2, "big")
            rt_data = bytes([0x00, 0x01, 0x35, 0xAA, 0xBB])
            return dest_mac + source_mac_bytes + ethertype + frame_id + rt_data

        # Parse frames from two different devices
        mac1 = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
        mac2 = bytes([0x00, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE])

        analyzer.parse_frame(create_frame(mac1, 0x8000), timestamp=0.0)
        analyzer.parse_frame(create_frame(mac1, 0x8001), timestamp=0.001)
        analyzer.parse_frame(create_frame(mac2, 0x8000), timestamp=0.002)
        analyzer.parse_frame(create_frame(mac2, 0xFC00), timestamp=0.003)

        assert len(analyzer.frames) == 4

        # Get statistics
        stats = analyzer._get_frame_type_statistics()
        assert stats["RT_CLASS_1"] == 3
        assert stats["RT_CLASS_2"] == 1

        # Export topology
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "network.json"
            analyzer.export_topology(output_path)
            assert output_path.exists()

    def test_mixed_frame_types(self) -> None:
        """Test handling mixed RT, IRT, and PTCP frames."""
        analyzer = ProfinetAnalyzer()

        def create_frame_with_id(frame_id_val: int) -> bytes:
            dest_mac = bytes([0x01, 0x0E, 0xCF, 0x00, 0x00, 0x01])
            source_mac = bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])
            ethertype = bytes([0x88, 0x92])
            frame_id = frame_id_val.to_bytes(2, "big")
            # Generic payload
            payload = bytes([0x00, 0x01, 0x35, 0xAA])
            return dest_mac + source_mac + ethertype + frame_id + payload

        # RT_CLASS_1
        analyzer.parse_frame(create_frame_with_id(0x8000), timestamp=0.0)

        # RT_CLASS_2 (IRT)
        analyzer.parse_frame(create_frame_with_id(0xFC00), timestamp=0.001)

        # RT_CLASS_3 (IRT with fragmentation)
        analyzer.parse_frame(create_frame_with_id(0xFD00), timestamp=0.002)

        # PTCP
        analyzer.parse_frame(create_frame_with_id(0xFF41), timestamp=0.003)

        assert len(analyzer.frames) == 4
        assert analyzer.frames[0].frame_type == "RT_CLASS_1"
        assert analyzer.frames[1].frame_type == "RT_CLASS_2"
        assert analyzer.frames[2].frame_type == "RT_CLASS_3"
        assert analyzer.frames[3].frame_type == "PTCP"
