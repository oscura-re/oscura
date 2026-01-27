"""Comprehensive unit tests for BLE protocol analyzer.

Tests for src/oscura/analyzers/protocols/ble/analyzer.py

This test suite provides comprehensive coverage of the BLE analyzer module,
including:
- BLE packet parsing and decoding
- Advertising data (AD) structure parsing
- ATT protocol operation decoding
- GATT service/characteristic discovery
- Custom UUID registration
- Export functionality (JSON, CSV)
- Edge cases and error handling

References:
    Bluetooth Core Specification v5.4
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from oscura.analyzers.protocols.ble.analyzer import (
    BLEAnalyzer,
    BLEPacket,
    GATTCharacteristic,
    GATTDescriptor,
    GATTService,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# =============================================================================
# Test Data Generators
# =============================================================================


def create_advertising_packet(
    name: str = "TestDevice",
    flags: int = 0x06,
    service_uuids: list[int] | None = None,
    tx_power: int | None = None,
) -> bytes:
    """Create BLE advertising packet data.

    Args:
        name: Device name.
        flags: Advertising flags.
        service_uuids: List of 16-bit service UUIDs.
        tx_power: Tx power level in dBm.

    Returns:
        Advertising data bytes.
    """
    ad_structures = []

    # Flags
    ad_structures.append(bytes([2, 0x01, flags]))

    # Complete Local Name
    name_bytes = name.encode("utf-8")
    ad_structures.append(bytes([len(name_bytes) + 1, 0x09]) + name_bytes)

    # Service UUIDs
    if service_uuids:
        uuid_bytes = b"".join(uuid.to_bytes(2, "little") for uuid in service_uuids)
        ad_structures.append(bytes([len(uuid_bytes) + 1, 0x03]) + uuid_bytes)

    # Tx Power
    if tx_power is not None:
        ad_structures.append(bytes([2, 0x0A, tx_power & 0xFF]))

    return b"".join(ad_structures)


def create_att_read_request(handle: int) -> bytes:
    """Create ATT Read Request packet.

    Args:
        handle: Attribute handle.

    Returns:
        ATT packet bytes.
    """
    return bytes([0x0A]) + handle.to_bytes(2, "little")


def create_att_read_response(value: bytes) -> bytes:
    """Create ATT Read Response packet.

    Args:
        value: Attribute value.

    Returns:
        ATT packet bytes.
    """
    return bytes([0x0B]) + value


def create_att_read_by_group_type_response(
    attributes: list[tuple[int, int, bytes]],
) -> bytes:
    """Create ATT Read By Group Type Response (service discovery).

    Args:
        attributes: List of (handle, end_handle, uuid_bytes) tuples.

    Returns:
        ATT packet bytes.
    """
    if not attributes:
        return bytes([0x11, 0])

    # Determine length from first attribute
    first_attr = attributes[0]
    attr_len = 2 + 2 + len(first_attr[2])  # handle + end_handle + uuid

    data = bytes([0x11, attr_len])

    for handle, end_handle, uuid_bytes in attributes:
        data += handle.to_bytes(2, "little")
        data += end_handle.to_bytes(2, "little")
        data += uuid_bytes

    return data


def create_att_read_by_type_response(
    attributes: list[tuple[int, bytes]],
) -> bytes:
    """Create ATT Read By Type Response (characteristic discovery).

    Args:
        attributes: List of (handle, data) tuples.

    Returns:
        ATT packet bytes.
    """
    if not attributes:
        return bytes([0x09, 0])

    # Determine length from first attribute
    attr_len = 2 + len(attributes[0][1])  # handle + data

    data = bytes([0x09, attr_len])

    for handle, attr_data in attributes:
        data += handle.to_bytes(2, "little")
        data += attr_data

    return data


# =============================================================================
# BLEPacket Tests
# =============================================================================


class TestBLEPacket:
    """Test BLEPacket dataclass."""

    def test_packet_creation(self):
        """Test creating BLE packet."""
        packet = BLEPacket(
            timestamp=1.23,
            packet_type="ADV_IND",
            source_address="AA:BB:CC:DD:EE:FF",
            data=b"\x02\x01\x06",
        )

        assert packet.timestamp == 1.23
        assert packet.packet_type == "ADV_IND"
        assert packet.source_address == "AA:BB:CC:DD:EE:FF"
        assert packet.data == b"\x02\x01\x06"
        assert packet.dest_address is None
        assert packet.rssi is None
        assert packet.decoded is None

    def test_packet_to_dict(self):
        """Test converting packet to dictionary."""
        packet = BLEPacket(
            timestamp=1.0,
            packet_type="ADV_IND",
            source_address="AA:BB:CC:DD:EE:FF",
            data=b"\x02\x01\x06",
            rssi=-50,
        )

        packet_dict = packet.to_dict()

        assert packet_dict["timestamp"] == 1.0
        assert packet_dict["packet_type"] == "ADV_IND"
        assert packet_dict["source_address"] == "AA:BB:CC:DD:EE:FF"
        assert packet_dict["data"] == "020106"
        assert packet_dict["rssi"] == -50


# =============================================================================
# GATT Data Structure Tests
# =============================================================================


class TestGATTDataStructures:
    """Test GATT service, characteristic, and descriptor dataclasses."""

    def test_gatt_descriptor_creation(self):
        """Test creating GATT descriptor."""
        descriptor = GATTDescriptor(
            uuid="0x2902",
            name="Client Characteristic Configuration",
            handle=10,
            value=b"\x01\x00",
        )

        assert descriptor.uuid == "0x2902"
        assert descriptor.name == "Client Characteristic Configuration"
        assert descriptor.handle == 10
        assert descriptor.value == b"\x01\x00"

    def test_gatt_characteristic_creation(self):
        """Test creating GATT characteristic."""
        char = GATTCharacteristic(
            uuid="0x2A37",
            name="Heart Rate Measurement",
            properties=["read", "notify"],
            handle=8,
            value_handle=9,
        )

        assert char.uuid == "0x2A37"
        assert char.name == "Heart Rate Measurement"
        assert char.properties == ["read", "notify"]
        assert char.handle == 8
        assert char.value_handle == 9

    def test_gatt_service_creation(self):
        """Test creating GATT service."""
        char = GATTCharacteristic(
            uuid="0x2A37",
            name="Heart Rate Measurement",
            properties=["read", "notify"],
            handle=8,
        )

        service = GATTService(
            uuid="0x180D",
            name="Heart Rate",
            characteristics=[char],
            handle_range=(7, 12),
        )

        assert service.uuid == "0x180D"
        assert service.name == "Heart Rate"
        assert len(service.characteristics) == 1
        assert service.handle_range == (7, 12)

    def test_gatt_service_to_dict(self):
        """Test converting GATT service to dictionary."""
        service = GATTService(
            uuid="0x180D",
            name="Heart Rate",
            characteristics=[],
            handle_range=(1, 10),
        )

        service_dict = service.to_dict()

        assert service_dict["uuid"] == "0x180D"
        assert service_dict["name"] == "Heart Rate"
        assert service_dict["handle_range"] == [1, 10]
        assert service_dict["characteristics"] == []


# =============================================================================
# BLEAnalyzer Initialization Tests
# =============================================================================


class TestBLEAnalyzerInit:
    """Test BLEAnalyzer initialization."""

    def test_default_initialization(self):
        """Test analyzer with default initialization."""
        analyzer = BLEAnalyzer()

        assert len(analyzer.packets) == 0
        assert len(analyzer.services) == 0
        assert len(analyzer.custom_uuids) == 0

    def test_custom_uuid_registration(self):
        """Test registering custom UUIDs."""
        analyzer = BLEAnalyzer()
        analyzer.register_custom_uuid("0xABCD", "My Custom Service")

        assert "0XABCD" in analyzer.custom_uuids  # Note: upper() converts 0x to 0X
        assert analyzer.custom_uuids["0XABCD"] == "My Custom Service"

        # Test retrieval
        name = analyzer.get_uuid_name("0xABCD", "service")
        assert name == "My Custom Service"


# =============================================================================
# Advertising Data Parsing Tests
# =============================================================================


class TestAdvertisingDataParsing:
    """Test BLE advertising data parsing."""

    def test_parse_flags(self):
        """Test parsing advertising flags."""
        analyzer = BLEAnalyzer()
        data = b"\x02\x01\x06"  # Length=2, Type=Flags, Value=0x06

        result = analyzer.parse_advertising_data(data)

        assert "flags" in result
        assert result["flags"]["value"] == 0x06
        assert result["flags"]["br_edr_not_supported"] is True
        assert result["flags"]["le_general_discoverable"] is True

    def test_parse_complete_local_name(self):
        """Test parsing complete local name."""
        analyzer = BLEAnalyzer()
        name = "TestDevice"
        name_bytes = name.encode("utf-8")
        data = bytes([len(name_bytes) + 1, 0x09]) + name_bytes

        result = analyzer.parse_advertising_data(data)

        assert "name" in result
        assert result["name"] == "TestDevice"

    def test_parse_tx_power(self):
        """Test parsing Tx power level."""
        analyzer = BLEAnalyzer()
        data = b"\x02\x0a\xf6"  # Length=2, Type=Tx Power, Value=-10 dBm

        result = analyzer.parse_advertising_data(data)

        assert "tx_power" in result
        assert result["tx_power"] == -10

    def test_parse_service_uuids(self):
        """Test parsing 16-bit service UUIDs."""
        analyzer = BLEAnalyzer()
        # Complete List of 16-bit Service UUIDs: 0x180D (Heart Rate), 0x180F (Battery)
        data = b"\x05\x03\x0d\x18\x0f\x18"

        result = analyzer.parse_advertising_data(data)

        assert "service_uuids" in result
        assert "0x180D" in result["service_uuids"]
        assert "0x180F" in result["service_uuids"]

    def test_parse_service_data(self):
        """Test parsing service data."""
        analyzer = BLEAnalyzer()
        # Service Data: UUID=0x180D, Data=0xABCD
        data = b"\x05\x16\x0d\x18\xab\xcd"

        result = analyzer.parse_advertising_data(data)

        assert "service_data" in result
        assert result["service_data"]["uuid"] == "0x180D"
        assert result["service_data"]["data"] == "abcd"

    def test_parse_manufacturer_data(self):
        """Test parsing manufacturer specific data."""
        analyzer = BLEAnalyzer()
        # Manufacturer Data: Company ID=0x004C (Apple), Data=arbitrary
        data = b"\x07\xff\x4c\x00\x12\x34\x56\x78"

        result = analyzer.parse_advertising_data(data)

        assert "manufacturer_data" in result
        assert result["manufacturer_data"]["company_id"] == "0x004C"
        assert result["manufacturer_data"]["data"] == "12345678"

    def test_parse_multiple_ad_structures(self):
        """Test parsing multiple AD structures in one packet."""
        analyzer = BLEAnalyzer()
        data = create_advertising_packet(
            name="MultiTest",
            flags=0x06,
            service_uuids=[0x180D, 0x180F],
            tx_power=-5,
        )

        result = analyzer.parse_advertising_data(data)

        assert "flags" in result
        assert "name" in result
        assert result["name"] == "MultiTest"
        assert "service_uuids" in result
        assert len(result["service_uuids"]) == 2
        assert "tx_power" in result

    def test_parse_empty_data(self):
        """Test parsing empty advertising data."""
        analyzer = BLEAnalyzer()
        result = analyzer.parse_advertising_data(b"")

        assert result == {}

    def test_parse_malformed_data(self):
        """Test parsing malformed advertising data."""
        analyzer = BLEAnalyzer()
        # Malformed: length indicates more data than available
        data = b"\xff\x01\x06"

        result = analyzer.parse_advertising_data(data)

        # Should handle gracefully without crashing
        assert isinstance(result, dict)


# =============================================================================
# ATT Operation Decoding Tests
# =============================================================================


class TestATTOperationDecoding:
    """Test ATT protocol operation decoding."""

    def test_decode_read_request(self):
        """Test decoding ATT Read Request."""
        analyzer = BLEAnalyzer()
        data = create_att_read_request(0x0003)

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0x0A"
        assert result["opcode_name"] == "Read Request"
        assert result["handle"] == 0x0003

    def test_decode_read_response(self):
        """Test decoding ATT Read Response."""
        analyzer = BLEAnalyzer()
        value = b"\x01\x02\x03\x04"
        data = create_att_read_response(value)

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0x0B"
        assert result["opcode_name"] == "Read Response"
        assert result["value"] == "01020304"

    def test_decode_write_request(self):
        """Test decoding ATT Write Request."""
        analyzer = BLEAnalyzer()
        handle = 0x0010
        value = b"\xab\xcd"
        data = bytes([0x12]) + handle.to_bytes(2, "little") + value

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0x12"
        assert result["opcode_name"] == "Write Request"
        assert result["handle"] == 0x0010
        assert result["value"] == "abcd"

    def test_decode_notification(self):
        """Test decoding ATT Handle Value Notification."""
        analyzer = BLEAnalyzer()
        handle = 0x0020
        value = b"\x64"  # Battery level 100%
        data = bytes([0x1B]) + handle.to_bytes(2, "little") + value

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0x1B"
        assert result["opcode_name"] == "Handle Value Notification"
        assert result["handle"] == 0x0020
        assert result["value"] == "64"

    def test_decode_mtu_exchange(self):
        """Test decoding MTU Exchange Request."""
        analyzer = BLEAnalyzer()
        mtu = 247
        data = bytes([0x02]) + mtu.to_bytes(2, "little")

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0x02"
        assert result["opcode_name"] == "Exchange MTU Request"
        assert result["mtu"] == 247

    def test_decode_error_response(self):
        """Test decoding ATT Error Response."""
        analyzer = BLEAnalyzer()
        # Error: Request opcode=0x0A, Handle=0x0003, Error=0x0A (Attribute Not Found)
        data = bytes([0x01, 0x0A, 0x03, 0x00, 0x0A])

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0x01"
        assert result["opcode_name"] == "Error Response"
        assert result["request_opcode"] == "0x0A"
        assert result["handle"] == 0x0003
        assert result["error_code"] == "0x0A"

    def test_decode_read_by_type_request(self):
        """Test decoding Read By Type Request."""
        analyzer = BLEAnalyzer()
        # Read By Type: start=0x0001, end=0xFFFF, uuid=0x2803 (Characteristic)
        data = (
            bytes([0x08])
            + (0x0001).to_bytes(2, "little")
            + (0xFFFF).to_bytes(2, "little")
            + (0x2803).to_bytes(2, "little")
        )

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0x08"
        assert result["start_handle"] == 0x0001
        assert result["end_handle"] == 0xFFFF
        assert result["uuid"] == "0x2803"

    def test_decode_empty_packet(self):
        """Test decoding empty ATT packet."""
        analyzer = BLEAnalyzer()
        result = analyzer.decode_att_operation(b"")

        assert "error" in result

    def test_decode_unknown_opcode(self):
        """Test decoding unknown ATT opcode."""
        analyzer = BLEAnalyzer()
        data = b"\xff\x01\x02\x03"

        result = analyzer.decode_att_operation(data)

        assert result["opcode"] == "0xFF"
        assert "Unknown Opcode" in result["opcode_name"]


# =============================================================================
# GATT Service Discovery Tests
# =============================================================================


class TestGATTServiceDiscovery:
    """Test GATT service and characteristic discovery."""

    def test_discover_single_service(self):
        """Test discovering a single GATT service."""
        analyzer = BLEAnalyzer()

        # Add service discovery response
        # Service: Heart Rate (0x180D), handles 0x0001-0x0005
        att_data = create_att_read_by_group_type_response(
            [(0x0001, 0x0005, (0x180D).to_bytes(2, "little"))]
        )

        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=att_data,
        )
        packet.decoded = analyzer.decode_att_operation(att_data)
        analyzer.packets.append(packet)

        services = analyzer.discover_services()

        assert len(services) == 1
        assert services[0].uuid == "0x180D"
        assert services[0].name == "Heart Rate"
        assert services[0].handle_range == (0x0001, 0x0005)

    def test_discover_multiple_services(self):
        """Test discovering multiple GATT services."""
        analyzer = BLEAnalyzer()

        # Add service discovery response with multiple services
        att_data = create_att_read_by_group_type_response(
            [
                (0x0001, 0x0005, (0x180D).to_bytes(2, "little")),  # Heart Rate
                (0x0006, 0x0008, (0x180F).to_bytes(2, "little")),  # Battery
                (0x0009, 0x000F, (0x180A).to_bytes(2, "little")),  # Device Info
            ]
        )

        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=att_data,
        )
        packet.decoded = analyzer.decode_att_operation(att_data)
        analyzer.packets.append(packet)

        services = analyzer.discover_services()

        assert len(services) == 3
        assert services[0].name == "Heart Rate"
        assert services[1].name == "Battery Service"
        assert services[2].name == "Device Information"

    def test_discover_characteristics(self):
        """Test discovering characteristics within a service."""
        analyzer = BLEAnalyzer()

        # First, add service discovery
        service_data = create_att_read_by_group_type_response(
            [(0x0001, 0x0010, (0x180D).to_bytes(2, "little"))]  # Heart Rate
        )
        service_packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=service_data,
        )
        service_packet.decoded = analyzer.decode_att_operation(service_data)
        analyzer.packets.append(service_packet)

        # Add characteristic discovery
        # Characteristic: properties=0x12 (read, notify), value_handle=0x0003,
        # uuid=0x2A37 (HR Measurement)
        char_declaration = bytes([0x12, 0x03, 0x00, 0x37, 0x2A])
        char_data = create_att_read_by_type_response([(0x0002, char_declaration)])

        char_packet = BLEPacket(
            timestamp=0.1,
            packet_type="ATT_READ_BY_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=char_data,
        )
        char_packet.decoded = analyzer.decode_att_operation(char_data)
        analyzer.packets.append(char_packet)

        services = analyzer.discover_services()

        assert len(services) == 1
        assert len(services[0].characteristics) == 1

        char = services[0].characteristics[0]
        assert char.uuid == "0x2A37"
        assert char.name == "Heart Rate Measurement"
        assert "read" in char.properties
        assert "notify" in char.properties
        assert char.value_handle == 0x0003

    def test_discover_custom_uuid_service(self):
        """Test discovering service with custom UUID."""
        analyzer = BLEAnalyzer()
        analyzer.register_custom_uuid("0xABCD", "Custom Sensor Service")

        # Add custom service
        service_data = create_att_read_by_group_type_response(
            [(0x0001, 0x0005, (0xABCD).to_bytes(2, "little"))]
        )

        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=service_data,
        )
        packet.decoded = analyzer.decode_att_operation(service_data)
        analyzer.packets.append(packet)

        services = analyzer.discover_services()

        assert len(services) == 1
        assert services[0].name == "Custom Sensor Service"

    def test_discover_no_services(self):
        """Test discovery when no service packets exist."""
        analyzer = BLEAnalyzer()

        # Add non-service packets
        adv_data = create_advertising_packet("TestDevice")
        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ADV_IND",
            source_address="AA:BB:CC:DD:EE:FF",
            data=adv_data,
        )
        analyzer.add_packet(packet)

        services = analyzer.discover_services()

        assert len(services) == 0


# =============================================================================
# Packet Management Tests
# =============================================================================


class TestPacketManagement:
    """Test BLE packet addition and management."""

    def test_add_advertising_packet(self):
        """Test adding advertising packet with auto-decode."""
        analyzer = BLEAnalyzer()
        adv_data = create_advertising_packet("TestDevice", flags=0x06)

        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ADV_IND",
            source_address="AA:BB:CC:DD:EE:FF",
            data=adv_data,
        )

        analyzer.add_packet(packet)

        assert len(analyzer.packets) == 1
        assert analyzer.packets[0].decoded is not None
        assert "name" in analyzer.packets[0].decoded

    def test_add_att_packet(self):
        """Test adding ATT packet with auto-decode."""
        analyzer = BLEAnalyzer()
        att_data = create_att_read_request(0x0010)

        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_REQ",
            source_address="AA:BB:CC:DD:EE:FF",
            data=att_data,
        )

        analyzer.add_packet(packet)

        assert len(analyzer.packets) == 1
        assert analyzer.packets[0].decoded is not None
        assert analyzer.packets[0].decoded["opcode_name"] == "Read Request"

    def test_add_multiple_packets(self):
        """Test adding multiple packets."""
        analyzer = BLEAnalyzer()

        for i in range(5):
            packet = BLEPacket(
                timestamp=float(i),
                packet_type="ADV_IND",
                source_address=f"AA:BB:CC:DD:EE:{i:02X}",
                data=create_advertising_packet(f"Device{i}"),
            )
            analyzer.add_packet(packet)

        assert len(analyzer.packets) == 5


# =============================================================================
# Export Functionality Tests
# =============================================================================


class TestExportFunctionality:
    """Test service export to JSON and CSV."""

    def test_export_json(self):
        """Test exporting services to JSON."""
        analyzer = BLEAnalyzer()

        # Add a service
        service_data = create_att_read_by_group_type_response(
            [(0x0001, 0x0005, (0x180D).to_bytes(2, "little"))]
        )
        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=service_data,
        )
        packet.decoded = analyzer.decode_att_operation(service_data)
        analyzer.packets.append(packet)
        analyzer.discover_services()

        # Export to JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "services.json"
            analyzer.export_services(output_path, format="json")

            assert output_path.exists()

            # Verify content
            with output_path.open() as f:
                data = json.load(f)

            assert "services" in data
            assert len(data["services"]) == 1
            assert data["services"][0]["uuid"] == "0x180D"
            assert data["services"][0]["name"] == "Heart Rate"

    def test_export_csv(self):
        """Test exporting services to CSV."""
        analyzer = BLEAnalyzer()

        # Add a service with characteristic
        service_data = create_att_read_by_group_type_response(
            [(0x0001, 0x0010, (0x180D).to_bytes(2, "little"))]
        )
        service_packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=service_data,
        )
        service_packet.decoded = analyzer.decode_att_operation(service_data)
        analyzer.packets.append(service_packet)

        # Add characteristic
        char_declaration = bytes([0x02, 0x03, 0x00, 0x37, 0x2A])  # Read property
        char_data = create_att_read_by_type_response([(0x0002, char_declaration)])
        char_packet = BLEPacket(
            timestamp=0.1,
            packet_type="ATT_READ_BY_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=char_data,
        )
        char_packet.decoded = analyzer.decode_att_operation(char_data)
        analyzer.packets.append(char_packet)

        analyzer.discover_services()

        # Export to CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "services.csv"
            analyzer.export_services(output_path, format="csv")

            assert output_path.exists()

            # Verify content
            with output_path.open() as f:
                content = f.read()

            assert "0x180D" in content
            assert "Heart Rate" in content
            assert "0x2A37" in content

    def test_export_invalid_format(self):
        """Test export with invalid format."""
        analyzer = BLEAnalyzer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "services.xml"

            with pytest.raises(ValueError, match="Unsupported format"):
                analyzer.export_services(output_path, format="xml")


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Test analyzer statistics generation."""

    def test_get_statistics_empty(self):
        """Test statistics for empty analyzer."""
        analyzer = BLEAnalyzer()
        stats = analyzer.get_statistics()

        assert stats["total_packets"] == 0
        assert stats["services_discovered"] == 0
        assert stats["total_characteristics"] == 0

    def test_get_statistics_with_packets(self):
        """Test statistics with packets."""
        analyzer = BLEAnalyzer()

        # Add various packet types
        for i in range(3):
            adv_packet = BLEPacket(
                timestamp=float(i),
                packet_type="ADV_IND",
                source_address="AA:BB:CC:DD:EE:FF",
                data=create_advertising_packet(f"Device{i}"),
            )
            analyzer.add_packet(adv_packet)

        for i in range(2):
            att_packet = BLEPacket(
                timestamp=float(i + 10),
                packet_type="ATT_READ_REQ",
                source_address="AA:BB:CC:DD:EE:FF",
                data=create_att_read_request(i),
            )
            analyzer.add_packet(att_packet)

        stats = analyzer.get_statistics()

        assert stats["total_packets"] == 5
        assert stats["packet_types"]["ADV_IND"] == 3
        assert stats["packet_types"]["ATT_READ_REQ"] == 2

    def test_get_statistics_with_services(self):
        """Test statistics with discovered services."""
        analyzer = BLEAnalyzer()

        # Add two services, one with a characteristic
        service_data = create_att_read_by_group_type_response(
            [
                (0x0001, 0x0010, (0x180D).to_bytes(2, "little")),
                (0x0011, 0x0020, (0x180F).to_bytes(2, "little")),
            ]
        )
        service_packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=service_data,
        )
        service_packet.decoded = analyzer.decode_att_operation(service_data)
        analyzer.packets.append(service_packet)

        # Add characteristic to first service
        char_declaration = bytes([0x02, 0x03, 0x00, 0x37, 0x2A])
        char_data = create_att_read_by_type_response([(0x0002, char_declaration)])
        char_packet = BLEPacket(
            timestamp=0.1,
            packet_type="ATT_READ_BY_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=char_data,
        )
        char_packet.decoded = analyzer.decode_att_operation(char_data)
        analyzer.packets.append(char_packet)

        analyzer.discover_services()
        stats = analyzer.get_statistics()

        assert stats["services_discovered"] == 2
        assert stats["total_characteristics"] == 1


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_truncated_advertising_data(self):
        """Test parsing truncated advertising data."""
        analyzer = BLEAnalyzer()
        # Length indicates 10 bytes but only 2 provided
        data = b"\x0a\x09AB"

        result = analyzer.parse_advertising_data(data)

        # Should handle gracefully
        assert isinstance(result, dict)

    def test_decode_truncated_att_packet(self):
        """Test decoding truncated ATT packet."""
        analyzer = BLEAnalyzer()
        # Read request requires 3 bytes, only provide 2
        data = b"\x0a\x03"

        result = analyzer.decode_att_operation(data)

        # Should not crash
        assert "opcode" in result

    def test_service_discovery_with_invalid_data(self):
        """Test service discovery with malformed response."""
        analyzer = BLEAnalyzer()

        # Create packet with invalid decoded data
        packet = BLEPacket(
            timestamp=0.0,
            packet_type="ATT_READ_BY_GROUP_TYPE_RSP",
            source_address="AA:BB:CC:DD:EE:FF",
            data=b"\x11\x06\x01\x00",  # Truncated
        )
        packet.decoded = {"opcode_name": "Read By Group Type Response", "attributes": []}
        analyzer.packets.append(packet)

        services = analyzer.discover_services()

        # Should handle gracefully
        assert len(services) == 0

    def test_uuid_name_lookup_with_unknown(self):
        """Test UUID name lookup for unknown UUIDs."""
        analyzer = BLEAnalyzer()

        # Unknown service UUID
        name = analyzer.get_uuid_name("0x9999", "service")
        assert "Unknown" in name or "9999" in name

        # Unknown characteristic UUID
        name = analyzer.get_uuid_name("0x8888", "characteristic")
        assert "Unknown" in name or "8888" in name
