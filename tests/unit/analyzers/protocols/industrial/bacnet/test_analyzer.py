"""Unit tests for BACnet protocol analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscura.analyzers.protocols.industrial.bacnet import (
    BACnetAnalyzer,
    BACnetDevice,
    BACnetMessage,
    BACnetObject,
)


class TestBACnetMessage:
    """Test BACnetMessage dataclass."""

    def test_bacnet_message_creation(self) -> None:
        """Test creating BACnet message."""
        msg = BACnetMessage(
            timestamp=1.0,
            protocol="BACnet/IP",
            npdu={"version": 1, "control": 0x20},
            apdu_type="Unconfirmed-REQ",
            service_choice=8,
            service_name="who-Is",
        )

        assert msg.timestamp == 1.0
        assert msg.protocol == "BACnet/IP"
        assert msg.apdu_type == "Unconfirmed-REQ"
        assert msg.service_name == "who-Is"


class TestBACnetObject:
    """Test BACnetObject dataclass."""

    def test_bacnet_object_creation(self) -> None:
        """Test creating BACnet object."""
        obj = BACnetObject(
            object_type="analog-input",
            instance_number=5,
            properties={"present-value": 72.5, "units": "degrees-fahrenheit"},
        )

        assert obj.object_type == "analog-input"
        assert obj.instance_number == 5
        assert obj.properties["present-value"] == 72.5


class TestBACnetDevice:
    """Test BACnetDevice dataclass."""

    def test_bacnet_device_creation(self) -> None:
        """Test creating BACnet device."""
        device = BACnetDevice(
            device_instance=8,
            device_name="VAV-101",
            vendor_id=260,
            model_name="Controller",
        )

        assert device.device_instance == 8
        assert device.device_name == "VAV-101"
        assert device.vendor_id == 260
        assert len(device.objects) == 0


class TestBACnetAnalyzer:
    """Test BACnetAnalyzer class."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = BACnetAnalyzer()

        assert len(analyzer.messages) == 0
        assert len(analyzer.devices) == 0
        assert analyzer.APDU_TYPES[0] == "Confirmed-REQ"
        assert analyzer.UNCONFIRMED_SERVICES[8] == "who-Is"

    def test_parse_npdu_minimal(self) -> None:
        """Test parsing minimal NPDU."""
        analyzer = BACnetAnalyzer()
        # Version 1, control 0x00 (no flags)
        data = bytes([0x01, 0x00])
        npdu, consumed = analyzer._parse_npdu(data)

        assert npdu["version"] == 1
        assert npdu["control"] == 0x00
        assert npdu["dest_specifier"] is False
        assert npdu["source_specifier"] is False
        assert npdu["expects_reply"] is False
        assert consumed == 2

    def test_parse_npdu_with_destination(self) -> None:
        """Test parsing NPDU with destination address."""
        analyzer = BACnetAnalyzer()
        # Version 1, control 0x20 (dest_specifier set)
        # Dest network 100, MAC length 1, MAC 0x42
        data = bytes([0x01, 0x20, 0x00, 0x64, 0x01, 0x42])
        npdu, consumed = analyzer._parse_npdu(data)

        assert npdu["dest_specifier"] is True
        assert npdu["dest_network"] == 100
        assert npdu["dest_mac"] == "42"
        assert consumed == 6

    def test_parse_npdu_with_source(self) -> None:
        """Test parsing NPDU with source address."""
        analyzer = BACnetAnalyzer()
        # Version 1, control 0x08 (source_specifier set)
        # Source network 200, MAC length 2, MAC 0xABCD
        data = bytes([0x01, 0x08, 0x00, 0xC8, 0x02, 0xAB, 0xCD])
        npdu, consumed = analyzer._parse_npdu(data)

        assert npdu["source_specifier"] is True
        assert npdu["source_network"] == 200
        assert npdu["source_mac"] == "abcd"
        assert consumed == 7

    def test_parse_npdu_invalid_version(self) -> None:
        """Test error with invalid NPDU version."""
        analyzer = BACnetAnalyzer()
        data = bytes([0x02, 0x00])  # Version 2 (invalid)

        with pytest.raises(ValueError, match="Invalid NPDU version"):
            analyzer._parse_npdu(data)

    def test_parse_apdu_confirmed_request(self) -> None:
        """Test parsing confirmed request APDU."""
        analyzer = BACnetAnalyzer()
        # APDU type 0 (Confirmed-REQ), no flags, max_seg/apdu = 0x00
        # Invoke ID = 5, service choice = 12 (ReadProperty)
        data = bytes([0x00, 0x00, 0x05, 0x0C])
        apdu = analyzer._parse_apdu(data)

        assert apdu["apdu_type"] == 0
        assert apdu["invoke_id"] == 5
        assert apdu["service_choice"] == 12
        assert apdu["service_name"] == "readProperty"
        assert apdu["segmented"] is False

    def test_parse_apdu_unconfirmed_request(self) -> None:
        """Test parsing unconfirmed request APDU."""
        analyzer = BACnetAnalyzer()
        # APDU type 1 (Unconfirmed-REQ), service choice = 8 (Who-Is)
        data = bytes([0x10, 0x08])
        apdu = analyzer._parse_apdu(data)

        assert apdu["apdu_type"] == 1
        assert apdu["service_choice"] == 8
        assert apdu["service_name"] == "who-Is"

    def test_parse_apdu_simple_ack(self) -> None:
        """Test parsing SimpleACK APDU."""
        analyzer = BACnetAnalyzer()
        # APDU type 2 (SimpleACK), invoke ID = 10, service choice = 15
        data = bytes([0x20, 0x0A, 0x0F])
        apdu = analyzer._parse_apdu(data)

        assert apdu["apdu_type"] == 2
        assert apdu["invoke_id"] == 10
        assert apdu["service_choice"] == 15

    def test_parse_apdu_complex_ack(self) -> None:
        """Test parsing ComplexACK APDU."""
        analyzer = BACnetAnalyzer()
        # APDU type 3 (ComplexACK), invoke ID = 7, service choice = 12
        data = bytes([0x30, 0x07, 0x0C, 0x3E, 0x21, 0x42, 0x3F])
        apdu = analyzer._parse_apdu(data)

        assert apdu["apdu_type"] == 3
        assert apdu["invoke_id"] == 7
        assert apdu["service_choice"] == 12
        assert apdu["service_name"] == "readProperty"

    def test_parse_bacnet_ip_who_is(self) -> None:
        """Test parsing BACnet/IP Who-Is message."""
        analyzer = BACnetAnalyzer()
        # BVLC: Type 0x81, Function 0x0A, Length 0x000C
        # NPDU: Version 0x01, Control 0x20 (expects reply)
        # APDU: Unconfirmed-REQ (0x10), Who-Is (0x08)
        data = bytes([0x81, 0x0A, 0x00, 0x0C, 0x01, 0x20, 0xFF, 0xFF, 0x00, 0xFF, 0x10, 0x08])

        message = analyzer.parse_bacnet_ip(data, timestamp=1.5)

        assert message.timestamp == 1.5
        assert message.protocol == "BACnet/IP"
        assert message.apdu_type == "Unconfirmed-REQ"
        assert message.service_name == "who-Is"
        assert len(analyzer.messages) == 1

    def test_parse_bacnet_ip_i_am(self) -> None:
        """Test parsing BACnet/IP I-Am message."""
        analyzer = BACnetAnalyzer()
        # BVLC header + NPDU + I-Am APDU
        # I-Am contains: device #100, max APDU 1476, segmentation 3, vendor 260
        bvlc = bytes([0x81, 0x0A, 0x00, 0x19])
        npdu = bytes([0x01, 0x20, 0xFF, 0xFF, 0x00, 0xFF])
        apdu = bytes(
            [
                0x10,
                0x00,  # Unconfirmed-REQ, I-Am
                0xC4,
                0x02,
                0x00,
                0x00,
                0x64,  # Device #100
                0x22,
                0x05,
                0xC4,  # Max APDU = 1476
                0x91,
                0x03,  # Segmentation = 3
                0x22,
                0x01,
                0x04,  # Vendor = 260
            ]
        )

        message = analyzer.parse_bacnet_ip(bvlc + npdu + apdu, timestamp=2.0)

        assert message.service_name == "i-Am"
        assert message.decoded_service["device_instance"] == 100
        assert message.decoded_service["vendor_id"] == 260
        assert 100 in analyzer.devices
        assert analyzer.devices[100].vendor_id == 260

    def test_parse_bacnet_ip_invalid_bvlc_type(self) -> None:
        """Test error with invalid BVLC type."""
        analyzer = BACnetAnalyzer()
        data = bytes([0x82, 0x0A, 0x00, 0x0C])  # Invalid type 0x82

        with pytest.raises(ValueError, match="Invalid BACnet/IP type"):
            analyzer.parse_bacnet_ip(data)

    def test_parse_bacnet_ip_too_short(self) -> None:
        """Test error when BACnet/IP message is too short."""
        analyzer = BACnetAnalyzer()
        data = bytes([0x81, 0x0A])  # Only 2 bytes

        with pytest.raises(ValueError, match="BACnet/IP message too short"):
            analyzer.parse_bacnet_ip(data)

    def test_parse_bacnet_mstp_frame(self) -> None:
        """Test parsing BACnet MSTP frame."""
        analyzer = BACnetAnalyzer()
        # Preamble: 0x55 0xFF
        # Frame type: 0x05 (data expecting reply)
        # Dest: 0x01, Source: 0x00
        # Length: 0x000A (10 bytes)
        # Header CRC: calculated
        # Data: NPDU + APDU
        # Data CRC: calculated

        preamble = bytes([0x55, 0xFF])
        header = bytes([0x05, 0x01, 0x00, 0x00, 0x0A])
        header_crc = analyzer._mstp_header_crc(header)

        npdu = bytes([0x01, 0x00])
        apdu = bytes([0x10, 0x08])  # Who-Is
        data = npdu + apdu + bytes([0x00] * 6)  # Pad to 10 bytes
        data_crc = analyzer._mstp_data_crc(data)

        frame = preamble + header + bytes([header_crc]) + data + data_crc.to_bytes(2, "big")

        message = analyzer.parse_bacnet_mstp(frame, timestamp=3.0)

        assert message.timestamp == 3.0
        assert message.protocol == "BACnet/MSTP"
        assert message.service_name == "who-Is"

    def test_parse_bacnet_mstp_invalid_preamble(self) -> None:
        """Test error with invalid MSTP preamble."""
        analyzer = BACnetAnalyzer()
        # Minimum 8 bytes, but wrong second preamble byte
        data = bytes([0x55, 0xAA, 0x05, 0x01, 0x00, 0x00, 0x00, 0x00])

        with pytest.raises(ValueError, match="Invalid MSTP preamble.*0x55 0xFF"):
            analyzer.parse_bacnet_mstp(data)

    def test_parse_bacnet_mstp_too_short(self) -> None:
        """Test error when MSTP frame is too short."""
        analyzer = BACnetAnalyzer()
        data = bytes([0x55, 0xFF, 0x05])  # Only 3 bytes

        with pytest.raises(ValueError, match="BACnet MSTP frame too short"):
            analyzer.parse_bacnet_mstp(data)

    def test_decode_service_who_is(self) -> None:
        """Test service decoding for Who-Is."""
        analyzer = BACnetAnalyzer()
        data = bytes([0x09, 0x00, 0x19, 0xFF])  # Range 0-255

        result = analyzer._decode_service(1, 8, data)  # Unconfirmed, Who-Is

        assert result.get("device_instance_range_low_limit") == 0
        assert result.get("device_instance_range_high_limit") == 255

    def test_decode_service_read_property(self) -> None:
        """Test service decoding for ReadProperty."""
        analyzer = BACnetAnalyzer()
        # Object: analog-input #10, Property: present-value
        data = bytes([0x0C, 0x00, 0x00, 0x00, 0x0A, 0x19, 0x55])

        result = analyzer._decode_service(0, 12, data)  # Confirmed, ReadProperty

        assert result["object_identifier"]["instance"] == 10
        assert result["property_name"] == "present-value"

    def test_export_devices_empty(self, tmp_path: Path) -> None:
        """Test exporting devices when none discovered."""
        analyzer = BACnetAnalyzer()
        output_file = tmp_path / "devices.json"

        analyzer.export_devices(output_file)

        assert output_file.exists()
        with output_file.open() as f:
            data = json.load(f)
        assert data == []

    def test_export_devices_with_data(self, tmp_path: Path) -> None:
        """Test exporting discovered devices."""
        analyzer = BACnetAnalyzer()

        # Add a device
        device = BACnetDevice(device_instance=8, vendor_id=260)
        device.objects.append(
            BACnetObject(
                object_type="analog-input", instance_number=5, properties={"present-value": 72.5}
            )
        )
        analyzer.devices[8] = device

        output_file = tmp_path / "devices.json"
        analyzer.export_devices(output_file)

        assert output_file.exists()
        with output_file.open() as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["device_instance"] == 8
        assert data[0]["vendor_id"] == 260
        assert len(data[0]["objects"]) == 1
        assert data[0]["objects"][0]["object_type"] == "analog-input"

    def test_mstp_header_crc(self) -> None:
        """Test MSTP header CRC calculation."""
        analyzer = BACnetAnalyzer()
        header = bytes([0x05, 0x01, 0x00, 0x00, 0x0A])
        crc = analyzer._mstp_header_crc(header)

        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFF

    def test_mstp_data_crc(self) -> None:
        """Test MSTP data CRC calculation."""
        analyzer = BACnetAnalyzer()
        data = bytes([0x01, 0x00, 0x10, 0x08])
        crc = analyzer._mstp_data_crc(data)

        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_update_device_info_from_i_am(self) -> None:
        """Test device info update from I-Am message."""
        analyzer = BACnetAnalyzer()

        # Create I-Am message
        message = BACnetMessage(
            timestamp=1.0,
            protocol="BACnet/IP",
            npdu={},
            apdu_type="Unconfirmed-REQ",
            service_name="i-Am",
            decoded_service={"device_instance": 42, "vendor_id": 123},
        )

        analyzer._update_device_info(message)

        assert 42 in analyzer.devices
        assert analyzer.devices[42].device_instance == 42
        assert analyzer.devices[42].vendor_id == 123

    def test_parse_bacnet_ip_read_property_request(self) -> None:
        """Test parsing complete ReadProperty request."""
        analyzer = BACnetAnalyzer()

        # BVLC header
        bvlc = bytes([0x81, 0x0A, 0x00, 0x14])
        # NPDU
        npdu = bytes([0x01, 0x04])  # Version, expects reply
        # Confirmed-REQ ReadProperty
        apdu = bytes(
            [
                0x00,
                0x05,
                0x01,
                0x0C,  # Confirmed-REQ, invoke 1, ReadProperty
                0x0C,
                0x00,
                0x00,
                0x00,
                0x05,  # Object: analog-input #5
                0x19,
                0x55,  # Property: present-value
            ]
        )

        message = analyzer.parse_bacnet_ip(bvlc + npdu + apdu, timestamp=4.5)

        assert message.apdu_type == "Confirmed-REQ"
        assert message.service_name == "readProperty"
        assert message.invoke_id == 1
        assert message.decoded_service["property_name"] == "present-value"

    def test_multiple_messages_stored(self) -> None:
        """Test that multiple messages are stored."""
        analyzer = BACnetAnalyzer()

        # Parse two Who-Is messages
        data = bytes([0x81, 0x0A, 0x00, 0x0A, 0x01, 0x20, 0xFF, 0xFF, 0x00, 0xFF, 0x10, 0x08])

        analyzer.parse_bacnet_ip(data, timestamp=1.0)
        analyzer.parse_bacnet_ip(data, timestamp=2.0)

        assert len(analyzer.messages) == 2
        assert analyzer.messages[0].timestamp == 1.0
        assert analyzer.messages[1].timestamp == 2.0
