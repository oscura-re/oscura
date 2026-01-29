"""Unit tests for BACnet service decoders."""

from __future__ import annotations

from oscura.analyzers.protocols.industrial.bacnet.services import (
    decode_i_am,
    decode_i_have,
    decode_read_property_ack,
    decode_read_property_request,
    decode_who_has,
    decode_who_is,
    decode_write_property_request,
    get_property_name,
)


class TestDecodeWhoIs:
    """Test decode_who_is function."""

    def test_decode_who_is_no_range(self) -> None:
        """Test Who-Is with no device instance range."""
        data = bytes([])
        result = decode_who_is(data)

        assert result == {}

    def test_decode_who_is_with_range(self) -> None:
        """Test Who-Is with device instance range."""
        # Context tag 0 (low limit), length 2, value 0
        # Context tag 1 (high limit), length 2, value 255
        # 0x09 = tag 0, context, length 1
        # 0x19 = tag 1, context, length 1
        data = bytes([0x09, 0x00, 0x19, 0xFF])
        result = decode_who_is(data)

        assert result.get("device_instance_range_low_limit") == 0
        assert result.get("device_instance_range_high_limit") == 255


class TestDecodeIAm:
    """Test decode_i_am function."""

    def test_decode_i_am_complete(self) -> None:
        """Test I-Am with all fields."""
        # Device object identifier: Application tag 12 (object-id), device #8
        # Max APDU: Application unsigned, value 1476 (0x05C4)
        # Segmentation: Application enumerated, value 3 (no-segmentation)
        # Vendor ID: Application unsigned, value 260 (0x0104)
        data = bytes(
            [
                0xC4,
                0x02,
                0x00,
                0x00,
                0x08,  # App tag 12, len 4, device #8
                0x22,
                0x05,
                0xC4,  # App tag 2 (unsigned), len 2, value 1476
                0x91,
                0x03,  # App tag 9 (enumerated), len 1, value 3
                0x22,
                0x01,
                0x04,  # App tag 2 (unsigned), len 2, value 260
            ]
        )
        result = decode_i_am(data)

        assert result.get("device_instance") == 8
        assert result.get("device_object_type") == "device"
        assert result.get("max_apdu_length") == 1476
        assert result.get("segmentation") == "no-segmentation"
        assert result.get("vendor_id") == 260

    def test_decode_i_am_minimal(self) -> None:
        """Test I-Am with only device ID."""
        data = bytes([0xC4, 0x02, 0x00, 0x00, 0x42])  # Device #66
        result = decode_i_am(data)

        assert result.get("device_instance") == 66
        assert "max_apdu_length" not in result


class TestDecodeWhoHas:
    """Test decode_who_has function."""

    def test_decode_who_has_with_object_id(self) -> None:
        """Test Who-Has with object identifier."""
        # Context tag 2 (object ID), length 4, analog-input #123
        data = bytes([0x2C, 0x00, 0x00, 0x00, 0x7B])
        result = decode_who_has(data)

        assert "object_identifier" in result
        assert result["object_identifier"]["object_type"] == 0
        assert result["object_identifier"]["instance"] == 123

    def test_decode_who_has_with_object_name(self) -> None:
        """Test Who-Has with object name."""
        # Context tag 3 (object name), length 4, "TEMP"
        data = bytes([0x3C, 0x54, 0x45, 0x4D, 0x50])
        result = decode_who_has(data)

        assert result["object_name"] == "TEMP"


class TestDecodeIHave:
    """Test decode_i_have function."""

    def test_decode_i_have(self) -> None:
        """Test I-Have service decoding."""
        # Device ID: application tag 12, device #100
        # Object ID: application tag 12, analog-input #5
        # Object name: application tag 7, "Temperature"
        data = (
            bytes(
                [
                    0xC4,
                    0x02,
                    0x00,
                    0x00,
                    0x64,  # Device #100
                    0xC4,
                    0x00,
                    0x00,
                    0x00,
                    0x05,  # Analog-input #5
                    0x7B,
                    0x00,
                ]
            )
            + b"Temperature"  # Object name
        )
        result = decode_i_have(data)

        assert "device_identifier" in result
        assert "object_identifier" in result


class TestDecodeReadPropertyRequest:
    """Test decode_read_property_request function."""

    def test_decode_read_property_request(self) -> None:
        """Test ReadProperty request decoding."""
        # Object ID: context tag 0, analog-input #10
        # Property ID: context tag 1, present-value (85)
        data = bytes(
            [
                0x0C,
                0x00,
                0x00,
                0x00,
                0x0A,  # Object ID
                0x19,
                0x55,  # Property ID = 85
            ]
        )
        result = decode_read_property_request(data)

        assert result["object_identifier"]["object_type"] == 0
        assert result["object_identifier"]["instance"] == 10
        assert result["property_identifier"] == 85
        assert result["property_name"] == "present-value"

    def test_decode_read_property_request_with_array_index(self) -> None:
        """Test ReadProperty request with array index."""
        # Object ID: context tag 0
        # Property ID: context tag 1
        # Array index: context tag 2, value 3
        data = bytes(
            [
                0x0C,
                0x02,
                0x00,
                0x00,
                0x08,  # Device #8
                0x19,
                0x4D,  # Property ID = 77 (object-name)
                0x29,
                0x03,  # Array index = 3
            ]
        )
        result = decode_read_property_request(data)

        assert result["property_identifier"] == 77
        assert result["property_name"] == "object-name"
        assert result["property_array_index"] == 3


class TestDecodeReadPropertyAck:
    """Test decode_read_property_ack function."""

    def test_decode_read_property_ack_simple_value(self) -> None:
        """Test ReadProperty-ACK with simple value."""
        # Object ID: context tag 0, analog-input #5
        # Property ID: context tag 1, present-value (85)
        # Value: opening tag 3, unsigned 72, closing tag 3
        data = bytes(
            [
                0x0C,
                0x00,
                0x00,
                0x00,
                0x05,  # Context tag 0, object ID (analog-input #5)
                0x19,
                0x55,  # Context tag 1, property ID = 85
                0x3E,  # Opening tag 3 (context)
                0x21,
                0x48,  # App tag 2 (unsigned), len 1, value 72
                0x3F,  # Closing tag 3 (context)
            ]
        )
        result = decode_read_property_ack(data)

        assert result.get("property_identifier") == 85
        assert result.get("property_name") == "present-value"
        assert result.get("property_value") == 72

    def test_decode_read_property_ack_multiple_values(self) -> None:
        """Test ReadProperty-ACK with multiple values (array)."""
        # Opening tag, value 1, value 2, value 3, closing tag
        data = bytes(
            [
                0x0C,
                0x00,
                0x00,
                0x00,
                0x01,  # Context tag 0, object ID
                0x19,
                0x55,  # Context tag 1, property ID
                0x3E,  # Opening tag 3
                0x21,
                0x01,  # Value 1
                0x21,
                0x02,  # Value 2
                0x21,
                0x03,  # Value 3
                0x3F,  # Closing tag 3
            ]
        )
        result = decode_read_property_ack(data)

        assert result.get("property_value") == [1, 2, 3]


class TestDecodeWritePropertyRequest:
    """Test decode_write_property_request function."""

    def test_decode_write_property_request(self) -> None:
        """Test WriteProperty request decoding."""
        # Object ID: context tag 0, binary-output #20
        # Property ID: context tag 1, present-value (85)
        # Value: opening tag 3, enumerated 1 (active), closing tag 3
        data = bytes(
            [
                0x0C,
                0x01,
                0x00,
                0x00,
                0x14,  # Context tag 0, binary-output #20
                0x19,
                0x55,  # Context tag 1, property ID = 85
                0x3E,  # Opening tag 3
                0x91,
                0x01,  # App tag 9 (enumerated), len 1, value 1
                0x3F,  # Closing tag 3
            ]
        )
        result = decode_write_property_request(data)

        assert result.get("object_identifier", {}).get("object_type") == 4
        assert result.get("object_identifier", {}).get("instance") == 20
        assert result.get("property_identifier") == 85
        assert result.get("property_value") == 1

    def test_decode_write_property_request_with_priority(self) -> None:
        """Test WriteProperty request with priority."""
        # Object ID, Property ID, Value, Priority: context tag 4, value 8
        data = bytes(
            [
                0x0C,
                0x01,
                0x00,
                0x00,
                0x05,  # Binary-output #5
                0x19,
                0x55,  # Property ID
                0x3E,
                0x91,
                0x00,
                0x3F,  # Value = 0
                0x49,
                0x08,  # Priority = 8
            ]
        )
        result = decode_write_property_request(data)

        assert result.get("property_value") == 0
        assert result.get("priority") == 8


class TestGetPropertyName:
    """Test get_property_name function."""

    def test_get_known_property_names(self) -> None:
        """Test getting known property names."""
        assert get_property_name(0) == "acked-transitions"
        assert get_property_name(28) == "description"
        assert get_property_name(77) == "object-name"
        assert get_property_name(85) == "present-value"
        assert get_property_name(120) == "vendor-identifier"

    def test_get_unknown_property_name(self) -> None:
        """Test getting unknown property name."""
        assert get_property_name(9999) == "property-9999"
