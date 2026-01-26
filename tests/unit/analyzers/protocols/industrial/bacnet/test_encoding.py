"""Unit tests for BACnet encoding utilities."""

from __future__ import annotations

import pytest

from oscura.analyzers.protocols.industrial.bacnet.encoding import (
    parse_application_tag,
    parse_character_string,
    parse_enumerated,
    parse_object_identifier,
    parse_tag,
    parse_unsigned,
)


class TestParseTag:
    """Test parse_tag function."""

    def test_parse_context_tag(self) -> None:
        """Test parsing context-specific tag."""
        # Context tag 0, length 4
        data = bytes([0x08])  # Tag number 0, context, length 0
        tag, consumed = parse_tag(data, 0)

        assert tag["tag_number"] == 0
        assert tag["context_specific"] is True
        assert tag["length"] == 0
        assert consumed == 1

    def test_parse_application_tag(self) -> None:
        """Test parsing application tag."""
        # Application tag 2 (unsigned), length 1
        data = bytes([0x21])  # Tag number 2, application, length 1
        tag, consumed = parse_tag(data, 0)

        assert tag["tag_number"] == 2
        assert tag["context_specific"] is False
        assert tag["length"] == 1
        assert consumed == 1

    def test_parse_extended_tag_number(self) -> None:
        """Test parsing extended tag number (tag 15 means next byte)."""
        # Tag number 15 (extended), next byte = 200
        data = bytes([0xF0, 200])
        tag, consumed = parse_tag(data, 0)

        assert tag["tag_number"] == 200
        assert consumed == 2

    def test_parse_extended_length(self) -> None:
        """Test parsing extended length field for application tag."""
        # Tag with extended length (lvt = 5 for application tags)
        # 0x25 = 0010 0101 = tag 2, application, lvt=5 (extended)
        data = bytes([0x25, 100])  # Tag 2, extended length, actual length 100
        tag, consumed = parse_tag(data, 0)

        assert tag["tag_number"] == 2
        assert tag["context_specific"] is False  # Application tag
        assert tag["length"] == 100
        assert consumed == 2

    def test_parse_opening_tag(self) -> None:
        """Test parsing opening tag (length field = 5)."""
        # 0x3E = 0011 1110 = tag 3, context, opening (lvt=6 but with context bit becomes 14=0xE)
        # Actually for opening tag: tag_num=3 (bits 4-7), context=1 (bit 3), lvt=6 (bits 0-2)
        # = 0011 1110 = 0x3E
        # Wait, opening tag has lvt=6 which needs context bit set
        # Let's use proper encoding: tag 3, context-specific, opening
        data = bytes([0x3E])  # Context tag 3, opening
        tag, consumed = parse_tag(data, 0)

        assert tag["tag_number"] == 3
        assert tag["context_specific"] is True  # Opening/closing tags are context-specific
        assert tag["is_opening"] is True
        assert tag["is_closing"] is False
        assert consumed == 1

    def test_parse_closing_tag(self) -> None:
        """Test parsing closing tag (length field = 6)."""
        data = bytes([0x3F])  # Context tag 3, closing
        tag, consumed = parse_tag(data, 0)

        assert tag["tag_number"] == 3
        assert tag["context_specific"] is True  # Opening/closing tags are context-specific
        assert tag["is_opening"] is False
        assert tag["is_closing"] is True
        assert consumed == 1

    def test_parse_tag_offset_beyond_data(self) -> None:
        """Test error when offset is beyond data length."""
        data = bytes([0x21])
        with pytest.raises(ValueError, match="Offset beyond data length"):
            parse_tag(data, 10)

    def test_parse_tag_missing_extended_tag(self) -> None:
        """Test error when extended tag number byte is missing."""
        data = bytes([0xF0])  # Extended tag marker but no next byte
        with pytest.raises(ValueError, match="Extended tag number missing"):
            parse_tag(data, 0)


class TestParseUnsigned:
    """Test parse_unsigned function."""

    def test_parse_1_byte_unsigned(self) -> None:
        """Test parsing 1-byte unsigned integer."""
        data = bytes([0x42])
        value, consumed = parse_unsigned(data, 0, 1)

        assert value == 66
        assert consumed == 1

    def test_parse_2_byte_unsigned(self) -> None:
        """Test parsing 2-byte unsigned integer."""
        data = bytes([0x01, 0x23])
        value, consumed = parse_unsigned(data, 0, 2)

        assert value == 0x0123
        assert consumed == 2

    def test_parse_4_byte_unsigned(self) -> None:
        """Test parsing 4-byte unsigned integer."""
        data = bytes([0x12, 0x34, 0x56, 0x78])
        value, consumed = parse_unsigned(data, 0, 4)

        assert value == 0x12345678
        assert consumed == 4

    def test_parse_unsigned_invalid_length(self) -> None:
        """Test error with invalid length (>4 bytes)."""
        data = bytes([0x00] * 10)
        with pytest.raises(ValueError, match="Invalid unsigned integer length"):
            parse_unsigned(data, 0, 5)

    def test_parse_unsigned_data_too_short(self) -> None:
        """Test error when data is too short."""
        data = bytes([0x12])
        with pytest.raises(ValueError, match="Data too short"):
            parse_unsigned(data, 0, 4)


class TestParseEnumerated:
    """Test parse_enumerated function."""

    def test_parse_enumerated(self) -> None:
        """Test parsing enumerated value (same as unsigned)."""
        data = bytes([0x03])
        value, consumed = parse_enumerated(data, 0, 1)

        assert value == 3
        assert consumed == 1


class TestParseObjectIdentifier:
    """Test parse_object_identifier function."""

    def test_parse_device_object(self) -> None:
        """Test parsing device object identifier (type 8)."""
        # Object type 8 (device), instance 8
        # 0x02000008 = 00000010 00000000 00000000 00001000
        # Top 10 bits: 0000001000 = 8, bottom 22 bits: 8
        data = bytes([0x02, 0x00, 0x00, 0x08])
        obj_id, consumed = parse_object_identifier(data, 0)

        assert obj_id["object_type"] == 8
        assert obj_id["object_type_name"] == "device"
        assert obj_id["instance"] == 8
        assert consumed == 4

    def test_parse_analog_input_object(self) -> None:
        """Test parsing analog-input object identifier (type 0)."""
        # Object type 0 (analog-input), instance 123
        # 0x0000007B = 00000000 00000000 00000000 01111011
        data = bytes([0x00, 0x00, 0x00, 0x7B])
        obj_id, consumed = parse_object_identifier(data, 0)

        assert obj_id["object_type"] == 0
        assert obj_id["object_type_name"] == "analog-input"
        assert obj_id["instance"] == 123
        assert consumed == 4

    def test_parse_binary_output_object(self) -> None:
        """Test parsing binary-output object identifier (type 4)."""
        # Object type 4 (binary-output), instance 456
        # Top 10 bits: 4, bottom 22 bits: 456
        # 0x010001C8
        data = bytes([0x01, 0x00, 0x01, 0xC8])
        obj_id, consumed = parse_object_identifier(data, 0)

        assert obj_id["object_type"] == 4
        assert obj_id["object_type_name"] == "binary-output"
        assert obj_id["instance"] == 456
        assert consumed == 4

    def test_parse_unknown_object_type(self) -> None:
        """Test parsing unknown object type."""
        # Object type 999, instance 1
        # Top 10 bits: 999 = 0x3E7, bottom 22 bits: 1
        # 0xF9C00001
        data = bytes([0xF9, 0xC0, 0x00, 0x01])
        obj_id, consumed = parse_object_identifier(data, 0)

        assert obj_id["object_type"] == 999
        assert obj_id["object_type_name"] == "type-999"
        assert obj_id["instance"] == 1
        assert consumed == 4

    def test_parse_object_identifier_data_too_short(self) -> None:
        """Test error when data is too short."""
        data = bytes([0x02, 0x00])
        with pytest.raises(ValueError, match="Data too short"):
            parse_object_identifier(data, 0)


class TestParseCharacterString:
    """Test parse_character_string function."""

    def test_parse_utf8_string(self) -> None:
        """Test parsing UTF-8 encoded string."""
        data = bytes([0x00]) + b"Building 1"  # 0x00 = UTF-8 encoding
        string, consumed = parse_character_string(data, 0, 11)

        assert string == "Building 1"
        assert consumed == 11

    def test_parse_latin1_string(self) -> None:
        """Test parsing ISO 8859-1 (Latin-1) encoded string."""
        data = bytes([0x05]) + b"Caf\xe9"  # 0x05 = ISO 8859-1
        string, consumed = parse_character_string(data, 0, 5)

        assert string == "Café"
        assert consumed == 5

    def test_parse_empty_string(self) -> None:
        """Test parsing empty string."""
        data = bytes([0x00])  # UTF-8 encoding, no data
        string, consumed = parse_character_string(data, 0, 1)

        assert string == ""
        assert consumed == 1

    def test_parse_unsupported_encoding(self) -> None:
        """Test parsing unsupported encoding returns hex."""
        data = bytes([0x02, 0x01, 0x02, 0x03])  # Encoding 2 (JIS C 6226)
        string, consumed = parse_character_string(data, 0, 4)

        assert string == "010203"  # Hex representation
        assert consumed == 4


class TestParseApplicationTag:
    """Test parse_application_tag function."""

    def test_parse_null_tag(self) -> None:
        """Test parsing null application tag."""
        data = bytes([0x00])  # Tag 0 (null)
        value, consumed = parse_application_tag(data, 0)

        assert value is None
        assert consumed == 1

    def test_parse_boolean_true(self) -> None:
        """Test parsing boolean true."""
        data = bytes([0x11])  # Tag 1 (boolean), length 1 (true)
        value, consumed = parse_application_tag(data, 0)

        assert value is True
        assert consumed == 1

    def test_parse_boolean_false(self) -> None:
        """Test parsing boolean false."""
        data = bytes([0x10])  # Tag 1 (boolean), length 0 (false)
        value, consumed = parse_application_tag(data, 0)

        assert value is False
        assert consumed == 1

    def test_parse_unsigned_integer(self) -> None:
        """Test parsing unsigned integer."""
        data = bytes([0x21, 0x42])  # Tag 2 (unsigned), length 1, value 66
        value, consumed = parse_application_tag(data, 0)

        assert value == 66
        assert consumed == 2

    def test_parse_signed_integer(self) -> None:
        """Test parsing signed integer."""
        data = bytes([0x31, 0xFF])  # Tag 3 (signed), length 1, value -1
        value, consumed = parse_application_tag(data, 0)

        assert value == -1
        assert consumed == 2

    def test_parse_real_float(self) -> None:
        """Test parsing real (IEEE 754 float)."""
        import struct

        float_bytes = struct.pack(">f", 3.14159)
        data = bytes([0x44]) + float_bytes  # Tag 4 (real), length 4
        value, consumed = parse_application_tag(data, 0)

        assert abs(value - 3.14159) < 0.001  # type: ignore[operator]
        assert consumed == 5

    def test_parse_double(self) -> None:
        """Test parsing double (IEEE 754 double)."""
        import struct

        double_bytes = struct.pack(">d", 2.718281828)
        # Tag 5 (double), application, extended length (lvt=5), length=8
        # 0x55 = 0101 0101 = tag 5, application, lvt=5 (extended length follows)
        data = bytes([0x55, 0x08]) + double_bytes  # Tag 5, length 8
        value, consumed = parse_application_tag(data, 0)

        assert abs(value - 2.718281828) < 0.000001  # type: ignore[operator]
        assert consumed == 10  # tag + len_byte + 8 data bytes

    def test_parse_character_string(self) -> None:
        """Test parsing character string."""
        # Tag 7 (character-string), application, extended length (lvt=5), length=5 (encoding byte + 4 chars)
        # 0x75 = 0111 0101 = tag 7, application, lvt=5 (extended)
        data = bytes([0x75, 0x05, 0x00]) + b"HVAC"  # Tag 7, len 5, UTF-8 encoding + "HVAC"
        value, consumed = parse_application_tag(data, 0)

        assert value == "HVAC"
        assert consumed == 7  # tag + len_byte + 5 data bytes

    def test_parse_enumerated(self) -> None:
        """Test parsing enumerated value."""
        data = bytes([0x91, 0x03])  # Tag 9 (enumerated), length 1, value 3
        value, consumed = parse_application_tag(data, 0)

        assert value == 3
        assert consumed == 2

    def test_parse_object_identifier(self) -> None:
        """Test parsing object identifier."""
        data = bytes([0xC4, 0x02, 0x00, 0x00, 0x08])  # Tag 12, length 4, device #8
        value, consumed = parse_application_tag(data, 0)

        assert isinstance(value, dict)
        assert value["object_type"] == 8
        assert value["object_type_name"] == "device"
        assert value["instance"] == 8
        assert consumed == 5

    def test_parse_context_tag_raises_error(self) -> None:
        """Test that context tags raise error in application tag parser."""
        data = bytes([0x08])  # Context tag
        with pytest.raises(ValueError, match="Expected application tag"):
            parse_application_tag(data, 0)

    def test_parse_opening_tag_raises_error(self) -> None:
        """Test that opening tags raise error (they are context-specific)."""
        data = bytes([0x3E])  # Opening tag (context-specific)
        with pytest.raises(ValueError, match="Expected application tag"):
            parse_application_tag(data, 0)
