"""Comprehensive unit tests for binary parsing utilities.

Tests binary parsing functionality:
"""

from __future__ import annotations

import struct

import pytest

from oscura.analyzers.packet.parser import (
    BinaryParser,
    PacketParser,
    TLVRecord,
    parse_tlv,
    parse_tlv_nested,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.packet]


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestBinaryParser:
    """Test BinaryParser for fast struct parsing."""

    def test_basic_parsing(self):
        """Test basic binary parsing."""
        parser = BinaryParser(">HH")  # Two big-endian unsigned shorts
        data = bytes([0x00, 0x01, 0x00, 0x02])

        result = parser.unpack(data)

        assert result == (1, 2)

    def test_format_property(self):
        """Test format string property."""
        parser = BinaryParser(">HBB")

        assert parser.format == ">HBB"

    def test_size_property(self):
        """Test size property calculation."""
        parser = BinaryParser(">HH")  # 2 + 2 = 4 bytes

        assert parser.size == 4

    def test_unpack_from_offset(self):
        """Test unpacking from offset."""
        parser = BinaryParser(">H")
        data = bytes([0xFF, 0xFF, 0x00, 0x42])

        result = parser.unpack_from(data, offset=2)

        assert result == (0x42,)

    def test_pack(self):
        """Test packing values."""
        parser = BinaryParser(">HH")

        packed = parser.pack(1, 2)

        assert packed == bytes([0x00, 0x01, 0x00, 0x02])

    def test_iter_unpack(self):
        """Test iterating over repeated structures."""
        parser = BinaryParser(">HH")
        data = bytes([0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04])

        results = list(parser.iter_unpack(data))

        assert results == [(1, 2), (3, 4)]

    def test_big_endian(self):
        """Test big-endian parsing."""
        parser = BinaryParser(">I")  # Big-endian unsigned int
        data = bytes([0x00, 0x00, 0x01, 0x00])

        result = parser.unpack(data)

        assert result == (256,)

    def test_little_endian(self):
        """Test little-endian parsing."""
        parser = BinaryParser("<I")  # Little-endian unsigned int
        data = bytes([0x00, 0x01, 0x00, 0x00])

        result = parser.unpack(data)

        assert result == (256,)

    def test_mixed_types(self):
        """Test parsing mixed data types."""
        parser = BinaryParser(">HHBBI")  # short, short, byte, byte, int
        data = struct.pack(">HHBBI", 1, 2, 3, 4, 5)

        result = parser.unpack(data)

        assert result == (1, 2, 3, 4, 5)

    def test_signed_integers(self):
        """Test signed integer parsing."""
        parser = BinaryParser(">hh")  # Signed shorts
        data = struct.pack(">hh", -1, 100)

        result = parser.unpack(data)

        assert result == (-1, 100)

    def test_float_parsing(self):
        """Test floating point parsing."""
        parser = BinaryParser(">ff")  # Two floats
        data = struct.pack(">ff", 1.5, 2.5)

        result = parser.unpack(data)

        assert result[0] == pytest.approx(1.5)
        assert result[1] == pytest.approx(2.5)

    def test_buffer_too_small(self):
        """Test error handling for insufficient data."""
        parser = BinaryParser(">HH")
        data = bytes([0x00, 0x01])  # Only 2 bytes, need 4

        with pytest.raises(struct.error):
            parser.unpack(data)

    def test_unpack_truncates_excess(self):
        """Test that unpack only reads required bytes."""
        parser = BinaryParser(">H")
        data = bytes([0x00, 0x42, 0xFF, 0xFF])  # Extra bytes

        result = parser.unpack(data)

        assert result == (0x42,)

    def test_network_byte_order(self):
        """Test network byte order (big-endian)."""
        parser = BinaryParser("!H")  # Network byte order
        data = bytes([0x01, 0x00])

        result = parser.unpack(data)

        assert result == (256,)


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestPacketParser:
    """Test PacketParser for multi-field packet parsing."""

    def test_basic_packet_parsing(self):
        """Test basic packet field parsing."""
        fields = [
            ("sync", "H"),
            ("length", "H"),
            ("type", "B"),
        ]
        parser = PacketParser(fields, byte_order=">")

        data = struct.pack(">HHB", 0xAA55, 10, 5)
        result = parser.parse(data)

        assert result["sync"] == 0xAA55
        assert result["length"] == 10
        assert result["type"] == 5

    def test_parse_from_offset(self):
        """Test parsing from buffer offset."""
        fields = [("value", "H")]
        parser = PacketParser(fields, byte_order=">")

        data = bytes([0xFF, 0xFF, 0x00, 0x42])
        result = parser.parse(data, offset=2)

        assert result["value"] == 0x42

    def test_pack_fields(self):
        """Test packing named fields."""
        fields = [
            ("sync", "H"),
            ("type", "B"),
        ]
        parser = PacketParser(fields, byte_order=">")

        packed = parser.pack(sync=0xAA55, type=5)

        assert packed == struct.pack(">HB", 0xAA55, 5)

    def test_size_property(self):
        """Test packet size calculation."""
        fields = [
            ("a", "H"),
            ("b", "B"),
            ("c", "I"),
        ]
        parser = PacketParser(fields)

        assert parser.size == 2 + 1 + 4

    def test_little_endian_packet(self):
        """Test little-endian packet parsing."""
        fields = [("value", "I")]
        parser = PacketParser(fields, byte_order="<")

        data = struct.pack("<I", 0x12345678)
        result = parser.parse(data)

        assert result["value"] == 0x12345678

    def test_multiple_field_types(self):
        """Test packet with various field types."""
        fields = [
            ("magic", "H"),
            ("version", "B"),
            ("flags", "B"),
            ("length", "I"),
        ]
        parser = PacketParser(fields, byte_order=">")

        data = struct.pack(">HBBI", 0xCAFE, 1, 0x80, 1024)
        result = parser.parse(data)

        assert result["magic"] == 0xCAFE
        assert result["version"] == 1
        assert result["flags"] == 0x80
        assert result["length"] == 1024


# =============================================================================
# =============================================================================


@pytest.mark.unit
class TestTLVParser:
    """Test Type-Length-Value record parsing."""

    def test_basic_tlv_parsing(self):
        """Test basic TLV record parsing."""
        # Type=1, Length=3, Value=[0x01, 0x02, 0x03]
        data = bytes([0x01, 0x03, 0x01, 0x02, 0x03])

        records = parse_tlv(data, type_size=1, length_size=1)

        assert len(records) == 1
        assert records[0].type_id == 1
        assert records[0].length == 3
        assert records[0].value == bytes([0x01, 0x02, 0x03])

    def test_multiple_tlv_records(self):
        """Test parsing multiple TLV records."""
        # Record 1: Type=1, Length=2
        # Record 2: Type=2, Length=1
        data = bytes([0x01, 0x02, 0xAA, 0xBB, 0x02, 0x01, 0xFF])

        records = parse_tlv(data, type_size=1, length_size=1)

        assert len(records) == 2
        assert records[0].type_id == 1
        assert records[0].value == bytes([0xAA, 0xBB])
        assert records[1].type_id == 2
        assert records[1].value == bytes([0xFF])

    def test_tlv_with_16bit_fields(self):
        """Test TLV with 16-bit type and length."""
        data = struct.pack(">HH", 0x0001, 0x0004) + bytes([0x01, 0x02, 0x03, 0x04])

        records = parse_tlv(data, type_size=2, length_size=2, big_endian=True)

        assert len(records) == 1
        assert records[0].type_id == 1
        assert records[0].length == 4

    def test_tlv_little_endian(self):
        """Test TLV with little-endian byte order."""
        data = struct.pack("<HH", 0x0001, 0x0002) + bytes([0xAA, 0xBB])

        records = parse_tlv(data, type_size=2, length_size=2, big_endian=False)

        assert len(records) == 1
        assert records[0].type_id == 1

    def test_tlv_include_length_in_length(self):
        """Test TLV where length includes header."""
        # Type=1, Length=5 (includes 2 bytes header), Value=3 bytes
        data = bytes([0x01, 0x05, 0x01, 0x02, 0x03])

        records = parse_tlv(data, type_size=1, length_size=1, include_length_in_length=True)

        assert len(records) == 1
        assert records[0].length == 3  # Adjusted length

    def test_tlv_truncated_record(self):
        """Test handling of truncated TLV record."""
        # Type=1, Length=10, but only 2 bytes of value
        data = bytes([0x01, 0x0A, 0x01, 0x02])

        records = parse_tlv(data, type_size=1, length_size=1)

        # Should stop parsing, not crash
        assert len(records) == 0

    def test_tlv_zero_length(self):
        """Test TLV with zero-length value."""
        data = bytes([0x01, 0x00])

        records = parse_tlv(data, type_size=1, length_size=1)

        assert len(records) == 1
        assert records[0].length == 0
        assert records[0].value == b""

    def test_tlv_offset_tracking(self):
        """Test that TLV records track their offset."""
        data = bytes([0x01, 0x02, 0xAA, 0xBB, 0x02, 0x01, 0xFF])

        records = parse_tlv(data, type_size=1, length_size=1)

        assert records[0].offset == 0
        assert records[1].offset == 4

    def test_tlv_32bit_fields(self):
        """Test TLV with 32-bit type and length."""
        data = struct.pack(">II", 0x00000001, 0x00000003) + bytes([0x01, 0x02, 0x03])

        records = parse_tlv(data, type_size=4, length_size=4, big_endian=True)

        assert len(records) == 1
        assert records[0].type_id == 1

    def test_tlv_empty_buffer(self):
        """Test parsing empty buffer."""
        records = parse_tlv(b"", type_size=1, length_size=1)

        assert len(records) == 0

    def test_tlv_incomplete_header(self):
        """Test with incomplete header."""
        data = bytes([0x01])  # Only type, no length

        records = parse_tlv(data, type_size=1, length_size=1)

        assert len(records) == 0


# =============================================================================
# Nested TLV Tests
# =============================================================================


@pytest.mark.unit
class TestNestedTLV:
    """Test nested TLV parsing."""

    def test_nested_tlv_basic(self):
        """Test basic nested TLV parsing."""
        # Container: Type=1, contains nested TLV
        nested_data = bytes([0x02, 0x02, 0xAA, 0xBB])  # Nested record
        data = bytes([0x01, len(nested_data)]) + nested_data

        result = parse_tlv_nested(data, type_size=1, length_size=1, container_types={1})

        assert 1 in result
        assert isinstance(result[1], dict)
        assert 2 in result[1]

    def test_nested_tlv_non_container(self):
        """Test nested TLV with non-container types."""
        data = bytes([0x01, 0x02, 0xAA, 0xBB])

        result = parse_tlv_nested(data, type_size=1, length_size=1, container_types=set())

        assert 1 in result
        assert result[1] == bytes([0xAA, 0xBB])

    def test_nested_tlv_mixed(self):
        """Test nested TLV with mixed container/value types."""
        # Type 1: regular value
        # Type 2: container with nested TLV
        nested = bytes([0x03, 0x01, 0xFF])
        data = bytes([0x01, 0x01, 0xAA, 0x02, len(nested)]) + nested

        result = parse_tlv_nested(data, type_size=1, length_size=1, container_types={2})

        assert result[1] == bytes([0xAA])
        assert isinstance(result[2], dict)
        assert 3 in result[2]

    def test_nested_tlv_empty_container(self):
        """Test nested TLV with empty container."""
        data = bytes([0x01, 0x00])  # Container with no data

        result = parse_tlv_nested(data, type_size=1, length_size=1, container_types={1})

        assert 1 in result
        assert result[1] == {}


# =============================================================================
# Data Class Tests
# =============================================================================


@pytest.mark.unit
class TestTLVDataClass:
    """Test TLVRecord data class."""

    def test_tlv_record_creation(self):
        """Test TLVRecord creation."""
        record = TLVRecord(type_id=1, length=3, value=b"\x01\x02\x03", offset=0)

        assert record.type_id == 1
        assert record.length == 3
        assert record.value == b"\x01\x02\x03"
        assert record.offset == 0

    def test_tlv_record_default_offset(self):
        """Test TLVRecord with default offset."""
        record = TLVRecord(type_id=1, length=2, value=b"\xaa\xbb")

        assert record.offset == 0


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit
class TestParserEdgeCases:
    """Test edge cases and error handling."""

    def test_parser_with_padding(self):
        """Test parsing with padding bytes."""
        parser = BinaryParser(">Hxx")  # Short with 2 padding bytes
        data = bytes([0x00, 0x42, 0xFF, 0xFF])

        result = parser.unpack(data)

        assert result == (0x42,)

    def test_parser_string_format(self):
        """Test parsing with string fields."""
        parser = BinaryParser(">4s")  # 4-byte string
        data = b"TEST"

        result = parser.unpack(data)

        assert result == (b"TEST",)

    def test_parser_bool_format(self):
        """Test parsing with boolean fields."""
        parser = BinaryParser(">??")  # Two booleans
        data = bytes([0x01, 0x00])

        result = parser.unpack(data)

        assert result == (True, False)

    def test_tlv_negative_length(self):
        """Test TLV handling of negative adjusted length."""
        # Include length in length, but length < header size
        data = bytes([0x01, 0x01, 0xFF])

        records = parse_tlv(data, type_size=1, length_size=1, include_length_in_length=True)

        # Should stop parsing when length is invalid
        assert len(records) == 0

    def test_parser_alignment(self):
        """Test parsing with alignment considerations."""
        parser = BinaryParser("=BHI")  # Native alignment
        data = struct.pack("=BHI", 1, 2, 3)

        result = parser.unpack(data)

        assert result[0] == 1
        assert result[1] == 2
        assert result[2] == 3
