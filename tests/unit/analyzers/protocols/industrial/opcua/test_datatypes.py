"""Tests for OPC UA data type parsers."""

from __future__ import annotations

from oscura.analyzers.protocols.industrial.opcua.datatypes import (
    parse_node_id,
    parse_string,
    parse_variant,
)


class TestParseString:
    """Test suite for OPC UA string parsing."""

    def test_parse_simple_string(self) -> None:
        """Test parsing a simple UTF-8 string."""
        # String "Hello" with length prefix
        data = b"\x05\x00\x00\x00Hello"
        result, consumed = parse_string(data, 0)

        assert result == "Hello"
        assert consumed == 9  # 4 bytes length + 5 bytes data

    def test_parse_empty_string(self) -> None:
        """Test parsing an empty string."""
        data = b"\x00\x00\x00\x00"
        result, consumed = parse_string(data, 0)

        assert result == ""
        assert consumed == 4

    def test_parse_null_string(self) -> None:
        """Test parsing a null string (length = -1)."""
        data = b"\xff\xff\xff\xff"
        result, consumed = parse_string(data, 0)

        assert result is None
        assert consumed == 4

    def test_parse_string_with_offset(self) -> None:
        """Test parsing string at non-zero offset."""
        data = b"\x00\x00\x00\x00\x03\x00\x00\x00ABC"
        result, consumed = parse_string(data, 4)

        assert result == "ABC"
        assert consumed == 7  # 4 bytes length + 3 bytes data

    def test_parse_string_insufficient_data(self) -> None:
        """Test parsing string with insufficient data."""
        # Claims 10 bytes but only 5 available
        data = b"\x0a\x00\x00\x00Hello"
        result, consumed = parse_string(data, 0)

        assert result is None
        assert consumed == 4  # Only length was consumed

    def test_parse_string_no_data(self) -> None:
        """Test parsing with no data."""
        data = b""
        result, consumed = parse_string(data, 0)

        assert result is None
        assert consumed == 0

    def test_parse_string_invalid_utf8(self) -> None:
        """Test parsing string with invalid UTF-8."""
        # Invalid UTF-8 sequence
        data = b"\x03\x00\x00\x00\xff\xfe\xfd"
        result, consumed = parse_string(data, 0)

        # Should use replacement characters
        assert result is not None
        assert consumed == 7

    def test_parse_long_string(self) -> None:
        """Test parsing a longer string."""
        text = "opc.tcp://localhost:4840/OPCUA/Server"
        data = len(text).to_bytes(4, "little") + text.encode("utf-8")
        result, consumed = parse_string(data, 0)

        assert result == text
        assert consumed == 4 + len(text)


class TestParseNodeId:
    """Test suite for OPC UA NodeId parsing."""

    def test_parse_twobyte_nodeid(self) -> None:
        """Test parsing TwoByte NodeId (ns=0, id < 256)."""
        # Encoding 0x00, identifier 42
        data = b"\x00\x2a"
        result, consumed = parse_node_id(data, 0)

        assert result == "i=42"
        assert consumed == 2

    def test_parse_fourbyte_nodeid(self) -> None:
        """Test parsing FourByte NodeId."""
        # Encoding 0x01, namespace 2, identifier 1001
        data = b"\x01\x02\xe9\x03"
        result, consumed = parse_node_id(data, 0)

        assert result == "ns=2;i=1001"
        assert consumed == 4

    def test_parse_fourbyte_nodeid_ns0(self) -> None:
        """Test parsing FourByte NodeId with namespace 0."""
        # Encoding 0x01, namespace 0, identifier 100
        data = b"\x01\x00\x64\x00"
        result, consumed = parse_node_id(data, 0)

        assert result == "i=100"
        assert consumed == 4

    def test_parse_numeric_nodeid(self) -> None:
        """Test parsing full Numeric NodeId."""
        # Encoding 0x02, namespace 5, identifier 100000
        data = b"\x02\x05\x00\xa0\x86\x01\x00"
        result, consumed = parse_node_id(data, 0)

        assert result == "ns=5;i=100000"
        assert consumed == 7

    def test_parse_numeric_nodeid_ns0(self) -> None:
        """Test parsing Numeric NodeId with namespace 0."""
        # Encoding 0x02, namespace 0, identifier 42
        data = b"\x02\x00\x00\x2a\x00\x00\x00"
        result, consumed = parse_node_id(data, 0)

        assert result == "i=42"
        assert consumed == 7

    def test_parse_string_nodeid(self) -> None:
        """Test parsing String NodeId."""
        # Encoding 0x03, namespace 2, string "MyVariable"
        string = "MyVariable"
        data = bytearray()
        data.append(0x03)  # String encoding
        data.extend((2).to_bytes(2, "little"))  # Namespace 2
        data.extend(len(string).to_bytes(4, "little"))
        data.extend(string.encode("utf-8"))

        result, consumed = parse_node_id(bytes(data), 0)

        assert result == "ns=2;s=MyVariable"
        assert consumed == 1 + 2 + 4 + len(string)

    def test_parse_string_nodeid_ns0(self) -> None:
        """Test parsing String NodeId with namespace 0."""
        # Encoding 0x03, namespace 0, string "Test"
        string = "Test"
        data = bytearray()
        data.append(0x03)
        data.extend((0).to_bytes(2, "little"))
        data.extend(len(string).to_bytes(4, "little"))
        data.extend(string.encode("utf-8"))

        result, consumed = parse_node_id(bytes(data), 0)

        assert result == "s=Test"
        assert consumed == 1 + 2 + 4 + len(string)

    def test_parse_guid_nodeid(self) -> None:
        """Test parsing GUID NodeId."""
        # Encoding 0x04, namespace 1, 16-byte GUID
        data = bytearray()
        data.append(0x04)
        data.extend((1).to_bytes(2, "little"))
        data.extend(b"\x12\x34\x56\x78" * 4)  # 16 bytes

        result, consumed = parse_node_id(bytes(data), 0)

        assert result.startswith("ns=1;g=")
        assert consumed == 19  # 1 + 2 + 16

    def test_parse_bytestring_nodeid(self) -> None:
        """Test parsing ByteString NodeId."""
        # Encoding 0x05, namespace 3, byte string
        bs = b"ABC"
        data = bytearray()
        data.append(0x05)
        data.extend((3).to_bytes(2, "little"))
        data.extend(len(bs).to_bytes(4, "little"))
        data.extend(bs)

        result, consumed = parse_node_id(bytes(data), 0)

        assert result == "ns=3;b=ABC"
        assert consumed == 1 + 2 + 4 + len(bs)

    def test_parse_nodeid_insufficient_data(self) -> None:
        """Test parsing NodeId with insufficient data."""
        # TwoByte encoding but no identifier byte
        data = b"\x00"
        result, consumed = parse_node_id(data, 0)

        assert result == "i=0"
        assert consumed == 1

    def test_parse_nodeid_empty_data(self) -> None:
        """Test parsing NodeId with no data."""
        data = b""
        result, consumed = parse_node_id(data, 0)

        assert result == "i=0"
        assert consumed == 0

    def test_parse_nodeid_with_offset(self) -> None:
        """Test parsing NodeId at non-zero offset."""
        data = b"\x00\x00\x00\x00\x01\x02\xe9\x03"
        result, consumed = parse_node_id(data, 4)

        assert result == "ns=2;i=1001"
        assert consumed == 4


class TestParseVariant:
    """Test suite for OPC UA Variant parsing."""

    def test_parse_boolean_variant(self) -> None:
        """Test parsing Boolean variant."""
        # Type ID 1 (Boolean), value True
        data = b"\x01\x01"
        result, consumed = parse_variant(data, 0)

        assert result is True
        assert consumed == 2

    def test_parse_byte_variant(self) -> None:
        """Test parsing Byte variant."""
        # Type ID 3 (Byte), value 42
        data = b"\x03\x2a"
        result, consumed = parse_variant(data, 0)

        assert result == 42
        assert consumed == 2

    def test_parse_int16_variant(self) -> None:
        """Test parsing Int16 variant."""
        # Type ID 4 (Int16), value -100
        data = b"\x04\x9c\xff"
        result, consumed = parse_variant(data, 0)

        assert result == -100
        assert consumed == 3

    def test_parse_uint16_variant(self) -> None:
        """Test parsing UInt16 variant."""
        # Type ID 5 (UInt16), value 1000
        data = b"\x05\xe8\x03"
        result, consumed = parse_variant(data, 0)

        assert result == 1000
        assert consumed == 3

    def test_parse_int32_variant(self) -> None:
        """Test parsing Int32 variant."""
        # Type ID 6 (Int32), value -50000
        data = b"\x06\xb0\x3c\xff\xff"
        result, consumed = parse_variant(data, 0)

        assert result == -50000
        assert consumed == 5

    def test_parse_uint32_variant(self) -> None:
        """Test parsing UInt32 variant."""
        # Type ID 7 (UInt32), value 100000
        data = b"\x07\xa0\x86\x01\x00"
        result, consumed = parse_variant(data, 0)

        assert result == 100000
        assert consumed == 5

    def test_parse_string_variant(self) -> None:
        """Test parsing String variant."""
        # Type ID 12 (String)
        string = "TestValue"
        data = bytearray()
        data.append(12)  # String type
        data.extend(len(string).to_bytes(4, "little"))
        data.extend(string.encode("utf-8"))

        result, consumed = parse_variant(bytes(data), 0)

        assert result == "TestValue"
        assert consumed == 1 + 4 + len(string)

    def test_parse_nodeid_variant(self) -> None:
        """Test parsing NodeId variant."""
        # Type ID 17 (NodeId), TwoByte encoding
        data = b"\x11\x00\x2a"  # NodeId type, encoding 0x00, id 42
        result, consumed = parse_variant(data, 0)

        assert result == "i=42"
        assert consumed == 3

    def test_parse_array_variant(self) -> None:
        """Test parsing array variant (simplified)."""
        # Type ID 7 (UInt32) with array flag (bit 6)
        data = b"\x47"  # 0x40 | 0x07 = array of UInt32
        result, consumed = parse_variant(data, 0)

        # Returns indication of array type
        assert isinstance(result, dict)
        assert result["array"] is True
        assert result["type_id"] == 7
        assert consumed == 1

    def test_parse_unsupported_variant_type(self) -> None:
        """Test parsing unsupported variant type."""
        # Type ID 10 (Float) - not fully implemented
        data = b"\x0a\x00\x00\x00\x00"
        result, consumed = parse_variant(data, 0)

        # Returns type indicator for unsupported types
        assert isinstance(result, dict)
        assert result["type_id"] == 10
        assert result["unsupported"] is True

    def test_parse_variant_insufficient_data(self) -> None:
        """Test parsing variant with insufficient data."""
        # UInt32 type but no value bytes
        data = b"\x07"
        result, consumed = parse_variant(data, 0)

        assert result is None
        assert consumed == 1

    def test_parse_variant_empty_data(self) -> None:
        """Test parsing variant with no data."""
        data = b""
        result, consumed = parse_variant(data, 0)

        assert result is None
        assert consumed == 0

    def test_parse_variant_with_offset(self) -> None:
        """Test parsing variant at non-zero offset."""
        data = b"\x00\x00\x00\x00\x03\x42"
        result, consumed = parse_variant(data, 4)

        assert result == 0x42
        assert consumed == 2
