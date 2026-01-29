"""Comprehensive tests for protocol_dsl module.


This module provides comprehensive test coverage for:
- FieldDefinition dataclass and properties
- ProtocolDefinition parsing from YAML and dict
- DecodedMessage container behavior
- ProtocolDecoder with all field types
- ProtocolEncoder with all field types
- Convenience functions (load_protocol, decode_message)
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import pytest
import yaml

from oscura.inference.protocol_dsl import (
    DecodedMessage,
    FieldDefinition,
    ProtocolDecoder,
    ProtocolDefinition,
    ProtocolEncoder,
    decode_message,
    load_protocol,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# FieldDefinition Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestFieldDefinition:
    """Test FieldDefinition dataclass."""

    def test_default_values(self) -> None:
        """Test FieldDefinition default values."""
        field = FieldDefinition(name="test_field")

        assert field.name == "test_field"
        assert field.field_type == "uint8"
        assert field.size is None
        assert field.offset is None
        assert field.endian == "big"
        assert field.condition is None
        assert field.enum is None
        assert field.validation is None
        assert field.default is None
        assert field.description == ""
        assert field.value is None
        assert field.size_ref is None
        assert field.element is None
        assert field.count_field is None
        assert field.count is None
        assert field.fields is None

    def test_custom_values(self) -> None:
        """Test FieldDefinition with custom values."""
        field = FieldDefinition(
            name="custom_field",
            field_type="uint32",
            size=4,
            offset=10,
            endian="little",
            condition="type == 1",
            enum={1: "A", 2: "B"},
            validation={"min": 0, "max": 100},
            default=42,
            description="A custom field",
            value=0x12345678,
        )

        assert field.name == "custom_field"
        assert field.field_type == "uint32"
        assert field.size == 4
        assert field.offset == 10
        assert field.endian == "little"
        assert field.condition == "type == 1"
        assert field.enum == {1: "A", 2: "B"}
        assert field.validation == {"min": 0, "max": 100}
        assert field.default == 42
        assert field.description == "A custom field"
        assert field.value == 0x12345678

    def test_size_ref_alias(self) -> None:
        """Test that size_ref is treated as alias for size."""
        field = FieldDefinition(
            name="data",
            field_type="bytes",
            size_ref="length",
        )

        # size_ref should be copied to size in __post_init__
        assert field.size == "length"
        assert field.size_ref == "length"

    def test_size_ref_does_not_override_size(self) -> None:
        """Test that size_ref doesn't override explicit size."""
        field = FieldDefinition(
            name="data",
            field_type="bytes",
            size=10,
            size_ref="length",
        )

        # Explicit size should be preserved
        assert field.size == 10

    def test_type_property_getter(self) -> None:
        """Test type property getter returns field_type."""
        field = FieldDefinition(name="test", field_type="uint16")

        assert field.type == "uint16"
        assert field.type == field.field_type

    def test_type_property_setter(self) -> None:
        """Test type property setter updates field_type."""
        field = FieldDefinition(name="test", field_type="uint8")

        field.type = "int32"

        assert field.field_type == "int32"
        assert field.type == "int32"

    def test_array_field_definition(self) -> None:
        """Test FieldDefinition for array type."""
        field = FieldDefinition(
            name="items",
            field_type="array",
            count=5,
            element={"type": "uint16"},
        )

        assert field.field_type == "array"
        assert field.count == 5
        assert field.element == {"type": "uint16"}

    def test_array_field_with_count_field(self) -> None:
        """Test array field with dynamic count from another field."""
        field = FieldDefinition(
            name="items",
            field_type="array",
            count_field="item_count",
            element={"type": "uint8"},
        )

        assert field.count_field == "item_count"
        assert field.count is None

    def test_struct_field_definition(self) -> None:
        """Test FieldDefinition for struct type with nested fields."""
        nested_field = FieldDefinition(name="inner", field_type="uint8")
        field = FieldDefinition(
            name="nested",
            field_type="struct",
            fields=[nested_field],
        )

        assert field.field_type == "struct"
        assert len(field.fields) == 1
        assert field.fields[0].name == "inner"


# =============================================================================
# ProtocolDefinition Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDefinition:
    """Test ProtocolDefinition class."""

    def test_from_dict_minimal(self) -> None:
        """Test creating ProtocolDefinition from minimal dict."""
        config = {"name": "MinimalProto"}

        proto = ProtocolDefinition.from_dict(config)

        assert proto.name == "MinimalProto"
        assert proto.version == "1.0"
        assert proto.description == ""
        assert proto.endian == "big"
        assert proto.fields == []
        assert proto.settings == {}
        assert proto.framing == {}

    def test_from_dict_full(self) -> None:
        """Test creating ProtocolDefinition from complete dict."""
        config = {
            "name": "FullProtocol",
            "version": "2.0",
            "description": "A full protocol",
            "endian": "little",
            "settings": {"debug": True},
            "framing": {"sync_pattern": "0xAA55"},
            "fields": [
                {"name": "header", "type": "uint16"},
                {"name": "length", "type": "uint8"},
            ],
            "computed_fields": [{"name": "crc", "formula": "sum(data)"}],
            "decoding": {"strict": True},
            "encoding": {"pad": True},
        }

        proto = ProtocolDefinition.from_dict(config)

        assert proto.name == "FullProtocol"
        assert proto.version == "2.0"
        assert proto.description == "A full protocol"
        assert proto.endian == "little"
        assert proto.settings == {"debug": True}
        assert proto.framing == {"sync_pattern": "0xAA55"}
        assert len(proto.fields) == 2
        assert proto.fields[0].name == "header"
        assert proto.fields[1].name == "length"
        assert proto.computed_fields == [{"name": "crc", "formula": "sum(data)"}]
        assert proto.decoding == {"strict": True}
        assert proto.encoding == {"pad": True}

    def test_from_dict_field_endian_inheritance(self) -> None:
        """Test that fields inherit default endianness."""
        config = {
            "name": "EndianTest",
            "endian": "little",
            "fields": [
                {"name": "field1", "type": "uint16"},  # Should inherit little
                {"name": "field2", "type": "uint16", "endian": "big"},  # Override
            ],
        }

        proto = ProtocolDefinition.from_dict(config)

        assert proto.fields[0].endian == "little"
        assert proto.fields[1].endian == "big"

    def test_from_dict_supports_field_type_key(self) -> None:
        """Test that 'field_type' key works as alternative to 'type'."""
        config = {
            "name": "Test",
            "fields": [
                {"name": "field1", "field_type": "uint32"},
            ],
        }

        proto = ProtocolDefinition.from_dict(config)

        assert proto.fields[0].field_type == "uint32"

    def test_from_dict_nested_struct_fields(self) -> None:
        """Test parsing nested struct fields."""
        config = {
            "name": "NestedTest",
            "fields": [
                {
                    "name": "outer",
                    "type": "struct",
                    "fields": [
                        {"name": "inner1", "type": "uint8"},
                        {"name": "inner2", "type": "uint16"},
                    ],
                },
            ],
        }

        proto = ProtocolDefinition.from_dict(config)

        assert proto.fields[0].field_type == "struct"
        assert len(proto.fields[0].fields) == 2
        assert proto.fields[0].fields[0].name == "inner1"
        assert proto.fields[0].fields[1].name == "inner2"

    def test_from_yaml(self) -> None:
        """Test loading ProtocolDefinition from YAML file."""
        config = {
            "name": "YamlProtocol",
            "version": "1.0",
            "endian": "big",
            "fields": [
                {"name": "sync", "type": "uint16"},
                {"name": "data", "type": "bytes", "size": 4},
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            proto = ProtocolDefinition.from_yaml(f.name)

            assert proto.name == "YamlProtocol"
            assert len(proto.fields) == 2

        # Cleanup
        Path(f.name).unlink()

    def test_from_yaml_with_path_object(self) -> None:
        """Test loading from Path object."""
        config = {"name": "PathTest", "fields": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            path = Path(f.name)

            proto = ProtocolDefinition.from_yaml(path)

            assert proto.name == "PathTest"

        path.unlink()

    def test_parse_field_definition_all_attributes(self) -> None:
        """Test _parse_field_definition with all attributes."""
        field_dict = {
            "name": "complex_field",
            "type": "uint32",
            "size": 4,
            "offset": 10,
            "endian": "little",
            "condition": "flag == 1",
            "enum": {1: "ON", 0: "OFF"},
            "validation": {"min": 0, "max": 255},
            "default": 100,
            "description": "Complex field",
            "value": 0xDEADBEEF,
            "size_ref": "len_field",
            "element": {"type": "uint8"},
            "count_field": "count",
            "count": 10,
        }

        field = ProtocolDefinition._parse_field_definition(field_dict, "big")

        assert field.name == "complex_field"
        assert field.field_type == "uint32"
        assert field.condition == "flag == 1"
        assert field.enum == {1: "ON", 0: "OFF"}
        assert field.validation == {"min": 0, "max": 255}
        assert field.default == 100
        assert field.description == "Complex field"
        assert field.value == 0xDEADBEEF
        assert field.element == {"type": "uint8"}
        assert field.count_field == "count"
        assert field.count == 10


# =============================================================================
# DecodedMessage Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestDecodedMessage:
    """Test DecodedMessage container class."""

    @pytest.fixture
    def sample_message(self) -> DecodedMessage:
        """Create a sample decoded message."""
        return DecodedMessage(
            fields={"sync": 0xAA55, "length": 10, "data": b"hello"},
            raw_data=b"\xaa\x55\x0a" + b"hello",
            size=8,
            valid=True,
            errors=[],
        )

    def test_contains_existing_field(self, sample_message: DecodedMessage) -> None:
        """Test __contains__ for existing field."""
        assert "sync" in sample_message
        assert "length" in sample_message
        assert "data" in sample_message

    def test_contains_missing_field(self, sample_message: DecodedMessage) -> None:
        """Test __contains__ for missing field."""
        assert "nonexistent" not in sample_message

    def test_getitem_existing_field(self, sample_message: DecodedMessage) -> None:
        """Test __getitem__ for existing field."""
        assert sample_message["sync"] == 0xAA55
        assert sample_message["length"] == 10
        assert sample_message["data"] == b"hello"

    def test_getitem_missing_field(self, sample_message: DecodedMessage) -> None:
        """Test __getitem__ for missing field raises KeyError."""
        with pytest.raises(KeyError):
            _ = sample_message["nonexistent"]

    def test_iter(self, sample_message: DecodedMessage) -> None:
        """Test __iter__ returns field names."""
        field_names = list(sample_message)

        assert "sync" in field_names
        assert "length" in field_names
        assert "data" in field_names
        assert len(field_names) == 3

    def test_keys(self, sample_message: DecodedMessage) -> None:
        """Test keys() returns field names."""
        keys = sample_message.keys()

        assert set(keys) == {"sync", "length", "data"}

    def test_values(self, sample_message: DecodedMessage) -> None:
        """Test values() returns field values."""
        values = list(sample_message.values())

        assert 0xAA55 in values
        assert 10 in values
        assert b"hello" in values

    def test_items(self, sample_message: DecodedMessage) -> None:
        """Test items() returns field name-value pairs."""
        items = dict(sample_message.items())

        assert items["sync"] == 0xAA55
        assert items["length"] == 10
        assert items["data"] == b"hello"

    def test_get_existing_field(self, sample_message: DecodedMessage) -> None:
        """Test get() for existing field."""
        assert sample_message.get("sync") == 0xAA55

    def test_get_missing_field_default_none(self, sample_message: DecodedMessage) -> None:
        """Test get() for missing field returns None."""
        assert sample_message.get("nonexistent") is None

    def test_get_missing_field_custom_default(self, sample_message: DecodedMessage) -> None:
        """Test get() for missing field returns custom default."""
        assert sample_message.get("nonexistent", "default_value") == "default_value"

    def test_invalid_message(self) -> None:
        """Test invalid message properties."""
        msg = DecodedMessage(
            fields={},
            raw_data=b"\xff\xff",
            size=2,
            valid=False,
            errors=["Field sync: validation failed"],
        )

        assert not msg.valid
        assert len(msg.errors) == 1
        assert "validation failed" in msg.errors[0]


# =============================================================================
# ProtocolDecoder Tests - Integer Types
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderIntegerTypes:
    """Test ProtocolDecoder integer type decoding."""

    def test_decode_uint8(self) -> None:
        """Test decoding uint8 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "uint8"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x42])

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == 0x42

    def test_decode_int8(self) -> None:
        """Test decoding int8 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "int8"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xFF])  # -1 in signed

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == -1

    def test_decode_uint16_big_endian(self) -> None:
        """Test decoding uint16 field in big endian."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "uint16"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x12, 0x34])

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == 0x1234

    def test_decode_uint16_little_endian(self) -> None:
        """Test decoding uint16 field in little endian."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "little",
                "fields": [{"name": "value", "type": "uint16"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x34, 0x12])

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == 0x1234

    def test_decode_int16(self) -> None:
        """Test decoding int16 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "int16"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xFF, 0xFE])  # -2 in big endian

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == -2

    def test_decode_uint32(self) -> None:
        """Test decoding uint32 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "uint32"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x12, 0x34, 0x56, 0x78])

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == 0x12345678

    def test_decode_int32(self) -> None:
        """Test decoding int32 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "int32"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xFF, 0xFF, 0xFF, 0xFF])  # -1

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == -1

    def test_decode_uint64(self) -> None:
        """Test decoding uint64 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "uint64"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0])

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == 0x123456789ABCDEF0

    def test_decode_int64(self) -> None:
        """Test decoding int64 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "little",
                "fields": [{"name": "value", "type": "int64"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])  # -2

        result = decoder.decode(data)

        assert result.valid
        assert result["value"] == -2


# =============================================================================
# ProtocolDecoder Tests - Float Types
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderFloatTypes:
    """Test ProtocolDecoder float type decoding."""

    def test_decode_float32_big_endian(self) -> None:
        """Test decoding float32 field in big endian."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "float32"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = struct.pack(">f", 3.14)

        result = decoder.decode(data)

        assert result.valid
        assert abs(result["value"] - 3.14) < 0.001

    def test_decode_float32_little_endian(self) -> None:
        """Test decoding float32 field in little endian."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "little",
                "fields": [{"name": "value", "type": "float32"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = struct.pack("<f", 2.71828)

        result = decoder.decode(data)

        assert result.valid
        assert abs(result["value"] - 2.71828) < 0.001

    def test_decode_float64(self) -> None:
        """Test decoding float64 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "float64"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = struct.pack(">d", 1.41421356)

        result = decoder.decode(data)

        assert result.valid
        assert abs(result["value"] - 1.41421356) < 0.00001


# =============================================================================
# ProtocolDecoder Tests - Bytes and String Types
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderBytesAndString:
    """Test ProtocolDecoder bytes and string type decoding."""

    def test_decode_bytes_fixed_size(self) -> None:
        """Test decoding bytes field with fixed size."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "data", "type": "bytes", "size": 5}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = b"hello world"

        result = decoder.decode(data)

        assert result.valid
        assert result["data"] == b"hello"

    def test_decode_bytes_variable_size(self) -> None:
        """Test decoding bytes field with variable size from length field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "length", "type": "uint8"},
                    {"name": "data", "type": "bytes", "size": "length"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([5]) + b"hello"

        result = decoder.decode(data)

        assert result.valid
        assert result["length"] == 5
        assert result["data"] == b"hello"

    def test_decode_bytes_remaining_size(self) -> None:
        """Test decoding bytes field with 'remaining' size."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "header", "type": "uint8"},
                    {"name": "data", "type": "bytes", "size": "remaining"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01]) + b"all remaining data"

        result = decoder.decode(data)

        assert result.valid
        assert result["data"] == b"all remaining data"

    def test_decode_bytes_size_exceeds_data(self) -> None:
        """Test decoding bytes when size exceeds available data."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "data", "type": "bytes", "size": 100}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = b"short"

        result = decoder.decode(data)

        # Should use remaining data
        assert result["data"] == b"short"

    def test_decode_string_fixed_size(self) -> None:
        """Test decoding string field with fixed size."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "text", "type": "string", "size": 10}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = b"hello\x00\x00\x00\x00\x00"

        result = decoder.decode(data)

        assert result.valid
        assert result["text"] == "hello"

    def test_decode_string_with_null_terminator(self) -> None:
        """Test that strings are stripped of null terminators."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "text", "type": "string", "size": 8}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = b"test\x00\x00\x00\x00"

        result = decoder.decode(data)

        assert result["text"] == "test"

    def test_decode_string_latin1_fallback(self) -> None:
        """Test string decoding falls back to latin-1 for non-UTF8."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "text", "type": "string", "size": 4}],
            }
        )
        decoder = ProtocolDecoder(proto)
        # Invalid UTF-8 sequence
        data = bytes([0xC0, 0xC1, 0xFE, 0xFF])

        result = decoder.decode(data)

        assert result.valid
        # Should decode using latin-1


# =============================================================================
# ProtocolDecoder Tests - Bitfield Type
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderBitfield:
    """Test ProtocolDecoder bitfield type decoding."""

    def test_decode_bitfield_1_byte(self) -> None:
        """Test decoding 1-byte bitfield."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "flags", "type": "bitfield", "size": 1}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0b10101010])

        result = decoder.decode(data)

        assert result.valid
        assert result["flags"] == 0b10101010

    def test_decode_bitfield_2_bytes(self) -> None:
        """Test decoding 2-byte bitfield."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "flags", "type": "bitfield", "size": 2}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x12, 0x34])

        result = decoder.decode(data)

        assert result.valid
        assert result["flags"] == 0x1234

    def test_decode_bitfield_4_bytes(self) -> None:
        """Test decoding 4-byte bitfield."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "little",
                "fields": [{"name": "flags", "type": "bitfield", "size": 4}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x78, 0x56, 0x34, 0x12])

        result = decoder.decode(data)

        assert result.valid
        assert result["flags"] == 0x12345678

    def test_decode_bitfield_default_size(self) -> None:
        """Test decoding bitfield with default size (1 byte)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "flags", "type": "bitfield"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xFF])

        result = decoder.decode(data)

        assert result.valid
        assert result["flags"] == 0xFF


# =============================================================================
# ProtocolDecoder Tests - Array Type
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderArray:
    """Test ProtocolDecoder array type decoding."""

    def test_decode_array_fixed_count(self) -> None:
        """Test decoding array with fixed count."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {
                        "name": "values",
                        "type": "array",
                        "count": 3,
                        "element": {"type": "uint16"},
                    },
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x00, 0x01, 0x00, 0x02, 0x00, 0x03])

        result = decoder.decode(data)

        assert result.valid
        assert result["values"] == [1, 2, 3]

    def test_decode_array_count_from_field(self) -> None:
        """Test decoding array with count from another field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "count", "type": "uint8"},
                    {
                        "name": "items",
                        "type": "array",
                        "count_field": "count",
                        "element": {"type": "uint8"},
                    },
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([4, 10, 20, 30, 40])

        result = decoder.decode(data)

        assert result.valid
        assert result["count"] == 4
        assert result["items"] == [10, 20, 30, 40]

    def test_decode_array_of_structs(self) -> None:
        """Test decoding array of struct elements."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {
                        "name": "records",
                        "type": "array",
                        "count": 2,
                        "element": {
                            "type": "struct",
                            "fields": [
                                {"name": "id", "type": "uint8"},
                                {"name": "value", "type": "uint8"},
                            ],
                        },
                    },
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([1, 10, 2, 20])

        result = decoder.decode(data)

        assert result.valid
        assert len(result["records"]) == 2
        assert result["records"][0]["id"] == 1
        assert result["records"][0]["value"] == 10
        assert result["records"][1]["id"] == 2
        assert result["records"][1]["value"] == 20

    def test_decode_array_little_endian_elements(self) -> None:
        """Test decoding array with little endian elements."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "little",
                "fields": [
                    {
                        "name": "values",
                        "type": "array",
                        "count": 2,
                        "element": {"type": "uint16"},
                    },
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x34, 0x12, 0x78, 0x56])

        result = decoder.decode(data)

        assert result.valid
        assert result["values"] == [0x1234, 0x5678]


# =============================================================================
# ProtocolDecoder Tests - Struct Type
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderStruct:
    """Test ProtocolDecoder struct type decoding."""

    def test_decode_simple_struct(self) -> None:
        """Test decoding simple nested struct."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {
                        "name": "header",
                        "type": "struct",
                        "fields": [
                            {"name": "version", "type": "uint8"},
                            {"name": "type", "type": "uint8"},
                        ],
                    },
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01, 0x02])

        result = decoder.decode(data)

        assert result.valid
        assert result["header"]["version"] == 1
        assert result["header"]["type"] == 2

    def test_decode_struct_with_condition(self) -> None:
        """Test decoding struct with conditional nested field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {
                        "name": "record",
                        "type": "struct",
                        "fields": [
                            {"name": "type", "type": "uint8"},
                            {"name": "extra", "type": "uint8", "condition": "type == 1"},
                        ],
                    },
                ],
            }
        )
        decoder = ProtocolDecoder(proto)

        # With type=1, extra should be decoded
        data1 = bytes([0x01, 0x42])
        result1 = decoder.decode(data1)
        assert result1["record"]["type"] == 1
        assert result1["record"]["extra"] == 0x42

        # With type=2, extra should be skipped
        data2 = bytes([0x02, 0x42])
        result2 = decoder.decode(data2)
        assert result2["record"]["type"] == 2
        assert "extra" not in result2["record"]


# =============================================================================
# ProtocolDecoder Tests - Conditional Fields
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderConditional:
    """Test ProtocolDecoder conditional field handling."""

    def test_condition_equal(self) -> None:
        """Test condition with equality check."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "type", "type": "uint8"},
                    {"name": "data_a", "type": "uint16", "condition": "type == 1"},
                    {"name": "data_b", "type": "uint32", "condition": "type == 2"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)

        # Type 1 - should decode data_a
        data1 = bytes([0x01, 0x12, 0x34])
        result1 = decoder.decode(data1)
        assert "data_a" in result1
        assert "data_b" not in result1
        assert result1["data_a"] == 0x1234

        # Type 2 - should decode data_b
        data2 = bytes([0x02, 0x12, 0x34, 0x56, 0x78])
        result2 = decoder.decode(data2)
        assert "data_a" not in result2
        assert "data_b" in result2
        assert result2["data_b"] == 0x12345678

    def test_condition_not_equal(self) -> None:
        """Test condition with not-equal check."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "flag", "type": "uint8"},
                    {"name": "optional", "type": "uint8", "condition": "flag != 0"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)

        # Flag 0 - optional should be skipped
        data0 = bytes([0x00, 0x42])
        result0 = decoder.decode(data0)
        assert "optional" not in result0

        # Flag 1 - optional should be decoded
        data1 = bytes([0x01, 0x42])
        result1 = decoder.decode(data1)
        assert result1["optional"] == 0x42

    def test_condition_greater_than(self) -> None:
        """Test condition with greater-than check."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "version", "type": "uint8"},
                    {"name": "extended", "type": "uint8", "condition": "version > 1"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)

        # Version 1 - extended should be skipped
        data1 = bytes([0x01, 0x42])
        result1 = decoder.decode(data1)
        assert "extended" not in result1

        # Version 2 - extended should be decoded
        data2 = bytes([0x02, 0x42])
        result2 = decoder.decode(data2)
        assert result2["extended"] == 0x42

    def test_condition_invalid_expression(self) -> None:
        """Test condition with invalid expression returns False."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "value", "type": "uint8"},
                    {"name": "optional", "type": "uint8", "condition": "invalid_syntax("},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01, 0x42])

        result = decoder.decode(data)

        # Invalid condition should be skipped
        assert "optional" not in result


# =============================================================================
# ProtocolDecoder Tests - Validation
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderValidation:
    """Test ProtocolDecoder validation functionality."""

    def test_validation_min_pass(self) -> None:
        """Test field validation with min constraint (pass)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "value", "type": "uint8", "validation": {"min": 0}},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x10])

        result = decoder.decode(data)

        assert result.valid
        assert len(result.errors) == 0

    def test_validation_min_fail(self) -> None:
        """Test field validation with min constraint (fail)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "value", "type": "uint8", "validation": {"min": 100}},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x10])  # 16 < 100

        result = decoder.decode(data)

        assert not result.valid
        assert len(result.errors) == 1
        assert "below minimum" in result.errors[0]

    def test_validation_max_pass(self) -> None:
        """Test field validation with max constraint (pass)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "value", "type": "uint8", "validation": {"max": 255}},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xFF])

        result = decoder.decode(data)

        assert result.valid

    def test_validation_max_fail(self) -> None:
        """Test field validation with max constraint (fail)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "value", "type": "uint8", "validation": {"max": 10}},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x20])  # 32 > 10

        result = decoder.decode(data)

        assert not result.valid
        assert len(result.errors) == 1
        assert "above maximum" in result.errors[0]

    def test_validation_value_pass(self) -> None:
        """Test field validation with expected value (pass)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "sync", "type": "uint8", "validation": {"value": 0xAA}},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xAA])

        result = decoder.decode(data)

        assert result.valid

    def test_validation_value_fail(self) -> None:
        """Test field validation with expected value (fail)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "sync", "type": "uint8", "validation": {"value": 0xAA}},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x55])

        result = decoder.decode(data)

        assert not result.valid
        assert "Expected" in result.errors[0]

    def test_validation_min_max_combined(self) -> None:
        """Test field validation with both min and max constraints."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "value", "type": "uint8", "validation": {"min": 10, "max": 100}},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)

        # Within range
        result_ok = decoder.decode(bytes([50]))
        assert result_ok.valid

        # Below min
        result_low = decoder.decode(bytes([5]))
        assert not result_low.valid

        # Above max
        result_high = decoder.decode(bytes([150]))
        assert not result_high.valid


# =============================================================================
# ProtocolDecoder Tests - Stream Decoding
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderStream:
    """Test ProtocolDecoder stream decoding functionality."""

    def test_decode_stream_no_sync(self) -> None:
        """Test decoding stream without sync pattern."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "a", "type": "uint8"},
                    {"name": "b", "type": "uint8"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([1, 2, 3, 4])

        messages = decoder.decode_stream(data)

        # Without sync pattern, stream decoder decodes multiple messages
        assert len(messages) == 2
        assert messages[0]["a"] == 1
        assert messages[0]["b"] == 2
        assert messages[1]["a"] == 3
        assert messages[1]["b"] == 4

    def test_decode_stream_with_sync(self) -> None:
        """Test decoding stream with sync pattern."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "framing": {"sync_pattern": "0xAA55"},
                "fields": [
                    {"name": "sync", "type": "uint16"},
                    {"name": "data", "type": "uint8"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        # Two messages with sync pattern
        data = bytes([0xAA, 0x55, 0x01, 0xAA, 0x55, 0x02])

        messages = decoder.decode_stream(data)

        assert len(messages) == 2
        assert messages[0]["data"] == 0x01
        assert messages[1]["data"] == 0x02

    def test_decode_stream_garbage_before_sync(self) -> None:
        """Test decoding stream with garbage data before sync pattern."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "framing": {"sync_pattern": "0xAA55"},
                "fields": [
                    {"name": "sync", "type": "uint16"},
                    {"name": "data", "type": "uint8"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        # Garbage, then valid message
        data = bytes([0xFF, 0xFF, 0xAA, 0x55, 0x42])

        messages = decoder.decode_stream(data)

        assert len(messages) == 1
        assert messages[0]["data"] == 0x42

    def test_decode_stream_empty_data(self) -> None:
        """Test decoding empty stream."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "uint8"}],
            }
        )
        decoder = ProtocolDecoder(proto)

        messages = decoder.decode_stream(b"")

        assert len(messages) == 0


# =============================================================================
# ProtocolDecoder Tests - Find Sync
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderFindSync:
    """Test ProtocolDecoder find_sync functionality."""

    def test_find_sync_no_pattern_configured(self) -> None:
        """Test find_sync when no sync pattern is configured."""
        proto = ProtocolDefinition.from_dict({"name": "Test", "fields": []})
        decoder = ProtocolDecoder(proto)

        offset = decoder.find_sync(b"data", 0)

        assert offset == 0  # Returns start when no pattern

    def test_find_sync_hex_pattern(self) -> None:
        """Test find_sync with hex string pattern."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "framing": {"sync_pattern": "0xAA55"},
                "fields": [],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x00, 0x00, 0xAA, 0x55, 0x00])

        offset = decoder.find_sync(data, 0)

        assert offset == 2

    def test_find_sync_string_pattern(self) -> None:
        """Test find_sync with string pattern."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "framing": {"sync_pattern": "SYNC"},
                "fields": [],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = b"garbageSYNCdata"

        offset = decoder.find_sync(data, 0)

        assert offset == 7

    def test_find_sync_bytes_pattern(self) -> None:
        """Test find_sync with bytes pattern."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "framing": {"sync_pattern": [0x12, 0x34]},
                "fields": [],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x00, 0x12, 0x34, 0x00])

        offset = decoder.find_sync(data, 0)

        assert offset == 1

    def test_find_sync_with_start_offset(self) -> None:
        """Test find_sync starting from non-zero offset."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "framing": {"sync_pattern": "0xAA"},
                "fields": [],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0xAA, 0x00, 0xAA, 0x00])

        # Start searching from offset 1
        offset = decoder.find_sync(data, 1)

        assert offset == 2

    def test_find_sync_pattern_not_found(self) -> None:
        """Test find_sync when pattern is not in data."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "framing": {"sync_pattern": "0xDEAD"},
                "fields": [],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x00, 0x00, 0x00, 0x00])

        offset = decoder.find_sync(data, 0)

        assert offset is None


# =============================================================================
# ProtocolDecoder Tests - Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolDecoderEdgeCases:
    """Test ProtocolDecoder edge cases and error handling."""

    def test_decode_empty_data(self) -> None:
        """Test decoding empty data."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "uint8"}],
            }
        )
        decoder = ProtocolDecoder(proto)

        result = decoder.decode(b"")

        assert not result.valid
        assert "Insufficient data" in result.errors[0]

    def test_decode_with_offset(self) -> None:
        """Test decoding with non-zero offset."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "uint8"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x00, 0x00, 0x42])

        result = decoder.decode(data, offset=2)

        assert result.valid
        assert result["value"] == 0x42

    def test_decode_insufficient_data_for_integer(self) -> None:
        """Test decoding when not enough data for integer type."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "uint32"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01, 0x02])  # Need 4 bytes, only have 2

        result = decoder.decode(data)

        assert not result.valid
        assert len(result.errors) > 0

    def test_decode_insufficient_data_for_float(self) -> None:
        """Test decoding when not enough data for float type."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "float32"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01, 0x02])  # Need 4 bytes

        result = decoder.decode(data)

        assert not result.valid

    def test_decode_unknown_field_type(self) -> None:
        """Test decoding with unknown field type."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "unknown_type"}],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01, 0x02, 0x03, 0x04])

        result = decoder.decode(data)

        assert not result.valid
        assert "Unknown field type" in result.errors[0]

    def test_decode_size_field_not_found(self) -> None:
        """Test decoding when size field reference not found."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "data", "type": "bytes", "size": "nonexistent_field"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01, 0x02, 0x03])

        result = decoder.decode(data)

        assert not result.valid
        assert "not found in context" in result.errors[0]

    def test_load_from_yaml(self) -> None:
        """Test ProtocolDecoder.load() class method."""
        config = {
            "name": "LoadTest",
            "fields": [{"name": "value", "type": "uint8"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            decoder = ProtocolDecoder.load(f.name)

            assert decoder.definition.name == "LoadTest"

        Path(f.name).unlink()

    def test_decode_message_size_tracking(self) -> None:
        """Test that decoded message size is tracked correctly."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "a", "type": "uint8"},
                    {"name": "b", "type": "uint16"},
                    {"name": "c", "type": "uint32"},
                ],
            }
        )
        decoder = ProtocolDecoder(proto)
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])

        result = decoder.decode(data)

        assert result.size == 7  # 1 + 2 + 4


# =============================================================================
# ProtocolEncoder Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestProtocolEncoder:
    """Test ProtocolEncoder class."""

    def test_encode_uint8(self) -> None:
        """Test encoding uint8 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "uint8"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": 0x42})

        assert result == bytes([0x42])

    def test_encode_int8(self) -> None:
        """Test encoding int8 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "int8"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": -1})

        assert result == bytes([0xFF])

    def test_encode_uint16_big_endian(self) -> None:
        """Test encoding uint16 field in big endian."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "uint16"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": 0x1234})

        assert result == bytes([0x12, 0x34])

    def test_encode_uint16_little_endian(self) -> None:
        """Test encoding uint16 field in little endian."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "little",
                "fields": [{"name": "value", "type": "uint16"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": 0x1234})

        assert result == bytes([0x34, 0x12])

    def test_encode_int16(self) -> None:
        """Test encoding int16 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "int16"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": -2})

        assert result == bytes([0xFF, 0xFE])

    def test_encode_uint32(self) -> None:
        """Test encoding uint32 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "uint32"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": 0x12345678})

        assert result == bytes([0x12, 0x34, 0x56, 0x78])

    def test_encode_int32(self) -> None:
        """Test encoding int32 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "int32"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": -1})

        assert result == bytes([0xFF, 0xFF, 0xFF, 0xFF])

    def test_encode_uint64(self) -> None:
        """Test encoding uint64 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "uint64"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": 0x123456789ABCDEF0})

        assert result == bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0])

    def test_encode_int64(self) -> None:
        """Test encoding int64 field (signed)."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "little",
                "fields": [{"name": "value", "type": "int64"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": -2})

        assert result == bytes([0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])

    def test_encode_float32(self) -> None:
        """Test encoding float32 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "float32"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": 3.14})

        expected = struct.pack(">f", 3.14)
        assert result == expected

    def test_encode_float64(self) -> None:
        """Test encoding float64 field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [{"name": "value", "type": "float64"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"value": 3.14159265358979})

        expected = struct.pack(">d", 3.14159265358979)
        assert result == expected

    def test_encode_bytes(self) -> None:
        """Test encoding bytes field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "data", "type": "bytes", "size": 5}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"data": b"hello"})

        assert result == b"hello"

    def test_encode_bytes_from_list(self) -> None:
        """Test encoding bytes field from list."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "data", "type": "bytes", "size": 3}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"data": [0x01, 0x02, 0x03]})

        assert result == bytes([0x01, 0x02, 0x03])

    def test_encode_bytes_from_tuple(self) -> None:
        """Test encoding bytes field from tuple."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "data", "type": "bytes", "size": 3}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"data": (0x01, 0x02, 0x03)})

        assert result == bytes([0x01, 0x02, 0x03])

    def test_encode_string(self) -> None:
        """Test encoding string field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "text", "type": "string", "size": 10}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"text": "hello"})

        assert result == b"hello"

    def test_encode_string_from_bytes(self) -> None:
        """Test encoding string field from bytes."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "text", "type": "string", "size": 5}],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"text": b"hello"})

        assert result == b"hello"

    def test_encode_array(self) -> None:
        """Test encoding array field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "endian": "big",
                "fields": [
                    {
                        "name": "values",
                        "type": "array",
                        "count": 3,
                        "element": {"type": "uint16"},
                    },
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"values": [1, 2, 3]})

        assert result == bytes([0x00, 0x01, 0x00, 0x02, 0x00, 0x03])

    def test_encode_array_of_structs(self) -> None:
        """Test encoding array of struct elements."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {
                        "name": "records",
                        "type": "array",
                        "count": 2,
                        "element": {
                            "type": "struct",
                            "fields": [
                                {"name": "id", "type": "uint8"},
                                {"name": "value", "type": "uint8"},
                            ],
                        },
                    },
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode(
            {
                "records": [
                    {"id": 1, "value": 10},
                    {"id": 2, "value": 20},
                ]
            }
        )

        assert result == bytes([1, 10, 2, 20])

    def test_encode_struct(self) -> None:
        """Test encoding struct field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {
                        "name": "header",
                        "type": "struct",
                        "fields": [
                            {"name": "version", "type": "uint8"},
                            {"name": "type", "type": "uint8"},
                        ],
                    },
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"header": {"version": 1, "type": 2}})

        assert result == bytes([1, 2])

    def test_encode_with_default_value(self) -> None:
        """Test encoding uses default value when field not provided."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "value", "type": "uint8", "default": 0x42},
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({})

        assert result == bytes([0x42])

    def test_encode_missing_required_field(self) -> None:
        """Test encoding raises error for missing required field."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "required_field", "type": "uint8"},
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        with pytest.raises(ValueError, match="Missing required field"):
            encoder.encode({})

    def test_encode_with_condition_true(self) -> None:
        """Test encoding conditional field when condition is true."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "type", "type": "uint8"},
                    {"name": "data", "type": "uint8", "condition": "type == 1"},
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"type": 1, "data": 0x42})

        assert result == bytes([0x01, 0x42])

    def test_encode_with_condition_false(self) -> None:
        """Test encoding conditional field when condition is false."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {"name": "type", "type": "uint8"},
                    {"name": "data", "type": "uint8", "condition": "type == 1"},
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"type": 2, "data": 0x42})

        assert result == bytes([0x02])  # data is skipped

    def test_encode_invalid_bytes_value(self) -> None:
        """Test encoding raises error for invalid bytes value."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "data", "type": "bytes", "size": 4}],
            }
        )
        encoder = ProtocolEncoder(proto)

        with pytest.raises(ValueError, match="Invalid bytes value"):
            encoder.encode({"data": 12345})  # Integer, not bytes

    def test_encode_unknown_field_type(self) -> None:
        """Test encoding raises error for unknown field type."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [{"name": "value", "type": "unknown_type"}],
            }
        )
        encoder = ProtocolEncoder(proto)

        with pytest.raises(ValueError, match="Unknown field type"):
            encoder.encode({"value": 42})

    def test_encode_struct_with_defaults(self) -> None:
        """Test encoding struct uses defaults for missing nested fields."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "Test",
                "fields": [
                    {
                        "name": "record",
                        "type": "struct",
                        "fields": [
                            {"name": "id", "type": "uint8"},
                            {"name": "value", "type": "uint8", "default": 0xFF},
                        ],
                    },
                ],
            }
        )
        encoder = ProtocolEncoder(proto)

        result = encoder.encode({"record": {"id": 1}})

        assert result == bytes([1, 0xFF])


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_load_protocol(self) -> None:
        """Test load_protocol convenience function."""
        config = {
            "name": "ConvenienceTest",
            "fields": [{"name": "value", "type": "uint8"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            proto = load_protocol(f.name)

            assert isinstance(proto, ProtocolDefinition)
            assert proto.name == "ConvenienceTest"

        Path(f.name).unlink()

    def test_load_protocol_with_path(self) -> None:
        """Test load_protocol with Path object."""
        config = {"name": "PathTest", "fields": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            path = Path(f.name)

            proto = load_protocol(path)

            assert proto.name == "PathTest"

        path.unlink()

    def test_decode_message_with_string_path(self) -> None:
        """Test decode_message with protocol path as string."""
        config = {
            "name": "DecodeTest",
            "fields": [{"name": "value", "type": "uint8"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()

            data = bytes([0x42])
            result = decode_message(data, f.name)

            assert result.valid
            assert result["value"] == 0x42

        Path(f.name).unlink()

    def test_decode_message_with_protocol_definition(self) -> None:
        """Test decode_message with ProtocolDefinition object."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "DirectDecode",
                "fields": [
                    {"name": "a", "type": "uint8"},
                    {"name": "b", "type": "uint8"},
                ],
            }
        )

        data = bytes([0x01, 0x02])
        result = decode_message(data, proto)

        assert result.valid
        assert result["a"] == 0x01
        assert result["b"] == 0x02


# =============================================================================
# Round-Trip Tests (Encode + Decode)
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestRoundTrip:
    """Test encode-decode round trips."""

    def test_roundtrip_simple_message(self) -> None:
        """Test round-trip of simple message."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "RoundTrip",
                "endian": "big",
                "fields": [
                    {"name": "sync", "type": "uint16"},
                    {"name": "length", "type": "uint8"},
                    {"name": "value", "type": "uint32"},
                ],
            }
        )

        original = {"sync": 0xAA55, "length": 4, "value": 0x12345678}

        encoder = ProtocolEncoder(proto)
        encoded = encoder.encode(original)

        decoder = ProtocolDecoder(proto)
        decoded = decoder.decode(encoded)

        assert decoded.valid
        assert decoded["sync"] == original["sync"]
        assert decoded["length"] == original["length"]
        assert decoded["value"] == original["value"]

    def test_roundtrip_with_array(self) -> None:
        """Test round-trip of message with array."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "ArrayRoundTrip",
                "fields": [
                    {"name": "count", "type": "uint8"},
                    {
                        "name": "items",
                        "type": "array",
                        "count_field": "count",
                        "element": {"type": "uint8"},
                    },
                ],
            }
        )

        original = {"count": 3, "items": [10, 20, 30]}

        encoder = ProtocolEncoder(proto)
        encoded = encoder.encode(original)

        decoder = ProtocolDecoder(proto)
        decoded = decoder.decode(encoded)

        assert decoded.valid
        assert decoded["count"] == 3
        assert decoded["items"] == [10, 20, 30]

    def test_roundtrip_with_struct(self) -> None:
        """Test round-trip of message with nested struct."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "StructRoundTrip",
                "fields": [
                    {
                        "name": "header",
                        "type": "struct",
                        "fields": [
                            {"name": "version", "type": "uint8"},
                            {"name": "flags", "type": "uint8"},
                        ],
                    },
                    {"name": "payload", "type": "uint16"},
                ],
            }
        )

        original = {
            "header": {"version": 2, "flags": 0xFF},
            "payload": 0x1234,
        }

        encoder = ProtocolEncoder(proto)
        encoded = encoder.encode(original)

        decoder = ProtocolDecoder(proto)
        decoded = decoder.decode(encoded)

        assert decoded.valid
        assert decoded["header"]["version"] == 2
        assert decoded["header"]["flags"] == 0xFF
        assert decoded["payload"] == 0x1234

    def test_roundtrip_mixed_endianness(self) -> None:
        """Test round-trip with mixed endianness."""
        proto = ProtocolDefinition.from_dict(
            {
                "name": "MixedEndian",
                "endian": "big",
                "fields": [
                    {"name": "big_value", "type": "uint16"},
                    {"name": "little_value", "type": "uint16", "endian": "little"},
                ],
            }
        )

        original = {"big_value": 0x1234, "little_value": 0x5678}

        encoder = ProtocolEncoder(proto)
        encoded = encoder.encode(original)

        decoder = ProtocolDecoder(proto)
        decoded = decoder.decode(encoded)

        assert decoded["big_value"] == 0x1234
        assert decoded["little_value"] == 0x5678
