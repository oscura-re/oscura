"""Tests for Wireshark dissector generation.

This module tests the generation of Wireshark Lua dissectors from Oscura
protocol definitions. It validates syntax, field mappings, and generated code.
"""

import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from oscura.export.wireshark import (
    WiresharkDissectorGenerator,
    check_luac_available,
    get_field_size,
    get_lua_reader_function,
    get_protofield_type,
    is_variable_length,
    validate_lua_syntax,
)
from oscura.inference.protocol_dsl import FieldDefinition, ProtocolDefinition


class TestTypeMapping:
    """Test type mapping functions."""

    def test_get_protofield_type_uint8(self) -> None:
        """Test uint8 mapping."""
        protofield, base = get_protofield_type("uint8")
        assert protofield == "ProtoField.uint8"
        assert base == "base.HEX"

    def test_get_protofield_type_uint16(self) -> None:
        """Test uint16 mapping."""
        protofield, base = get_protofield_type("uint16")
        assert protofield == "ProtoField.uint16"
        assert base == "base.HEX"

    def test_get_protofield_type_int32(self) -> None:
        """Test int32 mapping."""
        protofield, base = get_protofield_type("int32")
        assert protofield == "ProtoField.int32"
        assert base == "base.DEC"

    def test_get_protofield_type_string(self) -> None:
        """Test string mapping."""
        protofield, base = get_protofield_type("string")
        assert protofield == "ProtoField.string"
        assert base == "base.NONE"

    def test_get_protofield_type_bytes(self) -> None:
        """Test bytes mapping."""
        protofield, base = get_protofield_type("bytes")
        assert protofield == "ProtoField.bytes"
        assert base == "base.NONE"

    def test_get_protofield_type_with_custom_base(self) -> None:
        """Test custom display base."""
        protofield, base = get_protofield_type("uint16", "dec")
        assert protofield == "ProtoField.uint16"
        assert base == "base.DEC"

    def test_get_protofield_type_unknown(self) -> None:
        """Test unknown field type."""
        with pytest.raises(ValueError, match="Unknown field type"):
            get_protofield_type("unknown_type")

    def test_get_field_size_fixed(self) -> None:
        """Test fixed-size field sizes."""
        assert get_field_size("uint8") == 1
        assert get_field_size("uint16") == 2
        assert get_field_size("uint32") == 4
        assert get_field_size("int8") == 1
        assert get_field_size("float32") == 4
        assert get_field_size("float64") == 8

    def test_get_field_size_variable(self) -> None:
        """Test variable-size field sizes."""
        assert get_field_size("bytes") is None
        assert get_field_size("string") is None

    def test_is_variable_length(self) -> None:
        """Test variable length detection."""
        assert not is_variable_length("uint8")
        assert not is_variable_length("uint32")
        assert is_variable_length("bytes")
        assert is_variable_length("string")

    def test_get_lua_reader_function_big_endian(self) -> None:
        """Test Lua reader function for big endian."""
        assert get_lua_reader_function("uint8", "big") == "uint"
        assert get_lua_reader_function("uint16", "big") == "uint16"
        assert get_lua_reader_function("uint32", "big") == "uint32"

    def test_get_lua_reader_function_little_endian(self) -> None:
        """Test Lua reader function for little endian."""
        assert get_lua_reader_function("uint8", "little") == "uint"
        assert get_lua_reader_function("uint16", "little") == "le_uint16"
        assert get_lua_reader_function("uint32", "little") == "le_uint32"


class TestLuaValidator:
    """Test Lua syntax validation."""

    def test_validate_lua_syntax_valid(self) -> None:
        """Test validation of valid Lua code."""
        valid_code = """
        local x = 1
        local y = 2
        return x + y
        """
        is_valid, error = validate_lua_syntax(valid_code)
        assert is_valid
        assert error == "" or "skipped" in error.lower()

    def test_validate_lua_syntax_invalid(self) -> None:
        """Test validation of invalid Lua code."""
        if not check_luac_available():
            pytest.skip("luac not available")

        invalid_code = """
        local x = 1
        if x then
            print("test")
        -- Missing 'end'
        """
        is_valid, error = validate_lua_syntax(invalid_code)
        assert not is_valid
        assert error != ""

    def test_check_luac_available(self) -> None:
        """Test luac availability check."""
        # This should return True or False without errors
        result = check_luac_available()
        assert isinstance(result, bool)


class TestWiresharkDissectorGenerator:
    """Test Wireshark dissector generator."""

    def test_generate_simple_protocol(self) -> None:
        """Test generating dissector for simple protocol."""
        protocol = ProtocolDefinition(
            name="simple",
            description="Simple Test Protocol",
            version="1.0",
            fields=[
                FieldDefinition(name="type", field_type="uint8", size=1),
                FieldDefinition(name="length", field_type="uint16", size=2),
                FieldDefinition(name="data", field_type="bytes", size=4),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify basic structure
        assert "local simple_proto = Proto(" in lua_code
        assert '"simple"' in lua_code
        assert "Simple Test Protocol" in lua_code
        assert "f_type = ProtoField.uint8" in lua_code
        assert "f_length = ProtoField.uint16" in lua_code
        assert "f_data = ProtoField.bytes" in lua_code
        assert "function simple_proto.dissector(buffer, pinfo, tree)" in lua_code

    def test_generate_with_variable_length_field(self) -> None:
        """Test generating dissector with variable-length field."""
        protocol = ProtocolDefinition(
            name="varlen",
            description="Variable Length Protocol",
            fields=[
                FieldDefinition(name="length", field_type="uint16", size=2),
                FieldDefinition(
                    name="payload",
                    field_type="bytes",
                    size="length",  # References length field
                ),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify variable-length handling
        assert "Variable-length field: payload" in lua_code
        assert "payload_len" in lua_code

    def test_generate_with_tcp_registration(self) -> None:
        """Test generating dissector with TCP registration."""
        protocol = ProtocolDefinition(
            name="tcpproto",
            description="TCP Protocol",
            fields=[
                FieldDefinition(name="header", field_type="uint32", size=4),
            ],
            settings={"transport": "tcp", "port": 8000},
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify TCP registration
        assert "tcp_table = DissectorTable.get" in lua_code
        assert "tcp_table:add(8000" in lua_code

    def test_generate_with_udp_registration(self) -> None:
        """Test generating dissector with UDP registration."""
        protocol = ProtocolDefinition(
            name="udpproto",
            description="UDP Protocol",
            fields=[
                FieldDefinition(name="header", field_type="uint32", size=4),
            ],
            settings={"transport": "udp", "port": 5000},
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify UDP registration
        assert "udp_table = DissectorTable.get" in lua_code
        assert "udp_table:add(5000" in lua_code

    def test_generate_with_enum_field(self) -> None:
        """Test generating dissector with enum field."""
        protocol = ProtocolDefinition(
            name="enumproto",
            description="Protocol with Enum",
            fields=[
                FieldDefinition(
                    name="msg_type",
                    field_type="uint8",
                    size=1,
                    enum={0: "REQUEST", 1: "RESPONSE", 2: "ERROR"},
                ),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify enum value_string
        assert "[0] = " in lua_code
        assert "REQUEST" in lua_code
        assert "RESPONSE" in lua_code
        assert "ERROR" in lua_code

    def test_generate_with_conditional_field(self) -> None:
        """Test generating dissector with conditional field."""
        protocol = ProtocolDefinition(
            name="condproto",
            description="Protocol with Conditional Field",
            fields=[
                FieldDefinition(name="flags", field_type="uint8", size=1),
                FieldDefinition(
                    name="optional_data",
                    field_type="uint32",
                    size=4,
                    condition="flags == 1",
                ),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify conditional comment
        assert "Conditional field: optional_data" in lua_code
        assert "flags == 1" in lua_code

    def test_generate_with_little_endian(self) -> None:
        """Test generating dissector with little-endian fields."""
        protocol = ProtocolDefinition(
            name="leproto",
            description="Little Endian Protocol",
            endian="little",
            fields=[
                FieldDefinition(name="value", field_type="uint32", size=4, endian="little"),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify little endian is documented
        assert "little" in lua_code.lower()

    def test_generate_to_file(self) -> None:
        """Test generating dissector to file."""
        protocol = ProtocolDefinition(
            name="filetest",
            description="File Test Protocol",
            fields=[
                FieldDefinition(name="header", field_type="uint32", size=4),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "filetest.lua"
            generator = WiresharkDissectorGenerator(validate=False)
            generator.generate(protocol, output_path)

            # Verify file was created
            assert output_path.exists()

            # Verify content
            content = output_path.read_text()
            assert "filetest_proto" in content
            assert "File Test Protocol" in content

    def test_generate_empty_protocol_raises_error(self) -> None:
        """Test that empty protocol name raises error."""
        protocol = ProtocolDefinition(name="", description="Test")

        generator = WiresharkDissectorGenerator(validate=False)
        with pytest.raises(ValueError, match="Protocol name is required"):
            generator.generate_to_string(protocol)

    def test_generate_with_validation_valid(self) -> None:
        """Test generation with validation enabled for valid code."""
        if not check_luac_available():
            pytest.skip("luac not available")

        protocol = ProtocolDefinition(
            name="valid",
            description="Valid Protocol",
            fields=[
                FieldDefinition(name="data", field_type="uint8", size=1),
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "valid.lua"
            generator = WiresharkDissectorGenerator(validate=True)
            # Should not raise
            generator.generate(protocol, output_path)

    def test_min_length_calculation(self) -> None:
        """Test minimum length calculation."""
        protocol = ProtocolDefinition(
            name="minlen",
            description="Min Length Test",
            fields=[
                FieldDefinition(name="header", field_type="uint32", size=4),
                FieldDefinition(name="flags", field_type="uint8", size=1),
                FieldDefinition(name="data", field_type="bytes", size="remaining"),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Minimum length should be 5 (4 + 1, ignoring variable-length field)
        assert "if pktlen < 5 then" in lua_code

    def test_field_display_names(self) -> None:
        """Test field display name generation."""
        protocol = ProtocolDefinition(
            name="display",
            description="Display Name Test",
            fields=[
                FieldDefinition(
                    name="msg_type", field_type="uint8", size=1, description="Message Type"
                ),
                FieldDefinition(name="data_len", field_type="uint16", size=2),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Check display names
        assert "Message Type" in lua_code
        assert "Data Len" in lua_code

    def test_multiple_field_types(self) -> None:
        """Test protocol with multiple field types."""
        protocol = ProtocolDefinition(
            name="multitypes",
            description="Multiple Types Protocol",
            fields=[
                FieldDefinition(name="u8", field_type="uint8", size=1),
                FieldDefinition(name="u16", field_type="uint16", size=2),
                FieldDefinition(name="u32", field_type="uint32", size=4),
                FieldDefinition(name="i8", field_type="int8", size=1),
                FieldDefinition(name="i16", field_type="int16", size=2),
                FieldDefinition(name="i32", field_type="int32", size=4),
                FieldDefinition(name="f32", field_type="float32", size=4),
                FieldDefinition(name="str", field_type="string", size=10),
                FieldDefinition(name="data", field_type="bytes", size=8),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify all field types are present
        assert "ProtoField.uint8" in lua_code
        assert "ProtoField.uint16" in lua_code
        assert "ProtoField.uint32" in lua_code
        assert "ProtoField.int8" in lua_code
        assert "ProtoField.int16" in lua_code
        assert "ProtoField.int32" in lua_code
        assert "ProtoField.float" in lua_code
        assert "ProtoField.string" in lua_code
        assert "ProtoField.bytes" in lua_code


class TestRealWorldProtocols:
    """Test generation for real-world-like protocols."""

    def test_generate_uart_protocol(self) -> None:
        """Test generating dissector for UART-like protocol."""
        protocol = ProtocolDefinition(
            name="uart",
            description="UART Protocol",
            fields=[
                FieldDefinition(name="start_bit", field_type="uint8", size=1),
                FieldDefinition(name="data", field_type="bytes", size=8),
                FieldDefinition(name="parity", field_type="uint8", size=1),
                FieldDefinition(name="stop_bit", field_type="uint8", size=1),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        assert "uart_proto" in lua_code
        assert "UART Protocol" in lua_code
        assert "f_start_bit" in lua_code
        assert "f_data" in lua_code
        assert "f_parity" in lua_code
        assert "f_stop_bit" in lua_code

    def test_generate_modbus_like_protocol(self) -> None:
        """Test generating dissector for Modbus-like protocol."""
        protocol = ProtocolDefinition(
            name="modbus",
            description="Modbus Protocol",
            settings={"transport": "tcp", "port": 502},
            fields=[
                FieldDefinition(
                    name="function_code",
                    field_type="uint8",
                    size=1,
                    enum={
                        1: "Read Coils",
                        2: "Read Discrete Inputs",
                        3: "Read Holding Registers",
                        4: "Read Input Registers",
                    },
                ),
                FieldDefinition(name="address", field_type="uint16", size=2),
                FieldDefinition(name="count", field_type="uint16", size=2),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Verify Modbus characteristics
        assert "modbus_proto" in lua_code
        assert "tcp_table:add(502" in lua_code
        assert "Read Coils" in lua_code
        assert "Read Holding Registers" in lua_code


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_protocol_without_fields(self) -> None:
        """Test protocol with no fields."""
        protocol = ProtocolDefinition(name="empty", description="Empty Protocol", fields=[])

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Should generate valid code with empty fields
        assert "empty_proto" in lua_code
        assert ".fields = {" in lua_code

    def test_protocol_with_single_field(self) -> None:
        """Test protocol with single field."""
        protocol = ProtocolDefinition(
            name="single",
            description="Single Field Protocol",
            fields=[
                FieldDefinition(name="data", field_type="uint8", size=1),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        assert "single_proto" in lua_code
        assert "f_data" in lua_code

    def test_protocol_name_with_special_characters(self) -> None:
        """Test protocol name sanitization."""
        protocol = ProtocolDefinition(
            name="my-special_proto 2.0",
            description="Special Characters",
            fields=[
                FieldDefinition(name="data", field_type="uint8", size=1),
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Protocol variable should have special chars converted
        assert "my_special_proto" in lua_code

    def test_field_without_size(self) -> None:
        """Test field without explicit size uses default."""
        protocol = ProtocolDefinition(
            name="nosize",
            description="No Size Field",
            fields=[
                FieldDefinition(name="data", field_type="uint8"),  # No size specified
            ],
        )

        generator = WiresharkDissectorGenerator(validate=False)
        lua_code = generator.generate_to_string(protocol)

        # Should use default size for uint8 (1 byte)
        assert "f_data" in lua_code
