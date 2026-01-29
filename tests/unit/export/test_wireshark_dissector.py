"""Tests for Wireshark dissector generator.

This module tests the WiresharkDissectorGenerator class which generates
functional Wireshark Lua dissectors from ProtocolSpec objects.

Test Coverage:
    - Basic dissector generation
    - All field types (uint8, uint16, uint32, string, bytes)
    - Endianness handling (big/little)
    - CRC validation code generation
    - Test PCAP generation
    - Lua syntax validation
    - Protocol registration
    - Nested structures
    - Enum field handling
"""

from pathlib import Path

import pytest

from oscura.export.wireshark_dissector import DissectorConfig, WiresharkDissectorGenerator
from oscura.inference.crc_reverse import CRCParameters
from oscura.workflows.reverse_engineering import FieldSpec, ProtocolSpec


@pytest.fixture
def basic_spec() -> ProtocolSpec:
    """Create basic protocol spec for testing.

    Returns:
        ProtocolSpec with simple fields.
    """
    return ProtocolSpec(
        name="TestProtocol",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aa55",
        frame_length=10,
        fields=[
            FieldSpec(name="sync", offset=0, size=2, field_type="bytes"),
            FieldSpec(name="version", offset=2, size=1, field_type="uint8"),
            FieldSpec(name="length", offset=3, size=1, field_type="uint8"),
            FieldSpec(name="data", offset=4, size=4, field_type="bytes"),
            FieldSpec(name="checksum", offset=8, size=2, field_type="uint16"),
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.95,
    )


@pytest.fixture
def crc_spec() -> ProtocolSpec:
    """Create protocol spec with CRC validation.

    Returns:
        ProtocolSpec with CRC-16 checksum.
    """
    return ProtocolSpec(
        name="CRCProtocol",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="7e",
        frame_length=8,
        fields=[
            FieldSpec(name="sync", offset=0, size=1, field_type="bytes"),
            FieldSpec(name="data", offset=1, size=5, field_type="bytes"),
            FieldSpec(name="crc", offset=6, size=2, field_type="uint16"),
        ],
        checksum_type="crc16",
        checksum_position=-1,
        confidence=0.88,
    )


@pytest.fixture
def crc_params() -> CRCParameters:
    """Create CRC parameters for testing.

    Returns:
        CRCParameters for CRC-16-CCITT.
    """
    return CRCParameters(
        polynomial=0x1021,
        width=16,
        init=0xFFFF,
        xor_out=0x0000,
        reflect_in=False,
        reflect_out=False,
        confidence=0.95,
        test_pass_rate=1.0,
        algorithm_name="CRC-16-CCITT",
    )


@pytest.fixture
def all_field_types_spec() -> ProtocolSpec:
    """Create protocol spec with all supported field types.

    Returns:
        ProtocolSpec demonstrating all field types.
    """
    return ProtocolSpec(
        name="AllFieldTypes",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="ff",
        frame_length=20,
        fields=[
            FieldSpec(name="magic", offset=0, size=1, field_type="bytes"),
            FieldSpec(name="u8_field", offset=1, size=1, field_type="uint8"),
            FieldSpec(name="u16_field", offset=2, size=2, field_type="uint16"),
            FieldSpec(name="u32_field", offset=4, size=4, field_type="uint32"),
            FieldSpec(name="str_field", offset=8, size=8, field_type="string"),
            FieldSpec(name="bytes_field", offset=16, size=2, field_type="bytes"),
            FieldSpec(name="checksum", offset=18, size=2, field_type="checksum"),
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.92,
    )


def test_basic_dissector_generation(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test basic Lua dissector generation.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="TestProtocol", port=5000)
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "test.lua"
    lua_path, _ = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    # Verify file created
    assert lua_path.exists()
    assert lua_path.suffix == ".lua"

    # Read generated code
    lua_code = lua_path.read_text()

    # Verify key components
    assert "Proto(" in lua_code  # Protocol declaration
    assert "ProtoField" in lua_code  # Field declarations
    assert "function" in lua_code  # Dissector function
    assert "DissectorTable" in lua_code  # Registration
    assert "TestProtocol" in lua_code  # Protocol name


def test_dissector_with_crc(crc_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test dissector generation with CRC validation.

    Args:
        crc_spec: Protocol spec with CRC.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(
        protocol_name="CRCProtocol",
        port=None,
        include_crc_validation=True,
    )
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "crc.lua"
    lua_path, _ = generator.generate(
        crc_spec,
        sample_messages=[],
        output_path=output_path,
    )

    lua_code = lua_path.read_text()

    # Verify CRC validation function present
    assert "validate_crc16" in lua_code
    assert "0x1021" in lua_code  # CRC-16-CCITT polynomial
    assert "computed_crc" in lua_code
    assert "packet_crc" in lua_code


def test_all_field_types(all_field_types_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test dissector with all supported field types.

    Args:
        all_field_types_spec: Spec with all field types.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="AllFieldTypes")
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "all_types.lua"
    lua_path, _ = generator.generate(
        all_field_types_spec,
        sample_messages=[],
        output_path=output_path,
    )

    lua_code = lua_path.read_text()

    # Verify all field types represented
    assert "ProtoField.uint8" in lua_code
    assert "ProtoField.uint16" in lua_code
    assert "ProtoField.uint32" in lua_code
    assert "ProtoField.string" in lua_code
    assert "ProtoField.bytes" in lua_code

    # Verify all field names
    assert "u8_field" in lua_code
    assert "u16_field" in lua_code
    assert "u32_field" in lua_code
    assert "str_field" in lua_code
    assert "bytes_field" in lua_code


def test_endianness_handling(tmp_path: Path) -> None:
    """Test big and little endian field handling.

    Args:
        tmp_path: Temporary directory for output.
    """
    # Create spec with mixed endianness
    spec = ProtocolSpec(
        name="EndianTest",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=6,
        fields=[
            FieldSpec(name="big_endian", offset=0, size=2, field_type="uint16"),
            FieldSpec(name="little_endian", offset=2, size=2, field_type="uint16"),
            FieldSpec(name="mixed_u32", offset=4, size=4, field_type="uint32"),
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.9,
    )

    # Set endianness on fields (note: FieldSpec might not have endian attribute)
    # We'll need to add this in a way that works with the dataclass
    spec.fields[1].endian = "little"  # type: ignore[attr-defined]
    spec.fields[2].endian = "little"  # type: ignore[attr-defined]

    config = DissectorConfig(protocol_name="EndianTest")
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "endian.lua"
    lua_path, _ = generator.generate(spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify little-endian reader functions used
    assert "le_uint()" in lua_code


def test_pcap_generation(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test PCAP file generation with sample messages.

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(
        protocol_name="TestProtocol",
        port=5000,
        generate_test_pcap=True,
    )
    generator = WiresharkDissectorGenerator(config)

    sample_messages = [
        b"\xaa\x55\x01\x08testdata\x12\x34",
        b"\xaa\x55\x02\x08moredata\x56\x78",
        b"\xaa\x55\x03\x08lastdata\x9a\xbc",
    ]

    output_path = tmp_path / "test.lua"
    lua_path, pcap_path = generator.generate(
        basic_spec,
        sample_messages=sample_messages,
        output_path=output_path,
    )

    # Verify PCAP created
    assert pcap_path is not None
    assert pcap_path.exists()
    assert pcap_path.suffix == ".pcap"

    # Verify PCAP has content (should be larger than just header)
    pcap_data = pcap_path.read_bytes()
    assert len(pcap_data) > 24  # PCAP global header is 24 bytes

    # Verify PCAP magic number
    assert pcap_data[:4] == b"\xd4\xc3\xb2\xa1"  # Little-endian magic


def test_pcap_not_generated_when_disabled(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test that PCAP is not generated when disabled in config.

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(
        protocol_name="TestProtocol",
        generate_test_pcap=False,
    )
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "test.lua"
    lua_path, pcap_path = generator.generate(
        basic_spec,
        sample_messages=[b"\xaa\x55\x01\x02\x03\x04\x05\x06\x07\x08"],
        output_path=output_path,
    )

    # Verify PCAP not created
    assert pcap_path is None


def test_lua_syntax_validation(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test Lua syntax validation (if luac available).

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="TestProtocol")
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "test.lua"

    # Should not raise exception (validation passes or luac not available)
    lua_path, _ = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    # Verify Lua code looks syntactically correct
    lua_code = lua_path.read_text()
    assert lua_code.count("end") >= lua_code.count("function")  # Balanced


def test_protocol_registration_with_port(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test protocol registration on UDP/TCP port.

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="TestProtocol", port=12345)
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "test.lua"
    lua_path, _ = generator.generate(basic_spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify registration code
    assert "udp.port" in lua_code
    assert "tcp.port" in lua_code
    assert "12345" in lua_code
    assert ":add(" in lua_code


def test_protocol_registration_without_port(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test protocol registration without port (manual decode).

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="TestProtocol", port=None)
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "test.lua"
    lua_path, _ = generator.generate(basic_spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify manual registration instructions
    assert "Decode As" in lua_code
    assert "manual" in lua_code.lower()


def test_crc_function_from_params(
    crc_spec: ProtocolSpec, crc_params: CRCParameters, tmp_path: Path
) -> None:
    """Test CRC function generation with default parameters.

    Args:
        crc_spec: Protocol spec with CRC.
        crc_params: CRC parameters (not used in current implementation).
        tmp_path: Temporary directory for output.

    Note:
        ProtocolSpec does not have crc_info field, so we test default CRC
        generation for crc16 type. To support custom CRC parameters, the
        ProtocolSpec dataclass would need to be extended.
    """
    config = DissectorConfig(
        protocol_name="CRCProtocol",
        include_crc_validation=True,
    )
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "crc_params.lua"
    lua_path, _ = generator.generate(crc_spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify default CRC-16-CCITT parameters in code
    assert "0x1021" in lua_code  # Polynomial
    assert "0xFFFF" in lua_code  # Init value
    assert "validate_crc16" in lua_code
    assert "CRC-16" in lua_code


def test_validation_invalid_spec(tmp_path: Path) -> None:
    """Test that invalid specs raise ValueError.

    Args:
        tmp_path: Temporary directory for output.
    """
    # Spec with no name
    invalid_spec = ProtocolSpec(
        name="",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=5,
        fields=[],
        checksum_type=None,
        checksum_position=None,
        confidence=0.5,
    )

    config = DissectorConfig(protocol_name="Test")
    generator = WiresharkDissectorGenerator(config)

    with pytest.raises(ValueError, match="Protocol name is required"):
        generator.generate(invalid_spec, sample_messages=[], output_path=tmp_path / "test.lua")


def test_validation_no_fields(tmp_path: Path) -> None:
    """Test that spec with no fields raises ValueError.

    Args:
        tmp_path: Temporary directory for output.
    """
    invalid_spec = ProtocolSpec(
        name="NoFields",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=5,
        fields=[],
        checksum_type=None,
        checksum_position=None,
        confidence=0.5,
    )

    config = DissectorConfig(protocol_name="NoFields")
    generator = WiresharkDissectorGenerator(config)

    with pytest.raises(ValueError, match="at least one field"):
        generator.generate(invalid_spec, sample_messages=[], output_path=tmp_path / "test.lua")


def test_validation_unsupported_field_type(tmp_path: Path) -> None:
    """Test that unsupported field types raise ValueError.

    Args:
        tmp_path: Temporary directory for output.
    """
    invalid_spec = ProtocolSpec(
        name="BadField",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=5,
        fields=[
            FieldSpec(name="bad", offset=0, size=1, field_type="float64"),  # Not supported
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.5,
    )

    config = DissectorConfig(protocol_name="BadField")
    generator = WiresharkDissectorGenerator(config)

    with pytest.raises(ValueError, match="Unsupported field type"):
        generator.generate(invalid_spec, sample_messages=[], output_path=tmp_path / "test.lua")


def test_header_generation(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test header comment generation with metadata.

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="TestProtocol", wireshark_version="3.6+")
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "test.lua"
    lua_path, _ = generator.generate(basic_spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify header metadata
    assert "TestProtocol" in lua_code
    assert "115200 bps" in lua_code
    assert "8N1" in lua_code
    assert "aa55" in lua_code
    assert "0.95" in lua_code  # Confidence
    assert "3.6+" in lua_code  # Wireshark version
    assert "Installation:" in lua_code


def test_crc8_generation(tmp_path: Path) -> None:
    """Test CRC-8 validation function generation.

    Args:
        tmp_path: Temporary directory for output.
    """
    spec = ProtocolSpec(
        name="CRC8Protocol",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=5,
        fields=[
            FieldSpec(name="data", offset=0, size=4, field_type="bytes"),
            FieldSpec(name="crc", offset=4, size=1, field_type="uint8"),
        ],
        checksum_type="crc8",
        checksum_position=-1,
        confidence=0.9,
    )

    config = DissectorConfig(protocol_name="CRC8", include_crc_validation=True)
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "crc8.lua"
    lua_path, _ = generator.generate(spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify CRC-8 function
    assert "validate_crc8" in lua_code
    assert "0x07" in lua_code  # CRC-8 polynomial


def test_crc32_generation(tmp_path: Path) -> None:
    """Test CRC-32 validation function generation.

    Args:
        tmp_path: Temporary directory for output.
    """
    spec = ProtocolSpec(
        name="CRC32Protocol",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=10,
        fields=[
            FieldSpec(name="data", offset=0, size=6, field_type="bytes"),
            FieldSpec(name="crc", offset=6, size=4, field_type="uint32"),
        ],
        checksum_type="crc32",
        checksum_position=-1,
        confidence=0.9,
    )

    config = DissectorConfig(protocol_name="CRC32", include_crc_validation=True)
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "crc32.lua"
    lua_path, _ = generator.generate(spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify CRC-32 function
    assert "validate_crc32" in lua_code
    assert "0x04C11DB7" in lua_code  # CRC-32 polynomial


def test_multiple_messages_in_pcap(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test PCAP generation with multiple messages.

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="Test", port=5000, generate_test_pcap=True)
    generator = WiresharkDissectorGenerator(config)

    # Create 10 sample messages
    sample_messages = [b"\xaa\x55" + bytes([i, 8]) + b"data" + bytes([i, i]) for i in range(10)]

    output_path = tmp_path / "multi.lua"
    _, pcap_path = generator.generate(
        basic_spec,
        sample_messages=sample_messages,
        output_path=output_path,
    )

    assert pcap_path is not None
    pcap_data = pcap_path.read_bytes()

    # Should have global header + 10 packet headers + packet data
    # Rough size check (exact size depends on packet structure)
    assert len(pcap_data) > 24 + (10 * 16)  # Global + packet headers minimum


def test_enum_field_handling(tmp_path: Path) -> None:
    """Test enum field value_string table generation.

    Args:
        tmp_path: Temporary directory for output.
    """
    spec = ProtocolSpec(
        name="EnumProtocol",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=4,
        fields=[
            FieldSpec(
                name="type",
                offset=0,
                size=1,
                field_type="uint8",
            ),
            FieldSpec(name="data", offset=1, size=3, field_type="bytes"),
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.9,
    )

    # Add enum to type field
    spec.fields[0].enum = {  # type: ignore[attr-defined]
        0: "TYPE_A",
        1: "TYPE_B",
        2: "TYPE_C",
    }

    config = DissectorConfig(protocol_name="EnumTest")
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "enum.lua"
    lua_path, _ = generator.generate(spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify value_string table generated
    assert "vs_type" in lua_code
    assert "TYPE_A" in lua_code
    assert "TYPE_B" in lua_code
    assert "TYPE_C" in lua_code
    assert "[0]" in lua_code
    assert "[1]" in lua_code


def test_minimum_packet_length_check(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test minimum packet length validation in dissector.

    Args:
        basic_spec: Basic protocol spec.
        tmp_path: Temporary directory for output.
    """
    config = DissectorConfig(protocol_name="Test")
    generator = WiresharkDissectorGenerator(config)

    output_path = tmp_path / "test.lua"
    lua_path, _ = generator.generate(basic_spec, sample_messages=[], output_path=output_path)

    lua_code = lua_path.read_text()

    # Verify minimum length check
    assert "buffer:len()" in lua_code
    assert "return 0" in lua_code  # Return if too short
    assert f"< {basic_spec.frame_length}" in lua_code
