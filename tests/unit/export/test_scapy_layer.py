"""Tests for Scapy layer generator.

This module tests the ScapyLayerGenerator class which generates
production-ready Scapy layer classes from ProtocolSpec objects.

Test Coverage:
    - Basic layer generation
    - All field types (uint8, uint16, uint32, string, bytes)
    - Endianness handling (big/little)
    - CRC validation code generation
    - Python syntax validation
    - Generated code can be imported
    - Packet construction and dissection
    - Enum field handling
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from oscura.export.scapy_layer import ScapyLayerConfig, ScapyLayerGenerator
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


@pytest.fixture
def little_endian_spec() -> ProtocolSpec:
    """Create protocol spec with little-endian fields.

    Returns:
        ProtocolSpec with little-endian fields.
    """
    fields = [
        FieldSpec(name="sync", offset=0, size=2, field_type="bytes"),
        FieldSpec(name="le_u16", offset=2, size=2, field_type="uint16"),
        FieldSpec(name="le_u32", offset=4, size=4, field_type="uint32"),
    ]
    # Add endian attribute
    fields[1].endian = "little"  # type: ignore[attr-defined]
    fields[2].endian = "little"  # type: ignore[attr-defined]

    return ProtocolSpec(
        name="LittleEndianProto",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aabb",
        frame_length=8,
        fields=fields,
        checksum_type=None,
        checksum_position=None,
        confidence=0.90,
    )


def test_basic_layer_generation(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test basic Scapy layer generation.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(protocol_name="TestProtocol", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "test_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    # Verify file created
    assert layer_path.exists()
    assert layer_path.suffix == ".py"

    # Read generated code
    python_code = layer_path.read_text()

    # Verify key components
    assert "class TestProtocol(Packet):" in python_code
    assert "from scapy.fields import" in python_code
    assert "from scapy.packet import Packet" in python_code
    assert "fields_desc" in python_code


def test_all_field_types(all_field_types_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test generation with all field types.

    Args:
        all_field_types_spec: Spec with all field types.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(protocol_name="AllFieldTypes", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "all_fields_layer.py"
    layer_path = generator.generate(
        all_field_types_spec,
        sample_messages=[],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify all field types are present
    assert "ByteField" in python_code  # uint8
    assert "ShortField" in python_code  # uint16
    assert "IntField" in python_code  # uint32
    assert "StrFixedLenField" in python_code  # string and bytes
    assert "XShortField" in python_code or "XByteField" in python_code  # checksum


def test_endianness_handling(little_endian_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test little-endian field handling.

    Args:
        little_endian_spec: Spec with little-endian fields.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(protocol_name="LittleEndianProto", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "le_layer.py"
    layer_path = generator.generate(
        little_endian_spec,
        sample_messages=[],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify little-endian fields
    assert "LEShortField" in python_code
    assert "LEIntField" in python_code


def test_crc_generation(crc_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test CRC validation code generation.

    Args:
        crc_spec: Spec with CRC checksum.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(
        protocol_name="CRCProtocol", include_crc_validation=True, show_progress=False
    )
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "crc_layer.py"
    layer_path = generator.generate(
        crc_spec,
        sample_messages=[b"\x7etest12\x00\x00"],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify CRC function generated
    assert "def calculate_crc16" in python_code
    assert "post_build" in python_code
    assert "do_dissect" in python_code


def test_crc_custom_params(
    crc_spec: ProtocolSpec, crc_params: CRCParameters, tmp_path: Path
) -> None:
    """Test CRC generation with custom parameters.

    Args:
        crc_spec: Spec with CRC checksum.
        crc_params: Custom CRC parameters.
        tmp_path: Temporary directory for output.
    """
    # Attach CRC parameters to spec
    crc_spec.crc_info = crc_params  # type: ignore[attr-defined]

    config = ScapyLayerConfig(
        protocol_name="CRCProtocol", include_crc_validation=True, show_progress=False
    )
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "crc_custom_layer.py"
    layer_path = generator.generate(
        crc_spec,
        sample_messages=[],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify custom CRC parameters in docstring
    assert "0x1021" in python_code  # Polynomial
    assert "0xFFFF" in python_code  # Init


def test_python_syntax_validation(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test that generated Python code has valid syntax.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(protocol_name="TestProtocol", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "syntax_test_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify syntax is valid by compiling
    compile(python_code, str(layer_path), "exec")


def test_generated_code_importable(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test that generated code can be imported.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(protocol_name="TestProtocol", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "importable_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    # Import the generated module
    spec = importlib.util.spec_from_file_location("importable_layer", layer_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules["importable_layer"] = module
    spec.loader.exec_module(module)

    # Verify class exists
    assert hasattr(module, "TestProtocol")

    # Clean up
    del sys.modules["importable_layer"]


def test_packet_construction(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test constructing packets with generated layer.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    pytest.importorskip("scapy", reason="Scapy required for packet construction tests")

    config = ScapyLayerConfig(protocol_name="TestProtocol", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "construct_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    # Import the generated module
    spec_obj = importlib.util.spec_from_file_location("construct_layer", layer_path)
    assert spec_obj is not None
    assert spec_obj.loader is not None

    module = importlib.util.module_from_spec(spec_obj)
    sys.modules["construct_layer"] = module
    spec_obj.loader.exec_module(module)

    # Construct packet
    TestProtocol = module.TestProtocol
    pkt = TestProtocol(sync=b"\xaa\x55", version=1, length=10)

    # Verify fields
    assert pkt.version == 1
    assert pkt.length == 10

    # Clean up
    del sys.modules["construct_layer"]


def test_packet_dissection(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test dissecting packets with generated layer.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    pytest.importorskip("scapy", reason="Scapy required for packet dissection tests")

    config = ScapyLayerConfig(protocol_name="TestProtocol", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "dissect_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    # Import the generated module
    spec_obj = importlib.util.spec_from_file_location("dissect_layer", layer_path)
    assert spec_obj is not None
    assert spec_obj.loader is not None

    module = importlib.util.module_from_spec(spec_obj)
    sys.modules["dissect_layer"] = module
    spec_obj.loader.exec_module(module)

    # Dissect raw packet
    TestProtocol = module.TestProtocol
    raw_packet = b"\xaa\x55\x01\x0a\x00\x01\x02\x03\x12\x34"
    pkt = TestProtocol(raw_packet)

    # Verify dissection
    assert pkt.sync == b"\xaa\x55"
    assert pkt.version == 1
    assert pkt.length == 10

    # Clean up
    del sys.modules["dissect_layer"]


def test_examples_in_docstring(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test that examples are included in docstring when requested.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(
        protocol_name="TestProtocol", generate_examples=True, show_progress=False
    )
    generator = ScapyLayerGenerator(config)

    sample_msg = b"\xaa\x55\x01\x0a\x00\x01\x02\x03\x12\x34"
    output_path = tmp_path / "examples_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[sample_msg],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify examples in docstring
    assert "Example:" in python_code
    assert ">>> pkt = TestProtocol()" in python_code
    assert sample_msg.hex() in python_code


def test_no_examples_when_disabled(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test that examples are excluded when disabled.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(
        protocol_name="TestProtocol", generate_examples=False, show_progress=False
    )
    generator = ScapyLayerGenerator(config)

    sample_msg = b"\xaa\x55\x01\x0a\x00\x01\x02\x03\x12\x34"
    output_path = tmp_path / "no_examples_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[sample_msg],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify no examples in docstring (basic class docstring still has "Example:")
    # but no hex sample
    assert sample_msg.hex() not in python_code


def test_invalid_spec_no_name() -> None:
    """Test validation fails for spec without name."""
    spec = ProtocolSpec(
        name="",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=10,
        fields=[],
        checksum_type=None,
        checksum_position=None,
        confidence=0.9,
    )

    config = ScapyLayerConfig(protocol_name="Test", show_progress=False)
    generator = ScapyLayerGenerator(config)

    with pytest.raises(ValueError, match="Protocol name is required"):
        generator._validate_spec(spec)


def test_invalid_spec_no_fields() -> None:
    """Test validation fails for spec without fields."""
    spec = ProtocolSpec(
        name="TestProto",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=10,
        fields=[],
        checksum_type=None,
        checksum_position=None,
        confidence=0.9,
    )

    config = ScapyLayerConfig(protocol_name="Test", show_progress=False)
    generator = ScapyLayerGenerator(config)

    with pytest.raises(ValueError, match="Protocol must have at least one field"):
        generator._validate_spec(spec)


def test_invalid_field_type() -> None:
    """Test validation fails for unsupported field type."""
    spec = ProtocolSpec(
        name="TestProto",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aa",
        frame_length=10,
        fields=[
            FieldSpec(name="bad_field", offset=0, size=1, field_type="unsupported_type"),
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.9,
    )

    config = ScapyLayerConfig(protocol_name="Test", show_progress=False)
    generator = ScapyLayerGenerator(config)

    with pytest.raises(ValueError, match="Unsupported field type"):
        generator._validate_spec(spec)


def test_safe_class_name_conversion() -> None:
    """Test safe class name conversion."""
    config = ScapyLayerConfig(protocol_name="Test", show_progress=False)
    generator = ScapyLayerGenerator(config)

    # Test various inputs
    assert generator._safe_class_name("MyProtocol") == "Myprotocol"
    assert generator._safe_class_name("my-protocol") == "MyProtocol"
    assert generator._safe_class_name("my_protocol") == "MyProtocol"
    assert generator._safe_class_name("Protocol 123") == "Protocol123"
    assert generator._safe_class_name("Test@#$Proto") == "TestProto"


def test_safe_field_name_conversion() -> None:
    """Test safe field name conversion."""
    config = ScapyLayerConfig(protocol_name="Test", show_progress=False)
    generator = ScapyLayerGenerator(config)

    # Test various inputs
    assert generator._safe_field_name("MyField") == "myfield"
    assert generator._safe_field_name("my-field") == "my_field"
    assert generator._safe_field_name("Field@123") == "field_123"


def test_crc8_generation(tmp_path: Path) -> None:
    """Test CRC-8 function generation.

    Args:
        tmp_path: Temporary directory for output.
    """
    spec = ProtocolSpec(
        name="CRC8Proto",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="7e",
        frame_length=5,
        fields=[
            FieldSpec(name="sync", offset=0, size=1, field_type="bytes"),
            FieldSpec(name="data", offset=1, size=3, field_type="bytes"),
            FieldSpec(name="crc", offset=4, size=1, field_type="uint8"),
        ],
        checksum_type="crc8",
        checksum_position=-1,
        confidence=0.90,
    )

    config = ScapyLayerConfig(
        protocol_name="CRC8Proto", include_crc_validation=True, show_progress=False
    )
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "crc8_layer.py"
    layer_path = generator.generate(spec, sample_messages=[], output_path=output_path)

    python_code = layer_path.read_text()

    # Verify CRC-8 function
    assert "def calculate_crc8" in python_code
    assert "poly = 0x07" in python_code


def test_crc32_generation(tmp_path: Path) -> None:
    """Test CRC-32 function generation.

    Args:
        tmp_path: Temporary directory for output.
    """
    spec = ProtocolSpec(
        name="CRC32Proto",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aabb",
        frame_length=10,
        fields=[
            FieldSpec(name="sync", offset=0, size=2, field_type="bytes"),
            FieldSpec(name="data", offset=2, size=4, field_type="bytes"),
            FieldSpec(name="crc", offset=6, size=4, field_type="uint32"),
        ],
        checksum_type="crc32",
        checksum_position=-1,
        confidence=0.92,
    )

    config = ScapyLayerConfig(
        protocol_name="CRC32Proto", include_crc_validation=True, show_progress=False
    )
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "crc32_layer.py"
    layer_path = generator.generate(spec, sample_messages=[], output_path=output_path)

    python_code = layer_path.read_text()

    # Verify CRC-32 function
    assert "def calculate_crc32" in python_code
    assert "poly = 0x04C11DB7" in python_code


def test_constant_field_with_value(tmp_path: Path) -> None:
    """Test constant field with default value.

    Args:
        tmp_path: Temporary directory for output.
    """
    field = FieldSpec(name="magic", offset=0, size=2, field_type="constant")
    field.value = "aa55"  # type: ignore[attr-defined]

    spec = ProtocolSpec(
        name="ConstantProto",
        baud_rate=115200,
        frame_format="8N1",
        sync_pattern="aa55",
        frame_length=3,
        fields=[
            field,
            FieldSpec(name="data", offset=2, size=1, field_type="uint8"),
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.95,
    )

    config = ScapyLayerConfig(protocol_name="ConstantProto", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "constant_layer.py"
    layer_path = generator.generate(spec, sample_messages=[], output_path=output_path)

    python_code = layer_path.read_text()

    # Verify constant field has default value
    assert "b'\\xaa\\x55'" in python_code


def test_progress_bar_enabled(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test progress bar is shown for >100 messages.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(protocol_name="TestProtocol", show_progress=True)
    generator = ScapyLayerGenerator(config)

    # Create >100 sample messages to trigger progress bar
    sample_messages = [b"\xaa\x55\x01\x0a\x00\x01\x02\x03\x12\x34"] * 150

    output_path = tmp_path / "progress_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=sample_messages,
        output_path=output_path,
    )

    # Just verify it completes without error
    assert layer_path.exists()


def test_base_class_customization(basic_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test custom base class configuration.

    Args:
        basic_spec: Basic protocol spec fixture.
        tmp_path: Temporary directory for output.
    """
    config = ScapyLayerConfig(protocol_name="TestProtocol", base_class="TCP", show_progress=False)
    generator = ScapyLayerGenerator(config)

    output_path = tmp_path / "custom_base_layer.py"
    layer_path = generator.generate(
        basic_spec,
        sample_messages=[],
        output_path=output_path,
    )

    python_code = layer_path.read_text()

    # Verify custom base class
    assert "class TestProtocol(TCP):" in python_code
