"""Tests for Kaitai Struct generator.

This module tests the KaitaiStructGenerator class which generates
valid .ksy files from ProtocolSpec objects.

Test Coverage:
    - Basic .ksy generation
    - All field types (uint8, uint16, uint32, bytes, string)
    - Endianness handling (big-endian, little-endian)
    - Enum generation
    - Constant field validation
    - Checksum field marking
    - YAML syntax validation
    - Protocol ID sanitization
    - Documentation inclusion
    - Round-trip (generate .ksy, validate parseable)
"""

from pathlib import Path

import pytest
import yaml

from oscura.export.kaitai_struct import KaitaiConfig, KaitaiStructGenerator
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
            FieldSpec(name="checksum", offset=8, size=2, field_type="checksum"),
        ],
        checksum_type=None,
        checksum_position=None,
        confidence=0.95,
    )


@pytest.fixture
def all_field_types_spec() -> ProtocolSpec:
    """Create protocol spec with all supported field types.

    Returns:
        ProtocolSpec with comprehensive field coverage.
    """
    return ProtocolSpec(
        name="AllFieldTypes",
        baud_rate=9600,
        frame_format="8N1",
        sync_pattern="7e",
        frame_length=20,
        fields=[
            FieldSpec(name="sync", offset=0, size=1, field_type="constant", value="7e"),
            FieldSpec(name="byte_field", offset=1, size=1, field_type="uint8"),
            FieldSpec(name="word_field", offset=2, size=2, field_type="uint16"),
            FieldSpec(name="dword_field", offset=4, size=4, field_type="uint32"),
            FieldSpec(name="bytes_field", offset=8, size=4, field_type="bytes"),
            FieldSpec(name="string_field", offset=12, size=4, field_type="string"),
            FieldSpec(name="checksum_field", offset=16, size=2, field_type="checksum"),
        ],
        checksum_type="crc16",
        checksum_position=-1,
        confidence=0.88,
    )


@pytest.fixture
def enum_spec() -> ProtocolSpec:
    """Create protocol spec with enum fields.

    Returns:
        ProtocolSpec with enum field.
    """
    # Create field with enum
    msg_type_field = FieldSpec(
        name="message_type",
        offset=2,
        size=1,
        field_type="uint8",
    )
    # Add enum attribute
    msg_type_field.enum = {  # type: ignore[attr-defined]
        0x01: "init",
        0x02: "data",
        0x03: "ack",
        0x04: "error",
    }

    return ProtocolSpec(
        name="EnumProtocol",
        baud_rate=19200,
        frame_format="8N1",
        sync_pattern="a5",
        frame_length=8,
        fields=[
            FieldSpec(name="sync", offset=0, size=1, field_type="bytes"),
            FieldSpec(name="seq_num", offset=1, size=1, field_type="uint8"),
            msg_type_field,
            FieldSpec(name="payload", offset=3, size=4, field_type="bytes"),
            FieldSpec(name="crc", offset=7, size=1, field_type="checksum"),
        ],
        checksum_type="crc8",
        checksum_position=-1,
        confidence=0.92,
    )


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to output directory.
    """
    output_dir = tmp_path / "ksy_output"
    output_dir.mkdir()
    return output_dir


class TestKaitaiConfig:
    """Test KaitaiConfig validation and defaults."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = KaitaiConfig(protocol_id="test_proto")
        assert config.protocol_id == "test_proto"
        assert config.endian == "le"
        assert config.include_doc is True
        assert config.validate_syntax is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = KaitaiConfig(
            protocol_id="my_protocol",
            endian="be",
            include_doc=False,
            validate_syntax=False,
        )
        assert config.protocol_id == "my_protocol"
        assert config.endian == "be"
        assert config.include_doc is False
        assert config.validate_syntax is False


class TestKaitaiStructGenerator:
    """Test KaitaiStructGenerator basic functionality."""

    def test_basic_generation(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test basic .ksy file generation.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="test_protocol")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "test_protocol.ksy"

        result_path = generator.generate(basic_spec, output_path)

        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_yaml_is_valid(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that generated YAML is valid and parseable.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="test_protocol")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "test_protocol.ksy"

        generator.generate(basic_spec, output_path)

        # Parse YAML and verify structure
        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        assert "meta" in ksy_data
        assert "seq" in ksy_data
        assert ksy_data["meta"]["id"] == "test_protocol"
        assert ksy_data["meta"]["endian"] == "le"

    def test_field_types_mapping(
        self,
        all_field_types_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that all field types are correctly mapped to Kaitai types.

        Args:
            all_field_types_spec: Spec with all field types.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="all_types")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "all_types.ksy"

        generator.generate(all_field_types_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        fields = ksy_data["seq"]
        field_types = {f["id"]: f["type"] for f in fields}

        # Verify type mappings
        assert field_types["sync"] == "bytes"
        assert field_types["byte_field"] == "u1"
        assert field_types["word_field"] == "u2"
        assert field_types["dword_field"] == "u4"
        assert field_types["bytes_field"] == "bytes"
        assert field_types["string_field"] == "str"
        assert field_types["checksum_field"] == "bytes"

    def test_field_sizes(
        self,
        all_field_types_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that field sizes are correctly specified.

        Args:
            all_field_types_spec: Spec with all field types.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="all_types")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "all_types.ksy"

        generator.generate(all_field_types_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        fields = {f["id"]: f for f in ksy_data["seq"]}

        # Verify sizes
        assert fields["sync"]["size"] == 1
        assert fields["bytes_field"]["size"] == 4
        assert fields["string_field"]["size"] == 4
        assert fields["checksum_field"]["size"] == 2

        # Verify string encoding
        assert fields["string_field"]["encoding"] == "UTF-8"

    def test_endianness_configuration(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test endianness configuration.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        # Test little-endian
        config_le = KaitaiConfig(protocol_id="test_le", endian="le")
        generator_le = KaitaiStructGenerator(config_le)
        output_le = temp_output_dir / "test_le.ksy"
        generator_le.generate(basic_spec, output_le)

        with open(output_le, encoding="utf-8") as f:
            ksy_le = yaml.safe_load(f)
        assert ksy_le["meta"]["endian"] == "le"

        # Test big-endian
        config_be = KaitaiConfig(protocol_id="test_be", endian="be")
        generator_be = KaitaiStructGenerator(config_be)
        output_be = temp_output_dir / "test_be.ksy"
        generator_be.generate(basic_spec, output_be)

        with open(output_be, encoding="utf-8") as f:
            ksy_be = yaml.safe_load(f)
        assert ksy_be["meta"]["endian"] == "be"

    def test_enum_generation(
        self,
        enum_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test enum generation for fields with enum values.

        Args:
            enum_spec: Protocol spec with enum field.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="enum_test")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "enum_test.ksy"

        generator.generate(enum_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        # Verify enums section exists
        assert "enums" in ksy_data
        assert "message_type_enum" in ksy_data["enums"]

        # Verify enum values
        enum_values = ksy_data["enums"]["message_type_enum"]
        assert enum_values[0x01] == "init"
        assert enum_values[0x02] == "data"
        assert enum_values[0x03] == "ack"
        assert enum_values[0x04] == "error"

        # Verify field references enum
        msg_type_field = next(f for f in ksy_data["seq"] if f["id"] == "message_type")
        assert msg_type_field["enum"] == "message_type_enum"

    def test_constant_field_contents(
        self,
        all_field_types_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that constant fields include contents validation.

        Args:
            all_field_types_spec: Spec with constant field.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="const_test")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "const_test.ksy"

        generator.generate(all_field_types_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        # Find sync field (constant)
        sync_field = next(f for f in ksy_data["seq"] if f["id"] == "sync")
        assert "contents" in sync_field
        assert sync_field["contents"] == [0x7E]  # Hex value of "7e"

    def test_documentation_inclusion(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test documentation inclusion when enabled.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="doc_test", include_doc=True)
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "doc_test.ksy"

        generator.generate(basic_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        # Verify top-level doc
        assert "doc" in ksy_data
        assert "TestProtocol" in ksy_data["doc"]
        assert "115200" in ksy_data["doc"]

        # Verify field docs
        for field in ksy_data["seq"]:
            assert "doc" in field

        # Verify meta documentation
        assert "title" in ksy_data["meta"]
        assert ksy_data["meta"]["title"] == "TestProtocol"

    def test_documentation_exclusion(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test documentation exclusion when disabled.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="no_doc_test", include_doc=False)
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "no_doc_test.ksy"

        generator.generate(basic_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        # Verify no top-level doc
        assert "doc" not in ksy_data

        # Verify no field docs
        for field in ksy_data["seq"]:
            assert "doc" not in field

    def test_field_name_sanitization(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test that field names are sanitized for Kaitai compatibility.

        Args:
            temp_output_dir: Temporary output directory.
        """
        spec = ProtocolSpec(
            name="SanitizeTest",
            baud_rate=9600,
            frame_format="8N1",
            sync_pattern="aa",
            frame_length=5,
            fields=[
                FieldSpec(name="Field-With-Dashes", offset=0, size=1, field_type="uint8"),
                FieldSpec(name="Field With Spaces", offset=1, size=1, field_type="uint8"),
                FieldSpec(name="UPPERCASE_FIELD", offset=2, size=1, field_type="uint8"),
                FieldSpec(name="123numeric", offset=3, size=1, field_type="uint8"),
            ],
            checksum_type=None,
            checksum_position=None,
            confidence=0.9,
        )

        config = KaitaiConfig(protocol_id="sanitize_test")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "sanitize_test.ksy"

        generator.generate(spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        field_ids = [f["id"] for f in ksy_data["seq"]]

        # Verify sanitization
        assert "field_with_dashes" in field_ids
        assert "field_with_spaces" in field_ids
        assert "uppercase_field" in field_ids
        assert "field_123numeric" in field_ids

    def test_invalid_protocol_id(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test validation of invalid protocol IDs.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        # Test uppercase protocol ID
        config = KaitaiConfig(protocol_id="TestProtocol")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "test.ksy"

        with pytest.raises(ValueError, match="must be lowercase"):
            generator.generate(basic_spec, output_path)

        # Test protocol ID with invalid characters
        config2 = KaitaiConfig(protocol_id="test-protocol")
        generator2 = KaitaiStructGenerator(config2)

        with pytest.raises(ValueError, match="must be alphanumeric"):
            generator2.generate(basic_spec, output_path)

    def test_invalid_endianness(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test validation of invalid endianness.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="test_proto", endian="invalid")  # type: ignore[arg-type]
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "test.ksy"

        with pytest.raises(ValueError, match="endian must be"):
            generator.generate(basic_spec, output_path)

    def test_empty_protocol_name(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test validation of empty protocol name.

        Args:
            temp_output_dir: Temporary output directory.
        """
        spec = ProtocolSpec(
            name="",
            baud_rate=9600,
            frame_format="8N1",
            sync_pattern="aa",
            frame_length=5,
            fields=[
                FieldSpec(name="test", offset=0, size=1, field_type="uint8"),
            ],
            checksum_type=None,
            checksum_position=None,
            confidence=0.9,
        )

        config = KaitaiConfig(protocol_id="test_proto")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "test.ksy"

        with pytest.raises(ValueError, match="Protocol name is required"):
            generator.generate(spec, output_path)

    def test_no_fields(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test validation of protocol with no fields.

        Args:
            temp_output_dir: Temporary output directory.
        """
        spec = ProtocolSpec(
            name="TestProtocol",
            baud_rate=9600,
            frame_format="8N1",
            sync_pattern="aa",
            frame_length=5,
            fields=[],
            checksum_type=None,
            checksum_position=None,
            confidence=0.9,
        )

        config = KaitaiConfig(protocol_id="test_proto")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "test.ksy"

        with pytest.raises(ValueError, match="at least one field"):
            generator.generate(spec, output_path)

    def test_unsupported_field_type(
        self,
        temp_output_dir: Path,
    ) -> None:
        """Test validation of unsupported field types.

        Args:
            temp_output_dir: Temporary output directory.
        """
        spec = ProtocolSpec(
            name="TestProtocol",
            baud_rate=9600,
            frame_format="8N1",
            sync_pattern="aa",
            frame_length=5,
            fields=[
                FieldSpec(name="test", offset=0, size=1, field_type="unsupported"),
            ],
            checksum_type=None,
            checksum_position=None,
            confidence=0.9,
        )

        config = KaitaiConfig(protocol_id="test_proto")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "test.ksy"

        with pytest.raises(ValueError, match="Unsupported field type"):
            generator.generate(spec, output_path)

    def test_metadata_fields(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that metadata fields are correctly included.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="meta_test", include_doc=True)
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "meta_test.ksy"

        generator.generate(basic_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        meta = ksy_data["meta"]
        assert meta["id"] == "meta_test"
        assert meta["endian"] == "le"
        assert meta["title"] == "TestProtocol"
        assert "Oscura" in meta["application"]
        assert "0.95" in meta["application"]
        assert meta["file-extension"] == "bin"

    def test_round_trip_yaml_parsing(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that generated .ksy can be parsed and regenerated.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="roundtrip_test")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "roundtrip_test.ksy"

        # Generate initial .ksy
        generator.generate(basic_spec, output_path)

        # Parse YAML
        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        # Regenerate YAML from parsed data
        regenerated_yaml = yaml.dump(ksy_data, default_flow_style=False, sort_keys=False)

        # Parse regenerated YAML
        regenerated_data = yaml.safe_load(regenerated_yaml)

        # Verify structure is identical
        assert regenerated_data["meta"]["id"] == ksy_data["meta"]["id"]
        assert len(regenerated_data["seq"]) == len(ksy_data["seq"])
        for orig_field, regen_field in zip(ksy_data["seq"], regenerated_data["seq"], strict=True):
            assert orig_field["id"] == regen_field["id"]
            assert orig_field["type"] == regen_field["type"]

    def test_checksum_metadata(
        self,
        all_field_types_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that checksum information is included in documentation.

        Args:
            all_field_types_spec: Spec with checksum field.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="checksum_test", include_doc=True)
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "checksum_test.ksy"

        generator.generate(all_field_types_spec, output_path)

        with open(output_path, encoding="utf-8") as f:
            ksy_data = yaml.safe_load(f)

        doc = ksy_data["doc"]
        assert "crc16" in doc.lower()
        assert "end of frame" in doc.lower()

    def test_yaml_formatting(
        self,
        basic_spec: ProtocolSpec,
        temp_output_dir: Path,
    ) -> None:
        """Test that YAML is properly formatted and readable.

        Args:
            basic_spec: Basic protocol specification fixture.
            temp_output_dir: Temporary output directory.
        """
        config = KaitaiConfig(protocol_id="format_test")
        generator = KaitaiStructGenerator(config)
        output_path = temp_output_dir / "format_test.ksy"

        generator.generate(basic_spec, output_path)

        # Read raw content
        content = output_path.read_text(encoding="utf-8")

        # Verify block style (not flow style)
        assert "{" not in content  # No flow style dictionaries
        assert "[" not in content or "contents:" in content  # Arrays OK for contents

        # Verify proper indentation
        lines = content.split("\n")
        for line in lines:
            if line.strip():
                # Check indentation is spaces (not tabs)
                indent = len(line) - len(line.lstrip())
                assert indent % 2 == 0  # YAML uses 2-space indentation
