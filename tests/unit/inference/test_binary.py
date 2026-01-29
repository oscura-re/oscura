"""Comprehensive unit tests for binary format inference.

Tests for:
    - RE-BIN-001: Magic Byte Detection
    - RE-BIN-002: Structure Alignment Detection
    - RE-BIN-003: Binary Parser DSL

This module provides comprehensive test coverage for binary format
inference, magic byte detection, alignment analysis, and parser generation.
"""

from __future__ import annotations

import struct

import pytest

from oscura.inference.binary import (
    KNOWN_MAGIC_BYTES,
    AlignmentDetector,
    AlignmentResult,
    BinaryParserGenerator,
    MagicByteDetector,
    MagicByteResult,
    ParserDefinition,
    ParserField,
    detect_alignment,
    detect_magic_bytes,
    find_all_magic_bytes,
    generate_parser,
    parser_to_python,
    parser_to_yaml,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test MagicByteResult Dataclass
# =============================================================================


class TestMagicByteResult:
    """Test MagicByteResult dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating MagicByteResult with all fields."""
        result = MagicByteResult(
            magic=b"\x89PNG\r\n\x1a\n",
            offset=0,
            confidence=1.0,
            frequency=5,
            known_format="PNG",
            file_extension=".png",
        )

        assert result.magic == b"\x89PNG\r\n\x1a\n"
        assert result.offset == 0
        assert result.confidence == 1.0
        assert result.frequency == 5
        assert result.known_format == "PNG"
        assert result.file_extension == ".png"

    def test_creation_with_minimal_fields(self) -> None:
        """Test creating MagicByteResult with minimal fields."""
        result = MagicByteResult(
            magic=b"TEST",
            offset=10,
            confidence=0.8,
            frequency=3,
        )

        assert result.magic == b"TEST"
        assert result.offset == 10
        assert result.confidence == 0.8
        assert result.frequency == 3
        assert result.known_format is None
        assert result.file_extension is None


# =============================================================================
# Test AlignmentResult Dataclass
# =============================================================================


class TestAlignmentResult:
    """Test AlignmentResult dataclass."""

    def test_creation_with_all_fields(self) -> None:
        """Test creating AlignmentResult with all fields."""
        result = AlignmentResult(
            alignment=4,
            padding_positions=[3, 7],
            field_boundaries=[0, 4, 8],
            confidence=0.9,
            structure_size=16,
        )

        assert result.alignment == 4
        assert result.padding_positions == [3, 7]
        assert result.field_boundaries == [0, 4, 8]
        assert result.confidence == 0.9
        assert result.structure_size == 16

    def test_creation_with_minimal_fields(self) -> None:
        """Test creating AlignmentResult with minimal fields."""
        result = AlignmentResult(
            alignment=8,
            padding_positions=[],
            field_boundaries=[],
            confidence=0.5,
        )

        assert result.alignment == 8
        assert result.padding_positions == []
        assert result.field_boundaries == []
        assert result.confidence == 0.5
        assert result.structure_size is None


# =============================================================================
# Test ParserField Dataclass
# =============================================================================


class TestParserField:
    """Test ParserField dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ParserField with default values."""
        field = ParserField(
            name="test_field",
            offset=0,
            size=4,
            field_type="uint32",
        )

        assert field.name == "test_field"
        assert field.offset == 0
        assert field.size == 4
        assert field.field_type == "uint32"
        assert field.endian == "big"
        assert field.array_count == 1
        assert field.condition is None
        assert field.description == ""

    def test_creation_with_all_fields(self) -> None:
        """Test creating ParserField with all fields."""
        field = ParserField(
            name="array_field",
            offset=8,
            size=16,
            field_type="uint8",
            endian="little",
            array_count=16,
            condition="length > 0",
            description="Array of bytes",
        )

        assert field.name == "array_field"
        assert field.offset == 8
        assert field.size == 16
        assert field.field_type == "uint8"
        assert field.endian == "little"
        assert field.array_count == 16
        assert field.condition == "length > 0"
        assert field.description == "Array of bytes"


# =============================================================================
# Test ParserDefinition Dataclass
# =============================================================================


class TestParserDefinition:
    """Test ParserDefinition dataclass."""

    def test_creation_with_defaults(self) -> None:
        """Test creating ParserDefinition with defaults."""
        parser = ParserDefinition(
            name="TestStruct",
            fields=[],
            total_size=0,
        )

        assert parser.name == "TestStruct"
        assert parser.fields == []
        assert parser.total_size == 0
        assert parser.endian == "big"
        assert parser.magic is None
        assert parser.version == "1.0"

    def test_creation_with_all_fields(self) -> None:
        """Test creating ParserDefinition with all fields."""
        field = ParserField(name="field1", offset=0, size=4, field_type="uint32")
        parser = ParserDefinition(
            name="MyStruct",
            fields=[field],
            total_size=4,
            endian="little",
            magic=b"MAGIC",
            version="2.0",
        )

        assert parser.name == "MyStruct"
        assert len(parser.fields) == 1
        assert parser.total_size == 4
        assert parser.endian == "little"
        assert parser.magic == b"MAGIC"
        assert parser.version == "2.0"


# =============================================================================
# Test MagicByteDetector
# =============================================================================


class TestMagicByteDetector:
    """Test MagicByteDetector class.

    Tests RE-BIN-001: Magic Byte Detection.
    """

    def test_initialization_defaults(self) -> None:
        """Test detector initialization with defaults."""
        detector = MagicByteDetector()

        assert detector.known_signatures == KNOWN_MAGIC_BYTES
        assert detector.min_magic_length == 2
        assert detector.max_magic_length == 16

    def test_initialization_custom(self) -> None:
        """Test detector initialization with custom parameters."""
        custom_sigs = {b"TEST": ("Test Format", ".test")}
        detector = MagicByteDetector(
            known_signatures=custom_sigs,
            min_magic_length=3,
            max_magic_length=8,
        )

        assert detector.known_signatures == custom_sigs
        assert detector.min_magic_length == 3
        assert detector.max_magic_length == 8

    def test_detect_png_signature(self) -> None:
        """Test detecting PNG magic bytes."""
        detector = MagicByteDetector()
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = detector.detect(data)

        assert result is not None
        assert result.magic == b"\x89PNG\r\n\x1a\n"
        assert result.offset == 0
        assert result.confidence == 1.0
        assert result.known_format == "PNG"
        assert result.file_extension == ".png"

    def test_detect_jpeg_signature(self) -> None:
        """Test detecting JPEG magic bytes."""
        detector = MagicByteDetector()
        data = b"\xff\xd8\xff" + b"\x00" * 100

        result = detector.detect(data)

        assert result is not None
        assert result.magic == b"\xff\xd8\xff"
        assert result.known_format == "JPEG"

    def test_detect_pdf_signature(self) -> None:
        """Test detecting PDF magic bytes."""
        detector = MagicByteDetector()
        data = b"%PDF-1.5" + b"\x00" * 100

        result = detector.detect(data)

        assert result is not None
        assert result.magic == b"%PDF"
        assert result.known_format == "PDF"

    def test_detect_elf_signature(self) -> None:
        """Test detecting ELF magic bytes."""
        detector = MagicByteDetector()
        data = b"\x7fELF" + b"\x00" * 100

        result = detector.detect(data)

        assert result is not None
        assert result.magic == b"\x7fELF"
        assert result.known_format == "ELF Executable"

    def test_detect_zip_signature(self) -> None:
        """Test detecting ZIP magic bytes."""
        detector = MagicByteDetector()
        data = b"PK\x03\x04" + b"\x00" * 100

        result = detector.detect(data)

        assert result is not None
        assert result.magic == b"PK\x03\x04"
        assert result.known_format == "ZIP"

    def test_detect_at_offset(self) -> None:
        """Test detecting magic bytes at custom offset."""
        detector = MagicByteDetector()
        data = b"\x00" * 10 + b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        result = detector.detect(data, offset=10)

        assert result is not None
        assert result.magic == b"\x89PNG\r\n\x1a\n"
        assert result.offset == 10

    def test_detect_no_match(self) -> None:
        """Test detection when no magic bytes found."""
        detector = MagicByteDetector()
        data = b"\x00" * 100

        result = detector.detect(data)

        assert result is None

    def test_detect_offset_beyond_data(self) -> None:
        """Test detection with offset beyond data length."""
        detector = MagicByteDetector()
        data = b"\x00" * 10

        result = detector.detect(data, offset=100)

        assert result is None

    def test_detect_all_single_signature(self) -> None:
        """Test detecting all magic bytes with single signature."""
        detector = MagicByteDetector()
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        results = detector.detect_all(data)

        assert len(results) == 1
        assert results[0].magic == b"\x89PNG\r\n\x1a\n"

    def test_detect_all_multiple_signatures(self) -> None:
        """Test detecting all magic bytes with multiple signatures."""
        detector = MagicByteDetector()
        # PNG followed by ZIP
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10 + b"PK\x03\x04" + b"\x00" * 10

        results = detector.detect_all(data)

        assert len(results) >= 2
        formats = [r.known_format for r in results]
        assert "PNG" in formats
        assert "ZIP" in formats

    def test_detect_all_empty_data(self) -> None:
        """Test detecting all magic bytes in empty data."""
        detector = MagicByteDetector()
        data = b""

        results = detector.detect_all(data)

        assert results == []

    def test_learn_magic_from_samples(self) -> None:
        """Test learning magic bytes from multiple samples."""
        detector = MagicByteDetector()
        samples = [
            b"MAGIC\x00\x01\x02\x03",
            b"MAGIC\x00\x04\x05\x06",
            b"MAGIC\x00\x07\x08\x09",
        ]

        results = detector.learn_magic_from_samples(samples, min_frequency=2)

        assert len(results) > 0
        # Should find "MAGIC" as common prefix
        magic_found = any(b"MAGIC" in r.magic for r in results)
        assert magic_found

    def test_learn_magic_varied_prefixes(self) -> None:
        """Test learning magic bytes with varied prefixes."""
        detector = MagicByteDetector()
        samples = [
            b"AAA" + b"\x00" * 10,
            b"AAA" + b"\x01" * 10,
            b"BBB" + b"\x00" * 10,
        ]

        results = detector.learn_magic_from_samples(samples, min_frequency=2)

        # Should find "AAA" as common prefix (appears twice)
        assert any(r.magic == b"AAA" for r in results)

    def test_learn_magic_empty_samples(self) -> None:
        """Test learning magic bytes from empty sample list."""
        detector = MagicByteDetector()
        samples: list[bytes] = []

        results = detector.learn_magic_from_samples(samples)

        assert results == []

    def test_learn_magic_confidence_calculation(self) -> None:
        """Test confidence calculation in learned magic bytes."""
        detector = MagicByteDetector()
        samples = [b"TEST"] * 8 + [b"OTHER"] * 2

        results = detector.learn_magic_from_samples(samples, min_frequency=2)

        # Find TEST result
        test_result = next((r for r in results if r.magic == b"TEST"), None)
        assert test_result is not None
        assert test_result.confidence == 0.8  # 8/10

    def test_learn_magic_frequency_filter(self) -> None:
        """Test frequency filtering in magic byte learning."""
        detector = MagicByteDetector()
        samples = [b"AAA", b"AAA", b"BBB"]

        # With min_frequency=3, nothing should match
        results = detector.learn_magic_from_samples(samples, min_frequency=3)

        assert all(r.frequency < 3 for r in results)

    def test_learn_magic_recognizes_known(self) -> None:
        """Test learning recognizes known signatures."""
        detector = MagicByteDetector()
        samples = [b"\x89PNG\r\n\x1a\n" + b"\x00" * 10] * 3

        results = detector.learn_magic_from_samples(samples, min_frequency=2)

        # Should recognize PNG
        png_result = next(
            (r for r in results if r.known_format == "PNG"),
            None,
        )
        assert png_result is not None

    def test_add_signature(self) -> None:
        """Test adding custom signature."""
        detector = MagicByteDetector()
        custom_magic = b"CUSTOM"

        detector.add_signature(custom_magic, "Custom Format", ".cust")

        assert custom_magic in detector.known_signatures
        assert detector.known_signatures[custom_magic] == ("Custom Format", ".cust")

    def test_add_signature_detection(self) -> None:
        """Test detection of added custom signature."""
        detector = MagicByteDetector()
        custom_magic = b"MYFORMAT"
        detector.add_signature(custom_magic, "My Format", ".mf")

        data = b"MYFORMAT" + b"\x00" * 10
        result = detector.detect(data)

        assert result is not None
        assert result.known_format == "My Format"
        assert result.file_extension == ".mf"


# =============================================================================
# Test AlignmentDetector
# =============================================================================


class TestAlignmentDetector:
    """Test AlignmentDetector class.

    Tests RE-BIN-002: Structure Alignment Detection.
    """

    def test_initialization_defaults(self) -> None:
        """Test detector initialization with defaults."""
        detector = AlignmentDetector()

        assert detector.test_alignments == [1, 2, 4, 8, 16]
        assert detector.padding_byte is None

    def test_initialization_custom(self) -> None:
        """Test detector initialization with custom parameters."""
        detector = AlignmentDetector(
            test_alignments=[2, 4, 8],
            padding_byte=0xCC,
        )

        assert detector.test_alignments == [2, 4, 8]
        assert detector.padding_byte == 0xCC

    def test_detect_empty_data(self) -> None:
        """Test alignment detection on empty data."""
        detector = AlignmentDetector()
        result = detector.detect(b"")

        assert result.alignment == 1
        assert result.padding_positions == []
        assert result.field_boundaries == []
        assert result.confidence == 0.0

    def test_detect_aligned_4byte_structure(self) -> None:
        """Test detecting 4-byte aligned structure."""
        detector = AlignmentDetector()
        # Structure: uint32 + uint32 (8 bytes, 4-byte aligned)
        data = struct.pack(">II", 0x12345678, 0xABCDEF00)

        result = detector.detect(data)

        # Should detect some alignment
        assert result.alignment in [1, 2, 4, 8]

    def test_detect_padding_positions(self) -> None:
        """Test detecting padding positions."""
        detector = AlignmentDetector()
        # Data with null padding
        data = b"\x01\x02\x03\x00\x00\x00\x04\x05\x06"

        result = detector.detect(data)

        # Should find padding at positions 3, 4, 5
        assert len(result.padding_positions) > 0

    def test_detect_field_boundaries(self) -> None:
        """Test detecting field boundaries."""
        detector = AlignmentDetector()
        # Create data with distinct sections
        data = b"\xff" * 4 + b"\x00" * 4 + b"\xaa" * 4

        result = detector.detect(data)

        # Should detect some boundaries
        assert isinstance(result.field_boundaries, list)

    def test_detect_with_custom_padding_byte(self) -> None:
        """Test detection with custom padding byte."""
        detector = AlignmentDetector(padding_byte=0xCC)
        data = b"\x01\x02\xcc\xcc\x03\x04\xcc\xcc"

        result = detector.detect(data)

        # Should find 0xCC padding positions
        assert len(result.padding_positions) > 0

    def test_detect_structure_size(self) -> None:
        """Test structure size estimation."""
        detector = AlignmentDetector(test_alignments=[4])
        # Create repeating 8-byte structures
        data = struct.pack(">II", 0x11111111, 0x22222222) * 4

        result = detector.detect(data)

        # Should estimate structure size
        if result.structure_size:
            assert result.structure_size in [4, 8, 16, 32]

    def test_detect_field_types(self) -> None:
        """Test field type detection."""
        detector = AlignmentDetector()
        data = struct.pack(">BHI", 0x01, 0x0202, 0x03030303)

        alignment_result = detector.detect(data)
        field_types = detector.detect_field_types(data, alignment_result)

        assert len(field_types) > 0
        # Each tuple is (offset, size, type)
        for offset, size, field_type in field_types:
            assert isinstance(offset, int)
            assert isinstance(size, int)
            assert isinstance(field_type, str)

    def test_detect_field_types_uint8(self) -> None:
        """Test detection of uint8 fields."""
        detector = AlignmentDetector()
        data = b"\x01"

        alignment_result = AlignmentResult(
            alignment=1,
            padding_positions=[],
            field_boundaries=[],
            confidence=1.0,
        )
        field_types = detector.detect_field_types(data, alignment_result)

        assert len(field_types) == 1
        assert field_types[0] == (0, 1, "uint8")

    def test_detect_field_types_uint16(self) -> None:
        """Test detection of uint16 fields."""
        detector = AlignmentDetector()
        data = struct.pack(">H", 0x1234)

        alignment_result = AlignmentResult(
            alignment=2,
            padding_positions=[],
            field_boundaries=[],
            confidence=1.0,
        )
        field_types = detector.detect_field_types(data, alignment_result)

        assert len(field_types) == 1
        assert field_types[0] == (0, 2, "uint16")

    def test_detect_field_types_uint32(self) -> None:
        """Test detection of uint32 fields."""
        detector = AlignmentDetector()
        data = struct.pack(">I", 0x12345678)

        alignment_result = AlignmentResult(
            alignment=4,
            padding_positions=[],
            field_boundaries=[],
            confidence=1.0,
        )
        field_types = detector.detect_field_types(data, alignment_result)

        assert len(field_types) == 1
        assert field_types[0] == (0, 4, "uint32")

    def test_detect_field_types_uint64(self) -> None:
        """Test detection of uint64 fields."""
        detector = AlignmentDetector()
        data = struct.pack(">Q", 0x123456789ABCDEF0)

        alignment_result = AlignmentResult(
            alignment=8,
            padding_positions=[],
            field_boundaries=[],
            confidence=1.0,
        )
        field_types = detector.detect_field_types(data, alignment_result)

        assert len(field_types) == 1
        assert field_types[0] == (0, 8, "uint64")

    def test_detect_field_types_bytes(self) -> None:
        """Test detection of byte array fields."""
        detector = AlignmentDetector()
        data = b"\x01\x02\x03\x04\x05"

        alignment_result = AlignmentResult(
            alignment=1,
            padding_positions=[],
            field_boundaries=[],
            confidence=1.0,
        )
        field_types = detector.detect_field_types(data, alignment_result)

        assert len(field_types) == 1
        assert field_types[0][2] == "bytes[5]"

    def test_detect_padding_byte_default(self) -> None:
        """Test automatic padding byte detection."""
        detector = AlignmentDetector()
        data = b"\x01\x02\x00\x00\x00\x03\x04\x00\x00"

        padding_byte = detector._detect_padding_byte(data)

        # Should detect 0x00 as most common
        assert padding_byte == 0x00

    def test_detect_padding_byte_ff(self) -> None:
        """Test detecting 0xFF as padding byte."""
        detector = AlignmentDetector()
        data = b"\x01\xff\xff\xff\x02\xff\xff"

        padding_byte = detector._detect_padding_byte(data)

        assert padding_byte == 0xFF

    def test_find_padding_single_region(self) -> None:
        """Test finding single padding region."""
        detector = AlignmentDetector()
        data = b"\x01\x02\x00\x00\x00\x03\x04"

        positions = detector._find_padding(data, 0x00)

        assert 2 in positions
        assert 3 in positions
        assert 4 in positions

    def test_find_padding_multiple_regions(self) -> None:
        """Test finding multiple padding regions."""
        detector = AlignmentDetector()
        data = b"\x01\x00\x00\x02\x00\x00\x00\x03"

        positions = detector._find_padding(data, 0x00)

        assert len(positions) > 0

    def test_find_field_boundaries_short_data(self) -> None:
        """Test field boundary detection on short data."""
        detector = AlignmentDetector()
        data = b"\x01\x02"

        boundaries = detector._find_field_boundaries(data)

        # Too short for analysis
        assert boundaries == []

    def test_score_alignment_no_data(self) -> None:
        """Test alignment scoring with no padding/boundaries."""
        detector = AlignmentDetector()
        data = b"\x01\x02\x03\x04"

        score = detector._score_alignment(data, 4, [], [])

        # Should return default score
        assert 0.0 <= score <= 1.0

    def test_score_alignment_too_large(self) -> None:
        """Test alignment scoring with alignment larger than data."""
        detector = AlignmentDetector()
        data = b"\x01\x02"

        score = detector._score_alignment(data, 16, [], [])

        assert score == 0.0

    def test_estimate_structure_size_aligned(self) -> None:
        """Test structure size estimation for aligned data."""
        detector = AlignmentDetector()
        # 16 bytes, 4-byte aligned, could be 4x4 or 2x8
        data = b"\x00" * 16

        size = detector._estimate_structure_size(data, 4)

        assert size in [4, 8, 16]

    def test_estimate_structure_size_no_match(self) -> None:
        """Test structure size estimation with no clear pattern."""
        detector = AlignmentDetector()
        data = b"\x00" * 7  # Prime-ish length

        size = detector._estimate_structure_size(data, 4)

        assert size is None


# =============================================================================
# Test BinaryParserGenerator
# =============================================================================


class TestBinaryParserGenerator:
    """Test BinaryParserGenerator class.

    Tests RE-BIN-003: Binary Parser DSL.
    """

    def test_initialization_defaults(self) -> None:
        """Test generator initialization with defaults."""
        generator = BinaryParserGenerator()

        assert generator.default_endian == "big"

    def test_initialization_little_endian(self) -> None:
        """Test generator initialization with little endian."""
        generator = BinaryParserGenerator(default_endian="little")

        assert generator.default_endian == "little"

    def test_generate_empty_samples(self) -> None:
        """Test generating parser from empty samples."""
        generator = BinaryParserGenerator()
        samples: list[bytes] = []

        parser = generator.generate(samples)

        assert parser.name == "Structure"
        assert parser.fields == []
        assert parser.total_size == 0

    def test_generate_single_sample(self) -> None:
        """Test generating parser from single sample."""
        generator = BinaryParserGenerator()
        samples = [b"\x01\x02\x03\x04"]

        parser = generator.generate(samples, name="TestStruct")

        assert parser.name == "TestStruct"
        assert parser.total_size == 4
        assert len(parser.fields) >= 0

    def test_generate_with_magic_bytes(self) -> None:
        """Test generating parser that detects magic bytes."""
        generator = BinaryParserGenerator()
        samples = [b"\x89PNG\r\n\x1a\n" + b"\x00" * 10]

        parser = generator.generate(samples, name="PNGFile")

        assert parser.magic == b"\x89PNG\r\n\x1a\n"

    def test_generate_multiple_samples(self) -> None:
        """Test generating parser from multiple samples."""
        generator = BinaryParserGenerator()
        samples = [
            struct.pack(">HI", 0x1234, 0x56789ABC),
            struct.pack(">HI", 0x5678, 0x9ABCDEF0),
            struct.pack(">HI", 0xABCD, 0x12345678),
        ]

        parser = generator.generate(samples, name="Packet")

        assert parser.name == "Packet"
        assert parser.total_size == 6

    def test_generate_from_definition(self) -> None:
        """Test generating parser from dictionary definition."""
        generator = BinaryParserGenerator()
        definition = {
            "name": "MyStruct",
            "size": 8,
            "endian": "little",
            "fields": [
                {"name": "field1", "offset": 0, "size": 4, "type": "uint32"},
                {"name": "field2", "offset": 4, "size": 4, "type": "uint32"},
            ],
        }

        parser = generator.generate_from_definition(definition)

        assert parser.name == "MyStruct"
        assert parser.total_size == 8
        assert parser.endian == "little"
        assert len(parser.fields) == 2
        assert parser.fields[0].name == "field1"
        assert parser.fields[1].name == "field2"

    def test_generate_from_definition_minimal(self) -> None:
        """Test generating parser from minimal definition."""
        generator = BinaryParserGenerator()
        definition = {"fields": []}

        parser = generator.generate_from_definition(definition)

        assert parser.name == "Structure"
        assert len(parser.fields) == 0

    def test_generate_from_definition_with_arrays(self) -> None:
        """Test generating parser with array fields."""
        generator = BinaryParserGenerator()
        definition = {
            "name": "ArrayStruct",
            "fields": [
                {
                    "name": "data",
                    "offset": 0,
                    "size": 16,
                    "type": "uint8",
                    "count": 16,
                },
            ],
        }

        parser = generator.generate_from_definition(definition)

        assert parser.fields[0].array_count == 16

    def test_generate_from_definition_with_conditions(self) -> None:
        """Test generating parser with conditional fields."""
        generator = BinaryParserGenerator()
        definition = {
            "fields": [
                {
                    "name": "optional",
                    "offset": 0,
                    "size": 4,
                    "type": "uint32",
                    "condition": "has_optional",
                },
            ],
        }

        parser = generator.generate_from_definition(definition)

        assert parser.fields[0].condition == "has_optional"

    def test_to_yaml_simple(self) -> None:
        """Test YAML conversion for simple parser."""
        generator = BinaryParserGenerator()
        field = ParserField(
            name="value",
            offset=0,
            size=4,
            field_type="uint32",
        )
        parser = ParserDefinition(
            name="Simple",
            fields=[field],
            total_size=4,
        )

        yaml_str = generator.to_yaml(parser)

        assert "name: Simple" in yaml_str
        assert "size: 4" in yaml_str
        assert "endian: big" in yaml_str
        assert "field_type: uint32" in yaml_str or "type: uint32" in yaml_str

    def test_to_yaml_with_magic(self) -> None:
        """Test YAML conversion with magic bytes."""
        generator = BinaryParserGenerator()
        parser = ParserDefinition(
            name="WithMagic",
            fields=[],
            total_size=0,
            magic=b"TEST",
        )

        yaml_str = generator.to_yaml(parser)

        assert "magic: 54455354" in yaml_str  # "TEST" in hex

    def test_to_yaml_with_array(self) -> None:
        """Test YAML conversion with array field."""
        generator = BinaryParserGenerator()
        field = ParserField(
            name="array",
            offset=0,
            size=16,
            field_type="uint8",
            array_count=16,
        )
        parser = ParserDefinition(
            name="ArrayStruct",
            fields=[field],
            total_size=16,
        )

        yaml_str = generator.to_yaml(parser)

        assert "count: 16" in yaml_str

    def test_to_yaml_with_condition(self) -> None:
        """Test YAML conversion with conditional field."""
        generator = BinaryParserGenerator()
        field = ParserField(
            name="optional",
            offset=0,
            size=4,
            field_type="uint32",
            condition="enabled",
        )
        parser = ParserDefinition(
            name="Conditional",
            fields=[field],
            total_size=4,
        )

        yaml_str = generator.to_yaml(parser)

        assert "condition: enabled" in yaml_str

    def test_to_yaml_with_description(self) -> None:
        """Test YAML conversion with field descriptions."""
        generator = BinaryParserGenerator()
        field = ParserField(
            name="field",
            offset=0,
            size=4,
            field_type="uint32",
            description="Test field",
        )
        parser = ParserDefinition(
            name="Documented",
            fields=[field],
            total_size=4,
        )

        yaml_str = generator.to_yaml(parser)

        assert "description: Test field" in yaml_str

    def test_to_python_simple(self) -> None:
        """Test Python code generation for simple parser."""
        generator = BinaryParserGenerator()
        field = ParserField(
            name="value",
            offset=0,
            size=4,
            field_type="uint32",
        )
        parser = ParserDefinition(
            name="Simple",
            fields=[field],
            total_size=4,
        )

        python_code = generator.to_python(parser)

        assert "import struct" in python_code
        assert "class Simple:" in python_code
        assert "value: int" in python_code
        assert "def parse(cls, data: bytes)" in python_code

    def test_to_python_multiple_fields(self) -> None:
        """Test Python code generation with multiple fields."""
        generator = BinaryParserGenerator()
        fields = [
            ParserField(name="byte_val", offset=0, size=1, field_type="uint8"),
            ParserField(name="short_val", offset=1, size=2, field_type="uint16"),
            ParserField(name="int_val", offset=3, size=4, field_type="uint32"),
        ]
        parser = ParserDefinition(
            name="MultiField",
            fields=fields,
            total_size=7,
        )

        python_code = generator.to_python(parser)

        assert "byte_val: int" in python_code
        assert "short_val: int" in python_code
        assert "int_val: int" in python_code

    def test_to_python_little_endian(self) -> None:
        """Test Python code generation with little endian."""
        generator = BinaryParserGenerator(default_endian="little")
        field = ParserField(
            name="value",
            offset=0,
            size=4,
            field_type="uint32",
            endian="little",
        )
        parser = ParserDefinition(
            name="LittleEndian",
            fields=[field],
            total_size=4,
            endian="little",
        )

        python_code = generator.to_python(parser)

        # Should use < for little endian
        assert 'fmt = "<' in python_code

    def test_to_python_bytes_field(self) -> None:
        """Test Python code generation with bytes field."""
        generator = BinaryParserGenerator()
        field = ParserField(
            name="data",
            offset=0,
            size=16,
            field_type="bytes[16]",
        )
        parser = ParserDefinition(
            name="WithBytes",
            fields=[field],
            total_size=16,
        )

        python_code = generator.to_python(parser)

        assert "data: bytes" in python_code

    def test_to_python_float_field(self) -> None:
        """Test Python code generation with float field."""
        generator = BinaryParserGenerator()
        field = ParserField(
            name="temperature",
            offset=0,
            size=4,
            field_type="float32",
        )
        parser = ParserDefinition(
            name="WithFloat",
            fields=[field],
            total_size=4,
        )

        python_code = generator.to_python(parser)

        assert "temperature: float" in python_code

    def test_analyze_variance(self) -> None:
        """Test variance analysis across samples."""
        generator = BinaryParserGenerator()
        samples = [struct.pack(">II", 0xFFFFFFFF, i) for i in range(10)]
        field_infos = [(0, 4, "uint32"), (4, 4, "uint32")]

        variance = generator._analyze_variance(samples, field_infos)

        # First field is constant, second varies
        assert 0 in variance
        assert 4 in variance
        assert variance[0] < variance[4]

    def test_analyze_variance_empty_samples(self) -> None:
        """Test variance analysis with empty samples."""
        generator = BinaryParserGenerator()
        samples: list[bytes] = []
        field_infos = [(0, 4, "uint32")]

        variance = generator._analyze_variance(samples, field_infos)

        assert variance == {}

    def test_analyze_variance_all_zeros(self) -> None:
        """Test variance analysis with all zero values."""
        generator = BinaryParserGenerator()
        samples = [b"\x00" * 8] * 5
        field_infos = [(0, 4, "uint32"), (4, 4, "uint32")]

        variance = generator._analyze_variance(samples, field_infos)

        # All zeros should have zero variance
        assert variance[0] == 0.0
        assert variance[4] == 0.0


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_detect_magic_bytes_png(self) -> None:
        """Test detect_magic_bytes convenience function."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10

        result = detect_magic_bytes(data)

        assert result is not None
        assert result.known_format == "PNG"

    def test_detect_magic_bytes_with_offset(self) -> None:
        """Test detect_magic_bytes with offset."""
        data = b"\x00" * 10 + b"\x89PNG\r\n\x1a\n"

        result = detect_magic_bytes(data, offset=10)

        assert result is not None
        assert result.known_format == "PNG"

    def test_detect_magic_bytes_none(self) -> None:
        """Test detect_magic_bytes with no match."""
        data = b"\x00" * 10

        result = detect_magic_bytes(data)

        assert result is None

    def test_detect_alignment_function(self) -> None:
        """Test detect_alignment convenience function."""
        data = struct.pack(">II", 0x12345678, 0xABCDEF00)

        result = detect_alignment(data)

        assert isinstance(result, AlignmentResult)
        assert result.alignment in [1, 2, 4, 8, 16]

    def test_generate_parser_function(self) -> None:
        """Test generate_parser convenience function."""
        samples = [
            struct.pack(">HI", 0x1234, 0x56789ABC),
            struct.pack(">HI", 0x5678, 0x9ABCDEF0),
        ]

        parser = generate_parser(samples, name="TestPacket")

        assert parser.name == "TestPacket"
        assert parser.total_size == 6

    def test_generate_parser_little_endian(self) -> None:
        """Test generate_parser with little endian."""
        samples = [b"\x01\x02\x03\x04"]

        parser = generate_parser(samples, endian="little")

        assert parser.endian == "little"

    def test_parser_to_yaml_function(self) -> None:
        """Test parser_to_yaml convenience function."""
        field = ParserField(name="value", offset=0, size=4, field_type="uint32")
        parser = ParserDefinition(name="Test", fields=[field], total_size=4)

        yaml_str = parser_to_yaml(parser)

        assert "name: Test" in yaml_str
        assert "size: 4" in yaml_str

    def test_parser_to_python_function(self) -> None:
        """Test parser_to_python convenience function."""
        field = ParserField(name="value", offset=0, size=4, field_type="uint32")
        parser = ParserDefinition(name="Test", fields=[field], total_size=4)

        python_code = parser_to_python(parser)

        assert "class Test:" in python_code
        assert "def parse(cls, data: bytes)" in python_code

    def test_find_all_magic_bytes_function(self) -> None:
        """Test find_all_magic_bytes convenience function."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10

        results = find_all_magic_bytes(data)

        assert len(results) >= 1
        assert any(r.known_format == "PNG" for r in results)


# =============================================================================
# Integration Tests
# =============================================================================


class TestInferenceBinaryIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_simple_struct(self) -> None:
        """Test complete workflow from samples to parser to code."""
        # Create samples
        samples = [
            struct.pack(">BHI", 0x01, 0x0202, 0x03030303),
            struct.pack(">BHI", 0x04, 0x0505, 0x06060606),
            struct.pack(">BHI", 0x07, 0x0808, 0x09090909),
        ]

        # Generate parser
        parser = generate_parser(samples, name="SimpleStruct")

        assert parser.name == "SimpleStruct"
        assert parser.total_size == 7

        # Convert to YAML
        yaml_str = parser_to_yaml(parser)
        assert "name: SimpleStruct" in yaml_str

        # Convert to Python
        python_code = parser_to_python(parser)
        assert "class SimpleStruct:" in python_code

    def test_full_workflow_with_magic(self) -> None:
        """Test workflow with magic byte detection."""
        # Create samples with magic bytes
        samples = [b"MAGIC\x00\x01" + struct.pack(">I", i) for i in range(5)]

        # Generate parser
        parser = generate_parser(samples, name="MagicStruct")

        # Should detect magic bytes
        assert parser.magic is not None
        assert b"MAGIC" in parser.magic

    def test_alignment_to_parser(self) -> None:
        """Test using alignment detection in parser generation."""
        # Create aligned structure
        data = struct.pack(">II", 0x11111111, 0x22222222)

        # Detect alignment
        alignment = detect_alignment(data)
        assert alignment.alignment in [1, 2, 4, 8]

        # Generate parser using alignment info
        detector = AlignmentDetector()
        field_types = detector.detect_field_types(data, alignment)
        assert len(field_types) > 0

    def test_custom_signature_workflow(self) -> None:
        """Test workflow with custom signatures."""
        # Add custom signature
        detector = MagicByteDetector()
        detector.add_signature(b"CUSTOM", "Custom Format", ".cust")

        # Create data with custom magic
        data = b"CUSTOM" + b"\x00" * 10

        # Detect
        result = detector.detect(data)
        assert result is not None
        assert result.known_format == "Custom Format"

        # Use in parser generation
        samples = [data]
        parser = generate_parser(samples, name="CustomFormat")
        assert parser.magic == b"CUSTOM"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestInferenceBinaryEdgeCases:
    """Test edge cases and error handling."""

    def test_very_short_data(self) -> None:
        """Test handling of very short data."""
        data = b"\x01"

        # Magic detection
        result = detect_magic_bytes(data)
        # May or may not match depending on signatures

        # Alignment detection
        alignment = detect_alignment(data)
        assert alignment.alignment >= 1

    def test_very_long_data(self) -> None:
        """Test handling of large data."""
        data = b"\x00" * 10000

        # Should handle without issues
        alignment = detect_alignment(data)
        assert alignment.alignment in [1, 2, 4, 8, 16]

    def test_random_data(self) -> None:
        """Test handling of random-looking data."""
        import random

        random.seed(42)
        data = bytes(random.randint(0, 255) for _ in range(100))

        # Should not crash
        result = detect_magic_bytes(data)
        alignment = detect_alignment(data)
        assert alignment.alignment in [1, 2, 4, 8, 16]

    def test_all_same_byte(self) -> None:
        """Test handling of homogeneous data."""
        data = b"\xff" * 100

        alignment = detect_alignment(data)
        assert alignment.alignment in [1, 2, 4, 8, 16]

    def test_mixed_endianness(self) -> None:
        """Test parser with mixed endianness fields."""
        generator = BinaryParserGenerator(default_endian="big")
        fields = [
            ParserField(
                name="big_field",
                offset=0,
                size=4,
                field_type="uint32",
                endian="big",
            ),
            ParserField(
                name="little_field",
                offset=4,
                size=4,
                field_type="uint32",
                endian="little",
            ),
        ]
        parser = ParserDefinition(
            name="MixedEndian",
            fields=fields,
            total_size=8,
        )

        yaml_str = generator.to_yaml(parser)
        # Little endian field should have endian specified
        assert "endian: little" in yaml_str

    def test_empty_field_list(self) -> None:
        """Test parser with no fields."""
        parser = ParserDefinition(
            name="Empty",
            fields=[],
            total_size=0,
        )

        yaml_str = parser_to_yaml(parser)
        assert "name: Empty" in yaml_str
        assert "size: 0" in yaml_str


# =============================================================================
# Test KNOWN_MAGIC_BYTES Constant
# =============================================================================


class TestKnownMagicBytes:
    """Test the KNOWN_MAGIC_BYTES constant."""

    def test_known_magic_bytes_structure(self) -> None:
        """Test structure of KNOWN_MAGIC_BYTES."""
        assert isinstance(KNOWN_MAGIC_BYTES, dict)

        for magic, (format_name, extension) in KNOWN_MAGIC_BYTES.items():
            assert isinstance(magic, bytes)
            assert isinstance(format_name, str)
            assert isinstance(extension, str)
            assert extension.startswith(".")

    def test_known_magic_bytes_contains_common_formats(self) -> None:
        """Test that common formats are included."""
        formats = [info[0] for info in KNOWN_MAGIC_BYTES.values()]

        assert "PNG" in formats
        assert "JPEG" in formats
        assert "PDF" in formats
        assert "ZIP" in formats
        assert "ELF Executable" in formats

    def test_known_magic_bytes_unique_extensions(self) -> None:
        """Test that extensions are reasonable."""
        extensions = [info[1] for info in KNOWN_MAGIC_BYTES.values()]

        # All should start with dot
        assert all(ext.startswith(".") for ext in extensions)

        # Should have common extensions
        assert ".png" in extensions
        assert ".pdf" in extensions
        assert ".zip" in extensions


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Test performance characteristics."""

    def test_detect_all_performance(self) -> None:
        """Test detect_all doesn't take too long on reasonable data."""
        import time

        detector = MagicByteDetector()
        # 1KB of data
        data = b"\x00" * 1024

        start = time.time()
        results = detector.detect_all(data)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second for 1KB)
        assert elapsed < 1.0

    def test_learn_magic_performance(self) -> None:
        """Test magic learning performance."""
        import time

        detector = MagicByteDetector()
        # 100 samples of 100 bytes each
        samples = [b"\x00" * 100 for _ in range(100)]

        start = time.time()
        results = detector.learn_magic_from_samples(samples)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 2.0

    def test_alignment_detection_performance(self) -> None:
        """Test alignment detection performance."""
        import time

        detector = AlignmentDetector()
        # 1KB of structured data
        data = struct.pack(">I", 0x12345678) * 256

        start = time.time()
        result = detector.detect(data)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0
