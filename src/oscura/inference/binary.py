"""Binary format inference and parser generation.

    - RE-BIN-001: Magic Byte Detection
    - RE-BIN-002: Structure Alignment Detection
    - RE-BIN-003: Binary Parser DSL

This module provides tools for inferring binary file/message formats,
detecting magic bytes and file signatures, analyzing structure alignment,
and generating parser definitions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class MagicByteResult:
    """Result of magic byte detection.

    Implements RE-BIN-001: Magic byte detection result.

    Attributes:
        magic: Detected magic bytes.
        offset: Offset of magic bytes.
        confidence: Detection confidence (0-1).
        frequency: Number of occurrences.
        known_format: Known format name if recognized.
        file_extension: Suggested file extension.
    """

    magic: bytes
    offset: int
    confidence: float
    frequency: int
    known_format: str | None = None
    file_extension: str | None = None


@dataclass
class AlignmentResult:
    """Result of structure alignment detection.

    Implements RE-BIN-002: Alignment detection result.

    Attributes:
        alignment: Detected alignment (1, 2, 4, 8, etc.).
        padding_positions: Positions of detected padding.
        field_boundaries: Detected field boundaries.
        confidence: Detection confidence (0-1).
        structure_size: Estimated structure size.
    """

    alignment: int
    padding_positions: list[int]
    field_boundaries: list[int]
    confidence: float
    structure_size: int | None = None


@dataclass
class ParserField:
    """A field in a binary parser definition.

    Implements RE-BIN-003: Parser field definition.

    Attributes:
        name: Field name.
        offset: Byte offset.
        size: Field size in bytes.
        field_type: Data type (uint8, uint16, etc.).
        endian: Endianness (big or little).
        array_count: Array element count (1 for scalar).
        condition: Conditional expression.
        description: Field description.
    """

    name: str
    offset: int
    size: int
    field_type: str
    endian: Literal["big", "little"] = "big"
    array_count: int = 1
    condition: str | None = None
    description: str = ""


@dataclass
class ParserDefinition:
    """A complete binary parser definition.

    Implements RE-BIN-003: Parser definition.

    Attributes:
        name: Parser/structure name.
        fields: List of field definitions.
        total_size: Total structure size.
        endian: Default endianness.
        magic: Magic bytes if any.
        version: Parser version.
    """

    name: str
    fields: list[ParserField]
    total_size: int
    endian: Literal["big", "little"] = "big"
    magic: bytes | None = None
    version: str = "1.0"


# Known magic bytes database
KNOWN_MAGIC_BYTES: dict[bytes, tuple[str, str]] = {
    # Images
    b"\x89PNG\r\n\x1a\n": ("PNG", ".png"),
    b"\xff\xd8\xff": ("JPEG", ".jpg"),
    b"GIF87a": ("GIF", ".gif"),
    b"GIF89a": ("GIF", ".gif"),
    b"BM": ("BMP", ".bmp"),
    b"RIFF": ("RIFF", ".riff"),
    b"II*\x00": ("TIFF (LE)", ".tiff"),
    b"MM\x00*": ("TIFF (BE)", ".tiff"),
    # Archives
    b"PK\x03\x04": ("ZIP", ".zip"),
    b"\x1f\x8b\x08": ("GZIP", ".gz"),
    b"BZh": ("BZIP2", ".bz2"),
    b"\xfd7zXZ\x00": ("XZ", ".xz"),
    b"Rar!\x1a\x07": ("RAR", ".rar"),
    b"7z\xbc\xaf\x27\x1c": ("7Z", ".7z"),
    # Documents
    b"%PDF": ("PDF", ".pdf"),
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": ("OLE2", ".doc"),
    # Executables
    b"MZ": ("DOS/PE Executable", ".exe"),
    b"\x7fELF": ("ELF Executable", ".elf"),
    b"\xfe\xed\xfa\xce": ("Mach-O (32-bit)", ".macho"),
    b"\xfe\xed\xfa\xcf": ("Mach-O (64-bit)", ".macho"),
    b"\xca\xfe\xba\xbe": ("Java Class", ".class"),
    # Audio/Video
    b"ID3": ("MP3", ".mp3"),
    b"\xff\xfb": ("MP3", ".mp3"),
    b"OggS": ("OGG", ".ogg"),
    b"fLaC": ("FLAC", ".flac"),
    # Database
    b"SQLite format 3": ("SQLite", ".sqlite"),
    # Network
    b"\xd4\xc3\xb2\xa1": ("PCAP (LE)", ".pcap"),
    b"\xa1\xb2\xc3\xd4": ("PCAP (BE)", ".pcap"),
    b"\x0a\x0d\x0d\x0a": ("PCAPNG", ".pcapng"),
}


class MagicByteDetector:
    """Detect magic bytes and file signatures.

    Implements RE-BIN-001: Magic Byte Detection.

    Identifies file format signatures and common protocol headers.

    Example:
        >>> detector = MagicByteDetector()
        >>> result = detector.detect(data)
        >>> print(f"Detected: {result.known_format}")
    """

    def __init__(
        self,
        known_signatures: dict[bytes, tuple[str, str]] | None = None,
        min_magic_length: int = 2,
        max_magic_length: int = 16,
    ) -> None:
        """Initialize magic byte detector.

        Args:
            known_signatures: Dictionary of known magic bytes.
            min_magic_length: Minimum magic byte length.
            max_magic_length: Maximum magic byte length to consider.
        """
        self.known_signatures = known_signatures or KNOWN_MAGIC_BYTES
        self.min_magic_length = min_magic_length
        self.max_magic_length = max_magic_length

    def detect(self, data: bytes, offset: int = 0) -> MagicByteResult | None:
        """Detect magic bytes at offset.

        Implements RE-BIN-001: Magic byte detection.

        Args:
            data: Binary data.
            offset: Offset to check.

        Returns:
            MagicByteResult if magic bytes found, None otherwise.

        Example:
            >>> result = detector.detect(file_data)
            >>> if result:
            ...     print(f"Format: {result.known_format}")
        """
        if offset >= len(data):
            return None

        # Check known signatures
        for magic, (format_name, ext) in self.known_signatures.items():
            if len(data) >= offset + len(magic):
                if data[offset : offset + len(magic)] == magic:
                    return MagicByteResult(
                        magic=magic,
                        offset=offset,
                        confidence=1.0,
                        frequency=1,
                        known_format=format_name,
                        file_extension=ext,
                    )

        return None

    def detect_all(self, data: bytes) -> list[MagicByteResult]:
        """Detect all magic bytes in data.

        Implements RE-BIN-001: Scan for all magic bytes.

        Args:
            data: Binary data.

        Returns:
            List of all detected magic bytes.
        """
        results = []

        for offset in range(len(data)):
            result = self.detect(data, offset)
            if result:
                results.append(result)

        return results

    def learn_magic_from_samples(
        self,
        samples: Sequence[bytes],
        min_frequency: int = 2,
    ) -> list[MagicByteResult]:
        """Learn potential magic bytes from samples.

        Implements RE-BIN-001: Magic byte discovery.

        Args:
            samples: List of binary samples.
            min_frequency: Minimum occurrences to consider.

        Returns:
            List of potential magic byte patterns.
        """
        if not samples:
            return []

        # Collect common prefixes
        prefix_counts: Counter[bytes] = Counter()

        for length in range(self.min_magic_length, self.max_magic_length + 1):
            for sample in samples:
                if len(sample) >= length:
                    prefix = sample[:length]
                    prefix_counts[prefix] += 1

        # Filter by frequency and sort by frequency (desc) then length (desc)
        # This ensures longer magic bytes are preferred when frequencies are equal
        results = []
        for prefix, count in sorted(
            prefix_counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True
        ):
            if count >= min_frequency:
                # Check if known
                known_format = None
                file_ext = None
                if prefix in self.known_signatures:
                    known_format, file_ext = self.known_signatures[prefix]

                confidence = count / len(samples)

                results.append(
                    MagicByteResult(
                        magic=prefix,
                        offset=0,
                        confidence=confidence,
                        frequency=count,
                        known_format=known_format,
                        file_extension=file_ext,
                    )
                )

        return results

    def add_signature(self, magic: bytes, format_name: str, extension: str) -> None:
        """Add a custom signature.

        Args:
            magic: Magic bytes.
            format_name: Format name.
            extension: File extension.
        """
        self.known_signatures[magic] = (format_name, extension)


class AlignmentDetector:
    """Detect structure alignment in binary data.

    Implements RE-BIN-002: Structure Alignment Detection.

    Analyzes binary data to detect natural alignment boundaries
    and padding patterns typical of compiled structures.

    Example:
        >>> detector = AlignmentDetector()
        >>> result = detector.detect(structure_data)
        >>> print(f"Alignment: {result.alignment} bytes")
    """

    def __init__(
        self,
        test_alignments: list[int] | None = None,
        padding_byte: int | None = None,
    ) -> None:
        """Initialize alignment detector.

        Args:
            test_alignments: Alignments to test (default: [1, 2, 4, 8, 16]).
            padding_byte: Expected padding byte (auto-detect if None).
        """
        self.test_alignments = test_alignments or [1, 2, 4, 8, 16]
        self.padding_byte = padding_byte

    def detect(self, data: bytes) -> AlignmentResult:
        """Detect structure alignment.

        Implements RE-BIN-002: Alignment detection workflow.

        Args:
            data: Binary structure data.

        Returns:
            AlignmentResult with detected alignment.

        Example:
            >>> result = detector.detect(struct_data)
            >>> print(f"Fields at: {result.field_boundaries}")
        """
        if not data:
            return AlignmentResult(
                alignment=1,
                padding_positions=[],
                field_boundaries=[],
                confidence=0.0,
            )

        # Detect padding byte
        padding_byte = self._detect_padding_byte(data)

        # Find potential padding positions
        padding_positions = self._find_padding(data, padding_byte)

        # Find field boundaries using entropy transitions
        field_boundaries = self._find_field_boundaries(data)

        # Test each alignment
        best_alignment = 1
        best_score = 0.0

        for alignment in self.test_alignments:
            score = self._score_alignment(data, alignment, padding_positions, field_boundaries)
            if score > best_score:
                best_score = score
                best_alignment = alignment

        # Estimate structure size
        structure_size = self._estimate_structure_size(data, best_alignment)

        return AlignmentResult(
            alignment=best_alignment,
            padding_positions=padding_positions,
            field_boundaries=field_boundaries,
            confidence=best_score,
            structure_size=structure_size,
        )

    def detect_field_types(
        self,
        data: bytes,
        alignment: AlignmentResult,
    ) -> list[tuple[int, int, str]]:
        """Detect field types based on alignment.

        Implements RE-BIN-002: Field type inference.

        Args:
            data: Binary data.
            alignment: Alignment detection result.

        Returns:
            List of (offset, size, type) tuples.
        """
        fields = []
        boundaries = sorted(set([0] + alignment.field_boundaries + [len(data)]))

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            size = end - start

            # Infer type based on size
            field_type = self._infer_type(data[start:end], size)
            fields.append((start, size, field_type))

        return fields

    def _detect_padding_byte(self, data: bytes) -> int:
        """Detect most likely padding byte.

        Args:
            data: Binary data.

        Returns:
            Most likely padding byte value.
        """
        if self.padding_byte is not None:
            return self.padding_byte

        # Common padding bytes: 0x00, 0xFF, 0xCC, 0xAA
        candidates = [0x00, 0xFF, 0xCC, 0xAA]
        counts = {c: data.count(c) for c in candidates}
        return max(counts.keys(), key=lambda x: counts[x])

    def _find_padding(self, data: bytes, padding_byte: int) -> list[int]:
        """Find positions of potential padding.

        Args:
            data: Binary data.
            padding_byte: Padding byte value.

        Returns:
            List of padding positions.
        """
        positions: list[int] = []
        in_padding = False
        padding_start = 0

        for i, byte in enumerate(data):
            if byte == padding_byte:
                if not in_padding:
                    padding_start = i
                    in_padding = True
            else:
                if in_padding:
                    # End of padding region
                    if i - padding_start >= 1:  # At least 1 byte of padding
                        positions.extend(range(padding_start, i))
                    in_padding = False

        return positions

    def _find_field_boundaries(self, data: bytes) -> list[int]:
        """Find field boundaries using entropy analysis.

        Args:
            data: Binary data.

        Returns:
            List of boundary offsets.
        """
        if len(data) < 8:
            return []

        boundaries = []
        window = 4

        for i in range(window, len(data) - window):
            before = data[i - window : i]
            after = data[i : i + window]

            # Check for significant change in byte patterns
            before_unique = len(set(before))
            after_unique = len(set(after))

            if abs(before_unique - after_unique) >= 2:
                boundaries.append(i)

        return boundaries

    def _score_alignment(
        self,
        data: bytes,
        alignment: int,
        padding_positions: list[int],
        field_boundaries: list[int],
    ) -> float:
        """Score how well an alignment fits the data.

        Args:
            data: Binary data.
            alignment: Alignment value to test.
            padding_positions: Detected padding positions.
            field_boundaries: Detected field boundaries.

        Returns:
            Score (0-1) for this alignment.
        """
        if alignment > len(data):
            return 0.0

        score = 0.0
        checks = 0

        # Check if padding falls at alignment boundaries
        for pos in padding_positions:
            checks += 1
            if pos % alignment == alignment - 1:  # Padding before aligned position
                score += 1

        # Check if field boundaries fall at aligned positions
        for pos in field_boundaries:
            checks += 1
            if pos % alignment == 0:
                score += 1

        # Check natural field sizes
        common_sizes = [1, 2, 4, 8]
        for size in common_sizes:
            if alignment >= size and alignment % size == 0:
                score += 0.5

        if checks == 0:
            return 0.5  # No data to score

        return score / (checks + 2)

    def _estimate_structure_size(self, data: bytes, alignment: int) -> int | None:
        """Estimate structure size based on alignment.

        Args:
            data: Binary data.
            alignment: Detected alignment.

        Returns:
            Estimated structure size or None.
        """
        # Structure size is typically aligned
        for size in range(alignment, len(data) + 1, alignment):
            if len(data) % size == 0:
                count = len(data) // size
                if count >= 2:
                    return size

        return None

    def _infer_type(self, data: bytes, size: int) -> str:
        """Infer field type from data.

        Args:
            data: Field data.
            size: Field size.

        Returns:
            Inferred type string.
        """
        if size == 1:
            return "uint8"
        elif size == 2:
            return "uint16"
        elif size == 4:
            # Could be uint32 or float
            return "uint32"
        elif size == 8:
            return "uint64"
        else:
            return f"bytes[{size}]"


class BinaryParserGenerator:
    """Generate binary parser definitions.

    Implements RE-BIN-003: Binary Parser DSL.

    Creates parser definitions from analyzed binary data that can
    be used for decoding similar structures.

    Example:
        >>> generator = BinaryParserGenerator()
        >>> parser = generator.generate(samples, name="MyStruct")
        >>> print(parser.to_yaml())
    """

    def __init__(
        self,
        default_endian: Literal["big", "little"] = "big",
    ) -> None:
        """Initialize parser generator.

        Args:
            default_endian: Default endianness for fields.
        """
        self.default_endian = default_endian

    def generate(
        self,
        samples: Sequence[bytes],
        name: str = "Structure",
    ) -> ParserDefinition:
        """Generate parser definition from samples.

        Implements RE-BIN-003: Parser generation workflow.

        Args:
            samples: Binary data samples.
            name: Structure name.

        Returns:
            ParserDefinition for the data format.

        Example:
            >>> parser = generator.generate(packet_samples, name="Packet")
        """
        if not samples:
            return ParserDefinition(
                name=name,
                fields=[],
                total_size=0,
                endian=self.default_endian,
            )

        # Use first sample as reference
        reference = samples[0]
        total_size = len(reference)

        # Detect magic bytes - try known signatures first
        magic_detector = MagicByteDetector()
        magic_result = magic_detector.detect(reference)

        # If no known signature found, learn from samples
        if magic_result is None and len(samples) > 1:
            learned_magic = magic_detector.learn_magic_from_samples(samples)
            if learned_magic:
                # Use the most confident/frequent magic bytes
                magic_result = learned_magic[0]

        magic = magic_result.magic if magic_result else None

        # Detect alignment
        alignment_detector = AlignmentDetector()
        alignment_result = alignment_detector.detect(reference)

        # Detect field types
        field_infos = alignment_detector.detect_field_types(reference, alignment_result)

        # Analyze field variance across samples
        variance_info = self._analyze_variance(samples, field_infos)

        # Generate field definitions
        fields = []
        for _i, (offset, size, inferred_type) in enumerate(field_infos):
            variance = variance_info.get(offset, 0.0)

            # Name based on type and position
            if variance < 0.01:
                field_name = f"const_{offset}"
            elif inferred_type.startswith("uint"):
                field_name = f"field_{offset}"
            else:
                field_name = f"data_{offset}"

            fields.append(
                ParserField(
                    name=field_name,
                    offset=offset,
                    size=size,
                    field_type=inferred_type,
                    endian=self.default_endian,
                    description=f"Variance: {variance:.2f}",
                )
            )

        return ParserDefinition(
            name=name,
            fields=fields,
            total_size=total_size,
            endian=self.default_endian,
            magic=magic,
        )

    def generate_from_definition(
        self,
        definition: dict[str, Any],
    ) -> ParserDefinition:
        """Generate parser from dictionary definition.

        Implements RE-BIN-003: Parser from specification.

        Args:
            definition: Dictionary with parser specification.

        Returns:
            ParserDefinition object.
        """
        fields = []
        for field_def in definition.get("fields", []):
            fields.append(
                ParserField(
                    name=field_def["name"],
                    offset=field_def.get("offset", 0),
                    size=field_def.get("size", 1),
                    field_type=field_def.get("type", "uint8"),
                    endian=field_def.get("endian", self.default_endian),
                    array_count=field_def.get("count", 1),
                    condition=field_def.get("condition"),
                    description=field_def.get("description", ""),
                )
            )

        return ParserDefinition(
            name=definition.get("name", "Structure"),
            fields=fields,
            total_size=definition.get("size", sum(f.size for f in fields)),
            endian=definition.get("endian", self.default_endian),
            magic=definition.get("magic"),
            version=definition.get("version", "1.0"),
        )

    def to_yaml(self, parser: ParserDefinition) -> str:
        """Convert parser definition to YAML.

        Implements RE-BIN-003: YAML export.

        Args:
            parser: Parser definition.

        Returns:
            YAML string representation.
        """
        lines = [
            f"name: {parser.name}",
            f"version: {parser.version}",
            f"endian: {parser.endian}",
            f"size: {parser.total_size}",
        ]

        if parser.magic:
            lines.append(f"magic: {parser.magic.hex()}")

        lines.append("fields:")
        for field in parser.fields:
            lines.append(f"  - name: {field.name}")
            lines.append(f"    offset: {field.offset}")
            lines.append(f"    size: {field.size}")
            lines.append(f"    type: {field.field_type}")
            if field.endian != parser.endian:
                lines.append(f"    endian: {field.endian}")
            if field.array_count > 1:
                lines.append(f"    count: {field.array_count}")
            if field.condition:
                lines.append(f"    condition: {field.condition}")
            if field.description:
                lines.append(f"    description: {field.description}")

        return "\n".join(lines)

    def to_python(self, parser: ParserDefinition) -> str:
        """Generate Python struct unpacking code.

        Implements RE-BIN-003: Python code generation.

        Args:
            parser: Parser definition.

        Returns:
            Python code string.
        """
        endian_char = ">" if parser.endian == "big" else "<"
        format_chars = {
            "uint8": "B",
            "int8": "b",
            "uint16": "H",
            "int16": "h",
            "uint32": "I",
            "int32": "i",
            "uint64": "Q",
            "int64": "q",
            "float32": "f",
            "float64": "d",
        }

        lines = [
            "import struct",
            "from dataclasses import dataclass",
            "",
            "@dataclass",
            f"class {parser.name}:",
        ]

        # Add fields
        for field in parser.fields:
            if field.field_type.startswith("bytes"):
                py_type = "bytes"
            elif field.field_type in format_chars:
                if "int" in field.field_type:
                    py_type = "int"
                else:
                    py_type = "float"
            else:
                py_type = "int"
            lines.append(f"    {field.name}: {py_type}")

        # Add parse method
        lines.extend(
            [
                "",
                "    @classmethod",
                "    def parse(cls, data: bytes) -> '{parser.name}':",
            ]
        )

        # Generate struct format
        format_parts = []
        field_names = []
        for field in parser.fields:
            if field.field_type.startswith("bytes"):
                size = field.size
                format_parts.append(f"{size}s")
            elif field.field_type in format_chars:
                format_parts.append(format_chars[field.field_type])
            else:
                format_parts.append(f"{field.size}s")
            field_names.append(field.name)

        format_str = endian_char + "".join(format_parts)
        lines.append(f'        fmt = "{format_str}"')
        lines.append(f"        values = struct.unpack(fmt, data[:{parser.total_size}])")
        lines.append(
            f"        return cls({', '.join(f'values[{i}]' for i in range(len(field_names)))})"
        )

        return "\n".join(lines)

    def _analyze_variance(
        self,
        samples: Sequence[bytes],
        field_infos: list[tuple[int, int, str]],
    ) -> dict[int, float]:
        """Analyze field variance across samples.

        Args:
            samples: Binary samples.
            field_infos: List of (offset, size, type) tuples.

        Returns:
            Dictionary mapping offsets to variance scores.
        """
        variance_info = {}

        for offset, size, _ in field_infos:
            values = []
            for sample in samples:
                if offset + size <= len(sample):
                    field_bytes = sample[offset : offset + size]
                    # Convert to integer for comparison
                    value = int.from_bytes(field_bytes, "big")
                    values.append(value)

            if values:
                arr = np.array(values)
                if np.max(arr) > 0:
                    variance = np.std(arr) / np.max(arr)
                else:
                    variance = 0.0
                variance_info[offset] = float(variance)

        return variance_info


# =============================================================================
# Convenience functions
# =============================================================================


def detect_magic_bytes(data: bytes, offset: int = 0) -> MagicByteResult | None:
    """Detect magic bytes at offset.

    Implements RE-BIN-001: Magic Byte Detection.

    Args:
        data: Binary data.
        offset: Offset to check.

    Returns:
        MagicByteResult if detected, None otherwise.

    Example:
        >>> result = detect_magic_bytes(file_data)
        >>> if result:
        ...     print(f"Format: {result.known_format}")
    """
    detector = MagicByteDetector()
    return detector.detect(data, offset)


def detect_alignment(data: bytes) -> AlignmentResult:
    """Detect structure alignment in data.

    Implements RE-BIN-002: Structure Alignment Detection.

    Args:
        data: Binary structure data.

    Returns:
        AlignmentResult with detected alignment.

    Example:
        >>> result = detect_alignment(struct_data)
        >>> print(f"Alignment: {result.alignment} bytes")
    """
    detector = AlignmentDetector()
    return detector.detect(data)


def generate_parser(
    samples: Sequence[bytes],
    name: str = "Structure",
    endian: Literal["big", "little"] = "big",
) -> ParserDefinition:
    """Generate parser definition from samples.

    Implements RE-BIN-003: Binary Parser DSL.

    Args:
        samples: Binary data samples.
        name: Structure name.
        endian: Default endianness.

    Returns:
        ParserDefinition for the data format.

    Example:
        >>> parser = generate_parser(packet_samples, name="Packet")
        >>> print(parser_to_yaml(parser))
    """
    generator = BinaryParserGenerator(default_endian=endian)
    return generator.generate(samples, name)


def parser_to_yaml(parser: ParserDefinition) -> str:
    """Convert parser definition to YAML.

    Implements RE-BIN-003: YAML export.

    Args:
        parser: Parser definition.

    Returns:
        YAML string.
    """
    generator = BinaryParserGenerator()
    return generator.to_yaml(parser)


def parser_to_python(parser: ParserDefinition) -> str:
    """Convert parser definition to Python code.

    Implements RE-BIN-003: Python code generation.

    Args:
        parser: Parser definition.

    Returns:
        Python code string.
    """
    generator = BinaryParserGenerator()
    return generator.to_python(parser)


def find_all_magic_bytes(data: bytes) -> list[MagicByteResult]:
    """Find all magic bytes in data.

    Implements RE-BIN-001: Scan for all signatures.

    Args:
        data: Binary data.

    Returns:
        List of all detected magic bytes.
    """
    detector = MagicByteDetector()
    return detector.detect_all(data)


__all__ = [
    # Constants
    "KNOWN_MAGIC_BYTES",
    "AlignmentDetector",
    "AlignmentResult",
    "BinaryParserGenerator",
    # Classes
    "MagicByteDetector",
    # Data classes
    "MagicByteResult",
    "ParserDefinition",
    "ParserField",
    "detect_alignment",
    # Functions
    "detect_magic_bytes",
    "find_all_magic_bytes",
    "generate_parser",
    "parser_to_python",
    "parser_to_yaml",
]
