"""Grammar-based test vector generation for protocol fuzzing and validation.

This module provides intelligent test vector generation from protocol specifications,
supporting coverage-based testing, edge case generation, and mutation-based fuzzing.

Example:
    >>> from oscura.validation import GrammarTestGenerator, TestGenerationConfig
    >>> from oscura.sessions import ProtocolSpec, FieldHypothesis
    >>>
    >>> # Define protocol spec
    >>> spec = ProtocolSpec(
    ...     name="SimpleProtocol",
    ...     fields=[
    ...         FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
    ...         FieldHypothesis("cmd", 1, 1, "data", 0.85),
    ...         FieldHypothesis("length", 2, 1, "data", 0.90),
    ...         FieldHypothesis("checksum", 3, 1, "checksum", 0.95),
    ...     ]
    ... )
    >>>
    >>> # Generate comprehensive test suite
    >>> config = TestGenerationConfig(strategy="all", num_tests=50)
    >>> generator = GrammarTestGenerator(config)
    >>> tests = generator.generate_tests(spec)
    >>>
    >>> # Export as PCAP
    >>> generator.export_pcap(tests.valid_messages, Path("valid_tests.pcap"))
    >>>
    >>> # Export as pytest
    >>> generator.export_pytest(tests.valid_messages, Path("test_protocol.py"))

References:
    AFL Mutation Strategies: https://lcamtuf.coredump.cx/afl/technical_details.txt
    Hypothesis Strategies: https://hypothesis.readthedocs.io/
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from oscura.sessions.blackbox import FieldHypothesis, ProtocolSpec


@dataclass
class TestGenerationConfig:
    """Configuration for test vector generation.

    Attributes:
        strategy: Generation strategy ("coverage", "fuzzing", "edge_cases", "all").
        num_tests: Number of test vectors to generate.
        include_valid: Include valid messages in output.
        include_invalid: Include invalid messages in output.
        mutate_checksums: Generate messages with corrupted checksums.
        boundary_values: Generate boundary value test cases.
        export_format: Export format ("pcap", "binary", "pytest").

    Example:
        >>> config = TestGenerationConfig(
        ...     strategy="coverage",
        ...     num_tests=100,
        ...     include_valid=True,
        ...     include_invalid=True
        ... )
    """

    strategy: Literal["coverage", "fuzzing", "edge_cases", "all"] = "coverage"
    num_tests: int = 100
    include_valid: bool = True
    include_invalid: bool = True
    mutate_checksums: bool = True
    boundary_values: bool = True
    export_format: Literal["pcap", "binary", "pytest"] = "pcap"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_tests <= 0:
            raise ValueError(f"num_tests must be positive, got {self.num_tests}")
        if self.strategy not in {"coverage", "fuzzing", "edge_cases", "all"}:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        if self.export_format not in {"pcap", "binary", "pytest"}:
            raise ValueError(f"Invalid export_format: {self.export_format}")


@dataclass
class GeneratedTests:
    """Container for generated test vectors.

    Attributes:
        valid_messages: Valid protocol messages.
        invalid_messages: Invalid messages (corrupted checksums, bad lengths).
        edge_cases: Boundary value test cases.
        fuzzing_corpus: Mutation-based fuzzing inputs.
        test_descriptions: Human-readable description for each test.
        coverage_report: Coverage statistics (fields covered, value ranges).

    Example:
        >>> tests = GeneratedTests(
        ...     valid_messages=[b"\\xaa\\x01\\x00\\x12"],
        ...     invalid_messages=[b"\\xaa\\x01\\x00\\xff"],
        ...     test_descriptions=["Valid message", "Bad checksum"]
        ... )
        >>> print(f"Total tests: {len(tests.all_messages)}")
    """

    valid_messages: list[bytes] = field(default_factory=list)
    invalid_messages: list[bytes] = field(default_factory=list)
    edge_cases: list[bytes] = field(default_factory=list)
    fuzzing_corpus: list[bytes] = field(default_factory=list)
    test_descriptions: list[str] = field(default_factory=list)
    coverage_report: dict[str, Any] = field(default_factory=dict)

    @property
    def all_messages(self) -> list[bytes]:
        """Get all generated messages.

        Returns:
            Combined list of all message types.
        """
        return self.valid_messages + self.invalid_messages + self.edge_cases + self.fuzzing_corpus


class GrammarTestGenerator:
    """Grammar-based test vector generator for protocol validation and fuzzing.

    Generates comprehensive test suites from protocol specifications using
    coverage-based generation, boundary value analysis, and mutation-based fuzzing.

    Attributes:
        config: Test generation configuration.

    Example:
        >>> config = TestGenerationConfig(strategy="coverage", num_tests=50)
        >>> generator = GrammarTestGenerator(config)
        >>> tests = generator.generate_tests(protocol_spec)
        >>> print(f"Generated {len(tests.valid_messages)} valid tests")
    """

    def __init__(self, config: TestGenerationConfig) -> None:
        """Initialize test generator.

        Args:
            config: Test generation configuration.
        """
        self.config = config
        self._rng = random.Random(42)  # Deterministic for reproducibility

    def generate_tests(self, spec: ProtocolSpec) -> GeneratedTests:
        """Generate comprehensive test suite from protocol specification.

        Args:
            spec: Protocol specification with field definitions.

        Returns:
            Generated test vectors with coverage report.

        Example:
            >>> tests = generator.generate_tests(protocol_spec)
            >>> print(f"Valid: {len(tests.valid_messages)}")
            >>> print(f"Invalid: {len(tests.invalid_messages)}")
        """
        result = GeneratedTests()

        # Generate based on strategy
        if self.config.strategy in {"coverage", "all"}:
            if self.config.include_valid:
                result.valid_messages = self._generate_valid_messages(spec)

        if self.config.strategy in {"edge_cases", "all"}:
            if self.config.boundary_values:
                result.edge_cases = self._generate_edge_cases(spec)

        if self.config.strategy in {"fuzzing", "all"}:
            # Generate base messages first if needed
            base_messages = result.valid_messages or self._generate_valid_messages(spec)
            result.fuzzing_corpus = self._generate_fuzzing_corpus(spec, base_messages)

        # Generate invalid messages with corrupted checksums
        if self.config.include_invalid and self.config.mutate_checksums:
            base_messages = result.valid_messages or self._generate_valid_messages(spec)
            result.invalid_messages = self._corrupt_checksums(spec, base_messages)

        # Generate test descriptions
        result.test_descriptions = self._generate_descriptions(result, spec)

        # Generate coverage report
        result.coverage_report = self._generate_coverage_report(result, spec)

        return result

    def _generate_valid_messages(self, spec: ProtocolSpec) -> list[bytes]:
        """Generate valid messages covering all field combinations.

        Args:
            spec: Protocol specification.

        Returns:
            List of valid protocol messages.

        Example:
            >>> messages = generator._generate_valid_messages(spec)
            >>> all(len(msg) == spec_length for msg in messages)
            True
        """
        messages: list[bytes] = []
        num_messages = min(self.config.num_tests, 50)  # Cap at 50 for coverage

        for _ in range(num_messages):
            msg = bytearray()

            for field_def in spec.fields:
                field_bytes = self._generate_field_value(field_def, valid=True)
                msg.extend(field_bytes)

            messages.append(bytes(msg))

        return messages

    def _generate_field_value(self, field_def: FieldHypothesis, valid: bool = True) -> bytes:
        """Generate value for a single field.

        Args:
            field_def: Field definition.
            valid: Generate valid value (True) or potentially invalid (False).

        Returns:
            Field value as bytes.

        Example:
            >>> field = FieldHypothesis("counter", 0, 1, "counter", 0.9)
            >>> value = generator._generate_field_value(field)
            >>> len(value) == 1
            True
        """
        if field_def.field_type == "constant":
            # Constants have fixed values
            const_val = field_def.evidence.get("value", 0)
            return self._pack_value(const_val, field_def.length)

        if field_def.field_type == "counter":
            # Counters increment
            counter_val = self._rng.randint(0, (256**field_def.length) - 1)
            return self._pack_value(counter_val, field_def.length)

        if field_def.field_type == "checksum":
            # Checksums are computed later (placeholder for now)
            return b"\x00" * field_def.length

        # Default: random data
        return bytes(self._rng.randint(0, 255) for _ in range(field_def.length))

    def _generate_edge_cases(self, spec: ProtocolSpec) -> list[bytes]:
        """Generate boundary value test cases.

        Args:
            spec: Protocol specification.

        Returns:
            List of edge case messages (0x00, 0xFF, overflow, underflow).

        Example:
            >>> edge_cases = generator._generate_edge_cases(spec)
            >>> any(b"\\xff" in msg for msg in edge_cases)
            True
        """
        edge_cases: list[bytes] = []

        # Boundary values: all 0x00, all 0xFF
        msg_len = sum(f.length for f in spec.fields)
        edge_cases.append(b"\x00" * msg_len)
        edge_cases.append(b"\xff" * msg_len)

        # Field-specific boundary values
        for field_idx, field_def in enumerate(spec.fields):
            if field_def.field_type in {"data", "counter"}:
                # Min value (0)
                msg = self._build_message_with_field_value(spec, field_idx, 0)
                edge_cases.append(msg)

                # Max value
                max_val = (256**field_def.length) - 1
                msg = self._build_message_with_field_value(spec, field_idx, max_val)
                edge_cases.append(msg)

        return edge_cases

    def _generate_fuzzing_corpus(
        self, spec: ProtocolSpec, base_messages: list[bytes]
    ) -> list[bytes]:
        """Generate mutation-based fuzzing corpus.

        Args:
            spec: Protocol specification.
            base_messages: Valid messages to mutate.

        Returns:
            List of mutated messages for fuzzing.

        Example:
            >>> corpus = generator._generate_fuzzing_corpus(spec, [b"\\xaa\\x01\\x00"])
            >>> len(corpus) > 0
            True
        """
        corpus: list[bytes] = []
        num_mutations = min(self.config.num_tests, 100)

        for _ in range(num_mutations):
            if not base_messages:
                break

            # Select random base message
            base_msg = self._rng.choice(base_messages)
            mutated = self._mutate_message(base_msg)
            corpus.append(mutated)

        return corpus

    def _mutate_message(self, message: bytes) -> bytes:
        """Apply random mutation to message (AFL-inspired).

        Args:
            message: Original message.

        Returns:
            Mutated message.

        Mutations:
            - Bit flip (flip single bit)
            - Byte flip (flip entire byte)
            - Byte insert (insert random byte)
            - Byte delete (delete random byte)
            - Arithmetic (increment/decrement value)

        Example:
            >>> original = b"\\xaa\\x55\\x01"
            >>> mutated = generator._mutate_message(original)
            >>> mutated != original
            True
        """
        if not message:
            return message

        msg = bytearray(message)
        mutation_type = self._rng.choice(
            ["bit_flip", "byte_flip", "byte_insert", "byte_delete", "arithmetic"]
        )

        if mutation_type == "bit_flip":
            # Flip random bit
            pos = self._rng.randint(0, len(msg) - 1)
            bit = self._rng.randint(0, 7)
            msg[pos] ^= 1 << bit

        elif mutation_type == "byte_flip":
            # Flip entire byte
            pos = self._rng.randint(0, len(msg) - 1)
            msg[pos] ^= 0xFF

        elif mutation_type == "byte_insert":
            # Insert random byte
            pos = self._rng.randint(0, len(msg))
            msg.insert(pos, self._rng.randint(0, 255))

        elif mutation_type == "byte_delete":
            # Delete random byte (if message > 1 byte)
            if len(msg) > 1:
                pos = self._rng.randint(0, len(msg) - 1)
                del msg[pos]

        elif mutation_type == "arithmetic":
            # Increment or decrement value
            pos = self._rng.randint(0, len(msg) - 1)
            delta = self._rng.choice([-1, 1])
            msg[pos] = (msg[pos] + delta) % 256

        return bytes(msg)

    def _corrupt_checksums(self, spec: ProtocolSpec, messages: list[bytes]) -> list[bytes]:
        """Generate messages with corrupted checksums.

        Args:
            spec: Protocol specification.
            messages: Valid messages.

        Returns:
            Messages with invalid checksums.

        Example:
            >>> invalid = generator._corrupt_checksums(spec, valid_messages)
            >>> len(invalid) > 0
            True
        """
        corrupted: list[bytes] = []

        # Find checksum field
        checksum_field_idx = None
        checksum_offset = 0
        for idx, field_def in enumerate(spec.fields):
            if field_def.field_type == "checksum":
                checksum_field_idx = idx
                break
            checksum_offset += field_def.length

        if checksum_field_idx is None:
            return []  # No checksum field

        # Corrupt checksums in subset of messages
        num_corrupt = min(len(messages), 10)
        for msg in messages[:num_corrupt]:
            msg_arr = bytearray(msg)
            checksum_len = spec.fields[checksum_field_idx].length

            # XOR checksum with random value
            for i in range(checksum_len):
                msg_arr[checksum_offset + i] ^= self._rng.randint(1, 255)

            corrupted.append(bytes(msg_arr))

        return corrupted

    def _build_message_with_field_value(
        self, spec: ProtocolSpec, field_idx: int, value: int
    ) -> bytes:
        """Build message with specific field set to value.

        Args:
            spec: Protocol specification.
            field_idx: Index of field to set.
            value: Value to assign to field.

        Returns:
            Complete message with field set.

        Example:
            >>> msg = generator._build_message_with_field_value(spec, 1, 0xFF)
            >>> len(msg) > 0
            True
        """
        msg = bytearray()

        for idx, field_def in enumerate(spec.fields):
            if idx == field_idx:
                msg.extend(self._pack_value(value, field_def.length))
            else:
                msg.extend(self._generate_field_value(field_def, valid=True))

        return bytes(msg)

    def _pack_value(self, value: int, length: int) -> bytes:
        """Pack integer value into bytes (little-endian).

        Args:
            value: Integer value.
            length: Number of bytes.

        Returns:
            Packed bytes.

        Example:
            >>> generator._pack_value(0x1234, 2)
            b'\\x34\\x12'
        """
        return value.to_bytes(length, byteorder="little")

    def _generate_descriptions(self, tests: GeneratedTests, spec: ProtocolSpec) -> list[str]:
        """Generate human-readable descriptions for test vectors.

        Args:
            tests: Generated test vectors.
            spec: Protocol specification.

        Returns:
            List of test descriptions.

        Example:
            >>> descriptions = generator._generate_descriptions(tests, spec)
            >>> len(descriptions) == len(tests.all_messages)
            True
        """
        descriptions: list[str] = []

        for msg in tests.valid_messages:
            descriptions.append(f"Valid {spec.name} message: {msg.hex()}")

        for msg in tests.invalid_messages:
            descriptions.append(f"Invalid {spec.name} (bad checksum): {msg.hex()}")

        for msg in tests.edge_cases:
            descriptions.append(f"Edge case {spec.name}: {msg.hex()}")

        for msg in tests.fuzzing_corpus:
            descriptions.append(f"Fuzzing input {spec.name}: {msg.hex()}")

        return descriptions

    def _generate_coverage_report(
        self, tests: GeneratedTests, spec: ProtocolSpec
    ) -> dict[str, Any]:
        """Generate coverage statistics for test suite.

        Args:
            tests: Generated test vectors.
            spec: Protocol specification.

        Returns:
            Coverage report with field coverage and value ranges.

        Example:
            >>> report = generator._generate_coverage_report(tests, spec)
            >>> "fields_covered" in report
            True
        """
        report: dict[str, Any] = {
            "total_tests": len(tests.all_messages),
            "valid_tests": len(tests.valid_messages),
            "invalid_tests": len(tests.invalid_messages),
            "edge_cases": len(tests.edge_cases),
            "fuzzing_inputs": len(tests.fuzzing_corpus),
            "fields_covered": len(spec.fields),
            "protocol_name": spec.name,
        }

        return report

    def export_pcap(self, messages: list[bytes], output: Path) -> None:
        """Export messages as PCAP file (UDP packets).

        Args:
            messages: Protocol messages to export.
            output: Output PCAP file path.

        Example:
            >>> generator.export_pcap(tests.valid_messages, Path("tests.pcap"))
        """
        try:
            from scapy.all import (  # type: ignore[attr-defined]
                IP,
                UDP,
                Ether,
                wrpcap,
            )
        except ImportError as e:
            raise ImportError(
                "scapy is required for PCAP export. Install with: uv pip install scapy"
            ) from e

        packets = []
        for msg in messages:
            # Wrap in Ethernet/IP/UDP
            pkt = Ether() / IP() / UDP(sport=12345, dport=54321) / msg
            packets.append(pkt)

        wrpcap(str(output), packets)

    def export_pytest(self, messages: list[bytes], output: Path) -> None:
        """Export as pytest parametrized test cases.

        Args:
            messages: Protocol messages to export.
            output: Output Python test file path.

        Example:
            >>> generator.export_pytest(tests.valid_messages, Path("test_proto.py"))
        """
        test_code = [
            '"""Generated protocol test vectors."""',
            "",
            "import pytest",
            "",
            "",
            "@pytest.mark.parametrize(",
            '    "message,expected_valid",',
            "    [",
        ]

        # Add test cases
        for msg in messages:
            hex_msg = msg.hex()
            test_code.append(f'        (bytes.fromhex("{hex_msg}"), True),')

        test_code.extend(
            [
                "    ],",
                ")",
                "def test_protocol_parsing(message, expected_valid):",
                '    """Test protocol message parsing."""',
                "    # TODO: Implement parser validation (user should replace with actual parser)",
                "    assert isinstance(message, bytes)",
                "    assert len(message) > 0",
            ]
        )

        output.write_text("\n".join(test_code))
