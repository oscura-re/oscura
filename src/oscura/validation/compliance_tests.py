"""Compliance test generator for protocol standards validation.

This module provides comprehensive test suite generation for validating protocol
implementations against industry standards (IEEE, ISO, SAE, ANSI). Generates test
vectors for conformance, boundary conditions, and interoperability testing.

Example:
    >>> from oscura.validation import ComplianceTestGenerator, ComplianceConfig
    >>> from oscura.sessions import ProtocolSpec
    >>>
    >>> # Generate IEEE 802.3 Ethernet compliance tests
    >>> config = ComplianceConfig(standard="IEEE_802_3", test_types=["conformance"])
    >>> generator = ComplianceTestGenerator(config)
    >>> suite = generator.generate_suite(protocol_spec)
    >>>
    >>> # Export to pytest
    >>> generator.export_pytest(suite, Path("test_ethernet_compliance.py"))
    >>>
    >>> # Export to JSON test vectors
    >>> generator.export_json(suite, Path("ethernet_test_vectors.json"))

References:
    IEEE 802.3 Ethernet Standard
    SAE J1939 CAN Bus Protocol
    ISO 14229 Unified Diagnostic Services
    ISO 15765 ISO-TP (CAN Transport Protocol)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from oscura.sessions.blackbox import FieldHypothesis, ProtocolSpec


class StandardType(str, Enum):
    """Supported industry standards for compliance testing.

    Attributes:
        IEEE_802_3: Ethernet (CSMA/CD) protocol
        IEEE_1149_1: JTAG boundary scan
        SAE_J1939: CAN-based vehicle network
        ISO_14229: UDS diagnostic protocol
        ISO_15765: ISO-TP CAN transport
        MODBUS: Industrial Modbus RTU/TCP
        PROFINET: Real-time industrial Ethernet
        ETHERCAT: Industrial fieldbus
        MQTT: IoT messaging protocol
        COAP: Constrained application protocol
        LORAWAN: Long-range wide-area network
    """

    IEEE_802_3 = "IEEE_802_3"
    IEEE_1149_1 = "IEEE_1149_1"
    SAE_J1939 = "SAE_J1939"
    ISO_14229 = "ISO_14229"
    ISO_15765 = "ISO_15765"
    MODBUS = "MODBUS"
    PROFINET = "PROFINET"
    ETHERCAT = "ETHERCAT"
    MQTT = "MQTT"
    COAP = "COAP"
    LORAWAN = "LORAWAN"


class TestType(str, Enum):
    """Types of compliance tests to generate.

    Attributes:
        CONFORMANCE: Protocol conformance tests (message structure, timing, sequencing)
        BOUNDARY: Boundary value tests (min/max values, overflow, underflow)
        ERROR_HANDLING: Error handling tests (invalid checksums, malformed messages)
        STATE_MACHINE: State machine coverage tests (all transitions exercised)
        INTEROPERABILITY: Interoperability tests (multi-vendor compatibility)
    """

    CONFORMANCE = "conformance"
    BOUNDARY = "boundary"
    ERROR_HANDLING = "error_handling"
    STATE_MACHINE = "state_machine"
    INTEROPERABILITY = "interoperability"


@dataclass
class ComplianceConfig:
    """Configuration for compliance test generation.

    Attributes:
        standard: Target industry standard for compliance.
        test_types: Types of tests to generate.
        num_tests_per_type: Number of tests per test type.
        include_documentation: Include test documentation in output.
        export_format: Export format ("pytest", "json", "pcap", "markdown").
        strict_mode: Enforce strict compliance (no deviations allowed).

    Example:
        >>> config = ComplianceConfig(
        ...     standard=StandardType.SAE_J1939,
        ...     test_types=[TestType.CONFORMANCE, TestType.BOUNDARY],
        ...     num_tests_per_type=50
        ... )
    """

    standard: StandardType | str = StandardType.SAE_J1939
    test_types: list[TestType | str] = field(
        default_factory=lambda: [TestType.CONFORMANCE, TestType.BOUNDARY]
    )
    num_tests_per_type: int = 20
    include_documentation: bool = True
    export_format: Literal["pytest", "json", "pcap", "markdown"] = "pytest"
    strict_mode: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_tests_per_type <= 0:
            raise ValueError(f"num_tests_per_type must be positive, got {self.num_tests_per_type}")

        # Convert string to enum if needed
        if isinstance(self.standard, str):
            try:
                self.standard = StandardType(self.standard)
            except ValueError:
                # Allow custom standards (not in enum)
                pass

        # Convert test type strings to enums
        converted_types: list[TestType | str] = []
        for test_type in self.test_types:
            if isinstance(test_type, TestType):
                converted_types.append(test_type)
            elif isinstance(test_type, str):
                try:
                    converted_types.append(TestType(test_type))
                except ValueError:
                    converted_types.append(test_type)  # Keep custom string
        self.test_types = converted_types

        if self.export_format not in {"pytest", "json", "pcap", "markdown"}:
            raise ValueError(f"Invalid export_format: {self.export_format}")


@dataclass
class TestCase:
    """Individual compliance test case.

    Attributes:
        name: Test case name.
        description: Human-readable description.
        test_type: Type of test (conformance, boundary, etc.).
        input_data: Input data for test (bytes or dict).
        expected_output: Expected output (bytes, dict, or validation result).
        standard_reference: Reference to standard section (e.g., "IEEE 802.3 §4.2.1").
        severity: Test severity ("critical", "high", "medium", "low").
        metadata: Additional test metadata.

    Example:
        >>> test = TestCase(
        ...     name="ethernet_min_frame_size",
        ...     description="Verify minimum Ethernet frame size of 64 bytes",
        ...     test_type=TestType.BOUNDARY,
        ...     input_data=b"\\x00" * 64,
        ...     expected_output={"valid": True},
        ...     standard_reference="IEEE 802.3 §3.2.8",
        ...     severity="critical"
        ... )
    """

    name: str
    description: str
    test_type: TestType | str
    input_data: bytes | dict[str, Any]
    expected_output: bytes | dict[str, Any] | bool | None
    standard_reference: str
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceTestSuite:
    """Complete compliance test suite.

    Attributes:
        standard: Target standard.
        test_cases: List of test cases.
        metadata: Suite metadata (version, date, coverage stats).
        documentation: Test suite documentation.

    Example:
        >>> suite = ComplianceTestSuite(
        ...     standard=StandardType.SAE_J1939,
        ...     test_cases=[test1, test2, test3],
        ...     metadata={"total_tests": 3, "coverage": 85.5}
        ... )
        >>> print(f"Tests: {len(suite.test_cases)}")
    """

    standard: StandardType | str
    test_cases: list[TestCase] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    documentation: str = ""

    @property
    def total_tests(self) -> int:
        """Get total number of test cases."""
        return len(self.test_cases)

    def get_tests_by_type(self, test_type: TestType | str) -> list[TestCase]:
        """Get test cases by type.

        Args:
            test_type: Type of tests to retrieve.

        Returns:
            List of matching test cases.
        """
        return [tc for tc in self.test_cases if tc.test_type == test_type]

    def get_tests_by_severity(
        self, severity: Literal["critical", "high", "medium", "low"]
    ) -> list[TestCase]:
        """Get test cases by severity.

        Args:
            severity: Severity level to filter.

        Returns:
            List of matching test cases.
        """
        return [tc for tc in self.test_cases if tc.severity == severity]


class ComplianceTestGenerator:
    """Compliance test suite generator for protocol standards validation.

    Generates comprehensive test suites from protocol specifications using
    industry standard compliance requirements. Supports IEEE, SAE, ISO, ANSI standards.

    Attributes:
        config: Compliance test configuration.

    Example:
        >>> config = ComplianceConfig(standard=StandardType.IEEE_802_3)
        >>> generator = ComplianceTestGenerator(config)
        >>> suite = generator.generate_suite(protocol_spec)
        >>> generator.export_pytest(suite, Path("test_compliance.py"))
    """

    def __init__(self, config: ComplianceConfig) -> None:
        """Initialize compliance test generator.

        Args:
            config: Compliance test configuration.
        """
        self.config = config
        self._rng = random.Random(42)  # Deterministic for reproducibility

        # Standard-specific constraints
        self._standard_constraints = self._load_standard_constraints()

    def generate_suite(self, spec: ProtocolSpec) -> ComplianceTestSuite:
        """Generate complete compliance test suite from protocol specification.

        Args:
            spec: Protocol specification with field definitions.

        Returns:
            Complete compliance test suite with all test types.

        Example:
            >>> suite = generator.generate_suite(protocol_spec)
            >>> print(f"Generated {suite.total_tests} tests")
            >>> print(f"Conformance: {len(suite.get_tests_by_type('conformance'))}")
        """
        suite = ComplianceTestSuite(
            standard=self.config.standard,
            metadata={
                "protocol_name": spec.name,
                "standard": str(self.config.standard),
                "strict_mode": self.config.strict_mode,
                "test_types": [str(tt) for tt in self.config.test_types],
            },
        )

        # Generate tests for each type
        for test_type in self.config.test_types:
            if test_type == TestType.CONFORMANCE:
                suite.test_cases.extend(self._generate_conformance_tests(spec))
            elif test_type == TestType.BOUNDARY:
                suite.test_cases.extend(self._generate_boundary_tests(spec))
            elif test_type == TestType.ERROR_HANDLING:
                suite.test_cases.extend(self._generate_error_handling_tests(spec))
            elif test_type == TestType.STATE_MACHINE:
                suite.test_cases.extend(self._generate_state_machine_tests(spec))
            elif test_type == TestType.INTEROPERABILITY:
                suite.test_cases.extend(self._generate_interoperability_tests(spec))

        # Generate documentation
        if self.config.include_documentation:
            suite.documentation = self._generate_documentation(suite, spec)

        # Update metadata
        suite.metadata.update(
            {
                "total_tests": suite.total_tests,
                "conformance_tests": len(suite.get_tests_by_type(TestType.CONFORMANCE)),
                "boundary_tests": len(suite.get_tests_by_type(TestType.BOUNDARY)),
                "error_handling_tests": len(suite.get_tests_by_type(TestType.ERROR_HANDLING)),
                "state_machine_tests": len(suite.get_tests_by_type(TestType.STATE_MACHINE)),
                "critical_tests": len(suite.get_tests_by_severity("critical")),
                "high_tests": len(suite.get_tests_by_severity("high")),
            }
        )

        return suite

    def _generate_conformance_tests(self, spec: ProtocolSpec) -> list[TestCase]:
        """Generate protocol conformance tests.

        Tests message structure, field ordering, data types, and protocol sequences
        according to standard specifications.

        Args:
            spec: Protocol specification.

        Returns:
            List of conformance test cases.
        """
        tests: list[TestCase] = []
        constraints = self._standard_constraints.get(str(self.config.standard), {})

        # Test 1: Valid message structure
        for i in range(min(self.config.num_tests_per_type, 10)):
            msg = self._build_valid_message(spec, constraints)
            tests.append(
                TestCase(
                    name=f"{spec.name.lower()}_conformance_valid_{i}",
                    description=f"Valid {spec.name} message conforming to standard",
                    test_type=TestType.CONFORMANCE,
                    input_data=msg,
                    expected_output={"valid": True, "errors": []},
                    standard_reference=constraints.get(
                        "reference", f"{self.config.standard} General"
                    ),
                    severity="critical",
                    metadata={"test_id": f"CONF_{i:03d}"},
                )
            )

        # Test 2: Field ordering compliance
        if len(spec.fields) > 1:
            tests.append(
                TestCase(
                    name=f"{spec.name.lower()}_field_ordering",
                    description="Verify correct field ordering per standard",
                    test_type=TestType.CONFORMANCE,
                    input_data=self._build_valid_message(spec, constraints),
                    expected_output={"field_order_valid": True},
                    standard_reference=constraints.get("reference", "Field Order"),
                    severity="high",
                    metadata={"fields": [f.name for f in spec.fields]},
                )
            )

        # Test 3: Minimum message length
        min_length = sum(f.length for f in spec.fields)
        tests.append(
            TestCase(
                name=f"{spec.name.lower()}_min_message_length",
                description=f"Verify minimum message length of {min_length} bytes",
                test_type=TestType.CONFORMANCE,
                input_data=self._build_valid_message(spec, constraints),
                expected_output={"length": min_length, "valid": True},
                standard_reference=constraints.get("reference", "Message Length"),
                severity="critical",
                metadata={"min_length": min_length},
            )
        )

        return tests

    def _generate_boundary_tests(self, spec: ProtocolSpec) -> list[TestCase]:
        """Generate boundary value tests.

        Tests minimum/maximum values, overflow, underflow, and edge cases
        for all protocol fields.

        Args:
            spec: Protocol specification.

        Returns:
            List of boundary test cases.
        """
        tests: list[TestCase] = []

        # For each field, test min/max values
        for field_idx, field_def in enumerate(spec.fields):
            if field_def.field_type in {"data", "counter"}:
                # Min value (0)
                msg_min = self._build_message_with_field_value(spec, field_idx, 0)
                tests.append(
                    TestCase(
                        name=f"{spec.name.lower()}_field_{field_def.name}_min",
                        description=f"Test minimum value (0) for {field_def.name}",
                        test_type=TestType.BOUNDARY,
                        input_data=msg_min,
                        expected_output={"valid": True, "field_value": 0},
                        standard_reference=f"{self.config.standard} Boundary Values",
                        severity="high",
                        metadata={"field": field_def.name, "boundary": "min"},
                    )
                )

                # Max value
                max_val = (256**field_def.length) - 1
                msg_max = self._build_message_with_field_value(spec, field_idx, max_val)
                tests.append(
                    TestCase(
                        name=f"{spec.name.lower()}_field_{field_def.name}_max",
                        description=f"Test maximum value ({max_val}) for {field_def.name}",
                        test_type=TestType.BOUNDARY,
                        input_data=msg_max,
                        expected_output={"valid": True, "field_value": max_val},
                        standard_reference=f"{self.config.standard} Boundary Values",
                        severity="high",
                        metadata={"field": field_def.name, "boundary": "max"},
                    )
                )

        # All zeros message
        msg_len = sum(f.length for f in spec.fields)
        tests.append(
            TestCase(
                name=f"{spec.name.lower()}_all_zeros",
                description="Test message with all zero bytes",
                test_type=TestType.BOUNDARY,
                input_data=b"\x00" * msg_len,
                expected_output={"valid": self.config.strict_mode is False},
                standard_reference=f"{self.config.standard} Edge Cases",
                severity="medium",
                metadata={"pattern": "all_zeros"},
            )
        )

        # All ones message
        tests.append(
            TestCase(
                name=f"{spec.name.lower()}_all_ones",
                description="Test message with all 0xFF bytes",
                test_type=TestType.BOUNDARY,
                input_data=b"\xff" * msg_len,
                expected_output={"valid": self.config.strict_mode is False},
                standard_reference=f"{self.config.standard} Edge Cases",
                severity="medium",
                metadata={"pattern": "all_ones"},
            )
        )

        return tests

    def _generate_error_handling_tests(self, spec: ProtocolSpec) -> list[TestCase]:
        """Generate error handling tests.

        Tests invalid checksums, malformed messages, and protocol violations.

        Args:
            spec: Protocol specification.

        Returns:
            List of error handling test cases.
        """
        tests: list[TestCase] = []

        # Find checksum field
        checksum_field_idx = None
        checksum_offset = 0
        for idx, field_def in enumerate(spec.fields):
            if field_def.field_type == "checksum":
                checksum_field_idx = idx
                break
            checksum_offset += field_def.length

        # Test 1: Invalid checksum
        if checksum_field_idx is not None:
            msg = self._build_valid_message(spec, {})
            msg_arr = bytearray(msg)
            # Corrupt checksum
            checksum_len = spec.fields[checksum_field_idx].length
            for i in range(checksum_len):
                msg_arr[checksum_offset + i] ^= 0xFF

            tests.append(
                TestCase(
                    name=f"{spec.name.lower()}_invalid_checksum",
                    description="Test message with corrupted checksum",
                    test_type=TestType.ERROR_HANDLING,
                    input_data=bytes(msg_arr),
                    expected_output={"valid": False, "error": "checksum_mismatch"},
                    standard_reference=f"{self.config.standard} Error Detection",
                    severity="critical",
                    metadata={"error_type": "checksum"},
                )
            )

        # Test 2: Truncated message
        msg = self._build_valid_message(spec, {})
        truncated = msg[: len(msg) // 2]
        tests.append(
            TestCase(
                name=f"{spec.name.lower()}_truncated_message",
                description="Test truncated message (incomplete data)",
                test_type=TestType.ERROR_HANDLING,
                input_data=truncated,
                expected_output={"valid": False, "error": "incomplete_message"},
                standard_reference=f"{self.config.standard} Message Format",
                severity="high",
                metadata={"error_type": "truncation"},
            )
        )

        # Test 3: Oversized message
        msg = self._build_valid_message(spec, {})
        oversized = msg + b"\x00" * 10
        tests.append(
            TestCase(
                name=f"{spec.name.lower()}_oversized_message",
                description="Test oversized message (extra bytes)",
                test_type=TestType.ERROR_HANDLING,
                input_data=oversized,
                expected_output={"valid": self.config.strict_mode is False},
                standard_reference=f"{self.config.standard} Message Format",
                severity="medium",
                metadata={"error_type": "oversized"},
            )
        )

        return tests

    def _generate_state_machine_tests(self, spec: ProtocolSpec) -> list[TestCase]:
        """Generate state machine coverage tests.

        Tests all state transitions and verifies protocol state handling.

        Args:
            spec: Protocol specification.

        Returns:
            List of state machine test cases.
        """
        tests: list[TestCase] = []

        # Generate basic state transition tests
        transitions = [
            ("IDLE", "ACTIVE", "initialize"),
            ("ACTIVE", "ERROR", "fault"),
            ("ERROR", "RECOVERY", "reset"),
            ("RECOVERY", "IDLE", "complete"),
        ]

        for from_state, to_state, event in transitions:
            tests.append(
                TestCase(
                    name=f"{spec.name.lower()}_transition_{from_state}_to_{to_state}",
                    description=f"Test {from_state} -> {to_state} transition on {event}",
                    test_type=TestType.STATE_MACHINE,
                    input_data={"state": from_state, "event": event},
                    expected_output={"next_state": to_state, "valid": True},
                    standard_reference=f"{self.config.standard} State Machine",
                    severity="high",
                    metadata={"from": from_state, "to": to_state, "event": event},
                )
            )

        return tests

    def _generate_interoperability_tests(self, spec: ProtocolSpec) -> list[TestCase]:
        """Generate interoperability tests.

        Tests multi-vendor compatibility and protocol variant handling.

        Args:
            spec: Protocol specification.

        Returns:
            List of interoperability test cases.
        """
        tests: list[TestCase] = []

        # Generate tests for different protocol variants/implementations
        variants = ["vendor_a", "vendor_b", "reference_impl"]

        for variant in variants:
            msg = self._build_valid_message(spec, {})
            tests.append(
                TestCase(
                    name=f"{spec.name.lower()}_interop_{variant}",
                    description=f"Test interoperability with {variant} implementation",
                    test_type=TestType.INTEROPERABILITY,
                    input_data=msg,
                    expected_output={"compatible": True, "variant": variant},
                    standard_reference=f"{self.config.standard} Interoperability",
                    severity="medium",
                    metadata={"variant": variant},
                )
            )

        return tests

    def _build_valid_message(self, spec: ProtocolSpec, constraints: dict[str, Any]) -> bytes:
        """Build valid protocol message conforming to standard.

        Args:
            spec: Protocol specification.
            constraints: Standard-specific constraints.

        Returns:
            Valid message bytes.
        """
        msg = bytearray()

        for field_def in spec.fields:
            field_bytes = self._generate_field_value(field_def, constraints)
            msg.extend(field_bytes)

        return bytes(msg)

    def _generate_field_value(
        self, field_def: FieldHypothesis, constraints: dict[str, Any]
    ) -> bytes:
        """Generate value for a single field.

        Args:
            field_def: Field definition.
            constraints: Standard-specific constraints.

        Returns:
            Field value as bytes.
        """
        if field_def.field_type == "constant":
            const_val = field_def.evidence.get("value", 0)
            return self._pack_value(const_val, field_def.length)

        if field_def.field_type == "counter":
            counter_val = self._rng.randint(0, (256**field_def.length) - 1)
            return self._pack_value(counter_val, field_def.length)

        if field_def.field_type == "checksum":
            # Placeholder for checksum (computed later)
            return b"\x00" * field_def.length

        # Default: random data
        return bytes(self._rng.randint(0, 255) for _ in range(field_def.length))

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
        """
        msg = bytearray()

        for idx, field_def in enumerate(spec.fields):
            if idx == field_idx:
                msg.extend(self._pack_value(value, field_def.length))
            else:
                msg.extend(self._generate_field_value(field_def, {}))

        return bytes(msg)

    def _pack_value(self, value: int, length: int) -> bytes:
        """Pack integer value into bytes (little-endian).

        Args:
            value: Integer value.
            length: Number of bytes.

        Returns:
            Packed bytes.
        """
        return value.to_bytes(length, byteorder="little")

    def _load_standard_constraints(self) -> dict[str, Any]:
        """Load standard-specific constraints and requirements.

        Returns:
            Dictionary of constraints by standard name.
        """
        return {
            "StandardType.IEEE_802_3": {
                "min_frame_size": 64,
                "max_frame_size": 1518,
                "reference": "IEEE 802.3 §3.2.8",
            },
            "StandardType.SAE_J1939": {
                "max_pgn": 0x1FFFF,
                "priority_range": (0, 7),
                "reference": "SAE J1939/21",
            },
            "StandardType.ISO_14229": {
                "service_id_range": (0x01, 0xFF),
                "negative_response": 0x7F,
                "reference": "ISO 14229-1:2020",
            },
            "StandardType.MODBUS": {
                "max_address": 247,
                "function_code_range": (1, 127),
                "reference": "Modbus Application Protocol V1.1b3",
            },
        }

    def _generate_documentation(self, suite: ComplianceTestSuite, spec: ProtocolSpec) -> str:
        """Generate comprehensive test suite documentation.

        Args:
            suite: Test suite.
            spec: Protocol specification.

        Returns:
            Markdown documentation.
        """
        doc = [
            f"# Compliance Test Suite: {spec.name}",
            f"\n## Standard: {suite.standard}",
            f"\n**Total Tests:** {suite.total_tests}",
            "\n### Test Coverage",
            "",
        ]

        for test_type in self.config.test_types:
            count = len(suite.get_tests_by_type(test_type))
            doc.append(f"- **{test_type}**: {count} tests")

        doc.extend(
            [
                "\n### Severity Distribution",
                "",
                f"- Critical: {len(suite.get_tests_by_severity('critical'))}",
                f"- High: {len(suite.get_tests_by_severity('high'))}",
                f"- Medium: {len(suite.get_tests_by_severity('medium'))}",
                f"- Low: {len(suite.get_tests_by_severity('low'))}",
                "\n## Test Cases",
                "",
            ]
        )

        for test_case in suite.test_cases[:10]:  # First 10 for brevity
            doc.append(f"### {test_case.name}")
            doc.append(f"\n**Description:** {test_case.description}")
            doc.append(f"**Type:** {test_case.test_type}")
            doc.append(f"**Severity:** {test_case.severity}")
            doc.append(f"**Standard Reference:** {test_case.standard_reference}")
            doc.append("")

        if suite.total_tests > 10:
            doc.append(f"\n*... and {suite.total_tests - 10} more test cases*")

        return "\n".join(doc)

    def export_pytest(self, suite: ComplianceTestSuite, output: Path) -> None:
        """Export compliance tests as pytest parametrized test cases.

        Args:
            suite: Compliance test suite.
            output: Output Python test file path.

        Example:
            >>> generator.export_pytest(suite, Path("test_compliance.py"))
        """
        test_code = [
            f'"""Generated compliance tests for {suite.standard}."""',
            "",
            "import pytest",
            "",
            "",
            "@pytest.mark.parametrize(",
            '    "test_case",',
            "    [",
        ]

        # Add test cases
        for test_case in suite.test_cases:
            # Serialize test case
            test_dict = {
                "name": test_case.name,
                "description": test_case.description,
                "test_type": str(test_case.test_type),
                "input_data": (
                    test_case.input_data.hex()
                    if isinstance(test_case.input_data, bytes)
                    else test_case.input_data
                ),
                "expected_output": test_case.expected_output,
                "standard_reference": test_case.standard_reference,
                "severity": test_case.severity,
            }
            test_code.append(f"        {test_dict!r},")

        test_code.extend(
            [
                "    ],",
                ")",
                "def test_compliance(test_case):",
                f'    """Test compliance with {suite.standard}."""',
                "    # TODO: Implement compliance validation (user should replace with actual validator)",
                '    assert test_case["name"]',
                '    assert test_case["standard_reference"]',
                "",
                "",
                "def test_suite_coverage():",
                f'    """Verify test suite coverage for {suite.standard}."""',
                f"    assert {suite.total_tests} > 0  # Total tests",
            ]
        )

        output.write_text("\n".join(test_code))

    def export_json(self, suite: ComplianceTestSuite, output: Path) -> None:
        """Export compliance tests as JSON test vectors.

        Args:
            suite: Compliance test suite.
            output: Output JSON file path.

        Example:
            >>> generator.export_json(suite, Path("test_vectors.json"))
        """
        json_data = {
            "standard": str(suite.standard),
            "metadata": suite.metadata,
            "documentation": suite.documentation,
            "test_cases": [
                {
                    "name": tc.name,
                    "description": tc.description,
                    "test_type": str(tc.test_type),
                    "input_data": (
                        tc.input_data.hex() if isinstance(tc.input_data, bytes) else tc.input_data
                    ),
                    "expected_output": tc.expected_output,
                    "standard_reference": tc.standard_reference,
                    "severity": tc.severity,
                    "metadata": tc.metadata,
                }
                for tc in suite.test_cases
            ],
        }

        with output.open("w") as f:
            json.dump(json_data, f, indent=2)

    def export_pcap(self, suite: ComplianceTestSuite, output: Path) -> None:
        """Export compliance tests as PCAP file (UDP packets).

        Args:
            suite: Compliance test suite.
            output: Output PCAP file path.

        Example:
            >>> generator.export_pcap(suite, Path("compliance_tests.pcap"))
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
        for test_case in suite.test_cases:
            if isinstance(test_case.input_data, bytes):
                # Wrap in Ethernet/IP/UDP
                pkt = Ether() / IP() / UDP(sport=12345, dport=54321) / test_case.input_data
                packets.append(pkt)

        if packets:
            wrpcap(str(output), packets)

    def export_markdown(self, suite: ComplianceTestSuite, output: Path) -> None:
        """Export compliance tests as Markdown documentation.

        Args:
            suite: Compliance test suite.
            output: Output Markdown file path.

        Example:
            >>> generator.export_markdown(suite, Path("compliance_tests.md"))
        """
        output.write_text(suite.documentation)


__all__ = [
    "ComplianceConfig",
    "ComplianceTestGenerator",
    "ComplianceTestSuite",
    "StandardType",
    "TestCase",
    "TestType",
]
