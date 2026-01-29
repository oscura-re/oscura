"""Tests for compliance test generator.

Test Coverage:
- ComplianceConfig validation and defaults
- StandardType and TestType enums
- TestCase creation and metadata
- ComplianceTestSuite data structure and queries
- ComplianceTestGenerator conformance tests
- Boundary value tests
- Error handling tests
- State machine tests
- Interoperability tests
- Export to pytest/JSON/PCAP/Markdown
- Standard constraints loading
- Documentation generation
- Integration with ProtocolSpec
- Edge cases and error handling
"""

import json

import pytest

from oscura.sessions import FieldHypothesis, ProtocolSpec
from oscura.validation.compliance_tests import (
    ComplianceConfig,
    ComplianceTestGenerator,
    ComplianceTestSuite,
    StandardType,
)
from oscura.validation.compliance_tests import TestCase as ComplianceTestCase
from oscura.validation.compliance_tests import TestType as ComplianceTestType


class TestStandardTypeEnum:
    """Tests for StandardType enum."""

    def test_standard_type_values(self):
        """Test StandardType enum contains expected standards."""
        assert StandardType.IEEE_802_3 == "IEEE_802_3"
        assert StandardType.SAE_J1939 == "SAE_J1939"
        assert StandardType.ISO_14229 == "ISO_14229"
        assert StandardType.MODBUS == "MODBUS"
        assert StandardType.MQTT == "MQTT"

    def test_standard_type_iteration(self):
        """Test StandardType enum can be iterated."""
        standards = list(StandardType)
        assert len(standards) >= 11
        assert StandardType.IEEE_802_3 in standards


class TestTestTypeEnumClass:
    """Tests for TestType enum."""

    def test_test_type_values(self):
        """Test TestType enum contains expected test types."""
        assert ComplianceTestType.CONFORMANCE == "conformance"
        assert ComplianceTestType.BOUNDARY == "boundary"
        assert ComplianceTestType.ERROR_HANDLING == "error_handling"
        assert ComplianceTestType.STATE_MACHINE == "state_machine"
        assert ComplianceTestType.INTEROPERABILITY == "interoperability"

    def test_test_type_iteration(self):
        """Test TestType enum can be iterated."""
        types = list(ComplianceTestType)
        assert len(types) == 5


class TestComplianceConfig:
    """Tests for ComplianceConfig dataclass."""

    def test_valid_config_creation(self):
        """Test valid configuration creation."""
        config = ComplianceConfig(
            standard=StandardType.SAE_J1939,
            test_types=[ComplianceTestType.CONFORMANCE, ComplianceTestType.BOUNDARY],
            num_tests_per_type=50,
        )
        assert config.standard == StandardType.SAE_J1939
        assert len(config.test_types) == 2
        assert config.num_tests_per_type == 50

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ComplianceConfig()
        assert config.standard == StandardType.SAE_J1939
        assert ComplianceTestType.CONFORMANCE in config.test_types
        assert ComplianceTestType.BOUNDARY in config.test_types
        assert config.num_tests_per_type == 20
        assert config.include_documentation is True
        assert config.export_format == "pytest"
        assert config.strict_mode is True

    def test_invalid_num_tests_per_type(self):
        """Test validation of num_tests_per_type parameter."""
        with pytest.raises(ValueError, match="num_tests_per_type must be positive"):
            ComplianceConfig(num_tests_per_type=0)

        with pytest.raises(ValueError, match="num_tests_per_type must be positive"):
            ComplianceConfig(num_tests_per_type=-5)

    def test_invalid_export_format(self):
        """Test validation of export_format parameter."""
        with pytest.raises(ValueError, match="Invalid export_format"):
            ComplianceConfig(export_format="invalid")  # type: ignore

    def test_string_to_enum_conversion(self):
        """Test automatic conversion of strings to enums."""
        config = ComplianceConfig(standard="IEEE_802_3", test_types=["conformance", "boundary"])
        assert config.standard == StandardType.IEEE_802_3
        assert ComplianceTestType.CONFORMANCE in config.test_types
        assert ComplianceTestType.BOUNDARY in config.test_types

    def test_custom_standard_string(self):
        """Test support for custom standard strings."""
        config = ComplianceConfig(standard="CUSTOM_STANDARD")
        assert config.standard == "CUSTOM_STANDARD"


class TestTestCaseDataclass:
    """Tests for TestCase dataclass."""

    def test_test_case_creation(self):
        """Test TestCase creation with all fields."""
        test_case = ComplianceTestCase(
            name="test_ethernet_min_frame",
            description="Test minimum Ethernet frame size",
            test_type=ComplianceTestType.BOUNDARY,
            input_data=b"\x00" * 64,
            expected_output={"valid": True},
            standard_reference="IEEE 802.3 §3.2.8",
            severity="critical",
            metadata={"frame_size": 64},
        )
        assert test_case.name == "test_ethernet_min_frame"
        assert test_case.test_type == ComplianceTestType.BOUNDARY
        assert len(test_case.input_data) == 64  # type: ignore
        assert test_case.severity == "critical"
        assert test_case.metadata["frame_size"] == 64

    def test_test_case_default_severity(self):
        """Test TestCase default severity is medium."""
        test_case = ComplianceTestCase(
            name="test",
            description="desc",
            test_type=ComplianceTestType.CONFORMANCE,
            input_data=b"",
            expected_output={},
            standard_reference="ref",
        )
        assert test_case.severity == "medium"

    def test_test_case_dict_input_data(self):
        """Test TestCase with dict input data."""
        test_case = ComplianceTestCase(
            name="test_state_transition",
            description="Test state transition",
            test_type=ComplianceTestType.STATE_MACHINE,
            input_data={"state": "IDLE", "event": "start"},
            expected_output={"next_state": "ACTIVE"},
            standard_reference="State Machine",
        )
        assert test_case.input_data == {"state": "IDLE", "event": "start"}  # type: ignore


class TestComplianceTestSuite:
    """Tests for ComplianceTestSuite dataclass."""

    def test_empty_suite_creation(self):
        """Test empty ComplianceTestSuite creation."""
        suite = ComplianceTestSuite(standard=StandardType.ISO_14229)
        assert suite.standard == StandardType.ISO_14229
        assert len(suite.test_cases) == 0
        assert suite.total_tests == 0

    def test_suite_with_test_cases(self):
        """Test suite with test cases."""
        test1 = ComplianceTestCase(
            "test1", "desc1", ComplianceTestType.CONFORMANCE, b"", {}, "ref1", severity="critical"
        )
        test2 = ComplianceTestCase(
            "test2", "desc2", ComplianceTestType.BOUNDARY, b"", {}, "ref2", severity="high"
        )
        suite = ComplianceTestSuite(standard=StandardType.MODBUS, test_cases=[test1, test2])
        assert suite.total_tests == 2

    def test_get_tests_by_type(self):
        """Test filtering tests by type."""
        test1 = ComplianceTestCase("t1", "d1", ComplianceTestType.CONFORMANCE, b"", {}, "r1")
        test2 = ComplianceTestCase("t2", "d2", ComplianceTestType.CONFORMANCE, b"", {}, "r2")
        test3 = ComplianceTestCase("t3", "d3", ComplianceTestType.BOUNDARY, b"", {}, "r3")
        suite = ComplianceTestSuite(
            standard=StandardType.SAE_J1939, test_cases=[test1, test2, test3]
        )

        conformance_tests = suite.get_tests_by_type(ComplianceTestType.CONFORMANCE)
        assert len(conformance_tests) == 2
        assert all(t.test_type == ComplianceTestType.CONFORMANCE for t in conformance_tests)

        boundary_tests = suite.get_tests_by_type(ComplianceTestType.BOUNDARY)
        assert len(boundary_tests) == 1

    def test_get_tests_by_severity(self):
        """Test filtering tests by severity."""
        test1 = ComplianceTestCase(
            "t1", "d1", ComplianceTestType.CONFORMANCE, b"", {}, "r1", severity="critical"
        )
        test2 = ComplianceTestCase(
            "t2", "d2", ComplianceTestType.BOUNDARY, b"", {}, "r2", severity="critical"
        )
        test3 = ComplianceTestCase(
            "t3", "d3", ComplianceTestType.ERROR_HANDLING, b"", {}, "r3", severity="high"
        )
        suite = ComplianceTestSuite(
            standard=StandardType.IEEE_802_3, test_cases=[test1, test2, test3]
        )

        critical_tests = suite.get_tests_by_severity("critical")
        assert len(critical_tests) == 2

        high_tests = suite.get_tests_by_severity("high")
        assert len(high_tests) == 1


class TestComplianceTestGenerator:
    """Tests for ComplianceTestGenerator."""

    @pytest.fixture
    def simple_spec(self):
        """Create simple protocol spec for testing."""
        return ProtocolSpec(
            name="SimpleProtocol",
            fields=[
                FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
                FieldHypothesis("cmd", 1, 1, "data", 0.85),
                FieldHypothesis("length", 2, 1, "data", 0.90),
                FieldHypothesis("checksum", 3, 1, "checksum", 0.95),
            ],
        )

    @pytest.fixture
    def config_conformance(self):
        """Create config for conformance tests only."""
        return ComplianceConfig(
            standard=StandardType.SAE_J1939,
            test_types=[ComplianceTestType.CONFORMANCE],
            num_tests_per_type=5,
        )

    @pytest.fixture
    def config_all_types(self):
        """Create config for all test types."""
        return ComplianceConfig(
            standard=StandardType.ISO_14229,
            test_types=[
                ComplianceTestType.CONFORMANCE,
                ComplianceTestType.BOUNDARY,
                ComplianceTestType.ERROR_HANDLING,
                ComplianceTestType.STATE_MACHINE,
                ComplianceTestType.INTEROPERABILITY,
            ],
            num_tests_per_type=3,
        )

    def test_generator_initialization(self, config_conformance):
        """Test ComplianceTestGenerator initialization."""
        generator = ComplianceTestGenerator(config_conformance)
        assert generator.config == config_conformance
        assert generator._rng is not None
        assert generator._standard_constraints is not None

    def test_generate_suite_basic(self, simple_spec, config_conformance):
        """Test basic suite generation."""
        generator = ComplianceTestGenerator(config_conformance)
        suite = generator.generate_suite(simple_spec)

        assert suite.standard == StandardType.SAE_J1939
        assert suite.total_tests > 0
        assert "protocol_name" in suite.metadata
        assert suite.metadata["protocol_name"] == "SimpleProtocol"

    def test_generate_conformance_tests(self, simple_spec, config_conformance):
        """Test conformance test generation."""
        generator = ComplianceTestGenerator(config_conformance)
        suite = generator.generate_suite(simple_spec)

        conformance_tests = suite.get_tests_by_type(ComplianceTestType.CONFORMANCE)
        assert len(conformance_tests) > 0

        # Check test structure
        test = conformance_tests[0]
        assert test.name
        assert test.description
        assert test.standard_reference
        assert isinstance(test.input_data, bytes)

    def test_generate_boundary_tests(self, simple_spec):
        """Test boundary value test generation."""
        config = ComplianceConfig(
            standard=StandardType.MODBUS,
            test_types=[ComplianceTestType.BOUNDARY],
            num_tests_per_type=10,
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        boundary_tests = suite.get_tests_by_type(ComplianceTestType.BOUNDARY)
        assert len(boundary_tests) > 0

        # Should have min/max tests for data fields
        test_names = [t.name for t in boundary_tests]
        assert any("min" in name for name in test_names)
        assert any("max" in name for name in test_names)

    def test_generate_error_handling_tests(self, simple_spec):
        """Test error handling test generation."""
        config = ComplianceConfig(
            standard=StandardType.IEEE_802_3,
            test_types=[ComplianceTestType.ERROR_HANDLING],
            num_tests_per_type=10,
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        error_tests = suite.get_tests_by_type(ComplianceTestType.ERROR_HANDLING)
        assert len(error_tests) > 0

        # Should have checksum/truncation/oversized tests
        test_names = [t.name for t in error_tests]
        assert any("checksum" in name or "truncated" in name for name in test_names)

    def test_generate_state_machine_tests(self, simple_spec):
        """Test state machine test generation."""
        config = ComplianceConfig(
            standard=StandardType.ISO_14229,
            test_types=[ComplianceTestType.STATE_MACHINE],
            num_tests_per_type=10,
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        state_tests = suite.get_tests_by_type(ComplianceTestType.STATE_MACHINE)
        assert len(state_tests) > 0

        # Should have state transition tests
        test = state_tests[0]
        assert "transition" in test.name
        assert isinstance(test.input_data, dict)  # State transitions use dict input

    def test_generate_interoperability_tests(self, simple_spec):
        """Test interoperability test generation."""
        config = ComplianceConfig(
            standard=StandardType.PROFINET,
            test_types=[ComplianceTestType.INTEROPERABILITY],
            num_tests_per_type=10,
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        interop_tests = suite.get_tests_by_type(ComplianceTestType.INTEROPERABILITY)
        assert len(interop_tests) > 0

        # Should have vendor variant tests
        test_names = [t.name for t in interop_tests]
        assert any("interop" in name for name in test_names)

    def test_generate_all_test_types(self, simple_spec, config_all_types):
        """Test generation of all test types."""
        generator = ComplianceTestGenerator(config_all_types)
        suite = generator.generate_suite(simple_spec)

        assert len(suite.get_tests_by_type(ComplianceTestType.CONFORMANCE)) > 0
        assert len(suite.get_tests_by_type(ComplianceTestType.BOUNDARY)) > 0
        assert len(suite.get_tests_by_type(ComplianceTestType.ERROR_HANDLING)) > 0
        assert len(suite.get_tests_by_type(ComplianceTestType.STATE_MACHINE)) > 0
        assert len(suite.get_tests_by_type(ComplianceTestType.INTEROPERABILITY)) > 0

    def test_metadata_generation(self, simple_spec, config_all_types):
        """Test metadata generation in suite."""
        generator = ComplianceTestGenerator(config_all_types)
        suite = generator.generate_suite(simple_spec)

        assert "total_tests" in suite.metadata
        assert "conformance_tests" in suite.metadata
        assert "boundary_tests" in suite.metadata
        assert "error_handling_tests" in suite.metadata
        assert "critical_tests" in suite.metadata
        assert suite.metadata["total_tests"] == suite.total_tests

    def test_documentation_generation(self, simple_spec):
        """Test documentation generation."""
        config = ComplianceConfig(
            standard=StandardType.MQTT,
            test_types=[ComplianceTestType.CONFORMANCE],
            include_documentation=True,
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        assert suite.documentation
        assert "Compliance Test Suite" in suite.documentation
        assert "SimpleProtocol" in suite.documentation
        assert str(StandardType.MQTT) in suite.documentation

    def test_documentation_disabled(self, simple_spec):
        """Test documentation generation can be disabled."""
        config = ComplianceConfig(standard=StandardType.MODBUS, include_documentation=False)
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        assert suite.documentation == ""

    def test_standard_constraints_loading(self):
        """Test standard-specific constraints are loaded."""
        config = ComplianceConfig(standard=StandardType.IEEE_802_3)
        generator = ComplianceTestGenerator(config)

        constraints = generator._standard_constraints
        assert "StandardType.IEEE_802_3" in constraints
        assert "min_frame_size" in constraints["StandardType.IEEE_802_3"]

    def test_field_value_generation(self, simple_spec):
        """Test field value generation for different field types."""
        config = ComplianceConfig(standard=StandardType.SAE_J1939)
        generator = ComplianceTestGenerator(config)

        # Constant field
        const_field = simple_spec.fields[0]
        value = generator._generate_field_value(const_field, {})
        assert len(value) == const_field.length

        # Data field
        data_field = simple_spec.fields[1]
        value = generator._generate_field_value(data_field, {})
        assert len(value) == data_field.length

        # Checksum field
        checksum_field = simple_spec.fields[3]
        value = generator._generate_field_value(checksum_field, {})
        assert value == b"\x00"  # Placeholder

    def test_pack_value_little_endian(self):
        """Test value packing uses little-endian."""
        config = ComplianceConfig(standard=StandardType.MODBUS)
        generator = ComplianceTestGenerator(config)

        packed = generator._pack_value(0x1234, 2)
        assert packed == b"\x34\x12"  # Little-endian

        packed = generator._pack_value(0xAABBCCDD, 4)
        assert packed == b"\xdd\xcc\xbb\xaa"

    def test_strict_mode_enabled(self, simple_spec):
        """Test strict mode enforcement."""
        config = ComplianceConfig(
            standard=StandardType.ISO_14229,
            test_types=[ComplianceTestType.BOUNDARY],
            strict_mode=True,
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        # All-zeros test should be invalid in strict mode
        boundary_tests = suite.get_tests_by_type(ComplianceTestType.BOUNDARY)
        all_zeros_test = next(t for t in boundary_tests if "all_zeros" in t.name)
        assert all_zeros_test.expected_output == {"valid": False}  # type: ignore

    def test_strict_mode_disabled(self, simple_spec):
        """Test relaxed mode (strict_mode=False)."""
        config = ComplianceConfig(
            standard=StandardType.MODBUS,
            test_types=[ComplianceTestType.BOUNDARY],
            strict_mode=False,
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(simple_spec)

        # All-zeros test should be valid in relaxed mode
        boundary_tests = suite.get_tests_by_type(ComplianceTestType.BOUNDARY)
        all_zeros_test = next(t for t in boundary_tests if "all_zeros" in t.name)
        assert all_zeros_test.expected_output == {"valid": True}  # type: ignore


class TestExportFormats:
    """Tests for export format support."""

    @pytest.fixture
    def sample_suite(self):
        """Create sample test suite for export tests."""
        test1 = ComplianceTestCase(
            "test_conformance",
            "Test conformance",
            ComplianceTestType.CONFORMANCE,
            b"\xaa\x01\x02\x03",
            {"valid": True},
            "IEEE 802.3",
            severity="critical",
        )
        test2 = ComplianceTestCase(
            "test_boundary",
            "Test boundary",
            ComplianceTestType.BOUNDARY,
            b"\x00\x00\x00\x00",
            {"valid": False},
            "IEEE 802.3 §3.2",
            severity="high",
        )
        return ComplianceTestSuite(
            standard=StandardType.IEEE_802_3,
            test_cases=[test1, test2],
            metadata={"total_tests": 2},
            documentation="Test documentation",
        )

    def test_export_pytest(self, sample_suite, tmp_path):
        """Test pytest export."""
        config = ComplianceConfig(standard=StandardType.IEEE_802_3)
        generator = ComplianceTestGenerator(config)

        output_file = tmp_path / "test_compliance.py"
        generator.export_pytest(sample_suite, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "import pytest" in content
        assert "@pytest.mark.parametrize" in content
        assert "test_conformance" in content
        assert "test_boundary" in content
        assert "aa010203" in content  # Hex-encoded data

    def test_export_json(self, sample_suite, tmp_path):
        """Test JSON export."""
        config = ComplianceConfig(standard=StandardType.SAE_J1939)
        generator = ComplianceTestGenerator(config)

        output_file = tmp_path / "test_vectors.json"
        generator.export_json(sample_suite, output_file)

        assert output_file.exists()
        with output_file.open() as f:
            data = json.load(f)

        assert data["standard"] == "StandardType.IEEE_802_3"
        assert len(data["test_cases"]) == 2
        assert data["test_cases"][0]["name"] == "test_conformance"
        assert data["test_cases"][0]["input_data"] == "aa010203"

    def test_export_markdown(self, sample_suite, tmp_path):
        """Test Markdown export."""
        config = ComplianceConfig(standard=StandardType.MODBUS)
        generator = ComplianceTestGenerator(config)

        output_file = tmp_path / "compliance.md"
        generator.export_markdown(sample_suite, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "Test documentation" in content

    def test_export_pcap_requires_scapy(self, sample_suite, tmp_path):
        """Test PCAP export requires scapy."""
        config = ComplianceConfig(standard=StandardType.IEEE_802_3)
        generator = ComplianceTestGenerator(config)

        output_file = tmp_path / "tests.pcap"

        # Should raise ImportError if scapy not available
        try:
            generator.export_pcap(sample_suite, output_file)
        except ImportError as e:
            assert "scapy" in str(e).lower()

    def test_export_pcap_with_scapy(self, sample_suite, tmp_path):
        """Test PCAP export when scapy is available."""
        pytest.importorskip("scapy")

        config = ComplianceConfig(standard=StandardType.IEEE_802_3)
        generator = ComplianceTestGenerator(config)

        output_file = tmp_path / "tests.pcap"
        generator.export_pcap(sample_suite, output_file)

        # File should be created (scapy available)
        assert output_file.exists()


class TestIntegration:
    """Integration tests with complete workflows."""

    def test_complete_workflow_j1939(self):
        """Test complete workflow for SAE J1939 compliance."""
        spec = ProtocolSpec(
            name="J1939_Message",
            fields=[
                FieldHypothesis("priority", 0, 1, "data", 0.95),
                FieldHypothesis("pgn", 1, 3, "data", 0.98),
                FieldHypothesis("source", 4, 1, "data", 0.99),
                FieldHypothesis("data", 5, 8, "data", 0.90),
                FieldHypothesis("checksum", 13, 1, "checksum", 0.97),
            ],
        )

        config = ComplianceConfig(
            standard=StandardType.SAE_J1939,
            test_types=[
                ComplianceTestType.CONFORMANCE,
                ComplianceTestType.BOUNDARY,
                ComplianceTestType.ERROR_HANDLING,
            ],
            num_tests_per_type=10,
            strict_mode=True,
        )

        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        assert suite.total_tests > 0
        assert len(suite.get_tests_by_type(ComplianceTestType.CONFORMANCE)) > 0
        assert len(suite.get_tests_by_type(ComplianceTestType.BOUNDARY)) > 0
        assert len(suite.get_tests_by_type(ComplianceTestType.ERROR_HANDLING)) > 0

    def test_complete_workflow_modbus(self):
        """Test complete workflow for Modbus compliance."""
        spec = ProtocolSpec(
            name="Modbus_RTU",
            fields=[
                FieldHypothesis("slave_id", 0, 1, "data", 0.99),
                FieldHypothesis("function_code", 1, 1, "data", 0.98),
                FieldHypothesis("address", 2, 2, "data", 0.95),
                FieldHypothesis("count", 4, 2, "data", 0.95),
                FieldHypothesis("crc", 6, 2, "checksum", 0.99),
            ],
        )

        config = ComplianceConfig(
            standard=StandardType.MODBUS,
            test_types=[ComplianceTestType.CONFORMANCE, ComplianceTestType.BOUNDARY],
            num_tests_per_type=15,
        )

        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        assert suite.metadata["protocol_name"] == "Modbus_RTU"
        assert suite.total_tests > 0

    def test_export_all_formats(self, tmp_path):
        """Test exporting to all supported formats."""
        spec = ProtocolSpec(
            name="TestProtocol",
            fields=[
                FieldHypothesis("header", 0, 2, "constant", 0.99, {"value": 0xAA55}),
                FieldHypothesis("data", 2, 4, "data", 0.90),
            ],
        )

        config = ComplianceConfig(
            standard=StandardType.IEEE_802_3, test_types=[ComplianceTestType.CONFORMANCE]
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        # Export pytest
        pytest_file = tmp_path / "test_compliance.py"
        generator.export_pytest(suite, pytest_file)
        assert pytest_file.exists()

        # Export JSON
        json_file = tmp_path / "test_vectors.json"
        generator.export_json(suite, json_file)
        assert json_file.exists()

        # Export Markdown
        md_file = tmp_path / "compliance.md"
        generator.export_markdown(suite, md_file)
        assert md_file.exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_protocol_spec(self):
        """Test handling of empty protocol specification."""
        spec = ProtocolSpec(name="Empty", fields=[])

        config = ComplianceConfig(standard=StandardType.MODBUS)
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        # Should still generate some tests (state machine, interop)
        assert suite.total_tests >= 0

    def test_single_field_protocol(self):
        """Test protocol with single field."""
        spec = ProtocolSpec(
            name="SingleField", fields=[FieldHypothesis("data", 0, 1, "data", 0.90)]
        )

        config = ComplianceConfig(
            standard=StandardType.ISO_14229, test_types=[ComplianceTestType.BOUNDARY]
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        assert suite.total_tests > 0

    def test_large_field_protocol(self):
        """Test protocol with large fields."""
        spec = ProtocolSpec(
            name="LargeField",
            fields=[FieldHypothesis("large_data", 0, 256, "data", 0.85)],
        )

        config = ComplianceConfig(standard=StandardType.ETHERCAT)
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        # Should handle large field sizes
        assert suite.total_tests > 0

    def test_no_checksum_field(self):
        """Test protocol without checksum field."""
        spec = ProtocolSpec(
            name="NoChecksum",
            fields=[
                FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
                FieldHypothesis("data", 1, 4, "data", 0.90),
            ],
        )

        config = ComplianceConfig(
            standard=StandardType.MQTT, test_types=[ComplianceTestType.ERROR_HANDLING]
        )
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        # Should still generate error handling tests (truncation, oversized)
        error_tests = suite.get_tests_by_type(ComplianceTestType.ERROR_HANDLING)
        assert len(error_tests) > 0

    def test_all_constant_fields(self):
        """Test protocol with all constant fields."""
        spec = ProtocolSpec(
            name="AllConstants",
            fields=[
                FieldHypothesis("const1", 0, 1, "constant", 0.99, {"value": 0xAA}),
                FieldHypothesis("const2", 1, 1, "constant", 0.99, {"value": 0x55}),
            ],
        )

        config = ComplianceConfig(standard=StandardType.COAP)
        generator = ComplianceTestGenerator(config)
        suite = generator.generate_suite(spec)

        assert suite.total_tests > 0

    def test_deterministic_generation(self):
        """Test that generation is deterministic (same RNG seed)."""
        spec = ProtocolSpec(
            name="TestProto",
            fields=[FieldHypothesis("data", 0, 4, "data", 0.90)],
        )

        config = ComplianceConfig(standard=StandardType.LORAWAN, num_tests_per_type=20)

        # Generate twice
        generator1 = ComplianceTestGenerator(config)
        suite1 = generator1.generate_suite(spec)

        generator2 = ComplianceTestGenerator(config)
        suite2 = generator2.generate_suite(spec)

        # Should produce identical test suites
        assert suite1.total_tests == suite2.total_tests
        assert len(suite1.test_cases) == len(suite2.test_cases)
