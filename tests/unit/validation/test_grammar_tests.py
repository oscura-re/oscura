"""Tests for grammar-based test vector generation.

Test Coverage:
- GenConfig validation
- Valid message generation with all field types
- Edge case generation (boundary values)
- Fuzzing corpus generation with mutations
- Checksum corruption
- Coverage report generation
- PCAP export
- Pytest export
- Integration with hypothesis
"""

import random

import pytest

from oscura.sessions import FieldHypothesis, ProtocolSpec
from oscura.validation.grammar_tests import (
    GeneratedTests,
    GrammarTestGenerator,
)
from oscura.validation.grammar_tests import (
    TestGenerationConfig as GenConfig,  # Alias to avoid pytest collection confusion
)


class TestConfigValidation:
    """Tests for GenConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = GenConfig(
            strategy="coverage", num_tests=100, include_valid=True, include_invalid=True
        )
        assert config.strategy == "coverage"
        assert config.num_tests == 100
        assert config.include_valid is True

    def test_config_defaults(self):
        """Test default configuration values."""
        config = GenConfig()
        assert config.strategy == "coverage"
        assert config.num_tests == 100
        assert config.include_valid is True
        assert config.include_invalid is True
        assert config.mutate_checksums is True
        assert config.boundary_values is True
        assert config.export_format == "pcap"

    def test_invalid_num_tests(self):
        """Test validation of num_tests parameter."""
        with pytest.raises(ValueError, match="num_tests must be positive"):
            GenConfig(num_tests=0)

        with pytest.raises(ValueError, match="num_tests must be positive"):
            GenConfig(num_tests=-10)

    def test_invalid_strategy(self):
        """Test validation of strategy parameter."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            GenConfig(strategy="invalid")  # type: ignore

    def test_invalid_export_format(self):
        """Test validation of export_format parameter."""
        with pytest.raises(ValueError, match="Invalid export_format"):
            GenConfig(export_format="invalid")  # type: ignore


class TestGeneratedTestsContainer:
    """Tests for GeneratedTests data structure."""

    def test_empty_generated_tests(self):
        """Test empty GeneratedTests container."""
        tests = GeneratedTests()
        assert len(tests.valid_messages) == 0
        assert len(tests.invalid_messages) == 0
        assert len(tests.edge_cases) == 0
        assert len(tests.fuzzing_corpus) == 0
        assert len(tests.all_messages) == 0

    def test_all_messages_property(self):
        """Test all_messages property combines all message types."""
        tests = GeneratedTests(
            valid_messages=[b"\xaa\x01", b"\xaa\x02"],
            invalid_messages=[b"\xaa\xff"],
            edge_cases=[b"\x00\x00", b"\xff\xff"],
            fuzzing_corpus=[b"\xab\x01"],
        )
        assert len(tests.all_messages) == 6
        assert b"\xaa\x01" in tests.all_messages
        assert b"\xff\xff" in tests.all_messages


class TestGrammarTestGenerator:
    """Tests for GrammarTestGenerator."""

    @pytest.fixture
    def simple_spec(self):
        """Create simple protocol specification for testing."""
        return ProtocolSpec(
            name="SimpleProtocol",
            fields=[
                FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
                FieldHypothesis("cmd", 1, 1, "data", 0.85),
                FieldHypothesis("length", 2, 1, "counter", 0.90),
                FieldHypothesis("checksum", 3, 1, "checksum", 0.95),
            ],
        )

    @pytest.fixture
    def multi_byte_spec(self):
        """Create protocol spec with multi-byte fields."""
        return ProtocolSpec(
            name="MultiByteProtocol",
            fields=[
                FieldHypothesis("sync", 0, 2, "constant", 0.99, {"value": 0xAA55}),
                FieldHypothesis("id", 2, 2, "data", 0.85),
                FieldHypothesis("payload", 4, 4, "data", 0.80),
                FieldHypothesis("crc", 8, 2, "checksum", 0.95),
            ],
        )

    def test_generator_initialization(self):
        """Test GrammarTestGenerator initialization."""
        config = GenConfig(strategy="coverage", num_tests=50)
        generator = GrammarTestGenerator(config)
        assert generator.config == config
        assert generator._rng is not None

    def test_generate_valid_messages(self, simple_spec):
        """Test generation of valid protocol messages."""
        config = GenConfig(strategy="coverage", num_tests=10)
        generator = GrammarTestGenerator(config)

        messages = generator._generate_valid_messages(simple_spec)

        assert len(messages) == 10
        assert all(len(msg) == 4 for msg in messages)  # 1+1+1+1 bytes
        assert all(msg[0] == 0xAA for msg in messages)  # Constant header

    def test_generate_field_value_constant(self, simple_spec):
        """Test generation of constant field values."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        field = simple_spec.fields[0]  # header (constant)
        value = generator._generate_field_value(field, valid=True)

        assert len(value) == 1
        assert value[0] == 0xAA

    def test_generate_field_value_counter(self, simple_spec):
        """Test generation of counter field values."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        field = simple_spec.fields[2]  # length (counter)
        value = generator._generate_field_value(field, valid=True)

        assert len(value) == 1
        assert 0 <= value[0] <= 255

    def test_generate_field_value_checksum(self, simple_spec):
        """Test generation of checksum field values (placeholder)."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        field = simple_spec.fields[3]  # checksum
        value = generator._generate_field_value(field, valid=True)

        assert len(value) == 1
        assert value == b"\x00"  # Placeholder

    def test_generate_field_value_data(self, simple_spec):
        """Test generation of random data field values."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        field = simple_spec.fields[1]  # cmd (data)
        value = generator._generate_field_value(field, valid=True)

        assert len(value) == 1
        assert 0 <= value[0] <= 255

    def test_generate_edge_cases(self, simple_spec):
        """Test generation of edge case messages."""
        config = GenConfig(boundary_values=True)
        generator = GrammarTestGenerator(config)

        edge_cases = generator._generate_edge_cases(simple_spec)

        assert len(edge_cases) > 0
        # All zeros and all ones cases
        assert b"\x00\x00\x00\x00" in edge_cases
        assert b"\xff\xff\xff\xff" in edge_cases

    def test_generate_edge_cases_multi_byte_fields(self, multi_byte_spec):
        """Test edge case generation with multi-byte fields."""
        config = GenConfig(boundary_values=True)
        generator = GrammarTestGenerator(config)

        edge_cases = generator._generate_edge_cases(multi_byte_spec)

        assert len(edge_cases) > 0
        # Total message length: 2+2+4+2 = 10 bytes
        assert any(len(msg) == 10 for msg in edge_cases)

    def test_mutate_message_bit_flip(self, simple_spec):
        """Test bit flip mutation."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)
        generator._rng = random.Random(42)

        original = b"\xaa\x55\x01\x00"
        mutated = generator._mutate_message(original)

        # Message should be different (with high probability)
        assert isinstance(mutated, bytes)
        assert len(mutated) >= 3  # May delete or insert bytes

    def test_mutate_message_empty(self):
        """Test mutation of empty message."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        original = b""
        mutated = generator._mutate_message(original)

        assert mutated == b""

    def test_mutate_message_deterministic(self):
        """Test mutation is deterministic with same seed."""
        config = GenConfig()
        gen1 = GrammarTestGenerator(config)
        gen2 = GrammarTestGenerator(config)

        original = b"\xaa\x55\x01\x00"
        mutated1 = gen1._mutate_message(original)
        mutated2 = gen2._mutate_message(original)

        assert mutated1 == mutated2  # Same seed, same mutation

    def test_generate_fuzzing_corpus(self, simple_spec):
        """Test generation of fuzzing corpus."""
        config = GenConfig(strategy="fuzzing", num_tests=20)
        generator = GrammarTestGenerator(config)

        base_messages = [b"\xaa\x01\x00\x12", b"\xaa\x02\x01\x34"]
        corpus = generator._generate_fuzzing_corpus(simple_spec, base_messages)

        assert len(corpus) == 20
        assert all(isinstance(msg, bytes) for msg in corpus)

    def test_generate_fuzzing_corpus_empty_base(self, simple_spec):
        """Test fuzzing corpus generation with empty base messages."""
        config = GenConfig(strategy="fuzzing", num_tests=10)
        generator = GrammarTestGenerator(config)

        corpus = generator._generate_fuzzing_corpus(simple_spec, [])

        assert len(corpus) == 0

    def test_corrupt_checksums(self, simple_spec):
        """Test checksum corruption."""
        config = GenConfig(mutate_checksums=True)
        generator = GrammarTestGenerator(config)

        valid_messages = [b"\xaa\x01\x00\x12", b"\xaa\x02\x01\x34"]
        corrupted = generator._corrupt_checksums(simple_spec, valid_messages)

        assert len(corrupted) > 0
        # Checksums should be different from originals
        for orig, corr in zip(valid_messages[:2], corrupted[:2], strict=True):
            assert orig != corr

    def test_corrupt_checksums_no_checksum_field(self):
        """Test checksum corruption with no checksum field."""
        spec = ProtocolSpec(
            name="NoChecksum",
            fields=[
                FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
                FieldHypothesis("data", 1, 4, "data", 0.85),
            ],
        )
        config = GenConfig(mutate_checksums=True)
        generator = GrammarTestGenerator(config)

        corrupted = generator._corrupt_checksums(spec, [b"\xaa\x01\x02\x03\x04"])

        assert len(corrupted) == 0  # No checksum to corrupt

    def test_build_message_with_field_value(self, simple_spec):
        """Test building message with specific field value."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        msg = generator._build_message_with_field_value(simple_spec, 1, 0xFF)

        assert len(msg) == 4
        assert msg[0] == 0xAA  # Constant header
        assert msg[1] == 0xFF  # Set field value

    def test_pack_value_single_byte(self):
        """Test packing value as single byte."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        packed = generator._pack_value(0x42, 1)

        assert packed == b"\x42"

    def test_pack_value_multi_byte_little_endian(self):
        """Test packing value as multi-byte little-endian."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        packed = generator._pack_value(0x1234, 2)

        assert packed == b"\x34\x12"  # Little-endian

    def test_pack_value_four_bytes(self):
        """Test packing 4-byte value."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        packed = generator._pack_value(0x12345678, 4)

        assert packed == b"\x78\x56\x34\x12"  # Little-endian

    def test_generate_tests_coverage_strategy(self, simple_spec):
        """Test comprehensive test generation with coverage strategy."""
        config = GenConfig(strategy="coverage", num_tests=10)
        generator = GrammarTestGenerator(config)

        tests = generator.generate_tests(simple_spec)

        assert len(tests.valid_messages) == 10
        assert len(tests.coverage_report) > 0
        assert tests.coverage_report["protocol_name"] == "SimpleProtocol"

    def test_generate_tests_edge_cases_strategy(self, simple_spec):
        """Test test generation with edge_cases strategy."""
        config = GenConfig(strategy="edge_cases", num_tests=10)
        generator = GrammarTestGenerator(config)

        tests = generator.generate_tests(simple_spec)

        assert len(tests.edge_cases) > 0
        assert b"\x00\x00\x00\x00" in tests.edge_cases

    def test_generate_tests_fuzzing_strategy(self, simple_spec):
        """Test test generation with fuzzing strategy."""
        config = GenConfig(strategy="fuzzing", num_tests=20, include_valid=False)
        generator = GrammarTestGenerator(config)

        tests = generator.generate_tests(simple_spec)

        assert len(tests.fuzzing_corpus) == 20

    def test_generate_tests_all_strategy(self, simple_spec):
        """Test test generation with 'all' strategy."""
        config = GenConfig(strategy="all", num_tests=10)
        generator = GrammarTestGenerator(config)

        tests = generator.generate_tests(simple_spec)

        assert len(tests.valid_messages) > 0
        assert len(tests.edge_cases) > 0
        assert len(tests.fuzzing_corpus) > 0
        assert len(tests.invalid_messages) > 0  # Corrupted checksums

    def test_generate_tests_no_invalid(self, simple_spec):
        """Test test generation without invalid messages."""
        config = GenConfig(strategy="coverage", num_tests=10, include_invalid=False)
        generator = GrammarTestGenerator(config)

        tests = generator.generate_tests(simple_spec)

        assert len(tests.invalid_messages) == 0

    def test_generate_descriptions(self, simple_spec):
        """Test generation of test descriptions."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        tests = GeneratedTests(
            valid_messages=[b"\xaa\x01\x00\x12"],
            invalid_messages=[b"\xaa\x01\x00\xff"],
            edge_cases=[b"\x00\x00\x00\x00"],
            fuzzing_corpus=[b"\xab\x01\x00\x12"],
        )

        descriptions = generator._generate_descriptions(tests, simple_spec)

        assert len(descriptions) == 4
        assert "Valid SimpleProtocol" in descriptions[0]
        assert "Invalid SimpleProtocol" in descriptions[1]
        assert "Edge case SimpleProtocol" in descriptions[2]
        assert "Fuzzing input SimpleProtocol" in descriptions[3]

    def test_generate_coverage_report(self, simple_spec):
        """Test generation of coverage report."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        tests = GeneratedTests(
            valid_messages=[b"\xaa\x01\x00\x12"] * 10,
            invalid_messages=[b"\xaa\x01\x00\xff"] * 5,
            edge_cases=[b"\x00\x00\x00\x00"] * 3,
            fuzzing_corpus=[b"\xab\x01\x00\x12"] * 20,
        )

        report = generator._generate_coverage_report(tests, simple_spec)

        assert report["total_tests"] == 38
        assert report["valid_tests"] == 10
        assert report["invalid_tests"] == 5
        assert report["edge_cases"] == 3
        assert report["fuzzing_inputs"] == 20
        assert report["fields_covered"] == 4
        assert report["protocol_name"] == "SimpleProtocol"

    def test_export_pytest(self, simple_spec, tmp_path):
        """Test export as pytest parametrized tests."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        messages = [b"\xaa\x01\x00\x12", b"\xaa\x02\x01\x34"]
        output_file = tmp_path / "test_protocol.py"

        generator.export_pytest(messages, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "import pytest" in content
        assert "@pytest.mark.parametrize" in content
        assert "aa01001" in content  # Hex representation
        assert "def test_protocol_parsing" in content

    def test_export_pcap_without_scapy(self, simple_spec, tmp_path, monkeypatch):
        """Test PCAP export raises ImportError when scapy not available."""
        config = GenConfig()
        generator = GrammarTestGenerator(config)

        # Mock scapy import failure
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "scapy.all":
                raise ImportError("No module named 'scapy'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        messages = [b"\xaa\x01\x00\x12"]
        output_file = tmp_path / "test.pcap"

        with pytest.raises(ImportError, match="scapy is required"):
            generator.export_pcap(messages, output_file)

    @pytest.mark.requires_optional
    def test_export_pcap_with_scapy(self, simple_spec, tmp_path):
        """Test PCAP export with scapy installed."""
        pytest.importorskip("scapy")

        config = GenConfig()
        generator = GrammarTestGenerator(config)

        messages = [b"\xaa\x01\x00\x12", b"\xaa\x02\x01\x34"]
        output_file = tmp_path / "test.pcap"

        generator.export_pcap(messages, output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_integration_full_workflow(self, simple_spec, tmp_path):
        """Test complete workflow from spec to exported tests."""
        config = GenConfig(strategy="all", num_tests=10)
        generator = GrammarTestGenerator(config)

        # Generate tests
        tests = generator.generate_tests(simple_spec)

        # Verify all message types generated
        assert len(tests.valid_messages) > 0
        assert len(tests.edge_cases) > 0
        assert len(tests.fuzzing_corpus) > 0
        assert len(tests.invalid_messages) > 0

        # Verify coverage report
        assert tests.coverage_report["total_tests"] > 0
        assert tests.coverage_report["fields_covered"] == 4

        # Verify descriptions
        assert len(tests.test_descriptions) == len(tests.all_messages)

        # Export pytest
        pytest_file = tmp_path / "test_export.py"
        generator.export_pytest(tests.valid_messages, pytest_file)
        assert pytest_file.exists()

    def test_reproducibility(self, simple_spec):
        """Test that generation is reproducible with same seed."""
        config1 = GenConfig(strategy="all", num_tests=10)
        config2 = GenConfig(strategy="all", num_tests=10)

        gen1 = GrammarTestGenerator(config1)
        gen2 = GrammarTestGenerator(config2)

        tests1 = gen1.generate_tests(simple_spec)
        tests2 = gen2.generate_tests(simple_spec)

        # Same seed should produce same results
        assert tests1.valid_messages == tests2.valid_messages
        assert tests1.edge_cases == tests2.edge_cases

    def test_multi_byte_field_handling(self, multi_byte_spec):
        """Test handling of multi-byte fields."""
        config = GenConfig(strategy="coverage", num_tests=5)
        generator = GrammarTestGenerator(config)

        messages = generator._generate_valid_messages(multi_byte_spec)

        assert len(messages) == 5
        # Total length: 2 + 2 + 4 + 2 = 10 bytes
        assert all(len(msg) == 10 for msg in messages)
        # Sync field (constant 0xAA55, little-endian)
        assert all(msg[0] == 0x55 and msg[1] == 0xAA for msg in messages)

    def test_empty_protocol_spec(self):
        """Test handling of empty protocol specification."""
        spec = ProtocolSpec(name="EmptyProtocol", fields=[])
        config = GenConfig(strategy="coverage", num_tests=10)
        generator = GrammarTestGenerator(config)

        messages = generator._generate_valid_messages(spec)

        assert len(messages) == 10
        assert all(len(msg) == 0 for msg in messages)
