"""Comprehensive tests for protocol fuzzer.

Tests cover:
- Fuzzing configuration validation
- Mutation operators (bit flip, byte flip, arithmetic, etc.)
- Corpus management and minimization
- Coverage tracking
- Crash detection
- Report generation
- Export functionality (crashes, corpus, PCAP)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from oscura.validation import (
    FuzzingConfig,
    FuzzingReport,
    FuzzingResult,
    MutationOperator,
    ProtocolFuzzer,
)
from oscura.validation.fuzzer import TestResult as FuzzTestResult

if TYPE_CHECKING:
    from oscura.sessions.blackbox import ProtocolSpec


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_protocol_spec() -> ProtocolSpec:
    """Create simple protocol spec for testing."""
    from oscura.sessions.blackbox import FieldHypothesis, ProtocolSpec

    return ProtocolSpec(
        name="TestProtocol",
        fields=[
            FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA}),
            FieldHypothesis("cmd", 1, 1, "data", 0.85),
            FieldHypothesis("length", 2, 1, "data", 0.90),
            FieldHypothesis("payload", 3, 4, "data", 0.80),
            FieldHypothesis("checksum", 7, 1, "checksum", 0.95),
        ],
    )


@pytest.fixture
def default_config() -> FuzzingConfig:
    """Create default fuzzing configuration."""
    return FuzzingConfig(
        strategy="mutation",
        max_iterations=100,
        crash_detection=True,
        corpus_minimization=False,  # Disable for faster tests
        seed=42,
    )


@pytest.fixture
def fuzzer(default_config: FuzzingConfig) -> ProtocolFuzzer:
    """Create fuzzer instance."""
    return ProtocolFuzzer(default_config)


# =============================================================================
# Configuration Tests
# =============================================================================


def test_config_validation_max_iterations() -> None:
    """Test configuration validation for max_iterations."""
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        FuzzingConfig(max_iterations=0)

    with pytest.raises(ValueError, match="max_iterations must be positive"):
        FuzzingConfig(max_iterations=-1)


def test_config_validation_timeout() -> None:
    """Test configuration validation for timeout_ms."""
    with pytest.raises(ValueError, match="timeout_ms must be positive"):
        FuzzingConfig(timeout_ms=0)


def test_config_validation_strategy() -> None:
    """Test configuration validation for strategy."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        FuzzingConfig(strategy="invalid_strategy")  # type: ignore


def test_config_validation_corpus_size() -> None:
    """Test configuration validation for corpus size."""
    with pytest.raises(ValueError, match="min_corpus_size must be non-negative"):
        FuzzingConfig(min_corpus_size=-1)

    with pytest.raises(ValueError, match="max_corpus_size must be >= min_corpus_size"):
        FuzzingConfig(min_corpus_size=100, max_corpus_size=50)


def test_config_defaults() -> None:
    """Test default configuration values."""
    config = FuzzingConfig()
    assert config.strategy == "coverage_guided"
    assert config.max_iterations == 1000
    assert config.crash_detection is True
    assert config.corpus_minimization is True


# =============================================================================
# Fuzzer Initialization Tests
# =============================================================================


def test_fuzzer_initialization(default_config: FuzzingConfig) -> None:
    """Test fuzzer initialization."""
    fuzzer = ProtocolFuzzer(default_config)
    assert fuzzer.config == default_config
    assert len(fuzzer._corpus) == 0
    assert len(fuzzer._coverage_map) == 0
    assert len(fuzzer._crash_hashes) == 0


def test_fuzzer_seed_reproducibility() -> None:
    """Test that seed produces reproducible results."""
    config1 = FuzzingConfig(max_iterations=10, seed=42)
    config2 = FuzzingConfig(max_iterations=10, seed=42)

    fuzzer1 = ProtocolFuzzer(config1)
    fuzzer2 = ProtocolFuzzer(config2)

    # Generate same sequence of random numbers
    val1 = fuzzer1._rng.randint(0, 1000)
    val2 = fuzzer2._rng.randint(0, 1000)
    assert val1 == val2


# =============================================================================
# Corpus Generation Tests
# =============================================================================


def test_generate_seed_corpus(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test seed corpus generation from protocol spec."""
    corpus = fuzzer._generate_seed_corpus(simple_protocol_spec)

    assert len(corpus) >= 10  # At least min_corpus_size
    assert all(isinstance(msg, bytes) for msg in corpus)

    # Check message length matches spec
    expected_length = sum(f.length for f in simple_protocol_spec.fields)
    assert all(len(msg) == expected_length for msg in corpus)


def test_generate_field_value(fuzzer: ProtocolFuzzer) -> None:
    """Test field value generation."""
    from oscura.sessions.blackbox import FieldHypothesis

    # Constant field
    const_field = FieldHypothesis("header", 0, 1, "constant", 0.99, {"value": 0xAA})
    value = fuzzer._generate_field_value(const_field)
    assert value == b"\xaa"

    # Counter field
    counter_field = FieldHypothesis("counter", 0, 2, "counter", 0.95)
    value = fuzzer._generate_field_value(counter_field)
    assert len(value) == 2
    assert isinstance(value, bytes)

    # Checksum field (placeholder)
    checksum_field = FieldHypothesis("checksum", 0, 1, "checksum", 0.90)
    value = fuzzer._generate_field_value(checksum_field)
    assert value == b"\x00"

    # Data field
    data_field = FieldHypothesis("data", 0, 4, "data", 0.85)
    value = fuzzer._generate_field_value(data_field)
    assert len(value) == 4


# =============================================================================
# Mutation Operator Tests
# =============================================================================


def test_mutation_bit_flip(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test bit flip mutation."""
    original = b"\xaa\x55\x01\x00\x00\x00\x00\x00"
    mutated = fuzzer._apply_mutation(original, MutationOperator.BIT_FLIP, simple_protocol_spec)

    # Should differ by exactly 1 bit
    assert len(mutated) == len(original)
    assert mutated != original

    # Count differing bits
    diff_bits = sum(bin(a ^ b).count("1") for a, b in zip(original, mutated, strict=False))
    assert diff_bits == 1


def test_mutation_byte_flip(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test byte flip mutation."""
    original = b"\xaa\x55\x01\x00\x00\x00\x00\x00"
    mutated = fuzzer._apply_mutation(original, MutationOperator.BYTE_FLIP, simple_protocol_spec)

    assert len(mutated) == len(original)
    assert mutated != original

    # Exactly one byte should be XOR'd with 0xFF
    diff_count = sum(1 for a, b in zip(original, mutated, strict=False) if a != b)
    assert diff_count == 1


def test_mutation_arithmetic(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test arithmetic mutation."""
    original = b"\x10\x20\x30\x40\x50\x60\x70\x80"
    mutated = fuzzer._apply_mutation(original, MutationOperator.ARITHMETIC, simple_protocol_spec)

    assert len(mutated) == len(original)
    # Mutation may or may not change the data (due to modulo 256)


def test_mutation_boundary(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test boundary value mutation."""
    original = b"\x10\x20\x30\x40\x50\x60\x70\x80"
    mutated = fuzzer._apply_mutation(original, MutationOperator.BOUNDARY, simple_protocol_spec)

    assert len(mutated) == len(original)

    # At least one byte should be boundary value
    boundary_values = {0x00, 0xFF, 0x7F, 0x80}
    assert any(b in boundary_values for b in mutated)


def test_mutation_insert(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test byte insertion mutation."""
    original = b"\xaa\x55\x01\x00"
    mutated = fuzzer._apply_mutation(original, MutationOperator.INSERT, simple_protocol_spec)

    assert len(mutated) == len(original) + 1


def test_mutation_delete(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test byte deletion mutation."""
    original = b"\xaa\x55\x01\x00\x00\x00\x00\x00"
    mutated = fuzzer._apply_mutation(original, MutationOperator.DELETE, simple_protocol_spec)

    assert len(mutated) == len(original) - 1


def test_mutation_duplicate(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test region duplication mutation."""
    original = b"\xaa\x55\x01\x02\x03\x04\x05\x06"
    mutated = fuzzer._apply_mutation(original, MutationOperator.DUPLICATE, simple_protocol_spec)

    # Length should increase (duplicated region)
    assert len(mutated) >= len(original)


def test_mutation_swap(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test byte swap mutation."""
    original = b"\xaa\x55\x01\x02\x03\x04\x05\x06"
    mutated = fuzzer._apply_mutation(original, MutationOperator.SWAP, simple_protocol_spec)

    assert len(mutated) == len(original)
    # Two bytes should be swapped
    assert sorted(original) == sorted(mutated)


def test_mutation_checksum_corrupt(
    fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec
) -> None:
    """Test checksum corruption mutation."""
    original = b"\xaa\x01\x04\x00\x00\x00\x00\x12"  # Last byte is checksum
    mutated = fuzzer._apply_mutation(
        original, MutationOperator.CHECKSUM_CORRUPT, simple_protocol_spec
    )

    assert len(mutated) == len(original)
    # Checksum byte (last byte) should be different
    assert mutated[-1] != original[-1]


def test_mutation_length_corrupt(
    fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec
) -> None:
    """Test length field corruption mutation."""
    original = b"\xaa\x01\x04\x00\x00\x00\x00\x12"  # Byte 2 is "length"
    mutated = fuzzer._apply_mutation(
        original, MutationOperator.LENGTH_CORRUPT, simple_protocol_spec
    )

    assert len(mutated) == len(original)
    # Length byte should be corrupted
    # Note: May not change if field name doesn't contain "length"


def test_mutation_empty_input(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test mutation with empty input."""
    original = b""
    mutated = fuzzer._apply_mutation(original, MutationOperator.BIT_FLIP, simple_protocol_spec)
    # Should handle gracefully
    assert isinstance(mutated, bytes)


# =============================================================================
# Fuzzing Execution Tests
# =============================================================================


def test_fuzz_protocol_basic(fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec) -> None:
    """Test basic fuzzing execution without target function."""
    report = fuzzer.fuzz_protocol(simple_protocol_spec, seed_corpus=None, target_function=None)

    assert report.total_iterations == fuzzer.config.max_iterations
    assert report.corpus_size > 0
    assert report.execution_time_seconds >= 0


def test_fuzz_protocol_with_seed_corpus(
    fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec
) -> None:
    """Test fuzzing with provided seed corpus."""
    seed_corpus = [
        b"\xaa\x01\x04\x00\x00\x00\x00\x12",
        b"\xaa\x02\x04\x00\x00\x00\x00\x13",
    ]

    report = fuzzer.fuzz_protocol(
        simple_protocol_spec, seed_corpus=seed_corpus, target_function=None
    )

    assert report.total_iterations == fuzzer.config.max_iterations
    assert len(fuzzer._corpus) >= len(seed_corpus)


def test_fuzz_protocol_with_crashing_parser(
    fuzzer: ProtocolFuzzer, simple_protocol_spec: ProtocolSpec
) -> None:
    """Test fuzzing with parser that crashes on certain inputs."""

    def crashing_parser(data: bytes) -> dict:
        """Parser that crashes on specific pattern."""
        if len(data) > 3 and data[0] == 0xAA and data[2] == 0xFF:
            raise ValueError("Crash condition triggered")
        return {"valid": True}

    report = fuzzer.fuzz_protocol(simple_protocol_spec, None, crashing_parser)

    # Should detect at least some test results (may or may not crash)
    assert report.total_iterations == fuzzer.config.max_iterations


def test_fuzz_protocol_coverage_tracking(simple_protocol_spec: ProtocolSpec) -> None:
    """Test coverage tracking during fuzzing."""
    config = FuzzingConfig(
        strategy="coverage_guided", max_iterations=50, coverage_tracking=True, seed=42
    )
    fuzzer = ProtocolFuzzer(config)

    def simple_parser(data: bytes) -> dict:
        """Simple parser for coverage testing."""
        if len(data) < 4:
            return {"valid": False}
        return {"cmd": data[1], "length": data[2]}

    report = fuzzer.fuzz_protocol(simple_protocol_spec, None, simple_parser)

    # Coverage map should have some entries
    assert len(fuzzer._coverage_map) > 0


# =============================================================================
# Corpus Management Tests
# =============================================================================


def test_corpus_minimization(simple_protocol_spec: ProtocolSpec) -> None:
    """Test corpus minimization algorithm."""
    config = FuzzingConfig(
        max_iterations=100,
        corpus_minimization=True,
        min_corpus_size=5,
        max_corpus_size=20,
        seed=42,
    )
    fuzzer = ProtocolFuzzer(config)

    # Create large corpus
    fuzzer._corpus = [bytes([i % 256]) * 8 for i in range(50)]

    fuzzer._minimize_corpus()

    # Corpus should be reduced but not below min_corpus_size
    assert len(fuzzer._corpus) >= config.min_corpus_size
    assert len(fuzzer._corpus) <= config.max_corpus_size


def test_select_input_coverage_guided(fuzzer: ProtocolFuzzer) -> None:
    """Test input selection for coverage-guided fuzzing."""
    fuzzer.config.strategy = "coverage_guided"
    fuzzer._corpus = [b"\x01", b"\x02", b"\x03", b"\x04", b"\x05"]

    selected = fuzzer._select_input()
    assert selected in fuzzer._corpus


def test_select_input_empty_corpus(fuzzer: ProtocolFuzzer) -> None:
    """Test input selection with empty corpus."""
    fuzzer._corpus = []
    selected = fuzzer._select_input()
    assert selected == b""


# =============================================================================
# Coverage Tests
# =============================================================================


def test_coverage_hash_consistency(fuzzer: ProtocolFuzzer) -> None:
    """Test coverage hash computation consistency."""
    data = b"\xaa\x55\x01\x02\x03\x04"

    hash1 = fuzzer._compute_coverage_hash(data)
    hash2 = fuzzer._compute_coverage_hash(data)

    assert hash1 == hash2


def test_coverage_hash_uniqueness(fuzzer: ProtocolFuzzer) -> None:
    """Test coverage hash produces different values for different inputs."""
    data1 = b"\xaa\x55\x01\x02"
    data2 = b"\xbb\x66\x03\x04"

    hash1 = fuzzer._compute_coverage_hash(data1)
    hash2 = fuzzer._compute_coverage_hash(data2)

    # Hashes should likely be different (not guaranteed but highly probable)
    assert hash1 != hash2


# =============================================================================
# Crash Detection Tests
# =============================================================================


def test_crash_detection(fuzzer: ProtocolFuzzer) -> None:
    """Test crash detection mechanism."""

    def crashing_function(data: bytes) -> dict:
        """Function that always crashes."""
        raise RuntimeError("Intentional crash")

    result = fuzzer._execute_target(b"\xaa\x01\x02", crashing_function)

    assert result.result == FuzzTestResult.CRASH
    assert "Intentional crash" in result.error_message
    assert len(result.stack_trace) > 0


def test_crash_deduplication(simple_protocol_spec: ProtocolSpec) -> None:
    """Test that duplicate crashes are deduplicated."""
    config = FuzzingConfig(max_iterations=20, crash_detection=True, seed=42)
    fuzzer = ProtocolFuzzer(config)

    # Create multiple results with same crash
    crash_input = b"\xaa\xff\x00\x01\x02\x03\x04\x05"

    result1 = FuzzingResult(test_case=crash_input, result=FuzzTestResult.CRASH)
    result2 = FuzzingResult(test_case=crash_input, result=FuzzTestResult.CRASH)

    fuzzer._update_corpus(crash_input, result1)
    fuzzer._update_corpus(crash_input, result2)

    # Should only have one unique crash
    assert fuzzer._report.unique_crashes == 1


# =============================================================================
# Report Tests
# =============================================================================


def test_fuzzing_report_crash_rate() -> None:
    """Test crash rate calculation in report."""
    report = FuzzingReport(total_iterations=100, total_crashes=5)
    assert report.crash_rate == 0.05


def test_fuzzing_report_crash_rate_zero_iterations() -> None:
    """Test crash rate with zero iterations."""
    report = FuzzingReport(total_iterations=0, total_crashes=0)
    assert report.crash_rate == 0.0


def test_fuzzing_report_unique_crashes() -> None:
    """Test unique crash counting."""
    report = FuzzingReport(
        crashes=[
            b"\x01\x02",
            b"\x01\x02",  # Duplicate
            b"\x03\x04",
            b"\x03\x04",  # Duplicate
            b"\x05\x06",
        ]
    )
    assert report.unique_crashes == 3


# =============================================================================
# Export Tests
# =============================================================================


def test_export_crashes(fuzzer: ProtocolFuzzer, tmp_path: Path) -> None:
    """Test exporting crash-inducing inputs."""
    crashes = [
        b"\xaa\x01\x02\x03",
        b"\xbb\x04\x05\x06",
        b"\xcc\x07\x08\x09",
    ]
    fuzzer._report.crashes = crashes

    output_dir = tmp_path / "crashes"
    fuzzer.export_crashes(output_dir)

    assert output_dir.exists()

    crash_files = sorted(output_dir.glob("crash_*.bin"))
    assert len(crash_files) == 3

    # Verify content (sorted by filename)
    for idx, crash_file in enumerate(crash_files):
        assert crash_file.read_bytes() == crashes[idx]


def test_export_corpus(fuzzer: ProtocolFuzzer, tmp_path: Path) -> None:
    """Test exporting minimized corpus."""
    fuzzer._corpus = [b"\x01\x02", b"\x03\x04", b"\x05\x06"]

    output_dir = tmp_path / "corpus"
    fuzzer.export_corpus(output_dir)

    assert output_dir.exists()

    corpus_files = list(output_dir.glob("input_*.bin"))
    assert len(corpus_files) == 3


def test_export_report(fuzzer: ProtocolFuzzer, tmp_path: Path) -> None:
    """Test exporting fuzzing report as JSON."""
    import json

    fuzzer._report = FuzzingReport(
        total_iterations=1000,
        total_crashes=5,
        total_coverage_branches=42,
        corpus_size=127,
        execution_time_seconds=15.3,
    )

    output_file = tmp_path / "report.json"
    fuzzer.export_report(output_file)

    assert output_file.exists()

    report_data = json.loads(output_file.read_text())
    assert report_data["total_iterations"] == 1000
    assert report_data["total_crashes"] == 5
    assert report_data["total_coverage_branches"] == 42
    assert report_data["corpus_size"] == 127


def test_export_pcap(fuzzer: ProtocolFuzzer, tmp_path: Path) -> None:
    """Test exporting corpus as PCAP file."""
    pytest.importorskip("scapy", reason="scapy not available")

    messages = [b"\xaa\x01\x02\x03", b"\xbb\x04\x05\x06"]

    output_file = tmp_path / "corpus.pcap"
    fuzzer.export_pcap(messages, output_file)

    assert output_file.exists()
    assert output_file.stat().st_size > 0


# =============================================================================
# Statistics Tests
# =============================================================================


def test_update_statistics(fuzzer: ProtocolFuzzer) -> None:
    """Test statistics updating."""
    result = FuzzingResult(
        test_case=b"\xaa\x01",
        result=FuzzTestResult.PASS,
        coverage_delta=1,
    )

    fuzzer._update_statistics(result, MutationOperator.BIT_FLIP)

    assert fuzzer._report.total_iterations == 1
    assert fuzzer._report.total_coverage_branches == 1
    assert fuzzer._report.mutation_stats["BIT_FLIP"] == 1


def test_update_statistics_crash(fuzzer: ProtocolFuzzer) -> None:
    """Test statistics for crash results."""
    result = FuzzingResult(test_case=b"\xaa\x01", result=FuzzTestResult.CRASH)

    fuzzer._update_statistics(result, MutationOperator.CHECKSUM_CORRUPT)

    assert fuzzer._report.total_crashes == 1


# =============================================================================
# Utility Function Tests
# =============================================================================


def test_pack_value(fuzzer: ProtocolFuzzer) -> None:
    """Test integer packing."""
    assert fuzzer._pack_value(0x1234, 2) == b"\x34\x12"
    assert fuzzer._pack_value(0xABCDEF, 3) == b"\xef\xcd\xab"
    assert fuzzer._pack_value(0, 1) == b"\x00"
    assert fuzzer._pack_value(255, 1) == b"\xff"


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_fuzzing_workflow(simple_protocol_spec: ProtocolSpec, tmp_path: Path) -> None:
    """Test complete fuzzing workflow end-to-end."""
    # Configure fuzzer
    config = FuzzingConfig(
        strategy="coverage_guided",
        max_iterations=50,
        crash_detection=True,
        corpus_minimization=True,
        seed=42,
    )
    fuzzer = ProtocolFuzzer(config)

    # Define target parser
    def protocol_parser(data: bytes) -> dict:
        """Simple protocol parser."""
        if len(data) < 8:
            raise ValueError("Message too short")

        if data[0] != 0xAA:
            raise ValueError("Invalid header")

        # Trigger crash on specific pattern
        if len(data) > 5 and data[1] == 0xFF and data[5] == 0xFF:
            raise RuntimeError("Double 0xFF pattern detected")

        return {"header": data[0], "cmd": data[1], "valid": True}

    # Run fuzzing
    report = fuzzer.fuzz_protocol(simple_protocol_spec, None, protocol_parser)

    # Verify report
    assert report.total_iterations == 50
    assert report.corpus_size > 0

    # Export results
    fuzzer.export_crashes(tmp_path / "crashes")
    fuzzer.export_corpus(tmp_path / "corpus")
    fuzzer.export_report(tmp_path / "report.json")

    # Verify exports
    assert (tmp_path / "corpus").exists()
    assert (tmp_path / "report.json").exists()
