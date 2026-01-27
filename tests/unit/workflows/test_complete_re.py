"""Tests for complete one-function reverse engineering workflow.

This module tests the full_protocol_re() function that automates
the entire RE workflow from captures to dissectors.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from oscura.workflows.complete_re import (
    CompleteREResult,
    _calculate_entropy,
    _calculate_overall_confidence,
    _detect_crypto_regions,
    full_protocol_re,
)
from oscura.workflows.reverse_engineering import (
    FieldSpec,
    InferredFrame,
    ProtocolSpec,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_trace() -> MagicMock:
    """Create a mock WaveformTrace."""
    trace = MagicMock()
    # Create realistic UART-like digital signal
    sample_rate = 1_000_000  # 1 MS/s
    baud_rate = 9600
    samples_per_bit = sample_rate // baud_rate  # ~104 samples/bit

    # Generate simple digital signal with a few frames
    # Frame: start(0) + 8 data bits + stop(1) = 10 bits
    signal = np.ones(10000) * 5.0  # Idle high at 5V

    # Add a few UART frames
    for frame_start in [1000, 2000, 3000]:
        # Start bit (0)
        signal[frame_start : frame_start + samples_per_bit] = 0.0
        # Data bits (0xA5 = 10100101, LSB first)
        for bit_idx, bit_val in enumerate([1, 0, 1, 0, 0, 1, 0, 1]):
            start = frame_start + (bit_idx + 1) * samples_per_bit
            end = start + samples_per_bit
            signal[start:end] = 5.0 if bit_val else 0.0
        # Stop bit (1)
        start = frame_start + 9 * samples_per_bit
        end = start + samples_per_bit
        signal[start:end] = 5.0

    trace.data = signal
    trace.metadata = MagicMock()
    trace.metadata.sample_rate = float(sample_rate)
    return trace


@pytest.fixture
def temp_capture_file(tmp_path: Path, mock_trace: MagicMock) -> Path:
    """Create a temporary capture file."""
    capture_file = tmp_path / "capture.bin"
    # Write some binary data
    mock_trace.data.astype(np.float32).tofile(capture_file)
    return capture_file


@pytest.fixture
def temp_export_dir(tmp_path: Path) -> Path:
    """Create a temporary export directory."""
    export_dir = tmp_path / "output"
    export_dir.mkdir()
    return export_dir


@pytest.fixture
def sample_protocol_spec() -> ProtocolSpec:
    """Create a sample protocol specification."""
    return ProtocolSpec(
        name="TestProtocol",
        baud_rate=9600.0,
        frame_format="8N1",
        sync_pattern="aa55",
        frame_length=16,
        fields=[
            FieldSpec(name="sync", offset=0, size=2, field_type="constant", value="aa55"),
            FieldSpec(name="length", offset=2, size=1, field_type="uint8"),
            FieldSpec(name="data", offset=3, size="length - 5", field_type="bytes"),
            FieldSpec(name="checksum", offset=-1, size=1, field_type="checksum"),
        ],
        checksum_type="xor",
        checksum_position=-1,
        confidence=0.85,
    )


@pytest.fixture
def sample_frames() -> list[InferredFrame]:
    """Create sample decoded frames."""
    return [
        InferredFrame(
            start_bit=0,
            end_bit=80,
            raw_bits="0" * 80,
            raw_bytes=bytes([0xAA, 0x55, 0x10, 0x01, 0x02, 0x03, 0x04, 0x67]),
            checksum_valid=True,
        ),
        InferredFrame(
            start_bit=100,
            end_bit=180,
            raw_bits="0" * 80,
            raw_bytes=bytes([0xAA, 0x55, 0x10, 0x05, 0x06, 0x07, 0x08, 0x6C]),
            checksum_valid=True,
        ),
        InferredFrame(
            start_bit=200,
            end_bit=280,
            raw_bits="0" * 80,
            raw_bytes=bytes([0xAA, 0x55, 0x10, 0x09, 0x0A, 0x0B, 0x0C, 0x69]),
            checksum_valid=True,
        ),
    ]


# =============================================================================
# Unit Tests - Helper Functions
# =============================================================================


class TestCalculateEntropy:
    """Tests for Shannon entropy calculation."""

    def test_empty_data(self) -> None:
        """Test entropy of empty byte sequence."""
        assert _calculate_entropy(b"") == 0.0

    def test_uniform_data(self) -> None:
        """Test entropy of all same bytes (minimum entropy)."""
        data = b"\x00" * 100
        entropy = _calculate_entropy(data)
        assert entropy == 0.0

    def test_random_data(self) -> None:
        """Test entropy of random data (high entropy)."""
        np.random.seed(42)
        data = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
        entropy = _calculate_entropy(data)
        # Random data should have entropy close to 8 bits
        assert 7.0 < entropy < 8.0

    def test_binary_pattern(self) -> None:
        """Test entropy of alternating pattern."""
        data = b"\x00\xff" * 100
        entropy = _calculate_entropy(data)
        # Two equally probable bytes = 1 bit entropy
        assert 0.9 < entropy < 1.1


class TestDetectCryptoRegions:
    """Tests for crypto/entropy region detection."""

    def test_no_frames(self) -> None:
        """Test with empty frame list."""
        regions = _detect_crypto_regions([])
        assert regions == []

    def test_low_entropy_frames(self, sample_frames: list[InferredFrame]) -> None:
        """Test with low-entropy frames (no crypto detected)."""
        # Regular frames with low entropy
        regions = _detect_crypto_regions(sample_frames)
        # Should not detect crypto in structured data
        assert len(regions) == 0

    def test_high_entropy_frames(self) -> None:
        """Test with high-entropy frames (crypto detected)."""
        # Create frames with random high-entropy data (1000 bytes for proper entropy)
        np.random.seed(42)
        frames = [
            InferredFrame(
                start_bit=0,
                end_bit=80,
                raw_bits="0" * 80,
                raw_bytes=bytes(np.random.randint(0, 256, 1000, dtype=np.uint8)),
            )
            for _ in range(3)
        ]

        regions = _detect_crypto_regions(frames)
        # Should detect high entropy regions
        assert len(regions) > 0
        assert all(r["entropy"] > 7.0 for r in regions)


class TestCalculateOverallConfidence:
    """Tests for overall confidence calculation."""

    def test_base_confidence(self, sample_protocol_spec: ProtocolSpec) -> None:
        """Test with no warnings or partial results."""
        confidence = _calculate_overall_confidence(sample_protocol_spec, {}, [])
        # Should be close to protocol spec confidence
        assert 0.8 < confidence < 0.9

    def test_warning_penalty(self, sample_protocol_spec: ProtocolSpec) -> None:
        """Test confidence reduction with warnings."""
        warnings = ["Warning 1", "Warning 2", "Warning 3"]
        confidence = _calculate_overall_confidence(sample_protocol_spec, {}, warnings)
        # Should be lower due to warnings
        assert confidence < sample_protocol_spec.confidence

    def test_success_bonus(self, sample_protocol_spec: ProtocolSpec) -> None:
        """Test confidence boost from successful steps."""
        partial_results = {
            "traces": {},
            "reverse_engineering": {},
            "state_machine": {},
        }
        confidence = _calculate_overall_confidence(sample_protocol_spec, partial_results, [])
        # Should be higher due to successful steps
        assert confidence >= sample_protocol_spec.confidence

    def test_bounds(self, sample_protocol_spec: ProtocolSpec) -> None:
        """Test confidence stays within [0, 1] bounds."""
        # Many warnings
        confidence = _calculate_overall_confidence(sample_protocol_spec, {}, ["w"] * 100)
        assert 0.0 <= confidence <= 1.0

        # Many successes
        partial_results = {f"step_{i}": {} for i in range(100)}
        confidence = _calculate_overall_confidence(sample_protocol_spec, partial_results, [])
        assert 0.0 <= confidence <= 1.0


# =============================================================================
# Integration Tests - Full Workflow
# =============================================================================


class TestFullProtocolRE:
    """Tests for the complete workflow function."""

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_successful_workflow_single_capture(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
        mock_trace: MagicMock,
        sample_protocol_spec: ProtocolSpec,
        sample_frames: list[InferredFrame],
    ) -> None:
        """Test successful complete workflow with single capture."""
        # Mock loaders
        mock_load.return_value = {"capture": mock_trace}

        # Mock reverse engineering result
        mock_re_result = Mock()
        mock_re_result.protocol_spec = sample_protocol_spec
        mock_re_result.frames = sample_frames
        mock_re_result.baud_rate = 9600.0
        mock_re_result.bit_stream = "0" * 1000
        mock_re_result.byte_stream = b"\xaa\x55" * 50
        mock_re_result.sync_positions = [0, 100, 200]
        mock_re_result.characterization = {"threshold": 2.5}
        mock_re_result.confidence = 0.85
        mock_re_result.warnings = []
        mock_reverse_engineer.return_value = mock_re_result

        # Run workflow
        result = full_protocol_re(
            captures=str(temp_capture_file),
            export_dir=str(temp_export_dir),
            verbose=False,
        )

        # Verify result
        assert isinstance(result, CompleteREResult)
        assert result.protocol_spec == sample_protocol_spec
        assert result.confidence_score > 0.0
        assert result.execution_time > 0.0

        # Verify files generated
        assert result.dissector_path is not None
        assert result.scapy_layer_path is not None
        assert result.kaitai_path is not None
        assert result.report_path is not None

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_workflow_multiple_captures(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        tmp_path: Path,
        mock_trace: MagicMock,
        sample_protocol_spec: ProtocolSpec,
        sample_frames: list[InferredFrame],
    ) -> None:
        """Test workflow with multiple captures for differential analysis."""
        # Create multiple capture files
        capture1 = tmp_path / "idle.bin"
        capture2 = tmp_path / "button.bin"
        capture1.write_bytes(b"\x00" * 1000)
        capture2.write_bytes(b"\xff" * 1000)

        # Mock loaders
        mock_load.return_value = {"idle": mock_trace, "button": mock_trace}

        # Mock reverse engineering
        mock_re_result = Mock()
        mock_re_result.protocol_spec = sample_protocol_spec
        mock_re_result.frames = sample_frames
        mock_reverse_engineer.return_value = mock_re_result

        # Run workflow
        result = full_protocol_re(
            captures={"idle": str(capture1), "button": str(capture2)},
            export_dir=str(tmp_path / "output"),
            verbose=False,
        )

        # Verify differential analysis was performed
        assert "differential" in result.partial_results
        assert result.partial_results["differential"]["trace_count"] == 2

    def test_missing_capture_file(self, temp_export_dir: Path) -> None:
        """Test error handling for missing capture file."""
        with pytest.raises(FileNotFoundError, match="Capture file not found"):
            full_protocol_re(
                captures="nonexistent.bin",
                export_dir=str(temp_export_dir),
            )

    def test_empty_captures(self, temp_export_dir: Path) -> None:
        """Test error handling for empty captures dict."""
        with pytest.raises(ValueError, match="No captures provided"):
            full_protocol_re(
                captures={},
                export_dir=str(temp_export_dir),
            )

    @patch("oscura.workflows.complete_re._load_captures")
    def test_load_failure_raises_runtime_error(
        self,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
    ) -> None:
        """Test that load failures raise RuntimeError."""
        mock_load.side_effect = Exception("Load failed")

        with pytest.raises(RuntimeError, match="Failed to load captures"):
            full_protocol_re(
                captures=str(temp_capture_file),
                export_dir=str(temp_export_dir),
            )

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_graceful_degradation_decode_failure(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
        mock_trace: MagicMock,
    ) -> None:
        """Test graceful degradation when decoding fails."""
        mock_load.return_value = {"capture": mock_trace}
        mock_reverse_engineer.side_effect = Exception("Decode failed")

        # Should complete with minimal spec
        result = full_protocol_re(
            captures=str(temp_capture_file),
            export_dir=str(temp_export_dir),
            verbose=False,
        )

        # Should have warnings
        assert len(result.warnings) > 0
        assert any("decoding" in w.lower() for w in result.warnings)

        # Should have minimal protocol spec
        assert result.protocol_spec.name in ("unknown", "Unknown")
        assert result.protocol_spec.confidence == 0.0

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_optional_features_disabled(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
        mock_trace: MagicMock,
        sample_protocol_spec: ProtocolSpec,
        sample_frames: list[InferredFrame],
    ) -> None:
        """Test workflow with optional features disabled."""
        mock_load.return_value = {"capture": mock_trace}

        mock_re_result = Mock()
        mock_re_result.protocol_spec = sample_protocol_spec
        mock_re_result.frames = sample_frames
        mock_reverse_engineer.return_value = mock_re_result

        # Disable optional features
        result = full_protocol_re(
            captures=str(temp_capture_file),
            export_dir=str(temp_export_dir),
            auto_crc=False,
            detect_crypto=False,
            generate_tests=False,
            validate=False,
            verbose=False,
        )

        # Should still complete successfully
        assert result.protocol_spec == sample_protocol_spec

        # Crypto detection should be skipped
        assert "crypto" not in result.partial_results

        # Test vectors should not be generated
        assert result.test_vectors_path is None

        # Validation should not be performed
        assert result.validation_result is None

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_protocol_hint_used(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
        mock_trace: MagicMock,
        sample_protocol_spec: ProtocolSpec,
    ) -> None:
        """Test that protocol hint is used when provided."""
        mock_load.return_value = {"capture": mock_trace}

        mock_re_result = Mock()
        mock_re_result.protocol_spec = sample_protocol_spec
        mock_re_result.frames = []
        mock_reverse_engineer.return_value = mock_re_result

        result = full_protocol_re(
            captures=str(temp_capture_file),
            protocol_hint="spi",
            export_dir=str(temp_export_dir),
            verbose=False,
        )

        # Protocol hint should be used
        assert result.partial_results["protocol_detection"]["hint"] == "spi"
        assert result.partial_results["protocol_detection"]["confidence"] == 1.0

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_export_files_created(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
        mock_trace: MagicMock,
        sample_protocol_spec: ProtocolSpec,
        sample_frames: list[InferredFrame],
    ) -> None:
        """Test that all export files are created."""
        mock_load.return_value = {"capture": mock_trace}

        mock_re_result = Mock()
        mock_re_result.protocol_spec = sample_protocol_spec
        mock_re_result.frames = sample_frames
        mock_reverse_engineer.return_value = mock_re_result

        result = full_protocol_re(
            captures=str(temp_capture_file),
            export_dir=str(temp_export_dir),
            verbose=False,
        )

        # Verify files exist
        assert result.dissector_path is not None
        assert result.dissector_path.exists()
        assert result.dissector_path.suffix == ".lua"

        assert result.scapy_layer_path is not None
        assert result.scapy_layer_path.exists()
        assert result.scapy_layer_path.suffix == ".py"

        assert result.kaitai_path is not None
        assert result.kaitai_path.exists()
        assert result.kaitai_path.suffix == ".ksy"

        assert result.test_vectors_path is not None
        assert result.test_vectors_path.exists()
        assert result.test_vectors_path.suffix == ".json"

        # Verify test vectors format
        with open(result.test_vectors_path) as f:
            vectors = json.load(f)
        assert "protocol" in vectors
        assert "test_vectors" in vectors
        assert len(vectors["test_vectors"]) > 0

        assert result.report_path is not None
        assert result.report_path.exists()
        assert result.report_path.suffix == ".html"

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_warnings_collected(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
        mock_trace: MagicMock,
        sample_protocol_spec: ProtocolSpec,
        sample_frames: list[InferredFrame],
    ) -> None:
        """Test that warnings are collected from all steps."""
        mock_load.return_value = {"capture": mock_trace}

        # Create frames with high entropy to trigger crypto warning (1000 bytes for proper entropy)
        np.random.seed(42)
        crypto_frames = [
            InferredFrame(
                start_bit=0,
                end_bit=80,
                raw_bits="0" * 80,
                raw_bytes=bytes(np.random.randint(0, 256, 1000, dtype=np.uint8)),
            )
            for _ in range(3)
        ]

        mock_re_result = Mock()
        mock_re_result.protocol_spec = sample_protocol_spec
        mock_re_result.frames = crypto_frames
        mock_reverse_engineer.return_value = mock_re_result

        result = full_protocol_re(
            captures=str(temp_capture_file),
            export_dir=str(temp_export_dir),
            detect_crypto=True,
            verbose=False,
        )

        # Should have crypto warning
        assert len(result.warnings) > 0
        assert any("entropy" in w.lower() or "encryption" in w.lower() for w in result.warnings)


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.performance
class TestPerformance:
    """Performance tests for complete workflow."""

    @patch("oscura.workflows.complete_re._load_captures")
    @patch("oscura.workflows.complete_re.reverse_engineer_signal")
    def test_execution_time_recorded(
        self,
        mock_reverse_engineer: Mock,
        mock_load: Mock,
        temp_capture_file: Path,
        temp_export_dir: Path,
        mock_trace: MagicMock,
        sample_protocol_spec: ProtocolSpec,
    ) -> None:
        """Test that execution time is recorded."""
        mock_load.return_value = {"capture": mock_trace}

        mock_re_result = Mock()
        mock_re_result.protocol_spec = sample_protocol_spec
        mock_re_result.frames = []
        mock_reverse_engineer.return_value = mock_re_result

        result = full_protocol_re(
            captures=str(temp_capture_file),
            export_dir=str(temp_export_dir),
            verbose=False,
        )

        # Execution time should be positive
        assert result.execution_time > 0.0
        # Should complete quickly for mocked workflow
        assert result.execution_time < 10.0
