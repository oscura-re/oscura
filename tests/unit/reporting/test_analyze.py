"""Tests for comprehensive analysis report main entry point.

Tests verify the analyze() function correctly orchestrates the full
analysis pipeline from input to output.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.reporting.analyze import (
    UnsupportedFormatError,
    _detect_input_type_from_data,
    _detect_input_type_from_file,
    analyze,
)
from oscura.reporting.config import (
    AnalysisConfig,
    AnalysisDomain,
    AnalysisResult,
    InputType,
    ProgressInfo,
)

pytestmark = pytest.mark.unit


class TestDetectInputTypeFromFile:
    """Tests for _detect_input_type_from_file function."""

    def test_waveform_extensions(self):
        """Test waveform file extensions are detected."""
        waveform_exts = [".wfm", ".csv", ".npz", ".hdf5", ".h5", ".wav", ".tdms"]
        for ext in waveform_exts:
            path = Path(f"/data/signal{ext}")
            assert _detect_input_type_from_file(path) == InputType.WAVEFORM

    def test_digital_extensions(self):
        """Test digital file extensions are detected."""
        digital_exts = [".vcd", ".sr"]
        for ext in digital_exts:
            path = Path(f"/data/logic{ext}")
            assert _detect_input_type_from_file(path) == InputType.DIGITAL

    def test_binary_extensions(self):
        """Test binary file extensions are detected."""
        binary_exts = [".bin", ".raw"]
        for ext in binary_exts:
            path = Path(f"/data/capture{ext}")
            assert _detect_input_type_from_file(path) == InputType.BINARY

    def test_pcap_extensions(self):
        """Test PCAP file extensions are detected."""
        pcap_exts = [".pcap", ".pcapng"]
        for ext in pcap_exts:
            path = Path(f"/data/network{ext}")
            assert _detect_input_type_from_file(path) == InputType.PCAP

    def test_unsupported_extension_raises(self):
        """Test that unsupported extensions raise error."""
        path = Path("/data/file.xyz")
        with pytest.raises(UnsupportedFormatError):
            _detect_input_type_from_file(path)


class TestDetectInputTypeFromData:
    """Tests for _detect_input_type_from_data function."""

    def test_trace_object_detected_as_waveform(self):
        """Test Trace-like objects are detected as waveform."""

        class MockTrace:
            time = np.array([0, 1, 2])
            voltage = np.array([0, 1, 0])

        data = MockTrace()
        assert _detect_input_type_from_data(data) == InputType.WAVEFORM

    def test_bytes_detected_as_binary(self):
        """Test bytes are detected as binary."""
        data = b"\x00\x01\x02\x03"
        assert _detect_input_type_from_data(data) == InputType.BINARY

    def test_bytearray_detected_as_binary(self):
        """Test bytearray is detected as binary."""
        data = bytearray([0, 1, 2, 3])
        assert _detect_input_type_from_data(data) == InputType.BINARY

    def test_packet_list_detected(self):
        """Test list of packet-like objects detected as packets."""

        class MockPacket:
            timestamp = 0.0
            data = b"\x00"

        data = [MockPacket(), MockPacket()]
        assert _detect_input_type_from_data(data) == InputType.PACKETS


class TestAnalyzeValidation:
    """Tests for analyze() input validation."""

    def test_no_input_raises_error(self):
        """Test that providing no input raises ValueError."""
        with pytest.raises(ValueError, match="Either input_path or data must be provided"):
            analyze()

    def test_both_inputs_raises_error(self):
        """Test that providing both inputs raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".csv") as f:
            with pytest.raises(ValueError, match="Provide input_path OR data, not both"):
                analyze(input_path=f.name, data=np.array([1, 2, 3]))

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            analyze(input_path="/nonexistent/file.wfm")


class TestAnalyzeOutput:
    """Tests for analyze() output structure."""

    @pytest.fixture
    def mock_engine_result(self):
        """Create mock engine result."""
        return {
            "results": {
                AnalysisDomain.WAVEFORM: {"amplitude": 1.0},
            },
            "errors": [],
            "stats": {
                "total_analyses": 1,
                "successful_analyses": 1,
                "failed_analyses": 0,
                "skipped_analyses": 0,
            },
        }

    @patch("oscura.reporting.engine.AnalysisEngine")
    def test_returns_analysis_result(self, mock_engine_class, mock_engine_result):
        """Test that analyze returns AnalysisResult."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_engine_result
        mock_engine_class.return_value = mock_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock trace data
            data = MagicMock()
            data.time = np.linspace(0, 1, 100)
            data.voltage = np.sin(2 * np.pi * 10 * data.time)

            config = AnalysisConfig(
                domains=[AnalysisDomain.WAVEFORM],
                generate_plots=False,
            )

            result = analyze(data=data, output_dir=tmpdir, config=config)

            assert isinstance(result, AnalysisResult)
            assert result.output_dir.exists()

    @patch("oscura.reporting.engine.AnalysisEngine")
    def test_creates_timestamped_directory(self, mock_engine_class, mock_engine_result):
        """Test that analyze creates timestamped output directory."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = mock_engine_result
        mock_engine_class.return_value = mock_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            data = MagicMock()
            data.time = np.array([0, 1])
            data.voltage = np.array([0, 1])

            config = AnalysisConfig(generate_plots=False)
            result = analyze(data=data, output_dir=tmpdir, config=config)

            # Directory name should contain timestamp and "analysis"
            dirname = result.output_dir.name
            assert "_analysis" in dirname
            assert len(dirname.split("_")[0]) == 8  # YYYYMMDD


class TestAnalyzeProgressCallback:
    """Tests for analyze() progress callback functionality."""

    @patch("oscura.reporting.engine.AnalysisEngine")
    def test_progress_callback_called(self, mock_engine_class):
        """Test that progress callback is called during analysis."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = {
            "results": {},
            "errors": [],
            "stats": {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "skipped_analyses": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        progress_calls = []

        def on_progress(info: ProgressInfo):
            progress_calls.append(info)

        with tempfile.TemporaryDirectory() as tmpdir:
            data = MagicMock()
            data.time = np.array([0, 1])
            data.voltage = np.array([0, 1])

            config = AnalysisConfig(generate_plots=False)
            analyze(
                data=data,
                output_dir=tmpdir,
                config=config,
                progress_callback=on_progress,
            )

        # Should have been called at least for start and completion
        assert len(progress_calls) >= 2
        assert progress_calls[0].phase == "initializing"
        assert progress_calls[-1].phase == "complete"


class TestAnalyzeConfig:
    """Tests for analyze() configuration handling."""

    @patch("oscura.reporting.engine.AnalysisEngine")
    def test_uses_default_config(self, mock_engine_class):
        """Test that default config is used if not provided."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = {
            "results": {},
            "errors": [],
            "stats": {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "skipped_analyses": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            data = MagicMock()
            data.time = np.array([0, 1])
            data.voltage = np.array([0, 1])

            # No config provided - should use default
            result = analyze(data=data, output_dir=tmpdir)

            # Engine should have been initialized
            mock_engine_class.assert_called_once()

    @patch("oscura.reporting.engine.AnalysisEngine")
    def test_uses_provided_config(self, mock_engine_class):
        """Test that provided config is used."""
        mock_engine = MagicMock()
        mock_engine.run.return_value = {
            "results": {},
            "errors": [],
            "stats": {
                "total_analyses": 0,
                "successful_analyses": 0,
                "failed_analyses": 0,
                "skipped_analyses": 0,
            },
        }
        mock_engine_class.return_value = mock_engine

        with tempfile.TemporaryDirectory() as tmpdir:
            data = MagicMock()
            data.time = np.array([0, 1])
            data.voltage = np.array([0, 1])

            config = AnalysisConfig(
                domains=[AnalysisDomain.SPECTRAL],
                generate_plots=False,
            )
            analyze(data=data, output_dir=tmpdir, config=config)

            # Verify engine was initialized with correct config
            call_args = mock_engine_class.call_args
            assert call_args[0][0] == config
