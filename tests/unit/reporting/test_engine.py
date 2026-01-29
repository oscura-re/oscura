"""Tests for comprehensive analysis report engine.

Tests verify the AnalysisEngine correctly orchestrates analysis execution
across all domains with progress tracking and error handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.reporting.config import (
    AnalysisConfig,
    AnalysisDomain,
    InputType,
    ProgressInfo,
)
from oscura.reporting.engine import AnalysisEngine

pytestmark = pytest.mark.unit


class TestAnalysisEngineInit:
    """Tests for AnalysisEngine initialization."""

    def test_default_config(self):
        """Test engine uses default config if not provided."""
        engine = AnalysisEngine()
        assert engine.config is not None
        assert engine.config.continue_on_error is True

    def test_custom_config(self):
        """Test engine uses provided config."""
        config = AnalysisConfig(timeout_per_analysis=60.0)
        engine = AnalysisEngine(config)
        assert engine.config.timeout_per_analysis == 60.0


class TestAnalysisEngineInputTypeDetection:
    """Tests for AnalysisEngine.detect_input_type method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AnalysisEngine()

    def test_detect_waveform_from_path(self, engine):
        """Test detecting waveform from file extension."""
        for ext in [".wfm", ".csv", ".npz", ".h5", ".hdf5", ".wav", ".tdms"]:
            path = Path(f"/data/signal{ext}")
            result = engine.detect_input_type(path, None)
            assert result == InputType.WAVEFORM

    def test_detect_digital_from_path(self, engine):
        """Test detecting digital from file extension."""
        for ext in [".vcd", ".sr"]:
            path = Path(f"/data/logic{ext}")
            result = engine.detect_input_type(path, None)
            assert result == InputType.DIGITAL

    def test_detect_pcap_from_path(self, engine):
        """Test detecting PCAP from file extension."""
        for ext in [".pcap", ".pcapng"]:
            path = Path(f"/data/capture{ext}")
            result = engine.detect_input_type(path, None)
            assert result == InputType.PCAP

    def test_detect_binary_from_path(self, engine):
        """Test detecting binary from file extension."""
        for ext in [".bin", ".raw"]:
            path = Path(f"/data/file{ext}")
            result = engine.detect_input_type(path, None)
            assert result == InputType.BINARY

    def test_detect_waveform_from_trace_data(self, engine):
        """Test detecting waveform from Trace-like object."""
        mock_trace = MagicMock(spec=["data", "metadata"])
        mock_trace.data = np.array([1, 2, 3])
        mock_trace.metadata = MagicMock(spec=["is_digital"])
        mock_trace.metadata.is_digital = False

        result = engine.detect_input_type(None, mock_trace)
        assert result == InputType.WAVEFORM

    def test_detect_digital_from_trace_data(self, engine):
        """Test detecting digital from DigitalTrace-like object."""
        mock_trace = MagicMock(spec=["data", "metadata"])
        mock_trace.data = np.array([0, 1, 0])
        mock_trace.metadata = MagicMock(spec=["is_digital"])
        mock_trace.metadata.is_digital = True

        result = engine.detect_input_type(None, mock_trace)
        assert result == InputType.DIGITAL

    def test_detect_binary_from_bytes(self, engine):
        """Test detecting binary from bytes data."""
        data = b"\x00\x01\x02\x03"
        result = engine.detect_input_type(None, data)
        assert result == InputType.BINARY

    def test_detect_binary_from_bytearray(self, engine):
        """Test detecting binary from bytearray data."""
        data = bytearray([0, 1, 2, 3])
        result = engine.detect_input_type(None, data)
        assert result == InputType.BINARY

    def test_detect_packets_from_list(self, engine):
        """Test detecting packets from list data."""
        data = [{"timestamp": 0.0}, {"timestamp": 1.0}]
        result = engine.detect_input_type(None, data)
        assert result == InputType.PACKETS

    def test_detect_waveform_from_ndarray(self, engine):
        """Test detecting waveform from numpy array."""
        data = np.array([1.0, 2.0, 3.0])
        result = engine.detect_input_type(None, data)
        assert result == InputType.WAVEFORM


class TestAnalysisEngineRun:
    """Tests for AnalysisEngine.run method."""

    @pytest.fixture
    def engine(self):
        """Create engine with minimal config."""
        config = AnalysisConfig(
            domains=[AnalysisDomain.WAVEFORM],
            timeout_per_analysis=5.0,
            continue_on_error=True,
        )
        return AnalysisEngine(config)

    def test_run_requires_input(self, engine):
        """Test run raises error without input."""
        with pytest.raises(ValueError, match="Must provide either input_path or data"):
            engine.run()

    def test_run_with_file_not_found(self, engine):
        """Test run raises error for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            engine.run(input_path=Path("/nonexistent/file.wfm"))

    @patch("oscura.reporting.engine.importlib.import_module")
    def test_run_returns_result_structure(self, mock_import, engine):
        """Test run returns expected result structure."""
        # Mock module with no functions
        mock_module = MagicMock()
        mock_module.__name__ = "oscura.analyzers.waveform.measurements"
        mock_import.return_value = mock_module

        data = np.sin(np.linspace(0, 2 * np.pi, 100))
        result = engine.run(data=data)

        assert "results" in result
        assert "errors" in result
        assert "stats" in result
        assert isinstance(result["results"], dict)
        assert isinstance(result["errors"], list)
        assert isinstance(result["stats"], dict)

    def test_run_with_progress_callback(self, engine):
        """Test run calls progress callback."""
        progress_calls = []

        def on_progress(info: ProgressInfo):
            progress_calls.append(info)

        data = np.sin(np.linspace(0, 2 * np.pi, 100))

        with patch("oscura.reporting.engine.importlib.import_module"):
            engine.run(data=data, progress_callback=on_progress)

        assert len(progress_calls) >= 2
        phases = [p.phase for p in progress_calls]
        assert "loading" in phases
        assert "complete" in phases


class TestAnalysisEngineDomainExecution:
    """Tests for AnalysisEngine domain execution."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AnalysisEngine(AnalysisConfig(continue_on_error=True))

    @patch("oscura.reporting.engine.importlib.import_module")
    def test_executes_domain_modules(self, mock_import, engine):
        """Test that engine tries to import domain modules."""
        # Create a mock module with a test function
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        mock_import.return_value = mock_module

        data = np.array([1, 2, 3])
        engine._execute_domain(AnalysisDomain.WAVEFORM, data)

        # Should have tried to import modules
        assert mock_import.called

    @patch("oscura.reporting.engine.importlib.import_module")
    def test_handles_import_error(self, mock_import, engine):
        """Test that engine handles import errors gracefully."""
        mock_import.side_effect = ImportError("Module not found")

        data = np.array([1, 2, 3])
        results, errors = engine._execute_domain(AnalysisDomain.WAVEFORM, data)

        # Should not raise, should return empty/error results
        assert isinstance(results, dict)
        assert isinstance(errors, list)


class TestAnalysisEngineStats:
    """Tests for AnalysisEngine statistics calculation."""

    @patch("oscura.reporting.engine.importlib.import_module")
    def test_stats_include_required_fields(self, mock_import):
        """Test that stats include all required fields."""
        mock_module = MagicMock()
        mock_import.return_value = mock_module

        config = AnalysisConfig(domains=[AnalysisDomain.WAVEFORM])
        engine = AnalysisEngine(config)

        data = np.array([1, 2, 3])
        result = engine.run(data=data)

        stats = result["stats"]
        assert "input_type" in stats
        assert "total_domains" in stats
        assert "total_analyses" in stats
        assert "successful_analyses" in stats
        assert "failed_analyses" in stats
        assert "success_rate" in stats
        assert "duration_seconds" in stats


class TestAnalysisEngineParallelExecution:
    """Tests for AnalysisEngine parallel domain execution."""

    @pytest.fixture
    def engine_parallel(self):
        """Create engine with parallel execution enabled."""
        config = AnalysisConfig(
            domains=[AnalysisDomain.WAVEFORM, AnalysisDomain.STATISTICS],
            parallel_domains=True,
            timeout_per_analysis=5.0,
            continue_on_error=True,
        )
        return AnalysisEngine(config)

    @pytest.fixture
    def engine_sequential(self):
        """Create engine with parallel execution disabled."""
        config = AnalysisConfig(
            domains=[AnalysisDomain.WAVEFORM, AnalysisDomain.STATISTICS],
            parallel_domains=False,
            timeout_per_analysis=5.0,
            continue_on_error=True,
        )
        return AnalysisEngine(config)

    @patch("oscura.reporting.engine.importlib.import_module")
    def test_parallel_execution_enabled(self, mock_import, engine_parallel):
        """Test that parallel execution is used when enabled."""
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        mock_import.return_value = mock_module

        data = np.array([1, 2, 3])
        result = engine_parallel.run(data=data)

        # Should complete successfully with multiple domains
        assert result["stats"]["total_domains"] >= 1
        assert "results" in result

    @patch("oscura.reporting.engine.importlib.import_module")
    def test_sequential_fallback(self, mock_import, engine_sequential):
        """Test that sequential execution works when parallel is disabled."""
        mock_module = MagicMock()
        mock_module.__name__ = "test_module"
        mock_import.return_value = mock_module

        data = np.array([1, 2, 3])
        result = engine_sequential.run(data=data)

        # Should complete successfully with multiple domains
        assert result["stats"]["total_domains"] >= 1
        assert "results" in result

    @patch("oscura.reporting.engine.importlib.import_module")
    def test_parallel_handles_domain_errors(self, mock_import, engine_parallel):
        """Test that parallel execution handles errors in one domain without affecting others."""

        def side_effect(module_name):
            """Mock import that fails for specific modules."""
            if "waveform" in module_name:
                raise ImportError("Simulated waveform module error")
            mock_module = MagicMock()
            mock_module.__name__ = module_name
            return mock_module

        mock_import.side_effect = side_effect

        data = np.array([1, 2, 3])
        result = engine_parallel.run(data=data)

        # Should complete despite errors in one domain
        assert "results" in result
        assert "errors" in result
        assert isinstance(result["errors"], list)

    def test_single_domain_uses_sequential(self):
        """Test that single domain execution uses sequential path."""
        config = AnalysisConfig(
            domains=[AnalysisDomain.WAVEFORM],
            parallel_domains=True,  # Even with parallel enabled
            timeout_per_analysis=5.0,
        )
        engine = AnalysisEngine(config)

        data = np.array([1, 2, 3])

        with patch("oscura.reporting.engine.importlib.import_module"):
            result = engine.run(data=data)

        # Should complete successfully
        assert "results" in result
