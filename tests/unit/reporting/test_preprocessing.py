"""Unit tests for reporting engine preprocessing functionality.

Tests verify the AnalysisEngine correctly preprocesses data for domain-specific
requirements, particularly EYE domain and SPARAMS domain handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from oscura.analyzers.signal_integrity.sparams import SParameterData
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.reporting.argument_preparer import ArgumentPreparer
from oscura.reporting.config import AnalysisConfig, AnalysisDomain
from oscura.reporting.engine import AnalysisEngine

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestPreprocessForDomain:
    """Test AnalysisEngine._preprocess_for_domain method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AnalysisEngine(AnalysisConfig())

    def test_preprocess_returns_data_unchanged_for_waveform_domain(self, engine):
        """Test data passes through unchanged for WAVEFORM domain."""
        data = np.sin(np.linspace(0, 2 * np.pi, 100))

        result = engine._preprocess_for_domain(AnalysisDomain.WAVEFORM, data)

        # Should return same data
        assert result is data

    def test_preprocess_returns_data_unchanged_for_digital_domain(self, engine):
        """Test data passes through unchanged for DIGITAL domain."""
        data = np.array([0, 1, 0, 1, 1, 0])

        result = engine._preprocess_for_domain(AnalysisDomain.DIGITAL, data)

        assert result is data

    def test_preprocess_calls_eye_preprocessing_for_eye_domain(self, engine):
        """Test EYE domain triggers eye diagram preprocessing."""
        data = np.sin(np.linspace(0, 2 * np.pi, 1000))

        with patch.object(engine, "_preprocess_for_eye_domain") as mock_preprocess:
            mock_preprocess.return_value = MagicMock()
            engine._preprocess_for_domain(AnalysisDomain.EYE, data)

            mock_preprocess.assert_called_once_with(data)

    def test_preprocess_returns_data_unchanged_for_other_domains(self, engine):
        """Test data passes through for domains without special preprocessing."""
        data = np.random.randn(100)

        for domain in [
            AnalysisDomain.SPECTRAL,
            AnalysisDomain.PATTERNS,
            AnalysisDomain.JITTER,
        ]:
            result = engine._preprocess_for_domain(domain, data)
            # Should return same object
            assert result is data


@pytest.mark.unit
class TestPreprocessForEyeDomain:
    """Test AnalysisEngine._preprocess_for_eye_domain method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AnalysisEngine(AnalysisConfig())

    def test_preprocess_eye_returns_existing_eye_diagram(self, engine):
        """Test that existing EyeDiagram passes through unchanged."""
        # Create mock EyeDiagram
        mock_eye = MagicMock()
        mock_eye.samples_per_ui = 100
        mock_eye.time_axis = np.linspace(0, 2, 200)

        result = engine._preprocess_for_eye_domain(mock_eye)

        # Should return same object
        assert result is mock_eye

    def test_preprocess_eye_from_waveform_trace(self, engine):
        """Test generating eye diagram from WaveformTrace."""
        # Create a simple waveform with periodic pattern
        sample_rate = 1e9  # 1 GHz
        t = np.linspace(0, 1e-6, 1000)  # 1 microsecond
        data = np.sin(2 * np.pi * 100e6 * t)  # 100 MHz sine

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        with patch("oscura.analyzers.eye.diagram.generate_eye") as mock_generate:
            mock_eye = MagicMock()
            mock_eye.samples_per_ui = 100
            mock_eye.time_axis = np.linspace(0, 2, 200)
            mock_generate.return_value = mock_eye

            result = engine._preprocess_for_eye_domain(trace)

            # Should call generate_eye
            mock_generate.assert_called_once()
            assert result is mock_eye

    def test_preprocess_eye_from_numpy_array(self, engine):
        """Test generating eye diagram from numpy array."""
        data = np.sin(np.linspace(0, 20 * np.pi, 1000))

        with patch("oscura.analyzers.eye.diagram.generate_eye") as mock_generate:
            mock_eye = MagicMock()
            mock_eye.samples_per_ui = 100
            mock_eye.n_traces = 10
            mock_generate.return_value = mock_eye

            result = engine._preprocess_for_eye_domain(data)

            # Should call generate_eye
            mock_generate.assert_called_once()
            # Result should be the mocked eye diagram
            assert result is mock_eye

    def test_preprocess_eye_handles_insufficient_data(self, engine):
        """Test that insufficient data returns original data."""
        # Very short data that won't generate good eye
        data = np.array([1, 2, 3])

        result = engine._preprocess_for_eye_domain(data)

        # Should return original data on failure
        assert result is data

    def test_preprocess_eye_returns_data_on_exception(self, engine):
        """Test that exceptions during eye generation return original data."""
        data = np.sin(np.linspace(0, 2 * np.pi, 100))

        with patch("oscura.analyzers.eye.diagram.generate_eye") as mock_generate:
            mock_generate.side_effect = Exception("Test error")

            result = engine._preprocess_for_eye_domain(data)

            # Should return original data
            np.testing.assert_array_equal(result, data)

    def test_preprocess_eye_detects_unit_interval_from_zero_crossings(self, engine):
        """Test unit interval detection from signal zero crossings."""
        # Create periodic signal with known period
        sample_rate = 1e9
        freq = 100e6  # 100 MHz = 10ns period
        t = np.linspace(0, 1e-6, 1000)
        data = np.sin(2 * np.pi * freq * t)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        with patch("oscura.analyzers.eye.diagram.generate_eye") as mock_generate:
            mock_generate.return_value = MagicMock()
            engine._preprocess_for_eye_domain(trace)

            # Verify generate_eye was called with reasonable unit_interval
            call_args = mock_generate.call_args
            assert call_args is not None
            ui = call_args[1]["unit_interval"]
            # Unit interval should be in reasonable range (1ns to 100ns)
            assert 1e-9 < ui < 100e-9

    def test_preprocess_eye_handles_empty_data(self, engine):
        """Test handling of empty data."""
        data = np.array([])

        result = engine._preprocess_for_eye_domain(data)

        # Should return original empty data
        np.testing.assert_array_equal(result, data)

    def test_preprocess_eye_handles_none_data(self, engine):
        """Test handling of None data."""
        mock_trace = MagicMock()
        mock_trace.data = None
        mock_trace.metadata = MagicMock()

        result = engine._preprocess_for_eye_domain(mock_trace)

        # Should return original object
        assert result is mock_trace

    def test_preprocess_eye_uses_histogram_generation(self, engine):
        """Test that eye preprocessing enables histogram generation."""
        sample_rate = 1e9
        data = np.sin(np.linspace(0, 20 * np.pi, 1000))

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        with patch("oscura.analyzers.eye.diagram.generate_eye") as mock_generate:
            mock_generate.return_value = MagicMock()
            engine._preprocess_for_eye_domain(trace)

            # Verify generate_histogram=True was passed
            call_args = mock_generate.call_args
            assert call_args is not None
            assert call_args[1]["generate_histogram"] is True


@pytest.mark.unit
class TestPrepareArguments:
    """Test AnalysisEngine._prepare_arguments method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return AnalysisEngine(AnalysisConfig())

    def test_prepare_arguments_for_eye_diagram(self, engine, arg_preparer):
        """Test argument preparation for EyeDiagram objects."""
        # Create mock EyeDiagram
        mock_eye = MagicMock()

    @pytest.fixture
    def arg_preparer(self):
        """Create ArgumentPreparer instance."""
        return ArgumentPreparer(input_path=None, default_sample_rate=1e6)

    def test_prepare_arguments_skips_non_eye_functions_for_eye_data(self, engine, arg_preparer):
        """Test that non-eye functions are skipped for EyeDiagram data."""
        mock_eye = MagicMock()
        mock_eye.samples_per_ui = 100
        mock_eye.time_axis = np.linspace(0, 2, 200)

        mock_func = MagicMock()

        with patch("inspect.signature") as mock_sig:
            # Mock signature with "data" parameter (not "eye")
            mock_param = MagicMock()
            mock_sig.return_value.parameters = {"data": mock_param}

            args, kwargs = arg_preparer.prepare_arguments(mock_func, mock_eye)

            # Should return None (skip this function)
            assert args is None

    def test_prepare_arguments_for_s_parameters(self, engine, arg_preparer):
        """Test argument preparation for SParameterData objects."""
        # Create SParameterData
        freqs = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 2, 2), dtype=np.complex128)
        s_params = SParameterData(frequencies=freqs, s_matrix=s_matrix, n_ports=2)

        mock_func = MagicMock()

        with patch("inspect.signature") as mock_sig:
            # Mock signature with "s_params" parameter
            mock_param = MagicMock()
            mock_sig.return_value.parameters = {"s_params": mock_param}

            args, kwargs = arg_preparer.prepare_arguments(mock_func, s_params)

            # Should pass s_params as first argument
            assert args is not None
            assert len(args) == 1
            assert args[0] is s_params

    def test_prepare_arguments_for_s_params_variations(self, engine, arg_preparer):
        """Test various S-parameter function signatures."""
        freqs = np.array([1e9, 2e9])
        s_matrix = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params = SParameterData(frequencies=freqs, s_matrix=s_matrix, n_ports=2)

        mock_func = MagicMock()

        # Test different parameter names
        for param_name in ["s_params", "s_param", "s_data", "sparams"]:
            with patch("inspect.signature") as mock_sig:
                mock_param = MagicMock()
                mock_sig.return_value.parameters = {param_name: mock_param}

                args, kwargs = arg_preparer.prepare_arguments(mock_func, s_params)

                assert args is not None
                assert len(args) == 1
                assert args[0] is s_params

    def test_prepare_arguments_skips_non_sparam_functions_for_sparam_data(
        self, engine, arg_preparer
    ):
        """Test that non-S-parameter functions are skipped for S-parameter data."""
        freqs = np.array([1e9])
        s_matrix = np.zeros((1, 2, 2), dtype=np.complex128)
        s_params = SParameterData(frequencies=freqs, s_matrix=s_matrix, n_ports=2)

        mock_func = MagicMock()

        with patch("inspect.signature") as mock_sig:
            # Mock signature with incompatible parameter
            mock_param = MagicMock()
            mock_sig.return_value.parameters = {"data": mock_param}

            args, kwargs = arg_preparer.prepare_arguments(mock_func, s_params)

            # Should return None (skip this function)
            assert args is None

    def test_prepare_arguments_for_waveform_trace(self, engine, arg_preparer):
        """Test argument preparation for WaveformTrace objects."""
        metadata = TraceMetadata(sample_rate=1e9)
        trace = WaveformTrace(data=np.sin(np.linspace(0, 2 * np.pi, 100)), metadata=metadata)

        mock_func = MagicMock()

        with patch("inspect.signature") as mock_sig:
            mock_param = MagicMock()
            mock_param.annotation = "WaveformTrace"
            mock_sig.return_value.parameters = {"trace": mock_param}

            args, kwargs = arg_preparer.prepare_arguments(mock_func, trace)

            # Should pass trace directly
            assert args is not None
            assert args[0] is trace

    def test_prepare_arguments_extracts_data_from_trace(self, engine, arg_preparer):
        """Test extracting raw data from trace for data-parameter functions."""
        metadata = TraceMetadata(sample_rate=1e9)
        data = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = WaveformTrace(data=data, metadata=metadata)

        mock_func = MagicMock()

        with patch("inspect.signature") as mock_sig:
            mock_param = MagicMock()
            mock_param.annotation = None
            mock_sig.return_value.parameters = {"data": mock_param}

            args, kwargs = arg_preparer.prepare_arguments(mock_func, trace)

            # Should pass raw data array
            assert args is not None
            np.testing.assert_array_equal(args[0], data)

    def test_prepare_arguments_includes_sample_rate(self, engine, arg_preparer):
        """Test that sample_rate is included in kwargs when needed."""
        metadata = TraceMetadata(sample_rate=2.5e9)
        trace = WaveformTrace(data=np.array([1, 2, 3]), metadata=metadata)

        mock_func = MagicMock()

        with patch("inspect.signature") as mock_sig:
            mock_param = MagicMock()
            mock_sig.return_value.parameters = {"data": mock_param, "sample_rate": mock_param}

            args, kwargs = arg_preparer.prepare_arguments(mock_func, trace)

            # Should include sample_rate in kwargs
            assert "sample_rate" in kwargs
            assert kwargs["sample_rate"] == 2.5e9


@pytest.mark.unit
class TestIntegrationEyePreprocessing:
    """Integration tests for EYE domain preprocessing with real data."""

    def test_preprocess_sine_wave_for_eye(self):
        """Test preprocessing a clean sine wave for eye diagram analysis."""
        # Load a clean waveform from test data
        test_file = Path("test_data/comprehensive_validation/waveform/clean_sine_100hz.npz")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        # Load the test data
        from oscura.loaders import load

        trace = load(test_file)

        engine = AnalysisEngine(AnalysisConfig())

        # Preprocess for EYE domain
        result = engine._preprocess_for_eye_domain(trace)

        # Result should be either an EyeDiagram or the original trace
        # (depends on whether generation succeeded)
        assert result is not None

    def test_preprocess_noisy_signal_for_eye(self):
        """Test preprocessing a noisy signal for eye diagram."""
        test_file = Path("test_data/comprehensive_validation/waveform/noisy_20db.npz")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        from oscura.loaders import load

        trace = load(test_file)

        engine = AnalysisEngine(AnalysisConfig())
        result = engine._preprocess_for_eye_domain(trace)

        assert result is not None


@pytest.mark.unit
class TestIntegrationSParameterDetection:
    """Integration tests for S-parameter input type detection."""

    def test_detect_input_type_from_s2p_path(self):
        """Test detecting SPARAMS input type from .s2p extension."""
        engine = AnalysisEngine()

        test_file = Path("test_data/signal_integrity/touchstone/transmission_line_50ohm.s2p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        from oscura.loaders import load
        from oscura.reporting.config import InputType

        s_params = load(test_file)

        # Detect input type
        input_type = engine.detect_input_type(test_file, s_params)

        assert input_type == InputType.SPARAMS

    def test_detect_input_type_from_s4p_path(self):
        """Test detecting SPARAMS input type from .s4p extension."""
        engine = AnalysisEngine()

        test_file = Path("test_data/signal_integrity/touchstone/differential_pair.s4p")

        if not test_file.exists():
            pytest.skip(f"Test data file not found: {test_file}")

        from oscura.reporting.config import InputType

        # Test with just path
        input_type = engine.detect_input_type(test_file, None)

        assert input_type == InputType.SPARAMS

    def test_detect_input_type_from_s_parameter_data_object(self):
        """Test detecting SPARAMS from SParameterData object."""
        engine = AnalysisEngine()

        from oscura.reporting.config import InputType

        # Create SParameterData
        freqs = np.array([1e9, 2e9])
        s_matrix = np.zeros((2, 2, 2), dtype=np.complex128)
        s_params = SParameterData(frequencies=freqs, s_matrix=s_matrix, n_ports=2)

        # Detect from object characteristics
        input_type = engine.detect_input_type(None, s_params)

        assert input_type == InputType.SPARAMS
