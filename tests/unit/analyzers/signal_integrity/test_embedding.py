"""Unit tests for channel embedding and de-embedding.

This module tests the embedding.py module functionality:
- deembed: Remove fixture effects from measurements
- embed: Apply channel effects to signals
- cascade_deembed: De-embed multiple fixtures in sequence
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace

if TYPE_CHECKING:
    from oscura.analyzers.signal_integrity.sparams import SParameterData

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


# Helper functions
def make_waveform_trace(data: np.ndarray, sample_rate: float = 1e9) -> WaveformTrace:
    """Create a WaveformTrace from raw data for testing."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def make_s_parameter_data(
    n_freq: int = 100,
    n_ports: int = 2,
    z0: float = 50.0,
) -> SParameterData:
    """Create synthetic S-parameter data for testing."""
    from oscura.analyzers.signal_integrity.sparams import SParameterData

    frequencies = np.linspace(1e6, 10e9, n_freq)
    s_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=np.complex128)

    # Create realistic S-parameters
    for i in range(n_freq):
        freq = frequencies[i]
        # S11 (return loss) - starts low, increases with frequency
        s11_mag = 0.05 + 0.1 * (freq / 10e9)
        s11_phase = -np.pi / 4 * (freq / 10e9)
        s_matrix[i, 0, 0] = s11_mag * np.exp(1j * s11_phase)

        # S21 (insertion loss) - decreases with frequency
        s21_mag = 0.9 - 0.2 * (freq / 10e9)
        s21_phase = -np.pi * freq / 5e9
        s_matrix[i, 1, 0] = s21_mag * np.exp(1j * s21_phase)

        # S12 (reverse transmission) - assume reciprocal
        s_matrix[i, 0, 1] = s_matrix[i, 1, 0]

        # S22 (output return loss)
        s_matrix[i, 1, 1] = s_matrix[i, 0, 0] * 0.9

    return SParameterData(
        frequencies=frequencies,
        s_matrix=s_matrix,
        n_ports=n_ports,
        z0=z0,
    )


# =============================================================================
# De-embedding Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-002")
class TestDeembed:
    """Test de-embedding functionality."""

    def test_deembed_frequency_domain(self) -> None:
        """Test frequency domain de-embedding."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        # Create a simple test signal
        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)  # 1 GHz sine wave

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=100)

        result = deembed(trace, s_params, method="frequency_domain")

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples
        assert result.metadata.sample_rate == sample_rate

    def test_deembed_time_domain(self) -> None:
        """Test time domain de-embedding."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=100)

        result = deembed(trace, s_params, method="time_domain")

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples

    def test_deembed_invalid_method(self) -> None:
        """Test that invalid method raises ValueError."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)
        s_params = make_s_parameter_data(n_freq=50)

        with pytest.raises(ValueError, match="Unknown method"):
            deembed(trace, s_params, method="invalid_method")

    def test_deembed_requires_2port(self) -> None:
        """Test that de-embedding requires 2-port S-parameters."""
        from oscura.analyzers.signal_integrity.embedding import deembed
        from oscura.analyzers.signal_integrity.sparams import SParameterData
        from oscura.core.exceptions import AnalysisError

        # Create 1-port S-parameters
        frequencies = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 1, 1), dtype=np.complex128)

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=1,
        )

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        with pytest.raises(AnalysisError, match="2-port"):
            deembed(trace, s_params)


# =============================================================================
# Embedding Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("SI-003")
class TestEmbed:
    """Test channel embedding functionality."""

    def test_embed_basic(self) -> None:
        """Test basic channel embedding."""
        from oscura.analyzers.signal_integrity.embedding import embed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=100)

        result = embed(trace, s_params)

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples
        assert result.metadata.sample_rate == sample_rate

    def test_embed_attenuates_signal(self) -> None:
        """Test that embedding reduces signal amplitude (typical behavior)."""
        from oscura.analyzers.signal_integrity.embedding import embed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        # Use S-parameters with significant loss
        s_params = make_s_parameter_data(n_freq=100)

        result = embed(trace, s_params)

        # RMS of embedded signal should be less than original
        original_rms = np.sqrt(np.mean(signal**2))
        embedded_rms = np.sqrt(np.mean(result.data**2))

        assert embedded_rms < original_rms

    def test_embed_requires_2port(self) -> None:
        """Test that embedding requires 2-port S-parameters."""
        from oscura.analyzers.signal_integrity.embedding import embed
        from oscura.analyzers.signal_integrity.sparams import SParameterData
        from oscura.core.exceptions import AnalysisError

        frequencies = np.array([1e9, 2e9, 3e9])
        s_matrix = np.zeros((3, 1, 1), dtype=np.complex128)

        s_params = SParameterData(
            frequencies=frequencies,
            s_matrix=s_matrix,
            n_ports=1,
        )

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        with pytest.raises(AnalysisError, match="2-port"):
            embed(trace, s_params)


# =============================================================================
# Cascade De-embedding Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestCascadeDeembed:
    """Test cascade de-embedding functionality."""

    def test_cascade_deembed_empty_list(self) -> None:
        """Test cascade de-embedding with empty fixture list."""
        from oscura.analyzers.signal_integrity.embedding import cascade_deembed

        signal = np.sin(np.linspace(0, 2 * np.pi, 100))
        trace = make_waveform_trace(signal, sample_rate=1e9)

        result = cascade_deembed(trace, [])

        # With no fixtures, should return original trace
        np.testing.assert_array_equal(result.data, trace.data)

    def test_cascade_deembed_single_fixture(self) -> None:
        """Test cascade de-embedding with single fixture."""
        from oscura.analyzers.signal_integrity.embedding import cascade_deembed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        fixture = make_s_parameter_data(n_freq=100)

        result = cascade_deembed(trace, [fixture])

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples

    def test_cascade_deembed_multiple_fixtures(self) -> None:
        """Test cascade de-embedding with multiple fixtures."""
        from oscura.analyzers.signal_integrity.embedding import cascade_deembed

        n_samples = 1000
        sample_rate = 10e9
        t = np.arange(n_samples) / sample_rate
        signal = np.sin(2 * np.pi * 1e9 * t)

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        fixture1 = make_s_parameter_data(n_freq=100)
        fixture2 = make_s_parameter_data(n_freq=100)

        result = cascade_deembed(trace, [fixture1, fixture2])

        assert isinstance(result, WaveformTrace)
        assert len(result.data) == n_samples


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestEmbeddingEdgeCases:
    """Test edge cases in embedding/de-embedding."""

    def test_very_short_signal_embed(self) -> None:
        """Test embedding with very short signal."""
        from oscura.analyzers.signal_integrity.embedding import embed

        signal = np.array([1.0, -1.0, 1.0])
        trace = make_waveform_trace(signal, sample_rate=10e9)
        s_params = make_s_parameter_data(n_freq=50)

        result = embed(trace, s_params)

        assert len(result.data) == 3

    def test_high_frequency_content_deembed(self) -> None:
        """Test de-embedding with high frequency content."""
        from oscura.analyzers.signal_integrity.embedding import deembed

        n_samples = 1000
        sample_rate = 50e9
        t = np.arange(n_samples) / sample_rate
        # Signal with multiple frequency components
        signal = (
            np.sin(2 * np.pi * 1e9 * t)
            + 0.5 * np.sin(2 * np.pi * 5e9 * t)
            + 0.25 * np.sin(2 * np.pi * 10e9 * t)
        )

        trace = make_waveform_trace(signal, sample_rate=sample_rate)
        s_params = make_s_parameter_data(n_freq=200)

        result = deembed(trace, s_params)

        assert len(result.data) == n_samples
        assert np.all(np.isfinite(result.data))
