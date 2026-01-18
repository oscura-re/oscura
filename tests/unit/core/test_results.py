"""Tests for analysis result classes.

Tests the result objects with intermediate data access (API-005).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.results import (
    AnalysisResult,
    FFTResult,
    FilterResult,
    MeasurementResult,
    WaveletResult,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.core]


@pytest.fixture
def sample_trace() -> WaveformTrace:
    """Create sample waveform trace."""
    data = np.sin(2 * np.pi * 1000 * np.linspace(0, 0.01, 1000))
    metadata = TraceMetadata(sample_rate=100000.0)
    return WaveformTrace(data=data, metadata=metadata)


class TestAnalysisResult:
    """Test AnalysisResult base class."""

    def test_create_with_value(self) -> None:
        """Test creating result with value only."""
        result = AnalysisResult(value=42.5)
        assert result.value == 42.5
        assert result.intermediates == {}
        assert result.metadata == {}

    def test_create_with_intermediates(self) -> None:
        """Test creating result with intermediates."""
        intermediates = {"fft": np.array([1, 2, 3]), "freq": np.array([0, 1, 2])}
        result = AnalysisResult(value=10, intermediates=intermediates)
        assert result.value == 10
        assert "fft" in result.intermediates
        assert "freq" in result.intermediates

    def test_create_with_metadata(self) -> None:
        """Test creating result with metadata."""
        metadata = {"algorithm": "welch", "window": "hann"}
        result = AnalysisResult(value=5.0, metadata=metadata)
        assert result.metadata["algorithm"] == "welch"
        assert result.metadata["window"] == "hann"

    def test_get_intermediate_success(self) -> None:
        """Test getting existing intermediate."""
        intermediates = {"data": np.array([1, 2, 3])}
        result = AnalysisResult(value=1, intermediates=intermediates)
        data = result.get_intermediate("data")
        np.testing.assert_array_equal(data, np.array([1, 2, 3]))

    def test_get_intermediate_missing(self) -> None:
        """Test getting missing intermediate raises KeyError."""
        result = AnalysisResult(value=1, intermediates={"a": 1})
        with pytest.raises(KeyError, match="Intermediate 'missing' not found"):
            result.get_intermediate("missing")

    def test_has_intermediate(self) -> None:
        """Test checking if intermediate exists."""
        result = AnalysisResult(value=1, intermediates={"x": 10})
        assert result.has_intermediate("x")
        assert not result.has_intermediate("y")

    def test_list_intermediates(self) -> None:
        """Test listing intermediate keys."""
        intermediates = {"a": 1, "b": 2, "c": 3}
        result = AnalysisResult(value=0, intermediates=intermediates)
        keys = result.list_intermediates()
        assert set(keys) == {"a", "b", "c"}


class TestFFTResult:
    """Test FFTResult class."""

    def test_create_with_fft_data(self) -> None:
        """Test creating FFT result."""
        spectrum = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        frequencies = np.array([0.0, 1.0, 2.0])
        power = np.array([5.0, 25.0, 61.0])
        phase = np.array([0.0, 0.5, 1.0])

        result = FFTResult(
            value=spectrum,
            spectrum=spectrum,
            frequencies=frequencies,
            power=power,
            phase=phase,
        )

        np.testing.assert_array_equal(result.spectrum, spectrum)
        np.testing.assert_array_equal(result.frequencies, frequencies)
        np.testing.assert_array_equal(result.power, power)
        np.testing.assert_array_equal(result.phase, phase)

    def test_intermediates_populated(self) -> None:
        """Test that intermediates are auto-populated."""
        spectrum = np.array([1 + 0j, 2 + 0j])
        frequencies = np.array([0.0, 1.0])
        power = np.array([1.0, 4.0])
        phase = np.array([0.0, 0.0])

        result = FFTResult(
            value=None,
            spectrum=spectrum,
            frequencies=frequencies,
            power=power,
            phase=phase,
        )

        assert result.has_intermediate("spectrum")
        assert result.has_intermediate("frequencies")
        assert result.has_intermediate("power")
        assert result.has_intermediate("phase")

    def test_value_defaults_to_spectrum(self) -> None:
        """Test that value defaults to spectrum if None."""
        spectrum = np.array([1 + 2j, 3 + 4j])
        result = FFTResult(
            value=None,
            spectrum=spectrum,
            frequencies=np.array([0, 1]),
            power=np.array([5, 25]),
            phase=np.array([0, 0.5]),
        )
        np.testing.assert_array_equal(result.value, spectrum)

    def test_peak_frequency(self) -> None:
        """Test peak frequency calculation."""
        spectrum = np.array([1 + 0j, 5 + 0j, 2 + 0j])
        frequencies = np.array([100.0, 200.0, 300.0])
        power = np.array([1.0, 25.0, 4.0])  # Peak at 200 Hz

        result = FFTResult(
            value=None,
            spectrum=spectrum,
            frequencies=frequencies,
            power=power,
            phase=np.zeros(3),
        )

        assert result.peak_frequency == 200.0

    def test_peak_frequency_empty(self) -> None:
        """Test peak frequency with empty arrays."""
        result = FFTResult(value=None)
        assert result.peak_frequency == 0.0

    def test_magnitude(self) -> None:
        """Test magnitude spectrum calculation."""
        spectrum = np.array([3 + 4j, 0 + 1j, 1 + 0j])
        result = FFTResult(
            value=None,
            spectrum=spectrum,
            frequencies=np.array([0, 1, 2]),
            power=np.array([25, 1, 1]),
            phase=np.zeros(3),
        )

        expected_mag = np.array([5.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result.magnitude, expected_mag)

    def test_with_trace(self, sample_trace: WaveformTrace) -> None:
        """Test FFT result with original trace."""
        spectrum = np.fft.rfft(sample_trace.data)
        result = FFTResult(
            value=None,
            spectrum=spectrum,
            frequencies=np.fft.rfftfreq(
                len(sample_trace.data), 1 / sample_trace.metadata.sample_rate
            ),
            power=np.abs(spectrum) ** 2,
            phase=np.angle(spectrum),
            trace=sample_trace,
        )

        assert result.has_intermediate("trace")
        assert result.trace == sample_trace


class TestFilterResult:
    """Test FilterResult class."""

    def test_create_with_trace(self, sample_trace: WaveformTrace) -> None:
        """Test creating filter result with trace."""
        result = FilterResult(value=None, trace=sample_trace)
        assert result.trace == sample_trace
        assert result.has_intermediate("trace")

    def test_with_transfer_function(self) -> None:
        """Test filter result with transfer function."""
        freq = np.array([0, 100, 200])
        response = np.array([1.0, 0.7, 0.1])
        result = FilterResult(value=None, trace=None, frequency_response=(freq, response))
        assert result.frequency_response[0] is freq
        assert result.frequency_response[1] is response
        assert result.has_intermediate("frequency_response")

    def test_with_coefficients(self) -> None:
        """Test filter result with coefficients."""
        coeffs = np.array([[1.0, 0.5], [0.3, 0.2]])
        result = FilterResult(value=None, trace=None, filter_coefficients=coeffs)
        np.testing.assert_array_equal(result.filter_coefficients, coeffs)
        assert result.has_intermediate("filter_coefficients")

    def test_with_complex_transfer_function(self) -> None:
        """Test filter result with complex transfer function."""
        freq = np.array([0, 100, 200])
        H = np.array([1.0 + 0j, 0.7 + 0.1j, 0.1 + 0.05j])
        result = FilterResult(value=None, trace=None, transfer_function=(freq, H))
        assert result.transfer_function[0] is freq
        assert result.transfer_function[1] is H
        assert result.has_intermediate("transfer_function")

    def test_with_impulse_response(self) -> None:
        """Test filter result with impulse response."""
        impulse = np.array([1.0, 0.5, 0.25, 0.125])
        result = FilterResult(value=None, trace=None, impulse_response=impulse)
        np.testing.assert_array_equal(result.impulse_response, impulse)
        assert result.has_intermediate("impulse_response")


class TestWaveletResult:
    """Test WaveletResult class."""

    def test_create_with_coefficients(self) -> None:
        """Test creating wavelet result with coefficients."""
        coeffs = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        scales = np.array([1.0, 2.0, 4.0])
        result = WaveletResult(value=None, coeffs=coeffs, scales=scales)
        np.testing.assert_array_equal(result.coeffs, coeffs)
        np.testing.assert_array_equal(result.scales, scales)

    def test_intermediates_populated(self) -> None:
        """Test wavelet intermediates are populated."""
        coeffs = np.array([1 + 2j, 3 + 4j])
        scales = np.array([1.0])
        result = WaveletResult(value=None, coeffs=coeffs, scales=scales)
        assert result.has_intermediate("coeffs")
        assert result.has_intermediate("scales")

    def test_with_frequencies(self) -> None:
        """Test wavelet result with frequencies."""
        frequencies = np.array([0, 100, 200, 400])
        result = WaveletResult(value=None, frequencies=frequencies)
        np.testing.assert_array_equal(result.frequencies, frequencies)
        assert result.has_intermediate("frequencies")

    def test_with_trace(self, sample_trace: WaveformTrace) -> None:
        """Test wavelet result with trace."""
        result = WaveletResult(value=None, trace=sample_trace)
        assert result.trace == sample_trace
        assert result.has_intermediate("trace")

    def test_value_defaults_to_coeffs(self) -> None:
        """Test that value defaults to coeffs."""
        coeffs = np.array([1 + 0j, 2 + 0j])
        result = WaveletResult(value=None, coeffs=coeffs, scales=np.array([1, 2]))
        np.testing.assert_array_equal(result.value, coeffs)


class TestMeasurementResult:
    """Test MeasurementResult class."""

    def test_create_with_measurement(self) -> None:
        """Test creating measurement result."""
        result = MeasurementResult(value=3.3, units="V", method="vpp")
        assert result.value == 3.3
        assert result.units == "V"
        assert result.method == "vpp"

    def test_with_confidence(self) -> None:
        """Test measurement with confidence interval."""
        result = MeasurementResult(
            value=100.5, units="ns", method="rise_time", confidence=(99.0, 102.0)
        )
        assert result.confidence == (99.0, 102.0)
        assert result.metadata["confidence"] == (99.0, 102.0)

    def test_with_parameters(self) -> None:
        """Test measurement with parameters."""
        params = {"window": (0, 1e-3), "threshold": 0.5}
        result = MeasurementResult(value=42.0, units="Hz", method="frequency", parameters=params)
        assert result.parameters["window"] == (0, 1e-3)
        assert result.parameters["threshold"] == 0.5

    def test_str_representation(self) -> None:
        """Test string representation."""
        result = MeasurementResult(value=3.3, units="V")
        assert str(result) == "3.3 V"

    def test_str_without_units(self) -> None:
        """Test string representation without units."""
        result = MeasurementResult(value=42)
        assert str(result) == "42"

    def test_repr_representation(self) -> None:
        """Test repr representation."""
        result = MeasurementResult(value=3.3, units="V", method="vpp")
        repr_str = repr(result)
        assert "value=3.3" in repr_str
        assert "units='V'" in repr_str
        assert "method='vpp'" in repr_str

    def test_default_units_none(self) -> None:
        """Test that units defaults to None."""
        result = MeasurementResult(value=10)
        assert result.units is None


class TestCoreResultsIntegration:
    """Integration tests for result classes."""

    def test_nested_results(self) -> None:
        """Test storing one result as intermediate of another."""
        fft_result = FFTResult(
            value=None,
            spectrum=np.array([1 + 0j, 2 + 0j]),
            frequencies=np.array([0, 1]),
            power=np.array([1, 4]),
            phase=np.array([0, 0]),
        )

        measurement_result = MeasurementResult(
            value=fft_result.peak_frequency,
            units="Hz",
            method="dominant_frequency",
        )
        measurement_result.intermediates["fft_result"] = fft_result

        assert measurement_result.has_intermediate("fft_result")
        nested_fft = measurement_result.get_intermediate("fft_result")
        assert isinstance(nested_fft, FFTResult)

    def test_full_fft_workflow(self, sample_trace: WaveformTrace) -> None:
        """Test complete FFT analysis workflow."""
        # Simulate FFT computation
        spectrum = np.fft.rfft(sample_trace.data)
        freqs = np.fft.rfftfreq(len(sample_trace.data), 1 / sample_trace.metadata.sample_rate)
        power = np.abs(spectrum) ** 2
        phase = np.angle(spectrum)

        # Create result
        result = FFTResult(
            value=None,
            spectrum=spectrum,
            frequencies=freqs,
            power=power,
            phase=phase,
            trace=sample_trace,
        )

        # Access various aspects
        assert result.peak_frequency > 0
        assert len(result.magnitude) == len(spectrum)
        assert result.has_intermediate("trace")

    def test_measurement_from_filter(self, sample_trace: WaveformTrace) -> None:
        """Test creating measurement from filter result."""
        # Simulate filtered trace
        filtered_data = sample_trace.data * 0.7
        filtered_metadata = TraceMetadata(sample_rate=sample_trace.metadata.sample_rate)
        filtered_trace = WaveformTrace(data=filtered_data, metadata=filtered_metadata)

        filter_result = FilterResult(value=None, trace=filtered_trace)

        # Measure on filtered result
        measurement = MeasurementResult(
            value=filtered_trace.data.std(),
            units="V",
            method="rms",
        )
        measurement.intermediates["filter_result"] = filter_result

        assert measurement.has_intermediate("filter_result")
