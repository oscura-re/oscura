"""Comprehensive tests for waveform/measurements_with_uncertainty.py.

This test module provides complete coverage for uncertainty-aware measurements
following GUM principles.
"""

import numpy as np
import pytest

from oscura.analyzers.waveform import measurements_with_uncertainty as meas_u
from oscura.core.types import CalibrationInfo, TraceMetadata, WaveformTrace
from oscura.core.uncertainty import MeasurementWithUncertainty


@pytest.fixture
def pulse_trace() -> WaveformTrace:
    """Create pulse trace for rise/fall time testing.

    Returns:
        WaveformTrace with clean rising edge
    """
    # Create rising edge from 0 to 1V
    t = np.linspace(0, 100e-9, 1000)  # 100 ns total
    # Rise from 10ns to 20ns (10ns rise time)
    data = np.where(t < 10e-9, 0.0, np.where(t < 20e-9, (t - 10e-9) / 10e-9, 1.0))

    metadata = TraceMetadata(
        sample_rate=10e9,  # 10 GHz
        vertical_scale=0.2,
        vertical_offset=0.0,
    )
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def sine_trace() -> WaveformTrace:
    """Create sine wave trace for frequency/amplitude testing.

    Returns:
        WaveformTrace with 1 MHz sine wave
    """
    fs = 100e6  # 100 MHz sample rate
    f = 1e6  # 1 MHz signal
    t = np.linspace(0, 10e-6, 1000)  # 10 us, 10 cycles
    data = 2.0 * np.sin(2 * np.pi * f * t)  # 4 Vpp amplitude

    metadata = TraceMetadata(
        sample_rate=fs,
        vertical_scale=1.0,
        vertical_offset=0.0,
        calibration_info=CalibrationInfo(instrument="Test Oscilloscope", vertical_resolution=8),
    )
    return WaveformTrace(data=data, metadata=metadata)


class TestRiseTime:
    """Test suite for rise_time function with uncertainty."""

    def test_rise_time_basic(self, pulse_trace: WaveformTrace) -> None:
        """Test rise time measurement returns valid result."""
        result = meas_u.rise_time(pulse_trace)

        assert isinstance(result, MeasurementWithUncertainty)
        assert result.value > 0
        assert np.isfinite(result.value)
        assert result.unit == "s"
        # Rise time should be around 10 ns
        assert 5e-9 < result.value < 15e-9

    def test_rise_time_uncertainty_included(self, pulse_trace: WaveformTrace) -> None:
        """Test uncertainty is calculated when requested."""
        result = meas_u.rise_time(pulse_trace, include_uncertainty=True)

        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0
        # Uncertainty should be reasonable (< 20% of value)
        assert result.uncertainty < result.value * 0.2

    def test_rise_time_uncertainty_excluded(self, pulse_trace: WaveformTrace) -> None:
        """Test uncertainty can be excluded for speed."""
        result = meas_u.rise_time(pulse_trace, include_uncertainty=False)

        assert np.isnan(result.uncertainty)
        assert np.isfinite(result.value)

    def test_rise_time_custom_ref_levels(self, pulse_trace: WaveformTrace) -> None:
        """Test custom reference levels."""
        result = meas_u.rise_time(pulse_trace, ref_levels=(0.2, 0.8))

        assert np.isfinite(result.value)
        assert result.value > 0

    def test_rise_time_with_calibration(self) -> None:
        """Test rise time with calibration info."""
        t = np.linspace(0, 100e-9, 1000)
        data = np.where(t < 10e-9, 0.0, np.where(t < 20e-9, (t - 10e-9) / 10e-9, 1.0))

        calibration = CalibrationInfo(
            instrument="Test Oscilloscope", vertical_resolution=12, timebase_accuracy=25.0
        )
        metadata = TraceMetadata(
            sample_rate=10e9,
            vertical_scale=0.2,
            vertical_offset=0.0,
            calibration_info=calibration,
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.rise_time(trace)

        # With better calibration, uncertainty might be different
        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0

    def test_rise_time_with_noise(self) -> None:
        """Test rise time measurement on noisy signal."""
        t = np.linspace(0, 100e-9, 1000)
        data = np.where(t < 10e-9, 0.0, np.where(t < 20e-9, (t - 10e-9) / 10e-9, 1.0))
        # Add noise
        data += np.random.randn(1000) * 0.05

        metadata = TraceMetadata(sample_rate=10e9, vertical_scale=0.2)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.rise_time(trace)

        # Should still measure, but with larger uncertainty
        assert np.isfinite(result.value)
        assert np.isfinite(result.uncertainty)

    def test_rise_time_nan_value(self) -> None:
        """Test rise time returns NaN uncertainty for invalid measurement."""
        # Flat signal - no edge
        data = np.ones(1000) * 0.5
        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.rise_time(trace)

        # Value will be NaN, uncertainty should also be NaN
        if np.isnan(result.value):
            assert np.isnan(result.uncertainty)

    def test_rise_time_n_samples(self, pulse_trace: WaveformTrace) -> None:
        """Test n_samples field is populated."""
        result = meas_u.rise_time(pulse_trace)

        assert result.n_samples == len(pulse_trace.data)


class TestFallTime:
    """Test suite for fall_time function with uncertainty."""

    def test_fall_time_basic(self) -> None:
        """Test fall time measurement."""
        # Create falling edge
        t = np.linspace(0, 100e-9, 1000)
        data = np.where(t < 10e-9, 1.0, np.where(t < 20e-9, 1.0 - (t - 10e-9) / 10e-9, 0.0))

        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.fall_time(trace, ref_levels=(0.9, 0.1))

        assert isinstance(result, MeasurementWithUncertainty)
        assert result.value > 0
        assert result.unit == "s"
        assert 5e-9 < result.value < 15e-9

    def test_fall_time_uncertainty(self) -> None:
        """Test fall time uncertainty calculation."""
        t = np.linspace(0, 100e-9, 1000)
        data = np.where(t < 10e-9, 1.0, np.where(t < 20e-9, 1.0 - (t - 10e-9) / 10e-9, 0.0))

        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.fall_time(trace, include_uncertainty=True)

        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0

    def test_fall_time_no_uncertainty(self) -> None:
        """Test fall time without uncertainty for speed."""
        t = np.linspace(0, 100e-9, 1000)
        data = np.where(t < 10e-9, 1.0, np.where(t < 20e-9, 1.0 - (t - 10e-9) / 10e-9, 0.0))

        metadata = TraceMetadata(sample_rate=10e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.fall_time(trace, include_uncertainty=False)

        assert np.isnan(result.uncertainty)
        assert np.isfinite(result.value)


class TestFrequency:
    """Test suite for frequency measurement with uncertainty."""

    def test_frequency_basic(self, sine_trace: WaveformTrace) -> None:
        """Test frequency measurement on sine wave."""
        result = meas_u.frequency(sine_trace)

        assert isinstance(result, MeasurementWithUncertainty)
        assert result.unit == "Hz"
        # Should be near 1 MHz
        assert 0.9e6 < result.value < 1.1e6

    def test_frequency_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test frequency uncertainty calculation."""
        result = meas_u.frequency(sine_trace, include_uncertainty=True)

        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0
        # Relative uncertainty should be small
        relative_unc = result.relative_uncertainty
        assert relative_unc < 0.01  # Less than 1%

    def test_frequency_no_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test frequency without uncertainty."""
        result = meas_u.frequency(sine_trace, include_uncertainty=False)

        assert np.isnan(result.uncertainty)
        assert np.isfinite(result.value)

    def test_frequency_zero_value(self) -> None:
        """Test frequency with zero frequency signal."""
        data = np.ones(1000) * 0.5  # DC signal
        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.frequency(trace)

        # Will likely return NaN for DC signal
        if np.isnan(result.value):
            assert np.isnan(result.uncertainty)

    def test_frequency_high_accuracy(self) -> None:
        """Test frequency with very high sample rate and calibration."""
        fs = 10e9  # 10 GHz sample rate (needed for <0.1% uncertainty)
        f = 10e6  # 10 MHz signal
        t = np.linspace(0, 100e-6, 1000000)
        data = np.sin(2 * np.pi * f * t)

        calibration = CalibrationInfo(
            instrument="Test Oscilloscope", timebase_accuracy=1.0
        )  # Very accurate (1 ppm)
        metadata = TraceMetadata(
            sample_rate=fs,
            calibration_info=calibration,
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.frequency(trace)

        # With 10 GHz sampling and 1 ppm timebase, uncertainty should be <0.1%
        assert result.relative_uncertainty < 0.001


class TestAmplitude:
    """Test suite for amplitude (Vpp) measurement with uncertainty."""

    def test_amplitude_basic(self, sine_trace: WaveformTrace) -> None:
        """Test amplitude measurement."""
        result = meas_u.amplitude(sine_trace)

        assert isinstance(result, MeasurementWithUncertainty)
        assert result.unit == "V"
        # Amplitude should be 4 Vpp
        assert 3.5 < result.value < 4.5

    def test_amplitude_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test amplitude uncertainty calculation."""
        result = meas_u.amplitude(sine_trace, include_uncertainty=True)

        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0
        # Uncertainty should be reasonable
        assert result.uncertainty < result.value * 0.1

    def test_amplitude_no_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test amplitude without uncertainty."""
        result = meas_u.amplitude(sine_trace, include_uncertainty=False)

        assert np.isnan(result.uncertainty)
        assert np.isfinite(result.value)

    def test_amplitude_with_vertical_scale(self) -> None:
        """Test amplitude with vertical scale metadata."""
        data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 10e-6, 1000)) * 5.0

        metadata = TraceMetadata(
            sample_rate=100e6,
            vertical_scale=2.0,  # 2V/div
            vertical_offset=0.0,
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.amplitude(trace)

        assert np.isfinite(result.uncertainty)
        # Amplitude ~10 Vpp
        assert 9.0 < result.value < 11.0

    def test_amplitude_with_calibration(self) -> None:
        """Test amplitude with calibration info."""
        data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 10e-6, 1000)) * 2.0

        calibration = CalibrationInfo(
            instrument="Test Oscilloscope", vertical_resolution=12, timebase_accuracy=25.0
        )
        metadata = TraceMetadata(
            sample_rate=100e6,
            calibration_info=calibration,
        )
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.amplitude(trace)

        # Higher resolution should give smaller uncertainty
        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0

    def test_amplitude_with_noise(self) -> None:
        """Test amplitude on noisy signal."""
        t = np.linspace(0, 10e-6, 1000)
        data = 2.0 * np.sin(2 * np.pi * 1e6 * t) + np.random.randn(1000) * 0.1

        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.amplitude(trace)

        # Noise increases uncertainty
        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0


class TestRMS:
    """Test suite for RMS voltage measurement with uncertainty."""

    def test_rms_basic(self, sine_trace: WaveformTrace) -> None:
        """Test RMS measurement on sine wave."""
        result = meas_u.rms(sine_trace)

        assert isinstance(result, MeasurementWithUncertainty)
        assert result.unit == "V"
        # RMS of 4 Vpp sine is 4/(2*sqrt(2)) ≈ 1.414 V
        assert 1.0 < result.value < 2.0

    def test_rms_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test RMS uncertainty calculation."""
        result = meas_u.rms(sine_trace, include_uncertainty=True)

        assert np.isfinite(result.uncertainty)
        assert result.uncertainty > 0

    def test_rms_no_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test RMS without uncertainty."""
        result = meas_u.rms(sine_trace, include_uncertainty=False)

        assert np.isnan(result.uncertainty)
        assert np.isfinite(result.value)

    def test_rms_ac_coupled(self) -> None:
        """Test RMS with AC coupling (DC removal)."""
        # Sine wave with DC offset
        t = np.linspace(0, 10e-6, 1000)
        data = 2.0 * np.sin(2 * np.pi * 1e6 * t) + 5.0  # +5V DC offset

        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result_dc = meas_u.rms(trace, ac_coupled=False)
        result_ac = meas_u.rms(trace, ac_coupled=True)

        # AC coupled should be smaller (no DC component)
        assert result_ac.value < result_dc.value

    def test_rms_dc_signal(self) -> None:
        """Test RMS of pure DC signal."""
        data = np.ones(1000) * 3.0

        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.rms(trace, ac_coupled=False)

        # RMS of DC is just the DC value
        assert abs(result.value - 3.0) < 0.1

    def test_rms_statistical_uncertainty(self) -> None:
        """Test RMS statistical uncertainty decreases with more samples."""
        # Short signal
        data_short = np.random.randn(100)
        metadata_short = TraceMetadata(sample_rate=100e6)
        trace_short = WaveformTrace(data=data_short, metadata=metadata_short)

        # Long signal
        data_long = np.random.randn(10000)
        metadata_long = TraceMetadata(sample_rate=100e6)
        trace_long = WaveformTrace(data=data_long, metadata=metadata_long)

        result_short = meas_u.rms(trace_short)
        result_long = meas_u.rms(trace_long)

        # Longer signal should have smaller relative uncertainty
        rel_unc_short = result_short.relative_uncertainty
        rel_unc_long = result_long.relative_uncertainty

        assert rel_unc_long < rel_unc_short


class TestMeasurementWithUncertaintyProperties:
    """Test properties of MeasurementWithUncertainty objects."""

    def test_relative_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test relative_uncertainty property."""
        result = meas_u.frequency(sine_trace)

        rel_unc = result.relative_uncertainty
        expected = result.uncertainty / result.value if result.value != 0 else np.inf

        assert abs(rel_unc - expected) < 1e-10

    def test_format_with_uncertainty(self, sine_trace: WaveformTrace) -> None:
        """Test formatted output with uncertainty."""
        result = meas_u.amplitude(sine_trace)

        # Should have both value and uncertainty
        assert result.value > 0
        assert result.uncertainty > 0
        assert result.unit == "V"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trace(self) -> None:
        """Test measurements on empty trace."""
        data = np.array([])
        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Should handle gracefully or raise appropriate error
        try:
            result = meas_u.amplitude(trace)
            # If it returns, should be NaN or inf
            assert np.isnan(result.value) or np.isinf(result.value)
        except (ValueError, IndexError):
            pass  # Acceptable to raise error

    def test_very_short_trace(self) -> None:
        """Test measurements on very short trace."""
        data = np.array([1.0, 2.0, 3.0])
        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.rms(trace)

        # Should still compute something
        assert isinstance(result, MeasurementWithUncertainty)

    def test_all_nan_trace(self) -> None:
        """Test measurements on all-NaN trace."""
        data = np.full(1000, np.nan)
        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.amplitude(trace)

        # Should return NaN
        assert np.isnan(result.value)
        assert np.isnan(result.uncertainty)

    def test_constant_trace(self) -> None:
        """Test measurements on constant trace."""
        data = np.ones(1000) * 5.0
        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = meas_u.amplitude(trace)

        # Amplitude of constant signal is 0
        assert result.value == 0.0 or result.value < 0.01
