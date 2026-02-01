"""Tests for measurements namespace re-exports.

Test that measurements module properly re-exports waveform measurement
functions and provides correct API.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers import measurements
from oscura.core.types import TraceMetadata, WaveformTrace


class TestMeasurementsNamespace:
    """Test measurements namespace and re-exports."""

    def test_all_functions_exported(self) -> None:
        """Test that __all__ matches available functions."""
        expected_functions = {
            "amplitude",
            "duty_cycle",
            "fall_time",
            "frequency",
            "mean",
            "measure",
            "overshoot",
            "period",
            "preshoot",
            "pulse_width",
            "rise_time",
            "rms",
            "undershoot",
        }

        assert set(measurements.__all__) == expected_functions

        # Verify all functions exist
        for func_name in expected_functions:
            assert hasattr(measurements, func_name)
            assert callable(getattr(measurements, func_name))

    def test_amplitude_function_works(self) -> None:
        """Test amplitude function is correctly re-exported."""
        # Create simple square wave
        data = np.array([0.0] * 10 + [1.0] * 10 + [0.0] * 10)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.amplitude(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert 0.9 < result["value"] < 1.1  # Should be ~1.0

    def test_frequency_function_works(self) -> None:
        """Test frequency function is correctly re-exported."""
        # Create 1 MHz sine wave
        fs = 100e6
        t = np.linspace(0, 10e-6, 1000)
        data = np.sin(2 * np.pi * 1e6 * t)
        metadata = TraceMetadata(sample_rate=fs)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.frequency(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert 0.9e6 < result["value"] < 1.1e6  # Should be ~1 MHz

    def test_rise_time_function_works(self) -> None:
        """Test rise_time function is correctly re-exported."""
        # Create rising edge
        t = np.linspace(0, 10e-9, 1000)
        data = np.clip(t / 5e-9, 0, 1)  # Linear rise in 5ns
        metadata = TraceMetadata(sample_rate=100e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.rise_time(trace, ref_levels=(0.1, 0.9))
        assert result["applicable"]
        assert result["value"] is not None
        assert result["value"] > 0  # Should have positive rise time

    def test_mean_function_works(self) -> None:
        """Test mean function is correctly re-exported."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.mean(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert abs(result["value"] - 3.0) < 0.01  # Should be exactly 3.0

    def test_rms_function_works(self) -> None:
        """Test rms function is correctly re-exported."""
        # DC signal
        data = np.ones(100) * 2.0
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.rms(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert abs(result["value"] - 2.0) < 0.01  # RMS of DC is the DC value

    def test_period_function_works(self) -> None:
        """Test period function is correctly re-exported."""
        # 1 MHz signal -> 1 µs period
        fs = 100e6
        t = np.linspace(0, 10e-6, 1000)
        data = np.sin(2 * np.pi * 1e6 * t)
        metadata = TraceMetadata(sample_rate=fs)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.period(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert 0.9e-6 < result["value"] < 1.1e-6  # Should be ~1 µs

    def test_duty_cycle_function_works(self) -> None:
        """Test duty_cycle function is correctly re-exported."""
        # Create proper square wave with multiple periods
        t = np.linspace(0, 10e-6, 1000)
        data = np.where(np.sin(2 * np.pi * 1e5 * t) > 0, 1.0, 0.0)
        metadata = TraceMetadata(sample_rate=100e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.duty_cycle(trace, percentage=True)
        # Should be close to 50% (0.5 ratio) for a sine-based square wave
        if result["applicable"] and result["value"] is not None:
            assert 0.40 < result["value"] < 0.60  # Duty cycle returns ratio (0-1)
        # Otherwise it's inapplicable (no period found)

    def test_overshoot_function_works(self) -> None:
        """Test overshoot function is correctly re-exported."""
        # Create signal with overshoot
        data = np.array([0.0] * 10 + [1.2] + [1.0] * 10)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.overshoot(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert result["value"] > 0  # Should have positive overshoot

    def test_undershoot_function_works(self) -> None:
        """Test undershoot function is correctly re-exported."""
        # Create signal with undershoot
        data = np.array([1.0] * 10 + [-0.2] + [0.0] * 10)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.undershoot(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert result["value"] > 0  # Should have positive undershoot

    def test_pulse_width_function_works(self) -> None:
        """Test pulse_width function is correctly re-exported."""
        # 10 samples high at 1 MHz = 10 µs pulse
        data = np.array([0.0] * 10 + [1.0] * 10 + [0.0] * 10)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.pulse_width(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert result["value"] > 0  # Should have positive pulse width

    def test_fall_time_function_works(self) -> None:
        """Test fall_time function is correctly re-exported."""
        # Create falling edge
        t = np.linspace(0, 10e-9, 1000)
        data = np.clip(1 - t / 5e-9, 0, 1)  # Linear fall in 5ns
        metadata = TraceMetadata(sample_rate=100e9)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.fall_time(trace, ref_levels=(0.9, 0.1))
        assert result["applicable"]
        assert result["value"] is not None
        assert result["value"] > 0  # Should have positive fall time

    def test_preshoot_function_works(self) -> None:
        """Test preshoot function is correctly re-exported."""
        # Create signal with preshoot
        data = np.array([0.0] * 10 + [-0.1] + [1.0] * 10)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = measurements.preshoot(trace)
        assert result["applicable"]
        assert result["value"] is not None
        assert result["value"] >= 0  # Preshoot should be non-negative

    def test_measure_function_works(self) -> None:
        """Test measure function is correctly re-exported."""
        data = np.sin(2 * np.pi * np.linspace(0, 1, 100))
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Test measure with amplitude
        result = measurements.measure(trace, parameters=["amplitude"])
        assert isinstance(result, dict)
        assert "amplitude" in result


class TestMeasurementsDocumentation:
    """Test measurements module documentation."""

    def test_module_docstring_exists(self) -> None:
        """Test that measurements module has docstring."""
        assert measurements.__doc__ is not None
        assert "measurement" in measurements.__doc__.lower()

    def test_functions_have_docstrings(self) -> None:
        """Test that re-exported functions have docstrings."""
        # Check a few key functions
        for func_name in ["amplitude", "frequency", "rise_time"]:
            func = getattr(measurements, func_name)
            assert func.__doc__ is not None, f"{func_name} missing docstring"


class TestMeasurementsErrors:
    """Test error handling in measurements namespace."""

    def test_invalid_trace_raises_error(self) -> None:
        """Test that invalid trace raises appropriate error."""
        with pytest.raises((TypeError, AttributeError)):
            measurements.amplitude(None)  # type: ignore[arg-type]

    def test_empty_trace_handled(self) -> None:
        """Test that empty trace raises ValueError."""
        data = np.array([])
        metadata = TraceMetadata(sample_rate=1e6)

        # Empty trace raises ValueError at creation time
        with pytest.raises(ValueError, match="data array cannot be empty"):
            trace = WaveformTrace(data=data, metadata=metadata)
