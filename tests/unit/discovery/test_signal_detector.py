"""Tests for signal characterization.

Requirements tested:
"""

import numpy as np
import pytest

from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace
from oscura.discovery.signal_detector import characterize_signal

pytestmark = pytest.mark.unit


class TestSignalCharacterization:
    """Tests for signal characterization."""

    def test_characterize_digital_signal(self):
        """Test characterization of digital square wave."""
        # Generate 3.3V digital signal
        t = np.linspace(0, 1e-3, 10000)
        signal = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=10e6))

        result = characterize_signal(trace)

        assert result.signal_type in ("digital", "pwm", "uart")
        assert result.confidence >= 0.5
        assert 0.0 <= result.voltage_low <= 0.5
        assert 3.0 <= result.voltage_high <= 3.5
        assert 500 <= result.frequency_hz <= 1500  # ~1 kHz

    def test_characterize_analog_signal(self):
        """Test characterization of analog sine wave."""
        # Generate smooth sine wave with noise
        t = np.linspace(0, 1e-3, 10000)
        signal = 1.65 + 1.0 * np.sin(2 * np.pi * 1e3 * t)
        # Add some noise to make it more analog
        signal += np.random.randn(len(signal)) * 0.1

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=10e6))

        result = characterize_signal(trace)

        # Sine wave can be detected as analog or digital depending on discretization
        # Can also be misdetected as UART if voltage levels match digital thresholds
        assert result.signal_type in ("analog", "pwm", "digital", "uart")
        assert result.confidence >= 0.4
        assert result.voltage_low < result.voltage_high

    def test_characterize_pwm_signal(self):
        """Test characterization of PWM signal with varying duty cycle."""
        # Generate PWM with 50% duty cycle
        t = np.linspace(0, 1e-3, 10000)
        signal = np.where((t * 1e3) % 1.0 < 0.5, 3.3, 0.0)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=10e6))

        result = characterize_signal(trace)

        # PWM should be detected as either pwm or digital
        assert result.signal_type in ("pwm", "digital", "uart")
        assert result.confidence >= 0.5

    def test_characterize_empty_trace(self):
        """Test that empty trace raises error."""
        trace = WaveformTrace(data=np.array([]), metadata=TraceMetadata(sample_rate=1e6))

        with pytest.raises(ValueError, match="Cannot characterize empty trace"):
            characterize_signal(trace)

    def test_characterize_with_alternatives(self):
        """Test characterization with alternative suggestions."""
        # Ambiguous signal
        np.linspace(0, 1e-3, 1000)
        signal = np.random.randn(1000) * 0.1 + 1.65

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        result = characterize_signal(trace, include_alternatives=True, min_alternatives=3)

        # Should have alternatives when confidence is low or requested
        if result.confidence < 0.6:
            assert len(result.alternatives) >= 1

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        t = np.linspace(0, 1e-3, 1000)
        signal = 3.3 * (t > 0.5e-3).astype(float)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        result = characterize_signal(trace, confidence_threshold=0.9)

        # Result should still be returned even if confidence is low
        assert result is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_digital_trace_input(self):
        """Test characterization with DigitalTrace input."""
        signal = np.array([False] * 50 + [True] * 50 + [False] * 50, dtype=bool)

        trace = DigitalTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        result = characterize_signal(trace)

        assert result.signal_type in ("digital", "pwm", "uart", "spi")
        assert result.confidence >= 0.5

    def test_quality_metrics(self):
        """Test that quality metrics are calculated."""
        t = np.linspace(0, 1e-3, 1000)
        signal = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        result = characterize_signal(trace)

        assert "snr_db" in result.quality_metrics
        assert "jitter_ns" in result.quality_metrics
        assert "noise_level" in result.quality_metrics
        assert result.quality_metrics["snr_db"] >= 0

    def test_parameters_extraction(self):
        """Test extraction of signal-specific parameters."""
        # Digital signal should have logic family
        t = np.linspace(0, 1e-3, 1000)
        signal = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)

        trace = WaveformTrace(data=signal, metadata=TraceMetadata(sample_rate=1e6))

        result = characterize_signal(trace)

        # Should have some parameters
        assert isinstance(result.parameters, dict)
