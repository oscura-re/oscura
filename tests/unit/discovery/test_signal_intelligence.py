import pytest

"""Tests for signal classification and measurement intelligence.

This module tests the signal intelligence features that help users understand
signal characteristics and measurement suitability.

Requirements tested:
"""

import numpy as np

import oscura as osc
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


class TestSignalClassification:
    """Test signal type classification functionality."""

    def test_classify_dc_signal(self):
        """Test classification of DC signal."""
        # Create DC signal with minimal noise
        data = np.ones(1000) * 3.3 + np.random.normal(0, 0.001, 1000)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.classify_signal(trace)

        assert result["type"] == "dc"
        assert "constant" in result["characteristics"]
        assert result["dc_component"] is True
        assert result["frequency_estimate"] is None
        assert result["confidence"] > 0.9

    def test_classify_digital_signal(self):
        """Test classification of digital square wave."""
        # Create digital square wave with multiple periods
        # 10 periods at 1kHz = 10ms, sampled at 1MHz = 10000 samples
        t = np.linspace(0, 10e-3, 10000)
        freq = 1e3
        data = np.where(np.sin(2 * np.pi * freq * t) > 0, 3.3, 0.0)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.classify_signal(trace)

        assert result["type"] == "digital"
        assert "digital_levels" in result["characteristics"]
        assert "periodic" in result["characteristics"]
        assert result["frequency_estimate"] is not None
        assert abs(result["frequency_estimate"] - freq) < freq * 0.2  # Within 20%
        assert result["levels"] is not None
        assert "low" in result["levels"]
        assert "high" in result["levels"]

    def test_classify_analog_signal(self):
        """Test classification of analog sine wave."""
        # Create clean sine wave with multiple periods
        t = np.linspace(0, 10e-3, 10000)
        freq = 1e3
        data = np.sin(2 * np.pi * freq * t)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.classify_signal(trace)

        assert result["type"] == "analog"
        assert "periodic" in result["characteristics"]
        assert "clean" in result["characteristics"] or "low_noise" in result["characteristics"]
        assert result["frequency_estimate"] is not None
        assert abs(result["frequency_estimate"] - freq) < freq * 0.3  # Within 30%

    def test_classify_noisy_signal(self):
        """Test classification of noisy signal."""
        # Create noisy sine wave with multiple periods
        # Noise amplitude = 0.5, signal amplitude = 2.0, noise ratio = 0.25 (25%)
        t = np.linspace(0, 10e-3, 10000)
        freq = 1e3
        signal = np.sin(2 * np.pi * freq * t)
        noise = np.random.normal(0, 0.5, len(signal))
        data = signal + noise
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.classify_signal(trace)

        assert result["type"] in ("analog", "mixed")
        # With 25% noise ratio, should detect some level of noise
        assert any(
            noise_level in result["characteristics"]
            for noise_level in ["low_noise", "moderate_noise", "noisy"]
        )
        assert result["noise_level"] > 0.1

    def test_classify_aperiodic_signal(self):
        """Test classification of aperiodic/transient signal."""
        # Create single pulse
        data = np.zeros(1000)
        data[400:600] = 1.0
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.classify_signal(trace)

        assert "aperiodic" in result["characteristics"] or "transient" in result["characteristics"]
        # Frequency estimate may be None for aperiodic signal

    def test_classify_insufficient_data(self):
        """Test classification with insufficient data."""
        data = np.array([1.0, 2.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.classify_signal(trace)

        assert result["type"] == "unknown"
        assert "insufficient_data" in result["characteristics"]
        assert result["confidence"] == 0.0


class TestSignalQuality:
    """Test signal quality assessment functionality."""

    def test_assess_clean_signal(self):
        """Test quality assessment of clean signal."""
        # Create clean sine wave
        t = np.linspace(0, 1e-3, 10000)
        data = np.sin(2 * np.pi * 1e3 * t)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.assess_signal_quality(trace)

        assert result["snr"] is not None
        assert result["snr"] > 40  # Clean signal should have high SNR
        assert result["clipping"] is False
        assert result["saturation"] is False
        assert len(result["warnings"]) == 0

    def test_assess_clipped_signal(self):
        """Test detection of clipped signal."""
        # Create clipped sine wave
        t = np.linspace(0, 1e-3, 10000)
        data = np.clip(np.sin(2 * np.pi * 1e3 * t) * 2, -1.0, 1.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.assess_signal_quality(trace)

        assert result["clipping"] is True
        assert any("clipping" in w.lower() for w in result["warnings"])

    def test_assess_noisy_signal(self):
        """Test quality assessment of noisy signal."""
        # Create noisy signal
        t = np.linspace(0, 1e-3, 10000)
        signal = np.sin(2 * np.pi * 1e3 * t)
        noise = np.random.normal(0, 0.3, len(signal))
        data = signal + noise
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.assess_signal_quality(trace)

        assert result["snr"] is not None
        assert result["snr"] < 20  # Noisy signal should have low SNR
        assert result["noise_level"] > 0.1

    def test_assess_saturated_signal(self):
        """Test detection of saturated signal."""
        # Create saturated signal (stuck at one level)
        data = np.ones(1000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.assess_signal_quality(trace)

        assert result["saturation"] is True
        assert any("saturation" in w.lower() for w in result["warnings"])

    def test_assess_low_resolution_signal(self):
        """Test detection of quantization issues."""
        # Create signal with coarse quantization
        t = np.linspace(0, 1e-3, 10000)
        data = np.round(np.sin(2 * np.pi * 1e3 * t) * 10) / 10  # Only 20 levels
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.assess_signal_quality(trace)

        # Should detect low resolution
        assert any("resolution" in w.lower() for w in result["warnings"])

    def test_assess_insufficient_sample_rate(self):
        """Test detection of insufficient sample rate."""
        # Create 1 MHz signal sampled at 3 MHz (< 10x oversampling)
        t = np.linspace(0, 1e-3, 3000)
        data = np.sin(2 * np.pi * 1e6 * t)
        metadata = TraceMetadata(sample_rate=3e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.assess_signal_quality(trace)

        # Should warn about sample rate
        assert any("sample rate" in w.lower() for w in result["warnings"])

    def test_assess_crest_factor(self):
        """Test crest factor calculation."""
        # Create signal with high crest factor (pulsed)
        data = np.zeros(1000)
        data[450:550] = 10.0  # Short high pulse
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.assess_signal_quality(trace)

        assert result["crest_factor"] is not None
        assert result["crest_factor"] > 3  # High crest factor for pulsed signal


class TestMeasurementSuitability:
    """Test measurement suitability checking."""

    def test_frequency_unsuitable_for_dc(self):
        """Test that frequency measurement is unsuitable for DC signal."""
        # DC signal
        data = np.ones(1000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "frequency")

        assert result["suitable"] is False
        assert result["expected_result"] == "nan"
        assert any("dc" in w.lower() for w in result["warnings"])
        assert len(result["suggestions"]) > 0

    def test_rise_time_unsuitable_for_dc(self):
        """Test that rise_time measurement is unsuitable for DC signal."""
        # DC signal
        data = np.ones(1000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "rise_time")

        assert result["suitable"] is False
        assert result["expected_result"] == "nan"
        assert any("transition" in w.lower() or "edge" in w.lower() for w in result["warnings"])

    def test_frequency_suitable_for_periodic(self):
        """Test that frequency measurement is suitable for periodic signal."""
        # Periodic square wave
        t = np.linspace(0, 1e-3, 10000)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "frequency")

        assert result["suitable"] is True
        assert result["expected_result"] == "valid"

    def test_rise_time_suitable_for_digital(self):
        """Test that rise_time is suitable for digital signal with edges."""
        # Digital signal with edges
        t = np.linspace(0, 1e-3, 10000)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "rise_time")

        assert result["suitable"] is True
        assert result["expected_result"] in ("valid", "unreliable")

    def test_overshoot_warning_for_digital(self):
        """Test that overshoot measurement gives warning for digital signal."""
        # Clean digital signal (no overshoot expected)
        t = np.linspace(0, 1e-3, 10000)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "overshoot")

        # Should give warning that overshoot is designed for analog
        assert len(result["warnings"]) > 0 or result["expected_result"] == "unreliable"

    def test_clipping_affects_measurements(self):
        """Test that clipping affects measurement suitability."""
        # Clipped signal
        t = np.linspace(0, 1e-3, 10000)
        data = np.clip(np.sin(2 * np.pi * 1e3 * t) * 2, -1.0, 1.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "amplitude")

        assert any("clipping" in w.lower() for w in result["warnings"])

    def test_insufficient_data_for_fft(self):
        """Test that FFT is unsuitable for very short signals."""
        # Short signal
        data = np.sin(2 * np.pi * 1e3 * np.linspace(0, 1e-4, 100))
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "fft")

        assert result["expected_result"] == "unreliable"
        assert any("length" in w.lower() or "short" in w.lower() for w in result["warnings"])

    def test_low_sample_rate_warning(self):
        """Test warning for insufficient sample rate."""
        # 1 MHz signal sampled at 3 MHz
        t = np.linspace(0, 1e-3, 3000)
        data = np.sin(2 * np.pi * 1e6 * t)
        metadata = TraceMetadata(sample_rate=3e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "frequency")

        assert any("sample rate" in w.lower() for w in result["warnings"])

    def test_mean_always_suitable(self):
        """Test that mean measurement is always suitable."""
        # Any signal
        data = np.random.randn(1000)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = osc.check_measurement_suitability(trace, "mean")

        assert result["suitable"] is True


class TestMeasurementSuggestions:
    """Test smart measurement suggestions."""

    def test_suggestions_for_dc_signal(self):
        """Test measurement suggestions for DC signal."""
        # DC signal
        data = np.ones(1000) * 3.3
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        suggestions = osc.suggest_measurements(trace)

        # Should suggest statistical measurements
        names = [s["name"] for s in suggestions]
        assert "mean" in names
        assert "rms" in names
        # Should NOT suggest frequency or edges
        assert "frequency" not in names
        assert "rise_time" not in names

    def test_suggestions_for_periodic_digital(self):
        """Test measurement suggestions for periodic digital signal."""
        # Periodic square wave
        t = np.linspace(0, 1e-3, 10000)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        suggestions = osc.suggest_measurements(trace)

        names = [s["name"] for s in suggestions]
        # Should suggest frequency, timing, and edge measurements
        assert "frequency" in names
        assert "rise_time" in names or "fall_time" in names
        assert "duty_cycle" in names
        assert "amplitude" in names

    def test_suggestions_for_clean_periodic(self):
        """Test suggestions for clean periodic signal include spectral."""
        # Clean sine wave
        t = np.linspace(0, 1e-3, 10000)
        data = np.sin(2 * np.pi * 1e3 * t)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        suggestions = osc.suggest_measurements(trace)

        names = [s["name"] for s in suggestions]
        # Clean periodic signals should suggest spectral analysis
        assert "thd" in names or "snr" in names

    def test_suggestions_priority_ranking(self):
        """Test that suggestions are ranked by priority."""
        # Any signal
        t = np.linspace(0, 1e-3, 10000)
        data = np.sin(2 * np.pi * 1e3 * t)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        suggestions = osc.suggest_measurements(trace)

        # Check priorities are sequential
        priorities = [s["priority"] for s in suggestions]
        assert priorities == sorted(priorities)

    def test_suggestions_include_rationale(self):
        """Test that suggestions include rationale."""
        # Periodic signal
        t = np.linspace(0, 1e-3, 10000)
        data = np.sin(2 * np.pi * 1e3 * t)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        suggestions = osc.suggest_measurements(trace)

        # All suggestions should have rationale
        for s in suggestions:
            assert "rationale" in s
            assert len(s["rationale"]) > 0
            assert "confidence" in s
            assert 0.0 <= s["confidence"] <= 1.0

    def test_max_suggestions_limit(self):
        """Test that max_suggestions parameter works."""
        # Signal that would generate many suggestions
        t = np.linspace(0, 1e-3, 10000)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        suggestions = osc.suggest_measurements(trace, max_suggestions=3)

        assert len(suggestions) <= 3


class TestIntegrationScenarios:
    """Test realistic usage scenarios."""

    def test_troubleshoot_nan_frequency_on_dc(self):
        """Simulate troubleshooting NaN frequency result on DC signal."""
        # User tries to measure frequency on DC signal
        data = np.ones(1000) * 5.0
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Check why frequency gives NaN
        classification = osc.classify_signal(trace)
        assert classification["type"] == "dc"

        suitability = osc.check_measurement_suitability(trace, "frequency")
        assert suitability["suitable"] is False
        assert suitability["expected_result"] == "nan"

        # Get better suggestions
        suggestions = osc.suggest_measurements(trace)
        assert "mean" in [s["name"] for s in suggestions]

    def test_detect_clipping_issue(self):
        """Simulate detecting clipping affecting measurements."""
        # User has clipped signal
        t = np.linspace(0, 1e-3, 10000)
        data = np.clip(np.sin(2 * np.pi * 1e3 * t) * 2, -1.0, 1.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        quality = osc.assess_signal_quality(trace)
        assert quality["clipping"] is True
        assert len(quality["warnings"]) > 0

        # Check measurement reliability
        suitability = osc.check_measurement_suitability(trace, "amplitude")
        assert any("clipping" in w.lower() for w in suitability["warnings"])

    def test_understand_signal_characteristics(self):
        """Simulate understanding unknown signal characteristics."""
        # User loads unknown signal
        t = np.linspace(0, 1e-3, 10000)
        data = np.where(np.sin(2 * np.pi * 1e3 * t) > 0, 3.3, 0.0)
        metadata = TraceMetadata(sample_rate=10e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        # Classify signal
        classification = osc.classify_signal(trace)
        assert classification["type"] == "digital"
        assert "periodic" in classification["characteristics"]
        assert classification["frequency_estimate"] is not None

        # Get quality metrics
        quality = osc.assess_signal_quality(trace)
        assert quality["snr"] is not None

        # Get suitable measurements
        suggestions = osc.suggest_measurements(trace)
        assert len(suggestions) > 0
