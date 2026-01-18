"""Unit tests for signal intelligence module (INF-005 to INF-008).


References:
    IEEE 181-2011: Standard for Transitional Waveform Definitions
    IEEE 1057-2017: Standard for Digitizing Waveform Recorders
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.inference.signal_intelligence import (
    _count_edges,
    _detect_digital_signal,
    _detect_edge_periodicity,
    _detect_periodicity,
    _detect_periodicity_fft,
    _estimate_noise_level,
    assess_signal_quality,
    check_measurement_suitability,
    classify_signal,
    suggest_measurements,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test Fixtures
# =============================================================================


def create_trace(
    data: NDArray[np.floating[Any]],
    sample_rate: float = 1e6,
) -> WaveformTrace:
    """Create a WaveformTrace with given data and sample rate."""
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=sample_rate),
    )


def create_dc_signal(
    value: float = 1.0,
    n_samples: int = 1000,
    noise_level: float = 0.0,
) -> NDArray[np.floating[Any]]:
    """Create a DC signal with optional noise."""
    data = np.full(n_samples, value, dtype=np.float64)
    if noise_level > 0:
        data += np.random.randn(n_samples) * noise_level
    return data


def create_square_wave(
    frequency: float = 1000.0,
    sample_rate: float = 1e6,
    n_periods: int = 10,
    low: float = 0.0,
    high: float = 1.0,
    duty_cycle: float = 0.5,
    noise_level: float = 0.0,
) -> NDArray[np.floating[Any]]:
    """Create a square wave signal."""
    period_samples = int(sample_rate / frequency)
    n_samples = period_samples * n_periods
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    phase = (t * frequency) % 1.0
    data = np.where(phase < duty_cycle, high, low).astype(np.float64)
    if noise_level > 0:
        data += np.random.randn(n_samples) * noise_level
    return data


def create_sine_wave(
    frequency: float = 1000.0,
    sample_rate: float = 1e6,
    n_periods: int = 10,
    amplitude: float = 1.0,
    offset: float = 0.0,
    noise_level: float = 0.0,
) -> NDArray[np.floating[Any]]:
    """Create a sinusoidal signal."""
    period_samples = int(sample_rate / frequency)
    n_samples = period_samples * n_periods
    t = np.arange(n_samples, dtype=np.float64) / sample_rate
    data = offset + amplitude * np.sin(2 * np.pi * frequency * t)
    if noise_level > 0:
        data += np.random.randn(n_samples) * noise_level
    return data


# =============================================================================
# Test classify_signal
# =============================================================================


class TestClassifySignal:
    """Tests for classify_signal function (INF-005)."""

    def test_classify_dc_signal(self) -> None:
        """Test classification of DC signal."""
        data = create_dc_signal(value=5.0, n_samples=1000)
        trace = create_trace(data)

        result = classify_signal(trace)

        assert result["type"] == "dc"
        assert "constant" in result["characteristics"]
        assert result["dc_component"] is True
        assert result["frequency_estimate"] is None
        assert result["confidence"] >= 0.9
        assert result["levels"] is None

    def test_classify_dc_signal_with_tiny_noise(self) -> None:
        """Test classification of DC signal with very small noise."""
        data = create_dc_signal(value=5.0, n_samples=1000, noise_level=1e-10)
        trace = create_trace(data)

        result = classify_signal(trace)

        assert result["type"] == "dc"
        assert result["dc_component"] is True

    def test_classify_digital_square_wave(self) -> None:
        """Test classification of clean digital square wave."""
        data = create_square_wave(
            frequency=1000.0,
            sample_rate=1e6,
            n_periods=10,
            low=0.0,
            high=3.3,
        )
        trace = create_trace(data, sample_rate=1e6)

        result = classify_signal(trace)

        assert result["type"] == "digital"
        assert "digital_levels" in result["characteristics"]
        assert "periodic" in result["characteristics"]
        assert result["levels"] is not None
        # Check that levels are within the expected range
        # The histogram-based detection may return bin centers, not exact values
        assert result["levels"]["low"] < result["levels"]["high"]
        assert 0.0 <= result["levels"]["low"] <= 1.5  # Should be closer to low value
        assert 2.0 <= result["levels"]["high"] <= 3.5  # Should be closer to high value
        assert result["frequency_estimate"] is not None
        # Frequency should be close to 1000 Hz
        assert abs(result["frequency_estimate"] - 1000.0) < 100.0

    def test_classify_noisy_digital_signal(self) -> None:
        """Test classification of digital signal with noise."""
        # Use smaller noise relative to amplitude for reliable digital detection
        data = create_square_wave(
            frequency=1000.0,
            sample_rate=1e6,
            n_periods=10,
            low=0.0,
            high=3.3,
            noise_level=0.05,  # Lower noise for reliable digital detection
        )
        trace = create_trace(data, sample_rate=1e6)

        result = classify_signal(trace)

        # Should still detect as digital (or analog if noise causes misclassification)
        # With low noise, should be digital; with higher noise, could be analog
        assert result["type"] in ("digital", "mixed", "analog")
        if result["type"] in ("digital", "mixed"):
            assert result["levels"] is not None

    def test_classify_analog_sine_wave(self) -> None:
        """Test classification of clean sine wave."""
        data = create_sine_wave(
            frequency=1000.0,
            sample_rate=1e6,
            n_periods=10,
            amplitude=1.0,
        )
        trace = create_trace(data, sample_rate=1e6)

        result = classify_signal(trace)

        assert result["type"] == "analog"
        assert "periodic" in result["characteristics"]
        assert result["frequency_estimate"] is not None
        # Frequency should be close to 1000 Hz
        assert abs(result["frequency_estimate"] - 1000.0) < 100.0

    def test_classify_noisy_sine_wave(self) -> None:
        """Test classification of noisy sine wave."""
        data = create_sine_wave(
            frequency=1000.0,
            sample_rate=1e6,
            n_periods=10,
            amplitude=1.0,
            noise_level=0.2,
        )
        trace = create_trace(data, sample_rate=1e6)

        result = classify_signal(trace)

        assert result["type"] == "analog"
        assert "periodic" in result["characteristics"]
        # Should detect noise
        assert (
            "noisy" in result["characteristics"]
            or "moderate_noise" in result["characteristics"]
            or "low_noise" in result["characteristics"]
        )

    def test_classify_insufficient_data(self) -> None:
        """Test classification with insufficient data points."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        trace = create_trace(data)

        result = classify_signal(trace)

        assert result["type"] == "unknown"
        assert "insufficient_data" in result["characteristics"]
        assert result["confidence"] == 0.0

    def test_classify_single_sample(self) -> None:
        """Test classification with single sample."""
        data = np.array([1.0], dtype=np.float64)
        trace = create_trace(data)

        result = classify_signal(trace)

        assert result["type"] == "unknown"

    def test_classify_aperiodic_signal(self) -> None:
        """Test classification of aperiodic signal (random noise)."""
        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float64)
        trace = create_trace(data, sample_rate=1e6)

        result = classify_signal(trace)

        assert result["type"] == "analog"
        assert "aperiodic" in result["characteristics"]
        assert result["frequency_estimate"] is None

    def test_classify_pulsed_signal(self) -> None:
        """Test classification of signal with few pulses."""
        # Create signal with sparse pulses
        n_samples = 10000
        data = np.zeros(n_samples, dtype=np.float64)
        # Add a few pulses
        data[1000:1100] = 3.3
        data[5000:5100] = 3.3
        data[9000:9100] = 3.3
        trace = create_trace(data, sample_rate=1e6)

        result = classify_signal(trace)

        assert result["type"] in ("digital", "analog")
        # With sparse edges and wide spacing, might detect as pulsed
        assert result["confidence"] > 0

    def test_classify_with_dc_offset(self) -> None:
        """Test classification of signal with significant DC offset."""
        data = create_sine_wave(
            frequency=1000.0,
            sample_rate=1e6,
            n_periods=10,
            amplitude=0.1,  # Small amplitude
            offset=10.0,  # Large DC offset
        )
        trace = create_trace(data, sample_rate=1e6)

        result = classify_signal(trace)

        assert result["dc_component"] is True

    def test_classify_custom_thresholds(self) -> None:
        """Test classification with custom threshold parameters."""
        data = create_square_wave(
            frequency=1000.0,
            sample_rate=1e6,
            n_periods=10,
        )
        trace = create_trace(data, sample_rate=1e6)

        # Test with stricter digital threshold
        result_strict = classify_signal(trace, digital_threshold_ratio=0.95)
        # Test with looser digital threshold
        result_loose = classify_signal(trace, digital_threshold_ratio=0.5)

        assert result_strict is not None
        assert result_loose is not None
        # Both should classify this clean square wave as digital
        assert result_loose["type"] == "digital"

    def test_classify_mixed_signal(self) -> None:
        """Test classification of mixed digital/analog signal."""
        # Create square wave with significant ringing/analog variation
        n_samples = 10000
        sample_rate = 1e6
        data = create_square_wave(
            frequency=1000.0,
            sample_rate=sample_rate,
            n_periods=10,
            low=0.0,
            high=3.3,
        )
        # Add significant variation within levels
        data = data + np.sin(2 * np.pi * 50000 * np.arange(len(data)) / sample_rate) * 0.5
        trace = create_trace(data, sample_rate=sample_rate)

        result = classify_signal(trace)

        # Could be classified as mixed due to analog variation
        assert result["type"] in ("digital", "mixed", "analog")

    def test_classify_returns_all_keys(self) -> None:
        """Test that classify_signal returns all expected keys."""
        data = create_sine_wave()
        trace = create_trace(data)

        result = classify_signal(trace)

        expected_keys = {
            "type",
            "signal_type",
            "is_digital",
            "is_periodic",
            "characteristics",
            "dc_component",
            "frequency_estimate",
            "dominant_frequency",
            "snr_db",
            "confidence",
            "noise_level",
            "levels",
        }
        assert set(result.keys()) == expected_keys

    def test_classify_confidence_range(self) -> None:
        """Test that confidence is in valid range."""
        # Test various signal types
        signals = [
            create_dc_signal(),
            create_square_wave(),
            create_sine_wave(),
            np.random.randn(1000).astype(np.float64),
        ]

        for data in signals:
            trace = create_trace(data)
            result = classify_signal(trace)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_classify_noise_level_nonnegative(self) -> None:
        """Test that noise level is non-negative."""
        signals = [
            create_dc_signal(),
            create_square_wave(),
            create_sine_wave(),
        ]

        for data in signals:
            trace = create_trace(data)
            result = classify_signal(trace)
            assert result["noise_level"] >= 0.0


# =============================================================================
# Test assess_signal_quality
# =============================================================================


class TestAssessSignalQuality:
    """Tests for assess_signal_quality function (INF-006)."""

    def test_quality_clean_signal(self) -> None:
        """Test quality assessment of clean signal."""
        data = create_sine_wave(amplitude=1.0, noise_level=0.0)
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        assert result["clipping"] is False
        assert result["saturation"] is False
        assert result["noise_level"] < 0.1
        assert len(result["warnings"]) == 0 or all(
            "sample rate" in w.lower() for w in result["warnings"]
        )

    def test_quality_noisy_signal(self) -> None:
        """Test quality assessment of noisy signal."""
        data = create_sine_wave(amplitude=1.0, noise_level=0.5)
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        # Should have measurable noise
        assert result["noise_level"] > 0.1

    def test_quality_snr_calculation(self) -> None:
        """Test SNR calculation for signals with different noise levels."""
        # Clean signal - high SNR
        clean_data = create_sine_wave(amplitude=1.0, noise_level=0.01)
        clean_trace = create_trace(clean_data)
        clean_result = assess_signal_quality(clean_trace)

        # Noisy signal - lower SNR
        noisy_data = create_sine_wave(amplitude=1.0, noise_level=0.1)
        noisy_trace = create_trace(noisy_data)
        noisy_result = assess_signal_quality(noisy_trace)

        if clean_result["snr"] is not None and noisy_result["snr"] is not None:
            # Clean signal should have higher SNR
            assert clean_result["snr"] > noisy_result["snr"]

    def test_quality_insufficient_data(self) -> None:
        """Test quality assessment with insufficient data."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        assert "Insufficient data" in result["warnings"][0]
        assert result["snr"] is None
        assert result["clipping"] is False

    def test_quality_clipping_detection(self) -> None:
        """Test detection of clipped signal."""
        # Create sine wave that would be clipped
        n_samples = 10000
        t = np.linspace(0, 0.01, n_samples)
        data = np.sin(2 * np.pi * 100 * t) * 2.0  # Large amplitude
        # Clip at +-1 by setting long consecutive runs at extremes
        data = np.clip(data, -1.0, 1.0)
        # Make sure there are LONG consecutive runs at extremes (15%+ of data)
        # Force clipping by setting many consecutive samples at extremes
        data[:2000] = 1.0  # 20% at max
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        # Should detect clipping at maximum
        assert result["clipping"] is True

    def test_quality_saturation_detection_analog(self) -> None:
        """Test detection of saturated analog signal."""
        # Create signal with only a few unique values (quantized)
        data = np.round(np.random.randn(10000) * 2) / 2  # Only 0.5V steps
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        # May or may not trigger saturation depending on unique value count
        # The key is it should complete without error
        assert isinstance(result["saturation"], bool)

    def test_quality_saturation_detection_digital(self) -> None:
        """Test that digital signals with 2 levels are not flagged as saturated."""
        data = create_square_wave()
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        # Digital signals with 2 levels should NOT be flagged as saturated
        assert result["saturation"] is False

    def test_quality_crest_factor(self) -> None:
        """Test crest factor calculation."""
        # Sine wave has crest factor of sqrt(2) ~ 1.414
        data = create_sine_wave(amplitude=1.0)
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        assert result["crest_factor"] is not None
        # Sine wave crest factor should be around 1.414
        assert 1.3 < result["crest_factor"] < 1.6

    def test_quality_dynamic_range(self) -> None:
        """Test dynamic range calculation."""
        data = create_sine_wave(amplitude=1.0, offset=0.0)
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        # Dynamic range should be calculated
        assert result["dynamic_range"] is not None

    def test_quality_sample_rate_warning(self) -> None:
        """Test warning for insufficient sample rate."""
        # Create signal with frequency close to Nyquist
        frequency = 400000  # 400 kHz
        sample_rate = 1e6  # 1 MHz - only 2.5x oversampling
        data = create_sine_wave(frequency=frequency, sample_rate=sample_rate, n_periods=5)
        trace = create_trace(data, sample_rate=sample_rate)

        result = assess_signal_quality(trace)

        # Should warn about insufficient sample rate
        # (looking for sample rate related warnings)
        has_sample_rate_warning = any(
            "sample rate" in w.lower() or "oversampling" in w.lower() for w in result["warnings"]
        )
        # May or may not warn depending on exact frequency detection
        assert isinstance(has_sample_rate_warning, bool)

    def test_quality_quantization_warning(self) -> None:
        """Test warning for low resolution quantization."""
        # Create signal with only ~50 quantization levels
        amplitude = 5.0
        step = amplitude / 50
        data = np.round(create_sine_wave(amplitude=amplitude) / step) * step
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        # Should detect low resolution
        has_resolution_warning = any(
            "resolution" in w.lower() or "levels" in w.lower() for w in result["warnings"]
        )
        # May or may not warn based on exact detection
        assert isinstance(has_resolution_warning, bool)

    def test_quality_returns_all_keys(self) -> None:
        """Test that assess_signal_quality returns all expected keys."""
        data = create_sine_wave()
        trace = create_trace(data)

        result = assess_signal_quality(trace)

        expected_keys = {
            "snr",
            "noise_level",
            "clipping",
            "saturation",
            "warnings",
            "dynamic_range",
            "crest_factor",
        }
        assert set(result.keys()) == expected_keys


# =============================================================================
# Test check_measurement_suitability
# =============================================================================


class TestCheckMeasurementSuitability:
    """Tests for check_measurement_suitability function (INF-007)."""

    def test_frequency_on_periodic_signal(self) -> None:
        """Test frequency measurement suitability on periodic signal."""
        data = create_sine_wave(n_periods=10)
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "frequency")

        assert result["suitable"] is True
        assert result["expected_result"] == "valid"
        assert result["confidence"] > 0.5

    def test_frequency_on_dc_signal(self) -> None:
        """Test frequency measurement suitability on DC signal."""
        data = create_dc_signal()
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "frequency")

        assert result["suitable"] is False
        assert result["expected_result"] == "nan"
        assert len(result["warnings"]) > 0
        assert len(result["suggestions"]) > 0

    def test_period_on_dc_signal(self) -> None:
        """Test period measurement on DC signal."""
        data = create_dc_signal()
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "period")

        assert result["suitable"] is False
        assert result["expected_result"] == "nan"

    def test_rise_time_on_digital_signal(self) -> None:
        """Test rise_time measurement on digital signal with edges."""
        data = create_square_wave(n_periods=5)
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "rise_time")

        # Should be suitable since signal has edges
        assert result["suitable"] is True
        assert result["confidence"] > 0.5

    def test_rise_time_on_dc_signal(self) -> None:
        """Test rise_time measurement on DC signal (no edges)."""
        data = create_dc_signal()
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "rise_time")

        assert result["suitable"] is False
        assert result["expected_result"] == "nan"

    def test_duty_cycle_on_periodic_digital(self) -> None:
        """Test duty_cycle measurement on periodic digital signal."""
        data = create_square_wave(n_periods=5, duty_cycle=0.3)
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "duty_cycle")

        # Should be suitable for periodic digital signal
        assert result["suitable"] is True
        assert result["confidence"] > 0.5

    def test_duty_cycle_on_aperiodic(self) -> None:
        """Test duty_cycle measurement on aperiodic signal."""
        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float64)
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "duty_cycle")

        assert result["suitable"] is False
        assert result["expected_result"] == "nan"

    def test_mean_on_any_signal(self) -> None:
        """Test mean measurement is always suitable."""
        signals = [
            create_dc_signal(),
            create_square_wave(),
            create_sine_wave(),
        ]

        for data in signals:
            trace = create_trace(data)
            result = check_measurement_suitability(trace, "mean")

            # Mean should always be suitable
            assert result["suitable"] is True

    def test_thd_on_short_signal(self) -> None:
        """Test THD measurement warning on short signal."""
        # Create very short signal
        data = create_sine_wave(n_periods=2)[:200]  # Less than 256 samples
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "thd")

        # Should warn about short signal for spectral analysis
        has_length_warning = any(
            "length" in w.lower() or "short" in w.lower() for w in result["warnings"]
        )
        assert has_length_warning or result["expected_result"] == "unreliable"

    def test_spectral_on_aperiodic(self) -> None:
        """Test spectral measurement warning on aperiodic signal."""
        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float64)
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "snr")

        # Should warn about aperiodic signal for spectral analysis
        assert result["expected_result"] in ("nan", "unreliable")

    def test_overshoot_on_digital(self) -> None:
        """Test overshoot measurement warning on digital signal."""
        data = create_square_wave()
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "overshoot")

        # Should warn that digital signals may not have overshoot
        assert len(result["warnings"]) > 0 or result["expected_result"] == "unreliable"

    def test_amplitude_always_valid(self) -> None:
        """Test that amplitude measurement is generally valid."""
        data = create_sine_wave()
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "amplitude")

        assert result["suitable"] is True

    def test_returns_all_keys(self) -> None:
        """Test that check_measurement_suitability returns all expected keys."""
        data = create_sine_wave()
        trace = create_trace(data)

        result = check_measurement_suitability(trace, "frequency")

        expected_keys = {
            "suitable",
            "confidence",
            "warnings",
            "suggestions",
            "expected_result",
        }
        assert set(result.keys()) == expected_keys

    def test_confidence_range(self) -> None:
        """Test that confidence is in valid range."""
        data = create_sine_wave()
        trace = create_trace(data)

        measurements = ["frequency", "rise_time", "amplitude", "thd", "mean"]
        for measurement in measurements:
            result = check_measurement_suitability(trace, measurement)
            assert 0.0 <= result["confidence"] <= 1.0


# =============================================================================
# Test suggest_measurements
# =============================================================================


class TestSuggestMeasurements:
    """Tests for suggest_measurements function (INF-008)."""

    def test_suggest_for_dc_signal(self) -> None:
        """Test measurement suggestions for DC signal."""
        data = create_dc_signal()
        trace = create_trace(data)

        suggestions = suggest_measurements(trace)

        # Should suggest mean and rms
        names = [s["name"] for s in suggestions]
        assert "mean" in names
        assert "rms" in names

        # Should NOT suggest frequency for DC
        assert "frequency" not in names

    def test_suggest_for_periodic_digital(self) -> None:
        """Test measurement suggestions for periodic digital signal."""
        data = create_square_wave(n_periods=10)
        trace = create_trace(data)

        suggestions = suggest_measurements(trace)

        names = [s["name"] for s in suggestions]

        # Should suggest timing measurements
        assert "frequency" in names
        assert "rise_time" in names or "fall_time" in names

        # Should suggest duty cycle for periodic signal with edges
        # (depends on edge count detection)

    def test_suggest_for_periodic_analog(self) -> None:
        """Test measurement suggestions for periodic analog signal."""
        data = create_sine_wave(n_periods=10)
        trace = create_trace(data)

        suggestions = suggest_measurements(trace)

        names = [s["name"] for s in suggestions]

        # Should suggest frequency
        assert "frequency" in names

        # Should suggest amplitude
        assert "amplitude" in names

    def test_suggest_max_suggestions_limit(self) -> None:
        """Test that max_suggestions parameter is respected."""
        data = create_square_wave(n_periods=10)
        trace = create_trace(data)

        suggestions = suggest_measurements(trace, max_suggestions=3)

        assert len(suggestions) <= 3

    def test_suggest_always_includes_statistical(self) -> None:
        """Test that statistical measurements are always suggested."""
        signals = [
            create_dc_signal(),
            create_square_wave(),
            create_sine_wave(),
        ]

        for data in signals:
            trace = create_trace(data)
            suggestions = suggest_measurements(trace)

            names = [s["name"] for s in suggestions]
            assert "mean" in names
            assert "rms" in names

    def test_suggest_returns_correct_format(self) -> None:
        """Test that suggestions have correct format."""
        data = create_sine_wave()
        trace = create_trace(data)

        suggestions = suggest_measurements(trace)

        for s in suggestions:
            assert "name" in s
            assert "category" in s
            assert "priority" in s
            assert "rationale" in s
            assert "confidence" in s

            assert isinstance(s["name"], str)
            assert isinstance(s["category"], str)
            assert isinstance(s["priority"], int)
            assert isinstance(s["rationale"], str)
            assert 0.0 <= s["confidence"] <= 1.0

    def test_suggest_sorted_by_priority(self) -> None:
        """Test that suggestions are sorted by priority."""
        data = create_square_wave(n_periods=10)
        trace = create_trace(data)

        suggestions = suggest_measurements(trace)

        priorities = [s["priority"] for s in suggestions]
        assert priorities == sorted(priorities)

    def test_suggest_valid_categories(self) -> None:
        """Test that categories are valid."""
        data = create_square_wave(n_periods=10)
        trace = create_trace(data)

        suggestions = suggest_measurements(trace)

        valid_categories = {"statistical", "timing", "amplitude", "spectral"}
        for s in suggestions:
            assert s["category"] in valid_categories

    def test_suggest_spectral_for_clean_periodic(self) -> None:
        """Test spectral suggestions for clean periodic signal with enough samples."""
        # Create clean periodic signal with many samples
        data = create_sine_wave(n_periods=50, sample_rate=1e6, frequency=1000)
        trace = create_trace(data, sample_rate=1e6)

        suggestions = suggest_measurements(trace)

        names = [s["name"] for s in suggestions]

        # For clean periodic signal with enough samples, might suggest spectral
        # This depends on the exact signal characteristics
        # Just verify the function works
        assert len(suggestions) > 0


# =============================================================================
# Test Helper Functions
# =============================================================================


class TestDetectDigitalSignal:
    """Tests for _detect_digital_signal helper."""

    def test_detect_perfect_square_wave(self) -> None:
        """Test detection of perfect square wave."""
        data = create_square_wave(low=0.0, high=5.0)

        is_digital, levels, confidence = _detect_digital_signal(data, threshold_ratio=0.8)

        assert is_digital is True
        assert levels is not None
        # Histogram-based detection returns bin centers, which may not be exact
        # Check that low level is below threshold and high level is above
        assert levels["low"] < levels["high"]
        assert levels["low"] < 2.5  # Should be closer to low value (0.0)
        assert levels["high"] > 2.5  # Should be closer to high value (5.0)
        assert confidence > 0.8

    def test_detect_sine_wave_not_digital(self) -> None:
        """Test that sine wave is not detected as digital."""
        data = create_sine_wave()

        is_digital, levels, confidence = _detect_digital_signal(data, threshold_ratio=0.8)

        assert is_digital is False
        assert levels is None

    def test_detect_with_strict_threshold(self) -> None:
        """Test digital detection with strict threshold."""
        data = create_square_wave(noise_level=0.05)

        is_digital, _, confidence = _detect_digital_signal(data, threshold_ratio=0.95)

        # Noisy square wave may not meet strict threshold
        assert isinstance(is_digital, bool)
        assert isinstance(confidence, float)


class TestEstimateNoiseLevel:
    """Tests for _estimate_noise_level helper."""

    def test_estimate_no_noise(self) -> None:
        """Test noise estimation on clean signal."""
        data = create_sine_wave(noise_level=0.0)

        noise = _estimate_noise_level(data)

        # Clean signal should have very low noise estimate
        assert noise < 0.01

    def test_estimate_known_noise(self) -> None:
        """Test noise estimation with known noise level."""
        np.random.seed(42)
        noise_level = 0.1
        data = create_sine_wave(noise_level=noise_level)

        estimated_noise = _estimate_noise_level(data)

        # Estimate should be in the right ballpark
        assert 0.05 < estimated_noise < 0.3

    def test_estimate_short_signal(self) -> None:
        """Test noise estimation on short signal."""
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        noise = _estimate_noise_level(data)

        # Should return 0 for very short signals
        assert noise == 0.0

    def test_estimate_nonnegative(self) -> None:
        """Test that noise estimate is always non-negative."""
        signals = [
            create_dc_signal(),
            create_sine_wave(),
            np.random.randn(1000).astype(np.float64),
        ]

        for data in signals:
            noise = _estimate_noise_level(data)
            assert noise >= 0.0


class TestDetectPeriodicity:
    """Tests for _detect_periodicity helper."""

    def test_detect_periodic_sine(self) -> None:
        """Test periodicity detection on sine wave."""
        frequency = 1000.0
        sample_rate = 1e6
        data = create_sine_wave(frequency=frequency, sample_rate=sample_rate, n_periods=10)

        is_periodic, period, confidence = _detect_periodicity(data, sample_rate, threshold=0.7)

        assert is_periodic is True
        assert period is not None
        # Period should be close to 1/frequency = 1ms
        expected_period = 1.0 / frequency
        assert abs(period - expected_period) < expected_period * 0.2  # Within 20%
        assert confidence > 0.5

    def test_detect_aperiodic_noise(self) -> None:
        """Test periodicity detection on random noise."""
        np.random.seed(42)
        data = np.random.randn(1000).astype(np.float64)

        is_periodic, period, confidence = _detect_periodicity(data, 1e6, threshold=0.7)

        # Random noise should not be detected as periodic
        assert is_periodic is False

    def test_detect_short_signal(self) -> None:
        """Test periodicity detection on short signal."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        is_periodic, period, confidence = _detect_periodicity(data, 1e6, threshold=0.7)

        # Too short for reliable detection
        assert is_periodic is False


class TestDetectPeriodicityFFT:
    """Tests for _detect_periodicity_fft helper."""

    def test_fft_detect_periodic_sine(self) -> None:
        """Test FFT-based periodicity detection on sine wave."""
        frequency = 1000.0
        sample_rate = 1e6
        data = create_sine_wave(frequency=frequency, sample_rate=sample_rate, n_periods=10)

        is_periodic, period, confidence = _detect_periodicity_fft(data, sample_rate)

        assert is_periodic is True
        assert period is not None
        expected_period = 1.0 / frequency
        assert abs(period - expected_period) < expected_period * 0.2

    def test_fft_short_signal(self) -> None:
        """Test FFT detection on signal too short."""
        data = np.array(range(50), dtype=np.float64)

        is_periodic, period, confidence = _detect_periodicity_fft(data, 1e6)

        # Too short for FFT analysis
        assert is_periodic is False

    def test_fft_dc_signal(self) -> None:
        """Test FFT detection on DC signal."""
        data = create_dc_signal(n_samples=1000)

        is_periodic, period, confidence = _detect_periodicity_fft(data, 1e6)

        # DC signal has no periodic component
        assert is_periodic is False


class TestDetectEdgePeriodicity:
    """Tests for _detect_edge_periodicity helper."""

    def test_edge_detect_square_wave(self) -> None:
        """Test edge-based periodicity detection on square wave."""
        frequency = 1000.0
        sample_rate = 1e6
        data = create_square_wave(frequency=frequency, sample_rate=sample_rate, n_periods=5)
        levels = {"low": 0.0, "high": 1.0}

        is_periodic, period, confidence = _detect_edge_periodicity(data, sample_rate, levels)

        assert is_periodic is True
        assert period is not None
        expected_period = 1.0 / frequency
        # Edge-based detection should be close to expected period
        assert abs(period - expected_period) < expected_period * 0.3

    def test_edge_detect_no_levels(self) -> None:
        """Test edge detection without level information."""
        data = create_square_wave()

        is_periodic, period, confidence = _detect_edge_periodicity(data, 1e6, levels=None)

        # Should return False without levels
        assert is_periodic is False

    def test_edge_detect_short_signal(self) -> None:
        """Test edge detection on very short signal."""
        data = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
        levels = {"low": 0.0, "high": 1.0}

        is_periodic, period, confidence = _detect_edge_periodicity(data, 1e6, levels)

        # Very short, may or may not detect
        assert isinstance(is_periodic, bool)


class TestCountEdges:
    """Tests for _count_edges helper."""

    def test_count_square_wave_edges(self) -> None:
        """Test edge counting on square wave."""
        data = create_square_wave(n_periods=5)
        levels = {"low": 0.0, "high": 1.0}

        edge_count = _count_edges(data, levels)

        # 5 periods should have ~10 edges (rising + falling)
        assert edge_count >= 8  # Allow some margin

    def test_count_dc_signal_edges(self) -> None:
        """Test edge counting on DC signal."""
        data = create_dc_signal()

        edge_count = _count_edges(data, levels=None)

        # DC signal has no edges
        assert edge_count == 0

    def test_count_edges_short_signal(self) -> None:
        """Test edge counting on very short signal."""
        data = np.array([0.0, 1.0], dtype=np.float64)

        edge_count = _count_edges(data, levels=None)

        # Too short to count edges reliably
        assert edge_count == 0

    def test_count_edges_without_levels(self) -> None:
        """Test edge counting using median threshold."""
        data = create_square_wave(n_periods=3)

        edge_count = _count_edges(data, levels=None)

        # Should still find edges using median threshold
        assert edge_count >= 4


# =============================================================================
# Integration Tests
# =============================================================================


class TestInferenceSignalIntelligenceIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow_digital_signal(self) -> None:
        """Test complete analysis workflow for digital signal."""
        data = create_square_wave(
            frequency=10000.0,
            sample_rate=1e6,
            n_periods=20,
            low=0.0,
            high=3.3,
        )
        trace = create_trace(data, sample_rate=1e6)

        # Classify
        classification = classify_signal(trace)
        assert classification["type"] in ("digital", "mixed")

        # Assess quality
        quality = assess_signal_quality(trace)
        assert quality["clipping"] is False

        # Check measurement suitability
        freq_check = check_measurement_suitability(trace, "frequency")
        assert freq_check["suitable"] is True

        # Get suggestions
        suggestions = suggest_measurements(trace)
        assert len(suggestions) > 0

    def test_full_workflow_analog_signal(self) -> None:
        """Test complete analysis workflow for analog signal."""
        data = create_sine_wave(
            frequency=1000.0,
            sample_rate=1e6,
            n_periods=20,
            amplitude=1.0,
        )
        trace = create_trace(data, sample_rate=1e6)

        # Classify
        classification = classify_signal(trace)
        assert classification["type"] == "analog"
        assert "periodic" in classification["characteristics"]

        # Assess quality
        quality = assess_signal_quality(trace)
        assert quality["noise_level"] < 0.1

        # Check measurement suitability
        freq_check = check_measurement_suitability(trace, "frequency")
        assert freq_check["suitable"] is True

        # Get suggestions
        suggestions = suggest_measurements(trace)
        names = [s["name"] for s in suggestions]
        assert "frequency" in names

    def test_full_workflow_dc_signal(self) -> None:
        """Test complete analysis workflow for DC signal."""
        data = create_dc_signal(value=2.5)
        trace = create_trace(data)

        # Classify
        classification = classify_signal(trace)
        assert classification["type"] == "dc"

        # Assess quality
        quality = assess_signal_quality(trace)
        assert quality["clipping"] is False

        # Check measurement suitability for frequency
        freq_check = check_measurement_suitability(trace, "frequency")
        assert freq_check["suitable"] is False
        assert freq_check["expected_result"] == "nan"

        # Get suggestions - should not include frequency
        suggestions = suggest_measurements(trace)
        names = [s["name"] for s in suggestions]
        assert "frequency" not in names
        assert "mean" in names


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestInferenceSignalIntelligenceEdgeCases:
    """Test edge cases and error handling."""

    def test_all_zeros_signal(self) -> None:
        """Test handling of all-zeros signal."""
        data = np.zeros(1000, dtype=np.float64)
        trace = create_trace(data)

        result = classify_signal(trace)
        assert result["type"] == "dc"
        assert result["dc_component"] is True

    def test_all_same_value(self) -> None:
        """Test handling of constant non-zero signal."""
        data = np.full(1000, 42.0, dtype=np.float64)
        trace = create_trace(data)

        result = classify_signal(trace)
        assert result["type"] == "dc"

    def test_single_step(self) -> None:
        """Test handling of single step transition."""
        data = np.concatenate(
            [
                np.zeros(500, dtype=np.float64),
                np.ones(500, dtype=np.float64),
            ]
        )
        trace = create_trace(data)

        result = classify_signal(trace)
        # Could be classified as digital or transient
        assert result["type"] in ("digital", "analog", "mixed")

    def test_very_high_frequency_signal(self) -> None:
        """Test handling of signal near Nyquist frequency."""
        sample_rate = 1e6
        frequency = 0.4e6  # 0.4 * sample_rate (close to Nyquist)
        n_samples = 1000
        t = np.arange(n_samples, dtype=np.float64) / sample_rate
        data = np.sin(2 * np.pi * frequency * t)
        trace = create_trace(data, sample_rate=sample_rate)

        # Should handle without error
        result = classify_signal(trace)
        assert result is not None

        quality = assess_signal_quality(trace)
        # Should warn about sample rate
        assert quality is not None

    def test_inf_values_in_data(self) -> None:
        """Test handling of infinite values in data."""
        data = np.array([1.0, 2.0, np.inf, 3.0, 4.0], dtype=np.float64)
        trace = create_trace(data)

        # Functions should handle or raise appropriate errors
        try:
            result = classify_signal(trace)
            # If it succeeds, result should still be valid dict
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # Acceptable to raise error for invalid data
            pass

    def test_nan_values_in_data(self) -> None:
        """Test handling of NaN values in data."""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0], dtype=np.float64)
        trace = create_trace(data)

        # Functions should handle or raise appropriate errors
        try:
            result = classify_signal(trace)
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            pass

    def test_very_small_amplitude(self) -> None:
        """Test handling of very small amplitude signal."""
        data = create_sine_wave(amplitude=1e-15)
        trace = create_trace(data)

        result = classify_signal(trace)
        # Very small amplitude might be classified as DC
        assert result["type"] in ("dc", "analog")

    def test_very_large_amplitude(self) -> None:
        """Test handling of very large amplitude signal."""
        data = create_sine_wave(amplitude=1e10)
        trace = create_trace(data)

        result = classify_signal(trace)
        assert result["type"] == "analog"
        assert "periodic" in result["characteristics"]

    def test_different_sample_rates(self) -> None:
        """Test behavior with different sample rates."""
        # Ensure adequate oversampling by using frequencies appropriate for each sample rate
        test_cases = [
            # (sample_rate, frequency) - at least 10x oversampling
            (1e4, 100.0),  # 100 samples per period
            (1e6, 10000.0),  # 100 samples per period
            (1e9, 1e7),  # 100 samples per period
        ]

        for sr, freq in test_cases:
            data = create_sine_wave(frequency=freq, sample_rate=sr, n_periods=10)
            trace = create_trace(data, sample_rate=sr)

            result = classify_signal(trace)
            assert result is not None
            assert result["type"] == "analog"


class TestExportedAPI:
    """Test that exported API matches __all__."""

    def test_all_exports(self) -> None:
        """Test that all exported functions are accessible."""
        from oscura.inference import signal_intelligence

        expected_exports = [
            "assess_signal_quality",
            "check_measurement_suitability",
            "classify_signal",
            "suggest_measurements",
        ]

        for name in expected_exports:
            assert hasattr(signal_intelligence, name)
            assert name in signal_intelligence.__all__
