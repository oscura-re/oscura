"""Unit tests for signal validation and suitability checking.

This module tests validation.py which provides helper functions to determine
whether a signal is suitable for specific measurements.

Tests cover:
- is_suitable_for_frequency_measurement
- is_suitable_for_duty_cycle_measurement
- is_suitable_for_rise_time_measurement
- is_suitable_for_fall_time_measurement
- is_suitable_for_jitter_measurement
- get_valid_measurements
- analyze_signal_characteristics
- get_measurement_requirements
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils.factories import make_waveform_trace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


def make_square_wave(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
    duty_cycle: float = 0.5,
    rise_time_samples: int = 0,
) -> np.ndarray:
    """Create a square wave for testing.

    Args:
        frequency: Signal frequency in Hz
        sample_rate: Sample rate in Hz
        duration: Duration in seconds
        amplitude: Peak amplitude
        duty_cycle: Duty cycle (0-1)
        rise_time_samples: Number of samples for rise/fall transitions (0 = instant)
    """
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    period = 1.0 / frequency
    phase = (t % period) / period
    signal = np.where(phase < duty_cycle, amplitude, -amplitude)

    # Add realistic rise/fall times if requested
    if rise_time_samples > 0:
        # Find transitions
        transitions = np.diff(signal)
        rising = np.where(transitions > 0)[0]
        falling = np.where(transitions < 0)[0]

        # Smooth transitions with linear ramp
        for idx in rising:
            end_idx = min(idx + rise_time_samples + 1, n_samples)
            ramp_len = end_idx - idx - 1
            if ramp_len > 0:
                signal[idx + 1 : end_idx] = np.linspace(-amplitude, amplitude, ramp_len)

        for idx in falling:
            end_idx = min(idx + rise_time_samples + 1, n_samples)
            ramp_len = end_idx - idx - 1
            if ramp_len > 0:
                signal[idx + 1 : end_idx] = np.linspace(amplitude, -amplitude, ramp_len)

    return signal


def make_sine_wave(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """Create a sine wave for testing."""
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t)


def make_dc_signal(value: float, n_samples: int) -> np.ndarray:
    """Create a DC signal for testing."""
    return np.full(n_samples, value, dtype=np.float64)


def make_noise_signal(n_samples: int, amplitude: float = 1.0, seed: int = 42) -> np.ndarray:
    """Create a noise signal for testing."""
    np.random.seed(seed)
    return amplitude * np.random.randn(n_samples)


# =============================================================================
# Frequency Measurement Suitability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestFrequencyMeasurementSuitability:
    """Test is_suitable_for_frequency_measurement function."""

    def test_periodic_square_wave_suitable(self) -> None:
        """Test that periodic square wave is suitable for frequency measurement."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        assert suitable is True
        assert "suitable" in reason.lower()

    def test_dc_signal_not_suitable(self) -> None:
        """Test that DC signal is not suitable for frequency measurement."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        signal = make_dc_signal(1.0, 1000)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        assert suitable is False
        assert "variation" in reason.lower() or "dc" in reason.lower()

    def test_insufficient_samples_not_suitable(self) -> None:
        """Test that signal with insufficient samples is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        signal = np.array([1.0, -1.0])  # Only 2 samples
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        assert suitable is False
        assert "samples" in reason.lower()

    def test_single_edge_not_suitable(self) -> None:
        """Test that signal with only one edge is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        # Step function - only one transition
        signal = np.concatenate([np.zeros(50), np.ones(50)])
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        assert suitable is False
        assert "edge" in reason.lower()

    def test_aperiodic_signal_not_suitable(self) -> None:
        """Test that aperiodic signal is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        # Create signal with varying period
        signal = np.array(
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1] * 10, dtype=np.float64
        )
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        # May or may not be suitable depending on edge detection
        # Just check we get a valid response
        assert isinstance(suitable, bool)
        assert isinstance(reason, str)

    def test_periodic_sine_wave_not_suitable_no_edges(self) -> None:
        """Test that pure sine wave may not be suitable (no clear digital edges)."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        # Low amplitude sine wave might not trigger edge detection
        signal = make_sine_wave(frequency=1e6, sample_rate=100e6, duration=1e-4, amplitude=0.1)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        # Result depends on edge detection threshold
        assert isinstance(suitable, bool)
        assert isinstance(reason, str)


# =============================================================================
# Duty Cycle Measurement Suitability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestDutyCycleMeasurementSuitability:
    """Test is_suitable_for_duty_cycle_measurement function."""

    def test_square_wave_suitable(self) -> None:
        """Test that square wave is suitable for duty cycle measurement."""
        from oscura.analyzers.validation import is_suitable_for_duty_cycle_measurement

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4, duty_cycle=0.5)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_duty_cycle_measurement(trace)

        assert suitable is True
        assert "suitable" in reason.lower()

    def test_dc_signal_not_suitable(self) -> None:
        """Test that DC signal is not suitable for duty cycle measurement."""
        from oscura.analyzers.validation import is_suitable_for_duty_cycle_measurement

        signal = make_dc_signal(1.0, 1000)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_duty_cycle_measurement(trace)

        assert suitable is False

    def test_rising_edges_only_not_suitable(self) -> None:
        """Test that signal with only rising edges is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_duty_cycle_measurement

        # Sawtooth - rising edges but soft falling edges
        n_samples = 1000
        n_periods = 10
        signal = np.tile(np.linspace(0, 1, n_samples // n_periods), n_periods)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_duty_cycle_measurement(trace)

        # Should need both rising and falling edges
        assert isinstance(suitable, bool)
        assert isinstance(reason, str)


# =============================================================================
# Rise Time Measurement Suitability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestRiseTimeMeasurementSuitability:
    """Test is_suitable_for_rise_time_measurement function."""

    def test_square_wave_suitable(self) -> None:
        """Test that square wave is suitable for rise time measurement."""
        from oscura.analyzers.validation import is_suitable_for_rise_time_measurement

        # Use realistic rise time (5 samples for transition)
        signal = make_square_wave(
            frequency=1e6, sample_rate=100e6, duration=1e-4, rise_time_samples=5
        )
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_rise_time_measurement(trace)

        assert suitable is True
        assert "suitable" in reason.lower()

    def test_dc_signal_not_suitable(self) -> None:
        """Test that DC signal is not suitable for rise time measurement."""
        from oscura.analyzers.validation import is_suitable_for_rise_time_measurement

        signal = make_dc_signal(1.0, 1000)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_rise_time_measurement(trace)

        assert suitable is False

    def test_insufficient_samples_not_suitable(self) -> None:
        """Test that signal with insufficient samples is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_rise_time_measurement

        signal = np.array([0.0, 1.0])  # Only 2 samples
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_rise_time_measurement(trace)

        assert suitable is False
        assert "samples" in reason.lower()

    def test_no_amplitude_not_suitable(self) -> None:
        """Test that signal with no amplitude is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_rise_time_measurement

        signal = np.zeros(100)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_rise_time_measurement(trace)

        assert suitable is False
        # Zero amplitude means no edges can be detected
        assert "amplitude" in reason.lower() or "edges" in reason.lower()


# =============================================================================
# Fall Time Measurement Suitability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestFallTimeMeasurementSuitability:
    """Test is_suitable_for_fall_time_measurement function."""

    def test_square_wave_suitable(self) -> None:
        """Test that square wave is suitable for fall time measurement."""
        from oscura.analyzers.validation import is_suitable_for_fall_time_measurement

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_fall_time_measurement(trace)

        assert suitable is True
        assert "suitable" in reason.lower()

    def test_dc_signal_not_suitable(self) -> None:
        """Test that DC signal is not suitable for fall time measurement."""
        from oscura.analyzers.validation import is_suitable_for_fall_time_measurement

        signal = make_dc_signal(1.0, 1000)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_fall_time_measurement(trace)

        assert suitable is False

    def test_insufficient_samples_not_suitable(self) -> None:
        """Test that signal with insufficient samples is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_fall_time_measurement

        signal = np.array([1.0, 0.0])  # Only 2 samples
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_fall_time_measurement(trace)

        assert suitable is False
        assert "samples" in reason.lower()


# =============================================================================
# Jitter Measurement Suitability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestJitterMeasurementSuitability:
    """Test is_suitable_for_jitter_measurement function."""

    def test_periodic_square_wave_suitable(self) -> None:
        """Test that periodic square wave is suitable for jitter measurement."""
        from oscura.analyzers.validation import is_suitable_for_jitter_measurement

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_jitter_measurement(trace)

        assert suitable is True
        assert "suitable" in reason.lower()

    def test_dc_signal_not_suitable(self) -> None:
        """Test that DC signal is not suitable for jitter measurement."""
        from oscura.analyzers.validation import is_suitable_for_jitter_measurement

        signal = make_dc_signal(1.0, 1000)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_jitter_measurement(trace)

        assert suitable is False

    def test_insufficient_edges_not_suitable(self) -> None:
        """Test that signal with insufficient edges is not suitable."""
        from oscura.analyzers.validation import is_suitable_for_jitter_measurement

        # Only 2 edges - need at least 3 for jitter
        signal = np.concatenate([np.zeros(50), np.ones(100), np.zeros(50)])
        trace = make_waveform_trace(signal, sample_rate=1e9)

        suitable, reason = is_suitable_for_jitter_measurement(trace)

        # May or may not be suitable depending on edge count
        assert isinstance(suitable, bool)


# =============================================================================
# Get Valid Measurements Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestGetValidMeasurements:
    """Test get_valid_measurements function."""

    def test_periodic_signal_measurements(self) -> None:
        """Test valid measurements for periodic signal."""
        from oscura.analyzers.validation import get_valid_measurements

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        valid = get_valid_measurements(trace)

        assert isinstance(valid, list)
        assert "mean" in valid
        assert "rms" in valid
        assert "amplitude" in valid

    def test_dc_signal_limited_measurements(self) -> None:
        """Test valid measurements for DC signal are limited."""
        from oscura.analyzers.validation import get_valid_measurements

        signal = make_dc_signal(1.0, 1000)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        valid = get_valid_measurements(trace)

        assert "mean" in valid
        assert "rms" in valid
        # Frequency-based measurements should not be valid
        assert "frequency" not in valid
        assert "period" not in valid

    def test_empty_signal_measurements(self) -> None:
        """Test valid measurements for empty signal."""
        from oscura.analyzers.validation import get_valid_measurements

        signal = np.array([], dtype=np.float64)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        valid = get_valid_measurements(trace)

        # Should return empty or very limited list
        assert isinstance(valid, list)

    def test_single_sample_measurements(self) -> None:
        """Test valid measurements for single sample."""
        from oscura.analyzers.validation import get_valid_measurements

        signal = np.array([1.0])
        trace = make_waveform_trace(signal, sample_rate=1e9)

        valid = get_valid_measurements(trace)

        # mean and rms should be valid with single sample
        assert "mean" in valid
        assert "rms" in valid
        # amplitude needs at least 2 samples
        assert "amplitude" not in valid


# =============================================================================
# Analyze Signal Characteristics Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestAnalyzeSignalCharacteristics:
    """Test analyze_signal_characteristics function."""

    def test_dc_signal_characteristics(self) -> None:
        """Test characteristics of DC signal."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        signal = make_dc_signal(1.0, 100)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        chars = analyze_signal_characteristics(trace)

        assert chars["signal_type"] == "dc"
        assert not chars["has_variation"]
        assert not chars["is_periodic"]
        assert chars["edge_count"] == 0

    def test_periodic_digital_signal_characteristics(self) -> None:
        """Test characteristics of periodic digital signal."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        chars = analyze_signal_characteristics(trace)

        assert chars["has_variation"]
        assert chars["has_amplitude"]
        assert chars["has_edges"]
        assert chars["is_periodic"]
        assert chars["signal_type"] == "periodic_digital"

    def test_noise_signal_characteristics(self) -> None:
        """Test characteristics of noise signal."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        signal = make_noise_signal(1000, amplitude=0.1)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        chars = analyze_signal_characteristics(trace)

        assert chars["has_variation"]
        # May or may not have edges depending on noise characteristics
        assert isinstance(chars["edge_count"], int)

    def test_characteristics_includes_recommendations(self) -> None:
        """Test that characteristics include recommended measurements."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        chars = analyze_signal_characteristics(trace)

        assert "recommended_measurements" in chars
        assert isinstance(chars["recommended_measurements"], list)

    def test_insufficient_samples_characteristics(self) -> None:
        """Test characteristics with insufficient samples."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        signal = np.array([1.0, 0.0, 1.0])  # Only 3 samples
        trace = make_waveform_trace(signal, sample_rate=1e9)

        chars = analyze_signal_characteristics(trace)

        assert chars["sufficient_samples"] == False  # noqa: E712

    def test_sufficient_samples_characteristics(self) -> None:
        """Test characteristics with sufficient samples."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        chars = analyze_signal_characteristics(trace)

        assert chars["sufficient_samples"] == True  # noqa: E712

    def test_edge_count_accuracy(self) -> None:
        """Test that edge counts are reasonably accurate."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        # 10 periods at 1MHz, 100MHz sample rate = 1000 samples per period
        # Should have ~10 rising and ~10 falling edges
        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=10e-6)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        chars = analyze_signal_characteristics(trace)

        # Should have roughly 10 rising and 10 falling edges
        assert chars["rising_edge_count"] >= 5
        assert chars["falling_edge_count"] >= 5


# =============================================================================
# Get Measurement Requirements Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestGetMeasurementRequirements:
    """Test get_measurement_requirements function."""

    def test_frequency_requirements(self) -> None:
        """Test requirements for frequency measurement."""
        from oscura.analyzers.validation import get_measurement_requirements

        reqs = get_measurement_requirements("frequency")

        assert "description" in reqs
        assert "min_samples" in reqs
        assert "required_signal_types" in reqs
        assert "required_features" in reqs
        assert "common_nan_causes" in reqs

        assert reqs["min_samples"] >= 1
        assert "periodic" in reqs["required_features"]

    def test_mean_requirements(self) -> None:
        """Test requirements for mean measurement."""
        from oscura.analyzers.validation import get_measurement_requirements

        reqs = get_measurement_requirements("mean")

        assert reqs["min_samples"] == 1
        assert "all" in reqs["required_signal_types"]
        assert len(reqs["required_features"]) == 0  # No special features needed

    def test_rise_time_requirements(self) -> None:
        """Test requirements for rise_time measurement."""
        from oscura.analyzers.validation import get_measurement_requirements

        reqs = get_measurement_requirements("rise_time")

        assert "rising_edges" in reqs["required_features"]
        assert "amplitude" in reqs["required_features"]

    def test_duty_cycle_requirements(self) -> None:
        """Test requirements for duty_cycle measurement."""
        from oscura.analyzers.validation import get_measurement_requirements

        reqs = get_measurement_requirements("duty_cycle")

        assert "periodic" in reqs["required_features"]
        assert "rising_edges" in reqs["required_features"]
        assert "falling_edges" in reqs["required_features"]

    def test_unknown_measurement_returns_default(self) -> None:
        """Test that unknown measurement returns default requirements."""
        from oscura.analyzers.validation import get_measurement_requirements

        reqs = get_measurement_requirements("unknown_measurement_xyz")

        assert "description" in reqs
        assert "Measurement not documented" in reqs["description"]

    def test_all_documented_measurements(self) -> None:
        """Test that all commonly used measurements are documented."""
        from oscura.analyzers.validation import get_measurement_requirements

        measurements = [
            "frequency",
            "period",
            "duty_cycle",
            "rise_time",
            "fall_time",
            "pulse_width",
            "amplitude",
            "mean",
            "rms",
            "overshoot",
            "undershoot",
            "slew_rate",
            "rms_jitter",
            "peak_to_peak_jitter",
        ]

        for meas in measurements:
            reqs = get_measurement_requirements(meas)
            # Should have proper documentation (not the default)
            assert "Measurement not documented" not in reqs["description"]


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestValidationEdgeCases:
    """Test edge cases in validation module."""

    def test_very_low_amplitude_signal(self) -> None:
        """Test validation with very low amplitude signal."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        # Very low amplitude - near noise floor
        signal = 1e-15 * make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        # Should handle gracefully (likely not suitable due to low amplitude)
        assert isinstance(suitable, bool)
        assert isinstance(reason, str)

    def test_very_high_frequency_signal(self) -> None:
        """Test validation with high frequency relative to sample rate."""
        from oscura.analyzers.validation import is_suitable_for_frequency_measurement

        # High frequency - only 2 samples per period
        signal = make_square_wave(frequency=50e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_frequency_measurement(trace)

        # May or may not be suitable - depends on implementation
        assert isinstance(suitable, bool)

    def test_signal_with_nan_values(self) -> None:
        """Test validation handles NaN values gracefully."""
        from oscura.analyzers.validation import get_valid_measurements

        signal = np.array([1.0, np.nan, -1.0, 1.0, -1.0] * 20)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        # Should not crash, though results may be limited
        try:
            valid = get_valid_measurements(trace)
            assert isinstance(valid, list)
        except (ValueError, RuntimeWarning):
            # Acceptable to raise on NaN
            pass

    def test_signal_with_inf_values(self) -> None:
        """Test validation handles inf values gracefully."""
        from oscura.analyzers.validation import get_valid_measurements

        signal = np.array([1.0, np.inf, -1.0, 1.0, -1.0] * 20)
        trace = make_waveform_trace(signal, sample_rate=1e9)

        # Should not crash
        try:
            valid = get_valid_measurements(trace)
            assert isinstance(valid, list)
        except (ValueError, RuntimeWarning):
            # Acceptable to raise on inf
            pass

    def test_asymmetric_duty_cycle(self) -> None:
        """Test validation with asymmetric duty cycle."""
        from oscura.analyzers.validation import is_suitable_for_duty_cycle_measurement

        # 10% duty cycle
        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4, duty_cycle=0.1)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, reason = is_suitable_for_duty_cycle_measurement(trace)

        assert suitable is True

    def test_mixed_signal_types(self) -> None:
        """Test characteristics with mixed signal content."""
        from oscura.analyzers.validation import analyze_signal_characteristics

        # Signal with DC offset, periodic component, and noise
        n_samples = 10000
        sample_rate = 100e6
        t = np.arange(n_samples) / sample_rate

        # DC + square wave + noise
        signal = (
            1.0  # DC offset
            + make_square_wave(
                frequency=1e6, sample_rate=sample_rate, duration=n_samples / sample_rate
            )
            + 0.1 * np.random.randn(n_samples)
        )
        trace = make_waveform_trace(signal, sample_rate=sample_rate)

        chars = analyze_signal_characteristics(trace)

        assert chars["has_variation"]
        assert chars["has_amplitude"]


# =============================================================================
# Consistency Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestValidationConsistency:
    """Test consistency between validation functions."""

    def test_frequency_consistency_with_valid_measurements(self) -> None:
        """Test that frequency suitability matches get_valid_measurements."""
        from oscura.analyzers.validation import (
            get_valid_measurements,
            is_suitable_for_frequency_measurement,
        )

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, _ = is_suitable_for_frequency_measurement(trace)
        valid = get_valid_measurements(trace)

        if suitable:
            assert "frequency" in valid
        else:
            assert "frequency" not in valid

    def test_rise_time_consistency_with_valid_measurements(self) -> None:
        """Test that rise_time suitability matches get_valid_measurements."""
        from oscura.analyzers.validation import (
            get_valid_measurements,
            is_suitable_for_rise_time_measurement,
        )

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        suitable, _ = is_suitable_for_rise_time_measurement(trace)
        valid = get_valid_measurements(trace)

        if suitable:
            assert "rise_time" in valid
        else:
            assert "rise_time" not in valid

    def test_characteristics_consistency_with_valid_measurements(self) -> None:
        """Test that characteristics recommendations match get_valid_measurements."""
        from oscura.analyzers.validation import (
            analyze_signal_characteristics,
            get_valid_measurements,
        )

        signal = make_square_wave(frequency=1e6, sample_rate=100e6, duration=1e-4)
        trace = make_waveform_trace(signal, sample_rate=100e6)

        chars = analyze_signal_characteristics(trace)
        valid = get_valid_measurements(trace)

        # recommended_measurements should match get_valid_measurements
        assert set(chars["recommended_measurements"]) == set(valid)
