"""Unit tests for timing measurements module.

This module provides comprehensive tests for advanced timing measurements
including propagation delay, setup/hold time, slew rate, phase, skew,
clock recovery, and jitter analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.digital.timing import (
    ClockRecoveryResult,
    RMSJitterResult,
    TimingViolation,
    hold_time,
    peak_to_peak_jitter,
    phase,
    propagation_delay,
    recover_clock_edge,
    recover_clock_fft,
    rms_jitter,
    setup_time,
    skew,
    slew_rate,
    time_interval_error,
)
from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, TraceMetadata
from tests.utils.factories import make_waveform_trace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


# =============================================================================
# Helper Functions
# =============================================================================


def make_digital_trace(
    data: np.ndarray,
    sample_rate: float = 1e6,
) -> DigitalTrace:
    """Create a DigitalTrace from raw data for testing."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return DigitalTrace(data=data.astype(np.bool_), metadata=metadata)


def make_square_wave(
    frequency: float,
    sample_rate: float,
    duration: float,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Generate a square wave signal."""
    t = np.arange(0, duration, 1.0 / sample_rate)
    return amplitude * (np.sin(2 * np.pi * frequency * t) > 0).astype(np.float64) + offset


def make_clock_signal(
    frequency: float,
    sample_rate: float,
    n_cycles: int,
    jitter_std: float = 0.0,
    rng_seed: int = 42,
) -> np.ndarray:
    """Generate a clock signal with optional jitter."""
    period = 1.0 / frequency
    samples_per_cycle = int(sample_rate * period)

    # Create base pattern
    pattern = np.concatenate([np.ones(samples_per_cycle // 2), np.zeros(samples_per_cycle // 2)])

    # Add jitter if requested
    if jitter_std > 0:
        rng = np.random.default_rng(rng_seed)
        signal = []
        for _ in range(n_cycles):
            # Add timing jitter to each cycle
            jitter_samples = int(rng.normal(0, jitter_std * sample_rate))
            jittered_pattern = np.roll(pattern, jitter_samples)
            signal.append(jittered_pattern)
        return np.concatenate(signal)
    else:
        return np.tile(pattern, n_cycles)


# =============================================================================
# Test Data Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-007")
class TestDataClasses:
    """Test timing result data classes."""

    def test_clock_recovery_result(self) -> None:
        """Test ClockRecoveryResult dataclass."""
        result = ClockRecoveryResult(
            frequency=10e6,
            period=100e-9,
            method="fft",
            confidence=0.95,
        )

        assert result.frequency == 10e6
        assert result.period == 100e-9
        assert result.method == "fft"
        assert result.confidence == 0.95
        assert result.jitter_rms is None
        assert result.jitter_pp is None

    def test_clock_recovery_result_with_jitter(self) -> None:
        """Test ClockRecoveryResult with jitter data."""
        result = ClockRecoveryResult(
            frequency=10e6,
            period=100e-9,
            method="edge",
            confidence=0.90,
            jitter_rms=1e-12,
            jitter_pp=5e-12,
        )

        assert result.jitter_rms == 1e-12
        assert result.jitter_pp == 5e-12

    def test_timing_violation(self) -> None:
        """Test TimingViolation dataclass."""
        violation = TimingViolation(
            timestamp=1e-6,
            violation_type="setup",
            measured=0.5e-9,
            required=1e-9,
            margin=-0.5e-9,
        )

        assert violation.timestamp == 1e-6
        assert violation.violation_type == "setup"
        assert violation.measured == 0.5e-9
        assert violation.required == 1e-9
        assert violation.margin == -0.5e-9

    def test_rms_jitter_result(self) -> None:
        """Test RMSJitterResult dataclass."""
        result = RMSJitterResult(
            rms=1e-12,
            mean=100e-9,
            samples=1000,
            uncertainty=0.05e-12,
            edge_type="rising",
        )

        assert result.rms == 1e-12
        assert result.mean == 100e-9
        assert result.samples == 1000
        assert result.uncertainty == 0.05e-12
        assert result.edge_type == "rising"


# =============================================================================
# Propagation Delay Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-001")
class TestPropagationDelay:
    """Test propagation delay measurements."""

    def test_propagation_delay_basic(self) -> None:
        """Test basic propagation delay measurement."""
        sample_rate = 1e6
        frequency = 1000.0
        delay_samples = 10  # 10 samples delay = 10 µs at 1 MHz

        # Create input and delayed output
        input_signal = make_square_wave(frequency, sample_rate, 0.01)
        output_signal = np.roll(input_signal, delay_samples)

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        delay = propagation_delay(input_trace, output_trace)

        # Should measure approximately the expected delay
        expected_delay = delay_samples / sample_rate
        assert abs(delay - expected_delay) < 2 / sample_rate  # Within 2 samples

    def test_propagation_delay_rising_edges(self) -> None:
        """Test propagation delay using rising edges only."""
        sample_rate = 1e6
        frequency = 1000.0
        delay_samples = 5

        input_signal = make_square_wave(frequency, sample_rate, 0.01)
        output_signal = np.roll(input_signal, delay_samples)

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        delay = propagation_delay(input_trace, output_trace, edge_type="rising")

        expected_delay = delay_samples / sample_rate
        assert abs(delay - expected_delay) < 2 / sample_rate

    def test_propagation_delay_falling_edges(self) -> None:
        """Test propagation delay using falling edges only."""
        sample_rate = 1e6
        frequency = 1000.0
        delay_samples = 5

        input_signal = make_square_wave(frequency, sample_rate, 0.01)
        output_signal = np.roll(input_signal, delay_samples)

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        delay = propagation_delay(input_trace, output_trace, edge_type="falling")

        expected_delay = delay_samples / sample_rate
        assert abs(delay - expected_delay) < 2 / sample_rate

    def test_propagation_delay_return_all(self) -> None:
        """Test propagation delay with return_all=True."""
        sample_rate = 1e6
        frequency = 1000.0
        delay_samples = 5

        input_signal = make_square_wave(frequency, sample_rate, 0.01)
        output_signal = np.roll(input_signal, delay_samples)

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        delays = propagation_delay(input_trace, output_trace, return_all=True)

        assert isinstance(delays, np.ndarray)
        assert len(delays) > 0
        # All delays should be similar
        assert np.std(delays) < 1 / sample_rate

    def test_propagation_delay_no_input_edges(self) -> None:
        """Test propagation delay with no edges in input."""
        sample_rate = 1e6

        input_signal = np.ones(1000)  # Constant signal
        output_signal = make_square_wave(1000.0, sample_rate, 0.001)

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        with pytest.raises(InsufficientDataError) as exc_info:
            propagation_delay(input_trace, output_trace)

        assert "No edges found in input trace" in str(exc_info.value)

    def test_propagation_delay_no_output_edges(self) -> None:
        """Test propagation delay with no edges in output."""
        sample_rate = 1e6

        input_signal = make_square_wave(1000.0, sample_rate, 0.001)
        output_signal = np.ones(1000)  # Constant signal

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        with pytest.raises(InsufficientDataError) as exc_info:
            propagation_delay(input_trace, output_trace)

        assert "No edges found in output trace" in str(exc_info.value)

    def test_propagation_delay_digital_traces(self) -> None:
        """Test propagation delay with DigitalTrace objects."""
        sample_rate = 1e6
        frequency = 1000.0
        delay_samples = 5

        input_signal = make_square_wave(frequency, sample_rate, 0.01) > 0.5
        output_signal = np.roll(input_signal, delay_samples)

        input_trace = make_digital_trace(input_signal, sample_rate)
        output_trace = make_digital_trace(output_signal, sample_rate)

        delay = propagation_delay(input_trace, output_trace)

        expected_delay = delay_samples / sample_rate
        assert abs(delay - expected_delay) < 2 / sample_rate

    def test_propagation_delay_custom_ref_level(self) -> None:
        """Test propagation delay with custom reference level."""
        sample_rate = 1e6
        frequency = 1000.0
        delay_samples = 5

        # Signal with 0-2V range
        input_signal = make_square_wave(frequency, sample_rate, 0.01, amplitude=2.0)
        output_signal = np.roll(input_signal, delay_samples)

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        # Use 25% threshold instead of 50%
        delay = propagation_delay(input_trace, output_trace, ref_level=0.25)

        # Should still measure the delay
        assert delay > 0

    def test_propagation_delay_no_matching_edges(self) -> None:
        """Test propagation delay when no output edges follow input edges."""
        sample_rate = 1e6

        # Input has edges at end, output at beginning (so output edges come before input)
        # This creates a situation where there are edges, but they don't match up
        input_signal = np.concatenate([np.zeros(900), np.ones(50), np.zeros(50)])
        output_signal = np.concatenate([np.ones(50), np.zeros(50), np.ones(50), np.zeros(850)])

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        # When input edges happen after all output edges, we get no matches
        delay = propagation_delay(input_trace, output_trace, return_all=False)

        # Should return NaN when no matching edges
        assert np.isnan(delay)

    def test_propagation_delay_return_all_empty(self) -> None:
        """Test propagation delay return_all with no matching edges."""
        sample_rate = 1e6

        # Similar scenario - output edges before input edges
        input_signal = np.concatenate([np.zeros(900), np.ones(50), np.zeros(50)])
        output_signal = np.concatenate([np.ones(50), np.zeros(50), np.ones(50), np.zeros(850)])

        input_trace = make_waveform_trace(input_signal, sample_rate)
        output_trace = make_waveform_trace(output_signal, sample_rate)

        delays = propagation_delay(input_trace, output_trace, return_all=True)

        assert isinstance(delays, np.ndarray)
        assert len(delays) == 0


# =============================================================================
# Setup Time Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-002")
class TestSetupTime:
    """Test setup time measurements."""

    def test_setup_time_basic(self) -> None:
        """Test basic setup time measurement."""
        sample_rate = 1e6

        # Create data signal that changes before clock edge
        # Data edge at 100 samples, clock edge at 150 samples
        data_signal = np.concatenate([np.zeros(100), np.ones(900)])
        clock_signal = np.concatenate([np.zeros(150), np.ones(850)])

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_setup = setup_time(data_trace, clock_trace, clock_edge="rising")

        # Setup time should be ~50 samples (150 - 100)
        expected = 50 / sample_rate
        assert abs(t_setup - expected) < 3 / sample_rate

    def test_setup_time_rising_clock(self) -> None:
        """Test setup time with rising clock edge."""
        sample_rate = 1e6

        data_signal = np.tile([0, 0, 1, 1], 250)
        clock_signal = np.tile([0, 0, 0, 1], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_setup = setup_time(data_trace, clock_trace, clock_edge="rising")

        # Should have positive setup time
        assert t_setup > 0

    def test_setup_time_falling_clock(self) -> None:
        """Test setup time with falling clock edge."""
        sample_rate = 1e6

        data_signal = np.tile([1, 1, 0, 0], 250)
        clock_signal = np.tile([1, 1, 1, 0], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_setup = setup_time(data_trace, clock_trace, clock_edge="falling")

        assert t_setup > 0

    def test_setup_time_return_all(self) -> None:
        """Test setup time with return_all=True."""
        sample_rate = 1e6

        data_signal = np.tile([0, 0, 1, 1], 250)
        clock_signal = np.tile([0, 0, 0, 1], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        setup_times = setup_time(data_trace, clock_trace, return_all=True)

        assert isinstance(setup_times, np.ndarray)
        assert len(setup_times) > 0

    def test_setup_time_no_clock_edges(self) -> None:
        """Test setup time with no clock edges."""
        sample_rate = 1e6

        data_signal = make_square_wave(1000.0, sample_rate, 0.001)
        clock_signal = np.ones(1000)  # No edges

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_setup = setup_time(data_trace, clock_trace)

        assert np.isnan(t_setup)

    def test_setup_time_no_data_edges(self) -> None:
        """Test setup time with no data edges."""
        sample_rate = 1e6

        data_signal = np.ones(1000)  # No edges
        clock_signal = make_square_wave(1000.0, sample_rate, 0.001)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_setup = setup_time(data_trace, clock_trace)

        assert np.isnan(t_setup)

    def test_setup_time_digital_traces(self) -> None:
        """Test setup time with DigitalTrace objects."""
        sample_rate = 1e6

        data_signal = np.tile([0, 0, 1, 1], 250) > 0.5
        clock_signal = np.tile([0, 0, 0, 1], 250) > 0.5

        data_trace = make_digital_trace(data_signal, sample_rate)
        clock_trace = make_digital_trace(clock_signal, sample_rate)

        t_setup = setup_time(data_trace, clock_trace)

        assert t_setup > 0


# =============================================================================
# Hold Time Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-003")
class TestHoldTime:
    """Test hold time measurements."""

    def test_hold_time_basic(self) -> None:
        """Test basic hold time measurement."""
        sample_rate = 1e6

        # Clock edge at 100, data edge at 150
        clock_signal = np.concatenate([np.zeros(100), np.ones(900)])
        data_signal = np.concatenate([np.ones(150), np.zeros(850)])

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_hold = hold_time(data_trace, clock_trace, clock_edge="rising")

        # Hold time should be ~50 samples
        expected = 50 / sample_rate
        assert abs(t_hold - expected) < 3 / sample_rate

    def test_hold_time_rising_clock(self) -> None:
        """Test hold time with rising clock edge."""
        sample_rate = 1e6

        clock_signal = np.tile([0, 1, 1, 1], 250)
        data_signal = np.tile([1, 1, 0, 0], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_hold = hold_time(data_trace, clock_trace, clock_edge="rising")

        assert t_hold > 0

    def test_hold_time_falling_clock(self) -> None:
        """Test hold time with falling clock edge."""
        sample_rate = 1e6

        clock_signal = np.tile([1, 0, 0, 0], 250)
        data_signal = np.tile([0, 0, 1, 1], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_hold = hold_time(data_trace, clock_trace, clock_edge="falling")

        assert t_hold > 0

    def test_hold_time_return_all(self) -> None:
        """Test hold time with return_all=True."""
        sample_rate = 1e6

        clock_signal = np.tile([0, 1, 1, 1], 250)
        data_signal = np.tile([1, 1, 0, 0], 250)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        hold_times = hold_time(data_trace, clock_trace, return_all=True)

        assert isinstance(hold_times, np.ndarray)
        assert len(hold_times) > 0

    def test_hold_time_no_clock_edges(self) -> None:
        """Test hold time with no clock edges."""
        sample_rate = 1e6

        data_signal = make_square_wave(1000.0, sample_rate, 0.001)
        clock_signal = np.ones(1000)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_hold = hold_time(data_trace, clock_trace)

        assert np.isnan(t_hold)

    def test_hold_time_no_data_edges(self) -> None:
        """Test hold time with no data edges."""
        sample_rate = 1e6

        data_signal = np.ones(1000)
        clock_signal = make_square_wave(1000.0, sample_rate, 0.001)

        data_trace = make_waveform_trace(data_signal, sample_rate)
        clock_trace = make_waveform_trace(clock_signal, sample_rate)

        t_hold = hold_time(data_trace, clock_trace)

        assert np.isnan(t_hold)

    def test_hold_time_digital_traces(self) -> None:
        """Test hold time with DigitalTrace objects."""
        sample_rate = 1e6

        clock_signal = np.tile([0, 1, 1, 1], 250) > 0.5
        data_signal = np.tile([1, 1, 0, 0], 250) > 0.5

        data_trace = make_digital_trace(data_signal, sample_rate)
        clock_trace = make_digital_trace(clock_signal, sample_rate)

        t_hold = hold_time(data_trace, clock_trace)

        assert t_hold > 0


# =============================================================================
# Slew Rate Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-004")
class TestSlewRate:
    """Test slew rate measurements."""

    def test_slew_rate_rising_edge(self) -> None:
        """Test slew rate on rising edge."""
        sample_rate = 1e6

        # Create a ramp from 0 to 1 over 100 samples
        ramp = np.linspace(0, 1, 100)
        signal = np.concatenate([np.zeros(100), ramp, np.ones(800)])

        trace = make_waveform_trace(signal, sample_rate)

        sr = slew_rate(trace, edge_type="rising")

        # Slew rate should be positive for rising edge
        assert sr > 0

    def test_slew_rate_falling_edge(self) -> None:
        """Test slew rate on falling edge."""
        sample_rate = 1e6

        # Create a falling ramp
        ramp = np.linspace(1, 0, 100)
        signal = np.concatenate([np.ones(100), ramp, np.zeros(800)])

        trace = make_waveform_trace(signal, sample_rate)

        sr = slew_rate(trace, edge_type="falling")

        # Slew rate should be negative for falling edge
        assert sr < 0

    def test_slew_rate_both_edges(self) -> None:
        """Test slew rate on both rising and falling edges."""
        sample_rate = 1e6

        # Square wave with finite rise/fall times
        signal = make_square_wave(1000.0, sample_rate, 0.01)

        trace = make_waveform_trace(signal, sample_rate)

        sr = slew_rate(trace, edge_type="both")

        # Should measure something
        assert not np.isnan(sr)

    def test_slew_rate_custom_ref_levels(self) -> None:
        """Test slew rate with custom reference levels."""
        sample_rate = 1e6

        ramp = np.linspace(0, 1, 100)
        signal = np.concatenate([np.zeros(100), ramp, np.ones(800)])

        trace = make_waveform_trace(signal, sample_rate)

        # Use 10%-90% instead of 20%-80%
        sr = slew_rate(trace, ref_levels=(0.1, 0.9), edge_type="rising")

        assert sr > 0

    def test_slew_rate_return_all(self) -> None:
        """Test slew rate with return_all=True."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)
        trace = make_waveform_trace(signal, sample_rate)

        slew_rates = slew_rate(trace, edge_type="both", return_all=True)

        assert isinstance(slew_rates, np.ndarray)
        assert len(slew_rates) > 0

    def test_slew_rate_insufficient_data(self) -> None:
        """Test slew rate with insufficient data."""
        sample_rate = 1e6

        signal = np.array([0.0, 1.0])  # Only 2 samples
        trace = make_waveform_trace(signal, sample_rate)

        sr = slew_rate(trace)

        assert np.isnan(sr)

    def test_slew_rate_constant_signal(self) -> None:
        """Test slew rate on constant signal."""
        sample_rate = 1e6

        signal = np.ones(1000)
        trace = make_waveform_trace(signal, sample_rate)

        sr = slew_rate(trace)

        # No edges = NaN
        assert np.isnan(sr)

    def test_slew_rate_zero_amplitude(self) -> None:
        """Test slew rate with zero amplitude signal."""
        sample_rate = 1e6

        signal = np.zeros(1000)
        trace = make_waveform_trace(signal, sample_rate)

        sr = slew_rate(trace)

        assert np.isnan(sr)


# =============================================================================
# Phase Measurement Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-005")
class TestPhase:
    """Test phase difference measurements."""

    def test_phase_edge_method_zero_phase(self) -> None:
        """Test phase measurement with zero phase difference."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)

        phase_deg = phase(trace1, trace2, method="edge", unit="degrees")

        # Should be near zero
        assert abs(phase_deg) < 10

    def test_phase_edge_method_90_degrees(self) -> None:
        """Test phase measurement with 90 degree phase shift."""
        sample_rate = 1e6
        frequency = 1000.0
        period = 1.0 / frequency

        signal1 = make_square_wave(frequency, sample_rate, 0.01)
        # Shift by 1/4 period (90 degrees)
        shift_samples = int(sample_rate * period / 4)
        signal2 = np.roll(signal1, shift_samples)

        trace1 = make_waveform_trace(signal1, sample_rate)
        trace2 = make_waveform_trace(signal2, sample_rate)

        phase_deg = phase(trace1, trace2, method="edge", unit="degrees")

        # Should be near 90 degrees
        assert 70 < abs(phase_deg) < 110

    def test_phase_edge_method_180_degrees(self) -> None:
        """Test phase measurement with 180 degree phase shift."""
        sample_rate = 1e6
        frequency = 1000.0
        period = 1.0 / frequency

        signal1 = make_square_wave(frequency, sample_rate, 0.01)
        # Shift by 1/2 period (180 degrees)
        shift_samples = int(sample_rate * period / 2)
        signal2 = np.roll(signal1, shift_samples)

        trace1 = make_waveform_trace(signal1, sample_rate)
        trace2 = make_waveform_trace(signal2, sample_rate)

        phase_deg = phase(trace1, trace2, method="edge", unit="degrees")

        # Phase wraps around, so 180 degrees could be measured as -180 or 180
        # or even as near 0 if edges align differently. Accept a wide range
        # that includes the possibility of phase wrap-around
        assert abs(phase_deg) > 50 or abs(abs(phase_deg) - 180) < 30

    def test_phase_edge_method_radians(self) -> None:
        """Test phase measurement in radians."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)

        phase_rad = phase(trace1, trace2, method="edge", unit="radians")

        # Should be near zero
        assert abs(phase_rad) < 0.2

    def test_phase_fft_method(self) -> None:
        """Test phase measurement using FFT method."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)

        phase_deg = phase(trace1, trace2, method="fft", unit="degrees")

        # Should be near zero
        assert abs(phase_deg) < 10

    def test_phase_insufficient_edges(self) -> None:
        """Test phase with insufficient edges."""
        sample_rate = 1e6

        signal = np.ones(1000)  # No edges

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)

        phase_deg = phase(trace1, trace2, method="edge", unit="degrees")

        assert np.isnan(phase_deg)

    def test_phase_insufficient_samples_fft(self) -> None:
        """Test phase FFT method with insufficient samples."""
        sample_rate = 1e6

        signal = np.ones(10)  # Only 10 samples

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)

        phase_deg = phase(trace1, trace2, method="fft", unit="degrees")

        assert np.isnan(phase_deg)

    def test_phase_invalid_method(self) -> None:
        """Test phase with invalid method."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.001)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)

        with pytest.raises(ValueError, match="Unknown method"):
            phase(trace1, trace2, method="invalid")  # type: ignore


# =============================================================================
# Skew Measurement Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-006")
class TestSkew:
    """Test timing skew measurements."""

    def test_skew_identical_signals(self) -> None:
        """Test skew measurement with identical signals."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)
        trace3 = make_waveform_trace(signal, sample_rate)

        result = skew([trace1, trace2, trace3])

        assert len(result["skew_values"]) == 2  # Non-reference traces
        assert abs(result["mean"]) < 1e-6
        assert result["range"] < 1e-6

    def test_skew_with_delays(self) -> None:
        """Test skew measurement with delayed signals."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(np.roll(signal, 5), sample_rate)
        trace3 = make_waveform_trace(np.roll(signal, 10), sample_rate)

        result = skew([trace1, trace2, trace3])

        # Should detect skew
        assert result["range"] > 0
        assert result["max"] > result["min"]

    def test_skew_custom_reference(self) -> None:
        """Test skew measurement with custom reference index."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)

        trace1 = make_waveform_trace(np.roll(signal, 5), sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)  # Use as reference
        trace3 = make_waveform_trace(np.roll(signal, 10), sample_rate)

        result = skew([trace1, trace2, trace3], reference_idx=1)

        assert len(result["skew_values"]) == 2

    def test_skew_falling_edges(self) -> None:
        """Test skew measurement using falling edges."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(np.roll(signal, 5), sample_rate)

        result = skew([trace1, trace2], edge_type="falling")

        assert len(result["skew_values"]) == 1

    def test_skew_insufficient_traces(self) -> None:
        """Test skew with insufficient traces."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.001)

        trace1 = make_waveform_trace(signal, sample_rate)

        with pytest.raises(ValueError, match="at least 2 traces"):
            skew([trace1])

    def test_skew_invalid_reference_idx(self) -> None:
        """Test skew with invalid reference index."""
        sample_rate = 1e6
        signal = make_square_wave(1000.0, sample_rate, 0.001)

        trace1 = make_waveform_trace(signal, sample_rate)
        trace2 = make_waveform_trace(signal, sample_rate)

        with pytest.raises(ValueError, match="reference_idx.*out of range"):
            skew([trace1, trace2], reference_idx=5)

    def test_skew_no_edges(self) -> None:
        """Test skew with no edges in reference."""
        sample_rate = 1e6

        signal1 = np.ones(1000)  # No edges
        signal2 = make_square_wave(1000.0, sample_rate, 0.001)

        trace1 = make_waveform_trace(signal1, sample_rate)
        trace2 = make_waveform_trace(signal2, sample_rate)

        result = skew([trace1, trace2])

        assert len(result["skew_values"]) == 0
        assert np.isnan(result["mean"])

    def test_skew_digital_traces(self) -> None:
        """Test skew with DigitalTrace objects."""
        sample_rate = 1e6
        frequency = 1000.0

        signal = make_square_wave(frequency, sample_rate, 0.01) > 0.5

        trace1 = make_digital_trace(signal, sample_rate)
        trace2 = make_digital_trace(np.roll(signal, 5), sample_rate)

        result = skew([trace1, trace2])

        assert result["range"] > 0


# =============================================================================
# Clock Recovery FFT Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-003")
class TestRecoverClockFFT:
    """Test FFT-based clock recovery."""

    def test_recover_clock_fft_basic(self) -> None:
        """Test basic FFT clock recovery."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_fft(trace)

        # Should recover frequency within 5%
        assert abs(result.frequency - clock_freq) / clock_freq < 0.05
        assert result.method == "fft"
        assert result.confidence > 0

    def test_recover_clock_fft_with_freq_range(self) -> None:
        """Test FFT clock recovery with frequency range."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_fft(trace, min_freq=5e6, max_freq=15e6)

        assert 5e6 <= result.frequency <= 15e6

    def test_recover_clock_fft_digital_trace(self) -> None:
        """Test FFT clock recovery with DigitalTrace."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) > 0.5
        trace = make_digital_trace(signal, sample_rate)

        result = recover_clock_fft(trace)

        assert abs(result.frequency - clock_freq) / clock_freq < 0.1

    def test_recover_clock_fft_insufficient_samples(self) -> None:
        """Test FFT clock recovery with insufficient samples."""
        sample_rate = 1e6
        signal = np.ones(10)  # Only 10 samples (need 64)
        trace = make_waveform_trace(signal, sample_rate)

        with pytest.raises(InsufficientDataError) as exc_info:
            recover_clock_fft(trace)

        assert "at least 64 samples" in str(exc_info.value)
        assert "edge-based clock recovery" in str(exc_info.value)

    def test_recover_clock_fft_no_valid_frequencies(self) -> None:
        """Test FFT clock recovery with no frequencies in range."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        # Request impossible frequency range (10 MHz signal but looking at 90-100 MHz)
        with pytest.raises(ValueError) as exc_info:
            recover_clock_fft(trace, min_freq=90e6, max_freq=100e6)

        assert "No frequency components found" in str(exc_info.value)
        assert "90000000 Hz, 100000000 Hz" in str(exc_info.value)

    def test_recover_clock_fft_period(self) -> None:
        """Test that FFT recovery calculates correct period."""
        sample_rate = 100e6
        clock_freq = 10e6
        expected_period = 1.0 / clock_freq

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_fft(trace)

        assert abs(result.period - expected_period) / expected_period < 0.05


# =============================================================================
# Clock Recovery Edge Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("QUAL-004")
class TestRecoverClockEdge:
    """Test edge-based clock recovery."""

    def test_recover_clock_edge_basic(self) -> None:
        """Test basic edge-based clock recovery."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace)

        # Should recover frequency within 5%
        assert abs(result.frequency - clock_freq) / clock_freq < 0.05
        assert result.method == "edge"
        assert result.jitter_rms is not None
        assert result.jitter_pp is not None

    def test_recover_clock_edge_rising(self) -> None:
        """Test edge recovery using rising edges."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace, edge_type="rising")

        assert result.frequency > 0
        assert result.method == "edge"

    def test_recover_clock_edge_falling(self) -> None:
        """Test edge recovery using falling edges."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace, edge_type="falling")

        assert result.frequency > 0

    def test_recover_clock_edge_with_jitter(self) -> None:
        """Test edge recovery on signal with jitter."""
        sample_rate = 100e6
        clock_freq = 10e6
        jitter_std = 1e-9  # 1 ns jitter

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100, jitter_std=jitter_std)
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace)

        # Should still recover frequency
        assert abs(result.frequency - clock_freq) / clock_freq < 0.1
        # Jitter should be detected
        assert result.jitter_rms > 0
        assert result.jitter_pp > 0

    def test_recover_clock_edge_custom_threshold(self) -> None:
        """Test edge recovery with custom threshold."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) * 2.0
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace, threshold=0.75)

        assert result.frequency > 0

    def test_recover_clock_edge_insufficient_edges(self) -> None:
        """Test edge recovery with insufficient edges."""
        sample_rate = 1e6

        # Only one transition
        signal = np.concatenate([np.zeros(500), np.ones(500)])
        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace)

        assert np.isnan(result.frequency)
        assert result.confidence == 0.0

    def test_recover_clock_edge_digital_trace(self) -> None:
        """Test edge recovery with DigitalTrace."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) > 0.5
        trace = make_digital_trace(signal, sample_rate)

        result = recover_clock_edge(trace)

        assert abs(result.frequency - clock_freq) / clock_freq < 0.1

    def test_recover_clock_edge_confidence(self) -> None:
        """Test that confidence decreases with jitter."""
        sample_rate = 100e6
        clock_freq = 10e6

        # Clean signal
        clean_signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        clean_trace = make_waveform_trace(clean_signal, sample_rate)
        clean_result = recover_clock_edge(clean_trace)

        # Jittery signal
        jittery_signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100, jitter_std=2e-9)
        jittery_trace = make_waveform_trace(jittery_signal, sample_rate)
        jittery_result = recover_clock_edge(jittery_trace)

        # Clean signal should have higher confidence
        assert clean_result.confidence >= jittery_result.confidence


# =============================================================================
# RMS Jitter Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-007")
class TestRMSJitter:
    """Test RMS jitter measurements."""

    def test_rms_jitter_clean_signal(self) -> None:
        """Test RMS jitter on clean signal."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = rms_jitter(trace)

        # Clean signal should have low jitter
        assert result.rms >= 0
        assert result.samples > 0
        assert result.edge_type == "rising"

    def test_rms_jitter_with_noise(self) -> None:
        """Test RMS jitter on signal with jitter."""
        sample_rate = 100e6
        clock_freq = 10e6
        jitter_std = 1e-9

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100, jitter_std=jitter_std)
        trace = make_waveform_trace(signal, sample_rate)

        result = rms_jitter(trace)

        # Should detect jitter
        assert result.rms > 0
        assert result.uncertainty > 0

    def test_rms_jitter_rising_edges(self) -> None:
        """Test RMS jitter using rising edges."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = rms_jitter(trace, edge_type="rising")

        assert result.edge_type == "rising"
        assert result.samples > 0

    def test_rms_jitter_falling_edges(self) -> None:
        """Test RMS jitter using falling edges."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = rms_jitter(trace, edge_type="falling")

        assert result.edge_type == "falling"
        assert result.samples > 0

    def test_rms_jitter_both_edges(self) -> None:
        """Test RMS jitter using both edge types."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        result = rms_jitter(trace, edge_type="both")

        assert result.edge_type == "both"
        # Should have more samples than single edge type
        assert result.samples > 50

    def test_rms_jitter_custom_threshold(self) -> None:
        """Test RMS jitter with custom threshold."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) * 2.0
        trace = make_waveform_trace(signal, sample_rate)

        result = rms_jitter(trace, threshold=0.75)

        assert result.samples > 0

    def test_rms_jitter_insufficient_edges(self) -> None:
        """Test RMS jitter with insufficient edges."""
        sample_rate = 1e6

        signal = np.concatenate([np.zeros(500), np.ones(500)])
        trace = make_waveform_trace(signal, sample_rate)

        result = rms_jitter(trace)

        assert np.isnan(result.rms)
        assert result.samples == 0

    def test_rms_jitter_digital_trace(self) -> None:
        """Test RMS jitter with DigitalTrace."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) > 0.5
        trace = make_digital_trace(signal, sample_rate)

        result = rms_jitter(trace)

        assert result.samples > 0


# =============================================================================
# Peak-to-Peak Jitter Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-008")
class TestPeakToPeakJitter:
    """Test peak-to-peak jitter measurements."""

    def test_peak_to_peak_jitter_clean_signal(self) -> None:
        """Test pk-pk jitter on clean signal."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        jitter_pp = peak_to_peak_jitter(trace)

        # Clean signal should have low jitter
        assert jitter_pp >= 0

    def test_peak_to_peak_jitter_with_noise(self) -> None:
        """Test pk-pk jitter on signal with jitter."""
        sample_rate = 100e6
        clock_freq = 10e6
        jitter_std = 1e-9

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100, jitter_std=jitter_std)
        trace = make_waveform_trace(signal, sample_rate)

        jitter_pp = peak_to_peak_jitter(trace)

        # Should detect jitter
        assert jitter_pp > 0

    def test_peak_to_peak_jitter_edge_types(self) -> None:
        """Test pk-pk jitter with different edge types."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        jitter_rising = peak_to_peak_jitter(trace, edge_type="rising")
        jitter_falling = peak_to_peak_jitter(trace, edge_type="falling")
        jitter_both = peak_to_peak_jitter(trace, edge_type="both")

        assert jitter_rising >= 0
        assert jitter_falling >= 0
        assert jitter_both >= 0

    def test_peak_to_peak_jitter_custom_threshold(self) -> None:
        """Test pk-pk jitter with custom threshold."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) * 3.0
        trace = make_waveform_trace(signal, sample_rate)

        jitter_pp = peak_to_peak_jitter(trace, threshold=0.8)

        assert jitter_pp >= 0

    def test_peak_to_peak_jitter_insufficient_edges(self) -> None:
        """Test pk-pk jitter with insufficient edges."""
        sample_rate = 1e6

        signal = np.concatenate([np.zeros(500), np.ones(500)])
        trace = make_waveform_trace(signal, sample_rate)

        jitter_pp = peak_to_peak_jitter(trace)

        assert np.isnan(jitter_pp)

    def test_peak_to_peak_jitter_digital_trace(self) -> None:
        """Test pk-pk jitter with DigitalTrace."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) > 0.5
        trace = make_digital_trace(signal, sample_rate)

        jitter_pp = peak_to_peak_jitter(trace)

        assert jitter_pp >= 0

    def test_peak_to_peak_greater_than_rms(self) -> None:
        """Test that pk-pk jitter is greater than RMS jitter."""
        sample_rate = 100e6
        clock_freq = 10e6
        jitter_std = 1e-9

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100, jitter_std=jitter_std)
        trace = make_waveform_trace(signal, sample_rate)

        rms_result = rms_jitter(trace)
        pp_result = peak_to_peak_jitter(trace)

        # Pk-pk should be >= RMS
        assert pp_result >= rms_result.rms


# =============================================================================
# Time Interval Error Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("TIM-009")
class TestTimeIntervalError:
    """Test time interval error (TIE) measurements."""

    def test_tie_basic(self) -> None:
        """Test basic TIE measurement."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace)

        # Should return array of TIE values
        assert isinstance(tie, np.ndarray)
        assert len(tie) > 0

    def test_tie_rising_edges(self) -> None:
        """Test TIE using rising edges."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace, edge_type="rising")

        assert len(tie) > 0

    def test_tie_falling_edges(self) -> None:
        """Test TIE using falling edges."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace, edge_type="falling")

        assert len(tie) > 0

    def test_tie_with_nominal_period(self) -> None:
        """Test TIE with specified nominal period."""
        sample_rate = 100e6
        clock_freq = 10e6
        nominal_period = 1.0 / clock_freq

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace, nominal_period=nominal_period)

        assert len(tie) > 0

    def test_tie_clean_signal_near_zero(self) -> None:
        """Test that TIE of clean signal is near zero."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100)
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace)

        # Mean TIE should be near zero for clean signal
        assert abs(np.mean(tie)) < 1e-8

    def test_tie_with_jitter(self) -> None:
        """Test TIE on signal with jitter."""
        sample_rate = 100e6
        clock_freq = 10e6
        jitter_std = 1e-9

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100, jitter_std=jitter_std)
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace)

        # TIE should show variation
        assert np.std(tie) > 0

    def test_tie_insufficient_edges(self) -> None:
        """Test TIE with insufficient edges."""
        sample_rate = 1e6

        signal = np.concatenate([np.zeros(500), np.ones(500)])
        trace = make_waveform_trace(signal, sample_rate)

        with pytest.raises(InsufficientDataError) as exc_info:
            time_interval_error(trace)

        assert "at least 3 edges" in str(exc_info.value)

    def test_tie_custom_threshold(self) -> None:
        """Test TIE with custom threshold."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) * 2.0
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace, threshold=0.75)

        assert len(tie) > 0

    def test_tie_digital_trace(self) -> None:
        """Test TIE with DigitalTrace."""
        sample_rate = 100e6
        clock_freq = 10e6

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=100) > 0.5
        trace = make_digital_trace(signal, sample_rate)

        tie = time_interval_error(trace)

        assert len(tie) > 0

    def test_tie_length_matches_edges(self) -> None:
        """Test that TIE array length matches number of edges."""
        sample_rate = 100e6
        clock_freq = 10e6
        n_cycles = 50

        signal = make_clock_signal(clock_freq, sample_rate, n_cycles=n_cycles)
        trace = make_waveform_trace(signal, sample_rate)

        tie = time_interval_error(trace, edge_type="rising")

        # Should have approximately n_cycles edges
        assert 40 < len(tie) < 60  # Allow some tolerance


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestDigitalTimingEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_trace(self) -> None:
        """Test timing functions with empty trace."""
        empty = np.array([])
        trace = make_waveform_trace(empty, 1e6)

        # Most functions should handle gracefully
        sr = slew_rate(trace)
        assert np.isnan(sr)

    def test_single_sample_trace(self) -> None:
        """Test timing functions with single sample."""
        single = np.array([1.0])
        trace = make_waveform_trace(single, 1e6)

        sr = slew_rate(trace)
        assert np.isnan(sr)

    def test_very_high_sample_rate(self) -> None:
        """Test with very high sample rate."""
        sample_rate = 1e12  # 1 TSa/s
        signal = make_square_wave(1e9, sample_rate, 1e-8)

        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace)
        assert result.frequency > 0

    def test_very_low_frequency(self) -> None:
        """Test with very low frequency signal."""
        sample_rate = 1e6
        signal = make_square_wave(1.0, sample_rate, 10.0)  # 1 Hz

        trace = make_waveform_trace(signal, sample_rate)

        result = recover_clock_edge(trace)
        assert result.frequency > 0

    def test_mixed_trace_types(self) -> None:
        """Test mixing WaveformTrace and DigitalTrace."""
        sample_rate = 1e6

        analog = make_square_wave(1000.0, sample_rate, 0.01)
        digital = analog > 0.5

        analog_trace = make_waveform_trace(analog, sample_rate)
        digital_trace = make_digital_trace(digital, sample_rate)

        # Should work with mixed types
        delay = propagation_delay(analog_trace, digital_trace)
        # Delay should be reasonable (could be up to 1 period at 1kHz = 1ms)
        assert delay >= 0
        assert delay < 0.002  # Less than 2 periods


# =============================================================================
# Module Exports Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
class TestModuleExports:
    """Test that all public APIs are exported."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from oscura.analyzers.digital import timing

        expected_exports = {
            "ClockRecoveryResult",
            "RMSJitterResult",
            "TimingViolation",
            "propagation_delay",
            "setup_time",
            "hold_time",
            "slew_rate",
            "phase",
            "skew",
            "recover_clock_fft",
            "recover_clock_edge",
            "rms_jitter",
            "peak_to_peak_jitter",
            "time_interval_error",
        }

        assert hasattr(timing, "__all__")
        assert set(timing.__all__) == expected_exports

    def test_dataclasses_importable(self) -> None:
        """Test that all dataclasses are importable."""
        from oscura.analyzers.digital.timing import (
            ClockRecoveryResult,
            RMSJitterResult,
            TimingViolation,
        )

        assert ClockRecoveryResult is not None
        assert RMSJitterResult is not None
        assert TimingViolation is not None

    def test_functions_importable(self) -> None:
        """Test that all functions are importable."""
        from oscura.analyzers.digital.timing import (
            hold_time,
            peak_to_peak_jitter,
            phase,
            propagation_delay,
            recover_clock_edge,
            recover_clock_fft,
            rms_jitter,
            setup_time,
            skew,
            slew_rate,
            time_interval_error,
        )

        assert propagation_delay is not None
        assert setup_time is not None
        assert hold_time is not None
        assert slew_rate is not None
        assert phase is not None
        assert skew is not None
        assert recover_clock_fft is not None
        assert recover_clock_edge is not None
        assert rms_jitter is not None
        assert peak_to_peak_jitter is not None
        assert time_interval_error is not None
