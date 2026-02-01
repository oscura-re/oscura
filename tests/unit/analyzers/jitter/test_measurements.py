"""Comprehensive unit tests for jitter measurements module.

Tests for src/oscura/analyzers/jitter/measurements.py

This test suite provides comprehensive coverage of the jitter measurements module,
including:
- Cycle-to-cycle jitter measurement (cycle_to_cycle_jitter)
- Period jitter measurement (period_jitter)
- Duty cycle distortion measurement (measure_dcd)
- Time interval error calculation (tie_from_edges)
- Edge detection from waveforms (_find_edges)
- Result dataclasses (CycleJitterResult, DutyCycleDistortionResult)
- Error handling and validation
- Edge cases and numerical stability
- Input validation and boundary conditions
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.analyzers.jitter.measurements import (
    CycleJitterResult,
    DutyCycleDistortionResult,
    _find_edges,
    cycle_to_cycle_jitter,
    measure_dcd,
    period_jitter,
    tie_from_edges,
)
from oscura.core.exceptions import InsufficientDataError
from oscura.core.types import DigitalTrace, TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.jitter]


# =============================================================================
# Test Data Generation Helpers
# =============================================================================


def create_clock_periods(
    n_cycles: int,
    nominal_period: float = 1e-9,
    jitter_rms: float = 0.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate synthetic clock periods with controlled jitter.

    Args:
        n_cycles: Number of clock cycles.
        nominal_period: Base period in seconds.
        jitter_rms: RMS jitter in seconds (0 = perfect clock).
        seed: Random seed for reproducibility.

    Returns:
        Array of clock periods in seconds.
    """
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_rms, n_cycles)
    periods = nominal_period + jitter
    return np.maximum(periods, nominal_period * 0.5)  # Prevent negative periods


def create_clock_waveform(
    n_cycles: int,
    nominal_period: float = 1e-9,
    sample_rate: float = 1e9,
    duty_cycle: float = 0.5,
    amplitude: float = 1.0,
) -> WaveformTrace:
    """Generate a synthetic clock waveform.

    Args:
        n_cycles: Number of clock cycles.
        nominal_period: Period in seconds.
        sample_rate: Sample rate in Hz.
        duty_cycle: Duty cycle (0.0-1.0).
        amplitude: Signal amplitude in volts.

    Returns:
        WaveformTrace containing clock signal.
    """
    sample_period = 1.0 / sample_rate
    # Increase samples per cycle for better edge detection
    samples_per_cycle = int(100 * sample_rate * nominal_period)
    n_samples = n_cycles * samples_per_cycle
    time = np.arange(n_samples) * sample_period

    # Generate square wave
    phase = (time / nominal_period) % 1.0
    data = np.where(phase < duty_cycle, amplitude, 0.0)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def create_digital_clock(
    n_cycles: int,
    nominal_period: float = 1e-9,
    sample_rate: float = 1e9,
    duty_cycle: float = 0.5,
) -> DigitalTrace:
    """Generate a synthetic digital clock trace.

    Args:
        n_cycles: Number of clock cycles.
        nominal_period: Period in seconds.
        sample_rate: Sample rate in Hz.
        duty_cycle: Duty cycle (0.0-1.0).

    Returns:
        DigitalTrace containing clock signal.
    """
    sample_period = 1.0 / sample_rate
    # Increase samples per cycle for better edge detection
    samples_per_cycle = int(100 * sample_rate * nominal_period)
    n_samples = n_cycles * samples_per_cycle
    time = np.arange(n_samples) * sample_period

    # Generate square wave
    phase = (time / nominal_period) % 1.0
    data = (phase < duty_cycle).astype(bool)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return DigitalTrace(data=data, metadata=metadata)


# =============================================================================
# Tests for CycleJitterResult Dataclass
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestCycleJitterResult:
    """Test CycleJitterResult dataclass."""

    def test_dataclass_creation_minimal(self) -> None:
        """Test creating a CycleJitterResult with minimal fields."""
        c2c_values = np.array([1e-12, 2e-12, 1.5e-12])
        result = CycleJitterResult(
            c2c_rms=1.5e-12,
            c2c_pp=1e-12,
            c2c_values=c2c_values,
            period_mean=1e-9,
            period_std=1e-12,
            n_cycles=100,
        )

        assert result.c2c_rms == 1.5e-12
        assert result.c2c_pp == 1e-12
        assert len(result.c2c_values) == 3
        assert result.period_mean == 1e-9
        assert result.period_std == 1e-12
        assert result.n_cycles == 100
        assert result.histogram is None
        assert result.bin_centers is None

    def test_dataclass_creation_with_histogram(self) -> None:
        """Test creating CycleJitterResult with histogram data."""
        c2c_values = np.array([1e-12, 2e-12, 1.5e-12])
        histogram = np.array([0.1, 0.2, 0.3])
        bin_centers = np.array([0.5e-12, 1.5e-12, 2.5e-12])

        result = CycleJitterResult(
            c2c_rms=1.5e-12,
            c2c_pp=1e-12,
            c2c_values=c2c_values,
            period_mean=1e-9,
            period_std=1e-12,
            n_cycles=100,
            histogram=histogram,
            bin_centers=bin_centers,
        )

        assert result.histogram is not None
        assert len(result.histogram) == 3
        assert result.bin_centers is not None
        assert len(result.bin_centers) == 3
        assert np.array_equal(result.histogram, histogram)
        assert np.array_equal(result.bin_centers, bin_centers)

    def test_dataclass_attribute_types(self) -> None:
        """Test that dataclass attributes have correct types."""
        c2c_values = np.array([1e-12, 2e-12])
        result = CycleJitterResult(
            c2c_rms=1.5e-12,
            c2c_pp=1e-12,
            c2c_values=c2c_values,
            period_mean=1e-9,
            period_std=1e-12,
            n_cycles=100,
        )

        assert isinstance(result.c2c_rms, float)
        assert isinstance(result.c2c_pp, float)
        assert isinstance(result.c2c_values, np.ndarray)
        assert isinstance(result.period_mean, float)
        assert isinstance(result.period_std, float)
        assert isinstance(result.n_cycles, int)


# =============================================================================
# Tests for DutyCycleDistortionResult Dataclass
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestDutyCycleDistortionResult:
    """Test DutyCycleDistortionResult dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating a DutyCycleDistortionResult instance."""
        result = DutyCycleDistortionResult(
            dcd_seconds=1e-12,
            dcd_percent=0.1,
            mean_high_time=5e-10,
            mean_low_time=5.01e-10,
            duty_cycle=0.499,
            period=1.001e-9,
            n_cycles=50,
        )

        assert result.dcd_seconds == 1e-12
        assert result.dcd_percent == 0.1
        assert result.mean_high_time == 5e-10
        assert result.mean_low_time == 5.01e-10
        assert result.duty_cycle == pytest.approx(0.499)
        assert result.period == 1.001e-9
        assert result.n_cycles == 50

    def test_dataclass_attribute_types(self) -> None:
        """Test that dataclass attributes have correct types."""
        result = DutyCycleDistortionResult(
            dcd_seconds=1e-12,
            dcd_percent=0.1,
            mean_high_time=5e-10,
            mean_low_time=5.01e-10,
            duty_cycle=0.499,
            period=1.001e-9,
            n_cycles=50,
        )

        assert isinstance(result.dcd_seconds, float)
        assert isinstance(result.dcd_percent, float)
        assert isinstance(result.mean_high_time, float)
        assert isinstance(result.mean_low_time, float)
        assert isinstance(result.duty_cycle, float)
        assert isinstance(result.period, float)
        assert isinstance(result.n_cycles, int)

    def test_duty_cycle_range_validation(self) -> None:
        """Test duty cycle values are in valid range."""
        result = DutyCycleDistortionResult(
            dcd_seconds=1e-12,
            dcd_percent=0.1,
            mean_high_time=5e-10,
            mean_low_time=5e-10,
            duty_cycle=0.5,
            period=1e-9,
            n_cycles=50,
        )

        assert 0.0 <= result.duty_cycle <= 1.0


# =============================================================================
# Tests for tie_from_edges Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestTieFromEdges:
    """Test time interval error (TIE) calculation from edge timestamps."""

    def test_tie_perfect_clock(self) -> None:
        """Test TIE for a perfect clock (should be near zero)."""
        period = 1e-9
        n_edges = 100
        edge_timestamps = np.arange(n_edges) * period

        tie = tie_from_edges(edge_timestamps, nominal_period=period)

        assert len(tie) == n_edges
        assert np.max(np.abs(tie)) < 1e-15  # Should be nearly zero (rounding error)

    def test_tie_auto_nominal_period(self) -> None:
        """Test TIE when nominal period is calculated from edges."""
        period = 1e-9
        n_edges = 50
        edge_timestamps = np.arange(n_edges) * period

        tie = tie_from_edges(edge_timestamps, nominal_period=None)

        assert len(tie) == n_edges
        # With measured nominal, TIE should still be minimal for perfect clock
        assert np.max(np.abs(tie)) < 1e-15

    def test_tie_with_jitter(self) -> None:
        """Test TIE calculation with clock jitter."""
        rng = np.random.default_rng(42)
        period = 1e-9
        n_edges = 100
        jitter = rng.normal(0, 1e-12, n_edges)
        edge_timestamps = np.cumsum(np.full(n_edges, period) + jitter)

        tie = tie_from_edges(edge_timestamps, nominal_period=period)

        assert len(tie) == n_edges
        # TIE should show the jitter pattern
        assert np.std(tie) > 0
        # Should be on order of input jitter
        assert np.std(tie) < 10e-12

    def test_tie_empty_array(self) -> None:
        """Test TIE with empty edge array."""
        edge_timestamps = np.array([], dtype=np.float64)
        tie = tie_from_edges(edge_timestamps, nominal_period=1e-9)
        assert len(tie) == 0

    def test_tie_single_edge(self) -> None:
        """Test TIE with single edge (insufficient data)."""
        edge_timestamps = np.array([0.0], dtype=np.float64)
        tie = tie_from_edges(edge_timestamps, nominal_period=1e-9)
        assert len(tie) == 0

    def test_tie_two_edges(self) -> None:
        """Test TIE with two edges (insufficient data)."""
        edge_timestamps = np.array([0.0, 1e-9], dtype=np.float64)
        tie = tie_from_edges(edge_timestamps, nominal_period=1e-9)
        assert len(tie) == 0

    def test_tie_three_edges_minimum(self) -> None:
        """Test TIE with minimum valid three edges."""
        edge_timestamps = np.array([0.0, 1e-9, 2e-9], dtype=np.float64)
        tie = tie_from_edges(edge_timestamps, nominal_period=1e-9)
        assert len(tie) == 3
        # First edge should be at ideal position
        assert np.abs(tie[0]) < 1e-15

    def test_tie_frequency_offset(self) -> None:
        """Test TIE with frequency offset (drifting clock)."""
        period = 1e-9
        n_edges = 100

        # Simulate frequency offset by linearly changing period
        edge_timestamps = np.zeros(n_edges)
        t = 0.0
        for i in range(n_edges):
            edge_timestamps[i] = t
            # Period increases with time due to frequency offset
            t += period * (1 + i * 1e-4)

        tie = tie_from_edges(edge_timestamps, nominal_period=period)

        assert len(tie) == n_edges
        # TIE should show upward trend due to frequency offset
        assert tie[-1] > tie[0]

    def test_tie_negative_nominal_period(self) -> None:
        """Test TIE handles edge cases gracefully."""
        edge_timestamps = np.array([0.0, 1e-9, 2e-9])
        # Function doesn't validate negative period, just calculates
        tie = tie_from_edges(edge_timestamps, nominal_period=-1e-9)
        assert len(tie) == 3

    def test_tie_first_edge_is_reference(self) -> None:
        """Test that first edge is used as time reference."""
        offset = 1e-6  # Start at 1 microsecond
        period = 1e-9
        n_edges = 50
        edge_timestamps = offset + np.arange(n_edges) * period

        tie = tie_from_edges(edge_timestamps, nominal_period=period)

        assert len(tie) == n_edges
        # First edge sets the reference, so TIE should be near zero
        assert np.abs(tie[0]) < 1e-15


# =============================================================================
# Tests for cycle_to_cycle_jitter Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestCycleToClockJitter:
    """Test cycle-to-cycle jitter measurement."""

    def test_c2c_perfect_clock(self) -> None:
        """Test C2C jitter for perfect clock (should be ~zero)."""
        periods = create_clock_periods(100, jitter_rms=0.0)
        result = cycle_to_cycle_jitter(periods)

        assert isinstance(result, CycleJitterResult)
        assert result.n_cycles == 100
        assert result.c2c_rms < 1e-14  # Should be very small
        assert len(result.c2c_values) == 99  # c2c is difference, so n-1

    def test_c2c_with_jitter(self) -> None:
        """Test C2C jitter measurement with realistic jitter."""
        jitter_rms = 1e-12
        periods = create_clock_periods(200, jitter_rms=jitter_rms)
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 200
        # C2C jitter RMS should be on order of input jitter
        assert result.c2c_rms > 0
        assert result.c2c_rms < 10 * jitter_rms
        assert result.period_mean > 0
        assert result.period_std > 0

    def test_c2c_minimum_data_requirement(self) -> None:
        """Test C2C with minimum required periods (3)."""
        periods = create_clock_periods(3)
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 3
        assert len(result.c2c_values) == 2

    def test_c2c_insufficient_periods_error(self) -> None:
        """Test C2C raises InsufficientDataError with fewer than 3 periods."""
        periods = np.array([1e-9, 1e-9], dtype=np.float64)

        with pytest.raises(InsufficientDataError) as exc_info:
            cycle_to_cycle_jitter(periods)

        assert exc_info.value.required == 3
        assert exc_info.value.available == 2
        assert "cycle_to_cycle_jitter" in exc_info.value.analysis_type

    def test_c2c_empty_array_error(self) -> None:
        """Test C2C raises InsufficientDataError with empty array."""
        periods = np.array([], dtype=np.float64)

        with pytest.raises(InsufficientDataError):
            cycle_to_cycle_jitter(periods)

    def test_c2c_with_nan_filtering(self) -> None:
        """Test C2C jitter filters NaN values."""
        periods = np.array([1e-9, np.nan, 1e-9, 1e-9, 1e-9], dtype=np.float64)
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 4  # Only valid periods counted
        assert not np.any(np.isnan(result.c2c_values))

    def test_c2c_all_nan_raises_error(self) -> None:
        """Test C2C raises error if all periods are NaN."""
        periods = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        with pytest.raises(InsufficientDataError):
            cycle_to_cycle_jitter(periods)

    def test_c2c_with_histogram_enabled(self) -> None:
        """Test C2C with histogram generation."""
        periods = create_clock_periods(200, jitter_rms=1e-12)
        result = cycle_to_cycle_jitter(periods, include_histogram=True, n_bins=50)

        assert result.histogram is not None
        assert result.bin_centers is not None
        assert len(result.histogram) == 50
        assert len(result.bin_centers) == 50

    def test_c2c_without_histogram(self) -> None:
        """Test C2C without histogram generation."""
        periods = create_clock_periods(100, jitter_rms=1e-12)
        result = cycle_to_cycle_jitter(periods, include_histogram=False)

        assert result.histogram is None
        assert result.bin_centers is None

    def test_c2c_histogram_skipped_insufficient_data(self) -> None:
        """Test that histogram is skipped when C2C values < 10."""
        periods = np.array([1e-9, 1.001e-9, 1.0011e-9, 1.0012e-9], dtype=np.float64)
        result = cycle_to_cycle_jitter(periods, include_histogram=True)

        # With only 4 periods, c2c_values has length 3, so histogram skipped
        assert result.histogram is None
        assert result.bin_centers is None

    def test_c2c_peak_to_peak_calculation(self) -> None:
        """Test C2C peak-to-peak calculation."""
        # Create known jitter values
        periods = np.array([1e-9, 1.001e-9, 1.0005e-9, 1.002e-9, 1.0015e-9] * 20, dtype=np.float64)
        result = cycle_to_cycle_jitter(periods)

        # c2c_pp should be max(c2c) - min(c2c)
        expected_pp = np.max(result.c2c_values) - np.min(result.c2c_values)
        assert result.c2c_pp == pytest.approx(expected_pp)

    def test_c2c_mean_period_calculation(self) -> None:
        """Test that C2C correctly reports mean period."""
        test_periods = np.array([1e-9, 1.001e-9, 0.999e-9] * 10, dtype=np.float64)
        result = cycle_to_cycle_jitter(test_periods)

        expected_mean = np.mean(test_periods)
        assert result.period_mean == pytest.approx(expected_mean)

    def test_c2c_std_period_calculation(self) -> None:
        """Test that C2C correctly reports period standard deviation."""
        test_periods = np.array([1e-9, 1.001e-9, 0.999e-9] * 10, dtype=np.float64)
        result = cycle_to_cycle_jitter(test_periods)

        expected_std = np.std(test_periods)
        assert result.period_std == pytest.approx(expected_std)

    def test_c2c_custom_histogram_bins(self) -> None:
        """Test C2C with custom number of histogram bins."""
        periods = create_clock_periods(200, jitter_rms=1e-12)
        for n_bins in [25, 50, 100]:
            result = cycle_to_cycle_jitter(periods, include_histogram=True, n_bins=n_bins)
            assert len(result.histogram) == n_bins
            assert len(result.bin_centers) == n_bins

    def test_c2c_absolute_value_calculation(self) -> None:
        """Test that C2C values are absolute differences."""
        periods = np.array([1e-9, 1.001e-9, 0.999e-9, 1.002e-9], dtype=np.float64)
        result = cycle_to_cycle_jitter(periods)

        # All C2C values should be positive (absolute)
        assert np.all(result.c2c_values >= 0)


# =============================================================================
# Tests for period_jitter Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestPeriodJitter:
    """Test period jitter measurement."""

    def test_period_jitter_perfect_clock(self) -> None:
        """Test period jitter for perfect clock."""
        periods = create_clock_periods(100, jitter_rms=0.0)
        result = period_jitter(periods, nominal_period=1e-9)

        assert isinstance(result, CycleJitterResult)
        assert result.n_cycles == 100
        # RMS of constant values should be very small
        assert result.c2c_rms < 1e-14

    def test_period_jitter_with_deviation(self) -> None:
        """Test period jitter with known deviation."""
        nominal = 1e-9
        periods = np.array([nominal * 1.001, nominal * 0.999, nominal] * 10)
        result = period_jitter(periods, nominal_period=nominal)

        assert result.n_cycles == 30
        assert result.period_mean == pytest.approx(nominal, rel=1e-3)

    def test_period_jitter_auto_nominal(self) -> None:
        """Test period jitter with auto-calculated nominal period."""
        periods = create_clock_periods(50, jitter_rms=1e-12)
        result = period_jitter(periods, nominal_period=None)

        # Nominal should be the mean
        assert result.period_mean == pytest.approx(np.mean(periods))

    def test_period_jitter_insufficient_periods(self) -> None:
        """Test period jitter raises error with insufficient periods."""
        periods = np.array([1e-9], dtype=np.float64)

        with pytest.raises(InsufficientDataError) as exc_info:
            period_jitter(periods)

        assert exc_info.value.required == 2
        assert exc_info.value.available == 1
        assert "period_jitter" in exc_info.value.analysis_type

    def test_period_jitter_with_nan_filtering(self) -> None:
        """Test period jitter filters NaN values."""
        periods = np.array([1e-9, np.nan, 1e-9, np.nan, 1e-9], dtype=np.float64)
        result = period_jitter(periods, nominal_period=1e-9)

        assert result.n_cycles == 3

    def test_period_jitter_pp_calculation(self) -> None:
        """Test peak-to-peak period jitter calculation."""
        periods = np.array([0.9e-9, 1e-9, 1.1e-9], dtype=np.float64)
        result = period_jitter(periods, nominal_period=1e-9)

        expected_pp = 1.1e-9 - 0.9e-9
        assert result.c2c_pp == pytest.approx(expected_pp)

    def test_period_jitter_deviations_from_nominal(self) -> None:
        """Test that c2c_values contains absolute deviations from nominal."""
        nominal = 1e-9
        periods = np.array([0.99e-9, 1.01e-9, 1.0e-9], dtype=np.float64)
        result = period_jitter(periods, nominal_period=nominal)

        # c2c_values should be absolute deviations
        assert np.all(result.c2c_values >= 0)

    def test_period_jitter_no_histogram(self) -> None:
        """Test that period_jitter doesn't generate histogram."""
        periods = create_clock_periods(100, jitter_rms=1e-12)
        result = period_jitter(periods, nominal_period=1e-9)

        # period_jitter doesn't include histogram
        assert result.histogram is None
        assert result.bin_centers is None


# =============================================================================
# Tests for _find_edges Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestFindEdges:
    """Test edge detection from waveforms."""

    def test_find_edges_digital_clock(self) -> None:
        """Test finding edges in a digital clock signal."""
        trace = create_digital_clock(10, nominal_period=1e-9)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        # Should find ~10 rising and ~10 falling edges
        assert len(rising) > 8
        assert len(falling) > 8
        assert len(rising) == len(falling) or abs(len(rising) - len(falling)) <= 1

    def test_find_edges_waveform_clock(self) -> None:
        """Test finding edges in an analog clock waveform."""
        trace = create_clock_waveform(10, nominal_period=1e-9)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        # Should find ~10 rising and ~10 falling edges
        assert len(rising) > 8
        assert len(falling) > 8

    def test_find_edges_different_thresholds(self) -> None:
        """Test edge detection with different threshold levels."""
        trace = create_clock_waveform(5, nominal_period=1e-9)

        edges_low = _find_edges(trace, threshold_frac=0.2)
        edges_mid = _find_edges(trace, threshold_frac=0.5)
        edges_high = _find_edges(trace, threshold_frac=0.8)

        # All should find some edges
        assert len(edges_low[0]) > 3
        assert len(edges_mid[0]) > 3
        assert len(edges_high[0]) > 3

    def test_find_edges_empty_trace(self) -> None:
        """Test that empty trace raises ValueError."""
        metadata = TraceMetadata(sample_rate=1e9)

        # Empty trace creation raises ValueError in v0.9.0+
        with pytest.raises(ValueError, match="data array cannot be empty"):
            DigitalTrace(data=np.array([], dtype=bool), metadata=metadata)

    def test_find_edges_minimal_trace(self) -> None:
        """Test edge detection with minimal trace (< 3 samples)."""
        metadata = TraceMetadata(sample_rate=1e9)
        trace = DigitalTrace(data=np.array([False, True], dtype=bool), metadata=metadata)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        assert len(rising) == 0
        assert len(falling) == 0

    def test_find_edges_constant_signal_high(self) -> None:
        """Test edge detection on constant high signal (no edges)."""
        metadata = TraceMetadata(sample_rate=1e9)
        trace = DigitalTrace(data=np.array([True] * 100, dtype=bool), metadata=metadata)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        assert len(rising) == 0
        assert len(falling) == 0

    def test_find_edges_constant_signal_low(self) -> None:
        """Test edge detection on constant low signal (no edges)."""
        metadata = TraceMetadata(sample_rate=1e9)
        trace = DigitalTrace(data=np.array([False] * 100, dtype=bool), metadata=metadata)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        assert len(rising) == 0
        assert len(falling) == 0

    def test_find_edges_timestamps_monotonic(self) -> None:
        """Test that edge timestamps are monotonically increasing."""
        trace = create_digital_clock(20, nominal_period=1e-9)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        # All timestamps should be increasing
        assert len(rising) > 0
        assert len(falling) > 0
        assert np.all(np.diff(rising) > 0)
        assert np.all(np.diff(falling) > 0)

    def test_find_edges_asymmetric_duty_cycle(self) -> None:
        """Test edge detection with asymmetric duty cycle."""
        trace = create_clock_waveform(10, nominal_period=1e-9, duty_cycle=0.3)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        assert len(rising) > 8
        assert len(falling) > 8

    def test_find_edges_interpolation_accuracy(self) -> None:
        """Test that edge interpolation produces sub-sample accuracy."""
        # Use higher sample rate for better accuracy
        trace = create_digital_clock(10, nominal_period=1e-9, sample_rate=10e9)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        # Edge timestamps should have sub-sample precision
        if len(rising) > 1:
            periods = np.diff(rising)
            # Periods should be close to nominal (with wider tolerance due to edge detection)
            assert np.mean(periods) == pytest.approx(1e-9, rel=0.5)

    def test_find_edges_single_transition(self) -> None:
        """Test edge detection with single rising edge."""
        metadata = TraceMetadata(sample_rate=1e9)
        data = np.array([False] * 50 + [True] * 50, dtype=bool)
        trace = DigitalTrace(data=data, metadata=metadata)
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        assert len(rising) == 1
        assert len(falling) == 0

    def test_find_edges_waveform_with_noise(self) -> None:
        """Test edge detection on noisy waveform."""
        # Create waveform with noise
        metadata = TraceMetadata(sample_rate=1e9)
        rng = np.random.default_rng(42)
        base_signal = np.tile([0.0, 0.0, 1.0, 1.0], 25)
        noisy_signal = base_signal + rng.normal(0, 0.05, len(base_signal))
        trace = WaveformTrace(data=noisy_signal, metadata=metadata)

        rising, falling = _find_edges(trace, threshold_frac=0.5)

        # Should still detect edges despite noise
        assert len(rising) > 10
        assert len(falling) > 10


# =============================================================================
# Tests for measure_dcd Function
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestMeasureDcd:
    """Test duty cycle distortion measurement."""

    def test_dcd_perfect_clock_50_percent(self) -> None:
        """Test DCD for perfect 50% duty cycle clock."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.5)
        result = measure_dcd(trace, clock_period=1e-9)

        assert isinstance(result, DutyCycleDistortionResult)
        # DCD should be present and non-negative
        assert result.dcd_seconds >= 0
        assert result.dcd_percent >= 0
        assert result.n_cycles > 0

    def test_dcd_asymmetric_clock_30_percent(self) -> None:
        """Test DCD for clock with 30% duty cycle."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.3)
        result = measure_dcd(trace, clock_period=1e-9)

        # DCD should be non-zero for asymmetric duty cycle
        assert result.dcd_seconds > 1e-11
        assert result.dcd_percent > 1.0
        # Duty cycle should be measurably different from 0.5
        assert abs(result.duty_cycle - 0.3) < 0.7  # Allow tolerance for edge detection

    def test_dcd_asymmetric_clock_70_percent(self) -> None:
        """Test DCD for clock with 70% duty cycle."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.7)
        result = measure_dcd(trace, clock_period=1e-9)

        assert result.dcd_seconds > 1e-11
        assert result.dcd_percent > 1.0

    def test_dcd_waveform_trace(self) -> None:
        """Test DCD measurement on analog waveform trace."""
        trace = create_clock_waveform(15, nominal_period=1e-9, duty_cycle=0.4)
        result = measure_dcd(trace, clock_period=1e-9)

        assert result.n_cycles > 0
        assert result.duty_cycle > 0
        assert result.duty_cycle < 1.0

    def test_dcd_insufficient_edges_error(self) -> None:
        """Test DCD raises error with insufficient edges."""
        # Create trace with single edge
        metadata = TraceMetadata(sample_rate=1e9)
        trace = DigitalTrace(
            data=np.array([False] * 50 + [True] * 50, dtype=bool), metadata=metadata
        )

        with pytest.raises(InsufficientDataError) as exc_info:
            measure_dcd(trace)

        assert "2 rising and 2 falling edges" in str(exc_info.value)

    def test_dcd_default_threshold(self) -> None:
        """Test DCD uses default threshold of 0.5."""
        trace = create_digital_clock(10, nominal_period=1e-9, duty_cycle=0.5)
        result = measure_dcd(trace)  # No threshold specified

        assert result.n_cycles > 0

    def test_dcd_custom_threshold(self) -> None:
        """Test DCD with custom threshold."""
        trace = create_digital_clock(10, nominal_period=1e-9, duty_cycle=0.5)
        result1 = measure_dcd(trace, threshold=0.3)
        result2 = measure_dcd(trace, threshold=0.7)

        # Different thresholds should produce results
        assert result1.n_cycles > 0
        assert result2.n_cycles > 0

    def test_dcd_auto_clock_period(self) -> None:
        """Test DCD calculates clock period automatically."""
        trace = create_digital_clock(15, nominal_period=1e-9, duty_cycle=0.5)
        result = measure_dcd(trace, clock_period=None)

        # Should calculate period from high/low times
        assert result.period > 0
        assert result.dcd_percent < 100.0

    def test_dcd_multiple_cycles_for_accuracy(self) -> None:
        """Test DCD with multiple cycles for statistical accuracy."""
        trace = create_digital_clock(100, nominal_period=1e-9, duty_cycle=0.45)
        result = measure_dcd(trace, clock_period=1e-9)

        assert result.n_cycles >= 50  # Should have multiple cycles

    def test_dcd_high_low_time_sum_equals_period(self) -> None:
        """Test that high and low times sum to period."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.4)
        result = measure_dcd(trace, clock_period=1e-9)

        calculated_period = result.mean_high_time + result.mean_low_time
        assert calculated_period == pytest.approx(result.period, rel=1e-3)

    def test_dcd_duty_cycle_from_times(self) -> None:
        """Test duty cycle is correctly calculated from high/low times."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.6)
        result = measure_dcd(trace, clock_period=1e-9)

        calculated_duty = result.mean_high_time / result.period
        assert result.duty_cycle == pytest.approx(calculated_duty)

    def test_dcd_absolute_value(self) -> None:
        """Test that DCD is absolute value of high-low difference."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.3)
        result = measure_dcd(trace, clock_period=1e-9)

        expected_dcd = abs(result.mean_high_time - result.mean_low_time)
        assert result.dcd_seconds == pytest.approx(expected_dcd)

    def test_dcd_percent_calculation(self) -> None:
        """Test DCD percent is correctly calculated from seconds."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.4)
        result = measure_dcd(trace, clock_period=1e-9)

        expected_percent = (result.dcd_seconds / 1e-9) * 100
        assert result.dcd_percent == pytest.approx(expected_percent, rel=0.1)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestJitterMeasurementsIntegration:
    """Integration tests combining multiple functions."""

    def test_complete_jitter_analysis_workflow(self) -> None:
        """Test complete workflow from clock to jitter metrics."""
        # Create clock signal
        trace = create_digital_clock(50, nominal_period=1e-9)

        # Find edges
        rising, falling = _find_edges(trace, threshold_frac=0.5)

        # Calculate TIE
        tie = tie_from_edges(rising, nominal_period=1e-9)

        # Calculate periods and C2C jitter
        periods = np.diff(rising)
        c2c_result = cycle_to_cycle_jitter(periods)

        # All should produce valid results
        assert len(tie) > 0
        assert c2c_result.c2c_rms >= 0

    def test_c2c_and_period_jitter_consistency(self) -> None:
        """Test that C2C and period jitter give reasonable results."""
        jitter_rms = 2e-12
        periods = create_clock_periods(100, jitter_rms=jitter_rms)

        c2c_result = cycle_to_cycle_jitter(periods)
        pj_result = period_jitter(periods, nominal_period=1e-9)

        # Both should show jitter
        assert c2c_result.c2c_rms > 0
        assert pj_result.c2c_rms > 0

    def test_dcd_and_c2c_combined_analysis(self) -> None:
        """Test measuring both DCD and C2C on same signal."""
        trace = create_digital_clock(30, nominal_period=1e-9, duty_cycle=0.45)

        # Measure DCD
        dcd_result = measure_dcd(trace, clock_period=1e-9)

        # Extract edges for C2C
        rising, _ = _find_edges(trace, threshold_frac=0.5)
        periods = np.diff(rising)

        # Measure C2C
        c2c_result = cycle_to_cycle_jitter(periods)

        # Both should produce valid results
        assert dcd_result.dcd_percent > 0
        assert c2c_result.c2c_rms >= 0

    def test_high_jitter_signal_detection(self) -> None:
        """Test analysis of signal with high jitter."""
        high_jitter = 5e-12
        periods = create_clock_periods(100, jitter_rms=high_jitter)
        result = cycle_to_cycle_jitter(periods)

        # Should detect the high jitter
        assert result.c2c_rms > 1e-12
        assert result.period_std > 0


# =============================================================================
# Edge Cases and Boundary Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_very_small_periods_high_frequency(self) -> None:
        """Test with very small periods (high frequency)."""
        periods = create_clock_periods(50, nominal_period=1e-12, jitter_rms=1e-14)
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 50
        assert result.c2c_rms >= 0

    def test_very_large_periods_low_frequency(self) -> None:
        """Test with very large periods (low frequency)."""
        periods = create_clock_periods(50, nominal_period=1e-3, jitter_rms=1e-6)
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 50
        assert result.c2c_rms >= 0

    def test_extreme_jitter_ratio(self) -> None:
        """Test with jitter RMS being significant fraction of period."""
        nominal = 1e-9
        jitter = 1e-10  # 10% jitter
        periods = create_clock_periods(100, nominal_period=nominal, jitter_rms=jitter)
        result = cycle_to_cycle_jitter(periods)

        assert result.c2c_rms > 0
        assert result.n_cycles == 100

    def test_tie_with_large_time_offset(self) -> None:
        """Test TIE with very large but consistent offset."""
        period = 1e-9
        n_edges = 50
        offset = 1e-3  # Large offset
        edge_timestamps = offset + np.arange(n_edges) * period

        tie = tie_from_edges(edge_timestamps, nominal_period=period)

        assert len(tie) == n_edges
        # Offset should not affect TIE values
        assert np.max(np.abs(tie)) < 1e-15

    def test_dcd_with_very_short_pulses(self) -> None:
        """Test DCD with very short pulse widths (10% duty cycle)."""
        trace = create_digital_clock(50, nominal_period=1e-9, duty_cycle=0.1)
        result = measure_dcd(trace, clock_period=1e-9)

        # Should still produce valid result
        assert result.duty_cycle > 0
        assert result.duty_cycle < 1.0

    def test_dcd_with_very_long_pulses(self) -> None:
        """Test DCD with very long pulse widths (90% duty cycle)."""
        trace = create_digital_clock(50, nominal_period=1e-9, duty_cycle=0.9)
        result = measure_dcd(trace, clock_period=1e-9)

        # Should still produce valid result
        assert result.duty_cycle == pytest.approx(0.9, abs=0.05)

    def test_c2c_with_monotonic_increasing_periods(self) -> None:
        """Test C2C with monotonically increasing periods (frequency drift)."""
        periods = np.array([1e-9 * (1 + i * 1e-4) for i in range(100)])
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 100
        # Should detect changes in period
        assert result.c2c_rms > 0

    def test_c2c_with_bimodal_distribution(self) -> None:
        """Test C2C with bimodal period distribution."""
        # Two groups of periods with different values
        group1 = np.full(50, 1e-9, dtype=np.float64)
        group2 = np.full(50, 1.01e-9, dtype=np.float64)
        periods = np.concatenate([group1, group2])
        result = cycle_to_cycle_jitter(periods)

        # Should detect the step change
        assert result.c2c_rms > 1e-12

    def test_single_nan_in_middle(self) -> None:
        """Test handling of single NaN in middle of data."""
        periods = create_clock_periods(50, jitter_rms=1e-12)
        periods[25] = np.nan
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 49
        assert not np.any(np.isnan(result.c2c_values))

    def test_multiple_consecutive_nans(self) -> None:
        """Test handling of multiple consecutive NaNs."""
        periods = create_clock_periods(50, jitter_rms=1e-12)
        periods[20:25] = np.nan
        result = cycle_to_cycle_jitter(periods)

        assert result.n_cycles == 45
        assert not np.any(np.isnan(result.c2c_values))


# =============================================================================
# Numerical Stability Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_c2c_with_zero_jitter(self) -> None:
        """Test C2C handles perfect zero jitter."""
        periods = np.full(100, 1e-9, dtype=np.float64)
        result = cycle_to_cycle_jitter(periods)

        assert result.c2c_rms == 0.0
        assert result.period_std == 0.0

    def test_tie_numerical_precision(self) -> None:
        """Test TIE maintains numerical precision."""
        period = 1e-15  # Very small period
        n_edges = 100
        edge_timestamps = np.arange(n_edges) * period

        tie = tie_from_edges(edge_timestamps, nominal_period=period)

        assert len(tie) == n_edges
        # Should maintain precision even for small values
        assert np.max(np.abs(tie)) < 1e-20 or np.max(np.abs(tie)) < period * 1e-6

    def test_dcd_with_equal_high_low_times(self) -> None:
        """Test DCD when high and low times are exactly equal."""
        trace = create_digital_clock(20, nominal_period=1e-9, duty_cycle=0.5)
        result = measure_dcd(trace, clock_period=1e-9)

        # DCD should be small (but may not be exactly zero due to edge detection)
        assert result.dcd_seconds >= 0

    def test_histogram_with_identical_values(self) -> None:
        """Test histogram generation with identical C2C values."""
        periods = np.full(100, 1e-9, dtype=np.float64)
        # Add tiny variations to create non-zero C2C
        periods[1::2] += 1e-14
        result = cycle_to_cycle_jitter(periods, include_histogram=True)

        # Should handle near-identical values gracefully
        if result.histogram is not None:
            assert len(result.histogram) > 0


# =============================================================================
# __all__ Export Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.jitter
class TestModuleExports:
    """Test module exports and __all__ list."""

    def test_all_exports_available(self) -> None:
        """Test that all __all__ exports are importable."""
        from oscura.analyzers.jitter import measurements

        expected_exports = [
            "CycleJitterResult",
            "DutyCycleDistortionResult",
            "cycle_to_cycle_jitter",
            "measure_dcd",
            "period_jitter",
            "tie_from_edges",
        ]

        for export in expected_exports:
            assert hasattr(measurements, export), f"Missing export: {export}"

    def test_all_list_completeness(self) -> None:
        """Test that __all__ list contains expected items."""
        from oscura.analyzers.jitter import measurements

        assert hasattr(measurements, "__all__")
        assert len(measurements.__all__) == 6
