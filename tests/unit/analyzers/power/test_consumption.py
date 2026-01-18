"""Comprehensive unit tests for power consumption calculations.

This module tests all public functions in the power consumption module with
edge cases, error conditions, and various waveform scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.power]


def make_waveform_trace(data: NDArray[np.float64], sample_rate: float = 1e6) -> WaveformTrace:
    """Create a WaveformTrace from raw data for testing.

    Args:
        data: Raw waveform data.
        sample_rate: Sample rate in Hz.

    Returns:
        WaveformTrace with the given data and sample rate.
    """
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-001")
class TestInstantaneousPower:
    """Test instantaneous power calculation P(t) = V(t) * I(t)."""

    def test_instantaneous_power_basic(self) -> None:
        """Test basic instantaneous power calculation.

        Validates:
        - Correct multiplication of voltage and current
        - Result is WaveformTrace with correct metadata
        """
        from oscura.analyzers.power.basic import instantaneous_power

        v_data = np.array([10.0, 20.0, 15.0, 25.0, 30.0])
        i_data = np.array([2.0, 3.0, 2.5, 4.0, 5.0])
        v_trace = make_waveform_trace(v_data)
        i_trace = make_waveform_trace(i_data)

        power = instantaneous_power(v_trace, i_trace)

        expected = v_data * i_data  # [20, 60, 37.5, 100, 150]
        np.testing.assert_array_almost_equal(power.data, expected)
        assert power.metadata.sample_rate == 1e6

    def test_instantaneous_power_same_sample_rates(self) -> None:
        """Test with matching sample rates.

        Validates:
        - Output sample rate matches inputs
        """
        from oscura.analyzers.power.basic import instantaneous_power

        v_trace = make_waveform_trace(np.array([10.0] * 100), sample_rate=1e6)
        i_trace = make_waveform_trace(np.array([2.0] * 100), sample_rate=1e6)

        power = instantaneous_power(v_trace, i_trace)

        assert len(power.data) == 100
        assert power.metadata.sample_rate == 1e6
        np.testing.assert_array_almost_equal(power.data, np.array([20.0] * 100))

    def test_instantaneous_power_different_sample_rates_with_interpolation(self) -> None:
        """Test with different sample rates, interpolation enabled.

        Validates:
        - Voltage interpolated to match current (higher rate wins)
        - Output has correct sample rate
        """
        from oscura.analyzers.power.basic import instantaneous_power

        # Current at 1MHz, voltage at 500kHz
        v_trace = make_waveform_trace(np.array([10.0] * 50), sample_rate=5e5)
        i_trace = make_waveform_trace(np.array([2.0] * 100), sample_rate=1e6)

        power = instantaneous_power(v_trace, i_trace, interpolate_if_needed=True)

        # Should use higher sample rate
        assert power.metadata.sample_rate == 1e6
        assert len(power.data) == 100

    def test_instantaneous_power_different_sample_rates_no_interpolation(self) -> None:
        """Test with different sample rates, interpolation disabled.

        Validates:
        - Raises AnalysisError when rates don't match and interpolation disabled
        """
        from oscura.analyzers.power.basic import instantaneous_power

        v_trace = make_waveform_trace(np.array([10.0] * 50), sample_rate=5e5)
        i_trace = make_waveform_trace(np.array([2.0] * 100), sample_rate=1e6)

        with pytest.raises(AnalysisError, match="Sample rate mismatch"):
            instantaneous_power(v_trace, i_trace, interpolate_if_needed=False)

    def test_instantaneous_power_different_lengths(self) -> None:
        """Test with different trace lengths.

        Validates:
        - Uses minimum length
        - Truncates longer trace
        """
        from oscura.analyzers.power.basic import instantaneous_power

        v_trace = make_waveform_trace(np.array([10.0] * 100))
        i_trace = make_waveform_trace(np.array([2.0] * 50))

        power = instantaneous_power(v_trace, i_trace)

        assert len(power.data) == 50
        expected = np.array([20.0] * 50)
        np.testing.assert_array_almost_equal(power.data, expected)

    def test_instantaneous_power_zero_values(self) -> None:
        """Test with zero voltage/current.

        Validates:
        - Correctly produces zero power
        """
        from oscura.analyzers.power.basic import instantaneous_power

        v_trace = make_waveform_trace(np.array([0.0] * 100))
        i_trace = make_waveform_trace(np.array([5.0] * 100))

        power = instantaneous_power(v_trace, i_trace)

        np.testing.assert_array_almost_equal(power.data, np.array([0.0] * 100))

    def test_instantaneous_power_negative_values(self) -> None:
        """Test with negative voltage/current (regeneration).

        Validates:
        - Negative V and positive I produces negative power
        - Negative V and negative I produces positive power
        """
        from oscura.analyzers.power.basic import instantaneous_power

        v_trace = make_waveform_trace(np.array([-10.0, 10.0, -10.0, 10.0]))
        i_trace = make_waveform_trace(np.array([2.0, 2.0, -2.0, -2.0]))

        power = instantaneous_power(v_trace, i_trace)

        expected = np.array([-20.0, 20.0, 20.0, -20.0])
        np.testing.assert_array_almost_equal(power.data, expected)

    def test_instantaneous_power_ac_signals(self) -> None:
        """Test with AC waveforms (sine waves).

        Validates:
        - Correct power calculation for AC
        - Oscillating power with positive and negative values
        """
        from oscura.analyzers.power.basic import instantaneous_power

        t = np.linspace(0, 0.02, 1000)
        v_data = 10.0 * np.sin(2 * np.pi * 60 * t)  # 60 Hz, 10V peak
        i_data = 2.0 * np.sin(2 * np.pi * 60 * t)  # 60 Hz, 2A peak

        v_trace = make_waveform_trace(v_data)
        i_trace = make_waveform_trace(i_data)

        power = instantaneous_power(v_trace, i_trace)

        expected = v_data * i_data
        np.testing.assert_array_almost_equal(power.data, expected)

    def test_instantaneous_power_single_sample(self) -> None:
        """Test with single sample.

        Validates:
        - Handles minimum size
        """
        from oscura.analyzers.power.basic import instantaneous_power

        v_trace = make_waveform_trace(np.array([10.0]))
        i_trace = make_waveform_trace(np.array([2.0]))

        power = instantaneous_power(v_trace, i_trace)

        np.testing.assert_array_almost_equal(power.data, np.array([20.0]))

    def test_instantaneous_power_large_dataset(self) -> None:
        """Test with large dataset.

        Validates:
        - Handles large arrays efficiently
        """
        from oscura.analyzers.power.basic import instantaneous_power

        size = 1000000
        v_trace = make_waveform_trace(np.ones(size) * 10.0)
        i_trace = make_waveform_trace(np.ones(size) * 2.0)

        power = instantaneous_power(v_trace, i_trace)

        assert len(power.data) == size
        np.testing.assert_array_almost_equal(power.data, np.ones(size) * 20.0)


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-002")
class TestAveragePower:
    """Test average power calculation."""

    def test_average_power_from_power_trace(self) -> None:
        """Test average power calculation from power trace.

        Validates:
        - Correct mean calculation
        """
        from oscura.analyzers.power.basic import average_power

        power_data = np.array([10.0, 20.0, 30.0, 40.0])
        power_trace = make_waveform_trace(power_data)

        avg = average_power(power=power_trace)

        assert avg == 25.0

    def test_average_power_from_voltage_current(self) -> None:
        """Test average power from voltage and current traces.

        Validates:
        - Calculates instantaneous power first, then averages
        """
        from oscura.analyzers.power.basic import average_power

        v_trace = make_waveform_trace(np.array([10.0, 20.0, 30.0, 40.0]))
        i_trace = make_waveform_trace(np.array([2.0, 2.0, 2.0, 2.0]))

        avg = average_power(voltage=v_trace, current=i_trace)

        # Average of [20, 40, 60, 80] = 50
        assert avg == 50.0

    def test_average_power_missing_arguments(self) -> None:
        """Test error when required arguments are missing.

        Validates:
        - Raises AnalysisError if no valid inputs provided
        """
        from oscura.analyzers.power.basic import average_power

        with pytest.raises(AnalysisError, match="Either power trace or both voltage and current"):
            average_power()

    def test_average_power_only_voltage(self) -> None:
        """Test error with only voltage, no current.

        Validates:
        - Requires both V and I if power not provided
        """
        from oscura.analyzers.power.basic import average_power

        v_trace = make_waveform_trace(np.array([10.0] * 100))

        with pytest.raises(AnalysisError):
            average_power(voltage=v_trace)

    def test_average_power_zero_data(self) -> None:
        """Test with all-zero power.

        Validates:
        - Returns 0.0
        """
        from oscura.analyzers.power.basic import average_power

        power_trace = make_waveform_trace(np.zeros(100))

        avg = average_power(power=power_trace)

        assert avg == 0.0

    def test_average_power_negative_values(self) -> None:
        """Test with negative power values (regeneration).

        Validates:
        - Correctly handles negative average
        """
        from oscura.analyzers.power.basic import average_power

        power_data = np.array([-10.0, 10.0, -10.0, 10.0])
        power_trace = make_waveform_trace(power_data)

        avg = average_power(power=power_trace)

        assert avg == 0.0

    def test_average_power_single_sample(self) -> None:
        """Test with single sample.

        Validates:
        - Returns the value itself
        """
        from oscura.analyzers.power.basic import average_power

        power_trace = make_waveform_trace(np.array([42.5]))

        avg = average_power(power=power_trace)

        assert avg == 42.5


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-002")
class TestRMSPower:
    """Test RMS power calculation."""

    def test_rms_power_constant(self) -> None:
        """Test RMS of constant signal.

        Validates:
        - RMS equals value for constant signal
        """
        from oscura.analyzers.power.basic import rms_power

        power_trace = make_waveform_trace(np.array([10.0] * 100))

        rms = rms_power(power=power_trace)

        assert abs(rms - 10.0) < 1e-10

    def test_rms_power_sinusoidal(self) -> None:
        """Test RMS of sinusoidal signal.

        Validates:
        - RMS is peak/sqrt(2) for pure sine
        """
        from oscura.analyzers.power.basic import rms_power

        t = np.linspace(0, 1, 1000)
        # Sine with peak=10, mean should be 10/sqrt(2) ≈ 7.07
        power_data = 10.0 * np.sin(2 * np.pi * t)
        power_trace = make_waveform_trace(power_data)

        rms = rms_power(power=power_trace)

        # RMS of sine is peak/sqrt(2)
        expected = 10.0 / np.sqrt(2)
        assert abs(rms - expected) < 0.2  # Allow some numerical tolerance

    def test_rms_power_from_voltage_current(self) -> None:
        """Test RMS power from voltage and current.

        Validates:
        - Calculates power first, then RMS
        """
        from oscura.analyzers.power.basic import rms_power

        v_trace = make_waveform_trace(np.array([10.0, 20.0, 30.0, 40.0]))
        i_trace = make_waveform_trace(np.array([2.0, 2.0, 2.0, 2.0]))

        rms = rms_power(voltage=v_trace, current=i_trace)

        # Power = [20, 40, 60, 80]
        # RMS = sqrt(mean([400, 1600, 3600, 6400])) = sqrt(3000) ≈ 54.77
        expected = np.sqrt(np.mean(np.array([400.0, 1600.0, 3600.0, 6400.0])))
        assert abs(rms - expected) < 1e-10

    def test_rms_power_zero(self) -> None:
        """Test RMS of zero signal.

        Validates:
        - Returns 0.0
        """
        from oscura.analyzers.power.basic import rms_power

        power_trace = make_waveform_trace(np.zeros(100))

        rms = rms_power(power=power_trace)

        assert rms == 0.0

    def test_rms_power_negative_values(self) -> None:
        """Test RMS with negative values.

        Validates:
        - Correctly squares negative values
        """
        from oscura.analyzers.power.basic import rms_power

        power_data = np.array([-10.0, 10.0])
        power_trace = make_waveform_trace(power_data)

        rms = rms_power(power=power_trace)

        expected = np.sqrt(np.mean(np.array([100.0, 100.0])))
        assert abs(rms - expected) < 1e-10

    def test_rms_power_missing_arguments(self) -> None:
        """Test error when required arguments missing.

        Validates:
        - Raises AnalysisError
        """
        from oscura.analyzers.power.basic import rms_power

        with pytest.raises(AnalysisError):
            rms_power()


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-002")
class TestPeakPower:
    """Test peak power calculation."""

    def test_peak_power_absolute(self) -> None:
        """Test peak power with absolute=True.

        Validates:
        - Returns absolute maximum
        """
        from oscura.analyzers.power.basic import peak_power

        power_data = np.array([-50.0, 100.0, -30.0, 80.0])
        power_trace = make_waveform_trace(power_data)

        peak = peak_power(power=power_trace, absolute=True)

        assert peak == 100.0

    def test_peak_power_maximum(self) -> None:
        """Test peak power with absolute=False.

        Validates:
        - Returns maximum value (not absolute)
        """
        from oscura.analyzers.power.basic import peak_power

        power_data = np.array([-50.0, 100.0, -30.0, 80.0])
        power_trace = make_waveform_trace(power_data)

        peak = peak_power(power=power_trace, absolute=False)

        assert peak == 100.0

    def test_peak_power_negative_dominant(self) -> None:
        """Test peak with more negative values.

        Validates:
        - absolute=True catches largest magnitude
        - absolute=False returns max (which is less negative)
        """
        from oscura.analyzers.power.basic import peak_power

        power_data = np.array([-200.0, -100.0, 50.0])
        power_trace = make_waveform_trace(power_data)

        peak_abs = peak_power(power=power_trace, absolute=True)
        peak_max = peak_power(power=power_trace, absolute=False)

        assert peak_abs == 200.0
        assert peak_max == 50.0

    def test_peak_power_from_voltage_current(self) -> None:
        """Test peak power calculated from V and I.

        Validates:
        - Calculates power first
        """
        from oscura.analyzers.power.basic import peak_power

        v_trace = make_waveform_trace(np.array([10.0, -20.0, 30.0]))
        i_trace = make_waveform_trace(np.array([2.0, 2.0, 2.0]))

        peak = peak_power(voltage=v_trace, current=i_trace, absolute=True)

        # Power = [20, -40, 60]
        assert peak == 60.0

    def test_peak_power_zero(self) -> None:
        """Test peak of zero signal.

        Validates:
        - Returns 0.0
        """
        from oscura.analyzers.power.basic import peak_power

        power_trace = make_waveform_trace(np.zeros(100))

        peak = peak_power(power=power_trace)

        assert peak == 0.0

    def test_peak_power_single_sample(self) -> None:
        """Test with single sample.

        Validates:
        - Returns the value itself
        """
        from oscura.analyzers.power.basic import peak_power

        power_trace = make_waveform_trace(np.array([42.5]))

        peak = peak_power(power=power_trace)

        assert peak == 42.5

    def test_peak_power_missing_arguments(self) -> None:
        """Test error when arguments missing.

        Validates:
        - Raises AnalysisError
        """
        from oscura.analyzers.power.basic import peak_power

        with pytest.raises(AnalysisError):
            peak_power()


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-002")
class TestEnergy:
    """Test energy (integral of power) calculation."""

    def test_energy_constant_power(self) -> None:
        """Test energy from constant power.

        Validates:
        - Energy = Power * Time
        """
        from oscura.analyzers.power.basic import energy

        # 100W for 1 second = 100J
        sample_rate = 1000.0  # 1kHz -> 1ms per sample
        duration = 1.0  # 1 second
        num_samples = int(sample_rate * duration)
        power_data = np.ones(num_samples) * 100.0
        power_trace = make_waveform_trace(power_data, sample_rate=sample_rate)

        e = energy(power=power_trace)

        # Should be approximately 100J
        assert abs(e - 100.0) < 1.0

    def test_energy_from_voltage_current(self) -> None:
        """Test energy calculated from V and I.

        Validates:
        - Calculates power first, then integrates
        """
        from oscura.analyzers.power.basic import energy

        sample_rate = 1000.0
        v_trace = make_waveform_trace(np.ones(1000) * 10.0, sample_rate=sample_rate)
        i_trace = make_waveform_trace(np.ones(1000) * 2.0, sample_rate=sample_rate)

        e = energy(voltage=v_trace, current=i_trace)

        # Power = 20W for 1 second
        assert abs(e - 20.0) < 1.0

    def test_energy_with_time_limits(self) -> None:
        """Test energy with start_time and end_time.

        Validates:
        - Integration only over specified time window
        """
        from oscura.analyzers.power.basic import energy

        sample_rate = 100.0  # 100Hz -> 0.01s per sample
        duration = 1.0  # 1 second total = 100 samples
        power_data = np.ones(100) * 100.0  # 100W constant

        power_trace = make_waveform_trace(power_data, sample_rate=sample_rate)

        # Full integration
        e_full = energy(power=power_trace)

        # Half integration (0.5 seconds)
        e_half = energy(power=power_trace, start_time=0.0, end_time=0.5)

        assert abs(e_half - 50.0) < 1.0
        assert e_full > e_half

    def test_energy_partial_window_middle(self) -> None:
        """Test energy with window in middle.

        Validates:
        - start_time and end_time both used
        """
        from oscura.analyzers.power.basic import energy

        sample_rate = 100.0
        power_data = np.ones(100) * 100.0

        power_trace = make_waveform_trace(power_data, sample_rate=sample_rate)

        # 0.25 to 0.75 seconds = 0.5 seconds = 50J
        e = energy(power=power_trace, start_time=0.25, end_time=0.75)

        assert abs(e - 50.0) < 1.0

    def test_energy_start_time_only(self) -> None:
        """Test energy with only start_time.

        Validates:
        - Integrates from start_time to end
        """
        from oscura.analyzers.power.basic import energy

        sample_rate = 100.0
        power_data = np.ones(100) * 100.0

        power_trace = make_waveform_trace(power_data, sample_rate=sample_rate)

        # From 0.5s to end (0.5s) = 50J
        e = energy(power=power_trace, start_time=0.5)

        assert abs(e - 50.0) < 1.5

    def test_energy_end_time_only(self) -> None:
        """Test energy with only end_time.

        Validates:
        - Integrates from start to end_time
        """
        from oscura.analyzers.power.basic import energy

        sample_rate = 100.0
        power_data = np.ones(100) * 100.0

        power_trace = make_waveform_trace(power_data, sample_rate=sample_rate)

        # From start to 0.5s = 50J
        e = energy(power=power_trace, end_time=0.5)

        assert abs(e - 50.0) < 1.0

    def test_energy_zero_power(self) -> None:
        """Test energy with zero power.

        Validates:
        - Returns 0.0
        """
        from oscura.analyzers.power.basic import energy

        power_trace = make_waveform_trace(np.zeros(100))

        e = energy(power=power_trace)

        assert e == 0.0

    def test_energy_mixed_sign(self) -> None:
        """Test energy with positive and negative power.

        Validates:
        - Correctly integrates across sign changes
        """
        from oscura.analyzers.power.basic import energy

        # Trapezoid: 0.5 seconds of +100W, 0.5 seconds of -100W
        power_data = np.concatenate([np.ones(50) * 100.0, np.ones(50) * -100.0])
        power_trace = make_waveform_trace(power_data, sample_rate=100.0)

        e = energy(power=power_trace)

        # Should be close to 0 (slight numerical error expected)
        assert abs(e) < 5.0

    def test_energy_missing_arguments(self) -> None:
        """Test error when arguments missing.

        Validates:
        - Raises AnalysisError
        """
        from oscura.analyzers.power.basic import energy

        with pytest.raises(AnalysisError):
            energy()


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-002")
class TestPowerStatistics:
    """Test comprehensive power statistics calculation."""

    def test_power_statistics_basic(self) -> None:
        """Test basic power statistics calculation.

        Validates:
        - All required fields present
        - Correct values for simple case
        """
        from oscura.analyzers.power.basic import power_statistics

        power_data = np.array([10.0, 20.0, 30.0, 40.0])
        power_trace = make_waveform_trace(power_data)

        stats = power_statistics(power=power_trace)

        assert "average" in stats
        assert "rms" in stats
        assert "peak" in stats
        assert "peak_positive" in stats
        assert "peak_negative" in stats
        assert "energy" in stats
        assert "min" in stats
        assert "std" in stats

        assert stats["average"] == 25.0
        assert stats["peak_positive"] == 40.0
        assert stats["peak_negative"] == 10.0
        assert stats["min"] == 10.0

    def test_power_statistics_from_voltage_current(self) -> None:
        """Test statistics from voltage and current.

        Validates:
        - Calculates power first
        """
        from oscura.analyzers.power.basic import power_statistics

        v_trace = make_waveform_trace(np.array([10.0] * 100))
        i_trace = make_waveform_trace(np.array([2.0] * 100))

        stats = power_statistics(voltage=v_trace, current=i_trace)

        assert stats["average"] == 20.0
        assert stats["peak"] == 20.0
        assert stats["peak_positive"] == 20.0
        assert stats["peak_negative"] == 20.0

    def test_power_statistics_with_negative_power(self) -> None:
        """Test statistics with negative power (regeneration).

        Validates:
        - peak is absolute maximum
        - peak_negative is minimum
        - energy calculation correct
        """
        from oscura.analyzers.power.basic import power_statistics

        power_data = np.array([-50.0, 100.0, -30.0, 80.0])
        power_trace = make_waveform_trace(power_data, sample_rate=100.0)

        stats = power_statistics(power=power_trace)

        assert stats["peak"] == 100.0
        assert stats["peak_positive"] == 100.0
        assert stats["peak_negative"] == -50.0
        assert stats["min"] == -50.0

    def test_power_statistics_std_deviation(self) -> None:
        """Test standard deviation calculation.

        Validates:
        - Correct variance/std calculation
        """
        from oscura.analyzers.power.basic import power_statistics

        # Mean=25, values=[10,20,30,40]
        # Variance = [(10-25)² + (20-25)² + (30-25)² + (40-25)²]/4
        #          = [225 + 25 + 25 + 225]/4 = 125
        # Std = sqrt(125) ≈ 11.18
        power_data = np.array([10.0, 20.0, 30.0, 40.0])
        power_trace = make_waveform_trace(power_data)

        stats = power_statistics(power=power_trace)

        expected_std = np.std(power_data)
        assert abs(stats["std"] - expected_std) < 1e-10

    def test_power_statistics_sinusoidal(self) -> None:
        """Test statistics with sinusoidal power (AC).

        Validates:
        - Correct RMS for sine wave
        - Average and energy reflect AC nature
        """
        from oscura.analyzers.power.basic import power_statistics

        t = np.linspace(0, 1, 1000)
        power_data = 100.0 * np.sin(2 * np.pi * t)
        power_trace = make_waveform_trace(power_data)

        stats = power_statistics(power=power_trace)

        # RMS of sine is peak/sqrt(2)
        expected_rms = 100.0 / np.sqrt(2)
        assert abs(stats["rms"] - expected_rms) < 2.0

        # Average should be near 0 for AC
        assert abs(stats["average"]) < 5.0

    def test_power_statistics_zero_signal(self) -> None:
        """Test statistics with zero signal.

        Validates:
        - All values are 0
        """
        from oscura.analyzers.power.basic import power_statistics

        power_trace = make_waveform_trace(np.zeros(100))

        stats = power_statistics(power=power_trace)

        assert stats["average"] == 0.0
        assert stats["rms"] == 0.0
        assert stats["peak"] == 0.0
        assert stats["std"] == 0.0
        assert stats["energy"] == 0.0

    def test_power_statistics_missing_arguments(self) -> None:
        """Test error when arguments missing.

        Validates:
        - Raises AnalysisError
        """
        from oscura.analyzers.power.basic import power_statistics

        with pytest.raises(AnalysisError):
            power_statistics()

    def test_power_statistics_single_sample(self) -> None:
        """Test with single sample.

        Validates:
        - Std dev is 0
        """
        from oscura.analyzers.power.basic import power_statistics

        power_trace = make_waveform_trace(np.array([50.0]))

        stats = power_statistics(power=power_trace)

        assert stats["average"] == 50.0
        assert stats["peak"] == 50.0
        assert stats["std"] == 0.0


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-004")
class TestPowerProfile:
    """Test power profile (rolling statistics) calculation."""

    def test_power_profile_basic(self) -> None:
        """Test basic power profile calculation.

        Validates:
        - Returns all required fields
        - Output traces have correct shape
        """
        from oscura.analyzers.power.basic import power_profile

        v_trace = make_waveform_trace(np.ones(1000) * 10.0)
        i_trace = make_waveform_trace(np.ones(1000) * 2.0)

        profile = power_profile(v_trace, i_trace)

        assert "power_trace" in profile
        assert "rolling_avg" in profile
        assert "rolling_peak" in profile
        assert "cumulative_energy" in profile
        assert "statistics" in profile

        assert len(profile["power_trace"].data) == 1000
        assert len(profile["rolling_avg"].data) == 1000
        assert len(profile["rolling_peak"].data) == 1000
        assert len(profile["cumulative_energy"].data) == 1000

    def test_power_profile_rolling_average(self) -> None:
        """Test that rolling average is smoother than original.

        Validates:
        - Rolling avg reduces high-frequency noise
        """
        from oscura.analyzers.power.basic import power_profile

        # Sawtooth pattern to show smoothing
        power_data = np.tile([100.0, 200.0, 100.0, 200.0], 250)
        v_trace = make_waveform_trace(power_data)
        i_trace = make_waveform_trace(np.ones(1000))

        profile = power_profile(v_trace, i_trace)

        # Rolling average variance should be less than original
        orig_var = np.var(profile["power_trace"].data)
        rolling_var = np.var(profile["rolling_avg"].data)
        assert rolling_var < orig_var

    def test_power_profile_cumulative_energy_monotonic(self) -> None:
        """Test that cumulative energy is monotonically increasing.

        Validates:
        - Positive power always increases cumulative energy
        """
        from oscura.analyzers.power.basic import power_profile

        v_trace = make_waveform_trace(np.ones(1000) * 10.0)
        i_trace = make_waveform_trace(np.ones(1000) * 2.0)

        profile = power_profile(v_trace, i_trace)

        cumulative = profile["cumulative_energy"].data
        # Check monotonically increasing
        diffs = np.diff(cumulative)
        assert np.all(diffs >= 0)

    def test_power_profile_custom_window_size(self) -> None:
        """Test power profile with custom window size.

        Validates:
        - window_size parameter affects smoothing
        """
        from oscura.analyzers.power.basic import power_profile

        v_trace = make_waveform_trace(np.random.randn(1000) * 10 + 100)
        i_trace = make_waveform_trace(np.random.randn(1000) * 2 + 2)

        # Smaller window
        profile_small = power_profile(v_trace, i_trace, window_size=5)

        # Larger window
        profile_large = power_profile(v_trace, i_trace, window_size=51)

        # Larger window should be smoother (lower variance)
        var_small = np.var(profile_small["rolling_avg"].data)
        var_large = np.var(profile_large["rolling_avg"].data)
        assert var_large < var_small

    def test_power_profile_auto_window_size(self) -> None:
        """Test automatic window size selection.

        Validates:
        - window_size=None uses auto-selection
        - Results in valid profile
        """
        from oscura.analyzers.power.basic import power_profile

        v_trace = make_waveform_trace(np.ones(1000) * 10.0)
        i_trace = make_waveform_trace(np.ones(1000) * 2.0)

        profile = power_profile(v_trace, i_trace, window_size=None)

        assert len(profile["power_trace"].data) == 1000
        assert len(profile["rolling_avg"].data) == 1000

    def test_power_profile_small_dataset(self) -> None:
        """Test power profile with small dataset.

        Validates:
        - Handles small arrays
        """
        from oscura.analyzers.power.basic import power_profile

        v_trace = make_waveform_trace(np.array([10.0, 20.0, 30.0]))
        i_trace = make_waveform_trace(np.array([1.0, 2.0, 3.0]))

        profile = power_profile(v_trace, i_trace)

        assert len(profile["power_trace"].data) == 3

    def test_power_profile_statistics_included(self) -> None:
        """Test that overall statistics are included.

        Validates:
        - statistics field contains all required keys
        """
        from oscura.analyzers.power.basic import power_profile

        v_trace = make_waveform_trace(np.ones(1000) * 10.0)
        i_trace = make_waveform_trace(np.ones(1000) * 2.0)

        profile = power_profile(v_trace, i_trace)

        stats = profile["statistics"]
        assert "average" in stats
        assert "rms" in stats
        assert "peak" in stats
        assert "energy" in stats

    def test_power_profile_metadata_preserved(self) -> None:
        """Test that metadata is preserved in all traces.

        Validates:
        - Sample rate is preserved
        """
        from oscura.analyzers.power.basic import power_profile

        sample_rate = 5e6
        v_trace = make_waveform_trace(np.ones(1000) * 10.0, sample_rate=sample_rate)
        i_trace = make_waveform_trace(np.ones(1000) * 2.0, sample_rate=sample_rate)

        profile = power_profile(v_trace, i_trace)

        assert profile["power_trace"].metadata.sample_rate == sample_rate
        assert profile["rolling_avg"].metadata.sample_rate == sample_rate
        assert profile["rolling_peak"].metadata.sample_rate == sample_rate
        assert profile["cumulative_energy"].metadata.sample_rate == sample_rate

    def test_power_profile_with_negative_power(self) -> None:
        """Test profile with regenerative (negative) power.

        Validates:
        - Cumulative energy can decrease
        """
        from oscura.analyzers.power.basic import power_profile

        v_trace = make_waveform_trace(np.array([10.0, -10.0, 10.0, -10.0] * 250))
        i_trace = make_waveform_trace(np.array([2.0, 2.0, 2.0, 2.0] * 250))

        profile = power_profile(v_trace, i_trace)

        # Cumulative energy should oscillate up and down
        cumulative = profile["cumulative_energy"].data
        assert cumulative[-1] != cumulative[0]  # Net change


@pytest.mark.unit
@pytest.mark.analyzer
class TestPowerConsumptionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_values(self) -> None:
        """Test with very small power values (microWatts).

        Validates:
        - Maintains precision at small scales
        """
        from oscura.analyzers.power.basic import power_statistics

        power_data = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        power_trace = make_waveform_trace(power_data)

        stats = power_statistics(power=power_trace)

        assert stats["average"] > 0
        assert stats["rms"] > 0

    def test_very_large_values(self) -> None:
        """Test with very large power values (kilowatts).

        Validates:
        - No overflow with large numbers
        """
        from oscura.analyzers.power.basic import power_statistics

        power_data = np.array([1e6, 2e6, 3e6, 4e6])
        power_trace = make_waveform_trace(power_data)

        stats = power_statistics(power=power_trace)

        assert stats["average"] == 2.5e6

    def test_inf_nan_handling(self) -> None:
        """Test behavior with inf and NaN (should pass through).

        Validates:
        - Functions don't crash with inf/NaN
        """
        from oscura.analyzers.power.basic import instantaneous_power

        # Division by zero scenarios could lead to inf
        v_trace = make_waveform_trace(np.array([1.0, np.inf, -np.inf]))
        i_trace = make_waveform_trace(np.array([1.0, 1.0, 1.0]))

        power = instantaneous_power(v_trace, i_trace)

        assert len(power.data) == 3

    def test_very_high_frequency_ac(self) -> None:
        """Test with very high frequency AC signal.

        Validates:
        - Works with high frequency signals
        """
        from oscura.analyzers.power.basic import instantaneous_power

        sample_rate = 1e9  # 1 GHz
        t = np.arange(1000) / sample_rate
        frequency = 100e6  # 100 MHz

        v_trace = make_waveform_trace(
            10.0 * np.sin(2 * np.pi * frequency * t), sample_rate=sample_rate
        )
        i_trace = make_waveform_trace(
            2.0 * np.sin(2 * np.pi * frequency * t), sample_rate=sample_rate
        )

        power = instantaneous_power(v_trace, i_trace)

        assert len(power.data) == 1000

    def test_dtype_conversion_to_float64(self) -> None:
        """Test that output is always float64.

        Validates:
        - Proper dtype conversion
        """
        from oscura.analyzers.power.basic import instantaneous_power

        # Integer input
        v_data = np.array([10, 20, 30], dtype=np.int32)
        i_data = np.array([2, 2, 2], dtype=np.int32)

        v_trace = make_waveform_trace(v_data.astype(np.float64))
        i_trace = make_waveform_trace(i_data.astype(np.float64))

        power = instantaneous_power(v_trace, i_trace)

        assert power.data.dtype == np.float64


@pytest.mark.unit
@pytest.mark.analyzer
class TestPowerConsumptionIntegration:
    """Integration tests combining multiple functions."""

    def test_energy_from_instantaneous_vs_integrated(self) -> None:
        """Test that energy matches integral of instantaneous power.

        Validates:
        - energy function produces same result as integral
        """
        from oscura.analyzers.power.basic import energy, instantaneous_power

        v_trace = make_waveform_trace(np.ones(1000) * 10.0, sample_rate=1000.0)
        i_trace = make_waveform_trace(np.ones(1000) * 2.0, sample_rate=1000.0)

        power = instantaneous_power(v_trace, i_trace)
        e1 = energy(power=power)
        e2 = energy(voltage=v_trace, current=i_trace)

        assert abs(e1 - e2) < 1e-10

    def test_power_profile_statistics_match_direct(self) -> None:
        """Test that profile statistics match direct calculation.

        Validates:
        - power_profile statistics agree with power_statistics
        """
        from oscura.analyzers.power.basic import power_profile, power_statistics

        v_trace = make_waveform_trace(np.random.randn(1000) * 10 + 100)
        i_trace = make_waveform_trace(np.random.randn(1000) * 2 + 2)

        stats_direct = power_statistics(voltage=v_trace, current=i_trace)
        profile = power_profile(v_trace, i_trace)

        assert abs(stats_direct["average"] - profile["statistics"]["average"]) < 0.1
        assert abs(stats_direct["peak"] - profile["statistics"]["peak"]) < 0.1

    def test_average_rms_relationship(self) -> None:
        """Test mathematical relationship between average and RMS.

        Validates:
        - For constant signal, RMS = average
        """
        from oscura.analyzers.power.basic import average_power, rms_power

        power_trace = make_waveform_trace(np.ones(100) * 50.0)

        avg = average_power(power=power_trace)
        rms = rms_power(power=power_trace)

        assert abs(avg - rms) < 1e-10

    def test_peak_includes_average(self) -> None:
        """Test that peak power >= average power.

        Validates:
        - Mathematical relationship
        """
        from oscura.analyzers.power.basic import average_power, peak_power

        power_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        power_trace = make_waveform_trace(power_data)

        avg = average_power(power=power_trace)
        peak = peak_power(power=power_trace, absolute=True)

        assert peak >= avg
