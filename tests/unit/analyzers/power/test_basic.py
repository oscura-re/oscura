"""Unit tests for basic power analysis.


Tests cover all public functions including edge cases, error conditions,
and various power calculation scenarios (DC, AC, transient).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.power.basic import (
    average_power,
    energy,
    instantaneous_power,
    peak_power,
    power_profile,
    power_statistics,
    rms_power,
)
from oscura.core.exceptions import AnalysisError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.power]


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 10000.0  # 10 kHz


def create_trace(
    data: np.ndarray,
    sample_rate: float,
) -> WaveformTrace:
    """Create a waveform trace from data array."""
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data.astype(np.float64), metadata=metadata)


def create_dc_trace(
    voltage: float,
    sample_rate: float,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a constant DC waveform trace."""
    num_samples = int(sample_rate * duration)
    data = np.full(num_samples, voltage)
    return create_trace(data, sample_rate)


def create_sinusoidal_trace(
    amplitude: float,
    frequency: float,
    sample_rate: float,
    duration: float = 0.1,
    phase: float = 0.0,
    dc_offset: float = 0.0,
) -> WaveformTrace:
    """Create a sinusoidal waveform trace."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset
    return create_trace(data, sample_rate)


def create_square_wave_trace(
    amplitude: float,
    frequency: float,
    sample_rate: float,
    duration: float = 0.1,
    duty_cycle: float = 0.5,
) -> WaveformTrace:
    """Create a square wave trace."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = amplitude * (np.sin(2 * np.pi * frequency * t) > (1 - 2 * duty_cycle))
    return create_trace(data, sample_rate)


def create_transient_trace(
    v_low: float,
    v_high: float,
    sample_rate: float,
    transition_time: float = 0.05,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a trace with a transient (step change)."""
    num_samples = int(sample_rate * duration)
    transition_sample = int(sample_rate * transition_time)
    data = np.full(num_samples, v_low)
    data[transition_sample:] = v_high
    return create_trace(data, sample_rate)


@pytest.mark.unit
@pytest.mark.power
class TestInstantaneousPower:
    """Test instantaneous power calculation (P(t) = V(t) * I(t))."""

    def test_dc_power_calculation(self, sample_rate: float) -> None:
        """Test instantaneous power with DC voltage and current."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        power = instantaneous_power(voltage, current)

        # DC power: P = V * I = 12 * 2 = 24 W
        expected_power = 24.0
        assert np.allclose(power.data, expected_power)
        assert power.metadata.sample_rate == sample_rate

    def test_sinusoidal_power(self, sample_rate: float) -> None:
        """Test instantaneous power with sinusoidal voltage and current."""
        frequency = 60.0
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        power = instantaneous_power(voltage, current)

        # Instantaneous power varies but average should be V_rms * I_rms
        # For in-phase signals: P_avg = (V_pk * I_pk) / 2
        expected_avg = (120.0 * 10.0) / 2
        assert abs(np.mean(power.data) - expected_avg) < 10.0

    def test_power_with_phase_shift(self, sample_rate: float) -> None:
        """Test power calculation with phase shift between V and I."""
        frequency = 60.0
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 4)

        power = instantaneous_power(voltage, current)

        # Average power with phase shift: P = V_rms * I_rms * cos(phi)
        # Expected: (120/sqrt(2)) * (10/sqrt(2)) * cos(pi/4) = 600 * 0.707 = 424.2
        expected_avg = (120.0 / np.sqrt(2)) * (10.0 / np.sqrt(2)) * np.cos(np.pi / 4)
        assert abs(np.mean(power.data) - expected_avg) < 30.0

    def test_power_same_sample_rate(self, sample_rate: float) -> None:
        """Test that same sample rates are preserved."""
        voltage = create_dc_trace(5.0, sample_rate)
        current = create_dc_trace(1.0, sample_rate)

        power = instantaneous_power(voltage, current)

        assert power.metadata.sample_rate == sample_rate

    def test_power_different_sample_rates_interpolate(self, sample_rate: float) -> None:
        """Test power calculation with different sample rates (with interpolation)."""
        voltage = create_dc_trace(10.0, sample_rate, duration=0.1)
        current = create_dc_trace(2.0, sample_rate / 2, duration=0.1)

        power = instantaneous_power(voltage, current, interpolate_if_needed=True)

        # Should interpolate to higher sample rate
        assert power.metadata.sample_rate == sample_rate
        assert np.allclose(power.data, 20.0, atol=0.1)

    def test_power_different_sample_rates_no_interpolate(self, sample_rate: float) -> None:
        """Test that mismatched sample rates raise error without interpolation."""
        voltage = create_dc_trace(10.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate / 2)

        with pytest.raises(AnalysisError, match="Sample rate mismatch"):
            instantaneous_power(voltage, current, interpolate_if_needed=False)

    def test_power_current_higher_sample_rate(self, sample_rate: float) -> None:
        """Test power calculation when current has higher sample rate than voltage."""
        voltage = create_dc_trace(10.0, sample_rate / 2, duration=0.1)
        current = create_dc_trace(2.0, sample_rate, duration=0.1)

        power = instantaneous_power(voltage, current, interpolate_if_needed=True)

        # Should interpolate to higher sample rate (current's rate)
        assert power.metadata.sample_rate == sample_rate
        assert np.allclose(power.data, 20.0, atol=0.1)

    def test_power_different_lengths(self, sample_rate: float) -> None:
        """Test power calculation with different trace lengths."""
        voltage = create_dc_trace(10.0, sample_rate, duration=0.15)
        current = create_dc_trace(2.0, sample_rate, duration=0.10)

        power = instantaneous_power(voltage, current)

        # Should truncate to shorter length
        assert len(power.data) == int(sample_rate * 0.10)
        assert np.allclose(power.data, 20.0)

    def test_power_zero_voltage(self, sample_rate: float) -> None:
        """Test power with zero voltage."""
        voltage = create_dc_trace(0.0, sample_rate)
        current = create_dc_trace(5.0, sample_rate)

        power = instantaneous_power(voltage, current)

        assert np.allclose(power.data, 0.0)

    def test_power_zero_current(self, sample_rate: float) -> None:
        """Test power with zero current."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(0.0, sample_rate)

        power = instantaneous_power(voltage, current)

        assert np.allclose(power.data, 0.0)

    def test_power_negative_values(self, sample_rate: float) -> None:
        """Test power with negative voltage/current (regenerative power)."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(-2.0, sample_rate)

        power = instantaneous_power(voltage, current)

        # Negative power indicates regeneration
        assert np.allclose(power.data, -24.0)

    def test_power_transient_load(self, sample_rate: float) -> None:
        """Test power calculation during transient load change."""
        voltage = create_dc_trace(12.0, sample_rate, duration=0.1)
        current = create_transient_trace(1.0, 3.0, sample_rate, duration=0.1)

        power = instantaneous_power(voltage, current)

        # First half: 12V * 1A = 12W
        # Second half: 12V * 3A = 36W
        first_half = power.data[: len(power.data) // 2]
        second_half = power.data[len(power.data) // 2 :]
        assert np.mean(first_half) < np.mean(second_half)

    def test_power_output_dtype(self, sample_rate: float) -> None:
        """Test that power output is float64."""
        voltage = create_dc_trace(10.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        power = instantaneous_power(voltage, current)

        assert power.data.dtype == np.float64


@pytest.mark.unit
@pytest.mark.power
class TestAveragePower:
    """Test average (mean) power calculation."""

    def test_average_power_from_power_trace(self, sample_rate: float) -> None:
        """Test average power from pre-calculated power trace."""
        power_data = np.array([10.0, 20.0, 30.0, 40.0])
        power_trace = create_trace(power_data, sample_rate)

        p_avg = average_power(power_trace)

        assert p_avg == 25.0

    def test_average_power_from_voltage_current(self, sample_rate: float) -> None:
        """Test average power from voltage and current traces."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.5, sample_rate)

        p_avg = average_power(voltage=voltage, current=current)

        assert abs(p_avg - 30.0) < 0.1

    def test_average_power_sinusoidal(self, sample_rate: float) -> None:
        """Test average power with sinusoidal waveforms."""
        frequency = 60.0
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        p_avg = average_power(voltage=voltage, current=current)

        # P_avg = (V_pk * I_pk) / 2 for in-phase sine waves
        expected = (120.0 * 10.0) / 2
        assert abs(p_avg - expected) < 10.0

    def test_average_power_no_arguments_error(self) -> None:
        """Test that missing arguments raise error."""
        with pytest.raises(AnalysisError, match="Either power trace or both"):
            average_power()

    def test_average_power_missing_current_error(self, sample_rate: float) -> None:
        """Test that missing current raises error."""
        voltage = create_dc_trace(12.0, sample_rate)

        with pytest.raises(AnalysisError, match="Either power trace or both"):
            average_power(voltage=voltage)

    def test_average_power_zero_power(self, sample_rate: float) -> None:
        """Test average power when power is zero."""
        voltage = create_dc_trace(0.0, sample_rate)
        current = create_dc_trace(5.0, sample_rate)

        p_avg = average_power(voltage=voltage, current=current)

        assert abs(p_avg) < 1e-10

    def test_average_power_negative_power(self, sample_rate: float) -> None:
        """Test average power with regenerative (negative) power."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(-2.0, sample_rate)

        p_avg = average_power(voltage=voltage, current=current)

        assert abs(p_avg - (-24.0)) < 0.1


@pytest.mark.unit
@pytest.mark.power
class TestRMSPower:
    """Test RMS power calculation."""

    def test_rms_power_constant(self, sample_rate: float) -> None:
        """Test RMS power with constant power."""
        power_data = np.full(1000, 25.0)
        power_trace = create_trace(power_data, sample_rate)

        p_rms = rms_power(power_trace)

        # RMS of constant is the constant itself
        assert abs(p_rms - 25.0) < 0.1

    def test_rms_power_from_voltage_current(self, sample_rate: float) -> None:
        """Test RMS power from voltage and current."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        p_rms = rms_power(voltage=voltage, current=current)

        assert abs(p_rms - 24.0) < 0.1

    def test_rms_power_varying(self, sample_rate: float) -> None:
        """Test RMS power with varying power levels."""
        # Power varies: 0, 10, 20, 30, 40
        power_data = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        power_trace = create_trace(power_data, sample_rate)

        p_rms = rms_power(power_trace)

        # RMS = sqrt(mean(P^2)) = sqrt((0 + 100 + 400 + 900 + 1600)/5)
        # = sqrt(600) = 24.49
        expected_rms = np.sqrt(np.mean(power_data**2))
        assert abs(p_rms - expected_rms) < 0.1

    def test_rms_power_sinusoidal(self, sample_rate: float) -> None:
        """Test RMS power with sinusoidal voltage and current."""
        frequency = 60.0
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        p_rms = rms_power(voltage=voltage, current=current)

        # RMS should be higher than average for AC power
        assert p_rms > 0

    def test_rms_power_no_arguments_error(self) -> None:
        """Test that missing arguments raise error."""
        with pytest.raises(AnalysisError, match="Either power trace or both"):
            rms_power()


@pytest.mark.unit
@pytest.mark.power
class TestPeakPower:
    """Test peak power calculation."""

    def test_peak_power_constant(self, sample_rate: float) -> None:
        """Test peak power with constant power."""
        power_data = np.full(1000, 25.0)
        power_trace = create_trace(power_data, sample_rate)

        p_peak = peak_power(power_trace)

        assert abs(p_peak - 25.0) < 0.1

    def test_peak_power_from_voltage_current(self, sample_rate: float) -> None:
        """Test peak power from voltage and current."""
        voltage = create_transient_trace(5.0, 12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        p_peak = peak_power(voltage=voltage, current=current, absolute=True)

        # Peak is 12V * 2A = 24W
        assert abs(p_peak - 24.0) < 0.1

    def test_peak_power_absolute_mode(self, sample_rate: float) -> None:
        """Test peak power in absolute mode."""
        power_data = np.array([-30.0, -10.0, 0.0, 10.0, 20.0])
        power_trace = create_trace(power_data, sample_rate)

        p_peak = peak_power(power_trace, absolute=True)

        # Absolute peak is 30.0 (from -30.0)
        assert abs(p_peak - 30.0) < 0.1

    def test_peak_power_non_absolute_mode(self, sample_rate: float) -> None:
        """Test peak power in non-absolute mode."""
        power_data = np.array([-30.0, -10.0, 0.0, 10.0, 20.0])
        power_trace = create_trace(power_data, sample_rate)

        p_peak = peak_power(power_trace, absolute=False)

        # Maximum value is 20.0
        assert abs(p_peak - 20.0) < 0.1

    def test_peak_power_sinusoidal(self, sample_rate: float) -> None:
        """Test peak power with sinusoidal power waveform."""
        frequency = 60.0
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        p_peak = peak_power(voltage=voltage, current=current, absolute=True)

        # Peak instantaneous power for in-phase sine waves: V_pk * I_pk
        expected_peak = 120.0 * 10.0
        assert abs(p_peak - expected_peak) < 50.0

    def test_peak_power_no_arguments_error(self) -> None:
        """Test that missing arguments raise error."""
        with pytest.raises(AnalysisError, match="Either power trace or both"):
            peak_power()

    def test_peak_power_negative_regeneration(self, sample_rate: float) -> None:
        """Test peak power with regenerative (negative) power."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(-3.0, sample_rate)

        p_peak_abs = peak_power(voltage=voltage, current=current, absolute=True)
        p_peak_noabs = peak_power(voltage=voltage, current=current, absolute=False)

        # Absolute: 36W, Non-absolute: -36W
        assert abs(p_peak_abs - 36.0) < 0.1
        assert abs(p_peak_noabs - (-36.0)) < 0.1


@pytest.mark.unit
@pytest.mark.power
class TestEnergy:
    """Test energy (integral of power) calculation."""

    def test_energy_constant_power(self, sample_rate: float) -> None:
        """Test energy calculation with constant power."""
        # 100W for 0.1 seconds = 10 Joules
        power_data = np.full(int(sample_rate * 0.1), 100.0)
        power_trace = create_trace(power_data, sample_rate)

        e = energy(power_trace)

        expected_energy = 100.0 * 0.1  # P * t
        assert abs(e - expected_energy) < 0.5

    def test_energy_from_voltage_current(self, sample_rate: float) -> None:
        """Test energy from voltage and current traces."""
        duration = 0.1
        voltage = create_dc_trace(12.0, sample_rate, duration=duration)
        current = create_dc_trace(2.0, sample_rate, duration=duration)

        e = energy(voltage=voltage, current=current)

        # E = P * t = 24W * 0.1s = 2.4 J
        expected_energy = 24.0 * duration
        assert abs(e - expected_energy) < 0.1

    def test_energy_varying_power(self, sample_rate: float) -> None:
        """Test energy with varying power levels."""
        # Linear ramp from 0 to 100W over 0.1 seconds
        duration = 0.1
        num_samples = int(sample_rate * duration)
        power_data = np.linspace(0, 100, num_samples)
        power_trace = create_trace(power_data, sample_rate)

        e = energy(power_trace)

        # Energy = integral of linear ramp = (average power) * time
        # Average = 50W, time = 0.1s, E = 5J
        expected_energy = 50.0 * duration
        assert abs(e - expected_energy) < 0.5

    def test_energy_with_time_limits(self, sample_rate: float) -> None:
        """Test energy calculation with start and end time limits."""
        duration = 0.1
        power_data = np.full(int(sample_rate * duration), 100.0)
        power_trace = create_trace(power_data, sample_rate)

        # Calculate energy only between 0.02s and 0.08s (0.06s duration)
        e = energy(power_trace, start_time=0.02, end_time=0.08)

        expected_energy = 100.0 * 0.06
        assert abs(e - expected_energy) < 0.5

    def test_energy_start_time_only(self, sample_rate: float) -> None:
        """Test energy with only start time specified."""
        duration = 0.1
        power_data = np.full(int(sample_rate * duration), 100.0)
        power_trace = create_trace(power_data, sample_rate)

        # From 0.05s to end (0.05s duration)
        e = energy(power_trace, start_time=0.05)

        expected_energy = 100.0 * 0.05
        assert abs(e - expected_energy) < 0.5

    def test_energy_end_time_only(self, sample_rate: float) -> None:
        """Test energy with only end time specified."""
        duration = 0.1
        power_data = np.full(int(sample_rate * duration), 100.0)
        power_trace = create_trace(power_data, sample_rate)

        # From start to 0.05s (0.05s duration)
        e = energy(power_trace, end_time=0.05)

        expected_energy = 100.0 * 0.05
        assert abs(e - expected_energy) < 0.5

    def test_energy_sinusoidal(self, sample_rate: float) -> None:
        """Test energy calculation with sinusoidal power."""
        frequency = 60.0
        duration = 0.1
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, duration=duration)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, duration=duration)

        e = energy(voltage=voltage, current=current)

        # Energy = average power * time
        # Average power = (V_pk * I_pk) / 2 = 600W
        expected_energy = 600.0 * duration
        assert abs(e - expected_energy) < 5.0

    def test_energy_no_arguments_error(self) -> None:
        """Test that missing arguments raise error."""
        with pytest.raises(AnalysisError, match="Either power trace or both"):
            energy()

    def test_energy_negative_power(self, sample_rate: float) -> None:
        """Test energy with negative (regenerative) power."""
        duration = 0.1
        voltage = create_dc_trace(12.0, sample_rate, duration=duration)
        current = create_dc_trace(-2.0, sample_rate, duration=duration)

        e = energy(voltage=voltage, current=current)

        # Negative energy indicates regeneration
        expected_energy = -24.0 * duration
        assert abs(e - expected_energy) < 0.1


@pytest.mark.unit
@pytest.mark.power
class TestPowerStatistics:
    """Test comprehensive power statistics calculation."""

    def test_power_statistics_from_power_trace(self, sample_rate: float) -> None:
        """Test power statistics from pre-calculated power trace."""
        power_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        power_trace = create_trace(power_data, sample_rate)

        stats = power_statistics(power_trace)

        assert "average" in stats
        assert "rms" in stats
        assert "peak" in stats
        assert "peak_positive" in stats
        assert "peak_negative" in stats
        assert "energy" in stats
        assert "min" in stats
        assert "std" in stats

    def test_power_statistics_values_dc(self, sample_rate: float) -> None:
        """Test power statistics values with DC power."""
        duration = 0.1
        voltage = create_dc_trace(12.0, sample_rate, duration=duration)
        current = create_dc_trace(2.0, sample_rate, duration=duration)

        stats = power_statistics(voltage=voltage, current=current)

        # All stats should be 24W for constant power
        assert abs(stats["average"] - 24.0) < 0.1
        assert abs(stats["rms"] - 24.0) < 0.1
        assert abs(stats["peak"] - 24.0) < 0.1
        assert abs(stats["peak_positive"] - 24.0) < 0.1
        assert abs(stats["peak_negative"] - 24.0) < 0.1
        assert abs(stats["min"] - 24.0) < 0.1
        assert abs(stats["std"]) < 0.1  # No variation

    def test_power_statistics_varying_power(self, sample_rate: float) -> None:
        """Test power statistics with varying power."""
        power_data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        power_trace = create_trace(power_data, sample_rate)

        stats = power_statistics(power_trace)

        assert abs(stats["average"] - 30.0) < 0.1
        assert abs(stats["min"] - 10.0) < 0.1
        assert abs(stats["peak_positive"] - 50.0) < 0.1
        assert abs(stats["peak_negative"] - 10.0) < 0.1
        assert abs(stats["peak"] - 50.0) < 0.1
        assert stats["std"] > 0  # Has variation

    def test_power_statistics_with_negative_power(self, sample_rate: float) -> None:
        """Test power statistics with regenerative (negative) power."""
        power_data = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
        power_trace = create_trace(power_data, sample_rate)

        stats = power_statistics(power_trace)

        assert abs(stats["average"] - 0.0) < 0.1
        assert abs(stats["peak_positive"] - 20.0) < 0.1
        assert abs(stats["peak_negative"] - (-20.0)) < 0.1
        assert abs(stats["peak"] - 20.0) < 0.1  # Absolute peak
        assert abs(stats["min"] - (-20.0)) < 0.1

    def test_power_statistics_energy_integration(self, sample_rate: float) -> None:
        """Test that energy in statistics matches integration."""
        duration = 0.1
        power_data = np.full(int(sample_rate * duration), 100.0)
        power_trace = create_trace(power_data, sample_rate)

        stats = power_statistics(power_trace)

        # Energy should be 100W * 0.1s = 10J
        expected_energy = 100.0 * duration
        assert abs(stats["energy"] - expected_energy) < 0.5

    def test_power_statistics_no_arguments_error(self) -> None:
        """Test that missing arguments raise error."""
        with pytest.raises(AnalysisError, match="Either power trace or both"):
            power_statistics()

    def test_power_statistics_sinusoidal(self, sample_rate: float) -> None:
        """Test power statistics with sinusoidal waveforms."""
        frequency = 60.0
        duration = 0.1
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, duration=duration)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, duration=duration)

        stats = power_statistics(voltage=voltage, current=current)

        # Average power should be (V_pk * I_pk) / 2
        expected_avg = 600.0
        assert abs(stats["average"] - expected_avg) < 50.0

        # Peak should be V_pk * I_pk
        expected_peak = 1200.0
        assert abs(stats["peak"] - expected_peak) < 100.0


@pytest.mark.unit
@pytest.mark.power
class TestPowerProfile:
    """Test power profile with rolling statistics."""

    def test_power_profile_basic(self, sample_rate: float) -> None:
        """Test basic power profile generation."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        profile = power_profile(voltage, current)

        assert "power_trace" in profile
        assert "rolling_avg" in profile
        assert "rolling_peak" in profile
        assert "cumulative_energy" in profile
        assert "statistics" in profile

    def test_power_profile_traces_type(self, sample_rate: float) -> None:
        """Test that profile contains WaveformTrace objects."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        profile = power_profile(voltage, current)

        assert isinstance(profile["power_trace"], WaveformTrace)
        assert isinstance(profile["rolling_avg"], WaveformTrace)
        assert isinstance(profile["rolling_peak"], WaveformTrace)
        assert isinstance(profile["cumulative_energy"], WaveformTrace)
        assert isinstance(profile["statistics"], dict)

    def test_power_profile_constant_power(self, sample_rate: float) -> None:
        """Test power profile with constant power."""
        duration = 0.1
        voltage = create_dc_trace(12.0, sample_rate, duration=duration)
        current = create_dc_trace(2.0, sample_rate, duration=duration)

        profile = power_profile(voltage, current)

        # Constant power should have flat rolling average in center (edges have convolution artifacts)
        # Check the middle 50% of the data to avoid edge effects
        mid_start = len(profile["rolling_avg"].data) // 4
        mid_end = 3 * len(profile["rolling_avg"].data) // 4
        middle_section = profile["rolling_avg"].data[mid_start:mid_end]
        assert np.allclose(middle_section, 24.0, atol=0.5)

    def test_power_profile_transient(self, sample_rate: float) -> None:
        """Test power profile with transient load."""
        voltage = create_dc_trace(12.0, sample_rate, duration=0.1)
        current = create_transient_trace(1.0, 3.0, sample_rate, duration=0.1)

        profile = power_profile(voltage, current)

        # Rolling average should track the transition
        first_quarter = profile["rolling_avg"].data[: len(profile["rolling_avg"].data) // 4]
        last_quarter = profile["rolling_avg"].data[-len(profile["rolling_avg"].data) // 4 :]
        assert np.mean(last_quarter) > np.mean(first_quarter)

    def test_power_profile_cumulative_energy_increases(self, sample_rate: float) -> None:
        """Test that cumulative energy is monotonically increasing."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        profile = power_profile(voltage, current)

        # Cumulative energy should increase monotonically
        cum_energy = profile["cumulative_energy"].data
        assert np.all(np.diff(cum_energy) >= 0)

    def test_power_profile_auto_window_size(self, sample_rate: float) -> None:
        """Test auto-selection of window size."""
        voltage = create_dc_trace(12.0, sample_rate, duration=0.1)
        current = create_dc_trace(2.0, sample_rate, duration=0.1)

        # Should auto-select window size
        profile = power_profile(voltage, current, window_size=None)

        assert profile["rolling_avg"].data.shape == profile["power_trace"].data.shape

    def test_power_profile_custom_window_size(self, sample_rate: float) -> None:
        """Test custom window size."""
        voltage = create_dc_trace(12.0, sample_rate, duration=0.1)
        current = create_dc_trace(2.0, sample_rate, duration=0.1)

        # Use small window size
        profile = power_profile(voltage, current, window_size=51)

        assert profile["rolling_avg"].data.shape == profile["power_trace"].data.shape

    def test_power_profile_even_window_becomes_odd(self, sample_rate: float) -> None:
        """Test that even window sizes are converted to odd."""
        voltage = create_dc_trace(12.0, sample_rate, duration=0.1)
        current = create_dc_trace(2.0, sample_rate, duration=0.1)

        # Pass even window size - should be converted to odd (101)
        profile = power_profile(voltage, current, window_size=100)

        # Should complete without error
        assert profile is not None

    def test_power_profile_rolling_peak(self, sample_rate: float) -> None:
        """Test rolling peak calculation."""
        # Create power with a spike
        duration = 0.1
        num_samples = int(sample_rate * duration)
        power_data = np.full(num_samples, 10.0)
        power_data[num_samples // 2] = 100.0  # Spike in the middle

        voltage = create_trace(power_data, sample_rate)
        current = create_dc_trace(1.0, sample_rate, duration=duration)

        profile = power_profile(voltage, current, window_size=51)

        # Rolling peak should show elevated values near the spike
        assert np.max(profile["rolling_peak"].data) > 50.0

    def test_power_profile_statistics_match(self, sample_rate: float) -> None:
        """Test that statistics in profile match power_statistics."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        profile = power_profile(voltage, current)
        standalone_stats = power_statistics(voltage=voltage, current=current)

        # Should match
        assert abs(profile["statistics"]["average"] - standalone_stats["average"]) < 0.1
        assert abs(profile["statistics"]["peak"] - standalone_stats["peak"]) < 0.1


@pytest.mark.unit
@pytest.mark.power
class TestPowerBasicEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_traces(self, sample_rate: float) -> None:
        """Test that empty trace raises ValueError."""
        empty_data = np.array([])

        # Empty trace creation raises ValueError in v0.9.0+
        with pytest.raises(ValueError, match="data array cannot be empty"):
            create_trace(empty_data, sample_rate)

    def test_single_sample_traces(self, sample_rate: float) -> None:
        """Test functions with single-sample traces."""
        data = np.array([10.0])
        trace = create_trace(data, sample_rate)

        power = instantaneous_power(trace, trace)
        assert len(power.data) == 1
        assert power.data[0] == 100.0

    def test_very_short_traces(self, sample_rate: float) -> None:
        """Test functions with very short traces."""
        voltage = create_dc_trace(12.0, sample_rate, duration=0.001)
        current = create_dc_trace(2.0, sample_rate, duration=0.001)

        stats = power_statistics(voltage=voltage, current=current)
        assert stats["average"] > 0

    def test_very_long_traces(self, sample_rate: float) -> None:
        """Test functions with long traces."""
        # 1 second duration = 10,000 samples at 10 kHz
        voltage = create_dc_trace(12.0, sample_rate, duration=1.0)
        current = create_dc_trace(2.0, sample_rate, duration=1.0)

        p_avg = average_power(voltage=voltage, current=current)
        assert abs(p_avg - 24.0) < 0.1

    def test_very_large_power_values(self, sample_rate: float) -> None:
        """Test with very large power values."""
        voltage = create_dc_trace(10000.0, sample_rate)
        current = create_dc_trace(1000.0, sample_rate)

        p_avg = average_power(voltage=voltage, current=current)
        assert abs(p_avg - 10000000.0) < 100.0

    def test_very_small_power_values(self, sample_rate: float) -> None:
        """Test with very small power values."""
        voltage = create_dc_trace(0.001, sample_rate)
        current = create_dc_trace(0.001, sample_rate)

        p_avg = average_power(voltage=voltage, current=current)
        assert abs(p_avg - 0.000001) < 1e-9

    def test_nan_handling(self, sample_rate: float) -> None:
        """Test handling of NaN values in data."""
        data_with_nan = np.array([10.0, 20.0, np.nan, 30.0, 40.0])
        trace = create_trace(data_with_nan, sample_rate)

        # Functions should handle NaN (result will contain NaN)
        p_avg = average_power(trace)
        assert np.isnan(p_avg)

    def test_inf_handling(self, sample_rate: float) -> None:
        """Test handling of infinite values in data."""
        data_with_inf = np.array([10.0, 20.0, np.inf, 30.0, 40.0])
        trace = create_trace(data_with_inf, sample_rate)

        p_avg = average_power(trace)
        # Result will be inf
        assert np.isinf(p_avg)

    def test_mixed_positive_negative_power(self, sample_rate: float) -> None:
        """Test with both positive and negative power (bidirectional power flow)."""
        power_data = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
        power_trace = create_trace(power_data, sample_rate)

        stats = power_statistics(power_trace)

        assert abs(stats["average"] - 0.0) < 0.1
        assert abs(stats["peak_positive"] - 50.0) < 0.1
        assert abs(stats["peak_negative"] - (-50.0)) < 0.1

    def test_zero_sample_rate_error(self) -> None:
        """Test that zero sample rate raises error."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TraceMetadata(sample_rate=0.0)

    def test_negative_sample_rate_error(self) -> None:
        """Test that negative sample rate raises error."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            TraceMetadata(sample_rate=-1000.0)

    def test_power_profile_very_small_window(self, sample_rate: float) -> None:
        """Test power profile with very small window size."""
        voltage = create_dc_trace(12.0, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        # Minimum window of 3 samples
        profile = power_profile(voltage, current, window_size=3)

        assert profile is not None

    def test_energy_time_limits_out_of_range(self, sample_rate: float) -> None:
        """Test energy with time limits outside trace duration."""
        duration = 0.1
        power_data = np.full(int(sample_rate * duration), 100.0)
        power_trace = create_trace(power_data, sample_rate)

        # Request energy beyond trace duration
        e = energy(power_trace, start_time=0.2, end_time=0.3)

        # Should return 0 or very small value
        assert abs(e) < 1.0

    def test_energy_inverted_time_limits(self, sample_rate: float) -> None:
        """Test energy with end_time < start_time."""
        duration = 0.1
        power_data = np.full(int(sample_rate * duration), 100.0)
        power_trace = create_trace(power_data, sample_rate)

        # End time before start time - should return 0 or small value
        e = energy(power_trace, start_time=0.08, end_time=0.02)

        assert abs(e) < 1.0

    def test_interpolation_maintains_power_average(self, sample_rate: float) -> None:
        """Test that interpolation preserves average power reasonably."""
        duration = 0.1
        voltage_high_rate = create_dc_trace(12.0, sample_rate, duration=duration)
        current_low_rate = create_dc_trace(2.0, sample_rate / 4, duration=duration)

        power = instantaneous_power(voltage_high_rate, current_low_rate, interpolate_if_needed=True)

        # Average should still be close to 24W
        assert abs(np.mean(power.data) - 24.0) < 1.0

    def test_square_wave_power(self, sample_rate: float) -> None:
        """Test power calculation with square wave voltage."""
        frequency = 60.0
        voltage = create_square_wave_trace(120.0, frequency, sample_rate)
        current = create_dc_trace(2.0, sample_rate)

        p_avg = average_power(voltage=voltage, current=current)

        # Square wave alternates between 0 and 120V with 50% duty cycle
        # Average power = (0*2 + 120*2) / 2 = 120W
        assert abs(p_avg - 120.0) < 20.0

    def test_duty_cycle_affects_average_power(self, sample_rate: float) -> None:
        """Test that duty cycle affects average power."""
        frequency = 60.0
        voltage_50 = create_square_wave_trace(120.0, frequency, sample_rate, duty_cycle=0.5)
        voltage_25 = create_square_wave_trace(120.0, frequency, sample_rate, duty_cycle=0.25)
        current = create_dc_trace(2.0, sample_rate)

        p_avg_50 = average_power(voltage=voltage_50, current=current)
        p_avg_25 = average_power(voltage=voltage_25, current=current)

        # 25% duty cycle should have lower average power than 50%
        assert p_avg_25 < p_avg_50
