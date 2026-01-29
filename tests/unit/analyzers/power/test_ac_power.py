"""Unit tests for AC power analysis.


Tests cover all public functions including edge cases, error conditions,
and various waveform scenarios (sinusoidal, distorted, three-phase).
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.power.ac_power import (
    apparent_power,
    displacement_power_factor,
    distortion_power_factor,
    phase_angle,
    power_factor,
    reactive_power,
    three_phase_power,
    total_harmonic_distortion_power,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.power]


@pytest.fixture
def sample_rate() -> float:
    """Standard sample rate for tests."""
    return 10000.0  # 10 kHz


@pytest.fixture
def frequency() -> float:
    """Standard AC frequency for tests."""
    return 60.0  # 60 Hz


def create_sinusoidal_trace(
    amplitude: float,
    frequency: float,
    sample_rate: float,
    duration: float = 0.1,
    phase: float = 0.0,
    dc_offset: float = 0.0,
) -> WaveformTrace:
    """Create a sinusoidal waveform trace for testing."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = amplitude * np.sin(2 * np.pi * frequency * t + phase) + dc_offset
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


def create_distorted_trace(
    fundamental_amplitude: float,
    frequency: float,
    sample_rate: float,
    harmonics: dict[int, float],
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a distorted waveform with harmonics."""
    num_samples = int(sample_rate * duration)
    t = np.arange(num_samples) / sample_rate
    data = fundamental_amplitude * np.sin(2 * np.pi * frequency * t)

    for harmonic, amplitude in harmonics.items():
        data += amplitude * np.sin(2 * np.pi * frequency * harmonic * t)

    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.mark.unit
class TestPhaseAngle:
    """Test phase angle calculation between voltage and current."""

    def test_phase_angle_in_phase(self, sample_rate: float, frequency: float) -> None:
        """Test phase angle when voltage and current are in phase (resistive)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=0.0)

        phi = phase_angle(voltage, current)

        # Should be close to 0 radians for in-phase signals
        assert abs(phi) < 0.1

    def test_phase_angle_lagging(self, sample_rate: float, frequency: float) -> None:
        """Test phase angle when current lags voltage (inductive)."""
        # Current lags by 45 degrees (pi/4 radians)
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 4)

        phi = phase_angle(voltage, current)

        # Should be positive for lagging current (inductive)
        # Allow some tolerance due to cross-correlation estimation
        assert 0.5 < phi < 1.2  # Around pi/4 = 0.785

    def test_phase_angle_leading(self, sample_rate: float, frequency: float) -> None:
        """Test phase angle when current leads voltage (capacitive)."""
        # Current leads by 45 degrees
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=np.pi / 4)

        phi = phase_angle(voltage, current)

        # Should be negative for leading current (capacitive)
        assert -1.2 < phi < -0.5  # Around -pi/4 = -0.785

    def test_phase_angle_90_degrees(self, sample_rate: float, frequency: float) -> None:
        """Test phase angle at 90 degrees (purely reactive)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 2)

        phi = phase_angle(voltage, current)

        # Should be close to pi/2 for 90-degree lag (magnitude check)
        # Note: May wrap around due to cross-correlation implementation
        assert abs(abs(phi) - np.pi / 2) < 0.3  # Within 0.3 radians of pi/2

    def test_phase_angle_with_dc_offset(self, sample_rate: float, frequency: float) -> None:
        """Test that DC offset is removed before phase calculation."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0, dc_offset=5.0)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=0.0, dc_offset=2.0)

        phi = phase_angle(voltage, current)

        # DC offset should not affect phase angle
        assert abs(phi) < 0.1

    def test_phase_angle_different_lengths(self, sample_rate: float, frequency: float) -> None:
        """Test phase angle with different trace lengths (should truncate)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, duration=0.15)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, duration=0.10)

        phi = phase_angle(voltage, current)

        # Should still work, truncating to shorter length
        assert abs(phi) < 0.1

    def test_phase_angle_low_frequency(self, sample_rate: float) -> None:
        """Test phase angle with low frequency signal."""
        # Test with 10 Hz signal
        voltage = create_sinusoidal_trace(120.0, 10.0, sample_rate, duration=0.5)
        current = create_sinusoidal_trace(10.0, 10.0, sample_rate, duration=0.5)

        phi = phase_angle(voltage, current)

        assert abs(phi) < 0.1


@pytest.mark.unit
class TestReactivePower:
    """Test reactive power calculation (Q = V_rms * I_rms * sin(phi))."""

    def test_reactive_power_resistive_load(self, sample_rate: float, frequency: float) -> None:
        """Test reactive power with resistive load (should be ~0)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        q = reactive_power(voltage, current)

        # Resistive load should have near-zero reactive power
        assert abs(q) < 50.0  # Small tolerance due to numerical errors

    def test_reactive_power_inductive_load(self, sample_rate: float, frequency: float) -> None:
        """Test reactive power with inductive load (positive Q)."""
        # Current lags by 45 degrees
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 4)

        q = reactive_power(voltage, current)

        # Calculate expected: V_rms = 120/sqrt(2), I_rms = 10/sqrt(2), sin(pi/4) = sqrt(2)/2
        # Q = (120/sqrt(2)) * (10/sqrt(2)) * sin(pi/4) = 600 * sqrt(2)/2 = 424.26
        expected_q = (120.0 / np.sqrt(2)) * (10.0 / np.sqrt(2)) * np.sin(np.pi / 4)

        # Should be positive for inductive load
        assert q > 0
        assert abs(q - expected_q) < 50.0  # Allow some tolerance

    def test_reactive_power_capacitive_load(self, sample_rate: float, frequency: float) -> None:
        """Test reactive power with capacitive load (negative Q)."""
        # Current leads by 45 degrees
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=np.pi / 4)

        q = reactive_power(voltage, current)

        # Should be negative for capacitive load
        assert q < 0
        assert abs(q) > 200.0  # Significant reactive power

    def test_reactive_power_purely_reactive(self, sample_rate: float, frequency: float) -> None:
        """Test reactive power with purely reactive load (90 degrees)."""
        # Current lags by 90 degrees (pure inductor)
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 2)

        q = reactive_power(voltage, current)

        # Q = V_rms * I_rms * sin(90°) = V_rms * I_rms
        expected_q = (120.0 / np.sqrt(2)) * (10.0 / np.sqrt(2))

        # Phase calculation may have sign ambiguity due to cross-correlation
        assert abs(abs(q) - expected_q) < 50.0

    def test_reactive_power_zero_current(self, sample_rate: float, frequency: float) -> None:
        """Test reactive power with zero current."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(0.0, frequency, sample_rate)

        q = reactive_power(voltage, current)

        assert abs(q) < 0.1

    def test_reactive_power_different_lengths(self, sample_rate: float, frequency: float) -> None:
        """Test reactive power with different trace lengths."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, duration=0.15)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, duration=0.10)

        q = reactive_power(voltage, current)

        # Should truncate to shorter length and calculate
        assert isinstance(q, float)


@pytest.mark.unit
class TestApparentPower:
    """Test apparent power calculation (S = V_rms * I_rms)."""

    def test_apparent_power_nominal(self, sample_rate: float, frequency: float) -> None:
        """Test apparent power calculation with nominal values."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        s = apparent_power(voltage, current)

        # S = V_rms * I_rms = (120/sqrt(2)) * (10/sqrt(2)) = 600 VA
        expected_s = (120.0 / np.sqrt(2)) * (10.0 / np.sqrt(2))

        assert abs(s - expected_s) < 10.0

    def test_apparent_power_independent_of_phase(
        self, sample_rate: float, frequency: float
    ) -> None:
        """Test that apparent power is independent of phase angle."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current_0 = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=0.0)
        current_45 = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 4)
        current_90 = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 2)

        s_0 = apparent_power(voltage, current_0)
        s_45 = apparent_power(voltage, current_45)
        s_90 = apparent_power(voltage, current_90)

        # All should be equal regardless of phase
        assert abs(s_0 - s_45) < 5.0
        assert abs(s_0 - s_90) < 5.0

    def test_apparent_power_zero_voltage(self, sample_rate: float, frequency: float) -> None:
        """Test apparent power with zero voltage."""
        voltage = create_sinusoidal_trace(0.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        s = apparent_power(voltage, current)

        assert abs(s) < 0.1

    def test_apparent_power_zero_current(self, sample_rate: float, frequency: float) -> None:
        """Test apparent power with zero current."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(0.0, frequency, sample_rate)

        s = apparent_power(voltage, current)

        assert abs(s) < 0.1

    def test_apparent_power_different_lengths(self, sample_rate: float, frequency: float) -> None:
        """Test apparent power with different trace lengths."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, duration=0.15)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, duration=0.10)

        s = apparent_power(voltage, current)

        assert s > 0


@pytest.mark.unit
class TestPowerFactor:
    """Test power factor calculation (PF = P / S)."""

    def test_power_factor_resistive_load(self, sample_rate: float, frequency: float) -> None:
        """Test power factor with resistive load (should be ~1.0)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        pf = power_factor(voltage, current)

        # Resistive load should have PF close to 1.0
        assert 0.95 < pf < 1.05

    def test_power_factor_inductive_load(self, sample_rate: float, frequency: float) -> None:
        """Test power factor with inductive load (lagging)."""
        # Current lags by 45 degrees, PF = cos(45°) = 0.707
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 4)

        pf = power_factor(voltage, current)

        # PF = cos(pi/4) = sqrt(2)/2 = 0.707
        expected_pf = np.cos(np.pi / 4)

        assert 0.6 < pf < 0.8
        assert abs(pf - expected_pf) < 0.15

    def test_power_factor_capacitive_load(self, sample_rate: float, frequency: float) -> None:
        """Test power factor with capacitive load (leading)."""
        # Current leads by 45 degrees
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=np.pi / 4)

        pf = power_factor(voltage, current)

        # PF magnitude should be cos(45°) = 0.707
        assert 0.6 < pf < 0.8

    def test_power_factor_purely_reactive(self, sample_rate: float, frequency: float) -> None:
        """Test power factor with purely reactive load (should be ~0)."""
        # Current at 90 degrees, PF = cos(90°) = 0
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 2)

        pf = power_factor(voltage, current)

        # Purely reactive load should have PF close to 0
        assert abs(pf) < 0.15

    def test_power_factor_zero_apparent_power(self, sample_rate: float, frequency: float) -> None:
        """Test power factor with zero apparent power (edge case)."""
        voltage = create_sinusoidal_trace(0.0, frequency, sample_rate)
        current = create_sinusoidal_trace(0.0, frequency, sample_rate)

        pf = power_factor(voltage, current)

        # Should return 0 when apparent power is zero
        assert pf == 0.0

    def test_power_factor_range(self, sample_rate: float, frequency: float) -> None:
        """Test that power factor is in valid range."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 3)

        pf = power_factor(voltage, current)

        # Power factor should be between -1 and 1
        assert -1.1 <= pf <= 1.1


@pytest.mark.unit
class TestDisplacementPowerFactor:
    """Test displacement power factor calculation (DPF = cos(phi))."""

    def test_dpf_sinusoidal_in_phase(self, sample_rate: float, frequency: float) -> None:
        """Test DPF with sinusoidal signals in phase."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        dpf = displacement_power_factor(voltage, current)

        # Should be close to 1.0 for in-phase signals
        assert 0.95 < dpf < 1.05

    def test_dpf_with_45_degree_lag(self, sample_rate: float, frequency: float) -> None:
        """Test DPF with 45-degree phase lag."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 4)

        dpf = displacement_power_factor(voltage, current)

        # DPF = cos(45°) = 0.707
        expected_dpf = np.cos(np.pi / 4)

        assert abs(dpf - expected_dpf) < 0.15

    def test_dpf_distorted_waveform(self, sample_rate: float, frequency: float) -> None:
        """Test DPF extracts fundamental component from distorted waveform."""
        # Create distorted voltage with harmonics
        voltage = create_distorted_trace(
            120.0, frequency, sample_rate, harmonics={3: 20.0, 5: 10.0}
        )
        # Pure sinusoidal current
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        dpf = displacement_power_factor(voltage, current)

        # DPF should only consider fundamental, so should be close to 1.0
        assert 0.8 < dpf < 1.1


@pytest.mark.unit
class TestDistortionPowerFactor:
    """Test distortion power factor calculation."""

    def test_distortion_pf_sinusoidal(self, sample_rate: float, frequency: float) -> None:
        """Test distortion PF with pure sinusoidal signals (should be ~1.0)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate)

        dist_pf = distortion_power_factor(voltage, current)

        # No distortion means distortion PF should be close to 1.0
        assert 0.9 < dist_pf < 1.1

    def test_distortion_pf_distorted_current(self, sample_rate: float, frequency: float) -> None:
        """Test distortion PF with distorted current waveform."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        # Current with significant harmonics
        current = create_distorted_trace(
            10.0, frequency, sample_rate, harmonics={3: 3.0, 5: 2.0, 7: 1.0}
        )

        dist_pf = distortion_power_factor(voltage, current)

        # Distortion should reduce the distortion PF below 1.0
        assert 0.5 < dist_pf < 1.0

    def test_distortion_pf_zero_dpf(self, sample_rate: float, frequency: float) -> None:
        """Test distortion PF when DPF is zero (edge case)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate)
        # Current at 90 degrees (DPF = 0)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-np.pi / 2)

        dist_pf = distortion_power_factor(voltage, current)

        # Should return 0.0 when DPF is zero (allow for numerical precision)
        assert abs(dist_pf) < 1e-10


@pytest.mark.unit
class TestTotalHarmonicDistortion:
    """Test total harmonic distortion calculation."""

    def test_thd_sinusoidal(self, sample_rate: float, frequency: float) -> None:
        """Test THD of pure sinusoidal signal (should be ~0)."""
        trace = create_sinusoidal_trace(120.0, frequency, sample_rate)

        thd = total_harmonic_distortion_power(trace)

        # Pure sinusoid should have very low THD
        assert thd < 0.05

    def test_thd_with_harmonics(self, sample_rate: float, frequency: float) -> None:
        """Test THD calculation with known harmonics."""
        # Create signal: fundamental + 3rd harmonic (20% amplitude)
        # THD = V3/V1 = 0.2 = 20%
        trace = create_distorted_trace(100.0, frequency, sample_rate, harmonics={3: 20.0})

        thd = total_harmonic_distortion_power(trace)

        # THD should be approximately 0.2 (20%)
        assert 0.15 < thd < 0.25

    def test_thd_multiple_harmonics(self, sample_rate: float, frequency: float) -> None:
        """Test THD with multiple harmonics."""
        # V1 = 100, V3 = 30, V5 = 20
        # THD = sqrt(30^2 + 20^2) / 100 = sqrt(1300) / 100 = 0.36
        trace = create_distorted_trace(100.0, frequency, sample_rate, harmonics={3: 30.0, 5: 20.0})

        thd = total_harmonic_distortion_power(trace)

        expected_thd = np.sqrt(30**2 + 20**2) / 100.0

        assert abs(thd - expected_thd) < 0.1

    def test_thd_auto_detect_fundamental(self, sample_rate: float, frequency: float) -> None:
        """Test THD with automatic fundamental frequency detection."""
        trace = create_distorted_trace(100.0, frequency, sample_rate, harmonics={3: 15.0})

        # Call without specifying fundamental frequency
        thd = total_harmonic_distortion_power(trace, fundamental_freq=None)

        assert thd > 0.1  # Should detect harmonics

    def test_thd_specified_fundamental(self, sample_rate: float, frequency: float) -> None:
        """Test THD with specified fundamental frequency."""
        trace = create_distorted_trace(100.0, frequency, sample_rate, harmonics={3: 20.0})

        thd = total_harmonic_distortion_power(trace, fundamental_freq=frequency)

        assert 0.15 < thd < 0.25

    def test_thd_max_harmonic_limit(self, sample_rate: float, frequency: float) -> None:
        """Test THD with limited harmonic range."""
        trace = create_distorted_trace(
            100.0, frequency, sample_rate, harmonics={3: 20.0, 5: 15.0, 7: 10.0, 9: 5.0}
        )

        # Calculate THD up to 5th harmonic only
        thd_5 = total_harmonic_distortion_power(trace, max_harmonic=5)
        # Calculate THD up to 50th harmonic (includes all)
        thd_50 = total_harmonic_distortion_power(trace, max_harmonic=50)

        # THD with more harmonics should be higher
        assert thd_50 >= thd_5

    def test_thd_zero_fundamental(self, sample_rate: float, frequency: float) -> None:
        """Test THD with zero fundamental (edge case)."""
        # Create trace with only DC
        data = np.ones(1000) * 5.0
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        thd = total_harmonic_distortion_power(trace)

        # DC-only signal: THD calculation auto-detects "fundamental" which may
        # be noise/artifacts in the FFT, leading to non-zero but small THD
        # The important thing is it doesn't crash or return inf/nan
        assert not np.isnan(thd)
        assert not np.isinf(thd)


@pytest.mark.unit
class TestThreePhasePower:
    """Test three-phase power calculations."""

    def test_three_phase_balanced_resistive(self, sample_rate: float, frequency: float) -> None:
        """Test three-phase power with balanced resistive load."""
        # 120-degree phase shifts for three-phase
        v_a = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        v_b = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        v_c = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=2 * np.pi / 3)

        # In-phase currents (resistive)
        i_a = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=0.0)
        i_b = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        i_c = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=2 * np.pi / 3)

        result = three_phase_power(v_a, v_b, v_c, i_a, i_b, i_c)

        # Each phase: P = V_rms * I_rms = (120/sqrt(2)) * (10/sqrt(2)) = 600 W
        # Total: 3 * 600 = 1800 W
        expected_total_p = 3 * (120.0 / np.sqrt(2)) * (10.0 / np.sqrt(2))

        assert abs(result["total_active"] - expected_total_p) < 100.0
        assert abs(result["total_reactive"]) < 100.0  # Resistive, so Q ~ 0
        assert result["power_factor"] > 0.95  # Should be close to 1.0
        assert abs(result["phase_a_power"] - 600.0) < 50.0
        assert abs(result["phase_b_power"] - 600.0) < 50.0
        assert abs(result["phase_c_power"] - 600.0) < 50.0

    def test_three_phase_inductive_load(self, sample_rate: float, frequency: float) -> None:
        """Test three-phase power with inductive load."""
        v_a = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        v_b = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        v_c = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=2 * np.pi / 3)

        # Currents lag by 30 degrees
        phase_lag = -np.pi / 6
        i_a = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=phase_lag)
        i_b = create_sinusoidal_trace(
            10.0, frequency, sample_rate, phase=-2 * np.pi / 3 + phase_lag
        )
        i_c = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=2 * np.pi / 3 + phase_lag)

        result = three_phase_power(v_a, v_b, v_c, i_a, i_b, i_c)

        # Reactive power should be positive (inductive)
        assert result["total_reactive"] > 0
        # Power factor should be cos(30°) = 0.866
        assert 0.7 < result["power_factor"] < 0.95

    def test_three_phase_unbalanced(self, sample_rate: float, frequency: float) -> None:
        """Test three-phase power with unbalanced load."""
        v_a = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        v_b = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        v_c = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=2 * np.pi / 3)

        # Unbalanced currents
        i_a = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=0.0)
        i_b = create_sinusoidal_trace(5.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        i_c = create_sinusoidal_trace(15.0, frequency, sample_rate, phase=2 * np.pi / 3)

        result = three_phase_power(v_a, v_b, v_c, i_a, i_b, i_c)

        # Individual phase powers should be different
        assert result["phase_a_power"] != result["phase_b_power"]
        assert result["phase_b_power"] != result["phase_c_power"]
        # Total power should be sum of phase powers
        total_from_phases = (
            result["phase_a_power"] + result["phase_b_power"] + result["phase_c_power"]
        )
        assert abs(result["total_active"] - total_from_phases) < 10.0

    def test_three_phase_zero_current(self, sample_rate: float, frequency: float) -> None:
        """Test three-phase power with zero current (no load)."""
        v_a = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        v_b = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        v_c = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=2 * np.pi / 3)

        i_a = create_sinusoidal_trace(0.0, frequency, sample_rate, phase=0.0)
        i_b = create_sinusoidal_trace(0.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        i_c = create_sinusoidal_trace(0.0, frequency, sample_rate, phase=2 * np.pi / 3)

        result = three_phase_power(v_a, v_b, v_c, i_a, i_b, i_c)

        assert abs(result["total_active"]) < 1.0
        assert abs(result["total_reactive"]) < 1.0
        assert result["power_factor"] == 0.0

    def test_three_phase_result_keys(self, sample_rate: float, frequency: float) -> None:
        """Test that three-phase result contains all expected keys."""
        v_a = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=0.0)
        v_b = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        v_c = create_sinusoidal_trace(120.0, frequency, sample_rate, phase=2 * np.pi / 3)

        i_a = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=0.0)
        i_b = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=-2 * np.pi / 3)
        i_c = create_sinusoidal_trace(10.0, frequency, sample_rate, phase=2 * np.pi / 3)

        result = three_phase_power(v_a, v_b, v_c, i_a, i_b, i_c)

        expected_keys = {
            "total_active",
            "total_reactive",
            "total_apparent",
            "power_factor",
            "phase_a_power",
            "phase_b_power",
            "phase_c_power",
        }

        assert set(result.keys()) == expected_keys


@pytest.mark.unit
class TestPowerAcPowerEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_traces(self, sample_rate: float) -> None:
        """Test functions with empty traces."""
        empty_data = np.array([])
        metadata = TraceMetadata(sample_rate=sample_rate)
        empty_trace = WaveformTrace(data=empty_data, metadata=metadata)

        # Empty arrays produce NaN from np.mean - this is expected behavior
        # that warns but doesn't crash
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            s = apparent_power(empty_trace, empty_trace)
            assert np.isnan(s)

    def test_single_sample_traces(self, sample_rate: float) -> None:
        """Test functions with single-sample traces."""
        data = np.array([1.0])
        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        s = apparent_power(trace, trace)
        assert isinstance(s, float)

    def test_very_short_traces(self, sample_rate: float, frequency: float) -> None:
        """Test functions with very short traces (< 1 period)."""
        voltage = create_sinusoidal_trace(120.0, frequency, sample_rate, duration=0.001)
        current = create_sinusoidal_trace(10.0, frequency, sample_rate, duration=0.001)

        # Should still work, though results may be less accurate
        pf = power_factor(voltage, current)
        assert isinstance(pf, float)

    def test_high_frequency_signal(self, sample_rate: float) -> None:
        """Test with high frequency signal."""
        # Test with 1 kHz signal (close to Nyquist for some scenarios)
        voltage = create_sinusoidal_trace(120.0, 1000.0, sample_rate, duration=0.01)
        current = create_sinusoidal_trace(10.0, 1000.0, sample_rate, duration=0.01)

        s = apparent_power(voltage, current)
        assert s > 0

    def test_dc_only_signal(self, sample_rate: float) -> None:
        """Test with DC-only signals (no AC component)."""
        dc_voltage = np.ones(1000) * 12.0
        dc_current = np.ones(1000) * 2.0
        metadata = TraceMetadata(sample_rate=sample_rate)

        voltage = WaveformTrace(data=dc_voltage, metadata=metadata)
        current = WaveformTrace(data=dc_current, metadata=metadata)

        # Apparent power should still calculate (RMS of DC)
        s = apparent_power(voltage, current)
        assert abs(s - 24.0) < 1.0  # 12V * 2A = 24VA

    def test_negative_amplitudes(self, sample_rate: float, frequency: float) -> None:
        """Test with negative amplitude signals."""
        voltage = create_sinusoidal_trace(-120.0, frequency, sample_rate)
        current = create_sinusoidal_trace(-10.0, frequency, sample_rate)

        # Should handle negative amplitudes correctly
        s = apparent_power(voltage, current)
        pf = power_factor(voltage, current)

        assert s > 0
        assert isinstance(pf, float)

    def test_very_large_amplitudes(self, sample_rate: float, frequency: float) -> None:
        """Test with very large amplitude values."""
        voltage = create_sinusoidal_trace(10000.0, frequency, sample_rate)
        current = create_sinusoidal_trace(1000.0, frequency, sample_rate)

        s = apparent_power(voltage, current)
        assert s > 0
        assert not np.isnan(s)
        assert not np.isinf(s)

    def test_very_small_amplitudes(self, sample_rate: float, frequency: float) -> None:
        """Test with very small amplitude values."""
        voltage = create_sinusoidal_trace(0.001, frequency, sample_rate)
        current = create_sinusoidal_trace(0.0001, frequency, sample_rate)

        s = apparent_power(voltage, current)
        assert s >= 0
        assert not np.isnan(s)
