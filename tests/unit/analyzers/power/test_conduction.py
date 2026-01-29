"""Unit tests for conduction loss analysis.


Tests cover conduction loss calculations, on-resistance measurement,
forward voltage, temperature derating, and device-specific loss functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.power.conduction import (
    conduction_loss,
    diode_conduction_loss,
    duty_cycle_weighted_loss,
    forward_voltage,
    igbt_conduction_loss,
    mosfet_conduction_loss,
    on_resistance,
    temperature_derating,
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
    value: float,
    sample_rate: float,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a constant DC waveform trace."""
    num_samples = int(sample_rate * duration)
    data = np.full(num_samples, value)
    return create_trace(data, sample_rate)


def create_noisy_dc_trace(
    value: float,
    noise_amplitude: float,
    sample_rate: float,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a DC trace with added noise."""
    num_samples = int(sample_rate * duration)
    np.random.seed(42)  # Reproducible noise
    data = np.full(num_samples, value) + np.random.uniform(
        -noise_amplitude, noise_amplitude, num_samples
    )
    return create_trace(data, sample_rate)


def create_resistive_vi_traces(
    current: float,
    resistance: float,
    sample_rate: float,
    duration: float = 0.1,
) -> tuple[WaveformTrace, WaveformTrace]:
    """Create V-I traces following Ohm's law (V = I * R)."""
    num_samples = int(sample_rate * duration)
    i_data = np.full(num_samples, current)
    v_data = i_data * resistance
    return create_trace(v_data, sample_rate), create_trace(i_data, sample_rate)


def create_variable_current_traces(
    i_min: float,
    i_max: float,
    resistance: float,
    sample_rate: float,
    duration: float = 0.1,
) -> tuple[WaveformTrace, WaveformTrace]:
    """Create V-I traces with varying current."""
    num_samples = int(sample_rate * duration)
    i_data = np.linspace(i_min, i_max, num_samples)
    v_data = i_data * resistance
    return create_trace(v_data, sample_rate), create_trace(i_data, sample_rate)


@pytest.mark.unit
@pytest.mark.power
class TestConductionLoss:
    """Test conduction_loss function."""

    def test_conduction_loss_with_duty_cycle(self, sample_rate: float) -> None:
        """Test conduction loss calculation with duty cycle."""
        voltage = create_dc_trace(1.0, sample_rate)  # 1V V_on
        current = create_dc_trace(10.0, sample_rate)  # 10A

        p_cond = conduction_loss(voltage, current, duty_cycle=0.5)

        # P = V * I * D = 1 * 10 * 0.5 = 5W
        assert abs(p_cond - 5.0) < 0.1

    def test_conduction_loss_without_duty_cycle(self, sample_rate: float) -> None:
        """Test conduction loss without duty cycle (instantaneous average)."""
        voltage = create_dc_trace(1.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)

        p_cond = conduction_loss(voltage, current)

        # P = mean(V * I) = 1 * 10 = 10W
        assert abs(p_cond - 10.0) < 0.1

    def test_conduction_loss_full_duty(self, sample_rate: float) -> None:
        """Test conduction loss with 100% duty cycle."""
        voltage = create_dc_trace(2.0, sample_rate)
        current = create_dc_trace(5.0, sample_rate)

        p_cond = conduction_loss(voltage, current, duty_cycle=1.0)

        # P = V * I * D = 2 * 5 * 1.0 = 10W
        assert abs(p_cond - 10.0) < 0.1

    def test_conduction_loss_zero_duty(self, sample_rate: float) -> None:
        """Test conduction loss with 0% duty cycle."""
        voltage = create_dc_trace(2.0, sample_rate)
        current = create_dc_trace(5.0, sample_rate)

        p_cond = conduction_loss(voltage, current, duty_cycle=0.0)

        # P = V * I * D = 2 * 5 * 0.0 = 0W
        assert abs(p_cond) < 0.001

    def test_conduction_loss_varying_current(self, sample_rate: float) -> None:
        """Test conduction loss with varying current."""
        num_samples = 1000
        i_data = np.linspace(0, 20, num_samples)  # Ramp from 0 to 20A
        v_data = np.full(num_samples, 1.0)  # Constant 1V

        voltage = create_trace(v_data, sample_rate)
        current = create_trace(i_data, sample_rate)

        p_cond = conduction_loss(voltage, current)

        # mean(V * I) = 1 * mean(0 to 20) = 1 * 10 = 10W
        assert abs(p_cond - 10.0) < 0.5

    def test_conduction_loss_different_lengths(self, sample_rate: float) -> None:
        """Test conduction loss with different trace lengths."""
        voltage = create_dc_trace(1.0, sample_rate, duration=0.1)
        current = create_dc_trace(10.0, sample_rate, duration=0.05)

        p_cond = conduction_loss(voltage, current)

        # Should truncate to shorter length
        assert abs(p_cond - 10.0) < 0.1

    def test_conduction_loss_zero_voltage(self, sample_rate: float) -> None:
        """Test conduction loss with zero voltage."""
        voltage = create_dc_trace(0.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)

        p_cond = conduction_loss(voltage, current)

        assert abs(p_cond) < 0.001

    def test_conduction_loss_zero_current(self, sample_rate: float) -> None:
        """Test conduction loss with zero current."""
        voltage = create_dc_trace(1.0, sample_rate)
        current = create_dc_trace(0.0, sample_rate)

        p_cond = conduction_loss(voltage, current)

        assert abs(p_cond) < 0.001


@pytest.mark.unit
@pytest.mark.power
class TestOnResistance:
    """Test on_resistance function."""

    def test_on_resistance_basic(self, sample_rate: float) -> None:
        """Test basic on-resistance calculation with varying current."""
        r_expected = 0.010  # 10 mOhm
        # Use varying current to get a proper linear fit
        voltage, current = create_variable_current_traces(
            i_min=5.0, i_max=15.0, resistance=r_expected, sample_rate=sample_rate
        )

        r_on = on_resistance(voltage, current)

        # R = V / I, fitted via polyfit
        assert abs(r_on - r_expected) < 0.001

    def test_on_resistance_varying_current(self, sample_rate: float) -> None:
        """Test on-resistance with varying current."""
        r_expected = 0.020  # 20 mOhm
        voltage, current = create_variable_current_traces(
            i_min=5.0, i_max=20.0, resistance=r_expected, sample_rate=sample_rate
        )

        r_on = on_resistance(voltage, current)

        # Linear fit should give correct resistance
        assert abs(r_on - r_expected) < 0.005

    def test_on_resistance_min_current_filter(self, sample_rate: float) -> None:
        """Test on-resistance with minimum current filter."""
        r_expected = 0.015  # 15 mOhm
        voltage, current = create_variable_current_traces(
            i_min=0.0, i_max=20.0, resistance=r_expected, sample_rate=sample_rate
        )

        r_on = on_resistance(voltage, current, min_current=5.0)

        # Should only use samples where I >= 5A
        assert abs(r_on - r_expected) < 0.01

    def test_on_resistance_custom_min_current(self, sample_rate: float) -> None:
        """Test on-resistance with custom minimum current."""
        r_expected = 0.010
        # Use varying current for proper linear fit
        voltage, current = create_variable_current_traces(
            i_min=5.0, i_max=20.0, resistance=r_expected, sample_rate=sample_rate
        )

        r_on = on_resistance(voltage, current, min_current=5.0)

        assert abs(r_on - r_expected) < 0.005

    def test_on_resistance_all_below_min_returns_nan(self, sample_rate: float) -> None:
        """Test that returns NaN when all current below minimum."""
        voltage = create_dc_trace(0.1, sample_rate)
        current = create_dc_trace(1.0, sample_rate)  # All 1A

        r_on = on_resistance(voltage, current, min_current=10.0)  # Min 10A

        assert np.isnan(r_on)

    def test_on_resistance_high_resistance(self, sample_rate: float) -> None:
        """Test with high on-resistance value."""
        r_expected = 1.0  # 1 Ohm
        # Use varying current for proper linear fit
        voltage, current = create_variable_current_traces(
            i_min=1.0, i_max=10.0, resistance=r_expected, sample_rate=sample_rate
        )

        r_on = on_resistance(voltage, current)

        assert abs(r_on - r_expected) < 0.1

    def test_on_resistance_very_low(self, sample_rate: float) -> None:
        """Test with very low on-resistance."""
        r_expected = 0.001  # 1 mOhm
        # Use varying current for proper linear fit
        voltage, current = create_variable_current_traces(
            i_min=50.0, i_max=150.0, resistance=r_expected, sample_rate=sample_rate
        )

        r_on = on_resistance(voltage, current)

        assert abs(r_on - r_expected) < 0.0005


@pytest.mark.unit
@pytest.mark.power
class TestForwardVoltage:
    """Test forward_voltage function."""

    def test_forward_voltage_basic(self, sample_rate: float) -> None:
        """Test basic forward voltage extraction."""
        v_f_expected = 0.7  # Typical diode Vf
        voltage = create_dc_trace(v_f_expected, sample_rate)
        current = create_dc_trace(1.0, sample_rate)

        v_f = forward_voltage(voltage, current)

        assert abs(v_f - v_f_expected) < 0.1

    def test_forward_voltage_with_threshold(self, sample_rate: float) -> None:
        """Test forward voltage at specific current threshold."""
        # Create V-I characteristic with some resistance
        num_samples = 1000
        i_data = np.linspace(0.1, 10.0, num_samples)
        v_data = 0.7 + 0.05 * i_data  # V = 0.7 + 0.05*I

        voltage = create_trace(v_data, sample_rate)
        current = create_trace(i_data, sample_rate)

        v_f = forward_voltage(voltage, current, current_threshold=5.0)

        # At I=5A, V = 0.7 + 0.05*5 = 0.95V
        assert abs(v_f - 0.95) < 0.1

    def test_forward_voltage_interpolation(self, sample_rate: float) -> None:
        """Test forward voltage with interpolation when no exact match."""
        num_samples = 100
        i_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
        v_data = 0.7 + 0.1 * i_data

        voltage = create_trace(v_data, sample_rate)
        current = create_trace(i_data, sample_rate)

        # Request at current not exactly in data
        v_f = forward_voltage(voltage, current, current_threshold=2.5)

        # Should find closest match
        assert v_f > 0.7

    def test_forward_voltage_auto_threshold(self, sample_rate: float) -> None:
        """Test forward voltage with auto threshold (10% of peak)."""
        num_samples = 1000
        i_data = np.linspace(0.1, 10.0, num_samples)
        v_data = 0.6 + 0.02 * i_data

        voltage = create_trace(v_data, sample_rate)
        current = create_trace(i_data, sample_rate)

        v_f = forward_voltage(voltage, current)

        # Auto threshold = 10% of 10A = 1A
        # At I=1A, V = 0.6 + 0.02 = 0.62V
        assert 0.5 < v_f < 0.8


@pytest.mark.unit
@pytest.mark.power
class TestDutyCycleWeightedLoss:
    """Test duty_cycle_weighted_loss function."""

    def test_single_loss(self) -> None:
        """Test with single loss point."""
        losses = [(10.0, 0.5)]  # 10W at 50% duty

        total = duty_cycle_weighted_loss(losses)

        assert total == 5.0

    def test_multiple_losses(self) -> None:
        """Test with multiple loss points."""
        losses = [
            (10.0, 0.3),  # 10W at 30% duty
            (5.0, 0.5),  # 5W at 50% duty
        ]

        total = duty_cycle_weighted_loss(losses)

        # 10 * 0.3 + 5 * 0.5 = 3 + 2.5 = 5.5W
        assert abs(total - 5.5) < 0.001

    def test_duty_cycle_exceeds_one_raises_error(self) -> None:
        """Test that duty cycles > 1.0 raise error."""
        losses = [
            (10.0, 0.6),
            (5.0, 0.6),  # Total duty = 1.2
        ]

        with pytest.raises(AnalysisError, match="exceeds 1.0"):
            duty_cycle_weighted_loss(losses)

    def test_zero_duty_cycle(self) -> None:
        """Test with zero duty cycle."""
        losses = [(10.0, 0.0)]

        total = duty_cycle_weighted_loss(losses)

        assert total == 0.0

    def test_full_duty_cycle(self) -> None:
        """Test with 100% duty cycle."""
        losses = [(25.0, 1.0)]

        total = duty_cycle_weighted_loss(losses)

        assert total == 25.0

    def test_many_operating_points(self) -> None:
        """Test with many operating points."""
        losses = [
            (5.0, 0.1),
            (10.0, 0.2),
            (15.0, 0.3),
            (20.0, 0.2),
        ]

        total = duty_cycle_weighted_loss(losses)

        # 5*0.1 + 10*0.2 + 15*0.3 + 20*0.2 = 0.5 + 2 + 4.5 + 4 = 11W
        assert abs(total - 11.0) < 0.001

    def test_empty_list(self) -> None:
        """Test with empty losses list."""
        losses: list[tuple[float, float]] = []

        total = duty_cycle_weighted_loss(losses)

        assert total == 0.0


@pytest.mark.unit
@pytest.mark.power
class TestTemperatureDerating:
    """Test temperature_derating function."""

    def test_temperature_derating_at_25c(self) -> None:
        """Test that 25C returns original resistance."""
        r_on = temperature_derating(0.010, temperature=25.0)

        assert abs(r_on - 0.010) < 0.0001

    def test_temperature_derating_above_25c(self) -> None:
        """Test derating above 25C."""
        r_25c = 0.010  # 10 mOhm at 25C

        r_100c = temperature_derating(r_25c, temperature=100.0)

        # R = 0.010 * (1 + 0.004 * (100-25)) = 0.010 * 1.3 = 0.013 Ohm
        assert abs(r_100c - 0.013) < 0.0001

    def test_temperature_derating_below_25c(self) -> None:
        """Test derating below 25C."""
        r_25c = 0.010

        r_0c = temperature_derating(r_25c, temperature=0.0)

        # R = 0.010 * (1 + 0.004 * (0-25)) = 0.010 * 0.9 = 0.009 Ohm
        assert abs(r_0c - 0.009) < 0.0001

    def test_temperature_derating_custom_coefficient(self) -> None:
        """Test with custom temperature coefficient."""
        r_25c = 0.010

        r_100c = temperature_derating(r_25c, temperature=100.0, temp_coefficient=0.002)

        # R = 0.010 * (1 + 0.002 * 75) = 0.010 * 1.15 = 0.0115 Ohm
        assert abs(r_100c - 0.0115) < 0.0001

    def test_temperature_derating_high_temp(self) -> None:
        """Test at high temperature."""
        r_25c = 0.010

        r_150c = temperature_derating(r_25c, temperature=150.0)

        # R = 0.010 * (1 + 0.004 * 125) = 0.010 * 1.5 = 0.015 Ohm
        assert abs(r_150c - 0.015) < 0.0001

    def test_temperature_derating_negative_temp(self) -> None:
        """Test at negative temperature."""
        r_25c = 0.010

        r_neg25 = temperature_derating(r_25c, temperature=-25.0)

        # R = 0.010 * (1 + 0.004 * (-50)) = 0.010 * 0.8 = 0.008 Ohm
        assert abs(r_neg25 - 0.008) < 0.0001


@pytest.mark.unit
@pytest.mark.power
class TestMosfetConductionLoss:
    """Test mosfet_conduction_loss function."""

    def test_mosfet_conduction_loss_basic(self) -> None:
        """Test basic MOSFET conduction loss calculation."""
        p_cond = mosfet_conduction_loss(i_rms=10.0, r_ds_on=0.010)

        # P = I^2 * R = 100 * 0.01 = 1W
        assert abs(p_cond - 1.0) < 0.01

    def test_mosfet_conduction_loss_with_temperature(self) -> None:
        """Test MOSFET conduction loss with temperature derating."""
        p_cond = mosfet_conduction_loss(i_rms=10.0, r_ds_on=0.010, temperature=100.0)

        # At 100C: R = 0.01 * 1.3 = 0.013 Ohm
        # P = 100 * 0.013 = 1.3W
        assert abs(p_cond - 1.3) < 0.05

    def test_mosfet_conduction_loss_high_current(self) -> None:
        """Test with high current."""
        p_cond = mosfet_conduction_loss(i_rms=50.0, r_ds_on=0.005)

        # P = 2500 * 0.005 = 12.5W
        assert abs(p_cond - 12.5) < 0.1

    def test_mosfet_conduction_loss_custom_temp_coeff(self) -> None:
        """Test with custom temperature coefficient."""
        p_cond = mosfet_conduction_loss(
            i_rms=10.0, r_ds_on=0.010, temperature=100.0, temp_coefficient=0.002
        )

        # At 100C with alpha=0.002: R = 0.01 * 1.15 = 0.0115 Ohm
        # P = 100 * 0.0115 = 1.15W
        assert abs(p_cond - 1.15) < 0.05


@pytest.mark.unit
@pytest.mark.power
class TestDiodeConductionLoss:
    """Test diode_conduction_loss function."""

    def test_diode_conduction_loss_basic(self) -> None:
        """Test basic diode conduction loss."""
        p_cond = diode_conduction_loss(i_avg=5.0, i_rms=7.0, v_f=0.7)

        # P = Vf * I_avg = 0.7 * 5 = 3.5W
        assert abs(p_cond - 3.5) < 0.01

    def test_diode_conduction_loss_with_dynamic_resistance(self) -> None:
        """Test diode loss with dynamic resistance."""
        p_cond = diode_conduction_loss(i_avg=5.0, i_rms=7.0, v_f=0.7, r_d=0.01)

        # P = Vf * I_avg + r_d * I_rms^2 = 0.7*5 + 0.01*49 = 3.5 + 0.49 = 3.99W
        assert abs(p_cond - 3.99) < 0.01

    def test_diode_conduction_loss_zero_resistance(self) -> None:
        """Test diode loss with zero dynamic resistance."""
        p_cond = diode_conduction_loss(i_avg=10.0, i_rms=10.0, v_f=1.0, r_d=0.0)

        # P = 1.0 * 10 = 10W
        assert abs(p_cond - 10.0) < 0.01

    def test_diode_conduction_loss_high_dynamic_resistance(self) -> None:
        """Test with high dynamic resistance (Schottky diode)."""
        p_cond = diode_conduction_loss(i_avg=2.0, i_rms=3.0, v_f=0.4, r_d=0.05)

        # P = 0.4*2 + 0.05*9 = 0.8 + 0.45 = 1.25W
        assert abs(p_cond - 1.25) < 0.01


@pytest.mark.unit
@pytest.mark.power
class TestIgbtConductionLoss:
    """Test igbt_conduction_loss function."""

    def test_igbt_conduction_loss_basic(self) -> None:
        """Test basic IGBT conduction loss."""
        p_cond = igbt_conduction_loss(i_c=50.0, v_ce_sat=2.0)

        # P = V_ce * I_c = 2 * 50 = 100W
        assert abs(p_cond - 100.0) < 0.1

    def test_igbt_conduction_loss_with_resistance(self) -> None:
        """Test IGBT loss with collector resistance."""
        p_cond = igbt_conduction_loss(i_c=50.0, v_ce_sat=2.0, r_c=0.01)

        # P = V_ce*I_c + r_c*I_c^2 = 2*50 + 0.01*2500 = 100 + 25 = 125W
        assert abs(p_cond - 125.0) < 0.1

    def test_igbt_conduction_loss_zero_resistance(self) -> None:
        """Test IGBT loss with zero resistance."""
        p_cond = igbt_conduction_loss(i_c=100.0, v_ce_sat=1.5, r_c=0.0)

        # P = 1.5 * 100 = 150W
        assert abs(p_cond - 150.0) < 0.1

    def test_igbt_conduction_loss_low_current(self) -> None:
        """Test IGBT loss at low current."""
        p_cond = igbt_conduction_loss(i_c=1.0, v_ce_sat=1.8, r_c=0.02)

        # P = 1.8*1 + 0.02*1 = 1.8 + 0.02 = 1.82W
        assert abs(p_cond - 1.82) < 0.01


@pytest.mark.unit
@pytest.mark.power
class TestConductionEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_traces(self, sample_rate: float) -> None:
        """Test with empty traces."""
        voltage = create_trace(np.array([]), sample_rate)
        current = create_trace(np.array([]), sample_rate)

        # numpy mean of empty array gives warning but returns nan
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_cond = conduction_loss(voltage, current)

        assert np.isnan(p_cond)

    def test_single_sample(self, sample_rate: float) -> None:
        """Test with single sample."""
        voltage = create_trace(np.array([1.0]), sample_rate)
        current = create_trace(np.array([10.0]), sample_rate)

        p_cond = conduction_loss(voltage, current)

        assert abs(p_cond - 10.0) < 0.1

    def test_negative_voltage(self, sample_rate: float) -> None:
        """Test with negative voltage (regeneration)."""
        voltage = create_dc_trace(-1.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)

        p_cond = conduction_loss(voltage, current)

        # Negative power = regeneration
        assert abs(p_cond - (-10.0)) < 0.1

    def test_negative_current(self, sample_rate: float) -> None:
        """Test with negative current."""
        voltage = create_dc_trace(1.0, sample_rate)
        current = create_dc_trace(-10.0, sample_rate)

        p_cond = conduction_loss(voltage, current)

        # Negative power
        assert abs(p_cond - (-10.0)) < 0.1

    def test_very_small_values(self, sample_rate: float) -> None:
        """Test with very small voltage/current values."""
        voltage = create_dc_trace(0.001, sample_rate)
        current = create_dc_trace(0.001, sample_rate)

        p_cond = conduction_loss(voltage, current)

        assert abs(p_cond - 0.000001) < 1e-9

    def test_very_large_values(self, sample_rate: float) -> None:
        """Test with very large voltage/current values."""
        voltage = create_dc_trace(1000.0, sample_rate)
        current = create_dc_trace(1000.0, sample_rate)

        p_cond = conduction_loss(voltage, current)

        assert abs(p_cond - 1000000.0) < 100.0

    def test_noisy_measurements(self, sample_rate: float) -> None:
        """Test with noisy measurements."""
        voltage = create_noisy_dc_trace(1.0, 0.1, sample_rate)
        current = create_noisy_dc_trace(10.0, 0.5, sample_rate)

        p_cond = conduction_loss(voltage, current)

        # Should be approximately 10W despite noise
        assert abs(p_cond - 10.0) < 1.0

    def test_duty_cycle_boundary(self) -> None:
        """Test duty cycle at boundary (exactly 1.0)."""
        losses = [(10.0, 1.0)]  # Exactly 100% duty

        total = duty_cycle_weighted_loss(losses)

        assert total == 10.0

    def test_on_resistance_with_noise(self, sample_rate: float) -> None:
        """Test on-resistance calculation with noisy data."""
        r_expected = 0.010
        num_samples = 1000
        np.random.seed(42)

        i_data = 10.0 + np.random.uniform(-1.0, 1.0, num_samples)
        v_data = i_data * r_expected + np.random.uniform(-0.01, 0.01, num_samples)

        voltage = create_trace(v_data, sample_rate)
        current = create_trace(i_data, sample_rate)

        r_on = on_resistance(voltage, current)

        # Should be close to expected despite noise
        assert abs(r_on - r_expected) < 0.005

    def test_temperature_derating_at_absolute_zero(self) -> None:
        """Test temperature derating at very low temperature."""
        r_25c = 0.010

        # At -273C (near absolute zero)
        r_cold = temperature_derating(r_25c, temperature=-273.0)

        # R = 0.010 * (1 + 0.004 * (-298)) = 0.010 * -0.192 = -0.00192
        # This shows physical limitation of linear model at extreme temps
        assert r_cold < r_25c
