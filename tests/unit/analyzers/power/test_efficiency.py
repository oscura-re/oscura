"""Comprehensive unit tests for power efficiency calculations.

This module tests all public functions in the efficiency module with
edge cases and error conditions.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.utils.factories import make_waveform_trace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.power]


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-003")
class TestEfficiency:
    """Test basic efficiency calculation."""

    def test_efficiency_basic(self) -> None:
        """Test basic efficiency calculation with ideal converter.

        Validates:
        - Efficiency calculation for 90% efficient converter
        - Result is in ratio form (0-1)
        """
        from oscura.analyzers.power.efficiency import efficiency

        # 100W input, 90W output (90% efficient)
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([9.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert 0 <= eta <= 1
        assert abs(eta - 0.9) < 1e-10

    def test_efficiency_zero_input(self) -> None:
        """Test efficiency with zero input power.

        Validates:
        - Returns 0.0 for zero input
        - No division by zero error
        """
        from oscura.analyzers.power.efficiency import efficiency

        v_in = make_waveform_trace(np.array([0.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([5.0] * 100))
        i_out = make_waveform_trace(np.array([5.0] * 100))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert eta == 0.0

    def test_efficiency_negative_input(self) -> None:
        """Test efficiency with negative input power.

        Validates:
        - Returns 0.0 for negative input
        - Handles edge case gracefully
        """
        from oscura.analyzers.power.efficiency import efficiency

        v_in = make_waveform_trace(np.array([-10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([5.0] * 100))
        i_out = make_waveform_trace(np.array([5.0] * 100))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert eta == 0.0

    def test_efficiency_varying_waveforms(self) -> None:
        """Test efficiency with varying voltage/current waveforms.

        Validates:
        - Efficiency calculated from average power
        - Works with AC-like signals
        """
        from oscura.analyzers.power.efficiency import efficiency

        # Varying signals (simulate switching converter ripple)
        t = np.linspace(0, 1, 1000)
        v_in = make_waveform_trace(12.0 + 0.5 * np.sin(2 * np.pi * 10 * t))
        i_in = make_waveform_trace(8.0 + 0.3 * np.sin(2 * np.pi * 10 * t))
        v_out = make_waveform_trace(5.0 + 0.1 * np.sin(2 * np.pi * 10 * t))
        i_out = make_waveform_trace(15.0 + 0.5 * np.sin(2 * np.pi * 10 * t))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert 0 <= eta <= 1
        # Average power in ≈ 12*8 = 96W, out ≈ 5*15 = 75W → ~78%
        assert 0.7 < eta < 0.9

    def test_efficiency_greater_than_one_invalid(self) -> None:
        """Test that efficiency > 1 is possible (incorrect measurement).

        Validates:
        - Function returns calculated value even if > 1
        - Does not enforce physical constraints (user error)
        """
        from oscura.analyzers.power.efficiency import efficiency

        # Output > Input (measurement error)
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([15.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        eta = efficiency(v_in, i_in, v_out, i_out)

        # Should return calculated value (1.5), no clamping
        assert eta > 1.0

    def test_efficiency_mismatched_lengths(self) -> None:
        """Test efficiency with different array lengths.

        Validates:
        - Handles different trace lengths
        - Uses shortest common length
        """
        from oscura.analyzers.power.efficiency import efficiency

        v_in = make_waveform_trace(np.array([10.0] * 1000))
        i_in = make_waveform_trace(np.array([10.0] * 1000))
        v_out = make_waveform_trace(np.array([9.0] * 500))  # Shorter
        i_out = make_waveform_trace(np.array([10.0] * 750))  # Different

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert 0 <= eta <= 1


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-003")
class TestPowerConversionEfficiency:
    """Test power conversion efficiency from power values."""

    def test_power_conversion_efficiency_basic(self) -> None:
        """Test basic power conversion efficiency.

        Validates:
        - Correct calculation from power values
        - 90% efficiency case
        """
        from oscura.analyzers.power.efficiency import power_conversion_efficiency

        eta = power_conversion_efficiency(p_in=100.0, p_out=90.0)

        assert eta == 0.9

    def test_power_conversion_efficiency_zero_input(self) -> None:
        """Test with zero input power.

        Validates:
        - Returns 0.0 for zero input
        - No division by zero
        """
        from oscura.analyzers.power.efficiency import power_conversion_efficiency

        eta = power_conversion_efficiency(p_in=0.0, p_out=50.0)

        assert eta == 0.0

    def test_power_conversion_efficiency_negative_input(self) -> None:
        """Test with negative input power.

        Validates:
        - Returns 0.0 for negative input
        """
        from oscura.analyzers.power.efficiency import power_conversion_efficiency

        eta = power_conversion_efficiency(p_in=-100.0, p_out=50.0)

        assert eta == 0.0

    def test_power_conversion_efficiency_perfect(self) -> None:
        """Test perfect efficiency (100%).

        Validates:
        - Returns 1.0 for perfect conversion
        """
        from oscura.analyzers.power.efficiency import power_conversion_efficiency

        eta = power_conversion_efficiency(p_in=100.0, p_out=100.0)

        assert eta == 1.0

    def test_power_conversion_efficiency_zero_output(self) -> None:
        """Test with zero output power.

        Validates:
        - Returns 0.0 for zero output
        - All input is lost
        """
        from oscura.analyzers.power.efficiency import power_conversion_efficiency

        eta = power_conversion_efficiency(p_in=100.0, p_out=0.0)

        assert eta == 0.0

    def test_power_conversion_efficiency_very_low(self) -> None:
        """Test very low efficiency.

        Validates:
        - Correct calculation for 1% efficiency
        """
        from oscura.analyzers.power.efficiency import power_conversion_efficiency

        eta = power_conversion_efficiency(p_in=100.0, p_out=1.0)

        assert abs(eta - 0.01) < 1e-10


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-003")
class TestMultiOutputEfficiency:
    """Test multi-output power supply efficiency."""

    def test_multi_output_efficiency_single_output(self) -> None:
        """Test with single output (simplest case).

        Validates:
        - Correct efficiency calculation
        - All dictionary keys present
        """
        from oscura.analyzers.power.efficiency import multi_output_efficiency

        v_in = make_waveform_trace(np.array([12.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))

        v_out = make_waveform_trace(np.array([5.0] * 100))
        i_out = make_waveform_trace(np.array([20.0] * 100))

        result = multi_output_efficiency(v_in, i_in, [(v_out, i_out)])

        assert "total_efficiency" in result
        assert "output_1_efficiency" in result
        assert "output_1_power" in result
        assert "total_output_power" in result
        assert "input_power" in result
        assert "losses" in result

        # P_in = 120W, P_out = 100W
        assert abs(result["input_power"] - 120.0) < 1e-6
        assert abs(result["output_1_power"] - 100.0) < 1e-6
        assert abs(result["total_output_power"] - 100.0) < 1e-6
        assert abs(result["total_efficiency"] - (100.0 / 120.0)) < 1e-6
        assert abs(result["losses"] - 20.0) < 1e-6

    def test_multi_output_efficiency_three_outputs(self) -> None:
        """Test with three outputs.

        Validates:
        - Per-output calculations
        - Total efficiency is sum of outputs
        - Loss calculation
        """
        from oscura.analyzers.power.efficiency import multi_output_efficiency

        # 120W input
        v_in = make_waveform_trace(np.array([12.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))

        # Three outputs: 40W, 30W, 20W = 90W total
        outputs = [
            (
                make_waveform_trace(np.array([5.0] * 100)),
                make_waveform_trace(np.array([8.0] * 100)),
            ),  # 40W
            (
                make_waveform_trace(np.array([3.3] * 100)),
                make_waveform_trace(np.array([9.09] * 100)),
            ),  # ~30W
            (
                make_waveform_trace(np.array([1.8] * 100)),
                make_waveform_trace(np.array([11.11] * 100)),
            ),  # ~20W
        ]

        result = multi_output_efficiency(v_in, i_in, outputs)

        # Check all outputs present
        assert "output_1_power" in result
        assert "output_2_power" in result
        assert "output_3_power" in result
        assert "output_1_efficiency" in result
        assert "output_2_efficiency" in result
        assert "output_3_efficiency" in result

        # Check values
        assert abs(result["input_power"] - 120.0) < 1e-6
        assert abs(result["output_1_power"] - 40.0) < 1e-6
        assert abs(result["total_output_power"] - 89.997) < 0.1  # Sum of outputs
        assert abs(result["total_efficiency"] - 0.75) < 0.01  # ~75%
        assert result["losses"] > 0

    def test_multi_output_efficiency_zero_input(self) -> None:
        """Test with zero input power.

        Validates:
        - Returns 0.0 efficiency for all outputs
        - No division by zero
        """
        from oscura.analyzers.power.efficiency import multi_output_efficiency

        v_in = make_waveform_trace(np.array([0.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))

        outputs = [
            (
                make_waveform_trace(np.array([5.0] * 100)),
                make_waveform_trace(np.array([10.0] * 100)),
            ),
        ]

        result = multi_output_efficiency(v_in, i_in, outputs)

        assert result["total_efficiency"] == 0.0
        assert result["output_1_efficiency"] == 0.0

    def test_multi_output_efficiency_empty_outputs(self) -> None:
        """Test with no outputs.

        Validates:
        - Handles empty output list
        - Returns 0 efficiency and losses equal to input
        """
        from oscura.analyzers.power.efficiency import multi_output_efficiency

        v_in = make_waveform_trace(np.array([12.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))

        result = multi_output_efficiency(v_in, i_in, [])

        assert result["total_efficiency"] == 0.0
        assert result["total_output_power"] == 0.0
        assert abs(result["losses"] - 120.0) < 1e-6

    def test_multi_output_efficiency_negative_losses(self) -> None:
        """Test when output power exceeds input (measurement error).

        Validates:
        - Negative losses are reported
        - Function doesn't clamp to physical constraints
        """
        from oscura.analyzers.power.efficiency import multi_output_efficiency

        # 100W input
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))

        # 150W output (impossible, measurement error)
        outputs = [
            (
                make_waveform_trace(np.array([15.0] * 100)),
                make_waveform_trace(np.array([10.0] * 100)),
            ),
        ]

        result = multi_output_efficiency(v_in, i_in, outputs)

        assert result["losses"] < 0  # Negative losses
        assert result["total_efficiency"] > 1.0  # >100% (invalid)


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-003")
class TestEfficiencyVsLoad:
    """Test efficiency vs load curve calculation."""

    def test_efficiency_vs_load_basic(self) -> None:
        """Test basic efficiency vs load calculation.

        Validates:
        - Returns correct dictionary keys
        - Arrays have expected shape
        - Load percentages range from 0-100
        """
        from oscura.analyzers.power.efficiency import efficiency_vs_load

        # Constant efficiency converter for simplicity
        t = np.linspace(0, 1, 1000)
        # Varying load
        i_out = 1.0 + 9.0 * t  # 1A to 10A

        v_in = make_waveform_trace(np.array([12.0] * 1000))
        i_in = make_waveform_trace(0.45 + 0.9 * i_out)  # ~90% efficient
        v_out = make_waveform_trace(np.array([5.0] * 1000))
        i_out_trace = make_waveform_trace(i_out)

        result = efficiency_vs_load(v_in, i_in, v_out, i_out_trace, n_points=10)

        assert "load_percent" in result
        assert "efficiency" in result
        assert "output_power" in result
        assert "input_power" in result

        assert len(result["load_percent"]) > 0
        assert len(result["efficiency"]) == len(result["load_percent"])
        assert len(result["output_power"]) == len(result["load_percent"])
        assert len(result["input_power"]) == len(result["load_percent"])

        # Load should range from 0-100%
        assert result["load_percent"].min() >= 0
        assert result["load_percent"].max() <= 100

    def test_efficiency_vs_load_n_points(self) -> None:
        """Test with different number of points.

        Validates:
        - n_points parameter controls output size
        - Works with various n_points values
        """
        from oscura.analyzers.power.efficiency import efficiency_vs_load

        v_in = make_waveform_trace(np.array([12.0] * 1000))
        i_in = make_waveform_trace(np.array([10.0] * 1000))
        v_out = make_waveform_trace(np.array([5.0] * 1000))
        i_out = make_waveform_trace(np.linspace(1, 10, 1000))

        for n in [5, 10, 50, 100]:
            result = efficiency_vs_load(v_in, i_in, v_out, i_out, n_points=n)
            # Length may be <= n due to binning
            assert len(result["load_percent"]) <= n + 1

    def test_efficiency_vs_load_constant_load(self) -> None:
        """Test with constant load (no variation).

        Validates:
        - Handles constant load case
        - Returns single bin
        """
        from oscura.analyzers.power.efficiency import efficiency_vs_load

        v_in = make_waveform_trace(np.array([12.0] * 1000))
        i_in = make_waveform_trace(np.array([10.0] * 1000))
        v_out = make_waveform_trace(np.array([5.0] * 1000))
        i_out = make_waveform_trace(np.array([10.0] * 1000))  # Constant

        result = efficiency_vs_load(v_in, i_in, v_out, i_out, n_points=10)

        # All load values should be similar
        assert len(result["load_percent"]) > 0
        assert np.std(result["load_percent"]) < 10  # Low variance

    def test_efficiency_vs_load_zero_max_power(self) -> None:
        """Test with zero max power.

        Validates:
        - Handles zero max power gracefully
        - Returns zero load percentages
        """
        from oscura.analyzers.power.efficiency import efficiency_vs_load

        v_in = make_waveform_trace(np.array([12.0] * 100))
        i_in = make_waveform_trace(np.array([0.0] * 100))
        v_out = make_waveform_trace(np.array([0.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        result = efficiency_vs_load(v_in, i_in, v_out, i_out, n_points=10)

        # All load should be 0%
        assert all(lp == 0 for lp in result["load_percent"])

    def test_efficiency_vs_load_small_dataset(self) -> None:
        """Test with very small dataset.

        Validates:
        - Handles small arrays (< n_points)
        - Doesn't crash with bin_size=1
        """
        from oscura.analyzers.power.efficiency import efficiency_vs_load

        v_in = make_waveform_trace(np.array([12.0] * 5))
        i_in = make_waveform_trace(np.array([10.0] * 5))
        v_out = make_waveform_trace(np.array([5.0] * 5))
        i_out = make_waveform_trace(np.array([10.0] * 5))

        result = efficiency_vs_load(v_in, i_in, v_out, i_out, n_points=100)

        assert len(result["load_percent"]) > 0


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-003")
class TestLossBreakdown:
    """Test power loss breakdown analysis."""

    def test_loss_breakdown_no_known_losses(self) -> None:
        """Test loss breakdown with no known losses.

        Validates:
        - All losses go to 'other_loss'
        - Correct total loss calculation
        """
        from oscura.analyzers.power.efficiency import loss_breakdown

        # 100W in, 90W out, 10W loss
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([9.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        result = loss_breakdown(v_in, i_in, v_out, i_out)

        assert "total_loss" in result
        assert "other_loss" in result
        assert "efficiency" in result

        assert abs(result["total_loss"] - 10.0) < 1e-6
        assert abs(result["other_loss"] - 10.0) < 1e-6
        assert abs(result["efficiency"] - 0.9) < 1e-10

    def test_loss_breakdown_with_known_losses(self) -> None:
        """Test loss breakdown with known loss components.

        Validates:
        - Known losses are reported correctly
        - Other loss is calculated correctly
        - Percentages are calculated
        """
        from oscura.analyzers.power.efficiency import loss_breakdown

        # 100W in, 90W out, 10W loss
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([9.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        result = loss_breakdown(
            v_in,
            i_in,
            v_out,
            i_out,
            switching_loss=4.0,
            conduction_loss=3.0,
            magnetic_loss=2.0,
            gate_drive_loss=0.5,
        )

        assert abs(result["switching_loss"] - 4.0) < 1e-10
        assert abs(result["conduction_loss"] - 3.0) < 1e-10
        assert abs(result["magnetic_loss"] - 2.0) < 1e-10
        assert abs(result["gate_drive_loss"] - 0.5) < 1e-10
        # 10W total - 9.5W known = 0.5W other
        assert abs(result["other_loss"] - 0.5) < 1e-6

        # Check percentages
        assert abs(result["switching_loss_percent"] - 40.0) < 1e-6
        assert abs(result["conduction_loss_percent"] - 30.0) < 1e-6
        assert abs(result["magnetic_loss_percent"] - 20.0) < 1e-6

    def test_loss_breakdown_known_exceeds_total(self) -> None:
        """Test when known losses exceed total measured loss.

        Validates:
        - Other loss is clamped to 0
        - Does not go negative
        """
        from oscura.analyzers.power.efficiency import loss_breakdown

        # 100W in, 90W out, 10W total loss
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([9.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        # Known losses = 15W > 10W total
        result = loss_breakdown(
            v_in,
            i_in,
            v_out,
            i_out,
            switching_loss=10.0,
            conduction_loss=5.0,
        )

        # Other loss should be clamped to 0
        assert result["other_loss"] == 0.0

    def test_loss_breakdown_zero_loss(self) -> None:
        """Test with perfect converter (zero loss).

        Validates:
        - Handles zero total loss
        - Percentage calculations don't divide by zero
        """
        from oscura.analyzers.power.efficiency import loss_breakdown

        # 100W in, 100W out (perfect)
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([10.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        result = loss_breakdown(v_in, i_in, v_out, i_out, switching_loss=0.0, conduction_loss=0.0)

        assert result["total_loss"] == 0.0
        assert result["switching_loss_percent"] == 0.0
        assert result["conduction_loss_percent"] == 0.0
        assert result["magnetic_loss_percent"] == 0.0

    def test_loss_breakdown_all_fields_present(self) -> None:
        """Test that all expected fields are present in result.

        Validates:
        - Complete result dictionary
        """
        from oscura.analyzers.power.efficiency import loss_breakdown

        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([9.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        result = loss_breakdown(v_in, i_in, v_out, i_out)

        expected_fields = [
            "input_power",
            "output_power",
            "efficiency",
            "total_loss",
            "switching_loss",
            "conduction_loss",
            "magnetic_loss",
            "gate_drive_loss",
            "other_loss",
            "switching_loss_percent",
            "conduction_loss_percent",
            "magnetic_loss_percent",
        ]

        for field in expected_fields:
            assert field in result


@pytest.mark.unit
@pytest.mark.analyzer
@pytest.mark.requirement("PWR-003")
class TestThermalEfficiency:
    """Test thermal efficiency analysis."""

    def test_thermal_efficiency_basic(self) -> None:
        """Test basic thermal efficiency calculation.

        Validates:
        - Correct thermal calculations
        - Temperature rise calculation
        """
        from oscura.analyzers.power.efficiency import thermal_efficiency

        # 100W in, 90W out, 10W loss
        # 40°C rise / 2.5°C/W = 16W thermal
        result = thermal_efficiency(
            p_in=100.0, p_out=90.0, ambient_temp=25.0, case_temp=65.0, thermal_resistance=2.5
        )

        assert "efficiency" in result
        assert "electrical_losses" in result
        assert "thermal_estimated_losses" in result
        assert "temperature_rise" in result
        assert "loss_discrepancy" in result

        assert abs(result["efficiency"] - 0.9) < 1e-10
        assert abs(result["electrical_losses"] - 10.0) < 1e-10
        assert abs(result["thermal_estimated_losses"] - 16.0) < 1e-10
        assert abs(result["temperature_rise"] - 40.0) < 1e-10
        assert abs(result["loss_discrepancy"] - 6.0) < 1e-10

    def test_thermal_efficiency_perfect_match(self) -> None:
        """Test when electrical and thermal losses match.

        Validates:
        - Zero discrepancy when losses match
        """
        from oscura.analyzers.power.efficiency import thermal_efficiency

        # 100W in, 90W out, 10W loss
        # Set thermal to match: 10W * 2.5 = 25°C rise
        result = thermal_efficiency(
            p_in=100.0, p_out=90.0, ambient_temp=25.0, case_temp=50.0, thermal_resistance=2.5
        )

        assert abs(result["loss_discrepancy"]) < 1e-10

    def test_thermal_efficiency_zero_input(self) -> None:
        """Test with zero input power.

        Validates:
        - Returns 0 efficiency
        - Still calculates thermal losses
        """
        from oscura.analyzers.power.efficiency import thermal_efficiency

        result = thermal_efficiency(
            p_in=0.0, p_out=0.0, ambient_temp=25.0, case_temp=40.0, thermal_resistance=2.5
        )

        assert result["efficiency"] == 0.0
        assert result["electrical_losses"] == 0.0
        assert abs(result["thermal_estimated_losses"] - 6.0) < 1e-10

    def test_thermal_efficiency_no_temperature_rise(self) -> None:
        """Test with no temperature rise.

        Validates:
        - Zero thermal losses when case = ambient
        """
        from oscura.analyzers.power.efficiency import thermal_efficiency

        result = thermal_efficiency(
            p_in=100.0, p_out=90.0, ambient_temp=25.0, case_temp=25.0, thermal_resistance=2.5
        )

        assert result["temperature_rise"] == 0.0
        assert result["thermal_estimated_losses"] == 0.0

    def test_thermal_efficiency_high_resistance(self) -> None:
        """Test with high thermal resistance.

        Validates:
        - Correct calculation with large Rth
        """
        from oscura.analyzers.power.efficiency import thermal_efficiency

        # 10°C rise / 100°C/W = 0.1W thermal loss
        result = thermal_efficiency(
            p_in=100.0,
            p_out=90.0,
            ambient_temp=25.0,
            case_temp=35.0,
            thermal_resistance=100.0,
        )

        assert abs(result["thermal_estimated_losses"] - 0.1) < 1e-10

    def test_thermal_efficiency_negative_temp_rise(self) -> None:
        """Test with case temp below ambient (unusual but possible).

        Validates:
        - Handles negative temperature rise
        - Negative thermal losses
        """
        from oscura.analyzers.power.efficiency import thermal_efficiency

        result = thermal_efficiency(
            p_in=100.0, p_out=90.0, ambient_temp=25.0, case_temp=20.0, thermal_resistance=2.5
        )

        assert result["temperature_rise"] < 0
        assert result["thermal_estimated_losses"] < 0


@pytest.mark.unit
@pytest.mark.analyzer
class TestPowerEfficiencyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_numbers(self) -> None:
        """Test with very small power values (mW range).

        Validates:
        - Maintains precision at small scales
        """
        from oscura.analyzers.power.efficiency import efficiency

        # 1mW scale
        v_in = make_waveform_trace(np.array([0.001] * 100))
        i_in = make_waveform_trace(np.array([1.0] * 100))
        v_out = make_waveform_trace(np.array([0.0009] * 100))
        i_out = make_waveform_trace(np.array([1.0] * 100))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert abs(eta - 0.9) < 1e-6

    def test_very_large_numbers(self) -> None:
        """Test with very large power values (MW range).

        Validates:
        - No overflow with large numbers
        """
        from oscura.analyzers.power.efficiency import efficiency

        # MW scale
        v_in = make_waveform_trace(np.array([1000.0] * 100))
        i_in = make_waveform_trace(np.array([1000.0] * 100))
        v_out = make_waveform_trace(np.array([900.0] * 100))
        i_out = make_waveform_trace(np.array([1000.0] * 100))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert abs(eta - 0.9) < 1e-10

    def test_single_sample(self) -> None:
        """Test with single sample traces.

        Validates:
        - Handles minimum array size
        """
        from oscura.analyzers.power.efficiency import efficiency

        v_in = make_waveform_trace(np.array([10.0]))
        i_in = make_waveform_trace(np.array([10.0]))
        v_out = make_waveform_trace(np.array([9.0]))
        i_out = make_waveform_trace(np.array([10.0]))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert abs(eta - 0.9) < 1e-10

    def test_negative_output_power(self) -> None:
        """Test with negative output power (regenerative).

        Validates:
        - Handles negative power correctly
        - Can result in negative efficiency
        """
        from oscura.analyzers.power.efficiency import efficiency

        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([-5.0] * 100))  # Negative
        i_out = make_waveform_trace(np.array([10.0] * 100))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert eta < 0  # Negative efficiency (power flowing backwards)

    def test_efficiency_vs_load_mismatched_lengths(self) -> None:
        """Test efficiency_vs_load with different trace lengths.

        Validates:
        - Handles truncation to common length
        - No crashes with mismatched lengths
        """
        from oscura.analyzers.power.efficiency import efficiency_vs_load

        v_in = make_waveform_trace(np.linspace(12, 12, 1000))
        i_in = make_waveform_trace(np.linspace(8, 12, 1000))
        v_out = make_waveform_trace(np.linspace(5, 5, 500))  # Shorter
        i_out = make_waveform_trace(np.linspace(1, 20, 750))  # Different

        result = efficiency_vs_load(v_in, i_in, v_out, i_out, n_points=10)

        assert len(result["efficiency"]) > 0
        assert all(0 <= e <= 1 or e > 1 for e in result["efficiency"])

    def test_loss_breakdown_negative_input_power(self) -> None:
        """Test loss breakdown with negative input power.

        Validates:
        - Handles negative power gracefully
        - Efficiency set to 0 for negative input
        """
        from oscura.analyzers.power.efficiency import loss_breakdown

        v_in = make_waveform_trace(np.array([-10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([5.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        result = loss_breakdown(v_in, i_in, v_out, i_out)

        assert result["efficiency"] == 0.0

    def test_thermal_efficiency_zero_thermal_resistance(self) -> None:
        """Test thermal efficiency with zero thermal resistance.

        Validates:
        - Raises ZeroDivisionError for zero thermal resistance (edge case)
        """
        from oscura.analyzers.power.efficiency import thermal_efficiency

        # Zero resistance causes division by zero - expected behavior
        with pytest.raises(ZeroDivisionError):
            thermal_efficiency(
                p_in=100.0, p_out=90.0, ambient_temp=25.0, case_temp=40.0, thermal_resistance=0.0
            )

    def test_multi_output_efficiency_five_outputs(self) -> None:
        """Test with five outputs (stress test).

        Validates:
        - Scales to multiple outputs
        - Correct indexing and summation
        """
        from oscura.analyzers.power.efficiency import multi_output_efficiency

        # 100W input
        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))

        # Five equal 16W outputs (80W total)
        outputs = [
            (
                make_waveform_trace(np.array([3.2] * 100)),
                make_waveform_trace(np.array([5.0] * 100)),
            )
            for _ in range(5)
        ]

        result = multi_output_efficiency(v_in, i_in, outputs)

        # Check all five outputs are present
        for i in range(1, 6):
            assert f"output_{i}_power" in result
            assert f"output_{i}_efficiency" in result

        assert abs(result["total_output_power"] - 80.0) < 0.5
        assert abs(result["total_efficiency"] - 0.8) < 0.01

    def test_efficiency_precision_high_efficiency(self) -> None:
        """Test efficiency calculation at very high efficiency (99%+).

        Validates:
        - Precision at high efficiency values
        - Proper handling of small loss percentages
        """
        from oscura.analyzers.power.efficiency import efficiency

        v_in = make_waveform_trace(np.array([10.0] * 1000))
        i_in = make_waveform_trace(np.array([10.0] * 1000))
        v_out = make_waveform_trace(np.array([9.99] * 1000))
        i_out = make_waveform_trace(np.array([10.0] * 1000))

        eta = efficiency(v_in, i_in, v_out, i_out)

        assert 0.99 < eta < 1.0
        assert abs(eta - 0.999) < 0.001  # Within 0.1% of 99.9%

    def test_loss_breakdown_all_losses_accounted(self) -> None:
        """Test that all loss components sum to total loss.

        Validates:
        - Sum of all losses equals total loss
        - No loss of precision
        """
        from oscura.analyzers.power.efficiency import loss_breakdown

        v_in = make_waveform_trace(np.array([10.0] * 100))
        i_in = make_waveform_trace(np.array([10.0] * 100))
        v_out = make_waveform_trace(np.array([8.0] * 100))
        i_out = make_waveform_trace(np.array([10.0] * 100))

        result = loss_breakdown(
            v_in,
            i_in,
            v_out,
            i_out,
            switching_loss=5.0,
            conduction_loss=8.0,
            magnetic_loss=3.0,
            gate_drive_loss=2.0,
        )

        # Sum of all known losses should be less than or equal to total loss
        sum_known = (
            result["switching_loss"]
            + result["conduction_loss"]
            + result["magnetic_loss"]
            + result["gate_drive_loss"]
        )
        assert abs(sum_known + result["other_loss"] - result["total_loss"]) < 1e-6

    def test_efficiency_vs_load_descending_load(self) -> None:
        """Test efficiency_vs_load with descending load pattern.

        Validates:
        - Works with non-monotonic load variations
        - Correct sorting and binning
        """
        from oscura.analyzers.power.efficiency import efficiency_vs_load

        # Descending load
        t = np.linspace(0, 1, 500)
        i_out_desc = 10.0 - 9.0 * t  # 10A down to 1A

        v_in = make_waveform_trace(np.array([12.0] * 500))
        i_in = make_waveform_trace(0.6 + 0.85 * i_out_desc)  # ~85% efficiency
        v_out = make_waveform_trace(np.array([5.0] * 500))
        i_out = make_waveform_trace(i_out_desc)

        result = efficiency_vs_load(v_in, i_in, v_out, i_out, n_points=20)

        # Load should be sorted (internal sorting)
        assert result["load_percent"].min() >= 0
        assert result["load_percent"].max() <= 100
        # Load should be roughly monotonic after sorting
        assert len(result["load_percent"]) > 5
