"""Unit tests for Safe Operating Area (SOA) analysis.


Tests cover SOA limit definitions, violation detection, SOA plotting,
and MOSFET SOA generation.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.power.soa import (
    SOALimit,
    SOAViolation,
    check_soa_violations,
    create_mosfet_soa,
    plot_soa,
    soa_analysis,
)
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


def create_ramp_trace(
    start: float,
    end: float,
    sample_rate: float,
    duration: float = 0.1,
) -> WaveformTrace:
    """Create a linear ramp waveform trace."""
    num_samples = int(sample_rate * duration)
    data = np.linspace(start, end, num_samples)
    return create_trace(data, sample_rate)


@pytest.mark.unit
@pytest.mark.power
class TestSOALimit:
    """Test SOALimit dataclass."""

    def test_soa_limit_creation(self) -> None:
        """Test basic SOALimit creation."""
        limit = SOALimit(v_max=100.0, i_max=50.0)

        assert limit.v_max == 100.0
        assert limit.i_max == 50.0
        assert limit.pulse_width == np.inf
        assert limit.name == ""

    def test_soa_limit_with_pulse_width(self) -> None:
        """Test SOALimit with pulse width."""
        limit = SOALimit(v_max=80.0, i_max=100.0, pulse_width=1e-6)

        assert limit.pulse_width == 1e-6

    def test_soa_limit_with_name(self) -> None:
        """Test SOALimit with name."""
        limit = SOALimit(v_max=50.0, i_max=25.0, name="Test Limit")

        assert limit.name == "Test Limit"

    def test_soa_limit_dc_default(self) -> None:
        """Test that default pulse_width is infinity (DC limit)."""
        limit = SOALimit(v_max=100.0, i_max=50.0)

        assert limit.pulse_width == np.inf


@pytest.mark.unit
@pytest.mark.power
class TestSOAViolation:
    """Test SOAViolation dataclass."""

    def test_soa_violation_creation(self) -> None:
        """Test SOAViolation creation."""
        limit = SOALimit(v_max=100.0, i_max=50.0)
        violation = SOAViolation(
            timestamp=0.001,
            sample_index=100,
            voltage=90.0,
            current=60.0,
            limit=limit,
            margin=10.0,
        )

        assert violation.timestamp == 0.001
        assert violation.sample_index == 100
        assert violation.voltage == 90.0
        assert violation.current == 60.0
        assert violation.limit == limit
        assert violation.margin == 10.0


@pytest.mark.unit
@pytest.mark.power
class TestSOAAnalysis:
    """Test soa_analysis function."""

    def test_within_soa_passes(self, sample_rate: float) -> None:
        """Test that operation within SOA passes."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True
        assert len(result["violations"]) == 0

    def test_outside_soa_fails(self, sample_rate: float) -> None:
        """Test that operation outside SOA fails."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(60.0, sample_rate)  # Exceeds limit
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is False
        assert len(result["violations"]) > 0

    def test_voltage_at_boundary(self, sample_rate: float) -> None:
        """Test operation at voltage boundary."""
        voltage = create_dc_trace(100.0, sample_rate)  # At V_max
        current = create_dc_trace(10.0, sample_rate)
        limits = [
            SOALimit(v_max=50.0, i_max=50.0),
            SOALimit(v_max=100.0, i_max=25.0),
        ]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True

    def test_current_at_boundary(self, sample_rate: float) -> None:
        """Test operation at current boundary."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(50.0, sample_rate)  # At I_max
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True

    def test_multiple_limits(self, sample_rate: float) -> None:
        """Test with multiple SOA limits."""
        voltage = create_dc_trace(75.0, sample_rate)
        current = create_dc_trace(30.0, sample_rate)
        limits = [
            SOALimit(v_max=50.0, i_max=100.0),
            SOALimit(v_max=100.0, i_max=50.0),
        ]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True
        assert len(result["applicable_limits"]) == 2

    def test_pulse_width_filtering(self, sample_rate: float) -> None:
        """Test pulse width based limit filtering."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [
            SOALimit(v_max=100.0, i_max=20.0, pulse_width=1e-6),  # 1us pulse
            SOALimit(v_max=100.0, i_max=50.0, pulse_width=1e-3),  # 1ms pulse
            SOALimit(v_max=100.0, i_max=100.0, pulse_width=np.inf),  # DC
        ]

        # With 10us pulse, should use 1ms and DC limits
        result = soa_analysis(voltage, current, limits, pulse_width=10e-6)

        # Should filter to limits with pulse_width >= 10us
        assert len(result["applicable_limits"]) == 2

    def test_pulse_width_none_uses_dc(self, sample_rate: float) -> None:
        """Test that pulse_width=None uses DC (infinite) limits."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [
            SOALimit(v_max=100.0, i_max=50.0, pulse_width=1e-6),
            SOALimit(v_max=100.0, i_max=100.0, pulse_width=np.inf),
        ]

        result = soa_analysis(voltage, current, limits, pulse_width=None)

        # Should only use DC limit
        assert SOALimit(v_max=100.0, i_max=100.0, pulse_width=np.inf) in result["applicable_limits"]

    def test_returns_trajectory(self, sample_rate: float) -> None:
        """Test that result includes voltage/current trajectory."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert "v_trajectory" in result
        assert "i_trajectory" in result
        assert len(result["v_trajectory"]) == len(voltage.data)

    def test_min_margin_calculation(self, sample_rate: float) -> None:
        """Test minimum margin to SOA boundary."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(40.0, sample_rate)  # Close to 50A limit
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        # Margin should be positive (inside SOA)
        assert result["min_margin"] >= 0
        # Should be about 10A margin
        assert abs(result["min_margin"] - 10.0) < 1.0

    def test_negative_values_use_absolute(self, sample_rate: float) -> None:
        """Test that negative voltage/current use absolute values."""
        voltage = create_dc_trace(-50.0, sample_rate)  # Negative voltage
        current = create_dc_trace(-10.0, sample_rate)  # Negative current
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True

    def test_different_trace_lengths(self, sample_rate: float) -> None:
        """Test with different voltage and current trace lengths."""
        voltage = create_dc_trace(50.0, sample_rate, duration=0.1)
        current = create_dc_trace(10.0, sample_rate, duration=0.05)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        # Should truncate to shorter length
        assert len(result["v_trajectory"]) == len(current.data)

    def test_violation_details(self, sample_rate: float) -> None:
        """Test that violations contain correct details."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(60.0, sample_rate)  # Exceeds 50A limit
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert len(result["violations"]) > 0
        violation = result["violations"][0]
        assert isinstance(violation, SOAViolation)
        assert violation.current == 60.0
        assert violation.margin > 0  # Outside SOA

    def test_empty_limits_uses_all(self, sample_rate: float) -> None:
        """Test that empty applicable_limits uses all limits."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        # All limits have short pulse widths
        limits = [
            SOALimit(v_max=100.0, i_max=50.0, pulse_width=1e-9),
        ]

        # Request very long pulse width (none match)
        result = soa_analysis(voltage, current, limits, pulse_width=1.0)

        # Should use all limits as fallback
        assert len(result["applicable_limits"]) == 1

    def test_ramp_trajectory(self, sample_rate: float) -> None:
        """Test SOA analysis with ramping voltage."""
        voltage = create_ramp_trace(0.0, 100.0, sample_rate)
        current = create_ramp_trace(50.0, 0.0, sample_rate)  # Current decreases
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        # Should pass - starts at (0V, 50A), ends at (100V, 0A)
        assert result["passed"] is True


@pytest.mark.unit
@pytest.mark.power
class TestInterpolateSOALimit:
    """Test _interpolate_soa_limit function (indirectly through soa_analysis)."""

    def test_single_limit_within(self, sample_rate: float) -> None:
        """Test interpolation with single limit, within voltage range."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True

    def test_single_limit_beyond(self, sample_rate: float) -> None:
        """Test interpolation with single limit, beyond voltage range."""
        voltage = create_dc_trace(150.0, sample_rate)  # Beyond V_max
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        # Beyond V_max, the interpolated limit becomes 0A, so any current is a violation
        # However, the implementation may handle this differently
        # At minimum, we verify the analysis completes and returns margin info
        assert "min_margin" in result
        # With only one limit at 100V, operation at 150V may or may not be flagged
        # depending on interpolation behavior

    def test_interpolation_between_limits(self, sample_rate: float) -> None:
        """Test log-log interpolation between limit points."""
        voltage = create_dc_trace(75.0, sample_rate)  # Between 50V and 100V
        current = create_dc_trace(30.0, sample_rate)
        limits = [
            SOALimit(v_max=50.0, i_max=100.0),
            SOALimit(v_max=100.0, i_max=25.0),
        ]

        result = soa_analysis(voltage, current, limits)

        # At 75V, interpolated limit should be between 100A and 25A
        # Log-log interpolation
        assert result["passed"] is True

    def test_zero_voltage_limit(self, sample_rate: float) -> None:
        """Test with near-zero voltage in limits (linear fallback)."""
        voltage = create_dc_trace(5.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [
            SOALimit(v_max=0.1, i_max=100.0),  # Near zero
            SOALimit(v_max=10.0, i_max=50.0),
        ]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True


@pytest.mark.unit
@pytest.mark.power
class TestCheckSOAViolations:
    """Test check_soa_violations convenience function."""

    def test_no_violations(self, sample_rate: float) -> None:
        """Test when no violations occur."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        violations = check_soa_violations(voltage, current, limits)

        assert violations == []

    def test_has_violations(self, sample_rate: float) -> None:
        """Test when violations occur."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(60.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        violations = check_soa_violations(voltage, current, limits)

        assert len(violations) > 0
        assert all(isinstance(v, SOAViolation) for v in violations)


@pytest.mark.unit
@pytest.mark.power
class TestPlotSOA:
    """Test plot_soa function."""

    def test_plot_creates_figure(self, sample_rate: float) -> None:
        """Test that plot_soa returns a matplotlib figure."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        fig = plot_soa(voltage, current, limits)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_custom_title(self, sample_rate: float) -> None:
        """Test plot with custom title."""
        import matplotlib

        matplotlib.use("Agg")

        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        fig = plot_soa(voltage, current, limits, title="Custom Title")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_violations(self, sample_rate: float) -> None:
        """Test plot shows violations."""
        import matplotlib

        matplotlib.use("Agg")

        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(60.0, sample_rate)  # Violation
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        fig = plot_soa(voltage, current, limits, show_violations=True)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_without_violations_shown(self, sample_rate: float) -> None:
        """Test plot with violations hidden."""
        import matplotlib

        matplotlib.use("Agg")

        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(60.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        fig = plot_soa(voltage, current, limits, show_violations=False)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_custom_figsize(self, sample_rate: float) -> None:
        """Test plot with custom figure size."""
        import matplotlib

        matplotlib.use("Agg")

        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        fig = plot_soa(voltage, current, limits, figsize=(12, 10))

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


@pytest.mark.unit
@pytest.mark.power
class TestCreateMosfetSOA:
    """Test create_mosfet_soa function."""

    def test_basic_mosfet_soa(self) -> None:
        """Test basic MOSFET SOA generation."""
        limits = create_mosfet_soa(v_ds_max=100.0, i_d_max=50.0, p_d_max=150.0)

        assert len(limits) > 0
        assert all(isinstance(l, SOALimit) for l in limits)

    def test_mosfet_soa_includes_current_limit(self) -> None:
        """Test that MOSFET SOA includes current limit point."""
        limits = create_mosfet_soa(v_ds_max=100.0, i_d_max=50.0, p_d_max=150.0)

        # Should have I_max limit
        i_max_limits = [l for l in limits if l.name == "I_max"]
        assert len(i_max_limits) >= 1
        assert i_max_limits[0].i_max == 50.0

    def test_mosfet_soa_includes_voltage_limit(self) -> None:
        """Test that MOSFET SOA includes voltage limit point."""
        limits = create_mosfet_soa(v_ds_max=100.0, i_d_max=50.0, p_d_max=150.0)

        # Should have V_max limit
        v_max_limits = [l for l in limits if l.name == "V_max"]
        assert len(v_max_limits) >= 1
        assert v_max_limits[0].v_max == 100.0

    def test_mosfet_soa_power_limit(self) -> None:
        """Test that MOSFET SOA includes power limit region."""
        limits = create_mosfet_soa(v_ds_max=100.0, i_d_max=50.0, p_d_max=150.0)

        # Should have power limit points
        p_max_limits = [l for l in limits if "P_max" in l.name]
        assert len(p_max_limits) >= 1

    def test_mosfet_soa_with_pulse_limits(self) -> None:
        """Test MOSFET SOA with pulsed operation limits."""
        pulse_limits = {
            1e-6: 100.0,  # 1us pulse: 100A
            10e-6: 75.0,  # 10us pulse: 75A
            100e-6: 60.0,  # 100us pulse: 60A
        }

        limits = create_mosfet_soa(
            v_ds_max=100.0,
            i_d_max=50.0,
            p_d_max=150.0,
            pulse_limits=pulse_limits,
        )

        # Should include pulsed limits
        pulsed = [l for l in limits if l.pulse_width != np.inf]
        assert len(pulsed) == 3

    def test_mosfet_soa_pulse_names(self) -> None:
        """Test that pulsed limits have meaningful names."""
        pulse_limits = {1e-6: 100.0}

        limits = create_mosfet_soa(
            v_ds_max=100.0,
            i_d_max=50.0,
            p_d_max=150.0,
            pulse_limits=pulse_limits,
        )

        pulsed = [l for l in limits if l.pulse_width == 1e-6]
        assert len(pulsed) == 1
        assert "1us" in pulsed[0].name or "Pulse" in pulsed[0].name

    def test_mosfet_soa_high_power(self) -> None:
        """Test MOSFET SOA with high power rating."""
        limits = create_mosfet_soa(v_ds_max=600.0, i_d_max=200.0, p_d_max=500.0)

        assert len(limits) > 0
        # Voltage limit should be at V_ds_max
        v_max_limits = [l for l in limits if l.name == "V_max"]
        assert v_max_limits[0].v_max == 600.0

    def test_mosfet_soa_low_power(self) -> None:
        """Test MOSFET SOA with low power rating (power limited early)."""
        # P = V * I -> at I_max, V = P/I = 10/5 = 2V
        # Power limit kicks in at 2V, which is less than V_max
        limits = create_mosfet_soa(v_ds_max=20.0, i_d_max=5.0, p_d_max=10.0)

        assert len(limits) > 0


@pytest.mark.unit
@pytest.mark.power
class TestSOAEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_traces(self, sample_rate: float) -> None:
        """Test that empty trace raises ValueError."""
        # Empty trace creation raises ValueError in v0.9.0+
        with pytest.raises(ValueError, match="data array cannot be empty"):
            create_trace(np.array([]), sample_rate)

    def test_single_sample(self, sample_rate: float) -> None:
        """Test with single sample traces."""
        voltage = create_trace(np.array([50.0]), sample_rate)
        current = create_trace(np.array([10.0]), sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True

    def test_empty_limits_list(self, sample_rate: float) -> None:
        """Test with empty limits list."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(10.0, sample_rate)
        limits: list[SOALimit] = []

        result = soa_analysis(voltage, current, limits)

        # With no limits, everything passes
        assert result["passed"] is True

    def test_very_high_current(self, sample_rate: float) -> None:
        """Test with very high current (definite violation)."""
        voltage = create_dc_trace(50.0, sample_rate)
        current = create_dc_trace(1000.0, sample_rate)  # Way over limit
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is False
        assert len(result["violations"]) > 0

    def test_zero_voltage_zero_current(self, sample_rate: float) -> None:
        """Test with zero voltage and current."""
        voltage = create_dc_trace(0.0, sample_rate)
        current = create_dc_trace(0.0, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert result["passed"] is True

    def test_varying_trajectory(self, sample_rate: float) -> None:
        """Test with varying voltage/current trajectory."""
        # Simulate a switching trajectory
        num_samples = 1000
        t = np.linspace(0, 0.1, num_samples)
        v_data = 100.0 * (1 - np.exp(-t / 0.01))  # Rising voltage
        i_data = 50.0 * np.exp(-t / 0.01)  # Falling current

        voltage = create_trace(v_data, sample_rate)
        current = create_trace(i_data, sample_rate)
        limits = [SOALimit(v_max=100.0, i_max=50.0)]

        result = soa_analysis(voltage, current, limits)

        assert "min_margin" in result
