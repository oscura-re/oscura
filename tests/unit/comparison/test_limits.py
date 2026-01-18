"""Comprehensive unit tests for limit testing functionality.

This module tests limit specification, limit checking, and margin analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.comparison.limits import (
    LimitSpec,
    LimitTestResult,
    check_limits,
    create_limit_spec,
    margin_analysis,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


@pytest.fixture
def sine_trace() -> WaveformTrace:
    """Create a sine wave trace (-1 to 1)."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def constant_trace() -> WaveformTrace:
    """Create a constant trace at 0.5V."""
    data = np.ones(1000) * 0.5
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-002")
class TestLimitSpec:
    """Test limit specification creation and validation."""

    def test_create_with_upper_lower(self) -> None:
        """Test creating limit spec with upper and lower bounds."""
        spec = LimitSpec(upper=1.5, lower=-0.5, name="test_spec", unit="V")

        assert spec.upper == 1.5
        assert spec.lower == -0.5
        assert spec.name == "test_spec"
        assert spec.unit == "V"

    def test_create_upper_only(self) -> None:
        """Test creating limit spec with upper bound only."""
        spec = LimitSpec(upper=1.0)

        assert spec.upper == 1.0
        assert spec.lower is None

    def test_create_lower_only(self) -> None:
        """Test creating limit spec with lower bound only."""
        spec = LimitSpec(lower=-1.0)

        assert spec.upper is None
        assert spec.lower == -1.0

    def test_create_with_guardbands(self) -> None:
        """Test creating limit spec with guardbands."""
        spec = LimitSpec(upper=1.0, lower=0.0, upper_guardband=0.1, lower_guardband=0.1)

        assert spec.upper_guardband == 0.1
        assert spec.lower_guardband == 0.1

    def test_invalid_no_limits_raises(self) -> None:
        """Test that creating spec without limits raises error."""
        with pytest.raises(ValueError, match="At least one"):
            LimitSpec()

    def test_invalid_upper_lower_order_raises(self) -> None:
        """Test that upper < lower raises error."""
        with pytest.raises(ValueError, match="must be >="):
            LimitSpec(upper=0.0, lower=1.0)


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-002")
class TestCreateLimitSpec:
    """Test limit spec creation helper function."""

    def test_create_from_center_absolute_tolerance(self) -> None:
        """Test creating from center with absolute tolerance."""
        spec = create_limit_spec(center=1.0, tolerance=0.1)

        assert spec.upper == 1.1
        assert spec.lower == 0.9

    def test_create_from_center_percentage_tolerance(self) -> None:
        """Test creating from center with percentage tolerance."""
        spec = create_limit_spec(center=1.0, tolerance_pct=10)

        assert spec.upper == 1.1
        assert spec.lower == 0.9

    def test_create_with_guardband(self) -> None:
        """Test creating with guardband percentage."""
        spec = create_limit_spec(upper=1.0, lower=0.0, guardband_pct=10)

        assert spec.upper == 1.0
        assert spec.lower == 0.0
        expected_gb = 1.0 * 0.1  # 10% of range
        assert spec.upper_guardband == expected_gb
        assert spec.lower_guardband == expected_gb

    def test_create_invalid_center_without_tolerance_raises(self) -> None:
        """Test that center without tolerance raises error."""
        with pytest.raises(ValueError, match="requires tolerance"):
            create_limit_spec(center=1.0)

    def test_create_invalid_no_limits_raises(self) -> None:
        """Test that no limits raises error."""
        with pytest.raises(ValueError, match="Must specify limits"):
            create_limit_spec()


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-002")
class TestCheckLimits:
    """Test limit checking functionality."""

    def test_within_limits_pass(self, constant_trace: WaveformTrace) -> None:
        """Test data within limits passes."""
        result = check_limits(constant_trace, upper=1.0, lower=0.0)

        assert result.passed
        assert result.num_violations == 0
        assert result.violation_rate == 0.0
        assert result.max_value == 0.5
        assert result.min_value == 0.5
        assert result.upper_margin == 0.5
        assert result.lower_margin == 0.5

    def test_exceeds_upper_limit_fail(self, constant_trace: WaveformTrace) -> None:
        """Test data exceeding upper limit fails."""
        result = check_limits(constant_trace, upper=0.3, lower=0.0)

        assert not result.passed
        assert result.num_violations == 1000  # All samples exceed
        assert result.violation_rate == 1.0
        assert result.upper_violations is not None
        assert len(result.upper_violations) == 1000

    def test_below_lower_limit_fail(self, constant_trace: WaveformTrace) -> None:
        """Test data below lower limit fails."""
        result = check_limits(constant_trace, upper=1.0, lower=0.7)

        assert not result.passed
        assert result.num_violations == 1000
        assert result.lower_violations is not None
        assert len(result.lower_violations) == 1000

    def test_with_limit_spec_object(self, constant_trace: WaveformTrace) -> None:
        """Test checking with LimitSpec object."""
        spec = LimitSpec(upper=1.0, lower=0.0)
        result = check_limits(constant_trace, limits=spec)

        assert result.passed

    def test_sine_wave_limits(self, sine_trace: WaveformTrace) -> None:
        """Test limit checking on sine wave."""
        result = check_limits(sine_trace, upper=1.5, lower=-1.5)

        assert result.passed
        assert abs(result.max_value - 1.0) < 0.01
        assert abs(result.min_value - (-1.0)) < 0.01

    def test_partial_violations(self, sine_trace: WaveformTrace) -> None:
        """Test detecting partial violations."""
        # Limit that clips positive peaks
        result = check_limits(sine_trace, upper=0.5, lower=-1.5)

        assert not result.passed
        assert 0 < result.num_violations < len(sine_trace.data)
        assert result.violation_rate < 1.0

    def test_numpy_array_input(self) -> None:
        """Test checking limits on numpy array directly."""
        data = np.ones(100) * 0.5
        result = check_limits(data, upper=1.0, lower=0.0)

        assert result.passed

    def test_relative_limits(self) -> None:
        """Test relative limit mode."""
        spec = LimitSpec(upper=0.1, lower=-0.1, mode="relative")
        data = np.ones(100) * 1.0
        result = check_limits(data, limits=spec, reference=1.0)

        # Data at 1.0, limits at 1.0 +/- 0.1
        assert result.passed

    def test_guardband_detection(self, constant_trace: WaveformTrace) -> None:
        """Test guardband warning detection."""
        # constant_trace has data=0.5, so upper_margin = 1.0 - 0.5 = 0.5
        # Set guardband to 0.6 so that 0.5 < 0.6 triggers guardband warning
        spec = LimitSpec(upper=1.0, lower=0.0, upper_guardband=0.6)
        result = check_limits(constant_trace, limits=spec)

        assert result.passed
        assert result.within_guardband  # 0.5 margin is within 0.6 guardband


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-002")
class TestMarginAnalysis:
    """Test margin analysis functionality."""

    def test_margin_calculation_symmetric(self, constant_trace: WaveformTrace) -> None:
        """Test margin calculation with symmetric limits."""
        spec = LimitSpec(upper=1.0, lower=0.0)
        result = margin_analysis(constant_trace, spec)

        assert result.upper_margin == 0.5
        assert result.lower_margin == 0.5
        assert result.min_margin == 0.5
        assert result.critical_limit == "both"  # Equal margins

    def test_margin_calculation_upper_critical(self, constant_trace: WaveformTrace) -> None:
        """Test margin with upper limit critical."""
        spec = LimitSpec(upper=0.6, lower=0.0)
        result = margin_analysis(constant_trace, spec)

        # Use approximate comparison for floating point values
        assert abs(result.upper_margin - 0.1) < 1e-10
        assert abs(result.lower_margin - 0.5) < 1e-10
        assert abs(result.min_margin - 0.1) < 1e-10
        assert result.critical_limit == "upper"

    def test_margin_calculation_lower_critical(self, constant_trace: WaveformTrace) -> None:
        """Test margin with lower limit critical."""
        spec = LimitSpec(upper=1.0, lower=0.4)
        result = margin_analysis(constant_trace, spec)

        # Use approximate comparison for floating point values
        assert abs(result.upper_margin - 0.5) < 1e-10
        assert abs(result.lower_margin - 0.1) < 1e-10
        assert abs(result.min_margin - 0.1) < 1e-10
        assert result.critical_limit == "lower"

    def test_margin_percentage(self, constant_trace: WaveformTrace) -> None:
        """Test margin percentage calculation."""
        spec = LimitSpec(upper=1.0, lower=0.0)
        result = margin_analysis(constant_trace, spec)

        # Margin is 0.5 out of 1.0 range = 50%
        assert result.margin_percentage == 50.0

    def test_margin_status_pass(self, constant_trace: WaveformTrace) -> None:
        """Test margin status is pass with good margin."""
        spec = LimitSpec(upper=1.0, lower=0.0)
        result = margin_analysis(constant_trace, spec, warning_threshold_pct=20.0)

        assert result.margin_status == "pass"
        assert not result.warning

    def test_margin_status_warning(self, constant_trace: WaveformTrace) -> None:
        """Test margin status is warning with tight margin."""
        spec = LimitSpec(upper=0.55, lower=0.45)
        result = margin_analysis(constant_trace, spec, warning_threshold_pct=60.0)

        # Margin is 0.05 out of 0.1 range = 50% < 60% threshold
        assert result.margin_status == "warning"
        assert result.warning

    def test_margin_status_fail(self, constant_trace: WaveformTrace) -> None:
        """Test margin status is fail when limit exceeded."""
        spec = LimitSpec(upper=0.4, lower=0.0)
        result = margin_analysis(constant_trace, spec)

        assert result.margin_status == "fail"
        assert result.min_margin < 0

    def test_margin_with_single_limit_upper(self, constant_trace: WaveformTrace) -> None:
        """Test margin analysis with only upper limit."""
        spec = LimitSpec(upper=1.0)
        result = margin_analysis(constant_trace, spec)

        assert result.upper_margin == 0.5
        assert result.lower_margin is None
        assert result.critical_limit == "upper"

    def test_margin_with_single_limit_lower(self, constant_trace: WaveformTrace) -> None:
        """Test margin analysis with only lower limit."""
        spec = LimitSpec(lower=0.0)
        result = margin_analysis(constant_trace, spec)

        assert result.upper_margin is None
        assert result.lower_margin == 0.5
        assert result.critical_limit == "lower"

    def test_margin_numpy_array_input(self) -> None:
        """Test margin analysis on numpy array."""
        data = np.ones(100) * 0.5
        spec = LimitSpec(upper=1.0, lower=0.0)
        result = margin_analysis(data, spec)

        assert result.margin_percentage == 50.0


@pytest.mark.unit
@pytest.mark.comparison
class TestComparisonLimitsEdgeCases:
    """Test edge cases for limit testing."""

    def test_zero_range_data(self) -> None:
        """Test limit checking with constant data."""
        data = np.zeros(100)
        result = check_limits(data, upper=1.0, lower=-1.0)

        assert result.passed
        assert result.max_value == 0.0
        assert result.min_value == 0.0

    def test_single_sample(self) -> None:
        """Test with single sample."""
        data = np.array([0.5])
        result = check_limits(data, upper=1.0, lower=0.0)

        assert result.passed

    def test_empty_array(self) -> None:
        """Test with empty array."""
        data = np.array([])
        try:
            result = check_limits(data, upper=1.0, lower=0.0)
            # If it succeeds, violation rate should be 0
            assert result.violation_rate == 0.0
        except (ValueError, IndexError):
            # Acceptable to raise on empty data
            pass

    def test_limits_equal(self) -> None:
        """Test with upper == lower (single value limit)."""
        data = np.ones(100) * 1.0
        spec = LimitSpec(upper=1.0, lower=1.0)
        result = check_limits(data, limits=spec)

        # Should pass if data exactly matches limits
        assert result.passed

    def test_extreme_values(self) -> None:
        """Test with extreme values."""
        data = np.array([1e10, -1e10])
        result = check_limits(data, upper=2e10, lower=-2e10)

        assert result.passed

    def test_nan_values(self) -> None:
        """Test handling of NaN values."""
        data = np.array([0.5, np.nan, 0.5])
        try:
            result = check_limits(data, upper=1.0, lower=0.0)
            # NaN comparisons are always False, might appear as violations
            assert isinstance(result, LimitTestResult)
        except (ValueError, FloatingPointError):
            # Acceptable to raise on NaN
            pass
