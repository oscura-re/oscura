"""Comprehensive unit tests for arithmetic module.

This module tests all arithmetic operations defined in oscura.math.arithmetic:
- Trace-to-trace operations (add, subtract, multiply, divide)
- Trace-to-scalar operations
- Trace-to-array operations
- Convenience functions (scale, offset, invert, absolute)
- Calculus operations (differentiate, integrate)
- Mathematical expressions
- Edge cases (zeros, infinities, NaN, overflow)
- Array vectorization


Coverage target: >90% branch coverage
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.exceptions import AnalysisError, InsufficientDataError
from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.math.arithmetic import (
    absolute,
    add,
    differentiate,
    divide,
    integrate,
    invert,
    math_expression,
    multiply,
    offset,
    scale,
    subtract,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_rate() -> float:
    """Default sample rate for test traces (1 MHz)."""
    return 1_000_000.0


@pytest.fixture
def simple_metadata(sample_rate: float) -> TraceMetadata:
    """Create simple metadata for testing."""
    return TraceMetadata(
        sample_rate=sample_rate,
        vertical_scale=1.0,
        vertical_offset=0.0,
        channel_name="test_channel",
    )


@pytest.fixture
def simple_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a simple trace with values [1.0, 2.0, 3.0, 4.0, 5.0]."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    return WaveformTrace(data=data, metadata=simple_metadata)


@pytest.fixture
def second_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a second trace with values [2.0, 4.0, 6.0, 8.0, 10.0]."""
    data = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64)
    return WaveformTrace(data=data, metadata=simple_metadata)


@pytest.fixture
def zero_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a trace with all zeros."""
    data = np.zeros(5, dtype=np.float64)
    return WaveformTrace(data=data, metadata=simple_metadata)


@pytest.fixture
def sine_trace(sample_rate: float) -> WaveformTrace:
    """Create a sine wave trace for calculus operations."""
    duration = 0.001  # 1 ms
    t = np.arange(0, duration, 1 / sample_rate)
    frequency = 1000.0  # 1 kHz
    data = np.sin(2 * np.pi * frequency * t)
    metadata = TraceMetadata(sample_rate=sample_rate)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def ramp_trace(simple_metadata: TraceMetadata) -> WaveformTrace:
    """Create a linear ramp trace for derivative testing."""
    data = np.linspace(0.0, 100.0, 101)
    return WaveformTrace(data=data, metadata=simple_metadata)


# =============================================================================
# Addition Tests (ARITH-001)
# =============================================================================


@pytest.mark.unit
class TestAdd:
    """Test add() function."""

    def test_add_two_traces(self, simple_trace: WaveformTrace, second_trace: WaveformTrace):
        """Test adding two traces with same sample rate and length."""
        result = add(simple_trace, second_trace)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([3.0, 6.0, 9.0, 12.0, 15.0]),
        )
        assert result.metadata.sample_rate == simple_trace.metadata.sample_rate
        assert "sum" in result.metadata.channel_name

    def test_add_scalar_to_trace(self, simple_trace: WaveformTrace):
        """Test adding a scalar to trace."""
        result = add(simple_trace, 10.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([11.0, 12.0, 13.0, 14.0, 15.0]),
        )

    def test_add_negative_scalar(self, simple_trace: WaveformTrace):
        """Test adding a negative scalar."""
        result = add(simple_trace, -5.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([-4.0, -3.0, -2.0, -1.0, 0.0]),
        )

    def test_add_zero_scalar(self, simple_trace: WaveformTrace):
        """Test adding zero (should be identity)."""
        result = add(simple_trace, 0.0)

        np.testing.assert_array_almost_equal(result.data, simple_trace.data)

    def test_add_integer_scalar(self, simple_trace: WaveformTrace):
        """Test adding integer scalar (should be converted to float)."""
        result = add(simple_trace, 5)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([6.0, 7.0, 8.0, 9.0, 10.0]),
        )

    def test_add_array_to_trace(self, simple_trace: WaveformTrace):
        """Test adding numpy array to trace."""
        arr = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        result = add(simple_trace, arr)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([1.5, 3.0, 4.5, 6.0, 7.5]),
        )

    def test_add_array_length_mismatch(self, simple_trace: WaveformTrace):
        """Test adding array with wrong length raises error."""
        arr = np.array([1.0, 2.0, 3.0])  # Wrong length

        with pytest.raises(AnalysisError, match="Array length must match"):
            add(simple_trace, arr)

    def test_add_traces_different_lengths(
        self, simple_trace: WaveformTrace, simple_metadata: TraceMetadata
    ):
        """Test adding traces with different lengths (should truncate)."""
        long_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        long_trace = WaveformTrace(data=long_data, metadata=simple_metadata)

        result = add(simple_trace, long_trace)

        # Should truncate to shorter length (5)
        assert len(result.data) == 5
        np.testing.assert_array_almost_equal(
            result.data,
            np.array([2.0, 4.0, 6.0, 8.0, 10.0]),
        )

    def test_add_traces_incompatible_sample_rates(
        self, simple_trace: WaveformTrace, simple_metadata: TraceMetadata
    ):
        """Test adding traces with different sample rates raises error."""
        different_metadata = TraceMetadata(
            sample_rate=simple_metadata.sample_rate * 2.0,  # Double the rate
            channel_name="different",
        )
        different_trace = WaveformTrace(data=simple_trace.data.copy(), metadata=different_metadata)

        with pytest.raises(AnalysisError, match="Sample rates must match"):
            add(simple_trace, different_trace)

    def test_add_custom_channel_name(
        self, simple_trace: WaveformTrace, second_trace: WaveformTrace
    ):
        """Test custom channel name."""
        result = add(simple_trace, second_trace, channel_name="custom_sum")

        assert result.metadata.channel_name == "custom_sum"

    def test_add_preserves_metadata(self, simple_trace: WaveformTrace, second_trace: WaveformTrace):
        """Test that metadata is preserved from first trace."""
        result = add(simple_trace, second_trace)

        assert result.metadata.vertical_scale == simple_trace.metadata.vertical_scale
        assert result.metadata.vertical_offset == simple_trace.metadata.vertical_offset


# =============================================================================
# Subtraction Tests (ARITH-002)
# =============================================================================


@pytest.mark.unit
class TestSubtract:
    """Test subtract() function."""

    def test_subtract_two_traces(self, simple_trace: WaveformTrace, second_trace: WaveformTrace):
        """Test subtracting two traces."""
        result = subtract(simple_trace, second_trace)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([-1.0, -2.0, -3.0, -4.0, -5.0]),
        )
        assert "diff" in result.metadata.channel_name

    def test_subtract_scalar_from_trace(self, simple_trace: WaveformTrace):
        """Test subtracting a scalar from trace."""
        result = subtract(simple_trace, 2.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([-1.0, 0.0, 1.0, 2.0, 3.0]),
        )

    def test_subtract_array_from_trace(self, simple_trace: WaveformTrace):
        """Test subtracting numpy array from trace."""
        arr = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        result = subtract(simple_trace, arr)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        )

    def test_subtract_array_length_mismatch(self, simple_trace: WaveformTrace):
        """Test subtracting array with wrong length raises error."""
        arr = np.array([1.0, 2.0])  # Wrong length

        with pytest.raises(AnalysisError, match="Array length must match"):
            subtract(simple_trace, arr)

    def test_subtract_trace_from_itself(self, simple_trace: WaveformTrace):
        """Test subtracting trace from itself (should be zero)."""
        result = subtract(simple_trace, simple_trace)

        np.testing.assert_array_almost_equal(result.data, np.zeros(5))

    def test_subtract_dc_offset(self, simple_trace: WaveformTrace):
        """Test removing DC offset (common use case)."""
        dc_offset = np.mean(simple_trace.data)
        result = subtract(simple_trace, dc_offset)

        # Result should have zero mean (within numerical precision)
        assert abs(np.mean(result.data)) < 1e-10


# =============================================================================
# Multiplication Tests (ARITH-003)
# =============================================================================


@pytest.mark.unit
class TestMultiply:
    """Test multiply() function."""

    def test_multiply_two_traces(self, simple_trace: WaveformTrace, second_trace: WaveformTrace):
        """Test multiplying two traces."""
        result = multiply(simple_trace, second_trace)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([2.0, 8.0, 18.0, 32.0, 50.0]),
        )
        assert "mult" in result.metadata.channel_name

    def test_multiply_trace_by_scalar(self, simple_trace: WaveformTrace):
        """Test multiplying trace by scalar."""
        result = multiply(simple_trace, 2.5)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([2.5, 5.0, 7.5, 10.0, 12.5]),
        )

    def test_multiply_by_zero(self, simple_trace: WaveformTrace):
        """Test multiplying by zero."""
        result = multiply(simple_trace, 0.0)

        np.testing.assert_array_almost_equal(result.data, np.zeros(5))

    def test_multiply_by_one(self, simple_trace: WaveformTrace):
        """Test multiplying by one (identity)."""
        result = multiply(simple_trace, 1.0)

        np.testing.assert_array_almost_equal(result.data, simple_trace.data)

    def test_multiply_by_negative(self, simple_trace: WaveformTrace):
        """Test multiplying by negative scalar."""
        result = multiply(simple_trace, -2.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([-2.0, -4.0, -6.0, -8.0, -10.0]),
        )

    def test_multiply_array_elementwise(self, simple_trace: WaveformTrace):
        """Test element-wise multiplication with array."""
        arr = np.array([1.0, 0.5, 2.0, 0.25, 4.0])
        result = multiply(simple_trace, arr)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([1.0, 1.0, 6.0, 1.0, 20.0]),
        )

    def test_multiply_array_length_mismatch(self, simple_trace: WaveformTrace):
        """Test multiplying by array with wrong length raises error."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])  # Wrong length

        with pytest.raises(AnalysisError, match="Array length must match"):
            multiply(simple_trace, arr)


# =============================================================================
# Division Tests (ARITH-004)
# =============================================================================


@pytest.mark.unit
class TestDivide:
    """Test divide() function."""

    def test_divide_two_traces(self, second_trace: WaveformTrace, simple_trace: WaveformTrace):
        """Test dividing two traces."""
        result = divide(second_trace, simple_trace)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        )
        assert "div" in result.metadata.channel_name

    def test_divide_trace_by_scalar(self, simple_trace: WaveformTrace):
        """Test dividing trace by scalar."""
        result = divide(simple_trace, 2.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([0.5, 1.0, 1.5, 2.0, 2.5]),
        )

    def test_divide_by_zero_scalar(self, simple_trace: WaveformTrace):
        """Test dividing by zero scalar (should fill with NaN)."""
        result = divide(simple_trace, 0.0)

        assert np.all(np.isnan(result.data))

    def test_divide_by_zero_scalar_custom_fill(self, simple_trace: WaveformTrace):
        """Test dividing by zero with custom fill value."""
        result = divide(simple_trace, 0.0, fill_value=999.0)

        np.testing.assert_array_almost_equal(result.data, np.full(5, 999.0))

    def test_divide_by_trace_with_zeros(
        self, simple_trace: WaveformTrace, simple_metadata: TraceMetadata
    ):
        """Test dividing by trace with zero values."""
        divisor_data = np.array([1.0, 2.0, 0.0, 4.0, 0.0])
        divisor_trace = WaveformTrace(data=divisor_data, metadata=simple_metadata)

        result = divide(simple_trace, divisor_trace)

        # Check non-zero divisions
        assert result.data[0] == 1.0  # 1/1
        assert result.data[1] == 1.0  # 2/2
        assert result.data[3] == 1.0  # 4/4

        # Check zero divisions (should be NaN by default)
        assert np.isnan(result.data[2])
        assert np.isnan(result.data[4])

    def test_divide_by_trace_with_zeros_custom_fill(
        self, simple_trace: WaveformTrace, simple_metadata: TraceMetadata
    ):
        """Test dividing by trace with zeros using custom fill value."""
        divisor_data = np.array([1.0, 0.0, 3.0, 0.0, 5.0])
        divisor_trace = WaveformTrace(data=divisor_data, metadata=simple_metadata)

        result = divide(simple_trace, divisor_trace, fill_value=0.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([1.0, 0.0, 1.0, 0.0, 1.0]),
        )

    def test_divide_array_with_zeros(self, simple_trace: WaveformTrace):
        """Test dividing by array containing zeros."""
        arr = np.array([2.0, 0.0, 3.0, 4.0, 0.0])
        result = divide(simple_trace, arr)

        assert result.data[0] == 0.5  # 1/2
        assert np.isnan(result.data[1])  # 2/0
        assert result.data[2] == 1.0  # 3/3
        assert result.data[3] == 1.0  # 4/4
        assert np.isnan(result.data[4])  # 5/0

    def test_divide_array_length_mismatch(self, simple_trace: WaveformTrace):
        """Test dividing by array with wrong length raises error."""
        arr = np.array([1.0, 2.0, 3.0])  # Wrong length

        with pytest.raises(AnalysisError, match="Array length must match"):
            divide(simple_trace, arr)

    def test_divide_trace_by_itself(self, simple_trace: WaveformTrace):
        """Test dividing trace by itself (should be all ones)."""
        result = divide(simple_trace, simple_trace)

        np.testing.assert_array_almost_equal(result.data, np.ones(5))


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
class TestScale:
    """Test scale() convenience function."""

    def test_scale_positive_factor(self, simple_trace: WaveformTrace):
        """Test scaling by positive factor."""
        result = scale(simple_trace, 3.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([3.0, 6.0, 9.0, 12.0, 15.0]),
        )
        assert "scaled" in result.metadata.channel_name

    def test_scale_negative_factor(self, simple_trace: WaveformTrace):
        """Test scaling by negative factor."""
        result = scale(simple_trace, -1.5)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([-1.5, -3.0, -4.5, -6.0, -7.5]),
        )

    def test_scale_by_zero(self, simple_trace: WaveformTrace):
        """Test scaling by zero."""
        result = scale(simple_trace, 0.0)

        np.testing.assert_array_almost_equal(result.data, np.zeros(5))


@pytest.mark.unit
class TestOffset:
    """Test offset() convenience function."""

    def test_offset_positive(self, simple_trace: WaveformTrace):
        """Test adding positive offset."""
        result = offset(simple_trace, 10.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([11.0, 12.0, 13.0, 14.0, 15.0]),
        )
        assert "offset" in result.metadata.channel_name

    def test_offset_negative(self, simple_trace: WaveformTrace):
        """Test adding negative offset."""
        result = offset(simple_trace, -3.0)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0]),
        )


@pytest.mark.unit
class TestInvert:
    """Test invert() function."""

    def test_invert_trace(self, simple_trace: WaveformTrace):
        """Test inverting trace polarity."""
        result = invert(simple_trace)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([-1.0, -2.0, -3.0, -4.0, -5.0]),
        )
        assert "inverted" in result.metadata.channel_name

    def test_invert_twice_identity(self, simple_trace: WaveformTrace):
        """Test that inverting twice returns original."""
        result = invert(invert(simple_trace))

        np.testing.assert_array_almost_equal(result.data, simple_trace.data)


@pytest.mark.unit
class TestAbsolute:
    """Test absolute() function."""

    def test_absolute_positive_values(self, simple_trace: WaveformTrace):
        """Test absolute of positive values (should be unchanged)."""
        result = absolute(simple_trace)

        np.testing.assert_array_almost_equal(result.data, simple_trace.data)
        assert "abs" in result.metadata.channel_name

    def test_absolute_negative_values(self, simple_metadata: TraceMetadata):
        """Test absolute of negative values."""
        negative_data = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
        negative_trace = WaveformTrace(data=negative_data, metadata=simple_metadata)

        result = absolute(negative_trace)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        )

    def test_absolute_mixed_values(self, simple_metadata: TraceMetadata):
        """Test absolute of mixed positive and negative values."""
        mixed_data = np.array([-2.0, 3.0, -5.0, 1.0, -4.0])
        mixed_trace = WaveformTrace(data=mixed_data, metadata=simple_metadata)

        result = absolute(mixed_trace)

        np.testing.assert_array_almost_equal(
            result.data,
            np.array([2.0, 3.0, 5.0, 1.0, 4.0]),
        )

    def test_absolute_zero(self, zero_trace: WaveformTrace):
        """Test absolute of zeros."""
        result = absolute(zero_trace)

        np.testing.assert_array_almost_equal(result.data, np.zeros(5))


# =============================================================================
# Differentiation Tests (ARITH-005)
# =============================================================================


@pytest.mark.unit
class TestDifferentiate:
    """Test differentiate() function."""

    def test_differentiate_constant(self, simple_metadata: TraceMetadata):
        """Test derivative of constant (should be zero)."""
        constant_data = np.full(100, 5.0)
        constant_trace = WaveformTrace(data=constant_data, metadata=simple_metadata)

        result = differentiate(constant_trace)

        # Derivative of constant should be zero
        np.testing.assert_array_almost_equal(result.data, np.zeros(100), decimal=10)

    def test_differentiate_linear_ramp(self, ramp_trace: WaveformTrace):
        """Test derivative of linear ramp (should be constant)."""
        result = differentiate(ramp_trace)

        # Derivative of linear ramp should be constant (slope)
        # The ramp goes from 0 to 100 over 101 points with dt = 1/sample_rate
        # Slope = 100 / (100 * dt) = 100 * sample_rate / 100 = sample_rate
        expected_slope = 1.0 * ramp_trace.metadata.sample_rate  # 1 unit per sample * sample_rate
        np.testing.assert_array_almost_equal(result.data, np.full(101, expected_slope), decimal=0)

    def test_differentiate_sine_wave(self, sine_trace: WaveformTrace):
        """Test derivative of sine (should be cosine)."""
        result = differentiate(sine_trace, method="central")

        # Derivative of sin(2πft) is 2πf*cos(2πft)
        t = np.arange(len(sine_trace.data)) / sine_trace.metadata.sample_rate
        frequency = 1000.0
        expected = 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)

        # Central difference is quite accurate for smooth signals
        # Allow higher tolerance for numerical derivative
        np.testing.assert_array_almost_equal(result.data, expected, decimal=0)

    def test_differentiate_order_2(self, ramp_trace: WaveformTrace):
        """Test second derivative."""
        result = differentiate(ramp_trace, order=2)

        # Second derivative of linear ramp should be zero
        # Allow small numerical errors
        assert np.max(np.abs(result.data)) < 1e6  # Should be close to zero

    def test_differentiate_method_forward(self, simple_trace: WaveformTrace):
        """Test forward difference method."""
        result = differentiate(simple_trace, method="forward")

        # Forward difference should maintain array length
        assert len(result.data) == len(simple_trace.data)

    def test_differentiate_method_backward(self, simple_trace: WaveformTrace):
        """Test backward difference method."""
        result = differentiate(simple_trace, method="backward")

        # Backward difference should maintain array length
        assert len(result.data) == len(simple_trace.data)

    def test_differentiate_method_central(self, simple_trace: WaveformTrace):
        """Test central difference method (default)."""
        result = differentiate(simple_trace, method="central")

        # Central difference should maintain array length
        assert len(result.data) == len(simple_trace.data)

    def test_differentiate_invalid_method(self, simple_trace: WaveformTrace):
        """Test invalid differentiation method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            differentiate(simple_trace, method="invalid")

    def test_differentiate_insufficient_samples(self, simple_metadata: TraceMetadata):
        """Test differentiation with insufficient samples."""
        tiny_data = np.array([1.0])
        tiny_trace = WaveformTrace(data=tiny_data, metadata=simple_metadata)

        with pytest.raises(InsufficientDataError):
            differentiate(tiny_trace, order=1)

    def test_differentiate_negative_order(self, simple_trace: WaveformTrace):
        """Test negative order raises error."""
        with pytest.raises(ValueError, match="Order must be positive"):
            differentiate(simple_trace, order=0)

    def test_differentiate_custom_channel_name(self, simple_trace: WaveformTrace):
        """Test custom channel name."""
        result = differentiate(simple_trace, channel_name="velocity")

        assert result.metadata.channel_name == "velocity"

    def test_differentiate_clears_units(self, simple_trace: WaveformTrace):
        """Test that differentiation clears vertical scale/offset."""
        result = differentiate(simple_trace)

        # Units change during differentiation
        assert result.metadata.vertical_scale is None
        assert result.metadata.vertical_offset is None


# =============================================================================
# Integration Tests (ARITH-006)
# =============================================================================


@pytest.mark.unit
class TestIntegrate:
    """Test integrate() function."""

    def test_integrate_constant(self, simple_metadata: TraceMetadata):
        """Test integral of constant (should be linear ramp)."""
        constant_data = np.ones(100)
        constant_trace = WaveformTrace(data=constant_data, metadata=simple_metadata)

        result = integrate(constant_trace, method="trapezoid")

        # Integral of 1 should be linear ramp
        # Check that it's monotonically increasing
        assert np.all(np.diff(result.data) >= 0)

        # Last value should be approximately length * dt
        dt = constant_trace.metadata.time_base
        expected_final = 100 * dt
        np.testing.assert_almost_equal(result.data[-1], expected_final, decimal=6)

    def test_integrate_zero(self, zero_trace: WaveformTrace):
        """Test integral of zero (should be zero with initial=0)."""
        result = integrate(zero_trace, method="trapezoid", initial=0.0)

        np.testing.assert_array_almost_equal(result.data, np.zeros(5))

    def test_integrate_with_initial_value(self, simple_trace: WaveformTrace):
        """Test integral with initial value (only 0 is supported by scipy)."""
        result = integrate(simple_trace, method="trapezoid", initial=0.0)

        # First value should be initial value
        assert result.data[0] == 0.0

        # Check that integral is cumulative
        assert np.all(np.diff(result.data) >= 0)  # Should be monotonically increasing

    def test_integrate_method_cumsum(self, simple_trace: WaveformTrace):
        """Test cumsum integration method."""
        result = integrate(simple_trace, method="cumsum")

        # Should return cumulative sum * dt
        assert len(result.data) == len(simple_trace.data)

    def test_integrate_method_simpson(self, simple_trace: WaveformTrace):
        """Test Simpson's rule integration method."""
        result = integrate(simple_trace, method="simpson")

        # Simpson's currently uses trapezoid internally
        assert len(result.data) == len(simple_trace.data)

    def test_integrate_invalid_method(self, simple_trace: WaveformTrace):
        """Test invalid integration method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            integrate(simple_trace, method="invalid")

    def test_integrate_insufficient_samples(self, simple_metadata: TraceMetadata):
        """Test integration with insufficient samples."""
        tiny_data = np.array([1.0])
        tiny_trace = WaveformTrace(data=tiny_data, metadata=simple_metadata)

        with pytest.raises(InsufficientDataError):
            integrate(tiny_trace)

    def test_integrate_differentiate_identity(self, simple_metadata: TraceMetadata):
        """Test that integrate(differentiate(f)) ≈ f for simple functions."""
        # Use a quadratic function: f(x) = x^2
        x = np.linspace(0, 1, 100)
        data = x**2
        trace = WaveformTrace(data=data, metadata=simple_metadata)

        # Differentiate then integrate
        derivative = differentiate(trace, method="central")
        recovered = integrate(derivative, method="trapezoid", initial=0.0)

        # Should recover original (within numerical error)
        # Allow for larger error due to numerical integration/differentiation
        np.testing.assert_array_almost_equal(recovered.data, trace.data, decimal=1)

    def test_integrate_clears_units(self, simple_trace: WaveformTrace):
        """Test that integration clears vertical scale/offset."""
        result = integrate(simple_trace)

        # Units change during integration
        assert result.metadata.vertical_scale is None
        assert result.metadata.vertical_offset is None


# =============================================================================
# Mathematical Expression Tests
# =============================================================================


@pytest.mark.unit
class TestMathExpression:
    """Test math_expression() function."""

    def test_expression_add_traces(self, simple_trace: WaveformTrace, second_trace: WaveformTrace):
        """Test addition expression."""
        result = math_expression(
            "trace1 + trace2",
            {"trace1": simple_trace, "trace2": second_trace},
        )

        np.testing.assert_array_almost_equal(
            result.data,
            simple_trace.data + second_trace.data,
        )

    def test_expression_multiply_traces(
        self, simple_trace: WaveformTrace, second_trace: WaveformTrace
    ):
        """Test multiplication expression."""
        result = math_expression(
            "trace1 * trace2",
            {"trace1": simple_trace, "trace2": second_trace},
        )

        np.testing.assert_array_almost_equal(
            result.data,
            simple_trace.data * second_trace.data,
        )

    def test_expression_complex(self, simple_trace: WaveformTrace, second_trace: WaveformTrace):
        """Test complex expression."""
        result = math_expression(
            "(trace1 + trace2) / 2",
            {"trace1": simple_trace, "trace2": second_trace},
        )

        expected = (simple_trace.data + second_trace.data) / 2
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_expression_numpy_functions(self, simple_trace: WaveformTrace):
        """Test numpy functions in expression."""
        result = math_expression("sqrt(abs(trace))", {"trace": simple_trace})

        expected = np.sqrt(np.abs(simple_trace.data))
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_expression_trigonometric(self, sine_trace: WaveformTrace):
        """Test trigonometric functions."""
        result = math_expression("sin(trace) + cos(trace)", {"trace": sine_trace})

        expected = np.sin(sine_trace.data) + np.cos(sine_trace.data)
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_expression_with_constants(self, simple_trace: WaveformTrace):
        """Test expression with constants."""
        result = math_expression("trace * 2 + 1", {"trace": simple_trace})

        expected = simple_trace.data * 2 + 1
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_expression_with_pi(self, simple_trace: WaveformTrace):
        """Test expression using pi constant."""
        result = math_expression("trace * pi", {"trace": simple_trace})

        expected = simple_trace.data * np.pi
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_expression_power_calculation(
        self, simple_trace: WaveformTrace, second_trace: WaveformTrace
    ):
        """Test power calculation (voltage * current)."""
        result = math_expression(
            "voltage * current",
            {"voltage": simple_trace, "current": second_trace},
        )

        expected = simple_trace.data * second_trace.data
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_expression_no_traces_error(self):
        """Test expression with no traces raises error."""
        with pytest.raises(AnalysisError, match="No traces provided"):
            math_expression("1 + 1", {})

    def test_expression_length_mismatch(
        self, simple_trace: WaveformTrace, simple_metadata: TraceMetadata
    ):
        """Test expression with mismatched trace lengths raises error."""
        long_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        long_trace = WaveformTrace(data=long_data, metadata=simple_metadata)

        with pytest.raises(AnalysisError, match="different length"):
            math_expression("t1 + t2", {"t1": simple_trace, "t2": long_trace})

    def test_expression_sample_rate_mismatch(
        self, simple_trace: WaveformTrace, simple_metadata: TraceMetadata
    ):
        """Test expression with mismatched sample rates raises error."""
        different_metadata = TraceMetadata(sample_rate=simple_metadata.sample_rate * 2.0)
        different_trace = WaveformTrace(data=simple_trace.data.copy(), metadata=different_metadata)

        with pytest.raises(AnalysisError, match="different sample rate"):
            math_expression("t1 + t2", {"t1": simple_trace, "t2": different_trace})

    def test_expression_invalid_syntax(self, simple_trace: WaveformTrace):
        """Test invalid expression syntax."""
        with pytest.raises(AnalysisError, match="Invalid expression syntax"):
            math_expression("trace +", {"trace": simple_trace})

    def test_expression_invalid_characters(self, simple_trace: WaveformTrace):
        """Test expression with disallowed operations."""
        # AST-based evaluator prevents dangerous operations like import
        with pytest.raises(AnalysisError, match="not allowed|not defined"):
            math_expression("__import__('os')", {"trace": simple_trace})

    def test_expression_scalar_result(self, simple_trace: WaveformTrace):
        """Test expression returning scalar (should broadcast)."""
        result = math_expression("mean(trace)", {"trace": simple_trace})

        # Should broadcast scalar to array
        assert len(result.data) == len(simple_trace.data)
        assert np.all(result.data == np.mean(simple_trace.data))

    def test_expression_custom_channel_name(self, simple_trace: WaveformTrace):
        """Test custom channel name."""
        result = math_expression(
            "trace * 2",
            {"trace": simple_trace},
            channel_name="doubled",
        )

        assert result.metadata.channel_name == "doubled"

    def test_expression_channel_name_truncation(self, simple_trace: WaveformTrace):
        """Test that long expressions are truncated in channel name."""
        result = math_expression(
            "trace + trace + trace + trace + trace + trace",
            {"trace": simple_trace},
        )

        # Should be truncated to 20 chars plus "expr(" and ")"
        assert len(result.metadata.channel_name) <= 26  # "expr(" + 20 + ")"


# =============================================================================
# Edge Cases and Special Values
# =============================================================================


@pytest.mark.unit
class TestMathArithmeticEdgeCases:
    """Test edge cases with special values."""

    def test_add_infinity(self, simple_metadata: TraceMetadata):
        """Test addition with infinity."""
        inf_data = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        inf_trace = WaveformTrace(data=inf_data, metadata=simple_metadata)

        result = add(inf_trace, 10.0)

        assert result.data[0] == 11.0
        assert np.isinf(result.data[1])
        assert result.data[2] == 13.0

    def test_multiply_nan(self, simple_metadata: TraceMetadata):
        """Test multiplication with NaN."""
        nan_data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        nan_trace = WaveformTrace(data=nan_data, metadata=simple_metadata)

        result = multiply(nan_trace, 2.0)

        assert result.data[0] == 2.0
        assert np.isnan(result.data[1])
        assert result.data[2] == 6.0

    def test_divide_infinity_by_infinity(self, simple_metadata: TraceMetadata):
        """Test dividing infinity by infinity (should be NaN)."""
        inf_data = np.array([np.inf, np.inf, np.inf])
        inf_trace = WaveformTrace(data=inf_data, metadata=simple_metadata)

        result = divide(inf_trace, inf_trace)

        # inf/inf is NaN
        assert np.all(np.isnan(result.data))

    def test_overflow_handling(self, simple_metadata: TraceMetadata):
        """Test overflow in multiplication."""
        large_data = np.array([1e308, 2e308, 3e308])
        large_trace = WaveformTrace(data=large_data, metadata=simple_metadata)

        # Overflow may generate warnings, which is expected
        with np.errstate(over="ignore"):
            result = multiply(large_trace, 10.0)

        # Should handle overflow gracefully (may produce inf)
        assert len(result.data) == 3

    def test_very_small_numbers(self, simple_metadata: TraceMetadata):
        """Test operations with very small numbers."""
        tiny_data = np.array([1e-308, 2e-308, 3e-308])
        tiny_trace = WaveformTrace(data=tiny_data, metadata=simple_metadata)

        result = multiply(tiny_trace, 0.1)

        # Should handle underflow (may produce zero)
        assert len(result.data) == 3

    def test_empty_trace(self, simple_metadata: TraceMetadata):
        """Test operations on empty trace."""
        empty_data = np.array([])
        empty_trace = WaveformTrace(data=empty_data, metadata=simple_metadata)

        result = add(empty_trace, 10.0)

        assert len(result.data) == 0

    def test_single_sample_trace(self, simple_metadata: TraceMetadata):
        """Test operations on single-sample trace."""
        single_data = np.array([5.0])
        single_trace = WaveformTrace(data=single_data, metadata=simple_metadata)

        result = multiply(single_trace, 2.0)

        np.testing.assert_array_almost_equal(result.data, np.array([10.0]))

    def test_mixed_dtypes(self, simple_metadata: TraceMetadata):
        """Test operations with different dtypes."""
        int_data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        int_trace = WaveformTrace(data=int_data, metadata=simple_metadata)

        result = add(int_trace, 2.5)

        # Should convert to float64
        assert result.data.dtype == np.float64
        np.testing.assert_array_almost_equal(
            result.data,
            np.array([3.5, 4.5, 5.5, 6.5, 7.5]),
        )

    def test_sample_rate_tolerance(
        self, simple_trace: WaveformTrace, simple_metadata: TraceMetadata
    ):
        """Test that sample rate tolerance allows small differences."""
        # Within 0.1% tolerance
        close_rate = simple_metadata.sample_rate * 1.0005
        close_metadata = TraceMetadata(sample_rate=close_rate)
        close_trace = WaveformTrace(data=simple_trace.data.copy(), metadata=close_metadata)

        # Should not raise error
        result = add(simple_trace, close_trace)
        assert len(result.data) == len(simple_trace.data)


# =============================================================================
# Array Vectorization Tests
# =============================================================================


@pytest.mark.unit
class TestVectorization:
    """Test that operations are properly vectorized."""

    def test_large_array_addition(self, sample_rate: float):
        """Test addition on large arrays."""
        size = 1_000_000
        data1 = np.random.randn(size)
        data2 = np.random.randn(size)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace1 = WaveformTrace(data=data1, metadata=metadata)
        trace2 = WaveformTrace(data=data2, metadata=metadata)

        result = add(trace1, trace2)

        assert len(result.data) == size
        np.testing.assert_array_almost_equal(result.data, data1 + data2)

    def test_large_array_multiplication(self, sample_rate: float):
        """Test multiplication on large arrays."""
        size = 1_000_000
        data = np.random.randn(size)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = multiply(trace, 2.5)

        assert len(result.data) == size
        np.testing.assert_array_almost_equal(result.data, data * 2.5)

    def test_broadcasting_scalar(self, sample_rate: float):
        """Test that scalar operations are properly broadcast."""
        size = 10_000
        data = np.arange(size, dtype=np.float64)

        metadata = TraceMetadata(sample_rate=sample_rate)
        trace = WaveformTrace(data=data, metadata=metadata)

        result = add(trace, 100.0)

        np.testing.assert_array_almost_equal(result.data, data + 100.0)
