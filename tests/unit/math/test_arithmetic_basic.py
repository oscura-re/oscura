"""Tests for arithmetic operations module.

Tests requirements:
"""

import numpy as np
import pytest

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


@pytest.fixture
def sample_trace():
    """Create a sample trace for testing."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def ramp_trace():
    """Create a ramp trace for testing differentiation/integration."""
    data = np.linspace(0, 1, 1000)
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


class TestAddition:
    """Tests for trace addition (ARITH-001)."""

    def test_add_traces(self, sample_trace):
        """Test adding two traces."""
        result = add(sample_trace, sample_trace)
        np.testing.assert_allclose(result.data, sample_trace.data * 2)

    def test_add_scalar(self, sample_trace):
        """Test adding scalar to trace."""
        result = add(sample_trace, 1.0)
        np.testing.assert_allclose(result.data, sample_trace.data + 1.0)

    def test_add_preserves_sample_rate(self, sample_trace):
        """Test that sample rate is preserved."""
        result = add(sample_trace, sample_trace)
        assert result.metadata.sample_rate == sample_trace.metadata.sample_rate


class TestSubtraction:
    """Tests for trace subtraction (ARITH-002)."""

    def test_subtract_traces(self, sample_trace):
        """Test subtracting two traces."""
        result = subtract(sample_trace, sample_trace)
        np.testing.assert_allclose(result.data, np.zeros_like(sample_trace.data))

    def test_subtract_scalar(self, sample_trace):
        """Test subtracting scalar from trace."""
        result = subtract(sample_trace, 0.5)
        np.testing.assert_allclose(result.data, sample_trace.data - 0.5)


class TestMultiplication:
    """Tests for trace multiplication (ARITH-003)."""

    def test_multiply_traces(self, sample_trace):
        """Test multiplying two traces."""
        result = multiply(sample_trace, sample_trace)
        np.testing.assert_allclose(result.data, sample_trace.data**2)

    def test_multiply_scalar(self, sample_trace):
        """Test multiplying trace by scalar."""
        result = multiply(sample_trace, 2.0)
        np.testing.assert_allclose(result.data, sample_trace.data * 2.0)


class TestDivision:
    """Tests for trace division (ARITH-004)."""

    def test_divide_traces(self, sample_trace):
        """Test dividing two traces."""
        # Create non-zero trace
        data = np.ones(1000) * 2.0
        trace2 = WaveformTrace(data=data, metadata=sample_trace.metadata)
        result = divide(sample_trace, trace2)
        np.testing.assert_allclose(result.data, sample_trace.data / 2.0)

    def test_divide_by_zero(self, sample_trace):
        """Test division by zero handling."""
        result = divide(sample_trace, 0.0)
        assert np.all(np.isnan(result.data))


class TestScale:
    """Tests for trace scaling."""

    def test_scale(self, sample_trace):
        """Test scaling trace."""
        result = scale(sample_trace, 2.0)
        np.testing.assert_allclose(result.data, sample_trace.data * 2.0)


class TestOffset:
    """Tests for trace offset."""

    def test_offset(self, sample_trace):
        """Test offsetting trace."""
        result = offset(sample_trace, 1.0)
        np.testing.assert_allclose(result.data, sample_trace.data + 1.0)


class TestInvert:
    """Tests for trace inversion."""

    def test_invert(self, sample_trace):
        """Test inverting trace."""
        result = invert(sample_trace)
        np.testing.assert_allclose(result.data, -sample_trace.data)


class TestAbsolute:
    """Tests for absolute value."""

    def test_absolute(self, sample_trace):
        """Test absolute value."""
        result = absolute(sample_trace)
        np.testing.assert_allclose(result.data, np.abs(sample_trace.data))


class TestDifferentiate:
    """Tests for trace differentiation (ARITH-005)."""

    def test_differentiate_ramp(self, ramp_trace):
        """Test differentiating a ramp (should give constant)."""
        result = differentiate(ramp_trace)
        # Derivative of linear ramp should be approximately constant
        # Expected slope = 1 / duration = 1 / (1000 * 1e-6) = 1000
        expected = 1.0 / (1e-3)  # 1000
        # Middle region should be close to expected
        middle = result.data[10:-10]
        np.testing.assert_allclose(middle, expected, rtol=0.1)

    def test_differentiate_sine(self, sample_trace):
        """Test differentiating sine (should give cosine-like)."""
        result = differentiate(sample_trace)
        # Just check it doesn't crash and has reasonable values
        assert np.isfinite(result.data).all()


class TestIntegrate:
    """Tests for trace integration (ARITH-006)."""

    def test_integrate_constant(self):
        """Test integrating a constant (should give ramp)."""
        data = np.ones(1000) * 2.0
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)
        result = integrate(trace)
        # Integral of 2 over time should be 2*t
        # At end: 2 * 1e-3 = 2e-3
        assert result.data[-1] > result.data[0]


class TestMathExpression:
    """Tests for math expression evaluation."""

    def test_simple_expression(self, sample_trace):
        """Test simple math expression."""
        result = math_expression("CH1 + CH1", {"CH1": sample_trace})
        np.testing.assert_allclose(result.data, sample_trace.data * 2)

    def test_complex_expression(self, sample_trace):
        """Test complex math expression."""
        result = math_expression("abs(CH1) * 2", {"CH1": sample_trace})
        np.testing.assert_allclose(result.data, np.abs(sample_trace.data) * 2)
