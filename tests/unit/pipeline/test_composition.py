"""Comprehensive unit tests for oscura.pipeline.composition module.

Tests functional composition operators: compose(), pipe(), Composable,
make_composable(), and curry().
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.pipeline.composition import (
    Composable,
    compose,
    curry,
    make_composable,
    pipe,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_trace() -> WaveformTrace:
    """Create a sample WaveformTrace for testing."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    metadata = TraceMetadata(
        sample_rate=1000.0,
        channel_name="CH1",
    )
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def double_func() -> Callable[[WaveformTrace], WaveformTrace]:
    """Create a function that doubles trace values."""

    def double(trace: WaveformTrace) -> WaveformTrace:
        return WaveformTrace(
            data=trace.data * 2,
            metadata=trace.metadata,
        )

    double.__name__ = "double"
    return double


@pytest.fixture
def add_one_func() -> Callable[[WaveformTrace], WaveformTrace]:
    """Create a function that adds 1 to trace values."""

    def add_one(trace: WaveformTrace) -> WaveformTrace:
        return WaveformTrace(
            data=trace.data + 1,
            metadata=trace.metadata,
        )

    add_one.__name__ = "add_one"
    return add_one


@pytest.fixture
def square_func() -> Callable[[WaveformTrace], WaveformTrace]:
    """Create a function that squares trace values."""

    def square(trace: WaveformTrace) -> WaveformTrace:
        return WaveformTrace(
            data=trace.data**2,
            metadata=trace.metadata,
        )

    square.__name__ = "square"
    return square


# =============================================================================
# compose() Tests
# =============================================================================


class TestCompose:
    """Tests for compose() function."""

    def test_compose_single_function(self, sample_trace, double_func):
        """Test compose with single function returns that function."""
        composed = compose(double_func)
        result = composed(sample_trace)

        np.testing.assert_array_equal(result.data, sample_trace.data * 2)

    def test_compose_two_functions(self, sample_trace, double_func, add_one_func):
        """Test compose applies functions right-to-left."""
        # compose(f, g)(x) == f(g(x))
        # compose(double, add_one)(x) == double(add_one(x))
        composed = compose(double_func, add_one_func)
        result = composed(sample_trace)

        # First add_one: [1,2,3,4,5] -> [2,3,4,5,6]
        # Then double: [2,3,4,5,6] -> [4,6,8,10,12]
        expected = (sample_trace.data + 1) * 2
        np.testing.assert_array_equal(result.data, expected)

    def test_compose_three_functions(self, sample_trace, double_func, add_one_func, square_func):
        """Test compose with three functions."""
        # compose(f, g, h)(x) == f(g(h(x)))
        composed = compose(double_func, add_one_func, square_func)
        result = composed(sample_trace)

        # First square: [1,2,3,4,5] -> [1,4,9,16,25]
        # Then add_one: [1,4,9,16,25] -> [2,5,10,17,26]
        # Then double: [2,5,10,17,26] -> [4,10,20,34,52]
        expected = ((sample_trace.data**2) + 1) * 2
        np.testing.assert_array_equal(result.data, expected)

    def test_compose_empty_raises_error(self):
        """Test compose with no functions raises ValueError."""
        with pytest.raises(ValueError, match="at least one function"):
            compose()

    def test_compose_preserves_metadata(self, sample_trace, double_func, add_one_func):
        """Test compose preserves trace metadata through pipeline."""
        composed = compose(double_func, add_one_func)
        result = composed(sample_trace)

        assert result.metadata.sample_rate == sample_trace.metadata.sample_rate
        assert result.metadata.channel_name == sample_trace.metadata.channel_name

    def test_compose_function_name(self, double_func, add_one_func):
        """Test composed function has descriptive name."""
        composed = compose(double_func, add_one_func)

        assert "compose" in composed.__name__
        assert "double" in composed.__name__
        assert "add_one" in composed.__name__

    def test_compose_function_doc(self, double_func, add_one_func):
        """Test composed function has docstring."""
        composed = compose(double_func, add_one_func)

        assert composed.__doc__ is not None
        assert "2" in composed.__doc__  # Number of functions


# =============================================================================
# pipe() Tests
# =============================================================================


class TestPipe:
    """Tests for pipe() function."""

    def test_pipe_single_function(self, sample_trace, double_func):
        """Test pipe with single function."""
        result = pipe(sample_trace, double_func)

        np.testing.assert_array_equal(result.data, sample_trace.data * 2)

    def test_pipe_two_functions(self, sample_trace, double_func, add_one_func):
        """Test pipe applies functions left-to-right."""
        # pipe(x, f, g) == g(f(x))
        # pipe(x, double, add_one) == add_one(double(x))
        result = pipe(sample_trace, double_func, add_one_func)

        # First double: [1,2,3,4,5] -> [2,4,6,8,10]
        # Then add_one: [2,4,6,8,10] -> [3,5,7,9,11]
        expected = (sample_trace.data * 2) + 1
        np.testing.assert_array_equal(result.data, expected)

    def test_pipe_three_functions(self, sample_trace, double_func, add_one_func, square_func):
        """Test pipe with three functions."""
        # pipe(x, f, g, h) == h(g(f(x)))
        result = pipe(sample_trace, double_func, add_one_func, square_func)

        # First double: [1,2,3,4,5] -> [2,4,6,8,10]
        # Then add_one: [2,4,6,8,10] -> [3,5,7,9,11]
        # Then square: [3,5,7,9,11] -> [9,25,49,81,121]
        expected = ((sample_trace.data * 2) + 1) ** 2
        np.testing.assert_array_equal(result.data, expected)

    def test_pipe_no_functions(self, sample_trace):
        """Test pipe with no functions returns original data."""
        result = pipe(sample_trace)

        np.testing.assert_array_equal(result.data, sample_trace.data)

    def test_pipe_preserves_metadata(self, sample_trace, double_func, add_one_func):
        """Test pipe preserves trace metadata."""
        result = pipe(sample_trace, double_func, add_one_func)

        assert result.metadata.sample_rate == sample_trace.metadata.sample_rate

    def test_pipe_with_partial(self, sample_trace):
        """Test pipe with partial application."""

        def scale(trace: WaveformTrace, factor: float) -> WaveformTrace:
            return WaveformTrace(data=trace.data * factor, metadata=trace.metadata)

        result = pipe(
            sample_trace,
            partial(scale, factor=2.0),
            partial(scale, factor=3.0),
        )

        expected = sample_trace.data * 2.0 * 3.0
        np.testing.assert_array_equal(result.data, expected)


# =============================================================================
# Composable Class Tests
# =============================================================================


class TestComposable:
    """Tests for Composable mixin class."""

    def test_rshift_operator(self, double_func):
        """Test >> operator applies function."""

        class ComposableTrace(Composable):
            def __init__(self, data, metadata):
                self.data = data
                self.metadata = metadata

        data = np.array([1.0, 2.0, 3.0])
        metadata = TraceMetadata(sample_rate=1000.0, channel_name="C")

        # Create WaveformTrace for the transform function
        trace = WaveformTrace(data=data, metadata=metadata)
        composable = ComposableTrace(data=data, metadata=metadata)

        # The >> operator should apply the function
        # We need a function that works with ComposableTrace
        def double_composable(t):
            return ComposableTrace(data=t.data * 2, metadata=t.metadata)

        result = composable >> double_composable

        np.testing.assert_array_equal(result.data, data * 2)

    def test_rshift_chaining(self):
        """Test chaining multiple >> operators."""

        class ComposableValue(Composable):
            def __init__(self, value):
                self.value = value

        def add_10(x):
            return ComposableValue(x.value + 10)

        def double(x):
            return ComposableValue(x.value * 2)

        start = ComposableValue(5)
        result = start >> add_10 >> double

        # 5 + 10 = 15, 15 * 2 = 30
        assert result.value == 30

    def test_rshift_with_lambda(self):
        """Test >> with lambda function."""

        class ComposableValue(Composable):
            def __init__(self, value):
                self.value = value

        start = ComposableValue(10)
        result = start >> (lambda x: ComposableValue(x.value * 3))

        assert result.value == 30


# =============================================================================
# make_composable() Tests
# =============================================================================


class TestMakeComposable:
    """Tests for make_composable() decorator."""

    def test_immediate_application(self, sample_trace):
        """Test decorated function with trace as first argument."""

        @make_composable
        def scale(trace: WaveformTrace, factor: float = 1.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data * factor, metadata=trace.metadata)

        result = scale(sample_trace, factor=2.0)

        np.testing.assert_array_equal(result.data, sample_trace.data * 2.0)

    def test_partial_application(self, sample_trace):
        """Test decorated function returns partial when no trace given."""

        @make_composable
        def scale(trace: WaveformTrace, factor: float = 1.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data * factor, metadata=trace.metadata)

        # Get partial function
        double = scale(factor=2.0)

        # Apply to trace
        result = double(sample_trace)

        np.testing.assert_array_equal(result.data, sample_trace.data * 2.0)

    def test_use_in_compose(self, sample_trace):
        """Test decorated function works with compose()."""

        @make_composable
        def scale(trace: WaveformTrace, factor: float = 1.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data * factor, metadata=trace.metadata)

        @make_composable
        def offset(trace: WaveformTrace, value: float = 0.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data + value, metadata=trace.metadata)

        composed = compose(scale(factor=2.0), offset(value=1.0))
        result = composed(sample_trace)

        # offset first (right to left): [1,2,3,4,5] + 1 = [2,3,4,5,6]
        # then scale: [2,3,4,5,6] * 2 = [4,6,8,10,12]
        expected = (sample_trace.data + 1.0) * 2.0
        np.testing.assert_array_equal(result.data, expected)

    def test_use_in_pipe(self, sample_trace):
        """Test decorated function works with pipe()."""

        @make_composable
        def scale(trace: WaveformTrace, factor: float = 1.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data * factor, metadata=trace.metadata)

        @make_composable
        def offset(trace: WaveformTrace, value: float = 0.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data + value, metadata=trace.metadata)

        result = pipe(sample_trace, scale(factor=2.0), offset(value=1.0))

        # scale first (left to right): [1,2,3,4,5] * 2 = [2,4,6,8,10]
        # then offset: [2,4,6,8,10] + 1 = [3,5,7,9,11]
        expected = (sample_trace.data * 2.0) + 1.0
        np.testing.assert_array_equal(result.data, expected)

    def test_preserves_function_name(self):
        """Test decorator preserves function name."""

        @make_composable
        def my_transform(trace: WaveformTrace) -> WaveformTrace:
            return trace

        assert my_transform.__name__ == "my_transform"

    def test_preserves_function_doc(self):
        """Test decorator preserves function docstring."""

        @make_composable
        def my_transform(trace: WaveformTrace) -> WaveformTrace:
            """My custom transform."""
            return trace

        assert my_transform.__doc__ == "My custom transform."


# =============================================================================
# curry() Tests
# =============================================================================


class TestCurry:
    """Tests for curry() decorator."""

    def test_immediate_application(self, sample_trace):
        """Test curried function with trace as first argument."""

        @curry
        def scale_offset(trace: WaveformTrace, scale: float, offset: float) -> WaveformTrace:
            return WaveformTrace(data=trace.data * scale + offset, metadata=trace.metadata)

        result = scale_offset(sample_trace, 2.0, 1.0)

        expected = sample_trace.data * 2.0 + 1.0
        np.testing.assert_array_equal(result.data, expected)

    def test_curried_application(self, sample_trace):
        """Test curried function with partial arguments."""

        @curry
        def scale_offset(trace: WaveformTrace, scale: float, offset: float) -> WaveformTrace:
            return WaveformTrace(data=trace.data * scale + offset, metadata=trace.metadata)

        # Create specialized function
        double_shift = scale_offset(scale=2.0, offset=1.0)

        # Apply to trace
        result = double_shift(sample_trace)

        expected = sample_trace.data * 2.0 + 1.0
        np.testing.assert_array_equal(result.data, expected)

    def test_curried_with_kwargs(self, sample_trace):
        """Test curried function with keyword arguments."""

        @curry
        def transform(trace: WaveformTrace, a: float = 1.0, b: float = 0.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data * a + b, metadata=trace.metadata)

        # First application with some kwargs
        partial_transform = transform(a=3.0)

        # Complete with more kwargs
        result = partial_transform(sample_trace, b=5.0)

        expected = sample_trace.data * 3.0 + 5.0
        np.testing.assert_array_equal(result.data, expected)

    def test_curried_in_compose(self, sample_trace):
        """Test curried function works with compose()."""

        @curry
        def multiply(trace: WaveformTrace, factor: float) -> WaveformTrace:
            return WaveformTrace(data=trace.data * factor, metadata=trace.metadata)

        composed = compose(multiply(factor=2.0), multiply(factor=3.0))
        result = composed(sample_trace)

        # Right to left: *3 then *2
        expected = sample_trace.data * 3.0 * 2.0
        np.testing.assert_array_equal(result.data, expected)

    def test_preserves_function_name(self):
        """Test curry preserves function name."""

        @curry
        def my_curried_func(trace: WaveformTrace, x: float) -> WaveformTrace:
            return trace

        assert my_curried_func.__name__ == "my_curried_func"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineCompositionIntegration:
    """Integration tests for composition operators."""

    def test_compose_and_pipe_equivalence(self, sample_trace):
        """Test that compose and pipe produce equivalent results (with reversed order)."""

        def f(trace: WaveformTrace) -> WaveformTrace:
            return WaveformTrace(data=trace.data + 1, metadata=trace.metadata)

        def g(trace: WaveformTrace) -> WaveformTrace:
            return WaveformTrace(data=trace.data * 2, metadata=trace.metadata)

        # compose(f, g)(x) = f(g(x)) - g first, then f
        compose_result = compose(f, g)(sample_trace)

        # pipe(x, g, f) = f(g(x)) - g first, then f
        pipe_result = pipe(sample_trace, g, f)

        np.testing.assert_array_equal(compose_result.data, pipe_result.data)

    def test_complex_pipeline(self, sample_trace):
        """Test complex pipeline with multiple decorated functions."""

        @make_composable
        def scale(trace: WaveformTrace, factor: float = 1.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data * factor, metadata=trace.metadata)

        @make_composable
        def offset(trace: WaveformTrace, value: float = 0.0) -> WaveformTrace:
            return WaveformTrace(data=trace.data + value, metadata=trace.metadata)

        @make_composable
        def clip(
            trace: WaveformTrace, min_val: float = 0.0, max_val: float = 10.0
        ) -> WaveformTrace:
            clipped = np.clip(trace.data, min_val, max_val)
            return WaveformTrace(data=clipped, metadata=trace.metadata)

        # Create complex pipeline
        result = pipe(
            sample_trace,
            scale(factor=3.0),  # [1,2,3,4,5] -> [3,6,9,12,15]
            offset(value=-5.0),  # [3,6,9,12,15] -> [-2,1,4,7,10]
            clip(min_val=0.0, max_val=8.0),  # [-2,1,4,7,10] -> [0,1,4,7,8]
        )

        expected = np.array([0.0, 1.0, 4.0, 7.0, 8.0])
        np.testing.assert_array_equal(result.data, expected)

    def test_reusable_pipeline(self, sample_trace):
        """Test creating reusable pipeline with compose."""

        @make_composable
        def double(trace: WaveformTrace) -> WaveformTrace:
            return WaveformTrace(data=trace.data * 2, metadata=trace.metadata)

        @make_composable
        def half(trace: WaveformTrace) -> WaveformTrace:
            return WaveformTrace(data=trace.data / 2, metadata=trace.metadata)

        # Create reusable pipeline that should return to original
        normalize_pipeline = compose(half(), double(), half(), double())

        result = normalize_pipeline(sample_trace)

        # double -> half -> double -> half should give original
        np.testing.assert_array_almost_equal(result.data, sample_trace.data)


# =============================================================================
# Edge Cases
# =============================================================================


class TestPipelineCompositionEdgeCases:
    """Tests for edge cases."""

    def test_empty_trace_data(self):
        """Test operations on empty trace data."""
        trace = WaveformTrace(
            data=np.array([]),
            metadata=TraceMetadata(sample_rate=1000.0, channel_name="C"),
        )

        def identity(t: WaveformTrace) -> WaveformTrace:
            return t

        result = pipe(trace, identity)
        assert len(result.data) == 0

    def test_large_trace(self):
        """Test operations on large trace."""
        large_data = np.random.randn(100000)
        trace = WaveformTrace(
            data=large_data,
            metadata=TraceMetadata(sample_rate=1e9, channel_name="C"),
        )

        def scale(t: WaveformTrace) -> WaveformTrace:
            return WaveformTrace(data=t.data * 2, metadata=t.metadata)

        result = pipe(trace, scale, scale, scale)

        expected = large_data * 8
        np.testing.assert_array_almost_equal(result.data, expected)

    def test_compose_many_functions(self, sample_trace):
        """Test compose with many functions."""

        def add_one(t: WaveformTrace) -> WaveformTrace:
            return WaveformTrace(data=t.data + 1, metadata=t.metadata)

        # Compose 10 add_one functions
        composed = compose(*[add_one for _ in range(10)])
        result = composed(sample_trace)

        expected = sample_trace.data + 10
        np.testing.assert_array_equal(result.data, expected)

    def test_pipe_many_functions(self, sample_trace):
        """Test pipe with many functions."""

        def double(t: WaveformTrace) -> WaveformTrace:
            return WaveformTrace(data=t.data * 2, metadata=t.metadata)

        # Apply double 5 times: 2^5 = 32
        result = pipe(sample_trace, *[double for _ in range(5)])

        expected = sample_trace.data * 32
        np.testing.assert_array_equal(result.data, expected)

    def test_identity_function(self, sample_trace):
        """Test with identity function."""

        def identity(t: WaveformTrace) -> WaveformTrace:
            return t

        result = pipe(sample_trace, identity, identity, identity)

        np.testing.assert_array_equal(result.data, sample_trace.data)
        assert result is sample_trace  # Should be same object
