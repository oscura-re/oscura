"""Comprehensive tests for oscura/utils/lazy.py.

This test module provides complete coverage for lazy evaluation utilities,
including edge cases and error conditions.

Tests cover:
- LazyProxy abstract base class behavior
- LazyArray lazy computation and caching
- LazyOperation chaining and evaluation
- Progressive resolution analysis
- ROI selection and edge cases
- Error handling and validation
"""

import numpy as np
import pytest

from oscura.utils.lazy import (
    LazyArray,
    LazyOperation,
    ProgressiveResolution,
    auto_preview,
    lazy_operation,
    select_roi,
)

# Fixtures


@pytest.fixture
def sample_data() -> np.ndarray:
    """Create sample data for testing.

    Returns:
        1D numpy array with 1000 samples
    """
    np.random.seed(42)
    return np.random.randn(1000).astype(np.float64)


@pytest.fixture
def large_data() -> np.ndarray:
    """Create large dataset for memory testing.

    Returns:
        1D numpy array with 10 million samples
    """
    np.random.seed(42)
    return np.random.randn(10_000_000).astype(np.float64)


# LazyArray Tests


class TestLazyArray:
    """Test suite for LazyArray class.

    Tests cover:
    - Initialization and deferred computation
    - Caching behavior
    - Array-like interface (__len__, __getitem__, shape, dtype)
    - Reset functionality
    """

    def test_initialization(self) -> None:
        """Test LazyArray initializes without computing."""
        call_count = 0

        def expensive_func() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.arange(100).astype(np.float64)

        lazy = LazyArray(expensive_func)
        assert not lazy.is_computed()
        assert call_count == 0  # Not called yet

    def test_compute_defers_execution(self) -> None:
        """Test computation only happens on .compute() call."""
        call_count = 0

        def expensive_func() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.random.randn(1000).astype(np.float64)

        lazy = LazyArray(expensive_func)
        assert call_count == 0

        result = lazy.compute()
        assert call_count == 1
        assert result is not None
        assert len(result) == 1000

    def test_compute_caches_result(self) -> None:
        """Test subsequent .compute() calls use cached result."""
        call_count = 0

        def expensive_func() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.random.randn(100).astype(np.float64)

        lazy = LazyArray(expensive_func)
        result1 = lazy.compute()
        result2 = lazy.compute()

        assert call_count == 1  # Only called once
        assert np.array_equal(result1, result2)
        assert lazy.is_computed()

    def test_reset_clears_cache(self) -> None:
        """Test reset() forces recomputation."""
        call_count = 0

        def func() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.random.randn(50).astype(np.float64)

        lazy = LazyArray(func)
        lazy.compute()
        assert call_count == 1

        lazy.reset()
        assert not lazy.is_computed()

        lazy.compute()
        assert call_count == 2  # Called again after reset

    def test_len_triggers_computation(self) -> None:
        """Test __len__ triggers computation."""
        lazy = LazyArray(lambda: np.arange(100).astype(np.float64))
        assert not lazy.is_computed()

        length = len(lazy)
        assert length == 100
        assert lazy.is_computed()

    def test_getitem_triggers_computation(self) -> None:
        """Test __getitem__ triggers computation."""
        lazy = LazyArray(lambda: np.arange(10).astype(np.float64))
        assert not lazy.is_computed()

        value = lazy[5]
        assert value == 5.0
        assert lazy.is_computed()

    def test_shape_triggers_computation(self) -> None:
        """Test shape() method triggers computation."""
        lazy = LazyArray(lambda: np.zeros((10, 5)).astype(np.float64))
        assert not lazy.is_computed()

        shape = lazy.shape()
        assert shape == (10, 5)
        assert lazy.is_computed()

    def test_dtype_triggers_computation(self) -> None:
        """Test dtype() method triggers computation."""
        lazy = LazyArray(lambda: np.zeros(10).astype(np.float32))
        assert not lazy.is_computed()

        dtype = lazy.dtype()
        assert dtype == np.float32
        assert lazy.is_computed()

    def test_with_args_and_kwargs(self) -> None:
        """Test LazyArray with function arguments."""

        def func_with_args(a: int, b: int, c: int = 0) -> np.ndarray:
            return np.full(10, a + b + c).astype(np.float64)

        lazy = LazyArray(func_with_args, 5, 10, c=3)
        result = lazy.compute()
        assert np.all(result == 18.0)

    def test_slicing(self) -> None:
        """Test array slicing via __getitem__."""
        lazy = LazyArray(lambda: np.arange(100).astype(np.float64))

        slice_result = lazy[10:20]
        expected = np.arange(10, 20).astype(np.float64)
        assert np.array_equal(slice_result, expected)


# LazyOperation Tests


class TestLazyOperation:
    """Test suite for LazyOperation class.

    Tests cover:
    - Operation chaining
    - Evaluation of lazy operands
    - Mix of lazy and eager operands
    - Multiple argument operations
    """

    def test_simple_operation(self) -> None:
        """Test simple operation with eager data."""
        data = np.arange(10).astype(np.float64)
        op = LazyOperation(lambda x: x**2, data)

        assert not op.is_computed()
        result = op.compute()
        expected = data**2
        assert np.array_equal(result, expected)

    def test_operation_chaining(self) -> None:
        """Test chaining multiple lazy operations."""
        data = np.arange(10).astype(np.float64)

        # Chain: (x ** 2) + 1
        op1 = LazyOperation(lambda x: x**2, data)
        op2 = LazyOperation(lambda x: x + 1, op1)

        result = op2.compute()
        expected = (data**2) + 1
        assert np.array_equal(result, expected)

    def test_lazy_operand_evaluation(self) -> None:
        """Test lazy operands are evaluated automatically."""
        call_count = 0

        def track_calls() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.arange(10).astype(np.float64)

        lazy_data = LazyArray(track_calls)
        op = LazyOperation(lambda x: x * 2, lazy_data)

        assert call_count == 0
        result = op.compute()
        assert call_count == 1  # LazyArray computed
        expected = np.arange(10).astype(np.float64) * 2
        assert np.array_equal(result, expected)

    def test_multiple_operands(self) -> None:
        """Test operation with multiple operands."""
        a = np.array([1, 2, 3]).astype(np.float64)
        b = np.array([4, 5, 6]).astype(np.float64)

        op = LazyOperation(lambda x, y: x + y, a, b)
        result = op.compute()
        expected = np.array([5, 7, 9]).astype(np.float64)
        assert np.array_equal(result, expected)

    def test_kwargs_passed_to_operation(self) -> None:
        """Test keyword arguments passed to operation."""

        def func_with_kwargs(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
            return x * scale

        data = np.arange(5).astype(np.float64)
        op = LazyOperation(func_with_kwargs, data, scale=10.0)

        result = op.compute()
        expected = data * 10.0
        assert np.array_equal(result, expected)

    def test_complex_chaining(self) -> None:
        """Test complex operation chaining with multiple dependencies."""
        data = np.arange(10).astype(np.float64)

        # Build computation graph: (x + 1) * (x ** 2)
        op1 = LazyOperation(lambda x: x + 1, data)
        op2 = LazyOperation(lambda x: x**2, data)
        op3 = LazyOperation(lambda x, y: x * y, op1, op2)

        result = op3.compute()
        expected = (data + 1) * (data**2)
        assert np.array_equal(result, expected)

    def test_reset_invalidates_cache(self) -> None:
        """Test reset clears cached computation."""
        call_count = 0

        def tracked_op(x: np.ndarray) -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return x * 2

        data = np.arange(5).astype(np.float64)
        op = LazyOperation(tracked_op, data)

        op.compute()
        assert call_count == 1

        op.reset()
        op.compute()
        assert call_count == 2  # Recomputed


# lazy_operation helper tests


def test_lazy_operation_helper() -> None:
    """Test lazy_operation helper function."""
    data = np.arange(10).astype(np.float64)
    lazy_result = lazy_operation(np.fft.fft, data)

    assert isinstance(lazy_result, LazyOperation)
    assert not lazy_result.is_computed()

    result = lazy_result.compute()
    expected = np.fft.fft(data)
    assert np.allclose(result, expected)


# auto_preview tests


class TestAutoPreview:
    """Test suite for auto_preview function.

    Tests cover:
    - Preview generation for large datasets
    - Full data return for small datasets
    - Downsampling factor control
    """

    def test_small_data_returns_full(self, sample_data: np.ndarray) -> None:
        """Test small data returns full array."""
        result = auto_preview(sample_data, preview_only=False)
        assert np.array_equal(result, sample_data.astype(np.float64))

    def test_large_data_returns_downsampled(self, large_data: np.ndarray) -> None:
        """Test large data returns downsampled preview."""
        result = auto_preview(large_data, downsample_factor=100, preview_only=True)
        expected_len = len(large_data) // 100
        assert len(result) == expected_len
        assert result.dtype == np.float64

    def test_downsample_factor_respected(self, sample_data: np.ndarray) -> None:
        """Test downsampling factor is applied correctly."""
        factor = 5
        result = auto_preview(sample_data, downsample_factor=factor, preview_only=True)
        expected_len = len(sample_data) // factor
        assert len(result) == expected_len
        # Verify it's actually downsampled data
        assert np.array_equal(result, sample_data[::factor].astype(np.float64))

    def test_preview_only_flag(self, sample_data: np.ndarray) -> None:
        """Test preview_only flag controls behavior."""
        # With preview_only=True
        preview = auto_preview(sample_data, downsample_factor=10, preview_only=True)
        assert len(preview) < len(sample_data)

        # With preview_only=False and small data
        full = auto_preview(sample_data, downsample_factor=10, preview_only=False)
        assert len(full) == len(sample_data)


# select_roi tests


class TestSelectROI:
    """Test suite for select_roi function.

    Tests cover:
    - Sample-based selection
    - Time-based selection
    - Edge cases (empty, out of bounds)
    - Error conditions
    """

    def test_select_by_sample_indices(self, sample_data: np.ndarray) -> None:
        """Test ROI selection using sample indices."""
        roi = select_roi(sample_data, start=100, end=200)
        expected = sample_data[100:200].astype(np.float64)
        assert np.array_equal(roi, expected)

    def test_select_by_time(self, sample_data: np.ndarray) -> None:
        """Test ROI selection using time values."""
        sample_rate = 1000.0
        start_time = 0.1  # 100 samples at 1 kHz
        end_time = 0.2  # 200 samples

        roi = select_roi(
            sample_data,
            start_time=start_time,
            end_time=end_time,
            sample_rate=sample_rate,
        )

        expected = sample_data[100:200].astype(np.float64)
        assert np.array_equal(roi, expected)

    def test_time_without_sample_rate_raises(self, sample_data: np.ndarray) -> None:
        """Test time-based selection without sample_rate raises ValueError."""
        with pytest.raises(ValueError, match="sample_rate required"):
            select_roi(sample_data, start_time=0.1, end_time=0.2)

    def test_defaults_to_full_range(self, sample_data: np.ndarray) -> None:
        """Test None values default to full data range."""
        roi = select_roi(sample_data, start=None, end=None)
        assert np.array_equal(roi, sample_data.astype(np.float64))

    def test_clips_to_bounds(self, sample_data: np.ndarray) -> None:
        """Test out-of-bounds indices are clipped."""
        # Start before 0
        roi = select_roi(sample_data, start=-100, end=10)
        expected = sample_data[0:10].astype(np.float64)
        assert np.array_equal(roi, expected)

        # End beyond length
        roi = select_roi(sample_data, start=990, end=2000)
        expected = sample_data[990:].astype(np.float64)
        assert np.array_equal(roi, expected)

    def test_invalid_roi_raises(self, sample_data: np.ndarray) -> None:
        """Test start >= end raises ValueError."""
        with pytest.raises(ValueError, match="Invalid ROI"):
            select_roi(sample_data, start=500, end=100)

        with pytest.raises(ValueError, match="Invalid ROI"):
            select_roi(sample_data, start=500, end=500)

    def test_partial_specification(self, sample_data: np.ndarray) -> None:
        """Test specifying only start or only end."""
        # Only start
        roi = select_roi(sample_data, start=800)
        expected = sample_data[800:].astype(np.float64)
        assert np.array_equal(roi, expected)

        # Only end
        roi = select_roi(sample_data, end=200)
        expected = sample_data[:200].astype(np.float64)
        assert np.array_equal(roi, expected)


# ProgressiveResolution tests


class TestProgressiveResolution:
    """Test suite for ProgressiveResolution class.

    Tests cover:
    - Preview generation and caching
    - ROI extraction
    - Both eager and lazy data sources
    - Sample rate handling
    """

    @pytest.fixture
    def analyzer(self, sample_data: np.ndarray) -> ProgressiveResolution:
        """Create ProgressiveResolution instance for testing.

        Returns:
            Configured ProgressiveResolution analyzer
        """
        return ProgressiveResolution(sample_data, sample_rate=1000.0)

    def test_initialization(self, sample_data: np.ndarray) -> None:
        """Test ProgressiveResolution initialization."""
        analyzer = ProgressiveResolution(sample_data, sample_rate=1000.0)
        assert analyzer.sample_rate == 1000.0

    def test_get_preview(self, analyzer: ProgressiveResolution) -> None:
        """Test preview generation."""
        preview = analyzer.get_preview(downsample_factor=10)
        # Preview should be 1/10th the size
        assert len(preview) == 100  # 1000 / 10
        assert preview.dtype == np.float64

    def test_preview_caching(self, analyzer: ProgressiveResolution) -> None:
        """Test preview is cached and reused."""
        preview1 = analyzer.get_preview(downsample_factor=10)
        preview2 = analyzer.get_preview(downsample_factor=10)

        # Should be the same object (cached)
        assert preview1 is preview2

    def test_preview_recompute_different_factor(self, analyzer: ProgressiveResolution) -> None:
        """Test preview recomputes with different downsample factor."""
        preview1 = analyzer.get_preview(downsample_factor=10)
        preview2 = analyzer.get_preview(downsample_factor=5)

        assert len(preview1) != len(preview2)
        # Second preview should replace cache
        preview3 = analyzer.get_preview(downsample_factor=5)
        assert preview2 is preview3

    def test_force_recompute(self, analyzer: ProgressiveResolution) -> None:
        """Test force_recompute clears cache."""
        preview1 = analyzer.get_preview(downsample_factor=10)
        preview2 = analyzer.get_preview(downsample_factor=10, force_recompute=True)

        # Different objects (recomputed)
        assert preview1 is not preview2
        assert np.array_equal(preview1, preview2)

    def test_get_roi_by_samples(
        self, analyzer: ProgressiveResolution, sample_data: np.ndarray
    ) -> None:
        """Test ROI extraction using sample indices."""
        roi = analyzer.get_roi(start=100, end=200)
        expected = sample_data[100:200].astype(np.float64)
        assert np.array_equal(roi, expected)

    def test_get_roi_by_time(
        self, analyzer: ProgressiveResolution, sample_data: np.ndarray
    ) -> None:
        """Test ROI extraction using time values."""
        # Sample rate is 1000 Hz, so 0.1s = 100 samples
        roi = analyzer.get_roi(start_time=0.1, end_time=0.2)
        expected = sample_data[100:200].astype(np.float64)
        assert np.array_equal(roi, expected)

    def test_with_lazy_data(self) -> None:
        """Test ProgressiveResolution with LazyArray data source."""
        call_count = 0

        def expensive_func() -> np.ndarray:
            nonlocal call_count
            call_count += 1
            return np.random.randn(10000).astype(np.float64)

        lazy_data = LazyArray(expensive_func)
        analyzer = ProgressiveResolution(lazy_data, sample_rate=1000.0)

        # Should not compute yet
        assert call_count == 0

        # Get preview triggers computation
        preview = analyzer.get_preview(downsample_factor=10)
        assert call_count == 1
        assert len(preview) == 1000

        # Get ROI uses cached data
        roi = analyzer.get_roi(start=100, end=200)
        assert call_count == 1  # No additional calls
        assert len(roi) == 100

    def test_sample_rate_property(self, analyzer: ProgressiveResolution) -> None:
        """Test sample_rate property is accessible."""
        assert analyzer.sample_rate == 1000.0


# Integration tests


class TestLazyEvaluationIntegration:
    """Integration tests for lazy evaluation workflows.

    Tests realistic usage patterns combining multiple components.
    """

    def test_progressive_workflow(self, large_data: np.ndarray) -> None:
        """Test realistic progressive resolution workflow."""
        # Stage 1: Create analyzer
        analyzer = ProgressiveResolution(large_data, sample_rate=1e6)

        # Stage 2: Get preview for visualization
        preview = analyzer.get_preview(downsample_factor=100)
        assert len(preview) == len(large_data) // 100

        # Stage 3: User identifies interesting region and zooms in
        roi = analyzer.get_roi(start_time=0.5, end_time=0.6)
        expected_len = int(0.1 * 1e6)  # 0.1 seconds at 1 MHz
        assert len(roi) == expected_len

    def test_lazy_operation_pipeline(self) -> None:
        """Test lazy operation pipeline with chaining."""
        data = np.arange(1000).astype(np.float64)

        # Build processing pipeline (all lazy)
        op1 = lazy_operation(lambda x: x - np.mean(x), data)  # Remove DC
        op2 = lazy_operation(lambda x: x * np.hanning(len(x)), op1)  # Window
        op3 = lazy_operation(np.fft.fft, op2)  # FFT

        # Nothing computed yet
        assert not op1.is_computed()
        assert not op2.is_computed()
        assert not op3.is_computed()

        # Compute final result (triggers all dependencies)
        result = op3.compute()
        assert result is not None
        assert len(result) == 1000

        # All stages now computed
        assert op1.is_computed()
        assert op2.is_computed()
        assert op3.is_computed()

    def test_memory_efficient_analysis(self, large_data: np.ndarray) -> None:
        """Test memory-efficient analysis with lazy evaluation."""
        # Wrap large data in lazy proxy
        lazy_data = LazyArray(lambda: large_data)

        # Create operations without materializing
        squared = LazyOperation(lambda x: x**2, lazy_data)
        mean = LazyOperation(np.mean, squared)

        # Compute only what's needed
        result = mean.compute()
        expected = np.mean(large_data**2)
        assert np.isclose(result, expected)
