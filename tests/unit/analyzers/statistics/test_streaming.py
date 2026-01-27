"""Comprehensive tests for statistics/streaming.py.

This test module provides complete coverage for streaming statistics computation,
including edge cases and numerical stability tests.
"""

import numpy as np
import pytest

from oscura.analyzers.statistics.streaming import StreamingStats, StreamingStatsResult


class TestStreamingStatsResult:
    """Test suite for StreamingStatsResult dataclass."""

    def test_initialization(self) -> None:
        """Test successful initialization with valid parameters."""
        result = StreamingStatsResult(
            mean=5.0, variance=2.5, std=1.58, min=1.0, max=10.0, count=100
        )

        assert result.mean == 5.0
        assert result.variance == 2.5
        assert result.std == 1.58
        assert result.min == 1.0
        assert result.max == 10.0
        assert result.count == 100

    def test_dataclass_equality(self) -> None:
        """Test two results with same values are equal."""
        result1 = StreamingStatsResult(
            mean=5.0, variance=2.5, std=1.58, min=1.0, max=10.0, count=100
        )
        result2 = StreamingStatsResult(
            mean=5.0, variance=2.5, std=1.58, min=1.0, max=10.0, count=100
        )

        assert result1 == result2


class TestStreamingStats:
    """Test suite for StreamingStats class.

    Tests cover:
    - Initialization and validation
    - Normal operation with valid inputs
    - Edge cases (empty, single value, two values)
    - Numerical stability with large datasets
    - Accuracy compared to numpy
    - Incremental updates
    """

    @pytest.fixture
    def stats(self) -> StreamingStats:
        """Create StreamingStats instance for testing.

        Returns:
            Configured StreamingStats instance
        """
        return StreamingStats()

    def test_initialization(self, stats: StreamingStats) -> None:
        """Test successful initialization with default state."""
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.m2 == 0.0
        assert stats.min_val == float("inf")
        assert stats.max_val == float("-inf")

    def test_single_value(self, stats: StreamingStats) -> None:
        """Test update with single value."""
        stats.update(np.array([5.0]))
        result = stats.finalize()

        assert result.count == 1
        assert result.mean == 5.0
        assert result.variance == 0.0  # Single value has no variance
        assert result.std == 0.0
        assert result.min == 5.0
        assert result.max == 5.0

    def test_two_values(self, stats: StreamingStats) -> None:
        """Test update with two values."""
        stats.update(np.array([4.0, 6.0]))
        result = stats.finalize()

        assert result.count == 2
        assert result.mean == 5.0
        assert result.variance == 2.0  # Sample variance
        assert abs(result.std - np.sqrt(2.0)) < 1e-10
        assert result.min == 4.0
        assert result.max == 6.0

    def test_multiple_values(self, stats: StreamingStats) -> None:
        """Test update with multiple values in one chunk."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats.update(data)
        result = stats.finalize()

        expected_mean = np.mean(data)
        expected_var = np.var(data, ddof=1)
        expected_std = np.std(data, ddof=1)

        assert result.count == 5
        assert abs(result.mean - expected_mean) < 1e-10
        assert abs(result.variance - expected_var) < 1e-10
        assert abs(result.std - expected_std) < 1e-10
        assert result.min == 1.0
        assert result.max == 5.0

    def test_incremental_updates(self, stats: StreamingStats) -> None:
        """Test multiple incremental updates give correct results."""
        # Add data in chunks
        chunk1 = np.array([1.0, 2.0, 3.0])
        chunk2 = np.array([4.0, 5.0])
        chunk3 = np.array([6.0, 7.0, 8.0])

        stats.update(chunk1)
        stats.update(chunk2)
        stats.update(chunk3)

        result = stats.finalize()

        # Compare with numpy on full dataset
        full_data = np.concatenate([chunk1, chunk2, chunk3])
        expected_mean = np.mean(full_data)
        expected_var = np.var(full_data, ddof=1)
        expected_std = np.std(full_data, ddof=1)

        assert result.count == len(full_data)
        assert abs(result.mean - expected_mean) < 1e-10
        assert abs(result.variance - expected_var) < 1e-10
        assert abs(result.std - expected_std) < 1e-10
        assert result.min == 1.0
        assert result.max == 8.0

    def test_empty_data(self, stats: StreamingStats) -> None:
        """Test finalize with no data."""
        result = stats.finalize()

        assert result.count == 0
        assert result.mean == 0.0
        assert result.variance == 0.0
        assert result.std == 0.0
        assert result.min == 0.0  # Converts inf to 0
        assert result.max == 0.0  # Converts -inf to 0

    def test_large_dataset(self, stats: StreamingStats) -> None:
        """Test with large dataset for numerical stability."""
        # Generate large dataset
        np.random.seed(42)
        data = np.random.randn(1000000)

        # Process in chunks
        chunk_size = 10000
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            stats.update(chunk)

        result = stats.finalize()

        # Compare with numpy
        expected_mean = np.mean(data)
        expected_var = np.var(data, ddof=1)
        expected_std = np.std(data, ddof=1)

        assert result.count == len(data)
        assert abs(result.mean - expected_mean) < 1e-6  # Good accuracy
        assert abs(result.variance - expected_var) < 1e-4
        assert abs(result.std - expected_std) < 1e-4
        assert result.min <= np.min(data)
        assert result.max >= np.max(data)

    def test_numerical_stability_large_values(self, stats: StreamingStats) -> None:
        """Test numerical stability with large values."""
        # Large values that could cause overflow with naive algorithm
        data = np.array([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3])
        stats.update(data)
        result = stats.finalize()

        expected_mean = np.mean(data)
        expected_var = np.var(data, ddof=1)

        # Welford's algorithm should maintain accuracy
        assert abs(result.mean - expected_mean) < 1e-5
        assert abs(result.variance - expected_var) < 1.0  # Relative to large values

    def test_negative_values(self, stats: StreamingStats) -> None:
        """Test with negative values."""
        data = np.array([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
        stats.update(data)
        result = stats.finalize()

        assert result.count == 6
        assert abs(result.mean - 0.0) < 1e-10
        assert result.min == -5.0
        assert result.max == 5.0
        assert result.variance > 0

    def test_all_same_values(self, stats: StreamingStats) -> None:
        """Test with all identical values."""
        data = np.ones(100) * 7.0
        stats.update(data)
        result = stats.finalize()

        assert result.count == 100
        assert result.mean == 7.0
        assert result.variance == 0.0  # No variance
        assert result.std == 0.0
        assert result.min == 7.0
        assert result.max == 7.0

    def test_min_max_tracking(self, stats: StreamingStats) -> None:
        """Test min/max are correctly tracked across updates."""
        stats.update(np.array([5.0, 10.0]))
        stats.update(np.array([1.0, 3.0]))  # New min
        stats.update(np.array([7.0, 15.0]))  # New max
        stats.update(np.array([8.0, 9.0]))

        result = stats.finalize()

        assert result.min == 1.0
        assert result.max == 15.0

    def test_variance_formula_correctness(self, stats: StreamingStats) -> None:
        """Test variance calculation uses sample variance (n-1)."""
        data = np.array([2.0, 4.0, 6.0, 8.0])
        stats.update(data)
        result = stats.finalize()

        # Sample variance (ddof=1)
        expected_var = np.var(data, ddof=1)
        assert abs(result.variance - expected_var) < 1e-10

        # Should NOT match population variance (ddof=0)
        pop_var = np.var(data, ddof=0)
        assert abs(result.variance - pop_var) > 0.1

    def test_empty_array_update(self, stats: StreamingStats) -> None:
        """Test updating with empty array."""
        stats.update(np.array([]))
        result = stats.finalize()

        assert result.count == 0
        assert result.mean == 0.0
        assert result.variance == 0.0

    def test_mixed_chunk_sizes(self, stats: StreamingStats) -> None:
        """Test updates with varying chunk sizes."""
        stats.update(np.array([1.0]))  # Single
        stats.update(np.array([2.0, 3.0]))  # Two
        stats.update(np.array([4.0, 5.0, 6.0, 7.0, 8.0]))  # Five

        result = stats.finalize()

        full_data = np.arange(1.0, 9.0)
        expected_mean = np.mean(full_data)
        expected_var = np.var(full_data, ddof=1)

        assert result.count == 8
        assert abs(result.mean - expected_mean) < 1e-10
        assert abs(result.variance - expected_var) < 1e-10

    def test_zero_values(self, stats: StreamingStats) -> None:
        """Test with zero values."""
        data = np.array([0.0, 0.0, 0.0])
        stats.update(data)
        result = stats.finalize()

        assert result.count == 3
        assert result.mean == 0.0
        assert result.variance == 0.0
        assert result.min == 0.0
        assert result.max == 0.0

    def test_floating_point_precision(self, stats: StreamingStats) -> None:
        """Test maintains precision with small differences."""
        # Values very close together
        base = 1000000.0
        data = np.array([base, base + 0.001, base + 0.002, base + 0.003])
        stats.update(data)
        result = stats.finalize()

        expected_mean = np.mean(data)
        assert abs(result.mean - expected_mean) < 1e-9
        assert result.variance > 0  # Should detect small variance

    def test_nan_handling(self, stats: StreamingStats) -> None:
        """Test behavior with NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0])
        stats.update(data)
        result = stats.finalize()

        # NaN propagates
        assert result.count == 4
        # Mean will be NaN due to NaN in data
        assert np.isnan(result.mean) or result.mean != result.mean

    def test_inf_handling(self, stats: StreamingStats) -> None:
        """Test behavior with infinity values."""
        data = np.array([1.0, 2.0, np.inf, 4.0])

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            stats.update(data)
            result = stats.finalize()

        assert result.count == 4
        assert result.max == np.inf
        # Mean will be inf
        assert np.isinf(result.mean)

    def test_multidimensional_array_flattened(self, stats: StreamingStats) -> None:
        """Test 2D array is flattened correctly."""
        data_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats.update(data_2d)
        result = stats.finalize()

        # Should flatten to [1, 2, 3, 4]
        assert result.count == 4
        assert result.mean == 2.5
        assert result.min == 1.0
        assert result.max == 4.0

    def test_data_type_conversion(self, stats: StreamingStats) -> None:
        """Test integer data is converted to float."""
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        stats.update(data)
        result = stats.finalize()

        assert result.count == 5
        assert result.mean == 3.0
        assert isinstance(result.mean, float)

    def test_list_input_converted(self, stats: StreamingStats) -> None:
        """Test Python list input is converted to array."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats.update(data)  # type: ignore[arg-type]
        result = stats.finalize()

        assert result.count == 5
        assert abs(result.mean - 3.0) < 1e-10


class TestStreamingStatsComparison:
    """Compare streaming algorithm with numpy batch computation."""

    def test_random_normal_distribution(self) -> None:
        """Test accuracy on normal distribution."""
        np.random.seed(123)
        data = np.random.randn(10000)

        # Streaming
        stats = StreamingStats()
        stats.update(data)
        result = stats.finalize()

        # Numpy
        expected_mean = np.mean(data)
        expected_var = np.var(data, ddof=1)
        expected_std = np.std(data, ddof=1)

        assert abs(result.mean - expected_mean) < 1e-10
        assert abs(result.variance - expected_var) < 1e-8
        assert abs(result.std - expected_std) < 1e-8

    def test_uniform_distribution(self) -> None:
        """Test accuracy on uniform distribution."""
        np.random.seed(456)
        data = np.random.uniform(-10, 10, 5000)

        stats = StreamingStats()
        stats.update(data)
        result = stats.finalize()

        expected_mean = np.mean(data)
        expected_var = np.var(data, ddof=1)

        assert abs(result.mean - expected_mean) < 1e-10
        assert abs(result.variance - expected_var) < 1e-8

    def test_exponential_distribution(self) -> None:
        """Test accuracy on exponential distribution (skewed)."""
        np.random.seed(789)
        data = np.random.exponential(scale=2.0, size=5000)

        stats = StreamingStats()
        stats.update(data)
        result = stats.finalize()

        expected_mean = np.mean(data)
        expected_var = np.var(data, ddof=1)

        assert abs(result.mean - expected_mean) < 1e-10
        assert abs(result.variance - expected_var) < 1e-6


class TestStreamingStatsWelfordsAlgorithm:
    """Test Welford's algorithm specific properties."""

    def test_welford_intermediate_state(self) -> None:
        """Test intermediate state is valid after each update."""
        stats = StreamingStats()

        # Add values one at a time and check state
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        for i, val in enumerate(values, 1):
            stats.update(np.array([val]))

            # State should be valid
            assert stats.count == i
            assert np.isfinite(stats.mean)
            assert np.isfinite(stats.m2) or stats.m2 == 0.0

    def test_welford_delta_delta2_calculation(self) -> None:
        """Test the delta and delta2 calculation in Welford's algorithm."""
        stats = StreamingStats()

        # Manually verify Welford's algorithm steps
        stats.update(np.array([10.0]))
        assert stats.count == 1
        assert stats.mean == 10.0
        assert stats.m2 == 0.0

        stats.update(np.array([20.0]))
        assert stats.count == 2
        # delta = 20 - 10 = 10, mean = 10 + 10/2 = 15
        assert stats.mean == 15.0
        # delta2 = 20 - 15 = 5, m2 = 0 + 10*5 = 50
        assert stats.m2 == 50.0
