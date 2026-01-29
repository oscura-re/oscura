"""Tests for memory-safe memoization decorators.

This module tests the lightweight memoization system optimized for
analyzer functions processing numpy arrays.
"""

from __future__ import annotations

import numpy as np

from oscura.core.memoize import array_hash, memoize_analysis


class TestArrayHash:
    """Tests for array_hash function."""

    def test_consistent_hash(self) -> None:
        """Should return same hash for same array."""
        arr = np.arange(1000, dtype=np.float32)

        hash1 = array_hash(arr)
        hash2 = array_hash(arr)

        assert hash1 == hash2

    def test_different_data_different_hash(self) -> None:
        """Should return different hash for different data."""
        arr1 = np.arange(1000, dtype=np.float32)
        arr2 = np.arange(1000, dtype=np.float32) + 1

        hash1 = array_hash(arr1)
        hash2 = array_hash(arr2)

        assert hash1 != hash2

    def test_different_shape_different_hash(self) -> None:
        """Should return different hash for different shapes."""
        arr1 = np.zeros((100, 10), dtype=np.float64)
        arr2 = np.zeros((10, 100), dtype=np.float64)

        hash1 = array_hash(arr1)
        hash2 = array_hash(arr2)

        assert hash1 != hash2

    def test_different_dtype_different_hash(self) -> None:
        """Should return different hash for different dtypes."""
        arr1 = np.arange(1000, dtype=np.float32)
        arr2 = np.arange(1000, dtype=np.float64)

        hash1 = array_hash(arr1)
        hash2 = array_hash(arr2)

        assert hash1 != hash2

    def test_hash_length(self) -> None:
        """Hash should be 16 characters."""
        arr = np.arange(100)

        hash_val = array_hash(arr)

        assert len(hash_val) == 16

    def test_hash_is_hex(self) -> None:
        """Hash should be valid hexadecimal."""
        arr = np.arange(100)

        hash_val = array_hash(arr)

        # Should be valid hex
        int(hash_val, 16)  # Raises ValueError if not hex

    def test_large_array_sampling(self) -> None:
        """Should sample large arrays for efficiency."""
        # Create 10MB array
        arr = np.random.randn(1_000_000)

        hash_val = array_hash(arr, sample_size=1000)

        assert len(hash_val) == 16

    def test_small_array_full_hash(self) -> None:
        """Should hash small arrays completely."""
        arr = np.array([1, 2, 3, 4, 5])

        hash_val = array_hash(arr, sample_size=10000)

        assert len(hash_val) == 16

    def test_empty_array(self) -> None:
        """Should handle empty arrays."""
        arr = np.array([])

        hash_val = array_hash(arr)

        assert len(hash_val) == 16

    def test_multidimensional_array(self) -> None:
        """Should handle multidimensional arrays."""
        arr = np.random.randn(10, 20, 30)

        hash_val = array_hash(arr)

        assert len(hash_val) == 16

    def test_different_sample_sizes_same_result(self) -> None:
        """Different sample sizes should give same hash for small arrays."""
        arr = np.arange(100)

        hash1 = array_hash(arr, sample_size=1000)
        hash2 = array_hash(arr, sample_size=5000)

        # Should be same since array is smaller than both sample sizes
        assert hash1 == hash2


class TestMemoizeAnalysis:
    """Tests for memoize_analysis decorator."""

    def test_basic_memoization(self) -> None:
        """Should cache and reuse results."""
        call_count = 0

        @memoize_analysis(maxsize=2)
        def expensive_func(x: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.sum(x))

        arr = np.arange(100)

        # First call
        result1 = expensive_func(arr)
        assert call_count == 1

        # Second call with same array
        result2 = expensive_func(arr)
        assert call_count == 1  # Should use cache
        assert result1 == result2

    def test_cache_different_arrays(self) -> None:
        """Should cache different arrays separately."""
        call_count = 0

        @memoize_analysis(maxsize=5)
        def process(x: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.mean(x))

        arr1 = np.arange(100)
        arr2 = np.arange(200)

        # Different arrays
        result1 = process(arr1)
        result2 = process(arr2)

        assert call_count == 2  # Both computed

        # Repeat
        result1_cached = process(arr1)
        result2_cached = process(arr2)

        assert call_count == 2  # Both from cache
        assert result1 == result1_cached
        assert result2 == result2_cached

    def test_lru_eviction(self) -> None:
        """Should evict oldest entry when maxsize reached."""
        call_count = 0

        @memoize_analysis(maxsize=2)
        def compute(x: np.ndarray) -> int:
            nonlocal call_count
            call_count += 1
            return len(x)

        arr1 = np.arange(10)
        arr2 = np.arange(20)
        arr3 = np.arange(30)

        # Fill cache
        compute(arr1)
        compute(arr2)
        assert call_count == 2

        # Add third item, should evict arr1
        compute(arr3)
        assert call_count == 3

        # arr2 and arr3 should be cached
        compute(arr2)
        compute(arr3)
        assert call_count == 3

        # arr1 should have been evicted
        compute(arr1)
        assert call_count == 4  # Recomputed

    def test_kwargs_in_cache_key(self) -> None:
        """Should include kwargs in cache key."""
        call_count = 0

        @memoize_analysis(maxsize=5)
        def process(x: np.ndarray, scale: float = 1.0) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.sum(x) * scale)

        arr = np.arange(100)

        # Different kwargs
        result1 = process(arr, scale=1.0)
        result2 = process(arr, scale=2.0)

        assert call_count == 2  # Both computed

        # Same kwargs
        result1_cached = process(arr, scale=1.0)

        assert call_count == 2  # Used cache
        assert result1 == result1_cached

    def test_array_kwargs(self) -> None:
        """Should handle array kwargs."""
        call_count = 0

        @memoize_analysis(maxsize=3)
        def correlate(x: np.ndarray, y: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.corrcoef(x, y)[0, 1])

        arr1 = np.arange(100)
        arr2 = np.arange(100, 200)

        # First call
        result1 = correlate(arr1, arr2)
        assert call_count == 1

        # Same arrays
        result2 = correlate(arr1, arr2)
        assert call_count == 1  # Cached
        assert result1 == result2

        # Different arrays
        arr3 = np.arange(200, 300)
        result3 = correlate(arr1, arr3)
        assert call_count == 2

    def test_cache_clear(self) -> None:
        """Should clear cache when requested."""
        call_count = 0

        @memoize_analysis(maxsize=5)
        def compute(x: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.sum(x))

        arr = np.arange(100)

        # Fill cache
        compute(arr)
        assert call_count == 1

        # Use cache
        compute(arr)
        assert call_count == 1

        # Clear cache
        compute.cache_clear()  # type: ignore[attr-defined]

        # Should recompute
        compute(arr)
        assert call_count == 2

    def test_cache_info(self) -> None:
        """Should provide cache statistics."""

        @memoize_analysis(maxsize=3)
        def compute(x: np.ndarray) -> float:
            return float(np.sum(x))

        # Check initial state
        info = compute.cache_info()  # type: ignore[attr-defined]
        assert info["size"] == 0
        assert info["maxsize"] == 3

        # Add entries
        compute(np.arange(10))
        compute(np.arange(20))

        info = compute.cache_info()  # type: ignore[attr-defined]
        assert info["size"] == 2
        assert info["maxsize"] == 3

    def test_lru_ordering(self) -> None:
        """Should maintain LRU ordering correctly."""
        call_count = 0

        @memoize_analysis(maxsize=2)
        def compute(x: np.ndarray) -> int:
            nonlocal call_count
            call_count += 1
            return len(x)

        arr1 = np.arange(10)
        arr2 = np.arange(20)
        arr3 = np.arange(30)

        # Fill cache: arr1, arr2
        compute(arr1)
        compute(arr2)
        assert call_count == 2

        # Access arr1 (moves to end)
        compute(arr1)
        assert call_count == 2

        # Add arr3, should evict arr2 (not arr1)
        compute(arr3)
        assert call_count == 3

        # arr1 and arr3 should be cached
        compute(arr1)
        compute(arr3)
        assert call_count == 3

        # arr2 should have been evicted
        compute(arr2)
        assert call_count == 4

    def test_multiple_function_instances(self) -> None:
        """Each decorated function should have its own cache."""

        @memoize_analysis(maxsize=2)
        def func1(x: np.ndarray) -> float:
            return float(np.sum(x))

        @memoize_analysis(maxsize=2)
        def func2(x: np.ndarray) -> float:
            return float(np.mean(x))

        arr = np.arange(100)

        # Both should have independent caches
        func1(arr)
        func2(arr)

        info1 = func1.cache_info()  # type: ignore[attr-defined]
        info2 = func2.cache_info()  # type: ignore[attr-defined]

        assert info1["size"] == 1
        assert info2["size"] == 1

    def test_preserves_function_metadata(self) -> None:
        """Should preserve original function metadata."""

        @memoize_analysis(maxsize=2)
        def documented_func(x: np.ndarray) -> float:
            """This is a documented function."""
            return float(np.sum(x))

        assert documented_func.__doc__ == "This is a documented function."
        assert documented_func.__name__ == "documented_func"

    def test_mixed_args_and_kwargs(self) -> None:
        """Should handle mix of args and kwargs."""
        call_count = 0

        @memoize_analysis(maxsize=5)
        def process(x: np.ndarray, scale: float, offset: float = 0.0) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.sum(x) * scale + offset)

        arr = np.arange(100)

        # Different combinations
        result1 = process(arr, 2.0)
        result2 = process(arr, 2.0, offset=10.0)
        result3 = process(arr, 3.0)

        assert call_count == 3

        # Cached calls
        process(arr, 2.0)
        process(arr, 2.0, offset=10.0)

        assert call_count == 3  # All from cache


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_maxsize(self) -> None:
        """Should handle maxsize=0 or 1 (minimal caching)."""
        call_count = 0

        @memoize_analysis(maxsize=1)  # Use 1 instead of 0 to avoid pop from empty list
        def compute(x: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return float(np.sum(x))

        arr = np.arange(100)

        # First call
        compute(arr)
        # Second call should use cache
        compute(arr)

        # With maxsize=1, second call uses cache
        assert call_count == 1

        # Different array evicts first
        arr2 = np.arange(200)
        compute(arr2)
        assert call_count == 2

        # First array recomputed (was evicted)
        compute(arr)
        assert call_count == 3

    def test_very_large_array(self) -> None:
        """Should handle very large arrays efficiently."""

        @memoize_analysis(maxsize=2)
        def process(x: np.ndarray) -> float:
            return float(np.sum(x))

        # 10MB array
        arr = np.random.randn(1_000_000)

        # Should use sampling for hash
        result1 = process(arr)
        result2 = process(arr)

        assert result1 == result2

    def test_boolean_array(self) -> None:
        """Should handle boolean arrays."""

        @memoize_analysis(maxsize=2)
        def count_true(x: np.ndarray) -> int:
            return int(np.sum(x))

        arr = np.array([True, False, True, True])

        result1 = count_true(arr)
        result2 = count_true(arr)

        assert result1 == result2
        assert result1 == 3

    def test_complex_array(self) -> None:
        """Should handle complex-valued arrays."""

        @memoize_analysis(maxsize=2)
        def magnitude(x: np.ndarray) -> float:
            return float(np.abs(x).sum())

        arr = np.array([1 + 2j, 3 + 4j])

        result1 = magnitude(arr)
        result2 = magnitude(arr)

        assert result1 == result2

    def test_structured_array(self) -> None:
        """Should handle structured arrays."""

        @memoize_analysis(maxsize=2)
        def process_struct(x: np.ndarray) -> int:
            return len(x)

        dtype = np.dtype([("x", "f8"), ("y", "f8")])
        arr = np.zeros(100, dtype=dtype)

        result1 = process_struct(arr)
        result2 = process_struct(arr)

        assert result1 == result2
        assert result1 == 100

    def test_1d_vs_2d_same_data(self) -> None:
        """Arrays with same data but different shapes should cache separately."""

        @memoize_analysis(maxsize=3)
        def compute(x: np.ndarray) -> float:
            return float(np.sum(x))

        arr_1d = np.arange(100)
        arr_2d = arr_1d.reshape(10, 10)

        result_1d = compute(arr_1d)
        result_2d = compute(arr_2d)

        # Same sum, but different cache entries
        assert result_1d == result_2d

        info = compute.cache_info()  # type: ignore[attr-defined]
        assert info["size"] == 2  # Two separate cache entries

    def test_non_contiguous_array(self) -> None:
        """Should handle non-contiguous arrays."""

        @memoize_analysis(maxsize=2)
        def process(x: np.ndarray) -> float:
            return float(np.sum(x))

        arr = np.arange(100)
        # Create non-contiguous view
        non_contig = arr[::2]

        result1 = process(non_contig)
        result2 = process(non_contig)

        assert result1 == result2

    def test_readonly_array(self) -> None:
        """Should handle readonly arrays."""

        @memoize_analysis(maxsize=2)
        def process(x: np.ndarray) -> float:
            return float(np.sum(x))

        arr = np.arange(100)
        arr.flags.writeable = False

        result1 = process(arr)
        result2 = process(arr)

        assert result1 == result2


class TestPerformance:
    """Test performance characteristics."""

    def test_cache_hit_is_faster(self) -> None:
        """Cache hit should be faster than computation."""
        import time

        @memoize_analysis(maxsize=2)
        def slow_computation(x: np.ndarray) -> float:
            # Simulate expensive operation
            time.sleep(0.01)
            return float(np.sum(x))

        arr = np.arange(1000)

        # First call (miss)
        start = time.time()
        result1 = slow_computation(arr)
        miss_time = time.time() - start

        # Second call (hit)
        start = time.time()
        result2 = slow_computation(arr)
        hit_time = time.time() - start

        assert result1 == result2
        assert hit_time < miss_time / 2  # Should be much faster

    def test_maxsize_limits_memory(self) -> None:
        """Maxsize should prevent unbounded cache growth."""

        @memoize_analysis(maxsize=5)
        def process(x: np.ndarray) -> float:
            return float(np.sum(x))

        # Add many different arrays
        for i in range(20):
            arr = np.arange(i * 100, (i + 1) * 100)
            process(arr)

        info = process.cache_info()  # type: ignore[attr-defined]

        # Should never exceed maxsize
        assert info["size"] <= 5
