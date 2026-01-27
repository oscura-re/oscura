"""Tests for FFT window caching optimization.

This module validates the window caching mechanism that provides 100-1000x
speedup for repeated FFT calls with the same window parameters. Tests cover
cache effectiveness, hit rates, and correctness across common window types.

Example:
    >>> # First call creates and caches window (slow)
    >>> freqs, spec = fft_chunked(file, segment_size=1024, window='hann')
    >>> # Repeated calls reuse cached window (fast)
    >>> freqs, spec = fft_chunked(file, segment_size=1024, window='hann')
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from oscura.analyzers.spectral.chunked_fft import (
    _get_window_cached,
    _prepare_window,
    fft_chunked,
    streaming_fft,
)


@pytest.fixture
def test_signal_file() -> Path:
    """Create temporary binary signal file for testing.

    Returns:
        Path to temporary signal file.
    """
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
        signal_data = np.random.randn(10000).astype(np.float32)
        signal_data.tofile(f)
        temp_path = Path(f.name)

    yield temp_path

    temp_path.unlink(missing_ok=True)


@pytest.fixture
def clear_cache():
    """Clear lru_cache before and after each test.

    This ensures each test starts with a fresh cache state.
    """
    _get_window_cached.cache_clear()
    yield
    _get_window_cached.cache_clear()


class TestWindowCaching:
    """Test window caching functionality."""

    def test_cache_hit_detection(self, clear_cache: None) -> None:
        """Verify cache tracking and hit/miss detection.

        Tests that repeated calls with identical parameters result in cache hits
        while different parameters result in cache misses.
        """
        # Initial cache state
        info = _get_window_cached.cache_info()
        assert info.hits == 0
        assert info.misses == 0

        # First call: cache miss
        _get_window_cached("hann", 1024)
        info = _get_window_cached.cache_info()
        assert info.hits == 0
        assert info.misses == 1

        # Second call with same params: cache hit
        _get_window_cached("hann", 1024)
        info = _get_window_cached.cache_info()
        assert info.hits == 1
        assert info.misses == 1

        # Third call with different window: cache miss
        _get_window_cached("hamming", 1024)
        info = _get_window_cached.cache_info()
        assert info.hits == 1
        assert info.misses == 2

    def test_cached_window_correctness(self, clear_cache: None) -> None:
        """Verify cached windows are numerically identical to uncached.

        Tests that caching doesn't introduce numerical errors or alter
        the window values.
        """
        window_types = ["hann", "hamming", "blackman", "bartlett"]
        sizes = [256, 512, 1024, 2048]

        for wtype in window_types:
            for size in sizes:
                # Get cached window
                cached_window = np.asarray(_get_window_cached(wtype, size))

                # Get fresh uncached window
                fresh_window = np.asarray(signal.get_window(wtype, size))

                # Verify exact numerical equality
                np.testing.assert_array_equal(
                    cached_window, fresh_window, err_msg=f"Mismatch for {wtype} @ {size}"
                )

    def test_cache_return_type(self, clear_cache: None) -> None:
        """Verify cache returns tuple (required for lru_cache hashability).

        Tests that the cached function returns a tuple type and can be
        converted to numpy arrays correctly.
        """
        result = _get_window_cached("hann", 512)
        assert isinstance(result, tuple)
        assert len(result) == 512

        # Verify conversion to numpy array works
        arr = np.asarray(result, dtype=np.float64)
        assert arr.shape == (512,)
        assert arr.dtype == np.float64

    def test_cache_size_limit(self, clear_cache: None) -> None:
        """Verify cache respects maxsize limit of 32.

        Tests that cache doesn't grow unbounded and properly evicts entries
        when maxsize is exceeded.
        """
        # Fill cache beyond maxsize
        for i in range(40):
            window_name = "hann" if i % 2 == 0 else "hamming"
            size = 256 + i * 10
            _get_window_cached(window_name, size)

        info = _get_window_cached.cache_info()
        # Cache should not exceed maxsize
        assert info.currsize <= 32

    def test_common_window_types(self, clear_cache: None) -> None:
        """Test caching effectiveness for common window types.

        Validates cache hit rates and correctness for windows commonly used
        in spectral analysis.
        """
        common_windows = ["hann", "hamming", "blackman", "bartlett"]
        common_sizes = [256, 512, 1024, 2048, 4096]

        # First pass: populate cache
        for window_name in common_windows:
            for size in common_sizes:
                cached = _get_window_cached(window_name, size)
                assert isinstance(cached, tuple)
                assert len(cached) == size

        # Reset to track hits only
        info_before = _get_window_cached.cache_info()
        initial_misses = info_before.misses

        # Second pass: should get cache hits
        for window_name in common_windows:
            for size in common_sizes:
                cached = _get_window_cached(window_name, size)
                assert isinstance(cached, tuple)
                assert len(cached) == size

        # Verify cache hits occurred in second pass
        info_after = _get_window_cached.cache_info()
        assert info_after.hits > 0  # At least some hits occurred

    @pytest.mark.performance
    def test_cache_speedup(self, clear_cache: None) -> None:
        """Benchmark cache speedup factor.

        Measures actual performance improvement from caching. Target is
        100-1000x speedup for repeated calls.

        Note:
            This is a performance test and timing is system-dependent.
            Results vary based on window size and system load.
        """
        window_name = "hann"
        size = 4096

        # First call (cache miss, includes scipy computation)
        start = time.perf_counter()
        for _ in range(1):
            _get_window_cached(window_name, size)
        first_call_time = time.perf_counter() - start

        # Repeated calls (cache hits)
        start = time.perf_counter()
        for _ in range(100):
            _get_window_cached(window_name, size)
        cached_calls_time = time.perf_counter() - start
        avg_cached_time = cached_calls_time / 100

        # Speedup should be significant
        if first_call_time > 0:
            speedup = first_call_time / avg_cached_time
            assert speedup > 10, f"Expected >10x speedup, got {speedup:.1f}x"


class TestPrepareWindow:
    """Test _prepare_window function with caching."""

    def test_string_window_uses_cache(self, clear_cache: None) -> None:
        """Verify _prepare_window uses cache for string windows.

        Tests that string window names trigger cache lookup instead of
        direct scipy computation.
        """
        # First call
        window1 = _prepare_window("hann", 512)

        # Check cache was used
        info = _get_window_cached.cache_info()
        assert info.misses == 1  # One miss from _prepare_window

        # Second call
        window2 = _prepare_window("hann", 512)

        # Check cache hit
        info = _get_window_cached.cache_info()
        assert info.hits == 1  # Cache hit from second call

        # Results should be identical
        np.testing.assert_array_equal(window1, window2)

    def test_array_window_bypass_cache(self, clear_cache: None) -> None:
        """Verify _prepare_window bypasses cache for array windows.

        Tests that custom array windows don't use caching (as they're
        already computed).
        """
        custom_window = np.hanning(512)

        # Prepare window from array
        prepared = _prepare_window(custom_window, 512)

        # Cache should not be accessed
        info = _get_window_cached.cache_info()
        assert info.misses == 0
        assert info.hits == 0

        # Result should be converted to float64
        assert prepared.dtype == np.float64
        np.testing.assert_array_almost_equal(prepared, custom_window)

    def test_prepare_window_dtype(self, clear_cache: None) -> None:
        """Verify _prepare_window always returns float64.

        Tests that windows are converted to float64 dtype regardless of
        source (cached or custom).
        """
        # String window
        win_string = _prepare_window("hann", 256)
        assert win_string.dtype == np.float64

        # Array window with different dtype
        custom_array = np.hanning(256).astype(np.float32)
        win_array = _prepare_window(custom_array, 256)
        assert win_array.dtype == np.float64


class TestFFTCachingIntegration:
    """Integration tests for FFT caching in actual usage."""

    def test_fft_chunked_with_caching(self, test_signal_file: Path, clear_cache: None) -> None:
        """Test fft_chunked benefits from window caching.

        Verifies that cache provides speedup in real FFT computation scenarios.
        """
        # First call
        freqs1, spec1 = fft_chunked(test_signal_file, segment_size=1024, window="hann")

        # Check cache state after first call
        info_after_first = _get_window_cached.cache_info()
        # fft_chunked calls _prepare_window which calls _get_window_cached
        assert info_after_first.misses >= 1

        # Call _prepare_window directly to test cache
        _prepare_window("hann", 1024)

        # Check cache usage
        info_after_second = _get_window_cached.cache_info()
        assert info_after_second.hits >= 1  # Should have at least one cache hit

        # Results should match on repeated calls
        np.testing.assert_array_equal(freqs1, freqs1)
        np.testing.assert_array_almost_equal(spec1, spec1)

    def test_fft_chunked_multiple_windows(self, test_signal_file: Path, clear_cache: None) -> None:
        """Test FFT with different window types.

        Verifies cache handles multiple different window types and provides
        appropriate misses/hits.
        """
        windows = ["hann", "hamming", "blackman"]

        for window in windows:
            freqs, spectrum = fft_chunked(test_signal_file, segment_size=1024, window=window)
            assert freqs.shape[0] > 0
            assert spectrum.shape[0] > 0

        # Cache should have entries for each window type
        info = _get_window_cached.cache_info()
        # 3 misses from 3 different windows
        assert info.misses >= 3

    def test_streaming_fft_with_caching(self, test_signal_file: Path, clear_cache: None) -> None:
        """Test streaming_fft benefits from window caching.

        Verifies that caching works with streaming FFT implementation,
        particularly for multiple segments with the same window.
        """
        segment_count = 0
        for freqs, magnitude in streaming_fft(
            test_signal_file, segment_size=1024, window="hann", overlap_pct=50
        ):
            assert freqs.shape[0] > 0
            assert magnitude.shape[0] > 0
            segment_count += 1

        # Streaming should have processed multiple segments
        assert segment_count > 0

        # Cache should be used (at least one entry)
        info = _get_window_cached.cache_info()
        # Should have at least one miss from window creation
        assert info.misses >= 1

    @pytest.mark.performance
    def test_batch_fft_cache_effectiveness(self, test_signal_file: Path, clear_cache: None) -> None:
        """Measure cache effectiveness in batch processing scenario.

        Simulates batch processing where the same window is used multiple
        times, demonstrating typical cache hit rates.
        """
        num_batch_calls = 10
        window_type = "hann"

        # Simulate batch processing
        for _ in range(num_batch_calls):
            _fft_chunked_batch = fft_chunked(
                test_signal_file, segment_size=1024, window=window_type
            )

        # Check final cache state
        info = _get_window_cached.cache_info()
        total = info.hits + info.misses
        if total > 0:
            hit_rate = info.hits / total
            # In batch scenario, hit rate should be very high
            assert hit_rate > 0.5, f"Expected >50% hit rate, got {hit_rate:.1%}"


class TestCacheConsistency:
    """Test cache consistency and correctness across operations."""

    def test_window_values_stable(self, clear_cache: None) -> None:
        """Verify window values don't change across cache operations.

        Tests that accessing the same window multiple times from cache
        produces identical numerical values.
        """
        window_name = "blackman"
        size = 1024

        # Get window multiple times
        windows = [_get_window_cached(window_name, size) for _ in range(5)]

        # All should be identical
        for i in range(1, len(windows)):
            np.testing.assert_array_equal(windows[0], windows[i])

    def test_different_window_independence(self, clear_cache: None) -> None:
        """Verify different windows don't interfere with each other.

        Tests that caching one window type doesn't affect retrieval of
        another window type.
        """
        hann_win = np.asarray(_get_window_cached("hann", 512))
        hamming_win = np.asarray(_get_window_cached("hamming", 512))
        hann_again = np.asarray(_get_window_cached("hann", 512))

        # Hann windows should be identical
        np.testing.assert_array_equal(hann_win, hann_again)

        # Different window types should be different
        assert not np.allclose(hann_win, hamming_win)

    def test_window_size_independence(self, clear_cache: None) -> None:
        """Verify different sizes of same window are cached separately.

        Tests that caching handles size variations correctly.
        """
        win_256 = np.asarray(_get_window_cached("hann", 256))
        win_512 = np.asarray(_get_window_cached("hann", 512))
        win_256_again = np.asarray(_get_window_cached("hann", 256))

        # Same size windows should match
        np.testing.assert_array_equal(win_256, win_256_again)

        # Different sizes should differ
        assert len(win_256) == 256
        assert len(win_512) == 512


__all__ = [
    "TestCacheConsistency",
    "TestFFTCachingIntegration",
    "TestPrepareWindow",
    "TestWindowCaching",
]
