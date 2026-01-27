"""Tests for Numba-accelerated timing functions.

Validates correctness and performance of JIT-compiled timing calculations.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.digital import timing_numba


@pytest.mark.unit
@pytest.mark.performance
class TestTimingNumba:
    """Test Numba-accelerated timing functions."""

    def test_compute_delays_basic(self) -> None:
        """Test basic delay computation."""
        input_edges = np.array([1.0, 2.0, 3.0])
        output_edges = np.array([1.5, 2.5, 3.5])

        delays = timing_numba.compute_delays_fast(input_edges, output_edges)

        assert len(delays) == 3
        np.testing.assert_array_almost_equal(delays, [0.5, 0.5, 0.5])

    def test_compute_delays_no_output(self) -> None:
        """Test delays when no output edges after input."""
        input_edges = np.array([5.0, 6.0])
        output_edges = np.array([1.0, 2.0])

        delays = timing_numba.compute_delays_fast(input_edges, output_edges)

        assert len(delays) == 2
        assert np.all(np.isnan(delays))

    def test_compute_delays_mixed(self) -> None:
        """Test delays with some missing outputs."""
        input_edges = np.array([1.0, 3.0, 5.0])
        output_edges = np.array([1.5, 3.5])

        delays = timing_numba.compute_delays_fast(input_edges, output_edges)

        assert len(delays) == 3
        np.testing.assert_array_almost_equal(delays[:2], [0.5, 0.5])
        assert np.isnan(delays[2])

    def test_compute_setup_times_basic(self) -> None:
        """Test basic setup time computation."""
        data_edges = np.array([1.0, 2.0, 3.0])
        clock_edges = np.array([1.5, 2.5, 3.5])

        setup_times = timing_numba.compute_setup_times_fast(data_edges, clock_edges)

        assert len(setup_times) == 3
        np.testing.assert_array_almost_equal(setup_times, [0.5, 0.5, 0.5])

    def test_compute_setup_times_no_prior_data(self) -> None:
        """Test setup times when no prior data edges."""
        data_edges = np.array([5.0, 6.0])
        clock_edges = np.array([1.0, 2.0])

        setup_times = timing_numba.compute_setup_times_fast(data_edges, clock_edges)

        assert len(setup_times) == 2
        assert np.all(np.isnan(setup_times))

    def test_compute_hold_times_basic(self) -> None:
        """Test basic hold time computation."""
        data_edges = np.array([1.5, 2.5, 3.5])
        clock_edges = np.array([1.0, 2.0, 3.0])

        hold_times = timing_numba.compute_hold_times_fast(data_edges, clock_edges)

        assert len(hold_times) == 3
        np.testing.assert_array_almost_equal(hold_times, [0.5, 0.5, 0.5])

    def test_compute_hold_times_no_subsequent_data(self) -> None:
        """Test hold times when no subsequent data edges."""
        data_edges = np.array([1.0, 2.0])
        clock_edges = np.array([5.0, 6.0])

        hold_times = timing_numba.compute_hold_times_fast(data_edges, clock_edges)

        assert len(hold_times) == 2
        assert np.all(np.isnan(hold_times))

    def test_compute_delays_large_array(self) -> None:
        """Test delay computation with large arrays."""
        # Generate 1000 edges with wider spacing to avoid overlap
        input_edges = np.arange(0, 100, 0.1)  # 1000 edges, 0.1s apart
        output_edges = input_edges + 0.01  # Fixed 0.01s delay

        delays = timing_numba.compute_delays_fast(input_edges, output_edges)

        assert len(delays) == 1000
        # All delays should be approximately 0.01s
        assert np.allclose(delays, 0.01, rtol=1e-5, atol=1e-8)

    def test_compute_delays_empty_input(self) -> None:
        """Test delay computation with empty inputs."""
        delays = timing_numba.compute_delays_fast(np.array([]), np.array([1.0, 2.0]))

        assert len(delays) == 0

    def test_compute_delays_empty_output(self) -> None:
        """Test delay computation with empty outputs."""
        delays = timing_numba.compute_delays_fast(np.array([1.0, 2.0]), np.array([]))

        assert len(delays) == 2
        assert np.all(np.isnan(delays))

    def test_numba_availability(self) -> None:
        """Test that Numba status is tracked correctly."""
        # HAS_NUMBA flag should be set based on import success
        assert isinstance(timing_numba.HAS_NUMBA, bool)

        # Functions should work regardless of Numba availability
        input_edges = np.array([1.0, 2.0])
        output_edges = np.array([1.5, 2.5])

        delays = timing_numba.compute_delays_fast(input_edges, output_edges)
        assert len(delays) == 2
        np.testing.assert_array_almost_equal(delays, [0.5, 0.5])


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.skipif(not timing_numba.HAS_NUMBA, reason="Numba not installed")
class TestTimingNumbaPerformance:
    """Test performance characteristics of Numba JIT."""

    def test_delays_performance(self) -> None:
        """Test that delay computation is fast with Numba."""
        import time

        # Generate 100k edges
        input_edges = np.linspace(0, 100, 100000)
        output_edges = input_edges + 0.01

        # Warmup (JIT compilation)
        _ = timing_numba.compute_delays_fast(input_edges[:100], output_edges[:100])

        # Measure
        start = time.perf_counter()
        delays = timing_numba.compute_delays_fast(input_edges, output_edges)
        duration_ms = (time.perf_counter() - start) * 1000

        assert len(delays) == 100000
        # Should complete in <10ms with Numba (vs 30ms pure Python)
        assert duration_ms < 10.0, f"Too slow: {duration_ms:.1f}ms"

    def test_setup_times_performance(self) -> None:
        """Test that setup time computation is fast with Numba."""
        import time

        # Generate 100k edges
        data_edges = np.linspace(0, 100, 100000)
        clock_edges = data_edges + 0.01

        # Warmup
        _ = timing_numba.compute_setup_times_fast(data_edges[:100], clock_edges[:100])

        # Measure
        start = time.perf_counter()
        setup_times = timing_numba.compute_setup_times_fast(data_edges, clock_edges)
        duration_ms = (time.perf_counter() - start) * 1000

        assert len(setup_times) == 100000
        # Should complete in <10ms with Numba
        assert duration_ms < 10.0, f"Too slow: {duration_ms:.1f}ms"

    def test_compilation_caching(self) -> None:
        """Test that Numba caches compiled functions."""
        import time

        edges1 = np.array([1.0, 2.0, 3.0])
        edges2 = np.array([1.5, 2.5, 3.5])

        # First call (includes compilation)
        start1 = time.perf_counter()
        _ = timing_numba.compute_delays_fast(edges1, edges2)
        duration1_ms = (time.perf_counter() - start1) * 1000

        # Second call (cached)
        start2 = time.perf_counter()
        _ = timing_numba.compute_delays_fast(edges1, edges2)
        duration2_ms = (time.perf_counter() - start2) * 1000

        # Cached call should be much faster (<1ms vs potentially 100ms+)
        assert duration2_ms < 1.0, f"Cache not working: {duration2_ms:.1f}ms"


@pytest.mark.unit
class TestTimingNumbaFallback:
    """Test fallback behavior when Numba unavailable."""

    def test_fallback_works_without_numba(self) -> None:
        """Test that functions work even without Numba."""
        # Temporarily disable Numba
        original_has_numba = timing_numba.HAS_NUMBA

        try:
            timing_numba.HAS_NUMBA = False

            input_edges = np.array([1.0, 2.0, 3.0])
            output_edges = np.array([1.5, 2.5, 3.5])

            delays = timing_numba.compute_delays_fast(input_edges, output_edges)

            assert len(delays) == 3
            np.testing.assert_array_almost_equal(delays, [0.5, 0.5, 0.5])

        finally:
            timing_numba.HAS_NUMBA = original_has_numba
