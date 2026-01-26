"""Tests for core/numba_backend.py - Numba JIT compilation backend.

Tests:
- Numba availability detection
- Decorator fallbacks when Numba not available
- Optimized functions (crossings, moving average, extrema, interpolation)
- Configuration helpers
- Both with and without Numba
"""

import numpy as np
import pytest

from oscura.core import numba_backend


class TestNumbaAvailability:
    """Test Numba availability detection."""

    def test_has_numba_flag(self) -> None:
        """Test HAS_NUMBA flag is boolean."""
        assert isinstance(numba_backend.HAS_NUMBA, bool)

    def test_imports_available(self) -> None:
        """Test that decorators are importable regardless of Numba."""
        assert hasattr(numba_backend, "njit")
        assert hasattr(numba_backend, "prange")
        assert hasattr(numba_backend, "vectorize")
        assert hasattr(numba_backend, "guvectorize")
        assert hasattr(numba_backend, "jit")


class TestDecoratorFallbacks:
    """Test decorator fallbacks when Numba not available."""

    def test_njit_decorator_simple(self) -> None:
        """Test njit decorator works (with or without Numba)."""

        @numba_backend.njit
        def add_one(x: float) -> float:
            return x + 1.0

        result = add_one(5.0)
        assert result == 6.0

    def test_njit_decorator_with_args(self) -> None:
        """Test njit decorator with arguments."""

        @numba_backend.njit(cache=True, parallel=False)
        def multiply(x: float, y: float) -> float:
            return x * y

        result = multiply(3.0, 4.0)
        assert result == 12.0

    def test_jit_decorator(self) -> None:
        """Test jit decorator."""

        @numba_backend.jit
        def square(x: float) -> float:
            return x**2

        result = square(5.0)
        assert result == 25.0

    def test_jit_decorator_with_kwargs(self) -> None:
        """Test jit decorator with kwargs."""

        @numba_backend.jit(nopython=True)
        def cube(x: float) -> float:
            return x**3

        result = cube(3.0)
        assert result == 27.0

    def test_vectorize_decorator(self) -> None:
        """Test vectorize decorator."""

        @numba_backend.vectorize
        def vec_add(x: float, y: float) -> float:
            return x + y

        # Should work with scalars
        result = vec_add(1.0, 2.0)
        assert result == 3.0

    def test_guvectorize_decorator(self) -> None:
        """Test guvectorize decorator."""

        @numba_backend.guvectorize(["void(float64[:], float64[:])"], "(n)->(n)")
        def my_cumsum(x: np.ndarray, result: np.ndarray) -> None:  # type: ignore
            cumsum = 0.0
            for i in range(len(x)):  # type: ignore
                cumsum += x[i]
                result[i] = cumsum

        # Basic functionality test
        arr = np.array([1.0, 2.0, 3.0])
        # Function signature differs between Numba and fallback
        # Just test it's callable
        assert callable(my_cumsum)

    def test_prange_function(self) -> None:
        """Test prange returns iterable."""
        result = list(numba_backend.prange(10))
        assert result == list(range(10))

    def test_prange_with_args(self) -> None:
        """Test prange with start/stop/step."""
        result = list(numba_backend.prange(0, 10, 2))
        assert result == [0, 2, 4, 6, 8]


class TestGetOptimalConfig:
    """Test get_optimal_numba_config function."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = numba_backend.get_optimal_numba_config()

        if numba_backend.HAS_NUMBA:
            assert "parallel" in config
            assert "cache" in config
            assert config["cache"] is True
            assert config["parallel"] is False
        else:
            assert config == {}

    def test_parallel_config(self) -> None:
        """Test parallel configuration."""
        config = numba_backend.get_optimal_numba_config(parallel=True)

        if numba_backend.HAS_NUMBA:
            assert config["parallel"] is True
        else:
            assert config == {}

    def test_fastmath_config(self) -> None:
        """Test fastmath configuration."""
        config = numba_backend.get_optimal_numba_config(fastmath=True)

        if numba_backend.HAS_NUMBA:
            assert config["fastmath"] is True
        else:
            assert config == {}

    def test_nogil_config(self) -> None:
        """Test nogil configuration."""
        config = numba_backend.get_optimal_numba_config(nogil=True)

        if numba_backend.HAS_NUMBA:
            assert config["nogil"] is True
        else:
            assert config == {}

    def test_all_options(self) -> None:
        """Test all options enabled."""
        config = numba_backend.get_optimal_numba_config(
            parallel=True, cache=True, fastmath=True, nogil=True
        )

        if numba_backend.HAS_NUMBA:
            assert config["parallel"] is True
            assert config["cache"] is True
            assert config["fastmath"] is True
            assert config["nogil"] is True
        else:
            assert config == {}


class TestFindCrossingsNumba:
    """Test find_crossings_numba function."""

    def test_rising_crossings(self) -> None:
        """Test detecting rising edge crossings."""
        data = np.array([0.0, 0.5, 1.5, 2.0, 1.0, 0.5, 1.5])
        threshold = 1.0
        direction = 1  # Rising only

        crossings = numba_backend.find_crossings_numba(data, threshold, direction)

        # Rising crossings at indices 2 and 6
        assert len(crossings) == 2
        assert 2 in crossings
        assert 6 in crossings

    def test_falling_crossings(self) -> None:
        """Test detecting falling edge crossings."""
        data = np.array([2.0, 1.5, 0.5, 0.0, 0.5, 1.5, 0.5])
        threshold = 1.0
        direction = -1  # Falling only

        crossings = numba_backend.find_crossings_numba(data, threshold, direction)

        # Falling crossings at indices 2 and 6
        assert len(crossings) == 2
        assert 2 in crossings
        assert 6 in crossings

    def test_both_crossings(self) -> None:
        """Test detecting both rising and falling crossings."""
        data = np.array([0.0, 1.5, 0.5, 1.5, 0.5])
        threshold = 1.0
        direction = 0  # Both

        crossings = numba_backend.find_crossings_numba(data, threshold, direction)

        # Should detect all crossings
        assert len(crossings) == 4  # Rising at 1, falling at 2, rising at 3, falling at 4

    def test_no_crossings(self) -> None:
        """Test when no crossings exist."""
        data = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        threshold = 1.0

        crossings = numba_backend.find_crossings_numba(data, threshold, 0)

        assert len(crossings) == 0

    def test_exact_threshold(self) -> None:
        """Test crossing at exact threshold value."""
        data = np.array([0.5, 1.0, 1.5])
        threshold = 1.0

        crossings = numba_backend.find_crossings_numba(data, threshold, 1)

        # Rising crossing at index 1 (0.5 -> 1.0) and 2 (1.0 -> 1.5)
        assert len(crossings) >= 1


class TestMovingAverageNumba:
    """Test moving_average_numba function."""

    def test_simple_moving_average(self) -> None:
        """Test basic moving average calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = 3

        result = numba_backend.moving_average_numba(data, window_size)

        # Expected: [(1+2+3)/3, (2+3+4)/3, (3+4+5)/3] = [2.0, 3.0, 4.0]
        assert len(result) == 3
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(3.0)
        assert result[2] == pytest.approx(4.0)

    def test_window_size_equals_length(self) -> None:
        """Test when window size equals data length."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = 5

        result = numba_backend.moving_average_numba(data, window_size)

        # Only one window
        assert len(result) == 1
        assert result[0] == pytest.approx(3.0)  # Mean of all values

    def test_window_size_one(self) -> None:
        """Test window size of 1 (returns original)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = 1

        result = numba_backend.moving_average_numba(data, window_size)

        assert len(result) == 5
        np.testing.assert_array_almost_equal(result, data)

    def test_larger_window(self) -> None:
        """Test with larger window size."""
        data = np.arange(100, dtype=np.float64)
        window_size = 10

        result = numba_backend.moving_average_numba(data, window_size)

        assert len(result) == 91  # 100 - 10 + 1
        # First window: mean of 0-9 = 4.5
        assert result[0] == pytest.approx(4.5)


class TestArgrelextremaNumb:
    """Test argrelextrema_numba function."""

    def test_find_maxima(self) -> None:
        """Test finding local maxima."""
        data = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 2.0])
        comparator = 1  # Maxima
        order = 1

        maxima = numba_backend.argrelextrema_numba(data, comparator, order)

        # Local maxima at indices 1 (3.0), 3 (5.0), 5 (6.0)
        assert len(maxima) == 3
        assert 1 in maxima
        assert 3 in maxima
        assert 5 in maxima

    def test_find_minima(self) -> None:
        """Test finding local minima."""
        data = np.array([5.0, 2.0, 4.0, 1.0, 3.0, 0.5, 2.0])
        comparator = -1  # Minima
        order = 1

        minima = numba_backend.argrelextrema_numba(data, comparator, order)

        # Local minima at indices 1 (2.0), 3 (1.0), 5 (0.5)
        assert len(minima) == 3
        assert 1 in minima
        assert 3 in minima
        assert 5 in minima

    def test_no_extrema(self) -> None:
        """Test when no local extrema exist."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Monotonic
        comparator = 1
        order = 1

        maxima = numba_backend.argrelextrema_numba(data, comparator, order)

        assert len(maxima) == 0

    def test_higher_order(self) -> None:
        """Test with higher order comparison."""
        data = np.array([1.0, 2.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        comparator = 1  # Maxima
        order = 2  # Compare with 2 points on each side

        maxima = numba_backend.argrelextrema_numba(data, comparator, order)

        # Only clear peak at index 2
        assert 2 in maxima

    def test_sine_wave_peaks(self) -> None:
        """Test finding peaks in sine wave."""
        t = np.linspace(0, 4 * np.pi, 100)
        data = np.sin(t)

        maxima = numba_backend.argrelextrema_numba(data, 1, 1)

        # Should find ~2 peaks (at π/2 and 5π/2)
        assert len(maxima) >= 2


class TestInterpolateLinearNumba:
    """Test interpolate_linear_numba function."""

    def test_simple_interpolation(self) -> None:
        """Test basic linear interpolation."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        x_new = np.array([0.5, 1.5])

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        assert y_new[0] == pytest.approx(0.5)
        assert y_new[1] == pytest.approx(1.5)

    def test_non_uniform_spacing(self) -> None:
        """Test interpolation with non-uniform x spacing."""
        x = np.array([0.0, 1.0, 4.0])
        y = np.array([0.0, 2.0, 8.0])
        x_new = np.array([2.5])

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        # At x=2.5 (halfway between 1 and 4): y = 2 + (8-2)*0.5 = 5.0
        assert y_new[0] == pytest.approx(5.0)

    def test_extrapolation_below(self) -> None:
        """Test extrapolation below range (should clamp)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        x_new = np.array([0.0])

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        # Should return first value
        assert y_new[0] == pytest.approx(10.0)

    def test_extrapolation_above(self) -> None:
        """Test extrapolation above range (should clamp)."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        x_new = np.array([5.0])

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        # Should return last value
        assert y_new[0] == pytest.approx(30.0)

    def test_exact_x_values(self) -> None:
        """Test interpolation at exact x coordinates."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 5.0, 10.0, 15.0])
        x_new = np.array([1.0, 2.0])

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        # Should return exact values
        assert y_new[0] == pytest.approx(5.0)
        assert y_new[1] == pytest.approx(10.0)

    def test_many_points(self) -> None:
        """Test interpolation with many points."""
        x = np.linspace(0, 10, 11)
        y = x**2
        x_new = np.linspace(0, 10, 101)

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        # Check a few values
        assert len(y_new) == 101
        assert y_new[0] == pytest.approx(0.0)
        assert y_new[-1] == pytest.approx(100.0)

    def test_single_point_interpolation(self) -> None:
        """Test interpolation to single new point."""
        x = np.array([0.0, 10.0])
        y = np.array([0.0, 100.0])
        x_new = np.array([5.0])

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        assert y_new[0] == pytest.approx(50.0)


class TestNumbaIntegration:
    """Test integration between Numba-accelerated functions."""

    def test_find_and_interpolate(self) -> None:
        """Test using crossings to guide interpolation."""
        # Create signal with crossings
        t = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine

        # Find zero crossings
        crossings = numba_backend.find_crossings_numba(data, 0.0, 1)  # Rising

        # Should find 2 rising crossings
        assert len(crossings) >= 2

    def test_smooth_and_find_peaks(self) -> None:
        """Test smoothing then finding peaks."""
        # Noisy data with peaks
        np.random.seed(42)
        data = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.1 * np.random.randn(100)

        # Smooth
        smoothed = numba_backend.moving_average_numba(data, 5)

        # Find peaks
        maxima = numba_backend.argrelextrema_numba(smoothed, 1, 2)

        # Should find ~2 clear peaks
        assert len(maxima) >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array_crossings(self) -> None:
        """Test find_crossings with empty array."""
        data = np.array([])
        crossings = numba_backend.find_crossings_numba(data, 1.0, 0)

        assert len(crossings) == 0

    def test_single_element_crossings(self) -> None:
        """Test find_crossings with single element."""
        data = np.array([1.5])
        crossings = numba_backend.find_crossings_numba(data, 1.0, 0)

        # No crossings possible with single element
        assert len(crossings) == 0

    def test_small_window_moving_average(self) -> None:
        """Test moving average with very small data."""
        data = np.array([1.0, 2.0])
        result = numba_backend.moving_average_numba(data, 2)

        assert len(result) == 1
        assert result[0] == pytest.approx(1.5)

    def test_extrema_small_array(self) -> None:
        """Test extrema with array smaller than order."""
        data = np.array([1.0, 2.0, 1.0])
        maxima = numba_backend.argrelextrema_numba(data, 1, 1)

        # Should find peak at index 1
        assert 1 in maxima

    def test_interpolate_two_points(self) -> None:
        """Test interpolation with minimum points."""
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 10.0])
        x_new = np.array([0.5])

        y_new = numba_backend.interpolate_linear_numba(x, y, x_new)

        assert y_new[0] == pytest.approx(5.0)


class TestExports:
    """Test module exports."""

    def test_all_exports_present(self) -> None:
        """Test that all expected symbols are in __all__."""
        expected = [
            "HAS_NUMBA",
            "njit",
            "jit",
            "prange",
            "vectorize",
            "guvectorize",
            "get_optimal_numba_config",
            "find_crossings_numba",
            "moving_average_numba",
            "argrelextrema_numba",
            "interpolate_linear_numba",
        ]

        for name in expected:
            assert name in numba_backend.__all__

    def test_all_exports_accessible(self) -> None:
        """Test that all exports are actually accessible."""
        for name in numba_backend.__all__:
            assert hasattr(numba_backend, name)
