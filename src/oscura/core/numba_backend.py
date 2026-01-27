"""Numba JIT compilation backend for performance-critical code paths.

This module provides a unified interface for Numba JIT compilation with graceful
fallback when Numba is not available. Provides 10-50x speedup for numerical loops
that cannot be fully vectorized.

Usage:
    from oscura.core.numba_backend import njit, prange, HAS_NUMBA

    @njit(parallel=True, cache=True)
    def fast_function(data):
        result = np.zeros_like(data)
        for i in prange(len(data)):
            result[i] = expensive_computation(data[i])
        return result

Performance characteristics:
    - First call: Compilation overhead (~100-500ms)
    - Subsequent calls: 10-50x faster than Python loops
    - Parallel execution: Additional speedup on multi-core systems
    - Cache: Compilation results cached between runs

Example:
    >>> from oscura.core.numba_backend import njit, HAS_NUMBA
    >>> import numpy as np
    >>>
    >>> @njit(cache=True)
    >>> def sum_of_squares(arr):
    ...     total = 0.0
    ...     for i in range(len(arr)):
    ...         total += arr[i] ** 2
    ...     return total
    >>>
    >>> data = np.random.randn(1_000_000)
    >>> result = sum_of_squares(data)  # Fast on second call
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

# Try to import Numba
try:
    from numba import guvectorize as _numba_guvectorize
    from numba import jit as _numba_jit
    from numba import njit as _numba_njit
    from numba import prange as _numba_prange
    from numba import vectorize as _numba_vectorize

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# Type variable for generic functions
F = TypeVar("F", bound=Callable[..., Any])


if HAS_NUMBA:
    # Numba is available - use real implementations
    njit = _numba_njit
    prange = _numba_prange
    vectorize = _numba_vectorize
    guvectorize = _numba_guvectorize
    jit = _numba_jit

else:
    # Numba not available - provide fallback decorators that do nothing
    def njit(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        """No-op decorator when Numba is not available.

        This decorator does nothing but allows code to remain syntactically valid
        when Numba is not installed.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Decorator function or decorated function.
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
                return func(*call_args, **call_kwargs)

            return wrapper  # type: ignore[return-value]

        # Handle both @njit and @njit() syntax
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return decorator(args[0])  # type: ignore[no-any-return]
        return decorator

    def prange(*args: Any, **kwargs: Any) -> range:
        """Fallback to regular range when Numba is not available.

        Args:
            *args: Same as range().
            **kwargs: Same as range().

        Returns:
            Standard Python range object.
        """
        return range(*args, **kwargs)

    def vectorize(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        """No-op decorator when Numba is not available.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Decorator that returns the original function.
        """

        def decorator(func: F) -> F:
            return func

        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])  # type: ignore[no-any-return]
        return decorator

    def guvectorize(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        """No-op decorator when Numba is not available.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Decorator that returns the original function.
        """

        def decorator(func: F) -> F:
            return func

        if len(args) == 1 and callable(args[0]):
            return decorator(args[0])  # type: ignore[no-any-return]
        return decorator

    def jit(*args: Any, **kwargs: Any) -> Callable[[F], F]:
        """No-op decorator when Numba is not available.

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).

        Returns:
            Decorator that returns the original function.
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            def wrapper(*call_args: Any, **call_kwargs: Any) -> Any:
                return func(*call_args, **call_kwargs)

            return wrapper  # type: ignore[return-value]

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return decorator(args[0])  # type: ignore[no-any-return]
        return decorator


def get_optimal_numba_config(
    parallel: bool = False,
    cache: bool = True,
    fastmath: bool = False,
    nogil: bool = False,
) -> dict[str, Any]:
    """Get optimal Numba configuration for given requirements.

    Args:
        parallel: Enable parallel execution using prange.
        cache: Enable compilation caching for faster subsequent runs.
        fastmath: Enable fast math optimizations (may reduce precision).
        nogil: Release GIL during execution (useful for threading).

    Returns:
        Dictionary of Numba configuration options.

    Example:
        >>> config = get_optimal_numba_config(parallel=True, cache=True)
        >>> @njit(**config)
        >>> def my_function(data):
        ...     pass
    """
    if not HAS_NUMBA:
        return {}

    return {
        "parallel": parallel,
        "cache": cache,
        "fastmath": fastmath,
        "nogil": nogil,
    }


# Example Numba-optimized functions for common operations


@njit(cache=True)  # type: ignore[untyped-decorator]  # Numba JIT decorator
def find_crossings_numba(
    data: np.ndarray,  # type: ignore[type-arg]
    threshold: float,
    direction: int = 0,
) -> np.ndarray:  # type: ignore[type-arg]
    """Find threshold crossings with Numba acceleration.

    Args:
        data: Input signal data.
        threshold: Threshold value to detect crossings.
        direction: 0=both, 1=rising only, -1=falling only.

    Returns:
        Array of indices where crossings occur.
    """
    crossings = []
    for i in range(1, len(data)):
        prev_val = data[i - 1]
        curr_val = data[i]

        if direction >= 0:  # Rising or both
            if prev_val < threshold <= curr_val:
                crossings.append(i)
        if direction <= 0 and direction != 1:  # Falling or both
            if prev_val > threshold >= curr_val:
                crossings.append(i)

    return np.array(crossings, dtype=np.int64)


@njit(parallel=True, cache=True)  # type: ignore[untyped-decorator]  # Numba JIT decorator
def moving_average_numba(
    data: np.ndarray,  # type: ignore[type-arg]
    window_size: int,
) -> np.ndarray:  # type: ignore[type-arg]
    """Compute moving average with Numba parallel acceleration.

    Args:
        data: Input signal data.
        window_size: Size of the moving window.

    Returns:
        Array of moving averages.
    """
    n = len(data)
    result = np.zeros(n - window_size + 1, dtype=np.float64)

    for i in prange(len(result)):
        total = 0.0
        for j in range(window_size):
            total += data[i + j]
        result[i] = total / window_size

    return result


@njit(cache=True)  # type: ignore[untyped-decorator]  # Numba JIT decorator
def argrelextrema_numba(
    data: np.ndarray,  # type: ignore[type-arg]
    comparator: int,
    order: int = 1,
) -> np.ndarray:  # type: ignore[type-arg]
    """Find relative extrema (peaks/valleys) with Numba acceleration.

    Args:
        data: Input signal data.
        comparator: 1 for maxima (peaks), -1 for minima (valleys).
        order: How many points on each side to use for comparison.

    Returns:
        Array of indices where extrema occur.
    """
    extrema = []
    n = len(data)

    for i in range(order, n - order):
        is_extremum = True

        for j in range(1, order + 1):
            if comparator > 0:  # Maximum
                if data[i] <= data[i - j] or data[i] <= data[i + j]:
                    is_extremum = False
                    break
            else:  # Minimum
                if data[i] >= data[i - j] or data[i] >= data[i + j]:
                    is_extremum = False
                    break

        if is_extremum:
            extrema.append(i)

    return np.array(extrema, dtype=np.int64)


@njit(cache=True)  # type: ignore[untyped-decorator]  # Numba JIT decorator
def interpolate_linear_numba(
    x: np.ndarray,  # type: ignore[type-arg]
    y: np.ndarray,  # type: ignore[type-arg]
    x_new: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Linear interpolation with Numba acceleration.

    Args:
        x: Original x coordinates (must be sorted).
        y: Original y values.
        x_new: New x coordinates to interpolate.

    Returns:
        Interpolated y values at x_new.
    """
    n = len(x)
    m = len(x_new)
    y_new = np.zeros(m, dtype=np.float64)

    for i in range(m):
        xi = x_new[i]

        # Binary search for bracketing indices
        left = 0
        right = n - 1

        while left < right - 1:
            mid = (left + right) // 2
            if x[mid] <= xi:
                left = mid
            else:
                right = mid

        # Linear interpolation
        if xi <= x[0]:
            y_new[i] = y[0]
        elif xi >= x[n - 1]:
            y_new[i] = y[n - 1]
        else:
            x0, x1 = x[left], x[right]
            y0, y1 = y[left], y[right]
            t = (xi - x0) / (x1 - x0)
            y_new[i] = y0 + t * (y1 - y0)

    return y_new


__all__ = [
    "HAS_NUMBA",
    "argrelextrema_numba",
    "find_crossings_numba",
    "get_optimal_numba_config",
    "guvectorize",
    "interpolate_linear_numba",
    "jit",
    "moving_average_numba",
    "njit",
    "prange",
    "vectorize",
]
