"""Lazy evaluation utilities for deferred computation.

This module provides lazy evaluation proxies that defer computation until
results are actually needed, enabling memory-efficient operation chaining.


Example:
    >>> from oscura.utils.lazy import LazyArray, lazy_operation
    >>> # Operations are deferred until .compute() is called
    >>> lazy_result = lazy_operation(large_data, lambda x: x ** 2)
    >>> result = lazy_result.compute()  # Only now is computation performed

References:
    Dask documentation on lazy evaluation
    NumPy lazy evaluation patterns
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")

if TYPE_CHECKING:
    from collections.abc import Callable


class LazyProxy(ABC, Generic[T]):
    """Abstract base class for lazy evaluation proxies.

    Defers computation until explicitly requested via .compute().
    """

    def __init__(self) -> None:
        self._computed: bool = False
        self._result: T | None = None

    @abstractmethod
    def _evaluate(self) -> T:
        """Perform the actual computation.

        Returns:
            Computed result.
        """

    def compute(self) -> T:
        """Evaluate and return the result.

        Returns:
            Computed result (cached after first evaluation).

        Example:
            >>> lazy_obj = LazyArray(lambda: np.arange(1000))
            >>> result = lazy_obj.compute()
        """
        if not self._computed:
            self._result = self._evaluate()
            self._computed = True
        return self._result  # type: ignore[return-value]

    def is_computed(self) -> bool:
        """Check if result has been computed.

        Returns:
            True if compute() has been called.
        """
        return self._computed

    def reset(self) -> None:
        """Clear cached result, forcing re-evaluation on next compute()."""
        self._computed = False
        self._result = None


class LazyArray(LazyProxy[NDArray[np.floating[Any]]]):
    """Lazy evaluation proxy for numpy arrays.

    Wraps a computation that returns a numpy array, deferring
    execution until the result is needed.

    Args:
        func: Callable that returns a numpy array.
        args: Positional arguments for func.
        kwargs: Keyword arguments for func.

    Example:
        >>> def expensive_computation():
        ...     return np.random.randn(1000000)
        >>> lazy = LazyArray(expensive_computation)
        >>> # No computation yet
        >>> result = lazy.compute()  # Now it runs
    """

    def __init__(
        self,
        func: Callable[..., NDArray[np.floating[Any]]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def _evaluate(self) -> NDArray[np.floating[Any]]:
        """Execute the deferred computation."""
        return self._func(*self._args, **self._kwargs)

    def __len__(self) -> int:
        """Get length (triggers computation)."""
        return len(self.compute())

    def __getitem__(self, key: Any) -> Any:
        """Get item (triggers computation)."""
        return self.compute()[key]

    def shape(self) -> tuple[int, ...]:
        """Get shape (triggers computation)."""
        return self.compute().shape

    def dtype(self) -> np.dtype[Any]:
        """Get dtype (triggers computation)."""
        return self.compute().dtype


class LazyOperation(LazyProxy[Any]):
    """Lazy evaluation of an operation on data.

    Chains operations without intermediate materialization.

    Args:
        operation: Callable that performs the operation.
        *operands: Input data or other lazy proxies.
        **kwargs: Keyword arguments for the operation.

    Example:
        >>> data = np.arange(1000)
        >>> # Chain operations without computing intermediate results
        >>> op1 = LazyOperation(lambda x: x ** 2, data)
        >>> op2 = LazyOperation(lambda x: x + 1, op1)
        >>> result = op2.compute()
    """

    def __init__(
        self,
        operation: Callable[..., Any],
        *operands: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._operation = operation
        self._operands = operands
        self._kwargs = kwargs

    def _evaluate(self) -> Any:
        """Evaluate the operation, computing operands if needed."""
        # Evaluate any lazy operands
        evaluated_operands = []
        for operand in self._operands:
            if isinstance(operand, LazyProxy):
                evaluated_operands.append(operand.compute())
            else:
                evaluated_operands.append(operand)

        return self._operation(*evaluated_operands, **self._kwargs)


def lazy_operation(
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> LazyOperation:
    """Create a lazy operation from a function.

    Args:
        func: Function to defer.
        *args: Arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        LazyOperation that will execute func when computed.

    Example:
        >>> import numpy as np
        >>> data = np.arange(1000)
        >>> lazy_result = lazy_operation(np.fft.fft, data)
        >>> # Computation happens here
        >>> result = lazy_result.compute()
    """
    return LazyOperation(func, *args, **kwargs)


def auto_preview(
    data: NDArray[np.floating[Any]],
    *,
    downsample_factor: int = 10,
    preview_only: bool = False,
) -> NDArray[np.float64]:
    """Generate preview of large dataset with automatic downsampling.

    Two-stage analysis: quick preview before full processing.

    Args:
        data: Input data array.
        downsample_factor: Factor to downsample by for preview (default 10).
        preview_only: If True, return only preview. If False, return full data.

    Returns:
        Preview (downsampled) or full data based on preview_only flag.

    Example:
        >>> import numpy as np
        >>> large_data = np.random.randn(10_000_000)
        >>> # Quick preview
        >>> preview = auto_preview(large_data, preview_only=True)
        >>> print(f"Preview shape: {preview.shape}")
        >>> # Full data
        >>> full = auto_preview(large_data, preview_only=False)

    References:
        MEM-026: Two-Stage Analysis (Preview + Full)
    """
    if preview_only or len(data) > 1_000_000:
        # Generate downsampled preview
        preview = data[::downsample_factor].copy()
        return preview.astype(np.float64)
    else:
        # Small enough, return full data
        return data.astype(np.float64)


def select_roi(
    data: NDArray[np.floating[Any]],
    start: int | None = None,
    end: int | None = None,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    sample_rate: float | None = None,
) -> NDArray[np.float64]:
    """Select region of interest from data.

    Allows selection by sample indices or time values.

    Args:
        data: Input data array.
        start: Start sample index (inclusive).
        end: End sample index (exclusive).
        start_time: Start time in seconds (alternative to start).
        end_time: End time in seconds (alternative to end).
        sample_rate: Sample rate in Hz (required if using time-based selection).

    Returns:
        Selected region of interest.

    Raises:
        ValueError: If time-based selection used without sample_rate.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(10_000_000)
        >>> # Select by sample indices
        >>> roi = select_roi(data, start=1000, end=2000)
        >>> # Select by time
        >>> roi_time = select_roi(
        ...     data, start_time=0.001, end_time=0.002, sample_rate=1e6
        ... )

    References:
        MEM-027: Region-of-Interest Selection from Preview
    """
    # Convert time-based to sample-based
    if start_time is not None or end_time is not None:
        if sample_rate is None:
            raise ValueError("sample_rate required for time-based selection")

        if start_time is not None:
            start = int(start_time * sample_rate)
        if end_time is not None:
            end = int(end_time * sample_rate)

    # Apply defaults
    if start is None:
        start = 0
    if end is None:
        end = len(data)

    # Validate and clip to bounds
    start = max(0, start)
    end = min(len(data), end)

    if start >= end:
        raise ValueError(f"Invalid ROI: start ({start}) >= end ({end})")

    # Extract region
    return data[start:end].astype(np.float64)


class ProgressiveResolution:
    """Progressive resolution analyzer for large datasets.

    Implements coarse-to-fine analysis: preview then zoom into ROI.

    Args:
        data: Input data array or lazy proxy.
        sample_rate: Sample rate in Hz.

    Example:
        >>> import numpy as np
        >>> data = np.random.randn(100_000_000)
        >>> analyzer = ProgressiveResolution(data, sample_rate=1e6)
        >>> # Stage 1: Preview
        >>> preview = analyzer.get_preview(downsample_factor=100)
        >>> # Stage 2: User selects ROI
        >>> roi_data = analyzer.get_roi(start_time=0.5, end_time=0.6)

    References:
        MEM-013: Progressive Resolution (Coarse-to-Fine)
    """

    def __init__(
        self,
        data: NDArray[np.floating[Any]] | LazyProxy[NDArray[np.floating[Any]]],
        sample_rate: float,
    ) -> None:
        self._data = data
        self._sample_rate = sample_rate
        self._preview: NDArray[np.float64] | None = None
        self._preview_factor: int | None = None

    def get_preview(
        self,
        downsample_factor: int = 10,
        force_recompute: bool = False,
    ) -> NDArray[np.float64]:
        """Generate low-resolution preview.

        Args:
            downsample_factor: Factor to downsample by.
            force_recompute: If True, recompute even if cached.

        Returns:
            Downsampled preview of data.
        """
        if self._preview is not None and not force_recompute:
            if self._preview_factor == downsample_factor:
                return self._preview

        # Get full data
        data = self._data.compute() if isinstance(self._data, LazyProxy) else self._data

        # Downsample
        self._preview = data[::downsample_factor].copy().astype(np.float64)
        self._preview_factor = downsample_factor

        return self._preview

    def get_roi(
        self,
        start: int | None = None,
        end: int | None = None,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> NDArray[np.float64]:
        """Get high-resolution region of interest.

        Args:
            start: Start sample index.
            end: End sample index.
            start_time: Start time in seconds (alternative).
            end_time: End time in seconds (alternative).

        Returns:
            Full-resolution ROI data.
        """
        # Get full data
        data = self._data.compute() if isinstance(self._data, LazyProxy) else self._data

        return select_roi(
            data,
            start=start,
            end=end,
            start_time=start_time,
            end_time=end_time,
            sample_rate=self._sample_rate,
        )

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        return self._sample_rate


__all__ = [
    "LazyArray",
    "LazyOperation",
    "LazyProxy",
    "ProgressiveResolution",
    "auto_preview",
    "lazy_operation",
    "select_roi",
]
