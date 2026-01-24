"""Buffer utilities for streaming data.

This module provides circular buffer implementation for
streaming data with O(1) operations.


Example:
    >>> from oscura.utils.buffer import CircularBuffer
    >>> buf = CircularBuffer(1000)
    >>> buf.append(value)
    >>> recent = buf.get_last(100)

References:
    Ring buffer data structure
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

T = TypeVar("T")


class CircularBuffer(Generic[T]):
    """Fixed-size circular buffer with O(1) operations.

    Thread-safe for single producer, single consumer pattern.

    Args:
        capacity: Maximum buffer size.
        dtype: NumPy dtype for numeric buffers.

    Attributes:
        capacity: Buffer capacity.
        count: Current number of items.

    Example:
        >>> buf = CircularBuffer(1000, dtype=np.float64)
        >>> for value in stream:
        ...     buf.append(value)
        ...     if buf.is_full():
        ...         process(buf.get_all())
    """

    def __init__(
        self,
        capacity: int,
        dtype: DTypeLike | None = None,
    ) -> None:
        """Initialize circular buffer.

        Args:
            capacity: Maximum buffer size.
            dtype: NumPy dtype. If None, uses object array.
        """
        self._capacity = capacity
        self._dtype = dtype

        if dtype is not None:
            self._data: NDArray[Any] = np.zeros(capacity, dtype=dtype)
        else:
            self._data = np.empty(capacity, dtype=object)

        self._head = 0  # Next write position
        self._count = 0  # Number of valid items

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def count(self) -> int:
        """Get current item count."""
        return self._count

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._count == 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._count == self._capacity

    def append(self, value: T) -> None:
        """Append value to buffer.

        O(1) operation. Overwrites oldest value if full.

        Args:
            value: Value to append.
        """
        self._data[self._head] = value
        self._head = (self._head + 1) % self._capacity

        if self._count < self._capacity:
            self._count += 1

    def extend(self, values: list[T] | NDArray[Any]) -> None:
        """Extend buffer with multiple values.

        Args:
            values: Values to append.
        """
        for value in values:
            self.append(value)

    def get_last(self, n: int = 1) -> NDArray[Any]:
        """Get last n items.

        Args:
            n: Number of items (default 1).

        Returns:
            Array of last n items (newest first).
        """
        n = min(n, self._count)

        if n == 0:
            if self._dtype is not None:
                return np.array([], dtype=self._dtype)
            return np.array([], dtype=object)

        result = np.empty(n, dtype=self._data.dtype)

        for i in range(n):
            idx = (self._head - 1 - i) % self._capacity
            result[i] = self._data[idx]

        return result

    def get_first(self, n: int = 1) -> NDArray[Any]:
        """Get first n items (oldest).

        Args:
            n: Number of items.

        Returns:
            Array of first n items (oldest first).
        """
        n = min(n, self._count)

        if n == 0:
            if self._dtype is not None:
                return np.array([], dtype=self._dtype)
            return np.array([], dtype=object)

        # Calculate tail position (oldest item)
        tail = (self._head - self._count) % self._capacity

        result = np.empty(n, dtype=self._data.dtype)

        for i in range(n):
            idx = (tail + i) % self._capacity
            result[i] = self._data[idx]

        return result

    def get_all(self) -> NDArray[Any]:
        """Get all items in order.

        Returns:
            Array of all items (oldest first).
        """
        return self.get_first(self._count)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> NDArray[Any]: ...

    def __getitem__(self, index: int | slice) -> T | NDArray[Any]:
        """Get item(s) by index.

        Positive indices count from oldest (0 = oldest).
        Negative indices count from newest (-1 = newest).

        Args:
            index: Integer index or slice.

        Returns:
            Item or array of items.

        Raises:
            IndexError: If index out of range.
        """
        if isinstance(index, slice):
            # Convert slice to indices
            start, stop, step = index.indices(self._count)
            result = []
            for i in range(start, stop, step):
                result.append(self[i])
            return np.array(result, dtype=self._data.dtype)

        if index < 0:
            index = self._count + index

        if index < 0 or index >= self._count:
            raise IndexError(f"Index {index} out of range [0, {self._count})")

        # Calculate actual position
        tail = (self._head - self._count) % self._capacity
        actual_idx = (tail + index) % self._capacity

        return self._data[actual_idx]  # type: ignore[no-any-return]

    def __len__(self) -> int:
        """Get current item count."""
        return self._count

    def clear(self) -> None:
        """Clear all items."""
        self._head = 0
        self._count = 0

    def mean(self) -> float:
        """Compute mean of numeric buffer.

        Returns:
            Mean value, or NaN if empty.
        """
        if self._count == 0:
            return float("nan")

        return float(np.mean(self.get_all()))

    def std(self) -> float:
        """Compute standard deviation.

        Returns:
            Standard deviation, or NaN if empty.
        """
        if self._count < 2:
            return float("nan")

        return float(np.std(self.get_all()))

    def min(self) -> T:
        """Get minimum value.

        Returns:
            Minimum value.

        Raises:
            ValueError: If buffer is empty.
        """
        if self._count == 0:
            raise ValueError("Buffer is empty")

        return np.min(self.get_all())  # type: ignore[no-any-return]

    def max(self) -> T:
        """Get maximum value.

        Returns:
            Maximum value.

        Raises:
            ValueError: If buffer is empty.
        """
        if self._count == 0:
            raise ValueError("Buffer is empty")

        return np.max(self.get_all())  # type: ignore[no-any-return]


class SlidingWindow:
    """Sliding window for time-series analysis.

    Maintains a window of samples based on time or count.

    Args:
        window_size: Window size in samples or seconds.
        time_based: If True, window_size is in seconds.

    Example:
        >>> window = SlidingWindow(1000)  # 1000 samples
        >>> for sample, time in stream:
        ...     window.add(sample, time)
        ...     if window.is_ready():
        ...         result = analyze(window.get_data())
    """

    def __init__(
        self,
        window_size: int | float,
        time_based: bool = False,
        dtype: DTypeLike = np.float64,
    ) -> None:
        """Initialize sliding window.

        Args:
            window_size: Size in samples or seconds.
            time_based: True for time-based window.
            dtype: Data type for samples.
        """
        self._window_size = window_size
        self._time_based = time_based

        if time_based:
            # Use large buffer for time-based
            capacity = 100000
        else:
            capacity = int(window_size)

        self._data = CircularBuffer(capacity, dtype=dtype)  # type: ignore[var-annotated]
        self._times = CircularBuffer(capacity, dtype=np.float64)  # type: ignore[var-annotated]

    def add(self, value: float, timestamp: float | None = None) -> None:
        """Add sample to window.

        Args:
            value: Sample value.
            timestamp: Sample timestamp (required for time-based).

        Raises:
            ValueError: If timestamp is None for time-based window.
        """
        self._data.append(value)

        if self._time_based:
            if timestamp is None:
                raise ValueError("Timestamp required for time-based window")
            self._times.append(timestamp)

    def is_ready(self) -> bool:
        """Check if window is full."""
        if self._time_based:
            if self._times.count < 2:
                return False
            times = self._times.get_all()
            duration = times[-1] - times[0]
            return duration >= self._window_size  # type: ignore[no-any-return]
        else:
            return self._data.count >= self._window_size

    def get_data(self) -> NDArray[np.float64]:
        """Get window data.

        Returns:
            Array of samples in window.
        """
        if self._time_based:
            # Get samples within time window
            times = self._times.get_all()
            data = self._data.get_all()

            if len(times) == 0:
                return np.array([], dtype=np.float64)

            cutoff = times[-1] - self._window_size
            mask = times >= cutoff

            result: NDArray[np.float64] = data[mask]
            return result
        else:
            result_all: NDArray[np.float64] = self._data.get_all()
            return result_all

    def get_times(self) -> NDArray[np.float64]:
        """Get timestamps for time-based window.

        Returns:
            Array of timestamps.

        Raises:
            ValueError: If not a time-based window.
        """
        if not self._time_based:
            raise ValueError("Not a time-based window")

        return self._times.get_all()

    def clear(self) -> None:
        """Clear window."""
        self._data.clear()
        self._times.clear()


__all__ = [
    "CircularBuffer",
    "SlidingWindow",
]
