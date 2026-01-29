"""Progress tracking and cancellation support for Oscura operations.

This module provides progress callbacks, cancellation tokens, and memory warnings
for long-running operations.


Example:
    >>> from oscura.core.progress import ProgressCallback, CancellationToken
    >>> token = CancellationToken()
    >>> def progress_fn(current, total, message):
    ...     print(f"{current}/{total}: {message}")
    >>> # Use in analysis functions
    >>> result = analyze(data, progress_callback=progress_fn, cancel_token=token)

References:
    - WCAG 2.1 progress indication guidelines
    - Python threading and multiprocessing best practices
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Protocol

# Lazy import for optional system monitoring
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

if TYPE_CHECKING:
    from collections.abc import Callable


class ProgressCallback(Protocol):
    """Protocol for progress callback functions.

    : Progress callback parameter on all analysis functions.
    Callback receives (current, total, message) for progress reporting.

    Args:
        current: Current progress value (e.g., samples processed)
        total: Total expected value (e.g., total samples)
        message: Descriptive message about current operation

    Example:
        >>> def my_progress(current: int, total: int, message: str) -> None:
        ...     percent = 100 * current / total
        ...     print(f"{percent:.1f}%: {message}")

    References:
        PROG-001: Progress Indication for Long Operations
    """

    def __call__(self, current: int, total: int, message: str) -> None:
        """Progress callback signature.

        Args:
            current: Current progress (completed items)
            total: Total items to process
            message: Status message
        """
        ...


class CancellationToken:
    """Token for cancelling long-running operations.

    : Cancellation Support - cancel() method on operation handles.
    Allows graceful cancellation of operations with Ctrl+C support.

    Attributes:
        cancelled: Whether cancellation has been requested
        message: Optional cancellation message

    Example:
        >>> from oscura.core.progress import CancellationToken, CancelledError
        >>> token = CancellationToken()
        >>> # In analysis function:
        >>> for i in range(n_samples):
        ...     if token.is_cancelled():
        ...         raise CancelledError("Analysis cancelled by user")
        ...     # ... process sample ...

    References:
        PROG-002: Cancellation Support
    """

    def __init__(self) -> None:
        """Initialize cancellation token."""
        self._cancelled: bool = False
        self._message: str = ""
        self._cancelled_at: float | None = None

    def cancel(self, message: str = "Operation cancelled") -> None:
        """Request cancellation of the operation.

        Args:
            message: Reason for cancellation (default: "Operation cancelled")

        Example:
            >>> token = CancellationToken()
            >>> token.cancel("User requested stop")
            >>> assert token.is_cancelled()

        References:
            PROG-002: Cancellation Support
        """
        self._cancelled = True
        self._message = message
        self._cancelled_at = time.time()

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if operation should be cancelled

        Example:
            >>> token = CancellationToken()
            >>> if token.is_cancelled():
            ...     return  # Exit early

        References:
            PROG-002: Cancellation Support
        """
        return self._cancelled

    def check(self) -> None:
        """Check cancellation status and raise if cancelled.

        Raises:
            CancelledError: If cancellation has been requested

        Example:
            >>> token = CancellationToken()
            >>> token.cancel()
            >>> token.check()  # Raises CancelledError

        References:
            PROG-002: Cancellation Support
        """
        if self._cancelled:
            raise CancelledError(self._message)

    @property
    def message(self) -> str:
        """Get cancellation message.

        Returns:
            Cancellation message

        References:
            PROG-002: Cancellation Support
        """
        return self._message

    @property
    def cancelled_at(self) -> float | None:
        """Get timestamp when cancellation was requested.

        Returns:
            Timestamp in seconds since epoch, or None if not cancelled

        References:
            PROG-002: Cancellation Support
        """
        return self._cancelled_at


class CancelledError(Exception):
    """Exception raised when operation is cancelled.

    : Partial results available after cancellation.
    Operations can catch this to save partial results before exiting.

    Attributes:
        message: Reason for cancellation
        progress: Progress percentage at cancellation (0-100)

    Example:
        >>> from oscura.core.progress import CancelledError
        >>> try:
        ...     # ... long operation ...
        ...     raise CancelledError("User cancelled", progress=45.5)
        ... except CancelledError as e:
        ...     print(f"Cancelled at {e.progress}%: {e.message}")

    References:
        PROG-002: Cancellation Support
    """

    def __init__(self, message: str, *, progress: float = 0.0) -> None:
        """Initialize CancelledError.

        Args:
            message: Reason for cancellation
            progress: Progress percentage at cancellation (default: 0.0)
        """
        self.message = message
        self.progress = progress
        super().__init__(f"{message} ({progress:.1f}% complete)")


def create_progress_tracker(
    total: int,
    *,
    callback: Callable[[int, int, str], None] | None = None,
    update_interval: float = 0.1,
) -> ProgressTracker:
    """Create a progress tracker for an operation.

    : Progress callback receives (current, total, eta_seconds).
    Automatically calculates ETA and throttles updates.

    Args:
        total: Total number of items to process
        callback: Optional progress callback function
        update_interval: Minimum time between updates in seconds (default: 0.1)

    Returns:
        ProgressTracker instance

    Example:
        >>> from oscura.core.progress import create_progress_tracker
        >>> tracker = create_progress_tracker(1000, callback=my_progress)
        >>> for i in range(1000):
        ...     tracker.update(i + 1, "Processing item")

    References:
        PROG-001: Progress Indication for Long Operations
    """
    return ProgressTracker(total, callback=callback, update_interval=update_interval)


class ProgressTracker:
    """Progress tracker with ETA calculation and throttling.

    : Callback receives (current, total, eta_seconds).
    Tracks progress and calculates estimated time to completion.

    Args:
        total: Total number of items
        callback: Optional progress callback
        update_interval: Minimum seconds between updates

    Example:
        >>> from oscura.core.progress import ProgressTracker
        >>> tracker = ProgressTracker(1000)
        >>> for i in range(1000):
        ...     tracker.update(i + 1, "Processing")
        >>> tracker.finish("Complete")

    References:
        PROG-001: Progress Indication for Long Operations
    """

    def __init__(
        self,
        total: int,
        *,
        callback: Callable[[int, int, str], None] | None = None,
        update_interval: float = 0.1,
    ) -> None:
        """Initialize progress tracker.

        Args:
            total: Total items to process
            callback: Progress callback function
            update_interval: Minimum seconds between updates
        """
        self.total = total
        self.current = 0
        self.callback = callback
        self.update_interval = update_interval

        self._start_time = time.time()
        self._last_update_time = 0.0
        self._finished = False

    def update(self, current: int, message: str = "") -> None:
        """Update progress.

        Args:
            current: Current progress value
            message: Status message

        Example:
            >>> tracker.update(500, "Halfway done")

        References:
            PROG-001: Progress Indication for Long Operations
        """
        self.current = current

        # Throttle updates
        now = time.time()
        if now - self._last_update_time < self.update_interval:
            return

        self._last_update_time = now

        if self.callback:
            self.callback(current, self.total, message)

    def get_eta(self) -> float:
        """Calculate estimated time to completion.

        Returns:
            Estimated seconds remaining

        Example:
            >>> tracker.update(500, "Processing")
            >>> eta = tracker.get_eta()
            >>> print(f"ETA: {eta:.1f} seconds")

        References:
            PROG-001: Progress Indication for Long Operations
        """
        if self.current == 0:
            return 0.0

        elapsed = time.time() - self._start_time
        rate = self.current / elapsed
        remaining = self.total - self.current

        if rate > 0:
            return remaining / rate
        else:
            return 0.0

    def get_progress_percent(self) -> float:
        """Get progress as percentage.

        Returns:
            Progress percentage (0-100)

        Example:
            >>> tracker.update(250, "Processing")
            >>> print(f"Progress: {tracker.get_progress_percent():.1f}%")

        References:
            PROG-001: Progress Indication for Long Operations
        """
        if self.total == 0:
            return 100.0
        return 100.0 * self.current / self.total

    def finish(self, message: str = "Complete") -> None:
        """Mark operation as finished.

        Args:
            message: Completion message (default: "Complete")

        Example:
            >>> tracker.finish("Analysis complete")

        References:
            PROG-001: Progress Indication for Long Operations
        """
        self._finished = True
        self.current = self.total

        if self.callback:
            self.callback(self.total, self.total, message)


def estimate_memory_usage(
    n_samples: int,
    dtype_bytes: int = 8,
    *,
    n_channels: int = 1,
    scratch_multiplier: float = 2.0,
) -> int:
    """Estimate memory usage for an operation.

    : Estimate memory before large FFT/spectrograms.
    Calculates expected memory consumption including scratch space.

    Args:
        n_samples: Number of samples
        dtype_bytes: Bytes per sample (default: 8 for float64)
        n_channels: Number of channels (default: 1)
        scratch_multiplier: Multiplier for temporary arrays (default: 2.0)

    Returns:
        Estimated memory usage in bytes

    Example:
        >>> from oscura.core.progress import estimate_memory_usage
        >>> memory_bytes = estimate_memory_usage(1_000_000, dtype_bytes=8)
        >>> memory_mb = memory_bytes / (1024 ** 2)
        >>> print(f"Estimated: {memory_mb:.1f} MB")

    References:
        PROG-003: Memory Usage Warnings
    """
    # Base array size
    base_size = n_samples * dtype_bytes * n_channels

    # Include scratch space for operations (e.g., FFT)
    total_size = int(base_size * scratch_multiplier)

    return total_size


def check_memory_available(required_bytes: int, *, threshold: float = 0.8) -> bool:
    """Check if sufficient memory is available.

    : Warn if estimated > 80% of available RAM.
    Checks system memory availability before large operations.

    Args:
        required_bytes: Required memory in bytes
        threshold: Maximum fraction of available RAM to use (default: 0.8)

    Returns:
        True if sufficient memory is available

    Example:
        >>> from oscura.core.progress import check_memory_available
        >>> required = 1024 * 1024 * 1024  # 1 GB
        >>> if not check_memory_available(required):
        ...     print("Warning: Insufficient memory")

    References:
        PROG-003: Memory Usage Warnings
    """
    # If psutil not available, assume memory is sufficient
    if not _HAS_PSUTIL:
        return True

    memory = psutil.virtual_memory()
    available_bytes = memory.available
    threshold_bytes = available_bytes * threshold

    return required_bytes <= threshold_bytes  # type: ignore[no-any-return]


def warn_memory_usage(
    required_bytes: int,
    *,
    threshold: float = 0.8,
    suggest_chunked: bool = True,
) -> None:
    """Warn if operation may exceed available memory.

    : Warn before operations that may exceed available memory.
    Issues warning and suggests chunked processing if needed.

    Args:
        required_bytes: Required memory in bytes
        threshold: Maximum fraction of available RAM (default: 0.8)
        suggest_chunked: Suggest chunked processing (default: True)

    Example:
        >>> from oscura.core.progress import warn_memory_usage
        >>> required = estimate_memory_usage(10_000_000)
        >>> warn_memory_usage(required)

    References:
        PROG-003: Memory Usage Warnings
    """
    # If psutil not available, skip memory warning
    if not _HAS_PSUTIL:
        return

    memory = psutil.virtual_memory()
    available_bytes = memory.available
    threshold_bytes = available_bytes * threshold

    required_mb = required_bytes / (1024**2)
    available_mb = available_bytes / (1024**2)
    threshold_mb = threshold_bytes / (1024**2)

    if required_bytes > threshold_bytes:
        message = (
            f"Warning: Operation may require {required_mb:.1f} MB of memory, "
            f"but only {available_mb:.1f} MB is available "
            f"(threshold: {threshold_mb:.1f} MB)."
        )

        if suggest_chunked:
            message += " Consider using chunked processing or reducing the data size."

        warnings.warn(message, ResourceWarning, stacklevel=2)


def create_simple_progress(
    message_prefix: str = "Progress",
) -> Callable[[int, int, str], None]:
    """Create a simple text-based progress callback.

    : CLI shows progress bar for long operations.
    Returns a callback that prints progress to stdout.

    Args:
        message_prefix: Prefix for progress messages (default: "Progress")

    Returns:
        Progress callback function

    Example:
        >>> from oscura.core.progress import create_simple_progress
        >>> callback = create_simple_progress("Loading")
        >>> for i in range(100):
        ...     callback(i + 1, 100, "Processing")

    References:
        PROG-001: Progress Indication for Long Operations
    """

    def callback(current: int, total: int, message: str) -> None:
        percent = 100 * current / total if total > 0 else 0
        status = f"{message_prefix}: {percent:.1f}% ({current}/{total})"
        if message:
            status += f" - {message}"
        print(f"\r{status}", end="", flush=True)
        if current >= total:
            print()  # New line when complete

    return callback


__all__ = [
    "CancellationToken",
    "CancelledError",
    "ProgressCallback",
    "ProgressTracker",
    "check_memory_available",
    "create_progress_tracker",
    "create_simple_progress",
    "estimate_memory_usage",
    "warn_memory_usage",
]
