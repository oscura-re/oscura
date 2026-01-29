"""Enhanced cancellation support for Oscura operations.

This module provides advanced cancellation features including signal handling,
cleanup routines, and resume support for long-running operations.


Example:
    >>> from oscura.core.cancellation import CancellationManager
    >>> manager = CancellationManager()
    >>> with manager.cancellable_operation("Loading data"):
    ...     # ... long operation ...
    ...     manager.check_cancelled()

References:
    - Python threading best practices
    - Signal handling patterns
"""

from __future__ import annotations

import atexit
import signal
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class CancellationManager:
    """Manager for cancellable operations with cleanup support.

    : Ctrl+C handling, cleanup, and resume support.
    Provides graceful cancellation with automatic cleanup and signal handling.

    Args:
        cleanup_callback: Optional callback for cleanup on cancellation
        auto_cleanup: Automatically cleanup on exit (default: True)

    Example:
        >>> from oscura.core.cancellation import CancellationManager
        >>> def cleanup():
        ...     print("Cleaning up...")
        >>> manager = CancellationManager(cleanup_callback=cleanup)
        >>> manager.register_signal_handlers()
        >>> # Press Ctrl+C to trigger cancellation

    References:
        PROG-002: Cancellation Support
    """

    def __init__(
        self,
        *,
        cleanup_callback: Callable[[], None] | None = None,
        auto_cleanup: bool = True,
    ) -> None:
        """Initialize cancellation manager.

        Args:
            cleanup_callback: Function to call on cancellation
            auto_cleanup: Register cleanup at exit
        """
        self._cancelled = threading.Event()
        self._cleanup_callback = cleanup_callback
        self._cleanup_functions: list[Callable[[], None]] = []
        self._partial_results: dict[str, Any] = {}
        self._operation_name = ""
        self._start_time = 0.0
        self._signal_handlers_registered = False

        if auto_cleanup:
            atexit.register(self._cleanup)

    def register_signal_handlers(self) -> None:
        """Register signal handlers for Ctrl+C and SIGTERM.

        : Ctrl+C handling - graceful cancellation.
        Catches interrupt signals and triggers cancellation.

        Example:
            >>> manager.register_signal_handlers()
            >>> # Now Ctrl+C will trigger cancellation

        References:
            PROG-002: Ctrl+C handling
        """
        if self._signal_handlers_registered:
            return

        def signal_handler(signum: int, frame: Any) -> None:
            self.cancel(f"Received signal {signum}")

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        self._signal_handlers_registered = True

    def cancel(self, reason: str = "Operation cancelled") -> None:
        """Request cancellation of the operation.

        : cancel() method on operation handles.

        Args:
            reason: Reason for cancellation

        Example:
            >>> manager.cancel("User requested stop")

        References:
            PROG-002: cancel() method on operation handles
        """
        self._cancelled.set()
        self._operation_name = reason

    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if operation should be cancelled

        Example:
            >>> if manager.is_cancelled():
            ...     return  # Exit early

        References:
            PROG-002: Cancellation Support
        """
        return self._cancelled.is_set()

    def check_cancelled(self) -> None:
        """Check cancellation status and raise if cancelled.

        : Graceful cancellation with partial results.

        Raises:
            CancelledException: If cancellation has been requested

        Example:
            >>> manager.check_cancelled()  # Raises if cancelled

        References:
            PROG-002: Cancellation Support
        """
        if self._cancelled.is_set():
            self._cleanup()
            elapsed = time.time() - self._start_time if self._start_time > 0 else 0
            raise CancelledException(
                self._operation_name,
                partial_results=self._partial_results,
                elapsed_time=elapsed,
            )

    def add_cleanup(self, cleanup_fn: Callable[[], None]) -> None:
        """Add a cleanup function to be called on cancellation.

        : Cleanup on cancellation - no partial files.

        Args:
            cleanup_fn: Function to call for cleanup

        Example:
            >>> def cleanup_temp_files():
            ...     os.remove("temp.dat")
            >>> manager.add_cleanup(cleanup_temp_files)

        References:
            PROG-002: Cleanup on cancellation
        """
        self._cleanup_functions.append(cleanup_fn)

    def store_partial_result(self, key: str, value: Any) -> None:
        """Store partial result for retrieval after cancellation.

        : Partial results available after cancellation.

        Args:
            key: Result identifier
            value: Partial result value

        Example:
            >>> manager.store_partial_result("samples_processed", 1000)

        References:
            PROG-002: Partial results available after cancellation
        """
        self._partial_results[key] = value

    def get_partial_results(self) -> dict[str, Any]:
        """Get partial results collected before cancellation.

        Returns:
            Dictionary of partial results

        Example:
            >>> try:
            ...     # ... operation ...
            ... except CancelledException as e:
            ...     results = manager.get_partial_results()

        References:
            PROG-002: Partial results available after cancellation
        """
        return self._partial_results.copy()

    def _cleanup(self) -> None:
        """Execute all registered cleanup functions.

        References:
            PROG-002: Cleanup on cancellation
        """
        # Call user-provided cleanup
        if self._cleanup_callback is not None:
            try:
                self._cleanup_callback()
            except Exception:
                pass  # Ignore cleanup errors

        # Call registered cleanup functions
        for cleanup_fn in self._cleanup_functions:
            try:
                cleanup_fn()
            except Exception:
                pass  # Ignore cleanup errors

    @contextmanager
    def cancellable_operation(
        self,
        name: str = "Operation",
    ) -> Generator[CancellationManager, None, None]:
        """Context manager for cancellable operations.

        : Graceful cancellation with cleanup.

        Args:
            name: Operation name for logging

        Yields:
            CancellationManager instance

        Raises:
            CancelledException: If operation is cancelled or interrupted.

        Example:
            >>> with manager.cancellable_operation("Loading data") as ctx:
            ...     for i in range(1000):
            ...         ctx.check_cancelled()
            ...         # ... process ...

        References:
            PROG-002: Cancellation Support
        """
        self._operation_name = name
        self._start_time = time.time()
        try:
            yield self
        except CancelledException:
            raise
        except KeyboardInterrupt:
            self.cancel("Interrupted by user (Ctrl+C)")
            self._cleanup()
            raise CancelledException(
                f"{name} interrupted by user",
                partial_results=self._partial_results,
                elapsed_time=time.time() - self._start_time,
            )
        finally:
            if self._cancelled.is_set():
                self._cleanup()


class CancelledException(Exception):
    """Exception raised when operation is cancelled.

    : Partial results available after cancellation.

    Attributes:
        message: Cancellation message
        partial_results: Results collected before cancellation
        elapsed_time: Time elapsed before cancellation

    Example:
        >>> try:
        ...     manager.check_cancelled()
        ... except CancelledException as e:
        ...     print(f"Cancelled after {e.elapsed_time:.1f}s")
        ...     print(f"Partial results: {e.partial_results}")

    References:
        PROG-002: Partial results available after cancellation
    """

    def __init__(
        self,
        message: str,
        *,
        partial_results: dict[str, Any] | None = None,
        elapsed_time: float = 0.0,
    ) -> None:
        """Initialize CancelledException.

        Args:
            message: Cancellation message
            partial_results: Partial results dictionary
            elapsed_time: Elapsed time in seconds
        """
        self.message = message
        self.partial_results = partial_results or {}
        self.elapsed_time = elapsed_time
        super().__init__(
            f"{message} (elapsed: {elapsed_time:.1f}s, "
            f"partial results: {len(self.partial_results)} items)"
        )


class ResumableOperation:
    """Support for resumable operations after cancellation.

    : Resume support where possible.

    Args:
        checkpoint_callback: Function to save checkpoint state
        restore_callback: Function to restore from checkpoint

    Example:
        >>> def save_state(state):
        ...     with open("checkpoint.json", "w") as f:
        ...         json.dump(state, f)
        >>> def load_state():
        ...     with open("checkpoint.json") as f:
        ...         return json.load(f)
        >>> op = ResumableOperation(save_state, load_state)

    References:
        PROG-002: Resume support where possible
    """

    def __init__(
        self,
        checkpoint_callback: Callable[[dict], None],  # type: ignore[type-arg]
        restore_callback: Callable[[], dict],  # type: ignore[type-arg]
    ) -> None:
        """Initialize resumable operation.

        Args:
            checkpoint_callback: Function to save state
            restore_callback: Function to restore state
        """
        self._checkpoint_callback = checkpoint_callback
        self._restore_callback = restore_callback
        self._state: dict[str, Any] = {}

    def checkpoint(self, state: dict[str, Any]) -> None:
        """Save operation state for resume.

        Args:
            state: Current operation state

        Example:
            >>> op.checkpoint({"processed": 500, "total": 1000})

        References:
            PROG-002: Resume support
        """
        self._state = state
        self._checkpoint_callback(state)

    def restore(self) -> dict[str, Any]:
        """Restore operation state from checkpoint.

        Returns:
            Restored state dictionary

        Example:
            >>> state = op.restore()
            >>> start_index = state.get("processed", 0)

        References:
            PROG-002: Resume support
        """
        self._state = self._restore_callback()
        return self._state

    def has_checkpoint(self) -> bool:
        """Check if checkpoint exists.

        Returns:
            True if checkpoint is available

        References:
            PROG-002: Resume support
        """
        try:
            self._restore_callback()
            return True
        except Exception:
            return False


def confirm_cancellation(
    operation_name: str = "operation",
    *,
    destructive: bool = False,
) -> bool:
    """Confirm cancellation for destructive operations.

    : Cancel confirmation for destructive operations.

    Args:
        operation_name: Name of operation to cancel
        destructive: Whether operation is destructive

    Returns:
        True if user confirms cancellation

    Example:
        >>> if confirm_cancellation("Delete files", destructive=True):
        ...     # Proceed with cancellation

    References:
        PROG-002: Cancel confirmation for destructive operations
    """
    if not destructive:
        return True

    try:
        response = input(f"Cancel {operation_name}? This may lose data. [y/N]: ").strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return True  # Assume yes on Ctrl+C during prompt


__all__ = [
    "CancellationManager",
    "CancelledException",
    "ResumableOperation",
    "confirm_cancellation",
]
