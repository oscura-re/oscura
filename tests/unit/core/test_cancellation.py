"""Comprehensive unit tests for cancellation module.

Tests for PROG-002: Cancellation Support requirements.
"""

from __future__ import annotations

import signal
import threading
import time
from unittest.mock import Mock, patch

import pytest

from oscura.core.cancellation import (
    CancellationManager,
    CancelledException,
    ResumableOperation,
    confirm_cancellation,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestCancellationManager:
    """Test suite for CancellationManager class."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        manager = CancellationManager()
        assert not manager.is_cancelled()
        assert manager.get_partial_results() == {}

    def test_init_with_cleanup_callback(self) -> None:
        """Test initialization with cleanup callback."""
        cleanup_fn = Mock()
        manager = CancellationManager(cleanup_callback=cleanup_fn)
        assert manager._cleanup_callback is cleanup_fn

    @patch("atexit.register")
    def test_auto_cleanup_registration(self, mock_atexit: Mock) -> None:
        """Test that auto_cleanup=True registers cleanup at exit."""
        manager = CancellationManager(auto_cleanup=True)
        mock_atexit.assert_called_once()
        # Verify the registered function is the cleanup method
        registered_fn = mock_atexit.call_args[0][0]
        assert callable(registered_fn)

    @patch("atexit.register")
    def test_no_auto_cleanup(self, mock_atexit: Mock) -> None:
        """Test that auto_cleanup=False does not register cleanup."""
        manager = CancellationManager(auto_cleanup=False)
        mock_atexit.assert_not_called()

    def test_cancel_sets_flag(self) -> None:
        """Test that cancel() sets cancellation flag."""
        manager = CancellationManager(auto_cleanup=False)
        assert not manager.is_cancelled()
        manager.cancel("Test reason")
        assert manager.is_cancelled()

    def test_cancel_with_custom_reason(self) -> None:
        """Test cancel with custom reason."""
        manager = CancellationManager(auto_cleanup=False)
        manager.cancel("Custom cancellation reason")
        assert manager._operation_name == "Custom cancellation reason"

    def test_is_cancelled_returns_false_initially(self) -> None:
        """Test is_cancelled returns False initially."""
        manager = CancellationManager(auto_cleanup=False)
        assert manager.is_cancelled() is False

    def test_is_cancelled_returns_true_after_cancel(self) -> None:
        """Test is_cancelled returns True after cancel()."""
        manager = CancellationManager(auto_cleanup=False)
        manager.cancel()
        assert manager.is_cancelled() is True

    def test_check_cancelled_raises_when_cancelled(self) -> None:
        """Test check_cancelled raises CancelledException when cancelled."""
        manager = CancellationManager(auto_cleanup=False)
        manager.cancel("Test cancellation")
        with pytest.raises(CancelledException) as exc_info:
            manager.check_cancelled()
        assert "Test cancellation" in str(exc_info.value)

    def test_check_cancelled_does_not_raise_when_not_cancelled(self) -> None:
        """Test check_cancelled does not raise when not cancelled."""
        manager = CancellationManager(auto_cleanup=False)
        # Should not raise
        manager.check_cancelled()

    def test_add_cleanup_function(self) -> None:
        """Test adding cleanup function."""
        manager = CancellationManager(auto_cleanup=False)
        cleanup_fn = Mock()
        manager.add_cleanup(cleanup_fn)
        assert cleanup_fn in manager._cleanup_functions

    def test_cleanup_executes_all_functions(self) -> None:
        """Test that cleanup executes all registered functions."""
        cleanup1 = Mock()
        cleanup2 = Mock()
        cleanup_callback = Mock()

        manager = CancellationManager(cleanup_callback=cleanup_callback, auto_cleanup=False)
        manager.add_cleanup(cleanup1)
        manager.add_cleanup(cleanup2)

        manager._cleanup()

        cleanup_callback.assert_called_once()
        cleanup1.assert_called_once()
        cleanup2.assert_called_once()

    def test_cleanup_ignores_exceptions(self) -> None:
        """Test that cleanup continues even if a function raises."""
        cleanup1 = Mock(side_effect=RuntimeError("Cleanup error"))
        cleanup2 = Mock()

        manager = CancellationManager(auto_cleanup=False)
        manager.add_cleanup(cleanup1)
        manager.add_cleanup(cleanup2)

        # Should not raise
        manager._cleanup()

        cleanup1.assert_called_once()
        cleanup2.assert_called_once()

    def test_cleanup_callback_exception_ignored(self) -> None:
        """Test that cleanup callback exceptions are ignored."""
        cleanup_callback = Mock(side_effect=ValueError("Callback error"))
        cleanup_fn = Mock()

        manager = CancellationManager(cleanup_callback=cleanup_callback, auto_cleanup=False)
        manager.add_cleanup(cleanup_fn)

        # Should not raise
        manager._cleanup()

        cleanup_callback.assert_called_once()
        cleanup_fn.assert_called_once()

    def test_store_partial_result(self) -> None:
        """Test storing partial results."""
        manager = CancellationManager(auto_cleanup=False)
        manager.store_partial_result("key1", "value1")
        manager.store_partial_result("key2", 42)

        results = manager.get_partial_results()
        assert results == {"key1": "value1", "key2": 42}

    def test_get_partial_results_returns_copy(self) -> None:
        """Test that get_partial_results returns a copy."""
        manager = CancellationManager(auto_cleanup=False)
        manager.store_partial_result("key", "value")

        results1 = manager.get_partial_results()
        results1["new_key"] = "new_value"
        results2 = manager.get_partial_results()

        # Original should not be modified
        assert "new_key" not in results2
        assert results2 == {"key": "value"}

    def test_register_signal_handlers(self) -> None:
        """Test registering signal handlers."""
        manager = CancellationManager(auto_cleanup=False)

        with patch("signal.signal") as mock_signal:
            manager.register_signal_handlers()
            assert manager._signal_handlers_registered is True
            # Should register SIGINT and SIGTERM
            assert mock_signal.call_count == 2
            calls = mock_signal.call_args_list
            signals = [call[0][0] for call in calls]
            assert signal.SIGINT in signals
            assert signal.SIGTERM in signals

    def test_register_signal_handlers_idempotent(self) -> None:
        """Test that registering signal handlers multiple times is safe."""
        manager = CancellationManager(auto_cleanup=False)

        with patch("signal.signal") as mock_signal:
            manager.register_signal_handlers()
            manager.register_signal_handlers()
            manager.register_signal_handlers()
            # Should only register once
            assert mock_signal.call_count == 2

    def test_signal_handler_triggers_cancellation(self) -> None:
        """Test that signal handler triggers cancellation."""
        manager = CancellationManager(auto_cleanup=False)

        with patch("signal.signal") as mock_signal:
            manager.register_signal_handlers()
            # Get the registered handler
            sigint_handler = mock_signal.call_args_list[0][0][1]

            # Simulate signal
            assert not manager.is_cancelled()
            sigint_handler(signal.SIGINT, None)
            assert manager.is_cancelled()

    def test_cancellable_operation_context_manager(self) -> None:
        """Test cancellable_operation context manager."""
        manager = CancellationManager(auto_cleanup=False)

        with manager.cancellable_operation("Test operation") as ctx:
            assert ctx is manager
            assert manager._operation_name == "Test operation"
            assert manager._start_time > 0

    def test_cancellable_operation_propagates_cancelled_exception(self) -> None:
        """Test that cancellable_operation propagates CancelledException."""
        manager = CancellationManager(auto_cleanup=False)

        with pytest.raises(CancelledException):
            with manager.cancellable_operation("Test op"):
                manager.cancel("Test")
                manager.check_cancelled()

    def test_cancellable_operation_handles_keyboard_interrupt(self) -> None:
        """Test that cancellable_operation converts KeyboardInterrupt."""
        manager = CancellationManager(auto_cleanup=False)

        with pytest.raises(CancelledException) as exc_info:
            with manager.cancellable_operation("Test op"):
                raise KeyboardInterrupt

        assert "interrupted by user" in str(exc_info.value)

    def test_cancellable_operation_cleanup_on_exception(self) -> None:
        """Test that cleanup is called on exception in context."""
        cleanup_fn = Mock()
        manager = CancellationManager(auto_cleanup=False)
        manager.add_cleanup(cleanup_fn)

        with pytest.raises(CancelledException):
            with manager.cancellable_operation("Test op"):
                manager.cancel()
                manager.check_cancelled()

        cleanup_fn.assert_called()

    def test_cancellable_operation_includes_elapsed_time(self) -> None:
        """Test that CancelledException includes elapsed time."""
        manager = CancellationManager(auto_cleanup=False)

        with pytest.raises(CancelledException) as exc_info:
            with manager.cancellable_operation("Test op"):
                time.sleep(0.1)
                manager.cancel()
                manager.check_cancelled()

        assert exc_info.value.elapsed_time >= 0.1

    def test_cancellable_operation_includes_partial_results(self) -> None:
        """Test that CancelledException includes partial results."""
        manager = CancellationManager(auto_cleanup=False)

        with pytest.raises(CancelledException) as exc_info:
            with manager.cancellable_operation("Test op"):
                manager.store_partial_result("samples", 100)
                manager.cancel()
                manager.check_cancelled()

        assert exc_info.value.partial_results == {"samples": 100}

    def test_thread_safety(self) -> None:
        """Test that cancellation works across threads."""
        manager = CancellationManager(auto_cleanup=False)
        results = []

        def worker() -> None:
            try:
                with manager.cancellable_operation("Worker"):
                    for i in range(100):
                        manager.check_cancelled()
                        time.sleep(0.01)
                        results.append(i)
            except CancelledException:
                pass

        thread = threading.Thread(target=worker)
        thread.start()
        time.sleep(0.05)  # Let worker start
        manager.cancel("Stop worker")
        thread.join(timeout=2.0)

        # Worker should have been cancelled
        assert len(results) < 100

    def test_cancellable_operation_normal_completion(self) -> None:
        """Test that cancellable_operation works normally when not cancelled."""
        manager = CancellationManager(auto_cleanup=False)
        executed = False

        with manager.cancellable_operation("Normal op"):
            executed = True

        assert executed
        assert not manager.is_cancelled()


class TestCancelledException:
    """Test suite for CancelledException class."""

    def test_init_with_message_only(self) -> None:
        """Test initialization with message only."""
        exc = CancelledException("Test message")
        assert exc.message == "Test message"
        assert exc.partial_results == {}
        assert exc.elapsed_time == 0.0

    def test_init_with_all_parameters(self) -> None:
        """Test initialization with all parameters."""
        partial = {"key": "value"}
        exc = CancelledException("Test message", partial_results=partial, elapsed_time=1.5)
        assert exc.message == "Test message"
        assert exc.partial_results == partial
        assert exc.elapsed_time == 1.5

    def test_str_representation(self) -> None:
        """Test string representation."""
        exc = CancelledException(
            "Operation stopped", partial_results={"a": 1, "b": 2}, elapsed_time=2.3
        )
        exc_str = str(exc)
        assert "Operation stopped" in exc_str
        assert "2.3s" in exc_str
        assert "2 items" in exc_str

    def test_exception_can_be_raised_and_caught(self) -> None:
        """Test that exception can be raised and caught."""
        with pytest.raises(CancelledException) as exc_info:
            raise CancelledException("Test")
        assert exc_info.value.message == "Test"

    def test_partial_results_defaults_to_empty_dict(self) -> None:
        """Test that partial_results defaults to empty dict."""
        exc = CancelledException("Test", partial_results=None)
        assert exc.partial_results == {}
        assert isinstance(exc.partial_results, dict)


class TestResumableOperation:
    """Test suite for ResumableOperation class."""

    def test_init(self) -> None:
        """Test initialization."""
        checkpoint_fn = Mock()
        restore_fn = Mock()
        op = ResumableOperation(checkpoint_fn, restore_fn)
        assert op._checkpoint_callback is checkpoint_fn
        assert op._restore_callback is restore_fn
        assert op._state == {}

    def test_checkpoint_saves_state(self) -> None:
        """Test checkpoint saves state."""
        checkpoint_fn = Mock()
        restore_fn = Mock()
        op = ResumableOperation(checkpoint_fn, restore_fn)

        state = {"progress": 50, "total": 100}
        op.checkpoint(state)

        assert op._state == state
        checkpoint_fn.assert_called_once_with(state)

    def test_restore_loads_state(self) -> None:
        """Test restore loads state."""
        checkpoint_fn = Mock()
        saved_state = {"progress": 75, "total": 100}
        restore_fn = Mock(return_value=saved_state)

        op = ResumableOperation(checkpoint_fn, restore_fn)
        restored = op.restore()

        assert restored == saved_state
        assert op._state == saved_state
        restore_fn.assert_called_once()

    def test_has_checkpoint_returns_true_when_available(self) -> None:
        """Test has_checkpoint returns True when checkpoint exists."""
        checkpoint_fn = Mock()
        restore_fn = Mock(return_value={"data": "test"})

        op = ResumableOperation(checkpoint_fn, restore_fn)
        assert op.has_checkpoint() is True

    def test_has_checkpoint_returns_false_on_exception(self) -> None:
        """Test has_checkpoint returns False when restore fails."""
        checkpoint_fn = Mock()
        restore_fn = Mock(side_effect=FileNotFoundError())

        op = ResumableOperation(checkpoint_fn, restore_fn)
        assert op.has_checkpoint() is False

    def test_multiple_checkpoints(self) -> None:
        """Test multiple checkpoint/restore cycles."""
        states = []

        def save_state(state: dict) -> None:  # type: ignore[type-arg]
            states.append(state.copy())

        def load_state() -> dict:  # type: ignore[type-arg]
            return states[-1] if states else {}

        op = ResumableOperation(save_state, load_state)

        # First checkpoint
        op.checkpoint({"step": 1})
        assert len(states) == 1

        # Second checkpoint
        op.checkpoint({"step": 2})
        assert len(states) == 2

        # Restore latest
        restored = op.restore()
        assert restored == {"step": 2}

    def test_checkpoint_and_restore_integration(self) -> None:
        """Test checkpoint and restore work together."""
        saved_state = {}

        def save(state: dict) -> None:  # type: ignore[type-arg]
            saved_state.update(state)

        def load() -> dict:  # type: ignore[type-arg]
            return saved_state.copy()

        op = ResumableOperation(save, load)

        # Save state
        op.checkpoint({"processed": 100, "remaining": 50})

        # Create new operation and restore
        op2 = ResumableOperation(save, load)
        restored = op2.restore()

        assert restored == {"processed": 100, "remaining": 50}


class TestConfirmCancellation:
    """Test suite for confirm_cancellation function."""

    def test_non_destructive_returns_true(self) -> None:
        """Test non-destructive operations always return True."""
        result = confirm_cancellation("Test operation", destructive=False)
        assert result is True

    @patch("builtins.input", return_value="y")
    def test_destructive_with_yes_response(self, mock_input: Mock) -> None:
        """Test destructive operation with 'y' response."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is True
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="yes")
    def test_destructive_with_yes_word_response(self, mock_input: Mock) -> None:
        """Test destructive operation with 'yes' response."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is True

    @patch("builtins.input", return_value="n")
    def test_destructive_with_no_response(self, mock_input: Mock) -> None:
        """Test destructive operation with 'n' response."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is False

    @patch("builtins.input", return_value="no")
    def test_destructive_with_no_word_response(self, mock_input: Mock) -> None:
        """Test destructive operation with 'no' response."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is False

    @patch("builtins.input", return_value="")
    def test_destructive_with_empty_response(self, mock_input: Mock) -> None:
        """Test destructive operation with empty response (default no)."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is False

    @patch("builtins.input", return_value="  Y  ")
    def test_destructive_with_whitespace_trimmed(self, mock_input: Mock) -> None:
        """Test that whitespace is trimmed from response."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is True

    @patch("builtins.input", side_effect=EOFError)
    def test_destructive_with_eof_error(self, mock_input: Mock) -> None:
        """Test EOFError is treated as confirmation."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is True

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    def test_destructive_with_keyboard_interrupt(self, mock_input: Mock) -> None:
        """Test KeyboardInterrupt is treated as confirmation."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is True

    @patch("builtins.input", return_value="y")
    def test_prompt_includes_operation_name(self, mock_input: Mock) -> None:
        """Test that prompt includes the operation name."""
        confirm_cancellation("Custom Operation", destructive=True)
        call_args = mock_input.call_args[0][0]
        assert "Custom Operation" in call_args

    @patch("builtins.input", return_value="maybe")
    def test_destructive_with_invalid_response(self, mock_input: Mock) -> None:
        """Test invalid response is treated as no."""
        result = confirm_cancellation("Delete files", destructive=True)
        assert result is False


class TestModuleExports:
    """Test module's public API exports."""

    def test_all_exports(self) -> None:
        """Test that __all__ contains expected exports."""
        from oscura.core import cancellation

        expected = [
            "CancellationManager",
            "CancelledException",
            "ResumableOperation",
            "confirm_cancellation",
        ]
        assert set(cancellation.__all__) == set(expected)

    def test_all_exports_are_importable(self) -> None:
        """Test that all exported names are actually importable."""
        from oscura.core.cancellation import (
            CancellationManager,
            CancelledException,
            ResumableOperation,
            confirm_cancellation,
        )

        assert CancellationManager is not None
        assert CancelledException is not None
        assert ResumableOperation is not None
        assert confirm_cancellation is not None


class TestIntegrationScenarios:
    """Integration tests for real-world scenarios."""

    def test_full_cancellation_workflow(self) -> None:
        """Test complete cancellation workflow with cleanup."""
        cleanup_called = False

        def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        manager = CancellationManager(cleanup_callback=cleanup, auto_cleanup=False)
        manager.register_signal_handlers()

        with pytest.raises(CancelledException) as exc_info:
            with manager.cancellable_operation("Processing data"):
                for i in range(10):
                    manager.store_partial_result("processed", i)
                    if i == 5:
                        manager.cancel("Halfway done")
                    manager.check_cancelled()

        assert cleanup_called
        assert exc_info.value.partial_results["processed"] == 5

    def test_resumable_operation_with_cancellation(self) -> None:
        """Test resumable operation that gets cancelled and resumed."""
        checkpoint_data = {}

        def save_checkpoint(state: dict) -> None:  # type: ignore[type-arg]
            checkpoint_data.update(state)

        def load_checkpoint() -> dict:  # type: ignore[type-arg]
            return checkpoint_data.copy()

        resumable = ResumableOperation(save_checkpoint, load_checkpoint)
        manager = CancellationManager(auto_cleanup=False)

        # First attempt - cancelled partway through
        with pytest.raises(CancelledException):
            with manager.cancellable_operation("First attempt"):
                for i in range(10):
                    if i == 5:
                        resumable.checkpoint({"progress": i})
                        manager.cancel()
                    manager.check_cancelled()

        # Second attempt - resume from checkpoint
        assert resumable.has_checkpoint()
        state = resumable.restore()
        assert state["progress"] == 5

    def test_concurrent_cancellation_from_multiple_threads(self) -> None:
        """Test cancellation works correctly with multiple threads."""
        manager = CancellationManager(auto_cleanup=False)
        results: dict[str, list[int]] = {"thread1": [], "thread2": []}

        def worker(name: str) -> None:
            try:
                with manager.cancellable_operation(f"Worker {name}"):
                    for i in range(100):
                        manager.check_cancelled()
                        results[name].append(i)
                        time.sleep(0.001)
            except CancelledException:
                pass

        thread1 = threading.Thread(target=worker, args=("thread1",))
        thread2 = threading.Thread(target=worker, args=("thread2",))

        thread1.start()
        thread2.start()
        time.sleep(0.05)
        manager.cancel("Stop all workers")
        thread1.join(timeout=2.0)
        thread2.join(timeout=2.0)

        # Both workers should be cancelled
        assert len(results["thread1"]) < 100
        assert len(results["thread2"]) < 100

    def test_nested_cleanup_functions(self) -> None:
        """Test that nested cleanup functions all execute."""
        execution_order = []

        def cleanup1() -> None:
            execution_order.append("cleanup1")

        def cleanup2() -> None:
            execution_order.append("cleanup2")

        def cleanup3() -> None:
            execution_order.append("cleanup3")

        manager = CancellationManager(cleanup_callback=cleanup1, auto_cleanup=False)
        manager.add_cleanup(cleanup2)
        manager.add_cleanup(cleanup3)

        manager.cancel()
        manager._cleanup()

        assert "cleanup1" in execution_order
        assert "cleanup2" in execution_order
        assert "cleanup3" in execution_order

    def test_partial_results_survive_cleanup(self) -> None:
        """Test that partial results are accessible after cleanup."""
        manager = CancellationManager(auto_cleanup=False)

        with pytest.raises(CancelledException):
            with manager.cancellable_operation("Test"):
                manager.store_partial_result("key1", "value1")
                manager.store_partial_result("key2", 42)
                manager.cancel()
                manager.check_cancelled()

        # Results should still be accessible
        results = manager.get_partial_results()
        assert results == {"key1": "value1", "key2": 42}
