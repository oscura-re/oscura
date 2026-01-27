"""Comprehensive tests for cancellation module.

Tests cancellation manager, signal handling, cleanup, partial results,
and resumable operations.
"""

from __future__ import annotations

import signal
import time
from unittest.mock import Mock, patch

import pytest

from oscura.core.cancellation import (
    CancellationManager,
    CancelledException,
    ResumableOperation,
    confirm_cancellation,
)


@pytest.mark.unit
@pytest.mark.core
class TestCancellationManager:
    """Tests for CancellationManager class."""

    def test_initialization_default(self) -> None:
        """Should initialize with default settings."""
        manager = CancellationManager()

        assert manager._cancelled.is_set() is False
        assert manager._cleanup_callback is None
        assert len(manager._cleanup_functions) == 0

    def test_initialization_with_callback(self) -> None:
        """Should initialize with cleanup callback."""
        cleanup_called = False

        def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        manager = CancellationManager(cleanup_callback=cleanup)
        assert manager._cleanup_callback is not None

    def test_cancel_sets_flag(self) -> None:
        """Should set cancellation flag."""
        manager = CancellationManager()

        manager.cancel("Test cancellation")

        assert manager.is_cancelled() is True
        assert manager._operation_name == "Test cancellation"

    def test_check_cancelled_raises_when_cancelled(self) -> None:
        """Should raise CancelledException when cancelled."""
        manager = CancellationManager()
        manager.cancel("Operation cancelled")

        with pytest.raises(CancelledException) as exc_info:
            manager.check_cancelled()

        assert "Operation cancelled" in str(exc_info.value)

    def test_check_cancelled_no_raise_when_not_cancelled(self) -> None:
        """Should not raise when not cancelled."""
        manager = CancellationManager()

        # Should not raise
        manager.check_cancelled()

    def test_add_cleanup_function(self) -> None:
        """Should add cleanup function to list."""
        manager = CancellationManager()
        cleanup_fn = Mock()

        manager.add_cleanup(cleanup_fn)

        assert cleanup_fn in manager._cleanup_functions

    def test_cleanup_calls_callback(self) -> None:
        """Should call cleanup callback on cancellation."""
        cleanup_mock = Mock()
        manager = CancellationManager(cleanup_callback=cleanup_mock)

        manager.cancel()
        manager._cleanup()

        cleanup_mock.assert_called_once()

    def test_cleanup_calls_all_functions(self) -> None:
        """Should call all registered cleanup functions."""
        manager = CancellationManager()
        cleanup1 = Mock()
        cleanup2 = Mock()

        manager.add_cleanup(cleanup1)
        manager.add_cleanup(cleanup2)

        manager.cancel()
        manager._cleanup()

        cleanup1.assert_called_once()
        cleanup2.assert_called_once()

    def test_cleanup_ignores_errors(self) -> None:
        """Should ignore errors in cleanup functions."""
        manager = CancellationManager()

        def failing_cleanup() -> None:
            raise RuntimeError("Cleanup failed")

        manager.add_cleanup(failing_cleanup)

        # Should not raise
        manager._cleanup()

    def test_store_partial_result(self) -> None:
        """Should store partial results."""
        manager = CancellationManager()

        manager.store_partial_result("processed", 100)
        manager.store_partial_result("total", 1000)

        results = manager.get_partial_results()
        assert results["processed"] == 100
        assert results["total"] == 1000

    def test_get_partial_results_returns_copy(self) -> None:
        """Should return copy of partial results."""
        manager = CancellationManager()
        manager.store_partial_result("key", "value")

        results1 = manager.get_partial_results()
        results2 = manager.get_partial_results()

        # Modify one copy
        results1["new_key"] = "new_value"

        # Original should be unchanged
        assert "new_key" not in results2


@pytest.mark.unit
@pytest.mark.core
class TestCancellationManagerSignals:
    """Tests for signal handling in CancellationManager."""

    def test_register_signal_handlers(self) -> None:
        """Should register signal handlers."""
        manager = CancellationManager()

        manager.register_signal_handlers()

        assert manager._signal_handlers_registered is True

    def test_register_signal_handlers_idempotent(self) -> None:
        """Should be safe to call multiple times."""
        manager = CancellationManager()

        manager.register_signal_handlers()
        manager.register_signal_handlers()  # Second call should be safe

        assert manager._signal_handlers_registered is True

    @patch("signal.signal")
    def test_registers_sigint_handler(self, mock_signal: Mock) -> None:
        """Should register SIGINT (Ctrl+C) handler."""
        manager = CancellationManager()

        manager.register_signal_handlers()

        # Check that signal.signal was called for SIGINT
        calls = [call[0] for call in mock_signal.call_args_list]
        assert any(signal.SIGINT in call for call in calls)

    @patch("signal.signal")
    def test_registers_sigterm_handler(self, mock_signal: Mock) -> None:
        """Should register SIGTERM handler."""
        manager = CancellationManager()

        manager.register_signal_handlers()

        # Check that signal.signal was called for SIGTERM
        calls = [call[0] for call in mock_signal.call_args_list]
        assert any(signal.SIGTERM in call for call in calls)


@pytest.mark.unit
@pytest.mark.core
class TestCancellableOperationContext:
    """Tests for cancellable_operation context manager."""

    def test_context_manager_normal_completion(self) -> None:
        """Should complete normally without cancellation."""
        manager = CancellationManager()

        with manager.cancellable_operation("Test operation") as ctx:
            assert ctx is manager
            # Operation completes normally

        assert manager._operation_name == "Test operation"

    def test_context_manager_with_cancellation(self) -> None:
        """Should raise CancelledException when cancelled."""
        manager = CancellationManager()

        with pytest.raises(CancelledException):
            with manager.cancellable_operation("Test operation"):
                manager.cancel("User cancelled")
                manager.check_cancelled()

    def test_context_manager_keyboard_interrupt(self) -> None:
        """Should convert KeyboardInterrupt to CancelledException."""
        manager = CancellationManager()

        with pytest.raises(CancelledException) as exc_info:
            with manager.cancellable_operation("Test operation"):
                raise KeyboardInterrupt()

        assert "interrupted by user" in str(exc_info.value).lower()

    def test_context_manager_cleanup_on_cancellation(self) -> None:
        """Should call cleanup on cancellation."""
        cleanup_called = False

        def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        manager = CancellationManager(cleanup_callback=cleanup)

        with pytest.raises(CancelledException):
            with manager.cancellable_operation("Test"):
                manager.cancel()
                manager.check_cancelled()

        assert cleanup_called is True

    def test_context_manager_timing(self) -> None:
        """Should track operation timing."""
        manager = CancellationManager()

        with manager.cancellable_operation("Test"):
            time.sleep(0.1)

        # Should have recorded start time
        assert manager._start_time > 0


@pytest.mark.unit
@pytest.mark.core
class TestCancelledException:
    """Tests for CancelledException class."""

    def test_initialization_basic(self) -> None:
        """Should initialize with message."""
        exc = CancelledException("Operation cancelled")

        assert exc.message == "Operation cancelled"
        assert exc.partial_results == {}
        assert exc.elapsed_time == 0.0

    def test_initialization_with_partial_results(self) -> None:
        """Should initialize with partial results."""
        results = {"processed": 50, "total": 100}
        exc = CancelledException("Cancelled", partial_results=results, elapsed_time=5.2)

        assert exc.message == "Cancelled"
        assert exc.partial_results == results
        assert exc.elapsed_time == 5.2

    def test_str_representation(self) -> None:
        """Should format as string with details."""
        exc = CancelledException("Test", partial_results={"key": "value"}, elapsed_time=3.5)

        exc_str = str(exc)
        assert "Test" in exc_str
        assert "3.5" in exc_str
        assert "1 items" in exc_str

    def test_none_partial_results(self) -> None:
        """Should handle None partial_results."""
        exc = CancelledException("Test", partial_results=None)

        assert exc.partial_results == {}


@pytest.mark.unit
@pytest.mark.core
class TestResumableOperation:
    """Tests for ResumableOperation class."""

    def test_initialization(self) -> None:
        """Should initialize with callbacks."""
        save_fn = Mock()
        load_fn = Mock()

        op = ResumableOperation(save_fn, load_fn)

        assert op._checkpoint_callback is save_fn
        assert op._restore_callback is load_fn

    def test_checkpoint_saves_state(self) -> None:
        """Should call checkpoint callback with state."""
        save_fn = Mock()
        load_fn = Mock()
        op = ResumableOperation(save_fn, load_fn)

        state = {"processed": 500, "total": 1000}
        op.checkpoint(state)

        save_fn.assert_called_once_with(state)
        assert op._state == state

    def test_restore_loads_state(self) -> None:
        """Should call restore callback and return state."""
        save_fn = Mock()
        load_fn = Mock(return_value={"processed": 300})
        op = ResumableOperation(save_fn, load_fn)

        state = op.restore()

        load_fn.assert_called_once()
        assert state == {"processed": 300}
        assert op._state == {"processed": 300}

    def test_has_checkpoint_true(self) -> None:
        """Should return True when checkpoint exists."""
        save_fn = Mock()
        load_fn = Mock(return_value={"data": "exists"})
        op = ResumableOperation(save_fn, load_fn)

        assert op.has_checkpoint() is True

    def test_has_checkpoint_false(self) -> None:
        """Should return False when checkpoint missing."""
        save_fn = Mock()
        load_fn = Mock(side_effect=FileNotFoundError())
        op = ResumableOperation(save_fn, load_fn)

        assert op.has_checkpoint() is False

    def test_has_checkpoint_handles_errors(self) -> None:
        """Should handle errors gracefully."""
        save_fn = Mock()
        load_fn = Mock(side_effect=ValueError("Corrupt checkpoint"))
        op = ResumableOperation(save_fn, load_fn)

        # Should not raise, returns False
        assert op.has_checkpoint() is False


@pytest.mark.unit
@pytest.mark.core
class TestConfirmCancellation:
    """Tests for confirm_cancellation function."""

    def test_non_destructive_always_confirms(self) -> None:
        """Non-destructive operations should auto-confirm."""
        result = confirm_cancellation("operation", destructive=False)

        assert result is True

    @patch("builtins.input", return_value="y")
    def test_destructive_confirms_with_yes(self, mock_input: Mock) -> None:
        """Destructive operation should confirm with 'y'."""
        result = confirm_cancellation("delete", destructive=True)

        assert result is True
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="yes")
    def test_destructive_confirms_with_yes_word(self, mock_input: Mock) -> None:
        """Should accept 'yes' as confirmation."""
        result = confirm_cancellation("delete", destructive=True)

        assert result is True

    @patch("builtins.input", return_value="n")
    def test_destructive_rejects_with_no(self, mock_input: Mock) -> None:
        """Should reject with 'n'."""
        result = confirm_cancellation("delete", destructive=True)

        assert result is False

    @patch("builtins.input", return_value="")
    def test_destructive_default_is_no(self, mock_input: Mock) -> None:
        """Empty input should default to no."""
        result = confirm_cancellation("delete", destructive=True)

        assert result is False

    @patch("builtins.input", side_effect=KeyboardInterrupt())
    def test_handles_keyboard_interrupt(self, mock_input: Mock) -> None:
        """Should return True on Ctrl+C during prompt."""
        result = confirm_cancellation("delete", destructive=True)

        # Ctrl+C during prompt should assume yes (cancel the cancellation prompt)
        assert result is True

    @patch("builtins.input", side_effect=EOFError())
    def test_handles_eof_error(self, mock_input: Mock) -> None:
        """Should return True on EOF."""
        result = confirm_cancellation("delete", destructive=True)

        assert result is True

    @patch("builtins.input", return_value="Y")
    def test_case_insensitive(self, mock_input: Mock) -> None:
        """Should be case insensitive."""
        result = confirm_cancellation("delete", destructive=True)

        assert result is True


@pytest.mark.unit
@pytest.mark.core
class TestCancellationIntegration:
    """Integration tests for cancellation workflow."""

    def test_full_cancellation_workflow(self) -> None:
        """Should handle complete cancellation workflow."""
        cleanup_called = False
        partial_results = {}

        def cleanup() -> None:
            nonlocal cleanup_called
            cleanup_called = True

        manager = CancellationManager(cleanup_callback=cleanup)

        with pytest.raises(CancelledException) as exc_info:
            with manager.cancellable_operation("Processing"):
                for i in range(100):
                    manager.store_partial_result("count", i)
                    if i == 50:
                        manager.cancel("Halfway done")
                        manager.check_cancelled()

        # Verify cleanup called
        assert cleanup_called is True

        # Verify partial results available
        exc = exc_info.value
        assert exc.partial_results["count"] == 50

    def test_resumable_workflow(self) -> None:
        """Should support resume after cancellation."""
        state_storage: dict[str, int] = {}

        def save_state(state: dict[str, int]) -> None:
            state_storage.update(state)

        def load_state() -> dict[str, int]:
            return state_storage.copy()

        op = ResumableOperation(save_state, load_state)

        # First run: checkpoint at 50%
        op.checkpoint({"processed": 500, "total": 1000})

        # Resume: restore state
        restored = op.restore()
        assert restored["processed"] == 500
        assert restored["total"] == 1000

    def test_multiple_cleanup_functions_execution_order(self) -> None:
        """Should execute cleanup functions in order."""
        execution_order: list[int] = []

        def cleanup1() -> None:
            execution_order.append(1)

        def cleanup2() -> None:
            execution_order.append(2)

        def cleanup3() -> None:
            execution_order.append(3)

        manager = CancellationManager()
        manager.add_cleanup(cleanup1)
        manager.add_cleanup(cleanup2)
        manager.add_cleanup(cleanup3)

        manager.cancel()
        manager._cleanup()

        assert execution_order == [1, 2, 3]

    def test_auto_cleanup_registration(self) -> None:
        """Should register cleanup at exit when auto_cleanup=True."""
        # Test that atexit.register is called
        with patch("atexit.register") as mock_register:
            manager = CancellationManager(auto_cleanup=True)

            # Should have registered cleanup
            mock_register.assert_called_once()

    def test_no_auto_cleanup_registration(self) -> None:
        """Should not register cleanup when auto_cleanup=False."""
        with patch("atexit.register") as mock_register:
            manager = CancellationManager(auto_cleanup=False)

            # Should not have registered cleanup
            mock_register.assert_not_called()
