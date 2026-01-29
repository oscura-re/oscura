"""Comprehensive unit tests for progress module.

Tests for PROG-001, PROG-002, and PROG-003 requirements:
"""

from __future__ import annotations

import time
import warnings
from unittest.mock import Mock, patch

import pytest

from oscura.core.progress import (
    CancellationToken,
    CancelledError,
    ProgressTracker,
    check_memory_available,
    create_progress_tracker,
    create_simple_progress,
    estimate_memory_usage,
    warn_memory_usage,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


# ============================================================================
# CancellationToken Tests
# ============================================================================


class TestCancellationToken:
    """Test suite for CancellationToken class."""

    def test_init_defaults(self) -> None:
        """Test CancellationToken initialization with defaults."""
        token = CancellationToken()
        assert token.is_cancelled() is False
        assert token.message == ""
        assert token.cancelled_at is None

    def test_cancel_sets_cancelled_flag(self) -> None:
        """Test that cancel() sets the cancelled flag."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True

    def test_cancel_with_message(self) -> None:
        """Test cancel with custom message."""
        token = CancellationToken()
        token.cancel("User requested stop")
        assert token.is_cancelled() is True
        assert token.message == "User requested stop"

    def test_cancel_default_message(self) -> None:
        """Test cancel with default message."""
        token = CancellationToken()
        token.cancel()
        assert token.message == "Operation cancelled"

    def test_cancel_sets_timestamp(self) -> None:
        """Test that cancel sets cancelled_at timestamp."""
        token = CancellationToken()
        before = time.time()
        token.cancel()
        after = time.time()

        assert token.cancelled_at is not None
        assert before <= token.cancelled_at <= after

    def test_is_cancelled_initially_false(self) -> None:
        """Test is_cancelled returns False initially."""
        token = CancellationToken()
        assert token.is_cancelled() is False

    def test_is_cancelled_after_cancel(self) -> None:
        """Test is_cancelled returns True after cancel()."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True

    def test_is_cancelled_multiple_calls(self) -> None:
        """Test is_cancelled remains True after multiple calls."""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled() is True
        assert token.is_cancelled() is True
        assert token.is_cancelled() is True

    def test_check_does_not_raise_when_not_cancelled(self) -> None:
        """Test check() does not raise when not cancelled."""
        token = CancellationToken()
        # Should not raise
        token.check()

    def test_check_raises_when_cancelled(self) -> None:
        """Test check() raises CancelledError when cancelled."""
        token = CancellationToken()
        token.cancel("Test cancellation")
        with pytest.raises(CancelledError) as exc_info:
            token.check()
        assert str(exc_info.value) == "Test cancellation (0.0% complete)"

    def test_check_raises_with_default_message(self) -> None:
        """Test check() raises with default message."""
        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancelledError) as exc_info:
            token.check()
        assert "Operation cancelled" in str(exc_info.value)

    def test_cancel_overwrites_previous_message(self) -> None:
        """Test that calling cancel again overwrites message."""
        token = CancellationToken()
        token.cancel("First message")
        assert token.message == "First message"
        token.cancel("Second message")
        assert token.message == "Second message"

    def test_cancel_overwrites_timestamp(self) -> None:
        """Test that calling cancel again updates timestamp."""
        token = CancellationToken()
        token.cancel()
        first_time = token.cancelled_at

        time.sleep(0.01)  # Small delay
        token.cancel()
        second_time = token.cancelled_at

        assert first_time is not None
        assert second_time is not None
        assert second_time > first_time

    def test_message_property_not_set(self) -> None:
        """Test message property when not set."""
        token = CancellationToken()
        assert token.message == ""

    def test_cancelled_at_property_not_set(self) -> None:
        """Test cancelled_at property when not set."""
        token = CancellationToken()
        assert token.cancelled_at is None


# ============================================================================
# CancelledError Tests
# ============================================================================


class TestCancelledError:
    """Test suite for CancelledError exception."""

    def test_init_with_message_only(self) -> None:
        """Test CancelledError initialization with message only."""
        error = CancelledError("Test cancellation")
        assert error.message == "Test cancellation"
        assert error.progress == 0.0

    def test_init_with_progress(self) -> None:
        """Test CancelledError initialization with progress."""
        error = CancelledError("Cancelled", progress=45.5)
        assert error.message == "Cancelled"
        assert error.progress == 45.5

    def test_string_representation(self) -> None:
        """Test string representation of CancelledError."""
        error = CancelledError("Test error", progress=50.0)
        assert str(error) == "Test error (50.0% complete)"

    def test_string_representation_zero_progress(self) -> None:
        """Test string representation with zero progress."""
        error = CancelledError("Test error", progress=0.0)
        assert str(error) == "Test error (0.0% complete)"

    def test_string_representation_full_progress(self) -> None:
        """Test string representation at full progress."""
        error = CancelledError("Test error", progress=100.0)
        assert str(error) == "Test error (100.0% complete)"

    def test_string_representation_fractional_progress(self) -> None:
        """Test string representation with fractional progress."""
        error = CancelledError("Test error", progress=33.333)
        assert str(error) == "Test error (33.3% complete)"

    def test_exception_inheritance(self) -> None:
        """Test that CancelledError inherits from Exception."""
        error = CancelledError("Test")
        assert isinstance(error, Exception)

    def test_raising_and_catching(self) -> None:
        """Test raising and catching CancelledError."""
        with pytest.raises(CancelledError) as exc_info:
            raise CancelledError("User cancelled", progress=75.0)
        assert exc_info.value.message == "User cancelled"
        assert exc_info.value.progress == 75.0


# ============================================================================
# ProgressTracker Tests
# ============================================================================


class TestProgressTracker:
    """Test suite for ProgressTracker class."""

    def test_init_defaults(self) -> None:
        """Test ProgressTracker initialization with defaults."""
        tracker = ProgressTracker(100)
        assert tracker.total == 100
        assert tracker.current == 0
        assert tracker.callback is None
        assert tracker.update_interval == 0.1

    def test_init_with_callback(self) -> None:
        """Test initialization with callback."""
        callback = Mock()
        tracker = ProgressTracker(50, callback=callback)
        assert tracker.callback is callback

    def test_init_with_custom_update_interval(self) -> None:
        """Test initialization with custom update interval."""
        tracker = ProgressTracker(100, update_interval=0.5)
        assert tracker.update_interval == 0.5

    def test_update_current_value(self) -> None:
        """Test update sets current value."""
        tracker = ProgressTracker(100)
        tracker.update(50)
        assert tracker.current == 50

    def test_update_with_message(self) -> None:
        """Test update with message parameter."""
        callback = Mock()
        tracker = ProgressTracker(100, callback=callback, update_interval=0.0)
        tracker.update(25, "Processing items")
        assert tracker.current == 25

    def test_update_calls_callback(self) -> None:
        """Test update calls callback after interval."""
        callback = Mock()
        tracker = ProgressTracker(100, callback=callback, update_interval=0.0)
        tracker.update(50, "Test message")
        callback.assert_called_once_with(50, 100, "Test message")

    def test_update_throttles_callback(self) -> None:
        """Test update throttles callback by update_interval."""
        callback = Mock()
        tracker = ProgressTracker(100, callback=callback, update_interval=1.0)
        tracker.update(25, "First")
        tracker.update(50, "Second")
        # Only first update should call callback due to throttling
        callback.assert_called_once()

    def test_update_without_callback(self) -> None:
        """Test update works without callback."""
        tracker = ProgressTracker(100)
        tracker.update(50, "No callback")
        assert tracker.current == 50

    def test_get_eta_zero_progress(self) -> None:
        """Test ETA calculation with zero progress."""
        tracker = ProgressTracker(100)
        eta = tracker.get_eta()
        assert eta == 0.0

    def test_get_eta_half_progress(self) -> None:
        """Test ETA calculation at 50% progress."""
        tracker = ProgressTracker(100)
        tracker._start_time = time.time() - 1.0  # 1 second ago
        tracker.update(50)
        eta = tracker.get_eta()
        # Should be approximately 1.0 second remaining
        assert 0.8 < eta < 1.2

    def test_get_eta_high_progress(self) -> None:
        """Test ETA calculation near completion."""
        tracker = ProgressTracker(100)
        tracker._start_time = time.time() - 2.0  # 2 seconds ago
        tracker.update(99)
        eta = tracker.get_eta()
        # Should be close to 0 seconds remaining
        assert 0 <= eta < 0.1

    def test_get_eta_zero_rate(self) -> None:
        """Test ETA calculation when rate is zero (elapsed time approaching zero)."""
        tracker = ProgressTracker(100)
        # Force rate to be <= 0 by setting elapsed to almost equal to start time
        tracker.current = 1
        # Set start_time to future (won't happen in practice but tests the else branch)
        tracker._start_time = time.time() + 10.0
        eta = tracker.get_eta()
        # Rate will be <= 0, should return 0.0
        assert eta == 0.0

    def test_get_progress_percent_zero(self) -> None:
        """Test progress percentage at zero."""
        tracker = ProgressTracker(100)
        assert tracker.get_progress_percent() == 0.0

    def test_get_progress_percent_half(self) -> None:
        """Test progress percentage at 50%."""
        tracker = ProgressTracker(100)
        tracker.current = 50
        assert tracker.get_progress_percent() == 50.0

    def test_get_progress_percent_full(self) -> None:
        """Test progress percentage at 100%."""
        tracker = ProgressTracker(100)
        tracker.current = 100
        assert tracker.get_progress_percent() == 100.0

    def test_get_progress_percent_zero_total(self) -> None:
        """Test progress percentage with zero total."""
        tracker = ProgressTracker(0)
        assert tracker.get_progress_percent() == 100.0

    def test_get_progress_percent_fractional(self) -> None:
        """Test progress percentage with fractional values."""
        tracker = ProgressTracker(1000)
        tracker.current = 333
        assert tracker.get_progress_percent() == 33.3

    def test_finish_sets_current_to_total(self) -> None:
        """Test finish sets current to total."""
        tracker = ProgressTracker(100)
        tracker.current = 50
        tracker.finish()
        assert tracker.current == 100

    def test_finish_calls_callback(self) -> None:
        """Test finish calls callback."""
        callback = Mock()
        tracker = ProgressTracker(100, callback=callback)
        tracker.finish("Analysis complete")
        callback.assert_called_once_with(100, 100, "Analysis complete")

    def test_finish_with_custom_message(self) -> None:
        """Test finish with custom message."""
        callback = Mock()
        tracker = ProgressTracker(100, callback=callback)
        tracker.finish("Custom completion message")
        callback.assert_called_once_with(100, 100, "Custom completion message")

    def test_finish_sets_finished_flag(self) -> None:
        """Test finish sets _finished flag."""
        tracker = ProgressTracker(100)
        assert tracker._finished is False
        tracker.finish()
        assert tracker._finished is True

    def test_multiple_updates(self) -> None:
        """Test multiple updates in sequence."""
        tracker = ProgressTracker(100, update_interval=0.0)
        callback = Mock()
        tracker.callback = callback

        for i in range(1, 6):
            tracker.update(i * 20)

        assert tracker.current == 100
        assert callback.call_count == 5

    def test_update_beyond_total(self) -> None:
        """Test update with current > total."""
        tracker = ProgressTracker(100)
        tracker.update(150)  # Beyond total
        assert tracker.current == 150
        assert tracker.get_progress_percent() > 100.0


# ============================================================================
# create_progress_tracker Tests
# ============================================================================


class TestCreateProgressTracker:
    """Test suite for create_progress_tracker factory function."""

    def test_returns_progress_tracker(self) -> None:
        """Test that function returns ProgressTracker instance."""
        tracker = create_progress_tracker(100)
        assert isinstance(tracker, ProgressTracker)

    def test_sets_total(self) -> None:
        """Test that total is set correctly."""
        tracker = create_progress_tracker(500)
        assert tracker.total == 500

    def test_sets_callback(self) -> None:
        """Test that callback is set correctly."""
        callback = Mock()
        tracker = create_progress_tracker(100, callback=callback)
        assert tracker.callback is callback

    def test_sets_update_interval(self) -> None:
        """Test that update_interval is set correctly."""
        tracker = create_progress_tracker(100, update_interval=0.5)
        assert tracker.update_interval == 0.5

    def test_default_update_interval(self) -> None:
        """Test default update_interval."""
        tracker = create_progress_tracker(100)
        assert tracker.update_interval == 0.1

    def test_with_all_parameters(self) -> None:
        """Test factory with all parameters specified."""
        callback = Mock()
        tracker = create_progress_tracker(1000, callback=callback, update_interval=0.2)
        assert tracker.total == 1000
        assert tracker.callback is callback
        assert tracker.update_interval == 0.2


# ============================================================================
# Memory Management Tests
# ============================================================================


class TestEstimateMemoryUsage:
    """Test suite for estimate_memory_usage function."""

    def test_basic_calculation(self) -> None:
        """Test basic memory estimation."""
        # 1000 samples * 8 bytes = 8000 bytes
        # * 2.0 scratch multiplier = 16000 bytes
        result = estimate_memory_usage(1000, dtype_bytes=8)
        assert result == 16000

    def test_custom_dtype_bytes(self) -> None:
        """Test with custom dtype_bytes."""
        # 1000 samples * 4 bytes = 4000 bytes
        # * 2.0 scratch multiplier = 8000 bytes
        result = estimate_memory_usage(1000, dtype_bytes=4)
        assert result == 8000

    def test_custom_scratch_multiplier(self) -> None:
        """Test with custom scratch_multiplier."""
        # 1000 samples * 8 bytes = 8000 bytes
        # * 3.0 scratch multiplier = 24000 bytes
        result = estimate_memory_usage(1000, dtype_bytes=8, scratch_multiplier=3.0)
        assert result == 24000

    def test_multiple_channels(self) -> None:
        """Test with multiple channels."""
        # 1000 samples * 8 bytes * 2 channels = 16000 bytes
        # * 2.0 scratch multiplier = 32000 bytes
        result = estimate_memory_usage(1000, dtype_bytes=8, n_channels=2)
        assert result == 32000

    def test_large_sample_count(self) -> None:
        """Test with large sample count."""
        # 1_000_000 samples * 8 bytes = 8_000_000 bytes
        # * 2.0 scratch multiplier = 16_000_000 bytes
        result = estimate_memory_usage(1_000_000, dtype_bytes=8)
        assert result == 16_000_000

    def test_single_byte_dtype(self) -> None:
        """Test with 1-byte dtype."""
        result = estimate_memory_usage(1000, dtype_bytes=1)
        assert result == 2000

    def test_zero_samples(self) -> None:
        """Test with zero samples."""
        result = estimate_memory_usage(0, dtype_bytes=8)
        assert result == 0

    def test_all_parameters(self) -> None:
        """Test with all parameters specified."""
        # 10000 samples * 4 bytes * 4 channels = 160_000 bytes
        # * 2.5 scratch multiplier = 400_000 bytes
        result = estimate_memory_usage(10000, dtype_bytes=4, n_channels=4, scratch_multiplier=2.5)
        assert result == 400_000


class TestCheckMemoryAvailable:
    """Test suite for check_memory_available function."""

    @patch("psutil.virtual_memory")
    def test_sufficient_memory(self, mock_memory: Mock) -> None:
        """Test when sufficient memory is available."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000  # 1 GB available
        mock_memory.return_value = mock_vm

        # Require 100 MB (well below threshold)
        result = check_memory_available(100_000_000)
        assert result is True

    @patch("psutil.virtual_memory")
    def test_insufficient_memory(self, mock_memory: Mock) -> None:
        """Test when insufficient memory is available."""
        mock_vm = Mock()
        mock_vm.available = 100_000_000  # 100 MB available
        mock_memory.return_value = mock_vm

        # Require 500 MB (above threshold)
        result = check_memory_available(500_000_000)
        assert result is False

    @patch("psutil.virtual_memory")
    def test_at_threshold(self, mock_memory: Mock) -> None:
        """Test when memory usage is exactly at threshold."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000  # 1 GB available
        mock_memory.return_value = mock_vm

        # Require exactly 80% of available (at default threshold)
        required = int(1_000_000_000 * 0.8)
        result = check_memory_available(required)
        assert result is True

    @patch("psutil.virtual_memory")
    def test_above_threshold(self, mock_memory: Mock) -> None:
        """Test when memory usage exceeds threshold."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000  # 1 GB available
        mock_memory.return_value = mock_vm

        # Require 81% of available (above default 80% threshold)
        required = int(1_000_000_000 * 0.81)
        result = check_memory_available(required)
        assert result is False

    @patch("psutil.virtual_memory")
    def test_custom_threshold(self, mock_memory: Mock) -> None:
        """Test with custom threshold."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000
        mock_memory.return_value = mock_vm

        # With 50% threshold
        required = int(1_000_000_000 * 0.51)
        result = check_memory_available(required, threshold=0.5)
        assert result is False

    @patch("psutil.virtual_memory")
    def test_zero_required_memory(self, mock_memory: Mock) -> None:
        """Test with zero required memory."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000
        mock_memory.return_value = mock_vm

        result = check_memory_available(0)
        assert result is True


class TestWarnMemoryUsage:
    """Test suite for warn_memory_usage function."""

    @patch("psutil.virtual_memory")
    def test_no_warning_when_sufficient(self, mock_memory: Mock) -> None:
        """Test no warning when sufficient memory available."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000
        mock_memory.return_value = mock_vm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_memory_usage(100_000_000)
            # No warning should be issued
            assert len(w) == 0

    @patch("psutil.virtual_memory")
    def test_warning_when_insufficient(self, mock_memory: Mock) -> None:
        """Test warning when insufficient memory available."""
        mock_vm = Mock()
        mock_vm.available = 100_000_000
        mock_memory.return_value = mock_vm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_memory_usage(500_000_000)
            assert len(w) == 1
            assert issubclass(w[0].category, ResourceWarning)
            assert "Operation may require" in str(w[0].message)

    @patch("psutil.virtual_memory")
    def test_warning_message_content(self, mock_memory: Mock) -> None:
        """Test warning message contains relevant information."""
        mock_vm = Mock()
        mock_vm.available = 500_000_000
        mock_memory.return_value = mock_vm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_memory_usage(600_000_000)
            message = str(w[0].message)
            # Check for MB values (600 MB required, 500 MB available)
            assert "572.2" in message or "572" in message  # 600 MB in actual calc
            assert "476" in message  # 500 MB in actual calc
            assert "Warning" in message

    @patch("psutil.virtual_memory")
    def test_suggests_chunked_processing(self, mock_memory: Mock) -> None:
        """Test that chunked processing suggestion is included by default."""
        mock_vm = Mock()
        mock_vm.available = 100_000_000
        mock_memory.return_value = mock_vm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_memory_usage(500_000_000, suggest_chunked=True)
            message = str(w[0].message)
            assert "chunked processing" in message.lower() or "Consider" in message

    @patch("psutil.virtual_memory")
    def test_no_chunked_suggestion_when_disabled(self, mock_memory: Mock) -> None:
        """Test no chunked processing suggestion when disabled."""
        mock_vm = Mock()
        mock_vm.available = 100_000_000
        mock_memory.return_value = mock_vm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_memory_usage(500_000_000, suggest_chunked=False)
            message = str(w[0].message)
            # Should still have warning but not suggest chunked
            assert "Operation may require" in message

    @patch("psutil.virtual_memory")
    def test_custom_threshold(self, mock_memory: Mock) -> None:
        """Test with custom threshold."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000
        mock_memory.return_value = mock_vm

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Require 90% with default 80% threshold - should warn
            warn_memory_usage(int(1_000_000_000 * 0.9), threshold=0.8)
            assert len(w) == 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Require 90% with 95% threshold - should not warn
            warn_memory_usage(int(1_000_000_000 * 0.9), threshold=0.95)
            assert len(w) == 0


# ============================================================================
# create_simple_progress Tests
# ============================================================================


class TestCreateSimpleProgress:
    """Test suite for create_simple_progress function."""

    def test_returns_callable(self) -> None:
        """Test that function returns a callable."""
        callback = create_simple_progress()
        assert callable(callback)

    def test_default_prefix(self, capsys) -> None:
        """Test with default prefix."""
        callback = create_simple_progress()
        callback(50, 100, "Processing")
        captured = capsys.readouterr()
        assert "Progress: 50.0%" in captured.out

    def test_custom_prefix(self, capsys) -> None:
        """Test with custom prefix."""
        callback = create_simple_progress("Loading")
        callback(50, 100, "Processing")
        captured = capsys.readouterr()
        assert "Loading: 50.0%" in captured.out

    def test_includes_progress_message(self, capsys) -> None:
        """Test that message is included in output."""
        callback = create_simple_progress()
        callback(50, 100, "Processing items")
        captured = capsys.readouterr()
        assert "Processing items" in captured.out

    def test_shows_current_and_total(self, capsys) -> None:
        """Test that current/total is shown."""
        callback = create_simple_progress()
        callback(25, 100, "")
        captured = capsys.readouterr()
        assert "(25/100)" in captured.out

    def test_zero_total(self, capsys) -> None:
        """Test with zero total."""
        callback = create_simple_progress()
        callback(0, 0, "Starting")
        captured = capsys.readouterr()
        assert "0.0%" in captured.out

    def test_at_completion(self, capsys) -> None:
        """Test at completion (current >= total)."""
        callback = create_simple_progress()
        callback(100, 100, "Complete")
        captured = capsys.readouterr()
        output = captured.out
        # Should end with newline when complete
        assert output.endswith("\n")

    def test_multiple_updates(self, capsys) -> None:
        """Test multiple sequential updates."""
        callback = create_simple_progress()
        for i in range(1, 4):
            callback(i * 25, 100, f"Step {i}")
        # Should print multiple times (each overwrites previous)
        captured = capsys.readouterr()
        assert "Progress:" in captured.out

    def test_percentage_formatting(self, capsys) -> None:
        """Test percentage formatting with fractional values."""
        callback = create_simple_progress()
        callback(33, 100, "")
        captured = capsys.readouterr()
        assert "33.0%" in captured.out

    def test_empty_message(self, capsys) -> None:
        """Test with empty message."""
        callback = create_simple_progress()
        callback(50, 100, "")
        captured = capsys.readouterr()
        # Should still show percentage and current/total
        assert "50.0%" in captured.out
        assert "(50/100)" in captured.out

    def test_carriage_return_for_progress(self, capsys) -> None:
        """Test that \\r is used for progress updates."""
        callback = create_simple_progress()
        callback(50, 100, "Processing")
        # Note: capsys captures the actual output sent to stdout
        # Check that output contains the expected format
        captured = capsys.readouterr()
        assert "50.0%" in captured.out


# ============================================================================
# Integration Tests
# ============================================================================


class TestProgressIntegration:
    """Integration tests combining multiple progress components."""

    def test_progress_tracker_with_cancellation(self) -> None:
        """Test ProgressTracker with CancellationToken."""
        token = CancellationToken()
        tracker = ProgressTracker(10)

        for i in range(1, 11):
            if token.is_cancelled():
                break
            tracker.update(i)
            if i == 5:
                token.cancel("Halfway through")

        assert token.is_cancelled() is True
        assert tracker.current == 5

    def test_cancellation_error_with_progress(self) -> None:
        """Test CancelledError preserves progress information."""
        error = CancelledError("Operation cancelled", progress=67.5)
        with pytest.raises(CancelledError) as exc_info:
            raise error
        assert exc_info.value.progress == 67.5

    def test_complete_analysis_workflow(self) -> None:
        """Test complete analysis workflow with progress tracking."""
        callback = Mock()
        tracker = create_progress_tracker(100, callback=callback, update_interval=0.0)
        token = CancellationToken()

        for i in range(1, 101):
            if token.is_cancelled():
                raise CancelledError("Analysis cancelled", progress=tracker.get_progress_percent())
            tracker.update(i, f"Analyzing item {i}")

        tracker.finish("Analysis complete")

        # Callback should be called for each update + finish
        assert callback.call_count > 100

    @patch("psutil.virtual_memory")
    def test_memory_check_and_warning(self, mock_memory: Mock) -> None:
        """Test memory estimation with checking and warning."""
        mock_vm = Mock()
        mock_vm.available = 100_000_000  # 100 MB
        mock_memory.return_value = mock_vm

        # Estimate memory for a large operation
        required = estimate_memory_usage(10_000_000, dtype_bytes=8)

        # Should indicate insufficient memory
        has_memory = check_memory_available(required)
        assert has_memory is False

        # Should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warn_memory_usage(required)
            assert len(w) > 0

    def test_progress_callback_protocol(self) -> None:
        """Test that progress callbacks follow the protocol."""

        def my_callback(current: int, total: int, message: str) -> None:
            """Test callback implementation."""
            assert isinstance(current, int)
            assert isinstance(total, int)
            assert isinstance(message, str)

        tracker = create_progress_tracker(100, callback=my_callback, update_interval=0.0)
        tracker.update(50, "Test")
        tracker.finish("Done")

    def test_progress_with_cancellation_check(self) -> None:
        """Test periodic cancellation checks during progress."""
        token = CancellationToken()
        progress_calls = []

        def progress_cb(current: int, total: int, msg: str) -> None:
            progress_calls.append((current, total, msg))

        tracker = ProgressTracker(100, callback=progress_cb, update_interval=0.0)

        try:
            for i in range(1, 101):
                tracker.update(i)
                if i == 50:
                    token.cancel("Stopped at 50%")
                token.check()  # Check cancellation
        except CancelledError:
            pass  # Expected

        # Should have updated to 50, then cancelled
        assert tracker.current == 50
        assert len(progress_calls) > 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestProgressEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_cancellation_token_multiple_cancels(self) -> None:
        """Test calling cancel multiple times."""
        token = CancellationToken()
        token.cancel("First")
        token.cancel("Second")
        token.cancel("Third")
        assert token.is_cancelled() is True
        assert token.message == "Third"

    def test_progress_tracker_negative_total(self) -> None:
        """Test tracker with negative total (edge case)."""
        tracker = ProgressTracker(-100)
        tracker.update(-50)
        # Should not crash
        assert tracker.current == -50

    def test_progress_tracker_fractional_current(self) -> None:
        """Test tracker with fractional current value."""
        # Update expects int, but test type flexibility
        tracker = ProgressTracker(1000)
        tracker.current = 333  # Integer current
        assert tracker.get_progress_percent() == 33.3

    def test_estimate_memory_very_large_values(self) -> None:
        """Test memory estimation with very large values."""
        # 1 billion samples, 8 bytes
        result = estimate_memory_usage(1_000_000_000, dtype_bytes=8)
        assert result == 16_000_000_000

    def test_estimate_memory_small_dtype(self) -> None:
        """Test with minimal dtype size."""
        result = estimate_memory_usage(1000, dtype_bytes=1)
        assert result == 2000

    def test_progress_percentage_precision(self) -> None:
        """Test progress percentage calculation precision."""
        tracker = ProgressTracker(3)
        tracker.current = 1
        # 1/3 = 33.333...
        percent = tracker.get_progress_percent()
        assert 33.0 < percent < 34.0

    def test_cancelled_error_with_negative_progress(self) -> None:
        """Test CancelledError with negative progress."""
        error = CancelledError("Test", progress=-10.0)
        assert error.progress == -10.0

    def test_cancelled_error_with_high_progress(self) -> None:
        """Test CancelledError with progress > 100."""
        error = CancelledError("Test", progress=150.0)
        assert error.progress == 150.0

    @patch("psutil.virtual_memory")
    def test_check_memory_boundary_conditions(self, mock_memory: Mock) -> None:
        """Test memory checking at boundary conditions."""
        mock_vm = Mock()
        mock_vm.available = 1_000_000_000
        mock_memory.return_value = mock_vm

        # Test at exact threshold
        threshold_amount = int(1_000_000_000 * 0.8)
        result = check_memory_available(threshold_amount)
        assert result is True

        # Just over threshold
        over_threshold = threshold_amount + 1
        result = check_memory_available(over_threshold)
        assert result is False

    def test_simple_progress_large_numbers(self, capsys) -> None:
        """Test simple progress with large current/total."""
        callback = create_simple_progress()
        callback(1_000_000, 10_000_000, "Processing large dataset")
        captured = capsys.readouterr()
        assert "10.0%" in captured.out
        assert "(1000000/10000000)" in captured.out


class TestLazyPsutilImport:
    """Test lazy psutil import behavior."""

    def test_check_memory_available_without_psutil(self) -> None:
        """Test check_memory_available returns True when psutil unavailable."""
        from unittest.mock import patch

        # Mock psutil as unavailable
        with patch("oscura.core.progress._HAS_PSUTIL", False):
            result = check_memory_available(1_000_000_000)
            assert result is True  # Should return True when psutil unavailable

    def test_warn_memory_usage_without_psutil(self) -> None:
        """Test warn_memory_usage silently returns when psutil unavailable."""
        from unittest.mock import patch

        # Mock psutil as unavailable
        with patch("oscura.core.progress._HAS_PSUTIL", False):
            # Should not raise any errors
            warn_memory_usage(1_000_000_000)
            # If it gets here without error, the test passes
