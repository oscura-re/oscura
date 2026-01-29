"""Comprehensive unit tests for progress.py CLI module.

This module provides extensive testing for progress reporting utilities, including:
- Progress reporter initialization
- Stage tracking and completion
- Progress updates within stages
- Context manager support
- tqdm integration (when available)
- Quiet mode operation
- Time tracking and formatting

Test Coverage:
- ProgressReporter class initialization
- start_stage() stage initialization
- complete_stage() stage completion
- update_progress() within-stage updates
- finish() completion
- Context manager (__enter__/__exit__)
- tqdm integration (mocked)
- Quiet mode suppression
"""

from __future__ import annotations

import time
from unittest.mock import Mock, patch

import pytest

from oscura.cli.progress import ProgressReporter

pytestmark = [pytest.mark.unit, pytest.mark.cli]


# =============================================================================
# Test ProgressReporter Initialization
# =============================================================================


@pytest.mark.unit
def test_progress_reporter_basic_init():
    """Test basic ProgressReporter initialization."""
    reporter = ProgressReporter(stages=3)

    assert reporter.total_stages == 3
    assert reporter.current_stage == 0
    assert not reporter.quiet
    assert reporter.stage_name == ""


@pytest.mark.unit
def test_progress_reporter_quiet_mode():
    """Test ProgressReporter in quiet mode."""
    reporter = ProgressReporter(quiet=True, stages=5)

    assert reporter.quiet is True
    assert reporter.total_stages == 5


@pytest.mark.unit
def test_progress_reporter_detects_tqdm_available():
    """Test that ProgressReporter detects tqdm availability."""
    # tqdm is available in test environment
    reporter = ProgressReporter(stages=1)

    # Should auto-detect based on availability and TTY
    assert hasattr(reporter, "use_tqdm")


@pytest.mark.unit
def test_progress_reporter_force_tqdm():
    """Test forcing tqdm usage."""
    with patch("oscura.cli.progress.TQDM_AVAILABLE", True):
        reporter = ProgressReporter(stages=1, use_tqdm=True, quiet=False)

        assert reporter.use_tqdm is True


@pytest.mark.unit
def test_progress_reporter_disable_tqdm():
    """Test disabling tqdm explicitly."""
    reporter = ProgressReporter(stages=1, use_tqdm=False)

    assert reporter.use_tqdm is False


@pytest.mark.unit
def test_progress_reporter_tqdm_unavailable():
    """Test behavior when tqdm is not available."""
    with patch("oscura.cli.progress.TQDM_AVAILABLE", False):
        reporter = ProgressReporter(stages=1, use_tqdm=True)

        # Should fall back to non-tqdm mode
        assert reporter.use_tqdm is False


@pytest.mark.unit
def test_progress_reporter_tqdm_quiet_mode():
    """Test that quiet mode disables tqdm."""
    with patch("oscura.cli.progress.TQDM_AVAILABLE", True):
        reporter = ProgressReporter(stages=1, quiet=True, use_tqdm=None)

        assert reporter.use_tqdm is False


# =============================================================================
# Test start_stage()
# =============================================================================


@pytest.mark.unit
def test_start_stage_increments_counter():
    """Test that start_stage increments stage counter."""
    reporter = ProgressReporter(stages=3, quiet=True)

    reporter.start_stage("Stage 1")
    assert reporter.current_stage == 1

    reporter.start_stage("Stage 2")
    assert reporter.current_stage == 2


@pytest.mark.unit
def test_start_stage_sets_name():
    """Test that start_stage sets stage name."""
    reporter = ProgressReporter(stages=2, quiet=True)

    reporter.start_stage("Loading data")
    assert reporter.stage_name == "Loading data"


@pytest.mark.unit
def test_start_stage_updates_time():
    """Test that start_stage updates stage start time."""
    reporter = ProgressReporter(stages=1, quiet=True)

    start_time = time.time()
    reporter.start_stage("Test stage")

    # Stage start time should be recent
    assert reporter.stage_start >= start_time


@pytest.mark.unit
def test_start_stage_prints_message():
    """Test that start_stage prints progress message in non-quiet mode."""
    reporter = ProgressReporter(stages=2, quiet=False, use_tqdm=False)

    with patch("builtins.print") as mock_print:
        reporter.start_stage("Processing")

        # Should print stage info
        mock_print.assert_called()
        # Check that the call includes stage info
        call_args = str(mock_print.call_args)
        assert "Processing" in call_args


@pytest.mark.unit
def test_start_stage_quiet_mode_no_output():
    """Test that start_stage doesn't print in quiet mode."""
    reporter = ProgressReporter(stages=2, quiet=True)

    with patch("builtins.print") as mock_print:
        reporter.start_stage("Silent stage")

        mock_print.assert_not_called()


@pytest.mark.unit
def test_start_stage_tqdm_integration():
    """Test start_stage with tqdm enabled."""
    with patch("oscura.cli.progress.tqdm") as mock_tqdm_class:
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        reporter = ProgressReporter(stages=2, quiet=False, use_tqdm=True)
        reporter.pbar = mock_pbar

        reporter.start_stage("Stage 1")

        # Should update tqdm description
        mock_pbar.set_description.assert_called_once()


# =============================================================================
# Test complete_stage()
# =============================================================================


@pytest.mark.unit
def test_complete_stage_prints_message():
    """Test that complete_stage prints completion message."""
    reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=False)
    reporter.start_stage("Test stage")

    with patch("builtins.print") as mock_print:
        reporter.complete_stage()

        mock_print.assert_called()
        call_args = str(mock_print.call_args)
        assert "completed" in call_args.lower()


@pytest.mark.unit
def test_complete_stage_quiet_mode():
    """Test that complete_stage respects quiet mode."""
    reporter = ProgressReporter(stages=1, quiet=True)
    reporter.start_stage("Stage")

    with patch("builtins.print") as mock_print:
        reporter.complete_stage()

        mock_print.assert_not_called()


@pytest.mark.unit
def test_complete_stage_calculates_duration():
    """Test that complete_stage calculates stage duration."""
    reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=False)

    with patch("time.time") as mock_time:
        # Start at t=0, complete at t=1.5
        mock_time.side_effect = [0.0, 1.5, 1.5]

        reporter.start_stage("Timed stage")

        with patch("builtins.print") as mock_print:
            reporter.complete_stage()

            # Should print duration
            call_args = str(mock_print.call_args)
            assert "1.5" in call_args


@pytest.mark.unit
def test_complete_stage_tqdm_update():
    """Test complete_stage updates tqdm progress bar."""
    with patch("oscura.cli.progress.tqdm") as mock_tqdm_class:
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        reporter = ProgressReporter(stages=2, quiet=False, use_tqdm=True)
        reporter.pbar = mock_pbar
        reporter.start_stage("Stage 1")

        reporter.complete_stage()

        # Should call update(1)
        mock_pbar.update.assert_called_once_with(1)


# =============================================================================
# Test update_progress()
# =============================================================================


@pytest.mark.unit
def test_update_progress_basic():
    """Test basic progress update."""
    reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=False)

    with patch("builtins.print") as mock_print:
        reporter.update_progress(50, 100, "Processing")

        # Should print progress
        mock_print.assert_called()
        call_args = str(mock_print.call_args)
        assert "50/100" in call_args


@pytest.mark.unit
def test_update_progress_calculates_percentage():
    """Test that update_progress calculates percentage."""
    reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=False)

    with patch("builtins.print") as mock_print:
        reporter.update_progress(25, 100)

        call_args = str(mock_print.call_args)
        assert "25.0%" in call_args


@pytest.mark.unit
def test_update_progress_with_message():
    """Test update_progress includes message."""
    reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=False)

    with patch("builtins.print") as mock_print:
        reporter.update_progress(10, 20, "Loading files")

        call_args = str(mock_print.call_args)
        assert "Loading files" in call_args


@pytest.mark.unit
def test_update_progress_quiet_mode():
    """Test that update_progress respects quiet mode."""
    reporter = ProgressReporter(stages=1, quiet=True)

    with patch("builtins.print") as mock_print:
        reporter.update_progress(10, 100)

        mock_print.assert_not_called()


@pytest.mark.unit
def test_update_progress_tqdm_mode():
    """Test that update_progress is no-op with tqdm."""
    with patch("oscura.cli.progress.tqdm") as mock_tqdm_class:
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=True)

        with patch("builtins.print") as mock_print:
            reporter.update_progress(10, 100)

            # Should not print when using tqdm
            mock_print.assert_not_called()


@pytest.mark.unit
def test_update_progress_zero_total():
    """Test update_progress handles zero total gracefully."""
    reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=False)

    with patch("builtins.print") as mock_print:
        reporter.update_progress(0, 0)

        # Should not crash, should show 0%
        call_args = str(mock_print.call_args)
        assert "0.0%" in call_args


# =============================================================================
# Test finish()
# =============================================================================


@pytest.mark.unit
def test_finish_prints_summary():
    """Test that finish prints completion summary."""
    reporter = ProgressReporter(stages=2, quiet=False, use_tqdm=False)

    with patch("builtins.print") as mock_print:
        reporter.finish()

        mock_print.assert_called()
        call_args = str(mock_print.call_args)
        assert "complete" in call_args.lower()


@pytest.mark.unit
def test_finish_shows_total_duration():
    """Test that finish shows total duration."""
    with patch("time.time") as mock_time:
        # Start at t=0, finish at t=5.0 (provide enough values for all calls)
        mock_time.side_effect = [0.0, 5.0, 5.0, 5.0]

        # Create reporter (sets start_time at t=0)
        reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=False)

        with patch("builtins.print") as mock_print:
            reporter.finish()

            call_args = str(mock_print.call_args)
            assert "5.0" in call_args


@pytest.mark.unit
def test_finish_closes_tqdm():
    """Test that finish closes tqdm progress bar."""
    with patch("oscura.cli.progress.tqdm") as mock_tqdm_class:
        mock_pbar = Mock()
        mock_tqdm_class.return_value = mock_pbar

        reporter = ProgressReporter(stages=1, quiet=False, use_tqdm=True)
        reporter.pbar = mock_pbar

        reporter.finish()

        # Should close progress bar
        mock_pbar.close.assert_called_once()


@pytest.mark.unit
def test_finish_quiet_mode():
    """Test that finish respects quiet mode."""
    reporter = ProgressReporter(stages=1, quiet=True)

    with patch("builtins.print") as mock_print:
        reporter.finish()

        mock_print.assert_not_called()


# =============================================================================
# Test Context Manager
# =============================================================================


@pytest.mark.unit
def test_context_manager_basic():
    """Test ProgressReporter as context manager."""
    with patch("builtins.print"):
        with ProgressReporter(stages=1, quiet=False, use_tqdm=False) as reporter:
            assert isinstance(reporter, ProgressReporter)


@pytest.mark.unit
def test_context_manager_calls_finish():
    """Test that context manager calls finish on exit."""
    with patch("oscura.cli.progress.ProgressReporter.finish") as mock_finish:
        with ProgressReporter(stages=1, quiet=True) as reporter:
            pass

        # finish should be called on exit
        mock_finish.assert_called_once()


@pytest.mark.unit
def test_context_manager_returns_self():
    """Test that __enter__ returns self."""
    reporter = ProgressReporter(stages=1, quiet=True)

    result = reporter.__enter__()

    assert result is reporter


@pytest.mark.unit
def test_context_manager_full_workflow():
    """Test complete workflow using context manager."""
    with patch("builtins.print"):
        with ProgressReporter(stages=2, quiet=False, use_tqdm=False) as progress:
            progress.start_stage("Stage 1")
            progress.complete_stage()
            progress.start_stage("Stage 2")
            progress.complete_stage()

        # Should complete without errors


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_full_progress_workflow():
    """Test complete progress reporting workflow."""
    with patch("builtins.print"):
        reporter = ProgressReporter(stages=3, quiet=False, use_tqdm=False)

        reporter.start_stage("Loading")
        reporter.update_progress(50, 100, "Files")
        reporter.complete_stage()

        reporter.start_stage("Processing")
        reporter.complete_stage()

        reporter.start_stage("Saving")
        reporter.complete_stage()

        reporter.finish()

    # Should complete without errors
    assert reporter.current_stage == 3


@pytest.mark.unit
def test_progress_timing_consistency():
    """Test that timing values are consistent."""
    reporter = ProgressReporter(stages=1, quiet=True)

    start = reporter.start_time
    time.sleep(0.01)  # Small delay
    reporter.finish()

    # Finish time should be after start time
    assert time.time() >= start


@pytest.mark.unit
def test_tqdm_not_used_in_quiet_mode():
    """Test that tqdm is never used in quiet mode."""
    with patch("oscura.cli.progress.TQDM_AVAILABLE", True):
        reporter = ProgressReporter(stages=1, quiet=True, use_tqdm=None)

        assert reporter.pbar is None
        assert reporter.use_tqdm is False
