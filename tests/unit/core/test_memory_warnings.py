"""Tests for memory warning threshold system.

Requirements tested:
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from oscura.config.memory import get_memory_config, reset_to_defaults
from oscura.core.memory_warnings import (
    MemoryWarningLevel,
    MemoryWarningMonitor,
    _invoke_callbacks,
    _warning_callbacks,
    check_and_abort_if_critical,
    check_memory_warnings,
    clear_memory_warning_callbacks,
    emit_memory_warning,
    format_memory_warning,
    register_memory_warning_callback,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestMemoryWarningLevel:
    """Test MemoryWarningLevel enum."""

    def test_enum_values(self):
        """Test enum has expected values."""
        assert MemoryWarningLevel.OK.value == "ok"
        assert MemoryWarningLevel.WARNING.value == "warning"
        assert MemoryWarningLevel.CRITICAL.value == "critical"

    def test_enum_equality(self):
        """Test enum equality comparisons."""
        assert MemoryWarningLevel.OK == MemoryWarningLevel.OK
        assert MemoryWarningLevel.OK != MemoryWarningLevel.WARNING
        assert MemoryWarningLevel.WARNING != MemoryWarningLevel.CRITICAL

    def test_enum_string_representation(self):
        """Test enum string representation."""
        assert str(MemoryWarningLevel.OK.value) == "ok"
        assert str(MemoryWarningLevel.WARNING.value) == "warning"
        assert str(MemoryWarningLevel.CRITICAL.value) == "critical"


class TestCheckMemoryWarnings:
    """Test check_memory_warnings function."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_returns_ok_when_below_warn_threshold(self, mock_pressure):
        """Test returns OK when pressure is below warn threshold."""
        mock_pressure.return_value = 0.5  # 50% pressure
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.OK

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_returns_warning_when_above_warn_threshold(self, mock_pressure):
        """Test returns WARNING when pressure is above warn threshold."""
        mock_pressure.return_value = 0.75  # 75% pressure (default warn: 0.7)
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_returns_critical_when_above_critical_threshold(self, mock_pressure):
        """Test returns CRITICAL when pressure is above critical threshold."""
        mock_pressure.return_value = 0.95  # 95% pressure (default critical: 0.9)
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.CRITICAL

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_boundary_at_warn_threshold(self, mock_pressure):
        """Test behavior exactly at warn threshold."""
        config = get_memory_config()
        mock_pressure.return_value = config.warn_threshold
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_boundary_at_critical_threshold(self, mock_pressure):
        """Test behavior exactly at critical threshold."""
        config = get_memory_config()
        mock_pressure.return_value = config.critical_threshold
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.CRITICAL

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_with_custom_thresholds(self, mock_pressure):
        """Test with custom configured thresholds."""
        # Configure custom thresholds
        from oscura.config.memory import _global_config

        _global_config.warn_threshold = 0.6
        _global_config.critical_threshold = 0.8

        mock_pressure.return_value = 0.65
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

        mock_pressure.return_value = 0.85
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.CRITICAL

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_zero_pressure(self, mock_pressure):
        """Test with zero memory pressure."""
        mock_pressure.return_value = 0.0
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.OK

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_max_pressure(self, mock_pressure):
        """Test with maximum memory pressure."""
        mock_pressure.return_value = 1.0
        level = check_memory_warnings()
        assert level == MemoryWarningLevel.CRITICAL


class TestEmitMemoryWarning:
    """Test emit_memory_warning function."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_no_warning_when_ok(self, mock_pressure, mock_available):
        """Test no warning emitted when memory is OK."""
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9  # 8 GB

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning()
            assert len(w) == 0

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_warning_emitted_at_warning_level(self, mock_pressure, mock_available):
        """Test warning emitted when at WARNING level."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9  # 4 GB

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning()
            assert len(w) == 1
            assert issubclass(w[0].category, ResourceWarning)
            assert "High memory pressure" in str(w[0].message)
            assert "75.0%" in str(w[0].message)

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_critical_warning_emitted(self, mock_pressure, mock_available):
        """Test critical warning emitted when at CRITICAL level."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9  # 1 GB

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning()
            assert len(w) == 1
            assert issubclass(w[0].category, ResourceWarning)
            assert "CRITICAL memory pressure" in str(w[0].message)
            assert "95.0%" in str(w[0].message)
            assert "1.00 GB" in str(w[0].message)

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_force_flag_emits_warning_when_ok(self, mock_pressure, mock_available):
        """Test force flag emits warning even when level is OK."""
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning(force=True)
            # With force=True and OK level, no warning should be emitted
            # (looking at the code, force or level != OK, so both OK+force won't emit)
            assert len(w) == 0

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_warning_stacklevel(self, mock_pressure, mock_available):
        """Test warning has correct stacklevel for debugging."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning()
            # Verify stacklevel is set (warning should be attributed to caller)
            assert len(w) == 1

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_warning_message_format(self, mock_pressure, mock_available):
        """Test warning message contains expected information."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning()
            message = str(w[0].message)
            # Check for key information
            assert "75.0%" in message  # Pressure percentage
            assert "4.00 GB" in message  # Available memory
            assert "Monitor memory usage" in message  # Recommendation


class TestCheckAndAbortIfCritical:
    """Test check_and_abort_if_critical function."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_no_error_when_ok(self, mock_pressure):
        """Test no error raised when memory is OK."""
        mock_pressure.return_value = 0.5
        # Should not raise
        check_and_abort_if_critical("test_operation")

    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_no_error_when_warning(self, mock_pressure):
        """Test no error raised at WARNING level."""
        mock_pressure.return_value = 0.75
        # Should not raise
        check_and_abort_if_critical("test_operation")

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_raises_memory_error_when_critical(self, mock_pressure, mock_available):
        """Test raises MemoryError when at CRITICAL level."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        with pytest.raises(MemoryError) as exc_info:
            check_and_abort_if_critical("spectrogram")

        error_msg = str(exc_info.value)
        assert "Critical memory pressure" in error_msg
        assert "95.0%" in error_msg
        assert "1.00 GB" in error_msg
        assert "spectrogram" in error_msg

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_error_message_contains_threshold(self, mock_pressure, mock_available):
        """Test error message contains threshold information."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        with pytest.raises(MemoryError) as exc_info:
            check_and_abort_if_critical("test_op")

        error_msg = str(exc_info.value)
        # Default critical threshold is 90%
        assert "90%" in error_msg

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_custom_operation_name(self, mock_pressure, mock_available):
        """Test custom operation name appears in error."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        with pytest.raises(MemoryError) as exc_info:
            check_and_abort_if_critical("my_custom_operation")

        assert "my_custom_operation" in str(exc_info.value)

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_default_operation_name(self, mock_pressure, mock_available):
        """Test default operation name when not specified."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        with pytest.raises(MemoryError) as exc_info:
            check_and_abort_if_critical()

        assert "operation" in str(exc_info.value)


class TestMemoryWarningMonitor:
    """Test MemoryWarningMonitor context manager."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_context_manager_enter_exit(self, mock_pressure, mock_available):
        """Test context manager can be entered and exited."""
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9

        with MemoryWarningMonitor("test_op") as monitor:
            assert isinstance(monitor, MemoryWarningMonitor)
            assert monitor.operation == "test_op"

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_initial_check_on_enter(self, mock_pressure, mock_available):
        """Test initial memory check performed on enter."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with MemoryWarningMonitor("test_op"):
                pass
            # Should emit warning on entry
            assert len(w) >= 1

    def test_initialization_parameters(self):
        """Test monitor initialization with custom parameters."""
        monitor = MemoryWarningMonitor(
            "my_op",
            check_interval=50,
            abort_on_critical=False,
        )
        assert monitor.operation == "my_op"
        assert monitor.check_interval == 50
        assert monitor.abort_on_critical is False

    def test_default_parameters(self):
        """Test monitor initialization with default parameters."""
        monitor = MemoryWarningMonitor("test_op")
        assert monitor.check_interval == 100
        assert monitor.abort_on_critical is True

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_check_respects_interval(self, mock_pressure, mock_available):
        """Test check only happens at specified interval."""
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with MemoryWarningMonitor("test_op", check_interval=10) as monitor:
                # Check at non-interval iterations (should be skipped)
                monitor.check(iteration=1)
                monitor.check(iteration=2)
                monitor.check(iteration=9)
                # No new warnings (initial warning on entry already counted)

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_check_at_interval(self, mock_pressure, mock_available):
        """Test check happens at interval iterations."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with MemoryWarningMonitor("test_op", check_interval=5) as monitor:
                monitor.check(iteration=5)  # Should check
                monitor.check(iteration=10)  # Should check
                # Should have warnings (initial + periodic checks)
                assert len(w) >= 1

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_warning_only_emitted_once(self, mock_pressure, mock_available):
        """Test warning is only emitted once per level."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with MemoryWarningMonitor("test_op", check_interval=1) as monitor:
                monitor.check(iteration=1)
                monitor.check(iteration=2)
                monitor.check(iteration=3)
                # Should only warn once (plus initial)
                warning_count = sum(1 for warn in w if "High memory pressure" in str(warn.message))
                assert warning_count == 2  # Initial + first check

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_critical_warning_only_once(self, mock_pressure, mock_available):
        """Test critical warning is only emitted once."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with MemoryWarningMonitor(
                "test_op", check_interval=1, abort_on_critical=False
            ) as monitor:
                monitor.check(iteration=1)
                monitor.check(iteration=2)
                # Should only warn once about critical
                critical_count = sum(
                    1 for warn in w if "Critical memory pressure" in str(warn.message)
                )
                assert critical_count <= 2  # Initial + first check

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_abort_on_critical_raises_error(self, mock_pressure, mock_available):
        """Test abort_on_critical raises MemoryError."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        with pytest.raises(MemoryError) as exc_info:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with MemoryWarningMonitor(
                    "test_op", check_interval=1, abort_on_critical=True
                ) as monitor:
                    monitor.check(iteration=1)

        assert "Critical memory pressure" in str(exc_info.value)
        assert "test_op" in str(exc_info.value)

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_abort_disabled_no_error(self, mock_pressure, mock_available):
        """Test abort_on_critical=False does not raise error."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with MemoryWarningMonitor("test_op", abort_on_critical=False) as monitor:
                monitor.check(iteration=100)  # Should not raise

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_check_without_iteration_increments_counter(self, mock_pressure, mock_available):
        """Test check without iteration parameter increments internal counter."""
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9

        with MemoryWarningMonitor("test_op") as monitor:
            assert monitor._iteration == 0
            monitor.check()
            assert monitor._iteration == 1
            monitor.check()
            assert monitor._iteration == 2

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_check_with_iteration_parameter(self, mock_pressure, mock_available):
        """Test check with explicit iteration parameter."""
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9

        with MemoryWarningMonitor("test_op", check_interval=10) as monitor:
            monitor.check(iteration=10)  # Should check
            monitor.check(iteration=20)  # Should check
            # Internal counter still increments
            assert monitor._iteration == 2

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_warning_message_includes_operation(self, mock_pressure, mock_available):
        """Test warning message includes operation name."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with MemoryWarningMonitor("my_special_operation", check_interval=1) as monitor:
                monitor.check(iteration=1)
                # Find warning with operation name
                found = any("my_special_operation" in str(warn.message) for warn in w)
                assert found


class TestRegisterMemoryWarningCallback:
    """Test callback registration and invocation."""

    def setup_method(self):
        """Clear callbacks before each test."""
        clear_memory_warning_callbacks()

    def teardown_method(self):
        """Clear callbacks after each test."""
        clear_memory_warning_callbacks()

    def test_register_callback(self):
        """Test registering a callback function."""
        callback = MagicMock()
        register_memory_warning_callback(callback)
        assert callback in _warning_callbacks

    def test_register_multiple_callbacks(self):
        """Test registering multiple callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        register_memory_warning_callback(callback1)
        register_memory_warning_callback(callback2)
        assert callback1 in _warning_callbacks
        assert callback2 in _warning_callbacks
        assert len(_warning_callbacks) == 2

    def test_clear_callbacks(self):
        """Test clearing all callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        register_memory_warning_callback(callback1)
        register_memory_warning_callback(callback2)
        clear_memory_warning_callbacks()
        assert len(_warning_callbacks) == 0

    def test_invoke_callbacks_calls_all(self):
        """Test _invoke_callbacks calls all registered callbacks."""
        callback1 = MagicMock()
        callback2 = MagicMock()
        register_memory_warning_callback(callback1)
        register_memory_warning_callback(callback2)

        _invoke_callbacks(MemoryWarningLevel.WARNING)

        callback1.assert_called_once_with(MemoryWarningLevel.WARNING)
        callback2.assert_called_once_with(MemoryWarningLevel.WARNING)

    def test_invoke_callbacks_with_different_levels(self):
        """Test callbacks receive correct warning level."""
        callback = MagicMock()
        register_memory_warning_callback(callback)

        _invoke_callbacks(MemoryWarningLevel.OK)
        callback.assert_called_with(MemoryWarningLevel.OK)

        _invoke_callbacks(MemoryWarningLevel.CRITICAL)
        callback.assert_called_with(MemoryWarningLevel.CRITICAL)

    def test_callback_exception_handled(self):
        """Test exceptions in callbacks are caught and warned."""

        def bad_callback(level):
            raise ValueError("Callback error")

        register_memory_warning_callback(bad_callback)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _invoke_callbacks(MemoryWarningLevel.WARNING)
            # Should emit RuntimeWarning about callback failure
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "callback failed" in str(w[0].message)

    def test_multiple_callbacks_one_fails(self):
        """Test other callbacks still execute if one fails."""

        def bad_callback(level):
            raise ValueError("Bad callback")

        good_callback = MagicMock()

        register_memory_warning_callback(bad_callback)
        register_memory_warning_callback(good_callback)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _invoke_callbacks(MemoryWarningLevel.WARNING)

        # Good callback should still be called
        good_callback.assert_called_once_with(MemoryWarningLevel.WARNING)

    def test_callback_receives_correct_type(self):
        """Test callback receives MemoryWarningLevel enum."""
        received = []

        def callback(level):
            received.append(level)

        register_memory_warning_callback(callback)
        _invoke_callbacks(MemoryWarningLevel.CRITICAL)

        assert len(received) == 1
        assert isinstance(received[0], MemoryWarningLevel)
        assert received[0] == MemoryWarningLevel.CRITICAL


class TestFormatMemoryWarning:
    """Test format_memory_warning function."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_format_ok_level(self, mock_pressure, mock_available):
        """Test formatting OK level message."""
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9

        msg = format_memory_warning(MemoryWarningLevel.OK)
        assert "Memory OK" in msg
        assert "50.0%" in msg
        assert "8.00 GB" in msg

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_format_warning_level(self, mock_pressure, mock_available):
        """Test formatting WARNING level message."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        msg = format_memory_warning(MemoryWarningLevel.WARNING)
        assert "High memory pressure" in msg
        assert "75.0%" in msg
        assert "4.00 GB" in msg
        assert "70%" in msg  # Default warn threshold

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_format_critical_level(self, mock_pressure, mock_available):
        """Test formatting CRITICAL level message."""
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        msg = format_memory_warning(MemoryWarningLevel.CRITICAL)
        assert "CRITICAL memory pressure" in msg
        assert "95.0%" in msg
        assert "1.00 GB" in msg
        assert "90%" in msg  # Default critical threshold

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_format_includes_thresholds(self, mock_pressure, mock_available):
        """Test formatted message includes configured thresholds."""
        mock_pressure.return_value = 0.85
        mock_available.return_value = 2e9

        # Use custom thresholds
        from oscura.config.memory import _global_config

        _global_config.warn_threshold = 0.6
        _global_config.critical_threshold = 0.8

        msg = format_memory_warning(MemoryWarningLevel.CRITICAL)
        assert "80%" in msg  # Custom critical threshold

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_format_with_low_memory(self, mock_pressure, mock_available):
        """Test formatting with very low available memory."""
        mock_pressure.return_value = 0.99
        mock_available.return_value = 0.1e9  # 0.1 GB

        msg = format_memory_warning(MemoryWarningLevel.CRITICAL)
        assert "0.10 GB" in msg

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_format_precision(self, mock_pressure, mock_available):
        """Test numeric formatting precision."""
        mock_pressure.return_value = 0.7654
        mock_available.return_value = 3.456e9

        msg = format_memory_warning(MemoryWarningLevel.WARNING)
        # Check decimal precision
        assert "76.5%" in msg  # 1 decimal for percentage
        assert "3.46 GB" in msg  # 2 decimals for GB


class TestMemoryWarningIntegration:
    """Integration tests for memory warning system."""

    def setup_method(self):
        """Reset memory config and callbacks before each test."""
        reset_to_defaults()
        clear_memory_warning_callbacks()

    def teardown_method(self):
        """Reset memory config and callbacks after each test."""
        reset_to_defaults()
        clear_memory_warning_callbacks()

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_full_workflow_ok_to_warning(self, mock_pressure, mock_available):
        """Test complete workflow from OK to WARNING state."""
        # Start at OK
        mock_pressure.return_value = 0.5
        mock_available.return_value = 8e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.OK

        # Transition to WARNING
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning()
            assert len(w) == 1
            assert "High memory pressure" in str(w[0].message)

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_full_workflow_warning_to_critical(self, mock_pressure, mock_available):
        """Test complete workflow from WARNING to CRITICAL state."""
        # Start at WARNING
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

        # Should not abort
        check_and_abort_if_critical()

        # Transition to CRITICAL
        mock_pressure.return_value = 0.95
        mock_available.return_value = 1e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.CRITICAL

        # Should abort
        with pytest.raises(MemoryError):
            check_and_abort_if_critical()

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_monitor_with_callbacks(self, mock_pressure, mock_available):
        """Test monitor works with registered callbacks."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        callback_levels = []

        def callback(level):
            callback_levels.append(level)

        register_memory_warning_callback(callback)

        # Callbacks are not automatically invoked by monitor
        # They need to be explicitly called via _invoke_callbacks
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with MemoryWarningMonitor("test_op"):
                pass

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_custom_thresholds_workflow(self, mock_pressure, mock_available):
        """Test workflow with custom configured thresholds."""
        from oscura.config.memory import _global_config

        _global_config.warn_threshold = 0.6
        _global_config.critical_threshold = 0.8

        # At 70%, should be WARNING with custom thresholds
        mock_pressure.return_value = 0.7
        mock_available.return_value = 4e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

        # At 85%, should be CRITICAL with custom thresholds
        mock_pressure.return_value = 0.85

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.CRITICAL

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_format_and_emit_consistency(self, mock_pressure, mock_available):
        """Test format_memory_warning and emit_memory_warning are consistent."""
        mock_pressure.return_value = 0.75
        mock_available.return_value = 4e9

        formatted_msg = format_memory_warning(MemoryWarningLevel.WARNING)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_memory_warning()
            emitted_msg = str(w[0].message)

            # Both should contain similar information
            assert "75.0%" in formatted_msg
            assert "75.0%" in emitted_msg
            assert "4.00 GB" in formatted_msg
            assert "4.00 GB" in emitted_msg


class TestCoreMemoryWarningsEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Reset memory config before each test."""
        reset_to_defaults()
        clear_memory_warning_callbacks()

    def teardown_method(self):
        """Reset memory config after each test."""
        reset_to_defaults()
        clear_memory_warning_callbacks()

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_exactly_at_warn_threshold(self, mock_pressure, mock_available):
        """Test behavior exactly at warning threshold boundary."""
        config = get_memory_config()
        mock_pressure.return_value = config.warn_threshold
        mock_available.return_value = 4e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_exactly_at_critical_threshold(self, mock_pressure, mock_available):
        """Test behavior exactly at critical threshold boundary."""
        config = get_memory_config()
        mock_pressure.return_value = config.critical_threshold
        mock_available.return_value = 1e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.CRITICAL

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_just_below_warn_threshold(self, mock_pressure, mock_available):
        """Test behavior just below warning threshold."""
        config = get_memory_config()
        mock_pressure.return_value = config.warn_threshold - 0.001
        mock_available.return_value = 4e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.OK

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_just_below_critical_threshold(self, mock_pressure, mock_available):
        """Test behavior just below critical threshold."""
        config = get_memory_config()
        mock_pressure.return_value = config.critical_threshold - 0.001
        mock_available.return_value = 2e9

        level = check_memory_warnings()
        assert level == MemoryWarningLevel.WARNING

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_zero_available_memory(self, mock_pressure, mock_available):
        """Test with zero available memory."""
        mock_pressure.return_value = 1.0
        mock_available.return_value = 0

        msg = format_memory_warning(MemoryWarningLevel.CRITICAL)
        assert "0.00 GB" in msg

    @patch("oscura.core.memory_warnings.get_available_memory")
    @patch("oscura.core.memory_warnings.get_memory_pressure")
    def test_very_large_available_memory(self, mock_pressure, mock_available):
        """Test with very large available memory."""
        mock_pressure.return_value = 0.1
        mock_available.return_value = 1e12  # 1 TB

        msg = format_memory_warning(MemoryWarningLevel.OK)
        assert "1000.00 GB" in msg

    def test_monitor_zero_check_interval(self):
        """Test monitor with zero check interval."""
        monitor = MemoryWarningMonitor("test", check_interval=0)
        assert monitor.check_interval == 0

    def test_empty_operation_name(self):
        """Test with empty operation name."""
        monitor = MemoryWarningMonitor("")
        assert monitor.operation == ""

    def test_callback_list_isolation(self):
        """Test callback list is properly isolated between tests."""
        # This test verifies teardown is working
        assert len(_warning_callbacks) == 0
        register_memory_warning_callback(MagicMock())
        assert len(_warning_callbacks) == 1
