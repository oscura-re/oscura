"""Tests for debug mode and verbosity control.

Tests the debug mode infrastructure (LOG-007).
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock, patch

import pytest

from oscura.core.debug import (
    DebugLevel,
    configure_debug_from_env,
    debug_context,
    debug_log,
    disable_debug,
    enable_debug,
    get_debug_level,
    is_debug_enabled,
    should_log_debug,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


@pytest.fixture(autouse=True)
def clean_debug_state() -> None:
    """Clean debug state before each test."""
    # Reset to disabled
    from oscura.core.debug import _debug_level

    try:
        _debug_level.set(DebugLevel.DISABLED)
    except (LookupError, AttributeError):
        pass


class TestDebugLevel:
    """Test DebugLevel enum."""

    def test_debug_levels_exist(self) -> None:
        """Test that all debug levels are defined."""
        assert DebugLevel.DISABLED == 0
        assert DebugLevel.MINIMAL == 1
        assert DebugLevel.NORMAL == 2
        assert DebugLevel.VERBOSE == 3
        assert DebugLevel.TRACE == 4

    def test_debug_levels_ordered(self) -> None:
        """Test that debug levels are properly ordered."""
        assert DebugLevel.DISABLED < DebugLevel.MINIMAL
        assert DebugLevel.MINIMAL < DebugLevel.NORMAL
        assert DebugLevel.NORMAL < DebugLevel.VERBOSE
        assert DebugLevel.VERBOSE < DebugLevel.TRACE

    def test_debug_level_count(self) -> None:
        """Test number of debug levels."""
        assert len(DebugLevel) == 5

    def test_debug_level_comparison(self) -> None:
        """Test debug level comparison operations."""
        assert DebugLevel.NORMAL >= DebugLevel.MINIMAL
        assert DebugLevel.TRACE > DebugLevel.DISABLED
        assert DebugLevel.VERBOSE <= DebugLevel.TRACE


class TestEnableDebug:
    """Test enable_debug() function."""

    @patch("oscura.core.logging.set_log_level")
    def test_enable_minimal(self, mock_set_log_level: MagicMock) -> None:
        """Test enabling minimal debug level."""
        enable_debug(level="minimal")
        assert get_debug_level() == DebugLevel.MINIMAL
        mock_set_log_level.assert_called_once_with("INFO")

    @patch("oscura.core.logging.set_log_level")
    def test_enable_normal(self, mock_set_log_level: MagicMock) -> None:
        """Test enabling normal debug level."""
        enable_debug(level="normal")
        assert get_debug_level() == DebugLevel.NORMAL
        mock_set_log_level.assert_called_once_with("DEBUG")

    @patch("oscura.core.logging.set_log_level")
    def test_enable_verbose(self, mock_set_log_level: MagicMock) -> None:
        """Test enabling verbose debug level."""
        enable_debug(level="verbose")
        assert get_debug_level() == DebugLevel.VERBOSE
        mock_set_log_level.assert_called_once_with("DEBUG")

    @patch("oscura.core.logging.set_log_level")
    def test_enable_trace(self, mock_set_log_level: MagicMock) -> None:
        """Test enabling trace debug level."""
        enable_debug(level="trace")
        assert get_debug_level() == DebugLevel.TRACE
        mock_set_log_level.assert_called_once_with("DEBUG")

    @patch("oscura.core.logging.set_log_level")
    def test_enable_default_is_normal(self, mock_set_log_level: MagicMock) -> None:
        """Test that default level is normal."""
        enable_debug()
        assert get_debug_level() == DebugLevel.NORMAL


class TestDisableDebug:
    """Test disable_debug() function."""

    @patch("oscura.core.logging.set_log_level")
    def test_disable_sets_disabled_level(self, mock_set_log_level: MagicMock) -> None:
        """Test that disable sets level to DISABLED."""
        enable_debug(level="verbose")
        disable_debug()
        assert get_debug_level() == DebugLevel.DISABLED
        mock_set_log_level.assert_called_with("WARNING")

    @patch("oscura.core.logging.set_log_level")
    def test_disable_from_enabled(self, mock_set_log_level: MagicMock) -> None:
        """Test disabling from enabled state."""
        enable_debug(level="trace")
        mock_set_log_level.reset_mock()

        disable_debug()
        assert not is_debug_enabled()
        mock_set_log_level.assert_called_once_with("WARNING")


class TestIsDebugEnabled:
    """Test is_debug_enabled() function."""

    @patch("oscura.core.logging.set_log_level")
    def test_disabled_returns_false(self, mock_set_log_level: MagicMock) -> None:
        """Test that disabled state returns False."""
        disable_debug()
        assert not is_debug_enabled()

    @patch("oscura.core.logging.set_log_level")
    def test_minimal_enabled(self, mock_set_log_level: MagicMock) -> None:
        """Test minimal level returns True."""
        enable_debug(level="minimal")
        assert is_debug_enabled()
        assert is_debug_enabled(DebugLevel.MINIMAL)

    @patch("oscura.core.logging.set_log_level")
    def test_check_with_min_level(self, mock_set_log_level: MagicMock) -> None:
        """Test checking with minimum level requirement."""
        enable_debug(level="normal")
        assert is_debug_enabled(DebugLevel.MINIMAL)
        assert is_debug_enabled(DebugLevel.NORMAL)
        assert not is_debug_enabled(DebugLevel.VERBOSE)
        assert not is_debug_enabled(DebugLevel.TRACE)

    @patch("oscura.core.logging.set_log_level")
    def test_verbose_meets_lower_levels(self, mock_set_log_level: MagicMock) -> None:
        """Test that verbose level meets all lower requirements."""
        enable_debug(level="verbose")
        assert is_debug_enabled(DebugLevel.MINIMAL)
        assert is_debug_enabled(DebugLevel.NORMAL)
        assert is_debug_enabled(DebugLevel.VERBOSE)


class TestGetDebugLevel:
    """Test get_debug_level() function."""

    @patch("oscura.core.logging.set_log_level")
    def test_get_after_enable(self, mock_set_log_level: MagicMock) -> None:
        """Test getting level after enabling."""
        enable_debug(level="verbose")
        level = get_debug_level()
        assert level == DebugLevel.VERBOSE
        assert isinstance(level, DebugLevel)

    @patch("oscura.core.logging.set_log_level")
    def test_get_after_disable(self, mock_set_log_level: MagicMock) -> None:
        """Test getting level after disabling."""
        enable_debug(level="trace")
        disable_debug()
        level = get_debug_level()
        assert level == DebugLevel.DISABLED

    def test_get_initial_level(self) -> None:
        """Test getting initial debug level."""
        # Should be DISABLED by default (from fixture)
        level = get_debug_level()
        assert level == DebugLevel.DISABLED


class TestDebugContext:
    """Test debug_context context manager."""

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_context_sets_level(
        self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock
    ) -> None:
        """Test that context sets debug level."""
        mock_logger = MagicMock()
        mock_logger.level = logging.INFO
        mock_get_logger.return_value = mock_logger

        with debug_context(level="verbose"):
            assert get_debug_level() == DebugLevel.VERBOSE

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_context_with_minimal(
        self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock
    ) -> None:
        """Test context with minimal level."""
        mock_logger = MagicMock()
        mock_logger.level = logging.INFO
        mock_get_logger.return_value = mock_logger

        with debug_context(level="minimal"):
            assert get_debug_level() == DebugLevel.MINIMAL
        # Verify set_log_level was called with "INFO" for MINIMAL
        mock_set_log_level.assert_any_call("INFO")

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_context_restores_level(
        self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock
    ) -> None:
        """Test that context restores previous level."""
        mock_logger = MagicMock()
        mock_logger.level = logging.WARNING
        mock_get_logger.return_value = mock_logger

        enable_debug(level="minimal")
        with debug_context(level="trace"):
            assert get_debug_level() == DebugLevel.TRACE
        assert get_debug_level() == DebugLevel.MINIMAL

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_context_nested(
        self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock
    ) -> None:
        """Test nested debug contexts."""
        mock_logger = MagicMock()
        mock_logger.level = logging.DEBUG
        mock_get_logger.return_value = mock_logger

        with debug_context(level="normal"):
            assert get_debug_level() == DebugLevel.NORMAL
            with debug_context(level="trace"):
                assert get_debug_level() == DebugLevel.TRACE
            assert get_debug_level() == DebugLevel.NORMAL

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_context_with_exception(
        self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock
    ) -> None:
        """Test context restores level even with exception."""
        mock_logger = MagicMock()
        mock_logger.level = logging.INFO
        mock_get_logger.return_value = mock_logger

        enable_debug(level="minimal")
        try:
            with debug_context(level="verbose"):
                raise ValueError("Test error")
        except ValueError:
            pass
        assert get_debug_level() == DebugLevel.MINIMAL

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_context_disabled(
        self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock
    ) -> None:
        """Test context with disabled level."""
        mock_logger = MagicMock()
        mock_logger.level = logging.DEBUG
        mock_get_logger.return_value = mock_logger

        enable_debug(level="normal")
        with debug_context(level="disabled"):
            assert get_debug_level() == DebugLevel.DISABLED
        assert get_debug_level() == DebugLevel.NORMAL


class TestShouldLogDebug:
    """Test should_log_debug() function."""

    @patch("oscura.core.logging.set_log_level")
    def test_should_log_when_enabled(self, mock_set_log_level: MagicMock) -> None:
        """Test that should_log returns True when enabled."""
        enable_debug(level="normal")
        assert should_log_debug()

    @patch("oscura.core.logging.set_log_level")
    def test_should_not_log_when_disabled(self, mock_set_log_level: MagicMock) -> None:
        """Test that should_log returns False when disabled."""
        disable_debug()
        assert not should_log_debug()

    @patch("oscura.core.logging.set_log_level")
    def test_should_log_with_min_level(self, mock_set_log_level: MagicMock) -> None:
        """Test should_log with minimum level."""
        enable_debug(level="normal")
        assert should_log_debug(DebugLevel.MINIMAL)
        assert should_log_debug(DebugLevel.NORMAL)
        assert not should_log_debug(DebugLevel.VERBOSE)


class TestConfigureDebugFromEnv:
    """Test configure_debug_from_env() function."""

    @patch.dict(os.environ, {"OSCURA_DEBUG": "verbose"})
    @patch("oscura.core.logging.set_log_level")
    def test_configure_from_env_verbose(self, mock_set_log_level: MagicMock) -> None:
        """Test configuring from environment variable."""
        configure_debug_from_env()
        assert get_debug_level() == DebugLevel.VERBOSE

    @patch.dict(os.environ, {"OSCURA_DEBUG": "minimal"})
    @patch("oscura.core.logging.set_log_level")
    def test_configure_from_env_minimal(self, mock_set_log_level: MagicMock) -> None:
        """Test configuring minimal level from env."""
        configure_debug_from_env()
        assert get_debug_level() == DebugLevel.MINIMAL

    @patch.dict(os.environ, {"OSCURA_DEBUG": "TRACE"})
    @patch("oscura.core.logging.set_log_level")
    def test_configure_case_insensitive(self, mock_set_log_level: MagicMock) -> None:
        """Test that env variable is case-insensitive."""
        configure_debug_from_env()
        assert get_debug_level() == DebugLevel.TRACE

    @patch.dict(os.environ, {}, clear=True)
    @patch("oscura.core.logging.set_log_level")
    def test_configure_no_env_var(self, mock_set_log_level: MagicMock) -> None:
        """Test that missing env var doesn't change level."""
        current_level = get_debug_level()
        configure_debug_from_env()
        assert get_debug_level() == current_level

    @patch.dict(os.environ, {"OSCURA_DEBUG": "invalid"})
    @patch("oscura.core.logging.set_log_level")
    def test_configure_invalid_level(self, mock_set_log_level: MagicMock) -> None:
        """Test that invalid level is ignored."""
        current_level = get_debug_level()
        configure_debug_from_env()
        # Should not change level for invalid value
        assert get_debug_level() == current_level


class TestDebugLog:
    """Test debug_log() function."""

    @patch("oscura.core.logging.set_log_level")
    def test_debug_log_when_enabled(self, mock_set_log_level: MagicMock) -> None:
        """Test debug_log logs when enabled."""
        enable_debug(level="normal")
        mock_logger = MagicMock()
        debug_log(mock_logger, "Test message")
        mock_logger.debug.assert_called_once_with("Test message")

    @patch("oscura.core.logging.set_log_level")
    def test_debug_log_when_disabled(self, mock_set_log_level: MagicMock) -> None:
        """Test debug_log doesn't log when disabled."""
        disable_debug()
        mock_logger = MagicMock()
        debug_log(mock_logger, "Test message")
        mock_logger.debug.assert_not_called()

    @patch("oscura.core.logging.set_log_level")
    def test_debug_log_with_min_level(self, mock_set_log_level: MagicMock) -> None:
        """Test debug_log with minimum level requirement."""
        enable_debug(level="normal")
        mock_logger = MagicMock()

        # Should log (normal >= normal)
        debug_log(mock_logger, "Message 1", min_level=DebugLevel.NORMAL)
        assert mock_logger.debug.call_count == 1

        # Should not log (normal < verbose)
        debug_log(mock_logger, "Message 2", min_level=DebugLevel.VERBOSE)
        assert mock_logger.debug.call_count == 1  # Still 1

    @patch("oscura.core.logging.set_log_level")
    def test_debug_log_with_kwargs(self, mock_set_log_level: MagicMock) -> None:
        """Test debug_log passes kwargs to logger."""
        enable_debug(level="verbose")
        mock_logger = MagicMock()
        debug_log(mock_logger, "Test %s", min_level=DebugLevel.NORMAL, extra={"key": "value"})
        mock_logger.debug.assert_called_once_with("Test %s", extra={"key": "value"})


class TestCoreDebugIntegration:
    """Integration tests for debug mode."""

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_full_workflow(self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock) -> None:
        """Test complete debug workflow."""
        mock_logger = MagicMock()
        mock_logger.level = logging.WARNING
        mock_get_logger.return_value = mock_logger

        # Start disabled
        assert not is_debug_enabled()

        # Enable debug
        enable_debug(level="verbose")
        assert is_debug_enabled()
        assert get_debug_level() == DebugLevel.VERBOSE

        # Use debug context for temporary change
        with debug_context(level="trace"):
            assert get_debug_level() == DebugLevel.TRACE

        # Back to verbose
        assert get_debug_level() == DebugLevel.VERBOSE

        # Disable
        disable_debug()
        assert not is_debug_enabled()

    @patch("oscura.core.logging.set_log_level")
    @patch("oscura.core.logging.get_logger")
    def test_conditional_logging_pattern(
        self, mock_get_logger: MagicMock, mock_set_log_level: MagicMock
    ) -> None:
        """Test realistic conditional logging pattern."""
        mock_logger = MagicMock()
        mock_logger.level = logging.DEBUG
        mock_get_logger.return_value = mock_logger

        test_logger = logging.getLogger("test")

        # Enable verbose debugging
        enable_debug(level="verbose")

        # Log at different levels
        if should_log_debug(DebugLevel.MINIMAL):
            test_logger.debug("Basic info")

        if should_log_debug(DebugLevel.VERBOSE):
            test_logger.debug("Detailed info")

        if should_log_debug(DebugLevel.TRACE):
            test_logger.debug("Should not appear")

    @patch.dict(os.environ, {"OSCURA_DEBUG": "trace"})
    @patch("oscura.core.logging.set_log_level")
    def test_env_configuration_workflow(self, mock_set_log_level: MagicMock) -> None:
        """Test configuration from environment."""
        # Clear current state
        from oscura.core.debug import _debug_level

        _debug_level.set(DebugLevel.DISABLED)

        # Configure from env
        configure_debug_from_env()

        # Should now be at trace level
        assert get_debug_level() == DebugLevel.TRACE
        assert is_debug_enabled(DebugLevel.TRACE)

    @patch("oscura.core.logging.set_log_level")
    def test_debug_levels_hierarchy(self, mock_set_log_level: MagicMock) -> None:
        """Test that debug levels follow proper hierarchy."""
        enable_debug(level="verbose")

        # Verbose should include all lower levels
        assert is_debug_enabled(DebugLevel.MINIMAL)
        assert is_debug_enabled(DebugLevel.NORMAL)
        assert is_debug_enabled(DebugLevel.VERBOSE)

        # But not higher levels
        assert not is_debug_enabled(DebugLevel.TRACE)
