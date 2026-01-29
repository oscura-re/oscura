"""Unit tests for memory configuration module.

This module tests config/memory.py which provides global memory limit
configuration and settings.

Requirements tested:
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.unit


# =============================================================================
# MemoryConfiguration Dataclass Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-009")
class TestMemoryConfiguration:
    """Test MemoryConfiguration dataclass."""

    def test_default_configuration(self) -> None:
        """Test default configuration values."""
        from oscura.core.config.memory import MemoryConfiguration

        config = MemoryConfiguration()

        assert config.max_memory is None  # Auto-detect
        assert config.warn_threshold == 0.7
        assert config.critical_threshold == 0.9
        assert config.auto_degrade is False
        assert config.memory_reserve == 0

    def test_custom_configuration(self) -> None:
        """Test configuration with custom values."""
        from oscura.core.config.memory import MemoryConfiguration

        config = MemoryConfiguration(
            max_memory=4_000_000_000,
            warn_threshold=0.6,
            critical_threshold=0.85,
            auto_degrade=True,
            memory_reserve=500_000_000,
        )

        assert config.max_memory == 4_000_000_000
        assert config.warn_threshold == 0.6
        assert config.critical_threshold == 0.85
        assert config.auto_degrade is True
        assert config.memory_reserve == 500_000_000

    def test_warn_threshold_validation_too_low(self) -> None:
        """Test that warn_threshold below 0.0 raises ValueError."""
        from oscura.core.config.memory import MemoryConfiguration

        with pytest.raises(ValueError, match="warn_threshold"):
            MemoryConfiguration(warn_threshold=-0.1)

    def test_warn_threshold_validation_too_high(self) -> None:
        """Test that warn_threshold above 1.0 raises ValueError."""
        from oscura.core.config.memory import MemoryConfiguration

        with pytest.raises(ValueError, match="warn_threshold"):
            MemoryConfiguration(warn_threshold=1.1)

    def test_critical_threshold_validation_too_low(self) -> None:
        """Test that critical_threshold below 0.0 raises ValueError."""
        from oscura.core.config.memory import MemoryConfiguration

        with pytest.raises(ValueError, match="critical_threshold"):
            MemoryConfiguration(critical_threshold=-0.1)

    def test_critical_threshold_validation_too_high(self) -> None:
        """Test that critical_threshold above 1.0 raises ValueError."""
        from oscura.core.config.memory import MemoryConfiguration

        with pytest.raises(ValueError, match="critical_threshold"):
            MemoryConfiguration(critical_threshold=1.1)

    def test_warn_must_be_less_than_critical(self) -> None:
        """Test that warn_threshold must be less than critical_threshold."""
        from oscura.core.config.memory import MemoryConfiguration

        with pytest.raises(ValueError, match="warn_threshold.*less than.*critical_threshold"):
            MemoryConfiguration(warn_threshold=0.9, critical_threshold=0.7)

    def test_equal_thresholds_raises_error(self) -> None:
        """Test that equal thresholds raise ValueError."""
        from oscura.core.config.memory import MemoryConfiguration

        with pytest.raises(ValueError, match="warn_threshold.*less than.*critical_threshold"):
            MemoryConfiguration(warn_threshold=0.8, critical_threshold=0.8)

    def test_negative_memory_reserve_raises_error(self) -> None:
        """Test that negative memory_reserve raises ValueError."""
        from oscura.core.config.memory import MemoryConfiguration

        with pytest.raises(ValueError, match="memory_reserve.*non-negative"):
            MemoryConfiguration(memory_reserve=-100)

    def test_zero_memory_reserve_allowed(self) -> None:
        """Test that zero memory_reserve is allowed."""
        from oscura.core.config.memory import MemoryConfiguration

        config = MemoryConfiguration(memory_reserve=0)
        assert config.memory_reserve == 0

    def test_boundary_thresholds(self) -> None:
        """Test boundary values for thresholds."""
        from oscura.core.config.memory import MemoryConfiguration

        # 0.0 warn and 1.0 critical should be valid
        config = MemoryConfiguration(warn_threshold=0.0, critical_threshold=1.0)
        assert config.warn_threshold == 0.0
        assert config.critical_threshold == 1.0


# =============================================================================
# Memory Limit Setting Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-009")
class TestSetMemoryLimit:
    """Test set_memory_limit function."""

    def setup_method(self) -> None:
        """Reset to defaults before each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def teardown_method(self) -> None:
        """Reset to defaults after each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def test_set_memory_limit_integer(self) -> None:
        """Test setting memory limit as integer bytes."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit(4_000_000_000)

        config = get_memory_config()
        assert config.max_memory == 4_000_000_000

    def test_set_memory_limit_none_for_auto(self) -> None:
        """Test setting memory limit to None for auto-detection."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit(4_000_000_000)  # Set first
        set_memory_limit(None)  # Then reset to auto

        config = get_memory_config()
        assert config.max_memory is None

    def test_set_memory_limit_gb_string(self) -> None:
        """Test setting memory limit with GB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("4GB")

        config = get_memory_config()
        assert config.max_memory == 4_000_000_000

    def test_set_memory_limit_mb_string(self) -> None:
        """Test setting memory limit with MB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("512MB")

        config = get_memory_config()
        assert config.max_memory == 512_000_000

    def test_set_memory_limit_kb_string(self) -> None:
        """Test setting memory limit with KB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("1024KB")

        config = get_memory_config()
        assert config.max_memory == 1_024_000

    def test_set_memory_limit_gib_string(self) -> None:
        """Test setting memory limit with GiB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("2GiB")

        config = get_memory_config()
        expected = 2 * 1024**3
        assert config.max_memory == expected

    def test_set_memory_limit_mib_string(self) -> None:
        """Test setting memory limit with MiB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("256MiB")

        config = get_memory_config()
        expected = 256 * 1024**2
        assert config.max_memory == expected

    def test_set_memory_limit_kib_string(self) -> None:
        """Test setting memory limit with KiB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("512KiB")

        config = get_memory_config()
        expected = 512 * 1024
        assert config.max_memory == expected

    def test_set_memory_limit_case_insensitive(self) -> None:
        """Test that memory limit parsing is case insensitive."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("4gb")
        config = get_memory_config()
        assert config.max_memory == 4_000_000_000

        set_memory_limit("512Mb")
        config = get_memory_config()
        assert config.max_memory == 512_000_000

    def test_set_memory_limit_with_whitespace(self) -> None:
        """Test that memory limit parsing handles whitespace."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("  4GB  ")
        config = get_memory_config()
        assert config.max_memory == 4_000_000_000

    def test_set_memory_limit_invalid_string(self) -> None:
        """Test that invalid memory string raises ValueError."""
        from oscura.core.config.memory import set_memory_limit

        with pytest.raises(ValueError, match="Invalid memory limit"):
            set_memory_limit("invalid")

    def test_set_memory_limit_fractional_gb(self) -> None:
        """Test setting memory limit with fractional GB."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("1.5GB")

        config = get_memory_config()
        assert config.max_memory == 1_500_000_000


# =============================================================================
# Memory Threshold Setting Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-011")
class TestSetMemoryThresholds:
    """Test set_memory_thresholds function."""

    def setup_method(self) -> None:
        """Reset to defaults before each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def teardown_method(self) -> None:
        """Reset to defaults after each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def test_set_both_thresholds(self) -> None:
        """Test setting both warn and critical thresholds."""
        from oscura.core.config.memory import get_memory_config, set_memory_thresholds

        set_memory_thresholds(warn_threshold=0.6, critical_threshold=0.85)

        config = get_memory_config()
        assert config.warn_threshold == 0.6
        assert config.critical_threshold == 0.85

    def test_set_warn_threshold_only(self) -> None:
        """Test setting only warn threshold."""
        from oscura.core.config.memory import get_memory_config, set_memory_thresholds

        set_memory_thresholds(warn_threshold=0.5)

        config = get_memory_config()
        assert config.warn_threshold == 0.5
        assert config.critical_threshold == 0.9  # Default unchanged

    def test_set_critical_threshold_only(self) -> None:
        """Test setting only critical threshold."""
        from oscura.core.config.memory import get_memory_config, set_memory_thresholds

        set_memory_thresholds(critical_threshold=0.95)

        config = get_memory_config()
        assert config.warn_threshold == 0.7  # Default unchanged
        assert config.critical_threshold == 0.95

    def test_invalid_warn_threshold(self) -> None:
        """Test that invalid warn threshold raises ValueError."""
        from oscura.core.config.memory import set_memory_thresholds

        with pytest.raises(ValueError, match="warn_threshold"):
            set_memory_thresholds(warn_threshold=1.5)

    def test_invalid_critical_threshold(self) -> None:
        """Test that invalid critical threshold raises ValueError."""
        from oscura.core.config.memory import set_memory_thresholds

        with pytest.raises(ValueError, match="critical_threshold"):
            set_memory_thresholds(critical_threshold=-0.1)

    def test_warn_greater_than_critical_raises_error(self) -> None:
        """Test that warn >= critical raises ValueError."""
        from oscura.core.config.memory import set_memory_thresholds

        with pytest.raises(ValueError, match="warn_threshold.*less than"):
            set_memory_thresholds(warn_threshold=0.95, critical_threshold=0.8)


# =============================================================================
# Auto Degrade Setting Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-012")
class TestEnableAutoDegrade:
    """Test enable_auto_degrade function."""

    def setup_method(self) -> None:
        """Reset to defaults before each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def teardown_method(self) -> None:
        """Reset to defaults after each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def test_enable_auto_degrade(self) -> None:
        """Test enabling auto degrade."""
        from oscura.core.config.memory import enable_auto_degrade, get_memory_config

        enable_auto_degrade(True)

        config = get_memory_config()
        assert config.auto_degrade is True

    def test_disable_auto_degrade(self) -> None:
        """Test disabling auto degrade."""
        from oscura.core.config.memory import enable_auto_degrade, get_memory_config

        enable_auto_degrade(True)  # Enable first
        enable_auto_degrade(False)  # Then disable

        config = get_memory_config()
        assert config.auto_degrade is False

    def test_enable_auto_degrade_default_argument(self) -> None:
        """Test enable_auto_degrade with default argument."""
        from oscura.core.config.memory import enable_auto_degrade, get_memory_config

        enable_auto_degrade()  # Default is True

        config = get_memory_config()
        assert config.auto_degrade is True


# =============================================================================
# Memory Reserve Setting Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("MEM-002")
class TestSetMemoryReserve:
    """Test set_memory_reserve function."""

    def setup_method(self) -> None:
        """Reset to defaults before each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def teardown_method(self) -> None:
        """Reset to defaults after each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def test_set_memory_reserve_integer(self) -> None:
        """Test setting memory reserve as integer bytes."""
        from oscura.core.config.memory import get_memory_config, set_memory_reserve

        set_memory_reserve(500_000_000)

        config = get_memory_config()
        assert config.memory_reserve == 500_000_000

    def test_set_memory_reserve_gb_string(self) -> None:
        """Test setting memory reserve with GB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_reserve

        set_memory_reserve("1GB")

        config = get_memory_config()
        assert config.memory_reserve == 1_000_000_000

    def test_set_memory_reserve_mb_string(self) -> None:
        """Test setting memory reserve with MB string."""
        from oscura.core.config.memory import get_memory_config, set_memory_reserve

        set_memory_reserve("512MB")

        config = get_memory_config()
        assert config.memory_reserve == 512_000_000


# =============================================================================
# Environment Variable Configuration Tests
# =============================================================================


@pytest.mark.unit
class TestConfigureFromEnvironment:
    """Test configure_from_environment function."""

    def setup_method(self) -> None:
        """Reset to defaults and clear env vars before each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()
        # Clear any environment variables we might use
        self._env_vars = [
            "TK_MAX_MEMORY",
            "TK_MEMORY_RESERVE",
            "TK_MEMORY_WARN_THRESHOLD",
            "TK_MEMORY_CRITICAL_THRESHOLD",
            "TK_AUTO_DEGRADE",
        ]
        self._saved_env = {var: os.environ.get(var) for var in self._env_vars}
        for var in self._env_vars:
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self) -> None:
        """Reset to defaults and restore env vars after each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()
        # Restore environment variables
        for var, value in self._saved_env.items():
            if value is None:
                if var in os.environ:
                    del os.environ[var]
            else:
                os.environ[var] = value

    def test_configure_max_memory_from_env(self) -> None:
        """Test configuring max memory from environment."""
        from oscura.core.config.memory import configure_from_environment, get_memory_config

        os.environ["TK_MAX_MEMORY"] = "4GB"
        configure_from_environment()

        config = get_memory_config()
        assert config.max_memory == 4_000_000_000

    def test_configure_memory_reserve_from_env(self) -> None:
        """Test configuring memory reserve from environment."""
        from oscura.core.config.memory import configure_from_environment, get_memory_config

        os.environ["TK_MEMORY_RESERVE"] = "1GB"
        configure_from_environment()

        config = get_memory_config()
        assert config.memory_reserve == 1_000_000_000

    def test_configure_warn_threshold_from_env(self) -> None:
        """Test configuring warn threshold from environment."""
        from oscura.core.config.memory import configure_from_environment, get_memory_config

        os.environ["TK_MEMORY_WARN_THRESHOLD"] = "0.6"
        configure_from_environment()

        config = get_memory_config()
        assert config.warn_threshold == 0.6

    def test_configure_critical_threshold_from_env(self) -> None:
        """Test configuring critical threshold from environment."""
        from oscura.core.config.memory import configure_from_environment, get_memory_config

        os.environ["TK_MEMORY_CRITICAL_THRESHOLD"] = "0.95"
        configure_from_environment()

        config = get_memory_config()
        assert config.critical_threshold == 0.95

    def test_configure_auto_degrade_true_from_env(self) -> None:
        """Test enabling auto degrade from environment (various true values)."""
        from oscura.core.config.memory import configure_from_environment, get_memory_config

        for true_value in ["1", "true", "yes", "on", "TRUE", "YES"]:
            os.environ["TK_AUTO_DEGRADE"] = true_value
            from oscura.core.config.memory import reset_to_defaults

            reset_to_defaults()
            configure_from_environment()

            config = get_memory_config()
            assert config.auto_degrade is True, f"Failed for value: {true_value}"

    def test_configure_auto_degrade_false_from_env(self) -> None:
        """Test that other values don't enable auto degrade."""
        from oscura.core.config.memory import configure_from_environment, get_memory_config

        os.environ["TK_AUTO_DEGRADE"] = "0"
        configure_from_environment()

        config = get_memory_config()
        assert config.auto_degrade is False

    def test_invalid_threshold_ignored(self) -> None:
        """Test that invalid threshold values are silently ignored."""
        from oscura.core.config.memory import configure_from_environment, get_memory_config

        os.environ["TK_MEMORY_WARN_THRESHOLD"] = "invalid"
        configure_from_environment()

        config = get_memory_config()
        assert config.warn_threshold == 0.7  # Default unchanged


# =============================================================================
# Reset to Defaults Tests
# =============================================================================


@pytest.mark.unit
class TestResetToDefaults:
    """Test reset_to_defaults function."""

    def teardown_method(self) -> None:
        """Reset to defaults after each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def test_reset_restores_all_defaults(self) -> None:
        """Test that reset_to_defaults restores all default values."""
        from oscura.core.config.memory import (
            enable_auto_degrade,
            get_memory_config,
            reset_to_defaults,
            set_memory_limit,
            set_memory_reserve,
            set_memory_thresholds,
        )

        # Modify all settings
        set_memory_limit("8GB")
        set_memory_thresholds(warn_threshold=0.5, critical_threshold=0.8)
        enable_auto_degrade(True)
        set_memory_reserve("2GB")

        # Reset
        reset_to_defaults()

        # Verify all defaults
        config = get_memory_config()
        assert config.max_memory is None
        assert config.warn_threshold == 0.7
        assert config.critical_threshold == 0.9
        assert config.auto_degrade is False
        assert config.memory_reserve == 0


# =============================================================================
# Memory String Parsing Tests
# =============================================================================


@pytest.mark.unit
class TestParseMemoryString:
    """Test _parse_memory_string function."""

    def test_parse_gigabytes(self) -> None:
        """Test parsing gigabyte strings."""
        from oscura.core.config.memory import _parse_memory_string

        assert _parse_memory_string("4GB") == 4_000_000_000
        assert _parse_memory_string("1.5gb") == 1_500_000_000

    def test_parse_megabytes(self) -> None:
        """Test parsing megabyte strings."""
        from oscura.core.config.memory import _parse_memory_string

        assert _parse_memory_string("512MB") == 512_000_000
        assert _parse_memory_string("256mb") == 256_000_000

    def test_parse_kilobytes(self) -> None:
        """Test parsing kilobyte strings."""
        from oscura.core.config.memory import _parse_memory_string

        assert _parse_memory_string("1024KB") == 1_024_000

    def test_parse_gibibytes(self) -> None:
        """Test parsing gibibyte strings."""
        from oscura.core.config.memory import _parse_memory_string

        expected = 2 * 1024**3
        assert _parse_memory_string("2GiB") == expected

    def test_parse_mebibytes(self) -> None:
        """Test parsing mebibyte strings."""
        from oscura.core.config.memory import _parse_memory_string

        expected = 256 * 1024**2
        assert _parse_memory_string("256MiB") == expected

    def test_parse_kibibytes(self) -> None:
        """Test parsing kibibyte strings."""
        from oscura.core.config.memory import _parse_memory_string

        expected = 512 * 1024
        assert _parse_memory_string("512KiB") == expected

    def test_parse_plain_bytes(self) -> None:
        """Test parsing plain byte values."""
        from oscura.core.config.memory import _parse_memory_string

        assert _parse_memory_string("1000000") == 1_000_000

    def test_parse_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        from oscura.core.config.memory import _parse_memory_string

        with pytest.raises(ValueError, match="Invalid memory limit"):
            _parse_memory_string("not_a_number")

    def test_parse_empty_string(self) -> None:
        """Test that empty string raises ValueError."""
        from oscura.core.config.memory import _parse_memory_string

        with pytest.raises(ValueError):
            _parse_memory_string("")


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestMemoryConfigurationIntegration:
    """Integration tests for memory configuration."""

    def setup_method(self) -> None:
        """Reset to defaults before each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def teardown_method(self) -> None:
        """Reset to defaults after each test."""
        from oscura.core.config.memory import reset_to_defaults

        reset_to_defaults()

    def test_full_configuration_workflow(self) -> None:
        """Test complete configuration workflow."""
        from oscura.core.config.memory import (
            enable_auto_degrade,
            get_memory_config,
            reset_to_defaults,
            set_memory_limit,
            set_memory_reserve,
            set_memory_thresholds,
        )

        # Configure everything
        set_memory_limit("4GB")
        set_memory_thresholds(warn_threshold=0.6, critical_threshold=0.85)
        enable_auto_degrade(True)
        set_memory_reserve("512MB")

        # Verify configuration
        config = get_memory_config()
        assert config.max_memory == 4_000_000_000
        assert config.warn_threshold == 0.6
        assert config.critical_threshold == 0.85
        assert config.auto_degrade is True
        assert config.memory_reserve == 512_000_000

        # Reset and verify
        reset_to_defaults()
        config = get_memory_config()
        assert config.max_memory is None

    def test_get_memory_config_returns_same_instance(self) -> None:
        """Test that get_memory_config returns the global instance."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        config1 = get_memory_config()
        set_memory_limit("2GB")
        config2 = get_memory_config()

        # Should be the same instance
        assert config1 is config2
        assert config1.max_memory == 2_000_000_000

    def test_configuration_persists_across_calls(self) -> None:
        """Test that configuration persists across function calls."""
        from oscura.core.config.memory import get_memory_config, set_memory_limit

        set_memory_limit("8GB")

        # Multiple calls should return the same configured value
        for _ in range(5):
            config = get_memory_config()
            assert config.max_memory == 8_000_000_000
