"""Unit tests for default configuration values and injection.

Tests CFG-017
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from oscura.config.defaults import (
    DEFAULT_CONFIG,
    SCHEMA_DEFAULTS,
    deep_merge,
    get_default,
    get_effective_config,
    inject_defaults,
)

pytestmark = pytest.mark.unit


class TestDeepMerge:
    """Test deep_merge function."""

    def test_merge_empty_dicts(self) -> None:
        """Test merging two empty dictionaries."""
        result = deep_merge({}, {})
        assert result == {}

    def test_merge_base_only(self) -> None:
        """Test merging with empty override."""
        base = {"key1": "value1", "key2": "value2"}
        result = deep_merge(base, {})
        assert result == base
        assert result is not base  # Should be a copy

    def test_merge_override_only(self) -> None:
        """Test merging with empty base."""
        override = {"key1": "value1", "key2": "value2"}
        result = deep_merge({}, override)
        assert result == override
        assert result is not override  # Should be a copy

    def test_merge_simple_dicts(self) -> None:
        """Test merging two simple dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self) -> None:
        """Test merging nested dictionaries."""
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"d": 4, "e": 5}}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": {"c": 2, "d": 4, "e": 5}}

    def test_merge_deeply_nested(self) -> None:
        """Test merging deeply nested dictionaries."""
        base = {"level1": {"level2": {"level3": {"value": "old"}}}}
        override = {"level1": {"level2": {"level3": {"value": "new"}}}}
        result = deep_merge(base, override)
        assert result == {"level1": {"level2": {"level3": {"value": "new"}}}}

    def test_merge_preserves_base(self) -> None:
        """Test that merge doesn't modify base dictionary."""
        base = {"a": 1, "b": {"c": 2}}
        override = {"b": {"c": 3}}
        original_base = copy.deepcopy(base)

        deep_merge(base, override)

        assert base == original_base

    def test_merge_preserves_override(self) -> None:
        """Test that merge doesn't modify override dictionary."""
        base = {"a": 1}
        override = {"b": 2, "c": {"d": 3}}
        original_override = copy.deepcopy(override)

        deep_merge(base, override)

        assert override == original_override

    def test_merge_different_types(self) -> None:
        """Test merging when types differ (override wins)."""
        base = {"key": "string"}
        override = {"key": {"nested": "value"}}
        result = deep_merge(base, override)
        assert result == {"key": {"nested": "value"}}

    def test_merge_list_values(self) -> None:
        """Test merging with list values (override replaces)."""
        base = {"key": [1, 2, 3]}
        override = {"key": [4, 5, 6]}
        result = deep_merge(base, override)
        assert result == {"key": [4, 5, 6]}

    def test_merge_none_values(self) -> None:
        """Test merging with None values."""
        base = {"key1": "value1", "key2": None}
        override = {"key2": "value2", "key3": None}
        result = deep_merge(base, override)
        assert result == {"key1": "value1", "key2": "value2", "key3": None}

    def test_merge_mixed_types(self) -> None:
        """Test merging with various value types."""
        base = {
            "string": "text",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2],
            "dict": {"nested": "value"},
        }
        override = {
            "int": 100,
            "bool": False,
            "new_key": "new_value",
        }
        result = deep_merge(base, override)

        assert result["string"] == "text"
        assert result["int"] == 100
        assert result["float"] == 3.14
        assert result["bool"] is False
        assert result["list"] == [1, 2]
        assert result["new_key"] == "new_value"

    def test_merge_multiple_levels(self) -> None:
        """Test merging with multiple nested levels."""
        base = {
            "a": {"b": {"c": 1, "d": 2}, "e": 3},
            "f": 4,
        }
        override = {
            "a": {"b": {"d": 20}, "g": 30},
            "h": 40,
        }
        result = deep_merge(base, override)

        assert result == {
            "a": {"b": {"c": 1, "d": 20}, "e": 3, "g": 30},
            "f": 4,
            "h": 40,
        }


class TestInjectDefaults:
    """Test inject_defaults function."""

    def test_inject_protocol_defaults(self) -> None:
        """Test injecting protocol schema defaults."""
        config = {"name": "uart"}
        result = inject_defaults(config, "protocol")

        assert result["name"] == "uart"
        assert result["version"] == "1.0.0"
        assert "timing" in result

    def test_inject_pipeline_defaults(self) -> None:
        """Test injecting pipeline schema defaults."""
        config = {"name": "test_pipeline"}
        result = inject_defaults(config, "pipeline")

        assert result["name"] == "test_pipeline"
        assert result["version"] == "1.0.0"
        assert result["parallel_groups"] == []

    def test_inject_logic_family_defaults(self) -> None:
        """Test injecting logic family defaults."""
        config = {"name": "custom_ttl"}
        result = inject_defaults(config, "logic_family")

        assert result["name"] == "custom_ttl"
        assert "temperature_range" in result
        assert result["temperature_range"]["min"] == 0
        assert result["temperature_range"]["max"] == 70

    def test_inject_threshold_profile_defaults(self) -> None:
        """Test injecting threshold profile defaults."""
        config = {"name": "my_profile"}
        result = inject_defaults(config, "threshold_profile")

        assert result["name"] == "my_profile"
        assert result["tolerance"] == 0
        assert result["overrides"] == {}

    def test_inject_preferences_defaults(self) -> None:
        """Test injecting preferences defaults."""
        config = {"theme": "dark"}
        result = inject_defaults(config, "preferences")

        assert result["theme"] == "dark"
        assert "defaults" in result
        assert result["defaults"]["sample_rate"] == 1e6

    def test_inject_preserves_user_values(self) -> None:
        """Test that inject_defaults preserves user-provided values."""
        config = {
            "name": "uart",
            "version": "2.0.0",
            "timing": {"baud_rates": [115200]},
        }
        result = inject_defaults(config, "protocol")

        assert result["version"] == "2.0.0"  # User value preserved
        assert result["timing"]["baud_rates"] == [115200]

    def test_inject_unknown_schema_returns_copy(self) -> None:
        """Test that unknown schema returns deep copy of config."""
        config = {"custom": "config"}
        result = inject_defaults(config, "unknown_schema")

        assert result == config
        assert result is not config

    def test_inject_empty_config(self) -> None:
        """Test injecting defaults into empty config."""
        config: dict[str, Any] = {}
        result = inject_defaults(config, "protocol")

        assert result["version"] == "1.0.0"
        assert "timing" in result

    def test_inject_does_not_modify_original(self) -> None:
        """Test that inject_defaults doesn't modify original config."""
        config = {"name": "test"}
        original = copy.deepcopy(config)

        inject_defaults(config, "protocol")

        assert config == original


class TestGetEffectiveConfig:
    """Test get_effective_config function."""

    def test_get_effective_config_defaults_only(self) -> None:
        """Test getting effective config with no user config."""
        result = get_effective_config()

        assert "version" in result
        assert "defaults" in result
        assert result["defaults"]["sample_rate"] == 1e6

    def test_get_effective_config_with_user_config(self) -> None:
        """Test getting effective config with user overrides."""
        user_config = {"defaults": {"sample_rate": 2e6}}
        result = get_effective_config(user_config)

        assert result["defaults"]["sample_rate"] == 2e6
        # Other defaults should still be present
        assert "window_function" in result["defaults"]

    def test_get_effective_config_with_schema(self) -> None:
        """Test getting effective config with schema-specific defaults."""
        user_config = {"name": "test"}
        result = get_effective_config(user_config, "protocol")

        # Should have base defaults
        assert "defaults" in result
        # Should have schema defaults
        assert "version" in result
        assert "timing" in result
        # Should have user config
        assert result["name"] == "test"

    def test_get_effective_config_merge_order(self) -> None:
        """Test that merge order is correct: base -> schema -> user."""
        user_config = {
            "defaults": {"sample_rate": 3e6},
            "timing": {"data_bits": [7]},
        }
        result = get_effective_config(user_config, "protocol")

        # User value wins
        assert result["defaults"]["sample_rate"] == 3e6
        # Schema defaults present
        assert "timing" in result
        # User timing override
        assert result["timing"]["data_bits"] == [7]
        # Base defaults still present
        assert "loaders" in result

    def test_get_effective_config_none_values(self) -> None:
        """Test handling of None values."""
        result1 = get_effective_config(None, None)
        assert "defaults" in result1

        result2 = get_effective_config(None, "protocol")
        assert "version" in result2

    def test_get_effective_config_empty_user_config(self) -> None:
        """Test with empty user config."""
        result = get_effective_config({}, "protocol")

        # Should have all defaults
        assert "defaults" in result
        assert "version" in result
        assert "timing" in result


class TestGetDefault:
    """Test get_default function."""

    def test_get_default_top_level(self) -> None:
        """Test getting top-level default value."""
        result = get_default("version")
        assert result == "1.0"

    def test_get_default_nested(self) -> None:
        """Test getting nested default value."""
        result = get_default("defaults.sample_rate")
        assert result == 1e6

    def test_get_default_deeply_nested(self) -> None:
        """Test getting deeply nested value."""
        result = get_default("measurements.rise_time.ref_levels")
        assert result == [0.1, 0.9]

    def test_get_default_missing_key(self) -> None:
        """Test getting missing key returns None."""
        result = get_default("nonexistent.key.path")
        assert result is None

    def test_get_default_with_schema(self) -> None:
        """Test getting schema-specific default."""
        result = get_default("version", "protocol")
        assert result == "1.0.0"  # Schema default, not base default

    def test_get_default_schema_priority(self) -> None:
        """Test that schema defaults take priority over base."""
        # Version exists in both base and schema
        base_version = get_default("version")
        schema_version = get_default("version", "protocol")

        assert base_version == "1.0"
        assert schema_version == "1.0.0"

    def test_get_default_fallback_to_base(self) -> None:
        """Test fallback to base when schema doesn't have key."""
        result = get_default("defaults.sample_rate", "protocol")
        # Protocol schema doesn't have this, should fall back to base
        assert result == 1e6

    def test_get_default_various_types(self) -> None:
        """Test getting defaults of various types."""
        assert isinstance(get_default("version"), str)
        assert isinstance(get_default("defaults.sample_rate"), float)
        assert isinstance(get_default("defaults.fft_size"), int)
        assert isinstance(get_default("loaders.auto_detect"), bool)
        assert isinstance(get_default("loaders.formats"), list)
        assert isinstance(get_default("loaders.tektronix"), dict)


class TestDefaultConstants:
    """Test default configuration constants."""

    def test_default_config_structure(self) -> None:
        """Test DEFAULT_CONFIG has expected structure."""
        assert "version" in DEFAULT_CONFIG
        assert "defaults" in DEFAULT_CONFIG
        assert "loaders" in DEFAULT_CONFIG
        assert "measurements" in DEFAULT_CONFIG
        assert "spectral" in DEFAULT_CONFIG
        assert "visualization" in DEFAULT_CONFIG
        assert "export" in DEFAULT_CONFIG
        assert "logging" in DEFAULT_CONFIG

    def test_default_config_types(self) -> None:
        """Test DEFAULT_CONFIG value types."""
        assert isinstance(DEFAULT_CONFIG["version"], str)
        assert isinstance(DEFAULT_CONFIG["defaults"], dict)
        assert isinstance(DEFAULT_CONFIG["defaults"]["sample_rate"], float)
        assert isinstance(DEFAULT_CONFIG["defaults"]["fft_size"], int)

    def test_schema_defaults_structure(self) -> None:
        """Test SCHEMA_DEFAULTS has expected schemas."""
        assert "protocol" in SCHEMA_DEFAULTS
        assert "pipeline" in SCHEMA_DEFAULTS
        assert "logic_family" in SCHEMA_DEFAULTS
        assert "threshold_profile" in SCHEMA_DEFAULTS
        assert "preferences" in SCHEMA_DEFAULTS

    def test_schema_defaults_protocol(self) -> None:
        """Test protocol schema defaults."""
        protocol = SCHEMA_DEFAULTS["protocol"]
        assert protocol["version"] == "1.0.0"
        assert "timing" in protocol

    def test_schema_defaults_pipeline(self) -> None:
        """Test pipeline schema defaults."""
        pipeline = SCHEMA_DEFAULTS["pipeline"]
        assert pipeline["version"] == "1.0.0"
        assert pipeline["parallel_groups"] == []

    def test_default_config_immutability(self) -> None:
        """Test that DEFAULT_CONFIG is not accidentally modified."""
        original = copy.deepcopy(DEFAULT_CONFIG)

        # Attempt to use defaults (shouldn't modify original)
        config = inject_defaults({}, "protocol")
        config["version"] = "modified"

        assert original == DEFAULT_CONFIG


class TestConfigDefaultsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_deep_merge_with_none_dict(self) -> None:
        """Test deep_merge handles None as dict value."""
        base = {"key": None}
        override = {"key": {"nested": "value"}}
        result = deep_merge(base, override)
        assert result == {"key": {"nested": "value"}}

    def test_inject_defaults_with_nested_override(self) -> None:
        """Test inject_defaults with deeply nested user values."""
        config = {
            "timing": {
                "data_bits": [7, 8, 9],
                "stop_bits": [1, 2],
                "custom_field": "custom_value",
            }
        }
        result = inject_defaults(config, "protocol")

        # User values preserved
        assert result["timing"]["data_bits"] == [7, 8, 9]
        assert result["timing"]["custom_field"] == "custom_value"
        # Defaults merged
        assert "parity" in result["timing"]

    def test_get_default_empty_path(self) -> None:
        """Test get_default with empty path."""
        result = get_default("")
        assert result is None

    def test_get_default_single_dot(self) -> None:
        """Test get_default with single dot."""
        result = get_default(".")
        assert result is None

    def test_get_effective_config_complex_merge(self) -> None:
        """Test complex multi-level merge scenario."""
        user_config = {
            "defaults": {
                "sample_rate": 10e6,
                "window_function": "hamming",
            },
            "loaders": {
                "auto_detect": False,
                "csv": {"delimiter": ";"},
            },
            "custom_section": {
                "custom_key": "custom_value",
            },
        }

        result = get_effective_config(user_config)

        # User overrides
        assert result["defaults"]["sample_rate"] == 10e6
        assert result["defaults"]["window_function"] == "hamming"
        assert result["loaders"]["auto_detect"] is False
        assert result["loaders"]["csv"]["delimiter"] == ";"

        # Base defaults preserved
        assert result["defaults"]["fft_size"] == 1024
        assert result["loaders"]["csv"]["skip_header"] == 0

        # Custom section
        assert result["custom_section"]["custom_key"] == "custom_value"
