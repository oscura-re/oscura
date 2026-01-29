"""Unit tests for application settings management.

Tests CFG-019, CFG-020, CFG-021
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscura.core.config.settings import (
    AnalysisSettings,
    CLIDefaults,
    OutputSettings,
    Settings,
    get_settings,
    load_settings,
    reset_settings,
    save_settings,
    set_settings,
)
from oscura.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


class TestCLIDefaults:
    """Test CLIDefaults dataclass."""

    def test_cli_defaults_creation(self) -> None:
        """Test creating CLIDefaults with default values."""
        cli = CLIDefaults()

        assert cli.output_format == "text"
        assert cli.verbosity == 1
        assert cli.color_output is True
        assert cli.progress_bar is True
        assert cli.parallel_workers == 4

    def test_cli_defaults_custom_values(self) -> None:
        """Test creating CLIDefaults with custom values."""
        cli = CLIDefaults(
            output_format="json",
            verbosity=3,
            color_output=False,
            progress_bar=False,
            parallel_workers=8,
        )

        assert cli.output_format == "json"
        assert cli.verbosity == 3
        assert cli.color_output is False
        assert cli.progress_bar is False
        assert cli.parallel_workers == 8

    def test_cli_defaults_partial_override(self) -> None:
        """Test creating CLIDefaults with partial overrides."""
        cli = CLIDefaults(verbosity=2, parallel_workers=2)

        assert cli.output_format == "text"
        assert cli.verbosity == 2
        assert cli.color_output is True
        assert cli.parallel_workers == 2


class TestAnalysisSettings:
    """Test AnalysisSettings dataclass."""

    def test_analysis_settings_creation(self) -> None:
        """Test creating AnalysisSettings with default values."""
        analysis = AnalysisSettings()

        assert analysis.max_trace_size == 0
        assert analysis.enable_caching is True
        assert analysis.cache_dir is None
        assert analysis.timeout == 300.0
        assert analysis.streaming_mode is False

    def test_analysis_settings_custom_values(self) -> None:
        """Test creating AnalysisSettings with custom values."""
        analysis = AnalysisSettings(
            max_trace_size=1024**3,
            enable_caching=False,
            cache_dir="/tmp/cache",
            timeout=600.0,
            streaming_mode=True,
        )

        assert analysis.max_trace_size == 1024**3
        assert analysis.enable_caching is False
        assert analysis.cache_dir == "/tmp/cache"
        assert analysis.timeout == 600.0
        assert analysis.streaming_mode is True

    def test_analysis_settings_with_zero_timeout(self) -> None:
        """Test AnalysisSettings with zero timeout."""
        analysis = AnalysisSettings(timeout=0.0)
        assert analysis.timeout == 0.0


class TestOutputSettings:
    """Test OutputSettings dataclass."""

    def test_output_settings_creation(self) -> None:
        """Test creating OutputSettings with default values."""
        output = OutputSettings()

        assert output.default_format == "csv"
        assert output.include_raw_data is False
        assert output.compress_output is False
        assert output.decimal_places == 6
        assert output.timestamp_format == "iso8601"

    def test_output_settings_custom_values(self) -> None:
        """Test creating OutputSettings with custom values."""
        output = OutputSettings(
            default_format="json",
            include_raw_data=True,
            compress_output=True,
            decimal_places=3,
            timestamp_format="unix",
        )

        assert output.default_format == "json"
        assert output.include_raw_data is True
        assert output.compress_output is True
        assert output.decimal_places == 3
        assert output.timestamp_format == "unix"


class TestSettingsBasics:
    """Test Settings basic functionality."""

    def test_settings_creation(self) -> None:
        """Test creating Settings with default values."""
        settings = Settings()

        assert isinstance(settings.cli, CLIDefaults)
        assert isinstance(settings.analysis, AnalysisSettings)
        assert isinstance(settings.output, OutputSettings)
        assert settings.features == {}
        assert settings.custom == {}

    def test_settings_default_submodules(self) -> None:
        """Test that Settings creates default submodules."""
        settings = Settings()

        assert settings.cli.verbosity == 1
        assert settings.analysis.timeout == 300.0
        assert settings.output.default_format == "csv"

    def test_settings_custom_submodules(self) -> None:
        """Test creating Settings with custom submodules."""
        cli = CLIDefaults(verbosity=2)
        analysis = AnalysisSettings(timeout=600.0)
        output = OutputSettings(default_format="json")

        settings = Settings(cli=cli, analysis=analysis, output=output)

        assert settings.cli.verbosity == 2
        assert settings.analysis.timeout == 600.0
        assert settings.output.default_format == "json"

    def test_settings_with_features(self) -> None:
        """Test Settings with feature flags."""
        settings = Settings(features={"advanced_analysis": True, "beta": False})

        assert settings.features["advanced_analysis"] is True
        assert settings.features["beta"] is False

    def test_settings_with_custom(self) -> None:
        """Test Settings with custom settings."""
        custom = {"my_setting": 42, "my_dict": {"nested": "value"}}
        settings = Settings(custom=custom)

        assert settings.custom["my_setting"] == 42
        assert settings.custom["my_dict"]["nested"] == "value"


class TestFeatureFlags:
    """Test feature flag functionality."""

    def test_enable_feature(self) -> None:
        """Test enabling a feature."""
        settings = Settings()

        settings.enable_feature("advanced_analysis")

        assert settings.is_feature_enabled("advanced_analysis") is True

    def test_disable_feature(self) -> None:
        """Test disabling a feature."""
        settings = Settings()
        settings.enable_feature("advanced_analysis")

        settings.disable_feature("advanced_analysis")

        assert settings.is_feature_enabled("advanced_analysis") is False

    def test_is_feature_enabled_default_false(self) -> None:
        """Test that undefined features are disabled by default."""
        settings = Settings()

        assert settings.is_feature_enabled("undefined_feature") is False

    def test_enable_multiple_features(self) -> None:
        """Test enabling multiple features."""
        settings = Settings()

        settings.enable_feature("feature1")
        settings.enable_feature("feature2")
        settings.enable_feature("feature3")

        assert settings.is_feature_enabled("feature1") is True
        assert settings.is_feature_enabled("feature2") is True
        assert settings.is_feature_enabled("feature3") is True
        assert settings.is_feature_enabled("feature4") is False

    def test_enable_disable_toggle(self) -> None:
        """Test toggling feature on and off."""
        settings = Settings()

        settings.enable_feature("toggle_feature")
        assert settings.is_feature_enabled("toggle_feature") is True

        settings.disable_feature("toggle_feature")
        assert settings.is_feature_enabled("toggle_feature") is False

        settings.enable_feature("toggle_feature")
        assert settings.is_feature_enabled("toggle_feature") is True


class TestSettingsGet:
    """Test Settings.get() method."""

    def test_get_top_level_cli_value(self) -> None:
        """Test getting top-level CLI value."""
        settings = Settings()

        result = settings.get("cli.verbosity")

        assert result == 1

    def test_get_nested_value(self) -> None:
        """Test getting nested setting value."""
        settings = Settings()

        result = settings.get("cli.output_format")

        assert result == "text"

    def test_get_analysis_value(self) -> None:
        """Test getting analysis setting value."""
        settings = Settings()

        result = settings.get("analysis.timeout")

        assert result == 300.0

    def test_get_output_value(self) -> None:
        """Test getting output setting value."""
        settings = Settings()

        result = settings.get("output.decimal_places")

        assert result == 6

    def test_get_custom_value(self) -> None:
        """Test getting custom setting value."""
        settings = Settings(custom={"my_key": "my_value"})

        result = settings.get("custom.my_key")

        assert result == "my_value"

    def test_get_missing_key_returns_default(self) -> None:
        """Test that missing key returns default value."""
        settings = Settings()

        result = settings.get("missing.key", default="fallback")

        assert result == "fallback"

    def test_get_missing_key_no_default(self) -> None:
        """Test that missing key returns None when no default."""
        settings = Settings()

        result = settings.get("missing.key")

        assert result is None

    def test_get_nested_custom(self) -> None:
        """Test getting nested custom value."""
        settings = Settings(custom={"level1": {"level2": {"level3": "value"}}})

        result = settings.get("custom.level1.level2.level3")

        assert result == "value"

    def test_get_with_various_types(self) -> None:
        """Test getting values of various types."""
        settings = Settings(
            custom={
                "bool_val": True,
                "int_val": 42,
                "float_val": 3.14,
                "list_val": [1, 2, 3],
                "dict_val": {"nested": "value"},
            }
        )

        assert settings.get("custom.bool_val") is True
        assert settings.get("custom.int_val") == 42
        assert settings.get("custom.float_val") == 3.14
        assert settings.get("custom.list_val") == [1, 2, 3]
        assert settings.get("custom.dict_val") == {"nested": "value"}

    def test_get_nonexistent_nested_path_returns_default(self) -> None:
        """Test that nonexistent nested path returns default."""
        settings = Settings()

        result = settings.get("cli.nonexistent.path", default=42)

        assert result == 42


class TestSettingsSet:
    """Test Settings.set() method."""

    def test_set_cli_value(self) -> None:
        """Test setting CLI value."""
        settings = Settings()

        settings.set("cli.verbosity", 3)

        assert settings.cli.verbosity == 3

    def test_set_analysis_value(self) -> None:
        """Test setting analysis value."""
        settings = Settings()

        settings.set("analysis.timeout", 600.0)

        assert settings.analysis.timeout == 600.0

    def test_set_output_value(self) -> None:
        """Test setting output value."""
        settings = Settings()

        settings.set("output.default_format", "json")

        assert settings.output.default_format == "json"

    def test_set_custom_value(self) -> None:
        """Test setting custom value."""
        settings = Settings()

        settings.set("custom.my_key", "my_value")

        assert settings.custom["my_key"] == "my_value"

    def test_set_nested_custom_value(self) -> None:
        """Test setting nested custom value."""
        settings = Settings()

        settings.set("custom.level1.level2.level3", "deep_value")

        assert settings.custom["level1"]["level2"]["level3"] == "deep_value"

    def test_set_multiple_values(self) -> None:
        """Test setting multiple values."""
        settings = Settings()

        settings.set("cli.verbosity", 2)
        settings.set("analysis.timeout", 500.0)
        settings.set("output.default_format", "json")

        assert settings.cli.verbosity == 2
        assert settings.analysis.timeout == 500.0
        assert settings.output.default_format == "json"

    def test_set_invalid_path_raises_error(self) -> None:
        """Test that setting invalid path raises KeyError."""
        settings = Settings()

        with pytest.raises(KeyError, match="Invalid setting path"):
            settings.set("nonexistent.section.key", "value")

    def test_set_unknown_attribute_raises_error(self) -> None:
        """Test that setting unknown attribute raises KeyError."""
        settings = Settings()

        with pytest.raises(KeyError, match="Unknown setting"):
            settings.set("cli.nonexistent_attribute", "value")

    def test_set_overwrites_existing_value(self) -> None:
        """Test that set overwrites existing values."""
        settings = Settings(custom={"key": "original"})

        settings.set("custom.key", "updated")

        assert settings.custom["key"] == "updated"

    def test_set_with_various_types(self) -> None:
        """Test setting values of various types."""
        settings = Settings()

        settings.set("custom.bool_val", False)
        settings.set("custom.int_val", 100)
        settings.set("custom.float_val", 2.71)
        settings.set("custom.list_val", [4, 5, 6])
        settings.set("custom.dict_val", {"nested": "data"})

        assert settings.custom["bool_val"] is False
        assert settings.custom["int_val"] == 100
        assert settings.custom["float_val"] == 2.71
        assert settings.custom["list_val"] == [4, 5, 6]
        assert settings.custom["dict_val"] == {"nested": "data"}


class TestSettingsToDict:
    """Test Settings.to_dict() method."""

    def test_to_dict_structure(self) -> None:
        """Test that to_dict has correct structure."""
        settings = Settings()
        data = settings.to_dict()

        assert "cli" in data
        assert "analysis" in data
        assert "output" in data
        assert "features" in data
        assert "custom" in data

    def test_to_dict_cli_values(self) -> None:
        """Test CLI values in to_dict output."""
        settings = Settings()
        data = settings.to_dict()

        assert data["cli"]["output_format"] == "text"
        assert data["cli"]["verbosity"] == 1
        assert data["cli"]["color_output"] is True
        assert data["cli"]["progress_bar"] is True
        assert data["cli"]["parallel_workers"] == 4

    def test_to_dict_analysis_values(self) -> None:
        """Test analysis values in to_dict output."""
        settings = Settings()
        data = settings.to_dict()

        assert data["analysis"]["max_trace_size"] == 0
        assert data["analysis"]["enable_caching"] is True
        assert data["analysis"]["cache_dir"] is None
        assert data["analysis"]["timeout"] == 300.0
        assert data["analysis"]["streaming_mode"] is False

    def test_to_dict_output_values(self) -> None:
        """Test output values in to_dict output."""
        settings = Settings()
        data = settings.to_dict()

        assert data["output"]["default_format"] == "csv"
        assert data["output"]["include_raw_data"] is False
        assert data["output"]["compress_output"] is False
        assert data["output"]["decimal_places"] == 6
        assert data["output"]["timestamp_format"] == "iso8601"

    def test_to_dict_with_features(self) -> None:
        """Test features in to_dict output."""
        settings = Settings()
        settings.enable_feature("feature1")
        settings.enable_feature("feature2")

        data = settings.to_dict()

        assert data["features"]["feature1"] is True
        assert data["features"]["feature2"] is True

    def test_to_dict_with_custom(self) -> None:
        """Test custom settings in to_dict output."""
        settings = Settings(custom={"key1": "value1", "key2": 42})

        data = settings.to_dict()

        assert data["custom"]["key1"] == "value1"
        assert data["custom"]["key2"] == 42

    def test_to_dict_with_custom_values(self) -> None:
        """Test to_dict after setting custom values."""
        settings = Settings()
        settings.set("custom.test_key", "test_value")
        settings.set("cli.verbosity", 3)

        data = settings.to_dict()

        assert data["custom"]["test_key"] == "test_value"
        assert data["cli"]["verbosity"] == 3

    def test_to_dict_is_serializable(self) -> None:
        """Test that to_dict output is JSON serializable."""
        settings = Settings()
        settings.enable_feature("test")
        settings.set("custom.key", "value")

        data = settings.to_dict()
        json_str = json.dumps(data)

        assert isinstance(json_str, str)
        assert "test" in json_str


class TestSettingsFromDict:
    """Test Settings.from_dict() method."""

    def test_from_dict_empty_dict(self) -> None:
        """Test creating Settings from empty dict."""
        settings = Settings.from_dict({})

        # Should have defaults
        assert settings.cli.verbosity == 1
        assert settings.analysis.timeout == 300.0
        assert settings.output.default_format == "csv"

    def test_from_dict_cli_values(self) -> None:
        """Test from_dict with CLI values."""
        data = {
            "cli": {
                "output_format": "json",
                "verbosity": 2,
                "color_output": False,
                "progress_bar": False,
                "parallel_workers": 8,
            }
        }

        settings = Settings.from_dict(data)

        assert settings.cli.output_format == "json"
        assert settings.cli.verbosity == 2
        assert settings.cli.color_output is False
        assert settings.cli.progress_bar is False
        assert settings.cli.parallel_workers == 8

    def test_from_dict_analysis_values(self) -> None:
        """Test from_dict with analysis values."""
        data = {
            "analysis": {
                "max_trace_size": 1024**3,
                "enable_caching": False,
                "cache_dir": "/tmp/cache",
                "timeout": 600.0,
                "streaming_mode": True,
            }
        }

        settings = Settings.from_dict(data)

        assert settings.analysis.max_trace_size == 1024**3
        assert settings.analysis.enable_caching is False
        assert settings.analysis.cache_dir == "/tmp/cache"
        assert settings.analysis.timeout == 600.0
        assert settings.analysis.streaming_mode is True

    def test_from_dict_output_values(self) -> None:
        """Test from_dict with output values."""
        data = {
            "output": {
                "default_format": "hdf5",
                "include_raw_data": True,
                "compress_output": True,
                "decimal_places": 3,
                "timestamp_format": "unix",
            }
        }

        settings = Settings.from_dict(data)

        assert settings.output.default_format == "hdf5"
        assert settings.output.include_raw_data is True
        assert settings.output.compress_output is True
        assert settings.output.decimal_places == 3
        assert settings.output.timestamp_format == "unix"

    def test_from_dict_features(self) -> None:
        """Test from_dict with features."""
        data = {"features": {"feature1": True, "feature2": False, "feature3": True}}

        settings = Settings.from_dict(data)

        assert settings.is_feature_enabled("feature1") is True
        assert settings.is_feature_enabled("feature2") is False
        assert settings.is_feature_enabled("feature3") is True

    def test_from_dict_custom(self) -> None:
        """Test from_dict with custom settings."""
        data = {"custom": {"key1": "value1", "key2": 42, "nested": {"key3": "value3"}}}

        settings = Settings.from_dict(data)

        assert settings.custom["key1"] == "value1"
        assert settings.custom["key2"] == 42
        assert settings.custom["nested"]["key3"] == "value3"

    def test_from_dict_partial_data(self) -> None:
        """Test from_dict with partial data (missing sections)."""
        data = {
            "cli": {"verbosity": 2},
            "custom": {"my_key": "my_value"},
        }

        settings = Settings.from_dict(data)

        # CLI should be updated
        assert settings.cli.verbosity == 2
        # Other CLI values should be defaults
        assert settings.cli.output_format == "text"
        # Analysis should be defaults
        assert settings.analysis.timeout == 300.0
        # Custom should be set
        assert settings.custom["my_key"] == "my_value"

    def test_from_dict_roundtrip(self) -> None:
        """Test roundtrip: to_dict -> from_dict -> to_dict."""
        original = Settings()
        original.enable_feature("test_feature")
        original.set("cli.verbosity", 2)
        original.set("analysis.timeout", 500.0)
        original.set("custom.my_key", "my_value")

        data = original.to_dict()
        restored = Settings.from_dict(data)
        restored_data = restored.to_dict()

        assert restored_data == data


class TestLoadSettings:
    """Test load_settings function."""

    def test_load_settings_valid_json(self, tmp_path: Path) -> None:
        """Test loading settings from valid JSON file."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(
            json.dumps(
                {
                    "cli": {"verbosity": 2},
                    "analysis": {"timeout": 600.0},
                }
            )
        )

        settings = load_settings(settings_file)

        assert settings.cli.verbosity == 2
        assert settings.analysis.timeout == 600.0

    def test_load_settings_from_string_path(self, tmp_path: Path) -> None:
        """Test loading settings from string path."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"cli": {"verbosity": 3}}))

        settings = load_settings(str(settings_file))

        assert settings.cli.verbosity == 3

    def test_load_settings_with_expanduser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading settings with tilde expansion."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"cli": {"verbosity": 2}}))

        def mock_expanduser(self: Path) -> Path:
            path_str = str(self)
            if path_str.startswith("~"):
                return tmp_path / path_str[2:]
            return self

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        settings = load_settings("~/settings.json")

        assert settings.cli.verbosity == 2

    def test_load_settings_file_not_found(self, tmp_path: Path) -> None:
        """Test error when settings file not found."""
        settings_file = tmp_path / "nonexistent.json"

        with pytest.raises(ConfigurationError, match="not found"):
            load_settings(settings_file)

    def test_load_settings_invalid_json(self, tmp_path: Path) -> None:
        """Test error with invalid JSON."""
        settings_file = tmp_path / "invalid.json"
        settings_file.write_text("{invalid json}")

        with pytest.raises(ConfigurationError, match="Failed to parse settings JSON"):
            load_settings(settings_file)

    def test_load_settings_not_dict(self, tmp_path: Path) -> None:
        """Test error when JSON is not a dict."""
        settings_file = tmp_path / "array.json"
        settings_file.write_text(json.dumps([1, 2, 3]))

        with pytest.raises(ConfigurationError, match="must contain a JSON object"):
            load_settings(settings_file)

    def test_load_settings_empty_file(self, tmp_path: Path) -> None:
        """Test loading from empty file."""
        settings_file = tmp_path / "empty.json"
        settings_file.write_text("{}")

        settings = load_settings(settings_file)

        # Should have defaults
        assert settings.cli.verbosity == 1

    def test_load_settings_complex_config(self, tmp_path: Path) -> None:
        """Test loading complex settings file."""
        settings_file = tmp_path / "complex.json"
        config = {
            "cli": {
                "output_format": "json",
                "verbosity": 2,
                "color_output": False,
                "progress_bar": False,
                "parallel_workers": 8,
            },
            "analysis": {
                "max_trace_size": 1024**3,
                "enable_caching": False,
                "cache_dir": "/tmp/cache",
                "timeout": 600.0,
                "streaming_mode": True,
            },
            "output": {
                "default_format": "json",
                "include_raw_data": True,
                "compress_output": True,
                "decimal_places": 3,
                "timestamp_format": "unix",
            },
            "features": {"advanced_analysis": True, "beta": False},
            "custom": {"my_setting": 42, "nested": {"key": "value"}},
        }
        settings_file.write_text(json.dumps(config))

        settings = load_settings(settings_file)

        assert settings.cli.verbosity == 2
        assert settings.analysis.streaming_mode is True
        assert settings.output.default_format == "json"
        assert settings.is_feature_enabled("advanced_analysis") is True
        assert settings.custom["my_setting"] == 42


class TestSaveSettings:
    """Test save_settings function."""

    def test_save_settings_json(self, tmp_path: Path) -> None:
        """Test saving settings to JSON file."""
        settings_file = tmp_path / "settings.json"
        settings = Settings()
        settings.set("cli.verbosity", 2)
        settings.set("analysis.timeout", 600.0)

        save_settings(settings, settings_file)

        assert settings_file.exists()
        with open(settings_file) as f:
            data = json.load(f)
        assert data["cli"]["verbosity"] == 2
        assert data["analysis"]["timeout"] == 600.0

    def test_save_settings_with_string_path(self, tmp_path: Path) -> None:
        """Test saving settings with string path."""
        settings_file = tmp_path / "settings.json"
        settings = Settings()

        save_settings(settings, str(settings_file))

        assert settings_file.exists()

    def test_save_settings_with_expanduser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test saving settings with tilde expansion."""
        settings_file = tmp_path / "settings.json"

        def mock_expanduser(self: Path) -> Path:
            path_str = str(self)
            if path_str.startswith("~"):
                return tmp_path / path_str[2:]
            return self

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        settings = Settings()
        save_settings(settings, "~/settings.json")

        assert settings_file.exists()

    def test_save_settings_creates_parent_dir(self, tmp_path: Path) -> None:
        """Test that save_settings creates parent directories."""
        settings_file = tmp_path / "nested" / "dir" / "settings.json"
        settings = Settings()

        save_settings(settings, settings_file)

        assert settings_file.exists()

    def test_save_settings_overwrites_existing(self, tmp_path: Path) -> None:
        """Test that save_settings overwrites existing file."""
        settings_file = tmp_path / "settings.json"
        settings_file.write_text('{"old": "data"}')

        settings = Settings()
        settings.set("cli.verbosity", 3)
        save_settings(settings, settings_file)

        with open(settings_file) as f:
            data = json.load(f)
        assert "old" not in data
        assert data["cli"]["verbosity"] == 3

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Test save and load roundtrip."""
        settings_file = tmp_path / "settings.json"

        # Create and save settings
        original = Settings()
        original.set("cli.verbosity", 2)
        original.set("analysis.timeout", 500.0)
        original.enable_feature("test_feature")
        original.set("custom.my_key", "my_value")
        save_settings(original, settings_file)

        # Load and verify
        loaded = load_settings(settings_file)
        assert loaded.cli.verbosity == 2
        assert loaded.analysis.timeout == 500.0
        assert loaded.is_feature_enabled("test_feature") is True
        assert loaded.custom["my_key"] == "my_value"


class TestGlobalSettings:
    """Test global settings functions."""

    def test_get_settings_returns_singleton(self) -> None:
        """Test that get_settings returns same instance."""
        reset_settings()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_set_settings(self) -> None:
        """Test setting global settings."""
        reset_settings()

        new_settings = Settings()
        new_settings.set("cli.verbosity", 3)

        set_settings(new_settings)
        global_settings = get_settings()

        assert global_settings.cli.verbosity == 3

    def test_reset_settings(self) -> None:
        """Test resetting global settings."""
        # First, modify settings
        settings = get_settings()
        settings.set("cli.verbosity", 3)
        assert settings.cli.verbosity == 3

        # Reset
        reset_settings()
        settings = get_settings()

        assert settings.cli.verbosity == 1

    def test_get_settings_modify_global(self) -> None:
        """Test that modifying returned settings affects global state."""
        reset_settings()

        settings = get_settings()
        settings.set("cli.verbosity", 2)

        new_ref = get_settings()

        assert new_ref.cli.verbosity == 2

    def test_multiple_set_and_get(self) -> None:
        """Test multiple set and get operations."""
        reset_settings()

        settings = get_settings()
        settings.set("cli.verbosity", 2)
        settings.set("analysis.timeout", 600.0)
        settings.enable_feature("feature1")

        settings2 = get_settings()

        assert settings2.cli.verbosity == 2
        assert settings2.analysis.timeout == 600.0
        assert settings2.is_feature_enabled("feature1") is True


class TestSettingsEdgeCases:
    """Test edge cases and error conditions."""

    def test_settings_with_none_values(self) -> None:
        """Test settings with None values."""
        settings = Settings()
        settings.analysis.cache_dir = None

        data = settings.to_dict()
        assert data["analysis"]["cache_dir"] is None

        restored = Settings.from_dict(data)
        assert restored.analysis.cache_dir is None

    def test_settings_with_empty_strings(self) -> None:
        """Test settings with empty strings."""
        settings = Settings()
        settings.set("custom.empty_string", "")

        assert settings.get("custom.empty_string") == ""

    def test_settings_with_zero_values(self) -> None:
        """Test settings with zero values."""
        settings = Settings()
        settings.analysis.timeout = 0.0
        settings.output.decimal_places = 0
        settings.cli.parallel_workers = 0

        data = settings.to_dict()
        assert data["analysis"]["timeout"] == 0.0
        assert data["output"]["decimal_places"] == 0
        assert data["cli"]["parallel_workers"] == 0

    def test_settings_with_large_numbers(self) -> None:
        """Test settings with large numbers."""
        settings = Settings()
        large_number = 10**15

        settings.analysis.max_trace_size = large_number
        assert settings.get("analysis.max_trace_size") == large_number

    def test_settings_get_empty_path(self) -> None:
        """Test get with empty path segments."""
        settings = Settings()

        # Should handle empty segments gracefully
        result = settings.get("custom", default="fallback")
        assert result == {}

    def test_custom_dict_mutation(self) -> None:
        """Test that custom dict can be mutated."""
        settings = Settings()
        settings.custom["key1"] = "value1"
        settings.custom["key2"] = "value2"

        assert settings.custom["key1"] == "value1"
        assert settings.custom["key2"] == "value2"

    def test_feature_flag_with_special_names(self) -> None:
        """Test feature flags with various naming patterns."""
        settings = Settings()

        settings.enable_feature("advanced-analysis")
        settings.enable_feature("beta_feature")
        settings.enable_feature("Feature.With.Dots")

        assert settings.is_feature_enabled("advanced-analysis") is True
        assert settings.is_feature_enabled("beta_feature") is True
        assert settings.is_feature_enabled("Feature.With.Dots") is True

    def test_settings_type_preservation(self) -> None:
        """Test that types are preserved through serialization."""
        settings = Settings()
        settings.set("custom.int_val", 42)
        settings.set("custom.float_val", 3.14)
        settings.set("custom.bool_val", True)
        settings.set("custom.str_val", "text")
        settings.set("custom.list_val", [1, 2, 3])

        data = settings.to_dict()
        restored = Settings.from_dict(data)

        assert isinstance(restored.custom["int_val"], int)
        assert isinstance(restored.custom["float_val"], float)
        assert isinstance(restored.custom["bool_val"], bool)
        assert isinstance(restored.custom["str_val"], str)
        assert isinstance(restored.custom["list_val"], list)

    def test_json_serialization_with_special_floats(self, tmp_path: Path) -> None:
        """Test JSON serialization handles standard numeric values."""
        settings_file = tmp_path / "settings.json"
        settings = Settings()
        settings.analysis.timeout = 1e-6  # Very small number
        settings.analysis.max_trace_size = int(1e9)  # Large number

        save_settings(settings, settings_file)
        loaded = load_settings(settings_file)

        assert loaded.analysis.timeout == 1e-6
        assert loaded.analysis.max_trace_size == int(1e9)


class TestConfigSettingsIntegration:
    """Integration tests combining multiple functions."""

    def test_create_modify_save_load_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: create, modify, save, load."""
        settings_file = tmp_path / "workflow_settings.json"

        # Create and modify
        settings = Settings()
        settings.cli.verbosity = 2
        settings.analysis.timeout = 500.0
        settings.enable_feature("advanced_analysis")
        settings.set("custom.project_name", "test_project")

        # Save
        save_settings(settings, settings_file)

        # Load
        loaded = load_settings(settings_file)

        # Verify
        assert loaded.cli.verbosity == 2
        assert loaded.analysis.timeout == 500.0
        assert loaded.is_feature_enabled("advanced_analysis") is True
        assert loaded.get("custom.project_name") == "test_project"

    def test_global_and_file_settings_sync(self, tmp_path: Path) -> None:
        """Test syncing global settings with file settings."""
        reset_settings()
        settings_file = tmp_path / "sync_settings.json"

        # Modify global
        global_settings = get_settings()
        global_settings.set("cli.verbosity", 2)
        global_settings.enable_feature("feature1")

        # Save to file
        save_settings(global_settings, settings_file)

        # Reset global
        reset_settings()

        # Verify reset worked
        assert get_settings().cli.verbosity == 1

        # Load from file
        loaded = load_settings(settings_file)
        set_settings(loaded)

        # Verify restored
        assert get_settings().cli.verbosity == 2
        assert get_settings().is_feature_enabled("feature1") is True

    def test_multiple_settings_files(self, tmp_path: Path) -> None:
        """Test working with multiple settings files."""
        file1 = tmp_path / "settings1.json"
        file2 = tmp_path / "settings2.json"

        # Create first settings
        settings1 = Settings()
        settings1.set("cli.verbosity", 1)
        save_settings(settings1, file1)

        # Create second settings
        settings2 = Settings()
        settings2.set("cli.verbosity", 3)
        save_settings(settings2, file2)

        # Load and verify
        loaded1 = load_settings(file1)
        loaded2 = load_settings(file2)

        assert loaded1.cli.verbosity == 1
        assert loaded2.cli.verbosity == 3

    def test_settings_merging_pattern(self) -> None:
        """Test pattern for merging settings."""
        base = Settings()
        base.set("cli.verbosity", 1)

        override = Settings()
        override.set("cli.verbosity", 2)
        override.set("analysis.timeout", 500.0)

        # Manual merge
        for key, value in override.to_dict()["cli"].items():
            base.set(f"cli.{key}", value)
        for key, value in override.to_dict()["analysis"].items():
            base.set(f"analysis.{key}", value)

        assert base.cli.verbosity == 2
        assert base.analysis.timeout == 500.0
