"""Unit tests for configuration file loading utilities.

Tests CFG-002, CFG-010
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from oscura.core.config.loader import (
    _load_json,
    _load_yaml,
    get_config_value,
    load_config,
    load_config_file,
    save_config,
)
from oscura.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


class TestLoadConfigFile:
    """Test load_config_file function."""

    def test_load_yaml_file(self, tmp_path: Path) -> None:
        """Test loading a valid YAML file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("name: test\nversion: '1.0'\n")

        config = load_config_file(config_file, validate=False)

        assert config["name"] == "test"
        assert config["version"] == "1.0"

    def test_load_yml_extension(self, tmp_path: Path) -> None:
        """Test loading .yml extension."""
        config_file = tmp_path / "test.yml"
        config_file.write_text("key: value\n")

        config = load_config_file(config_file, validate=False)

        assert config["key"] == "value"

    def test_load_json_file(self, tmp_path: Path) -> None:
        """Test loading a valid JSON file."""
        config_file = tmp_path / "test.json"
        config_file.write_text('{"name": "test", "version": "1.0"}')

        config = load_config_file(config_file, validate=False)

        assert config["name"] == "test"
        assert config["version"] == "1.0"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Test error when file does not exist."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(ConfigurationError, match="not found"):
            load_config_file(config_file)

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        """Test error with unsupported file extension."""
        config_file = tmp_path / "test.xyz"
        config_file.write_text("data")

        with pytest.raises(ConfigurationError, match="Unsupported configuration format"):
            load_config_file(config_file)

    def test_expanduser_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test path expansion with tilde."""
        # Create a config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value\n")

        # Mock expanduser to return our tmp_path
        def mock_expanduser(self: Path) -> Path:
            path_str = str(self)
            if path_str.startswith("~"):
                return tmp_path / path_str[2:]
            return self

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        # Use tilde path
        config = load_config_file("~/config.yaml", validate=False)

        assert config["test"] == "value"

    def test_validation_with_schema(self, tmp_path: Path) -> None:
        """Test validation against schema."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("name: test\n")

        # Mock the validate_against_schema function
        with patch("oscura.core.config.schema.validate_against_schema") as mock_validate:
            config = load_config_file(config_file, schema="protocol", validate=True)

            # Verify validation was called
            mock_validate.assert_called_once()
            assert config["name"] == "test"

    def test_validation_disabled(self, tmp_path: Path) -> None:
        """Test validation can be disabled."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("name: test\n")

        with patch("oscura.core.config.schema.validate_against_schema") as mock_validate:
            load_config_file(config_file, schema="protocol", validate=False)

            # Validation should not be called
            mock_validate.assert_not_called()

    def test_inject_defaults_enabled(self, tmp_path: Path) -> None:
        """Test default injection when enabled."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("name: test\n")

        with patch("oscura.core.config.defaults.inject_defaults") as mock_inject:
            mock_inject.return_value = {"name": "test", "default": "value"}

            config = load_config_file(
                config_file,
                schema="protocol",
                validate=False,
                inject_defaults=True,
            )

            mock_inject.assert_called_once()
            assert config["name"] == "test"

    def test_inject_defaults_disabled(self, tmp_path: Path) -> None:
        """Test default injection can be disabled."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("name: test\n")

        with patch("oscura.core.config.defaults.inject_defaults") as mock_inject:
            load_config_file(
                config_file,
                schema="protocol",
                validate=False,
                inject_defaults=False,
            )

            mock_inject.assert_not_called()

    def test_no_schema_no_validation(self, tmp_path: Path) -> None:
        """Test that validation is skipped when schema is None."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("name: test\n")

        with patch("oscura.core.config.schema.validate_against_schema") as mock_validate:
            load_config_file(config_file, schema=None, validate=True)

            # No validation without schema
            mock_validate.assert_not_called()


class TestLoadYaml:
    """Test _load_yaml helper function."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        """Test loading valid YAML content."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\nlist:\n  - item1\n  - item2\n")

        result = _load_yaml(yaml_file)

        assert result["key"] == "value"
        assert result["list"] == ["item1", "item2"]

    def test_empty_yaml_file(self, tmp_path: Path) -> None:
        """Test loading empty YAML file returns empty dict."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        result = _load_yaml(yaml_file)

        assert result == {}

    def test_yaml_not_dict(self, tmp_path: Path) -> None:
        """Test error when YAML is not a dictionary."""
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n")

        with pytest.raises(ConfigurationError, match="must be a dictionary"):
            _load_yaml(yaml_file)

    def test_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        """Test error on invalid YAML syntax."""
        yaml_file = tmp_path / "invalid.yaml"
        # Use truly invalid YAML that will always fail
        yaml_file.write_text("key: [value\n")  # Unclosed bracket

        with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
            _load_yaml(yaml_file)

    def test_yaml_file_read_error(self, tmp_path: Path) -> None:
        """Test error when YAML file cannot be read."""
        yaml_file = tmp_path / "test.yaml"

        with pytest.raises(ConfigurationError, match="Failed to read"):
            _load_yaml(yaml_file)

    @patch("oscura.core.config.loader.YAML_AVAILABLE", False)
    def test_yaml_not_available(self, tmp_path: Path) -> None:
        """Test error when PyYAML is not installed."""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("key: value\n")

        with pytest.raises(
            ConfigurationError,
            match="YAML support not available",
        ):
            _load_yaml(yaml_file)


class TestLoadJson:
    """Test _load_json helper function."""

    def test_valid_json(self, tmp_path: Path) -> None:
        """Test loading valid JSON content."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "number": 42}')

        result = _load_json(json_file)

        assert result["key"] == "value"
        assert result["number"] == 42

    def test_empty_json_object(self, tmp_path: Path) -> None:
        """Test loading empty JSON object."""
        json_file = tmp_path / "empty.json"
        json_file.write_text("{}")

        result = _load_json(json_file)

        assert result == {}

    def test_json_not_dict(self, tmp_path: Path) -> None:
        """Test error when JSON is not a dictionary."""
        json_file = tmp_path / "array.json"
        json_file.write_text("[1, 2, 3]")

        with pytest.raises(ConfigurationError, match="must be a dictionary"):
            _load_json(json_file)

    def test_invalid_json_syntax(self, tmp_path: Path) -> None:
        """Test error on invalid JSON syntax."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text('{"key": "value",}')  # Trailing comma

        with pytest.raises(ConfigurationError, match="Failed to parse JSON"):
            _load_json(json_file)

    def test_json_file_read_error(self, tmp_path: Path) -> None:
        """Test error when JSON file cannot be read."""
        json_file = tmp_path / "test.json"

        with pytest.raises(ConfigurationError, match="Failed to read"):
            _load_json(json_file)

    def test_json_null_becomes_empty_dict(self, tmp_path: Path) -> None:
        """Test that JSON null value becomes empty dict."""
        json_file = tmp_path / "null.json"
        json_file.write_text("null")

        result = _load_json(json_file)

        assert result == {}


class TestLoadConfig:
    """Test load_config function."""

    def test_load_with_explicit_path(self, tmp_path: Path) -> None:
        """Test loading config with explicit path."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("custom: value\n")

        config = load_config(config_file, use_defaults=False)

        assert config["custom"] == "value"

    def test_load_with_defaults(self, tmp_path: Path) -> None:
        """Test loading config merges with defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("custom: value\n")

        config = load_config(config_file, use_defaults=True)

        # Should have both custom value and defaults
        assert config["custom"] == "value"
        assert "version" in config  # From DEFAULT_CONFIG

    def test_search_cwd_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test searching for oscura.yaml in current directory."""
        config_file = tmp_path / "oscura.yaml"
        config_file.write_text("found: cwd_yaml\n")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(use_defaults=False)

        assert config["found"] == "cwd_yaml"

    def test_search_hidden_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test searching for .oscura.yaml in current directory."""
        config_file = tmp_path / ".oscura.yaml"
        config_file.write_text("found: hidden_yaml\n")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(use_defaults=False)

        assert config["found"] == "hidden_yaml"

    def test_search_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test searching for oscura.json in current directory."""
        config_file = tmp_path / "oscura.json"
        config_file.write_text('{"found": "cwd_json"}')

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(use_defaults=False)

        assert config["found"] == "cwd_json"

    def test_search_home_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test searching in ~/.oscura/config.yaml."""
        config_dir = tmp_path / ".oscura"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("found: home_config\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path / "other")

        config = load_config(use_defaults=False)

        assert config["found"] == "home_config"

    def test_search_xdg_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test searching in ~/.config/oscura/config.yaml."""
        config_dir = tmp_path / ".config" / "oscura"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.yaml"
        config_file.write_text("found: xdg_config\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path / "other")

        config = load_config(use_defaults=False)

        assert config["found"] == "xdg_config"

    def test_no_config_file_found(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test when no config file is found."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(use_defaults=False)

        assert config == {}

    def test_no_config_with_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test defaults returned when no config file found."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(use_defaults=True)

        # Should have default config
        assert "version" in config
        assert "defaults" in config

    def test_search_priority(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that cwd config takes priority over home config."""
        # Create both configs
        (tmp_path / "oscura.yaml").write_text("priority: cwd\n")
        home_config = tmp_path / "home" / ".oscura"
        home_config.mkdir(parents=True)
        (home_config / "config.yaml").write_text("priority: home\n")

        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)

        config = load_config(use_defaults=False)

        # CWD should win
        assert config["priority"] == "cwd"


class TestSaveConfig:
    """Test save_config function."""

    def test_save_yaml(self, tmp_path: Path) -> None:
        """Test saving config as YAML."""
        config_file = tmp_path / "output.yaml"
        config = {"key": "value", "number": 42}

        save_config(config, config_file)

        # Verify file was created and contains correct data
        assert config_file.exists()
        loaded = _load_yaml(config_file)
        assert loaded == config

    def test_save_json(self, tmp_path: Path) -> None:
        """Test saving config as JSON."""
        config_file = tmp_path / "output.json"
        config = {"key": "value", "number": 42}

        save_config(config, config_file)

        # Verify file was created and contains correct data
        assert config_file.exists()
        loaded = _load_json(config_file)
        assert loaded == config

    def test_save_with_explicit_yaml_format(self, tmp_path: Path) -> None:
        """Test saving with explicit YAML format."""
        config_file = tmp_path / "output.txt"
        config = {"key": "value"}

        save_config(config, config_file, format="yaml")

        # Should be YAML despite .txt extension
        loaded = _load_yaml(config_file)
        assert loaded == config

    def test_save_with_explicit_json_format(self, tmp_path: Path) -> None:
        """Test saving with explicit JSON format."""
        config_file = tmp_path / "output.txt"
        config = {"key": "value"}

        save_config(config, config_file, format="json")

        # Should be JSON despite .txt extension
        loaded = _load_json(config_file)
        assert loaded == config

    def test_save_unknown_extension_defaults_yaml(self, tmp_path: Path) -> None:
        """Test unknown extension defaults to YAML."""
        config_file = tmp_path / "output.xyz"
        config = {"key": "value"}

        save_config(config, config_file)

        # Should default to YAML
        loaded = _load_yaml(config_file)
        assert loaded == config

    def test_save_creates_parent_directory(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        config_file = tmp_path / "subdir" / "nested" / "config.yaml"
        config = {"key": "value"}

        save_config(config, config_file)

        assert config_file.exists()
        loaded = _load_yaml(config_file)
        assert loaded == config

    def test_save_expanduser(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test path expansion with tilde."""

        # Mock expanduser to return our tmp_path
        def mock_expanduser(self: Path) -> Path:
            path_str = str(self)
            if path_str.startswith("~"):
                return tmp_path / path_str[2:]
            return self

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        config = {"key": "value"}
        save_config(config, "~/config.yaml")

        saved_file = tmp_path / "config.yaml"
        assert saved_file.exists()

    def test_save_yaml_write_error(self, tmp_path: Path) -> None:
        """Test error handling when YAML write fails."""
        # Create a directory with the same name as the target file
        config_file = tmp_path / "config.yaml"
        config_file.mkdir()

        config = {"key": "value"}

        with pytest.raises(ConfigurationError, match="Failed to save"):
            save_config(config, config_file)

    def test_save_json_write_error(self, tmp_path: Path) -> None:
        """Test error handling when JSON write fails."""
        # Create a directory with the same name as the target file
        config_file = tmp_path / "config.json"
        config_file.mkdir()

        config = {"key": "value"}

        with pytest.raises(ConfigurationError, match="Failed to save"):
            save_config(config, config_file)

    @patch("oscura.core.config.loader.YAML_AVAILABLE", False)
    def test_save_yaml_not_available(self, tmp_path: Path) -> None:
        """Test error when PyYAML not available for YAML save."""
        config_file = tmp_path / "config.yaml"
        config = {"key": "value"}

        with pytest.raises(ConfigurationError, match="YAML support not available"):
            save_config(config, config_file)


class TestGetConfigValue:
    """Test get_config_value function."""

    def test_get_top_level_value(self) -> None:
        """Test getting top-level config value."""
        config = {"key": "value"}

        result = get_config_value(config, "key")

        assert result == "value"

    def test_get_nested_value(self) -> None:
        """Test getting nested config value."""
        config = {"level1": {"level2": {"level3": "deep_value"}}}

        result = get_config_value(config, "level1.level2.level3")

        assert result == "deep_value"

    def test_get_intermediate_dict(self) -> None:
        """Test getting intermediate dictionary."""
        config = {"level1": {"level2": {"key": "value"}}}

        result = get_config_value(config, "level1.level2")

        assert result == {"key": "value"}

    def test_missing_key_returns_default(self) -> None:
        """Test that missing key returns default value."""
        config = {"key": "value"}

        result = get_config_value(config, "missing", default="fallback")

        assert result == "fallback"

    def test_missing_nested_key_returns_default(self) -> None:
        """Test that missing nested key returns default."""
        config = {"level1": {"level2": "value"}}

        result = get_config_value(config, "level1.missing.key", default=None)

        assert result is None

    def test_default_none_when_not_specified(self) -> None:
        """Test that default is None when not specified."""
        config = {"key": "value"}

        result = get_config_value(config, "missing")

        assert result is None

    def test_path_through_non_dict_returns_default(self) -> None:
        """Test that path through non-dict value returns default."""
        config = {"key": "value"}

        # Try to traverse through string value
        result = get_config_value(config, "key.nested", default="fallback")

        assert result == "fallback"

    def test_empty_path(self) -> None:
        """Test with empty path string."""
        config = {"": "empty_key"}

        result = get_config_value(config, "")

        assert result == "empty_key"

    def test_single_dot_segments(self) -> None:
        """Test path with various dot-separated segments."""
        config = {"a": {"b": {"c": {"d": "value"}}}}

        result = get_config_value(config, "a.b.c.d")

        assert result == "value"

    def test_get_numeric_value(self) -> None:
        """Test getting numeric values."""
        config = {"defaults": {"sample_rate": 1e6}}

        result = get_config_value(config, "defaults.sample_rate")

        assert result == 1e6

    def test_get_list_value(self) -> None:
        """Test getting list values."""
        config = {"formats": ["wfm", "csv", "json"]}

        result = get_config_value(config, "formats")

        assert result == ["wfm", "csv", "json"]

    def test_get_boolean_value(self) -> None:
        """Test getting boolean values."""
        config = {"settings": {"enabled": True}}

        result = get_config_value(config, "settings.enabled")

        assert result is True

    def test_default_with_various_types(self) -> None:
        """Test default values of various types."""
        config = {"key": "value"}

        assert get_config_value(config, "missing", default=0) == 0
        assert get_config_value(config, "missing", default=[]) == []
        assert get_config_value(config, "missing", default={}) == {}
        assert get_config_value(config, "missing", default=False) is False


class TestConfigLoaderEdgeCases:
    """Test edge cases and error conditions."""

    def test_load_config_file_with_string_path(self, tmp_path: Path) -> None:
        """Test load_config_file accepts string paths."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value\n")

        # Pass as string, not Path
        config = load_config_file(str(config_file), validate=False)

        assert config["key"] == "value"

    def test_save_config_with_string_path(self, tmp_path: Path) -> None:
        """Test save_config accepts string paths."""
        config_file = tmp_path / "test.yaml"
        config = {"key": "value"}

        # Pass as string, not Path
        save_config(config, str(config_file))

        assert config_file.exists()

    def test_load_config_with_string_path(self, tmp_path: Path) -> None:
        """Test load_config accepts string paths."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: value\n")

        # Pass as string, not Path
        config = load_config(str(config_file), use_defaults=False)

        assert config["key"] == "value"

    def test_complex_nested_config(self, tmp_path: Path) -> None:
        """Test loading complex nested configuration."""
        config_file = tmp_path / "complex.yaml"
        yaml_content = """
version: "1.0"
defaults:
  sample_rate: 1000000
  settings:
    enabled: true
    options:
      - opt1
      - opt2
protocols:
  uart:
    baud_rate: 115200
    data_bits: 8
  spi:
    clock: 1000000
"""
        config_file.write_text(yaml_content)

        config = load_config_file(config_file, validate=False)

        assert config["version"] == "1.0"
        assert config["defaults"]["sample_rate"] == 1000000
        assert config["protocols"]["uart"]["baud_rate"] == 115200
        assert config["defaults"]["settings"]["options"] == ["opt1", "opt2"]

    def test_unicode_in_config(self, tmp_path: Path) -> None:
        """Test handling Unicode characters in config."""
        config_file = tmp_path / "unicode.yaml"
        config_file.write_text("name: Test \u2713\ndescription: Café\n", encoding="utf-8")

        config = load_config_file(config_file, validate=False)

        assert "Test \u2713" in config["name"]
        assert config["description"] == "Café"

    def test_json_with_unicode(self, tmp_path: Path) -> None:
        """Test JSON with Unicode characters."""
        config_file = tmp_path / "unicode.json"
        config = {"name": "Test ✓", "description": "Café"}

        save_config(config, config_file)
        loaded = load_config_file(config_file, validate=False)

        assert loaded == config

    def test_yaml_with_special_characters(self, tmp_path: Path) -> None:
        """Test YAML with special characters requiring quoting."""
        config_file = tmp_path / "special.yaml"
        config_file.write_text('key: "value: with: colons"\n')

        config = load_config_file(config_file, validate=False)

        assert config["key"] == "value: with: colons"


class TestConfigLoaderIntegration:
    """Integration tests combining multiple functions."""

    def test_save_and_load_roundtrip_yaml(self, tmp_path: Path) -> None:
        """Test saving and loading config maintains data (YAML)."""
        config_file = tmp_path / "roundtrip.yaml"
        original = {
            "version": "1.0",
            "settings": {
                "enabled": True,
                "value": 42,
                "items": ["a", "b", "c"],
            },
        }

        save_config(original, config_file)
        loaded = load_config_file(config_file, validate=False)

        assert loaded == original

    def test_save_and_load_roundtrip_json(self, tmp_path: Path) -> None:
        """Test saving and loading config maintains data (JSON)."""
        config_file = tmp_path / "roundtrip.json"
        original = {
            "version": "1.0",
            "settings": {
                "enabled": True,
                "value": 42,
                "items": ["a", "b", "c"],
            },
        }

        save_config(original, config_file)
        loaded = load_config_file(config_file, validate=False)

        assert loaded == original

    def test_load_merge_save_workflow(self, tmp_path: Path) -> None:
        """Test realistic workflow: load, merge, save."""
        # Start with base config
        base_file = tmp_path / "base.yaml"
        base_file.write_text("version: '1.0'\nbase_key: base_value\n")

        # Load it
        config = load_config_file(base_file, validate=False)

        # Merge in new settings
        config["new_key"] = "new_value"
        config["version"] = "2.0"

        # Save to new file
        output_file = tmp_path / "updated.yaml"
        save_config(config, output_file)

        # Verify
        loaded = load_config_file(output_file, validate=False)
        assert loaded["version"] == "2.0"
        assert loaded["base_key"] == "base_value"
        assert loaded["new_key"] == "new_value"

    def test_get_config_value_on_loaded_config(self, tmp_path: Path) -> None:
        """Test get_config_value on a loaded config file."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("defaults:\n  sample_rate: 1000000\n  window: hann\n")

        config = load_config_file(config_file, validate=False)
        sample_rate = get_config_value(config, "defaults.sample_rate")

        assert sample_rate == 1000000
