"""Comprehensive unit tests for src/oscura/core/config.py

Tests coverage for:

Covers all public classes, functions, properties, and methods with
edge cases, error handling, and validation.
"""

from pathlib import Path

import pytest

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from oscura.core.config import (
    DEFAULT_CONFIG,
    SmartDefaults,
    _deep_merge,
    get_config_value,
    load_config,
    save_config,
)
from oscura.core.config.legacy import validate_config
from oscura.core.exceptions import ConfigurationError

pytestmark = [
    pytest.mark.unit,
    pytest.mark.core,
    pytest.mark.usefixtures("reset_logging_state"),
]


# ==============================================================================
# _deep_merge() Tests
# ==============================================================================


class TestDeepMerge:
    """Test _deep_merge() function."""

    def test_merge_empty_dicts(self) -> None:
        """Test merging empty dictionaries."""
        result = _deep_merge({}, {})
        assert result == {}

    def test_merge_simple_override(self) -> None:
        """Test simple override."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested_dicts(self) -> None:
        """Test merging nested dictionaries."""
        base = {"section": {"key1": 1, "key2": 2}}
        override = {"section": {"key2": 20, "key3": 30}}
        result = _deep_merge(base, override)
        assert result == {"section": {"key1": 1, "key2": 20, "key3": 30}}

    def test_merge_deeply_nested(self) -> None:
        """Test merging deeply nested dictionaries."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"d": 20, "e": 30}}}
        result = _deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 20, "e": 30}}}

    def test_merge_non_dict_override(self) -> None:
        """Test overriding dict with non-dict."""
        base = {"key": {"nested": "value"}}
        override = {"key": "string"}
        result = _deep_merge(base, override)
        assert result == {"key": "string"}

    def test_merge_preserves_base(self) -> None:
        """Test that original base is not modified."""
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert base == {"a": 1}  # Unchanged
        assert result == {"a": 1, "b": 2}


# ==============================================================================
# load_config() Tests
# ==============================================================================


class TestLoadConfig:
    """Test load_config() function."""

    def test_load_defaults_only(self) -> None:
        """Test loading with no file, only defaults."""
        # When config_path=None and use_defaults=True, should return defaults
        config = load_config(config_path=None, use_defaults=True)
        assert "defaults" in config
        assert "loaders" in config
        assert config["defaults"]["sample_rate"] == 1e6

    def test_load_no_defaults(self) -> None:
        """Test loading with defaults disabled."""
        config = load_config(config_path=None, use_defaults=False)
        assert config == {}

    def test_load_yaml_file(self, tmp_path: Path) -> None:
        """Test loading from YAML file."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            f.write("custom_key: custom_value\n")
            f.write("defaults:\n")
            f.write("  sample_rate: 2000000\n")

        config = load_config(config_path=config_file, use_defaults=False)
        assert config["custom_key"] == "custom_value"
        assert config["defaults"]["sample_rate"] == 2000000

    def test_load_yaml_with_defaults(self, tmp_path: Path) -> None:
        """Test loading YAML and merging with defaults."""
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            f.write("defaults:\n")
            f.write("  sample_rate: 5000000\n")

        config = load_config(config_path=config_file, use_defaults=True)
        # Should have merged defaults
        assert config["defaults"]["sample_rate"] == 5000000
        assert "window_function" in config["defaults"]  # From DEFAULT_CONFIG

    def test_load_nonexistent_file_raises_error(self) -> None:
        """Test that loading nonexistent file raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config(config_path="/nonexistent/path/config.yaml")

    def test_load_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        """Test that invalid YAML raises ConfigurationError."""
        config_file = tmp_path / "invalid.yaml"
        with open(config_file, "w") as f:
            f.write("invalid: yaml: : syntax\n")

        with pytest.raises(ConfigurationError, match="Failed to parse"):
            load_config(config_path=config_file)

    def test_load_expands_user_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that ~ is expanded in path."""
        # Create config in tmp_path
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            f.write("test: value\n")

        # Mock expanduser to return our tmp path
        def mock_expanduser(self: Path) -> Path:
            if str(self).startswith("~"):
                return config_file
            return self

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        config = load_config(config_path="~/config.yaml", use_defaults=False)
        assert config["test"] == "value"


# ==============================================================================
# validate_config() Tests
# ==============================================================================


class TestValidateConfig:
    """Test validate_config() function."""

    def test_validate_default_config(self) -> None:
        """Test that DEFAULT_CONFIG is valid."""
        assert validate_config(DEFAULT_CONFIG) is True

    def test_validate_minimal_config(self) -> None:
        """Test validating minimal config."""
        config = {"defaults": {}, "loaders": {}}
        assert validate_config(config) is True

    def test_validate_missing_required_section(self) -> None:
        """Test that missing required section raises error."""
        config = {"defaults": {}}  # Missing 'loaders'
        with pytest.raises(ConfigurationError, match="Missing required configuration section"):
            validate_config(config)

    def test_validate_invalid_sample_rate_type(self) -> None:
        """Test that invalid sample_rate type raises error."""
        config = {"defaults": {"sample_rate": "not_a_number"}, "loaders": {}}
        with pytest.raises(ConfigurationError, match="Invalid sample_rate"):
            validate_config(config)

    def test_validate_negative_sample_rate(self) -> None:
        """Test that negative sample_rate raises error."""
        config = {"defaults": {"sample_rate": -1000}, "loaders": {}}
        with pytest.raises(ConfigurationError, match="Invalid sample_rate"):
            validate_config(config)

    def test_validate_zero_sample_rate(self) -> None:
        """Test that zero sample_rate raises error."""
        config = {"defaults": {"sample_rate": 0}, "loaders": {}}
        with pytest.raises(ConfigurationError, match="Invalid sample_rate"):
            validate_config(config)

    def test_validate_invalid_formats_type(self) -> None:
        """Test that non-list formats raises error."""
        config = {"defaults": {}, "loaders": {"formats": "not_a_list"}}
        with pytest.raises(ConfigurationError, match="Invalid formats"):
            validate_config(config)

    def test_validate_invalid_ref_levels(self) -> None:
        """Test that invalid ref_levels raises error."""
        config = {
            "defaults": {},
            "loaders": {},
            "measurements": {"rise_time": {"ref_levels": [0.1]}},  # Only 1 value
        }
        with pytest.raises(ConfigurationError, match="Invalid ref_levels"):
            validate_config(config)

    def test_validate_ref_levels_not_list(self) -> None:
        """Test that non-list ref_levels raises error."""
        config = {
            "defaults": {},
            "loaders": {},
            "measurements": {"rise_time": {"ref_levels": "0.1,0.9"}},
        }
        with pytest.raises(ConfigurationError, match="Invalid ref_levels"):
            validate_config(config)


# ==============================================================================
# get_config_value() Tests
# ==============================================================================


class TestGetConfigValue:
    """Test get_config_value() function."""

    def test_get_top_level_key(self) -> None:
        """Test getting top-level key."""
        config = {"key": "value"}
        result = get_config_value(config, "key")
        assert result == "value"

    def test_get_nested_key(self) -> None:
        """Test getting nested key with dot notation."""
        config = {"section": {"subsection": {"key": "value"}}}
        result = get_config_value(config, "section.subsection.key")
        assert result == "value"

    def test_get_default_when_missing(self) -> None:
        """Test that default is returned when key missing."""
        config = {"key": "value"}
        result = get_config_value(config, "nonexistent", default="default_value")
        assert result == "default_value"

    def test_get_default_nested_missing(self) -> None:
        """Test default for missing nested key."""
        config = {"section": {"key": "value"}}
        result = get_config_value(config, "section.nonexistent.key", default=999)
        assert result == 999

    def test_get_none_default(self) -> None:
        """Test None as default value."""
        config = {}
        result = get_config_value(config, "missing")
        assert result is None

    def test_get_from_default_config(self) -> None:
        """Test getting from DEFAULT_CONFIG."""
        result = get_config_value(DEFAULT_CONFIG, "defaults.sample_rate")
        assert result == 1e6


# ==============================================================================
# save_config() Tests
# ==============================================================================


class TestSaveConfig:
    """Test save_config() function."""

    def test_save_simple_config(self, tmp_path: Path) -> None:
        """Test saving simple config."""
        config = {"test_key": "test_value", "number": 42}
        output_file = tmp_path / "output.yaml"

        save_config(config, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded["test_key"] == "test_value"
        assert loaded["number"] == 42

    def test_save_nested_config(self, tmp_path: Path) -> None:
        """Test saving nested config."""
        config = {"section": {"nested": {"key": "value"}}}
        output_file = tmp_path / "nested.yaml"

        save_config(config, output_file)

        assert output_file.exists()
        with open(output_file) as f:
            loaded = yaml.safe_load(f)
        assert loaded["section"]["nested"]["key"] == "value"

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that save creates parent directories."""
        config = {"test": "value"}
        output_file = tmp_path / "subdir" / "config.yaml"

        save_config(config, output_file)

        assert output_file.exists()

    def test_save_round_trip(self, tmp_path: Path) -> None:
        """Test that save -> load preserves config."""
        config = DEFAULT_CONFIG.copy()
        config["custom"] = {"key": 123}
        output_file = tmp_path / "roundtrip.yaml"

        save_config(config, output_file)
        loaded = load_config(config_path=output_file, use_defaults=False)

        assert loaded["custom"]["key"] == 123


# ==============================================================================
# SmartDefaults Tests
# ==============================================================================


class TestSmartDefaults:
    """Test SmartDefaults class."""

    def test_create_default(self) -> None:
        """Test creating SmartDefaults."""
        defaults = SmartDefaults()
        assert defaults.verbose is False
        assert defaults._log_buffer == []

    def test_create_verbose(self) -> None:
        """Test creating verbose SmartDefaults."""
        defaults = SmartDefaults(verbose=True)
        assert defaults.verbose is True

    def test_get_fft_size_small_signal(self) -> None:
        """Test FFT size for small signal."""
        defaults = SmartDefaults()
        size = defaults.get_fft_size(signal_length=100)
        # Next power of 2 >= 100 is 128, but clamped to min 256
        assert size == 256

    def test_get_fft_size_exact_power_of_2(self) -> None:
        """Test FFT size when signal is exact power of 2."""
        defaults = SmartDefaults()
        size = defaults.get_fft_size(signal_length=1024)
        assert size == 1024

    def test_get_fft_size_large_signal(self) -> None:
        """Test FFT size for large signal."""
        defaults = SmartDefaults()
        size = defaults.get_fft_size(signal_length=10000)
        # Next power of 2 >= 10000 is 16384
        assert size == 16384

    def test_get_fft_size_clamped_to_max(self) -> None:
        """Test FFT size clamped to maximum."""
        defaults = SmartDefaults()
        size = defaults.get_fft_size(signal_length=1000000, max_size=8192)
        assert size == 8192

    def test_get_window_function_general(self) -> None:
        """Test window function for general application."""
        defaults = SmartDefaults()
        window = defaults.get_window_function(application="general")
        assert window == "hann"

    def test_get_window_function_transient(self) -> None:
        """Test window function for transient analysis."""
        defaults = SmartDefaults()
        window = defaults.get_window_function(application="transient")
        assert window == "boxcar"

    def test_get_window_function_narrowband(self) -> None:
        """Test window function for narrowband analysis."""
        defaults = SmartDefaults()
        window = defaults.get_window_function(application="narrowband", dynamic_range_db=50)
        assert window == "flattop"

    def test_get_window_function_high_dynamic_range(self) -> None:
        """Test window for high dynamic range."""
        defaults = SmartDefaults()
        window = defaults.get_window_function(application="general", dynamic_range_db=85)
        assert window == "blackman-harris"

    def test_get_window_function_moderate_dynamic_range(self) -> None:
        """Test window for moderate dynamic range."""
        defaults = SmartDefaults()
        window = defaults.get_window_function(application="general", dynamic_range_db=70)
        assert window == "blackman"

    def test_get_overlap_welch_hann(self) -> None:
        """Test overlap for Welch method with Hann window."""
        defaults = SmartDefaults()
        overlap = defaults.get_overlap(method="welch", window="hann")
        assert overlap == 0.5

    def test_get_overlap_bartlett(self) -> None:
        """Test overlap for Bartlett method."""
        defaults = SmartDefaults()
        overlap = defaults.get_overlap(method="bartlett")
        assert overlap == 0.0

    def test_get_overlap_blackman_harris(self) -> None:
        """Test overlap for Blackman-Harris window."""
        defaults = SmartDefaults()
        overlap = defaults.get_overlap(method="welch", window="blackman-harris")
        assert overlap == 0.75

    def test_get_reference_levels_rise_time(self) -> None:
        """Test reference levels for rise time."""
        defaults = SmartDefaults()
        levels = defaults.get_reference_levels(measurement="rise_time")
        assert levels == (0.1, 0.9)

    def test_get_reference_levels_fall_time(self) -> None:
        """Test reference levels for fall time."""
        defaults = SmartDefaults()
        levels = defaults.get_reference_levels(measurement="fall_time")
        assert levels == (0.9, 0.1)

    def test_get_reference_levels_timing(self) -> None:
        """Test reference levels for timing measurements."""
        defaults = SmartDefaults()
        levels = defaults.get_reference_levels(measurement="propagation_delay")
        assert levels == (0.5, 0.5)

    def test_get_log_messages(self) -> None:
        """Test getting log messages."""
        defaults = SmartDefaults()
        defaults.get_fft_size(1000)
        defaults.get_window_function("general")

        messages = defaults.get_log_messages()
        assert len(messages) == 2
        assert "FFT size" in messages[0]
        assert "Window function" in messages[1]

    def test_clear_log(self) -> None:
        """Test clearing log buffer."""
        defaults = SmartDefaults()
        defaults.get_fft_size(1000)
        assert len(defaults._log_buffer) > 0

        defaults.clear_log()
        assert len(defaults._log_buffer) == 0

    def test_verbose_mode_logs_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test that verbose mode prints to stdout."""
        defaults = SmartDefaults(verbose=True)
        defaults.get_fft_size(1000)

        captured = capsys.readouterr()
        assert "[SmartDefaults]" in captured.out
        assert "FFT size" in captured.out


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestCoreConfigIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: load, modify, validate, save."""
        # Load defaults
        config = load_config(use_defaults=True)

        # Modify
        config["defaults"]["sample_rate"] = 10e6

        # Validate
        assert validate_config(config) is True

        # Save
        output_file = tmp_path / "modified_config.yaml"
        save_config(config, output_file)

        # Load again
        loaded = load_config(config_path=output_file, use_defaults=False)

        # Verify
        assert loaded["defaults"]["sample_rate"] == 10e6

    def test_smart_defaults_workflow(self) -> None:
        """Test SmartDefaults workflow."""
        defaults = SmartDefaults(verbose=False)

        # Get various parameters
        fft_size = defaults.get_fft_size(5000)
        window = defaults.get_window_function("general")
        overlap = defaults.get_overlap("welch", window)
        ref_levels = defaults.get_reference_levels("rise_time")

        # Verify reasonable values
        assert fft_size >= 5000
        assert window in ["hann", "hamming", "blackman", "blackman-harris", "flattop", "boxcar"]
        assert 0 <= overlap <= 1
        assert len(ref_levels) == 2

        # Check logging
        messages = defaults.get_log_messages()
        assert len(messages) == 4
