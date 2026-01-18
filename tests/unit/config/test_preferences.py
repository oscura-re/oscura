"""Comprehensive test suite for user preferences management.

Tests CFG-018: Preferences Persistence
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from oscura.config.preferences import (
    DefaultsPreferences,
    EditorPreferences,
    ExportPreferences,
    LoggingPreferences,
    PreferencesManager,
    UserPreferences,
    VisualizationPreferences,
    get_preferences,
    get_preferences_manager,
    save_preferences,
)
from oscura.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


# =============================================================================
# VisualizationPreferences Tests
# =============================================================================


class TestVisualizationPreferences:
    """Test VisualizationPreferences dataclass."""

    def test_default_visualization_preferences(self) -> None:
        """Test default visualization preferences."""
        prefs = VisualizationPreferences()

        assert prefs.style == "seaborn-v0_8-whitegrid"
        assert prefs.figure_size == (10, 6)
        assert prefs.dpi == 100
        assert prefs.colormap == "viridis"
        assert prefs.grid is True
        assert prefs.dark_mode is False

    def test_custom_visualization_preferences(self) -> None:
        """Test custom visualization preferences."""
        prefs = VisualizationPreferences(
            style="dark_background",
            figure_size=(12, 8),
            dpi=150,
            colormap="plasma",
            grid=False,
            dark_mode=True,
        )

        assert prefs.style == "dark_background"
        assert prefs.figure_size == (12, 8)
        assert prefs.dpi == 150
        assert prefs.colormap == "plasma"
        assert prefs.grid is False
        assert prefs.dark_mode is True


# =============================================================================
# DefaultsPreferences Tests
# =============================================================================


class TestDefaultsPreferences:
    """Test DefaultsPreferences dataclass."""

    def test_default_analysis_preferences(self) -> None:
        """Test default analysis preferences."""
        prefs = DefaultsPreferences()

        assert prefs.sample_rate == 1e9
        assert prefs.window_function == "hann"
        assert prefs.fft_size == 8192
        assert prefs.rise_time_thresholds == (10.0, 90.0)
        assert prefs.logic_family == "TTL"

    def test_custom_analysis_preferences(self) -> None:
        """Test custom analysis preferences."""
        prefs = DefaultsPreferences(
            sample_rate=2e9,
            window_function="hamming",
            fft_size=16384,
            rise_time_thresholds=(20.0, 80.0),
            logic_family="CMOS",
        )

        assert prefs.sample_rate == 2e9
        assert prefs.window_function == "hamming"
        assert prefs.fft_size == 16384
        assert prefs.rise_time_thresholds == (20.0, 80.0)
        assert prefs.logic_family == "CMOS"


# =============================================================================
# ExportPreferences Tests
# =============================================================================


class TestExportPreferences:
    """Test ExportPreferences dataclass."""

    def test_default_export_preferences(self) -> None:
        """Test default export preferences."""
        prefs = ExportPreferences()

        assert prefs.default_format == "csv"
        assert prefs.precision == 6
        assert prefs.include_metadata is True
        assert prefs.compression == "gzip"

    def test_custom_export_preferences(self) -> None:
        """Test custom export preferences."""
        prefs = ExportPreferences(
            default_format="hdf5",
            precision=8,
            include_metadata=False,
            compression="lzf",
        )

        assert prefs.default_format == "hdf5"
        assert prefs.precision == 8
        assert prefs.include_metadata is False
        assert prefs.compression == "lzf"


# =============================================================================
# LoggingPreferences Tests
# =============================================================================


class TestLoggingPreferences:
    """Test LoggingPreferences dataclass."""

    def test_default_logging_preferences(self) -> None:
        """Test default logging preferences."""
        prefs = LoggingPreferences()

        assert prefs.level == "WARNING"
        assert prefs.file is None
        assert "%(asctime)s" in prefs.format
        assert prefs.show_timestamps is False

    def test_custom_logging_preferences(self) -> None:
        """Test custom logging preferences."""
        prefs = LoggingPreferences(
            level="DEBUG",
            file="/var/log/oscura.log",
            format="%(levelname)s: %(message)s",
            show_timestamps=True,
        )

        assert prefs.level == "DEBUG"
        assert prefs.file == "/var/log/oscura.log"
        assert prefs.format == "%(levelname)s: %(message)s"
        assert prefs.show_timestamps is True


# =============================================================================
# EditorPreferences Tests
# =============================================================================


class TestEditorPreferences:
    """Test EditorPreferences dataclass."""

    def test_default_editor_preferences(self) -> None:
        """Test default editor preferences."""
        prefs = EditorPreferences()

        assert prefs.history_size == 1000
        assert prefs.auto_save is True
        assert prefs.syntax_highlighting is True
        assert prefs.tab_completion is True

    def test_custom_editor_preferences(self) -> None:
        """Test custom editor preferences."""
        prefs = EditorPreferences(
            history_size=500,
            auto_save=False,
            syntax_highlighting=False,
            tab_completion=False,
        )

        assert prefs.history_size == 500
        assert prefs.auto_save is False
        assert prefs.syntax_highlighting is False
        assert prefs.tab_completion is False


# =============================================================================
# UserPreferences Tests
# =============================================================================


class TestUserPreferences:
    """Test UserPreferences dataclass."""

    def test_default_user_preferences(self) -> None:
        """Test default user preferences."""
        prefs = UserPreferences()

        assert isinstance(prefs.visualization, VisualizationPreferences)
        assert isinstance(prefs.defaults, DefaultsPreferences)
        assert isinstance(prefs.export, ExportPreferences)
        assert isinstance(prefs.logging, LoggingPreferences)
        assert isinstance(prefs.editor, EditorPreferences)
        assert prefs.recent_files == []
        assert prefs.custom == {}

    def test_user_preferences_with_custom_sections(self) -> None:
        """Test user preferences with custom section values."""
        viz = VisualizationPreferences(dark_mode=True)
        defaults = DefaultsPreferences(sample_rate=2e9)
        export = ExportPreferences(default_format="json")
        logging = LoggingPreferences(level="DEBUG")
        editor = EditorPreferences(history_size=2000)

        prefs = UserPreferences(
            visualization=viz,
            defaults=defaults,
            export=export,
            logging=logging,
            editor=editor,
        )

        assert prefs.visualization.dark_mode is True
        assert prefs.defaults.sample_rate == 2e9
        assert prefs.export.default_format == "json"
        assert prefs.logging.level == "DEBUG"
        assert prefs.editor.history_size == 2000

    def test_user_preferences_with_recent_files(self) -> None:
        """Test user preferences with recent files."""
        prefs = UserPreferences(recent_files=["/path/to/file1.bin", "/path/to/file2.bin"])

        assert len(prefs.recent_files) == 2
        assert "/path/to/file1.bin" in prefs.recent_files

    def test_user_preferences_with_custom_data(self) -> None:
        """Test user preferences with custom data."""
        prefs = UserPreferences(custom={"my_setting": 42, "nested": {"key": "value"}})

        assert prefs.custom["my_setting"] == 42
        assert prefs.custom["nested"]["key"] == "value"


# =============================================================================
# UserPreferences Get/Set Tests
# =============================================================================


class TestUserPreferencesGetSet:
    """Test UserPreferences get/set methods."""

    def test_get_visualization_setting(self) -> None:
        """Test getting visualization setting."""
        prefs = UserPreferences()

        dpi = prefs.get("visualization.dpi")

        assert dpi == 100

    def test_get_defaults_setting(self) -> None:
        """Test getting defaults setting."""
        prefs = UserPreferences()

        sample_rate = prefs.get("defaults.sample_rate")

        assert sample_rate == 1e9

    def test_get_export_setting(self) -> None:
        """Test getting export setting."""
        prefs = UserPreferences()

        precision = prefs.get("export.precision")

        assert precision == 6

    def test_get_logging_setting(self) -> None:
        """Test getting logging setting."""
        prefs = UserPreferences()

        level = prefs.get("logging.level")

        assert level == "WARNING"

    def test_get_editor_setting(self) -> None:
        """Test getting editor setting."""
        prefs = UserPreferences()

        history_size = prefs.get("editor.history_size")

        assert history_size == 1000

    def test_get_custom_setting(self) -> None:
        """Test getting custom setting."""
        prefs = UserPreferences(custom={"my_key": "my_value"})

        value = prefs.get("custom.my_key")

        assert value == "my_value"

    def test_get_nested_custom_setting(self) -> None:
        """Test getting nested custom setting."""
        prefs = UserPreferences(custom={"level1": {"level2": {"level3": "deep"}}})

        value = prefs.get("custom.level1.level2.level3")

        assert value == "deep"

    def test_get_missing_setting_returns_default(self) -> None:
        """Test getting missing setting returns default."""
        prefs = UserPreferences()

        value = prefs.get("missing.setting", default="fallback")

        assert value == "fallback"

    def test_get_missing_setting_returns_none(self) -> None:
        """Test getting missing setting returns None."""
        prefs = UserPreferences()

        value = prefs.get("missing.setting")

        assert value is None

    def test_set_visualization_setting(self) -> None:
        """Test setting visualization setting."""
        prefs = UserPreferences()

        prefs.set("visualization.dpi", 150)

        assert prefs.visualization.dpi == 150

    def test_set_defaults_setting(self) -> None:
        """Test setting defaults setting."""
        prefs = UserPreferences()

        prefs.set("defaults.sample_rate", 2e9)

        assert prefs.defaults.sample_rate == 2e9

    def test_set_export_setting(self) -> None:
        """Test setting export setting."""
        prefs = UserPreferences()

        prefs.set("export.default_format", "hdf5")

        assert prefs.export.default_format == "hdf5"

    def test_set_logging_setting(self) -> None:
        """Test setting logging setting."""
        prefs = UserPreferences()

        prefs.set("logging.level", "DEBUG")

        assert prefs.logging.level == "DEBUG"

    def test_set_editor_setting(self) -> None:
        """Test setting editor setting."""
        prefs = UserPreferences()

        prefs.set("editor.history_size", 2000)

        assert prefs.editor.history_size == 2000

    def test_set_custom_setting(self) -> None:
        """Test setting custom setting."""
        prefs = UserPreferences()

        prefs.set("custom.my_key", "my_value")

        assert prefs.custom["my_key"] == "my_value"

    def test_set_nested_custom_setting(self) -> None:
        """Test setting nested custom setting."""
        prefs = UserPreferences()

        prefs.set("custom.level1.level2", "value")

        assert prefs.custom["level1"]["level2"] == "value"

    def test_set_invalid_path_raises_error(self) -> None:
        """Test setting invalid path raises KeyError."""
        prefs = UserPreferences()

        with pytest.raises(KeyError, match="Invalid preference path"):
            prefs.set("nonexistent.section.key", "value")

    def test_set_unknown_attribute_raises_error(self) -> None:
        """Test setting unknown attribute raises KeyError."""
        prefs = UserPreferences()

        with pytest.raises(KeyError, match="Unknown preference"):
            prefs.set("visualization.nonexistent", "value")


# =============================================================================
# UserPreferences Serialization Tests
# =============================================================================


class TestUserPreferencesSerialization:
    """Test UserPreferences to_dict/from_dict methods."""

    def test_to_dict_structure(self) -> None:
        """Test to_dict produces correct structure."""
        prefs = UserPreferences()

        data = prefs.to_dict()

        assert "visualization" in data
        assert "defaults" in data
        assert "export" in data
        assert "logging" in data
        assert "editor" in data
        assert "recent_files" in data
        assert "custom" in data

    def test_to_dict_values(self) -> None:
        """Test to_dict preserves values."""
        prefs = UserPreferences()
        prefs.visualization.dpi = 150
        prefs.defaults.sample_rate = 2e9
        prefs.custom["key"] = "value"

        data = prefs.to_dict()

        assert data["visualization"]["dpi"] == 150
        assert data["defaults"]["sample_rate"] == 2e9
        assert data["custom"]["key"] == "value"

    def test_to_dict_list_conversion(self) -> None:
        """Test to_dict converts tuples to lists."""
        prefs = UserPreferences()

        data = prefs.to_dict()

        # figure_size is a tuple, should be converted to list
        assert isinstance(data["visualization"]["figure_size"], list)

    def test_from_dict_empty(self) -> None:
        """Test from_dict with empty dict uses defaults."""
        prefs = UserPreferences.from_dict({})

        assert prefs.visualization.dpi == 100
        assert prefs.defaults.sample_rate == 1e9
        assert prefs.recent_files == []

    def test_from_dict_visualization(self) -> None:
        """Test from_dict loads visualization preferences."""
        data = {
            "visualization": {
                "style": "dark_background",
                "figure_size": [12, 8],
                "dpi": 150,
                "colormap": "plasma",
                "grid": False,
                "dark_mode": True,
            }
        }

        prefs = UserPreferences.from_dict(data)

        assert prefs.visualization.style == "dark_background"
        assert prefs.visualization.figure_size == (12, 8)
        assert prefs.visualization.dpi == 150
        assert prefs.visualization.colormap == "plasma"
        assert prefs.visualization.grid is False
        assert prefs.visualization.dark_mode is True

    def test_from_dict_defaults(self) -> None:
        """Test from_dict loads defaults preferences."""
        data = {
            "defaults": {
                "sample_rate": 2e9,
                "window_function": "hamming",
                "fft_size": 16384,
                "rise_time_thresholds": [20.0, 80.0],
                "logic_family": "CMOS",
            }
        }

        prefs = UserPreferences.from_dict(data)

        assert prefs.defaults.sample_rate == 2e9
        assert prefs.defaults.window_function == "hamming"
        assert prefs.defaults.fft_size == 16384
        assert prefs.defaults.rise_time_thresholds == (20.0, 80.0)
        assert prefs.defaults.logic_family == "CMOS"

    def test_from_dict_export(self) -> None:
        """Test from_dict loads export preferences."""
        data = {
            "export": {
                "default_format": "hdf5",
                "precision": 8,
                "include_metadata": False,
                "compression": "lzf",
            }
        }

        prefs = UserPreferences.from_dict(data)

        assert prefs.export.default_format == "hdf5"
        assert prefs.export.precision == 8
        assert prefs.export.include_metadata is False
        assert prefs.export.compression == "lzf"

    def test_from_dict_logging(self) -> None:
        """Test from_dict loads logging preferences."""
        data = {
            "logging": {
                "level": "DEBUG",
                "file": "/var/log/oscura.log",
                "format": "%(message)s",
                "show_timestamps": True,
            }
        }

        prefs = UserPreferences.from_dict(data)

        assert prefs.logging.level == "DEBUG"
        assert prefs.logging.file == "/var/log/oscura.log"
        assert prefs.logging.format == "%(message)s"
        assert prefs.logging.show_timestamps is True

    def test_from_dict_editor(self) -> None:
        """Test from_dict loads editor preferences."""
        data = {
            "editor": {
                "history_size": 2000,
                "auto_save": False,
                "syntax_highlighting": False,
                "tab_completion": False,
            }
        }

        prefs = UserPreferences.from_dict(data)

        assert prefs.editor.history_size == 2000
        assert prefs.editor.auto_save is False
        assert prefs.editor.syntax_highlighting is False
        assert prefs.editor.tab_completion is False

    def test_from_dict_recent_files(self) -> None:
        """Test from_dict loads recent files."""
        data = {"recent_files": ["/path/file1.bin", "/path/file2.bin"]}

        prefs = UserPreferences.from_dict(data)

        assert len(prefs.recent_files) == 2
        assert "/path/file1.bin" in prefs.recent_files

    def test_from_dict_custom(self) -> None:
        """Test from_dict loads custom settings."""
        data = {"custom": {"key1": "value1", "nested": {"key2": "value2"}}}

        prefs = UserPreferences.from_dict(data)

        assert prefs.custom["key1"] == "value1"
        assert prefs.custom["nested"]["key2"] == "value2"

    def test_from_dict_partial_data(self) -> None:
        """Test from_dict with partial data fills defaults."""
        data = {
            "visualization": {"dpi": 150},
            "custom": {"key": "value"},
        }

        prefs = UserPreferences.from_dict(data)

        # Provided values
        assert prefs.visualization.dpi == 150
        assert prefs.custom["key"] == "value"
        # Defaults for missing values
        assert prefs.visualization.colormap == "viridis"
        assert prefs.defaults.sample_rate == 1e9

    def test_roundtrip_serialization(self) -> None:
        """Test roundtrip to_dict -> from_dict."""
        original = UserPreferences()
        original.visualization.dpi = 150
        original.defaults.sample_rate = 2e9
        original.recent_files = ["/file1.bin"]
        original.custom = {"key": "value"}

        data = original.to_dict()
        restored = UserPreferences.from_dict(data)

        assert restored.visualization.dpi == 150
        assert restored.defaults.sample_rate == 2e9
        assert restored.recent_files == ["/file1.bin"]
        assert restored.custom["key"] == "value"


# =============================================================================
# PreferencesManager Tests
# =============================================================================


class TestPreferencesManager:
    """Test PreferencesManager class."""

    def test_manager_initialization_default_path(self) -> None:
        """Test manager initialization with default path."""
        manager = PreferencesManager()

        assert manager.path is not None
        assert "oscura" in str(manager.path)

    def test_manager_initialization_custom_path(self, tmp_path: Path) -> None:
        """Test manager initialization with custom path."""
        custom_path = tmp_path / "custom_prefs.yaml"
        manager = PreferencesManager(path=custom_path)

        assert manager.path == custom_path

    def test_load_creates_defaults_if_no_file(self, tmp_path: Path) -> None:
        """Test load creates default preferences if file doesn't exist."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        prefs = manager.load()

        assert prefs.visualization.dpi == 100
        assert prefs.defaults.sample_rate == 1e9

    def test_load_from_existing_file(self, tmp_path: Path) -> None:
        """Test loading from existing file."""
        prefs_file = tmp_path / "prefs.yaml"
        data = {
            "visualization": {"dpi": 150},
            "defaults": {"sample_rate": 2e9},
        }
        with open(prefs_file, "w") as f:
            yaml.dump(data, f)

        manager = PreferencesManager(path=prefs_file)
        prefs = manager.load()

        assert prefs.visualization.dpi == 150
        assert prefs.defaults.sample_rate == 2e9

    def test_load_caching(self, tmp_path: Path) -> None:
        """Test preferences caching."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        prefs1 = manager.load(use_cache=True)
        prefs2 = manager.load(use_cache=True)

        assert prefs1 is prefs2

    def test_load_bypass_cache(self, tmp_path: Path) -> None:
        """Test bypassing cache."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        prefs1 = manager.load(use_cache=True)
        prefs2 = manager.load(use_cache=False)

        # Should be different instances
        assert prefs1 is not prefs2

    def test_load_handles_invalid_yaml(self, tmp_path: Path) -> None:
        """Test load handles invalid YAML gracefully."""
        prefs_file = tmp_path / "invalid.yaml"
        prefs_file.write_text("invalid: [yaml\n")

        manager = PreferencesManager(path=prefs_file)
        prefs = manager.load()

        # Should return defaults on error
        assert prefs.visualization.dpi == 100

    def test_load_handles_empty_file(self, tmp_path: Path) -> None:
        """Test load handles empty file."""
        prefs_file = tmp_path / "empty.yaml"
        prefs_file.write_text("")

        manager = PreferencesManager(path=prefs_file)
        prefs = manager.load()

        assert prefs.visualization.dpi == 100

    def test_save_creates_file(self, tmp_path: Path) -> None:
        """Test save creates preferences file."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        prefs = UserPreferences()
        prefs.visualization.dpi = 150
        manager.save(prefs)

        assert prefs_file.exists()

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        """Test save and load roundtrip."""
        prefs_file = tmp_path / "roundtrip.yaml"
        manager = PreferencesManager(path=prefs_file)

        original = UserPreferences()
        original.visualization.dpi = 150
        original.defaults.sample_rate = 2e9
        manager.save(original)

        loaded = manager.load(use_cache=False)

        assert loaded.visualization.dpi == 150
        assert loaded.defaults.sample_rate == 2e9

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test save creates parent directories."""
        prefs_file = tmp_path / "nested" / "dir" / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        prefs = UserPreferences()
        manager.save(prefs)

        assert prefs_file.exists()

    def test_save_error_handling(self, tmp_path: Path) -> None:
        """Test save error handling."""
        # Create a directory with same name as file
        prefs_file = tmp_path / "prefs.yaml"
        prefs_file.mkdir()

        manager = PreferencesManager(path=prefs_file)
        prefs = UserPreferences()

        with pytest.raises(ConfigurationError):
            manager.save(prefs)

    def test_reset_to_defaults(self, tmp_path: Path) -> None:
        """Test reset to default preferences."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        # Create custom preferences
        prefs = UserPreferences()
        prefs.visualization.dpi = 150
        manager.save(prefs)

        # Reset
        reset_prefs = manager.reset()

        assert reset_prefs.visualization.dpi == 100
        assert prefs_file.exists()

    def test_add_recent_file(self, tmp_path: Path) -> None:
        """Test adding recent file."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        manager.add_recent_file("/path/to/file.bin")

        prefs = manager.load(use_cache=False)
        assert "/path/to/file.bin" in prefs.recent_files

    def test_add_recent_file_moves_to_front(self, tmp_path: Path) -> None:
        """Test adding existing recent file moves it to front."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        manager.add_recent_file("/file1.bin")
        manager.add_recent_file("/file2.bin")
        manager.add_recent_file("/file1.bin")  # Add again

        prefs = manager.load(use_cache=False)
        assert prefs.recent_files[0] == "/file1.bin"

    def test_add_recent_file_limits_count(self, tmp_path: Path) -> None:
        """Test adding recent files respects max count."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        for i in range(15):
            manager.add_recent_file(f"/file{i}.bin")

        prefs = manager.load(use_cache=False)
        assert len(prefs.recent_files) == 10  # Default max

    def test_get_recent_files(self, tmp_path: Path) -> None:
        """Test getting recent files."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        for i in range(5):
            manager.add_recent_file(f"/file{i}.bin")

        recent = manager.get_recent_files()

        assert len(recent) == 5
        assert recent[0] == "/file4.bin"  # Most recent

    def test_get_recent_files_with_limit(self, tmp_path: Path) -> None:
        """Test getting recent files with count limit."""
        prefs_file = tmp_path / "prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        for i in range(10):
            manager.add_recent_file(f"/file{i}.bin")

        recent = manager.get_recent_files(max_count=3)

        assert len(recent) == 3

    def test_default_path_uses_xdg_config(self, tmp_path: Path, monkeypatch) -> None:
        """Test default path uses XDG_CONFIG_HOME."""
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

        manager = PreferencesManager()

        assert str(tmp_path / "config") in str(manager.path)

    def test_default_path_falls_back_to_home(self, tmp_path: Path, monkeypatch) -> None:
        """Test default path falls back to home directory."""
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = PreferencesManager()

        assert str(tmp_path / ".config") in str(manager.path)


# =============================================================================
# Global Functions Tests
# =============================================================================


class TestGlobalFunctions:
    """Test global preferences functions."""

    def test_get_preferences_manager_singleton(self) -> None:
        """Test get_preferences_manager returns singleton."""
        manager1 = get_preferences_manager()
        manager2 = get_preferences_manager()

        assert manager1 is manager2

    def test_get_preferences(self, tmp_path: Path) -> None:
        """Test get_preferences function."""
        # Create a manager with known path
        with patch("oscura.config.preferences._manager", None):
            with patch("oscura.config.preferences.PreferencesManager") as mock_manager:
                mock_instance = MagicMock()
                mock_instance.load.return_value = UserPreferences()
                mock_manager.return_value = mock_instance

                prefs = get_preferences()

                assert isinstance(prefs, UserPreferences)
                mock_instance.load.assert_called_once()

    def test_save_preferences(self, tmp_path: Path) -> None:
        """Test save_preferences function."""
        with patch("oscura.config.preferences._manager", None):
            with patch("oscura.config.preferences.PreferencesManager") as mock_manager:
                mock_instance = MagicMock()
                mock_manager.return_value = mock_instance

                prefs = UserPreferences()
                save_preferences(prefs)

                mock_instance.save.assert_called_once_with(prefs)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPreferencesIntegration:
    """Integration tests for preferences system."""

    def test_full_preferences_workflow(self, tmp_path: Path) -> None:
        """Test complete preferences workflow."""
        prefs_file = tmp_path / "workflow_prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        # Create and customize preferences
        prefs = UserPreferences()
        prefs.visualization.dark_mode = True
        prefs.defaults.sample_rate = 2e9
        prefs.export.default_format = "hdf5"
        prefs.logging.level = "DEBUG"
        prefs.editor.history_size = 2000
        prefs.custom["project"] = "my_project"

        # Save
        manager.save(prefs)

        # Load and verify
        loaded = manager.load(use_cache=False)

        assert loaded.visualization.dark_mode is True
        assert loaded.defaults.sample_rate == 2e9
        assert loaded.export.default_format == "hdf5"
        assert loaded.logging.level == "DEBUG"
        assert loaded.editor.history_size == 2000
        assert loaded.custom["project"] == "my_project"

    def test_preferences_migration(self, tmp_path: Path) -> None:
        """Test preferences can handle missing fields gracefully."""
        prefs_file = tmp_path / "old_prefs.yaml"

        # Write old format with missing fields
        old_data = {"visualization": {"dpi": 150}}
        with open(prefs_file, "w") as f:
            yaml.dump(old_data, f)

        manager = PreferencesManager(path=prefs_file)
        prefs = manager.load(use_cache=False)

        # Should have loaded value
        assert prefs.visualization.dpi == 150
        # Should have defaults for missing values
        assert prefs.defaults.sample_rate == 1e9
        assert prefs.export.precision == 6

    def test_preferences_validation_warning(self, tmp_path: Path) -> None:
        """Test validation warnings don't break loading."""
        prefs_file = tmp_path / "prefs.yaml"

        # Create valid preferences
        data = {
            "visualization": {"dpi": 150},
            "unknown_section": {"key": "value"},  # Extra data
        }
        with open(prefs_file, "w") as f:
            yaml.dump(data, f)

        manager = PreferencesManager(path=prefs_file)

        # Should load successfully despite validation warning
        prefs = manager.load(use_cache=False)

        assert prefs.visualization.dpi == 150

    def test_concurrent_preference_updates(self, tmp_path: Path) -> None:
        """Test handling of concurrent preference updates."""
        prefs_file = tmp_path / "concurrent_prefs.yaml"
        manager = PreferencesManager(path=prefs_file)

        # Initial save
        prefs1 = UserPreferences()
        prefs1.visualization.dpi = 150
        manager.save(prefs1)

        # Load and modify (simulating concurrent access)
        prefs2 = manager.load(use_cache=False)
        prefs2.defaults.sample_rate = 2e9
        manager.save(prefs2)

        # Reload and verify last save wins
        final = manager.load(use_cache=False)
        assert final.defaults.sample_rate == 2e9

    def test_preference_get_set_integration(self) -> None:
        """Test integrated get/set workflow."""
        prefs = UserPreferences()

        # Set various preferences
        prefs.set("visualization.dpi", 150)
        prefs.set("defaults.sample_rate", 2e9)
        prefs.set("custom.my_setting", "value")

        # Get them back
        assert prefs.get("visualization.dpi") == 150
        assert prefs.get("defaults.sample_rate") == 2e9
        assert prefs.get("custom.my_setting") == "value"

        # Convert to dict and back
        data = prefs.to_dict()
        restored = UserPreferences.from_dict(data)

        # Verify all values preserved
        assert restored.get("visualization.dpi") == 150
        assert restored.get("defaults.sample_rate") == 2e9
        assert restored.get("custom.my_setting") == "value"
