"""User preferences management system.

This module provides persistent user preferences for Oscura including
visualization settings, default parameters, export options, and UI
preferences.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from oscura.core.config.schema import validate_against_schema
from oscura.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class VisualizationPreferences:
    """Visualization preferences.

    Attributes:
        style: Matplotlib style name
        figure_size: Default figure size (width, height) in inches
        dpi: Default DPI for figures
        colormap: Default colormap name
        grid: Whether to show grid by default
        dark_mode: Use dark theme
    """

    style: str = "seaborn-v0_8-whitegrid"
    figure_size: tuple[float, float] = (10, 6)
    dpi: int = 100
    colormap: str = "viridis"
    grid: bool = True
    dark_mode: bool = False


@dataclass
class DefaultsPreferences:
    """Default analysis parameters.

    Attributes:
        sample_rate: Default sample rate (Hz)
        window_function: Default FFT window
        fft_size: Default FFT size
        rise_time_thresholds: Default rise time thresholds (low, high)
        logic_family: Default logic family
    """

    sample_rate: float = 1e9  # 1 GHz
    window_function: str = "hann"
    fft_size: int = 8192
    rise_time_thresholds: tuple[float, float] = (10.0, 90.0)
    logic_family: str = "TTL"


@dataclass
class ExportPreferences:
    """Export preferences.

    Attributes:
        default_format: Default export format
        precision: Floating point precision (decimal places)
        include_metadata: Include metadata in exports
        compression: Default compression for HDF5
    """

    default_format: str = "csv"
    precision: int = 6
    include_metadata: bool = True
    compression: str = "gzip"


@dataclass
class LoggingPreferences:
    """Logging preferences.

    Attributes:
        level: Default log level
        file: Log file path (None for no file logging)
        format: Log format string
        show_timestamps: Show timestamps in console
    """

    level: str = "WARNING"
    file: str | None = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    show_timestamps: bool = False


@dataclass
class EditorPreferences:
    """Editor/REPL preferences.

    Attributes:
        history_size: Number of commands to keep in history
        auto_save: Auto-save session on exit
        syntax_highlighting: Enable syntax highlighting
        tab_completion: Enable tab completion
    """

    history_size: int = 1000
    auto_save: bool = True
    syntax_highlighting: bool = True
    tab_completion: bool = True


@dataclass
class UserPreferences:
    """Complete user preferences.

    Attributes:
        visualization: Visualization preferences
        defaults: Default analysis parameters
        export: Export preferences
        logging: Logging preferences
        editor: Editor/REPL preferences
        recent_files: List of recent file paths
        custom: Custom user-defined preferences

    Example:
        >>> prefs = UserPreferences()
        >>> prefs.visualization.dark_mode = True
        >>> prefs.defaults.sample_rate = 2e9
        >>> prefs.save()
    """

    visualization: VisualizationPreferences = field(default_factory=VisualizationPreferences)
    defaults: DefaultsPreferences = field(default_factory=DefaultsPreferences)
    export: ExportPreferences = field(default_factory=ExportPreferences)
    logging: LoggingPreferences = field(default_factory=LoggingPreferences)
    editor: EditorPreferences = field(default_factory=EditorPreferences)
    recent_files: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get preference value by dot-notation key.

        Args:
            key: Key path (e.g., "visualization.dpi")
            default: Default value if not found

        Returns:
            Preference value

        Example:
            >>> prefs.get("visualization.dpi")
            100
            >>> prefs.get("custom.my_setting", 42)
            42
        """
        parts = key.split(".")
        obj: Any = self

        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part, default)
                if obj is default:
                    return default
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                return default

        return obj

    def set(self, key: str, value: Any) -> None:
        """Set preference value by dot-notation key.

        Args:
            key: Key path (e.g., "visualization.dpi")
            value: Value to set

        Raises:
            KeyError: If preference path is invalid or unknown.

        Example:
            >>> prefs.set("visualization.dpi", 150)
            >>> prefs.set("custom.my_setting", 42)
        """
        parts = key.split(".")

        if parts[0] == "custom":
            # Custom preferences are a dict
            if len(parts) == 2:
                self.custom[parts[1]] = value
            else:
                # Nested custom preferences
                current = self.custom
                for part in parts[1:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            return

        # Navigate to parent
        obj: Any = self
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Invalid preference path: {key}")

        # Set value
        final_part = parts[-1]
        if hasattr(obj, final_part):
            setattr(obj, final_part, value)
        else:
            raise KeyError(f"Unknown preference: {key}")

    def to_dict(self) -> dict[str, Any]:
        """Convert preferences to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "visualization": {
                "style": self.visualization.style,
                "figure_size": list(self.visualization.figure_size),
                "dpi": self.visualization.dpi,
                "colormap": self.visualization.colormap,
                "grid": self.visualization.grid,
                "dark_mode": self.visualization.dark_mode,
            },
            "defaults": {
                "sample_rate": self.defaults.sample_rate,
                "window_function": self.defaults.window_function,
                "fft_size": self.defaults.fft_size,
                "rise_time_thresholds": list(self.defaults.rise_time_thresholds),
                "logic_family": self.defaults.logic_family,
            },
            "export": {
                "default_format": self.export.default_format,
                "precision": self.export.precision,
                "include_metadata": self.export.include_metadata,
                "compression": self.export.compression,
            },
            "logging": {
                "level": self.logging.level,
                "file": self.logging.file,
                "format": self.logging.format,
                "show_timestamps": self.logging.show_timestamps,
            },
            "editor": {
                "history_size": self.editor.history_size,
                "auto_save": self.editor.auto_save,
                "syntax_highlighting": self.editor.syntax_highlighting,
                "tab_completion": self.editor.tab_completion,
            },
            "recent_files": self.recent_files,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserPreferences:
        """Create preferences from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            UserPreferences instance
        """
        prefs = cls()

        # Visualization
        if "visualization" in data:
            v = data["visualization"]
            prefs.visualization = VisualizationPreferences(
                style=v.get("style", prefs.visualization.style),
                figure_size=tuple(v.get("figure_size", prefs.visualization.figure_size)),
                dpi=v.get("dpi", prefs.visualization.dpi),
                colormap=v.get("colormap", prefs.visualization.colormap),
                grid=v.get("grid", prefs.visualization.grid),
                dark_mode=v.get("dark_mode", prefs.visualization.dark_mode),
            )

        # Defaults
        if "defaults" in data:
            d = data["defaults"]
            prefs.defaults = DefaultsPreferences(
                sample_rate=d.get("sample_rate", prefs.defaults.sample_rate),
                window_function=d.get("window_function", prefs.defaults.window_function),
                fft_size=d.get("fft_size", prefs.defaults.fft_size),
                rise_time_thresholds=tuple(
                    d.get("rise_time_thresholds", prefs.defaults.rise_time_thresholds)
                ),
                logic_family=d.get("logic_family", prefs.defaults.logic_family),
            )

        # Export
        if "export" in data:
            e = data["export"]
            prefs.export = ExportPreferences(
                default_format=e.get("default_format", prefs.export.default_format),
                precision=e.get("precision", prefs.export.precision),
                include_metadata=e.get("include_metadata", prefs.export.include_metadata),
                compression=e.get("compression", prefs.export.compression),
            )

        # Logging
        if "logging" in data:
            lg = data["logging"]
            prefs.logging = LoggingPreferences(
                level=lg.get("level", prefs.logging.level),
                file=lg.get("file", prefs.logging.file),
                format=lg.get("format", prefs.logging.format),
                show_timestamps=lg.get("show_timestamps", prefs.logging.show_timestamps),
            )

        # Editor
        if "editor" in data:
            ed = data["editor"]
            prefs.editor = EditorPreferences(
                history_size=ed.get("history_size", prefs.editor.history_size),
                auto_save=ed.get("auto_save", prefs.editor.auto_save),
                syntax_highlighting=ed.get("syntax_highlighting", prefs.editor.syntax_highlighting),
                tab_completion=ed.get("tab_completion", prefs.editor.tab_completion),
            )

        prefs.recent_files = data.get("recent_files", [])
        prefs.custom = data.get("custom", {})

        return prefs


class PreferencesManager:
    """Manager for loading and saving user preferences.

    Handles preferences file location, loading, saving, and migration.

    Example:
        >>> manager = PreferencesManager()
        >>> prefs = manager.load()
        >>> prefs.visualization.dark_mode = True
        >>> manager.save(prefs)
    """

    def __init__(self, path: Path | None = None):
        """Initialize preferences manager.

        Args:
            path: Override preferences file path
        """
        self._path = path or self._get_default_path()
        self._cached: UserPreferences | None = None

    def _get_default_path(self) -> Path:
        """Get default preferences file path."""
        xdg_config = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg_config) if xdg_config else Path.home() / ".config"

        config_dir = base / "oscura"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "preferences.yaml"

    def load(self, use_cache: bool = True) -> UserPreferences:
        """Load preferences from file.

        Args:
            use_cache: Use cached preferences if available

        Returns:
            User preferences
        """
        if use_cache and self._cached is not None:
            return self._cached

        if not self._path.exists():
            logger.debug(f"No preferences file at {self._path}, using defaults")
            self._cached = UserPreferences()
            return self._cached

        try:
            with open(self._path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                data = {}

            # Validate against schema if available
            try:
                validate_against_schema(data, "preferences")
            except Exception as e:
                logger.warning(f"Preferences validation warning: {e}")

            self._cached = UserPreferences.from_dict(data)
            logger.debug(f"Loaded preferences from {self._path}")
            return self._cached

        except Exception as e:
            logger.warning(f"Failed to load preferences from {self._path}: {e}")
            self._cached = UserPreferences()
            return self._cached

    def save(self, prefs: UserPreferences | None = None) -> None:
        """Save preferences to file.

        Args:
            prefs: Preferences to save (uses cached if None)

        Raises:
            ConfigurationError: If saving to file fails.
        """
        prefs = prefs or self._cached
        if prefs is None:
            prefs = UserPreferences()

        try:
            data = prefs.to_dict()

            # Ensure parent directory exists
            self._path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)

            self._cached = prefs
            logger.debug(f"Saved preferences to {self._path}")

        except Exception as e:
            logger.error(f"Failed to save preferences to {self._path}: {e}")
            raise ConfigurationError(f"Failed to save preferences to {self._path}: {e}") from e

    def reset(self) -> UserPreferences:
        """Reset preferences to defaults.

        Returns:
            Default preferences
        """
        self._cached = UserPreferences()
        self.save(self._cached)
        logger.info("Reset preferences to defaults")
        return self._cached

    def add_recent_file(self, path: str | Path, max_recent: int = 10) -> None:
        """Add file to recent files list.

        Args:
            path: File path
            max_recent: Maximum number of recent files to keep
        """
        prefs = self.load()
        path_str = str(path)

        # Remove if already in list
        if path_str in prefs.recent_files:
            prefs.recent_files.remove(path_str)

        # Add to front
        prefs.recent_files.insert(0, path_str)

        # Trim to max
        prefs.recent_files = prefs.recent_files[:max_recent]

        self.save(prefs)

    def get_recent_files(self, max_count: int = 10) -> list[str]:
        """Get list of recent files.

        Args:
            max_count: Maximum number to return

        Returns:
            List of recent file paths
        """
        prefs = self.load()
        return prefs.recent_files[:max_count]

    @property
    def path(self) -> Path:
        """Get preferences file path."""
        return self._path


# Global preferences manager
_manager: PreferencesManager | None = None


def get_preferences_manager() -> PreferencesManager:
    """Get global preferences manager.

    Returns:
        Global PreferencesManager instance
    """
    global _manager
    if _manager is None:
        _manager = PreferencesManager()
    return _manager


def get_preferences() -> UserPreferences:
    """Get current user preferences.

    Returns:
        User preferences
    """
    return get_preferences_manager().load()


def save_preferences(prefs: UserPreferences | None = None) -> None:
    """Save user preferences.

    Args:
        prefs: Preferences to save
    """
    get_preferences_manager().save(prefs)


__all__ = [
    "DefaultsPreferences",
    "EditorPreferences",
    "ExportPreferences",
    "LoggingPreferences",
    "PreferencesManager",
    "UserPreferences",
    "VisualizationPreferences",
    "get_preferences",
    "get_preferences_manager",
    "save_preferences",
]
