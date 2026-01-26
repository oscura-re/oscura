"""Application settings management system.

This module provides application-wide settings management for Oscura,
including feature flags, CLI defaults, output formats, and runtime
configuration options.


Example:
    >>> from oscura.core.config.settings import get_settings, Settings
    >>> settings = get_settings()
    >>> settings.enable_feature("advanced_analysis")
    >>> if settings.is_feature_enabled("advanced_analysis"):
    ...     perform_advanced_analysis()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oscura.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class CLIDefaults:
    """CLI default settings.

    Attributes:
        output_format: Default output format (json, yaml, text)
        verbosity: Default verbosity level (0-3)
        color_output: Enable colored output
        progress_bar: Show progress bars
        parallel_workers: Number of parallel workers
    """

    output_format: str = "text"
    verbosity: int = 1
    color_output: bool = True
    progress_bar: bool = True
    parallel_workers: int = 4


@dataclass
class AnalysisSettings:
    """Analysis configuration settings.

    Attributes:
        max_trace_size: Maximum trace size in bytes (0 = unlimited)
        enable_caching: Enable result caching
        cache_dir: Cache directory path
        timeout: Default timeout for analysis in seconds
        streaming_mode: Enable streaming mode for large files
    """

    max_trace_size: int = 0
    enable_caching: bool = True
    cache_dir: str | None = None
    timeout: float = 300.0
    streaming_mode: bool = False


@dataclass
class OutputSettings:
    """Output and export configuration.

    Attributes:
        default_format: Default export format (csv, json, hdf5)
        include_raw_data: Include raw waveform data in exports
        compress_output: Compress output files
        decimal_places: Decimal precision for numeric output
        timestamp_format: Format for timestamps
    """

    default_format: str = "csv"
    include_raw_data: bool = False
    compress_output: bool = False
    decimal_places: int = 6
    timestamp_format: str = "iso8601"


@dataclass
class Settings:
    """Application-wide settings.

    Attributes:
        cli: CLI defaults
        analysis: Analysis settings
        output: Output settings
        features: Feature flags
        custom: Custom user-defined settings

    Example:
        >>> settings = Settings()
        >>> settings.cli.verbosity = 2
        >>> settings.analysis.max_trace_size = 1024**3  # 1 GB
    """

    cli: CLIDefaults = field(default_factory=CLIDefaults)
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    features: dict[str, bool] = field(default_factory=dict)
    custom: dict[str, Any] = field(default_factory=dict)

    def enable_feature(self, name: str) -> None:
        """Enable a feature flag.

        Args:
            name: Feature name

        Example:
            >>> settings.enable_feature("advanced_analysis")
        """
        self.features[name] = True
        logger.debug(f"Feature enabled: {name}")

    def disable_feature(self, name: str) -> None:
        """Disable a feature flag.

        Args:
            name: Feature name

        Example:
            >>> settings.disable_feature("experimental_mode")
        """
        self.features[name] = False
        logger.debug(f"Feature disabled: {name}")

    def is_feature_enabled(self, name: str) -> bool:
        """Check if a feature is enabled.

        Args:
            name: Feature name

        Returns:
            True if feature is enabled, False otherwise

        Example:
            >>> if settings.is_feature_enabled("advanced_analysis"):
            ...     perform_analysis()
        """
        return self.features.get(name, False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value by dot-notation key.

        Args:
            key: Key path (e.g., "cli.verbosity" or "custom.my_setting")
            default: Default value if not found

        Returns:
            Setting value

        Example:
            >>> settings.get("cli.verbosity")
            1
            >>> settings.get("custom.undefined", "fallback")
            'fallback'
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
        """Set setting value by dot-notation key.

        Args:
            key: Key path (e.g., "cli.verbosity" or "custom.my_setting")
            value: Value to set

        Raises:
            KeyError: If path is invalid

        Example:
            >>> settings.set("cli.verbosity", 2)
            >>> settings.set("custom.my_setting", 42)
        """
        parts = key.split(".")

        if parts[0] == "custom":
            # Custom settings are a dict
            if len(parts) == 2:
                self.custom[parts[1]] = value
            else:
                # Nested custom settings
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
                raise KeyError(f"Invalid setting path: {key}")

        # Set value
        final_part = parts[-1]
        if hasattr(obj, final_part):
            setattr(obj, final_part, value)
        else:
            raise KeyError(f"Unknown setting: {key}")

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "cli": {
                "output_format": self.cli.output_format,
                "verbosity": self.cli.verbosity,
                "color_output": self.cli.color_output,
                "progress_bar": self.cli.progress_bar,
                "parallel_workers": self.cli.parallel_workers,
            },
            "analysis": {
                "max_trace_size": self.analysis.max_trace_size,
                "enable_caching": self.analysis.enable_caching,
                "cache_dir": self.analysis.cache_dir,
                "timeout": self.analysis.timeout,
                "streaming_mode": self.analysis.streaming_mode,
            },
            "output": {
                "default_format": self.output.default_format,
                "include_raw_data": self.output.include_raw_data,
                "compress_output": self.output.compress_output,
                "decimal_places": self.output.decimal_places,
                "timestamp_format": self.output.timestamp_format,
            },
            "features": self.features,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        """Create settings from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Settings instance
        """
        settings = cls()

        # CLI settings
        if "cli" in data:
            c = data["cli"]
            settings.cli = CLIDefaults(
                output_format=c.get("output_format", settings.cli.output_format),
                verbosity=c.get("verbosity", settings.cli.verbosity),
                color_output=c.get("color_output", settings.cli.color_output),
                progress_bar=c.get("progress_bar", settings.cli.progress_bar),
                parallel_workers=c.get("parallel_workers", settings.cli.parallel_workers),
            )

        # Analysis settings
        if "analysis" in data:
            a = data["analysis"]
            settings.analysis = AnalysisSettings(
                max_trace_size=a.get("max_trace_size", settings.analysis.max_trace_size),
                enable_caching=a.get("enable_caching", settings.analysis.enable_caching),
                cache_dir=a.get("cache_dir", settings.analysis.cache_dir),
                timeout=a.get("timeout", settings.analysis.timeout),
                streaming_mode=a.get("streaming_mode", settings.analysis.streaming_mode),
            )

        # Output settings
        if "output" in data:
            o = data["output"]
            settings.output = OutputSettings(
                default_format=o.get("default_format", settings.output.default_format),
                include_raw_data=o.get("include_raw_data", settings.output.include_raw_data),
                compress_output=o.get("compress_output", settings.output.compress_output),
                decimal_places=o.get("decimal_places", settings.output.decimal_places),
                timestamp_format=o.get("timestamp_format", settings.output.timestamp_format),
            )

        settings.features = data.get("features", {})
        settings.custom = data.get("custom", {})

        return settings


# Global settings instance
_global_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global application settings.

    Returns:
        Global Settings instance
    """
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
    return _global_settings


def set_settings(settings: Settings) -> None:
    """Set global application settings.

    Args:
        settings: Settings instance to use globally
    """
    global _global_settings
    _global_settings = settings
    logger.debug("Global settings updated")


def reset_settings() -> None:
    """Reset settings to defaults."""
    global _global_settings
    _global_settings = Settings()
    logger.debug("Settings reset to defaults")


def load_settings(path: Path | str) -> Settings:
    """Load settings from a JSON file.

    Args:
        path: Path to settings file

    Returns:
        Loaded Settings instance

    Raises:
        ConfigurationError: If file cannot be read or parsed

    Example:
        >>> settings = load_settings("settings.json")
    """
    try:
        path_obj = Path(path).expanduser()

        if not path_obj.exists():
            raise ConfigurationError(f"Settings file not found: {path}")

        with open(path_obj, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ConfigurationError("Settings file must contain a JSON object")

        logger.debug(f"Loaded settings from {path_obj}")
        return Settings.from_dict(data)

    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Failed to parse settings JSON: {e}") from e
    except OSError as e:
        raise ConfigurationError(f"Failed to read settings file: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error loading settings: {e}") from e


def save_settings(settings: Settings, path: Path | str) -> None:
    """Save settings to a JSON file.

    Args:
        settings: Settings to save
        path: Path to save settings to

    Raises:
        ConfigurationError: If file cannot be written

    Example:
        >>> settings = get_settings()
        >>> save_settings(settings, "settings.json")
    """
    try:
        path_obj = Path(path).expanduser()

        # Create parent directory if needed
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(settings.to_dict(), f, indent=2)

        logger.debug(f"Saved settings to {path_obj}")

    except OSError as e:
        raise ConfigurationError(f"Failed to write settings file: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Error saving settings: {e}") from e


__all__ = [
    "AnalysisSettings",
    "CLIDefaults",
    "OutputSettings",
    "Settings",
    "get_settings",
    "load_settings",
    "reset_settings",
    "save_settings",
    "set_settings",
]
