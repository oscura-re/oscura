"""Configuration file loading utilities.

This module provides unified configuration loading from YAML and JSON files
with support for schema validation, default injection, and path resolution.


Example:
    >>> from oscura.core.config.loader import load_config_file
    >>> config = load_config_file("pipeline.yaml", schema="pipeline")
"""

import json
from pathlib import Path
from typing import Any

from oscura.core.exceptions import ConfigurationError

# Try to import yaml
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config_file(
    path: str | Path,
    *,
    schema: str | None = None,
    validate: bool = True,
    inject_defaults: bool = True,
) -> dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Automatically detects file format from extension.
    Optionally validates against schema and injects defaults.

    Args:
        path: Path to configuration file.
        schema: Schema name to validate against (e.g., "protocol").
        validate: If True and schema provided, validate configuration.
        inject_defaults: If True, inject default values from schema.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigurationError: If file cannot be loaded or parsed.

    Example:
        >>> config = load_config_file("uart.yaml", schema="protocol")
        >>> print(config["name"])
        'uart'
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise ConfigurationError(
            "Configuration file not found",
            config_key=str(path),
        )

    # Determine format from extension
    ext = path.suffix.lower()

    if ext in (".yaml", ".yml"):
        config = _load_yaml(path)
    elif ext == ".json":
        config = _load_json(path)
    else:
        # Try YAML first, then JSON
        try:
            config = _load_yaml(path)
        except ConfigurationError:
            try:
                config = _load_json(path)
            except ConfigurationError:
                raise ConfigurationError(
                    f"Unsupported configuration format: {ext}",
                    config_key=str(path),
                    fix_hint="Use .yaml, .yml, or .json extension",
                )

    # Validate against schema if requested
    if validate and schema is not None:
        from oscura.core.config.schema import validate_against_schema

        validate_against_schema(config, schema)

    # Inject defaults if requested
    if inject_defaults and schema is not None:
        from oscura.core.config.defaults import inject_defaults as do_inject

        config = do_inject(config, schema)

    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If YAML parsing fails.
    """
    if not YAML_AVAILABLE:
        raise ConfigurationError(
            "YAML support not available",
            fix_hint="Install PyYAML: pip install pyyaml",
        )

    try:
        with open(path, encoding="utf-8") as f:
            content = yaml.safe_load(f)

        if content is None:
            return {}

        if not isinstance(content, dict):
            raise ConfigurationError(
                "Configuration must be a dictionary",
                config_key=str(path),
                expected_type="object",
                actual_value=type(content).__name__,
            )

        return content

    except yaml.YAMLError as e:
        # Extract line number from YAML error
        line = None
        if hasattr(e, "problem_mark") and e.problem_mark is not None:
            line = e.problem_mark.line + 1

        raise ConfigurationError(
            "Failed to parse YAML configuration",
            config_key=str(path),
            details=f"Line {line}: {e}" if line else str(e),
        ) from e
    except OSError as e:
        raise ConfigurationError(
            "Failed to read configuration file",
            config_key=str(path),
            details=str(e),
        ) from e


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON configuration file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ConfigurationError: If JSON parsing fails.
    """
    try:
        with open(path, encoding="utf-8") as f:
            content = json.load(f)

        if content is None:
            return {}

        if not isinstance(content, dict):
            raise ConfigurationError(
                "Configuration must be a dictionary",
                config_key=str(path),
                expected_type="object",
                actual_value=type(content).__name__,
            )

        return content

    except json.JSONDecodeError as e:
        raise ConfigurationError(
            "Failed to parse JSON configuration",
            config_key=str(path),
            details=f"Line {e.lineno}, column {e.colno}: {e.msg}",
        ) from e
    except OSError as e:
        raise ConfigurationError(
            "Failed to read configuration file",
            config_key=str(path),
            details=str(e),
        ) from e


def load_config(
    config_path: str | Path | None = None,
    *,
    use_defaults: bool = True,
) -> dict[str, Any]:
    """Load Oscura configuration.

    Searches for configuration in standard locations if no path provided.

    Args:
        config_path: Path to configuration file. If None, searches
            standard locations.
        use_defaults: If True, merge with default configuration.

    Returns:
        Configuration dictionary.

    Example:
        >>> config = load_config()  # Auto-find config
        >>> config = load_config("~/.oscura/config.yaml")
    """
    from oscura.core.config.defaults import DEFAULT_CONFIG, deep_merge

    config: dict[str, Any] = {}

    if use_defaults:
        import copy

        config = copy.deepcopy(DEFAULT_CONFIG)

    # Search for config files if no explicit path provided
    # use_defaults flag only controls whether to merge with DEFAULT_CONFIG
    if config_path is None:
        # Search standard locations
        search_paths = [
            Path.cwd() / "oscura.yaml",
            Path.cwd() / ".oscura.yaml",
            Path.cwd() / "oscura.json",
            Path.home() / ".oscura" / "config.yaml",
            Path.home() / ".config" / "oscura" / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                config_path = path
                break

    if config_path is not None:
        user_config = load_config_file(
            config_path,
            validate=False,
            inject_defaults=False,
        )
        config = deep_merge(config, user_config) if use_defaults else user_config

    return config


def save_config(
    config: dict[str, Any],
    path: str | Path,
    *,
    format: str | None = None,
) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary to save.
        path: Output file path.
        format: Output format ("yaml" or "json"). Auto-detected from
            extension if not specified.

    Raises:
        ConfigurationError: If configuration cannot be saved.

    Example:
        >>> save_config(config, "config.yaml")
        >>> save_config(config, "config.json")
    """
    path = Path(path).expanduser().resolve()

    # Determine format
    if format is None:
        ext = path.suffix.lower()
        if ext in (".yaml", ".yml"):
            format = "yaml"
        elif ext == ".json":
            format = "json"
        else:
            format = "yaml"  # Default

    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if format == "yaml":
            if not YAML_AVAILABLE:
                raise ConfigurationError(
                    "YAML support not available",
                    fix_hint="Install PyYAML: pip install pyyaml",
                )
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)

    except OSError as e:
        raise ConfigurationError(
            "Failed to save configuration",
            config_key=str(path),
            details=str(e),
        ) from e


def get_config_value(
    config: dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """Get configuration value by dot-separated path.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., "defaults.sample_rate").
        default: Default value if key not found.

    Returns:
        Configuration value or default.

    Example:
        >>> get_config_value(config, "defaults.sample_rate", 1e6)
        1000000.0
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


__all__ = [
    "get_config_value",
    "load_config",
    "load_config_file",
    "save_config",
]
