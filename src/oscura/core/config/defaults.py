"""Default configuration values and injection.

This module provides default configuration values and utilities
for injecting defaults into user configurations.


Example:
    >>> from oscura.core.config.defaults import inject_defaults
    >>> config = {"name": "test"}
    >>> full_config = inject_defaults(config, "protocol")
"""

from __future__ import annotations

import copy
from typing import Any

# Default configuration values
DEFAULT_CONFIG: dict[str, Any] = {
    "version": "1.0",
    "defaults": {
        "sample_rate": 1e6,  # 1 MHz default
        "window_function": "hann",
        "fft_size": 1024,
    },
    "loaders": {
        "auto_detect": True,
        "formats": ["wfm", "csv", "npz", "hdf5", "tdms", "vcd", "sr", "wav", "pcap"],
        "tektronix": {"byte_order": "little"},
        "csv": {"delimiter": ",", "skip_header": 0},
    },
    "measurements": {
        "rise_time": {"ref_levels": [0.1, 0.9]},
        "fall_time": {"ref_levels": [0.9, 0.1]},
        "frequency": {"min_periods": 3},
    },
    "spectral": {
        "default_window": "hann",
        "overlap": 0.5,
        "nfft": None,  # Auto-determine from signal length
    },
    "visualization": {
        "default_style": "seaborn",
        "figure_size": [10, 6],
        "dpi": 100,
        "colormap": "viridis",
    },
    "export": {
        "csv": {"precision": 6},
        "hdf5": {"compression": "gzip", "compression_opts": 4},
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
}


# Schema-specific default values
SCHEMA_DEFAULTS: dict[str, dict[str, Any]] = {
    "protocol": {
        "version": "1.0.0",
        "timing": {
            "data_bits": [8],
            "stop_bits": [1],
            "parity": ["none"],
        },
    },
    "pipeline": {
        "version": "1.0.0",
        "parallel_groups": [],
    },
    "logic_family": {
        "temperature_range": {
            "min": 0,
            "max": 70,
        },
    },
    "threshold_profile": {
        "tolerance": 0,
        "overrides": {},
    },
    "preferences": {
        "defaults": {
            "sample_rate": 1e6,
            "window_function": "hann",
        },
        "visualization": {
            "style": "seaborn",
            "dpi": 100,
        },
        "export": {
            "default_format": "csv",
            "precision": 6,
        },
        "logging": {
            "level": "INFO",
        },
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Values from override take precedence. Nested dictionaries are
    merged recursively.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary (new instance).

    Example:
        >>> base = {"a": 1, "b": {"c": 2}}
        >>> override = {"b": {"d": 3}}
        >>> deep_merge(base, override)
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def inject_defaults(
    config: dict[str, Any],
    schema_name: str,
) -> dict[str, Any]:
    """Inject default values into configuration.

    Adds default values for missing fields based on schema type.

    Args:
        config: User configuration dictionary.
        schema_name: Schema name to determine defaults.

    Returns:
        Configuration with defaults injected.

    Example:
        >>> config = {"name": "uart", "timing": {"baud_rates": [9600]}}
        >>> full = inject_defaults(config, "protocol")
        >>> print(full["timing"]["data_bits"])
        [8]
    """
    defaults = SCHEMA_DEFAULTS.get(schema_name, {})

    if not defaults:
        return copy.deepcopy(config)

    # Merge defaults with config (config takes precedence)
    return deep_merge(defaults, config)


def get_effective_config(
    user_config: dict[str, Any] | None = None,
    schema_name: str | None = None,
) -> dict[str, Any]:
    """Get effective configuration with all defaults applied.

    Combines base defaults, schema-specific defaults, and user configuration.

    Args:
        user_config: User-provided configuration.
        schema_name: Schema to apply defaults for.

    Returns:
        Complete configuration with all defaults.

    Example:
        >>> config = get_effective_config({"defaults": {"sample_rate": 2e6}})
        >>> print(config["defaults"]["sample_rate"])
        2000000.0
    """
    # Start with base defaults
    result = copy.deepcopy(DEFAULT_CONFIG)

    # Add schema-specific defaults
    if schema_name and schema_name in SCHEMA_DEFAULTS:
        schema_defaults = SCHEMA_DEFAULTS[schema_name]
        result = deep_merge(result, schema_defaults)

    # Apply user configuration
    if user_config:
        result = deep_merge(result, user_config)

    return result


def get_default(
    key_path: str,
    schema_name: str | None = None,
) -> Any:
    """Get default value for a configuration key.

    Args:
        key_path: Dot-separated path (e.g., "defaults.sample_rate").
        schema_name: Optional schema for schema-specific defaults.

    Returns:
        Default value or None if not found.

    Example:
        >>> get_default("defaults.sample_rate")
        1000000.0
    """
    # Check schema-specific defaults first
    if schema_name and schema_name in SCHEMA_DEFAULTS:
        value = _get_nested(SCHEMA_DEFAULTS[schema_name], key_path)
        if value is not None:
            return value

    # Fall back to base defaults
    return _get_nested(DEFAULT_CONFIG, key_path)


def _get_nested(config: dict[str, Any], key_path: str) -> Any:
    """Get nested value by dot-separated path.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path.

    Returns:
        Value or None if not found.
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    return value


__all__ = [
    "DEFAULT_CONFIG",
    "SCHEMA_DEFAULTS",
    "deep_merge",
    "get_default",
    "get_effective_config",
    "inject_defaults",
]
