"""Oscura configuration loading and management.

This module provides configuration loading from YAML files with support
for nested structures, user overrides, and schema validation.


Example:
    >>> from oscura.core.config import load_config
    >>> config = load_config()
    >>> print(config["defaults"]["sample_rate"])
    1000000.0
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from oscura.core.exceptions import ConfigurationError

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
        "formats": ["wfm", "csv", "npz", "hdf5", "tdms", "vcd", "sr", "wav"],
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
    },
    "export": {
        "csv": {"precision": 6},
        "hdf5": {"compression": "gzip", "compression_opts": 4},
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries.

    Values from override take precedence. Nested dictionaries are
    merged recursively.

    Args:
        base: Base dictionary.
        override: Dictionary with values to override.

    Returns:
        Merged dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(
    config_path: str | Path | None = None,
    *,
    use_defaults: bool = True,
) -> dict[str, Any]:
    """Load configuration from YAML file.

    Loads configuration from the specified path and optionally merges
    with default values. If no path is specified, looks for configuration
    in standard locations.

    Args:
        config_path: Path to YAML configuration file. If None, searches
            for config in standard locations.
        use_defaults: If True, merge loaded config with defaults.

    Returns:
        Configuration dictionary.

    Raises:
        ConfigurationError: If configuration file cannot be loaded or parsed.

    Example:
        >>> config = load_config()
        >>> print(config["defaults"]["sample_rate"])
        1000000.0

        >>> config = load_config("~/.oscura/config.yaml")
        >>> print(config["measurements"]["rise_time"]["ref_levels"])
        [0.1, 0.9]
    """
    config: dict[str, Any] = {}

    if use_defaults:
        config = copy.deepcopy(DEFAULT_CONFIG)

    if config_path is None:
        # Search standard locations
        search_paths = [
            Path.cwd() / "oscura.yaml",
            Path.cwd() / ".oscura.yaml",
            Path.home() / ".oscura" / "config.yaml",
            Path.home() / ".config" / "oscura" / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                config_path = path
                break

    if config_path is not None:
        config_path = Path(config_path).expanduser()

        if not config_path.exists():
            raise ConfigurationError(
                "Configuration file not found",
                config_key=str(config_path),
                fix_hint=f"Create configuration file at {config_path}",
            )

        if not YAML_AVAILABLE:
            raise ConfigurationError(
                "YAML support not available",
                details="PyYAML package is required for configuration loading",
                fix_hint="Install PyYAML: pip install pyyaml",
            )

        try:
            with open(config_path, encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                "Failed to parse configuration file",
                config_key=str(config_path),
                details=str(e),
            ) from e
        except OSError as e:
            raise ConfigurationError(
                "Failed to read configuration file",
                config_key=str(config_path),
                details=str(e),
            ) from e

        if user_config is not None:
            config = _deep_merge(config, user_config) if use_defaults else user_config

    return config


def validate_config(config: dict[str, Any]) -> bool:
    """Validate configuration against schema.

    Checks that required fields exist and have valid types.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        True if configuration is valid.

    Raises:
        ConfigurationError: If configuration is invalid.

    Example:
        >>> config = load_config()
        >>> validate_config(config)
        True
    """
    # Required top-level sections
    required_sections = ["defaults", "loaders"]

    for section in required_sections:
        if section not in config:
            raise ConfigurationError(
                f"Missing required configuration section: {section}",
                config_key=section,
            )

    # Validate defaults section
    defaults = config.get("defaults", {})
    if "sample_rate" in defaults:
        sample_rate = defaults["sample_rate"]
        if not isinstance(sample_rate, int | float) or sample_rate <= 0:
            raise ConfigurationError(
                "Invalid sample_rate in defaults",
                config_key="defaults.sample_rate",
                expected_type="positive number",
                actual_value=sample_rate,
            )

    # Validate loaders section
    loaders = config.get("loaders", {})
    if "formats" in loaders:
        formats = loaders["formats"]
        if not isinstance(formats, list):
            raise ConfigurationError(
                "Invalid formats in loaders",
                config_key="loaders.formats",
                expected_type="list of strings",
                actual_value=type(formats).__name__,
            )

    # Validate measurements section
    measurements = config.get("measurements", {})
    for name, settings in measurements.items():
        if "ref_levels" in settings:
            ref_levels = settings["ref_levels"]
            if not isinstance(ref_levels, list) or len(ref_levels) != 2:
                raise ConfigurationError(
                    f"Invalid ref_levels for {name}",
                    config_key=f"measurements.{name}.ref_levels",
                    expected_type="list of 2 numbers",
                    actual_value=ref_levels,
                )

    return True


def get_config_value(
    config: dict[str, Any],
    key_path: str,
    default: Any = None,
) -> Any:
    """Get a configuration value by dot-separated path.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the value (e.g., "defaults.sample_rate").
        default: Default value if key not found.

    Returns:
        Configuration value or default.

    Example:
        >>> config = load_config()
        >>> get_config_value(config, "defaults.sample_rate", 1e6)
        1000000.0
        >>> get_config_value(config, "unknown.key", "default")
        'default'
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def save_config(config: dict[str, Any], config_path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Path to save configuration to.

    Raises:
        ConfigurationError: If configuration cannot be saved.

    Example:
        >>> config = load_config()
        >>> config["defaults"]["sample_rate"] = 2e6
        >>> save_config(config, "~/my_config.yaml")
    """
    if not YAML_AVAILABLE:
        raise ConfigurationError(
            "YAML support not available",
            details="PyYAML package is required for configuration saving",
            fix_hint="Install PyYAML: pip install pyyaml",
        )

    config_path = Path(config_path).expanduser()

    # Create parent directory if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    except OSError as e:
        raise ConfigurationError(
            "Failed to save configuration file",
            config_key=str(config_path),
            details=str(e),
        ) from e


class SmartDefaults:
    """Intelligent defaults configuration.

    Provides smart default parameters that work for 80% of use cases,
    with automatic parameter selection based on signal characteristics.

    All auto-selected parameters are logged with rationale for transparency.
    Users can override any parameter.

    Example:
        >>> from oscura.core.config import SmartDefaults
        >>> defaults = SmartDefaults()
        >>> # Get smart default for FFT size based on signal length
        >>> fft_size = defaults.get_fft_size(signal_length=10000)
        >>> print(fft_size)
        8192

    References:
        Best practices from scipy.signal, matplotlib, and numpy
    """

    def __init__(self, verbose: bool = False):
        """Initialize SmartDefaults.

        Args:
            verbose: If True, log parameter selection rationale.
        """
        self.verbose = verbose
        self._log_buffer: list[str] = []

    def _log(self, message: str) -> None:
        """Log a parameter selection message.

        Args:
            message: Log message.
        """
        self._log_buffer.append(message)
        if self.verbose:
            print(f"[SmartDefaults] {message}")

    def get_fft_size(
        self,
        signal_length: int,
        *,
        min_size: int = 256,
        max_size: int = 2**16,
    ) -> int:
        """Get smart default FFT size.

        Args:
            signal_length: Length of signal in samples.
            min_size: Minimum FFT size.
            max_size: Maximum FFT size.

        Returns:
            Recommended FFT size (power of 2).
        """
        # Next power of 2 at or above signal length
        nfft = 2 ** int(np.ceil(np.log2(signal_length)))

        # Clamp to reasonable range
        nfft = max(min_size, min(nfft, max_size))

        self._log(
            f"FFT size: {nfft} (signal_length={signal_length}, "
            f"next_power_of_2={2 ** int(np.ceil(np.log2(signal_length)))})"
        )

        return nfft  # type: ignore[no-any-return]

    def get_window_function(
        self,
        application: str = "general",
        *,
        dynamic_range_db: float = 60.0,
    ) -> str:
        """Get smart default window function.

        Args:
            application: Application type ('general', 'narrowband', 'transient').
            dynamic_range_db: Required dynamic range in dB.

        Returns:
            Window function name.
        """
        if application == "transient":
            window = "boxcar"
            reason = "rectangular window for transient analysis"
        elif dynamic_range_db > 80:
            window = "blackman-harris"
            reason = f"high dynamic range ({dynamic_range_db} dB) requires Blackman-Harris"
        elif dynamic_range_db > 60:
            window = "blackman"
            reason = f"moderate-high dynamic range ({dynamic_range_db} dB) uses Blackman"
        elif application == "narrowband":
            window = "flattop"
            reason = "narrowband analysis uses flat-top for amplitude accuracy"
        else:
            window = "hann"
            reason = "general purpose uses Hann window"

        self._log(f"Window function: {window} ({reason})")

        return window

    def get_overlap(
        self,
        method: str = "welch",
        window: str = "hann",
    ) -> float:
        """Get smart default overlap for windowed methods.

        Args:
            method: Analysis method ('welch', 'bartlett', 'stft').
            window: Window function name.

        Returns:
            Overlap fraction (0-1).
        """
        if method == "bartlett":
            overlap = 0.0
            reason = "Bartlett method uses no overlap"
        elif window in ["hann", "hamming", "blackman"]:
            overlap = 0.5
            reason = f"{window} window typically uses 50% overlap"
        elif window == "blackman-harris":
            overlap = 0.75
            reason = "Blackman-Harris uses 75% overlap for smoothness"
        else:
            overlap = 0.5
            reason = "default 50% overlap for general windows"

        self._log(f"Overlap: {overlap * 100:.0f}% ({reason})")

        return overlap

    def get_reference_levels(
        self,
        measurement: str = "rise_time",
    ) -> tuple[float, float]:
        """Get smart default reference levels for timing measurements.

        Args:
            measurement: Measurement type ('rise_time', 'fall_time', etc.).

        Returns:
            Tuple of (low_level, high_level) as fractions (0-1).
        """
        if measurement in ["rise_time", "slew_rate"]:
            levels = (0.1, 0.9)
            reason = "10%-90% is IEEE 181-2011 standard for rise time"
        elif measurement == "fall_time":
            levels = (0.9, 0.1)
            reason = "90%-10% for fall time per IEEE 181-2011"
        elif measurement in ["propagation_delay", "setup_time", "hold_time"]:
            levels = (0.5, 0.5)
            reason = "50% threshold for timing measurements"
        else:
            levels = (0.1, 0.9)
            reason = "default 10%-90% for general measurements"

        self._log(f"Reference levels: {levels[0]:.0%}-{levels[1]:.0%} ({reason})")

        return levels

    def get_log_messages(self) -> list[str]:
        """Get all logged parameter selection messages.

        Returns:
            List of log messages.
        """
        return self._log_buffer.copy()

    def clear_log(self) -> None:
        """Clear the log buffer."""
        self._log_buffer.clear()


__all__ = [
    "DEFAULT_CONFIG",
    "SmartDefaults",
    "get_config_value",
    "load_config",
    "save_config",
    "validate_config",
]
