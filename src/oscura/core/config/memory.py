"""Memory configuration module for Oscura.

This module provides global memory limit configuration and settings.


Example:
    >>> from oscura.core.config.memory import set_memory_limit, get_memory_config
    >>> set_memory_limit("4GB")
    >>> config = get_memory_config()
    >>> print(f"Max memory: {config.max_memory / 1e9:.1f} GB")

References:
    See oscura.utils.memory for memory estimation and checking functions.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass


@dataclass
class MemoryConfiguration:
    """Global memory configuration for Oscura operations.


    Attributes:
        max_memory: Global memory limit in bytes (None = auto-detect 80% available).
        warn_threshold: Memory pressure warning threshold (0.0-1.0).
        critical_threshold: Memory pressure critical threshold (0.0-1.0).
        auto_degrade: Automatically downsample if memory exceeded.
        memory_reserve: Reserved memory headroom in bytes.
    """

    max_memory: int | None = None
    warn_threshold: float = 0.7
    critical_threshold: float = 0.9
    auto_degrade: bool = False
    memory_reserve: int = 0

    def __post_init__(self) -> None:
        """Validate configuration on initialization."""
        if not 0.0 <= self.warn_threshold <= 1.0:
            raise ValueError(
                f"warn_threshold must be in range [0.0, 1.0], got {self.warn_threshold}"
            )
        if not 0.0 <= self.critical_threshold <= 1.0:
            raise ValueError(
                f"critical_threshold must be in range [0.0, 1.0], got {self.critical_threshold}"
            )
        if self.warn_threshold >= self.critical_threshold:
            raise ValueError(
                f"warn_threshold ({self.warn_threshold}) must be less than "
                f"critical_threshold ({self.critical_threshold})"
            )
        if self.memory_reserve < 0:
            raise ValueError(f"memory_reserve must be non-negative, got {self.memory_reserve}")


# Global memory configuration instance
_global_config = MemoryConfiguration()


def get_memory_config() -> MemoryConfiguration:
    """Get the current global memory configuration.

    Returns:
        Current MemoryConfiguration instance.

    Example:
        >>> config = get_memory_config()
        >>> print(f"Max memory: {config.max_memory or 'auto'}")
    """
    return _global_config


def set_memory_limit(limit: int | str | None) -> None:
    """Set global memory limit for all Oscura operations.


    Args:
        limit: Memory limit as bytes (int), string ("4GB", "512MB"), or None for auto.

    Example:
        >>> set_memory_limit("4GB")
        >>> set_memory_limit(4 * 1024**3)  # 4 GB in bytes
        >>> set_memory_limit(None)  # Auto (80% of available)

    Environment:
        Can also be set via TK_MAX_MEMORY environment variable.
    """
    global _global_config

    if limit is None:
        _global_config.max_memory = None
        return

    if isinstance(limit, str):
        _global_config.max_memory = _parse_memory_string(limit)
    else:
        _global_config.max_memory = int(limit)


def set_memory_thresholds(
    warn_threshold: float | None = None,
    critical_threshold: float | None = None,
) -> None:
    """Set memory pressure warning thresholds.


    Args:
        warn_threshold: Warning threshold (0.0-1.0, e.g., 0.7 for 70% utilization).
        critical_threshold: Critical threshold (0.0-1.0, e.g., 0.9 for 90% utilization).

    Example:
        >>> set_memory_thresholds(warn_threshold=0.7, critical_threshold=0.9)
        >>> set_memory_thresholds(critical_threshold=0.95)  # Keep warn unchanged
    """
    global _global_config

    if warn_threshold is not None:
        _global_config.warn_threshold = warn_threshold
    if critical_threshold is not None:
        _global_config.critical_threshold = critical_threshold

    # Re-validate
    _global_config.__post_init__()


def enable_auto_degrade(enabled: bool = True) -> None:
    """Enable or disable automatic downsampling when memory limits exceeded.


    Args:
        enabled: Whether to enable automatic downsampling.

    Example:
        >>> enable_auto_degrade(True)
        >>> # Operations will now auto-downsample if memory insufficient
    """
    global _global_config
    _global_config.auto_degrade = enabled


def set_memory_reserve(reserve: int | str) -> None:
    """Set memory headroom to reserve (not use for operations).


    Args:
        reserve: Reserved memory as bytes (int) or string ("1GB", "512MB").

    Example:
        >>> set_memory_reserve("1GB")  # Reserve 1 GB for system
        >>> set_memory_reserve(512 * 1024**2)  # 512 MB
    """
    global _global_config

    if isinstance(reserve, str):
        _global_config.memory_reserve = _parse_memory_string(reserve)
    else:
        _global_config.memory_reserve = int(reserve)


def configure_from_environment() -> None:
    """Configure memory settings from environment variables.


    Environment Variables:
        TK_MAX_MEMORY: Maximum memory limit (e.g., "4GB", "512MB").
        TK_MEMORY_RESERVE: Reserved memory headroom (e.g., "1GB").
        TK_MEMORY_WARN_THRESHOLD: Warning threshold (e.g., "0.7").
        TK_MEMORY_CRITICAL_THRESHOLD: Critical threshold (e.g., "0.9").
        TK_AUTO_DEGRADE: Enable auto downsampling ("1", "true", "yes").

    Example:
        >>> import os
        >>> os.environ['TK_MAX_MEMORY'] = '4GB'
        >>> configure_from_environment()
    """
    # Max memory
    if max_mem := os.environ.get("TK_MAX_MEMORY"):
        set_memory_limit(max_mem)

    # Memory reserve
    if reserve := os.environ.get("TK_MEMORY_RESERVE"):
        set_memory_reserve(reserve)

    # Thresholds
    warn = None
    critical = None
    if warn_str := os.environ.get("TK_MEMORY_WARN_THRESHOLD"):
        with contextlib.suppress(ValueError):
            warn = float(warn_str)
    if crit_str := os.environ.get("TK_MEMORY_CRITICAL_THRESHOLD"):
        with contextlib.suppress(ValueError):
            critical = float(crit_str)
    if warn is not None or critical is not None:
        set_memory_thresholds(warn, critical)

    # Auto degrade
    if auto_str := os.environ.get("TK_AUTO_DEGRADE"):
        enable_auto_degrade(auto_str.lower() in ("1", "true", "yes", "on"))


def reset_to_defaults() -> None:
    """Reset memory configuration to default values.

    Example:
        >>> reset_to_defaults()
        >>> config = get_memory_config()
        >>> assert config.max_memory is None  # Auto-detect
    """
    global _global_config
    _global_config = MemoryConfiguration()


def _parse_memory_string(limit_str: str) -> int:
    """Parse memory limit string to bytes.

    Args:
        limit_str: Memory string like "4GB", "512MB", "1024KB".

    Returns:
        Memory limit in bytes.

    Raises:
        ValueError: If format is invalid.

    Example:
        >>> _parse_memory_string("4GB")
        4000000000
        >>> _parse_memory_string("512MB")
        512000000
    """
    limit_upper = limit_str.upper().strip()

    try:
        if limit_upper.endswith("GB"):
            return int(float(limit_upper[:-2]) * 1e9)
        elif limit_upper.endswith("MB"):
            return int(float(limit_upper[:-2]) * 1e6)
        elif limit_upper.endswith("KB"):
            return int(float(limit_upper[:-2]) * 1e3)
        elif limit_upper.endswith("GIB"):
            return int(float(limit_upper[:-3]) * 1024**3)
        elif limit_upper.endswith("MIB"):
            return int(float(limit_upper[:-3]) * 1024**2)
        elif limit_upper.endswith("KIB"):
            return int(float(limit_upper[:-3]) * 1024)
        else:
            # Assume bytes
            return int(float(limit_upper))
    except ValueError as e:
        raise ValueError(f"Invalid memory limit format: {limit_str}") from e


# Auto-configure from environment on import
configure_from_environment()


__all__ = [
    "MemoryConfiguration",
    "configure_from_environment",
    "enable_auto_degrade",
    "get_memory_config",
    "reset_to_defaults",
    "set_memory_limit",
    "set_memory_reserve",
    "set_memory_thresholds",
]
