"""Plugin-specific logging with isolated namespaces.

This module provides plugin-specific loggers with isolated namespaces
and per-plugin configuration for proper log management.


Example:
    >>> from oscura.core.extensibility.logging import get_plugin_logger
    >>> logger = get_plugin_logger("my_plugin")
    >>> logger.info("Plugin initialized")

References:
"""

from __future__ import annotations

import logging
from typing import Any

from oscura.core.logging import get_logger

# Plugin logger namespace prefix
PLUGIN_LOGGER_PREFIX = "oscura.plugins"

# Per-plugin log level configuration
_plugin_log_levels: dict[str, int] = {}


def get_plugin_logger(plugin_name: str) -> logging.Logger:
    """Get a logger for a specific plugin.

    Creates a logger under the oscura.plugins.<plugin_name> namespace
    with plugin-specific configuration.

    Args:
        plugin_name: Name of the plugin.

    Returns:
        Configured logging.Logger for the plugin.

    Example:
        >>> logger = get_plugin_logger("my_decoder")
        >>> logger.info("Decoding frame", frame_id=42)

    References:
        LOG-014: Isolated Logger Namespaces for Plugins
    """
    logger_name = f"{PLUGIN_LOGGER_PREFIX}.{plugin_name}"
    logger = get_logger(logger_name)

    # Apply per-plugin log level if configured
    if plugin_name in _plugin_log_levels:
        logger.setLevel(_plugin_log_levels[plugin_name])

    return logger


def set_plugin_log_level(plugin_name: str, level: str | int) -> None:
    """Set log level for a specific plugin.

    Args:
        plugin_name: Name of the plugin.
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL or int).

    Example:
        >>> set_plugin_log_level("my_plugin", "DEBUG")

    References:
        LOG-014: Isolated Logger Namespaces for Plugins
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    _plugin_log_levels[plugin_name] = level  # type: ignore[assignment]

    # Update existing logger if it exists
    logger_name = f"{PLUGIN_LOGGER_PREFIX}.{plugin_name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)


def get_plugin_log_level(plugin_name: str) -> int | None:
    """Get the configured log level for a plugin.

    Args:
        plugin_name: Name of the plugin.

    Returns:
        Log level as integer, or None if not configured.
    """
    return _plugin_log_levels.get(plugin_name)


class PluginLoggerAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """Logger adapter that adds plugin context to all log messages.

    Automatically includes plugin name and version in all log records.

    Example:
        >>> adapter = PluginLoggerAdapter("my_plugin", "1.0.0")
        >>> adapter.info("Processing data")
        # Logs: "Processing data" with extra={'plugin': 'my_plugin', 'version': '1.0.0'}

    References:
        LOG-014: Isolated Logger Namespaces for Plugins
    """

    def __init__(
        self,
        plugin_name: str,
        version: str = "0.0.0",
        extra: dict[str, Any] | None = None,
    ):
        """Initialize plugin logger adapter.

        Args:
            plugin_name: Name of the plugin.
            version: Plugin version string.
            extra: Additional context to include in all logs.
        """
        logger = get_plugin_logger(plugin_name)
        base_extra = {
            "plugin": plugin_name,
            "plugin_version": version,
        }
        if extra:
            base_extra.update(extra)
        super().__init__(logger, base_extra)
        self.plugin_name = plugin_name
        self.version = version

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:  # type: ignore[override]
        """Process log message to include plugin context.

        Args:
            msg: Log message.
            kwargs: Keyword arguments.

        Returns:
            Tuple of (message, kwargs) with plugin context added.
        """
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def log_plugin_lifecycle(
    plugin_name: str,
    event: str,
    *,
    version: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Log a plugin lifecycle event.

    Standard logging for plugin discovery, loading, registration,
    unloading, and error events.

    Args:
        plugin_name: Name of the plugin.
        event: Lifecycle event (discovered, loading, loaded, registered,
               unloading, unloaded, error, reload).
        version: Plugin version if available.
        details: Additional event details.

    Example:
        >>> log_plugin_lifecycle("my_plugin", "loaded", version="1.0.0")
        >>> log_plugin_lifecycle("my_plugin", "error", details={"error": "Import failed"})

    References:
        LOG-014: Isolated Logger Namespaces for Plugins
    """
    logger = get_logger(PLUGIN_LOGGER_PREFIX)

    event_levels = {
        "discovered": logging.DEBUG,
        "loading": logging.DEBUG,
        "loaded": logging.INFO,
        "registered": logging.INFO,
        "unloading": logging.DEBUG,
        "unloaded": logging.INFO,
        "error": logging.ERROR,
        "reload": logging.INFO,
    }

    level = event_levels.get(event, logging.INFO)
    extra: dict[str, Any] = {
        "plugin": plugin_name,
        "lifecycle_event": event,
    }
    if version:
        extra["plugin_version"] = version
    if details:
        extra.update(details)

    logger.log(
        level,
        "Plugin '%s' %s",
        plugin_name,
        event,
        extra=extra,
    )


def configure_plugin_logging(
    default_level: str = "INFO",
    plugin_levels: dict[str, str] | None = None,
) -> None:
    """Configure logging for all plugins.

    Sets up default and per-plugin log levels for the plugin
    logger namespace.

    Args:
        default_level: Default log level for all plugins.
        plugin_levels: Dict mapping plugin names to log levels.

    Example:
        >>> configure_plugin_logging(
        ...     default_level="WARNING",
        ...     plugin_levels={
        ...         "debug_plugin": "DEBUG",
        ...         "noisy_plugin": "ERROR",
        ...     }
        ... )

    References:
        LOG-014: Isolated Logger Namespaces for Plugins
    """
    # Set default level for plugin namespace
    plugin_root = logging.getLogger(PLUGIN_LOGGER_PREFIX)
    plugin_root.setLevel(getattr(logging, default_level.upper()))

    # Set per-plugin levels
    if plugin_levels:
        for plugin_name, level in plugin_levels.items():
            set_plugin_log_level(plugin_name, level)


def list_plugin_loggers() -> list[str]:
    """List all registered plugin loggers.

    Returns:
        List of plugin names with configured loggers.
    """
    return list(_plugin_log_levels.keys())


__all__ = [
    "PLUGIN_LOGGER_PREFIX",
    "PluginLoggerAdapter",
    "configure_plugin_logging",
    "get_plugin_log_level",
    "get_plugin_logger",
    "list_plugin_loggers",
    "log_plugin_lifecycle",
    "set_plugin_log_level",
]
