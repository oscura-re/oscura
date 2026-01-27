"""Plugin architecture for third-party extensions.

This module implements entry point discovery for third-party plugins,
allowing custom decoders, measurements, and file formats to be loaded
dynamically.
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class PluginError(Exception):
    """Exception raised when plugin loading fails.

    This exception is used to isolate plugin failures so they don't
    crash the main application.

    Example:
        >>> try:
        ...     plugin = osc.load_plugin('oscura.decoders', 'flexray')
        ... except osc.PluginError as e:
        ...     print(f"Plugin failed: {e}")

    References:
        API-007: Plugin Architecture
    """


@dataclass
class PluginMetadata:
    """Metadata about a loaded plugin.

    Attributes:
        name: Plugin name.
        entry_point: Entry point group.
        version: Plugin version (if available).
        module: Module name.
        callable: The loaded plugin object.
        dependencies: Plugin dependencies (if available).

    Example:
        >>> plugin = load_plugin('oscura.decoders', 'can')
        >>> print(f"Loaded {plugin.name} v{plugin.version}")

    References:
        API-007: Plugin Architecture
    """

    name: str
    entry_point: str
    version: str | None = None
    module: str | None = None
    callable: Any | None = None
    dependencies: list[str] | None = None

    def __repr__(self) -> str:
        """String representation of plugin metadata.

        Returns:
            String representation showing plugin name, version, and module.
        """
        parts = [f"name='{self.name}'"]
        if self.version:
            parts.append(f"version='{self.version}'")
        if self.module:
            parts.append(f"module='{self.module}'")
        return f"PluginMetadata({', '.join(parts)})"


class PluginManager:
    """Manager for discovering and loading third-party plugins.

    Discovers plugins via setuptools entry points and provides lazy loading
    with error isolation. Supports multiple entry point groups for different
    plugin types.

    Entry point groups:
    - oscura.decoders: Protocol decoders
    - oscura.measurements: Custom measurements
    - oscura.loaders: File format loaders
    - oscura.exporters: Export format handlers

    Example:
        >>> import oscura as osc
        >>> # Plugins auto-discovered from installed packages
        >>> # Use plugin decoder
        >>> can_frames = osc.decode(trace, protocol='can', baudrate=500000)
        >>> # List available plugins
        >>> plugins = osc.list_plugins()
        >>> print(f"Installed plugins: {plugins}")

    Advanced Example:
        >>> # Manually load plugin with error handling
        >>> try:
        ...     plugin = osc.load_plugin('oscura.decoders', 'flexray')
        ...     print(f"Plugin loaded: {plugin.name} v{plugin.version}")
        ... except osc.PluginError as e:
        ...     print(f"Plugin failed to load: {e}")

    Plugin Package Example:
        In your package's pyproject.toml:
        ```toml
        [project.entry-points."oscura.decoders"]
        flexray = "my_package.flexray:FlexRayDecoder"
        ```

    References:
        API-007: Plugin Architecture
        importlib.metadata entry points
        https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/
    """

    # Standard entry point groups
    ENTRY_POINT_GROUPS = [
        "oscura.decoders",
        "oscura.measurements",
        "oscura.loaders",
        "oscura.exporters",
    ]

    def __init__(self) -> None:
        """Initialize plugin manager."""
        self._loaded_plugins: dict[tuple[str, str], PluginMetadata] = {}
        self._failed_plugins: dict[tuple[str, str], Exception] = {}

    def discover_plugins(self, group: str | None = None) -> dict[str, list[str]]:
        """Discover available plugins via entry points.

        Args:
            group: Specific entry point group to search. If None, searches
                all standard groups.

        Returns:
            Dictionary mapping group names to lists of plugin names.

        Example:
            >>> manager = PluginManager()
            >>> plugins = manager.discover_plugins()
            >>> print(plugins)
            {'oscura.decoders': ['uart', 'spi', 'can'], ...}

        References:
            importlib.metadata.entry_points
        """
        discovered: dict[str, list[str]] = {}
        groups = [group] if group else self.ENTRY_POINT_GROUPS

        for group_name in groups:
            try:
                # Get entry points for this group
                # Python 3.10+ API
                eps = importlib.metadata.entry_points(group=group_name)
                discovered[group_name] = [ep.name for ep in eps]
            except (AttributeError, TypeError):
                # Fallback for Python 3.9 and earlier
                try:
                    eps = importlib.metadata.entry_points().get(group_name, [])  # type: ignore[attr-defined]
                    discovered[group_name] = [ep.name for ep in eps]
                except Exception as e:
                    logger.warning(f"Failed to discover plugins for group '{group_name}': {e}")
                    discovered[group_name] = []

        return discovered

    def load_plugin(
        self,
        group: str,
        name: str,
        reload: bool = False,
    ) -> PluginMetadata:
        """Load a plugin by group and name.

        Loads the plugin lazily on first use. Subsequent calls return cached
        instance unless reload=True.

        Args:
            group: Entry point group.
            name: Plugin name.
            reload: Force reload even if already loaded. Default False.

        Returns:
            PluginMetadata with loaded plugin information.

        Raises:
            PluginError: If plugin fails to load.

        Example:
            >>> manager = PluginManager()
            >>> plugin = manager.load_plugin('oscura.decoders', 'can')
            >>> decoder = plugin.callable

        References:
            API-007: Plugin Architecture
        """
        plugin_key = (group, name)

        # Check if already loaded
        if not reload and plugin_key in self._loaded_plugins:
            return self._loaded_plugins[plugin_key]

        # Check if previously failed
        if not reload and plugin_key in self._failed_plugins:
            raise PluginError(
                f"Plugin '{name}' in group '{group}' previously failed to load: "
                f"{self._failed_plugins[plugin_key]}"
            )

        try:
            # Find entry point
            entry_point = self._find_entry_point(group, name)

            if entry_point is None:
                raise PluginError(f"Plugin '{name}' not found in group '{group}'")

            # Load the plugin
            logger.info(f"Loading plugin '{name}' from group '{group}'")
            plugin_obj = entry_point.load()

            # Get version if available
            version = None
            if hasattr(entry_point, "dist") and entry_point.dist:
                version = entry_point.dist.version

            # Create metadata
            metadata = PluginMetadata(
                name=name,
                entry_point=group,
                version=version,
                module=entry_point.value,
                callable=plugin_obj,
            )

            # Cache loaded plugin
            self._loaded_plugins[plugin_key] = metadata

            logger.info(f"Successfully loaded plugin '{name}' v{version}")
            return metadata

        except Exception as e:
            # Cache failure
            self._failed_plugins[plugin_key] = e
            logger.error(f"Failed to load plugin '{name}': {e}")
            raise PluginError(f"Failed to load plugin '{name}' from group '{group}': {e}") from e

    def get_plugin(self, group: str, name: str) -> Any:
        """Get loaded plugin callable.

        Convenience method that loads plugin if needed and returns the
        callable object.

        Args:
            group: Entry point group.
            name: Plugin name.

        Returns:
            The loaded plugin object.

        Example:
            >>> manager = PluginManager()
            >>> decoder = manager.get_plugin('oscura.decoders', 'can')
            >>> frames = decoder.decode(trace)
        """
        metadata = self.load_plugin(group, name)
        return metadata.callable

    def is_loaded(self, group: str, name: str) -> bool:
        """Check if plugin is already loaded.

        Args:
            group: Entry point group.
            name: Plugin name.

        Returns:
            True if plugin is loaded.

        Example:
            >>> if manager.is_loaded('oscura.decoders', 'can'):
            ...     print("CAN decoder already loaded")
        """
        return (group, name) in self._loaded_plugins

    def list_loaded_plugins(self) -> list[PluginMetadata]:
        """List all loaded plugins.

        Returns:
            List of PluginMetadata for loaded plugins.

        Example:
            >>> loaded = manager.list_loaded_plugins()
            >>> for plugin in loaded:
            ...     print(f"{plugin.name} v{plugin.version}")
        """
        return list(self._loaded_plugins.values())

    def unload_plugin(self, group: str, name: str) -> None:
        """Unload a plugin from cache.

        Args:
            group: Entry point group.
            name: Plugin name.

        Example:
            >>> manager.unload_plugin('oscura.decoders', 'can')
        """
        plugin_key = (group, name)
        if plugin_key in self._loaded_plugins:
            del self._loaded_plugins[plugin_key]
        if plugin_key in self._failed_plugins:
            del self._failed_plugins[plugin_key]

    def _find_entry_point(self, group: str, name: str) -> Any | None:
        """Find entry point by group and name.

        Args:
            group: Entry point group.
            name: Entry point name.

        Returns:
            Entry point object or None if not found.
        """
        try:
            # Python 3.10+ API
            eps = importlib.metadata.entry_points(group=group)
            for ep in eps:
                if ep.name == name:
                    return ep
        except (AttributeError, TypeError):
            # Fallback for Python 3.9 and earlier
            try:
                eps = importlib.metadata.entry_points().get(group, [])  # type: ignore[attr-defined]
                for ep in eps:
                    if ep.name == name:
                        return ep
            except Exception:
                pass

        return None


# Global plugin manager instance
_manager = PluginManager()


def load_plugin(group: str, name: str) -> PluginMetadata:
    """Load a plugin from the global plugin manager.

    Convenience function for loading plugins without accessing the manager
    directly.

    Args:
        group: Entry point group.
        name: Plugin name.

    Returns:
        PluginMetadata with loaded plugin.

    Example:
        >>> import oscura as osc
        >>> plugin = osc.load_plugin('oscura.decoders', 'flexray')
        >>> print(f"Loaded {plugin.name} v{plugin.version}")

    References:
        API-007: Plugin Architecture
    """
    return _manager.load_plugin(group, name)


def list_plugins(group: str | None = None) -> dict[str, list[str]]:
    """List available plugins.

    Args:
        group: Specific group to list. If None, lists all groups.

    Returns:
        Dictionary mapping group names to plugin names.

    Example:
        >>> import oscura as osc
        >>> plugins = osc.list_plugins()
        >>> print(f"Available decoders: {plugins['oscura.decoders']}")

    References:
        API-007: Plugin Architecture
    """
    return _manager.discover_plugins(group)


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance.

    Returns:
        Global PluginManager instance.

    Example:
        >>> manager = osc.get_plugin_manager()
        >>> loaded = manager.list_loaded_plugins()
    """
    return _manager


__all__ = [
    "PluginError",
    "PluginManager",
    "PluginMetadata",
    "get_plugin_manager",
    "list_plugins",
    "load_plugin",
]
