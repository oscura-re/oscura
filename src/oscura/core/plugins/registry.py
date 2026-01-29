"""Plugin registry and management.

This module provides the central plugin registry for loading,
registering, and accessing plugins.


Example:
    >>> from oscura.core.plugins.registry import register_plugin, get_plugin
    >>> register_plugin(MyDecoder)
    >>> decoder = get_plugin("my_decoder")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from oscura.core.plugins.discovery import (
    OSCURA_API_VERSION,
    DiscoveredPlugin,
    discover_plugins,
)

if TYPE_CHECKING:
    from oscura.core.plugins.base import PluginBase, PluginCapability, PluginMetadata

logger = logging.getLogger(__name__)


class PluginConflictError(Exception):
    """Plugin registration conflict.

    Raised when registering a plugin with a name that already exists.

    Attributes:
        existing: Metadata of existing plugin.
        new: Metadata of new plugin.
    """

    def __init__(
        self,
        message: str,
        existing: PluginMetadata,
        new: PluginMetadata,
    ) -> None:
        super().__init__(message)
        self.existing = existing
        self.new = new


class PluginVersionError(Exception):
    """Plugin version incompatibility.

    Raised when a plugin is not compatible with the current API.

    Attributes:
        plugin_api_version: Plugin's required API version.
        oscura_api_version: Current Oscura API version.
    """

    def __init__(
        self,
        message: str,
        plugin_api_version: str,
        oscura_api_version: str,
    ) -> None:
        super().__init__(message)
        self.plugin_api_version = plugin_api_version
        self.oscura_api_version = oscura_api_version


class PluginDependencyError(Exception):
    """Plugin dependency not satisfied.

    Attributes:
        plugin: Plugin name that has unmet dependency.
        dependency: Missing dependency name.
        required_version: Required dependency version.
    """

    def __init__(
        self,
        message: str,
        plugin: str,
        dependency: str,
        required_version: str,
    ) -> None:
        super().__init__(message)
        self.plugin = plugin
        self.dependency = dependency
        self.required_version = required_version


class PluginRegistry:
    """Central registry for plugins.

    Manages plugin registration, loading, and lookup.

    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(MyDecoder)
        >>> plugin = registry.get("my_decoder")
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._plugins: dict[str, PluginBase] = {}
        self._metadata: dict[str, PluginMetadata] = {}
        self._by_capability: dict[PluginCapability, list[str]] = {}
        self._discovered: list[DiscoveredPlugin] = []

    def register(
        self,
        plugin: type[PluginBase] | PluginBase,
        *,
        check_compatibility: bool = True,
        check_conflicts: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Register a plugin with the registry.

        Args:
            plugin: Plugin class or instance to register.
            check_compatibility: Verify API compatibility.
            check_conflicts: Check for duplicate names.
            config: Optional plugin configuration.

        Raises:
            PluginConflictError: If plugin name already registered.
            PluginVersionError: If plugin is not compatible.
        """
        # Get or create instance
        instance = plugin() if isinstance(plugin, type) else plugin

        metadata = instance.metadata

        # Check compatibility
        if check_compatibility and not metadata.is_compatible_with(OSCURA_API_VERSION):
            raise PluginVersionError(
                f"Plugin '{metadata.name}' requires API v{metadata.api_version}, "
                f"but Oscura API is v{OSCURA_API_VERSION}",
                plugin_api_version=metadata.api_version,
                oscura_api_version=OSCURA_API_VERSION,
            )

        # Check conflicts (PLUG-002: conflict detection for duplicate plugins)
        if check_conflicts and metadata.name in self._plugins:
            existing = self._metadata[metadata.name]

            # Provide detailed conflict information
            conflict_msg = (
                f"Plugin '{metadata.name}' already registered:\n"
                f"  Existing: v{existing.version} at {existing.path}\n"
                f"  New: v{metadata.version}"
            )

            # Check if same version
            if existing.version == metadata.version:
                conflict_msg += " (same version)"

            raise PluginConflictError(
                conflict_msg,
                existing=existing,
                new=metadata,
            )

        # Register
        self._plugins[metadata.name] = instance
        self._metadata[metadata.name] = metadata

        # Index by capability
        for cap in metadata.capabilities:
            if cap not in self._by_capability:
                self._by_capability[cap] = []
            self._by_capability[cap].append(metadata.name)

        # Configure and load
        if config:
            instance.on_configure(config)

        instance.on_load()

        logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")

    def unregister(self, name: str) -> None:
        """Unregister a plugin.

        Args:
            name: Plugin name to unregister.
        """
        if name not in self._plugins:
            return

        instance = self._plugins[name]
        metadata = self._metadata[name]

        # Call unload hook
        instance.on_unload()

        # Remove from capability index
        for cap in metadata.capabilities:
            if cap in self._by_capability and name in self._by_capability[cap]:
                self._by_capability[cap].remove(name)

        # Remove from registry
        del self._plugins[name]
        del self._metadata[name]

        logger.info(f"Unregistered plugin: {name}")

    def get(self, name: str) -> PluginBase | None:
        """Get plugin by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin instance or None.
        """
        return self._plugins.get(name)

    def get_metadata(self, name: str) -> PluginMetadata | None:
        """Get plugin metadata by name.

        Args:
            name: Plugin name.

        Returns:
            Plugin metadata or None.
        """
        return self._metadata.get(name)

    def list_plugins(
        self,
        *,
        capability: PluginCapability | None = None,
    ) -> list[PluginMetadata]:
        """List registered plugins.

        Args:
            capability: Filter by capability.

        Returns:
            List of plugin metadata.
        """
        if capability is not None:
            names = self._by_capability.get(capability, [])
            return [self._metadata[name] for name in names]

        return list(self._metadata.values())

    def has_plugin(self, name: str) -> bool:
        """Check if plugin is registered.

        Args:
            name: Plugin name.

        Returns:
            True if registered.
        """
        return name in self._plugins

    def is_compatible(self, name: str) -> bool:
        """Check if plugin is compatible with current API.

        Args:
            name: Plugin name.

        Returns:
            True if compatible.
        """
        metadata = self._metadata.get(name)
        if metadata is None:
            return False
        return metadata.is_compatible_with(OSCURA_API_VERSION)

    def discover_and_load(
        self,
        *,
        compatible_only: bool = True,
        config: dict[str, dict[str, Any]] | None = None,
    ) -> list[PluginMetadata]:
        """Discover and load all available plugins.

        Args:
            compatible_only: Only load compatible plugins.
            config: Configuration dict keyed by plugin name.

        Returns:
            List of loaded plugin metadata.
        """
        self._discovered = discover_plugins(compatible_only=compatible_only)
        loaded: list[PluginMetadata] = []

        for discovered in self._discovered:
            if discovered.load_error:
                logger.warning(
                    f"Skipping plugin {discovered.metadata.name}: {discovered.load_error}"
                )
                continue

            if not discovered.compatible and compatible_only:
                logger.debug(f"Skipping incompatible plugin: {discovered.metadata.name}")
                continue

            try:
                if config and discovered.metadata.name in config:
                    config[discovered.metadata.name]

                # For now, just store the metadata
                # Full loading requires importing the plugin module
                self._metadata[discovered.metadata.name] = discovered.metadata
                loaded.append(discovered.metadata)

            except Exception as e:
                logger.error(f"Failed to load plugin {discovered.metadata.name}: {e}")

        return loaded

    def get_providers(self, item_type: str, item_name: str) -> list[str]:
        """Find plugins that provide a specific capability.

        Args:
            item_type: Type of item (e.g., "protocols", "algorithms").
            item_name: Name of item.

        Returns:
            List of plugin names that provide the item.
        """
        providers: list[str] = []

        for name, metadata in self._metadata.items():
            if item_type in metadata.provides:
                if item_name in metadata.provides[item_type]:
                    providers.append(name)

        return providers


# Global registry instance
_global_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry.

    Returns:
        Global PluginRegistry instance.
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = PluginRegistry()

    return _global_registry


def register_plugin(
    plugin: type[PluginBase] | PluginBase,
    *,
    config: dict[str, Any] | None = None,
) -> None:
    """Register plugin with global registry.

    Args:
        plugin: Plugin class or instance.
        config: Plugin configuration.
    """
    get_plugin_registry().register(plugin, config=config)


def get_plugin(name: str) -> PluginBase | None:
    """Get plugin from global registry.

    Args:
        name: Plugin name.

    Returns:
        Plugin instance or None.
    """
    return get_plugin_registry().get(name)


def list_plugins(
    *,
    capability: PluginCapability | None = None,
) -> list[PluginMetadata]:
    """List plugins from global registry.

    Args:
        capability: Filter by capability.

    Returns:
        List of plugin metadata.
    """
    return get_plugin_registry().list_plugins(capability=capability)


def is_compatible(name: str) -> bool:
    """Check plugin compatibility.

    Args:
        name: Plugin name.

    Returns:
        True if compatible.
    """
    return get_plugin_registry().is_compatible(name)


__all__ = [
    "PluginConflictError",
    "PluginDependencyError",
    "PluginRegistry",
    "PluginVersionError",
    "get_plugin",
    "get_plugin_registry",
    "is_compatible",
    "list_plugins",
    "register_plugin",
]
