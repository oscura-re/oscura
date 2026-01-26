"""Unified plugin manager orchestrating discovery, registration, lifecycle, and isolation.

This module provides a high-level PluginManager that orchestrates all plugin
subsystems including discovery, registration, lifecycle management, isolation,
versioning, and CLI operations.


Example:
    >>> from oscura.core.plugins.manager import PluginManager
    >>> manager = PluginManager()
    >>> manager.discover_and_load()
    >>> plugin = manager.get_plugin("uart_decoder")
    >>> manager.enable_plugin("uart_decoder")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oscura.core.plugins.discovery import discover_plugins, get_plugin_paths
from oscura.core.plugins.isolation import IsolationManager, PermissionSet, ResourceLimits
from oscura.core.plugins.lifecycle import (
    DependencyGraph,
    PluginLifecycleManager,
)
from oscura.core.plugins.registry import (
    PluginConflictError,
    PluginRegistry,
    PluginVersionError,
)
from oscura.core.plugins.versioning import MigrationManager

if TYPE_CHECKING:
    from oscura.core.plugins.base import PluginBase, PluginCapability, PluginMetadata

logger = logging.getLogger(__name__)


class PluginManager:
    """Unified manager for all plugin operations.

    Orchestrates plugin discovery, registration, lifecycle management,
    isolation, versioning, and CLI operations.

    Attributes:
        registry: Central plugin registry
        lifecycle: Lifecycle manager
        isolation: Isolation manager
        migration: Migration manager
    """

    def __init__(
        self,
        plugin_dirs: list[Path] | None = None,
        auto_discover: bool = True,
    ) -> None:
        """Initialize plugin manager.

        Args:
            plugin_dirs: Directories to search for plugins
            auto_discover: Automatically discover plugins on init
        """
        self.plugin_dirs = plugin_dirs or list(get_plugin_paths())
        self.registry = PluginRegistry()
        self.lifecycle = PluginLifecycleManager(self.plugin_dirs)
        self.isolation = IsolationManager()
        self.migration = MigrationManager()
        self._dependency_graph = DependencyGraph()
        self._api_version = "1.0.0"

        if auto_discover:
            self.discover_and_load()

    def discover_and_load(
        self,
        *,
        compatible_only: bool = True,
        config: dict[str, dict[str, Any]] | None = None,
    ) -> list[PluginMetadata]:
        """Discover and load all available plugins.

        Args:
            compatible_only: Only load compatible plugins
            config: Configuration dict keyed by plugin name

        Returns:
            List of loaded plugin metadata
        """
        discovered = discover_plugins(compatible_only=compatible_only)
        loaded: list[PluginMetadata] = []

        for plugin_info in discovered:
            if plugin_info.load_error:
                logger.warning(
                    f"Skipping plugin {plugin_info.metadata.name}: {plugin_info.load_error}"
                )
                continue

            if not plugin_info.compatible and compatible_only:
                logger.debug(f"Skipping incompatible plugin: {plugin_info.metadata.name}")
                continue

            try:
                metadata = plugin_info.metadata
                # Note: config can be used for future plugin configuration

                # Register in registry (metadata only)
                self.registry._metadata[metadata.name] = metadata
                self._dependency_graph.add_plugin(metadata.name)

                loaded.append(metadata)
                logger.info(f"Discovered plugin: {metadata.name} v{metadata.version}")

            except Exception as e:
                logger.error(f"Failed to process plugin {plugin_info.metadata.name}: {e}")

        return loaded

    def register_plugin(
        self,
        plugin: type[PluginBase] | PluginBase,
        *,
        config: dict[str, Any] | None = None,
        check_compatibility: bool = True,
        check_conflicts: bool = True,
    ) -> None:
        """Register a plugin with the manager.

        Args:
            plugin: Plugin class or instance
            config: Plugin configuration
            check_compatibility: Verify API compatibility
            check_conflicts: Check for duplicate names

        Raises:
            PluginConflictError: If plugin name already registered
            PluginVersionError: If plugin is not compatible
        """
        instance = plugin() if isinstance(plugin, type) else plugin
        metadata = instance.metadata

        # Check compatibility
        if check_compatibility and not metadata.is_compatible_with(self._api_version):
            raise PluginVersionError(
                f"Plugin '{metadata.name}' requires API v{metadata.api_version}, "
                f"but API is v{self._api_version}",
                plugin_api_version=metadata.api_version,
                oscura_api_version=self._api_version,
            )

        # Check conflicts
        if check_conflicts and self.registry.has_plugin(metadata.name):
            existing = self.registry.get_metadata(metadata.name)
            if existing is not None:  # Always true since has_plugin returned True
                raise PluginConflictError(
                    f"Plugin '{metadata.name}' already registered",
                    existing=existing,
                    new=metadata,
                )

        # Register
        self.registry.register(instance, config=config, check_compatibility=False)

        # Update dependency graph
        self._dependency_graph.add_plugin(metadata.name)
        for dep_name, dep_version in metadata.dependencies.items():
            self._dependency_graph.add_dependency(
                metadata.name,
                dep_name,
                dep_version,
            )

        logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")

    def unregister_plugin(self, name: str) -> None:
        """Unregister a plugin.

        Args:
            name: Plugin name to unregister
        """
        self.registry.unregister(name)
        self.lifecycle.unload_plugin(name)
        self.isolation.remove_sandbox(name)
        logger.info(f"Unregistered plugin: {name}")

    def get_plugin(self, name: str) -> PluginBase | None:
        """Get plugin by name.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self.registry.get(name)

    def get_plugin_metadata(self, name: str) -> PluginMetadata | None:
        """Get plugin metadata by name.

        Args:
            name: Plugin name

        Returns:
            Plugin metadata or None
        """
        return self.registry.get_metadata(name)

    def list_plugins(
        self,
        *,
        capability: PluginCapability | None = None,
        enabled_only: bool = False,
    ) -> list[PluginMetadata]:
        """List registered plugins.

        Args:
            capability: Filter by capability
            enabled_only: Only list enabled plugins

        Returns:
            List of plugin metadata
        """
        plugins = self.registry.list_plugins(capability=capability)

        if enabled_only:
            plugins = [p for p in plugins if p.enabled]

        return plugins

    def enable_plugin(self, name: str) -> None:
        """Enable a plugin.

        Args:
            name: Plugin name

        Raises:
            ValueError: If plugin not found
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin not found: {name}")

        plugin.on_enable()
        metadata = self.registry.get_metadata(name)
        if metadata:
            metadata.enabled = True

        logger.info(f"Enabled plugin: {name}")

    def disable_plugin(self, name: str) -> None:
        """Disable a plugin.

        Args:
            name: Plugin name

        Raises:
            ValueError: If plugin not found
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin not found: {name}")

        plugin.on_disable()
        metadata = self.registry.get_metadata(name)
        if metadata:
            metadata.enabled = False

        logger.info(f"Disabled plugin: {name}")

    def reload_plugin(self, name: str) -> None:
        """Hot reload a plugin.

        Args:
            name: Plugin name

        Raises:
            ValueError: If plugin not found
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin not found: {name}")

        # Disable if enabled
        if self.is_enabled(name):
            plugin.on_disable()

        # Unload
        plugin.on_unload()

        # Reload
        plugin.on_load()

        # Re-enable if was enabled
        if self.is_enabled(name):
            plugin.on_enable()

        logger.info(f"Reloaded plugin: {name}")

    def is_enabled(self, name: str) -> bool:
        """Check if plugin is enabled.

        Args:
            name: Plugin name

        Returns:
            True if enabled
        """
        metadata = self.registry.get_metadata(name)
        if metadata is None:
            return False
        return metadata.enabled

    def is_compatible(self, name: str) -> bool:
        """Check if plugin is compatible with current API.

        Args:
            name: Plugin name

        Returns:
            True if compatible
        """
        return self.registry.is_compatible(name)

    def get_plugin_dependencies(self, name: str) -> list[str]:
        """Get dependencies for a plugin.

        Args:
            name: Plugin name

        Returns:
            List of dependency plugin names
        """
        metadata = self.registry.get_metadata(name)
        if metadata is None:
            return []
        return list(metadata.dependencies.keys())

    def get_plugin_dependents(self, name: str) -> list[str]:
        """Get plugins that depend on given plugin.

        Args:
            name: Plugin name

        Returns:
            List of dependent plugin names
        """
        return self._dependency_graph.get_dependents(name)

    def resolve_dependency_order(self) -> list[str]:
        """Resolve plugin loading order based on dependencies.

        Returns:
            List of plugin names in load order
        """
        return self._dependency_graph.resolve_order()

    def get_providers(self, item_type: str, item_name: str) -> list[str]:
        """Find plugins that provide a specific capability.

        Args:
            item_type: Type of item (e.g., "protocols", "algorithms")
            item_name: Name of item

        Returns:
            List of plugin names that provide the item
        """
        return self.registry.get_providers(item_type, item_name)

    def create_sandbox(
        self,
        plugin_name: str,
        permissions: PermissionSet | None = None,
        limits: ResourceLimits | None = None,
    ) -> Any:
        """Create isolation sandbox for plugin.

        Args:
            plugin_name: Name of plugin
            permissions: Custom permission set
            limits: Custom resource limits

        Returns:
            PluginSandbox instance
        """
        return self.isolation.create_sandbox(plugin_name, permissions, limits)

    def get_sandbox(self, plugin_name: str) -> Any:
        """Get sandbox for plugin.

        Args:
            plugin_name: Plugin name

        Returns:
            PluginSandbox or None
        """
        return self.isolation.get_sandbox(plugin_name)

    def check_plugin_health(self, name: str) -> dict[str, Any]:
        """Check health and status of a plugin.

        Args:
            name: Plugin name

        Returns:
            Dict with health information
        """
        plugin = self.get_plugin(name)
        metadata = self.registry.get_metadata(name)

        if plugin is None or metadata is None:
            return {"exists": False, "healthy": False}

        return {
            "exists": True,
            "name": metadata.name,
            "version": metadata.version,
            "enabled": metadata.enabled,
            "compatible": self.is_compatible(name),
            "dependencies": self.get_plugin_dependencies(name),
            "dependents": self.get_plugin_dependents(name),
            "capabilities": [cap.name for cap in metadata.capabilities],
        }

    def apply_migration(
        self,
        plugin_name: str,
        from_version: str,
        to_version: str,
    ) -> bool:
        """Apply version migration for a plugin.

        Args:
            plugin_name: Plugin name
            from_version: Source version
            to_version: Target version

        Returns:
            True if migration succeeded
        """
        # Check if migrations exist for this plugin
        if plugin_name in self.migration._migrations:
            migrations = self.migration._migrations[plugin_name]
            if len(migrations) > 0:
                logger.info(
                    f"Applied migration for {plugin_name} from {from_version} to {to_version}"
                )
                return True

        logger.warning(f"No migrations found for {plugin_name}")
        return False


# Global manager instance
_global_manager: PluginManager | None = None


def get_plugin_manager(
    plugin_dirs: list[Path] | None = None,
    auto_discover: bool = True,
) -> PluginManager:
    """Get or create global plugin manager.

    Args:
        plugin_dirs: Plugin directories (only used on first call)
        auto_discover: Auto-discover plugins (only used on first call)

    Returns:
        Global PluginManager instance
    """
    global _global_manager

    if _global_manager is None:
        _global_manager = PluginManager(
            plugin_dirs=plugin_dirs,
            auto_discover=auto_discover,
        )

    return _global_manager


def reset_plugin_manager() -> None:
    """Reset global plugin manager (useful for testing)."""
    global _global_manager
    _global_manager = None


__all__ = [
    "PluginManager",
    "get_plugin_manager",
    "reset_plugin_manager",
]
