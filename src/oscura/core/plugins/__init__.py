"""Oscura plugin system.

This package provides plugin discovery, registration, and management
for extending Oscura functionality.


Example:
    >>> from oscura.core.plugins import discover_plugins, get_plugin
    >>> plugins = discover_plugins()
    >>> for plugin in plugins:
    ...     print(f"{plugin.name} v{plugin.version}")
"""

from oscura.core.plugins.base import (
    PluginBase,
    PluginCapability,
    PluginMetadata,
)
from oscura.core.plugins.cli import (
    PluginInstaller,
    cli_disable_plugin,
    cli_enable_plugin,
    cli_install_plugin,
    cli_list_plugins,
    cli_plugin_info,
    cli_validate_plugin,
)
from oscura.core.plugins.discovery import (
    discover_plugins,
    get_plugin_paths,
    scan_directory,
)
from oscura.core.plugins.isolation import (
    IsolationManager,
    Permission,
    PermissionSet,
    PluginSandbox,
    ResourceLimits,
    get_isolation_manager,
)
from oscura.core.plugins.lifecycle import (
    DependencyGraph,
    DependencyInfo,
    PluginHandle,
    PluginLifecycleManager,
    PluginLoadError,
    PluginState,
    get_lifecycle_manager,
    set_plugin_directories,
)
from oscura.core.plugins.manager import (
    PluginManager,
    get_plugin_manager,
    reset_plugin_manager,
)
from oscura.core.plugins.registry import (
    PluginRegistry,
    get_plugin,
    get_plugin_registry,
    is_compatible,
    list_plugins,
    register_plugin,
)
from oscura.core.plugins.versioning import (
    Migration,
    MigrationManager,
    VersionCompatibilityLayer,
    VersionRange,
    get_migration_manager,
)

__all__ = [
    # Base
    "DependencyGraph",
    "DependencyInfo",
    # CLI (PLUG-007)
    "IsolationManager",
    # Isolation (PLUG-004)
    "Migration",
    "MigrationManager",
    "Permission",
    "PermissionSet",
    "PluginBase",
    "PluginCapability",
    # Lifecycle
    "PluginHandle",
    "PluginInstaller",
    "PluginLifecycleManager",
    "PluginLoadError",
    # Manager
    "PluginManager",
    "PluginMetadata",
    # Registry
    "PluginRegistry",
    "PluginSandbox",
    "PluginState",
    "ResourceLimits",
    # Versioning (PLUG-003)
    "VersionCompatibilityLayer",
    "VersionRange",
    "cli_disable_plugin",
    "cli_enable_plugin",
    "cli_install_plugin",
    "cli_list_plugins",
    "cli_plugin_info",
    "cli_validate_plugin",
    # Discovery
    "discover_plugins",
    "get_isolation_manager",
    "get_lifecycle_manager",
    "get_migration_manager",
    "get_plugin",
    "get_plugin_manager",
    "get_plugin_paths",
    "get_plugin_registry",
    "is_compatible",
    "list_plugins",
    "register_plugin",
    "reset_plugin_manager",
    "scan_directory",
    "set_plugin_directories",
]
