"""Plugin lifecycle management and dependency resolution.

This module provides advanced plugin lifecycle management including
dependency resolution, graceful enable/disable, lazy loading, and
hot reload capabilities.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from oscura.core.plugins.base import PluginBase, PluginMetadata

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states."""

    DISCOVERED = auto()  # Found but not loaded
    LOADING = auto()  # Currently loading
    LOADED = auto()  # Loaded but not configured
    CONFIGURED = auto()  # Configured and ready
    ENABLED = auto()  # Fully enabled
    DISABLED = auto()  # Disabled by user
    ERROR = auto()  # Load/configure error
    UNLOADING = auto()  # Currently unloading


@dataclass
class PluginLoadError:
    """Plugin load error details.

    Attributes:
        plugin_name: Name of plugin that failed
        error: Exception that occurred
        traceback: Traceback string
        stage: Stage where failure occurred
        recoverable: Whether error is recoverable
    """

    plugin_name: str
    error: Exception
    traceback: str = ""
    stage: str = "load"  # discovery, load, configure, enable
    recoverable: bool = True


@dataclass
class DependencyInfo:
    """Plugin dependency information.

    Attributes:
        name: Dependency plugin name
        version_spec: Version specification (semver)
        optional: Whether dependency is optional
        resolved: Whether dependency has been resolved
    """

    name: str
    version_spec: str = "*"
    optional: bool = False
    resolved: bool = False


@dataclass
class PluginHandle:
    """Handle for managing a plugin instance.

    Attributes:
        metadata: Plugin metadata
        instance: Plugin instance (None if not loaded)
        state: Current lifecycle state
        dependencies: Plugin dependencies
        dependents: Plugins that depend on this one
        errors: List of errors encountered
        load_time: Time taken to load (seconds)
    """

    metadata: PluginMetadata
    instance: PluginBase | None = None
    state: PluginState = PluginState.DISCOVERED
    dependencies: list[DependencyInfo] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    errors: list[PluginLoadError] = field(default_factory=list)
    load_time: float = 0.0


class DependencyGraph:
    """Dependency resolution graph for plugins.

    Resolves plugin dependencies using topological sort to ensure
    correct load order and detect cycles.

    Example:
        >>> graph = DependencyGraph()
        >>> graph.add_plugin("core")
        >>> graph.add_dependency("decoder", "core", ">=1.0.0")
        >>> order = graph.resolve_order()
        >>> print(order)  # ['core', 'decoder']

    References:
        PLUG-005: Dependency Resolution
    """

    def __init__(self) -> None:
        """Initialize empty dependency graph."""
        self._nodes: dict[str, list[DependencyInfo]] = {}
        self._in_degree: dict[str, int] = {}
        # Reverse adjacency: maps dependency -> list of dependents
        self._reverse_adj: dict[str, list[str]] = {}

    def add_plugin(self, name: str) -> None:
        """Add plugin node to graph.

        Args:
            name: Plugin name
        """
        if name not in self._nodes:
            self._nodes[name] = []
            self._in_degree[name] = 0
            self._reverse_adj[name] = []

    def add_dependency(
        self,
        plugin: str,
        depends_on: str,
        version_spec: str = "*",
        optional: bool = False,
    ) -> None:
        """Add dependency edge.

        Args:
            plugin: Plugin that has the dependency
            depends_on: Plugin being depended on
            version_spec: Version specification
            optional: Whether dependency is optional
        """
        self.add_plugin(plugin)
        self.add_plugin(depends_on)

        dep = DependencyInfo(name=depends_on, version_spec=version_spec, optional=optional)
        self._nodes[plugin].append(dep)
        self._in_degree[plugin] += 1
        # Track reverse edge: depends_on -> plugin (plugin depends on depends_on)
        self._reverse_adj[depends_on].append(plugin)

    def resolve_order(self) -> list[str]:
        """Resolve topological order for loading.

        Returns:
            List of plugin names in load order

        Raises:
            ValueError: If circular dependency detected

        References:
            PLUG-005: Dependency Resolution - circular dependency detection
        """
        # Kahn's algorithm
        in_degree = dict(self._in_degree)
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Decrement in_degree for nodes that depend on this one
            for dependent in self._reverse_adj.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            # Cycle detected - find the cycle
            remaining = set(self._nodes.keys()) - set(result)
            cycle = self._find_cycle(remaining)

            raise ValueError(f"Circular dependency detected: {' -> '.join([*cycle, cycle[0]])}")

        return result

    def _find_cycle(self, nodes: set[str]) -> list[str]:
        """Find a cycle in the dependency graph.

        Args:
            nodes: Set of nodes that may be in a cycle

        Returns:
            List of nodes forming a cycle

        References:
            PLUG-005: Dependency Resolution - circular dependency detection
        """
        visited: set[str] = set()
        rec_stack: list[str] = []

        def dfs(node: str) -> list[str] | None:
            visited.add(node)
            rec_stack.append(node)

            for dep in self._nodes.get(node, []):
                if dep.name not in nodes:
                    continue

                if dep.name not in visited:
                    cycle = dfs(dep.name)
                    if cycle:
                        return cycle
                elif dep.name in rec_stack:
                    # Found cycle
                    idx = rec_stack.index(dep.name)
                    return rec_stack[idx:]

            rec_stack.pop()
            return None

        for node in nodes:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle

        return []

    def get_dependencies(self, plugin: str) -> list[DependencyInfo]:
        """Get dependencies for a plugin.

        Args:
            plugin: Plugin name

        Returns:
            List of dependencies
        """
        return self._nodes.get(plugin, [])

    def get_dependents(self, plugin: str) -> list[str]:
        """Get plugins that depend on given plugin.

        Args:
            plugin: Plugin name

        Returns:
            List of dependent plugin names
        """
        dependents = []
        for name, deps in self._nodes.items():
            if any(d.name == plugin for d in deps):
                dependents.append(name)
        return dependents


class PluginLifecycleManager:
    """Manager for plugin lifecycle operations.

    Handles plugin loading, configuration, enabling/disabling,
    hot reload, and graceful degradation.

    Example:
        >>> manager = PluginLifecycleManager()
        >>> manager.discover_plugins()
        >>> manager.load_plugin("uart_decoder")
        >>> manager.enable_plugin("uart_decoder")

    References:
        PLUG-004: Plugin Lifecycle (enable/disable/reload)
        PLUG-005: Dependency Resolution
        PLUG-006: Graceful Degradation
        PLUG-007: Lazy Loading
        PLUG-008: Plugin Hot Reload
    """

    def __init__(self, plugin_dirs: list[Path] | None = None) -> None:
        """Initialize lifecycle manager.

        Args:
            plugin_dirs: Directories to search for plugins
        """
        self._plugin_dirs = plugin_dirs or []
        self._handles: dict[str, PluginHandle] = {}
        self._dependency_graph = DependencyGraph()
        self._lock = threading.RLock()
        self._lazy_loaders: dict[str, Callable[[], PluginBase]] = {}
        self._file_watchers: dict[str, float] = {}  # path -> mtime
        self._lifecycle_callbacks: list[Callable[[str, PluginState], None]] = []

    def discover_plugins(self) -> list[str]:
        """Discover available plugins.

        Scans plugin directories for plugin manifests and Python files.

        Returns:
            List of discovered plugin names

        References:
            PLUG-007: Lazy Loading
        """
        discovered = []

        for plugin_dir in self._plugin_dirs:
            if not plugin_dir.exists():
                continue

            for item in plugin_dir.iterdir():
                if item.is_dir() and (item / "__init__.py").exists():
                    # Package plugin
                    name = item.name
                    self._register_lazy_loader(name, item)
                    discovered.append(name)
                elif item.suffix == ".py" and not item.name.startswith("_"):
                    # Single file plugin
                    name = item.stem
                    self._register_lazy_loader(name, item)
                    discovered.append(name)

        logger.info(f"Discovered {len(discovered)} plugins")
        return discovered

    def _register_lazy_loader(self, name: str, path: Path) -> None:
        """Register lazy loader for a plugin.

        Args:
            name: Plugin name
            path: Path to plugin

        References:
            PLUG-007: Lazy Loading
        """

        def loader() -> PluginBase:
            return self._load_plugin_from_path(name, path)

        self._lazy_loaders[name] = loader

        # Create handle in DISCOVERED state
        handle = PluginHandle(
            metadata=PluginMetadata(name=name, version="0.0.0"),
            state=PluginState.DISCOVERED,
        )
        self._handles[name] = handle

        # Track file for hot reload
        if path.is_file():
            self._file_watchers[str(path)] = path.stat().st_mtime
        else:
            init_path = path / "__init__.py"
            if init_path.exists():
                self._file_watchers[str(init_path)] = init_path.stat().st_mtime

    def _load_plugin_from_path(self, name: str, path: Path) -> PluginBase:
        """Load plugin from path.

        Args:
            name: Plugin name
            path: Path to plugin

        Returns:
            Loaded plugin instance

        Raises:
            ImportError: If plugin cannot be loaded or no PluginBase subclass found
        """
        if path.is_dir():
            spec = importlib.util.spec_from_file_location(name, path / "__init__.py")
        else:
            spec = importlib.util.spec_from_file_location(name, path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin from {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

        # Find PluginBase subclass
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, PluginBase) and attr is not PluginBase:
                return attr()

        raise ImportError(f"No PluginBase subclass found in {path}")

    def load_plugin(
        self, name: str, *, lazy: bool = True, resolve_deps: bool = True
    ) -> PluginHandle:
        """Load a plugin.

        Args:
            name: Plugin name
            lazy: Use lazy loading if available
            resolve_deps: Resolve dependencies first

        Returns:
            Plugin handle

        Raises:
            Exception: If plugin loading or initialization fails.
            ValueError: If plugin not found or dependency resolution fails

        References:
            PLUG-004: Plugin Lifecycle
            PLUG-005: Dependency Resolution
            PLUG-007: Lazy Loading
        """
        with self._lock:
            if name not in self._handles:
                raise ValueError(f"Plugin '{name}' not discovered")

            handle = self._handles[name]

            if handle.state == PluginState.LOADED:
                return handle

            # Resolve dependencies first
            if resolve_deps:
                self._resolve_dependencies(name)

            handle.state = PluginState.LOADING
            self._notify_state_change(name, handle.state)

            try:
                import time

                start = time.time()

                # Use lazy loader if available
                if lazy and name in self._lazy_loaders:
                    instance = self._lazy_loaders[name]()
                else:
                    instance = self._load_plugin_from_path(name, self._get_plugin_path(name))

                handle.instance = instance
                handle.metadata = instance.metadata
                handle.load_time = time.time() - start

                # Call on_load
                instance.on_load()

                handle.state = PluginState.LOADED
                self._notify_state_change(name, handle.state)

                logger.info(
                    f"Loaded plugin '{name}' v{handle.metadata.version} in {handle.load_time:.3f}s"
                )

                return handle

            except Exception as e:
                import traceback

                error = PluginLoadError(
                    plugin_name=name,
                    error=e,
                    traceback=traceback.format_exc(),
                    stage="load",
                    recoverable=True,
                )
                handle.errors.append(error)
                handle.state = PluginState.ERROR
                self._notify_state_change(name, handle.state)

                logger.error(f"Failed to load plugin '{name}': {e}")
                raise

    def _resolve_dependencies(self, name: str) -> None:
        """Resolve and load dependencies.

        Args:
            name: Plugin name

        Raises:
            ValueError: If required dependency not found

        References:
            PLUG-005: Dependency Resolution
        """
        handle = self._handles[name]

        for dep in handle.dependencies:
            if dep.resolved:
                continue

            if dep.name not in self._handles:
                if dep.optional:
                    logger.warning(f"Optional dependency '{dep.name}' for '{name}' not found")
                    continue
                else:
                    raise ValueError(f"Required dependency '{dep.name}' for '{name}' not found")

            dep_handle = self._handles[dep.name]
            if dep_handle.state not in (PluginState.LOADED, PluginState.ENABLED):
                self.load_plugin(dep.name)

            dep.resolved = True

    def configure_plugin(self, name: str, config: dict[str, Any]) -> PluginHandle:
        """Configure a loaded plugin.

        Args:
            name: Plugin name
            config: Configuration dictionary

        Returns:
            Updated plugin handle

        Raises:
            ValueError: If plugin not found or in invalid state
            Exception: If configuration fails

        References:
            PLUG-004: Plugin Lifecycle
        """
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                raise ValueError(f"Plugin '{name}' not found")

            if handle.state not in (PluginState.LOADED, PluginState.CONFIGURED):
                raise ValueError(f"Cannot configure plugin in state {handle.state}")

            try:
                if handle.instance:
                    handle.instance.on_configure(config)
                handle.state = PluginState.CONFIGURED
                self._notify_state_change(name, handle.state)
                logger.info(f"Configured plugin '{name}'")
                return handle

            except Exception as e:
                import traceback

                error = PluginLoadError(
                    plugin_name=name,
                    error=e,
                    traceback=traceback.format_exc(),
                    stage="configure",
                )
                handle.errors.append(error)
                handle.state = PluginState.ERROR
                self._notify_state_change(name, handle.state)
                raise

    def enable_plugin(self, name: str) -> PluginHandle:
        """Enable a configured plugin.

        Args:
            name: Plugin name

        Returns:
            Updated plugin handle

        Raises:
            ValueError: If plugin not found

        References:
            PLUG-002: Plugin Registration - lifecycle hooks
            PLUG-004: Plugin Lifecycle
        """
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                raise ValueError(f"Plugin '{name}' not found")

            if handle.state == PluginState.ENABLED:
                return handle

            if handle.state == PluginState.DISCOVERED:
                self.load_plugin(name)
                handle = self._handles[name]

            if handle.state == PluginState.LOADED:
                self.configure_plugin(name, {})
                handle = self._handles[name]

            # Call on_enable hook
            if handle.instance:
                handle.instance.on_enable()

            handle.state = PluginState.ENABLED
            self._notify_state_change(name, handle.state)
            logger.info(f"Enabled plugin '{name}'")
            return handle

    def disable_plugin(self, name: str, force: bool = False) -> PluginHandle:
        """Disable a plugin.

        Args:
            name: Plugin name
            force: Force disable even if dependents exist

        Returns:
            Updated plugin handle

        Raises:
            ValueError: If dependents exist and force=False

        References:
            PLUG-002: Plugin Registration - lifecycle hooks
            PLUG-004: Plugin Lifecycle
            PLUG-006: Graceful Degradation
        """
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                raise ValueError(f"Plugin '{name}' not found")

            # Check for dependents
            dependents = [
                n
                for n, h in self._handles.items()
                if any(d.name == name for d in h.dependencies) and h.state == PluginState.ENABLED
            ]

            if dependents and not force:
                raise ValueError(f"Cannot disable '{name}': required by {dependents}")

            # Call on_disable hook
            if handle.instance:
                handle.instance.on_disable()

            handle.state = PluginState.DISABLED
            self._notify_state_change(name, handle.state)
            logger.info(f"Disabled plugin '{name}'")
            return handle

    def unload_plugin(self, name: str, force: bool = False) -> None:
        """Unload a plugin completely.

        Args:
            name: Plugin name
            force: Force unload even if enabled

        References:
            PLUG-004: Plugin Lifecycle
        """
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                return

            if handle.state == PluginState.ENABLED and not force:
                self.disable_plugin(name)

            handle.state = PluginState.UNLOADING
            self._notify_state_change(name, handle.state)

            if handle.instance:
                try:
                    handle.instance.on_unload()
                except Exception as e:
                    logger.warning(f"Error during unload of '{name}': {e}")

            handle.instance = None
            handle.state = PluginState.DISCOVERED
            self._notify_state_change(name, handle.state)
            logger.info(f"Unloaded plugin '{name}'")

    def reload_plugin(self, name: str) -> PluginHandle:
        """Hot reload a plugin.

        Args:
            name: Plugin name

        Returns:
            Updated plugin handle

        Raises:
            ValueError: If plugin not found

        References:
            PLUG-006: Plugin Hot Reload - state preservation, memory leak prevention
        """
        with self._lock:
            handle = self._handles.get(name)
            if handle is None:
                raise ValueError(f"Plugin '{name}' not found")

            was_enabled = handle.state == PluginState.ENABLED
            config = handle.instance._config if handle.instance else {}

            # Preserve plugin state for restoration
            saved_state = self._save_plugin_state(handle)

            # Unload and cleanup old references
            self.unload_plugin(name, force=True)
            self._cleanup_plugin_references(name)

            # Clear from sys.modules to force reimport
            modules_to_clear = [mod for mod in sys.modules if mod.startswith(f"{name}.")]
            for mod in modules_to_clear:
                del sys.modules[mod]
            if name in sys.modules:
                del sys.modules[name]

            # Reload
            handle = self.load_plugin(name)

            # Restore state
            self._restore_plugin_state(handle, saved_state)

            if config:
                self.configure_plugin(name, config)

            if was_enabled:
                self.enable_plugin(name)

            logger.info(f"Hot reloaded plugin '{name}'")
            return handle

    def _save_plugin_state(self, handle: PluginHandle) -> dict[str, Any]:
        """Save plugin state before reload.

        Args:
            handle: Plugin handle

        Returns:
            Saved state dictionary

        References:
            PLUG-006: Plugin Hot Reload - state preservation
        """
        state: dict[str, Any] = {
            "config": handle.instance._config if handle.instance else {},
            "registered_protocols": (
                handle.instance._registered_protocols.copy() if handle.instance else []
            ),
            "registered_algorithms": (
                handle.instance._registered_algorithms.copy() if handle.instance else []
            ),
        }
        return state

    def _restore_plugin_state(self, handle: PluginHandle, state: dict[str, Any]) -> None:
        """Restore plugin state after reload.

        Args:
            handle: Plugin handle
            state: Saved state dictionary

        References:
            PLUG-006: Plugin Hot Reload - state preservation
        """
        if handle.instance:
            handle.instance._config = state.get("config", {})
            handle.instance._registered_protocols = state.get("registered_protocols", [])
            handle.instance._registered_algorithms = state.get("registered_algorithms", [])

    def _cleanup_plugin_references(self, name: str) -> None:
        """Clean up plugin references to prevent memory leaks.

        Args:
            name: Plugin name

        References:
            PLUG-006: Plugin Hot Reload - memory leak prevention
        """
        import gc

        # Remove from lazy loaders
        if name in self._lazy_loaders:
            del self._lazy_loaders[name]

        # Force garbage collection to clean up old references
        gc.collect()

        logger.debug(f"Cleaned up references for plugin '{name}'")

    def check_for_changes(self) -> list[str]:
        """Check for plugin file changes.

        Returns:
            List of plugin names with changed files

        References:
            PLUG-008: Plugin Hot Reload
        """
        changed = []

        for path_str, old_mtime in self._file_watchers.items():
            path = Path(path_str)
            if path.exists():
                new_mtime = path.stat().st_mtime
                if new_mtime > old_mtime:
                    # Find plugin name
                    for name, handle in self._handles.items():
                        if handle.metadata.path and str(handle.metadata.path) in path_str:
                            changed.append(name)
                            break
                    self._file_watchers[path_str] = new_mtime

        return changed

    def auto_reload_changed(self) -> list[str]:
        """Automatically reload changed plugins.

        Returns:
            List of reloaded plugin names

        References:
            PLUG-008: Plugin Hot Reload
        """
        changed = self.check_for_changes()
        reloaded = []

        for name in changed:
            try:
                self.reload_plugin(name)
                reloaded.append(name)
            except Exception as e:
                logger.error(f"Failed to auto-reload '{name}': {e}")

        return reloaded

    def graceful_degradation(self, name: str) -> dict[str, Any]:
        """Handle plugin failure gracefully.

        Returns fallback options when a plugin fails.

        Args:
            name: Plugin name

        Returns:
            Dictionary with degradation options

        References:
            PLUG-006: Graceful Degradation
        """
        handle = self._handles.get(name)
        if handle is None:
            return {"status": "not_found", "alternatives": []}

        # Find alternatives
        alternatives = []
        if handle.instance:
            # Look for plugins with same capabilities
            for cap in handle.metadata.capabilities:
                for other_name, other_handle in self._handles.items():
                    if other_name != name and other_handle.state == PluginState.ENABLED:
                        if cap in other_handle.metadata.capabilities:
                            alternatives.append(other_name)

        return {
            "status": "degraded",
            "plugin": name,
            "error": str(handle.errors[-1].error) if handle.errors else None,
            "alternatives": alternatives,
            "recoverable": handle.errors[-1].recoverable if handle.errors else True,
        }

    def get_handle(self, name: str) -> PluginHandle | None:
        """Get plugin handle.

        Args:
            name: Plugin name

        Returns:
            Plugin handle or None
        """
        return self._handles.get(name)

    def get_enabled_plugins(self) -> list[str]:
        """Get list of enabled plugins.

        Returns:
            List of plugin names
        """
        return [
            name for name, handle in self._handles.items() if handle.state == PluginState.ENABLED
        ]

    def on_state_change(self, callback: Callable[[str, PluginState], None]) -> None:
        """Register state change callback.

        Args:
            callback: Function called with (plugin_name, new_state)
        """
        self._lifecycle_callbacks.append(callback)

    def _notify_state_change(self, name: str, state: PluginState) -> None:
        """Notify callbacks of state change."""
        for callback in self._lifecycle_callbacks:
            try:
                callback(name, state)
            except Exception as e:
                logger.warning(f"State change callback failed: {e}")

    def _get_plugin_path(self, name: str) -> Path:
        """Get path to plugin.

        Args:
            name: Plugin name

        Returns:
            Path to plugin

        Raises:
            ValueError: If plugin path not found
        """
        for plugin_dir in self._plugin_dirs:
            # Check for package
            pkg_path = plugin_dir / name
            if pkg_path.is_dir() and (pkg_path / "__init__.py").exists():
                return pkg_path
            # Check for single file
            file_path = plugin_dir / f"{name}.py"
            if file_path.exists():
                return file_path

        raise ValueError(f"Plugin path not found for '{name}'")


# Global lifecycle manager
_lifecycle_manager: PluginLifecycleManager | None = None


def get_lifecycle_manager() -> PluginLifecycleManager:
    """Get global lifecycle manager.

    Returns:
        Global PluginLifecycleManager instance
    """
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = PluginLifecycleManager()
    return _lifecycle_manager


def set_plugin_directories(directories: list[Path]) -> None:
    """Set plugin directories for global manager.

    Args:
        directories: List of plugin directories
    """
    global _lifecycle_manager
    _lifecycle_manager = PluginLifecycleManager(directories)


__all__ = [
    "DependencyGraph",
    "DependencyInfo",
    "PluginHandle",
    "PluginLifecycleManager",
    "PluginLoadError",
    "PluginState",
    "get_lifecycle_manager",
    "set_plugin_directories",
]
