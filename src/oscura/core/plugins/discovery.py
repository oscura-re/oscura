"""Plugin discovery and scanning.

This module provides plugin discovery from filesystem directories
and Python entry points.


Example:
    >>> from oscura.core.plugins.discovery import discover_plugins
    >>> plugins = discover_plugins()
    >>> for plugin in plugins:
    ...     print(f"Found: {plugin.name} v{plugin.version}")
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from oscura.core.plugins.base import PluginBase, PluginMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator

# Try to import yaml for plugin.yaml parsing
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Oscura API version for compatibility checking
OSCURA_API_VERSION = "1.0.0"


@dataclass
class DiscoveredPlugin:
    """Information about a discovered plugin.

    Attributes:
        metadata: Plugin metadata.
        path: Path to plugin directory or module.
        entry_point: Entry point name (if from entry points).
        compatible: Whether plugin is compatible with current API.
        load_error: Error message if plugin failed to load.
    """

    metadata: PluginMetadata
    path: Path | None = None
    entry_point: str | None = None
    compatible: bool = True
    load_error: str | None = None


def get_plugin_paths() -> list[Path]:
    """Get list of plugin search directories.

    Returns paths in priority order:
    1. Project plugins: ./plugins/
    2. User plugins: ~/.oscura/plugins/
    3. System plugins: /usr/lib/oscura/plugins/

    Returns:
        List of plugin directory paths.
    """
    paths: list[Path] = []

    # Project plugins (current directory)
    project_plugins = Path.cwd() / "plugins"
    if project_plugins.exists():
        paths.append(project_plugins)

    # User plugins
    user_plugins = Path.home() / ".oscura" / "plugins"
    paths.append(user_plugins)  # Include even if doesn't exist

    # XDG config home
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config:
        xdg_plugins = Path(xdg_config) / "oscura" / "plugins"
        paths.append(xdg_plugins)

    # System plugins (Linux)
    system_plugins = Path("/usr/lib/oscura/plugins")
    if system_plugins.exists():
        paths.append(system_plugins)

    # Local lib (Linux)
    local_plugins = Path("/usr/local/lib/oscura/plugins")
    if local_plugins.exists():
        paths.append(local_plugins)

    return paths


def discover_plugins(
    *,
    compatible_only: bool = False,
    include_disabled: bool = False,
) -> list[DiscoveredPlugin]:
    """Discover all available plugins.

    Scans plugin directories and Python entry points for plugins.

    Args:
        compatible_only: If True, only return compatible plugins.
        include_disabled: If True, include disabled plugins.

    Returns:
        List of discovered plugins.

    Example:
        >>> plugins = discover_plugins()
        >>> print(f"Found {len(plugins)} plugins")
    """
    plugins: list[DiscoveredPlugin] = []
    seen_names: set[str] = set()

    # Scan plugin directories
    for plugin_dir in get_plugin_paths():
        if plugin_dir.exists() and plugin_dir.is_dir():
            for plugin in scan_directory(plugin_dir):
                if plugin.metadata.name not in seen_names:
                    plugins.append(plugin)
                    seen_names.add(plugin.metadata.name)

    # Scan entry points
    for plugin in scan_entry_points():
        if plugin.metadata.name not in seen_names:
            plugins.append(plugin)
            seen_names.add(plugin.metadata.name)

    # Filter by compatibility
    if compatible_only:
        plugins = [p for p in plugins if p.compatible]

    # Filter disabled
    if not include_disabled:
        plugins = [p for p in plugins if p.metadata.enabled]

    return plugins


def scan_directory(directory: Path) -> Iterator[DiscoveredPlugin]:
    """Scan a directory for plugins.

    Each subdirectory with a plugin.yaml or Python package
    is considered a potential plugin.

    Args:
        directory: Directory to scan.

    Yields:
        DiscoveredPlugin for each found plugin.
    """
    if not directory.exists():
        return

    for item in directory.iterdir():
        if item.is_dir():
            # Check for plugin.yaml
            plugin_yaml = item / "plugin.yaml"
            plugin_yml = item / "plugin.yml"

            if plugin_yaml.exists():
                plugin = _load_plugin_from_yaml(plugin_yaml)
                if plugin:
                    yield plugin
            elif plugin_yml.exists():
                plugin = _load_plugin_from_yaml(plugin_yml)
                if plugin:
                    yield plugin

            # Check for Python package with __init__.py
            init_py = item / "__init__.py"
            if init_py.exists():
                plugin = _load_plugin_from_module(item)
                if plugin:
                    yield plugin


def scan_entry_points() -> Iterator[DiscoveredPlugin]:
    """Scan Python entry points for plugins.

    Looks for entry points in the "oscura.plugins" group.

    Yields:
        DiscoveredPlugin for each found entry point.
    """
    try:
        # Python 3.10+ has entry_points with group filtering
        if hasattr(importlib.metadata, "entry_points"):
            eps = importlib.metadata.entry_points()

            # Handle different API versions
            if hasattr(eps, "select"):
                # Python 3.10+
                plugins_eps = eps.select(group="oscura.plugins")
            elif hasattr(eps, "get"):
                # Python 3.9
                plugins_eps = eps.get("oscura.plugins", [])
            else:
                # Python 3.8 style (dict)
                plugins_eps = eps.get("oscura.plugins", [])  # type: ignore[attr-defined]

            for ep in plugins_eps:
                try:
                    plugin_class = ep.load()

                    if isinstance(plugin_class, type) and issubclass(plugin_class, PluginBase):
                        instance = plugin_class()
                        metadata = instance.metadata

                        compatible = metadata.is_compatible_with(OSCURA_API_VERSION)

                        yield DiscoveredPlugin(
                            metadata=metadata,
                            entry_point=ep.name,
                            compatible=compatible,
                        )
                except Exception as e:
                    # Create placeholder for failed load
                    yield DiscoveredPlugin(
                        metadata=PluginMetadata(
                            name=ep.name,
                            version="0.0.0",
                            description="Failed to load",
                        ),
                        entry_point=ep.name,
                        compatible=False,
                        load_error=str(e),
                    )

    except Exception:
        # Entry points not available or error
        pass


def _load_plugin_from_yaml(yaml_path: Path) -> DiscoveredPlugin | None:
    """Load plugin metadata from YAML file.

    Args:
        yaml_path: Path to plugin.yaml file.

    Returns:
        DiscoveredPlugin or None if load fails.
    """
    if not YAML_AVAILABLE:
        return None

    try:
        data = _read_yaml_file(yaml_path)
        if not isinstance(data, dict):
            return None

        metadata = _build_plugin_metadata(yaml_path, data)
        compatible = metadata.is_compatible_with(OSCURA_API_VERSION)

        return DiscoveredPlugin(
            metadata=metadata,
            path=yaml_path.parent,
            compatible=compatible,
        )

    except Exception as e:
        return _create_failed_plugin(yaml_path.parent, str(e))


def _read_yaml_file(yaml_path: Path) -> Any:
    """Read and parse YAML file.

    Args:
        yaml_path: Path to YAML file.

    Returns:
        Parsed YAML data.
    """
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_plugin_metadata(yaml_path: Path, data: dict[str, Any]) -> PluginMetadata:
    """Build plugin metadata from YAML data.

    Args:
        yaml_path: Path to plugin.yaml file.
        data: Parsed YAML data.

    Returns:
        PluginMetadata instance.
    """
    metadata = PluginMetadata(
        name=data.get("name", yaml_path.parent.name),
        version=data.get("version", "0.0.0"),
        api_version=data.get("api_version", "1.0.0"),
        author=data.get("author", ""),
        description=data.get("description", ""),
        homepage=data.get("homepage", ""),
        license=data.get("license", ""),
        path=yaml_path.parent,
        enabled=data.get("enabled", True),
    )

    _parse_plugin_dependencies(metadata, data)
    _parse_plugin_provides(metadata, data)

    return metadata


def _parse_plugin_dependencies(metadata: PluginMetadata, data: dict[str, Any]) -> None:
    """Parse plugin dependencies from YAML data.

    Args:
        metadata: PluginMetadata to update.
        data: Parsed YAML data.
    """
    if "dependencies" not in data:
        return

    deps = data["dependencies"]
    if not isinstance(deps, list):
        return

    for dep in deps:
        if not isinstance(dep, dict):
            continue

        if "plugin" in dep:
            metadata.dependencies[dep["plugin"]] = dep.get("version", "*")
        elif "package" in dep:
            metadata.dependencies[dep["package"]] = dep.get("version", "*")


def _parse_plugin_provides(metadata: PluginMetadata, data: dict[str, Any]) -> None:
    """Parse plugin provides from YAML data.

    Args:
        metadata: PluginMetadata to update.
        data: Parsed YAML data.
    """
    if "provides" not in data:
        return

    provides = data["provides"]
    if not isinstance(provides, list):
        return

    for item in provides:
        if not isinstance(item, dict):
            continue

        for key, value in item.items():
            if key not in metadata.provides:
                metadata.provides[key] = []
            metadata.provides[key].append(value)


def _load_plugin_from_module(module_path: Path) -> DiscoveredPlugin | None:
    """Load plugin from Python module.

    Args:
        module_path: Path to Python package directory.

    Returns:
        DiscoveredPlugin or None if load fails.
    """
    try:
        parent = str(module_path.parent)
        added_path = parent not in sys.path

        if added_path:
            sys.path.insert(0, parent)

        try:
            module = _import_plugin_module(module_path)
            if module is None:
                return None

            plugin_class = _find_plugin_class(module)
            if plugin_class is None:
                return None

            return _create_discovered_plugin(plugin_class, module_path)

        finally:
            if added_path:
                sys.path.remove(parent)

    except Exception as e:
        return _create_failed_plugin(module_path, str(e))


def _import_plugin_module(module_path: Path) -> Any:
    """Import plugin module from path.

    Args:
        module_path: Path to plugin package.

    Returns:
        Imported module or None.
    """
    module_name = module_path.name
    spec = importlib.util.spec_from_file_location(module_name, module_path / "__init__.py")

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_plugin_class(module: Any) -> type | None:
    """Find plugin class in module.

    Args:
        module: Imported module to search.

    Returns:
        Plugin class or None.
    """
    # Check explicit names first
    if hasattr(module, "Plugin"):
        plugin_attr = module.Plugin
        if isinstance(plugin_attr, type):
            return plugin_attr
    if hasattr(module, "plugin"):
        plugin_attr = module.plugin
        if isinstance(plugin_attr, type):
            return plugin_attr

    # Search for PluginBase subclass
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and issubclass(attr, PluginBase) and attr is not PluginBase:
            return attr

    return None


def _create_discovered_plugin(plugin_class: type, module_path: Path) -> DiscoveredPlugin:
    """Create DiscoveredPlugin from plugin class.

    Args:
        plugin_class: Plugin class to instantiate.
        module_path: Path to plugin module.

    Returns:
        DiscoveredPlugin instance.
    """
    instance = plugin_class()
    metadata = instance.metadata
    metadata.path = module_path
    compatible = metadata.is_compatible_with(OSCURA_API_VERSION)

    return DiscoveredPlugin(metadata=metadata, path=module_path, compatible=compatible)


def _create_failed_plugin(module_path: Path, error: str) -> DiscoveredPlugin:
    """Create DiscoveredPlugin for failed load.

    Args:
        module_path: Path to plugin.
        error: Error message.

    Returns:
        DiscoveredPlugin with error.
    """
    return DiscoveredPlugin(
        metadata=PluginMetadata(name=module_path.name, version="0.0.0", path=module_path),
        path=module_path,
        compatible=False,
        load_error=error,
    )


__all__ = [
    "OSCURA_API_VERSION",
    "DiscoveredPlugin",
    "discover_plugins",
    "get_plugin_paths",
    "scan_directory",
    "scan_entry_points",
]
