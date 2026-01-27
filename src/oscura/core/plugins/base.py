"""Plugin base classes and metadata.

This module defines the base class for Oscura plugins and
the metadata structures for plugin registration.


Example:
    >>> class MyDecoder(PluginBase):
    ...     name = "my_decoder"
    ...     version = "1.0.0"
    ...     api_version = "1.0.0"
    ...
    ...     def on_load(self):
    ...         self.register_protocol("my_protocol")
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class PluginCapability(Enum):
    """Plugin capability types."""

    PROTOCOL_DECODER = auto()
    """Protocol decoder (UART, SPI, etc.)"""
    FILE_LOADER = auto()
    """File format loader"""
    FILE_EXPORTER = auto()
    """File format exporter"""
    ANALYZER = auto()
    """Signal analyzer"""
    ALGORITHM = auto()
    """Analysis algorithm"""
    VISUALIZATION = auto()
    """Visualization component"""
    WORKFLOW = auto()
    """High-level workflow"""


@dataclass
class PluginMetadata:
    """Plugin metadata and configuration.

    Attributes:
        name: Unique plugin identifier.
        version: Plugin version (semver).
        api_version: Required Oscura API version.
        author: Plugin author.
        description: Human-readable description.
        homepage: Plugin homepage URL.
        license: License identifier (SPDX).
        capabilities: List of plugin capabilities.
        dependencies: Required plugins and packages.
        provides: What the plugin provides (protocols, algorithms, etc.).
        path: Path to plugin (set during discovery).
        enabled: Whether plugin is enabled.
    """

    name: str
    version: str
    api_version: str = "1.0.0"
    author: str = ""
    description: str = ""
    homepage: str = ""
    license: str = ""
    capabilities: list[PluginCapability] = field(default_factory=list)
    dependencies: dict[str, str] = field(default_factory=dict)
    provides: dict[str, list[str]] = field(default_factory=dict)
    path: Path | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if not self.name:
            raise ValueError("Plugin name cannot be empty")
        if not self.version:
            raise ValueError("Plugin version cannot be empty")

    @property
    def qualified_name(self) -> str:
        """Get fully qualified plugin name with version.

        Returns:
            Name in format "name@version".
        """
        return f"{self.name}@{self.version}"

    def is_compatible_with(self, api_version: str) -> bool:
        """Check if plugin is compatible with given API version.

        Uses semver compatibility rules (major version must match).

        Args:
            api_version: Oscura API version to check against.

        Returns:
            True if compatible.
        """
        try:
            plugin_major = int(self.api_version.split(".")[0])
            target_major = int(api_version.split(".")[0])
            return plugin_major == target_major
        except (ValueError, IndexError):
            return False


class PluginBase(ABC):
    """Base class for all Oscura plugins.

    Subclass this to create a plugin. Define class attributes for
    metadata and implement lifecycle methods.

    Example:
        >>> class UartDecoder(PluginBase):
        ...     name = "uart_decoder"
        ...     version = "1.0.0"
        ...     api_version = "1.0.0"
        ...     author = "Oscura Contributors"
        ...     description = "UART protocol decoder"
        ...
        ...     def on_load(self):
        ...         self.register_protocol("uart")
        ...
        ...     def on_configure(self, config):
        ...         self.baud_rate = config.get("baud_rate", 115200)
    """

    # Required class attributes (override in subclass)
    name: str = ""
    version: str = ""
    api_version: str = "1.0.0"

    # Optional class attributes
    author: str = ""
    description: str = ""
    homepage: str = ""
    license: str = ""
    capabilities: list[PluginCapability] = []
    requires_plugins: list[tuple[str, str]] = []  # (name, version_spec)

    def __init__(self) -> None:
        """Initialize plugin instance."""
        self._metadata: PluginMetadata | None = None
        self._registered_protocols: list[str] = []
        self._registered_algorithms: list[tuple[str, str, Callable]] = []  # type: ignore[type-arg]
        self._config: dict[str, Any] = {}

    @property
    def metadata(self) -> PluginMetadata:
        """Get plugin metadata.

        Returns:
            PluginMetadata instance.
        """
        if self._metadata is None:
            self._metadata = PluginMetadata(
                name=self.name,
                version=self.version,
                api_version=self.api_version,
                author=self.author,
                description=self.description,
                homepage=self.homepage,
                license=self.license,
                capabilities=list(self.capabilities),
                dependencies=dict(self.requires_plugins),
            )
        return self._metadata

    def on_load(self) -> None:
        """Called when plugin is loaded.

        Override to register capabilities, initialize resources, etc.

        References:
            PLUG-002: Plugin Registration - lifecycle hooks
        """

    def on_configure(self, config: dict[str, Any]) -> None:
        """Called when plugin is configured.

        Override to handle configuration changes.

        Args:
            config: Plugin configuration dictionary.

        References:
            PLUG-002: Plugin Registration - lifecycle hooks
        """
        self._config = config

    def on_enable(self) -> None:
        """Called when plugin is enabled.

        Override to activate plugin functionality, start services, etc.

        References:
            PLUG-002: Plugin Registration - lifecycle hooks
        """

    def on_disable(self) -> None:
        """Called when plugin is disabled.

        Override to pause plugin functionality, stop services, etc.

        References:
            PLUG-002: Plugin Registration - lifecycle hooks
        """

    def on_unload(self) -> None:
        """Called when plugin is unloaded.

        Override to clean up resources.

        References:
            PLUG-002: Plugin Registration - lifecycle hooks
        """

    def register_protocol(self, protocol_name: str) -> None:
        """Register a protocol decoder capability.

        Args:
            protocol_name: Protocol identifier (e.g., "uart").
        """
        self._registered_protocols.append(protocol_name)
        if "protocols" not in self.metadata.provides:
            self.metadata.provides["protocols"] = []
        self.metadata.provides["protocols"].append(protocol_name)

    def register_algorithm(
        self,
        category: str,
        name: str,
        func: Callable,  # type: ignore[type-arg]
    ) -> None:
        """Register an algorithm implementation.

        Args:
            category: Algorithm category (e.g., "edge_detection").
            name: Algorithm name.
            func: Algorithm function.
        """
        self._registered_algorithms.append((category, name, func))
        if "algorithms" not in self.metadata.provides:
            self.metadata.provides["algorithms"] = []
        self.metadata.provides["algorithms"].append(f"{category}:{name}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value.
        """
        return self._config.get(key, default)


__all__ = [
    "PluginBase",
    "PluginCapability",
    "PluginMetadata",
]
