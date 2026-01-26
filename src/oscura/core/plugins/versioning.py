"""Plugin versioning and migration support.

This module provides version compatibility checking, migration support
between plugin versions, and multi-version compatibility layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from oscura.core.plugins.base import PluginBase

logger = logging.getLogger(__name__)


@dataclass
class VersionRange:
    """Version range specification.

    Supports version range syntax:
    - "1.0.0" - exact version
    - ">=1.0.0" - greater than or equal
    - "<2.0.0" - less than
    - "^1.5.0" - compatible with (same major)
    - "~1.5.0" - approximately (same major.minor)
    - "*" - any version

    References:
        PLUG-005: Plugin Dependencies - version range support
    """

    spec: str

    def matches(self, version: str) -> bool:
        """Check if version matches this range.

        Args:
            version: Version string to check (semver format)

        Returns:
            True if version matches range

        References:
            PLUG-005: Plugin Dependencies - version range support
        """
        if self.spec == "*":
            return True

        # Parse version
        try:
            v_major, v_minor, v_patch = self._parse_version(version)
        except ValueError:
            return False

        # Handle different operators
        if self.spec.startswith(">="):
            target = self.spec[2:].strip()
            t_major, t_minor, t_patch = self._parse_version(target)
            return (v_major, v_minor, v_patch) >= (t_major, t_minor, t_patch)

        elif self.spec.startswith("<="):
            target = self.spec[2:].strip()
            t_major, t_minor, t_patch = self._parse_version(target)
            return (v_major, v_minor, v_patch) <= (t_major, t_minor, t_patch)

        elif self.spec.startswith(">"):
            target = self.spec[1:].strip()
            t_major, t_minor, t_patch = self._parse_version(target)
            return (v_major, v_minor, v_patch) > (t_major, t_minor, t_patch)

        elif self.spec.startswith("<"):
            target = self.spec[1:].strip()
            t_major, t_minor, t_patch = self._parse_version(target)
            return (v_major, v_minor, v_patch) < (t_major, t_minor, t_patch)

        elif self.spec.startswith("^"):
            # Compatible: same major version
            target = self.spec[1:].strip()
            t_major, t_minor, t_patch = self._parse_version(target)
            return v_major == t_major and (v_minor, v_patch) >= (t_minor, t_patch)

        elif self.spec.startswith("~"):
            # Approximately: same major.minor version
            target = self.spec[1:].strip()
            t_major, t_minor, t_patch = self._parse_version(target)
            return v_major == t_major and v_minor == t_minor and v_patch >= t_patch

        else:
            # Exact match
            return version == self.spec

    def _parse_version(self, version: str) -> tuple[int, int, int]:
        """Parse semver version string.

        Args:
            version: Version string (e.g., "1.2.3")

        Returns:
            Tuple of (major, minor, patch)

        Raises:
            ValueError: If version format is invalid
        """
        # Handle version with metadata (e.g., "1.2.3-beta+build")
        version = version.split("-")[0].split("+")[0]

        parts = version.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version}")

        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
            return (major, minor, patch)
        except ValueError as e:
            raise ValueError(f"Invalid version format: {version}") from e


@dataclass
class Migration:
    """Plugin migration definition.

    Defines migration path from one version to another.

    Attributes:
        from_version: Source version
        to_version: Target version
        migrate_func: Migration function
        description: Migration description

    References:
        PLUG-003: Plugin Versioning - migration support
    """

    from_version: str
    to_version: str
    migrate_func: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ""

    def apply(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply migration to configuration.

        Args:
            config: Plugin configuration

        Returns:
            Migrated configuration

        References:
            PLUG-003: Plugin Versioning - migration support
        """
        logger.info(f"Migrating plugin config from v{self.from_version} to v{self.to_version}")
        return self.migrate_func(config)


class VersionCompatibilityLayer:
    """Multi-version compatibility layer for plugins.

    Allows plugins to support multiple API versions by adapting
    the interface based on the current Oscura API version.

    References:
        PLUG-003: Plugin Versioning - multi-version compatibility layer
    """

    def __init__(self, plugin: PluginBase) -> None:
        """Initialize compatibility layer.

        Args:
            plugin: Plugin instance to wrap
        """
        self._plugin = plugin
        self._api_version = "1.0.0"  # Default
        self._adapters: dict[str, Callable] = {}  # type: ignore[type-arg]

    def set_api_version(self, api_version: str) -> None:
        """Set target API version.

        Args:
            api_version: Oscura API version

        References:
            PLUG-003: Plugin Versioning - multi-version compatibility
        """
        self._api_version = api_version
        logger.debug(f"Set API version to {api_version} for plugin {self._plugin.name}")

    def register_adapter(
        self,
        api_version: str,
        method_name: str,
        adapter: Callable,  # type: ignore[type-arg]
    ) -> None:
        """Register method adapter for specific API version.

        Args:
            api_version: API version this adapter is for
            method_name: Method name to adapt
            adapter: Adapter function

        References:
            PLUG-003: Plugin Versioning - multi-version compatibility
        """
        key = f"{api_version}:{method_name}"
        self._adapters[key] = adapter

    def call_adapted(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call plugin method with version adaptation.

        Args:
            method_name: Method to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Method result

        References:
            PLUG-003: Plugin Versioning - multi-version compatibility
        """
        key = f"{self._api_version}:{method_name}"

        if key in self._adapters:
            # Use adapter
            adapter = self._adapters[key]
            return adapter(self._plugin, *args, **kwargs)
        else:
            # Call directly
            method = getattr(self._plugin, method_name)
            return method(*args, **kwargs)


class MigrationManager:
    """Manager for plugin configuration migrations.

    Tracks and applies migrations between plugin versions.

    References:
        PLUG-003: Plugin Versioning - migration support
    """

    def __init__(self) -> None:
        """Initialize migration manager."""
        self._migrations: dict[str, list[Migration]] = {}

    def register_migration(self, plugin_name: str, migration: Migration) -> None:
        """Register a migration for a plugin.

        Args:
            plugin_name: Plugin name
            migration: Migration definition

        References:
            PLUG-003: Plugin Versioning - migration support
        """
        if plugin_name not in self._migrations:
            self._migrations[plugin_name] = []

        self._migrations[plugin_name].append(migration)
        logger.debug(
            f"Registered migration for {plugin_name}: "
            f"v{migration.from_version} -> v{migration.to_version}"
        )

    def get_migration_path(
        self,
        plugin_name: str,
        from_version: str,
        to_version: str,
    ) -> list[Migration]:
        """Get migration path between versions.

        Args:
            plugin_name: Plugin name
            from_version: Source version
            to_version: Target version

        Returns:
            List of migrations in order

        Raises:
            ValueError: If no migration path exists

        References:
            PLUG-003: Plugin Versioning - migration support
        """
        if plugin_name not in self._migrations:
            return []

        # Simple linear path (could be enhanced with graph search)
        migrations = self._migrations[plugin_name]
        path: list[Migration] = []

        current = from_version
        while current != to_version:
            # Find next migration
            next_migration = None
            for migration in migrations:
                if migration.from_version == current:
                    next_migration = migration
                    break

            if next_migration is None:
                raise ValueError(f"No migration path from v{from_version} to v{to_version}")

            path.append(next_migration)
            current = next_migration.to_version

        return path

    def migrate(
        self,
        plugin_name: str,
        config: dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> dict[str, Any]:
        """Migrate configuration between versions.

        Args:
            plugin_name: Plugin name
            config: Current configuration
            from_version: Source version
            to_version: Target version

        Returns:
            Migrated configuration

        References:
            PLUG-003: Plugin Versioning - migration support
        """
        if from_version == to_version:
            return config

        path = self.get_migration_path(plugin_name, from_version, to_version)

        result = config
        for migration in path:
            result = migration.apply(result)

        return result


# Global migration manager
_migration_manager: MigrationManager | None = None


def get_migration_manager() -> MigrationManager:
    """Get global migration manager.

    Returns:
        Global MigrationManager instance
    """
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    return _migration_manager


__all__ = [
    "Migration",
    "MigrationManager",
    "VersionCompatibilityLayer",
    "VersionRange",
    "get_migration_manager",
]
