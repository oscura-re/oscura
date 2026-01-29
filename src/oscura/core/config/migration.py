"""Schema migration system for Oscura configuration files.

This module provides schema migration functionality to automatically upgrade
configuration files between schema versions while preserving user data.


Example:
    >>> from oscura.core.config.migration import migrate_config, register_migration
    >>> # Register a migration function
    >>> def migrate_1_0_to_1_1(config: dict) -> dict:
    ...     config['new_field'] = 'default_value'
    ...     return config
    >>> register_migration("1.0.0", "1.1.0", migrate_1_0_to_1_1)
    >>> # Migrate config to latest version
    >>> old_config = {"version": "1.0.0", "name": "test"}
    >>> new_config = migrate_config(old_config)
    >>> print(new_config["version"])
    1.1.0
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from oscura.core.exceptions import ConfigurationError

# Type alias for migration functions
MigrationFunction = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Migration:
    """Schema migration definition.

    Attributes:
        from_version: Source schema version (semver).
        to_version: Target schema version (semver).
        migrate_fn: Function to perform migration.
        description: Human-readable description of changes.
    """

    from_version: str
    to_version: str
    migrate_fn: MigrationFunction
    description: str = ""

    def __post_init__(self) -> None:
        """Validate migration after initialization."""
        if not self.from_version:
            raise ValueError("from_version cannot be empty")
        if not self.to_version:
            raise ValueError("to_version cannot be empty")
        if not callable(self.migrate_fn):
            raise ValueError("migrate_fn must be callable")


class SchemaMigration:
    """Schema migration manager with version tracking.

    Manages registration of migration functions and execution of migration
    paths from older schema versions to newer ones.

    Supports forward migrations only (not backward) to maintain data integrity.
    Preserves unknown keys during migration to avoid data loss.

    Example:
        >>> migration = SchemaMigration()
        >>> migration.register_migration("1.0.0", "1.1.0", upgrade_fn)
        >>> config = {"version": "1.0.0", "data": "value"}
        >>> migrated = migration.migrate_config(config, "1.1.0")
        >>> print(migrated["version"])
        1.1.0
    """

    def __init__(self) -> None:
        """Initialize empty migration registry."""
        # Map of (from_version, to_version) -> Migration
        self._migrations: dict[tuple[str, str], Migration] = {}
        # Map of from_version -> list of to_versions for path finding
        self._version_graph: dict[str, list[str]] = {}

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migrate_fn: MigrationFunction,
        *,
        description: str = "",
    ) -> None:
        """Register a migration function.

        Args:
            from_version: Source schema version (semver).
            to_version: Target schema version (semver).
            migrate_fn: Function that takes config dict and returns migrated dict.
            description: Human-readable description of migration.

        Raises:
            ValueError: If migration already registered for this version pair.

        Example:
            >>> migration = SchemaMigration()
            >>> def upgrade(cfg): return {**cfg, "new_field": "default"}
            >>> migration.register_migration("1.0.0", "1.1.0", upgrade)
        """
        key = (from_version, to_version)

        if key in self._migrations:
            raise ValueError(f"Migration from {from_version} to {to_version} already registered")

        mig = Migration(
            from_version=from_version,
            to_version=to_version,
            migrate_fn=migrate_fn,
            description=description,
        )

        self._migrations[key] = mig

        # Update graph for path finding
        if from_version not in self._version_graph:
            self._version_graph[from_version] = []
        self._version_graph[from_version].append(to_version)

    def migrate_config(
        self,
        config: dict[str, Any],
        target_version: str | None = None,
    ) -> dict[str, Any]:
        """Migrate config to target version.

        Automatically finds migration path and applies migrations in sequence.
        Preserves unknown keys during migration.

        Args:
            config: Configuration dictionary to migrate.
            target_version: Target schema version. If None, migrate to latest.

        Returns:
            Migrated configuration dictionary.

        Raises:
            ConfigurationError: If migration path not found or migration fails.

        Example:
            >>> migration = SchemaMigration()
            >>> config = {"version": "1.0.0", "name": "test"}
            >>> migrated = migration.migrate_config(config, "2.0.0")
        """
        # Make a deep copy to avoid mutating input
        result = copy.deepcopy(config)

        # Get current version
        current_version = self.get_config_version(result)

        # If no version field, add default
        if current_version is None:
            result["version"] = "1.0.0"
            current_version = "1.0.0"

        # If no target specified, use latest available
        if target_version is None:
            target_version = self._get_latest_version(current_version)
            if target_version is None:
                # No migrations available, return as-is
                return result

        # Already at target version
        if current_version == target_version:
            return result

        # Find migration path
        path = self._find_migration_path(current_version, target_version)

        if path is None:
            available = self.list_migrations()
            raise ConfigurationError(
                f"No migration path from {current_version} to {target_version}",
                details=f"Available migrations: {available}",
                fix_hint=f"Register migrations to connect {current_version} to {target_version}",
            )

        # Apply migrations in sequence
        for from_ver, to_ver in path:
            migration = self._migrations[(from_ver, to_ver)]

            try:
                result = migration.migrate_fn(result)
                # Update version field
                result["version"] = to_ver
            except Exception as e:
                raise ConfigurationError(
                    f"Migration from {from_ver} to {to_ver} failed",
                    details=str(e),
                    fix_hint="Check migration function implementation",
                ) from e

        return result

    def get_config_version(self, config: dict[str, Any]) -> str | None:
        """Extract version from config.

        Args:
            config: Configuration dictionary.

        Returns:
            Version string or None if not present.

        Example:
            >>> migration = SchemaMigration()
            >>> config = {"version": "1.2.3", "data": "value"}
            >>> migration.get_config_version(config)
            '1.2.3'
        """
        return config.get("version")

    def list_migrations(self) -> list[tuple[str, str]]:
        """List available migrations.

        Returns:
            List of (from_version, to_version) tuples.

        Example:
            >>> migration = SchemaMigration()
            >>> migration.register_migration("1.0.0", "1.1.0", lambda c: c)
            >>> migration.list_migrations()
            [('1.0.0', '1.1.0')]
        """
        return sorted(self._migrations.keys())

    def has_migration(self, from_version: str, to_version: str) -> bool:
        """Check if migration exists.

        Args:
            from_version: Source version.
            to_version: Target version.

        Returns:
            True if migration exists.
        """
        return (from_version, to_version) in self._migrations

    def _find_migration_path(
        self,
        from_version: str,
        to_version: str,
    ) -> list[tuple[str, str]] | None:
        """Find shortest migration path using BFS.

        Args:
            from_version: Source version.
            to_version: Target version.

        Returns:
            List of (from, to) version pairs representing migration path,
            or None if no path exists.
        """
        if from_version == to_version:
            return []

        # BFS to find shortest path
        queue: list[tuple[str, list[tuple[str, str]]]] = [(from_version, [])]
        visited = {from_version}

        while queue:
            current, path = queue.pop(0)

            # Get all possible next versions from current
            if current in self._version_graph:
                for next_version in self._version_graph[current]:
                    if next_version in visited:
                        continue

                    new_path = [*path, (current, next_version)]

                    if next_version == to_version:
                        return new_path

                    visited.add(next_version)
                    queue.append((next_version, new_path))

        return None

    def _get_latest_version(self, from_version: str) -> str | None:
        """Get latest version reachable from given version.

        Args:
            from_version: Starting version.

        Returns:
            Latest version string or None if no migrations available.
        """
        if from_version not in self._version_graph:
            return None

        # Find all reachable versions
        reachable = set()
        queue = [from_version]
        visited = {from_version}

        while queue:
            current = queue.pop(0)

            if current in self._version_graph:
                for next_version in self._version_graph[current]:
                    if next_version not in visited:
                        visited.add(next_version)
                        reachable.add(next_version)
                        queue.append(next_version)

        if not reachable:
            return None

        # Sort versions and return latest (simple lexicographic sort)
        # For proper semver comparison, could use packaging.version
        return sorted(reachable, key=_parse_version)[-1]


# Global migration registry
_global_migration: SchemaMigration | None = None


def get_migration_registry() -> SchemaMigration:
    """Get the global migration registry.

    Initializes with built-in migrations on first call.

    Returns:
        Global SchemaMigration instance.
    """
    global _global_migration

    if _global_migration is None:
        _global_migration = SchemaMigration()
        _register_builtin_migrations(_global_migration)

    return _global_migration


def register_migration(
    from_version: str,
    to_version: str,
    migrate_fn: MigrationFunction,
    *,
    description: str = "",
) -> None:
    """Register a migration with the global registry.

    Args:
        from_version: Source schema version.
        to_version: Target schema version.
        migrate_fn: Migration function.
        description: Human-readable description.
    """
    get_migration_registry().register_migration(
        from_version, to_version, migrate_fn, description=description
    )


def migrate_config(
    config: dict[str, Any],
    target_version: str | None = None,
) -> dict[str, Any]:
    """Migrate configuration to target version using global registry.

    Args:
        config: Configuration to migrate.
        target_version: Target version or None for latest.

    Returns:
        Migrated configuration.
    """
    return get_migration_registry().migrate_config(config, target_version)


def get_config_version(config: dict[str, Any]) -> str | None:
    """Get version from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Version string or None.
    """
    return get_migration_registry().get_config_version(config)


def list_migrations() -> list[tuple[str, str]]:
    """List all registered migrations.

    Returns:
        List of (from_version, to_version) tuples.
    """
    return get_migration_registry().list_migrations()


def _parse_version(version: str) -> tuple[int, ...]:
    """Parse semver version string into tuple for comparison.

    Args:
        version: Version string (e.g., "1.2.3").

    Returns:
        Tuple of integers (e.g., (1, 2, 3)).
    """
    try:
        return tuple(int(part) for part in version.split("."))
    except (ValueError, AttributeError):
        # Return (0, 0, 0) for invalid versions
        return (0, 0, 0)


def _register_builtin_migrations(migration: SchemaMigration) -> None:
    """Register built-in migrations for core schemas.

    Args:
        migration: Migration registry to populate.
    """

    # Example migration for protocol schema (1.0.0 -> 1.1.0)
    # This is a placeholder - real migrations would be added as needed
    def _migrate_protocol_1_0_to_1_1(config: dict[str, Any]) -> dict[str, Any]:
        """Migrate protocol config from 1.0.0 to 1.1.0.

        Example migration that preserves all existing fields.

        Args:
            config: Configuration dictionary to migrate.

        Returns:
            Migrated configuration dictionary.
        """
        # Keep all existing fields (preserves unknown keys)
        # Add new optional fields with defaults if needed
        return config

    # Register when actual schema changes are needed
    # migration.register_migration(
    #     "1.0.0",
    #     "1.1.0",
    #     _migrate_protocol_1_0_to_1_1,
    #     description="Protocol schema update for new features",
    # )


__all__ = [
    "Migration",
    "MigrationFunction",
    "SchemaMigration",
    "get_config_version",
    "get_migration_registry",
    "list_migrations",
    "migrate_config",
    "register_migration",
]
