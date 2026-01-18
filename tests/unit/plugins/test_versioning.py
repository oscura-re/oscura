"""Unit tests for the plugin versioning module.

Tests version compatibility checking, migration support
between plugin versions, and multi-version compatibility layers.
"""

from unittest.mock import MagicMock

import pytest

from oscura.plugins.versioning import (
    Migration,
    MigrationManager,
    VersionCompatibilityLayer,
    VersionRange,
    get_migration_manager,
)

pytestmark = pytest.mark.unit


class TestVersionRange:
    """Tests for the VersionRange class."""

    def test_version_range_creation(self):
        """Test creating a VersionRange."""
        vr = VersionRange(spec=">=1.0.0")
        assert vr.spec == ">=1.0.0"

    def test_exact_version_match(self):
        """Test exact version matching."""
        vr = VersionRange(spec="1.2.3")

        assert vr.matches("1.2.3") is True
        assert vr.matches("1.2.4") is False
        assert vr.matches("1.2.2") is False

    def test_wildcard_matches_all(self):
        """Test wildcard version matching."""
        vr = VersionRange(spec="*")

        assert vr.matches("1.0.0") is True
        assert vr.matches("2.0.0") is True
        assert vr.matches("0.0.1") is True

    def test_greater_equal(self):
        """Test greater-than-or-equal matching."""
        vr = VersionRange(spec=">=1.5.0")

        assert vr.matches("1.5.0") is True
        assert vr.matches("1.5.1") is True
        assert vr.matches("1.6.0") is True
        assert vr.matches("2.0.0") is True
        assert vr.matches("1.4.9") is False
        assert vr.matches("0.9.0") is False

    def test_less_equal(self):
        """Test less-than-or-equal matching."""
        vr = VersionRange(spec="<=2.0.0")

        assert vr.matches("2.0.0") is True
        assert vr.matches("1.9.9") is True
        assert vr.matches("1.0.0") is True
        assert vr.matches("2.0.1") is False
        assert vr.matches("2.1.0") is False

    def test_greater_than(self):
        """Test greater-than matching."""
        vr = VersionRange(spec=">1.0.0")

        assert vr.matches("1.0.1") is True
        assert vr.matches("1.1.0") is True
        assert vr.matches("1.0.0") is False
        assert vr.matches("0.9.9") is False

    def test_less_than(self):
        """Test less-than matching."""
        vr = VersionRange(spec="<2.0.0")

        assert vr.matches("1.9.9") is True
        assert vr.matches("1.0.0") is True
        assert vr.matches("2.0.0") is False
        assert vr.matches("2.0.1") is False

    def test_caret_compatible(self):
        """Test caret (compatible) version matching."""
        vr = VersionRange(spec="^1.5.0")

        # Same major, >= minor.patch
        assert vr.matches("1.5.0") is True
        assert vr.matches("1.5.1") is True
        assert vr.matches("1.6.0") is True
        assert vr.matches("1.9.9") is True

        # Different major
        assert vr.matches("2.0.0") is False
        assert vr.matches("0.9.0") is False

        # Lower minor.patch
        assert vr.matches("1.4.9") is False

    def test_tilde_approximately(self):
        """Test tilde (approximately) version matching."""
        vr = VersionRange(spec="~1.5.0")

        # Same major.minor, >= patch
        assert vr.matches("1.5.0") is True
        assert vr.matches("1.5.1") is True
        assert vr.matches("1.5.9") is True

        # Different minor
        assert vr.matches("1.6.0") is False
        assert vr.matches("1.4.9") is False

        # Different major
        assert vr.matches("2.5.0") is False

    def test_version_with_metadata(self):
        """Test parsing versions with metadata."""
        vr = VersionRange(spec=">=1.0.0")

        # Versions with pre-release or build metadata
        assert vr.matches("1.0.0-beta") is True
        assert vr.matches("1.0.0+build123") is True
        assert vr.matches("1.0.0-beta+build") is True

    def test_invalid_version_returns_false(self):
        """Test that invalid versions return False."""
        vr = VersionRange(spec=">=1.0.0")

        assert vr.matches("invalid") is False
        assert vr.matches("1.0") is False
        assert vr.matches("1.0.0.0") is False
        assert vr.matches("") is False

    def test_parse_version_valid(self):
        """Test parsing valid version strings."""
        vr = VersionRange(spec="*")

        assert vr._parse_version("1.2.3") == (1, 2, 3)
        assert vr._parse_version("0.0.0") == (0, 0, 0)
        assert vr._parse_version("10.20.30") == (10, 20, 30)

    def test_parse_version_with_metadata(self):
        """Test parsing versions with pre-release/build metadata."""
        vr = VersionRange(spec="*")

        assert vr._parse_version("1.2.3-beta") == (1, 2, 3)
        assert vr._parse_version("1.2.3+build") == (1, 2, 3)
        assert vr._parse_version("1.2.3-alpha+build123") == (1, 2, 3)

    def test_parse_version_invalid(self):
        """Test parsing invalid version strings raises ValueError."""
        vr = VersionRange(spec="*")

        with pytest.raises(ValueError):
            vr._parse_version("invalid")

        with pytest.raises(ValueError):
            vr._parse_version("1.2")

        with pytest.raises(ValueError):
            vr._parse_version("a.b.c")


class TestMigration:
    """Tests for the Migration class."""

    def test_migration_creation(self):
        """Test creating a Migration."""

        def migrate(config):
            return config

        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=migrate,
            description="Test migration",
        )

        assert migration.from_version == "1.0.0"
        assert migration.to_version == "1.1.0"
        assert migration.description == "Test migration"

    def test_migration_apply(self):
        """Test applying a migration."""

        def migrate(config):
            config["version"] = "1.1.0"
            config["new_field"] = "added"
            return config

        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=migrate,
        )

        old_config = {"version": "1.0.0", "setting": "value"}
        new_config = migration.apply(old_config)

        assert new_config["version"] == "1.1.0"
        assert new_config["new_field"] == "added"
        assert new_config["setting"] == "value"

    def test_migration_default_description(self):
        """Test migration with default (empty) description."""
        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=lambda c: c,
        )

        assert migration.description == ""


class TestVersionCompatibilityLayer:
    """Tests for the VersionCompatibilityLayer class."""

    def test_compatibility_layer_creation(self):
        """Test creating a VersionCompatibilityLayer."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"

        layer = VersionCompatibilityLayer(mock_plugin)

        assert layer._plugin is mock_plugin
        assert layer._api_version == "1.0.0"

    def test_set_api_version(self):
        """Test setting API version."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"

        layer = VersionCompatibilityLayer(mock_plugin)
        layer.set_api_version("2.0.0")

        assert layer._api_version == "2.0.0"

    def test_register_adapter(self):
        """Test registering a method adapter."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"

        layer = VersionCompatibilityLayer(mock_plugin)

        def adapter(plugin, *args):
            return "adapted"

        layer.register_adapter("1.0.0", "process", adapter)

        assert "1.0.0:process" in layer._adapters

    def test_call_adapted_with_adapter(self):
        """Test calling method with version adapter."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"

        layer = VersionCompatibilityLayer(mock_plugin)
        layer.set_api_version("1.0.0")

        def adapter(plugin, arg1, arg2):
            return f"adapted: {arg1}, {arg2}"

        layer.register_adapter("1.0.0", "process", adapter)
        result = layer.call_adapted("process", "a", "b")

        assert result == "adapted: a, b"

    def test_call_adapted_without_adapter(self):
        """Test calling method without adapter (direct call)."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"
        mock_plugin.process.return_value = "direct result"

        layer = VersionCompatibilityLayer(mock_plugin)
        result = layer.call_adapted("process", "arg1")

        mock_plugin.process.assert_called_once_with("arg1")
        assert result == "direct result"

    def test_call_adapted_with_kwargs(self):
        """Test calling adapted method with keyword arguments."""
        mock_plugin = MagicMock()
        mock_plugin.name = "test_plugin"
        mock_plugin.process.return_value = "result"

        layer = VersionCompatibilityLayer(mock_plugin)
        result = layer.call_adapted("process", "arg1", key="value")

        mock_plugin.process.assert_called_once_with("arg1", key="value")
        assert result == "result"


class TestMigrationManager:
    """Tests for the MigrationManager class."""

    def test_migration_manager_creation(self):
        """Test creating a MigrationManager."""
        manager = MigrationManager()
        assert manager._migrations == {}

    def test_register_migration(self):
        """Test registering a migration."""
        manager = MigrationManager()

        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=lambda c: c,
        )

        manager.register_migration("test_plugin", migration)

        assert "test_plugin" in manager._migrations
        assert migration in manager._migrations["test_plugin"]

    def test_register_multiple_migrations(self):
        """Test registering multiple migrations for same plugin."""
        manager = MigrationManager()

        migration1 = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=lambda c: c,
        )
        migration2 = Migration(
            from_version="1.1.0",
            to_version="1.2.0",
            migrate_func=lambda c: c,
        )

        manager.register_migration("test_plugin", migration1)
        manager.register_migration("test_plugin", migration2)

        assert len(manager._migrations["test_plugin"]) == 2

    def test_get_migration_path_direct(self):
        """Test getting direct migration path."""
        manager = MigrationManager()

        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=lambda c: c,
        )
        manager.register_migration("plugin", migration)

        path = manager.get_migration_path("plugin", "1.0.0", "1.1.0")

        assert len(path) == 1
        assert path[0] is migration

    def test_get_migration_path_multi_step(self):
        """Test getting multi-step migration path."""
        manager = MigrationManager()

        migration1 = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=lambda c: c,
        )
        migration2 = Migration(
            from_version="1.1.0",
            to_version="1.2.0",
            migrate_func=lambda c: c,
        )
        migration3 = Migration(
            from_version="1.2.0",
            to_version="2.0.0",
            migrate_func=lambda c: c,
        )

        manager.register_migration("plugin", migration1)
        manager.register_migration("plugin", migration2)
        manager.register_migration("plugin", migration3)

        path = manager.get_migration_path("plugin", "1.0.0", "2.0.0")

        assert len(path) == 3
        assert path[0].to_version == "1.1.0"
        assert path[1].to_version == "1.2.0"
        assert path[2].to_version == "2.0.0"

    def test_get_migration_path_no_migrations(self):
        """Test getting path when no migrations registered."""
        manager = MigrationManager()

        path = manager.get_migration_path("unknown_plugin", "1.0.0", "2.0.0")

        assert path == []

    def test_get_migration_path_no_path(self):
        """Test getting path when no path exists."""
        manager = MigrationManager()

        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=lambda c: c,
        )
        manager.register_migration("plugin", migration)

        with pytest.raises(ValueError, match="No migration path"):
            manager.get_migration_path("plugin", "1.0.0", "3.0.0")

    def test_migrate_same_version(self):
        """Test migrating when source and target are same."""
        manager = MigrationManager()

        config = {"setting": "value"}
        result = manager.migrate("plugin", config, "1.0.0", "1.0.0")

        assert result is config  # Same object returned

    def test_migrate_applies_all_steps(self):
        """Test that migrate applies all migration steps."""
        manager = MigrationManager()

        def add_field_a(config):
            config["field_a"] = True
            return config

        def add_field_b(config):
            config["field_b"] = True
            return config

        migration1 = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_func=add_field_a,
        )
        migration2 = Migration(
            from_version="1.1.0",
            to_version="1.2.0",
            migrate_func=add_field_b,
        )

        manager.register_migration("plugin", migration1)
        manager.register_migration("plugin", migration2)

        config = {"original": "value"}
        result = manager.migrate("plugin", config, "1.0.0", "1.2.0")

        assert result["original"] == "value"
        assert result["field_a"] is True
        assert result["field_b"] is True


class TestGetMigrationManager:
    """Tests for the get_migration_manager function."""

    def test_get_migration_manager_returns_instance(self):
        """Test that get_migration_manager returns a MigrationManager."""
        manager = get_migration_manager()
        assert isinstance(manager, MigrationManager)

    def test_get_migration_manager_singleton(self):
        """Test that get_migration_manager returns same instance."""
        manager1 = get_migration_manager()
        manager2 = get_migration_manager()

        assert manager1 is manager2


class TestVersionRangeEdgeCases:
    """Edge case tests for version range matching."""

    def test_zero_versions(self):
        """Test matching zero versions."""
        vr = VersionRange(spec=">=0.0.0")

        assert vr.matches("0.0.0") is True
        assert vr.matches("0.0.1") is True

    def test_large_version_numbers(self):
        """Test matching large version numbers."""
        vr = VersionRange(spec=">=100.200.300")

        assert vr.matches("100.200.300") is True
        assert vr.matches("100.200.301") is True
        assert vr.matches("100.200.299") is False

    def test_spec_with_spaces(self):
        """Test spec with leading/trailing spaces in target."""
        vr = VersionRange(spec=">= 1.0.0")

        assert vr.matches("1.0.0") is True
        assert vr.matches("1.0.1") is True

    def test_caret_with_zero_major(self):
        """Test caret matching with zero major version."""
        vr = VersionRange(spec="^0.5.0")

        # Same major (0), >= minor.patch
        assert vr.matches("0.5.0") is True
        assert vr.matches("0.5.1") is True
        assert vr.matches("0.6.0") is True

        # Different major
        assert vr.matches("1.0.0") is False

    def test_tilde_with_zero_patch(self):
        """Test tilde matching with zero patch."""
        vr = VersionRange(spec="~1.5.0")

        assert vr.matches("1.5.0") is True
        assert vr.matches("1.5.100") is True
