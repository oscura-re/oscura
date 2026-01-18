"""Tests for schema migration system (CFG-016).

Requirements tested:
"""

import copy

import pytest

from oscura.config.migration import (
    Migration,
    SchemaMigration,
    get_config_version,
    get_migration_registry,
    list_migrations,
    migrate_config,
    register_migration,
)
from oscura.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


class TestMigration:
    """Test Migration dataclass."""

    def test_migration_creation(self):
        """Test creating a Migration instance."""

        def migrate_fn(config):
            return config

        migration = Migration(
            from_version="1.0.0",
            to_version="1.1.0",
            migrate_fn=migrate_fn,
            description="Test migration",
        )

        assert migration.from_version == "1.0.0"
        assert migration.to_version == "1.1.0"
        assert migration.migrate_fn is migrate_fn
        assert migration.description == "Test migration"

    def test_migration_requires_from_version(self):
        """Test that from_version is required."""
        with pytest.raises(ValueError, match="from_version cannot be empty"):
            Migration(
                from_version="",
                to_version="1.1.0",
                migrate_fn=lambda c: c,
            )

    def test_migration_requires_to_version(self):
        """Test that to_version is required."""
        with pytest.raises(ValueError, match="to_version cannot be empty"):
            Migration(
                from_version="1.0.0",
                to_version="",
                migrate_fn=lambda c: c,
            )

    def test_migration_requires_callable(self):
        """Test that migrate_fn must be callable."""
        with pytest.raises(ValueError, match="migrate_fn must be callable"):
            Migration(
                from_version="1.0.0",
                to_version="1.1.0",
                migrate_fn="not callable",  # type: ignore
            )


class TestSchemaMigration:
    """Test SchemaMigration class."""

    def test_initialization(self):
        """Test creating SchemaMigration instance."""
        migration = SchemaMigration()
        assert migration.list_migrations() == []

    def test_register_migration(self):
        """Test registering a migration."""
        migration = SchemaMigration()

        def upgrade(config):
            config["new_field"] = "added"
            return config

        migration.register_migration("1.0.0", "1.1.0", upgrade)

        migrations = migration.list_migrations()
        assert ("1.0.0", "1.1.0") in migrations

    def test_register_duplicate_migration_raises(self):
        """Test that registering duplicate migration raises error."""
        migration = SchemaMigration()

        migration.register_migration("1.0.0", "1.1.0", lambda c: c)

        with pytest.raises(
            ValueError, match=r"Migration from 1\.0\.0 to 1\.1\.0 already registered"
        ):
            migration.register_migration("1.0.0", "1.1.0", lambda c: c)

    def test_has_migration(self):
        """Test checking if migration exists."""
        migration = SchemaMigration()

        assert not migration.has_migration("1.0.0", "1.1.0")

        migration.register_migration("1.0.0", "1.1.0", lambda c: c)

        assert migration.has_migration("1.0.0", "1.1.0")
        assert not migration.has_migration("1.1.0", "1.2.0")

    def test_get_config_version(self):
        """Test extracting version from config."""
        migration = SchemaMigration()

        config = {"version": "1.2.3", "data": "value"}
        assert migration.get_config_version(config) == "1.2.3"

        config_no_version = {"data": "value"}
        assert migration.get_config_version(config_no_version) is None

    def test_migrate_config_single_step(self):
        """Test migrating config with single migration step."""
        migration = SchemaMigration()

        def upgrade_1_0_to_1_1(config):
            config["new_field"] = "default_value"
            return config

        migration.register_migration("1.0.0", "1.1.0", upgrade_1_0_to_1_1)

        config = {"version": "1.0.0", "name": "test"}
        result = migration.migrate_config(config, "1.1.0")

        assert result["version"] == "1.1.0"
        assert result["name"] == "test"  # Preserved
        assert result["new_field"] == "default_value"  # Added

    def test_migrate_config_multi_step(self):
        """Test migrating config through multiple versions."""
        migration = SchemaMigration()

        def upgrade_1_0_to_1_1(config):
            config["field_1_1"] = "added in 1.1"
            return config

        def upgrade_1_1_to_1_2(config):
            config["field_1_2"] = "added in 1.2"
            return config

        migration.register_migration("1.0.0", "1.1.0", upgrade_1_0_to_1_1)
        migration.register_migration("1.1.0", "1.2.0", upgrade_1_1_to_1_2)

        config = {"version": "1.0.0", "name": "test"}
        result = migration.migrate_config(config, "1.2.0")

        assert result["version"] == "1.2.0"
        assert result["name"] == "test"
        assert result["field_1_1"] == "added in 1.1"
        assert result["field_1_2"] == "added in 1.2"

    def test_migrate_config_preserves_unknown_keys(self):
        """Test that migration preserves unknown keys."""
        migration = SchemaMigration()

        def upgrade(config):
            config["new_field"] = "new"
            return config

        migration.register_migration("1.0.0", "1.1.0", upgrade)

        config = {
            "version": "1.0.0",
            "name": "test",
            "custom_field": "should be preserved",
            "nested": {"custom": "data"},
        }
        result = migration.migrate_config(config, "1.1.0")

        assert result["custom_field"] == "should be preserved"
        assert result["nested"]["custom"] == "data"

    def test_migrate_config_no_mutation(self):
        """Test that migration doesn't mutate input config."""
        migration = SchemaMigration()

        def upgrade(config):
            config["new_field"] = "new"
            return config

        migration.register_migration("1.0.0", "1.1.0", upgrade)

        original = {"version": "1.0.0", "name": "test"}
        original_copy = copy.deepcopy(original)

        migration.migrate_config(original, "1.1.0")

        assert original == original_copy  # Unchanged

    def test_migrate_config_adds_version_if_missing(self):
        """Test that migration adds version field if missing."""
        migration = SchemaMigration()

        migration.register_migration("1.0.0", "1.1.0", lambda c: {**c, "new": "field"})

        config = {"name": "test"}  # No version
        result = migration.migrate_config(config, "1.1.0")

        assert result["version"] == "1.1.0"
        assert result["name"] == "test"

    def test_migrate_config_already_at_target(self):
        """Test migrating config already at target version."""
        migration = SchemaMigration()

        config = {"version": "1.1.0", "name": "test"}
        result = migration.migrate_config(config, "1.1.0")

        assert result == config

    def test_migrate_config_no_path_raises(self):
        """Test that missing migration path raises error."""
        migration = SchemaMigration()

        migration.register_migration("1.0.0", "1.1.0", lambda c: c)

        config = {"version": "1.0.0", "name": "test"}

        with pytest.raises(ConfigurationError, match=r"No migration path from 1\.0\.0 to 2\.0\.0"):
            migration.migrate_config(config, "2.0.0")

    def test_migrate_config_migration_failure_raises(self):
        """Test that migration function failure raises error."""
        migration = SchemaMigration()

        def failing_migration(config):
            raise RuntimeError("Migration failed")

        migration.register_migration("1.0.0", "1.1.0", failing_migration)

        config = {"version": "1.0.0", "name": "test"}

        with pytest.raises(ConfigurationError, match=r"Migration from 1\.0\.0 to 1\.1\.0 failed"):
            migration.migrate_config(config, "1.1.0")

    def test_migrate_to_latest_version(self):
        """Test migrating to latest available version."""
        migration = SchemaMigration()

        migration.register_migration("1.0.0", "1.1.0", lambda c: {**c, "v1.1": True})
        migration.register_migration("1.1.0", "1.2.0", lambda c: {**c, "v1.2": True})

        config = {"version": "1.0.0", "name": "test"}
        result = migration.migrate_config(config)  # No target specified

        assert result["version"] == "1.2.0"
        assert result["v1.1"] is True
        assert result["v1.2"] is True

    def test_migrate_to_latest_no_migrations(self):
        """Test migrating to latest when no migrations available."""
        migration = SchemaMigration()

        config = {"version": "1.0.0", "name": "test"}
        result = migration.migrate_config(config)  # No migrations registered

        assert result == {"version": "1.0.0", "name": "test"}  # Unchanged

    def test_migration_path_finding_bfs(self):
        """Test that BFS finds shortest migration path."""
        migration = SchemaMigration()

        # Create a graph with multiple paths
        # 1.0.0 -> 1.1.0 -> 2.0.0 (2 steps)
        # 1.0.0 -> 2.0.0 (1 step, should be chosen)
        migration.register_migration("1.0.0", "1.1.0", lambda c: {**c, "long": True})
        migration.register_migration("1.1.0", "2.0.0", lambda c: {**c, "long2": True})
        migration.register_migration("1.0.0", "2.0.0", lambda c: {**c, "short": True})

        config = {"version": "1.0.0", "name": "test"}
        result = migration.migrate_config(config, "2.0.0")

        # Should take direct path
        assert result["version"] == "2.0.0"
        assert "short" in result
        assert "long" not in result  # Didn't take long path

    def test_list_migrations(self):
        """Test listing all migrations."""
        migration = SchemaMigration()

        migration.register_migration("1.0.0", "1.1.0", lambda c: c)
        migration.register_migration("1.1.0", "1.2.0", lambda c: c)
        migration.register_migration("2.0.0", "2.1.0", lambda c: c)

        migrations = migration.list_migrations()

        assert len(migrations) == 3
        assert ("1.0.0", "1.1.0") in migrations
        assert ("1.1.0", "1.2.0") in migrations
        assert ("2.0.0", "2.1.0") in migrations


class TestGlobalRegistry:
    """Test global migration registry functions."""

    def test_get_migration_registry(self):
        """Test getting global registry."""
        registry = get_migration_registry()
        assert isinstance(registry, SchemaMigration)

        # Should return same instance
        registry2 = get_migration_registry()
        assert registry is registry2

    def test_register_migration_global(self):
        """Test registering migration with global registry."""
        # Note: This modifies global state, so be careful

        def upgrade(config):
            config["global_test"] = True
            return config

        register_migration("99.0.0", "99.1.0", upgrade, description="Test migration")

        # Check it was registered
        migrations = list_migrations()
        assert ("99.0.0", "99.1.0") in migrations

    def test_migrate_config_global(self):
        """Test migrating config with global registry."""

        def upgrade(config):
            config["global_migration"] = True
            return config

        register_migration("88.0.0", "88.1.0", upgrade)

        config = {"version": "88.0.0", "name": "test"}
        result = migrate_config(config, "88.1.0")

        assert result["version"] == "88.1.0"
        assert result["global_migration"] is True

    def test_get_config_version_global(self):
        """Test getting config version via global function."""
        config = {"version": "3.2.1", "data": "value"}
        assert get_config_version(config) == "3.2.1"

    def test_list_migrations_global(self):
        """Test listing migrations via global function."""
        migrations = list_migrations()
        assert isinstance(migrations, list)


class TestComplexMigrationScenarios:
    """Test complex real-world migration scenarios."""

    def test_protocol_config_migration(self):
        """Test migrating protocol config with nested data."""
        migration = SchemaMigration()

        def upgrade_protocol_v1_to_v2(config):
            """Migrate protocol from v1 to v2 - add timing constraints."""
            if "timing" not in config:
                config["timing"] = {}
            config["timing"]["setup_time_ns"] = 10
            config["timing"]["hold_time_ns"] = 5
            return config

        migration.register_migration("1.0.0", "2.0.0", upgrade_protocol_v1_to_v2)

        old_config = {
            "version": "1.0.0",
            "name": "uart",
            "baud_rate": 115200,
            "custom_param": "should_be_preserved",
        }

        new_config = migration.migrate_config(old_config, "2.0.0")

        assert new_config["version"] == "2.0.0"
        assert new_config["name"] == "uart"
        assert new_config["baud_rate"] == 115200
        assert new_config["custom_param"] == "should_be_preserved"
        assert new_config["timing"]["setup_time_ns"] == 10
        assert new_config["timing"]["hold_time_ns"] == 5

    def test_pipeline_config_migration(self):
        """Test migrating pipeline config with step restructuring."""
        migration = SchemaMigration()

        def upgrade_pipeline_v1_to_v2(config):
            """Migrate pipeline - convert old step format to new."""
            if "steps" in config:
                for step in config["steps"]:
                    # Old format had 'options', new has 'params'
                    if "options" in step:
                        step["params"] = step.pop("options")
            return config

        migration.register_migration("1.0.0", "2.0.0", upgrade_pipeline_v1_to_v2)

        old_pipeline = {
            "version": "1.0.0",
            "name": "analysis_pipeline",
            "steps": [
                {"name": "load", "type": "loader", "options": {"format": "csv"}},
                {"name": "decode", "type": "decoder", "options": {"protocol": "uart"}},
            ],
        }

        new_pipeline = migration.migrate_config(old_pipeline, "2.0.0")

        assert new_pipeline["version"] == "2.0.0"
        assert "options" not in new_pipeline["steps"][0]
        assert new_pipeline["steps"][0]["params"]["format"] == "csv"
        assert new_pipeline["steps"][1]["params"]["protocol"] == "uart"

    def test_multi_hop_migration_chain(self):
        """Test migration through multiple intermediate versions."""
        migration = SchemaMigration()

        # Chain of 4 migrations: 1.0 -> 1.1 -> 1.2 -> 2.0 -> 2.1
        def v1_0_to_1_1(c):
            c["added_in_1_1"] = True
            return c

        def v1_1_to_1_2(c):
            c["added_in_1_2"] = True
            return c

        def v1_2_to_2_0(c):
            c["major_upgrade"] = True
            return c

        def v2_0_to_2_1(c):
            c["added_in_2_1"] = True
            return c

        migration.register_migration("1.0.0", "1.1.0", v1_0_to_1_1)
        migration.register_migration("1.1.0", "1.2.0", v1_1_to_1_2)
        migration.register_migration("1.2.0", "2.0.0", v1_2_to_2_0)
        migration.register_migration("2.0.0", "2.1.0", v2_0_to_2_1)

        config = {"version": "1.0.0", "name": "test"}
        result = migration.migrate_config(config, "2.1.0")

        assert result["version"] == "2.1.0"
        assert result["added_in_1_1"] is True
        assert result["added_in_1_2"] is True
        assert result["major_upgrade"] is True
        assert result["added_in_2_1"] is True

    def test_branching_version_tree(self):
        """Test migration with branching version tree."""
        migration = SchemaMigration()

        # Version tree:
        #     1.0.0
        #      / \
        #   1.1.0  2.0.0
        #     |
        #   1.2.0
        migration.register_migration("1.0.0", "1.1.0", lambda c: {**c, "branch": "1.x"})
        migration.register_migration("1.0.0", "2.0.0", lambda c: {**c, "branch": "2.x"})
        migration.register_migration("1.1.0", "1.2.0", lambda c: {**c, "v1.2": True})

        # Should take 1.0.0 -> 2.0.0 (direct)
        config1 = {"version": "1.0.0", "name": "test"}
        result1 = migration.migrate_config(config1, "2.0.0")
        assert result1["branch"] == "2.x"

        # Should take 1.0.0 -> 1.1.0 -> 1.2.0 (2 hops)
        config2 = {"version": "1.0.0", "name": "test"}
        result2 = migration.migrate_config(config2, "1.2.0")
        assert result2["branch"] == "1.x"
        assert result2["v1.2"] is True


class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""

    def test_migration_completes_under_100ms(self):
        """Test that migration completes in <100ms."""
        import time

        migration = SchemaMigration()

        # Create chain of 3 migrations
        migration.register_migration("1.0.0", "1.1.0", lambda c: {**c, "v1": True})
        migration.register_migration("1.1.0", "1.2.0", lambda c: {**c, "v2": True})
        migration.register_migration("1.2.0", "2.0.0", lambda c: {**c, "v3": True})

        config = {"version": "1.0.0", "name": "test", "data": list(range(100))}

        start = time.perf_counter()
        migration.migrate_config(config, "2.0.0")
        duration = time.perf_counter() - start

        assert duration < 0.1  # <100ms

    def test_migration_preserves_100_percent_user_data(self):
        """Test that migration preserves 100% of user data."""
        migration = SchemaMigration()

        def upgrade(config):
            # Only add new fields, don't modify existing
            config["new_required_field"] = "default"
            return config

        migration.register_migration("1.0.0", "1.1.0", upgrade)

        original_data = {
            "version": "1.0.0",
            "user_field_1": "value1",
            "user_field_2": 42,
            "user_nested": {"key": "value"},
            "user_list": [1, 2, 3],
        }

        result = migration.migrate_config(original_data, "1.1.0")

        # All original user data preserved
        assert result["user_field_1"] == "value1"
        assert result["user_field_2"] == 42
        assert result["user_nested"]["key"] == "value"
        assert result["user_list"] == [1, 2, 3]
        # New required field added
        assert result["new_required_field"] == "default"

    def test_empty_config_migration(self):
        """Test migrating empty config."""
        migration = SchemaMigration()

        migration.register_migration("1.0.0", "1.1.0", lambda c: {**c, "field": "val"})

        config = {}
        result = migration.migrate_config(config, "1.1.0")

        assert result["version"] == "1.1.0"
        assert result["field"] == "val"

    def test_large_config_migration(self):
        """Test migrating large configuration."""
        migration = SchemaMigration()

        def upgrade(config):
            config["migrated"] = True
            return config

        migration.register_migration("1.0.0", "1.1.0", upgrade)

        # Create large config
        large_config = {
            "version": "1.0.0",
            "data": [{"id": i, "value": f"val_{i}"} for i in range(1000)],
        }

        result = migration.migrate_config(large_config, "1.1.0")

        assert result["version"] == "1.1.0"
        assert result["migrated"] is True
        assert len(result["data"]) == 1000
