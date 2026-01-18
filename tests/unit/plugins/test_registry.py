"""Unit tests for the plugin registry module.

Tests the central plugin registry for loading, registering,
and accessing plugins.
"""

from unittest.mock import MagicMock, patch

import pytest

from oscura.plugins.registry import (
    PluginConflictError,
    PluginDependencyError,
    PluginRegistry,
    PluginVersionError,
    get_plugin,
    get_plugin_registry,
    is_compatible,
    list_plugins,
    register_plugin,
)

pytestmark = pytest.mark.unit


class MockPluginMetadata:
    """Mock plugin metadata for testing."""

    def __init__(
        self,
        name: str = "test_plugin",
        version: str = "1.0.0",
        api_version: str = "1.0.0",
        capabilities: list | None = None,
        path: str = "/path/to/plugin",
        provides: dict | None = None,
    ):
        self.name = name
        self.version = version
        self.api_version = api_version
        self.capabilities = capabilities or []
        self.path = path
        self.provides = provides or {}

    def is_compatible_with(self, api_version: str) -> bool:
        """Check compatibility with API version."""
        # Simple compatibility check for testing
        my_parts = self.api_version.split(".")
        api_parts = api_version.split(".")
        return my_parts[0] == api_parts[0]  # Same major version


class MockPlugin:
    """Mock plugin for testing."""

    def __init__(
        self,
        name: str = "test_plugin",
        version: str = "1.0.0",
        api_version: str = "1.0.0",
        capabilities: list | None = None,
        compatible: bool = True,
    ):
        self._name = name
        self._version = version
        self._api_version = api_version
        self._capabilities = capabilities or []
        self._compatible = compatible
        self._configured = False
        self._loaded = False
        self._unloaded = False

        self.metadata = MockPluginMetadata(
            name=name,
            version=version,
            api_version=api_version,
            capabilities=capabilities or [],
        )

        if not compatible:
            self.metadata.api_version = "0.1.0"  # Incompatible version

    @property
    def name(self) -> str:
        return self._name

    def on_configure(self, config: dict) -> None:
        self._configured = True

    def on_load(self) -> None:
        self._loaded = True

    def on_unload(self) -> None:
        self._unloaded = True


class TestPluginConflictError:
    """Tests for PluginConflictError exception."""

    def test_conflict_error_creation(self):
        """Test creating a PluginConflictError."""
        existing = MockPluginMetadata(name="test", version="1.0.0")
        new = MockPluginMetadata(name="test", version="1.1.0")

        error = PluginConflictError(
            "Plugin conflict",
            existing=existing,
            new=new,
        )

        assert "Plugin conflict" in str(error)
        assert error.existing is existing
        assert error.new is new


class TestPluginVersionError:
    """Tests for PluginVersionError exception."""

    def test_version_error_creation(self):
        """Test creating a PluginVersionError."""
        error = PluginVersionError(
            "Version incompatible",
            plugin_api_version="2.0.0",
            oscura_api_version="1.0.0",
        )

        assert "Version incompatible" in str(error)
        assert error.plugin_api_version == "2.0.0"
        assert error.oscura_api_version == "1.0.0"


class TestPluginDependencyError:
    """Tests for PluginDependencyError exception."""

    def test_dependency_error_creation(self):
        """Test creating a PluginDependencyError."""
        error = PluginDependencyError(
            "Missing dependency",
            plugin="my_plugin",
            dependency="base_plugin",
            required_version=">=1.0.0",
        )

        assert "Missing dependency" in str(error)
        assert error.plugin == "my_plugin"
        assert error.dependency == "base_plugin"
        assert error.required_version == ">=1.0.0"


class TestPluginRegistry:
    """Tests for PluginRegistry class."""

    def test_registry_creation(self):
        """Test creating an empty registry."""
        registry = PluginRegistry()

        assert registry._plugins == {}
        assert registry._metadata == {}
        assert registry._by_capability == {}

    def test_register_plugin(self):
        """Test registering a plugin."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="test_plugin")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        assert registry.has_plugin("test_plugin")
        assert plugin._loaded

    def test_register_plugin_class(self):
        """Test registering a plugin class (not instance)."""
        registry = PluginRegistry()

        class TestPluginClass:
            def __init__(self):
                self.metadata = MockPluginMetadata(name="class_plugin")

            def on_configure(self, config):
                pass

            def on_load(self):
                pass

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(TestPluginClass)

        assert registry.has_plugin("class_plugin")

    def test_register_with_config(self):
        """Test registering a plugin with configuration."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="configured_plugin")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin, config={"setting": "value"})

        assert plugin._configured

    def test_register_conflict_detection(self):
        """Test that duplicate registration raises PluginConflictError."""
        registry = PluginRegistry()
        plugin1 = MockPlugin(name="duplicate")
        plugin2 = MockPlugin(name="duplicate", version="1.1.0")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin1)

            with pytest.raises(PluginConflictError) as exc_info:
                registry.register(plugin2)

            assert exc_info.value.existing.name == "duplicate"
            assert exc_info.value.new.name == "duplicate"

    def test_register_skip_conflict_check(self):
        """Test registering with conflict check disabled."""
        registry = PluginRegistry()
        plugin1 = MockPlugin(name="duplicate")
        plugin2 = MockPlugin(name="duplicate", version="1.1.0")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin1)
            # Should not raise with check_conflicts=False
            registry.register(plugin2, check_conflicts=False)

        # Second plugin should replace first
        assert registry.get("duplicate") is plugin2

    def test_register_incompatible_version(self):
        """Test that incompatible version raises PluginVersionError."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="incompatible", compatible=False)

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            with pytest.raises(PluginVersionError) as exc_info:
                registry.register(plugin)

            assert exc_info.value.oscura_api_version == "1.0.0"

    def test_register_skip_compatibility_check(self):
        """Test registering with compatibility check disabled."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="incompatible", compatible=False)

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            # Should not raise with check_compatibility=False
            registry.register(plugin, check_compatibility=False)

        assert registry.has_plugin("incompatible")

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="to_remove")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        registry.unregister("to_remove")

        assert not registry.has_plugin("to_remove")
        assert plugin._unloaded

    def test_unregister_nonexistent(self):
        """Test unregistering a nonexistent plugin does nothing."""
        registry = PluginRegistry()

        # Should not raise
        registry.unregister("nonexistent")

    def test_get_plugin(self):
        """Test getting a registered plugin."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="retrievable")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        retrieved = registry.get("retrievable")
        assert retrieved is plugin

    def test_get_nonexistent_plugin(self):
        """Test getting a nonexistent plugin returns None."""
        registry = PluginRegistry()

        result = registry.get("nonexistent")
        assert result is None

    def test_get_metadata(self):
        """Test getting plugin metadata."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="with_metadata")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        metadata = registry.get_metadata("with_metadata")
        assert metadata is not None
        assert metadata.name == "with_metadata"

    def test_get_metadata_nonexistent(self):
        """Test getting metadata for nonexistent plugin."""
        registry = PluginRegistry()

        result = registry.get_metadata("nonexistent")
        assert result is None

    def test_list_plugins(self):
        """Test listing all registered plugins."""
        registry = PluginRegistry()
        plugin1 = MockPlugin(name="plugin1")
        plugin2 = MockPlugin(name="plugin2")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin1)
            registry.register(plugin2)

        plugins = registry.list_plugins()

        assert len(plugins) == 2
        names = [p.name for p in plugins]
        assert "plugin1" in names
        assert "plugin2" in names

    def test_list_plugins_by_capability(self):
        """Test listing plugins by capability."""
        registry = PluginRegistry()

        plugin1 = MockPlugin(name="decoder1", capabilities=["decode"])
        plugin2 = MockPlugin(name="decoder2", capabilities=["decode"])
        plugin3 = MockPlugin(name="analyzer", capabilities=["analyze"])

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin1)
            registry.register(plugin2)
            registry.register(plugin3)

        decoders = registry.list_plugins(capability="decode")

        assert len(decoders) == 2
        names = [p.name for p in decoders]
        assert "decoder1" in names
        assert "decoder2" in names
        assert "analyzer" not in names

    def test_has_plugin(self):
        """Test checking if plugin is registered."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="exists")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        assert registry.has_plugin("exists") is True
        assert registry.has_plugin("not_exists") is False

    def test_is_compatible(self):
        """Test checking plugin compatibility."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="compatible", api_version="1.0.0")

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)
            assert registry.is_compatible("compatible") is True

    def test_is_compatible_nonexistent(self):
        """Test compatibility check for nonexistent plugin."""
        registry = PluginRegistry()

        assert registry.is_compatible("nonexistent") is False

    def test_get_providers(self):
        """Test finding plugins that provide specific capabilities."""
        registry = PluginRegistry()

        # Create plugin with provides metadata
        plugin = MockPlugin(name="protocol_provider")
        plugin.metadata.provides = {
            "protocols": ["uart", "spi"],
            "algorithms": ["fft"],
        }

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        uart_providers = registry.get_providers("protocols", "uart")
        assert "protocol_provider" in uart_providers

        fft_providers = registry.get_providers("algorithms", "fft")
        assert "protocol_provider" in fft_providers

        i2c_providers = registry.get_providers("protocols", "i2c")
        assert "protocol_provider" not in i2c_providers


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_plugin_registry_returns_instance(self):
        """Test that get_plugin_registry returns a PluginRegistry."""
        registry = get_plugin_registry()
        assert isinstance(registry, PluginRegistry)

    def test_get_plugin_registry_singleton(self):
        """Test that get_plugin_registry returns same instance."""
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()

        assert registry1 is registry2

    @patch("oscura.plugins.registry.get_plugin_registry")
    def test_register_plugin_uses_global(self, mock_get_registry):
        """Test that register_plugin uses global registry."""
        mock_registry = MagicMock()
        mock_get_registry.return_value = mock_registry

        plugin = MockPlugin()
        register_plugin(plugin, config={"key": "value"})

        mock_registry.register.assert_called_once()

    @patch("oscura.plugins.registry.get_plugin_registry")
    def test_get_plugin_uses_global(self, mock_get_registry):
        """Test that get_plugin uses global registry."""
        mock_registry = MagicMock()
        mock_registry.get.return_value = "plugin_instance"
        mock_get_registry.return_value = mock_registry

        result = get_plugin("test")

        mock_registry.get.assert_called_once_with("test")
        assert result == "plugin_instance"

    @patch("oscura.plugins.registry.get_plugin_registry")
    def test_list_plugins_uses_global(self, mock_get_registry):
        """Test that list_plugins uses global registry."""
        mock_registry = MagicMock()
        mock_registry.list_plugins.return_value = ["plugin1", "plugin2"]
        mock_get_registry.return_value = mock_registry

        result = list_plugins()

        mock_registry.list_plugins.assert_called_once()
        assert result == ["plugin1", "plugin2"]

    @patch("oscura.plugins.registry.get_plugin_registry")
    def test_is_compatible_uses_global(self, mock_get_registry):
        """Test that is_compatible uses global registry."""
        mock_registry = MagicMock()
        mock_registry.is_compatible.return_value = True
        mock_get_registry.return_value = mock_registry

        result = is_compatible("test")

        mock_registry.is_compatible.assert_called_once_with("test")
        assert result is True


class TestPluginRegistryDiscovery:
    """Tests for plugin discovery and loading."""

    @patch("oscura.plugins.registry.discover_plugins")
    def test_discover_and_load(self, mock_discover):
        """Test discovering and loading plugins."""
        registry = PluginRegistry()

        # Create mock discovered plugins
        mock_metadata = MockPluginMetadata(name="discovered_plugin")
        mock_discovered = MagicMock()
        mock_discovered.metadata = mock_metadata
        mock_discovered.compatible = True
        mock_discovered.load_error = None

        mock_discover.return_value = [mock_discovered]

        loaded = registry.discover_and_load()

        assert len(loaded) == 1
        assert loaded[0].name == "discovered_plugin"

    @patch("oscura.plugins.registry.discover_plugins")
    def test_discover_skips_incompatible(self, mock_discover):
        """Test that incompatible plugins are skipped."""
        registry = PluginRegistry()

        mock_metadata = MockPluginMetadata(name="incompatible_plugin")
        mock_discovered = MagicMock()
        mock_discovered.metadata = mock_metadata
        mock_discovered.compatible = False
        mock_discovered.load_error = None

        mock_discover.return_value = [mock_discovered]

        loaded = registry.discover_and_load(compatible_only=True)

        assert len(loaded) == 0

    @patch("oscura.plugins.registry.discover_plugins")
    def test_discover_skips_errors(self, mock_discover):
        """Test that plugins with load errors are skipped."""
        registry = PluginRegistry()

        mock_metadata = MockPluginMetadata(name="error_plugin")
        mock_discovered = MagicMock()
        mock_discovered.metadata = mock_metadata
        mock_discovered.compatible = True
        mock_discovered.load_error = "Failed to import"

        mock_discover.return_value = [mock_discovered]

        loaded = registry.discover_and_load()

        assert len(loaded) == 0


class TestRegistryCapabilityIndex:
    """Tests for capability-based indexing."""

    def test_capability_index_on_register(self):
        """Test that capabilities are indexed on registration."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="multi_cap", capabilities=["decode", "analyze"])

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        assert "multi_cap" in registry._by_capability.get("decode", [])
        assert "multi_cap" in registry._by_capability.get("analyze", [])

    def test_capability_index_on_unregister(self):
        """Test that capabilities are removed on unregistration."""
        registry = PluginRegistry()
        plugin = MockPlugin(name="to_remove", capabilities=["decode"])

        with patch("oscura.plugins.registry.OSCURA_API_VERSION", "1.0.0"):
            registry.register(plugin)

        # Verify it's indexed
        assert "to_remove" in registry._by_capability.get("decode", [])

        registry.unregister("to_remove")

        # Verify it's removed from index
        assert "to_remove" not in registry._by_capability.get("decode", [])
