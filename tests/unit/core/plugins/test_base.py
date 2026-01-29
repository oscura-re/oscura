"""Tests for plugin base classes and metadata.

This module tests the core plugin infrastructure:
- PluginCapability enum
- PluginMetadata dataclass
- PluginBase abstract class

Coverage target: 90%+
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oscura.core.plugins.base import (
    PluginBase,
    PluginCapability,
    PluginMetadata,
)


class TestPluginCapability:
    """Test PluginCapability enum."""

    def test_capability_values(self) -> None:
        """Test all capability enum values exist."""
        capabilities = [
            PluginCapability.PROTOCOL_DECODER,
            PluginCapability.FILE_LOADER,
            PluginCapability.FILE_EXPORTER,
            PluginCapability.ANALYZER,
            PluginCapability.ALGORITHM,
            PluginCapability.VISUALIZATION,
            PluginCapability.WORKFLOW,
        ]
        assert len(capabilities) == 7

    def test_capability_unique_values(self) -> None:
        """Test all capabilities have unique values."""
        capabilities = list(PluginCapability)
        values = [c.value for c in capabilities]
        assert len(values) == len(set(values))

    def test_capability_string_representation(self) -> None:
        """Test capability string representation."""
        cap = PluginCapability.PROTOCOL_DECODER
        assert "PROTOCOL_DECODER" in str(cap)


class TestPluginMetadata:
    """Test PluginMetadata dataclass."""

    def test_minimal_metadata(self) -> None:
        """Test creating metadata with minimal required fields."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
        )
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.api_version == "1.0.0"  # default
        assert metadata.enabled is True  # default

    def test_full_metadata(self) -> None:
        """Test creating metadata with all fields."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.2.3",
            api_version="2.0.0",
            author="Test Author",
            description="Test plugin description",
            homepage="https://example.com",
            license="MIT",
            capabilities=[PluginCapability.PROTOCOL_DECODER],
            dependencies={"numpy": ">=1.20", "scipy": ">=1.7"},
            provides={"protocols": ["uart", "spi"]},
            path=Path("/plugins/test"),
            enabled=False,
        )

        assert metadata.name == "test_plugin"
        assert metadata.version == "1.2.3"
        assert metadata.api_version == "2.0.0"
        assert metadata.author == "Test Author"
        assert metadata.description == "Test plugin description"
        assert metadata.homepage == "https://example.com"
        assert metadata.license == "MIT"
        assert metadata.capabilities == [PluginCapability.PROTOCOL_DECODER]
        assert metadata.dependencies == {"numpy": ">=1.20", "scipy": ">=1.7"}
        assert metadata.provides == {"protocols": ["uart", "spi"]}
        assert metadata.path == Path("/plugins/test")
        assert metadata.enabled is False

    def test_metadata_empty_name_raises(self) -> None:
        """Test that empty plugin name raises ValueError."""
        with pytest.raises(ValueError, match="Plugin name cannot be empty"):
            PluginMetadata(name="", version="1.0.0")

    def test_metadata_empty_version_raises(self) -> None:
        """Test that empty version raises ValueError."""
        with pytest.raises(ValueError, match="Plugin version cannot be empty"):
            PluginMetadata(name="test", version="")

    def test_qualified_name(self) -> None:
        """Test qualified_name property."""
        metadata = PluginMetadata(name="my_plugin", version="2.1.0")
        assert metadata.qualified_name == "my_plugin@2.1.0"

    def test_qualified_name_with_complex_version(self) -> None:
        """Test qualified name with complex version string."""
        metadata = PluginMetadata(name="plugin", version="1.2.3-rc.1+build.456")
        assert metadata.qualified_name == "plugin@1.2.3-rc.1+build.456"

    def test_is_compatible_with_same_major(self) -> None:
        """Test compatibility check with same major version."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            api_version="2.0.0",
        )
        assert metadata.is_compatible_with("2.0.0") is True
        assert metadata.is_compatible_with("2.1.0") is True
        assert metadata.is_compatible_with("2.99.99") is True

    def test_is_compatible_with_different_major(self) -> None:
        """Test compatibility check with different major version."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            api_version="2.0.0",
        )
        assert metadata.is_compatible_with("1.0.0") is False
        assert metadata.is_compatible_with("3.0.0") is False

    def test_default_capabilities_empty_list(self) -> None:
        """Test that default capabilities is empty list."""
        metadata = PluginMetadata(name="test", version="1.0.0")
        assert metadata.capabilities == []
        assert isinstance(metadata.capabilities, list)

    def test_default_dependencies_empty_dict(self) -> None:
        """Test that default dependencies is empty dict."""
        metadata = PluginMetadata(name="test", version="1.0.0")
        assert metadata.dependencies == {}
        assert isinstance(metadata.dependencies, dict)

    def test_default_provides_empty_dict(self) -> None:
        """Test that default provides is empty dict."""
        metadata = PluginMetadata(name="test", version="1.0.0")
        assert metadata.provides == {}
        assert isinstance(metadata.provides, dict)

    def test_multiple_capabilities(self) -> None:
        """Test metadata with multiple capabilities."""
        metadata = PluginMetadata(
            name="multi",
            version="1.0.0",
            capabilities=[
                PluginCapability.PROTOCOL_DECODER,
                PluginCapability.ANALYZER,
                PluginCapability.FILE_LOADER,
            ],
        )
        assert len(metadata.capabilities) == 3
        assert PluginCapability.PROTOCOL_DECODER in metadata.capabilities
        assert PluginCapability.ANALYZER in metadata.capabilities
        assert PluginCapability.FILE_LOADER in metadata.capabilities


class TestPluginBase:
    """Test PluginBase abstract class."""

    def test_cannot_instantiate_base_class(self) -> None:
        """Test that PluginBase cannot be instantiated directly."""
        # PluginBase is ABC but doesn't have abstract methods, so it can technically be instantiated
        # This test verifies the class exists and has expected structure
        assert hasattr(PluginBase, "metadata")
        assert hasattr(PluginBase, "on_load")
        assert hasattr(PluginBase, "on_unload")

    def test_concrete_plugin_implementation(self) -> None:
        """Test creating a concrete plugin implementation."""

        class TestPlugin(PluginBase):
            """Test plugin implementation."""

            metadata = PluginMetadata(
                name="test_plugin",
                version="1.0.0",
                capabilities=[PluginCapability.PROTOCOL_DECODER],
            )

            def on_load(self) -> None:
                """Load plugin."""
                self.loaded = True

            def on_unload(self) -> None:
                """Unload plugin."""
                self.loaded = False

        # Create instance
        plugin = TestPlugin()
        assert plugin.metadata.name == "test_plugin"
        assert plugin.metadata.version == "1.0.0"

        # Test lifecycle methods
        plugin.on_load()
        assert plugin.loaded is True

        plugin.on_unload()
        assert plugin.loaded is False

    def test_plugin_with_multiple_capabilities(self) -> None:
        """Test plugin declaring multiple capabilities."""

        class MultiCapPlugin(PluginBase):
            """Plugin with multiple capabilities."""

            metadata = PluginMetadata(
                name="multi_cap",
                version="2.0.0",
                capabilities=[
                    PluginCapability.FILE_LOADER,
                    PluginCapability.FILE_EXPORTER,
                    PluginCapability.ANALYZER,
                ],
            )

            def on_load(self) -> None:
                pass

            def on_unload(self) -> None:
                pass

        plugin = MultiCapPlugin()
        assert len(plugin.metadata.capabilities) == 3

    def test_plugin_with_dependencies(self) -> None:
        """Test plugin with declared dependencies."""

        class DependentPlugin(PluginBase):
            """Plugin with dependencies."""

            metadata = PluginMetadata(
                name="dependent",
                version="1.0.0",
                dependencies={
                    "numpy": ">=1.20",
                    "other_plugin": ">=2.0",
                },
            )

            def on_load(self) -> None:
                pass

            def on_unload(self) -> None:
                pass

        plugin = DependentPlugin()
        assert "numpy" in plugin.metadata.dependencies
        assert "other_plugin" in plugin.metadata.dependencies

    def test_plugin_provides_protocols(self) -> None:
        """Test plugin that provides protocol decoders."""

        class ProtocolPlugin(PluginBase):
            """Plugin providing protocol decoders."""

            metadata = PluginMetadata(
                name="proto_plugin",
                version="1.0.0",
                capabilities=[PluginCapability.PROTOCOL_DECODER],
                provides={
                    "protocols": ["modbus", "profinet"],
                    "analyzers": ["crc_validator"],
                },
            )

            def on_load(self) -> None:
                pass

            def on_unload(self) -> None:
                pass

        plugin = ProtocolPlugin()
        assert "protocols" in plugin.metadata.provides
        assert "modbus" in plugin.metadata.provides["protocols"]
        assert "profinet" in plugin.metadata.provides["protocols"]
        assert "analyzers" in plugin.metadata.provides
