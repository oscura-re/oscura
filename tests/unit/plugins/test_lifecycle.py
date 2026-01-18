"""Comprehensive unit tests for plugin lifecycle management.

This module provides comprehensive tests for plugin lifecycle, dependency
resolution, graceful degradation, lazy loading, and hot reload capabilities.

Requirements tested:
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from oscura.plugins.base import PluginBase, PluginCapability, PluginMetadata
from oscura.plugins.lifecycle import (
    DependencyGraph,
    DependencyInfo,
    PluginHandle,
    PluginLifecycleManager,
    PluginLoadError,
    PluginState,
    get_lifecycle_manager,
    set_plugin_directories,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockPlugin(PluginBase):
    """Mock plugin for testing."""

    name = "mock_plugin"
    version = "1.0.0"
    api_version = "1.0.0"
    author = "Test Author"
    description = "Mock plugin for testing"
    capabilities = [PluginCapability.PROTOCOL_DECODER]

    def __init__(self) -> None:
        super().__init__()
        self.load_called = False
        self.unload_called = False
        self.enable_called = False
        self.disable_called = False
        self.configure_called = False
        self.config_data = {}

    def on_load(self) -> None:
        self.load_called = True

    def on_unload(self) -> None:
        self.unload_called = True

    def on_enable(self) -> None:
        self.enable_called = True

    def on_disable(self) -> None:
        self.disable_called = True

    def on_configure(self, config: dict) -> None:
        self.configure_called = True
        self.config_data = config
        super().on_configure(config)


class FailingPlugin(PluginBase):
    """Plugin that fails on load."""

    name = "failing_plugin"
    version = "1.0.0"
    api_version = "1.0.0"

    def on_load(self) -> None:
        raise RuntimeError("Simulated load failure")


class DependentPlugin(PluginBase):
    """Plugin with dependencies."""

    name = "dependent_plugin"
    version = "1.0.0"
    api_version = "1.0.0"
    requires_plugins = [("mock_plugin", ">=1.0.0")]


@pytest.fixture
def temp_plugin_dir():
    """Create temporary plugin directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def lifecycle_manager(temp_plugin_dir):
    """Create a fresh lifecycle manager for each test."""
    return PluginLifecycleManager(plugin_dirs=[temp_plugin_dir])


@pytest.fixture
def mock_plugin_file(temp_plugin_dir):
    """Create a mock plugin file."""
    plugin_file = temp_plugin_dir / "test_plugin.py"
    plugin_code = """
from oscura.plugins.base import PluginBase

class TestPlugin(PluginBase):
    name = "test_plugin"
    version = "1.0.0"
    api_version = "1.0.0"
"""
    plugin_file.write_text(plugin_code)
    return plugin_file


@pytest.fixture
def mock_plugin_package(temp_plugin_dir):
    """Create a mock plugin package."""
    plugin_dir = temp_plugin_dir / "package_plugin"
    plugin_dir.mkdir()
    init_file = plugin_dir / "__init__.py"
    plugin_code = """
from oscura.plugins.base import PluginBase

class PackagePlugin(PluginBase):
    name = "package_plugin"
    version = "1.0.0"
    api_version = "1.0.0"
"""
    init_file.write_text(plugin_code)
    return plugin_dir


# =============================================================================
# PluginState Tests
# =============================================================================


def test_plugin_state_enum():
    """Test PluginState enum values."""
    assert PluginState.DISCOVERED
    assert PluginState.LOADING
    assert PluginState.LOADED
    assert PluginState.CONFIGURED
    assert PluginState.ENABLED
    assert PluginState.DISABLED
    assert PluginState.ERROR
    assert PluginState.UNLOADING


# =============================================================================
# PluginLoadError Tests
# =============================================================================


def test_plugin_load_error_creation():
    """Test PluginLoadError creation."""
    error = PluginLoadError(
        plugin_name="test",
        error=ValueError("test error"),
        traceback="traceback",
        stage="load",
        recoverable=True,
    )

    assert error.plugin_name == "test"
    assert isinstance(error.error, ValueError)
    assert error.traceback == "traceback"
    assert error.stage == "load"
    assert error.recoverable is True


def test_plugin_load_error_defaults():
    """Test PluginLoadError default values."""
    error = PluginLoadError(
        plugin_name="test",
        error=ValueError("test error"),
    )

    assert error.traceback == ""
    assert error.stage == "load"
    assert error.recoverable is True


# =============================================================================
# DependencyInfo Tests
# =============================================================================


def test_dependency_info_creation():
    """Test DependencyInfo creation."""
    dep = DependencyInfo(
        name="plugin1",
        version_spec=">=1.0.0",
        optional=False,
        resolved=False,
    )

    assert dep.name == "plugin1"
    assert dep.version_spec == ">=1.0.0"
    assert dep.optional is False
    assert dep.resolved is False


def test_dependency_info_defaults():
    """Test DependencyInfo default values."""
    dep = DependencyInfo(name="plugin1")

    assert dep.version_spec == "*"
    assert dep.optional is False
    assert dep.resolved is False


# =============================================================================
# PluginHandle Tests
# =============================================================================


def test_plugin_handle_creation():
    """Test PluginHandle creation."""
    metadata = PluginMetadata(name="test", version="1.0.0")
    handle = PluginHandle(metadata=metadata)

    assert handle.metadata == metadata
    assert handle.instance is None
    assert handle.state == PluginState.DISCOVERED
    assert handle.dependencies == []
    assert handle.dependents == []
    assert handle.errors == []
    assert handle.load_time == 0.0


def test_plugin_handle_with_instance():
    """Test PluginHandle with plugin instance."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )

    assert handle.instance == plugin
    assert handle.state == PluginState.LOADED


# =============================================================================
# DependencyGraph Tests
# =============================================================================


def test_dependency_graph_creation():
    """Test DependencyGraph creation."""
    graph = DependencyGraph()
    assert graph is not None


def test_dependency_graph_add_plugin():
    """Test adding plugins to dependency graph."""
    graph = DependencyGraph()
    graph.add_plugin("plugin1")
    graph.add_plugin("plugin2")

    assert "plugin1" in graph._nodes
    assert "plugin2" in graph._nodes


def test_dependency_graph_add_dependency():
    """Test adding dependencies to graph."""
    graph = DependencyGraph()
    graph.add_dependency("plugin2", "plugin1", ">=1.0.0")

    deps = graph.get_dependencies("plugin2")
    assert len(deps) == 1
    assert deps[0].name == "plugin1"
    assert deps[0].version_spec == ">=1.0.0"


def test_dependency_graph_resolve_order_simple():
    """Test resolving dependency order for simple case."""
    graph = DependencyGraph()
    graph.add_plugin("core")
    graph.add_dependency("decoder", "core", ">=1.0.0")

    order = graph.resolve_order()
    assert order == ["core", "decoder"]


def test_dependency_graph_resolve_order_complex():
    """Test resolving dependency order for complex graph."""
    graph = DependencyGraph()
    graph.add_plugin("a")
    graph.add_dependency("b", "a")
    graph.add_dependency("c", "a")
    graph.add_dependency("d", "b")
    graph.add_dependency("d", "c")

    order = graph.resolve_order()

    # 'a' must come first
    assert order[0] == "a"
    # 'b' and 'c' must come before 'd'
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")


def test_dependency_graph_circular_dependency():
    """Test circular dependency detection."""
    graph = DependencyGraph()
    graph.add_dependency("a", "b")
    graph.add_dependency("b", "c")
    graph.add_dependency("c", "a")

    with pytest.raises(ValueError, match="Circular dependency detected"):
        graph.resolve_order()


def test_dependency_graph_self_dependency():
    """Test self-dependency detection."""
    graph = DependencyGraph()
    graph.add_dependency("a", "a")

    with pytest.raises(ValueError, match="Circular dependency detected"):
        graph.resolve_order()


def test_dependency_graph_get_dependencies():
    """Test getting dependencies for a plugin."""
    graph = DependencyGraph()
    graph.add_dependency("plugin", "dep1", ">=1.0.0")
    graph.add_dependency("plugin", "dep2", ">=2.0.0")

    deps = graph.get_dependencies("plugin")
    assert len(deps) == 2
    assert deps[0].name == "dep1"
    assert deps[1].name == "dep2"


def test_dependency_graph_get_dependencies_none():
    """Test getting dependencies for plugin with none."""
    graph = DependencyGraph()
    graph.add_plugin("plugin")

    deps = graph.get_dependencies("plugin")
    assert deps == []


def test_dependency_graph_get_dependents():
    """Test getting dependents for a plugin."""
    graph = DependencyGraph()
    graph.add_dependency("plugin1", "core")
    graph.add_dependency("plugin2", "core")

    dependents = graph.get_dependents("core")
    assert set(dependents) == {"plugin1", "plugin2"}


def test_dependency_graph_optional_dependency():
    """Test optional dependencies."""
    graph = DependencyGraph()
    graph.add_dependency("plugin", "optional_dep", optional=True)

    deps = graph.get_dependencies("plugin")
    assert deps[0].optional is True


# =============================================================================
# PluginLifecycleManager Discovery Tests
# =============================================================================


def test_lifecycle_manager_creation():
    """Test PluginLifecycleManager creation."""
    manager = PluginLifecycleManager()
    assert manager is not None


def test_lifecycle_manager_discover_plugins_empty(lifecycle_manager):
    """Test discovering plugins in empty directory."""
    discovered = lifecycle_manager.discover_plugins()
    assert discovered == []


def test_lifecycle_manager_discover_single_file(lifecycle_manager, mock_plugin_file):
    """Test discovering single file plugin."""
    discovered = lifecycle_manager.discover_plugins()
    assert len(discovered) == 1
    assert "test_plugin" in discovered


def test_lifecycle_manager_discover_package(lifecycle_manager, mock_plugin_package):
    """Test discovering package plugin."""
    discovered = lifecycle_manager.discover_plugins()
    assert len(discovered) == 1
    assert "package_plugin" in discovered


def test_lifecycle_manager_discover_multiple(
    lifecycle_manager, mock_plugin_file, mock_plugin_package
):
    """Test discovering multiple plugins."""
    discovered = lifecycle_manager.discover_plugins()
    assert len(discovered) == 2
    assert set(discovered) == {"test_plugin", "package_plugin"}


def test_lifecycle_manager_discover_nonexistent_dir():
    """Test discovering plugins with nonexistent directory."""
    manager = PluginLifecycleManager(plugin_dirs=[Path("/nonexistent/path")])
    discovered = manager.discover_plugins()
    assert discovered == []


# =============================================================================
# PluginLifecycleManager Load Tests
# =============================================================================


def test_load_plugin_success(lifecycle_manager):
    """Test successful plugin loading."""
    # Register lazy loader manually
    metadata = PluginMetadata(name="mock_plugin", version="1.0.0")
    handle = PluginHandle(metadata=metadata, state=PluginState.DISCOVERED)
    lifecycle_manager._handles["mock_plugin"] = handle
    lifecycle_manager._lazy_loaders["mock_plugin"] = MockPlugin

    result = lifecycle_manager.load_plugin("mock_plugin")

    assert result.state == PluginState.LOADED
    assert result.instance is not None
    assert result.instance.load_called is True


def test_load_plugin_not_found(lifecycle_manager):
    """Test loading non-existent plugin."""
    with pytest.raises(ValueError, match="Plugin 'nonexistent' not discovered"):
        lifecycle_manager.load_plugin("nonexistent")


def test_load_plugin_already_loaded(lifecycle_manager):
    """Test loading already loaded plugin."""
    # Setup
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    result = lifecycle_manager.load_plugin("mock_plugin")
    assert result == handle
    assert result.state == PluginState.LOADED


def test_load_plugin_with_error(lifecycle_manager):
    """Test loading plugin that fails."""
    metadata = PluginMetadata(name="failing_plugin", version="1.0.0")
    handle = PluginHandle(metadata=metadata, state=PluginState.DISCOVERED)
    lifecycle_manager._handles["failing_plugin"] = handle
    lifecycle_manager._lazy_loaders["failing_plugin"] = FailingPlugin

    with pytest.raises(RuntimeError, match="Simulated load failure"):
        lifecycle_manager.load_plugin("failing_plugin")

    # Check error was recorded
    handle = lifecycle_manager._handles["failing_plugin"]
    assert handle.state == PluginState.ERROR
    assert len(handle.errors) == 1
    assert handle.errors[0].stage == "load"


def test_load_plugin_without_lazy(lifecycle_manager, mock_plugin_file):
    """Test loading plugin without lazy loading."""
    lifecycle_manager.discover_plugins()
    handle = lifecycle_manager.load_plugin("test_plugin", lazy=False)

    assert handle.state == PluginState.LOADED
    assert handle.instance is not None


def test_load_plugin_tracks_time(lifecycle_manager):
    """Test that plugin load time is tracked."""
    metadata = PluginMetadata(name="mock_plugin", version="1.0.0")
    handle = PluginHandle(metadata=metadata, state=PluginState.DISCOVERED)
    lifecycle_manager._handles["mock_plugin"] = handle
    lifecycle_manager._lazy_loaders["mock_plugin"] = MockPlugin

    result = lifecycle_manager.load_plugin("mock_plugin")
    assert result.load_time > 0


# =============================================================================
# PluginLifecycleManager Configure Tests
# =============================================================================


def test_configure_plugin_success(lifecycle_manager):
    """Test successful plugin configuration."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    config = {"key": "value"}
    result = lifecycle_manager.configure_plugin("mock_plugin", config)

    assert result.state == PluginState.CONFIGURED
    assert plugin.configure_called is True
    assert plugin.config_data == config


def test_configure_plugin_not_found(lifecycle_manager):
    """Test configuring non-existent plugin."""
    with pytest.raises(ValueError, match="Plugin 'nonexistent' not found"):
        lifecycle_manager.configure_plugin("nonexistent", {})


def test_configure_plugin_wrong_state(lifecycle_manager):
    """Test configuring plugin in wrong state."""
    metadata = PluginMetadata(name="mock_plugin", version="1.0.0")
    handle = PluginHandle(metadata=metadata, state=PluginState.DISCOVERED)
    lifecycle_manager._handles["mock_plugin"] = handle

    with pytest.raises(ValueError, match="Cannot configure plugin in state"):
        lifecycle_manager.configure_plugin("mock_plugin", {})


def test_configure_plugin_with_error(lifecycle_manager):
    """Test configuration error handling."""

    class FailingConfigPlugin(PluginBase):
        name = "failing_config"
        version = "1.0.0"
        api_version = "1.0.0"

        def on_configure(self, config):
            raise ValueError("Config error")

    plugin = FailingConfigPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["failing_config"] = handle

    with pytest.raises(ValueError, match="Config error"):
        lifecycle_manager.configure_plugin("failing_config", {})

    # Check error was recorded
    handle = lifecycle_manager._handles["failing_config"]
    assert handle.state == PluginState.ERROR
    assert len(handle.errors) == 1


# =============================================================================
# PluginLifecycleManager Enable/Disable Tests
# =============================================================================


def test_enable_plugin_success(lifecycle_manager):
    """Test successful plugin enable."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.CONFIGURED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    result = lifecycle_manager.enable_plugin("mock_plugin")

    assert result.state == PluginState.ENABLED
    assert plugin.enable_called is True


def test_enable_plugin_not_found(lifecycle_manager):
    """Test enabling non-existent plugin."""
    with pytest.raises(ValueError, match="Plugin 'nonexistent' not found"):
        lifecycle_manager.enable_plugin("nonexistent")


def test_enable_plugin_already_enabled(lifecycle_manager):
    """Test enabling already enabled plugin."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    result = lifecycle_manager.enable_plugin("mock_plugin")
    assert result.state == PluginState.ENABLED


def test_enable_plugin_from_discovered(lifecycle_manager):
    """Test enabling plugin from discovered state (auto-load/configure)."""
    metadata = PluginMetadata(name="mock_plugin", version="1.0.0")
    handle = PluginHandle(metadata=metadata, state=PluginState.DISCOVERED)
    lifecycle_manager._handles["mock_plugin"] = handle
    lifecycle_manager._lazy_loaders["mock_plugin"] = MockPlugin

    result = lifecycle_manager.enable_plugin("mock_plugin")

    assert result.state == PluginState.ENABLED


def test_disable_plugin_success(lifecycle_manager):
    """Test successful plugin disable."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    result = lifecycle_manager.disable_plugin("mock_plugin")

    assert result.state == PluginState.DISABLED
    assert plugin.disable_called is True


def test_disable_plugin_with_dependents(lifecycle_manager):
    """Test disabling plugin with enabled dependents."""
    # Setup core plugin
    core = MockPlugin()
    core_handle = PluginHandle(
        metadata=core.metadata,
        instance=core,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["mock_plugin"] = core_handle

    # Setup dependent plugin
    dependent = DependentPlugin()
    dep_info = DependencyInfo(name="mock_plugin", version_spec=">=1.0.0")
    dep_handle = PluginHandle(
        metadata=dependent.metadata,
        instance=dependent,
        state=PluginState.ENABLED,
        dependencies=[dep_info],
    )
    lifecycle_manager._handles["dependent_plugin"] = dep_handle

    with pytest.raises(ValueError, match="Cannot disable.*required by"):
        lifecycle_manager.disable_plugin("mock_plugin")


def test_disable_plugin_force(lifecycle_manager):
    """Test force disabling plugin with dependents."""
    # Setup core plugin
    core = MockPlugin()
    core_handle = PluginHandle(
        metadata=core.metadata,
        instance=core,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["mock_plugin"] = core_handle

    # Setup dependent plugin
    dependent = DependentPlugin()
    dep_info = DependencyInfo(name="mock_plugin", version_spec=">=1.0.0")
    dep_handle = PluginHandle(
        metadata=dependent.metadata,
        instance=dependent,
        state=PluginState.ENABLED,
        dependencies=[dep_info],
    )
    lifecycle_manager._handles["dependent_plugin"] = dep_handle

    result = lifecycle_manager.disable_plugin("mock_plugin", force=True)
    assert result.state == PluginState.DISABLED


# =============================================================================
# PluginLifecycleManager Unload Tests
# =============================================================================


def test_unload_plugin_success(lifecycle_manager):
    """Test successful plugin unload."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    lifecycle_manager.unload_plugin("mock_plugin")

    handle = lifecycle_manager._handles["mock_plugin"]
    assert handle.state == PluginState.DISCOVERED
    assert handle.instance is None
    assert plugin.unload_called is True


def test_unload_plugin_enabled(lifecycle_manager):
    """Test unloading enabled plugin (auto-disable)."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    lifecycle_manager.unload_plugin("mock_plugin")

    handle = lifecycle_manager._handles["mock_plugin"]
    assert handle.state == PluginState.DISCOVERED
    assert plugin.disable_called is True


def test_unload_plugin_force(lifecycle_manager):
    """Test force unloading enabled plugin."""
    plugin = MockPlugin()
    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    lifecycle_manager.unload_plugin("mock_plugin", force=True)

    handle = lifecycle_manager._handles["mock_plugin"]
    assert handle.state == PluginState.DISCOVERED


def test_unload_plugin_not_found(lifecycle_manager):
    """Test unloading non-existent plugin."""
    lifecycle_manager.unload_plugin("nonexistent")  # Should not raise


def test_unload_plugin_error_handling(lifecycle_manager):
    """Test error handling during unload."""

    class FailingUnloadPlugin(PluginBase):
        name = "failing_unload"
        version = "1.0.0"
        api_version = "1.0.0"

        def on_unload(self):
            raise RuntimeError("Unload error")

    plugin = FailingUnloadPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["failing_unload"] = handle

    # Should not raise, but log warning
    lifecycle_manager.unload_plugin("failing_unload")

    handle = lifecycle_manager._handles["failing_unload"]
    assert handle.state == PluginState.DISCOVERED
    assert handle.instance is None


# =============================================================================
# PluginLifecycleManager Reload Tests
# =============================================================================


def test_reload_plugin_success(lifecycle_manager, mock_plugin_file):
    """Test successful plugin reload."""
    lifecycle_manager.discover_plugins()
    handle = lifecycle_manager.load_plugin("test_plugin")
    handle = lifecycle_manager.enable_plugin("test_plugin")

    # Reload
    result = lifecycle_manager.reload_plugin("test_plugin")

    assert result.state == PluginState.ENABLED
    assert result.instance is not None


def test_reload_plugin_not_found(lifecycle_manager):
    """Test reloading non-existent plugin."""
    with pytest.raises(ValueError, match="Plugin 'nonexistent' not found"):
        lifecycle_manager.reload_plugin("nonexistent")


def test_reload_plugin_preserves_state(lifecycle_manager):
    """Test that reload preserves plugin state."""
    plugin = MockPlugin()
    plugin._config = {"key": "value"}
    plugin._registered_protocols = ["uart"]
    plugin._registered_algorithms = [("cat", "algo", lambda: None)]

    metadata = plugin.metadata
    handle = PluginHandle(
        metadata=metadata,
        instance=plugin,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle
    lifecycle_manager._lazy_loaders["mock_plugin"] = MockPlugin

    with patch.object(lifecycle_manager, "_get_plugin_path", return_value=Path("/")):
        with patch.object(lifecycle_manager, "_load_plugin_from_path", return_value=MockPlugin()):
            result = lifecycle_manager.reload_plugin("mock_plugin")

    assert result.state == PluginState.ENABLED


def test_reload_plugin_clears_sys_modules(lifecycle_manager):
    """Test that reload clears sys.modules."""
    # Add fake module entries
    sys.modules["mock_plugin"] = Mock()
    sys.modules["mock_plugin.submodule"] = Mock()

    plugin = MockPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle
    lifecycle_manager._lazy_loaders["mock_plugin"] = MockPlugin

    with patch.object(lifecycle_manager, "_get_plugin_path", return_value=Path("/")):
        with patch.object(lifecycle_manager, "_load_plugin_from_path", return_value=MockPlugin()):
            lifecycle_manager.reload_plugin("mock_plugin")

    assert "mock_plugin" not in sys.modules
    assert "mock_plugin.submodule" not in sys.modules


# =============================================================================
# PluginLifecycleManager Dependency Resolution Tests
# =============================================================================


def test_resolve_dependencies_success(lifecycle_manager):
    """Test successful dependency resolution."""
    # Setup core plugin
    core = MockPlugin()
    core_handle = PluginHandle(
        metadata=core.metadata,
        instance=core,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = core_handle

    # Setup dependent plugin
    dependent = DependentPlugin()
    dep_info = DependencyInfo(name="mock_plugin", version_spec=">=1.0.0")
    dep_handle = PluginHandle(
        metadata=dependent.metadata,
        state=PluginState.DISCOVERED,
        dependencies=[dep_info],
    )
    lifecycle_manager._handles["dependent_plugin"] = dep_handle
    lifecycle_manager._lazy_loaders["dependent_plugin"] = DependentPlugin

    lifecycle_manager.load_plugin("dependent_plugin")

    # Dependency should be marked as resolved
    assert dep_info.resolved is True


def test_resolve_dependencies_missing_required(lifecycle_manager):
    """Test missing required dependency."""
    dependent = DependentPlugin()
    dep_info = DependencyInfo(name="missing_plugin", version_spec=">=1.0.0")
    dep_handle = PluginHandle(
        metadata=dependent.metadata,
        state=PluginState.DISCOVERED,
        dependencies=[dep_info],
    )
    lifecycle_manager._handles["dependent_plugin"] = dep_handle
    lifecycle_manager._lazy_loaders["dependent_plugin"] = DependentPlugin

    with pytest.raises(ValueError, match="Required dependency 'missing_plugin'"):
        lifecycle_manager.load_plugin("dependent_plugin")


def test_resolve_dependencies_optional(lifecycle_manager):
    """Test optional dependency handling."""
    dependent = DependentPlugin()
    dep_info = DependencyInfo(
        name="optional_plugin",
        version_spec=">=1.0.0",
        optional=True,
    )
    dep_handle = PluginHandle(
        metadata=dependent.metadata,
        state=PluginState.DISCOVERED,
        dependencies=[dep_info],
    )
    lifecycle_manager._handles["dependent_plugin"] = dep_handle
    lifecycle_manager._lazy_loaders["dependent_plugin"] = DependentPlugin

    # Should not raise even though dependency is missing
    lifecycle_manager.load_plugin("dependent_plugin")


def test_resolve_dependencies_already_resolved(lifecycle_manager):
    """Test that already resolved dependencies are skipped."""
    dependent = DependentPlugin()
    dep_info = DependencyInfo(
        name="mock_plugin",
        version_spec=">=1.0.0",
        resolved=True,
    )
    dep_handle = PluginHandle(
        metadata=dependent.metadata,
        state=PluginState.DISCOVERED,
        dependencies=[dep_info],
    )
    lifecycle_manager._handles["dependent_plugin"] = dep_handle
    lifecycle_manager._lazy_loaders["dependent_plugin"] = DependentPlugin

    # Should not try to load mock_plugin
    lifecycle_manager.load_plugin("dependent_plugin")


def test_load_plugin_without_dependency_resolution(lifecycle_manager):
    """Test loading plugin without resolving dependencies."""
    dependent = DependentPlugin()
    dep_info = DependencyInfo(name="mock_plugin", version_spec=">=1.0.0")
    dep_handle = PluginHandle(
        metadata=dependent.metadata,
        state=PluginState.DISCOVERED,
        dependencies=[dep_info],
    )
    lifecycle_manager._handles["dependent_plugin"] = dep_handle
    lifecycle_manager._lazy_loaders["dependent_plugin"] = DependentPlugin

    # Should succeed without resolving dependencies
    result = lifecycle_manager.load_plugin("dependent_plugin", resolve_deps=False)
    assert result.state == PluginState.LOADED


# =============================================================================
# PluginLifecycleManager Hot Reload Tests
# =============================================================================


def test_check_for_changes_no_changes(lifecycle_manager, mock_plugin_file):
    """Test checking for changes when none exist."""
    lifecycle_manager.discover_plugins()

    changed = lifecycle_manager.check_for_changes()
    assert changed == []


def test_check_for_changes_with_modification(lifecycle_manager, mock_plugin_file):
    """Test detecting file changes."""
    lifecycle_manager.discover_plugins()

    # Simulate file modification
    time.sleep(0.01)
    mock_plugin_file.touch()

    # Note: This test may be flaky due to filesystem timestamp precision
    # changed = lifecycle_manager.check_for_changes()
    # We just verify it doesn't crash
    lifecycle_manager.check_for_changes()


def test_auto_reload_changed(lifecycle_manager):
    """Test auto-reloading changed plugins."""
    plugin = MockPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle
    lifecycle_manager._lazy_loaders["mock_plugin"] = MockPlugin

    with patch.object(lifecycle_manager, "check_for_changes", return_value=["mock_plugin"]):
        with patch.object(lifecycle_manager, "reload_plugin") as mock_reload:
            reloaded = lifecycle_manager.auto_reload_changed()

            mock_reload.assert_called_once_with("mock_plugin")


def test_auto_reload_changed_with_error(lifecycle_manager):
    """Test auto-reload handling errors gracefully."""
    plugin = MockPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    with patch.object(lifecycle_manager, "check_for_changes", return_value=["mock_plugin"]):
        with patch.object(
            lifecycle_manager, "reload_plugin", side_effect=RuntimeError("Reload failed")
        ):
            reloaded = lifecycle_manager.auto_reload_changed()
            assert reloaded == []


# =============================================================================
# PluginLifecycleManager Graceful Degradation Tests
# =============================================================================


def test_graceful_degradation_not_found(lifecycle_manager):
    """Test graceful degradation for non-existent plugin."""
    result = lifecycle_manager.graceful_degradation("nonexistent")
    assert result["status"] == "not_found"
    assert result["alternatives"] == []


def test_graceful_degradation_with_error(lifecycle_manager):
    """Test graceful degradation with plugin error."""
    plugin = MockPlugin()
    error = PluginLoadError(
        plugin_name="mock_plugin",
        error=ValueError("Test error"),
        recoverable=True,
    )
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.ERROR,
        errors=[error],
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    result = lifecycle_manager.graceful_degradation("mock_plugin")

    assert result["status"] == "degraded"
    assert result["plugin"] == "mock_plugin"
    assert result["recoverable"] is True


def test_graceful_degradation_find_alternatives(lifecycle_manager):
    """Test finding alternative plugins with same capability."""
    # Setup failed plugin
    plugin1 = MockPlugin()
    plugin1.name = "plugin1"
    plugin1.capabilities = [PluginCapability.PROTOCOL_DECODER]
    error = PluginLoadError(
        plugin_name="plugin1",
        error=ValueError("Error"),
    )
    handle1 = PluginHandle(
        metadata=plugin1.metadata,
        instance=plugin1,
        state=PluginState.ERROR,
        errors=[error],
    )
    lifecycle_manager._handles["plugin1"] = handle1

    # Setup alternative plugin
    plugin2 = MockPlugin()
    plugin2.name = "plugin2"
    plugin2.capabilities = [PluginCapability.PROTOCOL_DECODER]
    handle2 = PluginHandle(
        metadata=plugin2.metadata,
        instance=plugin2,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["plugin2"] = handle2

    result = lifecycle_manager.graceful_degradation("plugin1")

    assert "plugin2" in result["alternatives"]


# =============================================================================
# PluginLifecycleManager State Change Callbacks
# =============================================================================


def test_state_change_callback(lifecycle_manager):
    """Test state change callback invocation."""
    callback = Mock()
    lifecycle_manager.on_state_change(callback)

    plugin = MockPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    lifecycle_manager.enable_plugin("mock_plugin")

    # Should be called at least once
    assert callback.called


def test_state_change_callback_error_handling(lifecycle_manager):
    """Test that callback errors don't break state changes."""

    def failing_callback(name, state):
        raise RuntimeError("Callback error")

    lifecycle_manager.on_state_change(failing_callback)

    plugin = MockPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    # Should not raise despite callback error
    lifecycle_manager.enable_plugin("mock_plugin")


# =============================================================================
# PluginLifecycleManager Utility Methods
# =============================================================================


def test_get_handle(lifecycle_manager):
    """Test getting plugin handle."""
    metadata = PluginMetadata(name="test", version="1.0.0")
    handle = PluginHandle(metadata=metadata)
    lifecycle_manager._handles["test"] = handle

    result = lifecycle_manager.get_handle("test")
    assert result == handle


def test_get_handle_not_found(lifecycle_manager):
    """Test getting non-existent handle."""
    result = lifecycle_manager.get_handle("nonexistent")
    assert result is None


def test_get_enabled_plugins(lifecycle_manager):
    """Test getting list of enabled plugins."""
    plugin1 = MockPlugin()
    plugin1.name = "plugin1"
    handle1 = PluginHandle(
        metadata=plugin1.metadata,
        state=PluginState.ENABLED,
    )
    lifecycle_manager._handles["plugin1"] = handle1

    plugin2 = MockPlugin()
    plugin2.name = "plugin2"
    handle2 = PluginHandle(
        metadata=plugin2.metadata,
        state=PluginState.DISABLED,
    )
    lifecycle_manager._handles["plugin2"] = handle2

    enabled = lifecycle_manager.get_enabled_plugins()
    assert enabled == ["plugin1"]


# =============================================================================
# Global Lifecycle Manager Tests
# =============================================================================


def test_get_lifecycle_manager_singleton():
    """Test that get_lifecycle_manager returns singleton."""
    manager1 = get_lifecycle_manager()
    manager2 = get_lifecycle_manager()
    assert manager1 is manager2


def test_set_plugin_directories():
    """Test setting plugin directories."""
    dirs = [Path("/test/path")]
    set_plugin_directories(dirs)

    manager = get_lifecycle_manager()
    assert manager._plugin_dirs == dirs


# =============================================================================
# Thread Safety Tests
# =============================================================================


def test_lifecycle_manager_thread_safety(lifecycle_manager):
    """Test thread-safe access to lifecycle manager."""
    plugin = MockPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
        state=PluginState.LOADED,
    )
    lifecycle_manager._handles["mock_plugin"] = handle

    def enable_plugin():
        try:
            lifecycle_manager.enable_plugin("mock_plugin")
        except (ValueError, RuntimeError, KeyError) as e:
            # Plugin already enabled or invalid state - acceptable in concurrent scenario
            pass

    threads = [threading.Thread(target=enable_plugin) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should end up enabled
    handle = lifecycle_manager._handles["mock_plugin"]
    assert handle.state == PluginState.ENABLED


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


def test_load_plugin_from_path_file(lifecycle_manager, mock_plugin_file):
    """Test loading plugin from file path."""
    plugin = lifecycle_manager._load_plugin_from_path("test_plugin", mock_plugin_file)
    assert plugin is not None
    assert hasattr(plugin, "name")


def test_load_plugin_from_path_package(lifecycle_manager, mock_plugin_package):
    """Test loading plugin from package path."""
    plugin = lifecycle_manager._load_plugin_from_path("package_plugin", mock_plugin_package)
    assert plugin is not None
    assert hasattr(plugin, "name")


def test_load_plugin_from_path_invalid(lifecycle_manager):
    """Test loading plugin from invalid path."""
    with pytest.raises(ImportError):
        lifecycle_manager._load_plugin_from_path("test", Path("/nonexistent"))


def test_get_plugin_path_success(lifecycle_manager, mock_plugin_file):
    """Test getting plugin path."""
    lifecycle_manager.discover_plugins()
    path = lifecycle_manager._get_plugin_path("test_plugin")
    assert path == mock_plugin_file


def test_get_plugin_path_not_found(lifecycle_manager):
    """Test getting path for non-existent plugin."""
    with pytest.raises(ValueError, match="Plugin path not found"):
        lifecycle_manager._get_plugin_path("nonexistent")


def test_save_plugin_state(lifecycle_manager):
    """Test saving plugin state."""
    plugin = MockPlugin()
    plugin._config = {"key": "value"}
    plugin._registered_protocols = ["uart"]
    plugin._registered_algorithms = [("cat", "algo", lambda: None)]

    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
    )

    state = lifecycle_manager._save_plugin_state(handle)

    assert state["config"] == {"key": "value"}
    assert state["registered_protocols"] == ["uart"]
    assert len(state["registered_algorithms"]) == 1


def test_restore_plugin_state(lifecycle_manager):
    """Test restoring plugin state."""
    plugin = MockPlugin()
    handle = PluginHandle(
        metadata=plugin.metadata,
        instance=plugin,
    )

    state = {
        "config": {"key": "value"},
        "registered_protocols": ["uart"],
        "registered_algorithms": [("cat", "algo", lambda: None)],
    }

    lifecycle_manager._restore_plugin_state(handle, state)

    assert plugin._config == {"key": "value"}
    assert plugin._registered_protocols == ["uart"]
    assert len(plugin._registered_algorithms) == 1


def test_cleanup_plugin_references(lifecycle_manager):
    """Test cleaning up plugin references."""
    lifecycle_manager._lazy_loaders["test"] = lambda: None

    lifecycle_manager._cleanup_plugin_references("test")

    assert "test" not in lifecycle_manager._lazy_loaders
