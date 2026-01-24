"""Tests for Phase 1 Critical Infrastructure requirements.

Tests for:
- EXT-001 through EXT-006: Extension Points
- CFG-003 through CFG-018: Configuration Architecture
- PLUG-004 through PLUG-008: Plugin Lifecycle
"""

import tempfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


class TestExtensionPoints:
    """Tests for EXT-001 through EXT-006."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test to ensure isolation."""
        from oscura.extensibility.extensions import get_registry

        registry = get_registry()
        # Clear all algorithms and hooks registered during tests
        registry.clear_hooks()
        # Clear test-specific algorithms by category
        for category in ["test_category", "selection_test", "benchmark_category"]:
            try:
                if category in registry._algorithms:
                    registry._algorithms[category].clear()
            except (AttributeError, KeyError):
                pass
        yield

    def test_ext001_extension_point_registry(self):
        """Test EXT-001: Extension Point Registry."""
        from oscura.extensibility.extensions import (
            ExtensionPointSpec,
            get_registry,
        )

        registry = get_registry()

        # Registry should have built-in extension points
        points = registry.list_points()
        assert len(points) > 0

        # Check protocol_decoder exists
        spec = registry.get_point("protocol_decoder")
        assert spec.name == "protocol_decoder"
        assert spec.version == "1.0.0"
        assert "decode" in spec.required_methods

        # Register custom extension point
        custom_spec = ExtensionPointSpec(
            name="test_extension",
            version="1.0.0",
            description="Test extension point",
            required_methods=["process"],
        )
        registry.register_point(custom_spec)
        assert registry.exists("test_extension")

    def test_ext002_algorithm_registration(self):
        """Test EXT-002: Algorithm Registration."""
        from oscura.extensibility.extensions import get_registry

        registry = get_registry()

        def my_algorithm(data, param1=1.0):
            return data * param1

        registry.register_algorithm(
            name="test_algo",
            func=my_algorithm,
            category="test_category",
            priority=75,
            performance={"speed": "fast", "accuracy": "medium"},
            supports=["float32", "float64"],
        )

        algo = registry.get_algorithm("test_category", "test_algo")
        assert algo.name == "test_algo"
        assert algo.priority == 75
        assert algo.performance["speed"] == "fast"

    def test_ext003_algorithm_selection(self):
        """Test EXT-003: Algorithm Selection."""
        from oscura.extensibility.extensions import get_registry

        registry = get_registry()

        # Register multiple algorithms
        def fast_algo(data):
            return data

        def accurate_algo(data):
            return data * 1.0001

        registry.register_algorithm(
            name="fast_test",
            func=fast_algo,
            category="selection_test",
            performance={"speed": "fast", "accuracy": "low"},
        )

        registry.register_algorithm(
            name="accurate_test",
            func=accurate_algo,
            category="selection_test",
            performance={"speed": "slow", "accuracy": "high"},
        )

        # Select for speed
        selected = registry.select_algorithm("selection_test", optimize_for="speed")
        assert selected.name == "fast_test"

        # Select for accuracy
        selected = registry.select_algorithm("selection_test", optimize_for="accuracy")
        assert selected.name == "accurate_test"

    def test_ext004_priority_system(self):
        """Test EXT-004: Priority System."""
        from oscura.extensibility.extensions import get_registry

        registry = get_registry()

        def algo1(data):
            return data

        def algo2(data):
            return data

        registry.register_algorithm(
            name="low_priority", func=algo1, category="priority_test", priority=25
        )

        registry.register_algorithm(
            name="high_priority", func=algo2, category="priority_test", priority=75
        )

        # List ordered should have high priority first
        algos = registry.list_algorithms("priority_test", ordered=True)
        assert algos[0].name == "high_priority"
        assert algos[1].name == "low_priority"

        # Configure priorities
        registry.configure_priorities({"priority_test": {"low_priority": 100}})

        algos = registry.list_algorithms("priority_test", ordered=True)
        assert algos[0].name == "low_priority"

    def test_ext005_hook_system(self):
        """Test EXT-005: Hook System."""
        from oscura.extensibility.extensions import (
            HookContext,
            get_registry,
        )

        registry = get_registry()

        # Track hook execution
        executed = []

        def hook1(context):
            executed.append("hook1")
            context.metadata["hook1"] = True
            return context

        def hook2(context):
            executed.append("hook2")
            context.metadata["hook2"] = True
            return context

        registry.register_hook("test_hook", hook1, priority=50)
        registry.register_hook("test_hook", hook2, priority=75)

        # Execute hooks
        context = HookContext(data="test", metadata={})
        result = registry.execute_hooks("test_hook", context)

        # Higher priority should execute first
        assert executed[0] == "hook2"
        assert executed[1] == "hook1"
        assert result.metadata["hook1"] is True
        assert result.metadata["hook2"] is True

    def test_ext006_custom_decoder_registration(self):
        """Test EXT-006: Custom Decoder Registration."""
        from oscura.extensibility.extensions import get_registry

        registry = get_registry()

        class MyDecoder:
            """Test decoder for custom protocol registration.

            This decoder demonstrates EXT-006 requirement compliance
            by providing documentation of its purpose and usage.
            """

            def decode(self, trace):
                return []

            def get_metadata(self):
                return {"name": "my_decoder"}

        registry.register_decoder("my_protocol", MyDecoder)
        decoders = registry.list_decoders()
        assert "my_protocol" in decoders

        decoder_class = registry.get_decoder("my_protocol")
        instance = decoder_class()
        assert instance.get_metadata()["name"] == "my_decoder"


class TestConfigurationArchitecture:
    """Tests for CFG-003 through CFG-018."""

    @pytest.fixture(autouse=True)
    def reset_threshold_overrides(self):
        """Reset threshold overrides before each test to ensure isolation."""
        from oscura.config.thresholds import get_threshold_registry

        registry = get_threshold_registry()
        registry.reset_overrides()
        yield
        registry.reset_overrides()  # Clean up after test too

    def test_cfg003_protocol_registry(self):
        """Test CFG-003: Protocol Definition Registry."""
        from oscura.config.protocol import (
            get_protocol_registry,
        )

        registry = get_protocol_registry()

        # Built-in protocols
        protocols = registry.list()
        assert len(protocols) >= 4  # uart, spi, i2c, can

        uart = registry.get("uart")
        assert uart.name == "uart"
        assert 115200 in uart.timing.get("baud_rates", [])

        # Query capabilities
        caps = registry.get_capabilities("uart")
        assert caps.supports_digital is True

    def test_cfg004_protocol_inheritance(self):
        """Test CFG-004: Protocol Inheritance."""
        from oscura.config.protocol import (
            ProtocolDefinition,
            get_protocol_registry,
            resolve_inheritance,
        )

        registry = get_protocol_registry()

        # Create variant that extends uart
        variant = ProtocolDefinition(
            name="uart_custom",
            version="1.0.0",
            extends="uart",
            timing={"baud_rates": [9600]},  # Override baud rates
        )

        registry.register(variant)
        resolved = resolve_inheritance(variant, registry)

        # Should have parent's voltage_levels merged
        assert resolved.voltage_levels.get("logic_family") == "TTL"
        # But child's timing override
        assert resolved.timing["baud_rates"] == [9600]

    def test_cfg005_logic_family_config(self):
        """Test CFG-005: Logic Family Config Files."""
        from oscura.config.thresholds import get_threshold_registry

        registry = get_threshold_registry()

        # Built-in families
        families = registry.list_families()
        assert "TTL" in families
        assert "CMOS_5V" in families

        ttl = registry.get_family("TTL")
        assert ttl.VIH == 2.0
        assert ttl.VIL == 0.8
        assert ttl.noise_margin_high is not None

    def test_cfg006_threshold_override(self):
        """Test CFG-006: Threshold Override."""
        from oscura.config.thresholds import get_threshold_registry

        registry = get_threshold_registry()

        # Set session override
        registry.set_threshold_override(VIH=2.5, VIL=0.7)

        ttl = registry.get_family("TTL")
        assert ttl.VIH == 2.5
        assert ttl.VIL == 0.7

        # Reset
        registry.reset_overrides()
        ttl = registry.get_family("TTL")
        assert ttl.VIH == 2.0

    def test_cfg007_custom_logic_family(self):
        """Test CFG-007: Custom Logic Family Definitions."""
        from oscura.config.thresholds import (
            LogicFamily,
            get_threshold_registry,
        )

        registry = get_threshold_registry()

        custom = LogicFamily(
            name="my_custom",
            VIH=3.0,
            VIL=1.0,
            VOH=3.5,
            VOL=0.3,
            VCC=5.0,
            description="Custom logic family",
        )

        registry.register_family(custom)
        assert registry.get_family("user.my_custom").VIH == 3.0

    def test_cfg008_threshold_profiles(self):
        """Test CFG-008: Threshold Profiles."""
        from oscura.config.thresholds import get_threshold_registry

        registry = get_threshold_registry()

        # Built-in profiles
        strict = registry.get_profile("strict")
        assert strict.tolerance == 0

        relaxed = registry.get_profile("relaxed")
        assert relaxed.tolerance == 20

        # Apply profile
        family = registry.apply_profile("relaxed")
        assert "relaxed" in family.name

    def test_cfg009_pipeline_definition(self):
        """Test CFG-009: Pipeline Definition Files."""
        from oscura.config.pipeline import PipelineDefinition, PipelineStep

        step = PipelineStep(name="load", type="input.file", params={"path": "trace.bin"})

        pipeline = PipelineDefinition(name="test_pipeline", version="1.0.0", steps=[step])

        assert pipeline.name == "test_pipeline"
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].type == "input.file"

    def test_cfg010_pipeline_loader(self):
        """Test CFG-010: Pipeline Loader."""
        from oscura.config.pipeline import Pipeline, load_pipeline

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "name": "test",
                    "version": "1.0.0",
                    "steps": [{"name": "step1", "type": "input.file", "params": {}}],
                },
                f,
            )
            f.flush()

            pipeline_def = load_pipeline(f.name)
            assert pipeline_def.name == "test"
            assert len(pipeline_def.steps) == 1

            pipeline = Pipeline(pipeline_def)
            # Dry run should work
            result = pipeline.execute(dry_run=True)
            assert result.success

    def test_cfg011_pipeline_templates(self):
        """Test CFG-011: Pipeline Templates."""
        from oscura.config.pipeline import _substitute_variables

        template = "path: ${INPUT_FILE}, rate: ${SAMPLE_RATE}"
        variables = {"INPUT_FILE": "trace.bin", "SAMPLE_RATE": "1e9"}

        result = _substitute_variables(template, variables)
        assert result == "path: trace.bin, rate: 1e9"

    def test_cfg012_pipeline_composition(self):
        """Test CFG-012: Pipeline Composition."""
        from oscura.config.pipeline import PipelineDefinition

        pipeline = PipelineDefinition(name="main", steps=[], includes=["common/base.yaml"])

        assert len(pipeline.includes) == 1

    def test_cfg013_conditional_steps(self):
        """Test CFG-013: Conditional Pipeline Steps."""
        from oscura.config.pipeline import Pipeline, PipelineDefinition, PipelineStep

        step = PipelineStep(
            name="conditional", type="analysis.statistics", condition="confidence > 0.8"
        )

        pipeline = Pipeline(PipelineDefinition(name="test", steps=[step]))

        # Test condition evaluation
        pipeline._state["confidence"] = 0.9
        assert pipeline._evaluate_condition("confidence > 0.8") is True

        pipeline._state["confidence"] = 0.5
        assert pipeline._evaluate_condition("confidence > 0.8") is False

    def test_cfg018_preferences_persistence(self):
        """Test CFG-018: Preferences Persistence."""
        from oscura.config.preferences import (
            PreferencesManager,
            UserPreferences,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            manager = PreferencesManager(Path(f.name))

            # Create and save preferences
            prefs = UserPreferences()
            prefs.visualization.dark_mode = True
            prefs.defaults.sample_rate = 2e9

            manager.save(prefs)

            # Load and verify
            loaded = manager.load(use_cache=False)
            assert loaded.visualization.dark_mode is True
            assert loaded.defaults.sample_rate == 2e9


class TestPluginLifecycle:
    """Tests for PLUG-004 through PLUG-008."""

    def test_plug004_lifecycle_states(self):
        """Test PLUG-004: Plugin Lifecycle (enable/disable/reload)."""
        from oscura.plugins.lifecycle import PluginState

        # All states should be defined
        assert PluginState.DISCOVERED
        assert PluginState.LOADING
        assert PluginState.LOADED
        assert PluginState.CONFIGURED
        assert PluginState.ENABLED
        assert PluginState.DISABLED
        assert PluginState.ERROR
        assert PluginState.UNLOADING

    def test_plug005_dependency_resolution(self):
        """Test PLUG-005: Dependency Resolution."""
        from oscura.plugins.lifecycle import DependencyGraph

        graph = DependencyGraph()

        graph.add_plugin("core")
        graph.add_plugin("decoder")
        graph.add_plugin("ui")

        graph.add_dependency("decoder", "core")
        graph.add_dependency("ui", "decoder")

        order = graph.resolve_order()

        # Core must come before decoder, decoder before ui
        assert order.index("core") < order.index("decoder")
        assert order.index("decoder") < order.index("ui")

    def test_plug005_circular_dependency_detection(self):
        """Test PLUG-005: Circular dependency detection."""
        from oscura.plugins.lifecycle import DependencyGraph

        graph = DependencyGraph()

        graph.add_dependency("a", "b")
        graph.add_dependency("b", "c")
        graph.add_dependency("c", "a")  # Circular!

        with pytest.raises(ValueError, match="Circular"):
            graph.resolve_order()

    def test_plug006_graceful_degradation(self):
        """Test PLUG-006: Graceful Degradation."""
        from oscura.plugins.base import PluginMetadata
        from oscura.plugins.lifecycle import (
            PluginHandle,
            PluginLifecycleManager,
            PluginLoadError,
            PluginState,
        )

        manager = PluginLifecycleManager()

        # Add a failed plugin handle
        handle = PluginHandle(
            metadata=PluginMetadata(name="failed_plugin", version="1.0.0"),
            state=PluginState.ERROR,
            errors=[
                PluginLoadError(
                    plugin_name="failed_plugin",
                    error=ImportError("Test error"),
                    recoverable=True,
                )
            ],
        )
        manager._handles["failed_plugin"] = handle

        # Get degradation info
        info = manager.graceful_degradation("failed_plugin")
        assert info["status"] == "degraded"
        assert info["recoverable"] is True

    def test_plug007_lazy_loading(self):
        """Test PLUG-007: Lazy Loading."""
        from oscura.plugins.lifecycle import PluginLifecycleManager, PluginState

        manager = PluginLifecycleManager()

        # Register a lazy loader
        loaded = [False]

        def lazy_loader():
            loaded[0] = True
            from oscura.plugins.base import PluginBase

            class TestPlugin(PluginBase):
                name = "lazy_test"
                version = "1.0.0"

            return TestPlugin()

        from oscura.plugins.base import PluginMetadata
        from oscura.plugins.lifecycle import PluginHandle

        handle = PluginHandle(
            metadata=PluginMetadata(name="lazy_test", version="1.0.0"),
            state=PluginState.DISCOVERED,
        )
        manager._handles["lazy_test"] = handle
        manager._lazy_loaders["lazy_test"] = lazy_loader

        # Not loaded yet
        assert loaded[0] is False

        # Load triggers lazy loader
        manager.load_plugin("lazy_test")
        assert loaded[0] is True

    def test_plug008_hot_reload(self):
        """Test PLUG-008: Plugin Hot Reload."""
        from oscura.plugins.lifecycle import PluginLifecycleManager

        manager = PluginLifecycleManager()

        # check_for_changes should return empty list with no watched files
        changed = manager.check_for_changes()
        assert changed == []


class TestRequirementsPhase1InfrastructureIntegration:
    """Integration tests for Phase 1."""

    def test_extension_and_plugin_integration(self):
        """Test extension points work with plugins."""
        from oscura.extensibility.extensions import get_registry
        from oscura.plugins.base import PluginBase, PluginCapability

        registry = get_registry()

        # Simulate plugin registering at extension point
        class TestPlugin(PluginBase):
            name = "test_integration"
            version = "1.0.0"
            capabilities = [PluginCapability.ALGORITHM]

            def on_load(self):
                def my_algo(data):
                    return data * 2

                registry.register_algorithm(
                    name="plugin_algo", func=my_algo, category="plugin_test"
                )

        plugin = TestPlugin()
        plugin.on_load()

        # Algorithm should be available
        algo = registry.get_algorithm("plugin_test", "plugin_algo")
        assert algo.func([1, 2]) == [1, 2, 1, 2]  # list * 2

    def test_config_and_extension_integration(self):
        """Test configuration works with extensions."""
        from oscura.config.protocol import get_protocol_registry
        from oscura.extensibility.extensions import get_registry

        ext_registry = get_registry()
        proto_registry = get_protocol_registry()

        # Register decoder via extension system
        class UartDecoder:
            """UART protocol decoder for testing config/extension integration.

            This decoder demonstrates integration between protocol configuration
            and extension registration systems (EXT-006 requirement).
            """

            def decode(self, trace):
                return []

            def get_metadata(self):
                proto = proto_registry.get("uart")
                return {"name": "uart", "baud_rates": proto.timing.get("baud_rates")}

        ext_registry.register_decoder("uart_ext", UartDecoder)

        # Decoder should use protocol config
        decoder_class = ext_registry.get_decoder("uart_ext")
        instance = decoder_class()
        meta = instance.get_metadata()
        assert 115200 in meta["baud_rates"]
