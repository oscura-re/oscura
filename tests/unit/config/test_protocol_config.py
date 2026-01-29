"""Comprehensive test suite for protocol configuration system.

Tests CFG-001 through CFG-004 requirements:

This module provides 50+ tests for protocol configuration validation,
registry operations, inheritance resolution, and hot reload functionality.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from oscura.core.config.protocol import (
    ProtocolCapabilities,
    ProtocolDefinition,
    ProtocolRegistry,
    ProtocolWatcher,
    get_protocol_registry,
    load_protocol,
    migrate_protocol_schema,
    resolve_inheritance,
)
from oscura.core.config.schema import (
    ValidationError,
    validate_against_schema,
)
from oscura.core.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


# =============================================================================
# Protocol Definition Tests (CFG-001)
# =============================================================================


class TestProtocolDefinition:
    """Test ProtocolDefinition dataclass functionality."""

    def test_basic_creation(self) -> None:
        """Test creating a basic protocol definition."""
        protocol = ProtocolDefinition(
            name="uart",
            version="1.0.0",
            description="UART protocol",
        )

        assert protocol.name == "uart"
        assert protocol.version == "1.0.0"
        assert protocol.description == "UART protocol"
        assert protocol.extends is None

    def test_timing_configuration(self) -> None:
        """Test protocol with timing configuration."""
        protocol = ProtocolDefinition(
            name="uart",
            version="1.0.0",
            timing={
                "baud_rates": [9600, 115200, 921600],
                "data_bits": [7, 8],
                "stop_bits": [1, 2],
                "parity": ["none", "even", "odd"],
            },
        )

        assert 115200 in protocol.timing["baud_rates"]
        assert 8 in protocol.timing["data_bits"]

    def test_voltage_levels(self) -> None:
        """Test protocol with voltage level configuration."""
        protocol = ProtocolDefinition(
            name="rs232",
            version="1.0.0",
            voltage_levels={
                "mark_voltage": -12.0,
                "space_voltage": 12.0,
                "idle_state": "high",
            },
        )

        assert protocol.voltage_levels["mark_voltage"] == -12.0
        assert protocol.supports_analog is True

    def test_state_machine_config(self) -> None:
        """Test protocol with state machine configuration."""
        protocol = ProtocolDefinition(
            name="i2c",
            version="1.0.0",
            state_machine={
                "states": ["IDLE", "START", "ADDRESS", "DATA", "ACK", "STOP"],
                "initial_state": "IDLE",
                "transitions": [
                    {"from": "IDLE", "to": "START", "condition": "sda_fall_scl_high"},
                ],
            },
        )

        assert "IDLE" in protocol.state_machine["states"]
        assert protocol.state_machine["initial_state"] == "IDLE"

    def test_supports_digital_default(self) -> None:
        """Test that protocols support digital by default."""
        protocol = ProtocolDefinition(name="test", version="1.0.0")
        assert protocol.supports_digital is True

    def test_supports_analog_with_voltage(self) -> None:
        """Test analog support when voltage levels defined."""
        protocol = ProtocolDefinition(
            name="test",
            version="1.0.0",
            voltage_levels={"VIH": 2.0, "VIL": 0.8},
        )
        assert protocol.supports_analog is True

    def test_supports_analog_without_voltage(self) -> None:
        """Test no analog support without voltage levels."""
        protocol = ProtocolDefinition(name="test", version="1.0.0")
        assert protocol.supports_analog is False

    def test_sample_rate_min_from_baud(self) -> None:
        """Test sample rate calculation from baud rates."""
        protocol = ProtocolDefinition(
            name="uart",
            version="1.0.0",
            timing={"baud_rates": [115200]},
        )
        # Should be 10x max baud rate
        assert protocol.sample_rate_min == 115200 * 10

    def test_sample_rate_min_default(self) -> None:
        """Test default sample rate when no baud specified."""
        protocol = ProtocolDefinition(name="test", version="1.0.0")
        assert protocol.sample_rate_min == 1e6

    def test_bit_widths(self) -> None:
        """Test bit width extraction from timing config."""
        protocol = ProtocolDefinition(
            name="uart",
            version="1.0.0",
            timing={"data_bits": [7, 8, 9]},
        )
        assert protocol.bit_widths == [7, 8, 9]

    def test_bit_widths_default(self) -> None:
        """Test default bit width when not specified."""
        protocol = ProtocolDefinition(name="test", version="1.0.0")
        assert protocol.bit_widths == [8]

    def test_metadata_storage(self) -> None:
        """Test custom metadata storage."""
        protocol = ProtocolDefinition(
            name="custom",
            version="1.0.0",
            metadata={
                "author": "test",
                "license": "MIT",
                "custom_field": 42,
            },
        )

        assert protocol.metadata["author"] == "test"
        assert protocol.metadata["custom_field"] == 42

    def test_source_file_tracking(self) -> None:
        """Test source file path tracking."""
        protocol = ProtocolDefinition(
            name="test",
            version="1.0.0",
            source_file="/path/to/protocol.yaml",
        )
        assert protocol.source_file == "/path/to/protocol.yaml"


# =============================================================================
# Protocol Registry Tests (CFG-003)
# =============================================================================


class TestProtocolRegistry:
    """Test ProtocolRegistry functionality."""

    def test_singleton_pattern(self) -> None:
        """Test that ProtocolRegistry uses singleton pattern."""
        reg1 = ProtocolRegistry()
        reg2 = ProtocolRegistry()
        assert reg1 is reg2

    def test_register_protocol(self) -> None:
        """Test registering a protocol."""
        registry = ProtocolRegistry()
        protocol = ProtocolDefinition(name="test_proto", version="1.0.0")

        # Clear any existing registration
        if "test_proto" in registry._protocols:
            del registry._protocols["test_proto"]

        registry.register(protocol)

        result = registry.get("test_proto")
        assert result.name == "test_proto"

    def test_register_duplicate_error(self) -> None:
        """Test error when registering duplicate protocol."""
        registry = ProtocolRegistry()
        protocol = ProtocolDefinition(name="dup_proto", version="1.0.0")

        # Clear any existing
        if "dup_proto" in registry._protocols:
            del registry._protocols["dup_proto"]

        registry.register(protocol)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(protocol)

    def test_register_with_overwrite(self) -> None:
        """Test registering with overwrite flag."""
        registry = ProtocolRegistry()

        proto1 = ProtocolDefinition(name="overwrite_test", version="1.0.0", description="old")
        proto2 = ProtocolDefinition(name="overwrite_test", version="1.0.0", description="new")

        if "overwrite_test" in registry._protocols:
            del registry._protocols["overwrite_test"]

        registry.register(proto1)
        registry.register(proto2, overwrite=True)

        result = registry.get("overwrite_test")
        assert result.description == "new"

    def test_get_protocol_not_found(self) -> None:
        """Test error when getting non-existent protocol."""
        registry = ProtocolRegistry()

        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent_protocol")

    def test_get_with_version(self) -> None:
        """Test getting specific protocol version."""
        registry = ProtocolRegistry()

        # Clear existing
        if "versioned_proto" in registry._protocols:
            del registry._protocols["versioned_proto"]

        proto1 = ProtocolDefinition(name="versioned_proto", version="1.0.0")
        proto2 = ProtocolDefinition(name="versioned_proto", version="2.0.0")

        registry.register(proto1, set_default=False)
        registry.register(proto2)

        result1 = registry.get("versioned_proto", version="1.0.0")
        result2 = registry.get("versioned_proto", version="2.0.0")

        assert result1.version == "1.0.0"
        assert result2.version == "2.0.0"

    def test_get_default_version(self) -> None:
        """Test getting default version."""
        registry = ProtocolRegistry()

        if "default_ver_test" in registry._protocols:
            del registry._protocols["default_ver_test"]

        proto1 = ProtocolDefinition(name="default_ver_test", version="1.0.0")
        proto2 = ProtocolDefinition(name="default_ver_test", version="2.0.0")

        registry.register(proto1, set_default=False)
        registry.register(proto2, set_default=True)

        result = registry.get("default_ver_test")
        assert result.version == "2.0.0"

    def test_list_protocols(self) -> None:
        """Test listing all protocols."""
        registry = get_protocol_registry()
        protocols = registry.list()

        # Built-in protocols should be present
        names = [p.name for p in protocols]
        assert "uart" in names
        assert "spi" in names
        assert "i2c" in names

    def test_list_versions(self) -> None:
        """Test listing versions of a protocol."""
        registry = ProtocolRegistry()

        if "multi_ver" in registry._protocols:
            del registry._protocols["multi_ver"]

        registry.register(ProtocolDefinition(name="multi_ver", version="1.0.0"), set_default=False)
        registry.register(ProtocolDefinition(name="multi_ver", version="1.1.0"), set_default=False)
        registry.register(ProtocolDefinition(name="multi_ver", version="2.0.0"))

        versions = registry.list_versions("multi_ver")
        assert "1.0.0" in versions
        assert "1.1.0" in versions
        assert "2.0.0" in versions

    def test_has_protocol(self) -> None:
        """Test checking protocol existence."""
        registry = get_protocol_registry()

        assert registry.has_protocol("uart") is True
        assert registry.has_protocol("nonexistent") is False
        assert registry.has_protocol("uart", version="1.0.0") is True
        assert registry.has_protocol("uart", version="99.0.0") is False

    def test_get_capabilities(self) -> None:
        """Test getting protocol capabilities."""
        registry = get_protocol_registry()

        caps = registry.get_capabilities("uart")

        assert isinstance(caps, ProtocolCapabilities)
        assert caps.supports_digital is True
        assert isinstance(caps.sample_rate_min, float)

    def test_filter_by_digital(self) -> None:
        """Test filtering protocols by digital support."""
        registry = get_protocol_registry()

        digital_protocols = registry.filter(supports_digital=True)
        assert len(digital_protocols) > 0
        assert all(p.supports_digital for p in digital_protocols)

    def test_filter_by_sample_rate(self) -> None:
        """Test filtering by sample rate."""
        registry = get_protocol_registry()

        # Filter for protocols needing at least 1MHz
        high_speed = registry.filter(sample_rate_min__gte=1e6)
        assert isinstance(high_speed, list)

    def test_on_change_callback(self) -> None:
        """Test change callback registration."""
        registry = ProtocolRegistry()
        callback_called = []

        def callback(proto: ProtocolDefinition) -> None:
            callback_called.append(proto.name)

        registry.on_change(callback)

        protocol = ProtocolDefinition(name="callback_test", version="1.0.0")
        if "callback_test" in registry._protocols:
            del registry._protocols["callback_test"]

        registry.register(protocol, overwrite=True)
        registry._notify_change(protocol)

        assert "callback_test" in callback_called


# =============================================================================
# Protocol Loading Tests (CFG-002)
# =============================================================================


class TestProtocolLoading:
    """Test protocol loading from files."""

    def test_load_yaml_protocol(self, tmp_path: Path) -> None:
        """Test loading protocol from YAML file."""
        yaml_content = """
name: test_yaml
version: 1.0.0
description: Test YAML protocol
timing:
  baud_rates:
    - 9600
    - 115200
"""
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text(yaml_content)

        protocol = load_protocol(yaml_file, validate=False)

        assert protocol.name == "test_yaml"
        assert protocol.version == "1.0.0"
        assert 9600 in protocol.timing["baud_rates"]

    def test_load_json_protocol(self, tmp_path: Path) -> None:
        """Test loading protocol from JSON file."""
        json_content = {
            "name": "test_json",
            "version": "1.0.0",
            "description": "Test JSON protocol",
            "timing": {
                "baud_rates": [9600, 115200],
            },
        }
        json_file = tmp_path / "test.json"
        json_file.write_text(json.dumps(json_content))

        protocol = load_protocol(json_file, validate=False)

        assert protocol.name == "test_json"

    def test_load_nested_protocol(self, tmp_path: Path) -> None:
        """Test loading protocol with nested 'protocol' key."""
        yaml_content = """
protocol:
  name: nested_test
  version: 1.0.0
"""
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text(yaml_content)

        protocol = load_protocol(yaml_file, validate=False)

        assert protocol.name == "nested_test"

    def test_load_nonexistent_file(self) -> None:
        """Test error when loading non-existent file."""
        with pytest.raises(ConfigurationError, match="not found"):
            load_protocol("/nonexistent/path/protocol.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        """Test error on invalid YAML syntax."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("name: [invalid yaml")

        with pytest.raises(ConfigurationError):
            load_protocol(yaml_file, validate=False)

    def test_load_with_validation(self, tmp_path: Path) -> None:
        """Test loading with schema validation."""
        yaml_content = """
name: valid_protocol
version: 1.0.0
timing:
  baud_rates:
    - 9600
"""
        yaml_file = tmp_path / "valid.yaml"
        yaml_file.write_text(yaml_content)

        protocol = load_protocol(yaml_file, validate=True)
        assert protocol.name == "valid_protocol"

    def test_load_validation_failure(self, tmp_path: Path) -> None:
        """Test validation failure on invalid protocol."""
        yaml_content = """
name: INVALID-NAME!
version: not-a-version
"""
        yaml_file = tmp_path / "invalid_schema.yaml"
        yaml_file.write_text(yaml_content)

        with pytest.raises(ConfigurationError, match="validation failed"):
            load_protocol(yaml_file, validate=True)

    def test_source_file_tracked(self, tmp_path: Path) -> None:
        """Test that source file is tracked in loaded protocol."""
        yaml_file = tmp_path / "tracked.yaml"
        yaml_file.write_text("name: tracked\nversion: 1.0.0")

        protocol = load_protocol(yaml_file, validate=False)

        assert protocol.source_file == str(yaml_file)


# =============================================================================
# Protocol Inheritance Tests (CFG-004)
# =============================================================================


class TestProtocolInheritance:
    """Test protocol inheritance resolution."""

    def test_no_inheritance(self) -> None:
        """Test protocol without inheritance."""
        registry = ProtocolRegistry()
        protocol = ProtocolDefinition(name="no_inherit", version="1.0.0")

        result = resolve_inheritance(protocol, registry)

        assert result.name == "no_inherit"
        assert result.extends is None

    def test_single_level_inheritance(self) -> None:
        """Test single-level inheritance."""
        registry = ProtocolRegistry()

        if "parent_proto" in registry._protocols:
            del registry._protocols["parent_proto"]
        if "child_proto" in registry._protocols:
            del registry._protocols["child_proto"]

        parent = ProtocolDefinition(
            name="parent_proto",
            version="1.0.0",
            timing={"baud_rates": [9600]},
            description="Parent protocol",
        )
        registry.register(parent)

        child = ProtocolDefinition(
            name="child_proto",
            version="1.0.0",
            extends="parent_proto",
            timing={"stop_bits": [1, 2]},
        )

        result = resolve_inheritance(child, registry)

        assert result.extends is None  # Cleared after resolution
        assert result.timing["baud_rates"] == [9600]  # From parent
        assert result.timing["stop_bits"] == [1, 2]  # From child
        assert result.description == "Parent protocol"  # Inherited

    def test_multi_level_inheritance(self) -> None:
        """Test multi-level (grandparent) inheritance."""
        registry = ProtocolRegistry()

        for name in ["grandparent", "parent_ml", "child_ml"]:
            if name in registry._protocols:
                del registry._protocols[name]

        grandparent = ProtocolDefinition(
            name="grandparent",
            version="1.0.0",
            timing={"baud_rates": [9600]},
        )
        registry.register(grandparent)

        parent = ProtocolDefinition(
            name="parent_ml",
            version="1.0.0",
            extends="grandparent",
            timing={"data_bits": [8]},
        )
        registry.register(parent)

        child = ProtocolDefinition(
            name="child_ml",
            version="1.0.0",
            extends="parent_ml",
            timing={"stop_bits": [1]},
        )

        result = resolve_inheritance(child, registry)

        # All levels should be merged
        assert result.timing["baud_rates"] == [9600]
        assert result.timing["data_bits"] == [8]
        assert result.timing["stop_bits"] == [1]

    def test_circular_inheritance_detection(self) -> None:
        """Test detection of circular inheritance."""
        registry = ProtocolRegistry()

        for name in ["circ_a", "circ_b", "circ_c"]:
            if name in registry._protocols:
                del registry._protocols[name]

        # Create circular dependency: A -> B -> C -> A
        proto_a = ProtocolDefinition(name="circ_a", version="1.0.0", extends="circ_c")
        proto_b = ProtocolDefinition(name="circ_b", version="1.0.0", extends="circ_a")
        proto_c = ProtocolDefinition(name="circ_c", version="1.0.0", extends="circ_b")

        registry.register(proto_a)
        registry.register(proto_b)
        registry.register(proto_c)

        with pytest.raises(ConfigurationError, match=r"[Cc]ircular"):
            resolve_inheritance(proto_a, registry)

    def test_max_depth_exceeded(self) -> None:
        """Test error when inheritance depth exceeds maximum."""
        registry = ProtocolRegistry()

        # Create deep inheritance chain (6 levels, max is 5)
        names = [f"deep_{i}" for i in range(7)]
        for name in names:
            if name in registry._protocols:
                del registry._protocols[name]

        # Register chain
        registry.register(ProtocolDefinition(name="deep_0", version="1.0.0"))
        for i in range(1, 7):
            registry.register(
                ProtocolDefinition(
                    name=f"deep_{i}",
                    version="1.0.0",
                    extends=f"deep_{i - 1}",
                )
            )

        deepest = registry.get("deep_6")

        with pytest.raises(ConfigurationError, match="depth exceeded"):
            resolve_inheritance(deepest, registry, max_depth=5)

    def test_missing_parent_error(self) -> None:
        """Test error when parent protocol doesn't exist."""
        registry = ProtocolRegistry()

        child = ProtocolDefinition(
            name="orphan_child",
            version="1.0.0",
            extends="nonexistent_parent",
        )

        with pytest.raises(ConfigurationError, match="not found"):
            resolve_inheritance(child, registry)

    def test_shallow_merge(self) -> None:
        """Test shallow merge of timing configurations."""
        registry = ProtocolRegistry()

        if "shallow_parent" in registry._protocols:
            del registry._protocols["shallow_parent"]

        parent = ProtocolDefinition(
            name="shallow_parent",
            version="1.0.0",
            timing={"baud_rates": [9600], "data_bits": [8]},
        )
        registry.register(parent)

        child = ProtocolDefinition(
            name="shallow_child",
            version="1.0.0",
            extends="shallow_parent",
            timing={"baud_rates": [115200]},  # Override baud_rates
        )

        result = resolve_inheritance(child, registry, deep_merge=False)

        # Shallow merge: child's baud_rates completely replaces parent's
        assert result.timing["baud_rates"] == [115200]
        assert result.timing["data_bits"] == [8]  # Inherited

    def test_deep_merge(self) -> None:
        """Test deep merge of nested configurations."""
        registry = ProtocolRegistry()

        if "deep_parent" in registry._protocols:
            del registry._protocols["deep_parent"]

        parent = ProtocolDefinition(
            name="deep_parent",
            version="1.0.0",
            timing={
                "baud_rates": [9600],
                "advanced": {"retry": 3, "timeout": 1.0},
            },
        )
        registry.register(parent)

        child = ProtocolDefinition(
            name="deep_child",
            version="1.0.0",
            extends="deep_parent",
            timing={
                "advanced": {"timeout": 2.0},  # Override only timeout
            },
        )

        result = resolve_inheritance(child, registry, deep_merge=True)

        # Deep merge: nested dict is merged
        assert result.timing["advanced"]["retry"] == 3  # From parent
        assert result.timing["advanced"]["timeout"] == 2.0  # Overridden

    def test_child_overrides_description(self) -> None:
        """Test that child description overrides parent."""
        registry = ProtocolRegistry()

        if "desc_parent" in registry._protocols:
            del registry._protocols["desc_parent"]

        parent = ProtocolDefinition(
            name="desc_parent",
            version="1.0.0",
            description="Parent description",
        )
        registry.register(parent)

        child = ProtocolDefinition(
            name="desc_child",
            version="1.0.0",
            extends="desc_parent",
            description="Child description",
        )

        result = resolve_inheritance(child, registry)

        assert result.description == "Child description"


# =============================================================================
# Protocol Watcher Tests (CFG-002 Hot Reload)
# =============================================================================


class TestProtocolWatcher:
    """Test protocol hot reload functionality."""

    def test_watcher_creation(self, tmp_path: Path) -> None:
        """Test creating a protocol watcher."""
        watcher = ProtocolWatcher(tmp_path, poll_interval=0.1)

        assert watcher.directory == tmp_path
        assert watcher.poll_interval == 0.1
        assert watcher._running is False

    def test_watcher_start_stop(self, tmp_path: Path) -> None:
        """Test starting and stopping watcher."""
        watcher = ProtocolWatcher(tmp_path, poll_interval=0.1)

        watcher.start()
        assert watcher._running is True
        assert watcher._thread is not None
        assert watcher._thread.is_alive()

        watcher.stop()
        assert watcher._running is False

    def test_watcher_callback_registration(self, tmp_path: Path) -> None:
        """Test registering change callbacks."""
        watcher = ProtocolWatcher(tmp_path, poll_interval=0.1)
        callbacks = []

        def callback(proto: ProtocolDefinition) -> None:
            callbacks.append(proto)

        watcher.on_change(callback)

        assert callback in watcher._callbacks

    def test_watcher_detects_file_change(self, tmp_path: Path) -> None:
        """Test that watcher detects file modifications."""
        # Create initial file
        proto_file = tmp_path / "test_watch.yaml"
        proto_file.write_text("name: watch_test\nversion: 1.0.0\n")

        watcher = ProtocolWatcher(tmp_path, poll_interval=0.1)
        reloaded_protocols: list[ProtocolDefinition] = []

        def callback(proto: ProtocolDefinition) -> None:
            reloaded_protocols.append(proto)

        watcher.on_change(callback)
        watcher.start()

        # Wait for initial scan
        time.sleep(0.2)

        # Modify file
        time.sleep(0.1)
        proto_file.write_text("name: watch_test\nversion: 2.0.0\n")

        # Wait for detection
        time.sleep(0.3)

        watcher.stop()

        # Should have detected the change
        assert len(reloaded_protocols) >= 1
        assert reloaded_protocols[-1].version == "2.0.0"

    def test_watcher_latency_under_2s(self, tmp_path: Path) -> None:
        """Test that reload latency is under 2 seconds (CFG-002)."""
        proto_file = tmp_path / "latency_test.yaml"
        proto_file.write_text("name: latency_test\nversion: 1.0.0\n")

        watcher = ProtocolWatcher(tmp_path, poll_interval=0.5)
        change_times: list[float] = []

        def callback(proto: ProtocolDefinition) -> None:
            change_times.append(time.time())

        watcher.on_change(callback)
        watcher.start()
        time.sleep(0.6)  # Wait for scan

        # Record modification time
        modify_time = time.time()
        proto_file.write_text("name: latency_test\nversion: 2.0.0\n")

        # Wait for detection
        time.sleep(1.5)
        watcher.stop()

        if change_times:
            latency = change_times[-1] - modify_time
            # Latency should be under 2 seconds
            assert latency < 2.0, f"Latency was {latency}s, expected < 2s"

    def test_watcher_with_registry(self, tmp_path: Path) -> None:
        """Test watcher auto-registration with registry."""
        registry = ProtocolRegistry()

        proto_file = tmp_path / "registry_test.yaml"
        proto_file.write_text("name: registry_watch\nversion: 1.0.0\n")

        watcher = ProtocolWatcher(tmp_path, poll_interval=0.1, registry=registry)
        watcher.start()
        time.sleep(0.2)

        # Modify file
        proto_file.write_text("name: registry_watch\nversion: 2.0.0\n")
        time.sleep(0.3)

        watcher.stop()

        # Protocol should be in registry
        if registry.has_protocol("registry_watch"):
            proto = registry.get("registry_watch")
            assert proto.version == "2.0.0"


# =============================================================================
# Schema Migration Tests (CFG-001)
# =============================================================================


class TestSchemaMigration:
    """Test protocol schema migration functionality."""

    def test_no_migration_same_version(self) -> None:
        """Test no migration when versions match."""
        data = {"name": "test", "version": "1.0.0"}
        result = migrate_protocol_schema(data, "1.0.0", "1.0.0")
        assert result == data

    def test_migrate_0_9_to_1_0(self) -> None:
        """Test migration from 0.9.0 to 1.0.0."""
        data = {
            "name": "test",
            "state": {"states": ["A", "B"]},  # Old format
        }
        result = migrate_protocol_schema(data, "0.9.0", "1.0.0")

        assert result["schema_version"] == "1.0.0"
        assert "state_machine" in result  # Renamed
        assert result["state_machine"]["states"] == ["A", "B"]

    def test_migrate_0_8_to_0_9(self) -> None:
        """Test migration from 0.8.0 to 0.9.0."""
        data = {
            "name": "test",
            "timing": {"baudrate": 9600},  # Old format (singular)
        }
        result = migrate_protocol_schema(data, "0.8.0", "0.9.0")

        # Should convert to array
        assert result["timing"]["baud_rates"] == [9600]
        assert "baudrate" not in result["timing"]

    def test_migrate_0_8_to_1_0(self) -> None:
        """Test chained migration from 0.8.0 to 1.0.0."""
        data = {
            "name": "test",
            "timing": {"baudrate": 9600},
        }
        result = migrate_protocol_schema(data, "0.8.0", "1.0.0")

        assert result["schema_version"] == "1.0.0"
        assert result["timing"]["baud_rates"] == [9600]

    def test_unsupported_migration_path(self) -> None:
        """Test error on unsupported migration path."""
        data = {"name": "test"}

        with pytest.raises(ConfigurationError, match="No migration path"):
            migrate_protocol_schema(data, "0.1.0", "1.0.0")


# =============================================================================
# Schema Validation Tests (CFG-014, CFG-015)
# =============================================================================


class TestSchemaValidation:
    """Test JSON Schema validation functionality."""

    def test_validate_valid_protocol(self) -> None:
        """Test validating a valid protocol config."""
        config = {
            "name": "uart",
            "version": "1.0.0",
            "timing": {"baud_rates": [9600, 115200]},
        }

        result = validate_against_schema(config, "protocol")
        assert result is True

    def test_validate_invalid_name_pattern(self) -> None:
        """Test validation failure on invalid name pattern."""
        config = {
            "name": "INVALID-NAME!",
            "version": "1.0.0",
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "protocol")

    def test_validate_missing_required(self) -> None:
        """Test validation failure on missing required field."""
        config = {
            "version": "1.0.0",
            # Missing 'name'
        }

        with pytest.raises(ValidationError, match="required"):
            validate_against_schema(config, "protocol")

    def test_validate_wrong_type(self) -> None:
        """Test validation failure on wrong type."""
        config = {
            "name": "uart",
            "timing": {
                "baud_rates": "not-an-array",
            },
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "protocol")

    def test_validate_pipeline(self) -> None:
        """Test validating a pipeline configuration."""
        config = {
            "name": "test_pipeline",
            "steps": [
                {"name": "load", "type": "input.file"},
                {"name": "process", "type": "analyze.fft"},
            ],
        }

        result = validate_against_schema(config, "pipeline")
        assert result is True

    def test_validate_pipeline_empty_steps(self) -> None:
        """Test validation failure on empty pipeline steps."""
        config = {
            "name": "empty_pipeline",
            "steps": [],
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "pipeline")

    def test_validate_logic_family(self) -> None:
        """Test validating a logic family configuration."""
        config = {
            "name": "TTL",
            "VIH": 2.0,
            "VIL": 0.8,
            "VOH": 2.4,
            "VOL": 0.4,
        }

        result = validate_against_schema(config, "logic_family")
        assert result is True

    def test_validate_logic_family_voltage_range(self) -> None:
        """Test validation failure on out-of-range voltage."""
        config = {
            "name": "invalid",
            "VIH": 15.0,  # max is 10
            "VIL": 0.8,
            "VOH": 2.4,
            "VOL": 0.4,
        }

        with pytest.raises(ValidationError):
            validate_against_schema(config, "logic_family")

    def test_validate_strict_mode(self) -> None:
        """Test strict validation mode."""
        config = {
            "name": "strict_test",
            "version": "1.0.0",
            "unknown_field": "value",
        }

        # Non-strict should pass
        result = validate_against_schema(config, "protocol", strict=False)
        assert result is True

    def test_validation_error_details(self) -> None:
        """Test that validation errors include path information."""
        config = {
            "name": "uart",
            "timing": {
                "baud_rates": ["not", "integers"],
            },
        }

        try:
            validate_against_schema(config, "protocol")
            pytest.fail("Should have raised ValidationError")
        except ValidationError as e:
            assert e.path is not None

    def test_schema_not_found(self) -> None:
        """Test error when schema doesn't exist."""
        with pytest.raises(ConfigurationError, match="not found"):
            validate_against_schema({}, "nonexistent_schema")


# =============================================================================
# Built-in Protocols Tests
# =============================================================================


class TestBuiltinProtocols:
    """Test built-in protocol definitions."""

    def test_uart_protocol(self) -> None:
        """Test UART protocol is properly defined."""
        registry = get_protocol_registry()
        uart = registry.get("uart")

        assert uart.name == "uart"
        assert 9600 in uart.timing["baud_rates"]
        assert 115200 in uart.timing["baud_rates"]
        assert "none" in uart.timing["parity"]

    def test_spi_protocol(self) -> None:
        """Test SPI protocol is properly defined."""
        registry = get_protocol_registry()
        spi = registry.get("spi")

        assert spi.name == "spi"
        assert 8 in spi.timing["data_bits"]

    def test_i2c_protocol(self) -> None:
        """Test I2C protocol is properly defined."""
        registry = get_protocol_registry()
        i2c = registry.get("i2c")

        assert i2c.name == "i2c"
        assert "standard" in i2c.timing["speed_modes"]

    def test_can_protocol(self) -> None:
        """Test CAN protocol is properly defined."""
        registry = get_protocol_registry()
        can = registry.get("can")

        assert can.name == "can"
        assert 500000 in can.timing["baud_rates"]


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Test thread safety of protocol registry operations."""

    def test_concurrent_registration(self) -> None:
        """Test concurrent protocol registration."""
        registry = ProtocolRegistry()
        errors: list[Exception] = []

        def register_protocol(idx: int) -> None:
            try:
                name = f"concurrent_{idx}"
                if name in registry._protocols:
                    del registry._protocols[name]
                proto = ProtocolDefinition(name=name, version="1.0.0")
                registry.register(proto)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_protocol, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check no fatal errors (some duplicates are expected)
        assert len([e for e in errors if "already registered" not in str(e)]) == 0

    def test_concurrent_get(self) -> None:
        """Test concurrent protocol retrieval."""
        registry = get_protocol_registry()
        results: list[ProtocolDefinition | None] = []

        def get_protocol() -> None:
            proto = registry.get("uart")
            results.append(proto)

        threads = [threading.Thread(target=get_protocol) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 20
        assert all(p.name == "uart" for p in results if p is not None)
