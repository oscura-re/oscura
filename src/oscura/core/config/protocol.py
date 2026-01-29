"""Protocol definition registry and loading.

This module provides protocol definition management including registry,
loading from YAML/JSON files, inheritance, hot reload support, version
migration, and circular dependency detection.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from oscura.core.config.schema import validate_against_schema
from oscura.core.exceptions import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class ProtocolDefinition:
    """Protocol definition with metadata and configuration.

    Attributes:
        name: Protocol identifier (e.g., "uart", "spi")
        version: Protocol version (semver)
        description: Human-readable description
        author: Protocol definition author
        timing: Timing configuration (baud rates, data bits, etc.)
        voltage_levels: Logic level configuration
        state_machine: Protocol state machine definition
        extends: Parent protocol name for inheritance
        metadata: Additional custom metadata
        source_file: Path to source file (for hot reload)
        schema_version: Schema version for migration support

    Example:
        >>> protocol = ProtocolDefinition(
        ...     name="uart",
        ...     version="1.0.0",
        ...     timing={"baud_rates": [9600, 115200]}
        ... )
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    timing: dict[str, Any] = field(default_factory=dict)
    voltage_levels: dict[str, Any] = field(default_factory=dict)
    state_machine: dict[str, Any] = field(default_factory=dict)
    extends: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_file: str | None = None
    schema_version: str = "1.0.0"

    @property
    def supports_digital(self) -> bool:
        """Check if protocol supports digital signals."""
        return True  # Most protocols are digital

    @property
    def supports_analog(self) -> bool:
        """Check if protocol requires analog threshold detection."""
        return bool(self.voltage_levels)

    @property
    def sample_rate_min(self) -> float:
        """Minimum sample rate required for decoding."""
        # Estimate from baud rate (need 10x oversampling typically)
        baud_rates = self.timing.get("baud_rates", [])
        if baud_rates:
            max_baud = max(baud_rates)
            return float(max_baud * 10)
        return 1e6  # Default 1 MHz

    @property
    def sample_rate_max(self) -> float | None:
        """Maximum useful sample rate for decoding."""
        return None  # No upper limit typically

    @property
    def bit_widths(self) -> list[int]:
        """Supported data bit widths."""
        return self.timing.get("data_bits", [8])  # type: ignore[no-any-return]


@dataclass
class ProtocolCapabilities:
    """Protocol capabilities for querying and filtering.

    Attributes:
        supports_digital: Whether protocol uses digital signals
        supports_analog: Whether protocol needs analog thresholds
        sample_rate_min: Minimum required sample rate (Hz)
        sample_rate_max: Maximum useful sample rate (Hz)
        bit_widths: Supported data widths
    """

    supports_digital: bool = True
    supports_analog: bool = False
    sample_rate_min: float = 1e6
    sample_rate_max: float | None = None
    bit_widths: list[int] = field(default_factory=lambda: [8])


class ProtocolRegistry:
    """Central registry of all protocol definitions.

    Provides O(1) lookup by name, version queries, capability filtering,
    and enumeration for UI integration.

    Example:
        >>> registry = ProtocolRegistry()
        >>> uart = registry.get("uart")
        >>> i2c = registry.get("i2c", version="2.1.0")
        >>> all_protocols = registry.list()
        >>> digital = registry.filter(supports_digital=True)
    """

    _instance: ProtocolRegistry | None = None

    def __new__(cls) -> ProtocolRegistry:
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._protocols: dict[str, dict[str, ProtocolDefinition]] = {}  # type: ignore[misc, attr-defined]
            cls._instance._default_versions: dict[str, str] = {}  # type: ignore[misc, attr-defined]
            cls._instance._watchers: list[Callable[[ProtocolDefinition], None]] = []  # type: ignore[misc, attr-defined]
        return cls._instance

    def register(
        self,
        protocol: ProtocolDefinition,
        *,
        set_default: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Register a protocol definition.

        Args:
            protocol: Protocol definition to register
            set_default: If True, set as default version
            overwrite: If True, allow overwriting existing registration

        Raises:
            ValueError: If protocol already registered and overwrite=False

        Example:
            >>> registry.register(uart_protocol)
        """
        if protocol.name not in self._protocols:  # type: ignore[attr-defined]
            self._protocols[protocol.name] = {}  # type: ignore[attr-defined]

        if protocol.version in self._protocols[protocol.name] and not overwrite:  # type: ignore[attr-defined]
            raise ValueError(f"Protocol '{protocol.name}' v{protocol.version} already registered")

        self._protocols[protocol.name][protocol.version] = protocol  # type: ignore[attr-defined]

        if set_default:
            self._default_versions[protocol.name] = protocol.version  # type: ignore[attr-defined]

        logger.debug(f"Registered protocol: {protocol.name} v{protocol.version}")

    def get(self, name: str, version: str | None = None) -> ProtocolDefinition:
        """Get protocol by name and optional version.

        Args:
            name: Protocol name
            version: Specific version or None for default

        Returns:
            Protocol definition

        Raises:
            KeyError: If protocol not found

        Example:
            >>> uart = registry.get("uart")
            >>> i2c = registry.get("i2c", version="2.1.0")
        """
        if name not in self._protocols:  # type: ignore[attr-defined]
            raise KeyError(
                f"Protocol '{name}' not found. Available: {list(self._protocols.keys())}"  # type: ignore[attr-defined]
            )

        if version is None:
            version = self._default_versions.get(name)  # type: ignore[attr-defined]
            if version is None:
                # Get latest version
                versions = sorted(self._protocols[name].keys())  # type: ignore[attr-defined]
                version = versions[-1] if versions else None

        if version is None or version not in self._protocols[name]:  # type: ignore[attr-defined]
            raise KeyError(
                f"Protocol '{name}' version '{version}' not found. "
                f"Available versions: {list(self._protocols[name].keys())}"  # type: ignore[attr-defined]
            )

        return self._protocols[name][version]  # type: ignore[no-any-return, attr-defined]

    def list(self) -> list[ProtocolDefinition]:
        """List all available protocols (default versions).

        Returns:
            Sorted list of protocol definitions

        Example:
            >>> for proto in registry.list():
            ...     print(f"{proto.name} v{proto.version}: {proto.description}")
        """
        protocols = []
        for name in sorted(self._protocols.keys()):  # type: ignore[attr-defined]
            version = self._default_versions.get(name)  # type: ignore[attr-defined]
            if version and version in self._protocols[name]:  # type: ignore[attr-defined]
                protocols.append(self._protocols[name][version])  # type: ignore[attr-defined]
            elif self._protocols[name]:  # type: ignore[attr-defined]
                # Get latest version
                latest = sorted(self._protocols[name].keys())[-1]  # type: ignore[attr-defined]
                protocols.append(self._protocols[name][latest])  # type: ignore[attr-defined]
        return protocols

    def get_capabilities(self, name: str) -> ProtocolCapabilities:
        """Query protocol capabilities.

        Args:
            name: Protocol name

        Returns:
            Protocol capabilities

        Example:
            >>> caps = registry.get_capabilities("uart")
            >>> print(f"Sample rate: {caps.sample_rate_min}-{caps.sample_rate_max} Hz")
        """
        protocol = self.get(name)
        return ProtocolCapabilities(
            supports_digital=protocol.supports_digital,
            supports_analog=protocol.supports_analog,
            sample_rate_min=protocol.sample_rate_min,
            sample_rate_max=protocol.sample_rate_max,
            bit_widths=protocol.bit_widths,
        )

    def filter(
        self,
        supports_digital: bool | None = None,
        supports_analog: bool | None = None,
        sample_rate_min__gte: float | None = None,
        sample_rate_max__lte: float | None = None,
    ) -> list[ProtocolDefinition]:  # type: ignore[valid-type]
        """Filter protocols by capabilities.

        Args:
            supports_digital: Filter by digital support
            supports_analog: Filter by analog support
            sample_rate_min__gte: Minimum sample rate >= value
            sample_rate_max__lte: Maximum sample rate <= value

        Returns:
            List of matching protocols

        Example:
            >>> digital = registry.filter(supports_digital=True)
            >>> high_speed = registry.filter(sample_rate_min__gte=1_000_000)
        """
        results = []
        for protocol in self.list():
            match = True

            if supports_digital is not None:
                if protocol.supports_digital != supports_digital:
                    match = False

            if supports_analog is not None:
                if protocol.supports_analog != supports_analog:
                    match = False

            if sample_rate_min__gte is not None:
                if protocol.sample_rate_min < sample_rate_min__gte:
                    match = False

            if sample_rate_max__lte is not None and (
                protocol.sample_rate_max and protocol.sample_rate_max > sample_rate_max__lte
            ):
                match = False

            if match:
                results.append(protocol)

        return results

    def has_protocol(self, name: str, version: str | None = None) -> bool:
        """Check if protocol is registered.

        Args:
            name: Protocol name
            version: Specific version or None for any

        Returns:
            True if registered
        """
        if name not in self._protocols:  # type: ignore[attr-defined]
            return False
        if version is None:
            return True
        return version in self._protocols[name]  # type: ignore[attr-defined]

    def list_versions(self, name: str) -> list[str]:  # type: ignore[valid-type]
        """List all versions of a protocol.

        Args:
            name: Protocol name

        Returns:
            List of version strings
        """
        if name not in self._protocols:  # type: ignore[attr-defined]
            return []
        return sorted(self._protocols[name].keys())  # type: ignore[attr-defined]

    def on_change(self, callback: Callable[[ProtocolDefinition], None]) -> None:
        """Register callback for protocol changes (hot reload support).

        Args:
            callback: Function to call when protocol is reloaded

        Example:
            >>> watcher = registry.on_change(lambda proto: print(f"Reloaded {proto.name}"))
        """
        self._watchers.append(callback)  # type: ignore[attr-defined]

    def _notify_change(self, protocol: ProtocolDefinition) -> None:
        """Notify watchers of protocol change."""
        for callback in self._watchers:  # type: ignore[attr-defined]
            try:
                callback(protocol)
            except Exception as e:
                logger.warning(f"Protocol change callback failed: {e}")


def load_protocol(path: str | Path, validate: bool = True) -> ProtocolDefinition:
    """Load protocol definition from YAML or JSON file.

    Args:
        path: Path to protocol definition file
        validate: If True, validate against schema

    Returns:
        Loaded protocol definition

    Raises:
        ConfigurationError: If file invalid or validation fails

    Example:
        >>> protocol = load_protocol("configs/uart.yaml")
        >>> protocol = load_protocol("configs/i2c.json")
    """
    path = Path(path)

    if not path.exists():
        raise ConfigurationError(
            f"Protocol definition file not found: {path.name}", details=f"File path: {path}"
        )

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
            if path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(content)
            else:
                import json

                data = json.loads(content)

    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"YAML parse error in {path.name}", details=f"File: {path}\nError: {e}"
        ) from e
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load protocol file: {path.name}", details=f"File: {path}\nError: {e}"
        ) from e

    # Handle nested 'protocol' key
    if "protocol" in data:
        data = data["protocol"]

    if validate:
        try:
            validate_against_schema(data, "protocol")
        except Exception as e:
            raise ConfigurationError(
                f"Protocol validation failed for {path.name}",
                details=f"File: {path}\nError: {e}",
            ) from e

    protocol = ProtocolDefinition(
        name=data.get("name", path.stem),
        version=data.get("version", "1.0.0"),
        description=data.get("description", ""),
        author=data.get("author", ""),
        timing=data.get("timing", {}),
        voltage_levels=data.get("voltage_levels", {}),
        state_machine=data.get("state_machine", {}),
        extends=data.get("extends"),
        metadata=data.get("metadata", {}),
        source_file=str(path),
    )

    logger.info(f"Loaded protocol: {protocol.name} v{protocol.version} from {path}")
    return protocol


def resolve_inheritance(
    protocol: ProtocolDefinition,
    registry: ProtocolRegistry,
    *,
    max_depth: int = 5,
    deep_merge: bool = False,
    _visited: set[str] | None = None,
) -> ProtocolDefinition:
    """Resolve protocol inheritance chain with circular detection.

    Supports multi-level inheritance (up to 5 levels deep) with both
    shallow and deep merge strategies for nested properties.

    Args:
        protocol: Protocol with potential inheritance
        registry: Registry to look up parent protocols
        max_depth: Maximum inheritance depth (default 5.)
        deep_merge: If True, recursively merge nested dicts; else shallow merge
        _visited: Set of visited protocols for cycle detection

    Returns:
        Protocol with inherited properties merged

    Raises:
        ConfigurationError: If circular inheritance or depth exceeded

    Example:
        >>> resolved = resolve_inheritance(spi_variant, registry)
        >>> resolved_deep = resolve_inheritance(spi_variant, registry, deep_merge=True)
    """
    if _visited is None:
        _visited = set()

    if not protocol.extends:
        return protocol

    # Cycle detection using DFS with visited set
    if protocol.name in _visited:
        cycle_list = [*list(_visited), protocol.name]
        cycle = " → ".join(cycle_list)
        raise ConfigurationError(
            f"Circular inheritance detected: {cycle}",
            details=f"Protocol inheritance forms a cycle. Remove 'extends' from one of: {', '.join(cycle_list)}",
            fix_hint=f"Break the cycle by removing the 'extends' field from {protocol.name}",
        )

    # Depth limit check
    if len(_visited) >= max_depth:
        chain = " → ".join([*list(_visited), protocol.name])
        raise ConfigurationError(
            f"Inheritance depth exceeded maximum of {max_depth}",
            details=f"Current chain: {chain}",
            fix_hint="Flatten the inheritance hierarchy or increase max_depth",
        )

    _visited.add(protocol.name)

    # Get parent protocol
    try:
        parent = registry.get(protocol.extends)
    except KeyError as e:
        available = ", ".join(registry._protocols.keys())  # type: ignore[attr-defined]
        raise ConfigurationError(
            f"Parent protocol '{protocol.extends}' not found",
            details=f"Protocol '{protocol.name}' extends missing parent. Available: {available}",
            fix_hint=f"Add protocol '{protocol.extends}' to registry or fix 'extends' field",
        ) from e

    # Recursively resolve parent
    resolved_parent = resolve_inheritance(
        parent, registry, max_depth=max_depth, deep_merge=deep_merge, _visited=_visited
    )

    # Merge properties (child overrides parent)
    if deep_merge:
        merged_timing = _deep_merge_dicts(resolved_parent.timing, protocol.timing)
        merged_voltage = _deep_merge_dicts(resolved_parent.voltage_levels, protocol.voltage_levels)
        merged_state = _deep_merge_dicts(resolved_parent.state_machine, protocol.state_machine)
        merged_metadata = _deep_merge_dicts(resolved_parent.metadata, protocol.metadata)
    else:
        # Shallow merge (default)
        merged_timing = {**resolved_parent.timing, **protocol.timing}
        merged_voltage = {**resolved_parent.voltage_levels, **protocol.voltage_levels}
        merged_state = {**resolved_parent.state_machine, **protocol.state_machine}
        merged_metadata = {**resolved_parent.metadata, **protocol.metadata}

    return ProtocolDefinition(
        name=protocol.name,
        version=protocol.version,
        description=protocol.description or resolved_parent.description,
        author=protocol.author or resolved_parent.author,
        timing=merged_timing,
        voltage_levels=merged_voltage,
        state_machine=merged_state,
        extends=None,  # Clear extends after resolution
        metadata=merged_metadata,
        source_file=protocol.source_file,
        schema_version=protocol.schema_version,
    )


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries recursively.

    Args:
        base: Base dictionary
        override: Override dictionary (takes precedence)

    Returns:
        Merged dictionary

    Example:
        >>> base = {"a": {"b": 1, "c": 2}}
        >>> override = {"a": {"c": 3, "d": 4}}
        >>> _deep_merge_dicts(base, override)
        {'a': {'b': 1, 'c': 3, 'd': 4}}
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class ProtocolWatcher:
    """File watcher for hot-reloading protocol definitions.

    Monitors a directory for protocol file changes and reloads
    automatically with <2s latency using background thread polling.

    Example:
        >>> watcher = ProtocolWatcher("configs/")
        >>> watcher.on_change(lambda proto: print(f"Reloaded {proto.name}"))
        >>> watcher.start()
        >>> # ... later ...
        >>> watcher.stop()
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        poll_interval: float = 1.0,
        registry: ProtocolRegistry | None = None,
    ):
        """Initialize watcher for directory.

        Args:
            directory: Directory to watch for protocol files
            poll_interval: Polling interval in seconds (default 1.0 for <2s latency)
            registry: Registry to auto-register reloaded protocols
        """
        self.directory = Path(directory)
        self.poll_interval = poll_interval
        self.registry = registry
        self._callbacks: list[Callable[[ProtocolDefinition], None]] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._file_mtimes: dict[str, float] = {}

    def on_change(self, callback: Callable[[ProtocolDefinition], None]) -> None:
        """Register callback for protocol changes.

        Args:
            callback: Function to call with reloaded protocol
        """
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start watching for file changes in background thread.

        The watcher polls the directory every poll_interval seconds,
        ensuring <2s latency for detecting changes.
        """
        if self._running:
            logger.warning("Protocol watcher already running")
            return

        self._running = True
        self._scan_files()

        # Start background polling thread
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

        logger.info(
            f"Started watching protocols in {self.directory} (poll interval: {self.poll_interval}s)"
        )

    def stop(self) -> None:
        """Stop watching for file changes."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("Stopped protocol watcher")

    def _watch_loop(self) -> None:
        """Background thread polling loop."""
        while self._running:
            try:
                self.check_changes()
            except Exception as e:
                logger.error(f"Error in protocol watcher: {e}")
            time.sleep(self.poll_interval)

    def check_changes(self) -> list[ProtocolDefinition]:
        """Check for changed files and reload.

        Returns:
            List of reloaded protocols
        """
        if not self._running:
            return []

        reloaded = []
        for file_path in self.directory.glob("**/*.yaml"):
            if not file_path.is_file():
                continue

            try:
                mtime = os.path.getmtime(file_path)
            except OSError:
                continue

            str_path = str(file_path)

            if str_path in self._file_mtimes and mtime > self._file_mtimes[str_path]:
                try:
                    protocol = load_protocol(file_path)
                    reloaded.append(protocol)

                    # Auto-register if registry provided
                    if self.registry:
                        self.registry.register(protocol, overwrite=True)
                        self.registry._notify_change(protocol)

                    self._notify(protocol)
                    logger.info(f"Hot-reloaded protocol: {protocol.name} from {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to reload {file_path}: {e}")

            self._file_mtimes[str_path] = mtime

        return reloaded

    def _scan_files(self) -> None:
        """Initial scan of directory."""
        for file_path in self.directory.glob("**/*.yaml"):
            if file_path.is_file():
                with contextlib.suppress(OSError):
                    self._file_mtimes[str(file_path)] = os.path.getmtime(file_path)

    def _notify(self, protocol: ProtocolDefinition) -> None:
        """Notify callbacks of protocol change."""
        for callback in self._callbacks:
            try:
                callback(protocol)
            except Exception as e:
                logger.warning(f"Protocol change callback failed: {e}")


# Global registry instance
_registry: ProtocolRegistry | None = None


def get_protocol_registry() -> ProtocolRegistry:
    """Get the global protocol registry.

    Returns:
        Global ProtocolRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ProtocolRegistry()
        _register_builtin_protocols(_registry)
    return _registry


def _register_builtin_protocols(registry: ProtocolRegistry) -> None:
    """Register built-in protocol definitions."""
    # UART
    registry.register(
        ProtocolDefinition(
            name="uart",
            version="1.0.0",
            description="Universal Asynchronous Receiver/Transmitter",
            timing={
                "baud_rates": [
                    9600,
                    19200,
                    38400,
                    57600,
                    115200,
                    230400,
                    460800,
                    921600,
                ],
                "data_bits": [7, 8],
                "stop_bits": [1, 1.5, 2],
                "parity": ["none", "even", "odd", "mark", "space"],
            },
            voltage_levels={"logic_family": "TTL", "idle_state": "high"},
            state_machine={
                "states": ["IDLE", "START", "DATA", "PARITY", "STOP"],
                "initial_state": "IDLE",
            },
        )
    )

    # SPI
    registry.register(
        ProtocolDefinition(
            name="spi",
            version="1.0.0",
            description="Serial Peripheral Interface",
            timing={
                "data_bits": [8, 16, 32],
                "clock_polarity": [0, 1],
                "clock_phase": [0, 1],
            },
            state_machine={"states": ["IDLE", "ACTIVE"], "initial_state": "IDLE"},
        )
    )

    # I2C
    registry.register(
        ProtocolDefinition(
            name="i2c",
            version="1.0.0",
            description="Inter-Integrated Circuit",
            timing={
                "speed_modes": ["standard", "fast", "fast_plus", "high_speed"],
                "data_bits": [8],
            },
            state_machine={
                "states": ["IDLE", "START", "ADDRESS", "DATA", "ACK", "STOP"],
                "initial_state": "IDLE",
            },
        )
    )

    # CAN
    registry.register(
        ProtocolDefinition(
            name="can",
            version="1.0.0",
            description="Controller Area Network",
            timing={"baud_rates": [125000, 250000, 500000, 1000000]},
            state_machine={
                "states": [
                    "IDLE",
                    "SOF",
                    "ARBITRATION",
                    "CONTROL",
                    "DATA",
                    "CRC",
                    "ACK",
                    "EOF",
                ],
                "initial_state": "IDLE",
            },
        )
    )


def migrate_protocol_schema(
    protocol_data: dict[str, Any], from_version: str, to_version: str = "1.0.0"
) -> dict[str, Any]:
    """Migrate protocol definition between schema versions.

    Args:
        protocol_data: Protocol data dictionary
        from_version: Source schema version
        to_version: Target schema version (default current)

    Returns:
        Migrated protocol data

    Raises:
        ConfigurationError: If migration fails or unsupported version

    Example:
        >>> old_proto = {"name": "uart", "timing": {...}}
        >>> new_proto = migrate_protocol_schema(old_proto, "0.9.0", "1.0.0")
    """
    if from_version == to_version:
        return protocol_data

    # Define migration paths
    migrations = {
        ("0.9.0", "1.0.0"): _migrate_0_9_to_1_0,
        ("0.8.0", "0.9.0"): _migrate_0_8_to_0_9,
        ("0.8.0", "1.0.0"): lambda d: _migrate_0_9_to_1_0(_migrate_0_8_to_0_9(d)),
    }

    migration_key = (from_version, to_version)
    if migration_key not in migrations:
        raise ConfigurationError(
            f"No migration path from schema {from_version} to {to_version}",
            details="Supported migrations: " + ", ".join(f"{k[0]}→{k[1]}" for k in migrations),
            fix_hint="Manually update the protocol definition or use an intermediate version",
        )

    logger.info(f"Migrating protocol schema from {from_version} to {to_version}")
    try:
        migrated = migrations[migration_key](protocol_data.copy())  # type: ignore[no-untyped-call]
        migrated["schema_version"] = to_version
        return migrated
    except Exception as e:
        raise ConfigurationError(
            f"Schema migration failed from {from_version} to {to_version}",
            details=str(e),
            fix_hint="Check migration logs and manually update protocol definition",
        ) from e


def _migrate_0_8_to_0_9(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from schema 0.8.0 to 0.9.0."""
    # Example migration: rename 'baudrate' to 'baud_rates' and convert to list
    if "baudrate" in data.get("timing", {}):
        data.setdefault("timing", {})
        data["timing"]["baud_rates"] = [data["timing"].pop("baudrate")]
    return data


def _migrate_0_9_to_1_0(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate from schema 0.9.0 to 1.0.0."""
    # Example migration: add required fields with defaults
    data.setdefault("version", "1.0.0")
    data.setdefault("description", "")
    data.setdefault("author", "")

    # Convert old state format if needed
    if "state" in data:
        data["state_machine"] = data.pop("state")

    return data


__all__ = [
    "ProtocolCapabilities",
    "ProtocolDefinition",
    "ProtocolRegistry",
    "ProtocolWatcher",
    "get_protocol_registry",
    "load_protocol",
    "migrate_protocol_schema",
    "resolve_inheritance",
]
