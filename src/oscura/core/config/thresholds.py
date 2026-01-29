"""Threshold configuration for voltage levels and logic families.

This module provides threshold configuration for digital signal analysis
including logic family definitions, threshold profiles, and per-analysis
overrides.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from oscura.core.config.schema import validate_against_schema
from oscura.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class LogicFamily:
    """Logic family voltage threshold definition.

    Defines voltage thresholds per IEEE/JEDEC standards for digital
    signal interpretation.

    Attributes:
        name: Logic family name (e.g., "TTL", "CMOS_3V3")
        VIH: Input high voltage threshold (V)
        VIL: Input low voltage threshold (V)
        VOH: Output high voltage (V)
        VOL: Output low voltage (V)
        VCC: Supply voltage (V)
        description: Human-readable description
        temperature_range: Operating temperature range (min, max) in C
        noise_margin_high: High state noise margin (V)
        noise_margin_low: Low state noise margin (V)
        source: Origin of definition (builtin, user, file path)

    Example:
        >>> ttl = LogicFamily(
        ...     name="TTL",
        ...     VIH=2.0, VIL=0.8,
        ...     VOH=2.4, VOL=0.4,
        ...     VCC=5.0
        ... )
        >>> print(f"TTL noise margin high: {ttl.noise_margin_high}V")
    """

    name: str
    VIH: float  # Input high threshold
    VIL: float  # Input low threshold
    VOH: float  # Output high level
    VOL: float  # Output low level
    VCC: float = 5.0
    description: str = ""
    temperature_range: tuple[float, float] = field(default_factory=lambda: (0, 70))
    noise_margin_high: float | None = None
    noise_margin_low: float | None = None
    source: str = "builtin"

    def __post_init__(self) -> None:
        """Validate thresholds and compute noise margins."""
        # Validate threshold ordering
        if self.VIH <= self.VIL:
            raise ConfigurationError(
                f"Invalid thresholds for {self.name}: VIH ({self.VIH}V) must be > VIL ({self.VIL}V)"
            )
        if self.VOH <= self.VOL:
            raise ConfigurationError(
                f"Invalid thresholds for {self.name}: VOH ({self.VOH}V) must be > VOL ({self.VOL}V)"
            )

        # Compute noise margins if not provided
        if self.noise_margin_high is None:
            self.noise_margin_high = self.VOH - self.VIH
        if self.noise_margin_low is None:
            self.noise_margin_low = self.VIL - self.VOL

    def get_threshold(self, percent: float = 50.0) -> float:
        """Get threshold voltage at given percentage between VIL and VIH.

        Args:
            percent: Percentage between VIL (0%) and VIH (100%)

        Returns:
            Threshold voltage
        """
        return self.VIL + (self.VIH - self.VIL) * (percent / 100.0)

    def with_temperature_derating(
        self, temperature: float, derating_factor: float = 0.002
    ) -> LogicFamily:
        """Create copy with temperature-derated thresholds.

        Args:
            temperature: Operating temperature in Celsius
            derating_factor: Derating factor per degree C (default 0.2%/C)

        Returns:
            New LogicFamily with adjusted thresholds
        """
        # Simple linear derating from nominal 25C
        delta_t = temperature - 25.0
        factor = 1.0 - (delta_t * derating_factor)

        return LogicFamily(
            name=f"{self.name}@{temperature}C",
            VIH=self.VIH * factor,
            VIL=self.VIL * factor,
            VOH=self.VOH * factor,
            VOL=self.VOL * factor,
            VCC=self.VCC,
            description=f"{self.description} (derated for {temperature}C)",
            temperature_range=self.temperature_range,
            source=self.source,
        )


@dataclass
class ThresholdProfile:
    """Named threshold profile combining logic family with adjustments.

    Profiles allow users to save and reuse threshold configurations
    for specific analysis scenarios.

    Attributes:
        name: Profile name
        base_family: Base logic family name
        overrides: Override values for specific thresholds
        tolerance: Tolerance percentage (0-100)
        description: Profile description

    Example:
        >>> profile = ThresholdProfile(
        ...     name="strict_ttl",
        ...     base_family="TTL",
        ...     overrides={"VIH": 2.2},
        ...     tolerance=0
        ... )
    """

    name: str
    base_family: str = "TTL"
    overrides: dict[str, float] = field(default_factory=dict)
    tolerance: float = 0.0  # 0-100%
    description: str = ""

    def apply_to(self, family: LogicFamily) -> LogicFamily:
        """Apply profile overrides to a logic family.

        Args:
            family: Base logic family

        Returns:
            New LogicFamily with overrides applied
        """
        # Apply tolerance
        factor = 1.0 + (self.tolerance / 100.0)

        return LogicFamily(
            name=f"{family.name}_{self.name}",
            VIH=self.overrides.get("VIH", family.VIH),
            VIL=self.overrides.get("VIL", family.VIL),
            VOH=self.overrides.get("VOH", family.VOH * factor),
            VOL=self.overrides.get("VOL", family.VOL / factor),
            VCC=self.overrides.get("VCC", family.VCC),
            description=f"{family.description} with {self.name} profile",
            temperature_range=family.temperature_range,
            source="profile",
        )


class ThresholdRegistry:
    """Registry for logic families and threshold profiles.

    Manages built-in and user-defined logic families with support
    for runtime overrides and profile switching.

    Thread-safe singleton implementation with locks protecting shared state.

    Example:
        >>> registry = ThresholdRegistry()
        >>> ttl = registry.get_family("TTL")
        >>> cmos = registry.get_family("CMOS_3V3")
        >>> families = registry.list_families()
    """

    _instance: ThresholdRegistry | None = None
    _lock: threading.Lock = threading.Lock()

    # Instance attributes (initialized in __new__)
    _families: dict[str, LogicFamily]
    _profiles: dict[str, ThresholdProfile]
    _session_overrides: dict[str, float]
    _state_lock: threading.Lock

    def __new__(cls) -> ThresholdRegistry:
        """Ensure singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._families = {}
                    cls._instance._profiles = {}
                    cls._instance._session_overrides = {}
                    cls._instance._state_lock = threading.Lock()
                    cls._instance._register_builtins()
        return cls._instance

    def _register_builtins(self) -> None:
        """Register built-in logic family definitions."""
        builtins = _get_builtin_logic_families()

        for family in builtins:
            self._families[family.name] = family

        # Built-in profiles
        builtin_profiles = _get_builtin_profiles()

        for profile in builtin_profiles:
            self._profiles[profile.name] = profile

    def get_family(self, name: str) -> LogicFamily:
        """Get logic family by name.

        Args:
            name: Logic family name (case-insensitive)

        Returns:
            Logic family definition

        Raises:
            KeyError: If family not found
        """
        with self._state_lock:
            family = self._lookup_family(name)
            return self._apply_overrides_if_needed(family)

    def _lookup_family(self, name: str) -> LogicFamily:
        """Look up family by name (exact or case-insensitive).

        Args:
            name: Family name to look up.

        Returns:
            Logic family definition.

        Raises:
            KeyError: If family not found.
        """
        # Try exact match first
        if name in self._families:
            return self._families[name]

        # Try case-insensitive match
        if name.upper() in self._families:
            return self._families[name.upper()]

        available = list(self._families.keys())
        raise KeyError(f"Logic family '{name}' not found. Available: {available}")

    def _apply_overrides_if_needed(self, family: LogicFamily) -> LogicFamily:
        """Apply session overrides to family if configured.

        Args:
            family: Base family.

        Returns:
            Family with overrides applied, or original if no overrides.
        """
        if not self._session_overrides:
            return family

        return LogicFamily(
            name=family.name,
            VIH=self._session_overrides.get("VIH", family.VIH),
            VIL=self._session_overrides.get("VIL", family.VIL),
            VOH=self._session_overrides.get("VOH", family.VOH),
            VOL=self._session_overrides.get("VOL", family.VOL),
            VCC=self._session_overrides.get("VCC", family.VCC),
            description=family.description,
            temperature_range=family.temperature_range,
            source="override",
        )

    def list_families(self) -> list[str]:
        """List all available logic families.

        Returns:
            List of family names
        """
        with self._state_lock:
            return sorted(self._families.keys())

    def register_family(self, family: LogicFamily, *, namespace: str = "user") -> None:
        """Register custom logic family.

        Args:
            family: Logic family definition
            namespace: Namespace prefix for custom families

        Example:
            >>> custom = LogicFamily(name="my_custom", VIH=2.5, VIL=1.0, VOH=3.0, VOL=0.5)
            >>> registry.register_family(custom)
            >>> # Available as "user.my_custom"
        """
        with self._state_lock:
            # Namespace custom families
            if namespace and not family.name.startswith(f"{namespace}."):
                name = f"{namespace}.{family.name}"
            else:
                name = family.name

            # Update family with new name
            family = LogicFamily(
                name=name,
                VIH=family.VIH,
                VIL=family.VIL,
                VOH=family.VOH,
                VOL=family.VOL,
                VCC=family.VCC,
                description=family.description,
                temperature_range=family.temperature_range,
                source=family.source,
            )

            self._families[name] = family
            logger.info(f"Registered custom logic family: {name}")

    def set_threshold_override(self, **kwargs: float) -> None:
        """Set session-level threshold overrides.

        Overrides persist for session lifetime until reset.

        Args:
            **kwargs: Threshold overrides (VIH, VIL, VOH, VOL, VCC)

        Raises:
            ValueError: If invalid threshold key or value out of range.

        Example:
            >>> registry.set_threshold_override(VIH=2.5, VIL=0.7)
        """
        valid_keys = {"VIH", "VIL", "VOH", "VOL", "VCC"}
        with self._state_lock:
            for key, value in kwargs.items():
                if key not in valid_keys:
                    raise ValueError(f"Invalid threshold key: {key}. Valid: {valid_keys}")
                if not 0.0 <= value <= 10.0:
                    raise ValueError(f"Threshold {key}={value}V out of range (0-10V)")
                self._session_overrides[key] = value

            logger.info(f"Set threshold overrides: {kwargs}")

    def reset_overrides(self) -> None:
        """Reset session threshold overrides."""
        with self._state_lock:
            self._session_overrides.clear()
            logger.info("Reset threshold overrides")

    def get_profile(self, name: str) -> ThresholdProfile:
        """Get threshold profile by name.

        Args:
            name: Profile name

        Returns:
            Threshold profile

        Raises:
            KeyError: If profile not found.
        """
        with self._state_lock:
            if name not in self._profiles:
                raise KeyError(
                    f"Profile '{name}' not found. Available: {list(self._profiles.keys())}"
                )
            return self._profiles[name]

    def apply_profile(self, name: str) -> LogicFamily:
        """Apply a threshold profile.

        Args:
            name: Profile name

        Returns:
            Logic family with profile applied
        """
        profile = self.get_profile(name)
        base_family = self.get_family(profile.base_family)
        return profile.apply_to(base_family)

    def save_profile(
        self, name: str, base_family: str | None = None, path: str | Path | None = None
    ) -> None:
        """Save current settings as named profile.

        Args:
            name: Profile name
            base_family: Base family name (default: "TTL")
            path: Optional file path to save

        Example:
            >>> registry.set_threshold_override(VIH=2.5)
            >>> registry.save_profile("my_profile")
        """
        with self._state_lock:
            profile = ThresholdProfile(
                name=name,
                base_family=base_family or "TTL",
                overrides=dict(self._session_overrides),
                description=f"User profile {name}",
            )
            self._profiles[name] = profile

            if path:
                path = Path(path)
                data = {
                    "name": profile.name,
                    "base_family": profile.base_family,
                    "overrides": profile.overrides,
                    "tolerance": profile.tolerance,
                    "description": profile.description,
                }
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(data, f)
                logger.info(f"Saved profile to {path}")


def _get_builtin_logic_families() -> list[LogicFamily]:
    """Get list of built-in logic family definitions.

    Returns:
        List of standard logic families.
    """
    return [
        # TTL
        LogicFamily(
            name="TTL",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
            VCC=5.0,
            description="Standard TTL (74xx series)",
        ),
        # CMOS variants
        LogicFamily(
            name="CMOS_5V",
            VIH=3.5,
            VIL=1.5,
            VOH=4.9,
            VOL=0.1,
            VCC=5.0,
            description="CMOS 5V (74HCxx series)",
        ),
        LogicFamily(
            name="LVTTL_3V3",
            VIH=2.0,
            VIL=0.8,
            VOH=2.4,
            VOL=0.4,
            VCC=3.3,
            description="Low Voltage TTL 3.3V",
        ),
        LogicFamily(
            name="LVCMOS_3V3",
            VIH=2.0,
            VIL=0.7,
            VOH=2.4,
            VOL=0.4,
            VCC=3.3,
            description="Low Voltage CMOS 3.3V",
        ),
        LogicFamily(
            name="LVCMOS_2V5",
            VIH=1.7,
            VIL=0.7,
            VOH=2.0,
            VOL=0.4,
            VCC=2.5,
            description="Low Voltage CMOS 2.5V",
        ),
        LogicFamily(
            name="LVCMOS_1V8",
            VIH=1.17,
            VIL=0.63,
            VOH=1.35,
            VOL=0.45,
            VCC=1.8,
            description="Low Voltage CMOS 1.8V",
        ),
        LogicFamily(
            name="LVCMOS_1V5",
            VIH=0.975,
            VIL=0.525,
            VOH=1.125,
            VOL=0.375,
            VCC=1.5,
            description="Low Voltage CMOS 1.5V",
        ),
        LogicFamily(
            name="LVCMOS_1V2",
            VIH=0.84,  # 0.7 * 1.2
            VIL=0.36,  # 0.3 * 1.2
            VOH=1.1,
            VOL=0.1,
            VCC=1.2,
            description="Low Voltage CMOS 1.2V",
        ),
        # ECL
        LogicFamily(
            name="ECL",
            VIH=-0.9,
            VIL=-1.7,
            VOH=-0.9,
            VOL=-1.75,
            VCC=-5.2,
            description="Emitter-Coupled Logic (ECL 10K)",
        ),
    ]


def _get_builtin_profiles() -> list[ThresholdProfile]:
    """Get list of built-in threshold profiles.

    Returns:
        List of standard threshold profiles.
    """
    return [
        ThresholdProfile(
            name="strict",
            base_family="TTL",
            tolerance=0,
            description="Exact specification values",
        ),
        ThresholdProfile(
            name="relaxed",
            base_family="TTL",
            tolerance=20,
            description="20% tolerance for real-world signals",
        ),
        ThresholdProfile(
            name="auto",
            base_family="TTL",
            tolerance=10,
            description="Auto-adjusted based on signal confidence",
        ),
    ]


def load_logic_family(path: str | Path) -> LogicFamily:
    """Load logic family from YAML/JSON file.

    Args:
        path: Path to file

    Returns:
        Loaded logic family
    """
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Validate against schema
    validate_against_schema(data, "logic_family")

    return LogicFamily(
        name=data["name"],
        VIH=data["VIH"],
        VIL=data["VIL"],
        VOH=data["VOH"],
        VOL=data["VOL"],
        VCC=data.get("VCC", 5.0),
        description=data.get("description", ""),
        temperature_range=tuple(data.get("temperature_range", {}).values()) or (0, 70),
        noise_margin_high=data.get("noise_margin_high"),
        noise_margin_low=data.get("noise_margin_low"),
        source=str(path),
    )


def get_threshold_registry() -> ThresholdRegistry:
    """Get the global threshold registry.

    Returns:
        Global ThresholdRegistry instance
    """
    return ThresholdRegistry()


def get_user_logic_families_dir() -> Path:
    """Get user directory for custom logic families.

    Returns:
        Path to ~/.oscura/logic_families/
    """
    home = Path.home()
    xdg_config = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg_config) if xdg_config else home / ".config"

    dir_path = base / "oscura" / "logic_families"
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def load_user_logic_families() -> list[LogicFamily]:
    """Load all user-defined logic families.

    Returns:
        List of loaded logic families
    """
    families = []
    user_dir = get_user_logic_families_dir()

    for file_path in user_dir.glob("*.yaml"):
        try:
            family = load_logic_family(file_path)
            families.append(family)
        except Exception as e:
            logger.warning(f"Failed to load logic family from {file_path}: {e}")

    return families


__all__ = [
    "LogicFamily",
    "ThresholdProfile",
    "ThresholdRegistry",
    "get_threshold_registry",
    "get_user_logic_families_dir",
    "load_logic_family",
    "load_user_logic_families",
]
