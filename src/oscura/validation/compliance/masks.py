"""EMC regulatory limit masks.

This module provides limit mask loading and management for EMC compliance testing.


Example:
    >>> from oscura.validation.compliance.masks import load_limit_mask, AVAILABLE_MASKS
    >>> print(AVAILABLE_MASKS)
    >>> mask = load_limit_mask('FCC_Part15_ClassB')
    >>> print(f"Frequency range: {mask.frequency_range}")

References:
    FCC Part 15 (47 CFR Part 15)
    CISPR 22/32 (EN 55022/55032)
    MIL-STD-461G
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class LimitMask:
    """EMC limit mask definition.

    Attributes:
        name: Standard name (e.g., 'FCC_Part15_ClassB')
        description: Human-readable description
        frequency: Frequency points in Hz
        limit: Limit values in dBuV (or specified unit)
        unit: Limit unit ('dBuV', 'dBm', 'dBuV/m')
        standard: Standard designation (e.g., 'FCC Part 15B', 'CISPR 32')
        distance: Measurement distance in meters
        detector: Required detector type ('peak', 'quasi-peak', 'average')
        frequency_range: (min, max) frequency in Hz
        regulatory_body: Regulatory body (FCC, CE, MIL)
        document: Reference document
    """

    name: str
    frequency: NDArray[np.float64]
    limit: NDArray[np.float64]
    description: str = ""
    unit: str = "dBuV"
    standard: str = ""
    distance: float = 3.0  # meters
    detector: str = "peak"
    regulatory_body: str = ""
    document: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frequency_range(self) -> tuple[float, float]:
        """Return (min, max) frequency range."""
        return (float(self.frequency.min()), float(self.frequency.max()))

    def get_limit_at_frequency(self, frequency: float) -> float:
        """Get limit value at a specific frequency.

        Args:
            frequency: Frequency in Hz.

        Returns:
            Limit value at the specified frequency (interpolated if needed).
        """
        return float(np.interp(frequency, self.frequency, self.limit))

    def interpolate(self, frequencies: NDArray[np.float64]) -> NDArray[np.float64]:
        """Interpolate limit values at given frequencies.

        Args:
            frequencies: Frequency points to interpolate to.

        Returns:
            Interpolated limit values.
        """
        return np.interp(frequencies, self.frequency, self.limit)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "frequency": self.frequency.tolist(),
            "limit": self.limit.tolist(),
            "unit": self.unit,
            "distance": self.distance,
            "detector": self.detector,
            "regulatory_body": self.regulatory_body,
            "document": self.document,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LimitMask:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            frequency=np.array(data["frequency"]),
            limit=np.array(data["limit"]),
            unit=data.get("unit", "dBuV"),
            distance=data.get("distance", 3.0),
            detector=data.get("detector", "peak"),
            regulatory_body=data.get("regulatory_body", ""),
            document=data.get("document", ""),
            metadata=data.get("metadata", {}),
        )


# Built-in EMC limit masks
_BUILTIN_MASKS: dict[str, dict[str, Any]] = {
    # FCC Part 15 - Unintentional Radiators (Radiated Emissions)
    "FCC_Part15_ClassA": {
        "description": "FCC Part 15 Class A (Commercial) Radiated Emissions",
        "frequency": np.array([30, 88, 216, 960, 1000]) * 1e6,  # Hz
        "limit": np.array([49.5, 54, 56.9, 60, 60]),  # dBuV/m at 10m
        "unit": "dBuV/m",
        "distance": 10.0,
        "detector": "quasi-peak",
        "regulatory_body": "FCC",
        "document": "47 CFR 15.109",
    },
    "FCC_Part15_ClassB": {
        "description": "FCC Part 15 Class B (Residential) Radiated Emissions",
        "frequency": np.array([30, 88, 216, 960, 1000]) * 1e6,
        "limit": np.array([40, 43.5, 46, 54, 54]),  # dBuV/m at 3m
        "unit": "dBuV/m",
        "distance": 3.0,
        "detector": "quasi-peak",
        "regulatory_body": "FCC",
        "document": "47 CFR 15.109",
    },
    "FCC_Part15_ClassB_Conducted": {
        "description": "FCC Part 15 Class B Conducted Emissions",
        "frequency": np.array([0.15, 0.5, 5, 30]) * 1e6,
        "limit": np.array([66, 56, 56, 60]),  # dBuV quasi-peak
        "unit": "dBuV",
        "distance": 0,
        "detector": "quasi-peak",
        "regulatory_body": "FCC",
        "document": "47 CFR 15.107",
    },
    # CE/CISPR - European Standards
    "CE_CISPR22_ClassA": {
        "description": "CISPR 22 Class A (Commercial) Radiated Emissions",
        "frequency": np.array([30, 230, 1000]) * 1e6,
        "limit": np.array([40, 47, 47]),  # dBuV/m at 10m
        "unit": "dBuV/m",
        "distance": 10.0,
        "detector": "quasi-peak",
        "regulatory_body": "CE",
        "document": "EN 55022 / CISPR 22",
    },
    "CE_CISPR22_ClassB": {
        "description": "CISPR 22 Class B (Residential) Radiated Emissions",
        "frequency": np.array([30, 230, 1000]) * 1e6,
        "limit": np.array([30, 37, 37]),  # dBuV/m at 10m
        "unit": "dBuV/m",
        "distance": 10.0,
        "detector": "quasi-peak",
        "regulatory_body": "CE",
        "document": "EN 55022 / CISPR 22",
    },
    "CE_CISPR32_ClassA": {
        "description": "CISPR 32 Class A (Commercial) Radiated Emissions",
        "frequency": np.array([30, 230, 1000]) * 1e6,
        "limit": np.array([40, 47, 47]),
        "unit": "dBuV/m",
        "distance": 10.0,
        "detector": "quasi-peak",
        "regulatory_body": "CE",
        "document": "EN 55032 / CISPR 32",
    },
    "CE_CISPR32_ClassB": {
        "description": "CISPR 32 Class B (Residential) Radiated Emissions",
        "frequency": np.array([30, 230, 1000]) * 1e6,
        "limit": np.array([30, 37, 37]),
        "unit": "dBuV/m",
        "distance": 10.0,
        "detector": "quasi-peak",
        "regulatory_body": "CE",
        "document": "EN 55032 / CISPR 32",
    },
    "CE_CISPR32_ClassB_Conducted": {
        "description": "CISPR 32 Class B Conducted Emissions",
        "frequency": np.array([0.15, 0.5, 5, 30]) * 1e6,
        "limit_qp": np.array([66, 56, 56, 60]),  # Quasi-peak
        "limit_avg": np.array([56, 46, 46, 50]),  # Average
        "limit": np.array([66, 56, 56, 60]),  # Use QP as default
        "unit": "dBuV",
        "distance": 0,
        "detector": "quasi-peak",
        "regulatory_body": "CE",
        "document": "EN 55032 / CISPR 32",
    },
    # MIL-STD-461G - Military Standards
    "MIL_STD_461G_CE102": {
        "description": "MIL-STD-461G CE102 Conducted Emissions (10kHz-10MHz)",
        "frequency": np.array([0.01, 0.15, 0.5, 2, 10]) * 1e6,
        "limit": np.array([94, 80, 80, 80, 80]),  # dBuV
        "unit": "dBuV",
        "distance": 0,
        "detector": "peak",
        "regulatory_body": "MIL",
        "document": "MIL-STD-461G",
    },
    "MIL_STD_461G_RE102": {
        "description": "MIL-STD-461G RE102 Radiated Emissions (2MHz-18GHz)",
        "frequency": np.array([2, 30, 100, 200, 1000, 18000]) * 1e6,
        "limit": np.array([54, 54, 34, 34, 34, 34]),  # dBuV/m at 1m
        "unit": "dBuV/m",
        "distance": 1.0,
        "detector": "peak",
        "regulatory_body": "MIL",
        "document": "MIL-STD-461G",
    },
    "MIL_STD_461G_CS101": {
        "description": "MIL-STD-461G CS101 Conducted Susceptibility",
        "frequency": np.array([0.03, 0.15, 50]) * 1e6,
        "limit": np.array([6, 6, 6]),  # Vrms
        "unit": "Vrms",
        "distance": 0,
        "detector": "average",
        "regulatory_body": "MIL",
        "document": "MIL-STD-461G",
    },
}

# List of available mask names
AVAILABLE_MASKS = list(_BUILTIN_MASKS.keys())

# Mask aliases for convenience
_MASK_ALIASES: dict[str, str] = {
    "CISPR11_ClassB": "CE_CISPR22_ClassB",  # CISPR 11 is similar to CISPR 22
    "CISPR22": "CE_CISPR22_ClassB",  # Short alias for CISPR 22 Class B
    "CISPR22_ClassB": "CE_CISPR22_ClassB",
    "CISPR22_ClassA": "CE_CISPR22_ClassA",
    "CISPR32": "CE_CISPR32_ClassB",  # Short alias for CISPR 32 Class B
    "CISPR32_ClassB": "CE_CISPR32_ClassB",
    "CISPR32_ClassA": "CE_CISPR32_ClassA",
}


def load_limit_mask(
    name: str,
    custom_path: str | Path | None = None,
) -> LimitMask:
    """Load an EMC limit mask by name.

    Args:
        name: Mask name (see AVAILABLE_MASKS) or path to custom mask file.
        custom_path: Optional path to directory containing custom mask files.

    Returns:
        LimitMask object.

    Raises:
        ValueError: If mask name is unknown and no custom file found.

    Example:
        >>> mask = load_limit_mask('FCC_Part15_ClassB')
        >>> print(f"Limit at 100MHz: {mask.interpolate(np.array([100e6]))[0]:.1f} dBuV/m")
    """
    # Resolve aliases
    resolved_name = _MASK_ALIASES.get(name, name)

    # Check built-in masks first
    if resolved_name in _BUILTIN_MASKS:
        mask_data = _BUILTIN_MASKS[resolved_name].copy()
        return LimitMask(name=name, **mask_data)

    # Check custom path
    if custom_path is not None:
        custom_file = Path(custom_path) / f"{name}.json"
        if custom_file.exists():
            with open(custom_file) as f:
                data = json.load(f)
            return LimitMask.from_dict(data)

    # Try loading as JSON file path
    if Path(name).exists():
        with open(name) as f:
            data = json.load(f)
        return LimitMask.from_dict(data)

    raise ValueError(f"Unknown limit mask: {name}. Available masks: {', '.join(AVAILABLE_MASKS)}")


def create_custom_mask(
    name: str,
    frequencies: list[float] | NDArray[np.float64],
    limits: list[float] | NDArray[np.float64],
    unit: str = "dBuV",
    description: str = "",
    **kwargs: Any,
) -> LimitMask:
    """Create a custom limit mask.

    Args:
        name: Mask name.
        frequencies: Frequency points in Hz.
        limits: Limit values in specified unit.
        unit: Limit unit ('dBuV', 'dBm', 'dBuV/m').
        description: Human-readable description.
        **kwargs: Additional LimitMask attributes.

    Returns:
        LimitMask object.

    Raises:
        ValueError: If frequencies and limits have different lengths.

    Example:
        >>> mask = create_custom_mask(
        ...     name="MyLimit",
        ...     frequencies=[30e6, 100e6, 1000e6],
        ...     limits=[40, 35, 30],
        ...     unit="dBuV/m",
        ...     description="Custom radiated limit"
        ... )
    """
    freq_array = np.array(frequencies)
    limit_array = np.array(limits)

    # Validate lengths match
    if len(freq_array) != len(limit_array):
        raise ValueError(
            f"frequencies and limits must have the same length "
            f"(got {len(freq_array)} frequencies and {len(limit_array)} limits)"
        )

    # Auto-sort by frequency if not sorted
    if len(freq_array) > 1 and not np.all(np.diff(freq_array) > 0):
        sort_indices = np.argsort(freq_array)
        freq_array = freq_array[sort_indices]
        limit_array = limit_array[sort_indices]

    return LimitMask(
        name=name,
        description=description,
        frequency=freq_array,
        limit=limit_array,
        unit=unit,
        **kwargs,
    )


__all__ = [
    "AVAILABLE_MASKS",
    "LimitMask",
    "create_custom_mask",
    "load_limit_mask",
]
