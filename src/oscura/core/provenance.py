"""Measurement provenance tracking for reproducibility.

This module provides provenance tracking to record the complete history
of how measurements were computed, including algorithms, parameters,
timestamps, and library versions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Oscura version dynamically imported from package metadata (SSOT: pyproject.toml)
try:
    from importlib.metadata import version

    OSCURA_VERSION = version("oscura")
except Exception:
    # Fallback for development/testing when package not installed
    OSCURA_VERSION = "0.0.0+dev"


@dataclass
class Provenance:
    """Provenance information for a computation.

    Tracks the complete chain of operations, parameters, and context
    for reproducibility and debugging.

    Attributes:
        algorithm: Name of algorithm or method used.
        parameters: Dictionary of parameters passed to the algorithm.
        timestamp: ISO 8601 timestamp of computation.
        library_version: Version of Oscura used.
        input_hash: Optional hash of input data for change detection.
        metadata: Additional context information.

    Example:
        >>> prov = Provenance(
        ...     algorithm='rise_time',
        ...     parameters={'ref_levels': (10, 90)},
        ...     timestamp='2025-12-21T10:30:00Z',
        ...     library_version='0.1.0'
        ... )

    References:
        API-011: Measurement Provenance Tracking
    """

    algorithm: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    library_version: str = OSCURA_VERSION
    input_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance to dictionary for serialization.

        Returns:
            Dictionary representation of provenance.

        Example:
            >>> prov_dict = prov.to_dict()
            >>> import json
            >>> json.dumps(prov_dict)
        """
        return {
            "algorithm": self.algorithm,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "library_version": self.library_version,
            "input_hash": self.input_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        """Create Provenance from dictionary.

        Args:
            data: Dictionary containing provenance fields.

        Returns:
            Provenance object.

        Example:
            >>> prov = Provenance.from_dict(prov_dict)
        """
        return cls(
            algorithm=data["algorithm"],
            parameters=data.get("parameters", {}),
            timestamp=data.get("timestamp", ""),
            library_version=data.get("library_version", OSCURA_VERSION),
            input_hash=data.get("input_hash"),
            metadata=data.get("metadata", {}),
        )

    def __str__(self) -> str:
        """Human-readable provenance summary."""
        lines = [
            f"Algorithm: {self.algorithm}",
            f"Timestamp: {self.timestamp}",
            f"Version: {self.library_version}",
        ]
        if self.parameters:
            params_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            lines.append(f"Parameters: {params_str}")
        if self.input_hash:
            lines.append(f"Input Hash: {self.input_hash[:16]}...")
        return "\n".join(lines)


@dataclass
class MeasurementResultWithProvenance:
    """Measurement result with full provenance tracking.

    Extends the basic measurement result with comprehensive provenance
    information for reproducibility and debugging.

    Attributes:
        value: Measured value.
        units: Units of measurement (e.g., 'V', 'Hz', 's').
        provenance: Provenance information.
        confidence: Optional confidence interval (low, high).

    Example:
        >>> result = MeasurementResultWithProvenance(
        ...     value=3.3,
        ...     units='V',
        ...     provenance=Provenance(
        ...         algorithm='peak_to_peak',
        ...         parameters={'window': (0, 1e-3)}
        ...     )
        ... )
        >>> print(result)
        3.3 V (peak_to_peak)

    References:
        API-011: Measurement Provenance Tracking
    """

    value: float
    units: str | None = None
    provenance: Provenance | None = None
    confidence: tuple[float, float] | None = None

    def is_equivalent(
        self,
        other: MeasurementResultWithProvenance,
        *,
        rtol: float = 1e-9,
        atol: float = 0.0,
        check_parameters: bool = True,
    ) -> bool:
        """Check if two results are equivalent.

        Compares values within tolerance and optionally checks if the
        same algorithm and parameters were used.

        Args:
            other: Other measurement result to compare.
            rtol: Relative tolerance for value comparison.
            atol: Absolute tolerance for value comparison.
            check_parameters: If True, also verify matching algorithm and parameters.

        Returns:
            True if results are equivalent.

        Example:
            >>> result1.is_equivalent(result2, rtol=1e-6)
            True
        """
        # Check value equivalence
        if not np.isclose(self.value, other.value, rtol=rtol, atol=atol):
            return False

        # Check units match
        if self.units != other.units:
            return False

        # Optionally check provenance
        if check_parameters and self.provenance and other.provenance:
            if self.provenance.algorithm != other.provenance.algorithm:
                return False
            # Check if critical parameters match
            if self.provenance.parameters != other.provenance.parameters:
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation including provenance.

        Example:
            >>> result_dict = result.to_dict()
            >>> import json
            >>> json_str = json.dumps(result_dict)
        """
        result: dict[str, Any] = {
            "value": self.value,
            "units": self.units,
        }
        if self.provenance:
            result["provenance"] = self.provenance.to_dict()
        if self.confidence:
            result["confidence"] = self.confidence
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementResultWithProvenance:
        """Create result from dictionary.

        Args:
            data: Dictionary containing result fields.

        Returns:
            MeasurementResultWithProvenance object.
        """
        provenance = None
        if "provenance" in data:
            provenance = Provenance.from_dict(data["provenance"])

        confidence = None
        if "confidence" in data:
            confidence = tuple(data["confidence"])

        return cls(
            value=data["value"],
            units=data.get("units"),
            provenance=provenance,
            confidence=confidence,
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [str(self.value)]
        if self.units:
            parts.append(self.units)
        if self.provenance:
            parts.append(f"({self.provenance.algorithm})")
        return " ".join(parts)

    def __repr__(self) -> str:
        """Detailed representation."""
        parts = [f"value={self.value}"]
        if self.units:
            parts.append(f"units='{self.units}'")
        if self.provenance:
            parts.append(f"algorithm='{self.provenance.algorithm}'")
        return f"MeasurementResultWithProvenance({', '.join(parts)})"

    def pretty_print(self) -> str:
        """Pretty-print result with full provenance.

        Returns:
            Multi-line formatted string with all details.

        Example:
            >>> print(result.pretty_print())
            Value: 3.3 V
            Algorithm: peak_to_peak
            Timestamp: 2025-12-21T10:30:00Z
            Version: 0.1.0
            Parameters: window=(0, 0.001)
        """
        lines = [f"Value: {self.value}"]
        if self.units:
            lines[-1] += f" {self.units}"

        if self.confidence:
            lines.append(f"Confidence: ({self.confidence[0]}, {self.confidence[1]})")

        if self.provenance:
            lines.append(str(self.provenance))

        return "\n".join(lines)


def compute_input_hash(data: NDArray[np.float64]) -> str:
    """Compute hash of input data for change detection.

    Uses SHA-256 hash of data array for reproducibility checks.

    Args:
        data: Input numpy array.

    Returns:
        Hexadecimal hash string.

    Example:
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> hash_str = compute_input_hash(data)

    References:
        API-011: Measurement Provenance Tracking
    """
    # Convert to bytes and hash
    data_bytes = data.tobytes()
    hash_obj = hashlib.sha256(data_bytes)
    return hash_obj.hexdigest()


def create_provenance(
    algorithm: str,
    parameters: dict[str, Any] | None = None,
    *,
    input_data: NDArray[np.float64] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Provenance:
    """Create provenance record for a computation.

    Convenience function to create provenance with automatic timestamp
    and optional input hash.

    Args:
        algorithm: Name of algorithm or method.
        parameters: Parameters used in computation.
        input_data: Optional input data to hash for change detection.
        metadata: Additional context information.

    Returns:
        Provenance object.

    Example:
        >>> import numpy as np
        >>> data = np.array([1.0, 2.0, 3.0])
        >>> prov = create_provenance(
        ...     algorithm='mean',
        ...     parameters={'axis': 0},
        ...     input_data=data
        ... )

    References:
        API-011: Measurement Provenance Tracking
    """
    input_hash = None
    if input_data is not None:
        input_hash = compute_input_hash(input_data)

    return Provenance(
        algorithm=algorithm,
        parameters=parameters or {},
        input_hash=input_hash,
        metadata=metadata or {},
    )


__all__ = [
    "MeasurementResultWithProvenance",
    "Provenance",
    "compute_input_hash",
    "create_provenance",
]
