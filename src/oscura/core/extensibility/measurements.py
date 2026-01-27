"""Custom measurement framework for user-defined measurements.

This module implements a framework for defining and registering custom
measurements that integrate seamlessly with batch processing and export.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .registry import AlgorithmRegistry

if TYPE_CHECKING:
    from collections.abc import Callable

    from oscura.core.types import WaveformTrace


@dataclass
class MeasurementDefinition:
    """Definition of a custom measurement with metadata.

    Defines a measurement function along with metadata about units, category,
    and documentation. Measurements can be registered globally and used in
    batch processing.

    Attributes:
        name: Unique name for the measurement.
        func: Callable that computes the measurement.
        units: Units of measurement (e.g., 'V', 'Hz', 's', 'ratio').
        category: Measurement category (e.g., 'amplitude', 'timing', 'frequency').
        description: Human-readable description.
        tags: Optional tags for categorization and search.

    Example:
        >>> import oscura as osc
        >>> def calculate_crest_factor(trace, **kwargs):
        ...     peak = abs(trace.data).max()
        ...     rms = (trace.data ** 2).mean() ** 0.5
        ...     return peak / rms
        >>> osc.register_measurement(
        ...     name='crest_factor',
        ...     func=calculate_crest_factor,
        ...     units='ratio',
        ...     category='amplitude'
        ... )
        >>> cf = osc.measure(trace, 'crest_factor')

    Advanced Example:
        >>> # Define measurement with full metadata
        >>> slew_rate_defn = osc.MeasurementDefinition(
        ...     name='max_slew_rate',
        ...     func=lambda trace: abs(trace.derivative()).max(),
        ...     units='V/s',
        ...     category='edge',
        ...     description='Maximum slew rate in trace',
        ...     tags=['edge', 'derivative', 'speed']
        ... )
        >>> osc.register_measurement(slew_rate_defn)

    References:
        API-008: Custom Measurement Framework
        API-006: Algorithm Override Hooks
    """

    name: str
    func: Callable[[WaveformTrace], float]
    units: str
    category: str
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate measurement definition.

        Raises:
            ValueError: If measurement name is empty.
            TypeError: If func is not callable or has invalid signature.
        """
        if not self.name:
            raise ValueError("Measurement name cannot be empty")

        if not callable(self.func):
            raise TypeError(f"Measurement func must be callable, got {type(self.func).__name__}")

        # Validate function signature
        self._validate_signature()

    def _validate_signature(self) -> None:
        """Validate that function has correct signature.

        Measurement functions should accept (trace, **kwargs) -> float.

        Raises:
            TypeError: If signature is invalid.
        """
        sig = inspect.signature(self.func)
        params = list(sig.parameters.values())

        # Should have at least one parameter (trace)
        if len(params) == 0:
            raise TypeError(
                f"Measurement function must accept at least one parameter "
                f"(trace). Got {self.func.__name__} with no parameters."
            )

        # Check if first parameter could accept WaveformTrace
        first_param = params[0]
        if first_param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"First parameter must be a regular parameter (trace), got {first_param.kind}"
            )

    def __call__(self, trace: WaveformTrace, **kwargs: Any) -> float:
        """Call measurement function.

        Args:
            trace: WaveformTrace to measure.
            **kwargs: Additional parameters for measurement.

        Returns:
            Measured value.

        Example:
            >>> defn = MeasurementDefinition(
            ...     name='peak',
            ...     func=lambda trace: abs(trace.data).max(),
            ...     units='V',
            ...     category='amplitude'
            ... )
            >>> value = defn(trace)
        """
        return self.func(trace, **kwargs)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            String representation of the measurement definition.
        """
        return (
            f"MeasurementDefinition(name='{self.name}', "
            f"units='{self.units}', category='{self.category}')"
        )


class MeasurementRegistry:
    """Registry for custom measurements.

    Manages registration and lookup of custom measurements. Integrates with
    the AlgorithmRegistry for storage.

    Example:
        >>> registry = MeasurementRegistry()
        >>> registry.register(
        ...     name='crest_factor',
        ...     func=calculate_crest_factor,
        ...     units='ratio',
        ...     category='amplitude'
        ... )
        >>> measurement = registry.get('crest_factor')
        >>> value = measurement(trace)

    References:
        API-008: Custom Measurement Framework
    """

    MEASUREMENT_CATEGORY = "measurement"

    def __init__(self) -> None:
        """Initialize measurement registry."""
        self._definitions: dict[str, MeasurementDefinition] = {}
        self._algorithm_registry = AlgorithmRegistry()

    def register(
        self,
        name: str | None = None,
        func: Callable[[WaveformTrace], float] | None = None,
        units: str | None = None,
        category: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        definition: MeasurementDefinition | None = None,
    ) -> None:
        """Register a custom measurement.

        Can be called with individual parameters or with a MeasurementDefinition.

        Args:
            name: Measurement name (required if definition not provided).
            func: Measurement function (required if definition not provided).
            units: Units of measurement (required if definition not provided).
            category: Measurement category (required if definition not provided).
            description: Optional description.
            tags: Optional tags.
            definition: Pre-built MeasurementDefinition (alternative to individual args).

        Raises:
            ValueError: If required parameters missing or name already exists.

        Example:
            >>> registry = MeasurementRegistry()
            >>> # Register with individual parameters
            >>> registry.register(
            ...     name='peak',
            ...     func=lambda trace: abs(trace.data).max(),
            ...     units='V',
            ...     category='amplitude'
            ... )
            >>> # Register with definition
            >>> defn = MeasurementDefinition(...)
            >>> registry.register(definition=defn)
        """
        # Handle definition argument
        if definition is not None:
            defn = definition
        else:
            # Validate required parameters
            if name is None or func is None or units is None or category is None:
                raise ValueError(
                    "Must provide either 'definition' or all of (name, func, units, category)"
                )

            defn = MeasurementDefinition(
                name=name,
                func=func,
                units=units,
                category=category,
                description=description,
                tags=tags or [],
            )

        # Check for duplicates
        if defn.name in self._definitions:
            raise ValueError(f"Measurement '{defn.name}' already registered")

        # Register in both registries
        self._definitions[defn.name] = defn
        self._algorithm_registry.register(
            name=defn.name,
            func=defn.func,
            category=self.MEASUREMENT_CATEGORY,
            validate=False,  # Already validated by MeasurementDefinition
        )

    def get(self, name: str) -> MeasurementDefinition:
        """Get measurement definition by name.

        Args:
            name: Measurement name.

        Returns:
            MeasurementDefinition for the measurement.

        Raises:
            KeyError: If measurement not found.

        Example:
            >>> measurement = registry.get('crest_factor')
            >>> value = measurement(trace)
        """
        if name not in self._definitions:
            available = list(self._definitions.keys())
            raise KeyError(f"Measurement '{name}' not found. Available: {available}")

        return self._definitions[name]

    def has_measurement(self, name: str) -> bool:
        """Check if measurement is registered.

        Args:
            name: Measurement name.

        Returns:
            True if measurement is registered.

        Example:
            >>> if registry.has_measurement('crest_factor'):
            ...     cf = registry.get('crest_factor')(trace)
        """
        return name in self._definitions

    def list_measurements(
        self,
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """List registered measurements.

        Args:
            category: Filter by category (optional).
            tags: Filter by tags (optional).

        Returns:
            List of measurement names.

        Example:
            >>> # List all measurements
            >>> all_measurements = registry.list_measurements()
            >>> # List amplitude measurements
            >>> amplitude = registry.list_measurements(category='amplitude')
            >>> # List measurements with 'edge' tag
            >>> edge_measurements = registry.list_measurements(tags=['edge'])
        """
        measurements = []

        for name, defn in self._definitions.items():
            # Filter by category
            if category is not None and defn.category != category:
                continue

            # Filter by tags
            if tags is not None and not any(tag in defn.tags for tag in tags):
                continue

            measurements.append(name)

        return measurements

    def get_metadata(self, name: str) -> dict[str, Any]:
        """Get metadata for a measurement.

        Args:
            name: Measurement name.

        Returns:
            Dictionary with measurement metadata.

        Example:
            >>> metadata = registry.get_metadata('crest_factor')
            >>> print(f"Units: {metadata['units']}")
            >>> print(f"Category: {metadata['category']}")
        """
        defn = self.get(name)
        return {
            "name": defn.name,
            "units": defn.units,
            "category": defn.category,
            "description": defn.description,
            "tags": defn.tags,
        }

    def unregister(self, name: str) -> None:
        """Unregister a measurement.

        Args:
            name: Measurement name.

        Example:
            >>> registry.unregister('crest_factor')
        """
        if name in self._definitions:
            del self._definitions[name]

        if self._algorithm_registry.has_algorithm(self.MEASUREMENT_CATEGORY, name):
            self._algorithm_registry.unregister(self.MEASUREMENT_CATEGORY, name)


# Global measurement registry
_registry = MeasurementRegistry()


def register_measurement(
    name: str | None = None,
    func: Callable[[WaveformTrace], float] | None = None,
    units: str | None = None,
    category: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
    definition: MeasurementDefinition | None = None,
) -> None:
    """Register a custom measurement in the global registry.

    Convenience function for registering measurements without accessing
    the registry directly.

    Args:
        name: Measurement name.
        func: Measurement function.
        units: Units of measurement.
        category: Measurement category.
        description: Optional description.
        tags: Optional tags.
        definition: Pre-built MeasurementDefinition.

    Example:
        >>> import oscura as osc
        >>> def calculate_crest_factor(trace, **kwargs):
        ...     peak = abs(trace.data).max()
        ...     rms = (trace.data ** 2).mean() ** 0.5
        ...     return peak / rms
        >>> osc.register_measurement(
        ...     name='crest_factor',
        ...     func=calculate_crest_factor,
        ...     units='ratio',
        ...     category='amplitude'
        ... )

    References:
        API-008: Custom Measurement Framework
    """
    _registry.register(
        name=name,
        func=func,
        units=units,
        category=category,
        description=description,
        tags=tags,
        definition=definition,
    )


def measure(trace: WaveformTrace, name: str, **kwargs: Any) -> float:
    """Execute a registered measurement.

    Args:
        trace: WaveformTrace to measure.
        name: Measurement name.
        **kwargs: Additional parameters for the measurement.

    Returns:
        Measured value.

    Example:
        >>> import oscura as osc
        >>> cf = osc.measure(trace, 'crest_factor')
        >>> print(f"Crest factor: {cf:.2f}")

    References:
        API-008: Custom Measurement Framework
    """
    defn = _registry.get(name)
    return defn(trace, **kwargs)


def list_measurements(
    category: str | None = None,
    tags: list[str] | None = None,
) -> list[str]:
    """List registered measurements.

    Args:
        category: Filter by category (optional).
        tags: Filter by tags (optional).

    Returns:
        List of measurement names.

    Example:
        >>> import oscura as osc
        >>> measurements = osc.list_measurements(category='amplitude')
        >>> print(f"Amplitude measurements: {measurements}")

    References:
        API-008: Custom Measurement Framework
    """
    return _registry.list_measurements(category=category, tags=tags)


def get_measurement_registry() -> MeasurementRegistry:
    """Get the global measurement registry.

    Returns:
        Global MeasurementRegistry instance.

    Example:
        >>> registry = osc.get_measurement_registry()
        >>> metadata = registry.get_metadata('crest_factor')
    """
    return _registry


__all__ = [
    "MeasurementDefinition",
    "MeasurementRegistry",
    "get_measurement_registry",
    "list_measurements",
    "measure",
    "register_measurement",
]
