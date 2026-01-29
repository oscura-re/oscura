"""JSON Schema validation system for Oscura configuration.

This module provides a flexible schema validation system using JSON Schema
for validating configuration files including protocols, pipelines, and
threshold configurations.


Example:
    >>> from oscura.core.config.schema import validate_against_schema
    >>> config = {"name": "uart", "baud_rate": 115200}
    >>> validate_against_schema(config, "protocol")
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Try to import jsonschema for full validation
try:
    import jsonschema  # noqa: F401
    from jsonschema import Draft7Validator
    from jsonschema import ValidationError as JsonSchemaError

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    JsonSchemaError = Exception

from oscura.core.exceptions import ConfigurationError
from oscura.core.exceptions import ValidationError as OscuraValidationError


class ValidationError(OscuraValidationError):
    """Schema validation error with detailed location information.

    Attributes:
        path: JSON path to the invalid field.
        line: Line number in source file (if available).
        column: Column number in source file (if available).
        schema_path: Path in schema where validation failed.
    """

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        column: int | None = None,
        schema_path: str | None = None,
        expected: Any = None,
        actual: Any = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize ValidationError.

        Args:
            message: Description of the validation failure.
            path: JSON path to invalid field (e.g., "protocol.timing.baud_rate").
            line: Line number in source file.
            column: Column number in source file.
            schema_path: Path in schema where validation failed.
            expected: Expected value or type.
            actual: Actual value found.
            suggestion: Suggested fix.
        """
        self.path = path
        self.line = line
        self.column = column
        self.schema_path = schema_path
        self.expected = expected
        self.actual = actual
        self.suggestion = suggestion

        # Build detailed message
        details_parts = []
        if path:
            details_parts.append(f"Path: {path}")
        if line is not None:
            location = f"Line {line}"
            if column is not None:
                location += f", column {column}"
            details_parts.append(location)
        if expected is not None:
            details_parts.append(f"Expected: {expected}")
        if actual is not None:
            details_parts.append(f"Got: {actual}")

        super().__init__(
            message,
            field=path,
            constraint=schema_path,
            value=actual,
        )


@dataclass
class ConfigSchema:
    """Schema definition with metadata.

    Attributes:
        name: Schema identifier (e.g., "protocol", "pipeline").
        version: Schema version (semver format).
        schema: JSON Schema dictionary.
        description: Human-readable description.
        uri: Optional URI for schema reference.
    """

    name: str
    version: str
    schema: dict[str, Any]
    description: str = ""
    uri: str | None = None

    def __post_init__(self) -> None:
        """Validate schema after initialization."""
        if not self.name:
            raise ValueError("Schema name cannot be empty")
        if not self.version:
            raise ValueError("Schema version cannot be empty")
        if not self.schema:
            raise ValueError("Schema cannot be empty")

    @property
    def full_uri(self) -> str:
        """Get full schema URI.

        Returns:
            URI for schema reference, or generated local path if not provided.
        """
        if self.uri:
            return self.uri
        return f"urn:oscura:schemas:{self.name}:v{self.version}"


class SchemaRegistry:
    """Central registry for all configuration schemas.

    Provides O(1) lookup of schemas by name and version.

    Example:
        >>> registry = SchemaRegistry()
        >>> registry.register(protocol_schema)
        >>> schema = registry.get("protocol")
    """

    def __init__(self) -> None:
        """Initialize empty schema registry."""
        self._schemas: dict[str, dict[str, ConfigSchema]] = {}
        self._default_versions: dict[str, str] = {}

    def register(
        self,
        schema: ConfigSchema,
        *,
        set_default: bool = True,
    ) -> None:
        """Register a schema with the registry.

        Args:
            schema: Schema to register.
            set_default: If True, set as default version for this schema name.

        Raises:
            ValueError: If schema with same name and version already exists.
        """
        if schema.name not in self._schemas:
            self._schemas[schema.name] = {}

        if schema.version in self._schemas[schema.name]:
            self._schemas[schema.name][schema.version]
            raise ValueError(f"Schema '{schema.name}' v{schema.version} already registered")

        self._schemas[schema.name][schema.version] = schema

        if set_default:
            self._default_versions[schema.name] = schema.version

    def get(
        self,
        name: str,
        version: str | None = None,
    ) -> ConfigSchema | None:
        """Get schema by name and optional version.

        Args:
            name: Schema name (e.g., "protocol").
            version: Specific version or None for default.

        Returns:
            ConfigSchema if found, None otherwise.
        """
        if name not in self._schemas:
            return None

        if version is None:
            version = self._default_versions.get(name)
            if version is None:
                return None

        return self._schemas[name].get(version)

    def list_schemas(self) -> list[str]:
        """List all registered schema names.

        Returns:
            List of schema names.
        """
        return list(self._schemas.keys())

    def list_versions(self, name: str) -> list[str]:
        """List all versions of a schema.

        Args:
            name: Schema name.

        Returns:
            List of version strings.
        """
        if name not in self._schemas:
            return []
        return list(self._schemas[name].keys())

    def has_schema(self, name: str, version: str | None = None) -> bool:
        """Check if schema exists.

        Args:
            name: Schema name.
            version: Specific version or None for any.

        Returns:
            True if schema exists.
        """
        if name not in self._schemas:
            return False
        if version is None:
            return True
        return version in self._schemas[name]


# Global schema registry
_global_registry: SchemaRegistry | None = None


def get_schema_registry() -> SchemaRegistry:
    """Get the global schema registry.

    Initializes with built-in schemas on first call.

    Returns:
        Global SchemaRegistry instance.
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = SchemaRegistry()
        _register_builtin_schemas(_global_registry)

    return _global_registry


def register_schema(
    schema: ConfigSchema,
    *,
    set_default: bool = True,
) -> None:
    """Register a schema with the global registry.

    Args:
        schema: Schema to register.
        set_default: If True, set as default version.
    """
    get_schema_registry().register(schema, set_default=set_default)


def validate_against_schema(
    config: dict[str, Any],
    schema_name: str,
    *,
    version: str | None = None,
    strict: bool = False,
) -> bool:
    """Validate configuration against a registered schema.

    Args:
        config: Configuration dictionary to validate.
        schema_name: Name of schema to validate against.
        version: Specific schema version or None for default.
        strict: If True, fail on additional properties.

    Returns:
        True if validation passes.

    Raises:
        ValidationError: If validation fails with detailed error info.
        ConfigurationError: If schema not found or jsonschema not available.
    """
    if not JSONSCHEMA_AVAILABLE:
        raise ConfigurationError(
            "JSON Schema validation not available",
            fix_hint="Install jsonschema: pip install jsonschema",
        )

    registry = get_schema_registry()
    schema_obj = registry.get(schema_name, version)

    if schema_obj is None:
        available = registry.list_schemas()
        raise ConfigurationError(
            f"Schema '{schema_name}' not found",
            details=f"Available schemas: {available}",
        )

    schema = schema_obj.schema.copy()

    # Add strict mode
    if strict and "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    try:
        validator = Draft7Validator(schema)
        errors = list(validator.iter_errors(config))

        if errors:
            # Get first error for main message
            error = errors[0]
            path = ".".join(str(p) for p in error.absolute_path) or "(root)"

            # Try to provide helpful suggestion
            suggestion = _get_error_suggestion(error)

            raise ValidationError(
                str(error.message),
                path=path,
                schema_path=".".join(str(p) for p in error.absolute_schema_path),
                expected=error.schema.get("type") or error.schema.get("enum"),
                actual=error.instance,
                suggestion=suggestion,
            )

        return True

    except JsonSchemaError as e:
        path = ".".join(str(p) for p in e.absolute_path) if e.absolute_path else None  # type: ignore[assignment]
        raise ValidationError(
            str(e.message),
            path=path,
            schema_path=".".join(str(p) for p in e.absolute_schema_path)
            if e.absolute_schema_path
            else None,
        ) from e


def _get_error_suggestion(error: Any) -> str | None:
    """Generate suggestion for common validation errors.

    Args:
        error: jsonschema ValidationError.

    Returns:
        Suggestion string or None.
    """
    msg = error.message.lower()

    if "is not of type" in msg:
        expected_type = error.schema.get("type", "unknown")
        return f"Convert value to {expected_type}"

    if "is not valid under any of the given schemas" in msg:
        return "Check value matches one of the allowed formats"

    if "is a required property" in msg:
        return "Add the missing required field"

    if "additional properties" in msg:
        return "Remove unrecognized fields or use additionalProperties: true"

    if "does not match" in msg:
        pattern = error.schema.get("pattern")
        if pattern:
            return f"Value must match pattern: {pattern}"

    return None


def _register_builtin_schemas(registry: SchemaRegistry) -> None:
    """Register all built-in schemas.

    Args:
        registry: Registry to populate.
    """
    _register_protocol_schema(registry)
    _register_pipeline_schema(registry)
    _register_logic_family_schema(registry)
    _register_threshold_profile_schema(registry)
    _register_preferences_schema(registry)


def _register_protocol_schema(registry: SchemaRegistry) -> None:
    """Register protocol definition schema.

    Args:
        registry: Schema registry to populate.
    """
    schema = _build_protocol_schema()
    registry.register(
        ConfigSchema(
            name="protocol",
            version="1.0.0",
            description="Protocol decoder configuration",
            schema=schema,
        )
    )


def _build_protocol_schema() -> dict[str, Any]:
    """Build protocol schema definition.

    Returns:
        JSON Schema dictionary for protocol configuration.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": _build_protocol_name_property(),
            "version": _build_semver_property(),
            "description": {"type": "string"},
            "author": {"type": "string"},
            "timing": _build_timing_property(),
            "voltage_levels": _build_voltage_levels_property(),
            "state_machine": _build_state_machine_property(),
        },
        "additionalProperties": True,
    }


def _build_protocol_name_property() -> dict[str, Any]:
    """Build protocol name property definition.

    Returns:
        Property schema for protocol name.
    """
    return {
        "type": "string",
        "description": "Protocol identifier",
        "pattern": "^[a-z][a-z0-9_]*$",
    }


def _build_semver_property() -> dict[str, Any]:
    """Build semantic version property definition.

    Returns:
        Property schema for semver strings.
    """
    return {
        "type": "string",
        "description": "Protocol version (semver)",
        "pattern": "^\\d+\\.\\d+\\.\\d+$",
    }


def _build_timing_property() -> dict[str, Any]:
    """Build timing configuration property.

    Returns:
        Property schema for timing parameters.
    """
    return {
        "type": "object",
        "properties": {
            "baud_rates": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
            },
            "data_bits": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1, "maximum": 32},
            },
            "stop_bits": {
                "type": "array",
                "items": {"type": "number", "minimum": 0.5, "maximum": 2},
            },
            "parity": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["none", "even", "odd", "mark", "space"],
                },
            },
        },
    }


def _build_voltage_levels_property() -> dict[str, Any]:
    """Build voltage levels property.

    Returns:
        Property schema for voltage level specifications.
    """
    return {
        "type": "object",
        "properties": {
            "logic_family": {"type": "string"},
            "idle_state": {"type": "string", "enum": ["high", "low"]},
            "mark_voltage": {"type": "number"},
            "space_voltage": {"type": "number"},
        },
    }


def _build_state_machine_property() -> dict[str, Any]:
    """Build state machine property.

    Returns:
        Property schema for state machine definitions.
    """
    return {
        "type": "object",
        "properties": {
            "states": {
                "type": "array",
                "items": {"type": "string"},
            },
            "initial_state": {"type": "string"},
            "transitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["from", "to", "condition"],
                    "properties": {
                        "from": {"type": "string"},
                        "to": {"type": "string"},
                        "condition": {"type": "string"},
                    },
                },
            },
        },
    }


def _register_pipeline_schema(registry: SchemaRegistry) -> None:
    """Register pipeline definition schema.

    Args:
        registry: Schema registry to populate.
    """
    schema = _build_pipeline_schema()
    registry.register(
        ConfigSchema(
            name="pipeline",
            version="1.0.0",
            description="Analysis pipeline configuration",
            schema=schema,
        )
    )


def _build_pipeline_schema() -> dict[str, Any]:
    """Build pipeline schema definition.

    Returns:
        JSON Schema dictionary for pipeline configuration.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["name", "steps"],
        "properties": {
            "name": {"type": "string", "description": "Pipeline identifier"},
            "version": _build_semver_property(),
            "description": {"type": "string"},
            "steps": _build_pipeline_steps_property(),
            "parallel_groups": _build_parallel_groups_property(),
        },
    }


def _build_pipeline_steps_property() -> dict[str, Any]:
    """Build pipeline steps property.

    Returns:
        Property schema for pipeline step definitions.
    """
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["name", "type"],
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "params": {"type": "object"},
                "inputs": {"type": "object"},
                "outputs": {"type": "object"},
            },
        },
    }


def _build_parallel_groups_property() -> dict[str, Any]:
    """Build parallel groups property.

    Returns:
        Property schema for parallel execution groups.
    """
    return {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "string"},
        },
    }


def _register_logic_family_schema(registry: SchemaRegistry) -> None:
    """Register logic family voltage threshold schema.

    Args:
        registry: Schema registry to populate.
    """
    schema = _build_logic_family_schema()
    registry.register(
        ConfigSchema(
            name="logic_family",
            version="1.0.0",
            description="Logic family voltage thresholds",
            schema=schema,
        )
    )


def _build_logic_family_schema() -> dict[str, Any]:
    """Build logic family schema definition.

    Returns:
        JSON Schema dictionary for logic family voltage specifications.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["name", "VIH", "VIL", "VOH", "VOL"],
        "properties": {
            "name": {"type": "string", "description": "Logic family name"},
            "description": {"type": "string"},
            "VIH": _build_voltage_property("Input high voltage threshold (V)"),
            "VIL": _build_voltage_property("Input low voltage threshold (V)"),
            "VOH": _build_voltage_property("Output high voltage (V)"),
            "VOL": _build_voltage_property("Output low voltage (V)"),
            "VCC": _build_supply_voltage_property(),
            "temperature_range": _build_temperature_range_property(),
            "noise_margin_high": {"type": "number", "description": "High state noise margin (V)"},
            "noise_margin_low": {"type": "number", "description": "Low state noise margin (V)"},
        },
    }


def _build_voltage_property(description: str) -> dict[str, Any]:
    """Build standard voltage property with 0-10V range.

    Args:
        description: Property description.

    Returns:
        Property schema for voltage value.
    """
    return {"type": "number", "description": description, "minimum": 0, "maximum": 10}


def _build_supply_voltage_property() -> dict[str, Any]:
    """Build supply voltage property.

    Returns:
        Property schema for VCC with 0-15V range.
    """
    return {"type": "number", "description": "Supply voltage (V)", "minimum": 0, "maximum": 15}


def _build_temperature_range_property() -> dict[str, Any]:
    """Build temperature range property.

    Returns:
        Property schema for temperature range.
    """
    return {
        "type": "object",
        "properties": {
            "min": {"type": "number"},
            "max": {"type": "number"},
        },
    }


def _register_threshold_profile_schema(registry: SchemaRegistry) -> None:
    """Register threshold profile schema.

    Args:
        registry: Schema registry to populate.
    """
    schema = _build_threshold_profile_schema()
    registry.register(
        ConfigSchema(
            name="threshold_profile",
            version="1.0.0",
            description="Analysis threshold profile",
            schema=schema,
        )
    )


def _build_threshold_profile_schema() -> dict[str, Any]:
    """Build threshold profile schema definition.

    Returns:
        JSON Schema dictionary for threshold profile configuration.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "base_family": {"type": "string", "description": "Base logic family to extend"},
            "overrides": {"type": "object", "additionalProperties": {"type": "number"}},
            "tolerance": _build_tolerance_property(),
        },
    }


def _build_tolerance_property() -> dict[str, Any]:
    """Build tolerance property.

    Returns:
        Property schema for tolerance percentage (0-100).
    """
    return {
        "type": "number",
        "description": "Tolerance percentage (0-100)",
        "minimum": 0,
        "maximum": 100,
        "default": 0,
    }


def _register_preferences_schema(registry: SchemaRegistry) -> None:
    """Register user preferences schema.

    Args:
        registry: Schema registry to populate.
    """
    schema = _build_preferences_schema()
    registry.register(
        ConfigSchema(
            name="preferences",
            version="1.0.0",
            description="User preferences",
            schema=schema,
        )
    )


def _build_preferences_schema() -> dict[str, Any]:
    """Build preferences schema definition.

    Returns:
        JSON Schema dictionary for user preferences.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "defaults": _build_defaults_property(),
            "visualization": _build_visualization_property(),
            "export": _build_export_property(),
            "logging": _build_logging_property(),
        },
    }


def _build_defaults_property() -> dict[str, Any]:
    """Build defaults property for preferences.

    Returns:
        Property schema for default settings.
    """
    return {
        "type": "object",
        "properties": {
            "sample_rate": {"type": "number", "minimum": 0},
            "window_function": {"type": "string"},
            "fft_size": {"type": "integer", "minimum": 1},
        },
    }


def _build_visualization_property() -> dict[str, Any]:
    """Build visualization property for preferences.

    Returns:
        Property schema for visualization settings.
    """
    return {
        "type": "object",
        "properties": {
            "style": {"type": "string"},
            "figure_size": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
            },
            "dpi": {"type": "integer", "minimum": 50, "maximum": 600},
            "colormap": {"type": "string"},
        },
    }


def _build_export_property() -> dict[str, Any]:
    """Build export property for preferences.

    Returns:
        Property schema for export settings.
    """
    return {
        "type": "object",
        "properties": {
            "default_format": {
                "type": "string",
                "enum": ["csv", "hdf5", "npz", "json"],
            },
            "precision": {"type": "integer", "minimum": 1, "maximum": 15},
        },
    }


def _build_logging_property() -> dict[str, Any]:
    """Build logging property for preferences.

    Returns:
        Property schema for logging settings.
    """
    return {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
            },
            "file": {"type": "string"},
        },
    }


__all__ = [
    "ConfigSchema",
    "SchemaRegistry",
    "ValidationError",
    "get_schema_registry",
    "register_schema",
    "validate_against_schema",
]
