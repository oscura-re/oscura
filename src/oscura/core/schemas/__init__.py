"""JSON Schema definitions for Oscura configuration types.

This module provides JSON Schema definitions for validating various
configuration file types used in Oscura, including packet formats,
device mappings, bus configurations, and protocol definitions.


Example:
    >>> from oscura.core.schemas import load_schema, validate_config
    >>> schema = load_schema("packet_format")
    >>> validate_config(config_dict, "packet_format")
    True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oscura.core.config.schema import (
    ConfigSchema,
    get_schema_registry,
    register_schema,
    validate_against_schema,
)

__all__ = [
    "SCHEMA_NAMES",
    "get_schema_path",
    "load_schema",
    "register_builtin_schemas",
    "validate_config",
]

# Schema type names
SCHEMA_NAMES = [
    "packet_format",
    "device_mapping",
    "bus_configuration",
    "protocol_definition",
]


def get_schema_path(schema_name: str) -> Path:
    """Get the file path for a schema definition.

    Args:
        schema_name: Schema name (e.g., "packet_format").

    Returns:
        Path to the JSON schema file.

    Raises:
        ValueError: If schema name is not recognized.
        FileNotFoundError: If schema file does not exist.
    """
    if schema_name not in SCHEMA_NAMES:
        raise ValueError(f"Unknown schema name: {schema_name}. Available schemas: {SCHEMA_NAMES}")

    schema_dir = Path(__file__).parent
    schema_file = schema_dir / f"{schema_name}.json"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    return schema_file


def load_schema(schema_name: str) -> dict[str, Any]:
    """Load a JSON schema definition from disk.

    Args:
        schema_name: Schema name (e.g., "packet_format").

    Returns:
        JSON schema dictionary.
    """
    schema_path = get_schema_path(schema_name)

    with open(schema_path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def validate_config(
    config: dict[str, Any],
    schema_name: str,
    *,
    strict: bool = False,
) -> bool:
    """Validate a configuration dictionary against a schema.

    Args:
        config: Configuration dictionary to validate.
        schema_name: Schema name (e.g., "packet_format").
        strict: If True, fail on additional properties.

    Returns:
        True if validation passes.

    Example:
        >>> config = {"name": "test", "packet": {...}, "header": {...}}
        >>> validate_config(config, "packet_format")
        True
    """
    # Use the existing validation system from config.schema
    return validate_against_schema(config, schema_name, strict=strict)


def register_builtin_schemas() -> None:
    """Register all built-in configuration schemas.

    This function registers the following schemas:
    - packet_format: Binary packet format configurations
    - device_mapping: Device ID to name mappings
    - bus_configuration: Parallel bus configurations
    - protocol_definition: Protocol DSL definitions



    The schemas are registered with the global SchemaRegistry and can
    be accessed via validate_against_schema() or get_schema_registry().
    """
    registry = get_schema_registry()

    # Load and register each schema
    for schema_name in SCHEMA_NAMES:
        # Skip if already registered
        if registry.has_schema(schema_name):
            continue

        schema_dict = load_schema(schema_name)

        # Extract version from $id if present
        version = "1.0.0"  # default
        if "$id" in schema_dict:
            # Extract version from URI like .../v1.0.0.json
            schema_id = schema_dict["$id"]
            if "/v" in schema_id:
                version_part = schema_id.split("/v")[-1]
                version = version_part.replace(".json", "")

        # Create ConfigSchema object
        config_schema = ConfigSchema(
            name=schema_name,
            version=version,
            schema=schema_dict,
            description=schema_dict.get("description", ""),
            uri=schema_dict.get("$id"),
        )

        # Register with global registry
        register_schema(config_schema, set_default=True)


# Auto-register schemas on module import
register_builtin_schemas()
