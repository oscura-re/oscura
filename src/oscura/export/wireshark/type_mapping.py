"""Type mapping between Oscura field types and Wireshark ProtoField types.

This module provides mappings from Oscura protocol field types to their
corresponding Wireshark Lua ProtoField types and display bases.

References:
    https://wiki.wireshark.org/LuaAPI/Proto
"""

from __future__ import annotations

from typing import Literal

# Map Oscura field types to Wireshark ProtoField types
FIELD_TYPE_MAPPING: dict[str, str] = {
    "uint8": "ProtoField.uint8",
    "uint16": "ProtoField.uint16",
    "uint32": "ProtoField.uint32",
    "uint64": "ProtoField.uint64",
    "int8": "ProtoField.int8",
    "int16": "ProtoField.int16",
    "int32": "ProtoField.int32",
    "int64": "ProtoField.int64",
    "bytes": "ProtoField.bytes",
    "string": "ProtoField.string",
    "bool": "ProtoField.bool",
    "float32": "ProtoField.float",
    "float64": "ProtoField.double",
}

# Map display base names to Wireshark base constants
BASE_MAPPING: dict[str, str] = {
    "dec": "base.DEC",
    "hex": "base.HEX",
    "oct": "base.OCT",
    "bin": "base.BIN",
    "none": "base.NONE",
}

# Default display base for each field type
DEFAULT_BASE: dict[str, str] = {
    "uint8": "hex",
    "uint16": "hex",
    "uint32": "hex",
    "uint64": "hex",
    "int8": "dec",
    "int16": "dec",
    "int32": "dec",
    "int64": "dec",
    "bytes": "none",
    "string": "none",
    "bool": "none",
    "float32": "none",
    "float64": "none",
}


def get_protofield_type(field_type: str, display_base: str | None = None) -> tuple[str, str]:
    """Map Oscura field type to Wireshark ProtoField and display base.

    Args:
        field_type: Oscura field type (uint8, uint16, string, etc.)
        display_base: Optional display base (dec, hex, oct, bin, none)

    Returns:
        Tuple of (protofield_type, base_constant)
        Example: ("ProtoField.uint16", "base.HEX")

    Raises:
        ValueError: If field type is unknown
    """
    if field_type not in FIELD_TYPE_MAPPING:
        raise ValueError(f"Unknown field type: {field_type}")

    protofield = FIELD_TYPE_MAPPING[field_type]

    # Determine display base
    if display_base is None:
        display_base = DEFAULT_BASE[field_type]

    if display_base not in BASE_MAPPING:
        raise ValueError(f"Unknown display base: {display_base}")

    base_constant = BASE_MAPPING[display_base]

    return protofield, base_constant


def get_field_size(field_type: str) -> int | None:
    """Get the fixed size in bytes for a field type.

    Args:
        field_type: Oscura field type

    Returns:
        Field size in bytes, or None for variable-length types
    """
    size_map: dict[str, int | None] = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 2,
        "uint32": 4,
        "int32": 4,
        "uint64": 8,
        "int64": 8,
        "float32": 4,
        "float64": 8,
        "bool": 1,
        "bytes": None,  # Variable length
        "string": None,  # Variable length
    }
    return size_map.get(field_type)


def is_variable_length(field_type: str) -> bool:
    """Check if a field type is variable length.

    Args:
        field_type: Oscura field type

    Returns:
        True if field is variable length
    """
    return get_field_size(field_type) is None


def get_lua_reader_function(field_type: str, endian: Literal["big", "little"] = "big") -> str:
    """Get the Lua buffer reader function for a field type.

    Args:
        field_type: Oscura field type
        endian: Byte order (big or little endian)

    Returns:
        Lua function name (e.g., "uint16", "le_uint16")

    Raises:
        ValueError: If field type is unknown
    """
    if field_type not in FIELD_TYPE_MAPPING:
        raise ValueError(f"Unknown field type: {field_type}")

    # Map field types to Lua buffer reader methods
    reader_map: dict[str, str] = {
        "uint8": "uint",
        "int8": "int",
        "uint16": "uint16",
        "int16": "int16",
        "uint32": "uint32",
        "int32": "int32",
        "uint64": "uint64",
        "int64": "int64",
        "float32": "float",
        "float64": "double",
        "bool": "uint",
        "bytes": "bytes",
        "string": "string",
    }

    reader = reader_map[field_type]

    # Add little-endian prefix if needed
    if endian == "little" and field_type not in ["uint8", "int8", "bool", "bytes", "string"]:
        return f"le_{reader}"

    return reader
