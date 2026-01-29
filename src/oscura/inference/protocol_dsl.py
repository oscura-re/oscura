"""Protocol Definition Language parser and decoder generator.

Requirements addressed: PSI-004

This module provides a declarative DSL for defining custom protocol formats
that can be used to generate decoders and encoders automatically.

Key capabilities:
- Parse YAML-based protocol definitions
- Support all common field types
- Conditional fields and length-prefixed data
- Generate efficient decoders and encoders
- Comprehensive error reporting
"""

from __future__ import annotations

import ast
import operator
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class FieldDefinition:
    """Protocol field definition.

    : Field specification.

    Attributes:
        name: Field name
        field_type: Field type (uint8, uint16, int32, float32, bytes, string, bitfield, array, struct)
                   Also accessible as 'type' for compatibility.
        size: Field size (literal or reference to length field)
        offset: Field offset (optional, auto-calculated if not provided)
        endian: Byte order ('big' or 'little')
        condition: Conditional expression for optional fields
        enum: Enumeration mapping for integer fields
        validation: Validation rules
        default: Default value
        description: Human-readable description
        value: Expected value for constant fields
        size_ref: Reference to length field (alias for size when string)
        element: Element definition for array types (contains type and optionally fields for struct)
        count_field: Field name that contains the array count
        count: Fixed array count
        fields: List of nested field definitions for struct types
    """

    name: str
    field_type: str = (
        "uint8"  # uint8, uint16, int32, float32, bytes, string, bitfield, array, struct
    )
    size: int | str | None = None  # Can be literal or reference to length field
    offset: int | None = None
    endian: Literal["big", "little"] = "big"
    condition: str | None = None  # Conditional field
    enum: dict[int, str] | dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    default: Any = None
    description: str = ""
    value: Any = None  # Expected constant value
    size_ref: str | None = None  # Alias for size reference
    # Array/struct specific fields
    element: dict[str, Any] | None = None  # Element definition for arrays
    count_field: str | None = None  # Field containing array count
    count: int | None = None  # Fixed array count
    fields: list[FieldDefinition] | None = None  # Nested fields for struct type

    def __post_init__(self) -> None:
        """Handle size_ref as alias for size."""
        if self.size_ref is not None and self.size is None:
            self.size = self.size_ref

    @property
    def type(self) -> str:
        """Alias for field_type for backward compatibility."""
        return self.field_type

    @type.setter
    def type(self, value: str) -> None:
        """Set field_type via type property."""
        self.field_type = value


@dataclass
class ProtocolDefinition:
    """Complete protocol definition.

    : Complete protocol specification.

    Attributes:
        name: Protocol name
        version: Protocol version
        description: Protocol description
        settings: Global settings (endianness, etc.)
        framing: Framing/sync configuration
        fields: List of field definitions
        computed_fields: Computed/derived fields
        decoding: Decoding settings
        encoding: Encoding settings
        endian: Default endianness for all fields
    """

    name: str
    description: str = ""
    version: str = "1.0"
    endian: Literal["big", "little"] = "big"
    fields: list[FieldDefinition] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    framing: dict[str, Any] = field(default_factory=dict)
    computed_fields: list[dict[str, Any]] = field(default_factory=list)
    decoding: dict[str, Any] = field(default_factory=dict)
    encoding: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProtocolDefinition:
        """Load protocol definition from YAML file.

        : YAML parsing.

        Args:
            path: Path to YAML file

        Returns:
            ProtocolDefinition instance
        """
        with open(path) as f:
            config = yaml.safe_load(f)

        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ProtocolDefinition:
        """Create from dictionary.

        : Configuration parsing.

        Args:
            config: Configuration dictionary

        Returns:
            ProtocolDefinition instance
        """
        # Parse field definitions
        field_defs = []
        default_endian = config.get("endian", "big")
        for field_dict in config.get("fields", []):
            field_def = cls._parse_field_definition(field_dict, default_endian)
            field_defs.append(field_def)

        return cls(
            name=config.get("name", "unknown"),
            version=config.get("version", "1.0"),
            description=config.get("description", ""),
            endian=default_endian,
            settings=config.get("settings", {}),
            framing=config.get("framing", {}),
            fields=field_defs,
            computed_fields=config.get("computed_fields", []),
            decoding=config.get("decoding", {}),
            encoding=config.get("encoding", {}),
        )

    @classmethod
    def _parse_field_definition(
        cls, field_dict: dict[str, Any], default_endian: str
    ) -> FieldDefinition:
        """Parse a single field definition from dictionary.

        Args:
            field_dict: Field configuration dictionary
            default_endian: Default endianness

        Returns:
            FieldDefinition instance
        """
        # Support both 'type' and 'field_type' attribute names
        field_type = field_dict.get("type") or field_dict.get("field_type", "uint8")

        # Parse nested fields for struct type
        nested_fields: list[FieldDefinition] | None = None
        if field_dict.get("fields"):
            nested_fields = [
                cls._parse_field_definition(f, default_endian) for f in field_dict["fields"]
            ]

        return FieldDefinition(
            name=field_dict["name"],
            field_type=field_type,
            size=field_dict.get("size"),
            offset=field_dict.get("offset"),
            endian=field_dict.get("endian", default_endian),
            condition=field_dict.get("condition"),
            enum=field_dict.get("enum"),
            validation=field_dict.get("validation"),
            default=field_dict.get("default"),
            description=field_dict.get("description", ""),
            value=field_dict.get("value"),
            size_ref=field_dict.get("size_ref"),
            element=field_dict.get("element"),
            count_field=field_dict.get("count_field"),
            count=field_dict.get("count"),
            fields=nested_fields,
        )


@dataclass
class DecodedMessage:
    """A decoded protocol message.

    : Decoded message representation.

    This class behaves like a dictionary for field access, supporting
    operations like `"field_name" in message` and `message["field_name"]`.

    Attributes:
        fields: Dictionary of field name -> value
        raw_data: Original binary data
        size: Message size in bytes
        valid: Whether message passed validation
        errors: List of validation errors
    """

    fields: dict[str, Any]
    raw_data: bytes
    size: int
    valid: bool
    errors: list[str]

    def __contains__(self, key: str) -> bool:
        """Check if field exists in message."""
        return key in self.fields

    def __getitem__(self, key: str) -> Any:
        """Get field value by name."""
        return self.fields[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over field names."""
        return iter(self.fields)

    def keys(self) -> Any:
        """Return field names."""
        return self.fields.keys()

    def values(self) -> Any:
        """Return field values."""
        return self.fields.values()

    def items(self) -> Any:
        """Return field name-value pairs."""
        return self.fields.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get field value with default."""
        return self.fields.get(key, default)


class _SafeConditionEvaluator(ast.NodeVisitor):
    """Safe evaluator for protocol field conditions.

    Only allows:
    - Comparisons: ==, !=, <, <=, >, >=
    - Logical operations: and, or, not
    - Constants: numbers, strings, booleans
    - Variable names from context

    Security:
        Uses AST parsing to safely evaluate conditions without eval().
    """

    def __init__(self, context: dict[str, Any]):
        """Initialize with field context.

        Args:
            context: Dictionary of field names to values
        """
        self.context = context
        self.compare_ops = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
        }

    def eval(self, expression: str) -> bool:
        """Evaluate condition expression.

        Args:
            expression: Condition string

        Returns:
            Boolean result
        """
        try:
            tree = ast.parse(expression, mode="eval")
            result = self.visit(tree.body)
            return bool(result)
        except Exception:
            # If evaluation fails, condition is false
            return False

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Visit comparison operation."""
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if type(op) not in self.compare_ops:
                return False
            right = self.visit(comparator)
            if not self.compare_ops[type(op)](left, right):
                return False
            left = right
        return True

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        """Visit boolean operation (and, or)."""
        if isinstance(node.op, ast.And):
            return all(self.visit(value) for value in node.values)
        elif isinstance(node.op, ast.Or):
            return any(self.visit(value) for value in node.values)
        return False

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Visit unary operation (not)."""
        if isinstance(node.op, ast.Not):
            return not self.visit(node.operand)
        return False

    def visit_Name(self, node: ast.Name) -> Any:
        """Visit variable name."""
        return self.context.get(node.id)

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Visit constant value.

        In Python 3.8+, ast.Constant replaces ast.Num, ast.Str, and ast.NameConstant.
        Since this project requires Python 3.12+, we only need visit_Constant.
        """
        return node.value

    def generic_visit(self, node: ast.AST) -> Any:
        """Disallow other node types."""
        return False


class ProtocolDecoder:
    """Decode binary data using protocol definition.

    : Protocol decoder with full field type support.
    """

    def __init__(self, definition: ProtocolDefinition):
        """Initialize decoder with protocol definition.

        Args:
            definition: Protocol definition
        """
        self.definition = definition
        self._endian_map: dict[str, str] = {"big": ">", "little": "<"}

    @classmethod
    def load(cls, path: str | Path) -> ProtocolDecoder:
        """Load decoder from YAML protocol definition.

        : Load decoder from file.

        Args:
            path: Path to YAML file

        Returns:
            ProtocolDecoder instance
        """
        definition = ProtocolDefinition.from_yaml(path)
        return cls(definition)

    def decode(self, data: bytes, offset: int = 0) -> DecodedMessage:
        """Decode single message from binary data.

        : Complete decoding with validation.

        Args:
            data: Binary data
            offset: Starting offset in data

        Returns:
            DecodedMessage instance
        """
        fields: dict[str, Any] = {}
        errors: list[str] = []
        current_offset = offset
        valid = True

        # Check minimum length
        if len(data) - offset < 1:
            return DecodedMessage(
                fields={}, raw_data=data[offset:], size=0, valid=False, errors=["Insufficient data"]
            )

        # Decode each field
        for field_def in self.definition.fields:
            # Check condition
            if field_def.condition:
                if not self._evaluate_condition(field_def.condition, fields):
                    continue  # Skip this field

            try:
                value, bytes_consumed = self._decode_field(data[current_offset:], field_def, fields)

                # Validate
                if field_def.validation:
                    validation_error = self._validate_field(value, field_def.validation)
                    if validation_error:
                        errors.append(f"{field_def.name}: {validation_error}")
                        valid = False

                # Store value
                fields[field_def.name] = value
                current_offset += bytes_consumed

            except Exception as e:
                errors.append(f"{field_def.name}: {e!s}")
                valid = False
                break

        total_size = current_offset - offset

        return DecodedMessage(
            fields=fields,
            raw_data=data[offset:current_offset],
            size=total_size,
            valid=valid and len(errors) == 0,
            errors=errors,
        )

    def decode_stream(self, data: bytes) -> list[DecodedMessage]:
        """Decode multiple messages from data stream.

        : Stream decoding with sync detection.

        Args:
            data: Binary data stream

        Returns:
            List of DecodedMessage instances
        """
        messages = []
        offset = 0

        # Check for sync pattern
        sync_pattern = self.definition.framing.get("sync_pattern")

        while offset < len(data):
            # Find sync if configured
            if sync_pattern:
                sync_offset = self.find_sync(data, offset)
                if sync_offset is None:
                    break  # No more sync patterns
                offset = sync_offset

            # Decode message
            msg = self.decode(data, offset)

            if msg.valid:
                messages.append(msg)
                offset += msg.size
            else:
                # Try to recover by finding next sync
                if sync_pattern:
                    offset += 1
                else:
                    break  # Can't recover without sync

        return messages

    def find_sync(self, data: bytes, start: int = 0) -> int | None:
        """Find sync pattern in data.

        : Sync pattern detection.

        Args:
            data: Binary data
            start: Starting offset

        Returns:
            Offset of sync pattern or None
        """
        sync_pattern = self.definition.framing.get("sync_pattern")
        if not sync_pattern:
            return start  # No sync pattern, start from beginning

        # Convert sync pattern (hex string or bytes)
        if isinstance(sync_pattern, str):
            if sync_pattern.startswith("0x"):
                # Hex string like "0xAA55"
                sync_bytes = bytes.fromhex(sync_pattern[2:])
            else:
                sync_bytes = sync_pattern.encode()
        else:
            sync_bytes = bytes(sync_pattern)

        # Search for pattern
        idx = data.find(sync_bytes, start)
        if idx == -1:
            return None
        return idx

    def _decode_bytes_field(
        self, data: bytes, field: FieldDefinition, context: dict[str, Any]
    ) -> tuple[bytes, int]:
        """Decode bytes field.

        Args:
            data: Binary data
            field: Field definition
            context: Previously decoded fields

        Returns:
            Tuple of (bytes value, bytes consumed)
        """
        size = self._resolve_size(field.size, context, data)
        if size > len(data):
            size = len(data)  # Use remaining data
        return bytes(data[:size]), size

    def _decode_string_field(
        self, data: bytes, field: FieldDefinition, context: dict[str, Any]
    ) -> tuple[str, int]:
        """Decode string field.

        Args:
            data: Binary data
            field: Field definition
            context: Previously decoded fields

        Returns:
            Tuple of (string value, bytes consumed)
        """
        size = self._resolve_size(field.size, context, data)
        if size > len(data):
            size = len(data)  # Use remaining data
        string_bytes = data[:size]

        # Try to decode as UTF-8, fall back to latin-1
        try:
            value = string_bytes.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            value = string_bytes.decode("latin-1").rstrip("\x00")

        return value, size

    def _decode_bitfield_field(
        self, data: bytes, field: FieldDefinition, endian: str
    ) -> tuple[int, int]:
        """Decode bitfield field.

        Args:
            data: Binary data
            field: Field definition
            endian: Endianness marker

        Returns:
            Tuple of (bitfield value, bytes consumed)

        Raises:
            ValueError: If bitfield size is unsupported
        """
        field_size = field.size if isinstance(field.size, int) else 1

        if field_size == 1:
            bitfield_value = int(data[0])
        elif field_size == 2:
            bitfield_value = struct.unpack(f"{endian}H", data[:2])[0]
        elif field_size == 4:
            bitfield_value = struct.unpack(f"{endian}I", data[:4])[0]
        else:
            raise ValueError(f"Unsupported bitfield size: {field_size}")

        return bitfield_value, field_size

    def _decode_field(
        self, data: bytes, field: FieldDefinition, context: dict[str, Any]
    ) -> tuple[Any, int]:
        """Decode single field.

        : Field decoding for all types.

        Args:
            data: Binary data
            field: Field definition
            context: Previously decoded fields

        Returns:
            Tuple of (value, bytes_consumed)

        Raises:
            ValueError: If bitfield size is unsupported or field type is unknown
        """
        endian = self._endian_map.get(field.endian, ">")
        field_type = field.field_type

        # Integer types
        if field_type in ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"]:
            return self._decode_integer(data, field_type, endian)

        # Float types
        if field_type in ["float32", "float64"]:
            return self._decode_float(data, field_type, endian)

        # Bytes
        if field_type == "bytes":
            return self._decode_bytes_field(data, field, context)

        # String
        if field_type == "string":
            return self._decode_string_field(data, field, context)

        # Bitfield
        if field_type == "bitfield":
            return self._decode_bitfield_field(data, field, endian)

        # Array
        if field_type == "array":
            return self._decode_array(data, field, context)

        # Struct (nested)
        if field_type == "struct":
            return self._decode_struct(data, field, context)

        raise ValueError(f"Unknown field type: {field_type}")

    def _decode_array(
        self, data: bytes, field: FieldDefinition, context: dict[str, Any]
    ) -> tuple[list[Any], int]:
        """Decode array field.

        : Array field decoding.

        Args:
            data: Binary data
            field: Field definition with element spec
            context: Previously decoded fields

        Returns:
            Tuple of (list of values, bytes_consumed)

        Raises:
            ValueError: If array field is missing element definition
        """
        elements = []
        total_consumed = 0

        # Determine element count
        count = None
        if field.count is not None:
            count = field.count
        elif field.count_field is not None and field.count_field in context:
            count = int(context[field.count_field])

        # Get element definition
        element_def = field.element
        if element_def is None:
            raise ValueError(f"Array field '{field.name}' missing element definition")

        element_type = element_def.get("type", "uint8")
        element_endian = element_def.get("endian", field.endian)

        # If no count, try to decode until data exhausted
        idx = 0
        while len(data) - total_consumed > 0:
            if count is not None and idx >= count:
                break

            # Create a temporary field definition for the element
            if element_type == "struct":
                # Nested struct in array
                nested_fields = element_def.get("fields", [])
                parsed_fields = [
                    ProtocolDefinition._parse_field_definition(f, element_endian)
                    for f in nested_fields
                ]
                elem_field = FieldDefinition(
                    name=f"{field.name}[{idx}]",
                    field_type="struct",
                    endian=element_endian,
                    fields=parsed_fields,
                )
                value, consumed = self._decode_struct(data[total_consumed:], elem_field, context)
            else:
                # Simple element type
                elem_field = FieldDefinition(
                    name=f"{field.name}[{idx}]",
                    field_type=element_type,
                    endian=element_endian,
                    size=element_def.get("size"),
                )
                value, consumed = self._decode_field(data[total_consumed:], elem_field, context)

            if consumed == 0:
                break  # Prevent infinite loop

            elements.append(value)
            total_consumed += consumed
            idx += 1

        return elements, total_consumed

    def _decode_struct(
        self, data: bytes, field: FieldDefinition, context: dict[str, Any]
    ) -> tuple[dict[str, Any], int]:
        """Decode struct field.

        : Nested struct field decoding.

        Args:
            data: Binary data
            field: Field definition with nested fields
            context: Previously decoded fields

        Returns:
            Tuple of (dict of field values, bytes_consumed)

        Raises:
            ValueError: If struct field is missing fields definition
        """
        struct_fields: dict[str, Any] = {}
        total_consumed = 0

        # Get nested field definitions
        nested_fields = field.fields
        if nested_fields is None:
            raise ValueError(f"Struct field '{field.name}' missing fields definition")

        # Decode each nested field
        for nested_field in nested_fields:
            if len(data) - total_consumed < 1:
                break  # Not enough data

            # Check condition if present
            if nested_field.condition:
                # Use combined context (parent context + struct fields decoded so far)
                combined_context = {**context, **struct_fields}
                if not self._evaluate_condition(nested_field.condition, combined_context):
                    continue

            value, consumed = self._decode_field(
                data[total_consumed:], nested_field, {**context, **struct_fields}
            )
            struct_fields[nested_field.name] = value
            total_consumed += consumed

        return struct_fields, total_consumed

    def _decode_integer(self, data: bytes, type_name: str, endian: str) -> tuple[int, int]:
        """Decode integer field.

        Args:
            data: Binary data
            type_name: Type name (uint8, int16, etc.)
            endian: Endian marker

        Returns:
            Tuple of (value, bytes_consumed)

        Raises:
            ValueError: If insufficient data for the integer type
        """
        format_map = {
            "uint8": ("B", 1),
            "int8": ("b", 1),
            "uint16": ("H", 2),
            "int16": ("h", 2),
            "uint32": ("I", 4),
            "int32": ("i", 4),
            "uint64": ("Q", 8),
            "int64": ("q", 8),
        }

        fmt_char, size = format_map[type_name]

        if len(data) < size:
            raise ValueError(f"Insufficient data for {type_name} (need {size}, have {len(data)})")

        # uint8/int8 don't use endianness
        if size == 1:
            value = struct.unpack(fmt_char, data[:size])[0]
        else:
            value = struct.unpack(f"{endian}{fmt_char}", data[:size])[0]

        return value, size

    def _decode_float(self, data: bytes, type_name: str, endian: str) -> tuple[float, int]:
        """Decode float field.

        Args:
            data: Binary data
            type_name: Type name (float32 or float64)
            endian: Endian marker

        Returns:
            Tuple of (value, bytes_consumed)

        Raises:
            ValueError: If insufficient data for the float type
        """
        if type_name == "float32":
            size = 4
            fmt = f"{endian}f"
        else:  # float64
            size = 8
            fmt = f"{endian}d"

        if len(data) < size:
            raise ValueError(f"Insufficient data for {type_name} (need {size}, have {len(data)})")

        value = struct.unpack(fmt, data[:size])[0]
        return value, size

    def _resolve_size(
        self, size_spec: int | str | None, context: dict[str, Any], data: bytes
    ) -> int:
        """Resolve field size (literal or reference).

        Args:
            size_spec: Size specification (int, field name, or 'remaining')
            context: Decoded fields
            data: Current data buffer (for 'remaining' size)

        Returns:
            Resolved size

        Raises:
            ValueError: If size field not found in context or size specification is invalid
        """
        if size_spec is None:
            # No size specified, return 0 (caller should handle)
            return 0
        elif isinstance(size_spec, int):
            return size_spec
        elif isinstance(size_spec, str):
            # Special case: 'remaining' means use all remaining data
            if size_spec == "remaining":
                return len(data)
            # Reference to another field
            if size_spec in context:
                return int(context[size_spec])
            else:
                raise ValueError(f"Size field '{size_spec}' not found in context")
        else:
            raise ValueError(f"Invalid size specification: {size_spec}")

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """Evaluate field condition against decoded context.

        : Conditional field evaluation.

        Args:
            condition: Condition expression (e.g., "msg_type == 0x02")
            context: Decoded fields

        Returns:
            True if condition is satisfied

        Security:
            Uses AST-based safe evaluation. Only comparisons and logical
            operations are permitted.
        """
        evaluator = _SafeConditionEvaluator(context)
        return evaluator.eval(condition)

    def _validate_field(self, value: Any, validation: dict[str, Any]) -> str | None:
        """Validate field value.

        Args:
            value: Field value
            validation: Validation rules

        Returns:
            Error message or None if valid
        """
        # Min/max validation
        if "min" in validation:
            if value < validation["min"]:
                return f"Value {value} below minimum {validation['min']}"

        if "max" in validation:
            if value > validation["max"]:
                return f"Value {value} above maximum {validation['max']}"

        # Value validation
        if "value" in validation:
            if value != validation["value"]:
                return f"Expected {validation['value']}, got {value}"

        return None


class ProtocolEncoder:
    """Encode data using protocol definition.

    : Protocol encoder.
    """

    def __init__(self, definition: ProtocolDefinition):
        """Initialize encoder.

        Args:
            definition: Protocol definition
        """
        self.definition = definition
        self._endian_map: dict[str, str] = {"big": ">", "little": "<"}

    def encode(self, fields: dict[str, Any]) -> bytes:
        """Encode field values to binary message.

        : Message encoding.

        Args:
            fields: Dictionary of field name -> value

        Returns:
            Encoded binary message

        Raises:
            ValueError: If required field is missing
        """
        result = bytearray()

        for field_def in self.definition.fields:
            # Check condition
            if field_def.condition:
                # Use safe evaluator instead of eval()
                evaluator = _SafeConditionEvaluator(fields)
                if not evaluator.eval(field_def.condition):
                    continue

            # Get value
            if field_def.name in fields:
                value = fields[field_def.name]
            elif field_def.default is not None:
                value = field_def.default
            else:
                raise ValueError(f"Missing required field: {field_def.name}")

            # Encode field
            encoded = self._encode_field(value, field_def)
            result.extend(encoded)

        return bytes(result)

    def _encode_field(self, value: Any, field: FieldDefinition) -> bytes:
        """Encode single field value.

        Args:
            value: Field value.
            field: Field definition.

        Returns:
            Encoded bytes.

        Raises:
            ValueError: If bytes value is invalid or field type is unknown for encoding.

        Example:
            >>> encoder._encode_field(42, FieldDefinition("counter", "uint16", "big"))
            b'\\x00*'
        """
        field_type = field.field_type

        # Dispatch to type-specific encoders
        if field_type in {"uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64"}:
            return self._encode_integer_field(value, field)
        elif field_type in {"float32", "float64"}:
            return self._encode_float_field(value, field)
        elif field_type == "bytes":
            return self._encode_bytes_field(value)
        elif field_type == "string":
            return self._encode_string_field(value)
        elif field_type == "array":
            return self._encode_array(value, field)
        elif field_type == "struct":
            return self._encode_struct(value, field)
        else:
            raise ValueError(f"Unknown field type for encoding: {field_type}")

    def _encode_integer_field(self, value: Any, field: FieldDefinition) -> bytes:
        """Encode integer field types.

        Args:
            value: Integer value.
            field: Field definition with type and endianness.

        Returns:
            Packed integer bytes.
        """
        endian = self._endian_map.get(field.endian, ">")
        field_type = field.field_type

        # Map field types to struct format characters
        _INT_FORMATS = {
            "uint8": "B",
            "int8": "b",
            "uint16": "H",
            "int16": "h",
            "uint32": "I",
            "int32": "i",
            "uint64": "Q",
            "int64": "q",
        }

        fmt_char = _INT_FORMATS[field_type]
        if fmt_char in {"B", "b"}:
            # uint8/int8 have no endianness
            return struct.pack(fmt_char, int(value))
        else:
            return struct.pack(f"{endian}{fmt_char}", int(value))

    def _encode_float_field(self, value: Any, field: FieldDefinition) -> bytes:
        """Encode floating-point field types.

        Args:
            value: Float value.
            field: Field definition with type and endianness.

        Returns:
            Packed float bytes.
        """
        endian = self._endian_map.get(field.endian, ">")

        if field.field_type == "float32":
            return struct.pack(f"{endian}f", float(value))
        elif field.field_type == "float64":
            return struct.pack(f"{endian}d", float(value))
        else:
            raise ValueError(f"Unknown float type: {field.field_type}")

    def _encode_bytes_field(self, value: Any) -> bytes:
        """Encode bytes field.

        Args:
            value: Bytes, list, or tuple of byte values.

        Returns:
            Byte sequence.

        Raises:
            ValueError: If value cannot be converted to bytes.
        """
        if isinstance(value, bytes):
            return value
        elif isinstance(value, list | tuple):
            return bytes(value)
        else:
            raise ValueError(f"Invalid bytes value: {value}")

    def _encode_string_field(self, value: Any) -> bytes:
        """Encode string field.

        Args:
            value: String or bytes.

        Returns:
            UTF-8 encoded bytes.
        """
        if isinstance(value, str):
            return value.encode("utf-8")
        else:
            return bytes(value)

    def _encode_array(self, value: list[Any], field: FieldDefinition) -> bytes:
        """Encode array field.

        Args:
            value: List of values
            field: Field definition

        Returns:
            Encoded bytes

        Raises:
            ValueError: If array field is missing element definition
        """
        result = bytearray()

        element_def = field.element
        if element_def is None:
            raise ValueError(f"Array field '{field.name}' missing element definition")

        element_type = element_def.get("type", "uint8")
        element_endian = element_def.get("endian", field.endian)

        for i, elem in enumerate(value):
            if element_type == "struct":
                # Nested struct
                nested_fields = element_def.get("fields", [])
                parsed_fields = [
                    ProtocolDefinition._parse_field_definition(f, element_endian)
                    for f in nested_fields
                ]
                elem_field = FieldDefinition(
                    name=f"{field.name}[{i}]",
                    field_type="struct",
                    endian=element_endian,
                    fields=parsed_fields,
                )
                result.extend(self._encode_struct(elem, elem_field))
            else:
                elem_field = FieldDefinition(
                    name=f"{field.name}[{i}]",
                    field_type=element_type,
                    endian=element_endian,
                )
                result.extend(self._encode_field(elem, elem_field))

        return bytes(result)

    def _encode_struct(self, value: dict[str, Any], field: FieldDefinition) -> bytes:
        """Encode struct field.

        Args:
            value: Dictionary of field values
            field: Field definition

        Returns:
            Encoded bytes

        Raises:
            ValueError: If struct field is missing fields definition
        """
        result = bytearray()

        nested_fields = field.fields
        if nested_fields is None:
            raise ValueError(f"Struct field '{field.name}' missing fields definition")

        for nested_field in nested_fields:
            if nested_field.name in value:
                result.extend(self._encode_field(value[nested_field.name], nested_field))
            elif nested_field.default is not None:
                result.extend(self._encode_field(nested_field.default, nested_field))

        return bytes(result)


def load_protocol(path: str | Path) -> ProtocolDefinition:
    """Load protocol definition from YAML.

    : Convenience function for loading protocols.

    Args:
        path: Path to YAML file

    Returns:
        ProtocolDefinition instance
    """
    return ProtocolDefinition.from_yaml(path)


def decode_message(data: bytes, protocol: str | ProtocolDefinition) -> DecodedMessage:
    """Decode message using protocol.

    : Convenience function for decoding.

    Args:
        data: Binary message data
        protocol: Protocol definition (path or instance)

    Returns:
        DecodedMessage instance
    """
    if isinstance(protocol, str):
        protocol_def = ProtocolDefinition.from_yaml(protocol)
    else:
        protocol_def = protocol

    decoder = ProtocolDecoder(protocol_def)
    return decoder.decode(data)
