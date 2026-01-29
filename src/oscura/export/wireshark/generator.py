"""Wireshark Lua dissector generator.

This module generates Wireshark Lua dissectors from Oscura protocol definitions.
The generated dissectors can be loaded into Wireshark for interactive protocol analysis.

Features:
- Generate ProtoField declarations for all field types
- Handle fixed and variable-length fields
- Support TCP/UDP port registration
- Checksum validation support
- Proper error handling for malformed packets

References:
    https://wiki.wireshark.org/lua/dissectors
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from oscura.inference.protocol_dsl import FieldDefinition, ProtocolDefinition

from .type_mapping import get_field_size, get_lua_reader_function, get_protofield_type
from .validator import validate_lua_syntax


class WiresharkDissectorGenerator:
    """Generate Wireshark Lua dissectors from protocol definitions.

    This class converts Oscura protocol definitions into Wireshark Lua
    dissectors that can be loaded into Wireshark for protocol analysis.

    Features:
    - Generate ProtoField declarations
    - Handle fixed and variable-length fields
    - Support nested protocols (dissector stacking)
    - TCP/UDP registration
    - Checksum validation
    - Registration on port/pattern

    Example:
        >>> from oscura.inference.protocol_dsl import ProtocolDefinition
        >>> protocol = ProtocolDefinition(name="simple", description="Simple Protocol")
        >>> generator = WiresharkDissectorGenerator()
        >>> generator.generate(protocol, Path("simple.lua"))

    References:
        https://wiki.wireshark.org/lua/dissectors
        https://wiki.wireshark.org/LuaAPI/Proto
    """

    def __init__(self, validate: bool = True) -> None:
        """Initialize the generator.

        Args:
            validate: Run luac syntax validation if available
        """
        self.validate = validate
        self.env = Environment(
            loader=PackageLoader("oscura.export.wireshark"),
            autoescape=select_autoescape(),
        )

    def generate(self, protocol: ProtocolDefinition, output_path: Path) -> None:
        """Generate Lua dissector file.

        Args:
            protocol: Protocol definition to export
            output_path: Where to write .lua file

        Raises:
            RuntimeError: If Lua syntax validation fails
        """
        lua_code = self.generate_to_string(protocol)

        # Validate if requested
        if self.validate:
            is_valid, error_msg = validate_lua_syntax(lua_code)
            if not is_valid:
                raise RuntimeError(f"Generated Lua code has syntax errors:\n{error_msg}")

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(lua_code)

    def generate_to_string(self, protocol: ProtocolDefinition) -> str:
        """Generate Lua dissector as string.

        Args:
            protocol: Protocol definition to export

        Returns:
            Generated Lua code

        Raises:
            ValueError: If protocol definition is invalid
        """
        # Validate protocol
        if not protocol.name:
            raise ValueError("Protocol name is required")

        # Prepare template context
        context = self._build_template_context(protocol)

        # Render template
        template = self.env.get_template("dissector.lua.j2")
        return str(template.render(**context))

    def _build_template_context(self, protocol: ProtocolDefinition) -> dict[str, Any]:
        """Build context dictionary for template rendering.

        Args:
            protocol: Protocol definition

        Returns:
            Template context dictionary
        """
        # Generate proto variable name (lowercase, no spaces)
        proto_var = protocol.name.lower().replace(" ", "_").replace("-", "_") + "_proto"

        # Process fields and calculate offsets
        field_data = []
        field_offsets = {}  # Track offset of each field by name
        current_offset = 0
        min_length = 0

        for field_def in protocol.fields:
            # Store offset for this field
            field_offsets[field_def.name] = current_offset

            field_info = self._process_field(field_def, protocol, field_offsets)
            field_data.append(field_info)

            # Update offset for next field (if fixed size)
            if field_info["size"] is not None and not field_info["is_variable_length"]:
                current_offset += field_info["size"]

            # Calculate minimum length (sum of fixed-size fields)
            if not field_info["is_variable_length"]:
                min_length += field_info["size"]

        # Extract transport settings
        transport = None
        port = None

        if "transport" in protocol.settings:
            transport = protocol.settings["transport"]
        if "port" in protocol.settings:
            port = protocol.settings["port"]

        # Check framing for transport/port
        if transport is None and "transport" in protocol.framing:
            transport = protocol.framing["transport"]
        if port is None and "port" in protocol.framing:
            port = protocol.framing["port"]

        context = {
            "protocol": protocol,
            "proto_var": proto_var,
            "fields": field_data,
            "min_length": max(min_length, 1),  # At least 1 byte
            "timestamp": datetime.now(UTC).isoformat(),
            "transport": transport,
            "port": port,
            "pattern": protocol.framing.get("sync_pattern"),
        }

        return context

    def _process_field(
        self,
        field_def: FieldDefinition,
        protocol: ProtocolDefinition,
        field_offsets: dict[str, int],
    ) -> dict[str, Any]:
        """Process a field definition for template rendering.

        Args:
            field_def: Field definition
            protocol: Parent protocol definition
            field_offsets: Dictionary mapping field names to byte offsets

        Returns:
            Dictionary with field information for template
        """
        # Get ProtoField type and display base
        protofield_type, display_base = get_protofield_type(field_def.field_type)

        # Get field size
        size = get_field_size(field_def.field_type)
        is_variable = size is None

        # Handle explicit size specification
        if field_def.size is not None:
            if isinstance(field_def.size, int):
                size = field_def.size
                is_variable = False
            elif isinstance(field_def.size, str):
                # Size is a reference to another field
                is_variable = True

        # Generate display name
        display_name = field_def.description or field_def.name.replace("_", " ").title()

        # Get Lua reader function
        reader_func = get_lua_reader_function(field_def.field_type, field_def.endian)

        field_info = {
            "name": field_def.name,
            "display_name": display_name,
            "protofield_type": protofield_type,
            "display_base": display_base,
            "size": size,
            "is_variable_length": is_variable,
            "condition": field_def.condition,
            "reader_function": reader_func,
            "endian": field_def.endian,
            "value_string": self._generate_value_string(field_def),
        }

        # Handle variable-length fields with size reference
        if is_variable and isinstance(field_def.size, str):
            # Find the size field
            size_field_def = self._find_field(protocol, field_def.size)
            if size_field_def:
                size_field_size = get_field_size(size_field_def.field_type)
                size_reader = get_lua_reader_function(
                    size_field_def.field_type, size_field_def.endian
                )

                # Get the offset of the size field
                size_field_offset = field_offsets.get(field_def.size, 0)

                field_info.update(
                    {
                        "size_field": field_def.size,
                        "size_offset": size_field_offset,
                        "size_field_size": size_field_size,
                        "size_reader": size_reader,
                    }
                )

        return field_info

    def _find_field(self, protocol: ProtocolDefinition, field_name: str) -> FieldDefinition | None:
        """Find a field definition by name.

        Args:
            protocol: Protocol definition
            field_name: Field name to find

        Returns:
            Field definition or None if not found
        """
        for field_def in protocol.fields:
            if field_def.name == field_name:
                return field_def
        return None

    def _generate_value_string(self, field_def: FieldDefinition) -> str | None:
        """Generate Wireshark value_string table for enum fields.

        Args:
            field_def: Field definition

        Returns:
            Lua table definition or None if field has no enum
        """
        if not field_def.enum:
            return None

        # Generate Lua table for enum
        entries = []
        for key, value in field_def.enum.items():
            if isinstance(key, int):
                entries.append(f'[{key}] = "{value}"')
            else:
                # String key - try to get value
                entries.append(f'["{key}"] = "{value}"')

        return "{" + ", ".join(entries) + "}"

    def _calculate_min_length(self, protocol: ProtocolDefinition) -> int:
        """Calculate the minimum packet length for the protocol.

        Args:
            protocol: Protocol definition

        Returns:
            Minimum length in bytes
        """
        min_length = 0

        for field_def in protocol.fields:
            # Skip conditional fields (they might not be present)
            if field_def.condition:
                continue

            # Get field size
            size = get_field_size(field_def.field_type)

            # Handle explicit size
            if field_def.size is not None and isinstance(field_def.size, int):
                size = field_def.size

            # Add to minimum length if fixed size
            if size is not None:
                min_length += size

        return max(min_length, 1)  # At least 1 byte
