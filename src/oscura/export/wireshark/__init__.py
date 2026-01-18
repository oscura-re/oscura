"""Wireshark dissector export module.

This module provides functionality to export Oscura protocol definitions
as Wireshark Lua dissectors for integration with Wireshark's protocol
analysis tools.

Example:
    >>> from oscura.inference.protocol_dsl import ProtocolDefinition
    >>> from oscura.export.wireshark import WiresharkDissectorGenerator
    >>> protocol = ProtocolDefinition(name="myproto", description="My Protocol")
    >>> generator = WiresharkDissectorGenerator()
    >>> generator.generate(protocol, Path("myproto.lua"))

Installation:
    Copy the generated .lua file to your Wireshark plugins directory:
    - Linux: ~/.local/lib/wireshark/plugins/
    - macOS: ~/.config/wireshark/plugins/
    - Windows: %APPDATA%\\Wireshark\\plugins\\
"""

from .generator import WiresharkDissectorGenerator
from .lua_builder import LuaCodeBuilder
from .type_mapping import (
    BASE_MAPPING,
    DEFAULT_BASE,
    FIELD_TYPE_MAPPING,
    get_field_size,
    get_lua_reader_function,
    get_protofield_type,
    is_variable_length,
)
from .validator import check_luac_available, validate_lua_file, validate_lua_syntax

__all__ = [
    "BASE_MAPPING",
    "DEFAULT_BASE",
    "FIELD_TYPE_MAPPING",
    "LuaCodeBuilder",
    "WiresharkDissectorGenerator",
    "check_luac_available",
    "get_field_size",
    "get_lua_reader_function",
    "get_protofield_type",
    "is_variable_length",
    "validate_lua_file",
    "validate_lua_syntax",
]
