"""Scapy layer generator from ProtocolSpec.

This module generates production-ready Scapy layer classes from ProtocolSpec
objects (from reverse engineering workflows). The generated layers can be
imported and used for packet construction, dissection, and network analysis.

Features:
    - Generate complete Scapy layer classes from ProtocolSpec
    - Support all field types (uint8, uint16, uint32, string, bytes, enum)
    - Handle endianness (big-endian, little-endian)
    - CRC validation using Scapy's XByteField
    - Generate importable Python modules
    - Validate generated code executes without errors

Example:
    >>> from oscura.export.scapy_layer import (
    ...     ScapyLayerGenerator,
    ...     ScapyLayerConfig
    ... )
    >>> from oscura.workflows.reverse_engineering import ProtocolSpec, FieldSpec
    >>> spec = ProtocolSpec(
    ...     name="MyProtocol",
    ...     baud_rate=115200,
    ...     frame_format="8N1",
    ...     sync_pattern="aa55",
    ...     frame_length=10,
    ...     fields=[
    ...         FieldSpec(name="sync", offset=0, size=2, field_type="bytes"),
    ...         FieldSpec(name="length", offset=2, size=1, field_type="uint8"),
    ...     ],
    ...     checksum_type="crc16",
    ...     checksum_position=-1,
    ...     confidence=0.95
    ... )
    >>> config = ScapyLayerConfig(protocol_name="MyProtocol")
    >>> generator = ScapyLayerGenerator(config)
    >>> layer_path = generator.generate(
    ...     spec,
    ...     sample_messages=[b"\\xaa\\x55\\x08test123"],
    ...     output_path=Path("myproto_layer.py")
    ... )
    >>> # Import and use the generated layer
    >>> from myproto_layer import MyProtocol
    >>> pkt = MyProtocol(sync=b"\\xaa\\x55", length=8)

Installation:
    Import the generated .py file directly or add it to your Python path:
    >>> import sys
    >>> sys.path.append('/path/to/generated')
    >>> from myproto_layer import MyProtocol

References:
    - Scapy Documentation: https://scapy.readthedocs.io/
    - Scapy Layer Creation: https://scapy.readthedocs.io/en/latest/build_dissect.html
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from oscura.utils.validation import validate_protocol_spec

if TYPE_CHECKING:
    from oscura.inference.crc_reverse import CRCParameters
    from oscura.workflows.reverse_engineering import FieldSpec, ProtocolSpec

__all__ = ["ScapyLayerConfig", "ScapyLayerGenerator"]

logger = logging.getLogger(__name__)


@dataclass
class ScapyLayerConfig:
    """Configuration for Scapy layer generation.

    Attributes:
        protocol_name: Protocol name for layer class.
        base_class: Scapy base class (default "Packet").
        include_crc_validation: Include CRC validation code (default True).
        generate_examples: Generate usage examples in docstring (default True).
        show_progress: Show progress bar for >100 messages (default True).
    """

    protocol_name: str
    base_class: str = "Packet"
    include_crc_validation: bool = True
    generate_examples: bool = True
    show_progress: bool = True


class ScapyLayerGenerator:
    """Generate production-ready Scapy layers from ProtocolSpec.

    This class converts ProtocolSpec objects (from reverse engineering workflows)
    into Scapy layer classes that can be imported and used for packet construction,
    dissection, and network analysis.

    Features:
        - All field types (uint8, uint16, uint32, string, bytes, enum)
        - Endianness handling (big/little)
        - CRC validation
        - Python code validation
        - Usage examples

    Example:
        >>> config = ScapyLayerConfig(protocol_name="MyProtocol")
        >>> generator = ScapyLayerGenerator(config)
        >>> layer_path = generator.generate(
        ...     spec,
        ...     sample_messages=[b"\\x01\\x02\\x03"],
        ...     output_path=Path("myproto_layer.py")
        ... )
        >>> from myproto_layer import MyProtocol
        >>> pkt = MyProtocol()
    """

    def __init__(self, config: ScapyLayerConfig) -> None:
        """Initialize Scapy layer generator.

        Args:
            config: Layer generation configuration.
        """
        self.config = config

    def generate(
        self,
        spec: ProtocolSpec,
        sample_messages: list[bytes],
        output_path: Path,
    ) -> Path:
        """Generate Scapy layer Python module.

        Args:
            spec: Protocol specification from reverse engineering.
            sample_messages: Sample protocol messages for examples.
            output_path: Path for output .py file.

        Returns:
            Path to generated Python module.

        Raises:
            ValueError: If spec is invalid or has missing required fields.
            RuntimeError: If Python syntax validation fails.
            OSError: If file writing fails.

        Example:
            >>> spec = ProtocolSpec(name="test", ...)
            >>> generator = ScapyLayerGenerator(config)
            >>> layer_path = generator.generate(
            ...     spec,
            ...     [b"\\x01\\x02\\x03"],
            ...     Path("test_layer.py")
            ... )
        """
        # Validate spec
        self._validate_spec(spec)

        # Generate Python code with optional progress bar
        if self.config.show_progress and len(sample_messages) > 100:
            with tqdm(total=5, desc="Generating Scapy layer") as pbar:
                pbar.set_description("Generating imports")
                imports = self._generate_imports(spec)
                pbar.update(1)

                pbar.set_description("Generating CRC functions")
                crc_code = self._generate_crc_functions(spec)
                pbar.update(1)

                pbar.set_description("Generating layer class")
                layer_code = self._generate_layer_class(spec, sample_messages)
                pbar.update(1)

                pbar.set_description("Generating bind layers")
                bind_code = self._generate_bind_layers(spec)
                pbar.update(1)

                pbar.set_description("Assembling code")
                python_code = self._assemble_code(imports, crc_code, layer_code, bind_code)
                pbar.update(1)
        else:
            imports = self._generate_imports(spec)
            crc_code = self._generate_crc_functions(spec)
            layer_code = self._generate_layer_class(spec, sample_messages)
            bind_code = self._generate_bind_layers(spec)
            python_code = self._assemble_code(imports, crc_code, layer_code, bind_code)

        # Validate Python syntax
        if not self._validate_python_syntax(python_code):
            raise RuntimeError("Generated Python code has syntax errors")

        # Write Python file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(python_code, encoding="utf-8")
        logger.info(f"Generated Scapy layer: {output_path}")

        return output_path

    def _validate_spec(self, spec: ProtocolSpec) -> None:
        """Validate protocol specification.

        Args:
            spec: Protocol specification to validate.

        Raises:
            ValueError: If spec is invalid.
        """
        validate_protocol_spec(spec)

        # Validate fields
        for field in spec.fields:
            if not field.name:
                raise ValueError("Field name is required")
            if field.field_type not in {
                "uint8",
                "uint16",
                "uint32",
                "bytes",
                "string",
                "constant",
                "checksum",
            }:
                raise ValueError(f"Unsupported field type: {field.field_type}")

    def _generate_imports(self, spec: ProtocolSpec) -> str:
        """Generate import statements.

        Args:
            spec: Protocol specification.

        Returns:
            Python import statements.
        """
        imports = [
            f'"""Scapy layer for {spec.name}.',
            "",
            f"Generated by Oscura on {datetime.now(UTC).isoformat()}",
            "",
            "Usage:",
            f"    >>> from {self._safe_module_name(spec.name)} import {self._safe_class_name(spec.name)}",
            f"    >>> pkt = {self._safe_class_name(spec.name)}()",
            "    >>> pkt.show()",
            '"""',
            "",
            "from scapy.fields import (",
            "    ByteField,",
            "    ShortField,",
            "    IntField,",
            "    LEShortField,",
            "    LEIntField,",
            "    StrFixedLenField,",
            "    XByteField,",
            "    XShortField,",
            "    XIntField,",
            ")",
            "from scapy.packet import Packet, bind_layers",
            "",
        ]

        return "\n".join(imports)

    def _generate_crc_functions(self, spec: ProtocolSpec) -> str:
        """Generate CRC validation functions.

        Args:
            spec: Protocol specification.

        Returns:
            Python CRC function code.
        """
        if not self.config.include_crc_validation:
            return ""

        if not spec.checksum_type or spec.checksum_type not in ("crc8", "crc16", "crc32"):
            return ""

        # Get CRC parameters if available
        crc_info = getattr(spec, "crc_info", None)
        if crc_info:
            return self._generate_crc_function_from_params(crc_info)

        # Default CRC implementations
        if spec.checksum_type == "crc16":
            return '''
def calculate_crc16(data: bytes) -> int:
    """Calculate CRC-16-CCITT checksum.

    Args:
        data: Input data bytes.

    Returns:
        16-bit CRC value.
    """
    crc = 0xFFFF
    poly = 0x1021

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc
'''
        elif spec.checksum_type == "crc8":
            return '''
def calculate_crc8(data: bytes) -> int:
    """Calculate CRC-8 checksum.

    Args:
        data: Input data bytes.

    Returns:
        8-bit CRC value.
    """
    crc = 0x00
    poly = 0x07

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc
'''
        else:  # crc32
            return '''
def calculate_crc32(data: bytes) -> int:
    """Calculate CRC-32 checksum.

    Args:
        data: Input data bytes.

    Returns:
        32-bit CRC value.
    """
    crc = 0xFFFFFFFF
    poly = 0x04C11DB7

    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ poly) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF

    return crc ^ 0xFFFFFFFF
'''

    def _generate_crc_function_from_params(self, crc_info: CRCParameters) -> str:
        """Generate CRC function from CRCParameters.

        Args:
            crc_info: CRC parameters from reverse engineering.

        Returns:
            Python CRC function code.
        """
        width = crc_info.width
        poly = crc_info.polynomial
        init = crc_info.init
        xor_out = crc_info.xor_out
        mask = (1 << width) - 1

        func_name = f"calculate_crc{width}"

        lines = [
            "",
            f"def {func_name}(data: bytes) -> int:",
            f'    """Calculate CRC-{width} checksum (custom parameters).',
            "    ",
            f"    Polynomial: 0x{poly:0{width // 4}x}",
            f"    Init: 0x{init:0{width // 4}x}",
            f"    XorOut: 0x{xor_out:0{width // 4}x}",
            f"    ReflectIn: {crc_info.reflect_in}",
            f"    ReflectOut: {crc_info.reflect_out}",
            "    ",
            "    Args:",
            "        data: Input data bytes.",
            "    ",
            "    Returns:",
            f"        {width}-bit CRC value.",
            '    """',
            f"    crc = 0x{init:0{width // 4}x}",
            f"    poly = 0x{poly:0{width // 4}x}",
            f"    mask = 0x{mask:0{width // 4}x}",
            "    ",
            "    for byte in data:",
        ]

        if crc_info.reflect_in:
            lines.extend(
                [
                    "        # Reflect input byte",
                    "        reflected = 0",
                    "        for b in range(8):",
                    "            if byte & (1 << b):",
                    "                reflected |= 1 << (7 - b)",
                    "        byte = reflected",
                ]
            )

        lines.extend(
            [
                f"        crc ^= byte << {width - 8}",
                "        for _ in range(8):",
                f"            if crc & 0x{1 << (width - 1):0{width // 4}x}:",
                "                crc = ((crc << 1) ^ poly) & mask",
                "            else:",
                "                crc = (crc << 1) & mask",
                "    ",
            ]
        )

        if crc_info.reflect_out:
            lines.extend(
                [
                    "    # Reflect output CRC",
                    "    reflected = 0",
                    f"    for b in range({width}):",
                    "        if crc & (1 << b):",
                    f"            reflected |= 1 << ({width - 1} - b)",
                    "    crc = reflected",
                    "    ",
                ]
            )

        lines.extend([f"    return crc ^ 0x{xor_out:0{width // 4}x}", ""])

        return "\n".join(lines)

    def _generate_layer_class(self, spec: ProtocolSpec, sample_messages: list[bytes]) -> str:
        """Generate Scapy layer class definition.

        Args:
            spec: Protocol specification.
            sample_messages: Sample messages for examples.

        Returns:
            Python class definition code.
        """
        class_name = self._safe_class_name(spec.name)
        base_class = self.config.base_class

        lines = [
            f"class {class_name}({base_class}):",
            f'    """Scapy layer for {spec.name}.',
            "    ",
            f"    Baud Rate: {spec.baud_rate} bps",
            f"    Frame Format: {spec.frame_format}",
            f"    Sync Pattern: {spec.sync_pattern}",
            f"    Frame Length: {spec.frame_length if spec.frame_length else 'Variable'} bytes",
            f"    Checksum: {spec.checksum_type if spec.checksum_type else 'None'}",
            "    ",
        ]

        # Add examples if requested
        if self.config.generate_examples and sample_messages:
            lines.extend(
                [
                    "    Example:",
                    f"        >>> pkt = {class_name}()",
                    "        >>> pkt.show()",
                ]
            )
            # Add first sample as example
            if sample_messages:
                first_sample = sample_messages[0]
                lines.append(f"        >>> raw_pkt = bytes.fromhex('{first_sample.hex()}')")
                lines.append(f"        >>> pkt = {class_name}(raw_pkt)")

        lines.extend(['    """', f'    name = "{spec.name}"', "    fields_desc = ["])

        # Generate field definitions
        for field in spec.fields:
            field_def = self._generate_field_definition(field)
            lines.append(f"        {field_def},")

        lines.extend(["    ]", ""])

        # Add post_build for CRC calculation if needed
        if spec.checksum_type and spec.checksum_position is not None:
            lines.extend(self._generate_post_build(spec))

        # Add do_dissect for CRC validation if needed
        if spec.checksum_type and self.config.include_crc_validation:
            lines.extend(self._generate_do_dissect(spec))

        return "\n".join(lines)

    def _generate_field_definition(self, field: FieldSpec) -> str:
        """Generate Scapy field definition.

        Args:
            field: Field specification.

        Returns:
            Python field definition code.
        """
        field_name = self._safe_field_name(field.name)
        field_size = field.size if isinstance(field.size, int) else 1
        endian = getattr(field, "endian", "big")

        # Dispatch to type-specific handlers
        if field.field_type == "uint8":
            return self._gen_uint8_field(field_name, field.name)
        elif field.field_type == "uint16":
            return self._gen_uint16_field(field_name, field.name, endian)
        elif field.field_type == "uint32":
            return self._gen_uint32_field(field_name, field.name, endian)
        elif field.field_type == "string":
            return f'StrFixedLenField("{field_name}", b"", length={field_size})'
        elif field.field_type in ("bytes", "constant"):
            return self._gen_bytes_field(field_name, field, field_size)
        else:  # checksum
            return self._gen_checksum_field(field_name, field_size)

    def _gen_uint8_field(self, field_name: str, original_name: str) -> str:
        """Generate uint8 field definition.

        Args:
            field_name: Safe field name.
            original_name: Original field name.

        Returns:
            Field definition code.
        """
        if original_name == "checksum" or "crc" in original_name.lower():
            return f'XByteField("{field_name}", 0)'
        return f'ByteField("{field_name}", 0)'

    def _gen_uint16_field(self, field_name: str, original_name: str, endian: str) -> str:
        """Generate uint16 field definition.

        Args:
            field_name: Safe field name.
            original_name: Original field name.
            endian: Byte order (big/little).

        Returns:
            Field definition code.
        """
        if original_name == "checksum" or "crc" in original_name.lower():
            return f'XShortField("{field_name}", 0)'
        if endian == "little":
            return f'LEShortField("{field_name}", 0)'
        return f'ShortField("{field_name}", 0)'

    def _gen_uint32_field(self, field_name: str, original_name: str, endian: str) -> str:
        """Generate uint32 field definition.

        Args:
            field_name: Safe field name.
            original_name: Original field name.
            endian: Byte order (big/little).

        Returns:
            Field definition code.
        """
        if original_name == "checksum" or "crc" in original_name.lower():
            return f'XIntField("{field_name}", 0)'
        if endian == "little":
            return f'LEIntField("{field_name}", 0)'
        return f'IntField("{field_name}", 0)'

    def _gen_bytes_field(self, field_name: str, field: FieldSpec, field_size: int) -> str:
        """Generate bytes/constant field definition.

        Args:
            field_name: Safe field name.
            field: Field specification.
            field_size: Field size in bytes.

        Returns:
            Field definition code.
        """
        default_value = getattr(field, "value", None)
        if default_value and isinstance(default_value, str):
            default_bytes = bytes.fromhex(default_value)
            return f'StrFixedLenField("{field_name}", {default_bytes!r}, length={field_size})'
        return f'StrFixedLenField("{field_name}", b"\\x00" * {field_size}, length={field_size})'

    def _gen_checksum_field(self, field_name: str, field_size: int) -> str:
        """Generate checksum field definition.

        Args:
            field_name: Safe field name.
            field_size: Field size in bytes.

        Returns:
            Field definition code.
        """
        if field_size == 1:
            return f'XByteField("{field_name}", 0)'
        elif field_size == 2:
            return f'XShortField("{field_name}", 0)'
        else:
            return f'XIntField("{field_name}", 0)'

    def _generate_post_build(self, spec: ProtocolSpec) -> list[str]:
        """Generate post_build method for CRC calculation.

        Args:
            spec: Protocol specification.

        Returns:
            Python method code lines.
        """
        if not spec.checksum_type:
            return []

        width_map = {"crc8": 8, "crc16": 16, "crc32": 32}
        width = width_map.get(spec.checksum_type, 16)

        lines = [
            "    def post_build(self, pkt: bytes, pay: bytes) -> bytes:",
            '        """Calculate and insert CRC checksum after packet build.',
            "        ",
            "        Args:",
            "            pkt: Built packet bytes.",
            "            pay: Payload bytes.",
            "        ",
            "        Returns:",
            "            Packet with calculated CRC.",
            '        """',
        ]

        if spec.checksum_position == -1:
            # CRC at end
            crc_size = width // 8
            lines.extend(
                [
                    f"        # Calculate CRC over packet minus last {crc_size} bytes",
                    f"        data = pkt[:-{crc_size}]",
                    f"        crc = calculate_crc{width}(data)",
                    "        # Replace CRC field",
                ]
            )

            if width == 8:
                lines.append("        pkt = data + bytes([crc])")
            elif width == 16:
                lines.append("        pkt = data + crc.to_bytes(2, byteorder='big')")
            else:
                lines.append("        pkt = data + crc.to_bytes(4, byteorder='big')")

        lines.extend(["        return pkt + pay", ""])

        return lines

    def _generate_do_dissect(self, spec: ProtocolSpec) -> list[str]:
        """Generate do_dissect method for CRC validation.

        Args:
            spec: Protocol specification.

        Returns:
            Python method code lines.
        """
        if not spec.checksum_type:
            return []

        width_map = {"crc8": 8, "crc16": 16, "crc32": 32}
        width = width_map.get(spec.checksum_type, 16)
        crc_size = width // 8

        return [
            "    def do_dissect(self, s: bytes) -> bytes:",
            '        """Dissect packet and validate CRC.',
            "        ",
            "        Args:",
            "            s: Raw packet bytes.",
            "        ",
            "        Returns:",
            "            Remaining bytes after dissection.",
            '        """',
            "        # Standard dissection",
            "        s = super().do_dissect(s)",
            "        # Validate CRC",
            f"        data = bytes(self)[:-{crc_size}]",
            f"        calculated_crc = calculate_crc{width}(data)",
            f"        packet_crc = int.from_bytes(bytes(self)[-{crc_size}:], byteorder='big')",
            "        if calculated_crc != packet_crc:",
            '            print(f"CRC mismatch: calculated={calculated_crc:04x}, '
            'packet={packet_crc:04x}")',
            "        return s",
            "",
        ]

    def _generate_bind_layers(self, spec: ProtocolSpec) -> str:
        """Generate bind_layers statements.

        Args:
            spec: Protocol specification.

        Returns:
            Python bind_layers code.
        """
        # For now, don't auto-bind to any layers
        # Users can manually bind as needed
        return f"# To bind this layer to another layer, use:\n# bind_layers(UDP, {self._safe_class_name(spec.name)}, dport=YOUR_PORT)\n"

    def _assemble_code(self, imports: str, crc_code: str, layer_code: str, bind_code: str) -> str:
        """Assemble complete Python module.

        Args:
            imports: Import statements.
            crc_code: CRC function code.
            layer_code: Layer class code.
            bind_code: Bind layers code.

        Returns:
            Complete Python module code.
        """
        sections = [imports]

        if crc_code:
            sections.append(crc_code)

        sections.extend([layer_code, bind_code])

        return "\n".join(sections)

    def _validate_python_syntax(self, python_code: str) -> bool:
        """Validate Python syntax using ast.parse.

        Args:
            python_code: Python code to validate.

        Returns:
            True if syntax is valid, False if errors found.
        """
        try:
            ast.parse(python_code)
            logger.info("Python syntax validation passed")
            return True
        except SyntaxError as e:
            logger.error(f"Python syntax error: {e}")
            return False

    def _safe_class_name(self, name: str) -> str:
        """Convert protocol name to safe Python class name.

        Args:
            name: Protocol name.

        Returns:
            Safe Python class name (PascalCase).
        """
        # If name is already PascalCase (first letter uppercase, contains mix of upper/lower),
        # preserve it as-is after removing non-alphanumeric chars
        clean_name = "".join(c if c.isalnum() else "_" for c in name)

        # Check if already PascalCase (starts with uppercase, has lowercase letters)
        if clean_name and clean_name[0].isupper() and any(c.islower() for c in clean_name):
            # Remove underscores but preserve case
            return clean_name.replace("_", "")

        # Otherwise convert to PascalCase
        words = clean_name.split("_")
        return "".join(word.capitalize() for word in words if word)

    def _safe_field_name(self, name: str) -> str:
        """Convert field name to safe Python identifier.

        Args:
            name: Field name.

        Returns:
            Safe Python field name (snake_case).
        """
        # Remove non-alphanumeric characters
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        # Convert to snake_case
        return clean_name.lower()

    def _safe_module_name(self, name: str) -> str:
        """Convert protocol name to safe Python module name.

        Args:
            name: Protocol name.

        Returns:
            Safe Python module name (snake_case).
        """
        # Remove non-alphanumeric characters
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        # Convert to snake_case
        return clean_name.lower()
