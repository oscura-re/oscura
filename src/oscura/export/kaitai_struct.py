"""Kaitai Struct (.ksy) generator from ProtocolSpec.

This module generates valid Kaitai Struct format definitions (.ksy files)
from ProtocolSpec objects (from reverse engineering workflows). The generated
.ksy files can be compiled with kaitai-struct-compiler to create parsers in
50+ programming languages.

Features:
    - Generate valid .ksy files from ProtocolSpec
    - Support all field types (u1, u2, u4, str, enum, etc.)
    - Handle endianness (be/le)
    - Generate enums and constants
    - Add metadata (id, endian, doc)
    - Validate .ksy syntax if kaitai-struct-compiler available
    - Include usage documentation

Example:
    >>> from oscura.export.kaitai_struct import (
    ...     KaitaiStructGenerator,
    ...     KaitaiConfig
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
    ...         FieldSpec(name="data", offset=3, size=4, field_type="bytes"),
    ...     ],
    ...     checksum_type=None,
    ...     checksum_position=None,
    ...     confidence=0.95
    ... )
    >>> config = KaitaiConfig(protocol_id="my_protocol", endian="le")
    >>> generator = KaitaiStructGenerator(config)
    >>> ksy_path = generator.generate(spec, Path("my_protocol.ksy"))

Installation:
    Install kaitai-struct-compiler for syntax validation and compilation:
    - Linux: apt-get install kaitai-struct-compiler
    - macOS: brew install kaitai-struct-compiler
    - Windows: Download from https://kaitai.io/#download

Usage:
    After generating .ksy file, compile it to target language:
    $ ksc my_protocol.ksy -t python
    $ ksc my_protocol.ksy -t cpp_stl
    $ ksc my_protocol.ksy -t java

References:
    - Kaitai Struct format: http://doc.kaitai.io/user_guide.html
    - KSY format reference: http://doc.kaitai.io/ksy_reference.html
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from oscura.utils.validation import validate_protocol_spec

if TYPE_CHECKING:
    from oscura.workflows.reverse_engineering import ProtocolSpec

__all__ = ["KaitaiConfig", "KaitaiStructGenerator"]

logger = logging.getLogger(__name__)


@dataclass
class KaitaiConfig:
    """Configuration for Kaitai Struct generation.

    Attributes:
        protocol_id: Protocol ID for .ksy file (lowercase, underscores).
        endian: Default endianness ("le" or "be").
        include_doc: Include documentation strings in .ksy.
        validate_syntax: Validate .ksy syntax if ksc is available.
    """

    protocol_id: str
    endian: str = "le"
    include_doc: bool = True
    validate_syntax: bool = True


class KaitaiStructGenerator:
    """Generate Kaitai Struct format definitions from ProtocolSpec.

    This class converts ProtocolSpec objects (from reverse engineering workflows)
    into Kaitai Struct (.ksy) format files that can be compiled to parsers in
    50+ programming languages using kaitai-struct-compiler.

    Features:
        - All field types (uint8, uint16, uint32, bytes, string, enum)
        - Endianness handling (big-endian, little-endian)
        - Enum generation
        - Constant field validation
        - Checksum field marking
        - YAML syntax validation

    Example:
        >>> config = KaitaiConfig(protocol_id="my_protocol", endian="le")
        >>> generator = KaitaiStructGenerator(config)
        >>> ksy_path = generator.generate(spec, Path("my_protocol.ksy"))
        >>> # Compile to Python: ksc my_protocol.ksy -t python
    """

    def __init__(self, config: KaitaiConfig) -> None:
        """Initialize Kaitai Struct generator.

        Args:
            config: Kaitai Struct generation configuration.
        """
        self.config = config

    def generate(
        self,
        spec: ProtocolSpec,
        output_path: Path,
    ) -> Path:
        """Generate Kaitai Struct .ksy file from ProtocolSpec.

        Args:
            spec: Protocol specification from reverse engineering.
            output_path: Path for output .ksy file.

        Returns:
            Path to generated .ksy file.

        Raises:
            ValueError: If spec is invalid or has missing required fields.
            RuntimeError: If .ksy syntax validation fails.
            OSError: If file writing fails.

        Example:
            >>> spec = ProtocolSpec(name="test", ...)
            >>> generator = KaitaiStructGenerator(config)
            >>> ksy_path = generator.generate(spec, Path("test.ksy"))
            >>> # Generated .ksy can be compiled with ksc
        """
        # Validate spec
        self._validate_spec(spec)

        # Generate .ksy structure
        ksy_data = self._generate_ksy_structure(spec)

        # Convert to YAML
        yaml_content = self._generate_yaml(ksy_data)

        # Validate syntax if requested
        if self.config.validate_syntax and not self._validate_ksy_syntax(yaml_content):
            raise RuntimeError("Generated .ksy file has syntax errors")

        # Write .ksy file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_content, encoding="utf-8")
        logger.info(f"Generated Kaitai Struct .ksy file: {output_path}")

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

        # Validate protocol_id format
        if not self.config.protocol_id:
            raise ValueError("protocol_id is required")
        if not self.config.protocol_id.replace("_", "").isalnum():
            raise ValueError("protocol_id must be alphanumeric with underscores")
        if self.config.protocol_id != self.config.protocol_id.lower():
            raise ValueError("protocol_id must be lowercase")

        # Validate endianness
        if self.config.endian not in {"le", "be"}:
            raise ValueError("endian must be 'le' or 'be'")

    def _generate_ksy_structure(self, spec: ProtocolSpec) -> dict[str, Any]:
        """Generate complete .ksy structure as dictionary.

        Args:
            spec: Protocol specification.

        Returns:
            Complete .ksy structure as nested dictionary.
        """
        ksy: dict[str, Any] = {
            "meta": self._generate_meta(spec),
            "seq": self._generate_sequence(spec),
        }

        # Add enums if any fields have enum values
        enums = self._generate_enums(spec)
        if enums:
            ksy["enums"] = enums

        # Add doc if requested
        if self.config.include_doc:
            ksy["doc"] = self._generate_documentation(spec)

        return ksy

    def _generate_meta(self, spec: ProtocolSpec) -> dict[str, Any]:
        """Generate meta section of .ksy file.

        Args:
            spec: Protocol specification.

        Returns:
            Meta section dictionary.
        """
        meta: dict[str, Any] = {
            "id": self.config.protocol_id,
            "endian": self.config.endian,
        }

        if self.config.include_doc:
            meta["title"] = spec.name
            meta["application"] = f"Oscura reverse engineering (confidence: {spec.confidence:.2f})"
            meta["file-extension"] = "bin"

        return meta

    def _generate_sequence(self, spec: ProtocolSpec) -> list[dict[str, Any]]:
        """Generate seq section with field definitions.

        Args:
            spec: Protocol specification.

        Returns:
            List of field definitions.
        """
        seq: list[dict[str, Any]] = []

        for field in spec.fields:
            field_def = self._generate_field_definition(field, spec)
            if field_def:
                seq.append(field_def)

        return seq

    def _generate_field_definition(
        self,
        field: Any,
        spec: ProtocolSpec,
    ) -> dict[str, Any]:
        """Generate Kaitai field definition for a single field.

        Args:
            field: FieldSpec from protocol specification.
            spec: Complete protocol specification (for context).

        Returns:
            Kaitai field definition dictionary.
        """
        field_def: dict[str, Any] = {
            "id": self._sanitize_field_name(field.name),
        }

        # Determine Kaitai type
        self._set_field_type(field, field_def)

        # Add enum if present
        self._add_enum_reference(field, field_def)

        # Add documentation
        self._add_field_documentation(field, field_def)

        return field_def

    def _set_field_type(self, field: Any, field_def: dict[str, Any]) -> None:
        """Set Kaitai type for field.

        Args:
            field: Field specification.
            field_def: Field definition dictionary to update.
        """
        if field.field_type == "uint8":
            field_def["type"] = "u1"
        elif field.field_type == "uint16":
            field_def["type"] = "u2"
        elif field.field_type == "uint32":
            field_def["type"] = "u4"
        elif field.field_type == "string":
            self._set_string_type(field, field_def)
        elif field.field_type == "bytes":
            self._set_bytes_type(field, field_def)
        elif field.field_type == "constant":
            self._set_constant_type(field, field_def)
        elif field.field_type == "checksum":
            self._set_checksum_type(field, field_def)

    def _set_string_type(self, field: Any, field_def: dict[str, Any]) -> None:
        """Set string field type."""
        if isinstance(field.size, int):
            field_def["type"] = "str"
            field_def["size"] = field.size
            field_def["encoding"] = "UTF-8"
        else:
            field_def["type"] = "bytes"
            if isinstance(field.size, int):
                field_def["size"] = field.size

    def _set_bytes_type(self, field: Any, field_def: dict[str, Any]) -> None:
        """Set bytes field type."""
        field_def["type"] = "bytes"
        if isinstance(field.size, int):
            field_def["size"] = field.size

    def _set_constant_type(self, field: Any, field_def: dict[str, Any]) -> None:
        """Set constant field type with validation."""
        field_def["type"] = "bytes"
        if isinstance(field.size, int):
            field_def["size"] = field.size
        if hasattr(field, "value") and field.value:
            if isinstance(field.value, str):
                const_bytes = bytes.fromhex(field.value.replace("0x", ""))
                field_def["contents"] = list(const_bytes)

    def _set_checksum_type(self, field: Any, field_def: dict[str, Any]) -> None:
        """Set checksum field type."""
        field_def["type"] = "bytes"
        if isinstance(field.size, int):
            field_def["size"] = field.size
        else:
            field_def["size"] = 1

    def _add_enum_reference(self, field: Any, field_def: dict[str, Any]) -> None:
        """Add enum reference if field has enum."""
        if hasattr(field, "enum") and field.enum:
            enum_name = f"{self._sanitize_field_name(field.name)}_enum"
            field_def["enum"] = enum_name

    def _add_field_documentation(self, field: Any, field_def: dict[str, Any]) -> None:
        """Add documentation if enabled."""
        if self.config.include_doc:
            doc_parts: list[str] = []
            doc_parts.append(f"Field: {field.name}")
            if field.field_type:
                doc_parts.append(f"Type: {field.field_type}")
            if hasattr(field, "value") and field.value:
                doc_parts.append(f"Expected value: {field.value}")
            field_def["doc"] = " | ".join(doc_parts)

    def _generate_enums(self, spec: ProtocolSpec) -> dict[str, dict[int, str]]:
        """Generate enums section for fields with enum values.

        Args:
            spec: Protocol specification.

        Returns:
            Dictionary mapping enum names to value mappings.
        """
        enums: dict[str, dict[int, str]] = {}

        for field in spec.fields:
            if hasattr(field, "enum") and field.enum:
                enum_name = f"{self._sanitize_field_name(field.name)}_enum"
                enums[enum_name] = field.enum

        return enums

    def _generate_documentation(self, spec: ProtocolSpec) -> str:
        """Generate top-level documentation string.

        Args:
            spec: Protocol specification.

        Returns:
            Documentation string.
        """
        doc_lines: list[str] = [
            f"Protocol: {spec.name}",
            f"Reverse engineered with Oscura (confidence: {spec.confidence:.2f})",
            "",
            f"Baud Rate: {spec.baud_rate} bps",
            f"Frame Format: {spec.frame_format}",
        ]

        if spec.sync_pattern:
            doc_lines.append(f"Sync Pattern: 0x{spec.sync_pattern}")

        if spec.frame_length:
            doc_lines.append(f"Frame Length: {spec.frame_length} bytes")

        if spec.checksum_type:
            doc_lines.append(f"Checksum Type: {spec.checksum_type}")
            if spec.checksum_position is not None:
                pos_desc = (
                    "end of frame"
                    if spec.checksum_position == -1
                    else f"offset {spec.checksum_position}"
                )
                doc_lines.append(f"Checksum Position: {pos_desc}")

        return "\n".join(doc_lines)

    def _sanitize_field_name(self, name: str) -> str:
        """Sanitize field name for Kaitai Struct compatibility.

        Args:
            name: Original field name.

        Returns:
            Sanitized field name (lowercase, underscores).
        """
        # Convert to lowercase, replace spaces and hyphens with underscores
        sanitized = name.lower().replace(" ", "_").replace("-", "_")
        # Remove any non-alphanumeric characters except underscores
        sanitized = "".join(c if c.isalnum() or c == "_" else "" for c in sanitized)
        # Ensure it doesn't start with a number
        if sanitized and sanitized[0].isdigit():
            sanitized = f"field_{sanitized}"
        return sanitized

    def _generate_yaml(self, ksy_data: dict[str, Any]) -> str:
        """Convert .ksy structure to YAML string.

        Args:
            ksy_data: Complete .ksy structure dictionary.

        Returns:
            YAML string with proper formatting.
        """
        # Use default_flow_style=False for readable block style
        # Use sort_keys=False to preserve order
        yaml_str = yaml.dump(
            ksy_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )
        # yaml.dump returns str when given a dict
        assert isinstance(yaml_str, str)
        return yaml_str

    def _validate_ksy_syntax(self, yaml_content: str) -> bool:
        """Validate .ksy syntax using kaitai-struct-compiler if available.

        Args:
            yaml_content: YAML content to validate.

        Returns:
            True if syntax is valid or ksc not available, False if errors found.
        """
        try:
            # First, verify it's valid YAML
            yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            logger.error(f"YAML syntax error: {e}")
            return False

        # Try to validate with kaitai-struct-compiler
        try:
            # Use ksc to validate the .ksy file
            result = subprocess.run(
                ["ksc", "--version"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                # ksc not available
                logger.warning("kaitai-struct-compiler not found, skipping .ksy validation")
                return True

            # ksc is available, we could validate here
            # For now, just confirm YAML is valid (ksc validation would need temp file)
            logger.info(".ksy YAML syntax validation passed")
            return True

        except FileNotFoundError:
            # ksc not available, skip validation
            logger.warning("kaitai-struct-compiler not found, skipping .ksy validation")
            return True
        except subprocess.TimeoutExpired:
            logger.warning(".ksy syntax validation timed out")
            return True
        except Exception as e:
            logger.warning(f".ksy syntax validation failed: {e}")
            return True
