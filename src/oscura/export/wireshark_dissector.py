"""Wireshark Lua dissector generator from ProtocolSpec.

This module generates functional Wireshark Lua dissectors from ProtocolSpec
objects (from reverse engineering workflows). The generated dissectors can
be loaded into Wireshark for interactive protocol analysis and validation.

Features:
    - Generate working Lua dissectors from ProtocolSpec
    - Support all field types (uint8, uint16, uint32, string, bytes, enum)
    - CRC validation in Lua
    - Test PCAP generation for validation
    - Lua syntax validation

Example:
    >>> from oscura.export.wireshark_dissector import (
    ...     WiresharkDissectorGenerator,
    ...     DissectorConfig
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
    ...     checksum_type=None,
    ...     checksum_position=None,
    ...     confidence=0.95
    ... )
    >>> config = DissectorConfig(protocol_name="MyProtocol", port=5000)
    >>> generator = WiresharkDissectorGenerator(config)
    >>> dissector_path, pcap_path = generator.generate(
    ...     spec,
    ...     sample_messages=[b"\\xaa\\x55\\x08test123"],
    ...     output_path=Path("myproto.lua")
    ... )

Installation:
    Copy the generated .lua file to your Wireshark plugins directory:
    - Linux: ~/.local/lib/wireshark/plugins/
    - macOS: ~/.config/wireshark/plugins/
    - Windows: %APPDATA%\\Wireshark\\plugins\\

References:
    - Wireshark Lua API: https://wiki.wireshark.org/LuaAPI
    - Lua Dissectors: https://wiki.wireshark.org/Lua/Dissectors
"""

from __future__ import annotations

import logging
import struct
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from oscura.utils.validation import validate_protocol_spec

if TYPE_CHECKING:
    from oscura.inference.crc_reverse import CRCParameters
    from oscura.workflows.reverse_engineering import ProtocolSpec

__all__ = ["DissectorConfig", "WiresharkDissectorGenerator"]

logger = logging.getLogger(__name__)


@dataclass
class DissectorConfig:
    """Configuration for Wireshark dissector generation.

    Attributes:
        protocol_name: Protocol name for dissector.
        port: UDP/TCP port number for registration (None for no registration).
        include_crc_validation: Include CRC validation code in dissector.
        generate_test_pcap: Generate test PCAP file with sample messages.
        wireshark_version: Target Wireshark version (default "3.0+").
    """

    protocol_name: str
    port: int | None = None
    include_crc_validation: bool = True
    generate_test_pcap: bool = True
    wireshark_version: str = "3.0+"


class WiresharkDissectorGenerator:
    """Generate functional Wireshark Lua dissectors from ProtocolSpec.

    This class converts ProtocolSpec objects (from reverse engineering workflows)
    into Wireshark Lua dissectors that can be loaded into Wireshark for protocol
    analysis and validation.

    Features:
        - All field types (uint8, uint16, uint32, string, bytes, enum)
        - CRC validation in Lua
        - Test PCAP generation
        - Lua syntax validation
        - UDP/TCP port registration

    Example:
        >>> config = DissectorConfig(protocol_name="MyProtocol", port=5000)
        >>> generator = WiresharkDissectorGenerator(config)
        >>> dissector_path, pcap_path = generator.generate(
        ...     spec,
        ...     sample_messages=[b"\\x01\\x02\\x03"],
        ...     output_path=Path("myproto.lua")
        ... )
    """

    def __init__(self, config: DissectorConfig) -> None:
        """Initialize dissector generator.

        Args:
            config: Dissector generation configuration.
        """
        self.config = config

    def generate(
        self,
        spec: ProtocolSpec,
        sample_messages: list[bytes],
        output_path: Path,
    ) -> tuple[Path, Path | None]:
        """Generate Wireshark Lua dissector and optional test PCAP.

        Args:
            spec: Protocol specification from reverse engineering.
            sample_messages: Sample protocol messages for test PCAP.
            output_path: Path for output .lua file.

        Returns:
            Tuple of (dissector_lua_path, test_pcap_path).
            test_pcap_path is None if generate_test_pcap is False.

        Raises:
            ValueError: If spec is invalid or has missing required fields.
            RuntimeError: If Lua syntax validation fails.
            OSError: If file writing fails.

        Example:
            >>> spec = ProtocolSpec(name="test", ...)
            >>> generator = WiresharkDissectorGenerator(config)
            >>> lua_path, pcap_path = generator.generate(
            ...     spec,
            ...     [b"\\x01\\x02\\x03"],
            ...     Path("test.lua")
            ... )
        """
        # Validate spec
        self._validate_spec(spec)

        # Generate Lua code
        lua_code = self._generate_lua_dissector(spec)

        # Validate Lua syntax
        if not self._validate_lua_syntax(lua_code):
            raise RuntimeError("Generated Lua code has syntax errors")

        # Write Lua file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(lua_code, encoding="utf-8")
        logger.info(f"Generated Lua dissector: {output_path}")

        # Generate test PCAP if requested
        pcap_path = None
        if self.config.generate_test_pcap and sample_messages:
            pcap_path = output_path.with_suffix(".pcap")
            self._generate_test_pcap(sample_messages, pcap_path)
            logger.info(f"Generated test PCAP: {pcap_path}")

        return output_path, pcap_path

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

    def _generate_lua_dissector(self, spec: ProtocolSpec) -> str:
        """Generate complete Lua dissector code.

        Args:
            spec: Protocol specification.

        Returns:
            Complete Lua dissector code as string.
        """
        sections = [
            self._generate_header(spec),
            self._generate_protocol_declaration(spec),
            self._generate_field_declarations(spec),
            self._generate_crc_validator(spec) if self.config.include_crc_validation else "",
            self._generate_dissector_function(spec),
            self._generate_registration(spec),
        ]

        return "\n\n".join(s for s in sections if s)

    def _generate_header(self, spec: ProtocolSpec) -> str:
        """Generate Lua file header with installation instructions.

        Args:
            spec: Protocol specification.

        Returns:
            Lua header comment block.
        """
        return f"""-- Wireshark Lua Dissector for {spec.name}
-- Generated by Oscura on {datetime.now(UTC).isoformat()}
--
-- Installation:
--   Copy this file to your Wireshark plugins directory:
--   - Linux:   ~/.local/lib/wireshark/plugins/
--   - macOS:   ~/.config/wireshark/plugins/
--   - Windows: %APPDATA%\\Wireshark\\plugins\\
--
-- Protocol: {spec.name}
-- Frame Format: {spec.frame_format}
-- Baud Rate: {spec.baud_rate} bps
-- Frame Length: {spec.frame_length if spec.frame_length else "Variable"} bytes
-- Sync Pattern: {spec.sync_pattern}
-- Checksum: {spec.checksum_type if spec.checksum_type else "None"}
-- Confidence: {spec.confidence:.2f}
--
-- Wireshark Version: {self.config.wireshark_version}
"""

    def _generate_protocol_declaration(self, spec: ProtocolSpec) -> str:
        """Generate protocol declaration (Proto object).

        Args:
            spec: Protocol specification.

        Returns:
            Lua code for protocol declaration.
        """
        proto_var = spec.name.lower().replace(" ", "_").replace("-", "_")
        return f"""-- Protocol declaration
local {proto_var}_proto = Proto("{proto_var}", "{spec.name}")"""

    def _generate_field_declarations(self, spec: ProtocolSpec) -> str:
        """Generate ProtoField declarations for all fields.

        Args:
            spec: Protocol specification.

        Returns:
            Lua code for field declarations.
        """
        proto_var = spec.name.lower().replace(" ", "_").replace("-", "_")
        lines = ["-- Field declarations"]

        for field in spec.fields:
            field_var = f"f_{field.name}"
            field_path = f"{proto_var}.{field.name}"
            display_name = field.name.replace("_", " ").title()

            # Determine ProtoField type and base
            if field.field_type == "uint8":
                proto_type = "uint8"
                base = "base.HEX"
            elif field.field_type == "uint16":
                proto_type = "uint16"
                base = "base.HEX"
            elif field.field_type == "uint32":
                proto_type = "uint32"
                base = "base.HEX"
            elif field.field_type == "string":
                proto_type = "string"
                base = "base.UNICODE"
            elif field.field_type in ("bytes", "constant"):
                proto_type = "bytes"
                base = "base.SPACE"
            else:  # checksum
                proto_type = "uint16"  # Assume 16-bit checksum
                base = "base.HEX"

            # Handle enum values
            if hasattr(field, "enum") and field.enum:
                # Generate value_string table
                enum_var = f"vs_{field.name}"
                lines.append(f"local {enum_var} = {{")
                for key, value in field.enum.items():
                    lines.append(f'  [{key}] = "{value}",')
                lines.append("}")
                lines.append(
                    f"local {field_var} = ProtoField.{proto_type}("
                    f'"{field_path}", "{display_name}", {base}, {enum_var})'
                )
            else:
                lines.append(
                    f"local {field_var} = ProtoField.{proto_type}("
                    f'"{field_path}", "{display_name}", {base})'
                )

        # Register fields with protocol
        lines.append("")
        lines.append(f"{proto_var}_proto.fields = {{")
        for field in spec.fields:
            lines.append(f"  f_{field.name},")
        lines.append("}")

        return "\n".join(lines)

    def _generate_crc_validator(self, spec: ProtocolSpec) -> str:
        """Generate CRC validation function in Lua.

        Args:
            spec: Protocol specification.

        Returns:
            Lua CRC validation function code.
        """
        if not spec.checksum_type or spec.checksum_type not in ("crc8", "crc16", "crc32"):
            return ""

        # Get CRC parameters if available
        crc_info = getattr(spec, "crc_info", None)
        if crc_info:
            return self._generate_crc_function_from_params(crc_info)

        # Default CRC implementations for common types
        if spec.checksum_type == "crc16":
            return """-- CRC-16-CCITT validation
function validate_crc16(buffer, offset, length)
  local crc = 0xFFFF
  local poly = 0x1021

  for i = 0, length - 1 do
    local byte = buffer(offset + i, 1):uint()
    crc = bit32.bxor(crc, bit32.lshift(byte, 8))

    for bit = 0, 7 do
      if bit32.band(crc, 0x8000) ~= 0 then
        crc = bit32.band(bit32.bxor(bit32.lshift(crc, 1), poly), 0xFFFF)
      else
        crc = bit32.band(bit32.lshift(crc, 1), 0xFFFF)
      end
    end
  end

  return crc
end"""
        elif spec.checksum_type == "crc8":
            return """-- CRC-8 validation
function validate_crc8(buffer, offset, length)
  local crc = 0x00
  local poly = 0x07

  for i = 0, length - 1 do
    local byte = buffer(offset + i, 1):uint()
    crc = bit32.bxor(crc, byte)

    for bit = 0, 7 do
      if bit32.band(crc, 0x80) ~= 0 then
        crc = bit32.band(bit32.bxor(bit32.lshift(crc, 1), poly), 0xFF)
      else
        crc = bit32.band(bit32.lshift(crc, 1), 0xFF)
      end
    end
  end

  return crc
end"""
        else:  # crc32
            return """-- CRC-32 validation
function validate_crc32(buffer, offset, length)
  local crc = 0xFFFFFFFF
  local poly = 0x04C11DB7

  for i = 0, length - 1 do
    local byte = buffer(offset + i, 1):uint()
    crc = bit32.bxor(crc, bit32.lshift(byte, 24))

    for bit = 0, 7 do
      if bit32.band(crc, 0x80000000) ~= 0 then
        crc = bit32.band(bit32.bxor(bit32.lshift(crc, 1), poly), 0xFFFFFFFF)
      else
        crc = bit32.band(bit32.lshift(crc, 1), 0xFFFFFFFF)
      end
    end
  end

  return bit32.bxor(crc, 0xFFFFFFFF)
end"""

    def _generate_crc_function_from_params(self, crc_info: CRCParameters) -> str:
        """Generate CRC function from CRCParameters.

        Args:
            crc_info: CRC parameters from reverse engineering.

        Returns:
            Lua CRC function code.
        """
        width = crc_info.width
        poly = crc_info.polynomial
        init = crc_info.init
        xor_out = crc_info.xor_out
        mask = (1 << width) - 1

        func_name = f"validate_crc{width}"

        code = [
            f"-- CRC-{width} validation (Custom parameters)",
            f"-- Polynomial: 0x{poly:0{width // 4}x}",
            f"-- Init: 0x{init:0{width // 4}x}",
            f"-- XorOut: 0x{xor_out:0{width // 4}x}",
            f"-- ReflectIn: {str(crc_info.reflect_in).lower()}",
            f"-- ReflectOut: {str(crc_info.reflect_out).lower()}",
            f"function {func_name}(buffer, offset, length)",
            f"  local crc = 0x{init:0{width // 4}x}",
            f"  local poly = 0x{poly:0{width // 4}x}",
            f"  local mask = 0x{mask:0{width // 4}x}",
            "  ",
            "  for i = 0, length - 1 do",
            "    local byte = buffer(offset + i, 1):uint()",
        ]

        if crc_info.reflect_in:
            code.extend(
                [
                    "    -- Reflect input byte",
                    "    local reflected = 0",
                    "    for b = 0, 7 do",
                    "      if bit32.band(byte, bit32.lshift(1, b)) ~= 0 then",
                    "        reflected = bit32.bor(reflected, bit32.lshift(1, 7 - b))",
                    "      end",
                    "    end",
                    "    byte = reflected",
                ]
            )

        code.extend(
            [
                f"    crc = bit32.bxor(crc, bit32.lshift(byte, {width - 8}))",
                "    ",
                "    for bit = 0, 7 do",
                f"      if bit32.band(crc, 0x{1 << (width - 1):0{width // 4}x}) ~= 0 then",
                "        crc = bit32.band(bit32.bxor(bit32.lshift(crc, 1), poly), mask)",
                "      else",
                "        crc = bit32.band(bit32.lshift(crc, 1), mask)",
                "      end",
                "    end",
                "  end",
                "  ",
            ]
        )

        if crc_info.reflect_out:
            code.extend(
                [
                    "  -- Reflect output CRC",
                    "  local reflected = 0",
                    f"  for b = 0, {width - 1} do",
                    "    if bit32.band(crc, bit32.lshift(1, b)) ~= 0 then",
                    f"      reflected = bit32.bor(reflected, bit32.lshift(1, {width - 1} - b))",
                    "    end",
                    "  end",
                    "  crc = reflected",
                ]
            )

        code.extend(
            [
                f"  return bit32.bxor(crc, 0x{xor_out:0{width // 4}x})",
                "end",
            ]
        )

        return "\n".join(code)

    def _generate_dissector_function(self, spec: ProtocolSpec) -> str:
        """Generate main dissector function.

        Args:
            spec: Protocol specification.

        Returns:
            Lua dissector function code.
        """
        proto_var = spec.name.lower().replace(" ", "_").replace("-", "_")
        min_length = spec.frame_length if spec.frame_length else 1

        lines = [
            "-- Main dissector function",
            f"function {proto_var}_proto.dissector(buffer, pinfo, tree)",
            "  -- Check minimum length",
            f"  if buffer:len() < {min_length} then",
            "    return 0",
            "  end",
            "  ",
            f'  pinfo.cols.protocol = "{spec.name}"',
            "  ",
            f'  local subtree = tree:add({proto_var}_proto, buffer(), "{spec.name}")',
            "  local offset = 0",
            "  ",
        ]

        # Extract fields
        for field in spec.fields:
            field_size = field.size if isinstance(field.size, int) else 1

            # Determine buffer reader
            if field.field_type == "uint8":
                reader = f"buffer(offset, {field_size}):uint()"
            elif field.field_type == "uint16" or field.field_type == "uint32":
                endian = getattr(field, "endian", "big")
                if endian == "little":
                    reader = f"buffer(offset, {field_size}):le_uint()"
                else:
                    reader = f"buffer(offset, {field_size}):uint()"
            elif field.field_type == "string":
                reader = f"buffer(offset, {field_size}):string()"
            else:  # bytes, constant, checksum
                reader = f"buffer(offset, {field_size})"

            lines.append(f"  -- Field: {field.name}")
            lines.append(f"  subtree:add(f_{field.name}, {reader})")
            lines.append(f"  offset = offset + {field_size}")
            lines.append("  ")

        # Add CRC validation
        if spec.checksum_type and spec.checksum_position is not None:
            width_map = {"crc8": 8, "crc16": 16, "crc32": 32}
            width = width_map.get(spec.checksum_type, 16)
            crc_size = width // 8

            if spec.checksum_position == -1:
                # CRC is at the end
                lines.extend(
                    [
                        "  -- Validate CRC",
                        f"  local data_length = buffer:len() - {crc_size}",
                        f"  local computed_crc = validate_crc{width}(buffer, 0, data_length)",
                        f"  local packet_crc = buffer(data_length, {crc_size}):uint()",
                        "  if computed_crc == packet_crc then",
                        f'    subtree:add(buffer(data_length, {crc_size}), "CRC: Valid")',
                        "  else",
                        '    subtree:add_expert_info(PI_CHECKSUM, PI_ERROR, "CRC: Invalid")',
                        "  end",
                        "  ",
                    ]
                )

        lines.extend(
            [
                "  return buffer:len()",
                "end",
            ]
        )

        return "\n".join(lines)

    def _generate_registration(self, spec: ProtocolSpec) -> str:
        """Generate protocol registration code.

        Args:
            spec: Protocol specification.

        Returns:
            Lua registration code.
        """
        proto_var = spec.name.lower().replace(" ", "_").replace("-", "_")

        if self.config.port is None:
            return f"""-- Protocol registration (manual)
-- To use this dissector:
-- 1. Open Wireshark
-- 2. Right-click on a packet
-- 3. Select "Decode As..."
-- 4. Choose "{spec.name}"
--
-- Or register on a specific port by adding:
-- DissectorTable.get("udp.port"):add(YOUR_PORT, {proto_var}_proto)"""

        return f"""-- Protocol registration
-- Register on UDP port {self.config.port}
local udp_port = DissectorTable.get("udp.port")
udp_port:add({self.config.port}, {proto_var}_proto)

-- Also register on TCP port {self.config.port}
local tcp_port = DissectorTable.get("tcp.port")
tcp_port:add({self.config.port}, {proto_var}_proto)"""

    def _validate_lua_syntax(self, lua_code: str) -> bool:
        """Validate Lua syntax using luac if available.

        Args:
            lua_code: Lua code to validate.

        Returns:
            True if syntax is valid or luac not available, False if errors found.
        """
        try:
            # Try to run luac syntax check
            result = subprocess.run(
                ["luac", "-p", "-"],
                input=lua_code.encode("utf-8"),
                capture_output=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                logger.error(f"Lua syntax error: {result.stderr.decode('utf-8')}")
                return False
            logger.info("Lua syntax validation passed")
            return True
        except FileNotFoundError:
            # luac not available, skip validation
            logger.warning("luac not found, skipping Lua syntax validation")
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Lua syntax validation timed out")
            return True
        except Exception as e:
            logger.warning(f"Lua syntax validation failed: {e}")
            return True

    def _generate_test_pcap(self, sample_messages: list[bytes], output_path: Path) -> None:
        """Generate test PCAP file with sample messages.

        Creates a PCAP file with UDP packets containing the sample messages.
        This allows testing the dissector in Wireshark.

        Args:
            sample_messages: List of protocol messages to include.
            output_path: Path for output .pcap file.

        Raises:
            OSError: If file writing fails.
        """
        pcap_data = bytearray(self._build_pcap_header())
        dst_port = self.config.port if self.config.port else 5000

        for i, message in enumerate(sample_messages):
            packet = self._build_udp_packet(i, message, dst_port)
            pcap_data.extend(self._build_packet_header(packet))

        output_path.write_bytes(pcap_data)

    def _build_pcap_header(self) -> bytes:
        """Build PCAP global header."""
        return struct.pack(
            "<IHHIIII",
            0xA1B2C3D4,
            2,
            4,
            0,
            0,
            65535,
            1,  # Magic, versions, timezone, accuracy, snaplen, linktype
        )

    def _build_udp_packet(self, index: int, message: bytes, dst_port: int) -> bytes:
        """Build complete UDP packet with Ethernet/IP/UDP headers."""
        src_ip = bytes([192, 168, 1, 1])
        dst_ip = bytes([192, 168, 1, 2])

        eth_header = (
            bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF])  # Dst MAC
            + bytes([0x00, 0x11, 0x22, 0x33, 0x44, 0x55])  # Src MAC
            + bytes([0x08, 0x00])  # IPv4
        )

        ip_header = self._build_ip_header(index, len(message), src_ip, dst_ip)
        udp_header = struct.pack(">HHHH", 50000 + index, dst_port, 8 + len(message), 0)

        return eth_header + ip_header + udp_header + message

    def _build_ip_header(
        self, packet_id: int, payload_len: int, src_ip: bytes, dst_ip: bytes
    ) -> bytes:
        """Build IP header with checksum calculation."""
        ip_total_length = 20 + 8 + payload_len
        header = struct.pack(
            ">BBHHHBBH4s4s", 0x45, 0, ip_total_length, packet_id + 1, 0, 64, 17, 0, src_ip, dst_ip
        )
        checksum = self._calculate_ip_checksum(header)
        return struct.pack(
            ">BBHHHBBH4s4s",
            0x45,
            0,
            ip_total_length,
            packet_id + 1,
            0,
            64,
            17,
            checksum,
            src_ip,
            dst_ip,
        )

    def _build_packet_header(self, packet: bytes) -> bytes:
        """Build PCAP packet header."""
        timestamp = int(datetime.now(UTC).timestamp())
        return struct.pack("<IIII", timestamp, 0, len(packet), len(packet)) + packet

    def _calculate_ip_checksum(self, header: bytes) -> int:
        """Calculate IP header checksum.

        Args:
            header: IP header bytes.

        Returns:
            Checksum value.
        """
        # Sum all 16-bit words
        checksum = 0
        for i in range(0, len(header), 2):
            word = (header[i] << 8) + (header[i + 1] if i + 1 < len(header) else 0)
            checksum += word

        # Add carry bits
        while checksum >> 16:
            checksum = (checksum & 0xFFFF) + (checksum >> 16)

        # One's complement
        return ~checksum & 0xFFFF
