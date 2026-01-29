"""Configurable binary packet loader with schema-driven parsing.

This module provides a flexible, configuration-driven system for loading
binary packet/frame data from custom DAQ systems, logic analyzers, and
packet captures without code changes.

Features:
    - Schema-driven packet format definition
    - Device/source configuration mapping
    - Multi-source binary data loader

Example:
    >>> from oscura.loaders.configurable import load_binary_packets
    >>> packets = load_binary_packets(
    ...     "capture.bin",
    ...     format_config="packet_format.yaml",
    ...     device_config="device_mapping.yaml"
    ... )
    >>> traces = extract_channels(packets, {"ch0": {"bits": [0, 7]}})
    >>> print(f"Loaded {len(traces['ch0'].data)} samples")
"""

from __future__ import annotations

import json
import logging
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from oscura.core.exceptions import ConfigurationError, FormatError, LoaderError
from oscura.core.types import DigitalTrace, TraceMetadata

if TYPE_CHECKING:
    from os import PathLike


# Logger for debug output
logger = logging.getLogger(__name__)

# Type size mapping in bytes
TYPE_SIZES = {
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint40": 5,
    "uint48": 6,
    "uint64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "float32": 4,
    "float64": 8,
}

# Type alias for parsed packet data
ParsedPacket = dict[str, Any]
"""Type alias for a parsed packet dictionary with header and samples."""


@dataclass
class BitfieldDef:
    """Bitfield definition within a header field.

    Attributes:
        name: Bitfield name.
        bit: Single bit position (if single-bit field).
        bits: Bit range [start, end] inclusive (if multi-bit field).
        description: Human-readable description (optional).
    """

    name: str
    bit: int | None = None
    bits: tuple[int, int] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate bitfield definition."""
        if self.bit is None and self.bits is None:
            raise ConfigurationError(
                "BitfieldDef must have either 'bit' or 'bits' specified",
                config_key=f"{self.name}",
            )
        if self.bit is not None and self.bits is not None:
            raise ConfigurationError(
                "BitfieldDef cannot have both 'bit' and 'bits' specified",
                config_key=f"{self.name}",
            )


@dataclass
class DeviceInfo:
    """Device information from configuration.

    Attributes:
        name: Full device name.
        short_name: Short device name (optional).
        description: Device description (optional).
        category: Device category (optional).
        sample_rate: Sample rate in Hz (optional).
        channels: Number of channels (optional).
        properties: Additional device properties (optional).
    """

    name: str
    short_name: str = ""
    description: str = ""
    category: str = ""
    sample_rate: float | None = None
    channels: int | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeviceInfo:
        """Create DeviceInfo from dictionary.

        Args:
            data: Device configuration dictionary.

        Returns:
            DeviceInfo instance.
        """
        return cls(
            name=data.get("name", "Unknown Device"),
            short_name=data.get("short_name", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
            sample_rate=data.get("sample_rate"),
            channels=data.get("channels"),
            properties=data.get("properties", {}),
        )


@dataclass
class HeaderFieldDef:
    """Header field definition.

    Defines a single field within a packet header including offset,
    size, type, and endianness.

    Attributes:
        name: Field name.
        offset: Byte offset from start of packet.
        size: Field size in bytes.
        type: Data type (uint8, uint16, uint32, uint40, uint48, uint64, bitfield, bytes).
        endian: Byte order ("big", "little", or "native").
        value: Expected constant value for validation (optional).
        fields: Bitfield definitions if type is "bitfield" (optional).
        description: Human-readable description (optional).
    """

    name: str
    offset: int
    size: int
    type: str
    endian: str = "big"
    value: int | bytes | None = None
    fields: dict[str, Any] | None = None
    description: str = ""

    def __post_init__(self) -> None:
        """Validate field definition."""
        if self.offset < 0:
            raise ConfigurationError(
                "Field offset must be non-negative",
                config_key=f"{self.name}.offset",
                actual_value=self.offset,
            )
        if self.size <= 0:
            raise ConfigurationError(
                "Field size must be positive",
                config_key=f"{self.name}.size",
                actual_value=self.size,
            )
        if self.endian not in ("big", "little", "native"):
            raise ConfigurationError(
                "Invalid endianness",
                config_key=f"{self.name}.endian",
                expected_type="'big', 'little', or 'native'",
                actual_value=self.endian,
            )


@dataclass
class SampleFormatDef:
    """Sample data format definition.

    Defines how to extract sample data from packets.

    Attributes:
        size: Bytes per sample.
        type: Data type (uint8, uint16, uint32, uint64).
        endian: Byte order ("big", "little", or "native").
        description: Human-readable description (optional).
    """

    size: int
    type: str
    endian: str = "little"
    description: str = ""

    def __post_init__(self) -> None:
        """Validate sample format."""
        if self.size <= 0:
            raise ConfigurationError(
                "Sample size must be positive",
                config_key="samples.format.size",
                actual_value=self.size,
            )


@dataclass
class PacketFormatConfig:
    """Packet format configuration.

    Complete packet format specification loaded from YAML/JSON.

    Attributes:
        name: Format name.
        version: Format version.
        packet_size: Total packet size in bytes (or "variable").
        byte_order: Default byte order ("big", "little", "native").
        length_field: Header field name containing packet length (for variable-length packets).
        length_includes_header: Whether length field includes header size (default True).
        header_size: Header size in bytes.
        header_fields: List of header field definitions.
        sample_offset: Offset where samples begin.
        sample_count: Number of samples per packet.
        sample_format: Sample format definition.
        channel_extraction: Channel extraction configuration (optional).
        validation: Validation rules (optional).
        description: Human-readable description (optional).
    """

    name: str
    version: str
    packet_size: int | str
    byte_order: str
    length_field: str | None = None
    length_includes_header: bool = True
    header_size: int = 0
    header_fields: list[HeaderFieldDef] = field(default_factory=list)
    sample_offset: int = 0
    sample_count: int = 0
    sample_format: SampleFormatDef | None = None
    channel_extraction: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    description: str = ""

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> PacketFormatConfig:
        """Load packet format from YAML or JSON file.

        Automatically detects file format based on extension.

        Args:
            path: Path to configuration file (.yaml, .yml, or .json).

        Returns:
            PacketFormatConfig instance.

        Example:
            >>> config = PacketFormatConfig.from_file("packet_format.yaml")
            >>> print(f"Loaded format: {config.name} v{config.version}")
        """
        path = Path(path)
        ext = path.suffix.lower()

        if ext in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif ext == ".json":
            return cls.from_json(path)
        else:
            # Try YAML by default
            logger.warning("Unknown file extension '%s', attempting YAML parsing", ext)
            return cls.from_yaml(path)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PacketFormatConfig:
        """Load packet format from dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            PacketFormatConfig instance.

        Raises:
            ConfigurationError: If configuration is invalid.

        Example:
            >>> config_dict = {
            ...     "name": "my_format",
            ...     "version": "1.0",
            ...     "packet": {"size": 1024, "byte_order": "big"},
            ...     "header": {"size": 16, "fields": []},
            ...     "samples": {"offset": 16, "count": 126, "format": {"size": 8, "type": "uint64"}}
            ... }
            >>> config = PacketFormatConfig.from_dict(config_dict)
        """
        # Validate required fields
        required = ["name", "version", "packet", "header", "samples"]
        missing = [key for key in required if key not in config]
        if missing:
            raise ConfigurationError(
                f"Missing required configuration keys: {', '.join(missing)}",
                fix_hint="Ensure configuration has all required sections.",
            )

        # Parse packet configuration
        packet_cfg = config["packet"]
        packet_size = packet_cfg.get("size", "variable")
        byte_order = packet_cfg.get("byte_order", "big")
        length_field = packet_cfg.get("length_field")
        length_includes_header = packet_cfg.get("length_includes_header", True)

        # Parse header configuration
        header_cfg = config["header"]
        header_size = header_cfg["size"]
        header_fields = []
        for field_cfg in header_cfg.get("fields", []):
            header_fields.append(
                HeaderFieldDef(
                    name=field_cfg["name"],
                    offset=field_cfg["offset"],
                    size=field_cfg["size"],
                    type=field_cfg["type"],
                    endian=field_cfg.get("endian", byte_order),
                    value=field_cfg.get("value"),
                    fields=field_cfg.get("fields"),
                    description=field_cfg.get("description", ""),
                )
            )

        # Parse samples configuration
        samples_cfg = config["samples"]
        sample_offset = samples_cfg["offset"]
        sample_count = samples_cfg["count"]
        sample_format = SampleFormatDef(
            size=samples_cfg["format"]["size"],
            type=samples_cfg["format"]["type"],
            endian=samples_cfg["format"].get("endian", "little"),
            description=samples_cfg["format"].get("description", ""),
        )

        # Optional configurations
        channel_extraction = samples_cfg.get("channel_extraction")
        validation = config.get("validation")

        return cls(
            name=config["name"],
            version=config["version"],
            packet_size=packet_size,
            byte_order=byte_order,
            length_field=length_field,
            length_includes_header=length_includes_header,
            header_size=header_size,
            header_fields=header_fields,
            sample_offset=sample_offset,
            sample_count=sample_count,
            sample_format=sample_format,
            channel_extraction=channel_extraction,
            validation=validation,
            description=config.get("description", ""),
        )

    @classmethod
    def from_yaml(cls, path: str | PathLike[str]) -> PacketFormatConfig:
        """Load packet format from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            PacketFormatConfig instance.

        Raises:
            LoaderError: If file cannot be read or configuration is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise LoaderError(
                "Configuration file not found",
                file_path=str(path),
            )

        try:
            with open(path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise LoaderError(
                "Failed to load configuration file",
                file_path=str(path),
                details=str(e),
            ) from e

        return cls.from_dict(config)

    @classmethod
    def from_json(cls, path: str | PathLike[str]) -> PacketFormatConfig:
        """Load packet format from JSON file.

        Args:
            path: Path to JSON configuration file.

        Returns:
            PacketFormatConfig instance.

        Raises:
            LoaderError: If file cannot be read or configuration is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise LoaderError(
                "Configuration file not found",
                file_path=str(path),
            )

        try:
            with open(path, encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            raise LoaderError(
                "Failed to load JSON configuration file",
                file_path=str(path),
                details=str(e),
            ) from e

        return cls.from_dict(config)


@dataclass
class DeviceConfig:
    """Device configuration mapping.

    Maps device IDs to names and parameters.

    Attributes:
        devices: Dictionary mapping device ID to device info.
        categories: Category definitions (optional).
        channels: Channel configuration (optional).
        unknown_policy: How to handle unknown devices ("error", "warn", "ignore").
    """

    devices: dict[int, dict[str, Any]]
    categories: dict[str, Any] = field(default_factory=dict)
    channels: dict[int, Any] = field(default_factory=dict)
    unknown_policy: str = "warn"

    @classmethod
    def from_yaml(cls, path: str | PathLike[str]) -> DeviceConfig:
        """Load device configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            DeviceConfig instance.

        Raises:
            LoaderError: If file cannot be read or configuration is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise LoaderError(
                "Device configuration file not found",
                file_path=str(path),
            )

        try:
            with open(path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            raise LoaderError(
                "Failed to load device configuration",
                file_path=str(path),
                details=str(e),
            ) from e

        # Parse device mappings
        devices = {}
        for dev_id_str, dev_info in config.get("devices", {}).items():
            # Convert hex or decimal string to int
            if isinstance(dev_id_str, str):
                dev_id = int(dev_id_str, 16 if dev_id_str.startswith("0x") else 10)
            else:
                dev_id = int(dev_id_str)
            devices[dev_id] = dev_info

        categories = config.get("categories", {})
        channels = config.get("channels", {})
        unknown_policy = config.get("unknown_device", {}).get("policy", "warn")

        return cls(
            devices=devices,
            categories=categories,
            channels=channels,
            unknown_policy=unknown_policy,
        )


class BitfieldExtractor:
    """Extract individual bits or bit ranges from integer values.

    Supports extracting single bits or bit ranges from multi-byte fields.

    Example:
        >>> extractor = BitfieldExtractor()
        >>> value = 0b1010_1100
        >>> extractor.extract_bit(value, 7)  # Most significant bit
        1
        >>> extractor.extract_bits(value, 4, 7)  # Upper nibble
        10
    """

    @staticmethod
    def extract_bit(value: int, bit: int) -> int:
        """Extract a single bit.

        Args:
            value: Integer value.
            bit: Bit position (0 = LSB).

        Returns:
            0 or 1.
        """
        return (value >> bit) & 1

    @staticmethod
    def extract_bits(value: int, start_bit: int, end_bit: int) -> int:
        """Extract a range of bits.

        Args:
            value: Integer value.
            start_bit: Starting bit position (inclusive).
            end_bit: Ending bit position (inclusive).

        Returns:
            Extracted value.
        """
        num_bits = end_bit - start_bit + 1
        mask = (1 << num_bits) - 1
        return (value >> start_bit) & mask


@dataclass
class PacketLoadResult:
    """Result of packet loading operation.

    Attributes:
        packets: List of loaded packets.
        packet_count: Number of packets loaded.
    """

    packets: list[dict[str, Any]]

    @property
    def packet_count(self) -> int:
        """Number of packets loaded."""
        return len(self.packets)


class ConfigurablePacketLoader:
    """Load binary packets using configuration-driven parsing.

    Parses binary files according to packet format configuration,
    extracting headers and sample data.

    Attributes:
        format_config: Packet format configuration.
        device_config: Device mapping configuration (optional).
    """

    def __init__(
        self,
        format_config: PacketFormatConfig,
        device_config: DeviceConfig | None = None,
    ) -> None:
        """Initialize configurable packet loader.

        Args:
            format_config: Packet format configuration.
            device_config: Device mapping configuration (optional).
        """
        self.format_config = format_config
        self.device_config = device_config
        self.bitfield_extractor = BitfieldExtractor()

    def load_packets(self, path: str | PathLike[str]) -> list[dict[str, Any]]:
        """Load and parse all packets from binary file.

        Args:
            path: Path to binary file.

        Returns:
            List of parsed packet dictionaries.

        Raises:
            LoaderError: If file cannot be read.
        """
        path = Path(path)
        if not path.exists():
            raise LoaderError(
                "Binary file not found",
                file_path=str(path),
            )

        packets = []
        for packet in self.load_packets_streaming(path):
            packets.append(packet)

        logger.info("Loaded %d packets from %s", len(packets), path)
        return packets

    def load_packets_streaming(
        self, path: str | PathLike[str], chunk_size: int = 1000
    ) -> Iterator[dict[str, Any]]:
        """Stream packets from binary file.

        Args:
            path: Path to binary file.
            chunk_size: Number of packets to buffer (for progress tracking).

        Yields:
            Parsed packet dictionaries.

        Raises:
            ConfigurationError: If packet configuration is invalid.
            LoaderError: If file cannot be read or packet limit exceeded.
            FormatError: If packet parsing fails.
        """
        MAX_PACKET_LIMIT = 10_000_000  # Prevent unbounded memory growth

        path = Path(path)

        # Validate and determine packet size mode
        is_variable_length, fixed_packet_size = self._determine_packet_mode()

        try:
            with open(path, "rb") as f:
                packet_index = 0
                while True:
                    # Check packet limit to prevent memory exhaustion
                    if packet_index >= MAX_PACKET_LIMIT:
                        raise LoaderError(
                            f"Exceeded maximum packet limit ({MAX_PACKET_LIMIT})",
                            file_path=str(path),
                            fix_hint="Process file in chunks or increase MAX_PACKET_LIMIT if needed",
                        )

                    # Read packet data
                    packet_data = self._read_next_packet(
                        f, packet_index, is_variable_length, fixed_packet_size
                    )
                    if packet_data is None:
                        break

                    # Parse and yield packet
                    try:
                        packet = self._parse_packet(packet_data, packet_index)
                        yield packet
                        packet_index += 1
                    except FormatError:
                        logger.exception("Failed to parse packet %d", packet_index)
                        raise

        except OSError as e:
            raise LoaderError(
                "Failed to read binary file",
                file_path=str(path),
                details=str(e),
            ) from e

    def _determine_packet_mode(self) -> tuple[bool, int | None]:
        """Determine if packets are variable-length and get fixed size if applicable.

        Returns:
            Tuple of (is_variable_length, fixed_packet_size).

        Raises:
            ConfigurationError: If variable-length config is invalid.
        """
        is_variable_length = (
            isinstance(self.format_config.packet_size, str)
            and self.format_config.packet_size == "variable"
        )

        if is_variable_length:
            # Validate configuration for variable-length packets
            if not self.format_config.length_field:
                raise ConfigurationError(
                    "Variable-length packets require 'length_field' in packet configuration",
                    config_key="packet.length_field",
                    fix_hint="Specify which header field contains the packet length",
                )
            return True, None

        # Determine fixed packet size
        if isinstance(self.format_config.packet_size, str):
            fixed_packet_size = int(self.format_config.packet_size)
        else:
            fixed_packet_size = self.format_config.packet_size

        return False, fixed_packet_size

    def _read_next_packet(
        self,
        file_handle: Any,
        packet_index: int,
        is_variable_length: bool,
        fixed_packet_size: int | None,
    ) -> bytes | None:
        """Read next packet from file.

        Args:
            file_handle: Open file handle.
            packet_index: Current packet index.
            is_variable_length: Whether packets are variable-length.
            fixed_packet_size: Fixed packet size (if not variable).

        Returns:
            Packet bytes or None if end of file.

        Raises:
            FormatError: If packet format is invalid.
        """
        if is_variable_length:
            return self._read_variable_length_packet(file_handle, packet_index)
        else:
            assert fixed_packet_size is not None
            return self._read_fixed_length_packet(file_handle, packet_index, fixed_packet_size)

    def _read_variable_length_packet(self, file_handle: Any, packet_index: int) -> bytes | None:
        """Read a variable-length packet.

        Args:
            file_handle: Open file handle.
            packet_index: Current packet index.

        Returns:
            Packet bytes or None if end of file.

        Raises:
            FormatError: If packet format is invalid.
        """
        # Read header first to determine packet size
        header_data = file_handle.read(self.format_config.header_size)
        if not header_data:
            return None

        if len(header_data) < self.format_config.header_size:
            logger.warning(
                "Incomplete header at end of file (packet %d): got %d bytes, expected %d",
                packet_index,
                len(header_data),
                self.format_config.header_size,
            )
            return None

        # Parse header to get length field
        header_dict = self._parse_header_fields(header_data)

        # Get packet length from header
        packet_length = self._extract_packet_length(header_dict, packet_index)

        # Calculate payload size
        payload_size = self._calculate_payload_size(packet_length)

        # Read remaining packet data
        payload_data = file_handle.read(payload_size)
        if len(payload_data) < payload_size:
            logger.warning(
                "Incomplete payload at end of file (packet %d): got %d bytes, expected %d",
                packet_index,
                len(payload_data),
                payload_size,
            )
            return None

        # Combine header and payload
        combined: bytes = header_data + payload_data
        return combined

    def _read_fixed_length_packet(
        self, file_handle: Any, packet_index: int, packet_size: int
    ) -> bytes | None:
        """Read a fixed-length packet.

        Args:
            file_handle: Open file handle.
            packet_index: Current packet index.
            packet_size: Expected packet size.

        Returns:
            Packet bytes or None if end of file.
        """
        packet_data = file_handle.read(packet_size)
        if not packet_data:
            return None

        if len(packet_data) < packet_size:
            logger.warning(
                "Incomplete packet at end of file (packet %d): got %d bytes, expected %d",
                packet_index,
                len(packet_data),
                packet_size,
            )
            return None

        result: bytes = bytes(packet_data)
        return result

    def _parse_header_fields(self, header_data: bytes) -> dict[str, Any]:
        """Parse all header fields from header data.

        Args:
            header_data: Raw header bytes.

        Returns:
            Dictionary of field name to value.
        """
        header_dict = {}
        for field_def in self.format_config.header_fields:
            value = self._extract_field(header_data, field_def)
            header_dict[field_def.name] = value
        return header_dict

    def _extract_packet_length(self, header_dict: dict[str, Any], packet_index: int) -> int:
        """Extract packet length from parsed header.

        Args:
            header_dict: Parsed header fields.
            packet_index: Current packet index.

        Returns:
            Packet length in bytes.

        Raises:
            FormatError: If length field not found.
        """
        if self.format_config.length_field not in header_dict:
            raise FormatError(
                f"Length field '{self.format_config.length_field}' not found in header (packet {packet_index})"
            )
        length: int = int(header_dict[self.format_config.length_field])
        return length

    def _calculate_payload_size(self, packet_length: int) -> int:
        """Calculate payload size from packet length.

        Args:
            packet_length: Total packet length from header.

        Returns:
            Payload size in bytes.
        """
        if self.format_config.length_includes_header:
            return packet_length - self.format_config.header_size
        else:
            return packet_length

    def _parse_packet(self, packet_data: bytes, packet_index: int) -> dict[str, Any]:
        """Parse a single packet.

        Args:
            packet_data: Raw packet bytes.
            packet_index: Packet index in file.

        Returns:
            Parsed packet dictionary with header and samples.
        """
        packet: dict[str, Any] = {
            "index": packet_index,
            "header": {},
            "samples": [],
        }

        # Parse header fields
        for field_def in self.format_config.header_fields:
            value = self._extract_field(packet_data, field_def)
            packet["header"][field_def.name] = value

        # Parse samples
        assert self.format_config.sample_format is not None
        sample_offset = self.format_config.sample_offset
        sample_count = self.format_config.sample_count
        sample_size = self.format_config.sample_format.size

        for i in range(sample_count):
            offset = sample_offset + (i * sample_size)
            if offset + sample_size > len(packet_data):
                logger.warning("Sample %d exceeds packet bounds (packet %d)", i, packet_index)
                break

            sample_bytes = packet_data[offset : offset + sample_size]
            sample_value = self._parse_sample(sample_bytes)
            packet["samples"].append(sample_value)

        return packet

    def _extract_field(self, packet_data: bytes, field_def: HeaderFieldDef) -> Any:
        """Extract a header field value.

        Args:
            packet_data: Raw packet bytes.
            field_def: Field definition.

        Returns:
            Extracted field value.

        Raises:
            ConfigurationError: If field type is unsupported.
            FormatError: If field cannot be extracted.
        """
        offset = field_def.offset
        size = field_def.size

        if offset + size > len(packet_data):
            raise FormatError(
                f"Field '{field_def.name}' exceeds packet bounds",
                expected=f"{offset + size} bytes",
                got=f"{len(packet_data)} bytes",
            )

        field_bytes = packet_data[offset : offset + size]

        # Handle different field types
        if field_def.type == "bytes":
            return field_bytes
        elif field_def.type == "bitfield":
            # Parse as integer first, then extract bitfields
            value = self._bytes_to_int(field_bytes, field_def.endian, signed=False)
            if field_def.fields:
                bitfields = {}
                for bf_name, bf_def in field_def.fields.items():
                    if "bit" in bf_def:
                        bitfields[bf_name] = self.bitfield_extractor.extract_bit(
                            value, bf_def["bit"]
                        )
                    elif "bits" in bf_def:
                        bit_range = bf_def["bits"]
                        bitfields[bf_name] = self.bitfield_extractor.extract_bits(
                            value, bit_range[0], bit_range[1]
                        )
                return bitfields
            return value
        elif field_def.type.startswith("uint"):
            return self._bytes_to_int(field_bytes, field_def.endian, signed=False)
        elif field_def.type.startswith("int"):
            return self._bytes_to_int(field_bytes, field_def.endian, signed=True)
        elif field_def.type == "float32":
            endian_char = "<" if field_def.endian == "little" else ">"
            return struct.unpack(f"{endian_char}f", field_bytes)[0]
        elif field_def.type == "float64":
            endian_char = "<" if field_def.endian == "little" else ">"
            return struct.unpack(f"{endian_char}d", field_bytes)[0]
        else:
            raise ConfigurationError(
                f"Unsupported field type: {field_def.type}",
                config_key=f"{field_def.name}.type",
            )

    def _bytes_to_int(self, data: bytes, endian: str, signed: bool) -> int:
        """Convert bytes to integer with specified endianness.

        Args:
            data: Byte data.
            endian: Byte order ("big", "little", or "native").
            signed: Whether to interpret as signed integer.

        Returns:
            Integer value.
        """
        from typing import Literal

        byte_order_str = endian if endian != "native" else "little"
        # Type assertion for mypy - we validate endian in __post_init__
        byte_order: Literal["little", "big"] = byte_order_str  # type: ignore[assignment]
        return int.from_bytes(data, byteorder=byte_order, signed=signed)

    def _parse_sample(self, sample_bytes: bytes) -> int:
        """Parse a sample value.

        Args:
            sample_bytes: Raw sample bytes.

        Returns:
            Sample value as integer.
        """
        assert self.format_config.sample_format is not None
        return self._bytes_to_int(
            sample_bytes, self.format_config.sample_format.endian, signed=False
        )

    def load(self, path: str | PathLike[str]) -> PacketLoadResult:
        """Load packets and return result object (test-compatible API).

        Args:
            path: Path to binary file.

        Returns:
            PacketLoadResult with loaded packets.
        """
        packets = self.load_packets(path)
        return PacketLoadResult(packets=packets)

    def stream(
        self, path: str | PathLike[str], chunk_size: int = 1000
    ) -> Iterator[PacketLoadResult]:
        """Stream packets in chunks (test-compatible API).

        Args:
            path: Path to binary file.
            chunk_size: Number of packets per chunk.

        Yields:
            PacketLoadResult objects with packet chunks.
        """
        chunk = []
        for packet in self.load_packets_streaming(path, chunk_size):
            chunk.append(packet)
            if len(chunk) >= chunk_size:
                yield PacketLoadResult(packets=chunk)
                chunk = []

        # Yield remaining packets
        if chunk:
            yield PacketLoadResult(packets=chunk)


class DeviceMapper:
    """Map device IDs to names and metadata.

    Provides human-readable names and configuration for devices
    identified in packet headers.

    Attributes:
        config: Device configuration.
    """

    def __init__(self, config: DeviceConfig) -> None:
        """Initialize device mapper.

        Args:
            config: Device configuration.
        """
        self.config = config

    @classmethod
    def from_file(cls, path: str | PathLike[str]) -> DeviceMapper:
        """Create DeviceMapper from configuration file.

        Args:
            path: Path to device configuration file.

        Returns:
            DeviceMapper instance.

        Example:
            >>> mapper = DeviceMapper.from_file("device_mapping.yaml")
            >>> device_name = mapper.get_device_name(0x2B)
        """
        config = DeviceConfig.from_yaml(path)
        return cls(config)

    def get_device(self, device_id: int) -> DeviceInfo | None:
        """Get device information object.

        Args:
            device_id: Device ID from packet header.

        Returns:
            DeviceInfo object or None if device not found.

        Raises:
            ConfigurationError: If device ID is unknown and unknown_policy is 'error'.

        Example:
            >>> device = mapper.get_device(0x2B)
            >>> if device:
            ...     print(f"{device.name}: {device.sample_rate} Hz")
        """
        if device_id in self.config.devices:
            return DeviceInfo.from_dict(self.config.devices[device_id])

        # Handle unknown device
        if self.config.unknown_policy == "error":
            raise ConfigurationError(
                f"Unknown device ID: 0x{device_id:02X}",
                fix_hint="Add device to device_mapping configuration or set unknown_policy to 'warn' or 'ignore'.",
            )
        elif self.config.unknown_policy == "warn":
            logger.warning("Unknown device ID: 0x%02X", device_id)

        return None

    def resolve_name(self, device_id: int) -> str:
        """Resolve device ID to human-readable name.

        Args:
            device_id: Device ID from packet header.

        Returns:
            Device name or "Unknown Device 0xXX".

        Example:
            >>> name = mapper.resolve_name(0x2B)
            >>> print(f"Device: {name}")
        """
        device = self.get_device(device_id)
        if device:
            return device.name
        return f"Unknown Device 0x{device_id:02X}"

    def get_device_name(self, device_id: int) -> str:
        """Get device name from ID.

        Args:
            device_id: Device ID from packet header.

        Returns:
            Device name or "Unknown Device 0xXX".
        """
        return self.resolve_name(device_id)

    def get_device_info(self, device_id: int) -> dict[str, Any]:
        """Get full device information as dictionary.

        Args:
            device_id: Device ID from packet header.

        Returns:
            Device information dictionary.
        """
        device = self.get_device(device_id)
        if device:
            return {
                "name": device.name,
                "short_name": device.short_name,
                "description": device.description,
                "category": device.category,
                "sample_rate": device.sample_rate,
                "channels": device.channels,
                "properties": device.properties,
            }

        return {
            "name": f"Unknown Device 0x{device_id:02X}",
            "category": "unknown",
        }


def load_binary_packets(
    path: str | PathLike[str],
    format_config: str | PathLike[str] | PacketFormatConfig,
    device_config: str | PathLike[str] | DeviceConfig | None = None,
) -> list[dict[str, Any]]:
    """Load binary packets from file using configuration.

    Main entry point for loading binary packet data.

    Args:
        path: Path to binary file.
        format_config: Packet format configuration (path or object).
        device_config: Device mapping configuration (path or object, optional).

    Returns:
        List of parsed packet dictionaries.

    Example:
        >>> packets = load_binary_packets(
        ...     "capture.bin",
        ...     "packet_format.yaml",
        ...     "device_mapping.yaml"
        ... )
        >>> print(f"Loaded {len(packets)} packets")
        >>> print(f"First packet device: {packets[0]['header']['device_id']}")
    """
    # Load configurations if paths provided
    fmt_cfg: PacketFormatConfig
    if isinstance(format_config, PacketFormatConfig):
        fmt_cfg = format_config
    else:
        fmt_cfg = PacketFormatConfig.from_yaml(format_config)

    dev_cfg: DeviceConfig | None = None
    if device_config is not None and isinstance(device_config, str | Path):
        dev_cfg = DeviceConfig.from_yaml(device_config)
    elif isinstance(device_config, DeviceConfig):
        dev_cfg = device_config

    # Create loader and load packets
    loader = ConfigurablePacketLoader(fmt_cfg, dev_cfg)
    return loader.load_packets(path)


def load_packets_streaming(
    path: str | PathLike[str],
    format_config: str | PathLike[str] | PacketFormatConfig,
    device_config: str | PathLike[str] | DeviceConfig | None = None,
    chunk_size: int = 1000,
) -> Iterator[dict[str, Any]]:
    """Stream binary packets from file using configuration.

    Memory-efficient streaming loader for large files.

    Args:
        path: Path to binary file.
        format_config: Packet format configuration (path or object).
        device_config: Device mapping configuration (path or object, optional).
        chunk_size: Number of packets to buffer.

    Yields:
        Parsed packet dictionaries.

    Example:
        >>> for packet in load_packets_streaming("large_capture.bin", "format.yaml"):
        ...     process_packet(packet)
    """
    # Load configurations if paths provided
    fmt_cfg: PacketFormatConfig
    if isinstance(format_config, PacketFormatConfig):
        fmt_cfg = format_config
    else:
        fmt_cfg = PacketFormatConfig.from_yaml(format_config)

    dev_cfg: DeviceConfig | None = None
    if device_config is not None and isinstance(device_config, str | Path):
        dev_cfg = DeviceConfig.from_yaml(device_config)
    elif isinstance(device_config, DeviceConfig):
        dev_cfg = device_config

    # Create loader and stream packets
    loader = ConfigurablePacketLoader(fmt_cfg, dev_cfg)
    yield from loader.load_packets_streaming(path, chunk_size=chunk_size)


def detect_source_type(path: str | PathLike[str]) -> str:
    """Detect binary data source type from file extension or content.

    Args:
        path: Path to file.

    Returns:
        Source type ("raw", "pcap", "sigrok", "vcd", "unknown").

    Example:
        >>> source_type = detect_source_type("capture.bin")
        >>> print(f"Detected: {source_type}")
        Detected: raw
    """
    path = Path(path)
    ext = path.suffix.lower()

    # Extension-based detection
    if ext in (".bin", ".dat", ".raw"):
        return "raw"
    elif ext in (".pcap", ".pcapng"):
        return "pcap"
    elif ext == ".sr":
        return "sigrok"
    elif ext == ".vcd":
        return "vcd"

    # Content-based detection for unknown extensions
    try:
        with open(path, "rb") as f:
            magic = f.read(8)

        # PCAP magic bytes
        if magic[:4] in (b"\xa1\xb2\xc3\xd4", b"\xd4\xc3\xb2\xa1"):
            return "pcap"

        # VCD starts with "$" commands
        if magic.startswith(b"$"):
            return "vcd"

    except Exception:
        pass

    return "unknown"


def extract_channels(
    packets: list[dict[str, Any]],
    channel_map: dict[str, dict[str, Any]],
    sample_rate: float | None = None,
) -> dict[str, DigitalTrace]:
    """Extract individual channels from packet samples.

    Extracts bit ranges from multi-bit samples to create individual
    channel traces.

    Args:
        packets: List of parsed packets.
        channel_map: Channel definitions with bit ranges.
        sample_rate: Sample rate in Hz. If None, defaults to 100 MHz
            (typical for high-speed digital). For accurate analysis,
            provide the actual sample rate from your acquisition system.

    Returns:
        Dictionary mapping channel names to DigitalTrace objects.

    Raises:
        ConfigurationError: If channel map is invalid.

    Example:
        >>> channel_map = {
        ...     "ch0": {"bits": [0, 7]},
        ...     "ch1": {"bits": [8, 15]},
        ... }
        >>> traces = extract_channels(packets, channel_map, sample_rate=1e9)
        >>> print(f"Channel 0: {len(traces['ch0'].data)} samples")
    """
    if not packets:
        raise ConfigurationError(
            "No packets to extract channels from",
            fix_hint="Ensure packets were loaded successfully.",
        )

    extractor = BitfieldExtractor()
    channels: dict[str, list[int]] = {name: [] for name in channel_map}

    # Extract samples for each channel
    for packet in packets:
        for sample in packet["samples"]:
            for ch_name, ch_def in channel_map.items():
                if "bits" in ch_def:
                    bit_range = ch_def["bits"]
                    value = extractor.extract_bits(sample, bit_range[0], bit_range[1])
                    channels[ch_name].append(value)
                elif "bit" in ch_def:
                    value = extractor.extract_bit(sample, ch_def["bit"])
                    channels[ch_name].append(value)

    # Use provided sample rate or default to 100 MHz (typical for high-speed digital)
    effective_sample_rate = sample_rate if sample_rate is not None else 100e6

    # Convert to DigitalTrace objects
    traces = {}
    for ch_name, samples in channels.items():
        # Convert to boolean array (0/1 -> False/True)
        data = np.array(samples, dtype=np.uint8).astype(np.bool_)

        # Create metadata with configurable sample rate
        metadata = TraceMetadata(
            sample_rate=effective_sample_rate,
            channel_name=ch_name,
        )

        traces[ch_name] = DigitalTrace(data=data, metadata=metadata)

    return traces


__all__ = [
    "BitfieldDef",
    "BitfieldExtractor",
    "ConfigurablePacketLoader",
    "DeviceConfig",
    "DeviceInfo",
    "DeviceMapper",
    "HeaderFieldDef",
    "PacketFormatConfig",
    "ParsedPacket",
    "SampleFormatDef",
    "detect_source_type",
    "extract_channels",
    "load_binary_packets",
    "load_packets_streaming",
]
