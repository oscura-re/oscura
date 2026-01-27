"""BLE (Bluetooth Low Energy) protocol analyzer with GATT service discovery.

This module provides comprehensive BLE packet analysis including:
- Advertising packet parsing (ADV_IND, SCAN_RSP, etc.)
- ATT protocol operation decoding (Read, Write, Notify, etc.)
- GATT service/characteristic/descriptor discovery
- Standard and custom UUID mapping
- Export to JSON/CSV formats

Example:
    >>> analyzer = BLEAnalyzer()
    >>> packet = BLEPacket(
    ...     timestamp=0.0,
    ...     packet_type="ADV_IND",
    ...     source_address="AA:BB:CC:DD:EE:FF",
    ...     data=adv_data,
    ... )
    >>> analyzer.add_packet(packet)
    >>> services = analyzer.discover_services()
    >>> analyzer.export_services(Path("services.json"))

References:
    Bluetooth Core Specification v5.4: https://www.bluetooth.com/specifications/specs/
    GATT Specification Supplement: https://www.bluetooth.com/specifications/specs/
"""

from __future__ import annotations

import csv
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oscura.analyzers.protocols.ble.uuids import (
    AD_TYPES,
    get_characteristic_name,
    get_descriptor_name,
    get_service_name,
    uuid_to_string,
)

# BLE Link Layer packet types (PDU Type field in advertising channel)
BLE_PACKET_TYPES: dict[int, str] = {
    0x00: "ADV_IND",  # Connectable undirected advertising
    0x01: "ADV_DIRECT_IND",  # Connectable directed advertising
    0x02: "ADV_NONCONN_IND",  # Non-connectable undirected advertising
    0x03: "SCAN_REQ",  # Scan request
    0x04: "SCAN_RSP",  # Scan response
    0x05: "CONNECT_REQ",  # Connection request
    0x06: "ADV_SCAN_IND",  # Scannable undirected advertising
}

# ATT Protocol Opcodes
ATT_OPCODES: dict[int, str] = {
    0x01: "Error Response",
    0x02: "Exchange MTU Request",
    0x03: "Exchange MTU Response",
    0x04: "Find Information Request",
    0x05: "Find Information Response",
    0x06: "Find By Type Value Request",
    0x07: "Find By Type Value Response",
    0x08: "Read By Type Request",
    0x09: "Read By Type Response",
    0x0A: "Read Request",
    0x0B: "Read Response",
    0x0C: "Read Blob Request",
    0x0D: "Read Blob Response",
    0x0E: "Read Multiple Request",
    0x0F: "Read Multiple Response",
    0x10: "Read By Group Type Request",
    0x11: "Read By Group Type Response",
    0x12: "Write Request",
    0x13: "Write Response",
    0x16: "Prepare Write Request",
    0x17: "Prepare Write Response",
    0x18: "Execute Write Request",
    0x19: "Execute Write Response",
    0x1B: "Handle Value Notification",
    0x1D: "Handle Value Indication",
    0x1E: "Handle Value Confirmation",
    0x52: "Write Command",
    0xD2: "Signed Write Command",
}

# GATT Characteristic properties (bit mask)
GATT_CHAR_PROPERTIES: dict[int, str] = {
    0x01: "broadcast",
    0x02: "read",
    0x04: "write_no_response",
    0x08: "write",
    0x10: "notify",
    0x20: "indicate",
    0x40: "authenticated_signed_writes",
    0x80: "extended_properties",
}


@dataclass
class GATTDescriptor:
    """GATT descriptor definition.

    Attributes:
        uuid: Descriptor UUID (16-bit or 128-bit).
        name: Human-readable name.
        handle: Attribute handle.
        value: Descriptor value (optional).
    """

    uuid: str
    name: str
    handle: int
    value: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "uuid": self.uuid,
            "name": self.name,
            "handle": self.handle,
            "value": self.value.hex() if self.value else None,
        }


@dataclass
class GATTCharacteristic:
    """GATT characteristic definition.

    Attributes:
        uuid: Characteristic UUID.
        name: Human-readable name.
        properties: List of properties (read, write, notify, etc.).
        handle: Attribute handle.
        value_handle: Value handle (for read/write operations).
        value: Characteristic value (optional).
        descriptors: List of descriptors.
    """

    uuid: str
    name: str
    properties: list[str]
    handle: int
    value_handle: int | None = None
    value: bytes | None = None
    descriptors: list[GATTDescriptor] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "uuid": self.uuid,
            "name": self.name,
            "properties": self.properties,
            "handle": self.handle,
            "value_handle": self.value_handle,
            "value": self.value.hex() if self.value else None,
            "descriptors": [d.to_dict() for d in self.descriptors],
        }


@dataclass
class GATTService:
    """GATT service definition.

    Attributes:
        uuid: Service UUID.
        name: Human-readable name.
        characteristics: List of characteristics.
        handle_range: (start_handle, end_handle) range.
    """

    uuid: str
    name: str
    characteristics: list[GATTCharacteristic]
    handle_range: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "uuid": self.uuid,
            "name": self.name,
            "handle_range": list(self.handle_range),
            "characteristics": [c.to_dict() for c in self.characteristics],
        }


@dataclass
class BLEPacket:
    """Represents a BLE packet.

    Attributes:
        timestamp: Packet timestamp in seconds.
        packet_type: Packet type (e.g., "ADV_IND", "ATT_READ_REQ").
        source_address: Source MAC address (AA:BB:CC:DD:EE:FF).
        dest_address: Destination MAC address (optional).
        rssi: Received Signal Strength Indicator in dBm (optional).
        data: Raw packet data.
        decoded: Decoded packet contents (optional).
    """

    timestamp: float
    packet_type: str
    source_address: str
    data: bytes
    dest_address: str | None = None
    rssi: int | None = None
    decoded: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "timestamp": self.timestamp,
            "packet_type": self.packet_type,
            "source_address": self.source_address,
            "dest_address": self.dest_address,
            "rssi": self.rssi,
            "data": self.data.hex(),
            "decoded": self.decoded,
        }


def _decode_error_response(data: bytes, result: dict[str, Any]) -> None:
    """Decode ATT Error Response packet.

    Args:
        data: ATT packet data.
        result: Result dictionary to populate.
    """
    if len(data) >= 5:
        result["request_opcode"] = f"0x{data[1]:02X}"
        result["handle"] = int.from_bytes(data[2:4], "little")
        result["error_code"] = f"0x{data[4]:02X}"


def _decode_mtu_operation(data: bytes, result: dict[str, Any]) -> None:
    """Decode ATT MTU Request/Response.

    Args:
        data: ATT packet data.
        result: Result dictionary to populate.
    """
    if len(data) >= 3:
        result["mtu"] = int.from_bytes(data[1:3], "little")


def _decode_read_request(data: bytes, result: dict[str, Any]) -> None:
    """Decode ATT Read Request/Blob Request.

    Args:
        data: ATT packet data.
        result: Result dictionary to populate.
    """
    if len(data) >= 3:
        result["handle"] = int.from_bytes(data[1:3], "little")


def _decode_read_response(data: bytes, result: dict[str, Any]) -> None:
    """Decode ATT Read Response.

    Args:
        data: ATT packet data.
        result: Result dictionary to populate.
    """
    result["value"] = data[1:].hex()


def _decode_read_by_type_request(data: bytes, result: dict[str, Any]) -> None:
    """Decode ATT Read By Type/Group Type Request.

    Args:
        data: ATT packet data.
        result: Result dictionary to populate.
    """
    if len(data) >= 7:
        result["start_handle"] = int.from_bytes(data[1:3], "little")
        result["end_handle"] = int.from_bytes(data[3:5], "little")
        uuid_data = data[5:]
        result["uuid"] = uuid_to_string(uuid_data)


def _decode_read_by_type_response(data: bytes, result: dict[str, Any]) -> None:
    """Decode ATT Read By Type/Group Type Response.

    Args:
        data: ATT packet data.
        result: Result dictionary to populate.
    """
    if len(data) >= 2:
        length = data[1]
        result["attribute_length"] = length

        attributes = []
        i = 2
        while i + length <= len(data):
            attr_data = data[i : i + length]
            if len(attr_data) >= 2:
                handle = int.from_bytes(attr_data[0:2], "little")
                attributes.append({"handle": handle, "data": attr_data[2:].hex()})
            i += length
        result["attributes"] = attributes


def _decode_write_operation(data: bytes, result: dict[str, Any]) -> None:
    """Decode ATT Write operations (Request/Notification/Indication/Command).

    Args:
        data: ATT packet data.
        result: Result dictionary to populate.
    """
    if len(data) >= 3:
        result["handle"] = int.from_bytes(data[1:3], "little")
        result["value"] = data[3:].hex()


class BLEAnalyzer:
    """BLE protocol analyzer with GATT service discovery.

    This analyzer processes BLE packets to extract advertising data,
    decode ATT operations, and discover GATT services/characteristics.

    Example:
        >>> analyzer = BLEAnalyzer()
        >>> analyzer.register_custom_uuid("0xABCD", "My Custom Service")
        >>> analyzer.add_packet(packet)
        >>> services = analyzer.discover_services()
        >>> print(f"Found {len(services)} services")
    """

    def __init__(self) -> None:
        """Initialize BLE analyzer."""
        self.packets: list[BLEPacket] = []
        self.services: list[GATTService] = []
        self.custom_uuids: dict[str, str] = {}
        self._service_cache: dict[int, GATTService] = {}
        self._char_cache: dict[int, GATTCharacteristic] = {}

    def add_packet(self, packet: BLEPacket) -> None:
        """Add BLE packet for analysis.

        Args:
            packet: BLE packet to add.

        Example:
            >>> packet = BLEPacket(
            ...     timestamp=0.0,
            ...     packet_type="ADV_IND",
            ...     source_address="AA:BB:CC:DD:EE:FF",
            ...     data=b"\\x02\\x01\\x06\\x09\\x09MyDevice",
            ... )
            >>> analyzer.add_packet(packet)
        """
        # Decode packet based on type
        if packet.packet_type.startswith("ADV_") or packet.packet_type == "SCAN_RSP":
            packet.decoded = self.parse_advertising_data(packet.data)
        elif packet.packet_type.startswith("ATT_"):
            packet.decoded = self.decode_att_operation(packet.data)

        self.packets.append(packet)

    def parse_advertising_data(self, data: bytes) -> dict[str, Any]:
        """Parse BLE advertising data (AD structures).

        Args:
            data: Advertising data payload.

        Returns:
            Dictionary of parsed AD structures.

        Example:
            >>> data = b"\\x02\\x01\\x06\\x09\\x09MyDevice"
            >>> result = analyzer.parse_advertising_data(data)
            >>> print(result["name"])
            'MyDevice'
        """
        result: dict[str, Any] = {}
        i = 0

        while i < len(data):
            if i >= len(data):
                break

            length = data[i]
            if length == 0 or i + length >= len(data):
                break

            ad_type = data[i + 1]
            ad_data = data[i + 2 : i + 1 + length]

            # Parse AD structure by type
            self._parse_ad_structure(ad_type, ad_data, result)

            i += 1 + length

        return result

    def _parse_ad_structure(self, ad_type: int, ad_data: bytes, result: dict[str, Any]) -> None:
        """Parse single AD structure into result dictionary.

        Args:
            ad_type: AD type code.
            ad_data: AD data bytes.
            result: Result dictionary to update.
        """
        if ad_type == 0x01:
            self._parse_flags(ad_data, result)
        elif ad_type in [0x08, 0x09]:
            result["name"] = ad_data.decode("utf-8", errors="ignore")
        elif ad_type == 0x0A:
            result["tx_power"] = struct.unpack("b", ad_data)[0]
        elif ad_type in [0x02, 0x03]:
            self._parse_service_uuids(ad_data, result)
        elif ad_type == 0x16:
            self._parse_service_data(ad_data, result)
        elif ad_type == 0x19:
            self._parse_appearance(ad_data, result)
        elif ad_type == 0xFF:
            self._parse_manufacturer_data(ad_data, result)
        else:
            ad_type_name = AD_TYPES.get(ad_type, f"Unknown Type 0x{ad_type:02X}")
            result[ad_type_name] = ad_data.hex()

    def _parse_flags(self, ad_data: bytes, result: dict[str, Any]) -> None:
        """Parse BLE flags AD structure.

        Args:
            ad_data: Flags data bytes.
            result: Result dictionary to update.
        """
        flags = int.from_bytes(ad_data, "little")
        result["flags"] = {
            "value": flags,
            "le_limited_discoverable": bool(flags & 0x01),
            "le_general_discoverable": bool(flags & 0x02),
            "br_edr_not_supported": bool(flags & 0x04),
            "le_br_edr_controller": bool(flags & 0x08),
            "le_br_edr_host": bool(flags & 0x10),
        }

    def _parse_service_uuids(self, ad_data: bytes, result: dict[str, Any]) -> None:
        """Parse 16-bit service UUIDs AD structure.

        Args:
            ad_data: UUID data bytes.
            result: Result dictionary to update.
        """
        uuids = []
        for j in range(0, len(ad_data), 2):
            uuid_val = int.from_bytes(ad_data[j : j + 2], "little")
            uuids.append(f"0x{uuid_val:04X}")
        result["service_uuids"] = uuids

    def _parse_service_data(self, ad_data: bytes, result: dict[str, Any]) -> None:
        """Parse service data AD structure.

        Args:
            ad_data: Service data bytes.
            result: Result dictionary to update.
        """
        if len(ad_data) >= 2:
            uuid_val = int.from_bytes(ad_data[0:2], "little")
            service_uuid = f"0x{uuid_val:04X}"
            result["service_data"] = {
                "uuid": service_uuid,
                "data": ad_data[2:].hex(),
            }

    def _parse_appearance(self, ad_data: bytes, result: dict[str, Any]) -> None:
        """Parse appearance AD structure.

        Args:
            ad_data: Appearance data bytes.
            result: Result dictionary to update.
        """
        if len(ad_data) >= 2:
            appearance = int.from_bytes(ad_data, "little")
            result["appearance"] = appearance

    def _parse_manufacturer_data(self, ad_data: bytes, result: dict[str, Any]) -> None:
        """Parse manufacturer data AD structure.

        Args:
            ad_data: Manufacturer data bytes.
            result: Result dictionary to update.
        """
        if len(ad_data) >= 2:
            company_id = int.from_bytes(ad_data[0:2], "little")
            result["manufacturer_data"] = {
                "company_id": f"0x{company_id:04X}",
                "data": ad_data[2:].hex(),
            }

    def decode_att_operation(self, data: bytes) -> dict[str, Any]:
        """Decode ATT protocol operation.

        Args:
            data: ATT packet payload.

        Returns:
            Dictionary of decoded operation details.

        Example:
            >>> data = b"\\x0A\\x03\\x00"  # Read Request, handle 0x0003
            >>> result = analyzer.decode_att_operation(data)
            >>> print(result["opcode_name"])
            'Read Request'
        """
        if len(data) < 1:
            return {"error": "Packet too short"}

        opcode = data[0]
        opcode_name = ATT_OPCODES.get(opcode, f"Unknown Opcode 0x{opcode:02X}")
        result: dict[str, Any] = {
            "opcode": f"0x{opcode:02X}",
            "opcode_name": opcode_name,
        }

        try:
            # Decode based on opcode category
            if opcode == 0x01:
                _decode_error_response(data, result)
            elif opcode in [0x02, 0x03]:
                _decode_mtu_operation(data, result)
            elif opcode in [0x0A, 0x0C]:
                _decode_read_request(data, result)
            elif opcode == 0x0B:
                _decode_read_response(data, result)
            elif opcode in [0x08, 0x10]:
                _decode_read_by_type_request(data, result)
            elif opcode in [0x09, 0x11]:
                _decode_read_by_type_response(data, result)
            elif opcode in [0x12, 0x1B, 0x1D, 0x52]:
                _decode_write_operation(data, result)

        except (struct.error, IndexError) as e:
            result["parse_error"] = str(e)

        return result

    def discover_services(self) -> list[GATTService]:
        """Discover GATT services from captured ATT packets.

        Analyzes Read By Group Type responses to build service hierarchy.

        Returns:
            List of discovered GATT services.

        Example:
            >>> services = analyzer.discover_services()
            >>> for service in services:
            ...     print(f"{service.name}: {service.uuid}")
        """
        self.services.clear()
        self._service_cache.clear()
        self._char_cache.clear()

        # Find service discovery responses (Read By Group Type Response, UUID 0x2800)
        for packet in self.packets:
            if not packet.decoded:
                continue

            opcode_name = packet.decoded.get("opcode_name", "")

            # Service discovery (Primary Service = 0x2800)
            if opcode_name == "Read By Group Type Response":
                attributes = packet.decoded.get("attributes", [])
                for attr in attributes:
                    try:
                        handle = attr["handle"]
                        data_hex = attr["data"]
                        data = bytes.fromhex(data_hex)

                        if len(data) >= 4:
                            # Format: end_handle (2) + UUID (2 or 16)
                            end_handle = int.from_bytes(data[0:2], "little")
                            uuid_data = data[2:]
                            uuid = uuid_to_string(uuid_data)

                            # Get service name
                            service_name = self.get_uuid_name(uuid, "service")

                            service = GATTService(
                                uuid=uuid,
                                name=service_name,
                                characteristics=[],
                                handle_range=(handle, end_handle),
                            )
                            self.services.append(service)
                            self._service_cache[handle] = service
                    except (ValueError, KeyError):
                        continue

            # Characteristic discovery (Characteristic Declaration = 0x2803)
            elif opcode_name == "Read By Type Response":
                attributes = packet.decoded.get("attributes", [])
                for attr in attributes:
                    try:
                        handle = attr["handle"]
                        data_hex = attr["data"]
                        data = bytes.fromhex(data_hex)

                        if len(data) >= 5:
                            # Format: properties (1) + value_handle (2) + UUID (2 or 16)
                            properties_byte = data[0]
                            value_handle = int.from_bytes(data[1:3], "little")
                            uuid_data = data[3:]
                            uuid = uuid_to_string(uuid_data)

                            # Parse properties
                            properties = self._parse_properties(properties_byte)

                            # Get characteristic name
                            char_name = self.get_uuid_name(uuid, "characteristic")

                            char = GATTCharacteristic(
                                uuid=uuid,
                                name=char_name,
                                properties=properties,
                                handle=handle,
                                value_handle=value_handle,
                            )

                            # Find parent service
                            for service in self.services:
                                if service.handle_range[0] <= handle <= service.handle_range[1]:
                                    service.characteristics.append(char)
                                    self._char_cache[handle] = char
                                    break
                    except (ValueError, KeyError):
                        continue

        return self.services

    def _parse_properties(self, properties_byte: int) -> list[str]:
        """Parse GATT characteristic properties byte.

        Args:
            properties_byte: Properties bit mask.

        Returns:
            List of property names.
        """
        properties = []
        for bit, name in GATT_CHAR_PROPERTIES.items():
            if properties_byte & bit:
                properties.append(name)
        return properties

    def register_custom_uuid(self, uuid: str, name: str) -> None:
        """Register custom service/characteristic UUID.

        Args:
            uuid: UUID string (e.g., "0xABCD" or full format).
            name: Human-readable name.

        Example:
            >>> analyzer.register_custom_uuid("0xABCD", "My Custom Service")
        """
        self.custom_uuids[uuid.upper()] = name

    def get_uuid_name(self, uuid: str, uuid_type: str = "service") -> str:
        """Get name for UUID (checks custom mappings first).

        Args:
            uuid: UUID string.
            uuid_type: Type of UUID ("service", "characteristic", "descriptor").

        Returns:
            Human-readable name.
        """
        # Check custom mappings first
        if uuid.upper() in self.custom_uuids:
            return self.custom_uuids[uuid.upper()]

        # Check standard mappings
        if uuid_type == "service":
            return get_service_name(uuid)
        elif uuid_type == "characteristic":
            return get_characteristic_name(uuid)
        elif uuid_type == "descriptor":
            return get_descriptor_name(uuid)
        else:
            return f"Unknown {uuid_type}"

    def export_services(self, output_path: Path, format: str = "json") -> None:
        """Export discovered services to file.

        Args:
            output_path: Output file path.
            format: Export format ("json" or "csv").

        Raises:
            ValueError: If format is not supported.

        Example:
            >>> analyzer.export_services(Path("services.json"))
            >>> analyzer.export_services(Path("services.csv"), format="csv")
        """
        if format == "json":
            self._export_json(output_path)
        elif format == "csv":
            self._export_csv(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, output_path: Path) -> None:
        """Export services as JSON.

        Args:
            output_path: Output file path.
        """
        data = {
            "services": [service.to_dict() for service in self.services],
            "packet_count": len(self.packets),
        }
        with output_path.open("w") as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, output_path: Path) -> None:
        """Export services as CSV.

        Args:
            output_path: Output file path.
        """
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Service UUID",
                    "Service Name",
                    "Handle Range",
                    "Characteristic UUID",
                    "Characteristic Name",
                    "Properties",
                ]
            )

            for service in self.services:
                if not service.characteristics:
                    writer.writerow(
                        [
                            service.uuid,
                            service.name,
                            f"{service.handle_range[0]}-{service.handle_range[1]}",
                            "",
                            "",
                            "",
                        ]
                    )
                else:
                    for char in service.characteristics:
                        writer.writerow(
                            [
                                service.uuid,
                                service.name,
                                f"{service.handle_range[0]}-{service.handle_range[1]}",
                                char.uuid,
                                char.name,
                                ", ".join(char.properties),
                            ]
                        )

    def get_statistics(self) -> dict[str, Any]:
        """Get analysis statistics.

        Returns:
            Dictionary of statistics.

        Example:
            >>> stats = analyzer.get_statistics()
            >>> print(f"Total packets: {stats['total_packets']}")
        """
        packet_types: dict[str, int] = {}
        for packet in self.packets:
            packet_types[packet.packet_type] = packet_types.get(packet.packet_type, 0) + 1

        return {
            "total_packets": len(self.packets),
            "packet_types": packet_types,
            "services_discovered": len(self.services),
            "total_characteristics": sum(len(s.characteristics) for s in self.services),
        }


__all__ = [
    "ATT_OPCODES",
    "BLE_PACKET_TYPES",
    "GATT_CHAR_PROPERTIES",
    "BLEAnalyzer",
    "BLEPacket",
    "GATTCharacteristic",
    "GATTDescriptor",
    "GATTService",
]
