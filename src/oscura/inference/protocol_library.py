"""Protocol Format Library for common protocol definitions.

    - RE-DSL-003: Protocol Format Library

This module provides a library of common protocol definitions for industrial,
IoT, and communication protocols to enable immediate decoding without
user-defined definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, overload

from oscura.inference.protocol_dsl import (
    FieldDefinition,
    ProtocolDecoder,
    ProtocolDefinition,
)


@dataclass
class ProtocolInfo:
    """Information about a protocol in the library.

    Attributes:
        name: Protocol name.
        category: Protocol category.
        version: Protocol version.
        description: Human-readable description.
        reference: Reference documentation URL.
        definition: Protocol definition for decoding.
    """

    name: str
    category: Literal["industrial", "iot", "network", "automotive", "building", "serial", "custom"]
    version: str
    description: str
    reference: str = ""
    definition: ProtocolDefinition | None = None


class ProtocolLibrary:
    """Library of common protocol definitions.

    Implements RE-DSL-003: Protocol Format Library.

    Provides pre-defined protocol definitions for common industrial,
    IoT, and communication protocols.

    Example:
        >>> library = ProtocolLibrary()
        >>> modbus = library.get("modbus_rtu")
        >>> decoder = library.get_decoder("modbus_rtu")
        >>> message = decoder.decode(data)
    """

    def __init__(self) -> None:
        """Initialize protocol library with built-in protocols."""
        self._protocols: dict[str, ProtocolInfo] = {}
        self._decoders: dict[str, ProtocolDecoder] = {}
        self._load_builtin_protocols()

    def _load_builtin_protocols(self) -> None:
        """Load all built-in protocol definitions."""
        # Industrial protocols
        self._add_modbus_rtu()
        self._add_modbus_tcp()
        self._add_dnp3()
        self._add_bacnet()

        # IoT protocols
        self._add_mqtt()
        self._add_coap()
        self._add_cbor()
        self._add_messagepack()

        # Automotive protocols
        self._add_obd2()
        self._add_j1939()
        self._add_can()

        # Network protocols
        self._add_http()
        self._add_dns()
        self._add_ntp()
        self._add_syslog()

        # Serial protocols
        self._add_nmea()
        self._add_xmodem()

        # Building automation
        self._add_knx()
        self._add_lonworks()

        # Custom/generic
        self._add_tlv()
        self._add_length_prefixed()

    def list_protocols(self, category: str | None = None) -> list[ProtocolInfo]:
        """List available protocols.

        Args:
            category: Filter by category (optional).

        Returns:
            List of ProtocolInfo for available protocols.

        Example:
            >>> protocols = library.list_protocols(category="industrial")
            >>> for p in protocols:
            ...     print(f"{p.name}: {p.description}")
        """
        protocols = list(self._protocols.values())
        if category:
            protocols = [p for p in protocols if p.category == category]
        return protocols

    def list_protocol_names(self, category: str | None = None) -> list[str]:
        """List available protocol names.

        Args:
            category: Filter by category (optional).

        Returns:
            List of protocol name strings.
        """
        protocols = self.list_protocols(category)
        return [p.name for p in protocols]

    def get(self, name: str) -> ProtocolInfo | None:
        """Get protocol information by name.

        Args:
            name: Protocol name (case-insensitive).

        Returns:
            ProtocolInfo or None if not found.
        """
        return self._protocols.get(name.lower())

    def get_decoder(self, name: str) -> ProtocolDecoder | None:
        """Get a decoder for a protocol.

        Args:
            name: Protocol name.

        Returns:
            ProtocolDecoder instance or None if not found.

        Example:
            >>> decoder = library.get_decoder("modbus_rtu")
            >>> if decoder:
            ...     message = decoder.decode(data)
        """
        name = name.lower()
        if name in self._decoders:
            return self._decoders[name]

        info = self._protocols.get(name)
        if info and info.definition:
            decoder = ProtocolDecoder(info.definition)
            self._decoders[name] = decoder
            return decoder

        return None

    def get_definition(self, name: str) -> ProtocolDefinition | None:
        """Get protocol definition by name.

        Args:
            name: Protocol name.

        Returns:
            ProtocolDefinition or None if not found.
        """
        info = self._protocols.get(name.lower())
        return info.definition if info else None

    def add_protocol(
        self,
        info: ProtocolInfo,
    ) -> None:
        """Add a custom protocol to the library.

        Args:
            info: Protocol information and definition.

        Example:
            >>> custom = ProtocolInfo(
            ...     name="my_protocol",
            ...     category="custom",
            ...     version="1.0",
            ...     description="My custom protocol",
            ...     definition=my_definition
            ... )
            >>> library.add_protocol(custom)
        """
        self._protocols[info.name.lower()] = info

    def categories(self) -> list[str]:
        """Get list of protocol categories.

        Returns:
            List of category names.
        """
        return sorted({p.category for p in self._protocols.values()})

    # =========================================================================
    # Built-in Protocol Definitions
    # =========================================================================

    def _add_modbus_rtu(self) -> None:
        """Add Modbus RTU protocol definition."""
        definition = ProtocolDefinition(
            name="modbus_rtu",
            description="Modbus RTU serial protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="address",
                    field_type="uint8",
                    description="Slave address (1-247)",
                ),
                FieldDefinition(
                    name="function_code",
                    field_type="uint8",
                    description="Function code",
                    enum={
                        1: "read_coils",
                        2: "read_discrete_inputs",
                        3: "read_holding_registers",
                        4: "read_input_registers",
                        5: "write_single_coil",
                        6: "write_single_register",
                        15: "write_multiple_coils",
                        16: "write_multiple_registers",
                    },
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Function-specific data",
                ),
                FieldDefinition(
                    name="crc",
                    field_type="uint16",
                    endian="little",
                    description="CRC-16 checksum",
                ),
            ],
        )

        self._protocols["modbus_rtu"] = ProtocolInfo(
            name="modbus_rtu",
            category="industrial",
            version="1.0",
            description="Modbus RTU serial protocol for industrial automation",
            reference="https://modbus.org/specs.php",
            definition=definition,
        )

    def _add_modbus_tcp(self) -> None:
        """Add Modbus TCP protocol definition."""
        definition = ProtocolDefinition(
            name="modbus_tcp",
            description="Modbus TCP/IP protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="transaction_id",
                    field_type="uint16",
                    description="Transaction identifier",
                ),
                FieldDefinition(
                    name="protocol_id",
                    field_type="uint16",
                    description="Protocol identifier (0 for Modbus)",
                ),
                FieldDefinition(
                    name="length",
                    field_type="uint16",
                    description="Length of remaining message",
                ),
                FieldDefinition(
                    name="unit_id",
                    field_type="uint8",
                    description="Unit identifier",
                ),
                FieldDefinition(
                    name="function_code",
                    field_type="uint8",
                    description="Function code",
                    enum={
                        1: "read_coils",
                        2: "read_discrete_inputs",
                        3: "read_holding_registers",
                        4: "read_input_registers",
                        5: "write_single_coil",
                        6: "write_single_register",
                        15: "write_multiple_coils",
                        16: "write_multiple_registers",
                    },
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Function-specific data",
                ),
            ],
        )

        self._protocols["modbus_tcp"] = ProtocolInfo(
            name="modbus_tcp",
            category="industrial",
            version="1.0",
            description="Modbus TCP/IP protocol for industrial automation",
            reference="https://modbus.org/specs.php",
            definition=definition,
        )

    def _add_dnp3(self) -> None:
        """Add DNP3 protocol definition."""
        definition = ProtocolDefinition(
            name="dnp3",
            description="Distributed Network Protocol 3",
            endian="little",
            fields=[
                FieldDefinition(
                    name="start",
                    field_type="uint16",
                    value=0x0564,
                    description="Start bytes (0x0564)",
                ),
                FieldDefinition(
                    name="length",
                    field_type="uint8",
                    description="Data link layer length",
                ),
                FieldDefinition(
                    name="control",
                    field_type="uint8",
                    description="Control byte",
                ),
                FieldDefinition(
                    name="destination",
                    field_type="uint16",
                    description="Destination address",
                ),
                FieldDefinition(
                    name="source",
                    field_type="uint16",
                    description="Source address",
                ),
                FieldDefinition(
                    name="crc",
                    field_type="uint16",
                    description="CRC-16 checksum",
                ),
                FieldDefinition(
                    name="transport_header",
                    field_type="uint8",
                    description="Transport layer header",
                ),
                FieldDefinition(
                    name="application_data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Application layer data",
                ),
            ],
        )

        self._protocols["dnp3"] = ProtocolInfo(
            name="dnp3",
            category="industrial",
            version="3.0",
            description="Distributed Network Protocol for SCADA systems",
            reference="https://www.dnp.org/",
            definition=definition,
        )

    def _add_bacnet(self) -> None:
        """Add BACnet protocol definition."""
        definition = ProtocolDefinition(
            name="bacnet",
            description="Building Automation and Control Networks",
            endian="big",
            fields=[
                FieldDefinition(
                    name="type",
                    field_type="uint8",
                    description="BVLC type (0x81 for BACnet/IP)",
                ),
                FieldDefinition(
                    name="function",
                    field_type="uint8",
                    description="BVLC function",
                    enum={
                        0x00: "bvlc_result",
                        0x01: "write_broadcast_table",
                        0x04: "forwarded_npdu",
                        0x0A: "original_unicast_npdu",
                        0x0B: "original_broadcast_npdu",
                    },
                ),
                FieldDefinition(
                    name="length",
                    field_type="uint16",
                    description="Total BVLC length",
                ),
                FieldDefinition(
                    name="npdu",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Network Protocol Data Unit",
                ),
            ],
        )

        self._protocols["bacnet"] = ProtocolInfo(
            name="bacnet",
            category="building",
            version="2020",
            description="Building Automation and Control Networks protocol",
            reference="https://www.bacnetinternational.org/",
            definition=definition,
        )

    def _add_http(self) -> None:
        """Add HTTP protocol definition (simplified for binary inspection)."""
        # HTTP is text-based, so we provide a simplified binary representation
        definition = ProtocolDefinition(
            name="http",
            description="Hypertext Transfer Protocol (simplified)",
            endian="big",
            fields=[
                FieldDefinition(
                    name="request_line",
                    field_type="string",
                    description="HTTP request/status line",
                ),
                FieldDefinition(
                    name="headers",
                    field_type="string",
                    size_ref="remaining",
                    description="HTTP headers and body",
                ),
            ],
        )

        self._protocols["http"] = ProtocolInfo(
            name="http",
            category="network",
            version="1.1",
            description="Hypertext Transfer Protocol",
            reference="https://tools.ietf.org/html/rfc2616",
            definition=definition,
        )

    def _add_mqtt(self) -> None:
        """Add MQTT protocol definition."""
        definition = ProtocolDefinition(
            name="mqtt",
            description="Message Queuing Telemetry Transport",
            endian="big",
            fields=[
                FieldDefinition(
                    name="fixed_header",
                    field_type="uint8",
                    description="Fixed header (type + flags)",
                ),
                FieldDefinition(
                    name="remaining_length",
                    field_type="uint8",
                    description="Remaining length (variable encoding)",
                ),
                FieldDefinition(
                    name="payload",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Variable header + payload",
                ),
            ],
        )

        self._protocols["mqtt"] = ProtocolInfo(
            name="mqtt",
            category="iot",
            version="5.0",
            description="MQTT IoT messaging protocol",
            reference="https://mqtt.org/",
            definition=definition,
        )

    def _add_coap(self) -> None:
        """Add CoAP protocol definition."""
        definition = ProtocolDefinition(
            name="coap",
            description="Constrained Application Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="version_type_tkl",
                    field_type="uint8",
                    description="Version (2b) + Type (2b) + Token Length (4b)",
                ),
                FieldDefinition(
                    name="code",
                    field_type="uint8",
                    description="Method/Response code",
                    enum={
                        0x01: "GET",
                        0x02: "POST",
                        0x03: "PUT",
                        0x04: "DELETE",
                        0x44: "2.04 Changed",
                        0x45: "2.05 Content",
                        0x81: "4.01 Unauthorized",
                        0x84: "4.04 Not Found",
                    },
                ),
                FieldDefinition(
                    name="message_id",
                    field_type="uint16",
                    description="Message ID",
                ),
                FieldDefinition(
                    name="token_options_payload",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Token + Options + Payload",
                ),
            ],
        )

        self._protocols["coap"] = ProtocolInfo(
            name="coap",
            category="iot",
            version="RFC 7252",
            description="Constrained Application Protocol for IoT",
            reference="https://tools.ietf.org/html/rfc7252",
            definition=definition,
        )

    def _add_cbor(self) -> None:
        """Add CBOR protocol definition (simplified header)."""
        definition = ProtocolDefinition(
            name="cbor",
            description="Concise Binary Object Representation",
            endian="big",
            fields=[
                FieldDefinition(
                    name="initial_byte",
                    field_type="uint8",
                    description="Major type (3b) + Additional info (5b)",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="CBOR data",
                ),
            ],
        )

        self._protocols["cbor"] = ProtocolInfo(
            name="cbor",
            category="iot",
            version="RFC 8949",
            description="Concise Binary Object Representation",
            reference="https://tools.ietf.org/html/rfc8949",
            definition=definition,
        )

    def _add_messagepack(self) -> None:
        """Add MessagePack protocol definition (simplified header)."""
        definition = ProtocolDefinition(
            name="messagepack",
            description="MessagePack binary serialization",
            endian="big",
            fields=[
                FieldDefinition(
                    name="format",
                    field_type="uint8",
                    description="Format byte",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="MessagePack data",
                ),
            ],
        )

        self._protocols["messagepack"] = ProtocolInfo(
            name="messagepack",
            category="iot",
            version="1.0",
            description="MessagePack binary serialization format",
            reference="https://msgpack.org/",
            definition=definition,
        )

    def _add_obd2(self) -> None:
        """Add OBD-II protocol definition."""
        definition = ProtocolDefinition(
            name="obd2",
            description="On-Board Diagnostics II",
            endian="big",
            fields=[
                FieldDefinition(
                    name="mode",
                    field_type="uint8",
                    description="Service/mode",
                    enum={
                        0x01: "show_current_data",
                        0x02: "show_freeze_frame",
                        0x03: "show_dtc",
                        0x04: "clear_dtc",
                        0x09: "request_vehicle_info",
                    },
                ),
                FieldDefinition(
                    name="pid",
                    field_type="uint8",
                    description="Parameter ID",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Parameter data",
                ),
            ],
        )

        self._protocols["obd2"] = ProtocolInfo(
            name="obd2",
            category="automotive",
            version="ISO 15031",
            description="On-Board Diagnostics II automotive protocol",
            reference="https://www.iso.org/standard/66369.html",
            definition=definition,
        )

    def _add_j1939(self) -> None:
        """Add SAE J1939 protocol definition."""
        definition = ProtocolDefinition(
            name="j1939",
            description="SAE J1939 Heavy Duty Vehicle Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="priority",
                    field_type="uint8",
                    size=3,
                    description="Message priority (0-7)",
                ),
                FieldDefinition(
                    name="pgn",
                    field_type="uint32",
                    size=18,
                    description="Parameter Group Number",
                ),
                FieldDefinition(
                    name="source_address",
                    field_type="uint8",
                    description="Source address",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size=8,
                    description="Data bytes (up to 8)",
                ),
            ],
        )

        self._protocols["j1939"] = ProtocolInfo(
            name="j1939",
            category="automotive",
            version="J1939-21",
            description="SAE J1939 protocol for heavy-duty vehicles",
            reference="https://www.sae.org/standards/content/j1939_202210/",
            definition=definition,
        )

    def _add_can(self) -> None:
        """Add CAN bus frame definition."""
        definition = ProtocolDefinition(
            name="can",
            description="Controller Area Network",
            endian="big",
            fields=[
                FieldDefinition(
                    name="identifier",
                    field_type="uint32",
                    description="CAN ID (11 or 29 bits)",
                ),
                FieldDefinition(
                    name="dlc",
                    field_type="uint8",
                    description="Data Length Code (0-8)",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size=8,
                    description="Data bytes",
                ),
            ],
        )

        self._protocols["can"] = ProtocolInfo(
            name="can",
            category="automotive",
            version="2.0B",
            description="Controller Area Network bus protocol",
            reference="https://www.iso.org/standard/63648.html",
            definition=definition,
        )

    def _add_dns(self) -> None:
        """Add DNS protocol definition."""
        definition = ProtocolDefinition(
            name="dns",
            description="Domain Name System",
            endian="big",
            fields=[
                FieldDefinition(
                    name="transaction_id",
                    field_type="uint16",
                    description="Transaction ID",
                ),
                FieldDefinition(
                    name="flags",
                    field_type="uint16",
                    description="Flags",
                ),
                FieldDefinition(
                    name="questions",
                    field_type="uint16",
                    description="Question count",
                ),
                FieldDefinition(
                    name="answer_rrs",
                    field_type="uint16",
                    description="Answer RR count",
                ),
                FieldDefinition(
                    name="authority_rrs",
                    field_type="uint16",
                    description="Authority RR count",
                ),
                FieldDefinition(
                    name="additional_rrs",
                    field_type="uint16",
                    description="Additional RR count",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Questions, answers, authority, additional",
                ),
            ],
        )

        self._protocols["dns"] = ProtocolInfo(
            name="dns",
            category="network",
            version="RFC 1035",
            description="Domain Name System protocol",
            reference="https://tools.ietf.org/html/rfc1035",
            definition=definition,
        )

    def _add_ntp(self) -> None:
        """Add NTP protocol definition."""
        definition = ProtocolDefinition(
            name="ntp",
            description="Network Time Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="flags",
                    field_type="uint8",
                    description="LI (2b) + VN (3b) + Mode (3b)",
                ),
                FieldDefinition(
                    name="stratum",
                    field_type="uint8",
                    description="Stratum level",
                ),
                FieldDefinition(
                    name="poll",
                    field_type="uint8",
                    description="Poll interval",
                ),
                FieldDefinition(
                    name="precision",
                    field_type="int8",
                    description="Clock precision",
                ),
                FieldDefinition(
                    name="root_delay",
                    field_type="uint32",
                    description="Root delay",
                ),
                FieldDefinition(
                    name="root_dispersion",
                    field_type="uint32",
                    description="Root dispersion",
                ),
                FieldDefinition(
                    name="reference_id",
                    field_type="bytes",
                    size=4,
                    description="Reference identifier",
                ),
                FieldDefinition(
                    name="reference_timestamp",
                    field_type="uint64",
                    description="Reference timestamp",
                ),
                FieldDefinition(
                    name="origin_timestamp",
                    field_type="uint64",
                    description="Origin timestamp",
                ),
                FieldDefinition(
                    name="receive_timestamp",
                    field_type="uint64",
                    description="Receive timestamp",
                ),
                FieldDefinition(
                    name="transmit_timestamp",
                    field_type="uint64",
                    description="Transmit timestamp",
                ),
            ],
        )

        self._protocols["ntp"] = ProtocolInfo(
            name="ntp",
            category="network",
            version="4",
            description="Network Time Protocol for time synchronization",
            reference="https://tools.ietf.org/html/rfc5905",
            definition=definition,
        )

    def _add_syslog(self) -> None:
        """Add Syslog protocol definition."""
        definition = ProtocolDefinition(
            name="syslog",
            description="Syslog Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="priority",
                    field_type="string",
                    description="Priority value in angle brackets",
                ),
                FieldDefinition(
                    name="message",
                    field_type="string",
                    size_ref="remaining",
                    description="Syslog message",
                ),
            ],
        )

        self._protocols["syslog"] = ProtocolInfo(
            name="syslog",
            category="network",
            version="RFC 5424",
            description="Syslog protocol for system logging",
            reference="https://tools.ietf.org/html/rfc5424",
            definition=definition,
        )

    def _add_nmea(self) -> None:
        """Add NMEA 0183 protocol definition."""
        definition = ProtocolDefinition(
            name="nmea",
            description="NMEA 0183 GPS Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="start",
                    field_type="string",
                    size=1,
                    value="$",
                    description="Start delimiter",
                ),
                FieldDefinition(
                    name="talker_id",
                    field_type="string",
                    size=2,
                    description="Talker identifier (GP, GN, etc.)",
                ),
                FieldDefinition(
                    name="sentence_id",
                    field_type="string",
                    size=3,
                    description="Sentence identifier (GGA, RMC, etc.)",
                ),
                FieldDefinition(
                    name="data",
                    field_type="string",
                    size_ref="remaining",
                    description="Comma-separated data fields",
                ),
            ],
        )

        self._protocols["nmea"] = ProtocolInfo(
            name="nmea",
            category="serial",
            version="0183",
            description="NMEA 0183 GPS/navigation protocol",
            reference="https://www.nmea.org/",
            definition=definition,
        )

    def _add_xmodem(self) -> None:
        """Add XMODEM protocol definition."""
        definition = ProtocolDefinition(
            name="xmodem",
            description="XMODEM File Transfer Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="header",
                    field_type="uint8",
                    description="SOH (0x01), EOT (0x04), or CAN (0x18)",
                    enum={
                        0x01: "SOH",
                        0x04: "EOT",
                        0x06: "ACK",
                        0x15: "NAK",
                        0x18: "CAN",
                    },
                ),
                FieldDefinition(
                    name="block_number",
                    field_type="uint8",
                    description="Block number (1-255)",
                ),
                FieldDefinition(
                    name="block_complement",
                    field_type="uint8",
                    description="One's complement of block number",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size=128,
                    description="Data block (128 bytes)",
                ),
                FieldDefinition(
                    name="checksum",
                    field_type="uint8",
                    description="Arithmetic sum checksum",
                ),
            ],
        )

        self._protocols["xmodem"] = ProtocolInfo(
            name="xmodem",
            category="serial",
            version="1.0",
            description="XMODEM file transfer protocol",
            reference="https://en.wikipedia.org/wiki/XMODEM",
            definition=definition,
        )

    def _add_knx(self) -> None:
        """Add KNX protocol definition."""
        definition = ProtocolDefinition(
            name="knx",
            description="KNX Building Automation Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="header_length",
                    field_type="uint8",
                    value=0x06,
                    description="Header length (always 6)",
                ),
                FieldDefinition(
                    name="protocol_version",
                    field_type="uint8",
                    value=0x10,
                    description="Protocol version (0x10)",
                ),
                FieldDefinition(
                    name="service_type",
                    field_type="uint16",
                    description="Service type identifier",
                    enum={
                        0x0201: "search_request",
                        0x0202: "search_response",
                        0x0203: "description_request",
                        0x0204: "description_response",
                        0x0205: "connect_request",
                        0x0206: "connect_response",
                        0x0420: "tunnelling_request",
                        0x0421: "tunnelling_ack",
                    },
                ),
                FieldDefinition(
                    name="total_length",
                    field_type="uint16",
                    description="Total length including header",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="Service-specific data",
                ),
            ],
        )

        self._protocols["knx"] = ProtocolInfo(
            name="knx",
            category="building",
            version="2.0",
            description="KNX building automation protocol",
            reference="https://www.knx.org/",
            definition=definition,
        )

    def _add_lonworks(self) -> None:
        """Add LonWorks protocol definition (simplified)."""
        definition = ProtocolDefinition(
            name="lonworks",
            description="LonWorks Control Network Protocol",
            endian="big",
            fields=[
                FieldDefinition(
                    name="npdu",
                    field_type="uint8",
                    description="Network PDU byte",
                ),
                FieldDefinition(
                    name="domain_length",
                    field_type="uint8",
                    description="Domain length (0, 1, 3, 6)",
                ),
                FieldDefinition(
                    name="data",
                    field_type="bytes",
                    size_ref="remaining",
                    description="LonWorks data",
                ),
            ],
        )

        self._protocols["lonworks"] = ProtocolInfo(
            name="lonworks",
            category="building",
            version="1.0",
            description="LonWorks control network protocol",
            reference="https://www.echelon.com/",
            definition=definition,
        )

    def _add_tlv(self) -> None:
        """Add generic TLV (Tag-Length-Value) format."""
        definition = ProtocolDefinition(
            name="tlv",
            description="Tag-Length-Value Generic Format",
            endian="big",
            fields=[
                FieldDefinition(
                    name="tag",
                    field_type="uint8",
                    description="Type/tag identifier",
                ),
                FieldDefinition(
                    name="length",
                    field_type="uint8",
                    description="Value length",
                ),
                FieldDefinition(
                    name="value",
                    field_type="bytes",
                    size_ref="length",
                    description="Value data",
                ),
            ],
        )

        self._protocols["tlv"] = ProtocolInfo(
            name="tlv",
            category="custom",
            version="1.0",
            description="Generic Tag-Length-Value encoding",
            definition=definition,
        )

    def _add_length_prefixed(self) -> None:
        """Add generic length-prefixed format."""
        definition = ProtocolDefinition(
            name="length_prefixed",
            description="Length-Prefixed Message Format",
            endian="big",
            fields=[
                FieldDefinition(
                    name="length",
                    field_type="uint16",
                    description="Message length",
                ),
                FieldDefinition(
                    name="payload",
                    field_type="bytes",
                    size_ref="length",
                    description="Message payload",
                ),
            ],
        )

        self._protocols["length_prefixed"] = ProtocolInfo(
            name="length_prefixed",
            category="custom",
            version="1.0",
            description="Generic length-prefixed message format",
            definition=definition,
        )


# Global library instance
_library: ProtocolLibrary | None = None


def get_library() -> ProtocolLibrary:
    """Get the global protocol library instance.

    Returns:
        ProtocolLibrary singleton instance.
    """
    global _library
    if _library is None:
        _library = ProtocolLibrary()
    return _library


@overload
def list_protocols(category: str | None = None, *, names_only: Literal[True]) -> list[str]: ...


@overload
def list_protocols(
    category: str | None = None, *, names_only: Literal[False] = ...
) -> list[ProtocolInfo]: ...


def list_protocols(
    category: str | None = None, *, names_only: bool = False
) -> list[str] | list[ProtocolInfo]:
    """List available protocols in the library.

    Implements RE-DSL-003: Protocol listing.

    Args:
        category: Optional category filter.
        names_only: If True, return list of protocol names (strings).
                   If False (default), return list of ProtocolInfo objects.

    Returns:
        List of available protocols (names or ProtocolInfo).

    Example:
        >>> # Get protocol names
        >>> names = list_protocols(names_only=True)
        >>> "http" in names
        True

        >>> # Get full protocol info
        >>> protocols = list_protocols()
        >>> protocols[0].name
        'modbus_rtu'
    """
    if names_only:
        return get_library().list_protocol_names(category)
    return get_library().list_protocols(category)


def get_protocol(name: str) -> ProtocolInfo | None:
    """Get protocol information by name.

    Args:
        name: Protocol name.

    Returns:
        ProtocolInfo or None.
    """
    return get_library().get(name)


def get_decoder(name: str) -> ProtocolDecoder | None:
    """Get a decoder for a protocol.

    Implements RE-DSL-003: Protocol decoding.

    Args:
        name: Protocol name.

    Returns:
        ProtocolDecoder or None.
    """
    return get_library().get_decoder(name)


__all__ = [
    "ProtocolInfo",
    "ProtocolLibrary",
    "get_decoder",
    "get_library",
    "get_protocol",
    "list_protocols",
]
