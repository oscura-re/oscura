"""MQTT protocol analyzer for versions 3.1.1 and 5.0.

This module provides comprehensive MQTT protocol analysis including packet
parsing, session tracking, and topic hierarchy discovery.

Example:
    >>> from oscura.iot.mqtt import MQTTAnalyzer
    >>> analyzer = MQTTAnalyzer()
    >>> packet = analyzer.parse_packet(data, timestamp=0.0)
    >>> topology = analyzer.get_topic_hierarchy()

References:
    MQTT 3.1.1: http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/
    MQTT 5.0: https://docs.oasis-open.org/mqtt/mqtt/v5.0/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from oscura.iot.mqtt.properties import parse_properties


@dataclass
class MQTTPacket:
    """MQTT control packet representation.

    Attributes:
        timestamp: Packet timestamp in seconds.
        packet_type: Packet type name ("CONNECT", "PUBLISH", etc.).
        protocol_version: MQTT protocol version ("3.1.1" or "5.0").
        flags: Packet flags (DUP, QoS, RETAIN, etc.).
        packet_id: Packet identifier for QoS > 0 (optional).
        topic: Topic name for PUBLISH/SUBSCRIBE packets (optional).
        payload: Packet payload bytes.
        properties: MQTT 5.0 properties dictionary.
        qos: Quality of Service level (0, 1, or 2).

    Example:
        >>> packet = MQTTPacket(
        ...     timestamp=0.0,
        ...     packet_type="PUBLISH",
        ...     protocol_version="3.1.1",
        ...     flags={"dup": False, "qos": 0, "retain": False},
        ...     topic="home/sensor/temperature",
        ...     payload=b"22.5",
        ...     qos=0,
        ... )
    """

    timestamp: float
    packet_type: str
    protocol_version: str
    flags: dict[str, bool | int]
    packet_id: int | None = None
    topic: str | None = None
    payload: bytes = b""
    properties: dict[str, Any] = field(default_factory=dict)
    qos: int = 0


@dataclass
class MQTTSession:
    """MQTT session information.

    Tracks connection details and topic subscriptions for a client.

    Attributes:
        client_id: Client identifier string.
        username: Authentication username (optional).
        protocol_version: MQTT protocol version ("3.1.1" or "5.0").
        keep_alive: Keep-alive interval in seconds.
        clean_session: Clean session flag (3.1.1) or clean start flag (5.0).
        will_topic: Last Will and Testament topic (optional).
        will_message: Last Will and Testament message payload (optional).
        subscribed_topics: List of subscribed topic filters.
        published_topics: Dictionary mapping topics to publish counts.

    Example:
        >>> session = MQTTSession(
        ...     client_id="sensor_01",
        ...     username="admin",
        ...     protocol_version="3.1.1",
        ...     keep_alive=60,
        ... )
    """

    client_id: str
    username: str | None = None
    protocol_version: str = "3.1.1"
    keep_alive: int = 60
    clean_session: bool = True
    will_topic: str | None = None
    will_message: bytes | None = None
    subscribed_topics: list[str] = field(default_factory=list)
    published_topics: dict[str, int] = field(default_factory=dict)


class MQTTAnalyzer:
    """MQTT protocol analyzer for versions 3.1.1 and 5.0.

    Parses MQTT control packets, tracks sessions, and builds topic hierarchies.

    Attributes:
        PACKET_TYPES: Mapping of packet type codes to names.

    Example:
        >>> analyzer = MQTTAnalyzer()
        >>> packet = analyzer.parse_packet(mqtt_data)
        >>> hierarchy = analyzer.get_topic_hierarchy()
        >>> analyzer.export_topology(Path("mqtt_topology.json"))
    """

    # MQTT control packet types
    PACKET_TYPES: ClassVar[dict[int, str]] = {
        1: "CONNECT",
        2: "CONNACK",
        3: "PUBLISH",
        4: "PUBACK",
        5: "PUBREC",
        6: "PUBREL",
        7: "PUBCOMP",
        8: "SUBSCRIBE",
        9: "SUBACK",
        10: "UNSUBSCRIBE",
        11: "UNSUBACK",
        12: "PINGREQ",
        13: "PINGRESP",
        14: "DISCONNECT",
        15: "AUTH",
    }

    def __init__(self) -> None:
        """Initialize MQTT analyzer.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> len(analyzer.packets)
            0
        """
        self.packets: list[MQTTPacket] = []
        self.sessions: dict[str, MQTTSession] = {}
        self.topics: set[str] = set()

    def _calculate_header_size(self, data: bytes) -> int:
        """Calculate fixed header size including length encoding.

        Args:
            data: Raw packet data

        Returns:
            Size of fixed header in bytes
        """
        header_size = 1
        temp_data = data[1:]
        while temp_data and (temp_data[0] & 0x80):
            header_size += 1
            temp_data = temp_data[1:]
        return header_size + 1

    def _parse_packet_data(
        self, packet_type: str, var_header_payload: bytes, flags: int
    ) -> dict[str, Any]:
        """Parse variable header and payload based on packet type.

        Args:
            packet_type: Type of MQTT packet
            var_header_payload: Variable header and payload bytes
            flags: Packet flags

        Returns:
            Parsed packet data dictionary
        """
        parsers: dict[str, Any] = {
            "CONNECT": lambda: self._parse_connect(var_header_payload),
            "PUBLISH": lambda: self._parse_publish(var_header_payload, flags),
            "SUBSCRIBE": lambda: self._parse_subscribe(var_header_payload),
            "SUBACK": lambda: self._parse_suback(var_header_payload),
            "UNSUBSCRIBE": lambda: self._parse_unsubscribe(var_header_payload),
            "CONNACK": lambda: self._parse_connack(var_header_payload),
            "DISCONNECT": lambda: self._parse_disconnect(var_header_payload),
            "AUTH": lambda: self._parse_auth(var_header_payload),
        }

        if packet_type in parsers:
            # Lambda returns dict[str, Any] from parser methods
            from typing import cast

            return cast("dict[str, Any]", parsers[packet_type]())

        if packet_type in ["PUBACK", "PUBREC", "PUBREL", "PUBCOMP", "UNSUBACK"]:
            return self._parse_ack(var_header_payload, packet_type)

        if packet_type in ["PINGREQ", "PINGRESP"]:
            return {}

        return {"payload": var_header_payload}

    def _track_packet_metadata(self, packet: MQTTPacket, parsed_data: dict[str, Any]) -> None:
        """Track packet metadata (topics, sessions, subscriptions).

        Args:
            packet: Parsed MQTT packet
            parsed_data: Raw parsed data
        """
        if packet.topic:
            self.topics.add(packet.topic)

        if packet.packet_type == "CONNECT" and "client_id" in parsed_data:
            self._track_session(parsed_data)

        if packet.packet_type == "SUBSCRIBE" and "topics" in parsed_data:
            for topic_filter, _ in parsed_data["topics"]:
                self.topics.add(topic_filter)

    def parse_packet(self, data: bytes, timestamp: float = 0.0) -> MQTTPacket:
        """Parse MQTT control packet.

        Args:
            data: Raw MQTT packet bytes.
            timestamp: Packet timestamp in seconds.

        Returns:
            Parsed MQTT packet.

        Raises:
            ValueError: If packet is malformed or incomplete.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> # CONNECT packet
            >>> data = b"\\x10\\x10\\x00\\x04MQTT\\x04\\x02\\x00\\x3c\\x00\\x04test"
            >>> packet = analyzer.parse_packet(data)
            >>> packet.packet_type
            'CONNECT'
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for MQTT packet")

        # Parse fixed header and extract variable header/payload
        packet_type, var_header_payload, flags = self._extract_packet_components(data)

        # Parse variable header and payload
        parsed_data = self._parse_packet_data(packet_type, var_header_payload, flags)

        # Create packet object
        packet = self._create_packet(timestamp, packet_type, parsed_data)

        # Track metadata
        self.packets.append(packet)
        self._track_packet_metadata(packet, parsed_data)

        return packet

    def _extract_packet_components(self, data: bytes) -> tuple[str, bytes, int]:
        """Extract packet type and variable header/payload from data.

        Args:
            data: Raw MQTT packet bytes.

        Returns:
            Tuple of (packet_type, var_header_payload, flags).

        Raises:
            ValueError: If packet is malformed or incomplete.
        """
        packet_type_code, flags, remaining_length = self._parse_fixed_header(data)

        if packet_type_code not in self.PACKET_TYPES:
            raise ValueError(f"Unknown packet type: {packet_type_code}")

        packet_type = self.PACKET_TYPES[packet_type_code]
        header_size = self._calculate_header_size(data)

        if len(data) < header_size + remaining_length:
            raise ValueError("Incomplete packet data")

        var_header_payload = data[header_size : header_size + remaining_length]
        return packet_type, var_header_payload, flags

    def _create_packet(
        self, timestamp: float, packet_type: str, parsed_data: dict[str, Any]
    ) -> MQTTPacket:
        """Create MQTTPacket from parsed data.

        Args:
            timestamp: Packet timestamp.
            packet_type: Packet type string.
            parsed_data: Parsed packet data dictionary.

        Returns:
            MQTTPacket instance.
        """
        return MQTTPacket(
            timestamp=timestamp,
            packet_type=packet_type,
            protocol_version=parsed_data.get("protocol_version", "3.1.1"),
            flags=parsed_data.get("flags", {}),
            packet_id=parsed_data.get("packet_id"),
            topic=parsed_data.get("topic"),
            payload=parsed_data.get("payload", b""),
            properties=parsed_data.get("properties", {}),
            qos=parsed_data.get("qos", 0),
        )

    def _parse_fixed_header(self, data: bytes) -> tuple[int, int, int]:
        """Parse MQTT fixed header.

        The fixed header consists of:
        - Byte 1: Control Packet type (bits 4-7) and Flags (bits 0-3)
        - Byte 2+: Remaining Length (variable length encoding)

        Args:
            data: Raw packet data.

        Returns:
            Tuple of (packet_type, flags, remaining_length).

        Raises:
            ValueError: If header is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> # PUBLISH packet: type=3, flags=0, length=10
            >>> data = b"\\x30\\x0a..."
            >>> ptype, flags, length = analyzer._parse_fixed_header(data)
            >>> ptype
            3
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for fixed header")

        byte1 = data[0]
        packet_type = (byte1 >> 4) & 0x0F
        flags = byte1 & 0x0F

        # Decode remaining length (variable byte integer)
        multiplier = 1
        value = 0
        index = 1

        while True:
            if index >= len(data):
                raise ValueError("Incomplete remaining length")

            encoded_byte = data[index]
            value += (encoded_byte & 0x7F) * multiplier

            if (encoded_byte & 0x80) == 0:
                break

            multiplier *= 128
            index += 1

            if multiplier > 128 * 128 * 128:
                raise ValueError("Remaining length exceeds maximum")

        return packet_type, flags, value

    def _parse_mqtt_string(self, data: bytes, offset: int) -> tuple[str, int]:
        """Parse MQTT UTF-8 encoded string.

        Args:
            data: Packet data buffer.
            offset: Current offset in buffer.

        Returns:
            Tuple of (parsed_string, new_offset).

        Raises:
            ValueError: If string data is incomplete.
        """
        if len(data) < offset + 2:
            raise ValueError("Incomplete string length field")

        str_len = int.from_bytes(data[offset : offset + 2], "big")
        offset += 2

        if len(data) < offset + str_len:
            raise ValueError("Incomplete string data")

        parsed_str = data[offset : offset + str_len].decode("utf-8")
        return parsed_str, offset + str_len

    def _parse_mqtt_binary(self, data: bytes, offset: int) -> tuple[bytes, int]:
        """Parse MQTT binary data field.

        Args:
            data: Packet data buffer.
            offset: Current offset in buffer.

        Returns:
            Tuple of (binary_data, new_offset).

        Raises:
            ValueError: If binary data is incomplete.
        """
        if len(data) < offset + 2:
            raise ValueError("Incomplete binary length field")

        bin_len = int.from_bytes(data[offset : offset + 2], "big")
        offset += 2

        if len(data) < offset + bin_len:
            raise ValueError("Incomplete binary data")

        binary_data = data[offset : offset + bin_len]
        return binary_data, offset + bin_len

    def _parse_connect_flags(self, connect_flags: int) -> dict[str, bool | int]:
        """Parse CONNECT packet flags byte.

        Args:
            connect_flags: Flags byte from CONNECT packet.

        Returns:
            Dictionary of parsed flag values.
        """
        return {
            "clean_session": bool(connect_flags & 0x02),
            "will_flag": bool(connect_flags & 0x04),
            "will_qos": (connect_flags >> 3) & 0x03,
            "will_retain": bool(connect_flags & 0x20),
            "username_flag": bool(connect_flags & 0x40),
            "password_flag": bool(connect_flags & 0x80),
        }

    def _parse_will_data(
        self, data: bytes, offset: int, protocol_version: str
    ) -> tuple[str, bytes, int]:
        """Parse Will topic and message from CONNECT packet.

        Args:
            data: Packet data buffer.
            offset: Current offset in buffer.
            protocol_version: MQTT protocol version.

        Returns:
            Tuple of (will_topic, will_message, new_offset).

        Raises:
            ValueError: If will data is incomplete.
        """
        # Will properties (MQTT 5.0 only)
        if protocol_version == "5.0":
            _, consumed = parse_properties(data, offset)
            offset += consumed

        # Will topic
        will_topic, offset = self._parse_mqtt_string(data, offset)

        # Will message
        will_message, offset = self._parse_mqtt_binary(data, offset)

        return will_topic, will_message, offset

    def _parse_connect(self, data: bytes) -> dict[str, Any]:
        """Parse CONNECT packet.

        Args:
            data: Variable header and payload bytes.

        Returns:
            Parsed CONNECT fields.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> # MQTT 3.1.1 CONNECT
            >>> data = b"\\x00\\x04MQTT\\x04\\x02\\x00\\x3c\\x00\\x04test"
            >>> result = analyzer._parse_connect(data)
            >>> result["protocol_version"]
            '3.1.1'
        """
        if len(data) < 10:
            raise ValueError("Insufficient data for CONNECT packet")

        offset = 0

        # Protocol name
        protocol_name, offset = self._parse_mqtt_string(data, offset)

        # Protocol level
        protocol_level = data[offset]
        offset += 1

        # Map protocol level to version
        version_map = {3: "3.1", 4: "3.1.1", 5: "5.0"}
        protocol_version = version_map.get(protocol_level, "3.1.1")

        # Connect flags
        connect_flags = data[offset]
        offset += 1
        flags = self._parse_connect_flags(connect_flags)

        # Keep alive
        keep_alive = int.from_bytes(data[offset : offset + 2], "big")
        offset += 2

        # MQTT 5.0 properties
        properties: dict[str, Any] = {}
        if protocol_version == "5.0":
            properties, consumed = parse_properties(data, offset)
            offset += consumed

        # Client ID
        client_id, offset = self._parse_mqtt_string(data, offset)

        # Will topic and message
        will_topic = None
        will_message = None
        if flags["will_flag"]:
            will_topic, will_message, offset = self._parse_will_data(data, offset, protocol_version)

        # Username
        username = None
        if flags["username_flag"]:
            username, offset = self._parse_mqtt_string(data, offset)

        # Password
        password = None
        if flags["password_flag"]:
            password, offset = self._parse_mqtt_binary(data, offset)

        return {
            "protocol_name": protocol_name,
            "protocol_version": protocol_version,
            "flags": flags,
            "keep_alive": keep_alive,
            "properties": properties,
            "client_id": client_id,
            "will_topic": will_topic,
            "will_message": will_message,
            "username": username,
            "password": password,
        }

    def _parse_publish(self, data: bytes, flags: int) -> dict[str, Any]:
        """Parse PUBLISH packet.

        Args:
            data: Variable header and payload bytes.
            flags: Fixed header flags byte.

        Returns:
            Parsed PUBLISH fields.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> # PUBLISH to "test" with QoS 0
            >>> data = b"\\x00\\x04test22.5"
            >>> result = analyzer._parse_publish(data, 0x00)
            >>> result["topic"]
            'test'
        """
        qos = (flags >> 1) & 0x03
        dup = bool(flags & 0x08)
        retain = bool(flags & 0x01)

        if len(data) < 2:
            raise ValueError("Insufficient data for topic name length")

        # Parse topic name
        topic_len = int.from_bytes(data[0:2], "big")
        if len(data) < 2 + topic_len:
            raise ValueError("Insufficient data for topic name")

        topic = data[2 : 2 + topic_len].decode("utf-8")
        offset = 2 + topic_len

        # Parse packet identifier (if QoS > 0)
        packet_id = None
        if qos > 0:
            if len(data) < offset + 2:
                raise ValueError("Insufficient data for packet ID")
            packet_id = int.from_bytes(data[offset : offset + 2], "big")
            offset += 2

        # Parse properties (MQTT 5.0) - detect by checking if next byte looks like property length
        properties: dict[str, Any] = {}
        # We'll skip property parsing here for simplicity in 3.1.1 mode
        # In 5.0, properties would be parsed before payload

        # Remaining data is payload
        payload = data[offset:]

        return {
            "topic": topic,
            "qos": qos,
            "flags": {"dup": dup, "qos": qos, "retain": retain},
            "packet_id": packet_id,
            "payload": payload,
            "properties": properties,
        }

    def _parse_subscribe(self, data: bytes) -> dict[str, Any]:
        """Parse SUBSCRIBE packet.

        Args:
            data: Variable header and payload bytes.

        Returns:
            Parsed SUBSCRIBE fields with topic filters.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> # SUBSCRIBE to "test/+" with QoS 1
            >>> data = b"\\x00\\x01\\x00\\x06test/+\\x01"
            >>> result = analyzer._parse_subscribe(data)
            >>> result["topics"][0]
            ('test/+', 1)
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for packet ID")

        # Packet identifier
        packet_id = int.from_bytes(data[0:2], "big")
        offset = 2

        # Properties (MQTT 5.0) - skip for simplicity
        properties: dict[str, Any] = {}

        # Topic filters
        topics = []
        while offset < len(data):
            if len(data) < offset + 2:
                raise ValueError("Incomplete topic filter")

            topic_len = int.from_bytes(data[offset : offset + 2], "big")
            offset += 2

            if len(data) < offset + topic_len:
                raise ValueError("Incomplete topic filter")

            topic_filter = data[offset : offset + topic_len].decode("utf-8")
            offset += topic_len

            if offset >= len(data):
                raise ValueError("Missing subscription options")

            # Subscription options (QoS in lower 2 bits for 3.1.1)
            options = data[offset]
            qos = options & 0x03
            offset += 1

            topics.append((topic_filter, qos))

        return {
            "packet_id": packet_id,
            "properties": properties,
            "topics": topics,
        }

    def _parse_ack(self, data: bytes, packet_type: str) -> dict[str, Any]:
        """Parse acknowledgment packets (PUBACK, PUBREC, PUBREL, PUBCOMP, UNSUBACK).

        Args:
            data: Variable header bytes.
            packet_type: Type of acknowledgment packet.

        Returns:
            Parsed acknowledgment fields.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> data = b"\\x00\\x01"  # Packet ID 1
            >>> result = analyzer._parse_ack(data, "PUBACK")
            >>> result["packet_id"]
            1
        """
        if len(data) < 2:
            raise ValueError(f"Insufficient data for {packet_type} packet ID")

        packet_id = int.from_bytes(data[0:2], "big")

        # MQTT 5.0 may have reason code and properties
        reason_code = None
        properties: dict[str, Any] = {}

        if len(data) > 2:
            reason_code = data[2]
            if len(data) > 3:
                properties, _ = parse_properties(data, 3)

        return {
            "packet_id": packet_id,
            "reason_code": reason_code,
            "properties": properties,
        }

    def _parse_suback(self, data: bytes) -> dict[str, Any]:
        """Parse SUBACK packet.

        Args:
            data: Variable header and payload bytes.

        Returns:
            Parsed SUBACK fields with return codes.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> data = b"\\x00\\x01\\x00\\x01"  # Packet ID 1, QoS 0 and 1 granted
            >>> result = analyzer._parse_suback(data)
            >>> result["return_codes"]
            [0, 1]
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for SUBACK packet ID")

        packet_id = int.from_bytes(data[0:2], "big")
        offset = 2

        # Properties (MQTT 5.0)
        properties: dict[str, Any] = {}

        # Return codes
        return_codes = list(data[offset:])

        return {
            "packet_id": packet_id,
            "properties": properties,
            "return_codes": return_codes,
        }

    def _parse_unsubscribe(self, data: bytes) -> dict[str, Any]:
        """Parse UNSUBSCRIBE packet.

        Args:
            data: Variable header and payload bytes.

        Returns:
            Parsed UNSUBSCRIBE fields with topic filters.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> data = b"\\x00\\x01\\x00\\x04test"
            >>> result = analyzer._parse_unsubscribe(data)
            >>> result["topics"]
            ['test']
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for packet ID")

        packet_id = int.from_bytes(data[0:2], "big")
        offset = 2

        # Properties (MQTT 5.0)
        properties: dict[str, Any] = {}

        # Topic filters
        topics = []
        while offset < len(data):
            if len(data) < offset + 2:
                raise ValueError("Incomplete topic filter")

            topic_len = int.from_bytes(data[offset : offset + 2], "big")
            offset += 2

            if len(data) < offset + topic_len:
                raise ValueError("Incomplete topic filter")

            topic_filter = data[offset : offset + topic_len].decode("utf-8")
            offset += topic_len

            topics.append(topic_filter)

        return {
            "packet_id": packet_id,
            "properties": properties,
            "topics": topics,
        }

    def _parse_connack(self, data: bytes) -> dict[str, Any]:
        """Parse CONNACK packet.

        Args:
            data: Variable header bytes.

        Returns:
            Parsed CONNACK fields.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> data = b"\\x00\\x00"  # Session present=0, return code=0 (success)
            >>> result = analyzer._parse_connack(data)
            >>> result["return_code"]
            0
        """
        if len(data) < 2:
            raise ValueError("Insufficient data for CONNACK")

        acknowledge_flags = data[0]
        session_present = bool(acknowledge_flags & 0x01)

        return_code = data[1]

        # Properties (MQTT 5.0)
        properties: dict[str, Any] = {}
        if len(data) > 2:
            properties, _ = parse_properties(data, 2)

        return {
            "flags": {"session_present": session_present},
            "return_code": return_code,
            "properties": properties,
        }

    def _parse_disconnect(self, data: bytes) -> dict[str, Any]:
        """Parse DISCONNECT packet.

        Args:
            data: Variable header bytes.

        Returns:
            Parsed DISCONNECT fields.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> data = b""  # Empty for MQTT 3.1.1
            >>> result = analyzer._parse_disconnect(data)
            >>> result
            {}
        """
        # MQTT 3.1.1 has no variable header
        if len(data) == 0:
            return {}

        # MQTT 5.0 has reason code and properties
        reason_code = data[0] if len(data) > 0 else 0
        properties: dict[str, Any] = {}

        if len(data) > 1:
            properties, _ = parse_properties(data, 1)

        return {
            "reason_code": reason_code,
            "properties": properties,
        }

    def _parse_auth(self, data: bytes) -> dict[str, Any]:
        """Parse AUTH packet (MQTT 5.0 only).

        Args:
            data: Variable header bytes.

        Returns:
            Parsed AUTH fields.

        Raises:
            ValueError: If packet is malformed.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> data = b"\\x00"  # Reason code
            >>> result = analyzer._parse_auth(data)
            >>> result["reason_code"]
            0
        """
        if len(data) < 1:
            raise ValueError("Insufficient data for AUTH")

        reason_code = data[0]
        properties: dict[str, Any] = {}

        if len(data) > 1:
            properties, _ = parse_properties(data, 1)

        return {
            "protocol_version": "5.0",
            "reason_code": reason_code,
            "properties": properties,
        }

    def _track_session(self, connect_data: dict[str, Any]) -> None:
        """Track MQTT session from CONNECT packet.

        Args:
            connect_data: Parsed CONNECT packet data.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> connect_data = {
            ...     "client_id": "test",
            ...     "username": "admin",
            ...     "protocol_version": "3.1.1",
            ...     "keep_alive": 60,
            ...     "flags": {"clean_session": True},
            ... }
            >>> analyzer._track_session(connect_data)
            >>> "test" in analyzer.sessions
            True
        """
        client_id = connect_data["client_id"]

        session = MQTTSession(
            client_id=client_id,
            username=connect_data.get("username"),
            protocol_version=connect_data.get("protocol_version", "3.1.1"),
            keep_alive=connect_data.get("keep_alive", 60),
            clean_session=connect_data.get("flags", {}).get("clean_session", True),
            will_topic=connect_data.get("will_topic"),
            will_message=connect_data.get("will_message"),
        )

        self.sessions[client_id] = session

    def get_topic_hierarchy(self) -> dict[str, Any]:
        """Build hierarchical topic tree from observed topics.

        Topics are split by '/' separator and organized into nested
        dictionaries representing the topic hierarchy.

        Returns:
            Nested dictionary representing topic tree.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> analyzer.topics = {"home/sensor/temperature", "home/sensor/humidity"}
            >>> tree = analyzer.get_topic_hierarchy()
            >>> "home" in tree
            True
            >>> "sensor" in tree["home"]
            True
        """
        tree: dict[str, Any] = {}

        for topic in self.topics:
            parts = topic.split("/")
            current = tree

            for part in parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        return tree

    def export_topology(self, output_path: Path) -> None:
        """Export topic hierarchy and session information as JSON.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = MQTTAnalyzer()
            >>> analyzer.topics = {"test/topic"}
            >>> analyzer.export_topology(Path("topology.json"))
        """
        topology = {
            "topic_hierarchy": self.get_topic_hierarchy(),
            "topics": sorted(self.topics),
            "sessions": {
                client_id: {
                    "client_id": session.client_id,
                    "username": session.username,
                    "protocol_version": session.protocol_version,
                    "keep_alive": session.keep_alive,
                    "clean_session": session.clean_session,
                    "will_topic": session.will_topic,
                    "subscribed_topics": session.subscribed_topics,
                    "published_topics": session.published_topics,
                }
                for client_id, session in self.sessions.items()
            },
            "packet_count": len(self.packets),
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(topology, f, indent=2)


__all__ = [
    "MQTTAnalyzer",
    "MQTTPacket",
    "MQTTSession",
]
