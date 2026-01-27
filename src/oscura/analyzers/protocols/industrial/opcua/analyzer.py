"""OPC UA (Unified Architecture) protocol analyzer.

This module provides comprehensive OPC UA binary protocol analysis for
industrial automation and SCADA systems communication. Supports message
parsing, service decoding, security policy handling, and address space export.

Example:
    >>> from oscura.analyzers.protocols.industrial.opcua import OPCUAAnalyzer
    >>> analyzer = OPCUAAnalyzer()
    >>> # Parse Hello message
    >>> hello = bytes([0x48, 0x45, 0x4C, 0x46, 0x1C, 0x00, 0x00, 0x00, ...])
    >>> msg = analyzer.parse_message(hello, timestamp=0.0)
    >>> print(f"{msg.message_type}: {msg.decoded_service}")
    >>> # Export discovered address space
    >>> analyzer.export_address_space(Path("opcua_nodes.json"))

References:
    OPC UA Part 6: Mappings (Binary Protocol)
    https://reference.opcfoundation.org/Core/Part6/v105/docs/

    OPC UA Part 4: Services
    https://reference.opcfoundation.org/Core/Part4/v105/docs/
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from oscura.analyzers.protocols.industrial.opcua.datatypes import parse_string
from oscura.analyzers.protocols.industrial.opcua.services import SERVICE_PARSERS


@dataclass
class OPCUAMessage:
    """OPC UA message representation.

    Attributes:
        timestamp: Message timestamp in seconds.
        message_type: Message type ("HEL", "ACK", "OPN", "CLO", "MSG", "ERR").
        is_final: True if final chunk ('F' flag).
        chunk_type: Chunk type character ('F', 'C', 'A') or None.
        secure_channel_id: Secure channel identifier.
        security_token_id: Security token ID (for secure messages).
        sequence_number: Message sequence number (for MSG chunks).
        request_id: Request identifier (for MSG chunks).
        service_id: Service type identifier (NodeId).
        service_name: Human-readable service name.
        payload: Raw payload bytes.
        decoded_service: Parsed service-specific data.
    """

    timestamp: float
    message_type: str  # "HEL", "ACK", "OPN", "CLO", "MSG", "ERR"
    is_final: bool  # 'F' flag
    chunk_type: str | None = None  # For chunked messages
    secure_channel_id: int = 0
    security_token_id: int | None = None
    sequence_number: int | None = None
    request_id: int | None = None
    service_id: int | None = None
    service_name: str | None = None
    payload: bytes = b""
    decoded_service: dict[str, Any] = field(default_factory=dict)


@dataclass
class OPCUANode:
    """OPC UA node in address space.

    Represents a node in the OPC UA information model address space.

    Attributes:
        node_id: Node identifier string (e.g., "ns=2;i=1001").
        node_class: Node class ("Object", "Variable", "Method", etc.).
        browse_name: Qualified name for browsing.
        display_name: Localized display name.
        value: Current value (for Variable nodes).
        data_type: Data type identifier (for Variable nodes).
        children: List of child node IDs.
    """

    node_id: str  # e.g., "ns=2;i=1001"
    node_class: str  # "Object", "Variable", "Method", etc.
    browse_name: str | None = None
    display_name: str | None = None
    value: Any = None
    data_type: str | None = None
    children: list[str] = field(default_factory=list)


class OPCUAAnalyzer:
    """OPC UA protocol analyzer for binary protocol.

    Provides comprehensive OPC UA protocol analysis including message parsing,
    service decoding, security handling, and address space discovery.

    Attributes:
        messages: List of parsed OPC UA messages.
        nodes: Dictionary of discovered nodes by node ID.
        security_mode: Current security mode ("None", "Sign", "SignAndEncrypt").

    Example:
        >>> analyzer = OPCUAAnalyzer()
        >>> # Parse Hello message
        >>> hello_msg = bytes([0x48, 0x45, 0x4C, 0x46, ...])  # HEL + F
        >>> msg = analyzer.parse_message(hello_msg)
        >>> assert msg.message_type == "HEL"
        >>> # Parse MSG chunk
        >>> msg_chunk = bytes([0x4D, 0x53, 0x47, 0x46, ...])  # MSG + F
        >>> msg = analyzer.parse_message(msg_chunk)
        >>> print(f"Service: {msg.service_name}")
    """

    # Message type constants (3-byte identifiers)
    MESSAGE_TYPES: ClassVar[dict[int, str]] = {
        0x48454C: "HEL",  # Hello (H E L)
        0x41434B: "ACK",  # Acknowledge (A C K)
        0x4F504E: "OPN",  # Open Secure Channel (O P N)
        0x434C4F: "CLO",  # Close Secure Channel (C L O)
        0x4D5347: "MSG",  # Message (M S G)
        0x455252: "ERR",  # Error (E R R)
    }

    # Service IDs (subset of most common services)
    SERVICE_IDS: ClassVar[dict[int, str]] = {
        421: "ReadRequest",
        424: "ReadResponse",
        673: "WriteRequest",
        676: "WriteResponse",
        527: "BrowseRequest",
        530: "BrowseResponse",
        631: "CreateSubscriptionRequest",
        634: "CreateSubscriptionResponse",
        826: "PublishRequest",
        829: "PublishResponse",
        445: "GetEndpointsRequest",
        448: "GetEndpointsResponse",
        461: "OpenSecureChannelRequest",
        464: "OpenSecureChannelResponse",
        465: "CloseSecureChannelRequest",
        468: "CloseSecureChannelResponse",
    }

    # Node class enumeration
    NODE_CLASSES: ClassVar[dict[int, str]] = {
        1: "Object",
        2: "Variable",
        4: "Method",
        8: "ObjectType",
        16: "VariableType",
        32: "ReferenceType",
        64: "DataType",
        128: "View",
    }

    def __init__(self) -> None:
        """Initialize OPC UA analyzer."""
        self.messages: list[OPCUAMessage] = []
        self.nodes: dict[str, OPCUANode] = {}
        self.security_mode: str = "None"

    def parse_message(self, data: bytes, timestamp: float = 0.0) -> OPCUAMessage:
        """Parse OPC UA binary protocol message.

        Message Header (8 bytes):
        - MessageType (3 bytes) - "HEL", "ACK", "OPN", "MSG", etc.
        - ChunkType (1 byte) - 'F' (Final), 'C' (Continue), 'A' (Abort)
        - MessageSize (4 bytes, little-endian) - Total message size

        Args:
            data: Complete message bytes including header.
            timestamp: Message timestamp in seconds.

        Returns:
            Parsed OPC UA message.

        Raises:
            ValueError: If message is invalid.

        Example:
            >>> analyzer = OPCUAAnalyzer()
            >>> # Hello message
            >>> hello = bytes([0x48, 0x45, 0x4C, 0x46, 0x1C, 0x00, 0x00, 0x00])
            >>> msg = analyzer.parse_message(hello)
            >>> assert msg.message_type == "HEL"
            >>> assert msg.is_final is True
        """
        if len(data) < 8:
            raise ValueError(f"OPC UA message too short: {len(data)} bytes (minimum 8)")

        # Parse common header
        header = self._parse_header(data)

        msg_type = header["message_type"]
        chunk_type = header["chunk_type"]
        is_final = chunk_type == "F"

        # Parse type-specific payload
        decoded: dict[str, Any] = {}
        payload = data[8:]

        if msg_type == "HEL":
            decoded = self._parse_hello(payload)
        elif msg_type == "ACK":
            decoded = self._parse_acknowledge(payload)
        elif msg_type == "OPN":
            decoded = self._parse_open_secure_channel(payload)
        elif msg_type == "CLO":
            decoded = self._parse_open_secure_channel(payload)  # Same format as OPN
        elif msg_type == "MSG":
            decoded = self._parse_message_chunk(payload)
        elif msg_type == "ERR":
            decoded = self._parse_error(payload)

        message = OPCUAMessage(
            timestamp=timestamp,
            message_type=msg_type,
            is_final=is_final,
            chunk_type=chunk_type,
            secure_channel_id=decoded.get("secure_channel_id", 0),
            security_token_id=decoded.get("security_token_id"),
            sequence_number=decoded.get("sequence_number"),
            request_id=decoded.get("request_id"),
            service_id=decoded.get("service_id"),
            service_name=self.SERVICE_IDS.get(decoded.get("service_id", 0)),
            payload=payload,
            decoded_service=decoded,
        )

        self.messages.append(message)
        return message

    def _parse_header(self, data: bytes) -> dict[str, Any]:
        """Parse message header (8 bytes).

        Header Format:
        - MessageType (3 bytes) - ASCII characters
        - ChunkType (1 byte) - 'F', 'C', or 'A'
        - MessageSize (4 bytes, little-endian)

        Args:
            data: Message data starting with header.

        Returns:
            Dictionary with header fields.

        Raises:
            ValueError: If header is invalid.
        """
        # Parse message type (3 bytes)
        msg_type_bytes = data[0:3]
        msg_type_val = (msg_type_bytes[0] << 16) | (msg_type_bytes[1] << 8) | msg_type_bytes[2]
        msg_type = self.MESSAGE_TYPES.get(msg_type_val)

        if msg_type is None:
            # Try to decode as ASCII for better error messages
            try:
                msg_type_str = msg_type_bytes.decode("ascii", errors="ignore")
                raise ValueError(
                    f"Unknown OPC UA message type: {msg_type_str} (0x{msg_type_val:06X})"
                )
            except UnicodeDecodeError as exc:
                raise ValueError(f"Invalid OPC UA message type: 0x{msg_type_val:06X}") from exc

        # Parse chunk type
        chunk_type = chr(data[3])
        if chunk_type not in ("F", "C", "A"):
            raise ValueError(f"Invalid chunk type: {chunk_type} (expected F/C/A)")

        # Parse message size
        message_size = int.from_bytes(data[4:8], "little")

        if message_size < 8:
            raise ValueError(f"Invalid message size: {message_size} (minimum 8)")

        if message_size != len(data):
            raise ValueError(f"Message size mismatch: header={message_size}, actual={len(data)}")

        return {
            "message_type": msg_type,
            "chunk_type": chunk_type,
            "message_size": message_size,
        }

    def _parse_hello(self, data: bytes) -> dict[str, Any]:
        """Parse Hello message payload.

        Hello Message Format (after 8-byte header):
        - ProtocolVersion (4 bytes, little-endian)
        - ReceiveBufferSize (4 bytes, little-endian)
        - SendBufferSize (4 bytes, little-endian)
        - MaxMessageSize (4 bytes, little-endian)
        - MaxChunkCount (4 bytes, little-endian)
        - EndpointUrl (String, length-prefixed UTF-8)

        Args:
            data: Hello payload (without header).

        Returns:
            Parsed Hello message data.

        Raises:
            ValueError: If payload is invalid.

        Example:
            >>> analyzer = OPCUAAnalyzer()
            >>> payload = bytes([0x00, 0x00, 0x00, 0x00, ...])  # Protocol version 0
            >>> hello = analyzer._parse_hello(payload)
            >>> assert 'protocol_version' in hello
        """
        if len(data) < 20:
            raise ValueError(f"Hello message too short: {len(data)} bytes (minimum 20)")

        protocol_version = int.from_bytes(data[0:4], "little")
        receive_buffer_size = int.from_bytes(data[4:8], "little")
        send_buffer_size = int.from_bytes(data[8:12], "little")
        max_message_size = int.from_bytes(data[12:16], "little")
        max_chunk_count = int.from_bytes(data[16:20], "little")

        # Parse endpoint URL (length-prefixed string)
        endpoint_url = None
        if len(data) >= 24:
            url_str, _ = parse_string(data, 20)
            endpoint_url = url_str

        return {
            "protocol_version": protocol_version,
            "receive_buffer_size": receive_buffer_size,
            "send_buffer_size": send_buffer_size,
            "max_message_size": max_message_size,
            "max_chunk_count": max_chunk_count,
            "endpoint_url": endpoint_url,
        }

    def _parse_acknowledge(self, data: bytes) -> dict[str, Any]:
        """Parse Acknowledge message payload.

        Acknowledge Message Format:
        - ProtocolVersion (4 bytes)
        - ReceiveBufferSize (4 bytes)
        - SendBufferSize (4 bytes)
        - MaxMessageSize (4 bytes)
        - MaxChunkCount (4 bytes)

        Args:
            data: Acknowledge payload.

        Returns:
            Parsed Acknowledge message data.
        """
        if len(data) < 20:
            raise ValueError(f"Acknowledge message too short: {len(data)} bytes (minimum 20)")

        return {
            "protocol_version": int.from_bytes(data[0:4], "little"),
            "receive_buffer_size": int.from_bytes(data[4:8], "little"),
            "send_buffer_size": int.from_bytes(data[8:12], "little"),
            "max_message_size": int.from_bytes(data[12:16], "little"),
            "max_chunk_count": int.from_bytes(data[16:20], "little"),
        }

    def _parse_open_secure_channel(self, data: bytes) -> dict[str, Any]:
        """Parse Open Secure Channel message payload.

        OpenSecureChannel Format:
        - SecureChannelId (4 bytes)
        - SecurityPolicyUri (String)
        - ... (complex, varies by security mode)

        Args:
            data: Open Secure Channel payload.

        Returns:
            Parsed Open Secure Channel data.
        """
        if len(data) < 4:
            raise ValueError("OpenSecureChannel message too short")

        secure_channel_id = int.from_bytes(data[0:4], "little")

        # Security policy URI at offset 4
        security_policy_uri = None
        if len(data) > 4:
            policy_uri, _ = parse_string(data, 4)
            security_policy_uri = policy_uri

        return {
            "secure_channel_id": secure_channel_id,
            "security_policy_uri": security_policy_uri,
        }

    def _parse_message_chunk(self, data: bytes) -> dict[str, Any]:
        """Parse MSG chunk payload containing service request/response.

        MSG Chunk Format:
        - SecureChannelId (4 bytes)
        - SecurityTokenId (4 bytes)
        - SequenceNumber (4 bytes)
        - RequestId (4 bytes)
        - Service payload (varies by service type)

        Args:
            data: MSG chunk payload.

        Returns:
            Parsed MSG chunk data with service information.

        Example:
            >>> analyzer = OPCUAAnalyzer()
            >>> # Simplified MSG chunk
            >>> payload = bytes([0x01, 0x00, 0x00, 0x00, ...])
            >>> msg = analyzer._parse_message_chunk(payload)
            >>> assert 'secure_channel_id' in msg
        """
        if len(data) < 16:
            raise ValueError(f"MSG chunk too short: {len(data)} bytes (minimum 16)")

        secure_channel_id = int.from_bytes(data[0:4], "little")
        security_token_id = int.from_bytes(data[4:8], "little")
        sequence_number = int.from_bytes(data[8:12], "little")
        request_id = int.from_bytes(data[12:16], "little")

        # Service payload starts at offset 16
        service_payload = data[16:]

        # Try to decode service type (NodeId at start of service payload)
        service_id = None
        service_data: dict[str, Any] = {}

        if len(service_payload) >= 4:
            # Service type is encoded as NodeId
            # For simplicity, assume numeric encoding (most common)
            # Real implementation would use parse_node_id
            try:
                # Try to extract service ID (simplified - assumes FourByte encoding)
                if service_payload[0] == 0x01:  # FourByte encoding
                    service_id = int.from_bytes(service_payload[2:4], "little")
                    service_data = self._decode_service(service_id, service_payload[4:])
            except (ValueError, IndexError):
                service_data = {"raw_payload_size": len(service_payload)}

        return {
            "secure_channel_id": secure_channel_id,
            "security_token_id": security_token_id,
            "sequence_number": sequence_number,
            "request_id": request_id,
            "service_id": service_id,
            "service_data": service_data,
        }

    def _parse_error(self, data: bytes) -> dict[str, Any]:
        """Parse Error message payload.

        Error Message Format:
        - Error (4 bytes, StatusCode)
        - Reason (String)

        Args:
            data: Error payload.

        Returns:
            Parsed error data.
        """
        if len(data) < 4:
            raise ValueError("Error message too short")

        error_code = int.from_bytes(data[0:4], "little")

        # Parse reason string
        reason = None
        if len(data) > 4:
            reason_str, _ = parse_string(data, 4)
            reason = reason_str

        return {
            "error_code": error_code,
            "reason": reason,
        }

    def _decode_service(self, service_id: int, payload: bytes) -> dict[str, Any]:
        """Decode service-specific payload.

        Args:
            service_id: Service type identifier.
            payload: Service payload bytes.

        Returns:
            Decoded service data.

        Example:
            >>> analyzer = OPCUAAnalyzer()
            >>> # ReadRequest service
            >>> service_data = analyzer._decode_service(421, b'...')
            >>> assert 'service' in service_data
        """
        # Check if we have a parser for this service
        if service_id in SERVICE_PARSERS:
            parser_request, parser_response = SERVICE_PARSERS[service_id]
            try:
                # Try request parser first (would need context to determine)
                if parser_request is not None:
                    result: dict[str, Any] = parser_request(payload)
                    return result
            except (ValueError, IndexError):
                pass

        # Fallback to basic info
        fallback: dict[str, Any] = {
            "service_id": service_id,
            "service_name": self.SERVICE_IDS.get(service_id, "Unknown"),
            "payload_size": len(payload),
        }
        return fallback

    def export_address_space(self, output_path: Path) -> None:
        """Export discovered address space as JSON.

        Exports all discovered nodes with their properties to a JSON file.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = OPCUAAnalyzer()
            >>> # ... parse messages and discover nodes ...
            >>> analyzer.export_address_space(Path("opcua_nodes.json"))
        """
        export_data = {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "node_class": node.node_class,
                    "browse_name": node.browse_name,
                    "display_name": node.display_name,
                    "value": str(node.value) if node.value is not None else None,
                    "data_type": node.data_type,
                    "children": node.children,
                }
                for node in self.nodes.values()
            ],
            "message_count": len(self.messages),
            "security_mode": self.security_mode,
        }

        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)


__all__ = ["OPCUAAnalyzer", "OPCUAMessage", "OPCUANode"]
