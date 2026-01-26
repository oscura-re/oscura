"""CoAP protocol analyzer with RFC 7252 support and extensions.

Provides comprehensive CoAP message parsing, request/response matching,
blockwise transfer support (RFC 7959), and observe extension (RFC 7641).

Example:
    >>> from oscura.iot.coap import CoAPAnalyzer
    >>> analyzer = CoAPAnalyzer()
    >>> data = bytes([0x40, 0x01, 0x12, 0x34])  # CON GET
    >>> message = analyzer.parse_message(data, timestamp=0.0)
    >>> print(message.msg_type, message.code)
    CON GET

References:
    RFC 7252: CoAP (Constrained Application Protocol)
    RFC 7959: Block-Wise Transfers in CoAP
    RFC 7641: Observing Resources in CoAP
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from oscura.iot.coap.options import (
    CONTENT_FORMATS,
    OPTIONS,
    OptionParser,
    format_block_option,
)


@dataclass
class CoAPMessage:
    """CoAP message representation.

    Attributes:
        timestamp: Message timestamp in seconds.
        version: CoAP version (always 1 for RFC 7252).
        msg_type: Message type ("CON", "NON", "ACK", "RST").
        code: Method code or response code (e.g., "GET", "2.05 Content").
        message_id: 16-bit message identifier for duplicate detection.
        token: Token for matching requests/responses (0-8 bytes).
        options: Parsed options dictionary (option name -> list of values).
        payload: Message payload bytes.
        is_request: True if request, False if response.
        uri: Reconstructed URI from Uri-* options (optional).

    Example:
        >>> msg = CoAPMessage(
        ...     timestamp=1.0,
        ...     version=1,
        ...     msg_type="CON",
        ...     code="GET",
        ...     message_id=0x1234,
        ...     token=b"\\x01",
        ... )
        >>> msg.is_request
        True
    """

    timestamp: float
    version: int
    msg_type: str
    code: str
    message_id: int
    token: bytes
    options: dict[str, list[Any]] = field(default_factory=dict)
    payload: bytes = b""
    is_request: bool = True
    uri: str | None = None


@dataclass
class CoAPExchange:
    """CoAP request-response exchange.

    Represents a complete request-response transaction, including
    support for multiple responses (observe pattern).

    Attributes:
        request: Initial CoAP request message.
        responses: List of response messages (multiple for observe).
        complete: True if exchange is complete (no more responses expected).
        observe: True if this is an observe relationship.

    Example:
        >>> request = CoAPMessage(...)
        >>> exchange = CoAPExchange(request=request)
        >>> exchange.responses.append(response_msg)
    """

    request: CoAPMessage
    responses: list[CoAPMessage] = field(default_factory=list)
    complete: bool = False
    observe: bool = False


class CoAPAnalyzer:
    """CoAP protocol analyzer supporting RFC 7252 and extensions.

    Analyzes CoAP messages including parsing message format, decoding options,
    matching requests with responses, and handling blockwise transfers and
    observe relationships.

    Attributes:
        MSG_TYPES: Message type code to name mapping.
        METHODS: Method code to name mapping.
        RESPONSE_CLASSES: Response class descriptions.

    Example:
        >>> analyzer = CoAPAnalyzer()
        >>> msg = analyzer.parse_message(data, timestamp=0.0)
        >>> analyzer.match_request_response()
        >>> analyzer.export_exchanges(Path("coap_traffic.json"))
    """

    # Message types (RFC 7252 Section 3)
    MSG_TYPES: ClassVar[dict[int, str]] = {
        0: "CON",  # Confirmable
        1: "NON",  # Non-confirmable
        2: "ACK",  # Acknowledgement
        3: "RST",  # Reset
    }

    # Method codes 0.xx (RFC 7252 Section 5.8)
    METHODS: ClassVar[dict[int, str]] = {
        1: "GET",
        2: "POST",
        3: "PUT",
        4: "DELETE",
        5: "FETCH",  # RFC 8132
        6: "PATCH",  # RFC 8132
        7: "iPATCH",  # RFC 8132
    }

    # Response code classes
    RESPONSE_CLASSES: ClassVar[dict[int, str]] = {
        2: "Success",
        4: "Client Error",
        5: "Server Error",
    }

    def __init__(self) -> None:
        """Initialize CoAP analyzer.

        Example:
            >>> analyzer = CoAPAnalyzer()
            >>> len(analyzer.messages)
            0
        """
        self.messages: list[CoAPMessage] = []
        self.exchanges: dict[bytes, CoAPExchange] = {}  # Token -> exchange
        self.message_id_map: dict[int, CoAPMessage] = {}  # Message ID -> message
        self.option_parser = OptionParser()

    def parse_message(self, data: bytes, timestamp: float = 0.0) -> CoAPMessage:
        """Parse CoAP message from bytes.

        Parses complete CoAP message including header, token, options, and payload
        according to RFC 7252 Section 3.

        Message Format:
         0                   1                   2                   3
         0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |Ver| T |  TKL  |      Code     |          Message ID           |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |   Token (if any, TKL bytes) ...
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |   Options (if any) ...
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |1 1 1 1 1 1 1 1|    Payload (if any) ...
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

        Args:
            data: Raw CoAP message bytes.
            timestamp: Message timestamp in seconds.

        Returns:
            Parsed CoAP message.

        Raises:
            ValueError: If message is malformed or too short.

        Example:
            >>> analyzer = CoAPAnalyzer()
            >>> data = bytes([0x40, 0x01, 0x12, 0x34])  # CON GET
            >>> msg = analyzer.parse_message(data)
            >>> msg.msg_type
            'CON'
            >>> msg.code
            'GET'
        """
        if len(data) < 4:
            raise ValueError(f"CoAP message too short: {len(data)} bytes (minimum 4)")

        # Parse header and token
        version, msg_type, code_str, is_request, message_id, token, offset = (
            self._parse_header_and_token(data)
        )

        # Parse options
        options = self._parse_options(data, offset)

        # Find payload boundary
        payload_start = self._find_payload_start(data, offset)

        # Extract payload
        payload = data[payload_start:] if payload_start < len(data) else b""

        # Reconstruct URI from options
        uri = self._reconstruct_uri(options)

        message = CoAPMessage(
            timestamp=timestamp,
            version=version,
            msg_type=msg_type,
            code=code_str,
            message_id=message_id,
            token=token,
            options=options,
            payload=payload,
            is_request=is_request,
            uri=uri,
        )

        self.messages.append(message)
        self.message_id_map[message_id] = message

        return message

    def _parse_header_and_token(self, data: bytes) -> tuple[int, str, str, bool, int, bytes, int]:
        """Parse CoAP header and token fields.

        Args:
            data: Raw message bytes.

        Returns:
            Tuple of (version, msg_type, code_str, is_request, message_id, token, offset).

        Raises:
            ValueError: If header or token is invalid.
        """
        byte0 = data[0]
        version = (byte0 >> 6) & 0x03
        msg_type_val = (byte0 >> 4) & 0x03
        tkl = byte0 & 0x0F

        if version != 1:
            raise ValueError(f"Unsupported CoAP version: {version} (expected 1)")

        if tkl > 8:
            raise ValueError(f"Invalid token length: {tkl} (maximum 8)")

        code_byte = data[1]
        message_id = int.from_bytes(data[2:4], "big")

        msg_type = self.MSG_TYPES.get(msg_type_val, f"UNKNOWN({msg_type_val})")
        code_str, is_request = self._parse_code(code_byte)

        offset = 4

        # Parse token (TKL bytes)
        token = b""
        if tkl > 0:
            if len(data) < offset + tkl:
                raise ValueError(f"Insufficient data for token: need {tkl} bytes")
            token = data[offset : offset + tkl]
            offset += tkl

        return version, msg_type, code_str, is_request, message_id, token, offset

    def _find_payload_start(self, data: bytes, start_offset: int) -> int:
        """Find start of payload after options.

        Args:
            data: Complete message data.
            start_offset: Offset where options start.

        Returns:
            Offset where payload starts.

        Raises:
            ValueError: If option encoding is invalid.
        """
        payload_start = start_offset
        while payload_start < len(data):
            if data[payload_start] == 0xFF:
                payload_start += 1  # Skip marker
                break
            if payload_start >= len(data):
                break

            option_byte = data[payload_start]
            delta = (option_byte >> 4) & 0x0F
            length = option_byte & 0x0F

            payload_start += 1

            # Handle extended delta
            if delta == 13:
                payload_start += 1
            elif delta == 14:
                payload_start += 2
            elif delta == 15:
                break  # Payload marker

            # Handle extended length
            if length == 13:
                payload_start += 1
            elif length == 14:
                payload_start += 2
            elif length == 15:
                raise ValueError("Invalid option length encoding (15)")

            # Skip option value
            if delta < 15:
                actual_length = self._calculate_option_length(data, payload_start, length)
                payload_start += actual_length

        return payload_start

    def _calculate_option_length(self, data: bytes, offset: int, length_base: int) -> int:
        """Calculate actual option length from extended encoding.

        Args:
            data: Message data.
            offset: Current offset (after length encoding bytes).
            length_base: Base length value from option byte.

        Returns:
            Actual option length in bytes.
        """
        if length_base < 13:
            return length_base
        elif length_base == 13 and offset <= len(data):
            return data[offset - 1] + 13
        elif length_base == 14 and offset + 1 <= len(data):
            return int.from_bytes(data[offset - 2 : offset], "big") + 269
        return 0

    def _parse_code(self, code: int) -> tuple[str, bool]:
        """Parse code byte into human-readable string and request flag.

        Code byte format: class (3 bits) . detail (5 bits)
        - 0.xx: Request methods
        - 2.xx: Success responses
        - 4.xx: Client error responses
        - 5.xx: Server error responses

        Args:
            code: Code byte value (0-255).

        Returns:
            Tuple of (code_string, is_request).

        Example:
            >>> analyzer = CoAPAnalyzer()
            >>> analyzer._parse_code(0x01)
            ('GET', True)
            >>> analyzer._parse_code(0x45)
            ('2.05 Content', False)
        """
        code_class = (code >> 5) & 0x07
        code_detail = code & 0x1F

        if code_class == 0:
            # Request method
            method = self.METHODS.get(code_detail, f"0.{code_detail:02d}")
            return method, True

        # Response code
        code_str = f"{code_class}.{code_detail:02d}"

        # Add common response names
        response_names = {
            0x41: "2.01 Created",
            0x42: "2.02 Deleted",
            0x43: "2.03 Valid",
            0x44: "2.04 Changed",
            0x45: "2.05 Content",
            0x5F: "2.31 Continue",
            0x80: "4.00 Bad Request",
            0x81: "4.01 Unauthorized",
            0x82: "4.02 Bad Option",
            0x83: "4.03 Forbidden",
            0x84: "4.04 Not Found",
            0x85: "4.05 Method Not Allowed",
            0x86: "4.06 Not Acceptable",
            0x8C: "4.12 Precondition Failed",
            0x8D: "4.13 Request Entity Too Large",
            0x8F: "4.15 Unsupported Content-Format",
            0xA0: "5.00 Internal Server Error",
            0xA1: "5.01 Not Implemented",
            0xA2: "5.02 Bad Gateway",
            0xA3: "5.03 Service Unavailable",
            0xA4: "5.04 Gateway Timeout",
            0xA5: "5.05 Proxying Not Supported",
        }

        return response_names.get(code, code_str), False

    def _parse_options(self, data: bytes, start_offset: int) -> dict[str, list[Any]]:
        """Parse CoAP options using delta encoding.

        Options are encoded with delta encoding where each option number
        is the sum of all previous deltas. Handles extended delta/length
        encoding for values >= 13.

        Args:
            data: Complete message data.
            start_offset: Offset where options start.

        Returns:
            Dictionary mapping option names to lists of decoded values.

        Raises:
            ValueError: If option encoding is invalid.

        Example:
            >>> analyzer = CoAPAnalyzer()
            >>> # Parse message with Uri-Path option
            >>> options = analyzer._parse_options(data, 5)
            >>> options.get("Uri-Path", [])
            ['temperature', 'sensor1']
        """
        options: dict[str, list[Any]] = {}
        offset = start_offset
        current_option_num = 0

        while offset < len(data):
            # Check for payload marker
            if data[offset] == 0xFF:
                break

            if offset >= len(data):
                break

            option_byte = data[offset]
            delta_base = (option_byte >> 4) & 0x0F
            length_base = option_byte & 0x0F
            offset += 1

            # Check for payload marker in delta
            if delta_base == 15:
                offset -= 1  # Back up to marker
                break

            # Parse extended delta
            try:
                delta, delta_bytes = self.option_parser.parse_extended_value(
                    delta_base, data, offset
                )
                offset += delta_bytes
            except ValueError as e:
                raise ValueError(f"Failed to parse option delta: {e}") from e

            # Parse extended length
            try:
                length, length_bytes = self.option_parser.parse_extended_value(
                    length_base, data, offset
                )
                offset += length_bytes
            except ValueError as e:
                raise ValueError(f"Failed to parse option length: {e}") from e

            # Calculate actual option number
            current_option_num += delta

            # Extract option value
            if len(data) < offset + length:
                raise ValueError(
                    f"Insufficient data for option value: need {length} bytes at offset {offset}"
                )

            option_value = data[offset : offset + length]
            offset += length

            # Decode option value
            decoded_value = self.option_parser.decode_value(current_option_num, option_value)

            # Store option
            option_name = OPTIONS.get(current_option_num, f"Option-{current_option_num}")

            if option_name not in options:
                options[option_name] = []

            options[option_name].append(decoded_value)

        return options

    def _reconstruct_uri(self, options: dict[str, list[Any]]) -> str | None:
        """Reconstruct URI from Uri-* options.

        Combines Uri-Host, Uri-Port, Uri-Path, and Uri-Query options
        into a complete URI string.

        Args:
            options: Parsed options dictionary.

        Returns:
            Reconstructed URI string, or None if no Uri options present.

        Example:
            >>> analyzer = CoAPAnalyzer()
            >>> options = {
            ...     "Uri-Host": ["example.com"],
            ...     "Uri-Path": ["sensors", "temperature"],
            ...     "Uri-Query": ["format=json"],
            ... }
            >>> analyzer._reconstruct_uri(options)
            'coap://example.com/sensors/temperature?format=json'
        """
        if not any(key.startswith("Uri-") for key in options):
            return None

        # Build URI components
        host = options.get("Uri-Host", [None])[0]
        port = options.get("Uri-Port", [5683])[0]  # Default CoAP port
        path_segments = options.get("Uri-Path", [])
        query_params = options.get("Uri-Query", [])

        # Construct URI
        uri_parts = []

        if host:
            scheme = "coap"
            if isinstance(port, int) and port != 5683:
                uri_parts.append(f"{scheme}://{host}:{port}")
            else:
                uri_parts.append(f"{scheme}://{host}")
        else:
            uri_parts.append("coap://")

        # Add path
        if path_segments:
            path = "/" + "/".join(str(seg) for seg in path_segments)
            uri_parts.append(path)

        # Add query string
        if query_params:
            query = "&".join(str(param) for param in query_params)
            uri_parts.append(f"?{query}")

        return "".join(uri_parts)

    def match_request_response(self) -> None:
        """Match requests with responses by token and message ID.

        Creates CoAPExchange objects linking requests with their responses.
        Supports observe relationships where multiple responses map to one request.

        Example:
            >>> analyzer = CoAPAnalyzer()
            >>> # Parse messages...
            >>> analyzer.match_request_response()
            >>> len(analyzer.exchanges)
            5
        """
        # Clear existing exchanges
        self.exchanges = {}

        for message in self.messages:
            if message.is_request:
                # Create new exchange for request
                observe = "Observe" in message.options
                exchange = CoAPExchange(request=message, observe=observe)
                self.exchanges[message.token] = exchange
            else:
                # Match response to request by token
                if message.token in self.exchanges:
                    exchange = self.exchanges[message.token]
                    exchange.responses.append(message)

                    # Mark complete if not observe or if ACK/RST
                    if not exchange.observe or message.msg_type in ("ACK", "RST"):
                        exchange.complete = True

    def export_exchanges(self, output_path: Path) -> None:
        """Export request-response exchanges as JSON.

        Exports all matched exchanges including request, responses, timing,
        and decoded options.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = CoAPAnalyzer()
            >>> # Parse and match messages...
            >>> analyzer.export_exchanges(Path("coap_exchanges.json"))
        """
        export_data: dict[str, Any] = {
            "summary": {
                "total_messages": len(self.messages),
                "total_exchanges": len(self.exchanges),
                "complete_exchanges": sum(1 for ex in self.exchanges.values() if ex.complete),
            },
            "exchanges": [],
        }

        for token, exchange in self.exchanges.items():
            request_data = self._message_to_dict(exchange.request)

            responses_data = [self._message_to_dict(resp) for resp in exchange.responses]

            exchange_entry = {
                "token": token.hex() if token else "",
                "observe": exchange.observe,
                "complete": exchange.complete,
                "request": request_data,
                "responses": responses_data,
                "response_count": len(exchange.responses),
            }

            export_data["exchanges"].append(exchange_entry)

        # Write JSON
        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)

    def _message_to_dict(self, message: CoAPMessage) -> dict[str, Any]:
        """Convert CoAPMessage to dictionary for JSON export.

        Args:
            message: CoAP message to convert.

        Returns:
            Dictionary representation of message.
        """
        # Format options for readability
        formatted_options: dict[str, list[Any]] = {}
        for name, values in message.options.items():
            formatted_values: list[Any] = []
            for value in values:
                if isinstance(value, bytes):
                    formatted_values.append(value.hex())
                elif isinstance(value, int) and name == "Content-Format":
                    # Add content format name
                    format_name = CONTENT_FORMATS.get(value, "unknown")
                    formatted_values.append({"code": value, "format": format_name})
                elif isinstance(value, int) and name in ("Block1", "Block2"):
                    # Decode block option
                    block_info = format_block_option(value)
                    formatted_values.append(block_info)
                else:
                    formatted_values.append(value)
            formatted_options[name] = formatted_values

        return {
            "timestamp": message.timestamp,
            "type": message.msg_type,
            "code": message.code,
            "message_id": f"0x{message.message_id:04X}",
            "token": message.token.hex() if message.token else "",
            "uri": message.uri,
            "options": formatted_options,
            "payload_length": len(message.payload),
            "payload": message.payload.hex()
            if len(message.payload) <= 64
            else f"{message.payload[:64].hex()}... ({len(message.payload)} bytes)",
        }


__all__ = ["CoAPAnalyzer", "CoAPExchange", "CoAPMessage"]
