"""MQTT 5.0 property parsing.

This module handles parsing of MQTT 5.0 properties used in variable headers
of control packets.

Example:
    >>> from oscura.iot.mqtt.properties import parse_properties
    >>> properties, consumed = parse_properties(data)

References:
    MQTT 5.0 Section 2.2.2: https://docs.oasis-open.org/mqtt/mqtt/v5.0/
"""

from __future__ import annotations

from typing import Any

# MQTT 5.0 Property identifiers
PROPERTY_IDS = {
    0x01: "payload_format_indicator",
    0x02: "message_expiry_interval",
    0x03: "content_type",
    0x08: "response_topic",
    0x09: "correlation_data",
    0x0B: "subscription_identifier",
    0x11: "session_expiry_interval",
    0x12: "assigned_client_identifier",
    0x13: "server_keep_alive",
    0x15: "authentication_method",
    0x16: "authentication_data",
    0x17: "request_problem_information",
    0x18: "will_delay_interval",
    0x19: "request_response_information",
    0x1A: "response_information",
    0x1C: "server_reference",
    0x1F: "reason_string",
    0x21: "receive_maximum",
    0x22: "topic_alias_maximum",
    0x23: "topic_alias",
    0x24: "maximum_qos",
    0x25: "retain_available",
    0x26: "user_property",
    0x27: "maximum_packet_size",
    0x28: "wildcard_subscription_available",
    0x29: "subscription_identifier_available",
    0x2A: "shared_subscription_available",
}


def _decode_variable_byte_integer(data: bytes, offset: int = 0) -> tuple[int, int]:
    """Decode variable byte integer from MQTT data.

    MQTT uses a variable length encoding scheme for integers. Each byte
    encodes 7 bits of data and 1 continuation bit.

    Args:
        data: Raw byte data.
        offset: Starting offset in data.

    Returns:
        Tuple of (decoded_value, bytes_consumed).

    Raises:
        ValueError: If data is insufficient or encoding is invalid.

    Example:
        >>> _decode_variable_byte_integer(b"\\x7f", 0)
        (127, 1)
        >>> _decode_variable_byte_integer(b"\\x80\\x01", 0)
        (128, 2)
    """
    multiplier = 1
    value = 0
    index = offset

    while True:
        if index >= len(data):
            raise ValueError("Incomplete variable byte integer")

        encoded_byte = data[index]
        value += (encoded_byte & 0x7F) * multiplier

        if (encoded_byte & 0x80) == 0:
            break

        multiplier *= 128
        index += 1

        if multiplier > 128 * 128 * 128:
            raise ValueError("Variable byte integer exceeds maximum")

    return value, index - offset + 1


def _decode_utf8_string(data: bytes, offset: int = 0) -> tuple[str, int]:
    """Decode UTF-8 encoded string from MQTT data.

    Args:
        data: Raw byte data.
        offset: Starting offset in data.

    Returns:
        Tuple of (decoded_string, bytes_consumed).

    Raises:
        ValueError: If data is insufficient or encoding is invalid.

    Example:
        >>> _decode_utf8_string(b"\\x00\\x05Hello", 0)
        ('Hello', 7)
    """
    if len(data) < offset + 2:
        raise ValueError("Insufficient data for string length")

    str_len = int.from_bytes(data[offset : offset + 2], "big")
    if len(data) < offset + 2 + str_len:
        raise ValueError("Insufficient data for string content")

    string_bytes = data[offset + 2 : offset + 2 + str_len]
    try:
        string_value = string_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Invalid UTF-8 encoding: {e}") from e

    return string_value, 2 + str_len


def _decode_binary_data(data: bytes, offset: int = 0) -> tuple[bytes, int]:
    """Decode binary data from MQTT data.

    Args:
        data: Raw byte data.
        offset: Starting offset in data.

    Returns:
        Tuple of (binary_data, bytes_consumed).

    Raises:
        ValueError: If data is insufficient.

    Example:
        >>> _decode_binary_data(b"\\x00\\x03ABC", 0)
        (b'ABC', 5)
    """
    if len(data) < offset + 2:
        raise ValueError("Insufficient data for binary length")

    bin_len = int.from_bytes(data[offset : offset + 2], "big")
    if len(data) < offset + 2 + bin_len:
        raise ValueError("Insufficient data for binary content")

    binary_value = data[offset + 2 : offset + 2 + bin_len]
    return binary_value, 2 + bin_len


def parse_properties(data: bytes, offset: int = 0) -> tuple[dict[str, Any], int]:
    """Parse MQTT 5.0 properties from packet data.

    Properties are encoded as: property_length (variable byte integer),
    followed by property_id (byte) and property_value pairs.

    Args:
        data: Raw packet data containing properties.
        offset: Starting offset in data.

    Returns:
        Tuple of (properties_dict, total_bytes_consumed).

    Raises:
        ValueError: If properties are malformed or incomplete.

    Example:
        >>> # Parse properties with message expiry interval (0x02)
        >>> data = b"\\x05\\x02\\x00\\x00\\x00\\x3c"  # 60 seconds
        >>> props, consumed = parse_properties(data)
        >>> props["message_expiry_interval"]
        60
    """
    properties: dict[str, Any] = {}

    # Decode properties length
    try:
        props_len, len_bytes = _decode_variable_byte_integer(data, offset)
    except ValueError as e:
        raise ValueError(f"Failed to decode properties length: {e}") from e

    total_consumed = len_bytes
    props_end = offset + len_bytes + props_len

    if len(data) < props_end:
        raise ValueError("Insufficient data for properties")

    current = offset + len_bytes

    # Parse individual properties
    while current < props_end:
        current = _parse_single_property(data, current, props_end, properties)

    total_consumed += props_len
    return properties, total_consumed


def _parse_single_property(
    data: bytes, current: int, props_end: int, properties: dict[str, Any]
) -> int:
    """Parse a single MQTT 5.0 property.

    Args:
        data: Raw packet data.
        current: Current offset in data.
        props_end: End offset of properties section.
        properties: Dictionary to update with parsed property.

    Returns:
        Updated offset after parsing property.

    Raises:
        ValueError: If property is malformed or unknown.
    """
    if current >= len(data):
        raise ValueError("Unexpected end of properties data")

    prop_id = data[current]
    current += 1

    if prop_id not in PROPERTY_IDS:
        raise ValueError(f"Unknown property ID: 0x{prop_id:02X}")

    prop_name = PROPERTY_IDS[prop_id]

    # Decode property value based on type
    try:
        return _decode_property_value(data, current, prop_id, prop_name, properties)
    except ValueError as e:
        raise ValueError(f"Failed to parse property {prop_name}: {e}") from e


def _decode_property_value(
    data: bytes, current: int, prop_id: int, prop_name: str, properties: dict[str, Any]
) -> int:
    """Decode property value based on property ID type.

    Args:
        data: Raw packet data.
        current: Current offset in data.
        prop_id: Property identifier.
        prop_name: Property name string.
        properties: Dictionary to update with decoded value.

    Returns:
        Updated offset after decoding value.

    Raises:
        ValueError: If data is insufficient or type is unhandled.
    """
    # Byte values
    if prop_id in [0x01, 0x24, 0x25, 0x28, 0x29, 0x2A]:
        if current >= len(data):
            raise ValueError(f"Insufficient data for property {prop_name}")
        properties[prop_name] = data[current]
        return current + 1

    # Two-byte integer values
    if prop_id in [0x13, 0x21, 0x22, 0x23]:
        if current + 2 > len(data):
            raise ValueError(f"Insufficient data for property {prop_name}")
        properties[prop_name] = int.from_bytes(data[current : current + 2], "big")
        return current + 2

    # Four-byte integer values
    if prop_id in [0x02, 0x11, 0x18, 0x27]:
        if current + 4 > len(data):
            raise ValueError(f"Insufficient data for property {prop_name}")
        properties[prop_name] = int.from_bytes(data[current : current + 4], "big")
        return current + 4

    # UTF-8 string values
    if prop_id in [0x03, 0x08, 0x12, 0x15, 0x1A, 0x1C, 0x1F]:
        string_value, consumed = _decode_utf8_string(data, current)
        properties[prop_name] = string_value
        return current + consumed

    # Binary data values
    if prop_id in [0x09, 0x16]:
        binary_value, consumed = _decode_binary_data(data, current)
        properties[prop_name] = binary_value
        return current + consumed

    # Subscription identifier (variable byte integer)
    if prop_id == 0x0B:
        sub_id, consumed = _decode_variable_byte_integer(data, current)
        properties[prop_name] = sub_id
        return current + consumed

    # User property (key-value pair)
    if prop_id == 0x26:
        if "user_property" not in properties:
            properties["user_property"] = []

        key, consumed1 = _decode_utf8_string(data, current)
        current += consumed1

        value, consumed2 = _decode_utf8_string(data, current)
        current += consumed2

        properties["user_property"].append((key, value))
        return current

    raise ValueError(f"Unhandled property type: 0x{prop_id:02X}")


__all__ = [
    "PROPERTY_IDS",
    "parse_properties",
]
