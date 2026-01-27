"""OPC UA data type parsers.

This module implements parsing for OPC UA built-in data types including
NodeId, Variant, String, DateTime, and other primitives.

References:
    OPC UA Part 6: Mappings - Section 5.1 Built-in Types
    https://reference.opcfoundation.org/Core/Part6/v105/docs/5.1
"""

from __future__ import annotations

from typing import Any


def parse_string(data: bytes, offset: int) -> tuple[str | None, int]:
    """Parse OPC UA String (length-prefixed UTF-8).

    String Format:
    - Length (4 bytes, little-endian, -1 for null)
    - Data (Length bytes, UTF-8 encoded)

    Args:
        data: Binary data containing the string.
        offset: Starting offset in data.

    Returns:
        Tuple of (parsed_string, bytes_consumed).
        Returns (None, 4) for null string.

    Example:
        >>> data = b'\\x05\\x00\\x00\\x00Hello'
        >>> s, consumed = parse_string(data, 0)
        >>> assert s == "Hello"
        >>> assert consumed == 9
    """
    if offset + 4 > len(data):
        return None, 0

    length = int.from_bytes(data[offset : offset + 4], "little", signed=True)
    consumed = 4

    if length == -1:
        # Null string
        return None, consumed

    if length < 0:
        # Invalid length
        return None, consumed

    if offset + 4 + length > len(data):
        # Not enough data
        return None, consumed

    try:
        string_value = data[offset + 4 : offset + 4 + length].decode("utf-8")
    except UnicodeDecodeError:
        # Invalid UTF-8
        string_value = data[offset + 4 : offset + 4 + length].decode("utf-8", errors="replace")

    consumed += length
    return string_value, consumed


def parse_node_id(data: bytes, offset: int) -> tuple[str, int]:
    """Parse OPC UA NodeId.

    NodeId Encoding:
    - EncodingByte (1 byte):
      * Bits 0-5: Encoding type
        - 0x00: TwoByte (ns=0, numeric id < 256)
        - 0x01: FourByte (ns < 256, numeric id < 65536)
        - 0x02: Numeric (full 32-bit namespace and identifier)
        - 0x03: String (namespace + UTF-8 string)
        - 0x04: Guid (namespace + 16-byte GUID)
        - 0x05: ByteString (namespace + byte string)
      * Bits 6-7: Namespace URI and Server Index flags
    - Namespace (varies by encoding)
    - Identifier (varies by encoding)

    Args:
        data: Binary data containing the NodeId.
        offset: Starting offset in data.

    Returns:
        Tuple of (node_id_string, bytes_consumed).
        NodeId string formats:
        - Numeric: "ns=X;i=Y" or "i=Y" (if ns=0)
        - String: "ns=X;s=string"
        - Guid: "ns=X;g=guid"
        - ByteString: "ns=X;b=base64"

    Example:
        >>> # FourByte numeric NodeId: ns=2, id=1001
        >>> data = bytes([0x01, 0x02, 0xE9, 0x03])
        >>> node_id, consumed = parse_node_id(data, 0)
        >>> assert node_id == "ns=2;i=1001"
        >>> assert consumed == 4
    """
    if offset >= len(data):
        return "i=0", 0

    encoding_byte = data[offset]
    encoding_type = encoding_byte & 0x3F
    consumed = 1

    if encoding_type == 0x00:
        return _parse_twobyte_nodeid(data, offset, consumed)
    elif encoding_type == 0x01:
        return _parse_fourbyte_nodeid(data, offset, consumed)
    elif encoding_type == 0x02:
        return _parse_numeric_nodeid(data, offset, consumed)
    elif encoding_type == 0x03:
        return _parse_string_nodeid(data, offset, consumed)
    elif encoding_type == 0x04:
        return _parse_guid_nodeid(data, offset, consumed)
    elif encoding_type == 0x05:
        return _parse_bytestring_nodeid(data, offset, consumed)
    else:
        return "i=0", consumed


def _parse_twobyte_nodeid(data: bytes, offset: int, consumed: int) -> tuple[str, int]:
    """Parse TwoByte NodeId (ns=0, identifier < 256).

    Args:
        data: Binary data.
        offset: Starting offset.
        consumed: Bytes already consumed.

    Returns:
        Tuple of (node_id_string, total_bytes_consumed).
    """
    if offset + 1 >= len(data):
        return "i=0", consumed
    identifier = data[offset + 1]
    return f"i={identifier}", consumed + 1


def _parse_fourbyte_nodeid(data: bytes, offset: int, consumed: int) -> tuple[str, int]:
    """Parse FourByte NodeId (ns < 256, identifier < 65536).

    Args:
        data: Binary data.
        offset: Starting offset.
        consumed: Bytes already consumed.

    Returns:
        Tuple of (node_id_string, total_bytes_consumed).
    """
    if offset + 3 >= len(data):
        return "i=0", consumed

    namespace = data[offset + 1]
    identifier = int.from_bytes(data[offset + 2 : offset + 4], "little")
    consumed += 3

    return _format_numeric_nodeid(namespace, identifier), consumed


def _parse_numeric_nodeid(data: bytes, offset: int, consumed: int) -> tuple[str, int]:
    """Parse Numeric NodeId (full 32-bit namespace and identifier).

    Args:
        data: Binary data.
        offset: Starting offset.
        consumed: Bytes already consumed.

    Returns:
        Tuple of (node_id_string, total_bytes_consumed).
    """
    if offset + 6 >= len(data):
        return "i=0", consumed

    namespace = int.from_bytes(data[offset + 1 : offset + 3], "little")
    identifier = int.from_bytes(data[offset + 3 : offset + 7], "little")
    consumed += 6

    return _format_numeric_nodeid(namespace, identifier), consumed


def _parse_string_nodeid(data: bytes, offset: int, consumed: int) -> tuple[str, int]:
    """Parse String NodeId (namespace + UTF-8 string).

    Args:
        data: Binary data.
        offset: Starting offset.
        consumed: Bytes already consumed.

    Returns:
        Tuple of (node_id_string, total_bytes_consumed).
    """
    if offset + 2 >= len(data):
        return "i=0", consumed

    namespace = int.from_bytes(data[offset + 1 : offset + 3], "little")
    consumed += 2

    string_value, string_consumed = parse_string(data, offset + consumed)
    consumed += string_consumed

    if string_value is None:
        return "i=0", consumed

    if namespace == 0:
        return f"s={string_value}", consumed
    return f"ns={namespace};s={string_value}", consumed


def _parse_guid_nodeid(data: bytes, offset: int, consumed: int) -> tuple[str, int]:
    """Parse Guid NodeId (namespace + 16-byte GUID).

    Args:
        data: Binary data.
        offset: Starting offset.
        consumed: Bytes already consumed.

    Returns:
        Tuple of (node_id_string, total_bytes_consumed).
    """
    if offset + 18 >= len(data):
        return "i=0", consumed

    namespace = int.from_bytes(data[offset + 1 : offset + 3], "little")
    guid_bytes = data[offset + 3 : offset + 19]
    consumed += 18
    guid_str = guid_bytes.hex()

    if namespace == 0:
        return f"g={guid_str}", consumed
    return f"ns={namespace};g={guid_str}", consumed


def _parse_bytestring_nodeid(data: bytes, offset: int, consumed: int) -> tuple[str, int]:
    """Parse ByteString NodeId (namespace + length-prefixed byte string).

    Args:
        data: Binary data.
        offset: Starting offset.
        consumed: Bytes already consumed.

    Returns:
        Tuple of (node_id_string, total_bytes_consumed).
    """
    if offset + 2 >= len(data):
        return "i=0", consumed

    namespace = int.from_bytes(data[offset + 1 : offset + 3], "little")
    consumed += 2

    bs_value, bs_consumed = parse_string(data, offset + consumed)
    consumed += bs_consumed

    if bs_value is None:
        return "i=0", consumed

    if namespace == 0:
        return f"b={bs_value}", consumed
    return f"ns={namespace};b={bs_value}", consumed


def _format_numeric_nodeid(namespace: int, identifier: int) -> str:
    """Format numeric NodeId as string.

    Args:
        namespace: Namespace index.
        identifier: Numeric identifier.

    Returns:
        Formatted NodeId string.
    """
    if namespace == 0:
        return f"i={identifier}"
    return f"ns={namespace};i={identifier}"


def parse_variant(data: bytes, offset: int) -> tuple[Any, int]:
    """Parse OPC UA Variant data type.

    Variant Encoding:
    - EncodingByte (1 byte):
      * Bits 0-5: Data type
      * Bit 6: Array flag
      * Bit 7: Array dimensions flag
    - Value (varies by type)
    - Array length (if array flag set)
    - Array dimensions (if dimensions flag set)

    Built-in type IDs:
    - 1: Boolean
    - 2: SByte
    - 3: Byte
    - 4: Int16
    - 5: UInt16
    - 6: Int32
    - 7: UInt32
    - 8: Int64
    - 9: UInt64
    - 10: Float
    - 11: Double
    - 12: String
    - 13: DateTime
    - 15: Guid
    - 17: NodeId
    - 22: LocalizedText

    Args:
        data: Binary data containing the variant.
        offset: Starting offset in data.

    Returns:
        Tuple of (parsed_value, bytes_consumed).

    Example:
        >>> # UInt32 variant
        >>> data = bytes([0x07, 0x2A, 0x00, 0x00, 0x00])
        >>> value, consumed = parse_variant(data, 0)
        >>> assert value == 42
        >>> assert consumed == 5
    """
    if offset >= len(data):
        return None, 0

    encoding_byte = data[offset]
    type_id = encoding_byte & 0x3F
    is_array = bool(encoding_byte & 0x40)
    consumed = 1

    # Handle arrays (simplified - just return indication)
    if is_array:
        return {"array": True, "type_id": type_id}, consumed

    # Parse scalar value based on type
    value, bytes_read = _parse_variant_scalar(data, offset + 1, type_id)
    consumed += bytes_read

    return value, consumed


def _parse_variant_scalar(data: bytes, offset: int, type_id: int) -> tuple[Any, int]:
    """Parse a scalar variant value.

    Args:
        data: Binary data.
        offset: Starting offset (after encoding byte).
        type_id: Variant type ID.

    Returns:
        Tuple of (parsed_value, bytes_consumed).
    """
    # Boolean (1 byte)
    if type_id == 1:
        return _parse_boolean_variant(data, offset)

    # Byte (1 byte)
    elif type_id == 3:
        return _parse_byte_variant(data, offset)

    # Int16 (2 bytes)
    elif type_id == 4:
        return _parse_int16_variant(data, offset)

    # UInt16 (2 bytes)
    elif type_id == 5:
        return _parse_uint16_variant(data, offset)

    # Int32 (4 bytes)
    elif type_id == 6:
        return _parse_int32_variant(data, offset)

    # UInt32 (4 bytes)
    elif type_id == 7:
        return _parse_uint32_variant(data, offset)

    # String (length-prefixed)
    elif type_id == 12:
        return _parse_string_variant(data, offset)

    # NodeId
    elif type_id == 17:
        return _parse_nodeid_variant(data, offset)

    # Unsupported types - return type indicator
    else:
        return {"type_id": type_id, "unsupported": True}, 0


def _parse_boolean_variant(data: bytes, offset: int) -> tuple[bool | None, int]:
    """Parse Boolean variant (1 byte)."""
    if offset < len(data):
        return bool(data[offset]), 1
    return None, 0


def _parse_byte_variant(data: bytes, offset: int) -> tuple[int | None, int]:
    """Parse Byte variant (1 byte)."""
    if offset < len(data):
        return data[offset], 1
    return None, 0


def _parse_int16_variant(data: bytes, offset: int) -> tuple[int | None, int]:
    """Parse Int16 variant (2 bytes)."""
    if offset + 1 < len(data):
        value = int.from_bytes(data[offset : offset + 2], "little", signed=True)
        return value, 2
    return None, 0


def _parse_uint16_variant(data: bytes, offset: int) -> tuple[int | None, int]:
    """Parse UInt16 variant (2 bytes)."""
    if offset + 1 < len(data):
        value = int.from_bytes(data[offset : offset + 2], "little")
        return value, 2
    return None, 0


def _parse_int32_variant(data: bytes, offset: int) -> tuple[int | None, int]:
    """Parse Int32 variant (4 bytes)."""
    if offset + 3 < len(data):
        value = int.from_bytes(data[offset : offset + 4], "little", signed=True)
        return value, 4
    return None, 0


def _parse_uint32_variant(data: bytes, offset: int) -> tuple[int | None, int]:
    """Parse UInt32 variant (4 bytes)."""
    if offset + 3 < len(data):
        value = int.from_bytes(data[offset : offset + 4], "little")
        return value, 4
    return None, 0


def _parse_string_variant(data: bytes, offset: int) -> tuple[str | None, int]:
    """Parse String variant (length-prefixed)."""
    str_value, str_consumed = parse_string(data, offset)
    return str_value, str_consumed


def _parse_nodeid_variant(data: bytes, offset: int) -> tuple[str, int]:
    """Parse NodeId variant."""
    node_id, node_consumed = parse_node_id(data, offset)
    return node_id, node_consumed


__all__ = ["parse_node_id", "parse_string", "parse_variant"]
