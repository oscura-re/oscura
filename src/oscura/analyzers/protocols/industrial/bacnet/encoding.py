"""BACnet encoding and decoding utilities.

This module provides low-level BACnet encoding/decoding functions for tags,
object identifiers, and application data types according to ASHRAE 135-2020.

References:
    ANSI/ASHRAE Standard 135-2020, Clause 20: Encoding of BACnet Tags
"""

from __future__ import annotations

from typing import Any


def parse_tag(data: bytes, offset: int) -> tuple[dict[str, Any], int]:
    """Parse BACnet tag (application or context).

    Args:
        data: Raw bytes to parse.
        offset: Starting offset in data.

    Returns:
        Tuple of (tag_dict, bytes_consumed) where tag_dict contains:
        - tag_number: Tag number (0-254)
        - context_specific: True if context tag, False if application tag
        - length: Value/length field
        - value: Decoded value (if applicable)

    Raises:
        ValueError: If data is too short or invalid tag format.

    Example:
        >>> data = bytes([0xC4, 0x02, 0x00, 0x00, 0x08])
        >>> tag, consumed = parse_tag(data, 0)
        >>> print(f"Tag {tag['tag_number']}, length {tag['length']}")
    """
    if offset >= len(data):
        raise ValueError("Offset beyond data length")

    initial_offset = offset
    tag_byte = data[offset]
    offset += 1

    # Parse tag number (bits 4-7)
    tag_number = (tag_byte >> 4) & 0x0F
    # Bit 3: class (0=application, 1=context)
    context_specific = bool(tag_byte & 0x08)
    # Bits 0-2: length/value/type
    length_value_type = tag_byte & 0x07

    # Extended tag number (tag number 15 means next byte has actual tag)
    if tag_number == 15:
        if offset >= len(data):
            raise ValueError("Extended tag number missing")
        tag_number = data[offset]
        offset += 1

    # Decode length/value/type field
    # For context tags: lvt=6 is opening tag, lvt=7 is closing tag
    # For application tags: lvt=5-7 are special (5 is used normally for length 5, but 6/7 are reserved)
    # Actually in BACnet: for application tags, lvt=7 is NOT used for opening/closing (those are context-only)
    # Instead, lvt in 0-4 gives the length directly, lvt=5 means length 5 for most tags except
    # for constructed data (which uses context tags with opening/closing)
    # Let me simplify: lvt=6/7 only have special meaning for context tags
    if context_specific and length_value_type == 6:  # Opening tag (context only)
        length = 0
        is_opening = True
        is_closing = False
    elif context_specific and length_value_type == 7:  # Closing tag (context only)
        length = 0
        is_opening = False
        is_closing = True
    elif not context_specific and length_value_type == 5:  # Extended length (application only)
        # For application tags, lvt=5 means extended length follows
        if offset >= len(data):
            raise ValueError("Extended length missing")
        length = data[offset]
        offset += 1
        if length == 254:  # 16-bit length
            if offset + 1 >= len(data):
                raise ValueError("16-bit length incomplete")
            length = int.from_bytes(data[offset : offset + 2], "big")
            offset += 2
        elif length == 255:  # 32-bit length
            if offset + 3 >= len(data):
                raise ValueError("32-bit length incomplete")
            length = int.from_bytes(data[offset : offset + 4], "big")
            offset += 4
        is_opening = False
        is_closing = False
    else:
        # For lengths 0-4, lvt directly encodes the length
        length = length_value_type
        is_opening = False
        is_closing = False

    tag_dict = {
        "tag_number": tag_number,
        "context_specific": context_specific,
        "length": length,
        "is_opening": is_opening,
        "is_closing": is_closing,
    }

    bytes_consumed = offset - initial_offset
    return tag_dict, bytes_consumed


def parse_unsigned(data: bytes, offset: int, length: int) -> tuple[int, int]:
    """Parse BACnet unsigned integer.

    Args:
        data: Raw bytes to parse.
        offset: Starting offset in data.
        length: Number of bytes (1-4).

    Returns:
        Tuple of (value, bytes_consumed).

    Raises:
        ValueError: If length is invalid or data too short.

    Example:
        >>> data = bytes([0x00, 0x01, 0x23, 0x45])
        >>> value, consumed = parse_unsigned(data, 0, 4)
        >>> print(f"Value: {value}")  # 74565
    """
    if length < 1 or length > 4:
        raise ValueError(f"Invalid unsigned integer length: {length}")
    if offset + length > len(data):
        raise ValueError("Data too short for unsigned integer")

    value = int.from_bytes(data[offset : offset + length], "big")
    return value, length


def parse_enumerated(data: bytes, offset: int, length: int) -> tuple[int, int]:
    """Parse BACnet enumerated value (same encoding as unsigned).

    Args:
        data: Raw bytes to parse.
        offset: Starting offset in data.
        length: Number of bytes (1-4).

    Returns:
        Tuple of (value, bytes_consumed).

    Raises:
        ValueError: If length is invalid or data too short.

    Example:
        >>> data = bytes([0x03])  # Segmentation: both
        >>> value, consumed = parse_enumerated(data, 0, 1)
    """
    return parse_unsigned(data, offset, length)


def parse_object_identifier(data: bytes, offset: int) -> tuple[dict[str, Any], int]:
    """Parse BACnet object identifier (32-bit: 10-bit type + 22-bit instance).

    Args:
        data: Raw bytes to parse.
        offset: Starting offset in data.

    Returns:
        Tuple of (obj_id_dict, bytes_consumed) where obj_id_dict contains:
        - object_type: Object type number
        - object_type_name: Human-readable type name
        - instance: Instance number

    Raises:
        ValueError: If data too short.

    Example:
        >>> data = bytes([0x02, 0x00, 0x00, 0x08])
        >>> obj_id, consumed = parse_object_identifier(data, 0)
        >>> print(f"{obj_id['object_type_name']} #{obj_id['instance']}")
    """
    if offset + 4 > len(data):
        raise ValueError("Data too short for object identifier")

    obj_id_bytes = int.from_bytes(data[offset : offset + 4], "big")
    object_type = (obj_id_bytes >> 22) & 0x3FF  # Top 10 bits
    instance = obj_id_bytes & 0x3FFFFF  # Bottom 22 bits

    # Object type names (ASHRAE 135-2020, Clause 21)
    object_type_names = {
        0: "analog-input",
        1: "analog-output",
        2: "analog-value",
        3: "binary-input",
        4: "binary-output",
        5: "binary-value",
        6: "calendar",
        7: "command",
        8: "device",
        9: "event-enrollment",
        10: "file",
        11: "group",
        12: "loop",
        13: "multi-state-input",
        14: "multi-state-output",
        15: "notification-class",
        16: "program",
        17: "schedule",
        18: "averaging",
        19: "multi-state-value",
        20: "trend-log",
        21: "life-safety-point",
        22: "life-safety-zone",
        23: "accumulator",
        24: "pulse-converter",
    }

    obj_id_dict = {
        "object_type": object_type,
        "object_type_name": object_type_names.get(object_type, f"type-{object_type}"),
        "instance": instance,
    }

    return obj_id_dict, 4


def parse_character_string(data: bytes, offset: int, length: int) -> tuple[str, int]:
    """Parse BACnet character string (encoding byte + UTF-8 string).

    Args:
        data: Raw bytes to parse.
        offset: Starting offset in data.
        length: Total length including encoding byte.

    Returns:
        Tuple of (string_value, bytes_consumed).

    Raises:
        ValueError: If data too short.

    Example:
        >>> data = bytes([0x00]) + b"Building 1"  # 0x00 = UTF-8
        >>> string, consumed = parse_character_string(data, 0, 11)
    """
    if offset + length > len(data):
        raise ValueError("Data too short for character string")

    encoding = data[offset]  # 0=UTF-8, 1=IBM/MS DBCS, 2=JIS C 6226, 3=UCS-4, 4=UCS-2, 5=ISO 8859-1
    string_data = data[offset + 1 : offset + length]

    if encoding == 0:  # UTF-8
        string_value = string_data.decode("utf-8", errors="replace")
    elif encoding == 5:  # ISO 8859-1 (Latin-1)
        string_value = string_data.decode("latin-1", errors="replace")
    else:
        # For unsupported encodings, return hex representation
        string_value = string_data.hex()

    return string_value, length


def _parse_signed_integer(
    data: bytes, value_offset: int, length: int, tag_size: int
) -> tuple[int, int]:
    """Parse signed integer application tag.

    Args:
        data: Raw bytes.
        value_offset: Offset to value data.
        length: Value length.
        tag_size: Size of tag header.

    Returns:
        Tuple of (value, bytes_consumed).

    Raises:
        ValueError: If data too short.
    """
    if value_offset + length > len(data):
        raise ValueError("Data too short for signed integer")
    int_value = int.from_bytes(data[value_offset : value_offset + length], "big", signed=True)
    return int_value, tag_size + length


def _parse_real(data: bytes, value_offset: int, length: int, tag_size: int) -> tuple[float, int]:
    """Parse real (float) application tag.

    Args:
        data: Raw bytes.
        value_offset: Offset to value data.
        length: Value length.
        tag_size: Size of tag header.

    Returns:
        Tuple of (value, bytes_consumed).

    Raises:
        ValueError: If invalid length or data too short.
    """
    import struct

    if length != 4:
        raise ValueError(f"Invalid real length: {length}")
    if value_offset + 4 > len(data):
        raise ValueError("Data too short for real")

    float_value = struct.unpack(">f", data[value_offset : value_offset + 4])[0]
    return float_value, tag_size + 4


def _parse_double(data: bytes, value_offset: int, length: int, tag_size: int) -> tuple[float, int]:
    """Parse double application tag.

    Args:
        data: Raw bytes.
        value_offset: Offset to value data.
        length: Value length.
        tag_size: Size of tag header.

    Returns:
        Tuple of (value, bytes_consumed).

    Raises:
        ValueError: If invalid length or data too short.
    """
    import struct

    if length != 8:
        raise ValueError(f"Invalid double length: {length}")
    if value_offset + 8 > len(data):
        raise ValueError("Data too short for double")

    double_value = struct.unpack(">d", data[value_offset : value_offset + 8])[0]
    return double_value, tag_size + 8


def _parse_object_id_tag(
    data: bytes, value_offset: int, length: int, tag_size: int
) -> tuple[dict[str, Any], int]:
    """Parse object identifier application tag.

    Args:
        data: Raw bytes.
        value_offset: Offset to value data.
        length: Value length.
        tag_size: Size of tag header.

    Returns:
        Tuple of (obj_id_dict, bytes_consumed).

    Raises:
        ValueError: If invalid length.
    """
    if length != 4:
        raise ValueError(f"Invalid object identifier length: {length}")
    obj_id, _ = parse_object_identifier(data, value_offset)
    return obj_id, tag_size + 4


def parse_application_tag(data: bytes, offset: int) -> tuple[Any, int]:
    """Parse application-tagged data (standard BACnet data types).

    Args:
        data: Raw bytes to parse.
        offset: Starting offset in data.

    Returns:
        Tuple of (decoded_value, bytes_consumed).

    Raises:
        ValueError: If invalid tag or data too short.

    Example:
        >>> data = bytes([0x21, 0x05])  # Unsigned int, length 1, value 5
        >>> value, consumed = parse_application_tag(data, 0)
    """
    tag, tag_size = parse_tag(data, offset)

    if tag["context_specific"]:
        raise ValueError("Expected application tag, got context tag")

    if tag["is_opening"] or tag["is_closing"]:
        raise ValueError("Unexpected opening/closing tag")

    value_offset = offset + tag_size
    length = tag["length"]
    tag_number = tag["tag_number"]

    # Application tag numbers (ASHRAE 135-2020, Clause 20.2.1)
    if tag_number == 0:  # Null
        return None, tag_size
    elif tag_number == 1:  # Boolean
        return bool(length), tag_size
    elif tag_number == 2:  # Unsigned Integer
        uint_value, _ = parse_unsigned(data, value_offset, length)
        return uint_value, tag_size + length
    elif tag_number == 3:  # Signed Integer
        return _parse_signed_integer(data, value_offset, length, tag_size)
    elif tag_number == 4:  # Real
        return _parse_real(data, value_offset, length, tag_size)
    elif tag_number == 5:  # Double
        return _parse_double(data, value_offset, length, tag_size)
    elif tag_number == 7:  # Character String
        str_value, _ = parse_character_string(data, value_offset, length)
        return str_value, tag_size + length
    elif tag_number == 9:  # Enumerated
        enum_value, _ = parse_enumerated(data, value_offset, length)
        return enum_value, tag_size + length
    elif tag_number == 12:  # Object Identifier
        return _parse_object_id_tag(data, value_offset, length, tag_size)
    else:
        # Unknown/unsupported type - return raw bytes
        if value_offset + length > len(data):
            raise ValueError(f"Data too short for tag type {tag_number}")
        return data[value_offset : value_offset + length], tag_size + length
