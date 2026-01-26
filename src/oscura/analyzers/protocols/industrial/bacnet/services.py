"""BACnet service decoders.

This module provides service-specific decoders for BACnet confirmed and unconfirmed
services according to ASHRAE 135-2020.

References:
    ANSI/ASHRAE Standard 135-2020, Clause 15-19: Application Layer Services
"""

from __future__ import annotations

from typing import Any

from oscura.analyzers.protocols.industrial.bacnet.encoding import (
    parse_application_tag,
    parse_object_identifier,
    parse_tag,
)


def decode_who_is(data: bytes) -> dict[str, Any]:
    """Decode Who-Is service (unconfirmed service for device discovery).

    Args:
        data: Service payload bytes.

    Returns:
        Dictionary with optional device_instance_range_low_limit and
        device_instance_range_high_limit if specified.

    Example:
        >>> data = bytes([0x09, 0x00, 0x19, 0xFF])  # Range 0-255
        >>> result = decode_who_is(data)
    """
    result: dict[str, Any] = {}
    offset = 0

    # Optional device instance range (context tags 0 and 1)
    if offset < len(data):
        try:
            tag, tag_size = parse_tag(data, offset)
            if tag["context_specific"] and tag["tag_number"] == 0:
                # Low limit
                value_offset = offset + tag_size
                low_limit = int.from_bytes(data[value_offset : value_offset + tag["length"]], "big")
                result["device_instance_range_low_limit"] = low_limit
                offset = value_offset + tag["length"]

                # High limit (context tag 1)
                if offset < len(data):
                    tag, tag_size = parse_tag(data, offset)
                    if tag["context_specific"] and tag["tag_number"] == 1:
                        value_offset = offset + tag_size
                        high_limit = int.from_bytes(
                            data[value_offset : value_offset + tag["length"]], "big"
                        )
                        result["device_instance_range_high_limit"] = high_limit
        except (ValueError, IndexError):
            pass

    return result


def decode_i_am(data: bytes) -> dict[str, Any]:
    """Decode I-Am service (unconfirmed device announcement).

    Args:
        data: Service payload bytes.

    Returns:
        Dictionary with device_instance, max_apdu_length, segmentation, and vendor_id.

    Example:
        >>> data = bytes([0xC4, 0x02, 0x00, 0x00, 0x08, 0x22, 0x05, 0x00, ...])
        >>> result = decode_i_am(data)
        >>> print(f"Device {result['device_instance']}")
    """
    result: dict[str, Any] = {}
    offset = 0

    try:
        # Device object identifier (application tag 12, object identifier)
        value, consumed = parse_application_tag(data, offset)
        if isinstance(value, dict) and "instance" in value:
            result["device_instance"] = value["instance"]
            result["device_object_type"] = value.get("object_type_name", "unknown")
        offset += consumed

        # Max APDU length accepted (unsigned)
        if offset < len(data):
            value, consumed = parse_application_tag(data, offset)
            result["max_apdu_length"] = value
            offset += consumed

        # Segmentation supported (enumerated)
        if offset < len(data):
            value, consumed = parse_application_tag(data, offset)
            segmentation_names = {
                0: "both",
                1: "transmit",
                2: "receive",
                3: "no-segmentation",
            }
            result["segmentation"] = segmentation_names.get(value, value)
            offset += consumed

        # Vendor ID (unsigned)
        if offset < len(data):
            value, consumed = parse_application_tag(data, offset)
            result["vendor_id"] = value
            offset += consumed

    except (ValueError, IndexError):
        pass

    return result


def decode_who_has(data: bytes) -> dict[str, Any]:
    """Decode Who-Has service (unconfirmed service to find objects).

    Args:
        data: Service payload bytes.

    Returns:
        Dictionary with optional device range and object identifier or name.

    Example:
        >>> data = bytes([0x3C, 0x02, 0x00, 0x00, 0x01, 0x3E, ...])
        >>> result = decode_who_has(data)
    """
    result: dict[str, Any] = {}
    offset = 0

    try:
        # Optional device instance range (context tags 0 and 1)
        tag, tag_size = parse_tag(data, offset)
        if tag["context_specific"] and tag["tag_number"] == 0:
            value_offset = offset + tag_size
            low_limit = int.from_bytes(data[value_offset : value_offset + tag["length"]], "big")
            result["device_instance_range_low_limit"] = low_limit
            offset = value_offset + tag["length"]

            tag, tag_size = parse_tag(data, offset)
            if tag["context_specific"] and tag["tag_number"] == 1:
                value_offset = offset + tag_size
                high_limit = int.from_bytes(
                    data[value_offset : value_offset + tag["length"]], "big"
                )
                result["device_instance_range_high_limit"] = high_limit
                offset = value_offset + tag["length"]

        # Object identifier or object name (choice: context 2 or 3)
        if offset < len(data):
            tag, tag_size = parse_tag(data, offset)
            if tag["context_specific"] and tag["tag_number"] == 2:
                # Object identifier
                obj_id, _ = parse_object_identifier(data, offset + tag_size)
                result["object_identifier"] = obj_id
            elif tag["context_specific"] and tag["tag_number"] == 3:
                # Object name
                value_offset = offset + tag_size
                name = data[value_offset : value_offset + tag["length"]].decode(
                    "utf-8", errors="replace"
                )
                result["object_name"] = name

    except (ValueError, IndexError):
        pass

    return result


def decode_i_have(data: bytes) -> dict[str, Any]:
    """Decode I-Have service (unconfirmed response to Who-Has).

    Args:
        data: Service payload bytes.

    Returns:
        Dictionary with device_identifier, object_identifier, and object_name.

    Example:
        >>> result = decode_i_have(data)
        >>> print(f"Device has {result['object_name']}")
    """
    result: dict[str, Any] = {}
    offset = 0

    try:
        # Device identifier
        value, consumed = parse_application_tag(data, offset)
        result["device_identifier"] = value
        offset += consumed

        # Object identifier
        value, consumed = parse_application_tag(data, offset)
        result["object_identifier"] = value
        offset += consumed

        # Object name
        value, consumed = parse_application_tag(data, offset)
        result["object_name"] = value
        offset += consumed

    except (ValueError, IndexError):
        pass

    return result


def decode_read_property_request(data: bytes) -> dict[str, Any]:
    """Decode ReadProperty service request.

    Args:
        data: Service payload bytes.

    Returns:
        Dictionary with object_identifier, property_identifier, and optional
        property_array_index.

    Example:
        >>> result = decode_read_property_request(data)
        >>> print(f"Read {result['property_identifier']} from {result['object_identifier']}")
    """
    result: dict[str, Any] = {}
    offset = 0

    try:
        # Object identifier (context tag 0)
        tag, tag_size = parse_tag(data, offset)
        if tag["context_specific"] and tag["tag_number"] == 0:
            obj_id, _ = parse_object_identifier(data, offset + tag_size)
            result["object_identifier"] = obj_id
            offset += tag_size + 4

        # Property identifier (context tag 1)
        if offset < len(data):
            tag, tag_size = parse_tag(data, offset)
            if tag["context_specific"] and tag["tag_number"] == 1:
                value_offset = offset + tag_size
                prop_id = int.from_bytes(data[value_offset : value_offset + tag["length"]], "big")
                result["property_identifier"] = prop_id
                result["property_name"] = get_property_name(prop_id)
                offset = value_offset + tag["length"]

        # Optional property array index (context tag 2)
        if offset < len(data):
            tag, tag_size = parse_tag(data, offset)
            if tag["context_specific"] and tag["tag_number"] == 2:
                value_offset = offset + tag_size
                array_index = int.from_bytes(
                    data[value_offset : value_offset + tag["length"]], "big"
                )
                result["property_array_index"] = array_index

    except (ValueError, IndexError):
        pass

    return result


def decode_read_property_ack(data: bytes) -> dict[str, Any]:
    """Decode ReadProperty-ACK service response.

    Args:
        data: Service payload bytes.

    Returns:
        Dictionary with object_identifier, property_identifier, and property_value.

    Example:
        >>> result = decode_read_property_ack(data)
        >>> print(f"Value: {result['property_value']}")
    """
    result: dict[str, Any] = {}
    offset = 0

    try:
        # Parse object identifier (context tag 0)
        obj_id, offset = _parse_bacnet_object_id(data, offset)
        if obj_id is not None:
            result["object_identifier"] = obj_id

        # Parse property identifier (context tag 1)
        prop_id, prop_name, offset = _parse_bacnet_property_id(data, offset)
        if prop_id is not None:
            result["property_identifier"] = prop_id
            result["property_name"] = prop_name

        # Parse optional property array index (context tag 2)
        array_index, offset = _parse_bacnet_array_index(data, offset)
        if array_index is not None:
            result["property_array_index"] = array_index

        # Parse property value (context tag 3)
        prop_value = _parse_bacnet_property_value(data, offset)
        if prop_value is not None:
            result["property_value"] = prop_value

    except (ValueError, IndexError):
        pass

    return result


def _parse_bacnet_object_id(data: bytes, offset: int) -> tuple[dict[str, Any] | None, int]:
    """Parse BACnet object identifier from context tag 0.

    Args:
        data: Payload bytes.
        offset: Current parsing offset.

    Returns:
        Tuple of (object_id_dict, new_offset). None if tag not found.
    """
    if offset >= len(data):
        return (None, offset)

    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 0:
        obj_id, _ = parse_object_identifier(data, offset + tag_size)
        return (obj_id, offset + tag_size + 4)

    return (None, offset)


def _parse_bacnet_property_id(data: bytes, offset: int) -> tuple[int | None, str | None, int]:
    """Parse BACnet property identifier from context tag 1.

    Args:
        data: Payload bytes.
        offset: Current parsing offset.

    Returns:
        Tuple of (property_id, property_name, new_offset). None if tag not found.
    """
    if offset >= len(data):
        return (None, None, offset)

    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 1:
        value_offset = offset + tag_size
        prop_id = int.from_bytes(data[value_offset : value_offset + tag["length"]], "big")
        prop_name = get_property_name(prop_id)
        return (prop_id, prop_name, value_offset + tag["length"])

    return (None, None, offset)


def _parse_bacnet_array_index(data: bytes, offset: int) -> tuple[int | None, int]:
    """Parse optional BACnet array index from context tag 2.

    Args:
        data: Payload bytes.
        offset: Current parsing offset.

    Returns:
        Tuple of (array_index, new_offset). None if tag not found.
    """
    if offset >= len(data):
        return (None, offset)

    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 2:
        value_offset = offset + tag_size
        array_index = int.from_bytes(data[value_offset : value_offset + tag["length"]], "big")
        return (array_index, value_offset + tag["length"])

    return (None, offset)


def _parse_bacnet_property_value(data: bytes, offset: int) -> Any:
    """Parse BACnet property value from context tag 3.

    Args:
        data: Payload bytes.
        offset: Current parsing offset.

    Returns:
        Property value (single value or list). None if tag not found.
    """
    if offset >= len(data):
        return None

    tag, tag_size = parse_tag(data, offset)
    if not (tag["context_specific"] and tag["tag_number"] == 3 and tag["is_opening"]):
        return None

    offset += tag_size

    # Parse values until closing tag
    values = []
    while offset < len(data):
        tag, tag_size = parse_tag(data, offset)
        if tag["is_closing"] and tag["tag_number"] == 3:
            break

        try:
            value, consumed = parse_application_tag(data, offset)
            values.append(value)
            offset += consumed
        except ValueError:
            # Skip unparseable data
            offset += tag_size + tag.get("length", 0)
            break

    return values[0] if len(values) == 1 else (values if values else None)


def decode_write_property_request(data: bytes) -> dict[str, Any]:
    """Decode WriteProperty service request.

    Args:
        data: Service payload bytes.

    Returns:
        Dictionary with object_identifier, property_identifier, property_value,
        and optional priority.

    Example:
        >>> result = decode_write_property_request(data)
        >>> print(f"Write {result['property_value']} to {result['property_name']}")
    """
    result: dict[str, Any] = {}
    offset = 0

    try:
        # Parse object identifier (context tag 0)
        obj_id, offset = _parse_write_property_object_id(data, offset, result)

        # Parse property identifier (context tag 1)
        offset = _parse_write_property_id(data, offset, result)

        # Parse optional property array index (context tag 2)
        offset = _parse_write_property_array_index(data, offset, result)

        # Parse property value (context tag 3)
        offset = _parse_write_property_value(data, offset, result)

        # Parse optional priority (context tag 4)
        _parse_write_property_priority(data, offset, result)

    except (ValueError, IndexError):
        pass

    return result


def _parse_write_property_object_id(
    data: bytes, offset: int, result: dict[str, Any]
) -> tuple[dict[str, Any] | None, int]:
    """Parse object identifier from WriteProperty request.

    Args:
        data: Payload bytes.
        offset: Current offset.
        result: Result dictionary to populate.

    Returns:
        Tuple of (object_id, new_offset).
    """
    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 0:
        obj_id, _ = parse_object_identifier(data, offset + tag_size)
        result["object_identifier"] = obj_id
        return obj_id, offset + tag_size + 4
    return None, offset


def _parse_write_property_id(data: bytes, offset: int, result: dict[str, Any]) -> int:
    """Parse property identifier from WriteProperty request.

    Args:
        data: Payload bytes.
        offset: Current offset.
        result: Result dictionary to populate.

    Returns:
        New offset.
    """
    if offset >= len(data):
        return offset

    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 1:
        value_offset = offset + tag_size
        length: int = int(tag["length"])
        prop_id = int.from_bytes(data[value_offset : value_offset + length], "big")
        result["property_identifier"] = prop_id
        result["property_name"] = get_property_name(prop_id)
        return value_offset + length

    return offset


def _parse_write_property_array_index(data: bytes, offset: int, result: dict[str, Any]) -> int:
    """Parse optional array index from WriteProperty request.

    Args:
        data: Payload bytes.
        offset: Current offset.
        result: Result dictionary to populate.

    Returns:
        New offset.
    """
    if offset >= len(data):
        return offset

    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 2:
        value_offset = offset + tag_size
        length: int = int(tag["length"])
        array_index = int.from_bytes(data[value_offset : value_offset + length], "big")
        result["property_array_index"] = array_index
        return value_offset + length

    return offset


def _parse_write_property_value(data: bytes, offset: int, result: dict[str, Any]) -> int:
    """Parse property value from WriteProperty request.

    Args:
        data: Payload bytes.
        offset: Current offset.
        result: Result dictionary to populate.

    Returns:
        New offset.
    """
    if offset >= len(data):
        return offset

    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 3 and tag["is_opening"]:
        offset += tag_size

        # Parse value(s) until closing tag
        values = []
        while offset < len(data):
            tag, tag_size = parse_tag(data, offset)
            if tag["is_closing"] and tag["tag_number"] == 3:
                offset += tag_size
                break

            try:
                value, consumed = parse_application_tag(data, offset)
                values.append(value)
                offset += consumed
            except ValueError:
                offset += tag_size + tag.get("length", 0)
                break

        result["property_value"] = values[0] if len(values) == 1 else values

    return offset


def _parse_write_property_priority(data: bytes, offset: int, result: dict[str, Any]) -> None:
    """Parse optional priority from WriteProperty request.

    Args:
        data: Payload bytes.
        offset: Current offset.
        result: Result dictionary to populate.
    """
    if offset >= len(data):
        return

    tag, tag_size = parse_tag(data, offset)
    if tag["context_specific"] and tag["tag_number"] == 4:
        value_offset = offset + tag_size
        priority = int.from_bytes(data[value_offset : value_offset + tag["length"]], "big")
        result["priority"] = priority


def get_property_name(property_id: int) -> str:
    """Get human-readable property name from property identifier.

    Args:
        property_id: BACnet property identifier.

    Returns:
        Property name string.

    Example:
        >>> name = get_property_name(85)  # "present-value"
    """
    # Common BACnet property identifiers (ASHRAE 135-2020, Clause 21)
    property_names = {
        0: "acked-transitions",
        1: "ack-required",
        4: "action",
        8: "all",
        28: "description",
        36: "event-state",
        41: "high-limit",
        44: "limit-enable",
        45: "local-date",
        46: "local-time",
        52: "low-limit",
        56: "max-pres-value",
        59: "min-pres-value",
        62: "notify-type",
        65: "object-identifier",
        77: "object-name",
        79: "object-type",
        85: "present-value",
        103: "reliability",
        107: "segmentation-supported",
        111: "status-flags",
        112: "system-status",
        117: "units",
        120: "vendor-identifier",
        121: "vendor-name",
        122: "vt-classes-supported",
        155: "event-enable",
        158: "ack-mode",
    }
    return property_names.get(property_id, f"property-{property_id}")
