"""Zigbee Cluster Library (ZCL) definitions and parsers.

This module contains standard ZCL cluster definitions and frame parsers
for common clusters including On/Off, Level Control, Temperature, etc.

References:
    Zigbee Cluster Library Specification (CSA-IOT)
"""

from __future__ import annotations

from typing import Any

# Standard ZCL cluster IDs (subset of most common clusters)
ZCL_CLUSTERS: dict[int, str] = {
    0x0000: "Basic",
    0x0001: "Power Configuration",
    0x0003: "Identify",
    0x0004: "Groups",
    0x0005: "Scenes",
    0x0006: "On/Off",
    0x0008: "Level Control",
    0x0009: "Alarms",
    0x000A: "Time",
    0x0020: "Poll Control",
    0x0201: "Thermostat",
    0x0300: "Color Control",
    0x0400: "Illuminance Measurement",
    0x0402: "Temperature Measurement",
    0x0403: "Pressure Measurement",
    0x0404: "Flow Measurement",
    0x0405: "Relative Humidity Measurement",
    0x0406: "Occupancy Sensing",
    0x0500: "IAS Zone",
    0x0702: "Metering",
    0x0B04: "Electrical Measurement",
}

# ZCL frame control field bits
ZCL_FRAME_TYPE_MASK = 0x03
ZCL_FRAME_TYPE_GLOBAL = 0x00
ZCL_FRAME_TYPE_CLUSTER = 0x01

ZCL_MANUFACTURER_SPECIFIC = 0x04
ZCL_DIRECTION_SERVER_TO_CLIENT = 0x08
ZCL_DISABLE_DEFAULT_RESPONSE = 0x10

# Global ZCL commands
ZCL_GLOBAL_COMMANDS = {
    0x00: "Read Attributes",
    0x01: "Read Attributes Response",
    0x02: "Write Attributes",
    0x03: "Write Attributes Undivided",
    0x04: "Write Attributes Response",
    0x05: "Write Attributes No Response",
    0x06: "Configure Reporting",
    0x07: "Configure Reporting Response",
    0x08: "Read Reporting Configuration",
    0x09: "Read Reporting Configuration Response",
    0x0A: "Report Attributes",
    0x0B: "Default Response",
    0x0C: "Discover Attributes",
    0x0D: "Discover Attributes Response",
    0x0E: "Read Attributes Structured",
    0x0F: "Write Attributes Structured",
    0x10: "Write Attributes Structured Response",
    0x11: "Discover Commands Received",
    0x12: "Discover Commands Received Response",
    0x13: "Discover Commands Generated",
    0x14: "Discover Commands Generated Response",
    0x15: "Discover Attributes Extended",
    0x16: "Discover Attributes Extended Response",
}

# On/Off cluster (0x0006) commands
ZCL_ONOFF_COMMANDS = {
    0x00: "Off",
    0x01: "On",
    0x02: "Toggle",
    0x40: "Off With Effect",
    0x41: "On With Recall Global Scene",
    0x42: "On With Timed Off",
}

# Level Control cluster (0x0008) commands
ZCL_LEVEL_CONTROL_COMMANDS = {
    0x00: "Move to Level",
    0x01: "Move",
    0x02: "Step",
    0x03: "Stop",
    0x04: "Move to Level (with On/Off)",
    0x05: "Move (with On/Off)",
    0x06: "Step (with On/Off)",
    0x07: "Stop (with On/Off)",
}

# Color Control cluster (0x0300) commands
ZCL_COLOR_CONTROL_COMMANDS = {
    0x00: "Move to Hue",
    0x01: "Move Hue",
    0x02: "Step Hue",
    0x03: "Move to Saturation",
    0x04: "Move Saturation",
    0x05: "Step Saturation",
    0x06: "Move to Hue and Saturation",
    0x07: "Move to Color",
    0x08: "Move Color",
    0x09: "Step Color",
    0x0A: "Move to Color Temperature",
}


def parse_zcl_frame(cluster_id: int, data: bytes) -> dict[str, Any]:
    """Parse ZCL frame for specific cluster.

    Args:
        cluster_id: ZCL cluster ID (e.g., 0x0006 for On/Off).
        data: ZCL frame payload.

    Returns:
        Parsed ZCL frame with cluster-specific details.

    Example:
        >>> data = bytes([0x01, 0x00, 0x01])  # On/Off cluster, On command
        >>> result = parse_zcl_frame(0x0006, data)
        >>> print(result['command_name'])
        On
    """
    if len(data) < 3:
        return {
            "error": "Insufficient ZCL data",
            "cluster_id": cluster_id,
            "cluster_name": ZCL_CLUSTERS.get(cluster_id, "Unknown"),
        }

    frame_control = data[0]
    transaction_seq = data[1]
    command_id = data[2]

    frame_type = frame_control & ZCL_FRAME_TYPE_MASK
    manufacturer_specific = bool(frame_control & ZCL_MANUFACTURER_SPECIFIC)
    direction = (
        "server_to_client"
        if (frame_control & ZCL_DIRECTION_SERVER_TO_CLIENT)
        else "client_to_server"
    )
    disable_default_response = bool(frame_control & ZCL_DISABLE_DEFAULT_RESPONSE)

    result: dict[str, Any] = {
        "cluster_id": cluster_id,
        "cluster_name": ZCL_CLUSTERS.get(cluster_id, f"Unknown (0x{cluster_id:04X})"),
        "frame_control": frame_control,
        "frame_type": "global" if frame_type == ZCL_FRAME_TYPE_GLOBAL else "cluster_specific",
        "manufacturer_specific": manufacturer_specific,
        "direction": direction,
        "disable_default_response": disable_default_response,
        "transaction_sequence": transaction_seq,
        "command_id": command_id,
    }

    # Parse manufacturer code if present
    offset = 3
    if manufacturer_specific:
        if len(data) >= 5:
            mfr_code = int.from_bytes(data[3:5], "little")
            result["manufacturer_code"] = mfr_code
            offset = 5
        else:
            result["error"] = "Insufficient data for manufacturer code"
            return result

    payload = data[offset:]
    result["payload"] = payload

    # Parse cluster-specific commands
    if frame_type == ZCL_FRAME_TYPE_GLOBAL:
        result["command_name"] = ZCL_GLOBAL_COMMANDS.get(
            command_id, f"Unknown Global (0x{command_id:02X})"
        )
        if command_id == 0x00:  # Read Attributes
            result["details"] = _parse_read_attributes(payload)
        elif command_id == 0x01:  # Read Attributes Response
            result["details"] = _parse_read_attributes_response(payload)
        elif command_id == 0x0A:  # Report Attributes
            result["details"] = _parse_report_attributes(payload)
    else:
        # Cluster-specific command
        if cluster_id == 0x0006:  # On/Off
            result["command_name"] = ZCL_ONOFF_COMMANDS.get(
                command_id, f"Unknown (0x{command_id:02X})"
            )
            if command_id in [0x00, 0x01, 0x02]:  # Off, On, Toggle
                result["details"] = {"simple_command": True}
        elif cluster_id == 0x0008:  # Level Control
            result["command_name"] = ZCL_LEVEL_CONTROL_COMMANDS.get(
                command_id, f"Unknown (0x{command_id:02X})"
            )
            if command_id == 0x00 and len(payload) >= 2:  # Move to Level
                result["details"] = {
                    "level": payload[0],
                    "transition_time": int.from_bytes(payload[1:3], "little")
                    if len(payload) >= 3
                    else None,
                }
        elif cluster_id == 0x0300:  # Color Control
            result["command_name"] = ZCL_COLOR_CONTROL_COMMANDS.get(
                command_id, f"Unknown (0x{command_id:02X})"
            )
        else:
            result["command_name"] = f"Cluster Command 0x{command_id:02X}"

    return result


def _parse_read_attributes(payload: bytes) -> dict[str, Any]:
    """Parse Read Attributes command payload.

    Args:
        payload: ZCL payload after command ID.

    Returns:
        Parsed attribute IDs.
    """
    if len(payload) < 2:
        return {"error": "Insufficient data"}

    attribute_ids = []
    offset = 0
    while offset + 2 <= len(payload):
        attr_id = int.from_bytes(payload[offset : offset + 2], "little")
        attribute_ids.append(attr_id)
        offset += 2

    return {"attribute_ids": attribute_ids}


def _parse_read_attributes_response(payload: bytes) -> dict[str, Any]:
    """Parse Read Attributes Response payload.

    Args:
        payload: ZCL payload after command ID.

    Returns:
        Parsed attribute records.
    """
    if len(payload) < 3:
        return {"error": "Insufficient data"}

    attributes = []
    offset = 0
    while offset + 3 <= len(payload):
        attr_id = int.from_bytes(payload[offset : offset + 2], "little")
        status = payload[offset + 2]
        offset += 3

        attr_record: dict[str, Any] = {
            "attribute_id": attr_id,
            "status": status,
        }

        if status == 0x00:  # Success
            if offset >= len(payload):
                break
            data_type = payload[offset]
            offset += 1

            # Parse value based on data type (simplified)
            if data_type == 0x10:  # Boolean
                if offset < len(payload):
                    attr_record["value"] = bool(payload[offset])
                    offset += 1
            elif data_type == 0x20:  # Uint8
                if offset < len(payload):
                    attr_record["value"] = payload[offset]
                    offset += 1
            elif data_type == 0x21:  # Uint16
                if offset + 2 <= len(payload):
                    attr_record["value"] = int.from_bytes(payload[offset : offset + 2], "little")
                    offset += 2
            elif data_type == 0x29:  # Int16
                if offset + 2 <= len(payload):
                    attr_record["value"] = int.from_bytes(
                        payload[offset : offset + 2], "little", signed=True
                    )
                    offset += 2
            else:
                attr_record["data_type"] = data_type
                # Skip unknown data type
                break

        attributes.append(attr_record)

    return {"attributes": attributes}


def _parse_report_attributes(payload: bytes) -> dict[str, Any]:
    """Parse Report Attributes payload.

    Args:
        payload: ZCL payload after command ID.

    Returns:
        Parsed attribute reports.
    """
    # Similar to Read Attributes Response but without status field
    if len(payload) < 3:
        return {"error": "Insufficient data"}

    attributes = []
    offset = 0
    while offset + 3 <= len(payload):
        attr_id = int.from_bytes(payload[offset : offset + 2], "little")
        data_type = payload[offset + 2]
        offset += 3

        attr_record: dict[str, Any] = {
            "attribute_id": attr_id,
            "data_type": data_type,
        }

        # Parse value based on data type (simplified)
        if data_type == 0x10:  # Boolean
            if offset < len(payload):
                attr_record["value"] = bool(payload[offset])
                offset += 1
        elif data_type == 0x20:  # Uint8
            if offset < len(payload):
                attr_record["value"] = payload[offset]
                offset += 1
        elif data_type == 0x21:  # Uint16
            if offset + 2 <= len(payload):
                attr_record["value"] = int.from_bytes(payload[offset : offset + 2], "little")
                offset += 2
        elif data_type == 0x29:  # Int16
            if offset + 2 <= len(payload):
                attr_record["value"] = int.from_bytes(
                    payload[offset : offset + 2], "little", signed=True
                )
                offset += 2
        else:
            # Skip unknown data type
            break

        attributes.append(attr_record)

    return {"attributes": attributes}


__all__ = ["ZCL_CLUSTERS", "parse_zcl_frame"]
