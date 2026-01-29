"""Standard BLE UUID mappings for services, characteristics, and descriptors.

This module provides mappings for standard Bluetooth SIG-assigned UUIDs
to human-readable names for GATT services, characteristics, and descriptors.

References:
    Assigned Numbers: https://www.bluetooth.com/specifications/assigned-numbers/
"""

from __future__ import annotations

# Standard GATT Services (16-bit UUIDs)
# Format: {uuid: name}
STANDARD_SERVICES: dict[int, str] = {
    # Generic Services
    0x1800: "Generic Access",
    0x1801: "Generic Attribute",
    # Health & Fitness Services
    0x180D: "Heart Rate",
    0x180F: "Battery Service",
    0x1810: "Blood Pressure",
    0x1811: "Alert Notification Service",
    0x1812: "Human Interface Device",
    0x1814: "Running Speed and Cadence",
    0x1816: "Cycling Speed and Cadence",
    0x1818: "Cycling Power",
    0x1819: "Location and Navigation",
    # Device Information
    0x180A: "Device Information",
    # Environmental Sensing
    0x181A: "Environmental Sensing",
    0x181C: "User Data",
    0x181D: "Weight Scale",
    # Automation
    0x1820: "Internet Protocol Support",
    0x1821: "Indoor Positioning",
    0x1822: "Pulse Oximeter",
    0x1823: "HTTP Proxy",
    0x1824: "Transport Discovery",
    0x1825: "Object Transfer",
    # Audio
    0x1826: "Fitness Machine",
    0x1827: "Mesh Provisioning",
    0x1828: "Mesh Proxy",
    # Nordic Services (vendor-specific but common)
    0xFE59: "Nordic UART Service",
}

# Standard GATT Characteristics (16-bit UUIDs)
STANDARD_CHARACTERISTICS: dict[int, str] = {
    # Generic Access
    0x2A00: "Device Name",
    0x2A01: "Appearance",
    0x2A02: "Peripheral Privacy Flag",
    0x2A03: "Reconnection Address",
    0x2A04: "Peripheral Preferred Connection Parameters",
    # Generic Attribute
    0x2A05: "Service Changed",
    # Device Information
    0x2A23: "System ID",
    0x2A24: "Model Number String",
    0x2A25: "Serial Number String",
    0x2A26: "Firmware Revision String",
    0x2A27: "Hardware Revision String",
    0x2A28: "Software Revision String",
    0x2A29: "Manufacturer Name String",
    # Battery Service
    0x2A19: "Battery Level",
    # Heart Rate
    0x2A37: "Heart Rate Measurement",
    0x2A38: "Body Sensor Location",
    0x2A39: "Heart Rate Control Point",
    # Blood Pressure
    0x2A35: "Blood Pressure Measurement",
    0x2A36: "Intermediate Cuff Pressure",
    0x2A49: "Blood Pressure Feature",
    # Environmental Sensing
    0x2A6E: "Temperature",
    0x2A6F: "Humidity",
    0x2A76: "UV Index",
    # HID
    0x2A4A: "HID Information",
    0x2A4B: "Report Map",
    0x2A4C: "HID Control Point",
    0x2A4D: "Report",
    0x2A4E: "Protocol Mode",
}

# Standard GATT Descriptors (16-bit UUIDs)
STANDARD_DESCRIPTORS: dict[int, str] = {
    0x2900: "Characteristic Extended Properties",
    0x2901: "Characteristic User Description",
    0x2902: "Client Characteristic Configuration",
    0x2903: "Server Characteristic Configuration",
    0x2904: "Characteristic Presentation Format",
    0x2905: "Characteristic Aggregate Format",
    0x2906: "Valid Range",
    0x2907: "External Report Reference",
    0x2908: "Report Reference",
}

# Advertising Data (AD) Type codes
AD_TYPES: dict[int, str] = {
    0x01: "Flags",
    0x02: "Incomplete List of 16-bit Service UUIDs",
    0x03: "Complete List of 16-bit Service UUIDs",
    0x04: "Incomplete List of 32-bit Service UUIDs",
    0x05: "Complete List of 32-bit Service UUIDs",
    0x06: "Incomplete List of 128-bit Service UUIDs",
    0x07: "Complete List of 128-bit Service UUIDs",
    0x08: "Shortened Local Name",
    0x09: "Complete Local Name",
    0x0A: "Tx Power Level",
    0x0D: "Class of Device",
    0x0E: "Simple Pairing Hash C",
    0x0F: "Simple Pairing Randomizer R",
    0x10: "Device ID",
    0x11: "Security Manager Out of Band Flags",
    0x12: "Slave Connection Interval Range",
    0x14: "List of 16-bit Service Solicitation UUIDs",
    0x15: "List of 128-bit Service Solicitation UUIDs",
    0x16: "Service Data - 16-bit UUID",
    0x17: "Public Target Address",
    0x18: "Random Target Address",
    0x19: "Appearance",
    0x1A: "Advertising Interval",
    0x1B: "LE Bluetooth Device Address",
    0x1C: "LE Role",
    0x1D: "Simple Pairing Hash C-256",
    0x1E: "Simple Pairing Randomizer R-256",
    0x1F: "List of 32-bit Service Solicitation UUIDs",
    0x20: "Service Data - 32-bit UUID",
    0x21: "Service Data - 128-bit UUID",
    0xFF: "Manufacturer Specific Data",
}


def uuid_to_string(uuid_bytes: bytes, short: bool = True) -> str:
    """Convert UUID bytes to standard string format.

    Args:
        uuid_bytes: UUID as bytes (2, 4, or 16 bytes).
        short: Use short format for 16/32-bit UUIDs (e.g., "0x180D").

    Returns:
        UUID string in standard format.

    Example:
        >>> uuid_to_string(b"\\x0d\\x18")  # 16-bit UUID
        '0x180D'
        >>> uuid_to_string(b"\\x00\\x00\\x18\\x0d...", short=False)
        '0000180d-0000-1000-8000-00805f9b34fb'
    """
    if len(uuid_bytes) == 2:
        # 16-bit UUID
        val = int.from_bytes(uuid_bytes, "little")
        return f"0x{val:04X}" if short else f"{val:08x}-0000-1000-8000-00805f9b34fb"
    elif len(uuid_bytes) == 4:
        # 32-bit UUID (rare)
        val = int.from_bytes(uuid_bytes, "little")
        return f"0x{val:08X}" if short else f"{val:08x}-0000-1000-8000-00805f9b34fb"
    elif len(uuid_bytes) == 16:
        # 128-bit UUID (full)
        # Format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        parts = [
            uuid_bytes[0:4][::-1].hex(),  # Reverse for little-endian
            uuid_bytes[4:6][::-1].hex(),
            uuid_bytes[6:8][::-1].hex(),
            uuid_bytes[8:10].hex(),  # These stay in order
            uuid_bytes[10:16].hex(),
        ]
        return "-".join(parts)
    else:
        return uuid_bytes.hex()


def string_to_uuid_bytes(uuid_str: str) -> bytes:
    """Convert UUID string to bytes.

    Args:
        uuid_str: UUID string (e.g., "0x180D" or full format).

    Returns:
        UUID as bytes.

    Example:
        >>> string_to_uuid_bytes("0x180D")
        b'\\x0d\\x18'
    """
    if uuid_str.startswith("0x") or uuid_str.startswith("0X"):
        # Short format
        val = int(uuid_str, 16)
        if val <= 0xFFFF:
            return val.to_bytes(2, "little")
        else:
            return val.to_bytes(4, "little")
    elif "-" in uuid_str:
        # Full format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        uuid_str = uuid_str.replace("-", "")
        if len(uuid_str) != 32:
            raise ValueError(f"Invalid UUID format: {uuid_str}")
        # Convert back to bytes (handle endianness)
        parts = [
            bytes.fromhex(uuid_str[0:8])[::-1],  # Reverse for little-endian
            bytes.fromhex(uuid_str[8:12])[::-1],
            bytes.fromhex(uuid_str[12:16])[::-1],
            bytes.fromhex(uuid_str[16:20]),  # These stay in order
            bytes.fromhex(uuid_str[20:32]),
        ]
        return b"".join(parts)
    else:
        raise ValueError(f"Invalid UUID format: {uuid_str}")


def get_service_name(uuid: int | str) -> str:
    """Get standard service name from UUID.

    Args:
        uuid: Service UUID (16-bit int or string).

    Returns:
        Service name or "Unknown Service".

    Example:
        >>> get_service_name(0x180D)
        'Heart Rate'
    """
    if isinstance(uuid, str):
        if uuid.startswith("0x"):
            uuid = int(uuid, 16)
        else:
            return "Custom Service"
    return STANDARD_SERVICES.get(uuid, "Unknown Service")


def get_characteristic_name(uuid: int | str) -> str:
    """Get standard characteristic name from UUID.

    Args:
        uuid: Characteristic UUID (16-bit int or string).

    Returns:
        Characteristic name or "Unknown Characteristic".

    Example:
        >>> get_characteristic_name(0x2A37)
        'Heart Rate Measurement'
    """
    if isinstance(uuid, str):
        if uuid.startswith("0x"):
            uuid = int(uuid, 16)
        else:
            return "Custom Characteristic"
    return STANDARD_CHARACTERISTICS.get(uuid, "Unknown Characteristic")


def get_descriptor_name(uuid: int | str) -> str:
    """Get standard descriptor name from UUID.

    Args:
        uuid: Descriptor UUID (16-bit int or string).

    Returns:
        Descriptor name or "Unknown Descriptor".

    Example:
        >>> get_descriptor_name(0x2902)
        'Client Characteristic Configuration'
    """
    if isinstance(uuid, str):
        if uuid.startswith("0x"):
            uuid = int(uuid, 16)
        else:
            return "Custom Descriptor"
    return STANDARD_DESCRIPTORS.get(uuid, "Unknown Descriptor")


__all__ = [
    "AD_TYPES",
    "STANDARD_CHARACTERISTICS",
    "STANDARD_DESCRIPTORS",
    "STANDARD_SERVICES",
    "get_characteristic_name",
    "get_descriptor_name",
    "get_service_name",
    "string_to_uuid_bytes",
    "uuid_to_string",
]
