"""EtherCAT mailbox protocol parsers.

This module provides parsers for EtherCAT mailbox protocols including
CoE (CAN over EtherCAT), FoE (File over EtherCAT), SoE (Servo over EtherCAT),
and EoE (Ethernet over EtherCAT).

Example:
    >>> from oscura.analyzers.protocols.industrial.ethercat.mailbox import parse_mailbox
    >>> mailbox_data = bytes([0x06, 0x00, 0x00, 0x00, 0x02, ...])
    >>> result = parse_mailbox(mailbox_data)
    >>> print(f"Protocol: {result['protocol']}")

References:
    ETG.1000.6 CoE Protocol
    ETG.8200 FoE Protocol
    ETG.1000.4 SoE Protocol
    ETG.8500 EoE Protocol
"""

from __future__ import annotations

from typing import Any


def parse_mailbox(data: bytes) -> dict[str, Any]:
    """Parse EtherCAT mailbox protocol data.

    Mailbox Header Format:
    - Length (2 bytes, little-endian) - Data length
    - Address (2 bytes, little-endian) - Station address
    - Channel (6 bits) - Priority and channel info
    - Priority (2 bits)
    - Type (4 bits) - Protocol type
    - Counter (3 bits)
    - Reserved (1 bit)

    Args:
        data: Mailbox data bytes.

    Returns:
        Parsed mailbox data dictionary.

    Raises:
        ValueError: If data is invalid.

    Example:
        >>> # CoE mailbox data
        >>> data = bytes([0x0A, 0x00, 0x01, 0x00, 0x02, 0x00, ...])
        >>> result = parse_mailbox(data)
        >>> assert result['protocol'] == 'CoE'
    """
    if len(data) < 6:
        raise ValueError(f"Mailbox data too short: {len(data)} bytes (minimum 6)")

    # Parse mailbox header
    length = int.from_bytes(data[0:2], "little")
    address = int.from_bytes(data[2:4], "little")
    channel_priority = data[4]
    type_counter = data[5]

    # Extract fields
    channel = channel_priority & 0x3F  # Lower 6 bits
    priority = (channel_priority >> 6) & 0x03  # Upper 2 bits
    protocol_type = type_counter & 0x0F  # Lower 4 bits
    counter = (type_counter >> 4) & 0x07  # Bits 4-6

    # Map protocol type
    protocol_names = {
        0x00: "ERR",  # Error
        0x02: "CoE",  # CAN application protocol over EtherCAT
        0x03: "FoE",  # File access over EtherCAT
        0x04: "SoE",  # Servo drive profile over EtherCAT
        0x05: "EoE",  # Ethernet over EtherCAT
    }

    protocol = protocol_names.get(protocol_type, f"Unknown (0x{protocol_type:X})")

    result: dict[str, Any] = {
        "length": length,
        "address": address,
        "channel": channel,
        "priority": priority,
        "protocol": protocol,
        "protocol_type": protocol_type,
        "counter": counter,
    }

    # Parse protocol-specific data
    if len(data) > 6:
        protocol_data = data[6:]

        if protocol_type == 0x02:  # CoE
            result["coe"] = _parse_coe(protocol_data)
        elif protocol_type == 0x03:  # FoE
            result["foe"] = _parse_foe(protocol_data)
        elif protocol_type == 0x04:  # SoE
            result["soe"] = _parse_soe(protocol_data)
        elif protocol_type == 0x05:  # EoE
            result["eoe"] = _parse_eoe(protocol_data)

    return result


def _parse_coe(data: bytes) -> dict[str, Any]:
    """Parse CoE (CAN over EtherCAT) protocol data.

    CoE Header Format:
    - Number (2 bytes, little-endian) - CAN ID or SDO index
    - Service (4 bits) - CoE service type
    - Reserved (4 bits)

    Args:
        data: CoE protocol data.

    Returns:
        Parsed CoE data.
    """
    if len(data) < 2:
        return {"error": "CoE data too short"}

    number = int.from_bytes(data[0:2], "little")

    result: dict[str, Any] = {
        "number": number,
    }

    if len(data) >= 3:
        service = data[2] & 0x0F

        # CoE service types
        service_names = {
            0x01: "SDO Request",
            0x02: "SDO Response",
            0x03: "TxPDO",
            0x04: "RxPDO",
            0x05: "TxPDO Remote Request",
            0x06: "RxPDO Remote Request",
            0x07: "SDO Information",
        }

        result["service"] = service_names.get(service, f"Unknown (0x{service:X})")
        result["service_type"] = service

        # Parse SDO data if present
        if service in {0x01, 0x02} and len(data) >= 10:  # SDO Request/Response
            result["sdo"] = _parse_sdo(data[3:])  # Skip number (2 bytes) + service (1 byte)

    return result


def _parse_sdo(data: bytes) -> dict[str, Any]:
    """Parse SDO (Service Data Object) data.

    Args:
        data: SDO data bytes.

    Returns:
        Parsed SDO data.
    """
    if len(data) < 8:
        return {"error": "SDO data too short"}

    command = data[0]
    index = int.from_bytes(data[1:3], "little")
    subindex = data[3]
    data_bytes = data[4:8]

    # SDO command specifiers
    command_type = command >> 5  # Upper 3 bits

    command_names = {
        0x01: "Download Initiate",
        0x02: "Download Segment",
        0x03: "Upload Initiate",
        0x04: "Upload Segment",
        0x05: "Abort",
    }

    return {
        "command": command_names.get(command_type, f"Unknown (0x{command_type:X})"),
        "command_code": command,
        "index": index,
        "subindex": subindex,
        "data": data_bytes.hex(),
    }


def _parse_foe(data: bytes) -> dict[str, Any]:
    """Parse FoE (File over EtherCAT) protocol data.

    FoE Header Format:
    - OpCode (1 byte) - Operation code
    - Reserved (1 byte)
    - Packet Number (4 bytes, little-endian) - For data transfer

    Args:
        data: FoE protocol data.

    Returns:
        Parsed FoE data.
    """
    if len(data) < 1:
        return {"error": "FoE data too short"}

    opcode = data[0]

    # FoE operation codes
    opcode_names = {
        0x01: "Read Request",
        0x02: "Write Request",
        0x03: "Data",
        0x04: "Ack",
        0x05: "Error",
        0x06: "Busy",
    }

    result: dict[str, Any] = {
        "opcode": opcode_names.get(opcode, f"Unknown (0x{opcode:02X})"),
        "opcode_code": opcode,
    }

    if len(data) >= 6:
        packet_number = int.from_bytes(data[2:6], "little")
        result["packet_number"] = packet_number

    return result


def _parse_soe(data: bytes) -> dict[str, Any]:
    """Parse SoE (Servo over EtherCAT) protocol data.

    SoE Header Format:
    - OpCode (3 bits) - Operation code
    - InComplete (1 bit)
    - Error (1 bit)
    - DriveNo (3 bits)
    - Elements (2 bytes, little-endian) - Number of elements
    - IDN (2 bytes, little-endian) - Identification number

    Args:
        data: SoE protocol data.

    Returns:
        Parsed SoE data.
    """
    if len(data) < 4:
        return {"error": "SoE data too short"}

    header = data[0]
    opcode = header & 0x07  # Lower 3 bits
    incomplete = bool(header & 0x08)  # Bit 3
    error = bool(header & 0x10)  # Bit 4
    drive_no = (header >> 5) & 0x07  # Upper 3 bits

    # SoE operation codes
    opcode_names = {
        0x00: "No Operation",
        0x01: "Read Request",
        0x02: "Read Response",
        0x03: "Write Request",
        0x04: "Write Response",
        0x05: "Notification Request",
        0x06: "Emergency",
    }

    result: dict[str, Any] = {
        "opcode": opcode_names.get(opcode, f"Unknown (0x{opcode:X})"),
        "opcode_code": opcode,
        "incomplete": incomplete,
        "error": error,
        "drive_number": drive_no,
    }

    if len(data) >= 4:
        idn = int.from_bytes(data[2:4], "little")
        result["idn"] = idn

    return result


def _parse_eoe(data: bytes) -> dict[str, Any]:
    """Parse EoE (Ethernet over EtherCAT) protocol data.

    EoE Header Format:
    - Type (4 bits) - Fragment type
    - Port (4 bits) - Port number
    - Last Fragment (1 bit)
    - Time Append (1 bit)
    - Time Request (1 bit)
    - Reserved (5 bits)
    - Fragment Number (6 bits)
    - Frame Offset (6 bits)
    - Frame Number (4 bytes, little-endian)

    Args:
        data: EoE protocol data.

    Returns:
        Parsed EoE data.
    """
    if len(data) < 4:
        return {"error": "EoE data too short"}

    type_port = data[0]
    flags = data[1]
    frag_offset = int.from_bytes(data[2:4], "little")

    frame_type = type_port & 0x0F  # Lower 4 bits
    port = (type_port >> 4) & 0x0F  # Upper 4 bits

    last_fragment = bool(flags & 0x01)  # Bit 0
    time_append = bool(flags & 0x02)  # Bit 1
    time_request = bool(flags & 0x04)  # Bit 2

    fragment_number = frag_offset & 0x3F  # Lower 6 bits
    frame_offset = (frag_offset >> 6) & 0x3F  # Bits 6-11

    # EoE frame types
    type_names = {
        0x00: "Fragment",
        0x01: "Init Request",
        0x02: "Init Response",
    }

    result: dict[str, Any] = {
        "type": type_names.get(frame_type, f"Unknown (0x{frame_type:X})"),
        "type_code": frame_type,
        "port": port,
        "last_fragment": last_fragment,
        "time_append": time_append,
        "time_request": time_request,
        "fragment_number": fragment_number,
        "frame_offset": frame_offset,
    }

    return result


__all__ = ["parse_mailbox"]
