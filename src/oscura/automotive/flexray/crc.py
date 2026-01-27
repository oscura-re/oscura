"""FlexRay CRC calculation algorithms.

This module implements FlexRay CRC algorithms:
- Header CRC-11 (protects frame header)
- Frame CRC-24 (protects header + payload)

References:
    FlexRay Communications System Protocol Specification Version 3.0.1
    Section 4.5: Error Detection Mechanisms
"""

from __future__ import annotations


def calculate_header_crc(
    reserved: int,
    payload_preamble: int,
    null_frame: int,
    sync_frame: int,
    startup_frame: int,
    frame_id: int,
    payload_length: int,
) -> int:
    """Calculate 11-bit FlexRay header CRC.

    The header CRC protects the first 29 bits of the header (excluding the CRC
    itself and the cycle count field).

    CRC polynomial: x^11 + x^9 + x^8 + x^7 + x^2 + 1 (0x385)
    Initial value: 0x01A

    Args:
        reserved: Reserved bit (1 bit).
        payload_preamble: Payload preamble indicator (1 bit).
        null_frame: Null frame indicator (1 bit).
        sync_frame: Sync frame indicator (1 bit).
        startup_frame: Startup frame indicator (1 bit).
        frame_id: Frame ID (11 bits, 1-2047).
        payload_length: Payload length in 16-bit words (7 bits, 0-127).

    Returns:
        11-bit header CRC value.

    Example:
        >>> crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)
        >>> print(f"Header CRC: 0x{crc:03X}")
    """
    # Combine header fields (29 bits without CRC and cycle count)
    # Bit positions from MSB:
    # [28] reserved
    # [27] payload_preamble
    # [26] null_frame
    # [25] sync_frame
    # [24] startup_frame
    # [23:13] frame_id (11 bits)
    # [12:6] payload_length (7 bits)
    # [5:0] padding (will be shifted out during calculation)
    data = (
        (reserved << 28)
        | (payload_preamble << 27)
        | (null_frame << 26)
        | (sync_frame << 25)
        | (startup_frame << 24)
        | (frame_id << 13)
        | (payload_length << 6)
    )

    # CRC-11 calculation
    crc = 0x01A  # Initial value
    polynomial = 0x385  # x^11 + x^9 + x^8 + x^7 + x^2 + 1

    # Process 29 data bits
    for i in range(29):
        bit = (data >> (28 - i)) & 1
        msb = (crc >> 10) & 1

        crc = (crc << 1) & 0x7FF  # Shift and keep 11 bits
        if msb ^ bit:
            crc ^= polynomial

    return crc & 0x7FF


def calculate_frame_crc(header: bytes, payload: bytes) -> int:
    """Calculate 24-bit FlexRay frame CRC.

    The frame CRC protects the entire frame (header + payload).

    CRC polynomial: x^24 + x^22 + x^20 + x^19 + x^18 + x^16 + x^14 +
                    x^13 + x^11 + x^10 + x^8 + x^7 + x^6 + x^3 + x^1 + 1
                    (0x5D6DCB)
    Initial value: 0xFEDCBA

    Args:
        header: Frame header bytes (5 bytes).
        payload: Frame payload bytes (0-254 bytes).

    Returns:
        24-bit frame CRC value.

    Example:
        >>> header = bytes([0x00, 0x64, 0x12, 0x34, 0x05])
        >>> payload = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        >>> crc = calculate_frame_crc(header, payload)
        >>> print(f"Frame CRC: 0x{crc:06X}")
    """
    data = header + payload

    crc = 0xFEDCBA  # Initial value
    polynomial = 0x5D6DCB  # FlexRay CRC-24 polynomial

    for byte in data:
        crc ^= byte << 16

        for _ in range(8):
            if crc & 0x800000:
                crc = ((crc << 1) ^ polynomial) & 0xFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFF

    return crc


def verify_header_crc(
    reserved: int,
    payload_preamble: int,
    null_frame: int,
    sync_frame: int,
    startup_frame: int,
    frame_id: int,
    payload_length: int,
    received_crc: int,
) -> bool:
    """Verify FlexRay header CRC.

    Args:
        reserved: Reserved bit.
        payload_preamble: Payload preamble indicator.
        null_frame: Null frame indicator.
        sync_frame: Sync frame indicator.
        startup_frame: Startup frame indicator.
        frame_id: Frame ID (11 bits).
        payload_length: Payload length in words (7 bits).
        received_crc: Received CRC value to verify.

    Returns:
        True if CRC is valid, False otherwise.

    Example:
        >>> valid = verify_header_crc(0, 0, 0, 0, 0, 100, 5, 0x3A5)
        >>> print(f"Header CRC valid: {valid}")
    """
    computed_crc = calculate_header_crc(
        reserved, payload_preamble, null_frame, sync_frame, startup_frame, frame_id, payload_length
    )
    return computed_crc == received_crc


def verify_frame_crc(header: bytes, payload: bytes, received_crc: int) -> bool:
    """Verify FlexRay frame CRC.

    Args:
        header: Frame header bytes (5 bytes).
        payload: Frame payload bytes.
        received_crc: Received CRC value to verify.

    Returns:
        True if CRC is valid, False otherwise.

    Example:
        >>> header = bytes([0x00, 0x64, 0x12, 0x34, 0x05])
        >>> payload = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        >>> valid = verify_frame_crc(header, payload, 0x123456)
        >>> print(f"Frame CRC valid: {valid}")
    """
    computed_crc = calculate_frame_crc(header, payload)
    return computed_crc == received_crc


__all__ = [
    "calculate_frame_crc",
    "calculate_header_crc",
    "verify_frame_crc",
    "verify_header_crc",
]
