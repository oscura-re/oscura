"""Modbus RTU CRC-16 calculation.

This module provides CRC-16 calculation and validation for Modbus RTU frames.

The Modbus RTU CRC-16 uses polynomial 0xA001 (reversed 0x8005).

Example:
    >>> from oscura.analyzers.protocols.industrial.modbus.crc import calculate_crc
    >>> data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
    >>> crc = calculate_crc(data)
    >>> print(f"CRC: 0x{crc:04X}")

References:
    Modbus over Serial Line Specification V1.02 (Section 6.2.2)
    https://modbus.org/docs/Modbus_over_serial_line_V1_02.pdf
"""

from __future__ import annotations


def calculate_crc(data: bytes) -> int:
    """Calculate Modbus RTU CRC-16.

    Uses the polynomial 0xA001 (bit-reversed representation of 0x8005).
    This is the standard CRC-16-IBM/CRC-16-ANSI with reflected input/output.

    Args:
        data: Data bytes to calculate CRC for (excludes CRC itself).

    Returns:
        16-bit CRC value as integer.

    Example:
        >>> data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        >>> crc = calculate_crc(data)
        >>> assert crc == 0xC5CD  # Expected CRC for this data
    """
    crc = 0xFFFF

    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1

    return crc


def verify_crc(data: bytes) -> bool:
    """Verify Modbus RTU CRC-16.

    Checks if the last 2 bytes of data contain the correct CRC for the
    preceding bytes.

    Args:
        data: Complete frame including CRC (last 2 bytes, little-endian).

    Returns:
        True if CRC is valid, False otherwise.

    Example:
        >>> frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A, 0xCD, 0xC5])
        >>> assert verify_crc(frame) is True
    """
    if len(data) < 4:  # Minimum: Address + FC + CRC
        return False

    # Calculate CRC for all bytes except last 2
    calculated = calculate_crc(data[:-2])

    # Extract CRC from last 2 bytes (little-endian)
    received = int.from_bytes(data[-2:], "little")

    return calculated == received


__all__ = ["calculate_crc", "verify_crc"]
