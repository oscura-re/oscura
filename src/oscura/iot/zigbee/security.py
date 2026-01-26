"""Zigbee security and encryption support.

This module provides utilities for handling Zigbee security frames
including network key management and frame decryption support.

Note: This module provides structure parsing only. Actual AES-CCM
decryption requires cryptography libraries and valid network keys.

References:
    Zigbee Specification Section 4.5 (Security)
"""

from __future__ import annotations

from typing import Any


def parse_security_header(data: bytes) -> dict[str, Any]:
    """Parse Zigbee security header from NWK auxiliary header.

    Args:
        data: Security header bytes.

    Returns:
        Parsed security header fields.

    Example:
        >>> header = parse_security_header(security_data)
        >>> print(header['security_level'])
        5
    """
    if len(data) < 5:
        return {"error": "Insufficient data for security header"}

    security_control = data[0]
    frame_counter = int.from_bytes(data[1:5], "little")

    security_level = security_control & 0x07
    key_identifier = (security_control >> 3) & 0x03
    extended_nonce = bool(security_control & 0x20)

    result: dict[str, Any] = {
        "security_control": security_control,
        "security_level": security_level,
        "key_identifier": key_identifier,
        "extended_nonce": extended_nonce,
        "frame_counter": frame_counter,
    }

    offset = 5

    # Extended nonce includes source address
    if extended_nonce:
        if len(data) < offset + 8:
            result["error"] = "Insufficient data for extended nonce"
            return result
        source_address = int.from_bytes(data[offset : offset + 8], "little")
        result["source_address"] = source_address
        offset += 8

    # Key sequence number (if key identifier indicates it)
    if key_identifier == 0x01:  # Network key
        if len(data) < offset + 1:
            result["error"] = "Insufficient data for key sequence"
            return result
        result["key_sequence_number"] = data[offset]
        offset += 1

    result["header_length"] = offset
    return result


def is_frame_encrypted(nwk_frame_control: int) -> bool:
    """Check if NWK frame is encrypted.

    Args:
        nwk_frame_control: NWK frame control field (2 bytes as int).

    Returns:
        True if frame is encrypted, False otherwise.

    Example:
        >>> frame_control = 0x0208  # Security enabled
        >>> is_frame_encrypted(frame_control)
        True
    """
    # Security bit is bit 1 of the high byte (bit 9 overall)
    return bool((nwk_frame_control >> 9) & 0x01)


def get_security_level_name(level: int) -> str:
    """Get human-readable security level name.

    Args:
        level: Security level (0-7).

    Returns:
        Security level name.

    Example:
        >>> get_security_level_name(5)
        'ENC-MIC-32'
    """
    levels = {
        0x00: "None",
        0x01: "MIC-32",
        0x02: "MIC-64",
        0x03: "MIC-128",
        0x04: "ENC",
        0x05: "ENC-MIC-32",
        0x06: "ENC-MIC-64",
        0x07: "ENC-MIC-128",
    }
    return levels.get(level, f"Unknown ({level})")


def decrypt_frame(
    encrypted_payload: bytes,
    network_key: bytes,
    nonce: bytes,
    security_level: int,
) -> bytes | None:
    """Decrypt Zigbee encrypted frame.

    Note: This is a placeholder implementation. Actual decryption
    requires AES-CCM implementation with proper nonce construction.

    Args:
        encrypted_payload: Encrypted payload with MIC.
        network_key: 128-bit AES network key.
        nonce: 13-byte nonce.
        security_level: Security level (determines MIC size).

    Returns:
        Decrypted payload or None if decryption not available.

    Example:
        >>> # Requires cryptography library
        >>> decrypted = decrypt_frame(payload, key, nonce, 5)
    """
    # Placeholder - real implementation would use AES-CCM
    # from cryptography.hazmat.primitives.ciphers.aead import AESCCM
    # cipher = AESCCM(network_key)
    # return cipher.decrypt(nonce, encrypted_payload, None)
    return None  # Not implemented without cryptography dependencies


__all__ = [
    "decrypt_frame",
    "get_security_level_name",
    "is_frame_encrypted",
    "parse_security_header",
]
