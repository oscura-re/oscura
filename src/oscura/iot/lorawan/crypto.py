"""LoRaWAN cryptographic operations.

This module provides AES-128 encryption/decryption and CMAC operations
for LoRaWAN payload security as defined in LoRaWAN Specification 1.0.3.

Security:
    Uses cryptography library (modern, actively maintained) instead of deprecated PyCrypto.
    All cryptographic operations follow LoRaWAN 1.0.3 specification.

References:
    LoRaWAN Specification 1.0.3: https://lora-alliance.org/resource_hub/lorawan-specification-v1-0-3/
    Section 4.3.3 - MAC Frame Payload Encryption (FRMPayload)
    Section 4.4 - Message Integrity Code (MIC)
"""

from __future__ import annotations

from typing import Literal


def decrypt_payload(
    frm_payload: bytes,
    key: bytes,
    dev_addr: int,
    fcnt: int,
    direction: Literal["up", "down"],
) -> bytes:
    """Decrypt LoRaWAN FRMPayload using AES-128 in CTR mode.

    Args:
        frm_payload: Encrypted payload bytes.
        key: AES-128 key (16 bytes) - AppSKey or NwkSKey.
        dev_addr: Device address (4 bytes).
        fcnt: Frame counter.
        direction: Direction "up" (uplink) or "down" (downlink).

    Returns:
        Decrypted payload bytes.

    Raises:
        ValueError: If key is not 16 bytes.
        ImportError: If cryptography library is not available.

    Example:
        >>> key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        >>> encrypted = bytes.fromhex("0123456789ABCDEF")
        >>> decrypted = decrypt_payload(encrypted, key, 0x01020304, 1, "up")
    """
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    except ImportError as exc:
        msg = (
            "cryptography library is required for LoRaWAN encryption. "
            "Install with: pip install cryptography"
        )
        raise ImportError(msg) from exc

    if len(key) != 16:
        msg = f"Key must be 16 bytes, got {len(key)}"
        raise ValueError(msg)

    if len(frm_payload) == 0:
        return b""

    # Direction byte: 0x00 for uplink, 0x01 for downlink
    dir_byte = 0x00 if direction == "up" else 0x01

    # Number of 16-byte blocks needed
    num_blocks = (len(frm_payload) + 15) // 16

    # Generate keystream by encrypting counter blocks
    keystream = b""
    for i in range(num_blocks):
        # Build encryption block A_i (16 bytes)
        # A_i = 0x01 | 0x00000000 | Dir | DevAddr | FCnt | 0x00 | i
        a = bytearray(16)
        a[0] = 0x01  # Encryption flag
        # a[1:5] = 0x00000000 (already zero)
        a[5] = dir_byte
        a[6:10] = dev_addr.to_bytes(4, "little")
        a[10:14] = fcnt.to_bytes(4, "little")
        # a[14] = 0x00 (already zero)
        a[15] = i + 1

        # Encrypt the block using AES in ECB mode
        # Security: ECB mode is required by LoRaWAN spec (Section 4.3.3)
        # Each counter block is unique, so ECB mode is safe here
        cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        keystream += encryptor.update(bytes(a)) + encryptor.finalize()

    # XOR payload with keystream
    decrypted = bytes(
        p ^ k for p, k in zip(frm_payload, keystream[: len(frm_payload)], strict=False)
    )
    return decrypted


def compute_mic(
    data: bytes,
    key: bytes,
    dev_addr: int,
    fcnt: int,
    direction: Literal["up", "down"],
) -> int:
    """Compute LoRaWAN Message Integrity Code (MIC) using AES-CMAC.

    Args:
        data: Message data (MHDR | FHDR | FPort | FRMPayload).
        key: Network session key (NwkSKey, 16 bytes).
        dev_addr: Device address (4 bytes).
        fcnt: Frame counter.
        direction: Direction "up" (uplink) or "down" (downlink).

    Returns:
        32-bit MIC value.

    Raises:
        ValueError: If key is not 16 bytes.
        ImportError: If cryptography library is not available.

    Example:
        >>> key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        >>> data = bytes.fromhex("40010203040001000100")
        >>> mic = compute_mic(data, key, 0x01020304, 1, "up")
    """
    try:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import cmac
        from cryptography.hazmat.primitives.ciphers import algorithms
    except ImportError as exc:
        msg = (
            "cryptography library is required for LoRaWAN MIC computation. "
            "Install with: pip install cryptography"
        )
        raise ImportError(msg) from exc

    if len(key) != 16:
        msg = f"Key must be 16 bytes, got {len(key)}"
        raise ValueError(msg)

    # Direction byte: 0x00 for uplink, 0x01 for downlink
    dir_byte = 0x00 if direction == "up" else 0x01

    # Build MIC computation block B_0 (16 bytes)
    # B_0 = 0x49 | 0x00000000 | Dir | DevAddr | FCnt | 0x00 | len(msg)
    b0 = bytearray(16)
    b0[0] = 0x49  # MIC flag
    # b0[1:5] = 0x00000000 (already zero)
    b0[5] = dir_byte
    b0[6:10] = dev_addr.to_bytes(4, "little")
    b0[10:14] = fcnt.to_bytes(4, "little")
    # b0[14] = 0x00 (already zero)
    b0[15] = len(data)

    # Compute CMAC over B_0 | msg
    c = cmac.CMAC(algorithms.AES(key), backend=default_backend())
    c.update(bytes(b0))
    c.update(data)
    mac = c.finalize()

    # Return first 4 bytes as 32-bit integer (little-endian)
    mic = int.from_bytes(mac[:4], "little")
    return mic


def verify_mic(
    data: bytes,
    received_mic: int,
    key: bytes,
    dev_addr: int,
    fcnt: int,
    direction: Literal["up", "down"],
) -> bool:
    """Verify LoRaWAN Message Integrity Code (MIC).

    Args:
        data: Message data (MHDR | FHDR | FPort | FRMPayload).
        received_mic: Received MIC value.
        key: Network session key (NwkSKey, 16 bytes).
        dev_addr: Device address (4 bytes).
        fcnt: Frame counter.
        direction: Direction "up" (uplink) or "down" (downlink).

    Returns:
        True if MIC is valid, False otherwise.

    Example:
        >>> key = bytes.fromhex("2B7E151628AED2A6ABF7158809CF4F3C")
        >>> data = bytes.fromhex("40010203040001000100")
        >>> is_valid = verify_mic(data, 0x12345678, key, 0x01020304, 1, "up")
    """
    try:
        computed_mic = compute_mic(data, key, dev_addr, fcnt, direction)
        return computed_mic == received_mic
    except (ImportError, ValueError):
        # If crypto is unavailable, we can't verify
        return False


__all__ = [
    "compute_mic",
    "decrypt_payload",
    "verify_mic",
]
