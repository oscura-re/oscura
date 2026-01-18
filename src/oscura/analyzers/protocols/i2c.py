"""I2C protocol decoder.

This module provides I2C (Inter-Integrated Circuit) protocol decoding
with ACK/NAK detection, arbitration monitoring, and multi-speed support.


Example:
    >>> from oscura.analyzers.protocols.i2c import I2CDecoder
    >>> decoder = I2CDecoder()
    >>> for packet in decoder.decode(sda=sda, scl=scl):
    ...     print(f"Address: 0x{packet.annotations['address']:02X}")

References:
    I2C Specification (NXP UM10204)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    ChannelDef,
    OptionDef,
    SyncDecoder,
)
from oscura.core.types import DigitalTrace, ProtocolPacket

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class I2CCondition(Enum):
    """I2C bus conditions."""

    START = "start"
    STOP = "stop"
    REPEATED_START = "repeated_start"
    ACK = "ack"
    NAK = "nak"


@dataclass
class I2CTransaction:
    """I2C transaction record.

    Attributes:
        address: 7-bit or 10-bit device address.
        read: True for read, False for write.
        data: Data bytes transferred.
        acks: List of ACK (True) / NAK (False) for each byte.
        errors: List of detected errors.
    """

    address: int
    read: bool
    data: list[int]
    acks: list[bool]
    errors: list[str]


class I2CDecoder(SyncDecoder):
    """I2C protocol decoder.

    Decodes I2C bus transactions with ACK/NAK detection,
    arbitration monitoring, and support for standard, fast,
    and high-speed modes.

    Example:
        >>> decoder = I2CDecoder()
        >>> for packet in decoder.decode(sda=sda, scl=scl, sample_rate=10e6):
        ...     print(f"Addr: 0x{packet.annotations['address']:02X}")
        ...     print(f"Data: {packet.data.hex()}")
    """

    id = "i2c"
    name = "I2C"
    longname = "Inter-Integrated Circuit"
    desc = "I2C bus protocol decoder"

    channels = [  # noqa: RUF012
        ChannelDef("scl", "SCL", "Clock line", required=True),
        ChannelDef("sda", "SDA", "Data line", required=True),
    ]

    optional_channels = []  # noqa: RUF012

    options = [  # noqa: RUF012
        OptionDef(
            "address_format",
            "Address format",
            "7-bit or 10-bit",
            default="auto",
            values=["auto", "7bit", "10bit"],
        ),
    ]

    annotations = [  # noqa: RUF012
        ("start", "Start condition"),
        ("stop", "Stop condition"),
        ("address", "Device address"),
        ("data", "Data byte"),
        ("ack", "ACK"),
        ("nak", "NAK"),
        ("error", "Error"),
    ]

    def __init__(
        self,
        address_format: str = "auto",
    ) -> None:
        """Initialize I2C decoder.

        Args:
            address_format: Address format ("auto", "7bit", "10bit").
        """
        super().__init__(address_format=address_format)
        self._address_format = address_format

    def decode(  # type: ignore[override]
        self,
        trace: DigitalTrace | None = None,
        *,
        scl: NDArray[np.bool_] | None = None,
        sda: NDArray[np.bool_] | None = None,
        sample_rate: float = 1.0,
    ) -> Iterator[ProtocolPacket]:
        """Decode I2C transactions.

        Args:
            trace: Optional primary trace.
            scl: Clock signal.
            sda: Data signal.
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded I2C transactions as ProtocolPacket objects.

        Example:
            >>> decoder = I2CDecoder()
            >>> for pkt in decoder.decode(scl=scl, sda=sda, sample_rate=10e6):
            ...     print(f"Address: 0x{pkt.annotations['address']:02X}")
        """
        if scl is None or sda is None:
            return

        n_samples = min(len(scl), len(sda))
        scl = scl[:n_samples]
        sda = sda[:n_samples]

        # Find start and stop conditions
        # START: SDA falls while SCL is high
        # STOP: SDA rises while SCL is high

        conditions = []

        for i in range(1, n_samples):
            if scl[i] and scl[i - 1]:  # SCL is high
                if sda[i - 1] and not sda[i]:  # SDA falling
                    conditions.append((i, I2CCondition.START))
                elif not sda[i - 1] and sda[i]:  # SDA rising
                    conditions.append((i, I2CCondition.STOP))

        if len(conditions) == 0:
            return

        # Process transactions between START and STOP
        trans_idx = 0
        i = 0

        while i < len(conditions):
            if conditions[i][1] != I2CCondition.START:
                i += 1
                continue

            start_idx = conditions[i][0]
            start_time = start_idx / sample_rate

            # Find corresponding STOP or next START
            end_cond_idx = i + 1
            while end_cond_idx < len(conditions):
                if conditions[end_cond_idx][1] == I2CCondition.STOP:
                    break
                if conditions[end_cond_idx][1] == I2CCondition.START:
                    # Repeated START
                    break
                end_cond_idx += 1

            if end_cond_idx >= len(conditions):
                break

            end_idx = conditions[end_cond_idx][0]
            is_repeated = conditions[end_cond_idx][1] == I2CCondition.START

            # Extract bytes from this transaction
            bytes_data, acks = self._extract_bytes(
                scl[start_idx:end_idx],
                sda[start_idx:end_idx],
            )

            if len(bytes_data) == 0:
                i = end_cond_idx
                continue

            # First byte is address + R/W
            address_byte = bytes_data[0]
            address = address_byte >> 1
            is_read = (address_byte & 1) == 1

            # Check for 10-bit address
            is_10bit = False
            actual_address = address

            if self._address_format == "10bit" or (
                self._address_format == "auto" and (address_byte >> 3) == 0b11110
            ):
                # 10-bit address format
                if len(bytes_data) >= 2:
                    is_10bit = True
                    high_bits = (address_byte >> 1) & 0b11
                    low_bits = bytes_data[1]
                    actual_address = (high_bits << 8) | low_bits
                    data_bytes = bytes_data[2:]
                    data_acks = acks[2:] if len(acks) > 2 else []
                else:
                    data_bytes = []
                    data_acks = []
            else:
                actual_address = address
                data_bytes = bytes_data[1:]
                data_acks = acks[1:] if len(acks) > 1 else []

            # Check for errors
            errors = []
            if len(acks) > 0 and not acks[0]:
                errors.append("NAK on address")

            for j, (_byte, ack) in enumerate(zip(data_bytes, data_acks, strict=False)):
                if not ack and not is_read:
                    errors.append(f"NAK on byte {j}")

            end_time = end_idx / sample_rate

            # Add annotations
            self.put_annotation(
                start_time,
                start_time + 1e-6,
                AnnotationLevel.BITS,
                "START" if not is_repeated else "Sr",
            )

            addr_text = f"0x{actual_address:02X}" if not is_10bit else f"0x{actual_address:03X}"
            self.put_annotation(
                start_time,
                end_time,
                AnnotationLevel.FIELDS,
                f"{addr_text} {'R' if is_read else 'W'}",
            )

            # Create packet
            annotations = {
                "address": actual_address,
                "address_10bit": is_10bit,
                "read": is_read,
                "bytes": bytes_data,
                "acks": acks,
                "transaction_num": trans_idx,
            }

            packet = ProtocolPacket(
                timestamp=start_time,
                protocol="i2c",
                data=bytes(data_bytes),
                annotations=annotations,
                errors=errors,
            )

            yield packet

            trans_idx += 1
            i = end_cond_idx

            if is_repeated:
                continue
            else:
                i += 1

    def _extract_bytes(
        self,
        scl: NDArray[np.bool_],
        sda: NDArray[np.bool_],
    ) -> tuple[list[int], list[bool]]:
        """Extract bytes from I2C transaction.

        Args:
            scl: Clock signal segment.
            sda: Data signal segment.

        Returns:
            (bytes, acks) - List of byte values and ACK flags.
        """
        # Find rising edges of SCL (data sampling points)
        rising_edges = np.where(~scl[:-1] & scl[1:])[0] + 1

        if len(rising_edges) < 9:  # Need at least 8 data bits + ACK
            return [], []

        bytes_data = []
        acks = []

        i = 0
        while i + 9 <= len(rising_edges):
            # Extract 8 data bits (MSB first)
            byte_val = 0
            for bit_idx in range(8):
                sample_idx = rising_edges[i + bit_idx]
                if sample_idx < len(sda):
                    bit = 1 if sda[sample_idx] else 0
                    byte_val = (byte_val << 1) | bit

            # Extract ACK bit (9th bit, low = ACK, high = NAK)
            ack_idx = rising_edges[i + 8]
            if ack_idx < len(sda):
                ack = not sda[ack_idx]  # Low = ACK
            else:
                ack = False

            bytes_data.append(byte_val)
            acks.append(ack)

            i += 9

        return bytes_data, acks


def decode_i2c(
    scl: NDArray[np.bool_],
    sda: NDArray[np.bool_],
    sample_rate: float = 1.0,
    address_format: str = "auto",
) -> list[ProtocolPacket]:
    """Convenience function to decode I2C transactions.

    Args:
        scl: Clock signal.
        sda: Data signal.
        sample_rate: Sample rate in Hz.
        address_format: Address format ("auto", "7bit", "10bit").

    Returns:
        List of decoded I2C transactions.

    Example:
        >>> packets = decode_i2c(scl, sda, sample_rate=10e6)
        >>> for pkt in packets:
        ...     print(f"Address: 0x{pkt.annotations['address']:02X}")
    """
    decoder = I2CDecoder(address_format=address_format)
    return list(decoder.decode(scl=scl, sda=sda, sample_rate=sample_rate))


__all__ = ["I2CCondition", "I2CDecoder", "I2CTransaction", "decode_i2c"]
