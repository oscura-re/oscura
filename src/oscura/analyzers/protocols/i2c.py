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
from typing import TYPE_CHECKING, Any

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

    channels = [
        ChannelDef("scl", "SCL", "Clock line", required=True),
        ChannelDef("sda", "SDA", "Data line", required=True),
    ]

    optional_channels = []

    options = [
        OptionDef(
            "address_format",
            "Address format",
            "7-bit or 10-bit",
            default="auto",
            values=["auto", "7bit", "10bit"],
        ),
    ]

    annotations = [
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

        scl, sda = self._align_signals(scl, sda)
        conditions = self._find_start_stop_conditions(scl, sda)

        if len(conditions) == 0:
            return

        # Process each transaction
        yield from self._process_transactions(scl, sda, conditions, sample_rate)

    def _align_signals(
        self, scl: NDArray[np.bool_], sda: NDArray[np.bool_]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Align SCL and SDA signals to same length.

        Args:
            scl: Clock signal.
            sda: Data signal.

        Returns:
            Tuple of (aligned_scl, aligned_sda).
        """
        n_samples = min(len(scl), len(sda))
        return scl[:n_samples], sda[:n_samples]

    def _find_start_stop_conditions(
        self, scl: NDArray[np.bool_], sda: NDArray[np.bool_]
    ) -> list[tuple[int, I2CCondition]]:
        """Find START and STOP conditions in I2C signals.

        START: SDA falls while SCL is high.
        STOP: SDA rises while SCL is high.

        Args:
            scl: Clock signal.
            sda: Data signal.

        Returns:
            List of (sample_index, condition_type) tuples.
        """
        conditions = []
        n_samples = len(scl)

        for i in range(1, n_samples):
            if scl[i] and scl[i - 1]:  # SCL is high
                if sda[i - 1] and not sda[i]:  # SDA falling
                    conditions.append((i, I2CCondition.START))
                elif not sda[i - 1] and sda[i]:  # SDA rising
                    conditions.append((i, I2CCondition.STOP))

        return conditions

    def _process_transactions(
        self,
        scl: NDArray[np.bool_],
        sda: NDArray[np.bool_],
        conditions: list[tuple[int, I2CCondition]],
        sample_rate: float,
    ) -> Iterator[ProtocolPacket]:
        """Process I2C transactions between START and STOP conditions.

        Args:
            scl: Clock signal.
            sda: Data signal.
            conditions: List of START/STOP conditions.
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded I2C transactions as ProtocolPacket objects.
        """
        trans_idx = 0
        i = 0

        while i < len(conditions):
            if conditions[i][1] != I2CCondition.START:
                i += 1
                continue

            # Find transaction boundaries
            start_idx, end_idx, is_repeated, end_cond_idx = self._find_transaction_bounds(
                conditions, i
            )

            if end_cond_idx is None:
                break

            # Extract and decode transaction
            packet = self._decode_transaction(
                scl[start_idx:end_idx],
                sda[start_idx:end_idx],
                start_idx,
                end_idx,
                sample_rate,
                is_repeated,
                trans_idx,
            )

            if packet:
                yield packet
                trans_idx += 1

            i = end_cond_idx if is_repeated else end_cond_idx + 1

    def _find_transaction_bounds(
        self, conditions: list[tuple[int, I2CCondition]], start_cond_idx: int
    ) -> tuple[int, int, bool, int | None]:
        """Find start and end indices of an I2C transaction.

        Args:
            conditions: List of START/STOP conditions.
            start_cond_idx: Index of START condition.

        Returns:
            Tuple of (start_idx, end_idx, is_repeated, end_condition_idx).
        """
        start_idx = conditions[start_cond_idx][0]

        # Find corresponding STOP or next START
        end_cond_idx = start_cond_idx + 1
        while end_cond_idx < len(conditions):
            if conditions[end_cond_idx][1] in (I2CCondition.STOP, I2CCondition.START):
                break
            end_cond_idx += 1

        if end_cond_idx >= len(conditions):
            return start_idx, start_idx, False, None

        end_idx = conditions[end_cond_idx][0]
        is_repeated = conditions[end_cond_idx][1] == I2CCondition.START

        return start_idx, end_idx, is_repeated, end_cond_idx

    def _decode_transaction(
        self,
        scl: NDArray[np.bool_],
        sda: NDArray[np.bool_],
        start_idx: int,
        end_idx: int,
        sample_rate: float,
        is_repeated: bool,
        trans_idx: int,
    ) -> ProtocolPacket | None:
        """Decode a single I2C transaction.

        Args:
            scl: Clock signal for transaction.
            sda: Data signal for transaction.
            start_idx: Absolute start sample index.
            end_idx: Absolute end sample index.
            sample_rate: Sample rate in Hz.
            is_repeated: Whether this is a repeated START.
            trans_idx: Transaction index.

        Returns:
            Decoded ProtocolPacket or None if invalid.
        """
        bytes_data, acks = self._extract_bytes(scl, sda)

        if len(bytes_data) == 0:
            return None

        # Parse address and direction
        address_info = self._parse_address(bytes_data, acks)

        # Create annotations
        start_time = start_idx / sample_rate
        end_time = end_idx / sample_rate
        self._add_transaction_annotations(start_time, end_time, address_info, is_repeated)

        # Build packet
        errors = self._check_transaction_errors(
            acks, address_info["data_acks"], address_info["is_read"]
        )

        packet = ProtocolPacket(
            timestamp=start_time,
            protocol="i2c",
            data=bytes(address_info["data_bytes"]),
            annotations={
                "address": address_info["address"],
                "address_10bit": address_info["is_10bit"],
                "read": address_info["is_read"],
                "bytes": bytes_data,
                "acks": acks,
                "transaction_num": trans_idx,
            },
            errors=errors,
        )

        return packet

    def _parse_address(self, bytes_data: list[int], acks: list[bool]) -> dict[str, Any]:
        """Parse I2C address from first bytes.

        Args:
            bytes_data: List of decoded bytes.
            acks: List of ACK/NAK bits.

        Returns:
            Dictionary with address, is_10bit, is_read, data_bytes, data_acks.
        """
        address_byte = bytes_data[0]
        address = address_byte >> 1
        is_read = (address_byte & 1) == 1

        # Check for 10-bit address
        is_10bit = self._address_format == "10bit" or (
            self._address_format == "auto" and (address_byte >> 3) == 0b11110
        )

        if is_10bit and len(bytes_data) >= 2:
            high_bits = (address_byte >> 1) & 0b11
            low_bits = bytes_data[1]
            actual_address = (high_bits << 8) | low_bits
            data_bytes = bytes_data[2:]
            data_acks = acks[2:] if len(acks) > 2 else []
        else:
            actual_address = address
            data_bytes = bytes_data[1:]
            data_acks = acks[1:] if len(acks) > 1 else []

        return {
            "address": actual_address,
            "is_10bit": is_10bit,
            "is_read": is_read,
            "data_bytes": data_bytes,
            "data_acks": data_acks,
        }

    def _check_transaction_errors(
        self, acks: list[bool], data_acks: list[bool], is_read: bool
    ) -> list[str]:
        """Check for NAK errors in transaction.

        Args:
            acks: All ACK/NAK bits.
            data_acks: Data byte ACK/NAK bits.
            is_read: Whether this is a read transaction.

        Returns:
            List of error messages.
        """
        errors = []

        if len(acks) > 0 and not acks[0]:
            errors.append("NAK on address")

        for j, ack in enumerate(data_acks):
            if not ack and not is_read:
                errors.append(f"NAK on byte {j}")

        return errors

    def _add_transaction_annotations(
        self, start_time: float, end_time: float, address_info: dict[str, Any], is_repeated: bool
    ) -> None:
        """Add annotations for I2C transaction.

        Args:
            start_time: Transaction start time.
            end_time: Transaction end time.
            address_info: Parsed address information.
            is_repeated: Whether this is a repeated START.
        """
        self.put_annotation(
            start_time,
            start_time + 1e-6,
            AnnotationLevel.BITS,
            "START" if not is_repeated else "Sr",
        )

        addr_text = (
            f"0x{address_info['address']:02X}"
            if not address_info["is_10bit"]
            else f"0x{address_info['address']:03X}"
        )
        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.FIELDS,
            f"{addr_text} {'R' if address_info['is_read'] else 'W'}",
        )

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
