"""SWD protocol decoder.

This module provides ARM Serial Wire Debug (SWD) protocol decoding
with DP/AP access detection and ACK/WAIT/FAULT response handling.


Example:
    >>> from oscura.analyzers.protocols.swd import SWDDecoder
    >>> decoder = SWDDecoder()
    >>> for packet in decoder.decode(swclk=swclk, swdio=swdio):
    ...     print(f"Request: {packet.annotations['request_type']}")

References:
    ARM Debug Interface Architecture Specification ADIv5.0 to ADIv5.2
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    ChannelDef,
    SyncDecoder,
)
from oscura.core.types import DigitalTrace, ProtocolPacket

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class SWDResponse(Enum):
    """SWD ACK responses."""

    OK = 0b001
    WAIT = 0b010
    FAULT = 0b100


class SWDDecoder(SyncDecoder):
    """SWD protocol decoder.

    Decodes ARM Serial Wire Debug transactions including read/write
    operations to Debug Port (DP) and Access Port (AP) registers.

    Attributes:
        id: "swd"
        name: "SWD"
        channels: [swclk, swdio] (required)

    Example:
        >>> decoder = SWDDecoder()
        >>> for packet in decoder.decode(swclk=swclk, swdio=swdio, sample_rate=10e6):
        ...     print(f"ACK: {packet.annotations['ack']}")
    """

    id = "swd"
    name = "SWD"
    longname = "Serial Wire Debug"
    desc = "ARM Serial Wire Debug protocol decoder"

    channels = [
        ChannelDef("swclk", "SWCLK", "Serial Wire Clock", required=True),
        ChannelDef("swdio", "SWDIO", "Serial Wire Data I/O", required=True),
    ]

    optional_channels = []

    options = []

    annotations = [
        ("request", "Request packet"),
        ("ack", "ACK response"),
        ("data", "Data phase"),
        ("parity", "Parity bit"),
        ("error", "Error"),
    ]

    def __init__(self) -> None:
        """Initialize SWD decoder."""
        super().__init__()

    def _is_line_reset_sequence(
        self,
        swdio: NDArray[np.bool_],
        start_idx: int,
        edge_idx: int,
        rising_edges: NDArray[np.intp],
    ) -> bool:
        """Check if current start bit is part of a line reset sequence.

        Args:
            swdio: SWDIO signal data.
            start_idx: Index of potential start bit.
            edge_idx: Current edge index in rising_edges array.
            rising_edges: Array of rising edge indices.

        Returns:
            True if this appears to be a line reset sequence.
        """
        if start_idx == 0 or edge_idx == 0:
            return False

        prev_edge_idx = rising_edges[edge_idx - 1]
        swdio_between = swdio[prev_edge_idx:start_idx]

        # If SWDIO stayed high between edges, likely line reset
        return bool(len(swdio_between) > 0 and np.all(swdio_between))

    def _parse_swd_request(
        self, swdio: NDArray[np.bool_], edge_idx: int, rising_edges: NDArray[np.intp]
    ) -> tuple[dict[str, int], list[str], int] | None:
        """Parse SWD request packet (8 bits).

        Args:
            swdio: SWDIO signal data.
            edge_idx: Starting edge index.
            rising_edges: Array of rising edge indices.

        Returns:
            Tuple of (fields dict, errors list, register_addr) or None if insufficient edges.
        """
        if edge_idx + 8 > len(rising_edges):
            return None

        request_bits = []
        for i in range(8):
            bit_idx = rising_edges[edge_idx + i]
            request_bits.append(1 if swdio[bit_idx] else 0)

        # Extract fields
        fields = {
            "start_bit": request_bits[0],
            "apndp": request_bits[1],
            "rnw": request_bits[2],
            "addr_2": request_bits[3],
            "addr_3": request_bits[4],
            "parity": request_bits[5],
            "stop_bit": request_bits[6],
            "park_bit": request_bits[7],
        }

        # Validate request format
        errors = []
        if fields["start_bit"] != 1:
            errors.append("Invalid start bit")
        if fields["stop_bit"] != 0:
            errors.append("Invalid stop bit")
        if fields["park_bit"] != 1:
            errors.append("Invalid park bit")

        # Check parity (odd parity of APnDP, RnW, A[2:3])
        expected_parity = (
            fields["apndp"] + fields["rnw"] + fields["addr_2"] + fields["addr_3"]
        ) % 2
        if fields["parity"] != expected_parity:
            errors.append("Request parity error")

        # Construct register address
        register_addr = (fields["addr_3"] << 3) | (fields["addr_2"] << 2)

        return (fields, errors, register_addr)

    def _parse_swd_ack(
        self, swdio: NDArray[np.bool_], edge_idx: int, rising_edges: NDArray[np.intp]
    ) -> tuple[int, str, list[str]] | None:
        """Parse SWD ACK response (3 bits).

        Args:
            swdio: SWDIO signal data.
            edge_idx: Starting edge index.
            rising_edges: Array of rising edge indices.

        Returns:
            Tuple of (ack_value, ack_str, errors) or None if insufficient edges.
        """
        if edge_idx + 3 > len(rising_edges):
            return None

        ack_bits = []
        for i in range(3):
            bit_idx = rising_edges[edge_idx + i]
            ack_bits.append(1 if swdio[bit_idx] else 0)

        ack_value = (ack_bits[2] << 2) | (ack_bits[1] << 1) | ack_bits[0]

        # Decode ACK
        errors = []
        if ack_value == SWDResponse.OK.value:
            ack_str = "OK"
        elif ack_value == SWDResponse.WAIT.value:
            ack_str = "WAIT"
            errors.append("Target responded with WAIT")
        elif ack_value == SWDResponse.FAULT.value:
            ack_str = "FAULT"
            errors.append("Target responded with FAULT")
        else:
            ack_str = "INVALID"
            errors.append(f"Invalid ACK: 0b{ack_value:03b}")

        return (ack_value, ack_str, errors)

    def _parse_swd_data(
        self, swdio: NDArray[np.bool_], edge_idx: int, rising_edges: NDArray[np.intp]
    ) -> tuple[int, list[str]] | None:
        """Parse SWD data phase (32 bits + parity).

        Args:
            swdio: SWDIO signal data.
            edge_idx: Starting edge index.
            rising_edges: Array of rising edge indices.

        Returns:
            Tuple of (data_value, errors) or None if insufficient edges.
        """
        if edge_idx + 33 > len(rising_edges):
            return None

        data_bits = []
        for i in range(32):
            bit_idx = rising_edges[edge_idx + i]
            data_bits.append(1 if swdio[bit_idx] else 0)

        # Convert to value (LSB first)
        data_value = 0
        for i, bit in enumerate(data_bits):
            data_value |= bit << i

        # Parity bit
        parity_idx = rising_edges[edge_idx + 32]
        data_parity = 1 if swdio[parity_idx] else 0

        # Check data parity (odd parity)
        errors = []
        expected_data_parity = sum(data_bits) % 2
        if data_parity != expected_data_parity:
            errors.append("Data parity error")

        return (data_value, errors)

    def decode(  # type: ignore[override]
        self,
        trace: DigitalTrace | None = None,
        *,
        swclk: NDArray[np.bool_] | None = None,
        swdio: NDArray[np.bool_] | None = None,
        sample_rate: float = 1.0,
    ) -> Iterator[ProtocolPacket]:
        """Decode SWD transactions.

        Args:
            trace: Optional primary trace.
            swclk: Serial Wire Clock signal.
            swdio: Serial Wire Data I/O signal.
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded SWD transactions as ProtocolPacket objects.

        Example:
            >>> decoder = SWDDecoder()
            >>> for pkt in decoder.decode(swclk=swclk, swdio=swdio, sample_rate=1e6):
            ...     print(f"R/W: {'Read' if pkt.annotations['read'] else 'Write'}")
        """
        if swclk is None or swdio is None:
            return

        # Prepare signals
        swclk_trimmed, swdio_trimmed = self._prepare_signals(swclk, swdio)
        rising_edges = self._find_rising_edges(swclk_trimmed)

        if len(rising_edges) == 0:
            return

        # Decode transactions
        yield from self._decode_transactions(swdio_trimmed, rising_edges, sample_rate)

    def _prepare_signals(
        self,
        swclk: NDArray[np.bool_],
        swdio: NDArray[np.bool_],
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        """Trim signals to same length."""
        n_samples = min(len(swclk), len(swdio))
        return swclk[:n_samples], swdio[:n_samples]

    def _find_rising_edges(self, swclk: NDArray[np.bool_]) -> NDArray[np.intp]:
        """Find rising edges of SWCLK."""
        return np.where(~swclk[:-1] & swclk[1:])[0] + 1

    def _decode_transactions(
        self,
        swdio: NDArray[np.bool_],
        rising_edges: NDArray[np.intp],
        sample_rate: float,
    ) -> Iterator[ProtocolPacket]:
        """Decode all SWD transactions from edge stream."""
        trans_num = 0
        edge_idx = 0

        while edge_idx < len(rising_edges):
            # Check for valid start bit
            start_idx = rising_edges[edge_idx]
            if not swdio[start_idx]:
                edge_idx += 1
                continue

            # Skip line reset sequences
            if self._is_line_reset_sequence(swdio, start_idx, edge_idx, rising_edges):
                edge_idx += 1
                continue

            # Parse complete transaction
            packet_result = self._parse_transaction(
                swdio, rising_edges, edge_idx, trans_num, sample_rate
            )
            if packet_result is None:
                break

            packet, edge_advance = packet_result
            yield packet

            trans_num += 1
            edge_idx += edge_advance

    def _parse_transaction(
        self,
        swdio: NDArray[np.bool_],
        rising_edges: NDArray[np.intp],
        edge_idx: int,
        trans_num: int,
        sample_rate: float,
    ) -> tuple[ProtocolPacket, int] | None:
        """Parse complete SWD transaction."""
        # Parse request
        request_result = self._parse_swd_request(swdio, edge_idx, rising_edges)
        if request_result is None:
            return None
        fields, errors, register_addr = request_result

        current_idx = edge_idx + 8 + 1  # Request + turnaround

        # Parse ACK
        ack_result = self._parse_swd_ack(swdio, current_idx, rising_edges)
        if ack_result is None:
            return None
        ack_value, ack_str, ack_errors = ack_result
        errors.extend(ack_errors)

        current_idx += 3  # ACK bits

        # Parse data if ACK is OK
        data_value = self._parse_data_phase(swdio, rising_edges, current_idx, ack_value, errors)

        if ack_value == SWDResponse.OK.value:
            current_idx += 1 + 33  # Turnaround + data + parity

        # Build packet
        start_idx = rising_edges[edge_idx]
        packet = self._build_packet(
            fields,
            register_addr,
            ack_value,
            ack_str,
            data_value,
            errors,
            trans_num,
            start_idx,
            current_idx,
            rising_edges,
            sample_rate,
        )

        edge_advance = current_idx - edge_idx + 1
        return packet, edge_advance

    def _parse_data_phase(
        self,
        swdio: NDArray[np.bool_],
        rising_edges: NDArray[np.intp],
        current_idx: int,
        ack_value: int,
        errors: list[str],
    ) -> int:
        """Parse data phase if ACK is OK."""
        if ack_value != SWDResponse.OK.value:
            return 0

        data_result = self._parse_swd_data(swdio, current_idx + 1, rising_edges)
        if data_result is None:
            return 0

        data_value, data_errors = data_result
        errors.extend(data_errors)
        return data_value

    def _build_packet(
        self,
        fields: dict[str, int],
        register_addr: int,
        ack_value: int,
        ack_str: str,
        data_value: int,
        errors: list[str],
        trans_num: int,
        start_idx: int,
        end_idx: int,
        rising_edges: NDArray[np.intp],
        sample_rate: float,
    ) -> ProtocolPacket:
        """Build protocol packet from parsed data."""
        start_time = start_idx / sample_rate
        end_time = rising_edges[min(end_idx - 1, len(rising_edges) - 1)] / sample_rate

        # Add annotation
        port_type = "AP" if fields["apndp"] else "DP"
        access_type = "Read" if fields["rnw"] else "Write"
        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.PACKETS,
            f"{port_type} {access_type} @ 0x{register_addr:02X}: {ack_str}",
        )

        # Create annotations dict
        annotations = {
            "transaction_num": trans_num,
            "apndp": "AP" if fields["apndp"] else "DP",
            "read": bool(fields["rnw"]),
            "register_addr": register_addr,
            "ack": ack_str,
            "ack_value": ack_value,
        }

        if ack_value == SWDResponse.OK.value:
            annotations["data"] = data_value

        # Encode data bytes
        data_bytes = data_value.to_bytes(4, "little") if ack_value == SWDResponse.OK.value else b""

        return ProtocolPacket(
            timestamp=start_time,
            protocol="swd",
            data=data_bytes,
            annotations=annotations,
            errors=errors,
        )


def decode_swd(
    swclk: NDArray[np.bool_],
    swdio: NDArray[np.bool_],
    sample_rate: float = 1.0,
) -> list[ProtocolPacket]:
    """Convenience function to decode SWD transactions.

    Args:
        swclk: Serial Wire Clock signal.
        swdio: Serial Wire Data I/O signal.
        sample_rate: Sample rate in Hz.

    Returns:
        List of decoded SWD transactions.

    Example:
        >>> packets = decode_swd(swclk, swdio, sample_rate=10e6)
        >>> for pkt in packets:
        ...     print(f"ACK: {pkt.annotations['ack']}")
    """
    decoder = SWDDecoder()
    return list(decoder.decode(swclk=swclk, swdio=swdio, sample_rate=sample_rate))


__all__ = ["SWDDecoder", "SWDResponse", "decode_swd"]
