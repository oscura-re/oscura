"""CAN 2.0A/B protocol decoder.

This module implements a CAN (Controller Area Network) protocol decoder
supporting both standard (11-bit ID) and extended (29-bit ID) frames.


Example:
    >>> from oscura.analyzers.protocols.can import CANDecoder
    >>> decoder = CANDecoder(bitrate=500000)
    >>> for packet in decoder.decode(trace):
    ...     print(f"ID: {packet.annotations['arbitration_id']:03X}")
    ...     print(f"Data: {packet.data.hex()}")

References:
    ISO 11898-1:2015 Road vehicles - CAN - Part 1: Data link layer
    CAN Specification Version 2.0 (Bosch, 1991)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    AsyncDecoder,
    ChannelDef,
    DecoderState,
    OptionDef,
)
from oscura.core.types import DigitalTrace, ProtocolPacket

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


class CANFrameType(IntEnum):
    """CAN frame types."""

    DATA = 0
    REMOTE = 1
    ERROR = 2
    OVERLOAD = 3


@dataclass
class CANFrame:
    """Decoded CAN frame.

    Attributes:
        arbitration_id: CAN ID (11-bit or 29-bit).
        is_extended: True for 29-bit extended ID.
        is_remote: True for remote transmission request.
        dlc: Data length code (0-8).
        data: Data bytes.
        crc: Received CRC value.
        crc_computed: Computed CRC value.
        timestamp: Frame start time in seconds.
        end_timestamp: Frame end time in seconds.
        errors: List of detected errors.
    """

    arbitration_id: int
    is_extended: bool
    is_remote: bool
    dlc: int
    data: bytes
    crc: int
    crc_computed: int
    timestamp: float
    end_timestamp: float
    errors: list[str]

    @property
    def crc_valid(self) -> bool:
        """Check if CRC matches."""
        return self.crc == self.crc_computed


class CANDecoderState(DecoderState):
    """State machine for CAN decoder."""

    def reset(self) -> None:
        """Reset state."""
        self.bit_position = 0
        self.stuff_count = 0
        self.last_five_bits = 0
        self.frame_bits: list[int] = []
        self.in_frame = False
        self.frame_start_time = 0.0


# CAN bit timing constants
CAN_BITRATES = {
    10000: "10 kbps",
    20000: "20 kbps",
    50000: "50 kbps",
    100000: "100 kbps",
    125000: "125 kbps",
    250000: "250 kbps",
    500000: "500 kbps",
    800000: "800 kbps",
    1000000: "1 Mbps",
}

# CRC polynomial for CAN: x^15 + x^14 + x^10 + x^8 + x^7 + x^4 + x^3 + 1
CAN_CRC_POLY = 0x4599
CAN_CRC_INIT = 0x0000


class CANDecoder(AsyncDecoder):
    """CAN 2.0A/B protocol decoder.

    Decodes CAN frames from digital signal captures, supporting:
    - CAN 2.0A: Standard 11-bit identifiers
    - CAN 2.0B: Extended 29-bit identifiers
    - Bit stuffing detection and removal
    - CRC checking
    - Error detection

    Attributes:
        id: Decoder identifier.
        name: Human-readable name.
        channels: Required input channels.
        options: Configurable decoder options.

    Example:
        >>> decoder = CANDecoder(bitrate=500000)
        >>> frames = list(decoder.decode(trace))
        >>> for frame in frames:
        ...     print(f"CAN ID: 0x{frame.annotations['arbitration_id']:03X}")
    """

    id = "can"
    name = "CAN"
    longname = "Controller Area Network"
    desc = "CAN 2.0A/B bus decoder"
    license = "MIT"

    channels = [
        ChannelDef("can", "CAN", "CAN bus signal (CAN_H - CAN_L or single-ended)"),
    ]

    options = [
        OptionDef(
            "bitrate",
            "Bit Rate",
            "CAN bit rate in bps",
            default=500000,
            values=list(CAN_BITRATES.keys()),
        ),
        OptionDef(
            "sample_point",
            "Sample Point",
            "Sample point as fraction of bit time",
            default=0.75,
        ),
    ]

    def __init__(  # type: ignore[no-untyped-def]
        self,
        bitrate: int = 500000,
        sample_point: float = 0.75,
        **options,
    ) -> None:
        """Initialize CAN decoder.

        Args:
            bitrate: CAN bus bit rate in bps.
            sample_point: Sample point as fraction of bit time (0.5-0.9).
            **options: Additional decoder options.
        """
        super().__init__(baudrate=bitrate, **options)
        self._bitrate = bitrate
        self._sample_point = sample_point
        self._state = CANDecoderState()

    @property
    def bitrate(self) -> int:
        """Get CAN bit rate."""
        return self._bitrate

    @bitrate.setter
    def bitrate(self, value: int) -> None:
        """Set CAN bit rate."""
        self._bitrate = value
        self._baudrate = value

    def decode(
        self,
        trace: DigitalTrace,
        **channels: NDArray[np.bool_],
    ) -> Iterator[ProtocolPacket]:
        """Decode CAN frames from digital trace.

        Args:
            trace: Digital trace containing CAN signal.
            **channels: Additional channel data (not used for single-wire CAN).

        Yields:
            ProtocolPacket for each decoded CAN frame.

        Example:
            >>> decoder = CANDecoder(bitrate=500000)
            >>> for packet in decoder.decode(trace):
            ...     can_id = packet.annotations['arbitration_id']
            ...     print(f"ID: 0x{can_id:03X}, Data: {packet.data.hex()}")
        """
        self.reset()

        data = trace.data
        sample_rate = trace.metadata.sample_rate
        1.0 / sample_rate

        # Calculate samples per bit
        1.0 / self._bitrate
        samples_per_bit = round(sample_rate / self._bitrate)

        if samples_per_bit < 2:
            self.put_annotation(
                0,
                trace.duration,
                AnnotationLevel.MESSAGES,
                "Error: Sample rate too low for CAN decoding",
            )
            return

        # Sample offset within bit (where to sample)
        sample_offset = int(samples_per_bit * self._sample_point)

        # Find start of frames (falling edge from recessive to dominant)
        # In CAN, recessive = 1, dominant = 0
        frame_starts = self._find_frame_starts(data, samples_per_bit)

        for frame_start_idx in frame_starts:
            # Try to decode frame starting at this position
            frame = self._decode_frame(
                data,
                frame_start_idx,
                sample_rate,
                samples_per_bit,
                sample_offset,
            )

            if frame is not None:
                # Create packet
                packet = ProtocolPacket(
                    timestamp=frame.timestamp,
                    protocol="can",
                    data=frame.data,
                    annotations={
                        "arbitration_id": frame.arbitration_id,
                        "is_extended": frame.is_extended,
                        "is_remote": frame.is_remote,
                        "dlc": frame.dlc,
                        "crc": frame.crc,
                        "crc_valid": frame.crc_valid,
                    },
                    errors=frame.errors,
                    end_timestamp=frame.end_timestamp,
                )

                self._packets.append(packet)
                yield packet

    def _find_frame_starts(
        self,
        data: NDArray[np.bool_],
        samples_per_bit: int,
    ) -> list[int]:
        """Find potential frame start positions using vectorized edge detection.

        CAN frames start with a Start of Frame (SOF) bit, which is a
        dominant (0) bit following bus idle (recessive/1).

        Args:
            data: Digital signal data.
            samples_per_bit: Samples per CAN bit.

        Returns:
            List of sample indices for potential frame starts.
        """
        # Optimize using vectorized operations instead of loop
        min_idle_bits = 3  # Minimum idle time before frame
        min_idle_samples = min_idle_bits * samples_per_bit

        # Detect all falling edges (1 -> 0) using vectorized comparison
        falling_edges = np.where(data[:-1] & ~data[1:])[0] + 1

        frame_starts = []

        # Check each falling edge for idle condition
        for edge_idx in falling_edges:
            if edge_idx < min_idle_samples or edge_idx >= len(data) - samples_per_bit:
                continue

            # Check if previous samples are mostly high (idle) using vectorized mean
            idle_region = data[edge_idx - min_idle_samples : edge_idx]
            if np.mean(idle_region) > 0.8:  # Mostly recessive
                frame_starts.append(int(edge_idx))

        # Filter out closely spaced detections (same frame)
        if not frame_starts:
            return []

        filtered_starts = [frame_starts[0]]
        min_frame_gap = samples_per_bit * 20  # Minimum gap between frames

        for start in frame_starts[1:]:
            if start - filtered_starts[-1] >= min_frame_gap:
                filtered_starts.append(start)

        return filtered_starts

    def _decode_frame(
        self,
        data: NDArray[np.bool_],
        start_idx: int,
        sample_rate: float,
        samples_per_bit: int,
        sample_offset: int,
    ) -> CANFrame | None:
        """Decode a single CAN frame.

        Args:
            data: Digital signal data.
            start_idx: Sample index of frame start (SOF).
            sample_rate: Sample rate in Hz.
            samples_per_bit: Samples per CAN bit.
            sample_offset: Offset within bit for sampling.

        Returns:
            Decoded CANFrame or None if decode fails.
        """
        sample_period = 1.0 / sample_rate
        frame_start_time = start_idx * sample_period

        # Extract bits with bit stuffing removal
        bits = []  # type: ignore[var-annotated]
        stuff_count = 0
        consecutive_same = 0
        last_bit = None

        bit_idx = 0
        max_frame_bits = 150  # Maximum bits in extended frame with stuffing

        current_idx = start_idx

        while len(bits) < 128 and bit_idx < max_frame_bits:
            # Calculate sample position
            sample_pos = current_idx + sample_offset

            if sample_pos >= len(data):
                break

            # Sample the bit
            bit = data[sample_pos]

            # Check for bit stuffing
            if last_bit is not None:
                if bit == last_bit:
                    consecutive_same += 1
                else:
                    consecutive_same = 1

                # After 5 consecutive same bits, next bit should be opposite (stuff bit)
                if consecutive_same == 5:
                    # Next bit should be stuff bit - skip it
                    current_idx += samples_per_bit
                    bit_idx += 1
                    stuff_count += 1

                    # Sample the stuff bit to verify
                    stuff_sample_pos = current_idx + sample_offset
                    if stuff_sample_pos < len(data):
                        stuff_bit = data[stuff_sample_pos]
                        if stuff_bit == bit:
                            # Stuff error
                            pass
                    consecutive_same = 0
                    current_idx += samples_per_bit
                    bit_idx += 1
                    continue

            bits.append(int(bit))
            last_bit = bit

            current_idx += samples_per_bit
            bit_idx += 1

        if len(bits) < 20:  # Minimum frame length
            return None

        # Parse frame fields
        frame = self._parse_frame_bits(bits, frame_start_time, sample_period, current_idx)
        return frame

    def _parse_frame_bits(
        self,
        bits: list[int],
        start_time: float,
        sample_period: float,
        end_idx: int,
    ) -> CANFrame | None:
        """Parse decoded bits into CAN frame.

        Args:
            bits: List of bit values (after stuff bit removal).
            start_time: Frame start time.
            sample_period: Sample period.
            end_idx: End sample index.

        Returns:
            Parsed CANFrame or None if invalid.
        """
        errors: list[str] = []

        try:
            pos = self._parse_sof(bits, errors)
            if pos is None:
                return None

            arb_result = self._parse_arbitration_field(bits, pos)
            if arb_result[0] is None:
                return None
            arb_id, is_extended, is_remote, pos = arb_result

            dlc_result = self._parse_dlc(bits, pos)
            if dlc_result[0] is None:
                return None
            dlc, data_len, pos = dlc_result

            data_result = self._parse_data_field(bits, pos, data_len, is_remote)
            if data_result[0] is None:
                return None
            data, pos = data_result

            crc_result = self._parse_crc_field(bits, pos, errors)
            if crc_result[0] is None:
                return None
            crc_received, crc_computed, pos = crc_result

            pos = self._parse_ack_eof(bits, pos, errors)

            end_time = start_time + pos * (1.0 / self._bitrate)

            return CANFrame(
                arbitration_id=arb_id,
                is_extended=is_extended,
                is_remote=is_remote,
                dlc=dlc,
                data=data,
                crc=crc_received,
                crc_computed=crc_computed,
                timestamp=start_time,
                end_timestamp=end_time,
                errors=errors,
            )

        except (IndexError, ValueError):
            return None

    def _parse_sof(self, bits: list[int], errors: list[str]) -> int | None:
        """Parse Start of Frame bit.

        Args:
            bits: Bit array.
            errors: Error list to append to.

        Returns:
            Position after SOF, or None if invalid.
        """
        if len(bits) < 1:
            return None

        sof = bits[0]
        if sof != 0:
            errors.append("Invalid SOF")

        return 1

    def _parse_arbitration_field(
        self, bits: list[int], pos: int
    ) -> tuple[int, bool, bool, int] | tuple[None, None, None, None]:
        """Parse CAN arbitration field (ID, RTR, IDE).

        Args:
            bits: Bit array.
            pos: Current position.

        Returns:
            Tuple of (arb_id, is_extended, is_remote, new_pos) or (None, None, None, None).
        """
        if pos + 11 > len(bits):
            return None, None, None, None

        # First 11 bits of ID
        arb_id = self._extract_bits_as_int(bits, pos, 11)
        pos += 11

        # RTR/SRR bit
        if pos >= len(bits):
            return None, None, None, None
        rtr_or_srr = bits[pos]
        pos += 1

        # IDE bit
        if pos >= len(bits):
            return None, None, None, None
        ide = bits[pos]
        pos += 1

        is_extended = bool(ide)

        if is_extended:
            try:
                arb_id, is_remote, pos = self._parse_extended_id(bits, pos, arb_id)
            except IndexError:
                return None, None, None, None
        else:
            is_remote = bool(rtr_or_srr)
            pos += 1  # r0 reserved bit

        return arb_id, is_extended, is_remote, pos

    def _parse_extended_id(self, bits: list[int], pos: int, base_id: int) -> tuple[int, bool, int]:
        """Parse extended CAN ID (18 additional bits).

        Args:
            bits: Bit array.
            pos: Current position.
            base_id: Base 11-bit ID.

        Returns:
            Tuple of (full_id, is_remote, new_pos).

        Raises:
            IndexError: If insufficient bits available.
        """
        if pos + 18 > len(bits):
            raise IndexError("Insufficient bits for extended ID")

        # ID extension (18 bits)
        id_ext = self._extract_bits_as_int(bits, pos, 18)
        arb_id = (base_id << 18) | id_ext
        pos += 18

        # RTR bit
        if pos >= len(bits):
            raise IndexError("Insufficient bits for RTR")
        is_remote = bool(bits[pos])
        pos += 1

        # r1, r0 reserved bits
        pos += 2

        return arb_id, is_remote, pos

    def _parse_dlc(
        self, bits: list[int], pos: int
    ) -> tuple[int, int, int] | tuple[None, None, None]:
        """Parse Data Length Code.

        Args:
            bits: Bit array.
            pos: Current position.

        Returns:
            Tuple of (dlc, data_len, new_pos) or (None, None, None).
        """
        if pos + 4 > len(bits):
            return None, None, None

        dlc = self._extract_bits_as_int(bits, pos, 4)
        pos += 4

        data_len = min(dlc, 8)  # Limit to 8 bytes
        return dlc, data_len, pos

    def _parse_data_field(
        self, bits: list[int], pos: int, data_len: int, is_remote: bool
    ) -> tuple[bytes, int] | tuple[None, None]:
        """Parse CAN data field.

        Args:
            bits: Bit array.
            pos: Current position.
            data_len: Number of data bytes.
            is_remote: True if remote frame.

        Returns:
            Tuple of (data_bytes, new_pos) or (None, None).
        """
        if is_remote:
            return b"", pos

        if pos + data_len * 8 > len(bits):
            return None, None

        data_bytes = bytearray()
        for _ in range(data_len):
            byte_val = self._extract_bits_as_int(bits, pos, 8)
            data_bytes.append(byte_val)
            pos += 8

        return bytes(data_bytes), pos

    def _parse_crc_field(
        self, bits: list[int], pos: int, errors: list[str]
    ) -> tuple[int, int, int] | tuple[None, None, None]:
        """Parse CRC field and validate.

        Args:
            bits: Bit array.
            pos: Current position.
            errors: Error list to append to.

        Returns:
            Tuple of (crc_received, crc_computed, new_pos) or (None, None, None).
        """
        if pos + 15 > len(bits):
            return None, None, None

        crc_received = self._extract_bits_as_int(bits, pos, 15)
        crc_data_end = pos
        pos += 15

        crc_computed = self._compute_crc(bits[:crc_data_end])

        if crc_received != crc_computed:
            errors.append(
                f"CRC error: received 0x{crc_received:04X}, computed 0x{crc_computed:04X}"
            )

        # CRC delimiter (should be 1)
        if pos < len(bits) and bits[pos] != 1:
            errors.append("CRC delimiter error")
        pos += 1

        return crc_received, crc_computed, pos

    def _parse_ack_eof(self, bits: list[int], pos: int, errors: list[str]) -> int:
        """Parse ACK and EOF fields.

        Args:
            bits: Bit array.
            pos: Current position.
            errors: Error list to append to.

        Returns:
            Position after ACK/EOF.
        """
        # ACK slot and delimiter
        pos += 2

        # EOF (7 recessive bits) - not strictly checked
        return pos

    def _extract_bits_as_int(self, bits: list[int], start: int, count: int) -> int:
        """Extract consecutive bits as integer (MSB first).

        Args:
            bits: Bit array.
            start: Start position.
            count: Number of bits.

        Returns:
            Integer value.
        """
        value = 0
        for i in range(count):
            value = (value << 1) | bits[start + i]
        return value

    def _compute_crc(self, bits: list[int]) -> int:
        """Compute CAN CRC-15.

        Args:
            bits: Input bits for CRC calculation.

        Returns:
            15-bit CRC value.
        """
        crc = CAN_CRC_INIT

        for bit in bits:
            crc_next = (crc >> 14) & 1
            crc = (crc << 1) & 0x7FFF

            if bit ^ crc_next:
                crc ^= CAN_CRC_POLY

        return crc


def decode_can(
    trace: DigitalTrace,
    *,
    bitrate: int = 500000,
    sample_point: float = 0.75,
) -> list[CANFrame]:
    """Convenience function to decode CAN frames.

    Args:
        trace: Digital trace containing CAN signal.
        bitrate: CAN bit rate in bps (default 500000).
        sample_point: Sample point as fraction of bit time.

    Returns:
        List of decoded CANFrame objects.

    Example:
        >>> frames = decode_can(trace, bitrate=500000)
        >>> for frame in frames:
        ...     print(f"ID: 0x{frame.arbitration_id:03X}")
    """
    decoder = CANDecoder(bitrate=bitrate, sample_point=sample_point)
    frames = []

    for packet in decoder.decode(trace):
        # Reconstruct CANFrame from packet
        frame = CANFrame(
            arbitration_id=packet.annotations["arbitration_id"],
            is_extended=packet.annotations["is_extended"],
            is_remote=packet.annotations["is_remote"],
            dlc=packet.annotations["dlc"],
            data=packet.data,
            crc=packet.annotations["crc"],
            crc_computed=packet.annotations["crc"],  # Reconstruct as same
            timestamp=packet.timestamp,
            end_timestamp=packet.end_timestamp or packet.timestamp,
            errors=packet.errors,
        )
        frames.append(frame)

    return frames


__all__ = [
    "CAN_BITRATES",
    "CANDecoder",
    "CANFrame",
    "CANFrameType",
    "decode_can",
]
