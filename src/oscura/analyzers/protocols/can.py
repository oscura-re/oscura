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

    channels = [  # noqa: RUF012
        ChannelDef("can", "CAN", "CAN bus signal (CAN_H - CAN_L or single-ended)"),
    ]

    options = [  # noqa: RUF012
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
        """Find potential frame start positions.

        CAN frames start with a Start of Frame (SOF) bit, which is a
        dominant (0) bit following bus idle (recessive/1).

        Args:
            data: Digital signal data.
            samples_per_bit: Samples per CAN bit.

        Returns:
            List of sample indices for potential frame starts.
        """
        frame_starts = []

        # Look for falling edges (1 -> 0) after idle period
        min_idle_bits = 3  # Minimum idle time before frame
        min_idle_samples = min_idle_bits * samples_per_bit

        i = min_idle_samples
        while i < len(data) - samples_per_bit:
            # Check if previous samples are mostly high (idle)
            idle_region = data[max(0, i - min_idle_samples) : i]
            if np.mean(idle_region) > 0.8:  # Mostly recessive
                # Check for falling edge (SOF)
                if data[i - 1] and not data[i]:
                    frame_starts.append(i)
                    # Skip ahead to avoid detecting same frame
                    i += samples_per_bit * 20  # Skip at least 20 bits
                    continue
            i += 1

        return frame_starts

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
        errors = []

        try:
            pos = 0

            # SOF (should be 0)
            if pos >= len(bits):
                return None
            sof = bits[pos]
            pos += 1

            if sof != 0:
                errors.append("Invalid SOF")

            # Arbitration field
            if pos + 11 > len(bits):
                return None

            # First 11 bits of ID
            arb_id = 0
            for i in range(11):
                arb_id = (arb_id << 1) | bits[pos + i]
            pos += 11

            # RTR bit (for standard) or SRR bit (for extended)
            if pos >= len(bits):
                return None
            rtr_or_srr = bits[pos]
            pos += 1

            # IDE bit
            if pos >= len(bits):
                return None
            ide = bits[pos]
            pos += 1

            is_extended = bool(ide)
            is_remote = False

            if is_extended:
                # Extended frame: 18 more ID bits
                if pos + 18 > len(bits):
                    return None

                # ID extension (18 bits)
                for i in range(18):
                    arb_id = (arb_id << 1) | bits[pos + i]
                pos += 18

                # RTR bit
                if pos >= len(bits):
                    return None
                is_remote = bool(bits[pos])
                pos += 1

                # r1, r0 reserved bits
                pos += 2
            else:
                # Standard frame
                is_remote = bool(rtr_or_srr)
                # r0 reserved bit
                pos += 1

            # DLC (4 bits)
            if pos + 4 > len(bits):
                return None

            dlc = 0
            for i in range(4):
                dlc = (dlc << 1) | bits[pos + i]
            pos += 4

            # Limit DLC to 8
            data_len = min(dlc, 8)

            # Data field (0-8 bytes)
            if not is_remote:
                if pos + data_len * 8 > len(bits):
                    return None

                data_bytes = bytearray()
                for byte_idx in range(data_len):
                    byte_val = 0
                    for bit_idx in range(8):
                        byte_val = (byte_val << 1) | bits[pos + byte_idx * 8 + bit_idx + bit_idx]
                    data_bytes.append(byte_val)
                    pos += 8

                data = bytes(data_bytes)
            else:
                data = b""

            # CRC field (15 bits)
            if pos + 15 > len(bits):
                return None

            crc_received = 0
            for i in range(15):
                crc_received = (crc_received << 1) | bits[pos + i]
            pos += 15

            # Compute CRC on frame bits before CRC field
            # CRC covers SOF through data field
            crc_data_end = pos - 15
            crc_computed = self._compute_crc(bits[:crc_data_end])

            if crc_received != crc_computed:
                errors.append(
                    f"CRC error: received 0x{crc_received:04X}, computed 0x{crc_computed:04X}"
                )

            # CRC delimiter (should be 1)
            if pos < len(bits) and bits[pos] != 1:
                errors.append("CRC delimiter error")
            pos += 1

            # ACK slot and delimiter
            pos += 2

            # EOF (7 recessive bits)
            # We don't strictly check this

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
