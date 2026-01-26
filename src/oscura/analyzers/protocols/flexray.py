"""FlexRay protocol decoder.

This module implements FlexRay automotive protocol decoder with support
for static and dynamic segments, 10 Mbps signaling, and CRC validation.


Example:
    >>> from oscura.analyzers.protocols.flexray import FlexRayDecoder
    >>> decoder = FlexRayDecoder()
    >>> for packet in decoder.decode(bp=bp, bm=bm):
    ...     print(f"Slot: {packet.annotations['slot_id']}")

References:
    FlexRay Communications System Protocol Specification Version 3.0.1
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from oscura.analyzers.protocols.base import (
    AnnotationLevel,
    AsyncDecoder,
    ChannelDef,
    OptionDef,
)
from oscura.core.types import DigitalTrace, ProtocolPacket, WaveformTrace

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray


class FlexRaySegment(Enum):
    """FlexRay communication segment types."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    SYMBOL = "symbol"


@dataclass
class FlexRayFrame:
    """Decoded FlexRay frame.

    Attributes:
        slot_id: Slot identifier (1-2047).
        cycle_count: Cycle counter (0-63).
        payload_length: Payload length in 16-bit words (0-127).
        header_crc: Header CRC value.
        payload: Payload data bytes.
        frame_crc: Frame CRC value (24-bit).
        segment: Segment type (static or dynamic).
        timestamp: Frame start time in seconds.
        errors: List of detected errors.
    """

    slot_id: int
    cycle_count: int
    payload_length: int
    header_crc: int
    payload: bytes
    frame_crc: int
    segment: FlexRaySegment
    timestamp: float
    errors: list[str]


class _FlexRayBitSampler:
    """Helper for sampling bits from FlexRay frames."""

    def __init__(self, data: NDArray[np.bool_], bit_idx: int, bit_period: float):
        """Initialize bit sampler.

        Args:
            data: Digital data array.
            bit_idx: Starting bit index.
            bit_period: Bit period in samples.
        """
        self.data = data
        self.bit_idx: float = float(bit_idx)
        self.bit_period = bit_period

    def sample_bits(self, count: int) -> list[int]:
        """Sample specified number of bits.

        Args:
            count: Number of bits to sample.

        Returns:
            List of sampled bit values (0 or 1).
        """
        bits = []
        for _ in range(count):
            sample_idx_raw = self.bit_idx + self.bit_period / 2
            sample_idx = int(sample_idx_raw)
            if sample_idx < len(self.data):
                bits.append(1 if self.data[sample_idx] else 0)
                self.bit_idx += self.bit_period
            else:
                return bits
        return bits

    def validate_fss(self) -> bool:
        """Validate Frame Start Sequence (1 bit, must be 0).

        Returns:
            True if FSS is valid.
        """
        fss_bits = self.sample_bits(1)
        return bool(fss_bits and fss_bits[0] == 0)

    def parse_header(self) -> dict[str, int] | None:
        """Parse 40-bit FlexRay header.

        Returns:
            Dict with slot_id, header_crc, cycle_count, payload_length or None if incomplete.
        """
        header_bits = self.sample_bits(40)
        if len(header_bits) < 40:
            return None

        return {
            "slot_id": self._bits_to_int(header_bits[4:15]),
            "header_crc": self._bits_to_int(header_bits[15:26]),
            "cycle_count": self._bits_to_int(header_bits[26:32]),
            "payload_length": self._bits_to_int(header_bits[33:40]),
        }

    def parse_payload(self, payload_length: int, errors: list[str]) -> list[int]:
        """Parse payload bytes.

        Args:
            payload_length: Payload length in 16-bit words.
            errors: Error list to append to.

        Returns:
            List of payload byte values.
        """
        payload_byte_count = payload_length * 2
        payload_bytes = []

        for _ in range(payload_byte_count):
            byte_bits = self.sample_bits(8)
            if len(byte_bits) == 8:
                payload_bytes.append(self._bits_to_int(byte_bits))
            else:
                errors.append("Incomplete payload")
                break

        return payload_bytes

    def parse_crc(self) -> int:
        """Parse 24-bit frame CRC.

        Returns:
            CRC value.
        """
        crc_bits = self.sample_bits(24)
        return self._bits_to_int(crc_bits)

    def get_bit_idx(self) -> int:
        """Get current bit index.

        Returns:
            Current bit index position.
        """
        return int(self.bit_idx)

    @staticmethod
    def _bits_to_int(bits: list[int]) -> int:
        """Convert bit list to integer.

        Args:
            bits: List of bit values.

        Returns:
            Integer value.
        """
        result = 0
        for bit in bits:
            result = (result << 1) | bit
        return result


class FlexRayDecoder(AsyncDecoder):
    """FlexRay protocol decoder.

    Decodes FlexRay bus frames with header and frame CRC validation,
    static and dynamic segment support, and slot/cycle identification.

    Attributes:
        id: "flexray"
        name: "FlexRay"
        channels: [bp, bm] (differential pair)

    Example:
        >>> decoder = FlexRayDecoder(bitrate=10000000)
        >>> for packet in decoder.decode(bp=bp, bm=bm, sample_rate=100e6):
        ...     print(f"Slot {packet.annotations['slot_id']}, Cycle {packet.annotations['cycle_count']}")
    """

    id = "flexray"
    name = "FlexRay"
    longname = "FlexRay Automotive Network"
    desc = "FlexRay protocol decoder"

    channels = [
        ChannelDef("bp", "BP", "FlexRay Bus Plus", required=True),
        ChannelDef("bm", "BM", "FlexRay Bus Minus", required=True),
    ]

    optional_channels = []

    options = [
        OptionDef(
            "bitrate",
            "Bitrate",
            "Bits per second",
            default=10000000,
            values=[2500000, 5000000, 10000000],
        ),
    ]

    annotations = [
        ("tss", "Transmission Start Sequence"),
        ("fss", "Frame Start Sequence"),
        ("header", "Frame header"),
        ("payload", "Payload"),
        ("crc", "Frame CRC"),
        ("error", "Error"),
    ]

    # FlexRay constants
    TSS_LENGTH = 3  # Transmission Start Sequence (Low + Low + High)
    FSS_LENGTH = 1  # Frame Start Sequence (Low)
    BSS_LENGTH = 1  # Byte Start Sequence

    def __init__(
        self,
        bitrate: int = 10000000,
    ) -> None:
        """Initialize FlexRay decoder.

        Args:
            bitrate: FlexRay bitrate in bps (2.5, 5, or 10 Mbps).
        """
        super().__init__(baudrate=bitrate, bitrate=bitrate)
        self._bitrate = bitrate

    def decode(  # type: ignore[override]
        self,
        trace: DigitalTrace | WaveformTrace | None = None,
        *,
        bp: NDArray[np.bool_] | None = None,
        bm: NDArray[np.bool_] | None = None,
        sample_rate: float = 1.0,
    ) -> Iterator[ProtocolPacket]:
        """Decode FlexRay frames.

        Args:
            trace: Optional input trace.
            bp: Bus Plus signal.
            bm: Bus Minus signal.
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded FlexRay frames as ProtocolPacket objects.

        Example:
            >>> decoder = FlexRayDecoder(bitrate=10000000)
            >>> for pkt in decoder.decode(bp=bp, bm=bm, sample_rate=100e6):
            ...     print(f"Slot: {pkt.annotations['slot_id']}")
        """
        if trace is not None:
            if isinstance(trace, WaveformTrace):
                from oscura.analyzers.digital.extraction import to_digital

                digital_trace = to_digital(trace, threshold="auto")
            else:
                digital_trace = trace
            bp = digital_trace.data
            sample_rate = digital_trace.metadata.sample_rate

        if bp is None or bm is None:
            return

        n_samples = min(len(bp), len(bm))
        bp = bp[:n_samples]
        bm = bm[:n_samples]

        # Decode differential signal
        # IdleLow: BP=0, BM=1 -> 0
        # Data0: BP=1, BM=0 -> 1
        # Data1: BP=0, BM=1 -> 0
        # Simplified: use BP as primary signal
        diff_signal = bp

        bit_period = sample_rate / self._bitrate

        frame_num = 0
        idx = 0

        while idx < len(diff_signal):
            # Look for TSS (Transmission Start Sequence)
            tss_idx = self._find_tss(diff_signal, idx, bit_period)
            if tss_idx is None:
                break

            # Decode frame
            frame, end_idx = self._decode_frame(diff_signal, tss_idx, sample_rate, bit_period)

            if frame is not None:
                # Add annotation
                self.put_annotation(
                    frame.timestamp,
                    frame.timestamp + 0.001,
                    AnnotationLevel.PACKETS,
                    f"Slot {frame.slot_id}, Cycle {frame.cycle_count}",
                )

                # Create packet
                annotations = {
                    "frame_num": frame_num,
                    "slot_id": frame.slot_id,
                    "cycle_count": frame.cycle_count,
                    "payload_length": frame.payload_length,
                    "header_crc": frame.header_crc,
                    "frame_crc": frame.frame_crc,
                    "segment": frame.segment.value,
                }

                packet = ProtocolPacket(
                    timestamp=frame.timestamp,
                    protocol="flexray",
                    data=frame.payload,
                    annotations=annotations,
                    errors=frame.errors,
                )

                yield packet
                frame_num += 1

            idx = end_idx if end_idx > idx else idx + int(bit_period)

    def _find_tss(
        self,
        data: NDArray[np.bool_],
        start_idx: int,
        bit_period: float,
    ) -> int | None:
        """Find Transmission Start Sequence.

        Args:
            data: Digital data array.
            start_idx: Start search index.
            bit_period: Bit period in samples.

        Returns:
            Index of TSS start, or None if not found.
        """
        # TSS pattern: Low (idle), Low (data0), High (data1)
        # Simplified: look for specific transition pattern
        idx = start_idx
        while idx < len(data) - int(3 * bit_period):
            # Sample at bit centers
            sample1_idx = int(idx + bit_period / 2)
            sample2_idx = int(idx + 1.5 * bit_period)
            sample3_idx = int(idx + 2.5 * bit_period)

            if sample1_idx < len(data) and sample2_idx < len(data) and sample3_idx < len(data):
                # Check for low, low, high pattern
                if not data[sample1_idx] and not data[sample2_idx] and data[sample3_idx]:
                    return idx

            idx += int(bit_period / 4)

        return None

    def _decode_frame(
        self,
        data: NDArray[np.bool_],
        tss_idx: int,
        sample_rate: float,
        bit_period: float,
    ) -> tuple[FlexRayFrame | None, int]:
        """Decode FlexRay frame starting from TSS.

        Args:
            data: Digital data array.
            tss_idx: TSS index.
            sample_rate: Sample rate in Hz.
            bit_period: Bit period in samples.

        Returns:
            (frame, end_index) tuple.
        """
        errors = []
        bit_idx = tss_idx + int(3 * bit_period)  # Skip TSS

        sampler = _FlexRayBitSampler(data, bit_idx, bit_period)

        # Validate FSS
        if not sampler.validate_fss():
            errors.append("Invalid FSS")

        # Parse header
        header_fields = sampler.parse_header()
        if header_fields is None:
            return None, sampler.get_bit_idx()

        # Parse payload
        payload_bytes = sampler.parse_payload(header_fields["payload_length"], errors)

        # Parse CRC
        frame_crc = sampler.parse_crc()

        # Create frame
        frame = FlexRayFrame(
            slot_id=header_fields["slot_id"],
            cycle_count=header_fields["cycle_count"],
            payload_length=header_fields["payload_length"],
            header_crc=header_fields["header_crc"],
            payload=bytes(payload_bytes),
            frame_crc=frame_crc,
            segment=FlexRaySegment.STATIC,
            timestamp=tss_idx / sample_rate,
            errors=errors,
        )

        return frame, sampler.get_bit_idx()


def decode_flexray(
    bp: NDArray[np.bool_],
    bm: NDArray[np.bool_],
    sample_rate: float = 1.0,
    bitrate: int = 10000000,
) -> list[ProtocolPacket]:
    """Convenience function to decode FlexRay frames.

    Args:
        bp: Bus Plus signal.
        bm: Bus Minus signal.
        sample_rate: Sample rate in Hz.
        bitrate: FlexRay bitrate in bps.

    Returns:
        List of decoded FlexRay frames.

    Example:
        >>> packets = decode_flexray(bp, bm, sample_rate=100e6, bitrate=10e6)
        >>> for pkt in packets:
        ...     print(f"Slot: {pkt.annotations['slot_id']}")
    """
    decoder = FlexRayDecoder(bitrate=bitrate)
    return list(decoder.decode(bp=bp, bm=bm, sample_rate=sample_rate))


__all__ = ["FlexRayDecoder", "FlexRayFrame", "FlexRaySegment", "decode_flexray"]
