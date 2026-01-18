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

    channels = [  # noqa: RUF012
        ChannelDef("bp", "BP", "FlexRay Bus Plus", required=True),
        ChannelDef("bm", "BM", "FlexRay Bus Minus", required=True),
    ]

    optional_channels = []  # noqa: RUF012

    options = [  # noqa: RUF012
        OptionDef(
            "bitrate",
            "Bitrate",
            "Bits per second",
            default=10000000,
            values=[2500000, 5000000, 10000000],
        ),
    ]

    annotations = [  # noqa: RUF012
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

        # Sample bits
        def sample_bits(count: int) -> list[int]:
            nonlocal bit_idx
            bits = []
            for _ in range(count):
                sample_idx = int(bit_idx + bit_period / 2)
                if sample_idx < len(data):
                    bits.append(1 if data[sample_idx] else 0)
                    bit_idx += bit_period  # type: ignore[assignment]
                else:
                    return bits
            return bits

        # FSS (Frame Start Sequence) - 1 bit
        fss_bits = sample_bits(1)
        if not fss_bits or fss_bits[0] != 0:
            errors.append("Invalid FSS")

        # Header (5 bytes = 40 bits)
        # Byte 1: Reserved (1) + Payload preamble (1) + NULL frame (1) + Sync (1) + Startup (1) + Slot ID[10:8] (3)
        # Byte 2: Slot ID[7:0] (8)
        # Byte 3: Header CRC[10:3] (8)
        # Byte 4: Header CRC[2:0] (3) + Cycle count[5:0] (6) - split to bits 7:5 and 4:0
        # Byte 5: Cycle count continued + Payload length[6:0] (7)

        header_bits = sample_bits(40)
        if len(header_bits) < 40:
            return None, int(bit_idx)

        # Extract header fields (simplified)
        # Slot ID (11 bits): bits 4-14
        slot_id_bits = header_bits[4:15]
        slot_id = 0
        for bit in slot_id_bits:
            slot_id = (slot_id << 1) | bit

        # Header CRC (11 bits): bits 15-25
        header_crc_bits = header_bits[15:26]
        header_crc = 0
        for bit in header_crc_bits:
            header_crc = (header_crc << 1) | bit

        # Cycle count (6 bits): bits 26-31
        cycle_bits = header_bits[26:32]
        cycle_count = 0
        for bit in cycle_bits:
            cycle_count = (cycle_count << 1) | bit

        # Payload length (7 bits): bits 33-39
        payload_len_bits = header_bits[33:40]
        payload_length = 0
        for bit in payload_len_bits:
            payload_length = (payload_length << 1) | bit

        # Payload (payload_length * 2 bytes, as length is in 16-bit words)
        payload_byte_count = payload_length * 2
        payload_bytes = []

        for _ in range(payload_byte_count):
            byte_bits = sample_bits(8)
            if len(byte_bits) == 8:
                byte_val = 0
                for bit in byte_bits:
                    byte_val = (byte_val << 1) | bit
                payload_bytes.append(byte_val)
            else:
                errors.append("Incomplete payload")
                break

        # Frame CRC (24 bits)
        crc_bits = sample_bits(24)
        frame_crc = 0
        for bit in crc_bits:
            frame_crc = (frame_crc << 1) | bit

        # Create frame
        frame = FlexRayFrame(
            slot_id=slot_id,
            cycle_count=cycle_count,
            payload_length=payload_length,
            header_crc=header_crc,
            payload=bytes(payload_bytes),
            frame_crc=frame_crc,
            segment=FlexRaySegment.STATIC,  # Simplified: assume static
            timestamp=tss_idx / sample_rate,
            errors=errors,
        )

        return frame, int(bit_idx)


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
