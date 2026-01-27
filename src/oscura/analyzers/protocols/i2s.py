"""I2S protocol decoder.

This module provides Inter-IC Sound (I2S) audio protocol decoding
with support for standard, left-justified, and right-justified modes.


Example:
    >>> from oscura.analyzers.protocols.i2s import I2SDecoder
    >>> decoder = I2SDecoder(bit_depth=16)
    >>> for packet in decoder.decode(bck=bck, ws=ws, sd=sd):
    ...     print(f"Left: {packet.annotations['left_sample']}")

References:
    I2S Bus Specification (Philips Semiconductors)
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal

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


class I2SMode(Enum):
    """I2S alignment modes."""

    STANDARD = "standard"  # MSB 1 clock after WS change
    LEFT_JUSTIFIED = "left_justified"  # MSB at WS change
    RIGHT_JUSTIFIED = "right_justified"  # MSB before WS change


class I2SDecoder(SyncDecoder):
    """I2S protocol decoder.

    Decodes I2S audio bus transactions with configurable bit depth
    and alignment modes (standard, left-justified, right-justified).

    Attributes:
        id: "i2s"
        name: "I2S"
        channels: [bck, ws, sd] (required)

    Example:
        >>> decoder = I2SDecoder(bit_depth=24, mode="standard")
        >>> for packet in decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=1e6):
        ...     print(f"Stereo: L={packet.annotations['left']} R={packet.annotations['right']}")
    """

    id = "i2s"
    name = "I2S"
    longname = "Inter-IC Sound"
    desc = "I2S audio bus protocol decoder"

    channels = [
        ChannelDef("bck", "BCK", "Bit Clock (SCLK)", required=True),
        ChannelDef("ws", "WS", "Word Select (LRCLK)", required=True),
        ChannelDef("sd", "SD", "Serial Data", required=True),
    ]

    optional_channels = []

    options = [
        OptionDef(
            "bit_depth",
            "Bit depth",
            "Bits per sample",
            default=16,
            values=[8, 16, 24, 32],
        ),
        OptionDef(
            "mode",
            "Mode",
            "Alignment mode",
            default="standard",
            values=["standard", "left_justified", "right_justified"],
        ),
    ]

    annotations = [
        ("left", "Left channel sample"),
        ("right", "Right channel sample"),
        ("word", "Word boundary"),
    ]

    def __init__(
        self,
        bit_depth: int = 16,
        mode: Literal["standard", "left_justified", "right_justified"] = "standard",
    ) -> None:
        """Initialize I2S decoder.

        Args:
            bit_depth: Bits per sample (8, 16, 24, 32).
            mode: Alignment mode.
        """
        super().__init__(bit_depth=bit_depth, mode=mode)
        self._bit_depth = bit_depth
        self._mode = I2SMode(mode)

    def decode(  # type: ignore[override]
        self,
        trace: DigitalTrace | None = None,
        *,
        bck: NDArray[np.bool_] | None = None,
        ws: NDArray[np.bool_] | None = None,
        sd: NDArray[np.bool_] | None = None,
        sample_rate: float = 1.0,
    ) -> Iterator[ProtocolPacket]:
        """Decode I2S audio data.

        Args:
            trace: Optional primary trace.
            bck: Bit Clock signal.
            ws: Word Select signal (0=left, 1=right).
            sd: Serial Data signal.
            sample_rate: Sample rate in Hz.

        Yields:
            Decoded I2S samples as ProtocolPacket objects.

        Example:
            >>> decoder = I2SDecoder(bit_depth=16)
            >>> for pkt in decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=1e6):
            ...     print(f"Left: {pkt.annotations['left_sample']}")
        """
        if bck is None or ws is None or sd is None:
            return

        bck, ws, sd = self._align_signals(bck, ws, sd)
        rising_edges = np.where(~bck[:-1] & bck[1:])[0] + 1
        ws_transitions = np.where(ws[:-1] != ws[1:])[0] + 1

        if len(rising_edges) == 0 or len(ws_transitions) == 0:
            return

        trans_num = 0
        ws_idx = 0
        left_sample = 0
        right_sample = 0
        first_start_time = 0.0

        while ws_idx < len(ws_transitions) - 1:
            word_start_idx = ws_transitions[ws_idx]
            word_end_idx = ws_transitions[ws_idx + 1]

            is_left = not ws[word_start_idx]
            word_edges = self._get_word_edges(rising_edges, word_start_idx, word_end_idx)

            if len(word_edges) == 0:
                ws_idx += 1
                continue

            data_edges = self._select_data_edges(word_edges)
            sample_value = self._extract_sample(sd, data_edges)

            start_time = word_start_idx / sample_rate
            end_time = word_end_idx / sample_rate

            # Accumulate stereo pair
            if ws_idx % 2 == 0:
                left_sample = sample_value if is_left else 0
                right_sample = 0 if is_left else sample_value
                first_start_time = start_time
            else:
                if is_left:
                    left_sample = sample_value
                else:
                    right_sample = sample_value

                packet = self._build_stereo_packet(
                    left_sample, right_sample, first_start_time, end_time, trans_num
                )

                yield packet
                trans_num += 1

            ws_idx += 1

    def _align_signals(
        self, bck: NDArray[np.bool_], ws: NDArray[np.bool_], sd: NDArray[np.bool_]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_]]:
        """Align all signals to minimum length.

        Args:
            bck: Bit clock signal.
            ws: Word select signal.
            sd: Serial data signal.

        Returns:
            Aligned signals.
        """
        n_samples = min(len(bck), len(ws), len(sd))
        return bck[:n_samples], ws[:n_samples], sd[:n_samples]

    def _get_word_edges(
        self, rising_edges: NDArray[np.intp], start_idx: int, end_idx: int
    ) -> NDArray[np.intp]:
        """Get clock edges within word period.

        Args:
            rising_edges: All rising edge indices.
            start_idx: Word start index.
            end_idx: Word end index.

        Returns:
            Edges within word period.
        """
        return rising_edges[(rising_edges >= start_idx) & (rising_edges < end_idx)]

    def _select_data_edges(self, word_edges: NDArray[np.intp]) -> NDArray[np.intp]:
        """Select data edges based on I2S mode.

        Args:
            word_edges: All edges in word period.

        Returns:
            Edges containing valid data.
        """
        if self._mode == I2SMode.STANDARD:
            # Skip first edge (data starts on second edge)
            return word_edges[1:] if len(word_edges) > 1 else np.array([])
        elif self._mode == I2SMode.LEFT_JUSTIFIED:
            return word_edges
        else:  # RIGHT_JUSTIFIED
            # Take last bit_depth edges
            if len(word_edges) >= self._bit_depth:
                return word_edges[-self._bit_depth :]
            return word_edges

    def _extract_sample(self, sd: NDArray[np.bool_], data_edges: NDArray[np.intp]) -> int:
        """Extract and convert sample value.

        Args:
            sd: Serial data signal.
            data_edges: Edges to sample.

        Returns:
            Signed sample value.
        """
        sample_bits = []
        for edge_idx in data_edges[: self._bit_depth]:
            if edge_idx < len(sd):
                sample_bits.append(1 if sd[edge_idx] else 0)

        # Pad incomplete samples
        if len(sample_bits) < self._bit_depth:
            sample_bits.extend([0] * (self._bit_depth - len(sample_bits)))

        # Convert to signed integer (MSB first, two's complement)
        sample_value = 0
        for bit in sample_bits:
            sample_value = (sample_value << 1) | bit

        # Convert to signed
        if sample_bits[0] == 1:  # Negative number
            sample_value = sample_value - (1 << self._bit_depth)

        return sample_value

    def _build_stereo_packet(
        self, left: int, right: int, start_time: float, end_time: float, sample_num: int
    ) -> ProtocolPacket:
        """Build I2S stereo sample packet.

        Args:
            left: Left channel sample.
            right: Right channel sample.
            start_time: Packet start time.
            end_time: Packet end time.
            sample_num: Sample number.

        Returns:
            Protocol packet.
        """
        self.put_annotation(
            start_time,
            end_time,
            AnnotationLevel.PACKETS,
            f"L: {left} / R: {right}",
        )

        annotations = {
            "sample_num": sample_num,
            "left_sample": left,
            "right_sample": right,
            "bit_depth": self._bit_depth,
            "mode": self._mode.value,
        }

        byte_count = (self._bit_depth + 7) // 8
        left_bytes = left.to_bytes(byte_count, "little", signed=True)
        right_bytes = right.to_bytes(byte_count, "little", signed=True)
        data_bytes = left_bytes + right_bytes

        return ProtocolPacket(
            timestamp=start_time,
            protocol="i2s",
            data=data_bytes,
            annotations=annotations,
            errors=[],
        )


def decode_i2s(
    bck: NDArray[np.bool_],
    ws: NDArray[np.bool_],
    sd: NDArray[np.bool_],
    sample_rate: float = 1.0,
    bit_depth: int = 16,
    mode: Literal["standard", "left_justified", "right_justified"] = "standard",
) -> list[ProtocolPacket]:
    """Convenience function to decode I2S audio data.

    Args:
        bck: Bit Clock signal.
        ws: Word Select signal.
        sd: Serial Data signal.
        sample_rate: Sample rate in Hz.
        bit_depth: Bits per sample (8, 16, 24, 32).
        mode: Alignment mode.

    Returns:
        List of decoded I2S stereo samples.

    Example:
        >>> packets = decode_i2s(bck, ws, sd, sample_rate=1e6, bit_depth=16)
        >>> for pkt in packets:
        ...     print(f"L={pkt.annotations['left_sample']}, R={pkt.annotations['right_sample']}")
    """
    decoder = I2SDecoder(bit_depth=bit_depth, mode=mode)
    return list(decoder.decode(bck=bck, ws=ws, sd=sd, sample_rate=sample_rate))


__all__ = ["I2SDecoder", "I2SMode", "decode_i2s"]
