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

    channels = [  # noqa: RUF012
        ChannelDef("bck", "BCK", "Bit Clock (SCLK)", required=True),
        ChannelDef("ws", "WS", "Word Select (LRCLK)", required=True),
        ChannelDef("sd", "SD", "Serial Data", required=True),
    ]

    optional_channels = []  # noqa: RUF012

    options = [  # noqa: RUF012
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

    annotations = [  # noqa: RUF012
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

        n_samples = min(len(bck), len(ws), len(sd))
        bck = bck[:n_samples]
        ws = ws[:n_samples]
        sd = sd[:n_samples]

        # Find rising edges of BCK (data sampled on rising edge in I2S)
        rising_edges = np.where(~bck[:-1] & bck[1:])[0] + 1

        # Find WS transitions to identify word boundaries
        ws_transitions = np.where(ws[:-1] != ws[1:])[0] + 1

        if len(rising_edges) == 0 or len(ws_transitions) == 0:
            return

        trans_num = 0
        ws_idx = 0

        while ws_idx < len(ws_transitions) - 1:
            # Get word boundaries
            word_start_idx = ws_transitions[ws_idx]
            word_end_idx = ws_transitions[ws_idx + 1]

            # Determine channel (WS=0 is left, WS=1 is right in standard I2S)
            is_left = not ws[word_start_idx]

            # Find BCK edges in this word period
            word_edges = rising_edges[
                (rising_edges >= word_start_idx) & (rising_edges < word_end_idx)
            ]

            if len(word_edges) == 0:
                ws_idx += 1
                continue

            # In standard I2S mode, data starts 1 clock after WS change
            # In left-justified mode, data starts at WS change
            # In right-justified mode, data is aligned to end of word period
            if self._mode == I2SMode.STANDARD:
                # Skip first edge (data starts on second edge)
                data_edges = word_edges[1:] if len(word_edges) > 1 else []
            elif self._mode == I2SMode.LEFT_JUSTIFIED:
                data_edges = word_edges
            else:  # RIGHT_JUSTIFIED
                # Take last bit_depth edges
                data_edges = (
                    word_edges[-self._bit_depth :]
                    if len(word_edges) >= self._bit_depth
                    else word_edges
                )

            # Extract sample data (MSB first)
            sample_bits = []
            for edge_idx in data_edges[: self._bit_depth]:
                if edge_idx < len(sd):
                    sample_bits.append(1 if sd[edge_idx] else 0)

            if len(sample_bits) < self._bit_depth:
                # Incomplete sample, pad with zeros
                sample_bits.extend([0] * (self._bit_depth - len(sample_bits)))

            # Convert to signed integer value (MSB first, two's complement)
            sample_value = 0
            for bit in sample_bits:
                sample_value = (sample_value << 1) | bit

            # Convert from unsigned to signed (two's complement)
            if sample_bits[0] == 1:  # Negative number
                sample_value = sample_value - (1 << self._bit_depth)

            # Calculate timing
            start_time = word_start_idx / sample_rate
            end_time = word_end_idx / sample_rate

            # Store left and right channels
            if ws_idx % 2 == 0:
                # First word of stereo pair
                left_sample = sample_value if is_left else 0
                right_sample = 0 if is_left else sample_value
                first_start_time = start_time
            else:
                # Second word of stereo pair - emit packet
                if is_left:
                    left_sample = sample_value
                else:
                    right_sample = sample_value

                # Add annotation
                self.put_annotation(
                    first_start_time,
                    end_time,
                    AnnotationLevel.PACKETS,
                    f"L: {left_sample} / R: {right_sample}",
                )

                # Create packet
                annotations = {
                    "sample_num": trans_num,
                    "left_sample": left_sample,
                    "right_sample": right_sample,
                    "bit_depth": self._bit_depth,
                    "mode": self._mode.value,
                }

                # Encode as bytes (little-endian, signed)
                byte_count = (self._bit_depth + 7) // 8
                left_bytes = left_sample.to_bytes(byte_count, "little", signed=True)
                right_bytes = right_sample.to_bytes(byte_count, "little", signed=True)
                data_bytes = left_bytes + right_bytes

                packet = ProtocolPacket(
                    timestamp=first_start_time,
                    protocol="i2s",
                    data=data_bytes,
                    annotations=annotations,
                    errors=[],
                )

                yield packet
                trans_num += 1

            ws_idx += 1


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
