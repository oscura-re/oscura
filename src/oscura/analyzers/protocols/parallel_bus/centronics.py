"""Centronics parallel printer protocol decoder.

Decoder for the Centronics parallel printer interface, commonly used
in legacy printer connections (pre-USB).

Protocol Overview:
    - 8 data lines (D0-D7)
    - STROBE: Latches data (active low)
    - BUSY: Printer busy signal (active high)
    - ACK: Acknowledge data received (active low)

References:
    - Centronics Data Computer Corp. Parallel Interface Standard
    - IEEE 1284-1994 (modern parallel port standard based on Centronics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class CentronicsFrame:
    """Decoded Centronics frame.

    Attributes:
        timestamp: Frame timestamp in seconds
        data: Data byte value
        character: ASCII character if printable, None otherwise
    """

    timestamp: float
    data: int
    character: str | None


def decode_centronics(
    data_lines: list[NDArray[np.bool_]],
    strobe: NDArray[np.bool_],
    busy: NDArray[np.bool_],
    ack: NDArray[np.bool_],
    sample_rate: float,
) -> list[CentronicsFrame]:
    """Decode Centronics parallel printer protocol.

    Args:
        data_lines: List of 8 data line arrays (D0-D7)
        strobe: Strobe signal (active low, latches data)
        busy: Busy signal (active high when printer busy)
        ack: Acknowledge signal (active low when data accepted)
        sample_rate: Sample rate in Hz

    Returns:
        List of decoded Centronics frames

    Example:
        >>> frames = decode_centronics(data_lines, strobe, busy, ack, 1e6)
        >>> text = ''.join(f.character or f'\\x{f.data:02x}' for f in frames)
        >>> print(f"Printer data: {text}")
    """
    frames: list[CentronicsFrame] = []

    # Find STROBE falling edges (data latch events)
    strobe_falling = np.where((strobe[:-1] == True) & (strobe[1:] == False))[0]  # noqa: E712

    for edge_idx in strobe_falling:
        # Sample data at STROBE falling edge
        data_byte = 0
        for bit_idx, data_line in enumerate(data_lines):
            if edge_idx < len(data_line) and data_line[edge_idx]:
                data_byte |= 1 << bit_idx

        timestamp = edge_idx / sample_rate

        # Convert to character if printable ASCII
        character = chr(data_byte) if 32 <= data_byte < 127 else None

        frame = CentronicsFrame(
            timestamp=timestamp,
            data=data_byte,
            character=character,
        )

        frames.append(frame)

    return frames
