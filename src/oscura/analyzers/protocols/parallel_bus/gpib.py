"""GPIB (IEEE-488) protocol decoder.

IEEE 488-1978 General Purpose Interface Bus (GPIB) decoder for instrument control.

References:
    - IEEE 488.1-2003: IEEE Standard Digital Interface for Programmable Instrumentation
    - HP Application Note 1298: Understanding GPIB Addressing
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GPIBMessageType(Enum):
    """GPIB message types."""

    TALK_ADDRESS = "talk_address"
    LISTEN_ADDRESS = "listen_address"
    DATA = "data"
    COMMAND = "command"


@dataclass
class GPIBFrame:
    """Decoded GPIB frame.

    Attributes:
        timestamp: Frame timestamp in seconds
        data: Data byte value
        message_type: Type of GPIB message
        description: Human-readable description
    """

    timestamp: float
    data: int
    message_type: GPIBMessageType
    description: str


def decode_gpib(
    dio_lines: list[NDArray[np.bool_]],
    dav: NDArray[np.bool_],
    nrfd: NDArray[np.bool_],
    ndac: NDArray[np.bool_],
    eoi: NDArray[np.bool_],
    atn: NDArray[np.bool_],
    sample_rate: float,
) -> list[GPIBFrame]:
    """Decode GPIB (IEEE-488) protocol.

    Args:
        dio_lines: List of 8 DIO line arrays (data lines)
        dav: Data Valid signal (active low)
        nrfd: Not Ready For Data (active low)
        ndac: Not Data Accepted (active low)
        eoi: End or Identify (active low)
        atn: Attention (active low for command, high for data)
        sample_rate: Sample rate in Hz

    Returns:
        List of decoded GPIB frames

    Example:
        >>> frames = decode_gpib(dio_lines, dav, nrfd, ndac, eoi, atn, 1e6)
        >>> for frame in frames:
        ...     print(f"{frame.timestamp*1e6:.1f}µs: {frame.description}")
    """
    frames: list[GPIBFrame] = []

    # Find DAV falling edges (data valid transitions)
    dav_falling = np.where((dav[:-1] == True) & (dav[1:] == False))[0]  # noqa: E712

    for edge_idx in dav_falling:
        # Sample data at DAV falling edge
        data_byte = 0
        for bit_idx, dio_line in enumerate(dio_lines):
            if edge_idx < len(dio_line) and dio_line[edge_idx]:
                data_byte |= 1 << bit_idx

        timestamp = edge_idx / sample_rate

        # Determine message type from ATN and EOI
        atn_active = not atn[edge_idx] if edge_idx < len(atn) else False
        eoi_active = not eoi[edge_idx] if edge_idx < len(eoi) else False

        if atn_active:
            # Command/address mode
            if data_byte & 0x40:  # Talk address
                address = data_byte & 0x1F
                frame = GPIBFrame(
                    timestamp=timestamp,
                    data=data_byte,
                    message_type=GPIBMessageType.TALK_ADDRESS,
                    description=f"Talk address {address}",
                )
            elif data_byte & 0x20:  # Listen address
                address = data_byte & 0x1F
                frame = GPIBFrame(
                    timestamp=timestamp,
                    data=data_byte,
                    message_type=GPIBMessageType.LISTEN_ADDRESS,
                    description=f"Listen address {address}",
                )
            else:  # Command
                frame = GPIBFrame(
                    timestamp=timestamp,
                    data=data_byte,
                    message_type=GPIBMessageType.COMMAND,
                    description=f"Command 0x{data_byte:02X}",
                )
        else:
            # Data mode
            char = chr(data_byte) if 32 <= data_byte < 127 else ""
            desc = f"Data 0x{data_byte:02X}"
            if char:
                desc += f" ('{char}')"
            if eoi_active:
                desc += " [EOI]"

            frame = GPIBFrame(
                timestamp=timestamp,
                data=data_byte,
                message_type=GPIBMessageType.DATA,
                description=desc,
            )

        frames.append(frame)

    return frames
