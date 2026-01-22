"""Parallel bus protocol decoders for vintage systems.

Implements decoders for classic parallel bus protocols:
- IEEE-488 (GPIB): General Purpose Interface Bus for instruments
- Centronics: Parallel printer interface
- ISA: Industry Standard Architecture bus

Example:
    >>> from oscura.analyzers.protocols.parallel_bus import decode_gpib, decode_centronics
    >>> frames = decode_gpib(dio_lines, dav, nrfd, ndac, eoi, atn)
    >>> print_data = decode_centronics(data_lines, strobe, busy, ack)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


# =============================================================================
# IEEE-488 (GPIB) Protocol Decoder
# =============================================================================


class GPIBMessageType(Enum):
    """GPIB message types."""

    DATA = "data"  # Data bytes (ATN=0)
    COMMAND = "command"  # Command bytes (ATN=1)
    TALK_ADDRESS = "talk_address"  # Talk address
    LISTEN_ADDRESS = "listen_address"  # Listen address
    SECONDARY_ADDRESS = "secondary_address"  # Secondary address
    UNIVERSAL_COMMAND = "universal_command"  # Universal command
    ADDRESSED_COMMAND = "addressed_command"  # Addressed command


@dataclass
class GPIBFrame:
    """A decoded GPIB frame.

    Attributes:
        timestamp: Frame timestamp in seconds.
        data: Data byte value (0-255).
        message_type: Type of GPIB message.
        eoi: End-Or-Identify asserted.
        description: Human-readable description.
    """

    timestamp: float
    data: int
    message_type: GPIBMessageType
    eoi: bool
    description: str


def decode_gpib(
    dio_lines: list[NDArray[np.bool_]],  # DIO1-DIO8 (8 data lines)
    dav: NDArray[np.bool_],  # Data Valid
    nrfd: NDArray[np.bool_],  # Not Ready For Data
    ndac: NDArray[np.bool_],  # Not Data Accepted
    eoi: NDArray[np.bool_],  # End Or Identify
    atn: NDArray[np.bool_],  # Attention
    sample_rate: float = 1.0,
) -> list[GPIBFrame]:
    """Decode IEEE-488 (GPIB) bus transactions.

    Args:
        dio_lines: List of 8 digital traces for DIO1-DIO8.
        dav: Data Valid signal (active low).
        nrfd: Not Ready For Data signal (active low).
        ndac: Not Data Accepted signal (active low).
        eoi: End Or Identify signal (active low).
        atn: Attention signal (active low).
        sample_rate: Sample rate in Hz.

    Returns:
        List of GPIBFrame objects.

    Example:
        >>> frames = decode_gpib(dio, dav, nrfd, ndac, eoi, atn, 1e6)
        >>> for frame in frames:
        ...     print(f"{frame.timestamp*1e6:.1f}us: {frame.description}")
    """
    if len(dio_lines) != 8:
        raise ValueError("GPIB requires exactly 8 DIO lines")

    frames: list[GPIBFrame] = []
    time_base = 1.0 / sample_rate

    # Combine DIO lines into data bus
    data_bus = np.zeros(len(dio_lines[0]), dtype=np.uint8)
    for i, line in enumerate(dio_lines):
        data_bus |= (line.astype(np.uint8) << i).astype(np.uint8)

    # Detect DAV falling edges (data valid)
    dav_falling = np.where(np.diff(dav.astype(np.int8)) == -1)[0]

    for idx in dav_falling:
        # Sample data after falling edge
        sample_idx = idx + 1
        if sample_idx >= len(data_bus):
            continue

        timestamp = idx * time_base
        data_byte = int(data_bus[sample_idx])
        eoi_active = not eoi[sample_idx]  # Active low
        atn_active = not atn[sample_idx]  # Active low

        # Decode message type based on ATN and data
        if atn_active:
            # Check address bytes first (bit patterns for talk/listen)
            if data_byte & 0x40:
                # Talk address (bit 6 set)
                address = data_byte & 0x1F
                msg_type = GPIBMessageType.TALK_ADDRESS
                desc = f"Talk address {address}"
            elif data_byte & 0x20:
                # Listen address (bit 5 set, bit 6 clear)
                address = data_byte & 0x1F
                msg_type = GPIBMessageType.LISTEN_ADDRESS
                desc = f"Listen address {address}"
            elif 0x10 <= data_byte <= 0x1F:
                # Universal commands
                msg_type = GPIBMessageType.UNIVERSAL_COMMAND
                desc = _gpib_universal_command_name(data_byte)
            else:
                # Addressed commands (0x00-0x0F and others)
                msg_type = GPIBMessageType.ADDRESSED_COMMAND
                desc = _gpib_addressed_command_name(data_byte)
        else:
            # Data byte
            msg_type = GPIBMessageType.DATA
            desc = f"Data: 0x{data_byte:02X}"
            if 32 <= data_byte <= 126:
                desc += f" ('{chr(data_byte)}')"

        if eoi_active:
            desc += " [EOI]"

        frames.append(
            GPIBFrame(
                timestamp=timestamp,
                data=data_byte,
                message_type=msg_type,
                eoi=eoi_active,
                description=desc,
            )
        )

    return frames


def _gpib_universal_command_name(cmd: int) -> str:
    """Get name of GPIB universal command."""
    commands = {
        0x11: "DCL (Device Clear)",
        0x14: "GET (Group Execute Trigger)",
        0x15: "GTL (Go To Local)",
        0x08: "LLO (Local Lockout)",
        0x01: "SPD (Serial Poll Disable)",
        0x18: "SPE (Serial Poll Enable)",
        0x13: "PPU (Parallel Poll Unconfigure)",
    }
    return commands.get(cmd, f"Unknown universal command 0x{cmd:02X}")


def _gpib_addressed_command_name(cmd: int) -> str:
    """Get name of GPIB addressed command."""
    commands = {
        0x04: "SDC (Selected Device Clear)",
        0x05: "PPC (Parallel Poll Configure)",
        0x09: "TCT (Take Control)",
    }
    return commands.get(cmd, f"Unknown addressed command 0x{cmd:02X}")


# =============================================================================
# Centronics Parallel Printer Protocol Decoder
# =============================================================================


@dataclass
class CentronicsFrame:
    """A decoded Centronics printer frame.

    Attributes:
        timestamp: Frame timestamp in seconds.
        data: Data byte value (0-255).
        character: ASCII character (if printable).
        control: Control signal states.
    """

    timestamp: float
    data: int
    character: str | None
    control: dict[str, bool]


def decode_centronics(
    data_lines: list[NDArray[np.bool_]],  # D0-D7 (8 data lines)
    strobe: NDArray[np.bool_],  # Strobe signal (active low)
    busy: NDArray[np.bool_],  # Busy signal
    ack: NDArray[np.bool_],  # Acknowledge signal (active low)
    sample_rate: float = 1.0,
    *,
    select: NDArray[np.bool_] | None = None,  # Select signal
    paper_out: NDArray[np.bool_] | None = None,  # Paper Out signal
    error: NDArray[np.bool_] | None = None,  # Error signal
) -> list[CentronicsFrame]:
    """Decode Centronics parallel printer protocol.

    Args:
        data_lines: List of 8 digital traces for D0-D7.
        strobe: Strobe signal (active low).
        busy: Busy signal (high when printer busy).
        ack: Acknowledge signal (active low pulse).
        sample_rate: Sample rate in Hz.
        select: Optional select signal.
        paper_out: Optional paper out signal.
        error: Optional error signal.

    Returns:
        List of CentronicsFrame objects.

    Example:
        >>> frames = decode_centronics(data, strobe, busy, ack, 1e6)
        >>> for frame in frames:
        ...     if frame.character:
        ...         print(frame.character, end='')
    """
    if len(data_lines) != 8:
        raise ValueError("Centronics requires exactly 8 data lines")

    frames: list[CentronicsFrame] = []
    time_base = 1.0 / sample_rate

    # Combine data lines into bytes
    data_bus = np.zeros(len(data_lines[0]), dtype=np.uint8)
    for i, line in enumerate(data_lines):
        data_bus |= (line.astype(np.uint8) << i).astype(np.uint8)

    # Detect STROBE falling edges (data latch)
    strobe_falling = np.where(np.diff(strobe.astype(np.int8)) == -1)[0]

    for idx in strobe_falling:
        # Sample data after falling edge
        sample_idx = idx + 1
        if sample_idx >= len(data_bus):
            continue

        timestamp = idx * time_base
        data_byte = int(data_bus[sample_idx])

        # Check if printable ASCII
        char = chr(data_byte) if 32 <= data_byte <= 126 else None

        # Capture control signals
        control = {
            "busy": bool(busy[sample_idx]),
            "ack": not ack[sample_idx],  # Active low
        }

        if select is not None:
            control["select"] = bool(select[sample_idx])
        if paper_out is not None:
            control["paper_out"] = bool(paper_out[sample_idx])
        if error is not None:
            control["error"] = bool(error[sample_idx])

        frames.append(
            CentronicsFrame(
                timestamp=timestamp,
                data=data_byte,
                character=char,
                control=control,
            )
        )

    return frames


# =============================================================================
# ISA Bus Protocol Analyzer
# =============================================================================


class ISACycleType(Enum):
    """ISA bus cycle types."""

    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    DMA = "dma"
    INTERRUPT = "interrupt"


@dataclass
class ISATransaction:
    """An ISA bus transaction.

    Attributes:
        timestamp: Transaction timestamp in seconds.
        cycle_type: Type of bus cycle.
        address: Address (20-bit for memory, 16-bit for I/O).
        data: Data byte (if applicable).
        description: Human-readable description.
    """

    timestamp: float
    cycle_type: ISACycleType
    address: int
    data: int | None
    description: str


def decode_isa_bus(
    address_lines: list[NDArray[np.bool_]],  # SA0-SA19 (20 address lines)
    data_lines: list[NDArray[np.bool_]],  # SD0-SD7 (8 data lines)
    ior: NDArray[np.bool_],  # I/O Read (active low)
    iow: NDArray[np.bool_],  # I/O Write (active low)
    memr: NDArray[np.bool_],  # Memory Read (active low)
    memw: NDArray[np.bool_],  # Memory Write (active low)
    ale: NDArray[np.bool_],  # Address Latch Enable
    sample_rate: float = 1.0,
) -> list[ISATransaction]:
    """Decode ISA bus transactions.

    Args:
        address_lines: List of 20 digital traces for SA0-SA19.
        data_lines: List of 8 digital traces for SD0-SD7.
        ior: I/O Read signal (active low).
        iow: I/O Write signal (active low).
        memr: Memory Read signal (active low).
        memw: Memory Write signal (active low).
        ale: Address Latch Enable.
        sample_rate: Sample rate in Hz.

    Returns:
        List of ISATransaction objects.

    Example:
        >>> trans = decode_isa_bus(addr, data, ior, iow, memr, memw, ale, 1e6)
        >>> for t in trans:
        ...     print(f"{t.timestamp*1e6:.1f}us: {t.description}")
    """
    if len(address_lines) < 16:
        raise ValueError("ISA bus requires at least 16 address lines")
    if len(data_lines) != 8:
        raise ValueError("ISA bus requires exactly 8 data lines")

    transactions: list[ISATransaction] = []
    time_base = 1.0 / sample_rate

    # Combine address lines
    address_bus = np.zeros(len(address_lines[0]), dtype=np.uint32)
    for i, line in enumerate(address_lines):
        address_bus |= (line.astype(np.uint32) << i).astype(np.uint32)

    # Combine data lines
    data_bus = np.zeros(len(data_lines[0]), dtype=np.uint8)
    for i, line in enumerate(data_lines):
        data_bus |= (line.astype(np.uint8) << i).astype(np.uint8)

    # Detect ALE falling edges (address latch)
    ale_falling = np.where(np.diff(ale.astype(np.int8)) == -1)[0]

    for idx in ale_falling:
        if idx >= len(address_bus):
            continue

        timestamp = idx * time_base
        address = int(address_bus[idx])

        # Look ahead for read/write strobes (larger window to catch delayed strobes)
        search_window = min(idx + 200, len(ior))

        ior_active = np.any(~ior[idx:search_window])
        iow_active = np.any(~iow[idx:search_window])
        memr_active = np.any(~memr[idx:search_window])
        memw_active = np.any(~memw[idx:search_window])

        # Determine cycle type
        data_val = None

        if ior_active:
            cycle_type = ISACycleType.IO_READ
            desc = f"I/O Read from 0x{address:04X}"
            # Find data at IOR falling edge
            ior_idx = np.where(~ior[idx:search_window])[0]
            if len(ior_idx) > 0:
                data_val = int(data_bus[idx + ior_idx[0]])
                desc += f" = 0x{data_val:02X}"

        elif iow_active:
            cycle_type = ISACycleType.IO_WRITE
            iow_idx = np.where(~iow[idx:search_window])[0]
            if len(iow_idx) > 0:
                data_val = int(data_bus[idx + iow_idx[0]])
            desc = f"I/O Write 0x{data_val:02X} to 0x{address:04X}"

        elif memr_active:
            cycle_type = ISACycleType.MEMORY_READ
            desc = f"Memory Read from 0x{address:05X}"
            memr_idx = np.where(~memr[idx:search_window])[0]
            if len(memr_idx) > 0:
                data_val = int(data_bus[idx + memr_idx[0]])
                desc += f" = 0x{data_val:02X}"

        elif memw_active:
            cycle_type = ISACycleType.MEMORY_WRITE
            memw_idx = np.where(~memw[idx:search_window])[0]
            if len(memw_idx) > 0:
                data_val = int(data_bus[idx + memw_idx[0]])
            desc = f"Memory Write 0x{data_val:02X} to 0x{address:05X}"

        else:
            # No control signals active
            continue

        transactions.append(
            ISATransaction(
                timestamp=timestamp,
                cycle_type=cycle_type,
                address=address,
                data=data_val,
                description=desc,
            )
        )

    return transactions


__all__ = [
    "CentronicsFrame",
    "GPIBFrame",
    "GPIBMessageType",
    "ISACycleType",
    "ISATransaction",
    "decode_centronics",
    "decode_gpib",
    "decode_isa_bus",
]
