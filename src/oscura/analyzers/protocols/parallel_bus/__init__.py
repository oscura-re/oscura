"""Parallel bus protocol decoders.

This module provides decoders for parallel bus protocols including:
- GPIB (IEEE-488): General Purpose Interface Bus for instrument control
- Centronics: Parallel printer interface
- ISA: Industry Standard Architecture bus

Example:
    >>> from oscura.analyzers.protocols.parallel_bus import decode_gpib
    >>> frames = decode_gpib(dio_lines, dav, nrfd, ndac, eoi, atn, sample_rate)
    >>> for frame in frames:
    ...     print(f"Type: {frame.message_type}, Data: 0x{frame.data:02X}")
"""

from __future__ import annotations

from oscura.analyzers.protocols.parallel_bus.centronics import decode_centronics
from oscura.analyzers.protocols.parallel_bus.gpib import decode_gpib

__all__ = ["decode_centronics", "decode_gpib"]
