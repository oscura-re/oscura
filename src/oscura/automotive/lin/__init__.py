"""LIN (Local Interconnect Network) protocol analysis.

This module provides comprehensive LIN protocol analysis capabilities including:
- LIN 2.x frame parsing with enhanced checksum support
- Protected ID calculation with parity bits
- Signal decoding and extraction
- Diagnostic services (master request/slave response)
- Schedule table inference from captured traffic
- LDF (LIN Description File) generation

Key features:
- Both classic and enhanced checksum validation
- Diagnostic frame parsing (0x3C, 0x3D)
- Event-triggered frame handling
- Signal mapping and decoding
- Automatic LDF generation from traffic

Example:
    >>> from oscura.automotive.lin import LINAnalyzer, LINSignal
    >>> analyzer = LINAnalyzer()
    >>> # Parse captured LIN frame
    >>> frame = analyzer.parse_frame(data=b'\\x55\\x80\\x01\\x02\\x03\\xFA', timestamp=1.0)
    >>> print(f"Frame ID: {frame.frame_id}, Checksum: {frame.checksum_type}")
    Frame ID: 0, Checksum: enhanced
    >>> # Add signal definition
    >>> analyzer.add_signal(LINSignal("Speed", frame_id=0, start_bit=0, bit_length=16))
    >>> # Generate LDF from captured traffic
    >>> analyzer.generate_ldf(Path("output.ldf"), baudrate=19200)

References:
    LIN Specification 2.2A
    ISO 17987 (LIN standard)
"""

__all__ = [
    "LINAnalyzer",
    "LINFrame",
    "LINScheduleEntry",
    "LINSignal",
]

from oscura.automotive.lin.analyzer import (
    LINAnalyzer,
    LINFrame,
    LINScheduleEntry,
    LINSignal,
)
