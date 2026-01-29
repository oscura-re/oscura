"""FlexRay automotive protocol analysis.

This module provides FlexRay-specific analysis tools including frame parsing,
CRC validation, signal decoding, and FIBEX format support.

Example:
    >>> from oscura.automotive.flexray import FlexRayAnalyzer, FlexRaySignal
    >>> analyzer = FlexRayAnalyzer()
    >>> frame = analyzer.parse_frame(data, timestamp=0.0, channel="A")
    >>> print(f"Slot {frame.header.frame_id}, CRC valid: {frame.crc_valid}")
"""

from oscura.automotive.flexray.analyzer import (
    FlexRayAnalyzer,
    FlexRayFrame,
    FlexRayHeader,
    FlexRaySignal,
)
from oscura.automotive.flexray.crc import calculate_frame_crc, calculate_header_crc
from oscura.automotive.flexray.fibex import FIBEXExporter, FIBEXImporter

__all__ = [
    "FIBEXExporter",
    "FIBEXImporter",
    "FlexRayAnalyzer",
    "FlexRayFrame",
    "FlexRayHeader",
    "FlexRaySignal",
    "calculate_frame_crc",
    "calculate_header_crc",
]
