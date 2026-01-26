"""PROFINET IO protocol analyzer package.

This package provides comprehensive PROFINET protocol analysis for real-time
industrial Ethernet communication, including RT, IRT, DCP, and PTCP protocols.

Example:
    >>> from oscura.analyzers.protocols.industrial.profinet import ProfinetAnalyzer
    >>> analyzer = ProfinetAnalyzer()
    >>> frame = analyzer.parse_frame(ethernet_data, timestamp=0.0)
    >>> devices = analyzer.discover_devices()

References:
    PROFINET Specification V2.4 (IEC 61158 / IEC 61784):
    https://www.profibus.com/download/profinet-specification/
"""

from oscura.analyzers.protocols.industrial.profinet.analyzer import (
    ProfinetAnalyzer,
    ProfinetDevice,
    ProfinetFrame,
)

__all__ = ["ProfinetAnalyzer", "ProfinetDevice", "ProfinetFrame"]
