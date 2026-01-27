"""OPC UA (Unified Architecture) protocol analyzer.

This package provides comprehensive OPC UA binary protocol analysis for
industrial automation and SCADA systems.

Example:
    >>> from oscura.analyzers.protocols.industrial.opcua import OPCUAAnalyzer
    >>> analyzer = OPCUAAnalyzer()
    >>> # Parse OPC UA Hello message
    >>> hello = bytes([0x48, 0x45, 0x4C, 0x46, ...])  # "HEL" + "F"
    >>> msg = analyzer.parse_message(hello, timestamp=0.0)
    >>> print(f"{msg.message_type}: {msg.decoded_service}")
"""

from oscura.analyzers.protocols.industrial.opcua.analyzer import (
    OPCUAAnalyzer,
    OPCUAMessage,
    OPCUANode,
)

__all__ = ["OPCUAAnalyzer", "OPCUAMessage", "OPCUANode"]
