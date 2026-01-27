"""EtherCAT protocol analyzer package.

EtherCAT (Ethernet for Control Automation Technology) is a high-performance
industrial fieldbus system based on Ethernet.

Example:
    >>> from oscura.analyzers.protocols.industrial.ethercat import EtherCATAnalyzer
    >>> analyzer = EtherCATAnalyzer()
    >>> frame = analyzer.parse_frame(ethernet_payload, timestamp=0.0)
    >>> print(f"Datagrams: {len(frame.datagrams)}")

References:
    IEC 61158 Type 12: https://www.iec.ch/
    ETG.1000 EtherCAT Protocol Specification
    ETG.2000 EtherCAT AL Protocol
"""

from oscura.analyzers.protocols.industrial.ethercat.analyzer import (
    EtherCATAnalyzer,
    EtherCATDatagram,
    EtherCATFrame,
    EtherCATSlave,
)

__all__ = [
    "EtherCATAnalyzer",
    "EtherCATDatagram",
    "EtherCATFrame",
    "EtherCATSlave",
]
