"""BACnet protocol analyzer package.

This package provides BACnet (Building Automation and Control Networks) protocol
analysis for both BACnet/IP (UDP) and BACnet/MSTP (serial) variants. Supports
NPDU/APDU parsing, service decoding, and device discovery for HVAC and building
automation systems.

Example:
    >>> from oscura.analyzers.protocols.industrial.bacnet import BACnetAnalyzer
    >>> analyzer = BACnetAnalyzer()
    >>> # Parse BACnet/IP message
    >>> message = analyzer.parse_bacnet_ip(udp_payload, timestamp=0.0)
    >>> # Parse BACnet MSTP frame
    >>> message = analyzer.parse_bacnet_mstp(serial_data, timestamp=0.0)

References:
    ANSI/ASHRAE Standard 135-2020 (BACnet):
    https://www.ashrae.org/technical-resources/bookstore/bacnet
"""

from oscura.analyzers.protocols.industrial.bacnet.analyzer import (
    BACnetAnalyzer,
    BACnetDevice,
    BACnetMessage,
    BACnetObject,
)

__all__ = [
    "BACnetAnalyzer",
    "BACnetDevice",
    "BACnetMessage",
    "BACnetObject",
]
