"""BLE (Bluetooth Low Energy) protocol analysis.

This module provides comprehensive BLE protocol decoding with GATT service
discovery, advertising packet parsing, and ATT operation decoding.

Example:
    >>> from oscura.analyzers.protocols.ble import BLEAnalyzer
    >>> analyzer = BLEAnalyzer()
    >>> analyzer.add_packet(packet)
    >>> services = analyzer.discover_services()

References:
    Bluetooth Core Specification v5.4: https://www.bluetooth.com/specifications/specs/
"""

from oscura.analyzers.protocols.ble.analyzer import (
    BLEAnalyzer,
    BLEPacket,
    GATTCharacteristic,
    GATTDescriptor,
    GATTService,
)
from oscura.analyzers.protocols.ble.uuids import (
    STANDARD_CHARACTERISTICS,
    STANDARD_DESCRIPTORS,
    STANDARD_SERVICES,
)

__all__ = [
    "STANDARD_CHARACTERISTICS",
    "STANDARD_DESCRIPTORS",
    "STANDARD_SERVICES",
    "BLEAnalyzer",
    "BLEPacket",
    "GATTCharacteristic",
    "GATTDescriptor",
    "GATTService",
]
