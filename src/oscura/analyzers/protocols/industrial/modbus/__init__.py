"""Modbus RTU/TCP protocol support.

This module provides comprehensive Modbus protocol analysis for both RTU (serial)
and TCP (Ethernet) variants, including all standard function codes, CRC validation,
and device state tracking.

Example:
    >>> from oscura.analyzers.protocols.industrial.modbus import ModbusAnalyzer
    >>> analyzer = ModbusAnalyzer()
    >>> message = analyzer.parse_rtu(frame_bytes)
    >>> print(f"{message.function_name}: {message.parsed_data}")

References:
    Modbus Application Protocol V1.1b3
    Modbus over Serial Line V1.02
"""

from oscura.analyzers.protocols.industrial.modbus.analyzer import (
    ModbusAnalyzer,
    ModbusDevice,
    ModbusMessage,
)
from oscura.analyzers.protocols.industrial.modbus.crc import calculate_crc, verify_crc

__all__ = [
    "ModbusAnalyzer",
    "ModbusDevice",
    "ModbusMessage",
    "calculate_crc",
    "verify_crc",
]
