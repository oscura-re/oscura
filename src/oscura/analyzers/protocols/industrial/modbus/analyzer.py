"""Modbus RTU/TCP protocol analyzer.

This module provides comprehensive Modbus protocol analysis supporting both
RTU (serial) and TCP (Ethernet) variants. Decodes all standard Modbus function
codes, validates CRC for RTU, tracks device states, and exports register maps.

Example:
    >>> from oscura.analyzers.protocols.industrial.modbus.analyzer import ModbusAnalyzer
    >>> analyzer = ModbusAnalyzer()
    >>> # Parse RTU frame
    >>> frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A, 0xCD, 0xC5])
    >>> message = analyzer.parse_rtu(frame, timestamp=0.0)
    >>> print(f"{message.function_name}: {message.parsed_data}")
    >>> # Parse TCP frame
    >>> tcp_frame = bytes([0x00, 0x01, 0x00, 0x00, 0x00, 0x06, 0x01, 0x03, ...])
    >>> message = analyzer.parse_tcp(tcp_frame, timestamp=0.0)

References:
    Modbus Application Protocol V1.1b3:
    https://modbus.org/docs/Modbus_Application_Protocol_V1_1b3.pdf

    Modbus over Serial Line V1.02:
    https://modbus.org/docs/Modbus_over_serial_line_V1_02.pdf
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from oscura.analyzers.protocols.industrial.modbus.crc import verify_crc
from oscura.analyzers.protocols.industrial.modbus.functions import (
    parse_read_coils_request,
    parse_read_coils_response,
    parse_read_discrete_inputs_request,
    parse_read_discrete_inputs_response,
    parse_read_holding_registers_request,
    parse_read_holding_registers_response,
    parse_read_input_registers_request,
    parse_read_input_registers_response,
    parse_write_multiple_coils_request,
    parse_write_multiple_coils_response,
    parse_write_multiple_registers_request,
    parse_write_multiple_registers_response,
    parse_write_single_coil,
    parse_write_single_register,
)


@dataclass
class ModbusMessage:
    """Modbus message representation.

    Attributes:
        timestamp: Message timestamp in seconds.
        variant: Protocol variant ("RTU" or "TCP").
        is_request: True if request, False if response.
        transaction_id: TCP transaction ID (TCP only).
        unit_id: Slave address (RTU) or unit identifier (TCP).
        function_code: Modbus function code.
        function_name: Human-readable function name.
        data: Raw function data bytes.
        exception_code: Exception code if error response.
        parsed_data: Parsed function-specific data.
        crc_valid: CRC validation result (RTU only).
    """

    timestamp: float
    variant: str  # "RTU" or "TCP"
    is_request: bool
    transaction_id: int | None = None  # TCP only
    unit_id: int = 0
    function_code: int = 0
    function_name: str = ""
    data: bytes = b""
    exception_code: int | None = None
    parsed_data: dict[str, Any] = field(default_factory=dict)
    crc_valid: bool | None = None  # RTU only


@dataclass
class ModbusDevice:
    """Modbus device state information.

    Tracks the state of a Modbus device including register and coil values
    observed in communication.

    Attributes:
        unit_id: Device unit ID / slave address.
        function_codes_seen: Set of function codes observed.
        coils: Coil states (address -> value).
        discrete_inputs: Discrete input states (address -> value).
        holding_registers: Holding register values (address -> value).
        input_registers: Input register values (address -> value).
    """

    unit_id: int
    function_codes_seen: set[int] = field(default_factory=set)
    coils: dict[int, bool] = field(default_factory=dict)
    discrete_inputs: dict[int, bool] = field(default_factory=dict)
    holding_registers: dict[int, int] = field(default_factory=dict)
    input_registers: dict[int, int] = field(default_factory=dict)


class ModbusAnalyzer:
    """Modbus protocol analyzer for RTU and TCP variants.

    Provides comprehensive Modbus protocol analysis including frame parsing,
    function code decoding, CRC validation, and device state tracking.

    Attributes:
        messages: List of parsed Modbus messages.
        devices: Dictionary of device states by unit ID.

    Example:
        >>> analyzer = ModbusAnalyzer()
        >>> # Parse RTU frame
        >>> rtu_frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A, 0xCD, 0xC5])
        >>> msg = analyzer.parse_rtu(rtu_frame)
        >>> print(f"Function: {msg.function_name}, CRC Valid: {msg.crc_valid}")
        >>> # Export device register map
        >>> analyzer.export_register_map(Path("registers.json"))
    """

    # Standard Modbus function codes
    FUNCTION_CODES: ClassVar[dict[int, str]] = {
        1: "Read Coils",
        2: "Read Discrete Inputs",
        3: "Read Holding Registers",
        4: "Read Input Registers",
        5: "Write Single Coil",
        6: "Write Single Register",
        15: "Write Multiple Coils",
        16: "Write Multiple Registers",
        23: "Read/Write Multiple Registers",
        # Diagnostic and maintenance functions
        8: "Diagnostics",
        11: "Get Comm Event Counter",
        12: "Get Comm Event Log",
        17: "Report Server ID",
        # File and queue operations
        20: "Read File Record",
        21: "Write File Record",
        22: "Mask Write Register",
        24: "Read FIFO Queue",
        43: "Encapsulated Interface Transport",
    }

    # Exception codes
    EXCEPTION_CODES: ClassVar[dict[int, str]] = {
        1: "Illegal Function",
        2: "Illegal Data Address",
        3: "Illegal Data Value",
        4: "Server Device Failure",
        5: "Acknowledge",
        6: "Server Device Busy",
        8: "Memory Parity Error",
        10: "Gateway Path Unavailable",
        11: "Gateway Target Device Failed to Respond",
    }

    def __init__(self) -> None:
        """Initialize Modbus analyzer."""
        self.messages: list[ModbusMessage] = []
        self.devices: dict[int, ModbusDevice] = {}

    def parse_rtu(self, data: bytes, timestamp: float = 0.0) -> ModbusMessage:
        """Parse Modbus RTU frame.

        RTU Frame Format:
        - Slave Address (1 byte)
        - Function Code (1 byte)
        - Data (N bytes, function-specific)
        - CRC-16 (2 bytes, little-endian)

        Args:
            data: Complete RTU frame including CRC.
            timestamp: Message timestamp in seconds.

        Returns:
            Parsed Modbus message.

        Raises:
            ValueError: If frame is invalid.

        Example:
            >>> analyzer = ModbusAnalyzer()
            >>> frame = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A, 0xCD, 0xC5])
            >>> msg = analyzer.parse_rtu(frame)
            >>> assert msg.unit_id == 1
            >>> assert msg.function_code == 3
            >>> assert msg.crc_valid is True
        """
        if len(data) < 4:  # Minimum: Address + FC + CRC
            raise ValueError(f"RTU frame too short: {len(data)} bytes (minimum 4)")

        # Verify CRC
        crc_valid = verify_crc(data)

        unit_id = data[0]
        function_code = data[1]
        function_data = data[2:-2]

        # Check for exception response (high bit set in function code)
        is_exception = bool(function_code & 0x80)
        exception_code = None
        parsed_data: dict[str, Any] = {}

        if is_exception:
            actual_fc = function_code & 0x7F
            if len(function_data) > 0:
                exception_code = function_data[0]
                parsed_data = {
                    "exception": self.EXCEPTION_CODES.get(exception_code, "Unknown Exception")
                }
        else:
            actual_fc = function_code
            try:
                # Parse function-specific data
                parsed_data = self._parse_function(actual_fc, function_data, is_request=True)
            except ValueError as e:
                parsed_data = {"parse_error": str(e)}

        message = ModbusMessage(
            timestamp=timestamp,
            variant="RTU",
            is_request=True,  # Determined by context in real usage
            unit_id=unit_id,
            function_code=actual_fc,
            function_name=self.FUNCTION_CODES.get(actual_fc, f"Unknown (0x{actual_fc:02X})"),
            data=function_data,
            exception_code=exception_code,
            parsed_data=parsed_data,
            crc_valid=crc_valid,
        )

        self.messages.append(message)
        self.update_device_state(message)

        return message

    def parse_tcp(self, data: bytes, timestamp: float = 0.0) -> ModbusMessage:
        """Parse Modbus TCP frame.

        TCP Frame Format (MBAP Header + PDU):
        - Transaction ID (2 bytes, big-endian)
        - Protocol ID (2 bytes, always 0x0000)
        - Length (2 bytes, big-endian, remaining bytes)
        - Unit ID (1 byte)
        - Function Code (1 byte)
        - Data (N bytes, function-specific)

        Args:
            data: Complete TCP frame including MBAP header.
            timestamp: Message timestamp in seconds.

        Returns:
            Parsed Modbus message.

        Raises:
            ValueError: If frame is invalid.

        Example:
            >>> analyzer = ModbusAnalyzer()
            >>> # Read Holding Registers request
            >>> frame = bytes([0x00, 0x01, 0x00, 0x00, 0x00, 0x06,
            ...                0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
            >>> msg = analyzer.parse_tcp(frame)
            >>> assert msg.transaction_id == 1
            >>> assert msg.function_code == 3
        """
        if len(data) < 8:  # Minimum: MBAP (7) + FC (1)
            raise ValueError(f"TCP frame too short: {len(data)} bytes (minimum 8)")

        # Parse MBAP header
        transaction_id = int.from_bytes(data[0:2], "big")
        protocol_id = int.from_bytes(data[2:4], "big")
        length = int.from_bytes(data[4:6], "big")
        unit_id = data[6]
        function_code = data[7]
        function_data = data[8:]

        if protocol_id != 0:
            raise ValueError(f"Invalid Modbus TCP protocol ID: {protocol_id} (expected 0)")

        # Verify length field
        expected_length = len(data) - 6  # Everything after protocol ID and length
        if length != expected_length:
            raise ValueError(f"Length mismatch: {length} != {expected_length}")

        # Check for exception response
        is_exception = bool(function_code & 0x80)
        exception_code = None
        parsed_data: dict[str, Any] = {}

        if is_exception:
            actual_fc = function_code & 0x7F
            if len(function_data) > 0:
                exception_code = function_data[0]
                parsed_data = {
                    "exception": self.EXCEPTION_CODES.get(exception_code, "Unknown Exception")
                }
        else:
            actual_fc = function_code
            try:
                parsed_data = self._parse_function(actual_fc, function_data, is_request=True)
            except ValueError as e:
                parsed_data = {"parse_error": str(e)}

        message = ModbusMessage(
            timestamp=timestamp,
            variant="TCP",
            is_request=True,
            transaction_id=transaction_id,
            unit_id=unit_id,
            function_code=actual_fc,
            function_name=self.FUNCTION_CODES.get(actual_fc, f"Unknown (0x{actual_fc:02X})"),
            data=function_data,
            exception_code=exception_code,
            parsed_data=parsed_data,
        )

        self.messages.append(message)
        self.update_device_state(message)

        return message

    def _parse_function(self, function_code: int, data: bytes, is_request: bool) -> dict[str, Any]:
        """Parse function-specific data.

        Args:
            function_code: Modbus function code.
            data: Function data bytes.
            is_request: True if request, False if response.

        Returns:
            Parsed function data.

        Raises:
            ValueError: If parsing fails.
        """
        # Read operations (1-4)
        if function_code in (1, 2, 3, 4):
            return self._parse_read_function(function_code, data, is_request)

        # Write operations (5, 6, 15, 16)
        if function_code in (5, 6, 15, 16):
            return self._parse_write_function(function_code, data, is_request)

        # Unsupported function codes
        return {"raw_data": data.hex()}

    def _parse_read_function(
        self, function_code: int, data: bytes, is_request: bool
    ) -> dict[str, Any]:
        """Parse read function codes (1-4).

        Args:
            function_code: Function code (1-4).
            data: Function data bytes.
            is_request: True if request, False if response.

        Returns:
            Parsed function data.
        """
        parsers = {
            1: (parse_read_coils_request, parse_read_coils_response),
            2: (parse_read_discrete_inputs_request, parse_read_discrete_inputs_response),
            3: (parse_read_holding_registers_request, parse_read_holding_registers_response),
            4: (parse_read_input_registers_request, parse_read_input_registers_response),
        }

        request_parser, response_parser = parsers[function_code]
        return request_parser(data) if is_request else response_parser(data)

    def _parse_write_function(
        self, function_code: int, data: bytes, is_request: bool
    ) -> dict[str, Any]:
        """Parse write function codes (5, 6, 15, 16).

        Args:
            function_code: Function code (5, 6, 15, or 16).
            data: Function data bytes.
            is_request: True if request, False if response.

        Returns:
            Parsed function data.
        """
        # Single write operations (no request/response distinction)
        if function_code == 5:
            return parse_write_single_coil(data)
        if function_code == 6:
            return parse_write_single_register(data)

        # Multiple write operations
        parsers = {
            15: (parse_write_multiple_coils_request, parse_write_multiple_coils_response),
            16: (parse_write_multiple_registers_request, parse_write_multiple_registers_response),
        }

        request_parser, response_parser = parsers[function_code]
        return request_parser(data) if is_request else response_parser(data)

    def update_device_state(self, message: ModbusMessage) -> None:
        """Update device state based on message.

        Tracks coil and register values observed in Modbus communication.

        Args:
            message: Parsed Modbus message.
        """
        device = self._ensure_device_exists(message.unit_id)
        device.function_codes_seen.add(message.function_code)

        # Don't update state for exceptions
        if message.exception_code is not None:
            return

        # Dispatch to function-specific handlers
        self._update_state_by_function_code(device, message)

    def _ensure_device_exists(self, unit_id: int) -> ModbusDevice:
        """Ensure device exists in registry.

        Args:
            unit_id: Modbus unit ID.

        Returns:
            ModbusDevice instance.
        """
        if unit_id not in self.devices:
            self.devices[unit_id] = ModbusDevice(unit_id=unit_id)
        return self.devices[unit_id]

    def _update_state_by_function_code(self, device: ModbusDevice, message: ModbusMessage) -> None:
        """Dispatch state update based on function code.

        Args:
            device: Device to update.
            message: Modbus message.
        """
        parsed = message.parsed_data
        fc = message.function_code

        # Coil operations
        if fc == 5:
            self._update_single_coil(device, parsed)

        # Register operations
        elif fc == 6:
            self._update_single_register(device, parsed)
        elif fc == 16:
            self._update_multiple_registers(device, parsed)

    def _update_single_coil(self, device: ModbusDevice, parsed: dict[str, Any]) -> None:
        """Update single coil value.

        Args:
            device: Device to update.
            parsed: Parsed message data.
        """
        if "output_address" in parsed and "coil_state" in parsed:
            device.coils[parsed["output_address"]] = parsed["coil_state"]

    def _update_single_register(self, device: ModbusDevice, parsed: dict[str, Any]) -> None:
        """Update single register value.

        Args:
            device: Device to update.
            parsed: Parsed message data.
        """
        if "register_address" in parsed and "register_value" in parsed:
            device.holding_registers[parsed["register_address"]] = parsed["register_value"]

    def _update_multiple_registers(self, device: ModbusDevice, parsed: dict[str, Any]) -> None:
        """Update multiple register values.

        Args:
            device: Device to update.
            parsed: Parsed message data.
        """
        if "starting_address" in parsed and "registers" in parsed:
            start_addr = parsed["starting_address"]
            for i, value in enumerate(parsed["registers"]):
                device.holding_registers[start_addr + i] = value

    def export_register_map(self, output_path: Path) -> None:
        """Export register map for all devices as JSON.

        Args:
            output_path: Path to output JSON file.

        Example:
            >>> analyzer = ModbusAnalyzer()
            >>> # ... parse messages ...
            >>> analyzer.export_register_map(Path("modbus_registers.json"))
        """
        export_data = {
            "devices": [
                {
                    "unit_id": device.unit_id,
                    "function_codes": sorted(device.function_codes_seen),
                    "coils": {str(k): v for k, v in sorted(device.coils.items())},
                    "discrete_inputs": {
                        str(k): v for k, v in sorted(device.discrete_inputs.items())
                    },
                    "holding_registers": {
                        str(k): v for k, v in sorted(device.holding_registers.items())
                    },
                    "input_registers": {
                        str(k): v for k, v in sorted(device.input_registers.items())
                    },
                }
                for device in self.devices.values()
            ],
            "message_count": len(self.messages),
        }

        with output_path.open("w") as f:
            json.dump(export_data, f, indent=2)


__all__ = ["ModbusAnalyzer", "ModbusDevice", "ModbusMessage"]
