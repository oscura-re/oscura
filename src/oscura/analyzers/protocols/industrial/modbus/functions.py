"""Modbus function code parsers.

This module provides parsers for all standard Modbus function codes,
supporting both request and response parsing.

References:
    Modbus Application Protocol V1.1b3
    https://modbus.org/docs/Modbus_Application_Protocol_V1_1b3.pdf
"""

from __future__ import annotations

from typing import Any


def parse_read_coils_request(data: bytes) -> dict[str, Any]:
    """Parse Read Coils (FC 01) request.

    Request Format:
    - Starting Address (2 bytes, big-endian)
    - Quantity of Coils (2 bytes, big-endian, 1-2000)

    Args:
        data: Function data (excludes address and function code).

    Returns:
        Parsed request data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 4:
        raise ValueError("Insufficient data for Read Coils request")

    starting_address = int.from_bytes(data[0:2], "big")
    quantity = int.from_bytes(data[2:4], "big")

    if quantity < 1 or quantity > 2000:
        raise ValueError(f"Invalid quantity: {quantity} (must be 1-2000)")

    return {
        "starting_address": starting_address,
        "quantity": quantity,
    }


def parse_read_coils_response(data: bytes) -> dict[str, Any]:
    """Parse Read Coils (FC 01) response.

    Response Format:
    - Byte Count (1 byte)
    - Coil Status (N bytes, LSB first)

    Args:
        data: Function data.

    Returns:
        Parsed response data with coil values.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 1:
        raise ValueError("Insufficient data for Read Coils response")

    byte_count = data[0]
    if len(data) < 1 + byte_count:
        raise ValueError("Insufficient coil data")

    # Parse coil values (each byte contains 8 coils, LSB first)
    coils = []
    for i in range(byte_count):
        byte_val = data[1 + i]
        for bit in range(8):
            coils.append(bool(byte_val & (1 << bit)))

    return {
        "byte_count": byte_count,
        "coils": coils,
    }


def parse_read_discrete_inputs_request(data: bytes) -> dict[str, Any]:
    """Parse Read Discrete Inputs (FC 02) request.

    Same format as Read Coils request.

    Args:
        data: Function data.

    Returns:
        Parsed request data.
    """
    return parse_read_coils_request(data)


def parse_read_discrete_inputs_response(data: bytes) -> dict[str, Any]:
    """Parse Read Discrete Inputs (FC 02) response.

    Same format as Read Coils response.

    Args:
        data: Function data.

    Returns:
        Parsed response data.
    """
    return parse_read_coils_response(data)


def parse_read_holding_registers_request(data: bytes) -> dict[str, Any]:
    """Parse Read Holding Registers (FC 03) request.

    Request Format:
    - Starting Address (2 bytes, big-endian)
    - Quantity of Registers (2 bytes, big-endian, 1-125)

    Args:
        data: Function data.

    Returns:
        Parsed request data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 4:
        raise ValueError("Insufficient data for Read Holding Registers request")

    starting_address = int.from_bytes(data[0:2], "big")
    quantity = int.from_bytes(data[2:4], "big")

    if quantity < 1 or quantity > 125:
        raise ValueError(f"Invalid quantity: {quantity} (must be 1-125)")

    return {
        "starting_address": starting_address,
        "quantity": quantity,
    }


def parse_read_holding_registers_response(data: bytes) -> dict[str, Any]:
    """Parse Read Holding Registers (FC 03) response.

    Response Format:
    - Byte Count (1 byte)
    - Register Values (N bytes, 2 bytes per register, big-endian)

    Args:
        data: Function data.

    Returns:
        Parsed response data with register values.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 1:
        raise ValueError("Insufficient data for Read Holding Registers response")

    byte_count = data[0]
    if len(data) < 1 + byte_count:
        raise ValueError("Insufficient register data")

    if byte_count % 2 != 0:
        raise ValueError(f"Byte count must be even, got {byte_count}")

    # Parse registers (2 bytes each, big-endian)
    registers = []
    for i in range(byte_count // 2):
        offset = 1 + i * 2
        reg_value = int.from_bytes(data[offset : offset + 2], "big")
        registers.append(reg_value)

    return {
        "byte_count": byte_count,
        "registers": registers,
    }


def parse_read_input_registers_request(data: bytes) -> dict[str, Any]:
    """Parse Read Input Registers (FC 04) request.

    Same format as Read Holding Registers request.

    Args:
        data: Function data.

    Returns:
        Parsed request data.
    """
    return parse_read_holding_registers_request(data)


def parse_read_input_registers_response(data: bytes) -> dict[str, Any]:
    """Parse Read Input Registers (FC 04) response.

    Same format as Read Holding Registers response.

    Args:
        data: Function data.

    Returns:
        Parsed response data.
    """
    return parse_read_holding_registers_response(data)


def parse_write_single_coil(data: bytes) -> dict[str, Any]:
    """Parse Write Single Coil (FC 05) request/response.

    Format:
    - Output Address (2 bytes, big-endian)
    - Output Value (2 bytes, 0x0000 or 0xFF00)

    Args:
        data: Function data.

    Returns:
        Parsed data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 4:
        raise ValueError("Insufficient data for Write Single Coil")

    output_address = int.from_bytes(data[0:2], "big")
    output_value = int.from_bytes(data[2:4], "big")

    if output_value not in (0x0000, 0xFF00):
        raise ValueError(f"Invalid coil value: 0x{output_value:04X} (must be 0x0000 or 0xFF00)")

    return {
        "output_address": output_address,
        "output_value": output_value,
        "coil_state": output_value == 0xFF00,
    }


def parse_write_single_register(data: bytes) -> dict[str, Any]:
    """Parse Write Single Register (FC 06) request/response.

    Format:
    - Register Address (2 bytes, big-endian)
    - Register Value (2 bytes, big-endian)

    Args:
        data: Function data.

    Returns:
        Parsed data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 4:
        raise ValueError("Insufficient data for Write Single Register")

    register_address = int.from_bytes(data[0:2], "big")
    register_value = int.from_bytes(data[2:4], "big")

    return {
        "register_address": register_address,
        "register_value": register_value,
    }


def parse_write_multiple_coils_request(data: bytes) -> dict[str, Any]:
    """Parse Write Multiple Coils (FC 15) request.

    Request Format:
    - Starting Address (2 bytes, big-endian)
    - Quantity of Outputs (2 bytes, big-endian, 1-1968)
    - Byte Count (1 byte)
    - Output Values (N bytes)

    Args:
        data: Function data.

    Returns:
        Parsed request data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 5:
        raise ValueError("Insufficient data for Write Multiple Coils request")

    starting_address = int.from_bytes(data[0:2], "big")
    quantity = int.from_bytes(data[2:4], "big")
    byte_count = data[4]

    if quantity < 1 or quantity > 1968:
        raise ValueError(f"Invalid quantity: {quantity} (must be 1-1968)")

    if len(data) < 5 + byte_count:
        raise ValueError("Insufficient coil data")

    # Parse coil values
    coils: list[bool] = []
    for i in range(byte_count):
        byte_val = data[5 + i]
        for bit in range(8):
            if len(coils) < quantity:
                coils.append(bool(byte_val & (1 << bit)))

    return {
        "starting_address": starting_address,
        "quantity": quantity,
        "byte_count": byte_count,
        "coils": coils,
    }


def parse_write_multiple_coils_response(data: bytes) -> dict[str, Any]:
    """Parse Write Multiple Coils (FC 15) response.

    Response Format:
    - Starting Address (2 bytes, big-endian)
    - Quantity of Outputs (2 bytes, big-endian)

    Args:
        data: Function data.

    Returns:
        Parsed response data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 4:
        raise ValueError("Insufficient data for Write Multiple Coils response")

    starting_address = int.from_bytes(data[0:2], "big")
    quantity = int.from_bytes(data[2:4], "big")

    return {
        "starting_address": starting_address,
        "quantity": quantity,
    }


def parse_write_multiple_registers_request(data: bytes) -> dict[str, Any]:
    """Parse Write Multiple Registers (FC 16) request.

    Request Format:
    - Starting Address (2 bytes, big-endian)
    - Quantity of Registers (2 bytes, big-endian, 1-123)
    - Byte Count (1 byte)
    - Register Values (N bytes, 2 bytes per register)

    Args:
        data: Function data.

    Returns:
        Parsed request data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 5:
        raise ValueError("Insufficient data for Write Multiple Registers request")

    starting_address = int.from_bytes(data[0:2], "big")
    quantity = int.from_bytes(data[2:4], "big")
    byte_count = data[4]

    if quantity < 1 or quantity > 123:
        raise ValueError(f"Invalid quantity: {quantity} (must be 1-123)")

    if byte_count != quantity * 2:
        raise ValueError(f"Byte count mismatch: {byte_count} != {quantity * 2}")

    if len(data) < 5 + byte_count:
        raise ValueError("Insufficient register data")

    # Parse registers
    registers = []
    for i in range(quantity):
        offset = 5 + i * 2
        reg_value = int.from_bytes(data[offset : offset + 2], "big")
        registers.append(reg_value)

    return {
        "starting_address": starting_address,
        "quantity": quantity,
        "byte_count": byte_count,
        "registers": registers,
    }


def parse_write_multiple_registers_response(data: bytes) -> dict[str, Any]:
    """Parse Write Multiple Registers (FC 16) response.

    Response Format:
    - Starting Address (2 bytes, big-endian)
    - Quantity of Registers (2 bytes, big-endian)

    Args:
        data: Function data.

    Returns:
        Parsed response data.

    Raises:
        ValueError: If data is invalid.
    """
    if len(data) < 4:
        raise ValueError("Insufficient data for Write Multiple Registers response")

    starting_address = int.from_bytes(data[0:2], "big")
    quantity = int.from_bytes(data[2:4], "big")

    return {
        "starting_address": starting_address,
        "quantity": quantity,
    }


__all__ = [
    "parse_read_coils_request",
    "parse_read_coils_response",
    "parse_read_discrete_inputs_request",
    "parse_read_discrete_inputs_response",
    "parse_read_holding_registers_request",
    "parse_read_holding_registers_response",
    "parse_read_input_registers_request",
    "parse_read_input_registers_response",
    "parse_write_multiple_coils_request",
    "parse_write_multiple_coils_response",
    "parse_write_multiple_registers_request",
    "parse_write_multiple_registers_response",
    "parse_write_single_coil",
    "parse_write_single_register",
]
