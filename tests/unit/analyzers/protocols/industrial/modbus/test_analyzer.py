"""Tests for Modbus RTU/TCP protocol analyzer.

This module tests comprehensive Modbus protocol analysis including RTU/TCP
frame parsing, CRC validation, function code decoding, and device state tracking.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from oscura.analyzers.protocols.industrial.modbus import ModbusAnalyzer
from oscura.analyzers.protocols.industrial.modbus.crc import calculate_crc


class TestModbusCRC:
    """Test Modbus RTU CRC-16 calculation."""

    def test_calculate_crc_basic(self) -> None:
        """Test CRC calculation for basic frame."""
        # Read Holding Registers request: Address 1, FC 03, Start 0, Qty 10
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        crc = calculate_crc(data)
        assert crc == 0xCDC5  # Known good CRC (0xC5CD in little-endian bytes)

    def test_calculate_crc_different_frames(self) -> None:
        """Test CRC for different frame types."""
        # Write Single Register: Address 1, FC 06, Reg 0, Value 123
        data = bytes([0x01, 0x06, 0x00, 0x00, 0x00, 0x7B])
        crc = calculate_crc(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_crc_roundtrip(self) -> None:
        """Test CRC calculation and verification roundtrip."""
        from oscura.analyzers.protocols.industrial.modbus.crc import verify_crc

        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        crc = calculate_crc(data)
        # Append CRC in little-endian
        frame = data + crc.to_bytes(2, "little")
        assert verify_crc(frame) is True


class TestModbusRTU:
    """Test Modbus RTU frame parsing."""

    def test_parse_read_holding_registers_request(self) -> None:
        """Test parsing Read Holding Registers request."""
        analyzer = ModbusAnalyzer()

        # Read Holding Registers: Unit 1, FC 03, Start 0, Qty 10
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        message = analyzer.parse_rtu(frame, timestamp=1.5)

        assert message.variant == "RTU"
        assert message.unit_id == 1
        assert message.function_code == 3
        assert message.function_name == "Read Holding Registers"
        assert message.crc_valid is True
        assert message.timestamp == 1.5
        assert message.parsed_data["starting_address"] == 0
        assert message.parsed_data["quantity"] == 10

    def test_parse_read_holding_registers_response(self) -> None:
        """Test parsing Read Holding Registers response."""
        analyzer = ModbusAnalyzer()

        # Response with 2 registers: 0x1234, 0x5678
        data = bytes([0x01, 0x03, 0x04, 0x12, 0x34, 0x56, 0x78])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        # Mark as response by setting is_request appropriately
        message = analyzer.parse_rtu(frame)
        # Override for testing (normally determined by context)
        message.is_request = False
        parsed = analyzer._parse_function(3, bytes([0x04, 0x12, 0x34, 0x56, 0x78]), False)

        assert parsed["byte_count"] == 4
        assert parsed["registers"] == [0x1234, 0x5678]

    def test_parse_write_single_coil(self) -> None:
        """Test parsing Write Single Coil request."""
        analyzer = ModbusAnalyzer()

        # Write Single Coil: Unit 1, FC 05, Address 172, Value ON (0xFF00)
        data = bytes([0x01, 0x05, 0x00, 0xAC, 0xFF, 0x00])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        message = analyzer.parse_rtu(frame)

        assert message.function_code == 5
        assert message.parsed_data["output_address"] == 172
        assert message.parsed_data["output_value"] == 0xFF00
        assert message.parsed_data["coil_state"] is True

    def test_parse_write_single_register(self) -> None:
        """Test parsing Write Single Register request."""
        analyzer = ModbusAnalyzer()

        # Write Single Register: Unit 1, FC 06, Address 1, Value 3
        data = bytes([0x01, 0x06, 0x00, 0x01, 0x00, 0x03])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        message = analyzer.parse_rtu(frame)

        assert message.function_code == 6
        assert message.parsed_data["register_address"] == 1
        assert message.parsed_data["register_value"] == 3

    def test_parse_exception_response(self) -> None:
        """Test parsing Modbus exception response."""
        analyzer = ModbusAnalyzer()

        # Exception response: Unit 1, FC 83 (03 + 0x80), Exception 02 (Illegal Data Address)
        data = bytes([0x01, 0x83, 0x02])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        message = analyzer.parse_rtu(frame)

        assert message.function_code == 3  # Original function code
        assert message.exception_code == 2
        assert "Illegal Data Address" in message.parsed_data["exception"]

    def test_parse_invalid_crc(self) -> None:
        """Test handling of invalid CRC."""
        analyzer = ModbusAnalyzer()

        # Valid data with wrong CRC
        data = bytes([0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])
        frame = data + bytes([0x00, 0x00])  # Wrong CRC

        message = analyzer.parse_rtu(frame)
        assert message.crc_valid is False

    def test_parse_frame_too_short(self) -> None:
        """Test handling of frames that are too short."""
        analyzer = ModbusAnalyzer()

        with pytest.raises(ValueError, match="RTU frame too short"):
            analyzer.parse_rtu(bytes([0x01, 0x03]))


class TestModbusTCP:
    """Test Modbus TCP frame parsing."""

    def test_parse_read_holding_registers_request(self) -> None:
        """Test parsing TCP Read Holding Registers request."""
        analyzer = ModbusAnalyzer()

        # MBAP Header: Trans ID 1, Proto 0, Len 6, Unit 1
        # PDU: FC 03, Start 0, Qty 10
        frame = bytes(
            [
                0x00,
                0x01,  # Transaction ID
                0x00,
                0x00,  # Protocol ID
                0x00,
                0x06,  # Length
                0x01,  # Unit ID
                0x03,  # Function code
                0x00,
                0x00,  # Starting address
                0x00,
                0x0A,  # Quantity
            ]
        )

        message = analyzer.parse_tcp(frame, timestamp=2.0)

        assert message.variant == "TCP"
        assert message.transaction_id == 1
        assert message.unit_id == 1
        assert message.function_code == 3
        assert message.timestamp == 2.0
        assert message.parsed_data["starting_address"] == 0
        assert message.parsed_data["quantity"] == 10

    def test_parse_read_coils_response(self) -> None:
        """Test parsing TCP Read Coils response."""
        analyzer = ModbusAnalyzer()

        # Response with 2 bytes of coil data: 0b11001101, 0b00000011
        frame = bytes(
            [
                0x00,
                0x02,  # Transaction ID
                0x00,
                0x00,  # Protocol ID
                0x00,
                0x05,  # Length (5 bytes: Unit + FC + ByteCount + 2 data bytes)
                0x01,  # Unit ID
                0x01,  # Function code (Read Coils)
                0x02,  # Byte count
                0b11001101,  # Coil data byte 1
                0b00000011,  # Coil data byte 2
            ]
        )

        message = analyzer.parse_tcp(frame)
        # Parse as response
        parsed = analyzer._parse_function(1, bytes([0x02, 0b11001101, 0b00000011]), False)

        assert parsed["byte_count"] == 2
        assert len(parsed["coils"]) == 16  # 2 bytes * 8 bits
        assert parsed["coils"][0] is True  # LSB first
        assert parsed["coils"][1] is False

    def test_parse_write_multiple_registers_request(self) -> None:
        """Test parsing Write Multiple Registers request."""
        analyzer = ModbusAnalyzer()

        # Write 2 registers starting at address 0: values 10, 20
        frame = bytes(
            [
                0x00,
                0x03,  # Transaction ID
                0x00,
                0x00,  # Protocol ID
                0x00,
                0x0B,  # Length (11 bytes)
                0x01,  # Unit ID
                0x10,  # Function code (16 = Write Multiple Registers)
                0x00,
                0x00,  # Starting address
                0x00,
                0x02,  # Quantity
                0x04,  # Byte count
                0x00,
                0x0A,  # Register 1: 10
                0x00,
                0x14,  # Register 2: 20
            ]
        )

        message = analyzer.parse_tcp(frame)

        assert message.function_code == 16
        assert message.parsed_data["starting_address"] == 0
        assert message.parsed_data["quantity"] == 2
        assert message.parsed_data["registers"] == [10, 20]

    def test_parse_invalid_protocol_id(self) -> None:
        """Test handling of invalid protocol ID."""
        analyzer = ModbusAnalyzer()

        # Invalid protocol ID (should be 0x0000)
        frame = bytes([0x00, 0x01, 0xFF, 0xFF, 0x00, 0x06, 0x01, 0x03, 0x00, 0x00, 0x00, 0x0A])

        with pytest.raises(ValueError, match="Invalid Modbus TCP protocol ID"):
            analyzer.parse_tcp(frame)

    def test_parse_tcp_frame_too_short(self) -> None:
        """Test handling of TCP frames that are too short."""
        analyzer = ModbusAnalyzer()

        with pytest.raises(ValueError, match="TCP frame too short"):
            analyzer.parse_tcp(bytes([0x00, 0x01, 0x00, 0x00]))


class TestDeviceState:
    """Test device state tracking."""

    def test_device_state_write_single_coil(self) -> None:
        """Test device state updates for Write Single Coil."""
        analyzer = ModbusAnalyzer()

        # Write Single Coil: Address 172, Value ON
        data = bytes([0x01, 0x05, 0x00, 0xAC, 0xFF, 0x00])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        analyzer.parse_rtu(frame)

        assert 1 in analyzer.devices
        device = analyzer.devices[1]
        assert 5 in device.function_codes_seen
        assert device.coils[172] is True

    def test_device_state_write_single_register(self) -> None:
        """Test device state updates for Write Single Register."""
        analyzer = ModbusAnalyzer()

        # Write Single Register: Address 1, Value 999
        data = bytes([0x01, 0x06, 0x00, 0x01, 0x03, 0xE7])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        analyzer.parse_rtu(frame)

        device = analyzer.devices[1]
        assert device.holding_registers[1] == 999

    def test_device_state_write_multiple_registers(self) -> None:
        """Test device state updates for Write Multiple Registers."""
        analyzer = ModbusAnalyzer()

        # Write Multiple Registers: Start 0, values [100, 200, 300]
        frame = bytes(
            [
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x0D,  # Length
                0x02,  # Unit ID
                0x10,  # FC 16
                0x00,
                0x00,  # Start address
                0x00,
                0x03,  # Quantity
                0x06,  # Byte count
                0x00,
                0x64,  # 100
                0x00,
                0xC8,  # 200
                0x01,
                0x2C,  # 300
            ]
        )

        analyzer.parse_tcp(frame)

        device = analyzer.devices[2]
        assert device.holding_registers[0] == 100
        assert device.holding_registers[1] == 200
        assert device.holding_registers[2] == 300

    def test_multiple_devices(self) -> None:
        """Test tracking multiple devices."""
        analyzer = ModbusAnalyzer()

        # Device 1
        data1 = bytes([0x01, 0x06, 0x00, 0x01, 0x00, 0x0A])
        crc1 = calculate_crc(data1)
        analyzer.parse_rtu(data1 + crc1.to_bytes(2, "little"))

        # Device 2
        data2 = bytes([0x02, 0x06, 0x00, 0x01, 0x00, 0x14])
        crc2 = calculate_crc(data2)
        analyzer.parse_rtu(data2 + crc2.to_bytes(2, "little"))

        assert len(analyzer.devices) == 2
        assert analyzer.devices[1].holding_registers[1] == 10
        assert analyzer.devices[2].holding_registers[1] == 20


class TestRegisterMapExport:
    """Test register map export functionality."""

    def test_export_register_map(self) -> None:
        """Test exporting register map to JSON."""
        analyzer = ModbusAnalyzer()

        # Add some data
        data = bytes([0x01, 0x06, 0x00, 0x01, 0x00, 0x7B])
        crc = calculate_crc(data)
        analyzer.parse_rtu(data + crc.to_bytes(2, "little"))

        # Export to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            analyzer.export_register_map(temp_path)

            # Read and verify
            import json

            with temp_path.open() as f:
                exported = json.load(f)

            assert "devices" in exported
            assert "message_count" in exported
            assert exported["message_count"] == 1
            assert len(exported["devices"]) == 1
            assert exported["devices"][0]["unit_id"] == 1
            assert "1" in exported["devices"][0]["holding_registers"]
            assert exported["devices"][0]["holding_registers"]["1"] == 123

        finally:
            if temp_path.exists():
                temp_path.unlink()


class TestFunctionParsers:
    """Test individual function code parsers."""

    def test_parse_read_coils_request(self) -> None:
        """Test Read Coils request parser."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_read_coils_request,
        )

        data = bytes([0x00, 0x13, 0x00, 0x25])  # Start 19, Qty 37
        parsed = parse_read_coils_request(data)

        assert parsed["starting_address"] == 19
        assert parsed["quantity"] == 37

    def test_parse_write_multiple_coils_request(self) -> None:
        """Test Write Multiple Coils request parser."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_write_multiple_coils_request,
        )

        # Write 10 coils starting at 20: first byte 0b11001101
        data = bytes([0x00, 0x14, 0x00, 0x0A, 0x02, 0b11001101, 0b00000011])
        parsed = parse_write_multiple_coils_request(data)

        assert parsed["starting_address"] == 20
        assert parsed["quantity"] == 10
        assert parsed["byte_count"] == 2
        assert len(parsed["coils"]) == 10

    def test_parse_invalid_quantity_ranges(self) -> None:
        """Test validation of quantity ranges."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_read_coils_request,
            parse_read_holding_registers_request,
        )

        # Coils: quantity too large (max 2000)
        with pytest.raises(ValueError, match="Invalid quantity"):
            parse_read_coils_request(bytes([0x00, 0x00, 0x07, 0xD1]))  # 2001

        # Registers: quantity too large (max 125)
        with pytest.raises(ValueError, match="Invalid quantity"):
            parse_read_holding_registers_request(bytes([0x00, 0x00, 0x00, 0x7E]))  # 126

    def test_parse_write_multiple_coils_response(self) -> None:
        """Test Write Multiple Coils response parser."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_write_multiple_coils_response,
        )

        data = bytes([0x00, 0x13, 0x00, 0x0A])
        parsed = parse_write_multiple_coils_response(data)

        assert parsed["starting_address"] == 19
        assert parsed["quantity"] == 10

    def test_parse_write_multiple_registers_response(self) -> None:
        """Test Write Multiple Registers response parser."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_write_multiple_registers_response,
        )

        data = bytes([0x00, 0x01, 0x00, 0x02])
        parsed = parse_write_multiple_registers_response(data)

        assert parsed["starting_address"] == 1
        assert parsed["quantity"] == 2

    def test_insufficient_data_errors(self) -> None:
        """Test handling of insufficient data in parsers."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_read_coils_request,
            parse_read_coils_response,
            parse_read_holding_registers_response,
            parse_write_multiple_coils_request,
            parse_write_multiple_registers_request,
            parse_write_single_coil,
            parse_write_single_register,
        )

        # All parsers should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Insufficient"):
            parse_read_coils_request(bytes([0x00]))

        with pytest.raises(ValueError, match="Insufficient"):
            parse_read_coils_response(bytes([]))

        with pytest.raises(ValueError, match="Insufficient"):
            parse_read_holding_registers_response(bytes([0x02, 0x12]))

        with pytest.raises(ValueError, match="Insufficient"):
            parse_write_single_coil(bytes([0x00, 0xAC]))

        with pytest.raises(ValueError, match="Insufficient"):
            parse_write_single_register(bytes([0x00, 0x01]))

        with pytest.raises(ValueError, match="Insufficient"):
            parse_write_multiple_coils_request(bytes([0x00, 0x14, 0x00]))

        with pytest.raises(ValueError, match="Insufficient"):
            parse_write_multiple_registers_request(bytes([0x00, 0x01, 0x00]))

    def test_byte_count_validation(self) -> None:
        """Test validation of byte counts."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_read_holding_registers_response,
            parse_write_multiple_registers_request,
        )

        # Odd byte count (must be even for registers)
        with pytest.raises(ValueError, match="Byte count must be even"):
            parse_read_holding_registers_response(bytes([0x03, 0x12, 0x34, 0x56]))

        # Byte count mismatch
        with pytest.raises(ValueError, match="Byte count mismatch"):
            parse_write_multiple_registers_request(
                bytes([0x00, 0x01, 0x00, 0x02, 0x03, 0x12, 0x34])  # Says 2 regs but only 3 bytes
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_function_data(self) -> None:
        """Test handling of empty function data."""
        analyzer = ModbusAnalyzer()

        # Frame with no function data (will fail parsing)
        data = bytes([0x01, 0x03])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        message = analyzer.parse_rtu(frame)
        assert "parse_error" in message.parsed_data

    def test_unsupported_function_code(self) -> None:
        """Test handling of unsupported function codes."""
        analyzer = ModbusAnalyzer()

        # Function code 99 (not standard)
        data = bytes([0x01, 0x63, 0x01, 0x02, 0x03])
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        message = analyzer.parse_rtu(frame)
        assert message.function_code == 99
        assert "Unknown" in message.function_name
        assert "raw_data" in message.parsed_data

    def test_message_accumulation(self) -> None:
        """Test that messages accumulate in analyzer."""
        analyzer = ModbusAnalyzer()

        # Parse multiple messages
        for i in range(5):
            data = bytes([0x01, 0x06, 0x00, i, 0x00, i])
            crc = calculate_crc(data)
            analyzer.parse_rtu(data + crc.to_bytes(2, "little"))

        assert len(analyzer.messages) == 5

    def test_coil_value_validation(self) -> None:
        """Test validation of coil values in Write Single Coil."""
        from oscura.analyzers.protocols.industrial.modbus.functions import (
            parse_write_single_coil,
        )

        # Invalid coil value (must be 0x0000 or 0xFF00)
        with pytest.raises(ValueError, match="Invalid coil value"):
            parse_write_single_coil(bytes([0x00, 0xAC, 0x01, 0x23]))

    def test_parse_all_read_functions(self) -> None:
        """Test parsing all read function codes."""
        analyzer = ModbusAnalyzer()

        # FC 01: Read Coils
        data = bytes([0x01, 0x01, 0x00, 0x13, 0x00, 0x25])
        crc = calculate_crc(data)
        msg1 = analyzer.parse_rtu(data + crc.to_bytes(2, "little"))
        assert msg1.function_code == 1

        # FC 02: Read Discrete Inputs
        data = bytes([0x01, 0x02, 0x00, 0x00, 0x00, 0x16])
        crc = calculate_crc(data)
        msg2 = analyzer.parse_rtu(data + crc.to_bytes(2, "little"))
        assert msg2.function_code == 2

        # FC 04: Read Input Registers
        data = bytes([0x01, 0x04, 0x00, 0x08, 0x00, 0x01])
        crc = calculate_crc(data)
        msg4 = analyzer.parse_rtu(data + crc.to_bytes(2, "little"))
        assert msg4.function_code == 4

    def test_parse_write_multiple_functions(self) -> None:
        """Test parsing write multiple functions."""
        analyzer = ModbusAnalyzer()

        # FC 15: Write Multiple Coils
        data = bytes([0x01, 0x0F, 0x00, 0x13, 0x00, 0x0A, 0x02, 0xCD, 0x01])
        crc = calculate_crc(data)
        msg = analyzer.parse_rtu(data + crc.to_bytes(2, "little"))
        assert msg.function_code == 15
        assert msg.parsed_data["starting_address"] == 19
        assert msg.parsed_data["quantity"] == 10

    def test_tcp_length_validation(self) -> None:
        """Test TCP frame length validation."""
        analyzer = ModbusAnalyzer()

        # Frame with incorrect length field
        frame = bytes(
            [
                0x00,
                0x01,  # Transaction ID
                0x00,
                0x00,  # Protocol ID
                0x00,
                0xFF,  # Wrong length (should be 6)
                0x01,  # Unit ID
                0x03,  # Function code
                0x00,
                0x00,  # Starting address
                0x00,
                0x0A,  # Quantity
            ]
        )

        with pytest.raises(ValueError, match="Length mismatch"):
            analyzer.parse_tcp(frame)

    def test_exception_without_data(self) -> None:
        """Test exception response without exception code."""
        analyzer = ModbusAnalyzer()

        # Exception response with no data
        data = bytes([0x01, 0x83])  # FC 03 + 0x80, no exception code
        crc = calculate_crc(data)
        frame = data + crc.to_bytes(2, "little")

        message = analyzer.parse_rtu(frame)
        assert message.exception_code is None
