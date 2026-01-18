"""Tests for J1939 protocol decoder.

This module tests J1939 (SAE J1939) protocol decoding including PGN extraction,
PDU format detection, and message component parsing.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

import pytest

from oscura.automotive.can.models import CANMessage
from oscura.automotive.j1939.decoder import J1939Decoder, J1939Message, extract_pgn


class TestExtractPGN:
    """Tests for PGN extraction from 29-bit CAN IDs."""

    def test_extract_pgn_pdu1_format(self):
        """Test PGN extraction for PDU1 format (destination-specific).

        PDU1: PDU Format < 240, PDU Specific is destination address.
        """
        # Create 29-bit ID with:
        # Priority: 3
        # Data Page: 0
        # PDU Format: 239 (< 240, so PDU1)
        # PDU Specific (destination): 0x10
        # Source: 0x20
        can_id = (3 << 26) | (0 << 24) | (239 << 16) | (0x10 << 8) | 0x20

        pgn, priority, dest, source = extract_pgn(can_id)

        assert priority == 3
        assert dest == 0x10
        assert source == 0x20
        # PGN for PDU1: data_page << 16 | pdu_format << 8
        assert pgn == (0 << 16) | (239 << 8)

    def test_extract_pgn_pdu2_format(self):
        """Test PGN extraction for PDU2 format (broadcast).

        PDU2: PDU Format >= 240, PDU Specific is group extension.
        """
        # Create 29-bit ID with:
        # Priority: 6
        # Data Page: 0
        # PDU Format: 240 (>= 240, so PDU2)
        # PDU Specific (group extension): 0xEE
        # Source: 0x33
        can_id = (6 << 26) | (0 << 24) | (240 << 16) | (0xEE << 8) | 0x33

        pgn, priority, dest, source = extract_pgn(can_id)

        assert priority == 6
        assert dest == 0xFF  # Broadcast
        assert source == 0x33
        # PGN for PDU2: data_page << 16 | pdu_format << 8 | pdu_specific
        assert pgn == (0 << 16) | (240 << 8) | 0xEE

    def test_extract_pgn_with_data_page_set(self):
        """Test PGN extraction with data page bit set."""
        # Data Page: 1
        # PDU Format: 250
        # PDU Specific: 0x04
        can_id = (5 << 26) | (1 << 24) | (250 << 16) | (0x04 << 8) | 0x50

        pgn, priority, dest, source = extract_pgn(can_id)

        # PGN includes data page
        assert pgn == (1 << 16) | (250 << 8) | 0x04
        assert priority == 5
        assert source == 0x50

    def test_extract_pgn_common_eec1(self):
        """Test extraction of common PGN 0xF004 (EEC1)."""
        # EEC1: PGN = 0xF004
        # Data Page: 0, PDU Format: 0xF0 (240), PDU Specific: 0x04
        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x04 << 8) | 0x00

        pgn, priority, dest, source = extract_pgn(can_id)

        assert pgn == 0xF004
        assert dest == 0xFF  # Broadcast (PDU2)

    def test_extract_pgn_priority_range(self):
        """Test that priority is correctly extracted (0-7)."""
        for prio in range(8):
            can_id = (prio << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | 0x00
            pgn, priority, dest, source = extract_pgn(can_id)
            assert priority == prio

    def test_extract_pgn_source_range(self):
        """Test extraction of various source addresses."""
        for src in [0x00, 0x01, 0x50, 0xFE, 0xFF]:
            can_id = (6 << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | src
            pgn, priority, dest, source = extract_pgn(can_id)
            assert source == src

    def test_extract_pgn_boundary_pdu_format_239(self):
        """Test boundary case: PDU Format = 239 (last PDU1)."""
        can_id = (3 << 26) | (0 << 24) | (239 << 16) | (0x50 << 8) | 0x20

        pgn, priority, dest, source = extract_pgn(can_id)

        # PDU1 format
        assert dest == 0x50  # Specific destination
        assert pgn == (0 << 16) | (239 << 8)

    def test_extract_pgn_boundary_pdu_format_240(self):
        """Test boundary case: PDU Format = 240 (first PDU2)."""
        can_id = (3 << 26) | (0 << 24) | (240 << 16) | (0x50 << 8) | 0x20

        pgn, priority, dest, source = extract_pgn(can_id)

        # PDU2 format
        assert dest == 0xFF  # Broadcast
        assert pgn == (0 << 16) | (240 << 8) | 0x50


class TestIsJ1939:
    """Tests for J1939 message identification."""

    def test_is_j1939_with_extended_id(self):
        """Test that extended ID messages are identified as J1939."""
        msg = CANMessage(arbitration_id=0x18EF1234, timestamp=1.0, data=bytes(8), is_extended=True)
        assert J1939Decoder.is_j1939(msg) is True

    def test_is_j1939_with_standard_id(self):
        """Test that standard ID messages are not identified as J1939."""
        msg = CANMessage(arbitration_id=0x123, timestamp=1.0, data=bytes(8), is_extended=False)
        assert J1939Decoder.is_j1939(msg) is False


class TestDecode:
    """Tests for J1939 message decoding."""

    def test_decode_basic_message(self):
        """Test decoding basic J1939 message."""
        # Create extended ID message
        # Priority: 6, Data Page: 0, PDU Format: 240, PDU Specific: 0x04, Source: 0x00
        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x04 << 8) | 0x00
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
        msg = CANMessage(arbitration_id=can_id, timestamp=1.5, data=data, is_extended=True)

        result = J1939Decoder.decode(msg)

        assert isinstance(result, J1939Message)
        assert result.pgn == 0xF004
        assert result.priority == 6
        assert result.source_address == 0x00
        assert result.destination_address == 0xFF  # Broadcast
        assert result.data == data
        assert result.timestamp == 1.5

    def test_decode_pdu1_message(self):
        """Test decoding PDU1 (destination-specific) message."""
        # PDU Format < 240
        can_id = (3 << 26) | (0 << 24) | (200 << 16) | (0x25 << 8) | 0x50
        data = bytes([0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x11, 0x22])
        msg = CANMessage(arbitration_id=can_id, timestamp=2.0, data=data, is_extended=True)

        result = J1939Decoder.decode(msg)

        assert result.destination_address == 0x25  # Specific destination
        assert result.source_address == 0x50

    def test_decode_with_standard_id_raises_error(self):
        """Test that decoding standard ID message raises ValueError."""
        msg = CANMessage(arbitration_id=0x123, timestamp=1.0, data=bytes(8), is_extended=False)

        with pytest.raises(ValueError, match="J1939 requires extended"):
            J1939Decoder.decode(msg)

    def test_decode_eec1_message(self):
        """Test decoding common EEC1 message (engine controller)."""
        # EEC1 PGN = 0xF004
        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x04 << 8) | 0x00
        data = bytes([0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF])
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)

        result = J1939Decoder.decode(msg)

        assert result.pgn == 0xF004

    def test_decode_preserves_data(self):
        """Test that decoding preserves message data unchanged."""
        can_id = (6 << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | 0x00
        original_data = bytes([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0])
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=original_data, is_extended=True)

        result = J1939Decoder.decode(msg)

        assert result.data == original_data

    def test_decode_variable_length_data(self):
        """Test decoding messages with variable data lengths."""
        can_id = (6 << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | 0x00

        # Short message
        short_data = bytes([0x01, 0x02])
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=short_data, is_extended=True)
        result = J1939Decoder.decode(msg)
        assert len(result.data) == 2

        # Full message
        full_data = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08])
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=full_data, is_extended=True)
        result = J1939Decoder.decode(msg)
        assert len(result.data) == 8


class TestGetPGNName:
    """Tests for PGN name lookup."""

    def test_get_known_pgn_names(self):
        """Test retrieving names for known PGNs."""
        assert "Electronic Engine Controller 1" in J1939Decoder.get_pgn_name(0xF004)
        assert "Electronic Engine Controller 2" in J1939Decoder.get_pgn_name(0xF003)
        assert "Cruise Control" in J1939Decoder.get_pgn_name(0xFEF1)
        assert "Fuel Economy" in J1939Decoder.get_pgn_name(0xFEF2)

    def test_get_unknown_pgn_name(self):
        """Test that unknown PGN returns hex string."""
        name = J1939Decoder.get_pgn_name(0x12345)
        assert name == "PGN_0x12345" or "0x12345" in name

    def test_pgn_names_dictionary(self):
        """Test that PGN_NAMES contains expected entries."""
        assert 0xF004 in J1939Decoder.PGN_NAMES
        assert 0xF003 in J1939Decoder.PGN_NAMES
        assert 0xFEF1 in J1939Decoder.PGN_NAMES
        assert 0xFEEE in J1939Decoder.PGN_NAMES  # Engine Temperature
        assert 0xFEEF in J1939Decoder.PGN_NAMES  # Engine Fluid Level/Pressure


class TestCommonPGNs:
    """Tests for decoding common J1939 PGNs."""

    def test_decode_eec2(self):
        """Test decoding EEC2 message."""
        # EEC2 PGN = 0xF003
        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x03 << 8) | 0x00
        data = bytes(8)
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)

        result = J1939Decoder.decode(msg)
        assert result.pgn == 0xF003

    def test_decode_cruise_control(self):
        """Test decoding cruise control/vehicle speed message."""
        # PGN = 0xFEF1
        can_id = (6 << 26) | (0 << 24) | (0xFE << 16) | (0xF1 << 8) | 0x00
        data = bytes(8)
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)

        result = J1939Decoder.decode(msg)
        assert result.pgn == 0xFEF1

    def test_decode_engine_temperature(self):
        """Test decoding engine temperature message."""
        # PGN = 0xFEEE
        can_id = (6 << 26) | (0 << 24) | (0xFE << 16) | (0xEE << 8) | 0x00
        data = bytes(8)
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)

        result = J1939Decoder.decode(msg)
        assert result.pgn == 0xFEEE


class TestPriorityLevels:
    """Tests for J1939 priority handling."""

    def test_decode_different_priorities(self):
        """Test decoding messages with different priority levels."""
        # J1939 priority: 0 = highest, 7 = lowest
        for priority in range(8):
            can_id = (priority << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | 0x00
            msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)

            result = J1939Decoder.decode(msg)
            assert result.priority == priority

    def test_high_priority_message(self):
        """Test decoding high-priority message (priority 0)."""
        can_id = (0 << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | 0x00
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)

        result = J1939Decoder.decode(msg)
        assert result.priority == 0  # Highest priority

    def test_low_priority_message(self):
        """Test decoding low-priority message (priority 7)."""
        can_id = (7 << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | 0x00
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)

        result = J1939Decoder.decode(msg)
        assert result.priority == 7  # Lowest priority


class TestSourceDestinationAddresses:
    """Tests for source and destination address handling."""

    def test_broadcast_destination(self):
        """Test that PDU2 messages have broadcast destination."""
        # PDU Format >= 240
        can_id = (6 << 26) | (0 << 24) | (250 << 16) | (0x00 << 8) | 0x55
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)

        result = J1939Decoder.decode(msg)
        assert result.destination_address == 0xFF  # Broadcast

    def test_specific_destination(self):
        """Test that PDU1 messages have specific destination."""
        # PDU Format < 240
        can_id = (6 << 26) | (0 << 24) | (200 << 16) | (0x42 << 8) | 0x55
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)

        result = J1939Decoder.decode(msg)
        assert result.destination_address == 0x42  # Specific destination

    def test_various_source_addresses(self):
        """Test decoding messages from various source addresses."""
        for source_addr in [0x00, 0x10, 0x50, 0xFE]:
            can_id = (6 << 26) | (0 << 24) | (240 << 16) | (0x00 << 8) | source_addr
            msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)

            result = J1939Decoder.decode(msg)
            assert result.source_address == source_addr


class TestJ1939MessageDataclass:
    """Tests for J1939Message dataclass."""

    def test_j1939_message_attributes(self):
        """Test that J1939Message has expected attributes."""
        msg = J1939Message(
            pgn=0xF004,
            priority=6,
            source_address=0x00,
            destination_address=0xFF,
            data=bytes([0x01, 0x02]),
            timestamp=1.5,
        )

        assert msg.pgn == 0xF004
        assert msg.priority == 6
        assert msg.source_address == 0x00
        assert msg.destination_address == 0xFF
        assert msg.data == bytes([0x01, 0x02])
        assert msg.timestamp == 1.5


class TestExpandedPGNSupport:
    """Tests for expanded PGN support (100+ PGNs)."""

    def test_pgn_count_exceeds_100(self):
        """Test that decoder supports 100+ PGNs."""
        assert len(J1939Decoder.PGN_NAMES) >= 100

    def test_engine_parameter_pgns(self):
        """Test engine parameter PGNs are defined."""
        # Core engine PGNs
        assert 0xF000 in J1939Decoder.PGN_NAMES  # ERC1
        assert 0xF001 in J1939Decoder.PGN_NAMES  # EBC1
        assert 0xF002 in J1939Decoder.PGN_NAMES  # ETC1
        assert 0xF003 in J1939Decoder.PGN_NAMES  # EEC2
        assert 0xF004 in J1939Decoder.PGN_NAMES  # EEC1
        assert 0xF005 in J1939Decoder.PGN_NAMES  # ETC2

    def test_transmission_pgns(self):
        """Test transmission and drivetrain PGNs."""
        assert 0xFEC0 in J1939Decoder.PGN_NAMES  # Transmission Configuration
        assert 0xFEC1 in J1939Decoder.PGN_NAMES  # High Resolution Vehicle Distance
        assert 0xFEF8 in J1939Decoder.PGN_NAMES  # Transmission Fluids 1

    def test_diagnostic_pgns(self):
        """Test diagnostic message PGNs."""
        assert 0xFECA in J1939Decoder.PGN_NAMES  # DM1
        assert 0xFECB in J1939Decoder.PGN_NAMES  # DM2
        assert 0xFECE in J1939Decoder.PGN_NAMES  # DM5
        assert 0xFED3 in J1939Decoder.PGN_NAMES  # DM11
        assert 0xFED4 in J1939Decoder.PGN_NAMES  # DM12
        assert 0xFED5 in J1939Decoder.PGN_NAMES  # DM13

    def test_aftertreatment_pgns(self):
        """Test aftertreatment system PGNs."""
        assert 0xFE40 in J1939Decoder.PGN_NAMES  # AT1 SCR Exhaust Gas Temp
        assert 0xFE42 in J1939Decoder.PGN_NAMES  # AT1 Intake NOx
        assert 0xFE43 in J1939Decoder.PGN_NAMES  # AT1 Outlet NOx
        assert 0xFE56 in J1939Decoder.PGN_NAMES  # AT1 DEF Tank Info
        assert 0xFEC7 in J1939Decoder.PGN_NAMES  # AT1 DPF 1

    def test_brake_and_wheel_pgns(self):
        """Test brake and wheel system PGNs."""
        assert 0xFE80 in J1939Decoder.PGN_NAMES  # Tire Condition
        assert 0xFE81 in J1939Decoder.PGN_NAMES  # Tire Pressure
        assert 0xFEBF in J1939Decoder.PGN_NAMES  # Wheel Speed Information

    def test_decimal_pgn_compatibility(self):
        """Test that decimal PGN values work (same as hex)."""
        # Verify decimal and hex values refer to same PGN
        assert J1939Decoder.get_pgn_name(65262) == J1939Decoder.get_pgn_name(0xFEEE)
        assert J1939Decoder.get_pgn_name(65265) == J1939Decoder.get_pgn_name(0xFEF1)
        assert J1939Decoder.get_pgn_name(65226) == J1939Decoder.get_pgn_name(0xFECA)

    def test_decode_new_pgn_messages(self):
        """Test decoding messages with newly added PGNs."""
        # Test ERC1 (0xF000)
        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x00 << 8) | 0x00
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)
        result = J1939Decoder.decode(msg)
        assert result.pgn == 0xF000
        assert "Retarder" in J1939Decoder.get_pgn_name(result.pgn)

        # Test Transmission Configuration (0xFEC0)
        can_id = (6 << 26) | (0 << 24) | (0xFE << 16) | (0xC0 << 8) | 0x00
        msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)
        result = J1939Decoder.decode(msg)
        assert result.pgn == 0xFEC0
        assert "Transmission" in J1939Decoder.get_pgn_name(result.pgn)


class TestSignalExtraction:
    """Tests for signal extraction from J1939 messages."""

    def test_extract_signal_basic(self):
        """Test extracting a simple 8-bit signal."""
        data = bytes([0x00, 0xFF, 0x42, 0x00])
        value = J1939Decoder.extract_signal(data, byte_pos=1, bit_pos=0, length_bits=8)
        assert value == 0xFF

    def test_extract_signal_with_bit_offset(self):
        """Test extracting signal with bit offset."""
        # Byte = 0b11110000, extract 4 bits starting at bit 4
        data = bytes([0xF0])
        value = J1939Decoder.extract_signal(data, byte_pos=0, bit_pos=4, length_bits=4)
        assert value == 0x0F

    def test_extract_signal_16_bit(self):
        """Test extracting 16-bit signal across two bytes."""
        # Little-endian: 0x1234 = [0x34, 0x12]
        data = bytes([0x00, 0x00, 0x34, 0x12])
        value = J1939Decoder.extract_signal(data, byte_pos=2, bit_pos=0, length_bits=16)
        assert value == 0x1234

    def test_extract_signal_32_bit(self):
        """Test extracting 32-bit signal."""
        # Little-endian: 0x12345678 = [0x78, 0x56, 0x34, 0x12]
        data = bytes([0x78, 0x56, 0x34, 0x12, 0x00, 0x00])
        value = J1939Decoder.extract_signal(data, byte_pos=0, bit_pos=0, length_bits=32)
        assert value == 0x12345678

    def test_extract_signal_out_of_bounds(self):
        """Test extracting signal beyond data length."""
        data = bytes([0x01, 0x02])
        value = J1939Decoder.extract_signal(data, byte_pos=5, bit_pos=0, length_bits=8)
        assert value == 0

    def test_decode_eec1_engine_speed(self):
        """Test decoding engine speed from EEC1 message."""
        # Create EEC1 message with engine speed = 1600 rpm
        # Engine speed in bytes 3-4, scale 0.125, offset 0
        # Raw value = 1600 / 0.125 = 12800 = 0x3200
        data = bytes([0x00, 0x00, 0x00, 0x00, 0x32, 0x00, 0x00, 0x00])

        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x04 << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signal = J1939Decoder.decode_signal(j1939_msg, "engine_speed")
        assert signal["value"] == pytest.approx(1600.0, rel=0.1)
        assert signal["unit"] == "rpm"
        assert signal["raw"] == 12800

    def test_decode_ccvs1_vehicle_speed(self):
        """Test decoding vehicle speed from CCVS1 message."""
        # Vehicle speed in bytes 1-2, scale 1/256, offset 0
        # Speed = 100 km/h -> raw = 100 * 256 = 25600 = 0x6400
        data = bytes([0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00])

        can_id = (6 << 26) | (0 << 24) | (0xFE << 16) | (0xF1 << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signal = J1939Decoder.decode_signal(j1939_msg, "wheel_based_speed")
        assert signal["value"] == pytest.approx(100.0, rel=0.1)
        assert signal["unit"] == "km/h"

    def test_decode_engine_temperature(self):
        """Test decoding coolant temperature from Engine Temperature 1."""
        # Coolant temp in byte 0, scale 1, offset -40
        # Temperature = 90°C -> raw = 90 + 40 = 130 = 0x82
        data = bytes([0x82, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

        can_id = (6 << 26) | (0 << 24) | (0xFE << 16) | (0xEE << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signal = J1939Decoder.decode_signal(j1939_msg, "coolant_temperature")
        assert signal["value"] == 90
        assert signal["unit"] == "°C"
        assert signal["raw"] == 130

    def test_decode_all_signals_eec1(self):
        """Test decoding all signals from EEC1 message."""
        # Create EEC1 with multiple signals
        data = bytes([0x00, 0x7D, 0x7D, 0x00, 0x32, 0x00, 0x00, 0x00])

        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x04 << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signals = J1939Decoder.decode_all_signals(j1939_msg)

        assert "engine_speed" in signals
        assert "driver_demand_torque" in signals
        assert "actual_engine_torque" in signals
        assert signals["engine_speed"]["value"] == pytest.approx(1600.0, rel=0.1)

    def test_decode_signal_unknown_pgn(self):
        """Test decoding signal from PGN with no signal definitions."""
        can_id = (6 << 26) | (0 << 24) | (0xFF << 16) | (0xFF << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signal = J1939Decoder.decode_signal(j1939_msg, "nonexistent_signal")
        assert signal["value"] is None
        assert signal["unit"] is None
        assert signal["raw"] is None

    def test_decode_signal_unknown_signal_name(self):
        """Test decoding unknown signal from known PGN."""
        can_id = (6 << 26) | (0 << 24) | (0xF0 << 16) | (0x04 << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=bytes(8), is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signal = J1939Decoder.decode_signal(j1939_msg, "nonexistent_signal")
        assert signal["value"] is None

    def test_decode_fuel_economy(self):
        """Test decoding fuel economy signals."""
        # Fuel rate: bytes 0-1, scale 0.05, offset 0
        # Fuel rate = 10 L/h -> raw = 10 / 0.05 = 200 = 0xC8
        data = bytes([0xC8, 0x00, 0x00, 0x10, 0x00, 0x08, 0x00, 0x00])

        can_id = (6 << 26) | (0 << 24) | (0xFE << 16) | (0xF2 << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signal = J1939Decoder.decode_signal(j1939_msg, "fuel_rate")
        assert signal["value"] == pytest.approx(10.0, rel=0.1)
        assert signal["unit"] == "L/h"

    def test_decode_ambient_conditions(self):
        """Test decoding ambient condition signals."""
        # Barometric pressure: byte 0, scale 0.5, offset 0
        # Pressure = 101 kPa -> raw = 101 / 0.5 = 202 = 0xCA
        data = bytes([0xCA, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])

        can_id = (6 << 26) | (0 << 24) | (0xFE << 16) | (0xF5 << 8) | 0x00
        can_msg = CANMessage(arbitration_id=can_id, timestamp=1.0, data=data, is_extended=True)
        j1939_msg = J1939Decoder.decode(can_msg)

        signal = J1939Decoder.decode_signal(j1939_msg, "barometric_pressure")
        assert signal["value"] == pytest.approx(101.0, rel=0.1)
        assert signal["unit"] == "kPa"


class TestPGNSignalDefinitions:
    """Tests for PGN signal definition completeness."""

    def test_pgn_signals_defined(self):
        """Test that signal definitions exist for key PGNs."""
        assert 0xF004 in J1939Decoder.PGN_SIGNALS  # EEC1
        assert 0xF003 in J1939Decoder.PGN_SIGNALS  # EEC2
        assert 0xFEF1 in J1939Decoder.PGN_SIGNALS  # CCVS1
        assert 0xFEEE in J1939Decoder.PGN_SIGNALS  # Engine Temperature
        assert 0xFEEF in J1939Decoder.PGN_SIGNALS  # Engine Fluid Level/Pressure
        assert 0xFEF2 in J1939Decoder.PGN_SIGNALS  # Fuel Economy
        assert 0xFEF5 in J1939Decoder.PGN_SIGNALS  # Ambient Conditions

    def test_signal_definitions_have_required_fields(self):
        """Test that signal definitions have all required fields."""
        for signals in J1939Decoder.PGN_SIGNALS.values():
            for signal_name, sig_def in signals.items():
                assert "byte" in sig_def, f"{signal_name} missing 'byte'"
                assert "bit" in sig_def, f"{signal_name} missing 'bit'"
                assert "length" in sig_def, f"{signal_name} missing 'length'"
                assert "scale" in sig_def, f"{signal_name} missing 'scale'"
                assert "offset" in sig_def, f"{signal_name} missing 'offset'"
                assert "unit" in sig_def, f"{signal_name} missing 'unit'"
