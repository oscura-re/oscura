"""Tests for J1939 protocol analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscura.automotive.j1939.analyzer import (
    J1939SPN,
    J1939Analyzer,
)


class TestJ1939Identifier:
    """Test J1939 identifier decoding."""

    def test_decode_identifier_pdu2_format(self):
        """Test decoding PDU2 format identifier (broadcast)."""
        analyzer = J1939Analyzer()
        # 0x18FEF100: Priority=6, DP=0, PF=0xFE, PS=0xF1, SA=0x00
        ident = analyzer._decode_identifier(0x18FEF100)

        assert ident.priority == 6
        assert ident.reserved == 0
        assert ident.data_page == 0
        assert ident.pdu_format == 0xFE
        assert ident.pdu_specific == 0xF1
        assert ident.source_address == 0x00
        assert ident.pgn == 65265  # CCVS1

    def test_decode_identifier_pdu1_format(self):
        """Test decoding PDU1 format identifier (destination-specific)."""
        analyzer = J1939Analyzer()
        # 0x0CEF0400: Priority=3, DP=0, PF=0xEF (239, PDU1), PS=0x04 (dest), SA=0x00
        ident = analyzer._decode_identifier(0x0CEF0400)

        assert ident.priority == 3
        assert ident.data_page == 0
        assert ident.pdu_format == 0xEF
        assert ident.pdu_specific == 0x04  # Destination address
        assert ident.source_address == 0x00
        assert ident.pgn == 0xEF00  # PGN with PS=0x00

    def test_decode_identifier_with_data_page(self):
        """Test decoding identifier with data page set."""
        analyzer = J1939Analyzer()
        # 0x19F00400: Priority=6, DP=1, PF=0xF0 (240, PDU2), PS=0x04, SA=0x00
        ident = analyzer._decode_identifier(0x19F00400)

        assert ident.data_page == 1
        assert ident.pgn == 0x1F004  # DP=1, PF=0xF0, PS=0x04

    def test_decode_identifier_priority_range(self):
        """Test all priority values (0-7)."""
        analyzer = J1939Analyzer()

        for priority in range(8):
            can_id = (priority << 26) | 0x00FEF100
            ident = analyzer._decode_identifier(can_id)
            assert ident.priority == priority

    def test_invalid_can_id(self):
        """Test error on invalid CAN ID (>29 bits)."""
        analyzer = J1939Analyzer()

        with pytest.raises(ValueError, match="Invalid 29-bit CAN ID"):
            analyzer.parse_message(0x20000000, b"\x00")


class TestPGNCalculation:
    """Test PGN calculation."""

    def test_calculate_pgn_pdu1_format(self):
        """Test PGN calculation for PDU1 format."""
        analyzer = J1939Analyzer()

        # PDU1: PF < 240, PS is destination, set to 0 for PGN
        pgn = analyzer._calculate_pgn(0xEF, 0x12, 0)
        assert pgn == 0xEF00

    def test_calculate_pgn_pdu2_format(self):
        """Test PGN calculation for PDU2 format."""
        analyzer = J1939Analyzer()

        # PDU2: PF >= 240, PS is group extension
        pgn = analyzer._calculate_pgn(0xFE, 0xF1, 0)
        assert pgn == 0xFEF1

    def test_calculate_pgn_boundary(self):
        """Test PGN calculation at PDU1/PDU2 boundary."""
        analyzer = J1939Analyzer()

        # PF=239 (0xEF): PDU1
        pgn1 = analyzer._calculate_pgn(239, 0xFF, 0)
        assert pgn1 == 0xEF00

        # PF=240 (0xF0): PDU2
        pgn2 = analyzer._calculate_pgn(240, 0xFF, 0)
        assert pgn2 == 0xF0FF

    def test_is_pdu1_format(self):
        """Test PDU1 format detection."""
        analyzer = J1939Analyzer()

        assert analyzer._is_pdu1_format(0)
        assert analyzer._is_pdu1_format(239)
        assert not analyzer._is_pdu1_format(240)
        assert not analyzer._is_pdu1_format(255)


class TestMessageParsing:
    """Test J1939 message parsing."""

    def test_parse_simple_message(self):
        """Test parsing simple single-frame message."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x0CF00400, b"\xff" * 8, timestamp=1.0)

        assert msg.timestamp == 1.0
        assert msg.can_id == 0x0CF00400
        assert msg.identifier.pgn == 61444  # EEC1
        assert msg.data == b"\xff" * 8
        assert not msg.is_transport_protocol
        assert msg.transport_info is None
        assert len(analyzer.messages) == 1

    def test_parse_message_with_known_pgn(self):
        """Test parsing message with known PGN name."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x0CF00400, b"\x00" * 8)

        assert msg.pgn_name == "Electronic Engine Controller 1"

    def test_parse_message_with_unknown_pgn(self):
        """Test parsing message with unknown PGN."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x18123456, b"\x00" * 8)

        assert msg.pgn_name is None

    def test_parse_message_validates_data_length(self):
        """Test error on data >8 bytes."""
        analyzer = J1939Analyzer()

        with pytest.raises(ValueError, match="Data too long"):
            analyzer.parse_message(0x18FEF100, b"\x00" * 9)


class TestTransportProtocol:
    """Test transport protocol parsing."""

    def test_parse_tp_cm_rts(self):
        """Test parsing TP.CM Request To Send."""
        analyzer = J1939Analyzer()

        # TP.CM RTS: 32 bytes, 5 packets, PGN 61680 (0xF0F0)
        # PGN 60160 (0xEB00) = TP.CM: CAN ID with PF=0xEB (235, PDU1)
        # PS=0xF1 (dest=0xF1), SA=0x00 -> CAN ID = 0x18EBF100
        data = b"\x10\x20\x00\x05\xff\xf0\xf0\x00"  # PGN 61680 in bytes 5-7
        msg = analyzer.parse_message(0x18EBF100, data)

        assert msg.is_transport_protocol
        assert msg.transport_info is not None
        assert msg.transport_info["type"] == "TP.CM"
        assert msg.transport_info["control"] == "RTS"
        assert msg.transport_info["total_size"] == 32
        assert msg.transport_info["total_packets"] == 5
        assert msg.transport_info["data_pgn"] == 61680

    def test_parse_tp_cm_cts(self):
        """Test parsing TP.CM Clear To Send."""
        analyzer = J1939Analyzer()

        # TP.CM CTS: 3 max packets
        # PGN 60160 (0xEB00) = TP.CM
        data = b"\x11\x00\x00\x00\x03\x00\x00\x00"
        msg = analyzer.parse_message(0x18EBF100, data)

        assert msg.transport_info["control"] == "CTS"
        assert msg.transport_info["max_packets"] == 3

    def test_parse_tp_cm_bam(self):
        """Test parsing TP.CM Broadcast Announce Message."""
        analyzer = J1939Analyzer()

        # TP.CM BAM: 14 bytes, 2 packets, PGN 61680
        # PGN 60160 (0xEB00) = TP.CM
        data = b"\x20\x0e\x00\x02\xff\xf0\x04\x00"
        msg = analyzer.parse_message(0x18EBF100, data)

        assert msg.transport_info["control"] == "BAM"
        assert msg.transport_info["total_size"] == 14
        assert msg.transport_info["total_packets"] == 2

    def test_parse_tp_cm_abort(self):
        """Test parsing TP.CM Abort."""
        analyzer = J1939Analyzer()

        # PGN 60160 (0xEB00) = TP.CM
        data = b"\xff\x00\x00\x00\x00\x00\x00\x00"
        msg = analyzer.parse_message(0x18EBF100, data)

        assert msg.transport_info["control"] == "ABORT"

    def test_parse_tp_dt(self):
        """Test parsing TP.DT data transfer."""
        analyzer = J1939Analyzer()

        # TP.DT: sequence 1, 7 bytes data
        # PGN 60416 (0xEC00) = TP.DT: CAN ID with PF=0xEC (236, PDU1)
        # PS=0xF1 (dest), SA=0x00 -> CAN ID = 0x18ECF100
        data = b"\x01\x00\x01\x02\x03\x04\x05\x06"
        msg = analyzer.parse_message(0x18ECF100, data)

        assert msg.is_transport_protocol
        assert msg.transport_info["type"] == "TP.DT"
        assert msg.transport_info["sequence"] == 1
        assert msg.transport_info["data"] == "00010203040506"

    def test_parse_tp_cm_invalid_length(self):
        """Test TP.CM with insufficient data."""
        analyzer = J1939Analyzer()

        # Only 5 bytes instead of 8 - PGN 60160 (TP.CM)
        msg = analyzer.parse_message(0x18EBF100, b"\x10\x20\x00\x05\xff")

        assert not msg.is_transport_protocol


class TestSPNDecoding:
    """Test SPN (Suspect Parameter Number) decoding."""

    def test_decode_spn_simple(self):
        """Test decoding simple SPN."""
        analyzer = J1939Analyzer()

        # Add SPN definition
        spn = J1939SPN(
            spn=190,
            name="Engine Speed",
            start_bit=24,
            bit_length=16,
            resolution=0.125,
            unit="rpm",
        )
        analyzer.add_spn_definition(61444, spn)

        # Parse message with engine speed = 16 (raw) = 2.0 rpm
        data = b"\x00\x00\x00\x10\x00\x00\x00\x00"
        msg = analyzer.parse_message(0x0CF00400, data)

        assert "Engine Speed" in msg.decoded_spns
        assert msg.decoded_spns["Engine Speed"] == 2.0

    def test_decode_spn_with_offset(self):
        """Test decoding SPN with offset."""
        analyzer = J1939Analyzer()

        spn = J1939SPN(
            spn=513,
            name="Actual Engine Percent Torque",
            start_bit=16,
            bit_length=8,
            resolution=1.0,
            offset=-125.0,
            unit="%",
        )
        analyzer.add_spn_definition(61444, spn)

        # Raw value = 150, scaled = 150 - 125 = 25%
        data = b"\x00\x00\x96\x00\x00\x00\x00\x00"
        msg = analyzer.parse_message(0x0CF00400, data)

        assert msg.decoded_spns["Actual Engine Percent Torque"] == 25.0

    def test_decode_multiple_spns(self):
        """Test decoding multiple SPNs from same message."""
        analyzer = J1939Analyzer()

        spn1 = J1939SPN(
            spn=91, name="Accel Pedal 1", start_bit=0, bit_length=8, resolution=0.4, unit="%"
        )
        spn2 = J1939SPN(
            spn=92, name="Engine Load", start_bit=8, bit_length=8, resolution=1.0, unit="%"
        )
        analyzer.add_spn_definition(61443, spn1)
        analyzer.add_spn_definition(61443, spn2)

        # Accel=100 (raw) = 40%, Load=50 (raw) = 50%
        data = b"\x64\x32\x00\x00\x00\x00\x00\x00"
        msg = analyzer.parse_message(0x0CF00300, data)

        assert msg.decoded_spns["Accel Pedal 1"] == 40.0
        assert msg.decoded_spns["Engine Load"] == 50.0

    def test_decode_spn_no_definitions(self):
        """Test decoding with no SPN definitions."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x0CF00400, b"\x00" * 8)

        assert msg.decoded_spns == {}

    def test_decode_spn_wrong_pgn(self):
        """Test SPN not decoded for wrong PGN."""
        analyzer = J1939Analyzer()

        spn = J1939SPN(spn=190, name="Engine Speed", start_bit=24, bit_length=16)
        analyzer.add_spn_definition(61444, spn)

        # Parse different PGN
        msg = analyzer.parse_message(0x18FEF100, b"\x00" * 8)

        assert msg.decoded_spns == {}


class TestMessageExport:
    """Test message export functionality."""

    def test_export_messages_json(self, tmp_path: Path):
        """Test exporting messages to JSON."""
        analyzer = J1939Analyzer()

        # Parse some messages
        analyzer.parse_message(0x0CF00400, b"\xff" * 8, timestamp=1.0)
        analyzer.parse_message(0x18FEF100, b"\x00" * 8, timestamp=2.0)

        output_file = tmp_path / "messages.json"
        analyzer.export_messages(output_file)

        assert output_file.exists()

        with output_file.open() as f:
            data = json.load(f)

        assert "messages" in data
        assert "total_messages" in data
        assert data["total_messages"] == 2
        assert len(data["messages"]) == 2

    def test_export_messages_content(self, tmp_path: Path):
        """Test exported message content."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x0CF00400, b"\x12\x34\x56\x78", timestamp=1.5)

        output_file = tmp_path / "messages.json"
        analyzer.export_messages(output_file)

        with output_file.open() as f:
            data = json.load(f)

        msg_data = data["messages"][0]
        assert msg_data["timestamp"] == 1.5
        assert msg_data["can_id"] == "0x0CF00400"
        assert msg_data["pgn"] == 61444  # EEC1
        assert msg_data["priority"] == 3
        assert msg_data["source_address"] == 0
        assert msg_data["data"] == "12345678"


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, tmp_path: Path):
        """Test complete J1939 analysis workflow."""
        analyzer = J1939Analyzer()

        # Add SPN definitions
        spn = J1939SPN(
            spn=190,
            name="Engine Speed",
            start_bit=24,
            bit_length=16,
            resolution=0.125,
            unit="rpm",
        )
        analyzer.add_spn_definition(61444, spn)

        # Parse various messages
        analyzer.parse_message(0x0CF00400, b"\x00\x00\x00\x10\x00\x00\x00\x00", 1.0)
        analyzer.parse_message(0x18FEF100, b"\x00" * 8, 2.0)
        analyzer.parse_message(0x18ECF100, b"\x20\x0e\x00\x02\xff\xf0\x04\x00", 3.0)

        # Export
        output_file = tmp_path / "workflow.json"
        analyzer.export_messages(output_file)

        assert len(analyzer.messages) == 3
        assert output_file.exists()

    def test_multiple_analyzers(self):
        """Test multiple independent analyzer instances."""
        analyzer1 = J1939Analyzer()
        analyzer2 = J1939Analyzer()

        analyzer1.parse_message(0x0CF00400, b"\x00" * 8)
        analyzer2.parse_message(0x18FEF100, b"\xff" * 8)

        assert len(analyzer1.messages) == 1
        assert len(analyzer2.messages) == 1
        assert analyzer1.messages[0].identifier.pgn != analyzer2.messages[0].identifier.pgn


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_can_id(self):
        """Test parsing CAN ID 0."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x00000000, b"\x00")

        assert msg.identifier.priority == 0
        assert msg.identifier.source_address == 0
        assert msg.identifier.pgn == 0

    def test_max_can_id(self):
        """Test parsing maximum valid 29-bit CAN ID."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x1FFFFFFF, b"\x00")

        assert msg.identifier.priority == 7
        assert msg.identifier.source_address == 0xFF

    def test_empty_data(self):
        """Test parsing message with no data."""
        analyzer = J1939Analyzer()

        msg = analyzer.parse_message(0x0CF00400, b"")

        assert msg.data == b""
        assert len(msg.data) == 0

    def test_spn_bit_extraction_boundaries(self):
        """Test SPN extraction at byte boundaries."""
        analyzer = J1939Analyzer()

        # Test SPN at bit 0
        # Use PDU2 format: PF=0xF0, PS=0x01 -> PGN=0xF001
        spn1 = J1939SPN(spn=1, name="SPN1", start_bit=0, bit_length=8)
        analyzer.add_spn_definition(0xF001, spn1)

        msg = analyzer.parse_message(0x18F00100, b"\x42\x00\x00\x00\x00\x00\x00\x00")
        assert msg.decoded_spns["SPN1"] == 66

        # Test SPN at bit 56
        analyzer2 = J1939Analyzer()
        spn2 = J1939SPN(spn=2, name="SPN2", start_bit=56, bit_length=8)
        analyzer2.add_spn_definition(0xF002, spn2)

        msg2 = analyzer2.parse_message(0x18F00200, b"\x00\x00\x00\x00\x00\x00\x00\x99")
        assert msg2.decoded_spns["SPN2"] == 153
