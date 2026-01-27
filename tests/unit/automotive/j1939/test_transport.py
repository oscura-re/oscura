"""Tests for J1939 transport protocol."""

from __future__ import annotations

import pytest

from oscura.automotive.j1939.transport import TransportProtocol, TransportSession


class TestTransportSession:
    """Test TransportSession dataclass."""

    def test_session_creation(self):
        """Test creating transport session."""
        session = TransportSession(
            source_address=0x00,
            dest_address=0xFF,
            data_pgn=61444,
            total_size=32,
            total_packets=5,
        )

        assert session.source_address == 0x00
        assert session.dest_address == 0xFF
        assert session.data_pgn == 61444
        assert session.total_size == 32
        assert session.total_packets == 5
        assert not session.is_broadcast
        assert len(session.packets) == 0

    def test_session_is_complete_empty(self):
        """Test is_complete with no packets."""
        session = TransportSession(0, 0xFF, 61444, 32, 5)

        assert not session.is_complete()

    def test_session_is_complete_partial(self):
        """Test is_complete with some packets."""
        session = TransportSession(0, 0xFF, 61444, 32, 5)
        session.packets[1] = b"\x00" * 7
        session.packets[2] = b"\x00" * 7

        assert not session.is_complete()

    def test_session_is_complete_full(self):
        """Test is_complete with all packets."""
        session = TransportSession(0, 0xFF, 61444, 32, 5)

        for i in range(1, 6):
            session.packets[i] = b"\x00" * 7

        assert session.is_complete()

    def test_session_reassemble_complete(self):
        """Test reassembling complete session."""
        session = TransportSession(0, 0xFF, 61444, 14, 2)
        session.packets[1] = b"\x00\x01\x02\x03\x04\x05\x06"
        session.packets[2] = b"\x07\x08\x09\x0a\x0b\x0c\x0d"

        data = session.reassemble()

        assert len(data) == 14
        assert data == b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d"

    def test_session_reassemble_with_padding(self):
        """Test reassembling trims padding from last packet."""
        session = TransportSession(0, 0xFF, 61444, 10, 2)
        session.packets[1] = b"\x00\x01\x02\x03\x04\x05\x06"
        session.packets[2] = b"\x07\x08\x09\xff\xff\xff\xff"  # Padded

        data = session.reassemble()

        assert len(data) == 10
        assert data == b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"

    def test_session_reassemble_incomplete_raises(self):
        """Test reassembling incomplete session raises error."""
        session = TransportSession(0, 0xFF, 61444, 14, 2)
        session.packets[1] = b"\x00" * 7

        with pytest.raises(ValueError, match="Session incomplete"):
            session.reassemble()

    def test_session_broadcast_flag(self):
        """Test broadcast session flag."""
        session = TransportSession(0, 0xFF, 61444, 32, 5, is_broadcast=True)

        assert session.is_broadcast


class TestTransportProtocolParsing:
    """Test transport protocol message parsing."""

    def test_parse_cm_rts(self):
        """Test parsing TP.CM Request To Send."""
        tp = TransportProtocol()

        data = b"\x10\x20\x00\x05\xff\xf4\x00\x00"
        info = tp.parse_cm(data)

        assert info is not None
        assert info["control"] == "RTS"
        assert info["control_byte"] == 16
        assert info["total_size"] == 32
        assert info["total_packets"] == 5
        assert info["max_packets"] == 0xFF
        assert info["data_pgn"] == 244  # 0x0000F4

    def test_parse_cm_cts(self):
        """Test parsing TP.CM Clear To Send."""
        tp = TransportProtocol()

        data = b"\x11\x00\x00\x00\x03\x00\x00\x00"
        info = tp.parse_cm(data)

        assert info["control"] == "CTS"
        assert info["max_packets"] == 3

    def test_parse_cm_eom_ack(self):
        """Test parsing TP.CM End of Message ACK."""
        tp = TransportProtocol()

        data = b"\x13\x20\x00\x05\xff\xf0\x04\x00"
        info = tp.parse_cm(data)

        assert info["control"] == "EOM_ACK"

    def test_parse_cm_bam(self):
        """Test parsing TP.CM Broadcast Announce Message."""
        tp = TransportProtocol()

        data = b"\x20\x0e\x00\x02\xff\xf4\x00\x00"
        info = tp.parse_cm(data)

        assert info["control"] == "BAM"
        assert info["total_size"] == 14
        assert info["total_packets"] == 2
        assert info["data_pgn"] == 244  # 0x0000F4

    def test_parse_cm_abort(self):
        """Test parsing TP.CM Abort."""
        tp = TransportProtocol()

        data = b"\xff\x00\x00\x01\x00\x00\x00\x00"
        info = tp.parse_cm(data)

        assert info["control"] == "ABORT"

    def test_parse_cm_unknown_control(self):
        """Test parsing unknown control byte."""
        tp = TransportProtocol()

        data = b"\x99\x00\x00\x00\x00\x00\x00\x00"
        info = tp.parse_cm(data)

        assert "Unknown" in info["control"]

    def test_parse_cm_invalid_length(self):
        """Test parsing TP.CM with insufficient data."""
        tp = TransportProtocol()

        info = tp.parse_cm(b"\x10\x20\x00")

        assert info is None

    def test_parse_dt_valid(self):
        """Test parsing TP.DT data transfer."""
        tp = TransportProtocol()

        data = b"\x01\x00\x01\x02\x03\x04\x05\x06"
        info = tp.parse_dt(data)

        assert info is not None
        assert info["sequence"] == 1
        assert info["data"] == b"\x00\x01\x02\x03\x04\x05\x06"

    def test_parse_dt_different_sequences(self):
        """Test parsing TP.DT with different sequence numbers."""
        tp = TransportProtocol()

        for seq in [1, 5, 255]:
            data = bytes([seq]) + b"\x00" * 7
            info = tp.parse_dt(data)

            assert info["sequence"] == seq

    def test_parse_dt_invalid_length(self):
        """Test parsing TP.DT with no data."""
        tp = TransportProtocol()

        info = tp.parse_dt(b"")

        assert info is None


class TestTransportProtocolSessions:
    """Test transport protocol session management."""

    def test_start_session_rts(self):
        """Test starting session from RTS."""
        tp = TransportProtocol()

        cm_info = tp.parse_cm(b"\x10\x20\x00\x05\xff\xf4\x00\x00")
        session = tp.start_session(0x00, 0xFF, cm_info, 1.0)

        assert session is not None
        assert session.source_address == 0x00
        assert session.dest_address == 0xFF
        assert session.data_pgn == 244  # 0x0000F4
        assert session.total_size == 32
        assert session.total_packets == 5
        assert not session.is_broadcast
        assert session.started_at == 1.0

    def test_start_session_bam(self):
        """Test starting session from BAM."""
        tp = TransportProtocol()

        cm_info = tp.parse_cm(b"\x20\x0e\x00\x02\xff\xf0\x04\x00")
        session = tp.start_session(0x00, 0xFF, cm_info, 1.0)

        assert session is not None
        assert session.is_broadcast

    def test_start_session_invalid_control(self):
        """Test starting session from non-RTS/BAM returns None."""
        tp = TransportProtocol()

        cm_info = tp.parse_cm(b"\x11\x00\x00\x00\x03\x00\x00\x00")  # CTS
        session = tp.start_session(0x00, 0xFF, cm_info, 1.0)

        assert session is None

    def test_start_session_stores_in_sessions(self):
        """Test started session is stored."""
        tp = TransportProtocol()

        cm_info = tp.parse_cm(b"\x10\x20\x00\x05\xff\xf0\x04\x00")
        tp.start_session(0x00, 0xFF, cm_info, 1.0)

        assert (0x00, 0xFF) in tp.sessions

    def test_add_packet_to_session(self):
        """Test adding packet to active session."""
        tp = TransportProtocol()

        # Start session
        cm_info = tp.parse_cm(b"\x10\x0e\x00\x02\xff\xf0\x04\x00")
        tp.start_session(0x00, 0xFF, cm_info, 1.0)

        # Add first packet
        dt_info = tp.parse_dt(b"\x01\x00\x01\x02\x03\x04\x05\x06")
        result = tp.add_packet(0x00, 0xFF, dt_info, 1.1)

        assert result is None  # Not complete yet
        assert len(tp.sessions[(0x00, 0xFF)].packets) == 1

    def test_add_packet_completes_session(self):
        """Test adding final packet completes session."""
        tp = TransportProtocol()

        # Start session (14 bytes, 2 packets)
        cm_info = tp.parse_cm(b"\x20\x0e\x00\x02\xff\xf0\x04\x00")
        tp.start_session(0x00, 0xFF, cm_info, 1.0)

        # Add packets
        dt1 = tp.parse_dt(b"\x01\x00\x01\x02\x03\x04\x05\x06")
        result1 = tp.add_packet(0x00, 0xFF, dt1, 1.1)

        assert result1 is None

        dt2 = tp.parse_dt(b"\x02\x07\x08\x09\x0a\x0b\x0c\x0d")
        result2 = tp.add_packet(0x00, 0xFF, dt2, 1.2)

        assert result2 is not None
        assert len(result2) == 14
        assert result2 == b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d"

    def test_add_packet_cleans_up_session(self):
        """Test completed session is removed."""
        tp = TransportProtocol()

        cm_info = tp.parse_cm(b"\x20\x0e\x00\x02\xff\xf0\x04\x00")
        tp.start_session(0x00, 0xFF, cm_info, 1.0)

        dt1 = tp.parse_dt(b"\x01" + b"\x00" * 7)
        tp.add_packet(0x00, 0xFF, dt1, 1.1)

        dt2 = tp.parse_dt(b"\x02" + b"\x00" * 7)
        tp.add_packet(0x00, 0xFF, dt2, 1.2)

        assert (0x00, 0xFF) not in tp.sessions

    def test_add_packet_no_session(self):
        """Test adding packet with no active session."""
        tp = TransportProtocol()

        dt_info = tp.parse_dt(b"\x01" + b"\x00" * 7)
        result = tp.add_packet(0x00, 0xFF, dt_info, 1.0)

        assert result is None

    def test_multiple_sessions(self):
        """Test multiple concurrent sessions."""
        tp = TransportProtocol()

        # Start session 1: SA=0x00, DA=0xFF
        cm1 = tp.parse_cm(b"\x10\x20\x00\x05\xff\xf0\x04\x00")
        tp.start_session(0x00, 0xFF, cm1, 1.0)

        # Start session 2: SA=0x01, DA=0xFF
        cm2 = tp.parse_cm(b"\x10\x0e\x00\x02\xff\xf1\x04\x00")
        tp.start_session(0x01, 0xFF, cm2, 1.0)

        assert len(tp.sessions) == 2
        assert (0x00, 0xFF) in tp.sessions
        assert (0x01, 0xFF) in tp.sessions


class TestTransportProtocolIntegration:
    """Integration tests for transport protocol."""

    def test_complete_rts_flow(self):
        """Test complete RTS/CTS flow."""
        tp = TransportProtocol()

        # 1. RTS
        cm_rts = tp.parse_cm(b"\x10\x15\x00\x03\xff\xf0\x04\x00")  # 21 bytes, 3 packets
        session = tp.start_session(0x00, 0x01, cm_rts, 1.0)

        assert session is not None

        # 2. Add TP.DT packets
        packets = [
            b"\x01\x00\x01\x02\x03\x04\x05\x06",
            b"\x02\x07\x08\x09\x0a\x0b\x0c\x0d",
            b"\x03\x0e\x0f\x10\x11\x12\x13\x14",
        ]

        for i, packet in enumerate(packets):
            dt_info = tp.parse_dt(packet)
            result = tp.add_packet(0x00, 0x01, dt_info, 1.0 + i * 0.1)

            if i < 2:
                assert result is None
            else:
                assert result is not None
                assert len(result) == 21

    def test_complete_bam_flow(self):
        """Test complete BAM broadcast flow."""
        tp = TransportProtocol()

        # BAM announcement
        cm_bam = tp.parse_cm(b"\x20\x0a\x00\x02\xff\xf0\x04\x00")  # 10 bytes, 2 packets
        session = tp.start_session(0x00, 0xFF, cm_bam, 1.0)

        assert session.is_broadcast

        # Add packets
        dt1 = tp.parse_dt(b"\x01\x00\x01\x02\x03\x04\x05\x06")
        tp.add_packet(0x00, 0xFF, dt1, 1.1)

        dt2 = tp.parse_dt(b"\x02\x07\x08\x09\xff\xff\xff\xff")
        result = tp.add_packet(0x00, 0xFF, dt2, 1.2)

        assert len(result) == 10


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_cm_all_zeros(self):
        """Test parsing TP.CM with all zeros."""
        tp = TransportProtocol()

        info = tp.parse_cm(b"\x00\x00\x00\x00\x00\x00\x00\x00")

        assert info is not None
        assert info["total_size"] == 0
        assert info["total_packets"] == 0

    def test_parse_cm_max_values(self):
        """Test parsing TP.CM with maximum values."""
        tp = TransportProtocol()

        # Max size (65535), max packets (255), max PGN
        data = b"\x10\xff\xff\xff\xff\xff\xff\xff"
        info = tp.parse_cm(data)

        assert info["total_size"] == 65535
        assert info["total_packets"] == 255
        assert info["data_pgn"] == 0xFFFFFF

    def test_session_single_packet(self):
        """Test session with single packet."""
        session = TransportSession(0, 0xFF, 61444, 7, 1)
        session.packets[1] = b"\x00\x01\x02\x03\x04\x05\x06"

        assert session.is_complete()
        data = session.reassemble()
        assert len(data) == 7

    def test_session_many_packets(self):
        """Test session with many packets."""
        session = TransportSession(0, 0xFF, 61444, 255 * 7, 255)

        for i in range(1, 256):
            session.packets[i] = bytes([i % 256]) * 7

        assert session.is_complete()
        data = session.reassemble()
        assert len(data) == 255 * 7
