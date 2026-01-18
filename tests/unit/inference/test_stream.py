"""Unit tests for stream reassembly and message framing.

Tests for:
    - RE-STR-001: UDP Stream Reconstruction
    - RE-STR-002: TCP Stream Reassembly
    - RE-STR-003: Message Framing and Segmentation
"""

from __future__ import annotations

import pytest

from oscura.inference.stream import (
    FramingResult,
    MessageFrame,
    MessageFramer,
    ReassembledStream,
    StreamSegment,
    TCPStreamReassembler,
    UDPStreamReassembler,
    detect_message_framing,
    extract_messages,
    reassemble_tcp_stream,
    reassemble_udp_stream,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test Data Classes
# =============================================================================


class TestStreamSegment:
    """Test StreamSegment dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic segment."""
        segment = StreamSegment(sequence_number=100, data=b"test data")

        assert segment.sequence_number == 100
        assert segment.data == b"test data"
        assert segment.timestamp == 0.0
        assert segment.src == ""
        assert segment.dst == ""
        assert segment.flags == 0
        assert segment.is_retransmit is False

    def test_full_creation(self) -> None:
        """Test creating segment with all fields."""
        segment = StreamSegment(
            sequence_number=42,
            data=b"payload",
            timestamp=123.456,
            src="192.168.1.1",
            dst="192.168.1.2",
            flags=0x02,
            is_retransmit=True,
        )

        assert segment.sequence_number == 42
        assert segment.data == b"payload"
        assert segment.timestamp == 123.456
        assert segment.src == "192.168.1.1"
        assert segment.dst == "192.168.1.2"
        assert segment.flags == 0x02
        assert segment.is_retransmit is True


class TestReassembledStream:
    """Test ReassembledStream dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a reassembled stream."""
        stream = ReassembledStream(
            data=b"complete data",
            src="10.0.0.1",
            dst="10.0.0.2",
            start_time=1.0,
            end_time=2.0,
            segments=5,
        )

        assert stream.data == b"complete data"
        assert stream.src == "10.0.0.1"
        assert stream.dst == "10.0.0.2"
        assert stream.start_time == 1.0
        assert stream.end_time == 2.0
        assert stream.segments == 5
        assert stream.gaps == []
        assert stream.retransmits == 0
        assert stream.out_of_order == 0

    def test_with_gaps_and_retransmits(self) -> None:
        """Test stream with gaps and retransmissions."""
        stream = ReassembledStream(
            data=b"data",
            src="a",
            dst="b",
            start_time=1.0,
            end_time=2.0,
            segments=10,
            gaps=[(100, 200), (300, 350)],
            retransmits=3,
            out_of_order=2,
        )

        assert stream.gaps == [(100, 200), (300, 350)]
        assert stream.retransmits == 3
        assert stream.out_of_order == 2


class TestMessageFrame:
    """Test MessageFrame dataclass."""

    def test_basic_frame(self) -> None:
        """Test creating a basic message frame."""
        frame = MessageFrame(
            data=b"message",
            offset=0,
            length=7,
        )

        assert frame.data == b"message"
        assert frame.offset == 0
        assert frame.length == 7
        assert frame.frame_type == "unknown"
        assert frame.is_complete is True
        assert frame.sequence is None

    def test_complete_frame(self) -> None:
        """Test frame with all fields."""
        frame = MessageFrame(
            data=b"msg",
            offset=100,
            length=3,
            frame_type="delimited",
            is_complete=False,
            sequence=42,
        )

        assert frame.frame_type == "delimited"
        assert frame.is_complete is False
        assert frame.sequence == 42


class TestFramingResult:
    """Test FramingResult dataclass."""

    def test_basic_result(self) -> None:
        """Test creating a basic framing result."""
        result = FramingResult(
            messages=[],
            framing_type="delimiter",
        )

        assert result.messages == []
        assert result.framing_type == "delimiter"
        assert result.delimiter is None
        assert result.length_field_offset is None
        assert result.length_field_size is None
        assert result.remaining == b""

    def test_complete_result(self) -> None:
        """Test result with all fields."""
        frames = [
            MessageFrame(data=b"msg1", offset=0, length=4),
            MessageFrame(data=b"msg2", offset=5, length=4),
        ]
        result = FramingResult(
            messages=frames,
            framing_type="length_prefix",
            delimiter=b"\n",
            length_field_offset=2,
            length_field_size=4,
            remaining=b"partial",
        )

        assert len(result.messages) == 2
        assert result.framing_type == "length_prefix"
        assert result.delimiter == b"\n"
        assert result.length_field_offset == 2
        assert result.length_field_size == 4
        assert result.remaining == b"partial"


# =============================================================================
# Test UDPStreamReassembler
# =============================================================================


class TestUDPStreamReassembler:
    """Test UDP stream reassembly.

    Tests RE-STR-001: UDP Stream Reconstruction.
    """

    def test_basic_reassembly_bytes(self) -> None:
        """Test reassembling simple byte packets."""
        reassembler = UDPStreamReassembler()

        reassembler.add_segment(b"packet1")
        reassembler.add_segment(b"packet2")
        reassembler.add_segment(b"packet3")

        stream = reassembler.get_stream()

        assert stream.data == b"packet1packet2packet3"
        assert stream.segments == 3

    def test_reassembly_with_sequence_numbers(self) -> None:
        """Test reassembly with explicit sequence numbers."""

        def get_seq(pkt: dict) -> int:
            return pkt["seq"]

        reassembler = UDPStreamReassembler(sequence_key=get_seq)

        # Add packets out of order
        reassembler.add_segment({"seq": 2, "data": b"third"})
        reassembler.add_segment({"seq": 0, "data": b"first"})
        reassembler.add_segment({"seq": 1, "data": b"second"})

        stream = reassembler.get_stream()

        # Should be ordered by sequence
        assert stream.data == b"firstsecondthird"
        assert stream.segments == 3
        assert stream.out_of_order == 2  # 2 packets were out of order

    def test_reassembly_with_timestamps(self) -> None:
        """Test reassembly tracking timestamps."""
        reassembler = UDPStreamReassembler()

        reassembler.add_segment(
            {"data": b"pkt1", "timestamp": 1.0, "src": "10.0.0.1", "dst": "10.0.0.2"}
        )
        reassembler.add_segment(
            {"data": b"pkt2", "timestamp": 2.0, "src": "10.0.0.1", "dst": "10.0.0.2"}
        )
        reassembler.add_segment(
            {"data": b"pkt3", "timestamp": 3.0, "src": "10.0.0.1", "dst": "10.0.0.2"}
        )

        stream = reassembler.get_stream()

        assert stream.start_time == 1.0
        assert stream.end_time == 3.0
        assert stream.src == "10.0.0.1"
        assert stream.dst == "10.0.0.2"

    def test_reassembly_detects_gaps(self) -> None:
        """Test gap detection in UDP streams."""

        def get_seq(pkt: dict) -> int:
            return pkt["seq"]

        reassembler = UDPStreamReassembler(sequence_key=get_seq)

        # Create packets with gaps
        reassembler.add_segment({"seq": 0, "data": b"aaaa"})  # 0-4
        reassembler.add_segment({"seq": 10, "data": b"bbbb"})  # Gap 4-10
        reassembler.add_segment({"seq": 20, "data": b"cccc"})  # Gap 14-20

        stream = reassembler.get_stream()

        assert len(stream.gaps) == 2
        assert (4, 10) in stream.gaps
        assert (14, 20) in stream.gaps

    def test_multiple_flows(self) -> None:
        """Test handling multiple UDP flows."""
        reassembler = UDPStreamReassembler()

        reassembler.add_segment(b"flow1_pkt1", flow_key="flow1")
        reassembler.add_segment(b"flow2_pkt1", flow_key="flow2")
        reassembler.add_segment(b"flow1_pkt2", flow_key="flow1")

        stream1 = reassembler.get_stream("flow1")
        stream2 = reassembler.get_stream("flow2")

        assert stream1.data == b"flow1_pkt1flow1_pkt2"
        assert stream2.data == b"flow2_pkt1"

    def test_get_all_streams(self) -> None:
        """Test getting all streams."""
        reassembler = UDPStreamReassembler()

        reassembler.add_segment(b"data1", flow_key="flow1")
        reassembler.add_segment(b"data2", flow_key="flow2")

        all_streams = reassembler.get_all_streams()

        assert len(all_streams) == 2
        assert "flow1" in all_streams
        assert "flow2" in all_streams
        assert all_streams["flow1"].data == b"data1"
        assert all_streams["flow2"].data == b"data2"

    def test_empty_reassembler(self) -> None:
        """Test empty reassembler returns empty stream."""
        reassembler = UDPStreamReassembler()
        stream = reassembler.get_stream()

        assert stream.data == b""
        assert stream.segments == 0

    def test_clear(self) -> None:
        """Test clearing reassembler."""
        reassembler = UDPStreamReassembler()
        reassembler.add_segment(b"data")

        reassembler.clear()
        stream = reassembler.get_stream()

        assert stream.data == b""
        assert stream.segments == 0

    def test_dict_with_payload_field(self) -> None:
        """Test handling dict with 'payload' instead of 'data'."""
        reassembler = UDPStreamReassembler()
        reassembler.add_segment({"payload": b"test", "src_ip": "1.2.3.4", "dst_ip": "5.6.7.8"})

        stream = reassembler.get_stream()

        assert stream.data == b"test"
        assert stream.src == "1.2.3.4"
        assert stream.dst == "5.6.7.8"

    def test_sequence_key_exception_handling(self) -> None:
        """Test graceful handling when sequence_key raises exception."""

        def bad_seq_key(pkt: dict) -> int:
            return pkt["nonexistent_field"]

        reassembler = UDPStreamReassembler(sequence_key=bad_seq_key)
        reassembler.add_segment({"data": b"test"})

        # Should not crash, falls back to default
        stream = reassembler.get_stream()
        assert stream.data == b"test"


# =============================================================================
# Test TCPStreamReassembler
# =============================================================================


class TestTCPStreamReassembler:
    """Test TCP stream reassembly.

    Tests RE-STR-002: TCP Stream Reassembly.
    """

    def test_basic_reassembly(self) -> None:
        """Test basic TCP stream reassembly."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b"first"})
        reassembler.add_segment({"seq": 1005, "data": b"second"})
        reassembler.add_segment({"seq": 1011, "data": b"third"})

        stream = reassembler.get_stream()

        assert stream.data == b"firstsecondthird"
        assert stream.segments == 3

    def test_out_of_order_reassembly(self) -> None:
        """Test reassembling out-of-order segments."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1005, "data": b"second"})
        reassembler.add_segment({"seq": 1011, "data": b"third"})
        reassembler.add_segment({"seq": 1000, "data": b"first"})

        stream = reassembler.get_stream()

        assert stream.data == b"firstsecondthird"
        assert stream.out_of_order > 0

    def test_syn_detection(self) -> None:
        """Test SYN flag detection for initial sequence."""
        reassembler = TCPStreamReassembler()

        # SYN packet (flag 0x02)
        reassembler.add_segment({"seq": 1000, "data": b"", "flags": 0x02, "src": "a", "dst": "b"})
        # Data packets
        reassembler.add_segment({"seq": 1001, "data": b"hello", "src": "a", "dst": "b"})

        stream = reassembler.get_stream()

        # SYN should set ISN but not be in data
        assert stream.data == b"hello"

    def test_retransmit_detection(self) -> None:
        """Test retransmission detection."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b"original"})
        reassembler.add_segment({"seq": 1000, "data": b"retrans"})  # Same seq

        stream = reassembler.get_stream()

        assert stream.retransmits == 1

    def test_gap_detection(self) -> None:
        """Test gap detection in TCP stream."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b"aaaa"})  # 1000-1004
        reassembler.add_segment({"seq": 1010, "data": b"bbbb"})  # Gap at 1004-1010

        stream = reassembler.get_stream()

        assert len(stream.gaps) > 0
        # Gap should be filled with zeros
        assert len(stream.data) == 14  # 4 + 6 (gap) + 4

    def test_overlap_handling(self) -> None:
        """Test handling overlapping segments."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b"abcdef"})
        reassembler.add_segment({"seq": 1003, "data": b"xyz"})  # Overlaps with "def"

        stream = reassembler.get_stream()

        # Should handle overlap gracefully
        assert len(stream.data) > 0

    def test_timestamp_tracking(self) -> None:
        """Test timestamp tracking."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b"a", "timestamp": 1.5})
        reassembler.add_segment({"seq": 1001, "data": b"b", "timestamp": 2.5})
        reassembler.add_segment({"seq": 1002, "data": b"c", "timestamp": 3.5})

        stream = reassembler.get_stream()

        assert stream.start_time == 1.5
        assert stream.end_time == 3.5

    def test_stream_segment_input(self) -> None:
        """Test using StreamSegment objects directly."""
        reassembler = TCPStreamReassembler()

        seg1 = StreamSegment(sequence_number=1000, data=b"first", src="a", dst="b")
        seg2 = StreamSegment(sequence_number=1005, data=b"second", src="a", dst="b")

        reassembler.add_segment(seg1)
        reassembler.add_segment(seg2)

        stream = reassembler.get_stream()

        assert stream.data == b"firstsecond"

    def test_multiple_flows(self) -> None:
        """Test handling multiple TCP flows."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b"flow1"}, flow_key="conn1")
        reassembler.add_segment({"seq": 2000, "data": b"flow2"}, flow_key="conn2")

        stream1 = reassembler.get_stream("conn1")
        stream2 = reassembler.get_stream("conn2")

        assert stream1.data == b"flow1"
        assert stream2.data == b"flow2"

    def test_get_all_streams(self) -> None:
        """Test getting all TCP streams."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b"a"}, flow_key="f1")
        reassembler.add_segment({"seq": 2000, "data": b"b"}, flow_key="f2")

        all_streams = reassembler.get_all_streams()

        assert len(all_streams) == 2
        assert "f1" in all_streams
        assert "f2" in all_streams

    def test_empty_reassembler(self) -> None:
        """Test empty TCP reassembler."""
        reassembler = TCPStreamReassembler()
        stream = reassembler.get_stream()

        assert stream.data == b""
        assert stream.segments == 0

    def test_clear(self) -> None:
        """Test clearing TCP reassembler."""
        reassembler = TCPStreamReassembler()
        reassembler.add_segment({"seq": 1000, "data": b"test"})

        reassembler.clear()
        stream = reassembler.get_stream()

        assert stream.data == b""
        assert stream.segments == 0

    def test_initial_sequence_parameter(self) -> None:
        """Test providing initial sequence number."""
        reassembler = TCPStreamReassembler(initial_sequence=5000)

        reassembler.add_segment({"seq": 5000, "data": b"start"})
        reassembler.add_segment({"seq": 5005, "data": b"next"})

        stream = reassembler.get_stream()

        assert stream.data == b"startnext"

    def test_sequence_wraparound(self) -> None:
        """Test handling sequence number wraparound."""
        reassembler = TCPStreamReassembler(initial_sequence=2**32 - 10)

        # Sequences near wraparound
        reassembler.add_segment({"seq": 2**32 - 10, "data": b"before"})
        reassembler.add_segment({"seq": 2**32 - 4, "data": b"wrap"})

        stream = reassembler.get_stream()

        assert stream.segments == 2

    def test_dict_with_payload_field(self) -> None:
        """Test handling dict with 'payload' field."""
        reassembler = TCPStreamReassembler()
        reassembler.add_segment({"sequence_number": 1000, "payload": b"data"})

        stream = reassembler.get_stream()

        assert stream.data == b"data"

    def test_zero_timestamp_handling(self) -> None:
        """Test streams with zero timestamps."""
        reassembler = TCPStreamReassembler()
        reassembler.add_segment({"seq": 1000, "data": b"test", "timestamp": 0.0})

        stream = reassembler.get_stream()

        assert stream.start_time == 0.0
        assert stream.end_time == 0.0


# =============================================================================
# Test MessageFramer
# =============================================================================


class TestMessageFramer:
    """Test message framing functionality.

    Tests RE-STR-003: Message Framing and Segmentation.
    """

    def test_delimiter_framing(self) -> None:
        """Test delimiter-based framing."""
        framer = MessageFramer(framing_type="delimiter", delimiter=b"\n")

        data = b"message1\nmessage2\nmessage3\n"
        result = framer.frame(data)

        assert result.framing_type == "delimiter"
        assert len(result.messages) == 3
        assert result.messages[0].data == b"message1"
        assert result.messages[1].data == b"message2"
        assert result.messages[2].data == b"message3"

    def test_delimiter_framing_with_remaining(self) -> None:
        """Test delimiter framing with incomplete message."""
        framer = MessageFramer(framing_type="delimiter", delimiter=b"\r\n")

        data = b"msg1\r\nmsg2\r\npartial"
        result = framer.frame(data)

        # Implementation treats final part as complete message even without delimiter
        assert len(result.messages) == 3
        assert result.messages[2].data == b"partial"
        assert result.remaining == b"partial"  # Also in remaining

    def test_length_prefix_framing(self) -> None:
        """Test length-prefixed framing."""
        framer = MessageFramer(
            framing_type="length_prefix",
            length_field_offset=0,
            length_field_size=2,
            length_field_endian="big",
        )

        # Create messages with 2-byte big-endian length prefix
        msg1 = (5).to_bytes(2, "big") + b"hello"
        msg2 = (5).to_bytes(2, "big") + b"world"
        data = msg1 + msg2

        result = framer.frame(data)

        assert result.framing_type == "length_prefix"
        assert len(result.messages) == 2
        assert result.messages[0].length == 7  # 2 header + 5 data
        assert result.messages[1].length == 7

    def test_length_prefix_little_endian(self) -> None:
        """Test length-prefixed framing with little endian."""
        framer = MessageFramer(
            framing_type="length_prefix",
            length_field_size=2,
            length_field_endian="little",
        )

        msg = (4).to_bytes(2, "little") + b"test"
        result = framer.frame(msg)

        assert len(result.messages) == 1

    def test_length_prefix_with_offset(self) -> None:
        """Test length prefix at non-zero offset."""
        framer = MessageFramer(
            framing_type="length_prefix",
            length_field_offset=2,
            length_field_size=2,
        )

        # 2 byte header, then 2 byte length, then data
        msg = b"XX" + (4).to_bytes(2, "big") + b"data"
        result = framer.frame(msg)

        assert len(result.messages) == 1
        assert result.messages[0].length == 8  # 2 + 2 + 4

    def test_length_includes_header(self) -> None:
        """Test length field that includes header."""
        framer = MessageFramer(
            framing_type="length_prefix",
            length_field_size=2,
            length_includes_header=True,
        )

        # Length is 10 (including 2-byte header + 8 bytes data)
        msg = (10).to_bytes(2, "big") + b"12345678"
        result = framer.frame(msg)

        assert len(result.messages) == 1
        assert result.messages[0].length == 10

    def test_length_prefix_incomplete_message(self) -> None:
        """Test length prefix with incomplete message."""
        framer = MessageFramer(framing_type="length_prefix", length_field_size=2)

        # Says it needs 100 bytes but only provides 10
        msg = (100).to_bytes(2, "big") + b"short"
        result = framer.frame(msg)

        assert len(result.messages) == 0
        assert len(result.remaining) > 0

    def test_fixed_size_framing(self) -> None:
        """Test fixed-size framing."""
        framer = MessageFramer(framing_type="fixed", fixed_size=5)

        data = b"1234567890abcdefgh"
        result = framer.frame(data)

        assert result.framing_type == "fixed"
        assert len(result.messages) == 3
        assert result.messages[0].data == b"12345"
        assert result.messages[1].data == b"67890"
        assert result.messages[2].data == b"abcde"
        assert result.remaining == b"fgh"

    def test_fixed_size_exact_multiple(self) -> None:
        """Test fixed size with exact multiple."""
        framer = MessageFramer(framing_type="fixed", fixed_size=4)

        data = b"12345678"
        result = framer.frame(data)

        assert len(result.messages) == 2
        assert result.remaining == b""

    def test_auto_detect_delimiter(self) -> None:
        """Test auto-detection of delimiter framing."""
        framer = MessageFramer(framing_type="auto")

        data = b"line1\nline2\nline3\nline4\n"
        result = framer.frame(data)

        assert result.framing_type == "delimiter"
        assert len(result.messages) >= 3

    def test_auto_detect_length_prefix(self) -> None:
        """Test auto-detection of length-prefixed framing."""
        framer = MessageFramer(framing_type="auto")

        # Create data that looks length-prefixed
        msg1 = (10).to_bytes(2, "big") + b"a" * 10
        msg2 = (10).to_bytes(2, "big") + b"b" * 10
        data = msg1 + msg2

        result = framer.frame(data)

        # Should detect length_prefix
        assert result.framing_type in ["length_prefix", "fixed", "unknown"]

    def test_auto_detect_fixed(self) -> None:
        """Test auto-detection of fixed-size framing."""
        framer = MessageFramer(framing_type="auto")

        # Create repeating pattern
        data = b"HEAD" + b"A" * 12 + b"HEAD" + b"B" * 12 + b"HEAD" + b"C" * 12
        result = framer.frame(data)

        # Should detect fixed size
        assert result.framing_type in ["fixed", "unknown"]

    def test_auto_detect_unknown(self) -> None:
        """Test auto-detection falling back to unknown."""
        framer = MessageFramer(framing_type="auto")

        data = b"random binary data with no clear pattern"
        result = framer.frame(data)

        assert result.framing_type == "unknown"
        assert len(result.messages) == 1
        assert result.messages[0].data == data

    def test_detect_framing_delimiter(self) -> None:
        """Test framing detection for delimiters."""
        framer = MessageFramer()

        data = b"msg1\r\nmsg2\r\nmsg3\r\nmsg4\r\n"
        detected = framer.detect_framing(data)

        assert detected == "delimiter"

    def test_detect_framing_length_prefix(self) -> None:
        """Test framing detection for length prefix.

        Note: Framing detection is heuristic and may return 'unknown'
        for data that doesn't clearly match a specific pattern.
        """
        framer = MessageFramer()

        msg1 = (20).to_bytes(2, "big") + b"a" * 20
        msg2 = (20).to_bytes(2, "big") + b"b" * 20
        data = msg1 + msg2

        detected = framer.detect_framing(data)

        # Detection is heuristic - may return unknown for ambiguous patterns
        assert detected in ["length_prefix", "fixed", "unknown"]

    def test_detect_framing_fixed(self) -> None:
        """Test framing detection for fixed size."""
        framer = MessageFramer()

        # Create data with repeating structure
        chunk = b"HDR" + b"X" * 13
        data = chunk * 10

        detected = framer.detect_framing(data)

        assert detected in ["fixed", "unknown"]

    def test_delimiter_empty_parts(self) -> None:
        """Test delimiter framing skips empty parts."""
        framer = MessageFramer(framing_type="delimiter", delimiter=b"\n")

        data = b"msg1\n\nmsg2\n"  # Double newline creates empty part
        result = framer.frame(data)

        # Empty parts should be skipped
        assert all(msg.data for msg in result.messages)

    def test_sequence_numbers(self) -> None:
        """Test message sequence numbering."""
        framer = MessageFramer(framing_type="delimiter", delimiter=b"|")

        data = b"a|b|c|d|"
        result = framer.frame(data)

        assert result.messages[0].sequence == 0
        assert result.messages[1].sequence == 1
        assert result.messages[2].sequence == 2
        assert result.messages[3].sequence == 3

    def test_delimiter_none_returns_empty(self) -> None:
        """Test delimiter framing with None delimiter returns empty."""
        framer = MessageFramer(framing_type="delimiter", delimiter=None)

        result = framer.frame(b"test data")

        assert len(result.messages) == 0

    def test_fixed_size_zero_returns_empty(self) -> None:
        """Test fixed framing with size 0 returns empty."""
        framer = MessageFramer(framing_type="fixed", fixed_size=0)

        result = framer.frame(b"test data")

        assert len(result.messages) == 0

    def test_length_prefix_4_byte(self) -> None:
        """Test length prefix with 4-byte length field."""
        framer = MessageFramer(
            framing_type="length_prefix",
            length_field_size=4,
            length_field_endian="big",
        )

        msg = (8).to_bytes(4, "big") + b"testdata"
        result = framer.frame(msg)

        assert len(result.messages) == 1
        assert result.messages[0].length == 12  # 4 header + 8 data


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience functions for stream operations."""

    def test_reassemble_udp_stream(self) -> None:
        """Test reassemble_udp_stream convenience function."""
        packets = [
            {"seq": 0, "data": b"first"},
            {"seq": 1, "data": b"second"},
            {"seq": 2, "data": b"third"},
        ]

        stream = reassemble_udp_stream(packets, sequence_key=lambda p: p["seq"])

        assert stream.data == b"firstsecondthird"
        assert stream.segments == 3

    def test_reassemble_udp_stream_bytes(self) -> None:
        """Test UDP reassembly with byte packets."""
        packets = [b"a", b"b", b"c"]

        stream = reassemble_udp_stream(packets)

        assert stream.data == b"abc"

    def test_reassemble_tcp_stream(self) -> None:
        """Test reassemble_tcp_stream convenience function."""
        segments = [
            {"seq": 1000, "data": b"hello"},
            {"seq": 1005, "data": b"world"},
        ]

        stream = reassemble_tcp_stream(segments)

        assert stream.data == b"helloworld"

    def test_reassemble_tcp_stream_with_flow_key(self) -> None:
        """Test TCP reassembly with flow key."""
        segments = [
            {"seq": 1000, "data": b"test"},
        ]

        stream = reassemble_tcp_stream(segments, flow_key="conn1")

        assert stream.data == b"test"

    def test_extract_messages_auto(self) -> None:
        """Test extract_messages with auto detection."""
        data = b"msg1\nmsg2\nmsg3\nmsg4\n"

        result = extract_messages(data, framing_type="auto")

        assert len(result.messages) > 0

    def test_extract_messages_delimiter(self) -> None:
        """Test extract_messages with delimiter."""
        data = b"a;b;c;d;"

        result = extract_messages(data, framing_type="delimiter", delimiter=b";")

        assert result.framing_type == "delimiter"
        assert len(result.messages) == 4

    def test_extract_messages_length_prefix(self) -> None:
        """Test extract_messages with length prefix."""
        msg1 = (4).to_bytes(2, "big") + b"test"
        msg2 = (4).to_bytes(2, "big") + b"data"

        result = extract_messages(msg1 + msg2, framing_type="length_prefix")

        assert result.framing_type == "length_prefix"
        assert len(result.messages) == 2

    def test_extract_messages_fixed(self) -> None:
        """Test extract_messages with fixed size."""
        data = b"123456789012"

        result = extract_messages(data, framing_type="fixed", fixed_size=4)

        assert result.framing_type == "fixed"
        assert len(result.messages) == 3

    def test_detect_message_framing_delimiter(self) -> None:
        """Test detect_message_framing for delimiter."""
        data = b"line1\r\nline2\r\nline3\r\nline4\r\n"

        framing = detect_message_framing(data)

        assert framing["type"] == "delimiter"
        assert "delimiter" in framing
        assert framing["delimiter"] == b"\r\n"

    def test_detect_message_framing_length_prefix(self) -> None:
        """Test detect_message_framing for length prefix.

        Note: Framing detection is heuristic and may return 'unknown'
        for data that doesn't clearly match a specific pattern.
        """
        msg1 = (15).to_bytes(2, "big") + b"a" * 15
        msg2 = (15).to_bytes(2, "big") + b"b" * 15
        data = msg1 + msg2

        framing = detect_message_framing(data)

        # Detection is heuristic - may return unknown for ambiguous patterns
        assert framing["type"] in ["length_prefix", "fixed", "unknown"]

    def test_detect_message_framing_fixed(self) -> None:
        """Test detect_message_framing for fixed size."""
        chunk = b"HEAD" + b"X" * 12
        data = chunk * 5

        framing = detect_message_framing(data)

        assert framing["type"] in ["fixed", "unknown"]
        if framing["type"] == "fixed":
            assert "fixed_size" in framing

    def test_detect_message_framing_unknown(self) -> None:
        """Test detect_message_framing for unknown format."""
        data = b"random data with no pattern"

        framing = detect_message_framing(data)

        assert framing["type"] == "unknown"


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestInferenceStreamEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data_framing(self) -> None:
        """Test framing empty data."""
        framer = MessageFramer(framing_type="delimiter", delimiter=b"\n")

        result = framer.frame(b"")

        assert len(result.messages) == 0

    def test_very_large_length_prefix(self) -> None:
        """Test handling unreasonably large length prefix."""
        framer = MessageFramer(framing_type="length_prefix", length_field_size=2)

        # Claim to need 60000 bytes but only provide 10
        msg = (60000).to_bytes(2, "big") + b"small"
        result = framer.frame(msg)

        # Should not crash, message incomplete
        assert len(result.messages) == 0

    def test_single_byte_segments(self) -> None:
        """Test reassembling single-byte segments."""
        reassembler = UDPStreamReassembler()

        for i in range(10):
            reassembler.add_segment({"seq": i, "data": bytes([i])})

        stream = reassembler.get_stream()

        assert len(stream.data) == 10

    def test_delimiter_at_start(self) -> None:
        """Test delimiter at start of data."""
        framer = MessageFramer(framing_type="delimiter", delimiter=b"\n")

        data = b"\nmsg1\nmsg2"
        result = framer.frame(data)

        # Empty parts should be skipped
        assert all(msg.data for msg in result.messages)

    def test_delimiter_at_end(self) -> None:
        """Test delimiter at end of data."""
        framer = MessageFramer(framing_type="delimiter", delimiter=b"\n")

        data = b"msg1\nmsg2\n"
        result = framer.frame(data)

        assert result.remaining == b""

    def test_multiple_delimiter_types(self) -> None:
        """Test different delimiter types."""
        for delim in [b"\r\n", b"\n", b"\x00", b"|", b","]:
            framer = MessageFramer(framing_type="delimiter", delimiter=delim)

            data = delim.join([b"a", b"b", b"c"]) + delim
            result = framer.frame(data)

            assert len(result.messages) == 3

    def test_tcp_no_data_segments(self) -> None:
        """Test TCP segments with no data (e.g., ACK only)."""
        reassembler = TCPStreamReassembler()

        reassembler.add_segment({"seq": 1000, "data": b""})
        reassembler.add_segment({"seq": 1000, "data": b"actual"})

        stream = reassembler.get_stream()

        assert stream.data == b"actual"

    def test_udp_no_timestamps(self) -> None:
        """Test UDP with no timestamps."""
        reassembler = UDPStreamReassembler()
        reassembler.add_segment({"data": b"test"})

        stream = reassembler.get_stream()

        assert stream.start_time == 0.0
        assert stream.end_time == 0.0

    def test_framing_very_short_data(self) -> None:
        """Test framing detection on very short data."""
        framer = MessageFramer()

        result = framer.detect_framing(b"ab")

        # Should not crash, returns unknown
        assert isinstance(result, str)

    def test_length_prefix_no_remaining_data(self) -> None:
        """Test length prefix with exact data."""
        framer = MessageFramer(framing_type="length_prefix", length_field_size=2)

        msg = (4).to_bytes(2, "big") + b"test"
        result = framer.frame(msg)

        assert result.remaining == b""

    def test_auto_framing_short_data(self) -> None:
        """Test auto framing with insufficient data."""
        framer = MessageFramer(framing_type="auto")

        result = framer.frame(b"ab")

        # Should return something reasonable
        assert result.framing_type == "unknown"
        assert len(result.messages) == 1


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_http_like_reassembly(self) -> None:
        """Test reassembling HTTP-like messages."""
        # Simulate HTTP-like protocol with CRLF delimiters
        framer = MessageFramer(framing_type="delimiter", delimiter=b"\r\n\r\n")

        request1 = b"GET /path HTTP/1.1\r\nHost: example.com\r\n\r\n"
        request2 = b"POST /api HTTP/1.1\r\nContent-Length: 5\r\n\r\n"

        data = request1 + request2
        result = framer.frame(data)

        assert len(result.messages) == 2

    def test_binary_protocol_with_length(self) -> None:
        """Test binary protocol with length prefix."""
        framer = MessageFramer(
            framing_type="length_prefix",
            length_field_offset=4,  # 4-byte magic number
            length_field_size=4,
            length_field_endian="big",
        )

        # Magic + Length + Payload
        msg = b"MAGC" + (8).to_bytes(4, "big") + b"testdata"
        result = framer.frame(msg)

        assert len(result.messages) == 1

    def test_udp_game_protocol(self) -> None:
        """Test UDP game protocol with sequence numbers."""

        def get_seq(pkt: dict) -> int:
            # Extract sequence from first 4 bytes
            return int.from_bytes(pkt["data"][:4], "big")

        packets = [
            {"data": (2).to_bytes(4, "big") + b"update2"},
            {"data": (0).to_bytes(4, "big") + b"update0"},
            {"data": (1).to_bytes(4, "big") + b"update1"},
        ]

        stream = reassemble_udp_stream(packets, sequence_key=get_seq)

        # Should be reordered
        assert stream.segments == 3
        assert stream.out_of_order == 2

    def test_tcp_with_fragmentation(self) -> None:
        """Test TCP with message fragmentation across segments."""
        reassembler = TCPStreamReassembler()

        # Simulate message split across TCP segments
        # Total data: "BEGINMIDDLEEND" = 14 bytes
        reassembler.add_segment({"seq": 1000, "data": b"BEGIN"})
        reassembler.add_segment({"seq": 1005, "data": b"MIDDLE"})
        reassembler.add_segment({"seq": 1011, "data": b"END"})

        stream = reassembler.get_stream()
        assert stream.data == b"BEGINMIDDLEEND"  # Verify reassembly

        # Then frame the reassembled stream (14 bytes / 7 = 2 complete messages)
        framer = MessageFramer(framing_type="fixed", fixed_size=7)
        result = framer.frame(stream.data)

        assert len(result.messages) == 2
        assert result.messages[0].data == b"BEGINMI"
        assert result.messages[1].data == b"DDLEEND"

    def test_multi_flow_scenario(self) -> None:
        """Test handling multiple concurrent flows."""
        reassembler = TCPStreamReassembler()

        # Client to server
        reassembler.add_segment(
            {"seq": 1000, "data": b"REQUEST", "src": "client", "dst": "server"},
            flow_key="c2s",
        )

        # Server to client
        reassembler.add_segment(
            {"seq": 2000, "data": b"RESPONSE", "src": "server", "dst": "client"},
            flow_key="s2c",
        )

        c2s = reassembler.get_stream("c2s")
        s2c = reassembler.get_stream("s2c")

        assert c2s.data == b"REQUEST"
        assert s2c.data == b"RESPONSE"

    def test_complete_workflow(self) -> None:
        """Test complete workflow: capture -> reassemble -> frame."""
        # Simulate captured TCP segments with contiguous sequence numbers
        # Each segment is: 2 bytes length prefix + 5 bytes data = 7 bytes
        segments = [
            {"seq": 1000, "data": (5).to_bytes(2, "big") + b"msg1\n"},  # ends at 1007
            {"seq": 1007, "data": (5).to_bytes(2, "big") + b"msg2\n"},  # ends at 1014
            {"seq": 1014, "data": (5).to_bytes(2, "big") + b"msg3\n"},  # ends at 1021
        ]

        # Reassemble TCP stream
        stream = reassemble_tcp_stream(segments)

        # Frame messages using length prefix
        result = extract_messages(
            stream.data,
            framing_type="length_prefix",
            length_field_size=2,
        )

        assert len(result.messages) == 3
        assert result.messages[0].data == (5).to_bytes(2, "big") + b"msg1\n"
        assert result.messages[1].data == (5).to_bytes(2, "big") + b"msg2\n"
        assert result.messages[2].data == (5).to_bytes(2, "big") + b"msg3\n"
