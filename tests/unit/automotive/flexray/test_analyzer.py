"""Tests for FlexRay protocol analyzer.

Tests cover:
- Header parsing and field extraction
- Frame parsing and validation
- CRC verification (header and frame)
- Signal decoding with scaling
- Segment type determination
- Frame statistics
- Multi-channel support
- Error handling
"""

import pytest

from oscura.automotive.flexray import (
    FlexRayAnalyzer,
    FlexRayHeader,
    FlexRaySignal,
)
from oscura.automotive.flexray.crc import calculate_frame_crc, calculate_header_crc


class TestFlexRayHeader:
    """Tests for FlexRay header parsing."""

    def test_header_creation(self) -> None:
        """Test creating a FlexRay header."""
        header = FlexRayHeader(
            reserved=0,
            payload_preamble=0,
            null_frame=False,
            sync_frame=True,
            startup_frame=False,
            frame_id=100,
            payload_length=10,
            header_crc=0x3A5,
            cycle_count=5,
        )

        assert header.frame_id == 100
        assert header.cycle_count == 5
        assert header.payload_length == 10
        assert header.sync_frame is True
        assert header.startup_frame is False

    def test_header_flags(self) -> None:
        """Test header flag fields."""
        # Startup frame
        header = FlexRayHeader(
            reserved=0,
            payload_preamble=0,
            null_frame=False,
            sync_frame=False,
            startup_frame=True,
            frame_id=1,
            payload_length=0,
            header_crc=0,
            cycle_count=0,
        )
        assert header.startup_frame is True

        # Null frame
        header = FlexRayHeader(
            reserved=0,
            payload_preamble=0,
            null_frame=True,
            sync_frame=False,
            startup_frame=False,
            frame_id=2,
            payload_length=0,
            header_crc=0,
            cycle_count=0,
        )
        assert header.null_frame is True


class TestFlexRayAnalyzer:
    """Tests for FlexRay analyzer."""

    def test_analyzer_initialization(self) -> None:
        """Test analyzer initialization."""
        analyzer = FlexRayAnalyzer()
        assert len(analyzer.frames) == 0
        assert len(analyzer.signals) == 0
        assert analyzer.cluster_config == {}

    def test_analyzer_with_config(self) -> None:
        """Test analyzer initialization with cluster config."""
        config = {
            "static_slot_count": 150,
            "dynamic_slot_count": 100,
            "cycle_length": 5000,
        }
        analyzer = FlexRayAnalyzer(cluster_config=config)
        assert analyzer.cluster_config["static_slot_count"] == 150
        assert analyzer.cluster_config["dynamic_slot_count"] == 100

    def test_parse_simple_frame(self) -> None:
        """Test parsing a simple FlexRay frame."""
        analyzer = FlexRayAnalyzer()

        # Create header bytes
        # Frame ID = 100 (0x064), payload length = 5 words (10 bytes)
        # Header structure (40 bits):
        # [39] reserved = 0
        # [38] payload_preamble = 0
        # [37] null_frame = 0
        # [36] sync_frame = 0
        # [35] startup_frame = 0
        # [34:24] frame_id = 100 (11 bits)
        # [23:17] payload_length = 5 words (7 bits)
        # [16:6] header_crc (11 bits) - calculated
        # [5:0] cycle_count = 0 (6 bits)

        # Calculate correct header CRC
        header_crc = calculate_header_crc(
            reserved=0,
            payload_preamble=0,
            null_frame=0,
            sync_frame=0,
            startup_frame=0,
            frame_id=100,
            payload_length=5,  # words
        )

        # Build 40-bit header
        header_int = (
            (0 << 39)  # reserved
            | (0 << 38)  # payload_preamble
            | (0 << 37)  # null_frame
            | (0 << 36)  # sync_frame
            | (0 << 35)  # startup_frame
            | (100 << 24)  # frame_id
            | (5 << 17)  # payload_length (words)
            | (header_crc << 6)  # header_crc
            | (0 << 0)  # cycle_count
        )

        header_bytes = header_int.to_bytes(5, "big")

        # Create payload (10 bytes)
        payload = bytes(range(10))

        # Calculate frame CRC
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")

        # Combine into complete frame
        frame_data = header_bytes + payload + crc_bytes

        # Parse frame
        frame = analyzer.parse_frame(frame_data, timestamp=1.0, channel="A")

        assert frame.header.frame_id == 100
        assert frame.header.cycle_count == 0
        assert frame.header.payload_length == 10
        assert frame.channel == "A"
        assert frame.timestamp == 1.0
        assert len(frame.payload) == 10
        assert frame.crc_valid is True

    def test_parse_frame_with_sync_flag(self) -> None:
        """Test parsing frame with sync flag set."""
        analyzer = FlexRayAnalyzer()

        # Frame with sync flag
        header_crc = calculate_header_crc(
            reserved=0,
            payload_preamble=0,
            null_frame=0,
            sync_frame=1,  # Sync frame
            startup_frame=0,
            frame_id=1,
            payload_length=0,
        )

        header_int = (
            (0 << 39)  # reserved
            | (0 << 38)  # payload_preamble
            | (0 << 37)  # null_frame
            | (1 << 36)  # sync_frame = 1
            | (0 << 35)  # startup_frame
            | (1 << 24)  # frame_id
            | (0 << 17)  # payload_length
            | (header_crc << 6)
            | (0 << 0)  # cycle_count
        )

        header_bytes = header_int.to_bytes(5, "big")
        payload = b""
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")

        frame_data = header_bytes + payload + crc_bytes
        frame = analyzer.parse_frame(frame_data)

        assert frame.header.sync_frame is True
        assert frame.header.startup_frame is False

    def test_parse_frame_with_startup_flag(self) -> None:
        """Test parsing frame with startup flag set."""
        analyzer = FlexRayAnalyzer()

        header_crc = calculate_header_crc(
            reserved=0,
            payload_preamble=0,
            null_frame=0,
            sync_frame=0,
            startup_frame=1,  # Startup frame
            frame_id=1,
            payload_length=0,
        )

        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (1 << 35)  # startup_frame = 1
            | (1 << 24)
            | (0 << 17)
            | (header_crc << 6)
            | (0 << 0)
        )

        header_bytes = header_int.to_bytes(5, "big")
        payload = b""
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")

        frame_data = header_bytes + payload + crc_bytes
        frame = analyzer.parse_frame(frame_data)

        assert frame.header.startup_frame is True

    def test_parse_frame_invalid_data_too_short(self) -> None:
        """Test parsing frame with insufficient data."""
        analyzer = FlexRayAnalyzer()

        # Only 5 bytes (need at least 8: 5 header + 0 payload + 3 CRC)
        with pytest.raises(ValueError, match="Frame data too short"):
            analyzer.parse_frame(bytes([0, 1, 2, 3, 4]))

    def test_parse_frame_invalid_frame_id(self) -> None:
        """Test parsing frame with invalid frame ID."""
        analyzer = FlexRayAnalyzer()

        # Frame ID = 0 (invalid, must be 1-2047)
        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 1, 0)  # Use valid ID for CRC

        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (0 << 24)  # Invalid frame_id = 0
            | (0 << 17)
            | (header_crc << 6)
            | (0 << 0)
        )

        header_bytes = header_int.to_bytes(5, "big")
        payload = b""
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")

        frame_data = header_bytes + payload + crc_bytes

        with pytest.raises(ValueError, match="Invalid frame ID"):
            analyzer.parse_frame(frame_data)

    def test_parse_frame_payload_length_mismatch(self) -> None:
        """Test parsing frame with payload length mismatch."""
        analyzer = FlexRayAnalyzer()

        # Header says 5 words (10 bytes), but provide only 5 bytes
        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 5)

        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (100 << 24)
            | (5 << 17)  # 5 words = 10 bytes
            | (header_crc << 6)
            | (0 << 0)
        )

        header_bytes = header_int.to_bytes(5, "big")
        payload = bytes([0, 1, 2, 3, 4])  # Only 5 bytes, not 10
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")

        frame_data = header_bytes + payload + crc_bytes

        with pytest.raises(ValueError, match="Payload length mismatch"):
            analyzer.parse_frame(frame_data)

    def test_segment_type_determination(self) -> None:
        """Test determining static vs dynamic segment."""
        config = {"static_slot_count": 100}
        analyzer = FlexRayAnalyzer(cluster_config=config)

        # Test static segment
        assert analyzer._determine_segment_type(50) == "static"
        assert analyzer._determine_segment_type(100) == "static"

        # Test dynamic segment
        assert analyzer._determine_segment_type(101) == "dynamic"
        assert analyzer._determine_segment_type(200) == "dynamic"

    def test_add_signal(self) -> None:
        """Test adding signal definitions."""
        analyzer = FlexRayAnalyzer()

        signal = FlexRaySignal(
            name="EngineSpeed",
            frame_id=100,
            start_bit=0,
            bit_length=16,
            factor=0.25,
            offset=0,
            unit="rpm",
        )

        analyzer.add_signal(signal)
        assert len(analyzer.signals) == 1
        assert analyzer.signals[0].name == "EngineSpeed"

    def test_decode_signals(self) -> None:
        """Test decoding signals from payload."""
        analyzer = FlexRayAnalyzer()

        # Add signal definition
        signal = FlexRaySignal(
            name="Speed",
            frame_id=100,
            start_bit=0,
            bit_length=16,
            byte_order="big_endian",
            factor=0.1,
            offset=0,
            unit="km/h",
        )
        analyzer.add_signal(signal)

        # Create payload with value 1000 (0x03E8)
        payload = bytes([0x03, 0xE8, 0x00, 0x00])

        # Decode
        decoded = analyzer.decode_signals(100, payload)

        assert "Speed" in decoded
        assert decoded["Speed"] == pytest.approx(100.0)  # 1000 * 0.1 = 100.0

    def test_decode_signals_with_offset(self) -> None:
        """Test decoding signals with offset."""
        analyzer = FlexRayAnalyzer()

        signal = FlexRaySignal(
            name="Temperature",
            frame_id=200,
            start_bit=0,
            bit_length=8,
            factor=1.0,
            offset=-40.0,  # Temperature offset
            unit="°C",
        )
        analyzer.add_signal(signal)

        # Raw value = 100, physical = 100 * 1.0 + (-40) = 60°C
        payload = bytes([100, 0, 0, 0])
        decoded = analyzer.decode_signals(200, payload)

        assert decoded["Temperature"] == pytest.approx(60.0)

    def test_decode_signals_little_endian(self) -> None:
        """Test decoding little-endian signals."""
        analyzer = FlexRayAnalyzer()

        signal = FlexRaySignal(
            name="Counter",
            frame_id=300,
            start_bit=0,
            bit_length=16,
            byte_order="little_endian",
            factor=1.0,
        )
        analyzer.add_signal(signal)

        # Little-endian 0x1234 = [0x34, 0x12]
        payload = bytes([0x34, 0x12, 0x00, 0x00])
        decoded = analyzer.decode_signals(300, payload)

        assert decoded["Counter"] == 0x1234

    def test_extract_signal_beyond_payload(self) -> None:
        """Test extracting signal that extends beyond payload."""
        analyzer = FlexRayAnalyzer()

        signal = FlexRaySignal(
            name="OutOfBounds",
            frame_id=100,
            start_bit=64,  # Beyond 4-byte payload
            bit_length=16,
        )
        analyzer.add_signal(signal)

        payload = bytes([0, 1, 2, 3])  # Only 4 bytes

        with pytest.raises(ValueError, match="extends beyond payload"):
            analyzer.decode_signals(100, payload)

    def test_get_frame_statistics(self) -> None:
        """Test getting frame statistics."""
        analyzer = FlexRayAnalyzer()

        # Create and parse multiple frames
        for i in range(5):
            header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100 + i, 0)
            header_int = (
                (0 << 39)
                | (0 << 38)
                | (0 << 37)
                | (0 << 36)
                | (0 << 35)
                | ((100 + i) << 24)
                | (0 << 17)
                | (header_crc << 6)
                | (0 << 0)
            )
            header_bytes = header_int.to_bytes(5, "big")
            payload = b""
            frame_crc = calculate_frame_crc(header_bytes, payload)
            crc_bytes = frame_crc.to_bytes(3, "big")
            frame_data = header_bytes + payload + crc_bytes

            channel = "A" if i % 2 == 0 else "B"
            analyzer.parse_frame(frame_data, timestamp=float(i), channel=channel)

        stats = analyzer.get_frame_statistics()

        assert stats["total_frames"] == 5
        assert stats["frames_by_channel"]["A"] == 3
        assert stats["frames_by_channel"]["B"] == 2
        assert stats["crc_errors"] == 0

    def test_multi_channel_support(self) -> None:
        """Test parsing frames from multiple channels."""
        analyzer = FlexRayAnalyzer()

        # Create frame
        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 0)
        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (100 << 24)
            | (0 << 17)
            | (header_crc << 6)
            | (0 << 0)
        )
        header_bytes = header_int.to_bytes(5, "big")
        payload = b""
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")
        frame_data = header_bytes + payload + crc_bytes

        # Parse on channel A
        frame_a = analyzer.parse_frame(frame_data, channel="A")
        assert frame_a.channel == "A"

        # Parse on channel B
        frame_b = analyzer.parse_frame(frame_data, channel="B")
        assert frame_b.channel == "B"

        assert len(analyzer.frames) == 2

    def test_cycle_count_extraction(self) -> None:
        """Test extracting cycle count from header."""
        analyzer = FlexRayAnalyzer()

        for cycle in [0, 31, 63]:  # Test various cycle counts
            header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, 0)
            header_int = (
                (0 << 39)
                | (0 << 38)
                | (0 << 37)
                | (0 << 36)
                | (0 << 35)
                | (100 << 24)
                | (0 << 17)
                | (header_crc << 6)
                | (cycle << 0)  # cycle_count
            )
            header_bytes = header_int.to_bytes(5, "big")
            payload = b""
            frame_crc = calculate_frame_crc(header_bytes, payload)
            crc_bytes = frame_crc.to_bytes(3, "big")
            frame_data = header_bytes + payload + crc_bytes

            frame = analyzer.parse_frame(frame_data)
            assert frame.header.cycle_count == cycle

    def test_maximum_payload_length(self) -> None:
        """Test parsing frame with maximum payload length."""
        analyzer = FlexRayAnalyzer()

        # Maximum: 127 words = 254 bytes
        max_payload_words = 127
        max_payload_bytes = 254

        header_crc = calculate_header_crc(0, 0, 0, 0, 0, 100, max_payload_words)
        header_int = (
            (0 << 39)
            | (0 << 38)
            | (0 << 37)
            | (0 << 36)
            | (0 << 35)
            | (100 << 24)
            | (max_payload_words << 17)
            | (header_crc << 6)
            | (0 << 0)
        )
        header_bytes = header_int.to_bytes(5, "big")
        payload = bytes(range(max_payload_bytes))
        frame_crc = calculate_frame_crc(header_bytes, payload)
        crc_bytes = frame_crc.to_bytes(3, "big")
        frame_data = header_bytes + payload + crc_bytes

        frame = analyzer.parse_frame(frame_data)
        assert frame.header.payload_length == max_payload_bytes
        assert len(frame.payload) == max_payload_bytes
