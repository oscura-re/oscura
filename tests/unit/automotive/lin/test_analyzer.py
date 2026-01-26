"""Unit tests for LIN protocol analyzer.

Tests comprehensive LIN 2.x protocol analysis including:
- Protected ID calculation with parity bits
- Classic and enhanced checksum validation
- Diagnostic frame parsing
- Signal decoding
- Schedule table inference
- LDF generation
"""

from pathlib import Path

import pytest

from oscura.automotive.lin import LINAnalyzer, LINSignal


class TestProtectedIDCalculation:
    """Test protected ID calculation with parity bits."""

    def test_calculate_protected_id_frame_0(self) -> None:
        """Test protected ID for frame ID 0."""
        analyzer = LINAnalyzer()
        protected_id = analyzer._calculate_protected_id(0)

        # Frame ID 0 (0b000000):
        # P0 = ID0 XOR ID1 XOR ID2 XOR ID4 = 0 XOR 0 XOR 0 XOR 0 = 0
        # P1 = NOT(ID1 XOR ID3 XOR ID4 XOR ID5) = NOT(0) = 1
        # Protected ID = 0b10000000 = 0x80
        assert protected_id == 0x80

    def test_calculate_protected_id_frame_1(self) -> None:
        """Test protected ID for frame ID 1."""
        analyzer = LINAnalyzer()
        protected_id = analyzer._calculate_protected_id(1)

        # Frame ID 1 (0b000001):
        # P0 = 1 XOR 0 XOR 0 XOR 0 = 1
        # P1 = NOT(0 XOR 0 XOR 0 XOR 0) = 1
        # Protected ID = 0b11000001 = 0xC1
        assert protected_id == 0xC1

    def test_calculate_protected_id_frame_2(self) -> None:
        """Test protected ID for frame ID 2."""
        analyzer = LINAnalyzer()
        protected_id = analyzer._calculate_protected_id(2)

        # Frame ID 2 (0b000010):
        # P0 = 0 XOR 1 XOR 0 XOR 0 = 1
        # P1 = NOT(1 XOR 0 XOR 0 XOR 0) = 0
        # Protected ID = 0b01000010 = 0x42
        assert protected_id == 0x42

    def test_calculate_protected_id_all_frame_ids(self) -> None:
        """Test protected ID calculation for all valid frame IDs (0-63)."""
        analyzer = LINAnalyzer()

        # Test all 64 valid frame IDs
        for frame_id in range(64):
            protected_id = analyzer._calculate_protected_id(frame_id)

            # Verify frame ID is preserved in lower 6 bits
            assert (protected_id & 0x3F) == frame_id

            # Verify protected ID is 8 bits
            assert 0 <= protected_id <= 0xFF

    def test_calculate_protected_id_invalid_frame_id(self) -> None:
        """Test that invalid frame IDs raise ValueError."""
        analyzer = LINAnalyzer()

        with pytest.raises(ValueError, match="exceeds 6 bits"):
            analyzer._calculate_protected_id(64)

        with pytest.raises(ValueError, match="exceeds 6 bits"):
            analyzer._calculate_protected_id(255)


class TestClassicChecksum:
    """Test classic checksum calculation (LIN 1.x)."""

    def test_calculate_classic_checksum_simple(self) -> None:
        """Test classic checksum with simple data."""
        analyzer = LINAnalyzer()
        checksum = analyzer._calculate_classic_checksum(b"\x01\x02\x03")

        # Sum = 1 + 2 + 3 = 6 = 0x06
        # Checksum = NOT(0x06) = 0xF9
        assert checksum == 0xF9

    def test_calculate_classic_checksum_with_carry(self) -> None:
        """Test classic checksum with carry propagation."""
        analyzer = LINAnalyzer()
        checksum = analyzer._calculate_classic_checksum(b"\xff\xff")

        # Sum = 255 + 255 = 510 = 0x01FE
        # Carry: 0xFE + 0x01 = 0xFF
        # Checksum = NOT(0xFF) = 0x00
        assert checksum == 0x00

    def test_calculate_classic_checksum_empty(self) -> None:
        """Test classic checksum with empty data."""
        analyzer = LINAnalyzer()
        checksum = analyzer._calculate_classic_checksum(b"")

        # Sum = 0
        # Checksum = NOT(0) = 0xFF
        assert checksum == 0xFF


class TestEnhancedChecksum:
    """Test enhanced checksum calculation (LIN 2.x)."""

    def test_calculate_enhanced_checksum_simple(self) -> None:
        """Test enhanced checksum with simple data."""
        analyzer = LINAnalyzer()
        protected_id = 0x80  # Frame ID 0
        checksum = analyzer._calculate_enhanced_checksum(protected_id, b"\x01\x02\x03")

        # Sum = 0x80 + 1 + 2 + 3 = 134 = 0x86
        # Checksum = NOT(0x86) = 0x79
        assert checksum == 0x79

    def test_calculate_enhanced_checksum_with_carry(self) -> None:
        """Test enhanced checksum with carry propagation."""
        analyzer = LINAnalyzer()
        protected_id = 0xFF
        checksum = analyzer._calculate_enhanced_checksum(protected_id, b"\xff")

        # Sum = 0xFF + 0xFF = 510
        # Carry: 0xFE + 1 = 0xFF
        # Checksum = NOT(0xFF) = 0x00
        assert checksum == 0x00

    def test_calculate_enhanced_checksum_frame_0(self) -> None:
        """Test enhanced checksum for frame ID 0 with known data."""
        analyzer = LINAnalyzer()
        protected_id = analyzer._calculate_protected_id(0)  # 0x80
        data = b"\x01\x02"

        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)

        # Sum = 0x80 + 0x01 + 0x02 = 131 = 0x83
        # Checksum = NOT(0x83) = 0x7C
        assert checksum == 0x7C


class TestFrameParsing:
    """Test LIN frame parsing."""

    def test_parse_frame_valid_enhanced_checksum(self) -> None:
        """Test parsing frame with valid enhanced checksum."""
        analyzer = LINAnalyzer()

        # Frame: sync=0x55, protected_id=0x80 (ID=0), data=0x01,0x02, checksum
        protected_id = analyzer._calculate_protected_id(0)  # 0x80
        data = b"\x01\x02"
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)  # 0x7C

        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])

        frame = analyzer.parse_frame(frame_bytes, timestamp=1.0)

        assert frame.frame_id == 0
        assert frame.data == b"\x01\x02"
        assert frame.checksum == checksum
        assert frame.checksum_valid is True
        assert frame.checksum_type == "enhanced"
        assert frame.parity_bits == (protected_id >> 6)
        assert frame.timestamp == 1.0

    def test_parse_frame_valid_classic_checksum(self) -> None:
        """Test parsing frame with valid classic checksum."""
        analyzer = LINAnalyzer()

        # Frame with classic checksum
        protected_id = analyzer._calculate_protected_id(1)  # 0xC1
        data = b"\x10\x20"
        checksum = analyzer._calculate_classic_checksum(data)  # NOT(0x30) = 0xCF

        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])

        frame = analyzer.parse_frame(frame_bytes, timestamp=2.0, checksum_type="classic")

        assert frame.frame_id == 1
        assert frame.data == b"\x10\x20"
        assert frame.checksum_valid is True
        assert frame.checksum_type == "classic"

    def test_parse_frame_invalid_checksum(self) -> None:
        """Test parsing frame with invalid checksum."""
        analyzer = LINAnalyzer()

        # Frame with incorrect checksum
        protected_id = analyzer._calculate_protected_id(0)
        data = b"\x01\x02"
        wrong_checksum = 0xFF  # Incorrect

        frame_bytes = bytes([0x55, protected_id]) + data + bytes([wrong_checksum])

        frame = analyzer.parse_frame(frame_bytes, timestamp=1.0)

        assert frame.checksum_valid is False
        assert frame.checksum == wrong_checksum

    def test_parse_frame_invalid_sync(self) -> None:
        """Test that invalid sync byte raises ValueError."""
        analyzer = LINAnalyzer()

        frame_bytes = b"\xaa\x80\x01\x02\xfa"  # Wrong sync byte

        with pytest.raises(ValueError, match="Invalid sync byte"):
            analyzer.parse_frame(frame_bytes)

    def test_parse_frame_too_short(self) -> None:
        """Test that too-short frame raises ValueError."""
        analyzer = LINAnalyzer()

        frame_bytes = b"\x55\x80\x01"  # Only 3 bytes

        with pytest.raises(ValueError, match="too short"):
            analyzer.parse_frame(frame_bytes)

    def test_parse_frame_invalid_parity(self) -> None:
        """Test that invalid protected ID parity raises ValueError."""
        analyzer = LINAnalyzer()

        # Use wrong protected ID (no parity bits)
        frame_bytes = b"\x55\x00\x01\x02\xfa"  # Protected ID should be 0x80, not 0x00

        with pytest.raises(ValueError, match="Invalid protected ID parity"):
            analyzer.parse_frame(frame_bytes)

    def test_parse_frame_stores_in_list(self) -> None:
        """Test that parsed frames are stored in analyzer."""
        analyzer = LINAnalyzer()

        # Parse multiple frames
        for frame_id in range(3):
            protected_id = analyzer._calculate_protected_id(frame_id)
            data = bytes([frame_id])
            checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
            frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])

            analyzer.parse_frame(frame_bytes, timestamp=float(frame_id))

        assert len(analyzer.frames) == 3
        assert analyzer.frames[0].frame_id == 0
        assert analyzer.frames[1].frame_id == 1
        assert analyzer.frames[2].frame_id == 2


class TestDiagnosticFrames:
    """Test diagnostic frame parsing."""

    def test_parse_diagnostic_master_request(self) -> None:
        """Test parsing master request diagnostic frame (0x3C)."""
        analyzer = LINAnalyzer()

        # Diagnostic frame: 0x3C (master request)
        # NAD=0x01, PCI=0x06, SID=0xB6 (ReadById), ID=0x0001
        protected_id = analyzer._calculate_protected_id(0x3C)
        data = b"\x01\x06\xb6\x00\x01\x00\x00\x00"
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)

        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])

        frame = analyzer.parse_frame(frame_bytes, timestamp=1.0)

        assert frame.is_diagnostic is True
        assert frame.frame_id == 0x3C
        assert frame.decoded_signals["nad"] == 0x01
        assert frame.decoded_signals["service_id"] == 0xB6
        assert frame.decoded_signals["service_name"] == "ReadById"
        assert frame.decoded_signals["frame_type"] == "MasterRequest"
        assert frame.decoded_signals["identifier"] == 0x0001

    def test_parse_diagnostic_slave_response(self) -> None:
        """Test parsing slave response diagnostic frame (0x3D)."""
        analyzer = LINAnalyzer()

        # Diagnostic frame: 0x3D (slave response)
        # NAD=0x01, PCI=0x06, SID=0xF6 (positive response to ReadById)
        protected_id = analyzer._calculate_protected_id(0x3D)
        data = b"\x01\x06\xf6\x12\x34\x00\x00\x00"
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)

        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])

        frame = analyzer.parse_frame(frame_bytes, timestamp=2.0)

        assert frame.is_diagnostic is True
        assert frame.frame_id == 0x3D
        assert frame.decoded_signals["frame_type"] == "SlaveResponse"

    def test_parse_diagnostic_assign_nad(self) -> None:
        """Test parsing AssignNAD diagnostic service."""
        analyzer = LINAnalyzer()

        protected_id = analyzer._calculate_protected_id(0x3C)
        data = b"\x7f\x06\xb1\x01\x02\x03\x04\x05"  # SID=0xB1 (AssignNAD)
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)

        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])

        frame = analyzer.parse_frame(frame_bytes, timestamp=1.0)

        assert frame.decoded_signals["service_name"] == "AssignNAD"


class TestSignalDecoding:
    """Test signal decoding from frame data."""

    def test_add_signal(self) -> None:
        """Test adding signal definition."""
        analyzer = LINAnalyzer()

        signal = LINSignal(
            name="EngineSpeed",
            frame_id=0x10,
            start_bit=0,
            bit_length=16,
            publisher="Master",
        )

        analyzer.add_signal(signal)

        assert len(analyzer.signals) == 1
        assert analyzer.signals[0].name == "EngineSpeed"

    def test_decode_single_signal(self) -> None:
        """Test decoding single signal from frame data."""
        analyzer = LINAnalyzer()

        # Add signal: 16-bit value at bit 0
        analyzer.add_signal(
            LINSignal(name="Speed", frame_id=0, start_bit=0, bit_length=16, publisher="Master")
        )

        # Data: 0x1027 (little-endian) = 10000 decimal
        decoded = analyzer.decode_signals(0, b"\x10\x27")

        assert "Speed" in decoded
        assert decoded["Speed"] == 0x2710  # 10000

    def test_decode_multiple_signals(self) -> None:
        """Test decoding multiple signals from same frame."""
        analyzer = LINAnalyzer()

        # Add multiple signals
        analyzer.add_signal(
            LINSignal(name="Signal1", frame_id=0, start_bit=0, bit_length=8, publisher="Master")
        )
        analyzer.add_signal(
            LINSignal(name="Signal2", frame_id=0, start_bit=8, bit_length=8, publisher="Master")
        )
        analyzer.add_signal(
            LINSignal(name="Signal3", frame_id=0, start_bit=16, bit_length=4, publisher="Master")
        )

        # Data: [0x12, 0x34, 0x56]
        decoded = analyzer.decode_signals(0, b"\x12\x34\x56")

        assert decoded["Signal1"] == 0x12
        assert decoded["Signal2"] == 0x34
        assert decoded["Signal3"] == 0x06  # Lower 4 bits of 0x56

    def test_decode_signals_empty_data(self) -> None:
        """Test decoding signals from empty data."""
        analyzer = LINAnalyzer()

        analyzer.add_signal(
            LINSignal(name="Signal", frame_id=0, start_bit=0, bit_length=8, publisher="Master")
        )

        decoded = analyzer.decode_signals(0, b"")

        assert decoded == {}

    def test_decode_signals_no_matching_frame(self) -> None:
        """Test decoding when no signals match frame ID."""
        analyzer = LINAnalyzer()

        analyzer.add_signal(
            LINSignal(name="Signal", frame_id=1, start_bit=0, bit_length=8, publisher="Master")
        )

        # Request decoding for frame ID 0 (no signals defined)
        decoded = analyzer.decode_signals(0, b"\x12\x34")

        assert decoded == {}


class TestScheduleTableInference:
    """Test schedule table inference from frame timing."""

    def test_infer_schedule_table_simple(self) -> None:
        """Test inferring schedule table from regular frames."""
        analyzer = LINAnalyzer()

        # Create frames with regular timing: Frame 0 every 10ms, Frame 1 every 20ms
        timestamps = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]  # 10ms intervals
        frame_ids = [0, 1, 0, 1, 0, 1]

        for ts, fid in zip(timestamps, frame_ids, strict=False):
            protected_id = analyzer._calculate_protected_id(fid)
            data = bytes([fid])
            checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
            frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
            analyzer.parse_frame(frame_bytes, timestamp=ts)

        schedule = analyzer.infer_schedule_table()

        # Should have entries for both frame IDs
        assert len(schedule) >= 2
        frame_ids_in_schedule = {entry.frame_id for entry in schedule}
        assert 0 in frame_ids_in_schedule
        assert 1 in frame_ids_in_schedule

    def test_infer_schedule_table_no_frames(self) -> None:
        """Test inferring schedule table with no frames."""
        analyzer = LINAnalyzer()

        schedule = analyzer.infer_schedule_table()

        assert schedule == []

    def test_infer_schedule_table_single_frame(self) -> None:
        """Test inferring schedule table with single frame."""
        analyzer = LINAnalyzer()

        protected_id = analyzer._calculate_protected_id(0)
        data = b"\x01"
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
        analyzer.parse_frame(frame_bytes, timestamp=1.0)

        schedule = analyzer.infer_schedule_table()

        assert schedule == []


class TestLDFGeneration:
    """Test LDF (LIN Description File) generation."""

    def test_generate_ldf_basic(self, tmp_path: Path) -> None:
        """Test basic LDF generation."""
        analyzer = LINAnalyzer()

        # Add some signals
        analyzer.add_signal(
            LINSignal(
                name="EngineSpeed",
                frame_id=0x10,
                start_bit=0,
                bit_length=16,
                publisher="Master",
            )
        )
        analyzer.add_signal(
            LINSignal(
                name="VehicleSpeed", frame_id=0x20, start_bit=0, bit_length=8, publisher="Slave1"
            )
        )

        # Parse some frames
        for frame_id in [0x10, 0x20]:
            protected_id = analyzer._calculate_protected_id(frame_id)
            data = bytes([frame_id, 0x00])
            checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
            frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
            analyzer.parse_frame(frame_bytes, timestamp=float(frame_id) / 1000.0)

        # Generate LDF
        ldf_path = tmp_path / "test.ldf"
        analyzer.generate_ldf(ldf_path, baudrate=19200)

        # Verify file was created
        assert ldf_path.exists()

        # Read and verify content
        ldf_content = ldf_path.read_text()

        assert "LIN_description_file;" in ldf_content
        assert 'LIN_protocol_version = "2.1";' in ldf_content
        assert "LIN_speed = 19.2 kbps;" in ldf_content
        assert "Nodes {" in ldf_content
        assert "Master: Master" in ldf_content
        assert "Slaves: Slave1" in ldf_content
        assert "Signals {" in ldf_content
        assert "EngineSpeed:" in ldf_content
        assert "VehicleSpeed:" in ldf_content
        assert "Frames {" in ldf_content
        assert "Frame_10:" in ldf_content
        assert "Frame_20:" in ldf_content

    def test_generate_ldf_with_schedule(self, tmp_path: Path) -> None:
        """Test LDF generation with schedule table."""
        analyzer = LINAnalyzer()

        # Parse frames with timing
        timestamps = [0.0, 0.01, 0.02, 0.03]
        frame_ids = [0x10, 0x20, 0x10, 0x20]

        for ts, fid in zip(timestamps, frame_ids, strict=False):
            protected_id = analyzer._calculate_protected_id(fid)
            data = bytes([fid])
            checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
            frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
            analyzer.parse_frame(frame_bytes, timestamp=ts)

        # Generate LDF
        ldf_path = tmp_path / "scheduled.ldf"
        analyzer.generate_ldf(ldf_path)

        # Verify schedule table is present
        ldf_content = ldf_path.read_text()
        assert "Schedule_tables {" in ldf_content
        assert "NormalTable {" in ldf_content
        assert "delay" in ldf_content

    def test_generate_ldf_different_baudrate(self, tmp_path: Path) -> None:
        """Test LDF generation with different baudrate."""
        analyzer = LINAnalyzer()

        # Add minimal frame
        protected_id = analyzer._calculate_protected_id(0)
        data = b"\x01"
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
        analyzer.parse_frame(frame_bytes, timestamp=1.0)

        # Generate LDF with 9600 baud
        ldf_path = tmp_path / "slow.ldf"
        analyzer.generate_ldf(ldf_path, baudrate=9600)

        ldf_content = ldf_path.read_text()
        assert "LIN_speed = 9.6 kbps;" in ldf_content

    def test_generate_ldf_no_frames_raises_error(self, tmp_path: Path) -> None:
        """Test that generating LDF without frames raises ValueError."""
        analyzer = LINAnalyzer()

        ldf_path = tmp_path / "empty.ldf"

        with pytest.raises(ValueError, match="No frames captured"):
            analyzer.generate_ldf(ldf_path)

    def test_generate_ldf_skips_diagnostic_frames(self, tmp_path: Path) -> None:
        """Test that diagnostic frames are excluded from LDF."""
        analyzer = LINAnalyzer()

        # Parse regular frame
        protected_id = analyzer._calculate_protected_id(0x10)
        data = b"\x01"
        checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
        frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
        analyzer.parse_frame(frame_bytes, timestamp=1.0)

        # Parse diagnostic frame (should be skipped)
        protected_id_diag = analyzer._calculate_protected_id(0x3C)
        data_diag = b"\x01\x06\xb6\x00\x01\x00\x00\x00"
        checksum_diag = analyzer._calculate_enhanced_checksum(protected_id_diag, data_diag)
        frame_bytes_diag = bytes([0x55, protected_id_diag]) + data_diag + bytes([checksum_diag])
        analyzer.parse_frame(frame_bytes_diag, timestamp=2.0)

        # Generate LDF
        ldf_path = tmp_path / "no_diag.ldf"
        analyzer.generate_ldf(ldf_path)

        ldf_content = ldf_path.read_text()

        # Should have Frame_10 but not Frame_3C
        assert "Frame_10:" in ldf_content
        assert "Frame_3C:" not in ldf_content


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow: parse frames, decode signals, generate LDF."""
        analyzer = LINAnalyzer()

        # Define signals
        analyzer.add_signal(
            LINSignal(
                name="EngineSpeed",
                frame_id=0x10,
                start_bit=0,
                bit_length=16,
                init_value=0,
                publisher="Master",
            )
        )
        analyzer.add_signal(
            LINSignal(
                name="Throttle",
                frame_id=0x10,
                start_bit=16,
                bit_length=8,
                init_value=0,
                publisher="Master",
            )
        )

        # Simulate captured LIN traffic
        test_data = [
            (0.000, 0x10, b"\x10\x27\x64"),  # Engine speed = 10000, Throttle = 100
            (0.010, 0x10, b"\x20\x4e\x80"),  # Engine speed = 20000, Throttle = 128
            (0.020, 0x10, b"\x30\x75\xff"),  # Engine speed = 30000, Throttle = 255
        ]

        for ts, frame_id, data in test_data:
            protected_id = analyzer._calculate_protected_id(frame_id)
            checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
            frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
            frame = analyzer.parse_frame(frame_bytes, timestamp=ts)

            # Verify signal decoding
            assert "EngineSpeed" in frame.decoded_signals
            assert "Throttle" in frame.decoded_signals

        # Verify frames were captured
        assert len(analyzer.frames) == 3

        # Infer schedule
        schedule = analyzer.infer_schedule_table()
        assert len(schedule) > 0

        # Generate LDF
        ldf_path = tmp_path / "vehicle.ldf"
        analyzer.generate_ldf(ldf_path, baudrate=19200)

        # Verify LDF content
        ldf_content = ldf_path.read_text()
        assert "EngineSpeed: 16, 0, Master;" in ldf_content
        assert "Throttle: 8, 0, Master;" in ldf_content
        assert "Frame_10: 16, Master, 3 {" in ldf_content

    def test_multiple_frame_ids(self) -> None:
        """Test parsing multiple different frame IDs."""
        analyzer = LINAnalyzer()

        frame_ids = [0x00, 0x01, 0x10, 0x20, 0x3F]  # Range of valid IDs

        for frame_id in frame_ids:
            protected_id = analyzer._calculate_protected_id(frame_id)
            data = bytes([frame_id])
            checksum = analyzer._calculate_enhanced_checksum(protected_id, data)
            frame_bytes = bytes([0x55, protected_id]) + data + bytes([checksum])
            frame = analyzer.parse_frame(frame_bytes, timestamp=float(frame_id))

            assert frame.frame_id == frame_id
            assert frame.checksum_valid is True

        assert len(analyzer.detected_frame_ids) == len(frame_ids)
        assert analyzer.detected_frame_ids == set(frame_ids)
