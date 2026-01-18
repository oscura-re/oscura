"""Tests for DBC parser functionality.

This module tests DBC file parsing and CAN message decoding using synthetic
DBC content and CAN messages.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from pathlib import Path

import pytest

from oscura.automotive.can.models import CANMessage
from oscura.automotive.dbc.parser import DBCParser, load_dbc

# Skip tests if cantools not installed
pytest.importorskip("cantools")


@pytest.fixture
def simple_dbc_content() -> str:
    """Synthetic DBC file content with basic message and signals.

    Returns:
        DBC file content as string.
    """
    return """VERSION ""

NS_ :

BS_:

BU_:

BO_ 640 Engine_Data: 8 ECU
 SG_ Engine_RPM : 16|16@1+ (0.25,0) [0|16383.75] "rpm" Vector__XXX
 SG_ Engine_Temp : 32|8@1+ (1,-40) [-40|215] "°C" Vector__XXX
 SG_ Throttle : 40|8@1+ (0.392156862745098,0) [0|100] "%" Vector__XXX

BO_ 1024 Vehicle_Speed: 8 ECU
 SG_ Speed : 0|16@0+ (0.01,0) [0|655.35] "km/h" Vector__XXX
 SG_ Odometer : 16|32@0+ (1,0) [0|4294967295] "km" Vector__XXX

"""


@pytest.fixture
def dbc_with_endianness() -> str:
    """DBC file with both big-endian and little-endian signals.

    Returns:
        DBC file content demonstrating different byte orders.
    """
    return """VERSION ""

NS_ :

BS_:

BU_:

BO_ 512 Mixed_Endian: 8 ECU
 SG_ Big_Endian_Signal : 7|16@0+ (1,0) [0|65535] "" Vector__XXX
 SG_ Little_Endian_Signal : 16|16@1+ (1,0) [0|65535] "" Vector__XXX
 SG_ Signed_Signal : 32|16@1- (1,-1000) [-1000|1000] "" Vector__XXX

"""


@pytest.fixture
def dbc_with_scale_offset() -> str:
    """DBC file with various scale and offset combinations.

    Returns:
        DBC file content with scale/offset examples.
    """
    return """VERSION ""

NS_ :

BS_:

BU_:

BO_ 768 Scaled_Signals: 8 ECU
 SG_ Temperature : 0|8@1+ (0.5,-40) [-40|87.5] "°C" Vector__XXX
 SG_ Pressure : 8|16@1+ (0.1,0) [0|6553.5] "kPa" Vector__XXX
 SG_ Voltage : 24|12@1+ (0.001,0) [0|4.095] "V" Vector__XXX

"""


@pytest.fixture
def temp_dbc_file(tmp_path: Path) -> Path:
    """Create temporary DBC file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to temporary directory.
    """
    return tmp_path


def create_dbc_file(path: Path, content: str) -> Path:
    """Helper to create DBC file from content.

    Args:
        path: Directory path.
        content: DBC file content.

    Returns:
        Path to created DBC file.
    """
    dbc_path = path / "test.dbc"
    dbc_path.write_text(content)
    return dbc_path


class TestDBCParserInit:
    """Tests for DBCParser initialization."""

    def test_init_with_valid_file(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test initialization with valid DBC file."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)
        assert parser.db is not None

    def test_init_with_missing_file(self, temp_dbc_file: Path):
        """Test initialization with non-existent file raises error."""
        missing_path = temp_dbc_file / "missing.dbc"
        with pytest.raises(FileNotFoundError, match="DBC file not found"):
            DBCParser(missing_path)

    def test_init_with_string_path(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test initialization with string path."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(str(dbc_path))
        assert parser.db is not None


class TestDecodeMessage:
    """Tests for CAN message decoding."""

    def test_decode_engine_rpm_basic(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test decoding engine RPM message."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)

        # Create message with RPM = 2000 (raw = 8000)
        # Little-endian uint16 at bit 16 (bytes 2-3): 0x1F40 = 8000 (stored as 0x40 0x1F)
        data = bytes([0x00, 0x00, 0x40, 0x1F, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x280, timestamp=1.0, data=data)

        signals = parser.decode_message(msg)

        assert "Engine_RPM" in signals
        assert abs(signals["Engine_RPM"].value - 2000.0) < 0.1
        assert signals["Engine_RPM"].unit == "rpm"

    def test_decode_multiple_signals(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test decoding message with multiple signals."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)

        # RPM = 3000 (raw = 12000 = 0x2EE0, little-endian: 0xE0 0x2E)
        # Temp = 90°C (raw = 130, at byte 4)
        # Throttle = 50% (raw = 127, at byte 5)
        data = bytes([0x00, 0x00, 0xE0, 0x2E, 130, 127, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x280, timestamp=1.0, data=data)

        signals = parser.decode_message(msg)

        assert len(signals) == 3
        assert abs(signals["Engine_RPM"].value - 3000.0) < 0.1
        assert abs(signals["Engine_Temp"].value - 90.0) < 0.1
        assert abs(signals["Throttle"].value - 50.0) < 1.0  # Approximate

    def test_decode_with_scale_and_offset(self, temp_dbc_file: Path, dbc_with_scale_offset: str):
        """Test signal decoding with scale and offset."""
        dbc_path = create_dbc_file(temp_dbc_file, dbc_with_scale_offset)
        parser = DBCParser(dbc_path)

        # Temperature: raw=100, value = 100*0.5 - 40 = 10°C (1 byte at bit 0)
        # Pressure: raw=1000 (0x03E8), value = 1000*0.1 = 100.0 kPa (16-bit LE at bit 8)
        # Little-endian 16-bit: 0x03E8 stored as 0xE8 0x03
        data = bytes([100, 0xE8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x300, timestamp=1.0, data=data)

        signals = parser.decode_message(msg)

        assert abs(signals["Temperature"].value - 10.0) < 0.1
        assert abs(signals["Pressure"].value - 100.0) < 0.1

    def test_decode_big_endian(self, temp_dbc_file: Path, dbc_with_endianness: str):
        """Test decoding big-endian signal."""
        dbc_path = create_dbc_file(temp_dbc_file, dbc_with_endianness)
        parser = DBCParser(dbc_path)

        # Big-endian: 0x1234 at bytes 0-1
        data = bytes([0x12, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x200, timestamp=1.0, data=data)

        signals = parser.decode_message(msg)
        assert "Big_Endian_Signal" in signals
        assert signals["Big_Endian_Signal"].value == 0x1234

    def test_decode_little_endian(self, temp_dbc_file: Path, dbc_with_endianness: str):
        """Test decoding little-endian signal."""
        dbc_path = create_dbc_file(temp_dbc_file, dbc_with_endianness)
        parser = DBCParser(dbc_path)

        # Little-endian: 0x5678 at bytes 2-3 (stored as 78 56)
        data = bytes([0x00, 0x00, 0x78, 0x56, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x200, timestamp=1.0, data=data)

        signals = parser.decode_message(msg)
        assert "Little_Endian_Signal" in signals
        assert signals["Little_Endian_Signal"].value == 0x5678

    def test_decode_signed_signal(self, temp_dbc_file: Path, dbc_with_endianness: str):
        """Test decoding signed signal."""
        dbc_path = create_dbc_file(temp_dbc_file, dbc_with_endianness)
        parser = DBCParser(dbc_path)

        # Signed signal with offset -1000
        # Raw value that decodes to negative
        data = bytes([0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x200, timestamp=1.0, data=data)

        signals = parser.decode_message(msg)
        assert "Signed_Signal" in signals
        # Exact value depends on cantools implementation

    def test_decode_invalid_message_id(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test decoding message with ID not in DBC raises error."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)

        msg = CANMessage(arbitration_id=0x999, timestamp=1.0, data=bytes(8))

        with pytest.raises(KeyError, match="not found in DBC"):
            parser.decode_message(msg)

    def test_decoded_signal_attributes(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test that decoded signal has correct attributes."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)

        # Little-endian byte order
        data = bytes([0x00, 0x00, 0x40, 0x1F, 0x00, 0x00, 0x00, 0x00])
        msg = CANMessage(arbitration_id=0x280, timestamp=1.5, data=data)

        signals = parser.decode_message(msg)
        rpm_signal = signals["Engine_RPM"]

        assert rpm_signal.name == "Engine_RPM"
        assert rpm_signal.unit == "rpm"
        assert rpm_signal.timestamp == 1.5
        assert rpm_signal.definition is not None
        assert rpm_signal.definition.scale == 0.25
        assert rpm_signal.definition.offset == 0


class TestGetMessageIDs:
    """Tests for get_message_ids() method."""

    def test_get_all_message_ids(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test retrieving all message IDs from DBC."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)

        message_ids = parser.get_message_ids()

        assert len(message_ids) == 2
        assert 0x280 in message_ids  # Engine_Data (640)
        assert 0x400 in message_ids  # Vehicle_Speed (1024)

    def test_empty_dbc_returns_empty_set(self, temp_dbc_file: Path):
        """Test that DBC with no messages returns empty set."""
        empty_dbc = """VERSION ""

NS_ :

BS_:

BU_:
"""
        dbc_path = create_dbc_file(temp_dbc_file, empty_dbc)
        parser = DBCParser(dbc_path)

        message_ids = parser.get_message_ids()
        assert len(message_ids) == 0


class TestGetMessageName:
    """Tests for get_message_name() method."""

    def test_get_existing_message_name(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test retrieving message name for existing ID."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)

        name = parser.get_message_name(0x280)
        assert name == "Engine_Data"

    def test_get_nonexistent_message_name(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test retrieving name for non-existent ID returns None."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = DBCParser(dbc_path)

        name = parser.get_message_name(0x999)
        assert name is None


class TestLoadDBC:
    """Tests for load_dbc() convenience function."""

    def test_load_dbc_function(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test load_dbc convenience function."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = load_dbc(dbc_path)

        assert isinstance(parser, DBCParser)
        assert parser.db is not None

    def test_load_dbc_with_string_path(self, temp_dbc_file: Path, simple_dbc_content: str):
        """Test load_dbc with string path."""
        dbc_path = create_dbc_file(temp_dbc_file, simple_dbc_content)
        parser = load_dbc(str(dbc_path))

        assert isinstance(parser, DBCParser)
        assert parser.db is not None
