"""Comprehensive tests for DBC file parser.

This module tests DBC file loading and message decoding.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.automotive]

from oscura.automotive.can.models import CANMessage
from oscura.automotive.dbc.parser import DBCParser, load_dbc


@pytest.fixture
def sample_dbc_content() -> str:
    """Sample DBC file content for testing."""
    return """VERSION ""

NS_ :
    NS_DESC_
    CM_
    BA_DEF_
    BA_
    VAL_
    CAT_DEF_
    CAT_
    FILTER
    BA_DEF_DEF_
    EV_DATA_
    ENVVAR_DATA_
    SGTYPE_
    SGTYPE_VAL_
    BA_DEF_SGTYPE_
    BA_SGTYPE_
    SIG_TYPE_REF_
    VAL_TABLE_
    SIG_GROUP_
    SIG_VALTYPE_
    SIGTYPE_VALTYPE_
    BO_TX_BU_
    CAT_
    FILTER

BS_:

BU_: ECU_1 ECU_2

BO_ 256 Engine_Status: 8 ECU_1
 SG_ Engine_RPM : 16|16@1+ (0.25,0) [0|8000] "rpm" ECU_2
 SG_ Engine_Temp : 32|8@1+ (1,-40) [-40|215] "degC" ECU_2
 SG_ Engine_Load : 40|8@1+ (0.392157,0) [0|100] "%" ECU_2

BO_ 512 Vehicle_Speed: 8 ECU_1
 SG_ Speed : 0|16@1+ (0.01,0) [0|655.35] "km/h" ECU_2
 SG_ Gear : 16|3@1+ (1,0) [0|7] "" ECU_2

CM_ SG_ 256 Engine_RPM "Engine speed in RPM";
CM_ SG_ 512 Speed "Vehicle speed";
"""


@pytest.fixture
def sample_dbc_file(sample_dbc_content) -> Path:
    """Create temporary DBC file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False) as f:
        f.write(sample_dbc_content)
        return Path(f.name)


class TestDBCParser:
    """Tests for DBCParser class."""

    def test_parser_creation(self, sample_dbc_file):
        """Test creating DBC parser."""
        parser = DBCParser(sample_dbc_file)
        assert parser is not None
        assert parser.db is not None

        # Clean up
        sample_dbc_file.unlink()

    def test_parser_nonexistent_file(self):
        """Test loading non-existent DBC file."""
        with pytest.raises(FileNotFoundError):
            DBCParser("nonexistent.dbc")

    def test_parser_missing_cantools(self, sample_dbc_file, monkeypatch):
        """Test error when cantools not installed."""
        # Mock import error
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cantools":
                raise ImportError("No module named 'cantools'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="cantools is required"):
            DBCParser(sample_dbc_file)

        # Clean up
        sample_dbc_file.unlink()

    def test_decode_message_basic(self, sample_dbc_file):
        """Test decoding a CAN message."""
        parser = DBCParser(sample_dbc_file)

        # Create message for ID 256 (Engine_Status)
        # RPM = 2000 rpm -> raw = 2000/0.25 = 8000 = 0x1F40
        # Temp = 85 degC -> raw = 85 - (-40) = 125 = 0x7D
        # Load = 50% -> raw = 50/0.392157 = ~127 = 0x7F
        data = bytearray(8)
        data[0] = 0x00
        data[1] = 0x00
        data[2] = 0x1F  # RPM high byte
        data[3] = 0x40  # RPM low byte
        data[4] = 0x7D  # Temp
        data[5] = 0x7F  # Load
        data[6] = 0x00
        data[7] = 0x00

        msg = CANMessage(
            arbitration_id=256,
            timestamp=1.0,
            data=bytes(data),
            is_extended=False,
        )

        decoded = parser.decode_message(msg)

        # Should have all 3 signals
        assert "Engine_RPM" in decoded
        assert "Engine_Temp" in decoded
        assert "Engine_Load" in decoded

        # Check RPM value (approximately 2000)
        assert 1990 < decoded["Engine_RPM"].value < 2010

        # Clean up
        sample_dbc_file.unlink()

    def test_decode_message_invalid_id(self, sample_dbc_file):
        """Test decoding message with unknown ID."""
        parser = DBCParser(sample_dbc_file)

        msg = CANMessage(
            arbitration_id=999,  # Not in DBC
            timestamp=1.0,
            data=bytes([0] * 8),
            is_extended=False,
        )

        with pytest.raises(KeyError, match="not found in DBC"):
            parser.decode_message(msg)

        # Clean up
        sample_dbc_file.unlink()

    def test_decode_message_invalid_data(self, sample_dbc_file):
        """Test decoding message with invalid data."""
        parser = DBCParser(sample_dbc_file)

        msg = CANMessage(
            arbitration_id=256,
            timestamp=1.0,
            data=bytes([0, 1]),  # Too short
            is_extended=False,
        )

        with pytest.raises(ValueError, match="Failed to decode"):
            parser.decode_message(msg)

        # Clean up
        sample_dbc_file.unlink()

    def test_get_message_ids(self, sample_dbc_file):
        """Test getting all message IDs from DBC."""
        parser = DBCParser(sample_dbc_file)

        message_ids = parser.get_message_ids()

        # Should have IDs 256 and 512
        assert 256 in message_ids
        assert 512 in message_ids
        assert len(message_ids) == 2

        # Clean up
        sample_dbc_file.unlink()

    def test_get_message_name_found(self, sample_dbc_file):
        """Test getting message name for existing ID."""
        parser = DBCParser(sample_dbc_file)

        name = parser.get_message_name(256)
        assert name == "Engine_Status"

        name = parser.get_message_name(512)
        assert name == "Vehicle_Speed"

        # Clean up
        sample_dbc_file.unlink()

    def test_get_message_name_not_found(self, sample_dbc_file):
        """Test getting message name for unknown ID."""
        parser = DBCParser(sample_dbc_file)

        name = parser.get_message_name(999)
        assert name is None

        # Clean up
        sample_dbc_file.unlink()

    def test_decoded_signal_structure(self, sample_dbc_file):
        """Test structure of decoded signals."""
        parser = DBCParser(sample_dbc_file)

        data = bytearray(8)
        data[2] = 0x10
        data[3] = 0x00  # RPM = 4000 * 0.25 = 1000

        msg = CANMessage(
            arbitration_id=256,
            timestamp=2.5,
            data=bytes(data),
            is_extended=False,
        )

        decoded = parser.decode_message(msg)
        signal = decoded["Engine_RPM"]

        # Check DecodedSignal structure
        assert signal.name == "Engine_RPM"
        assert signal.unit == "rpm"
        assert signal.timestamp == 2.5
        assert signal.definition is not None
        assert signal.definition.scale == 0.25

        # Clean up
        sample_dbc_file.unlink()

    def test_decode_multiple_messages(self, sample_dbc_file):
        """Test decoding multiple messages."""
        parser = DBCParser(sample_dbc_file)

        # Engine status message
        msg1 = CANMessage(
            arbitration_id=256,
            timestamp=1.0,
            data=bytes([0, 0, 0x10, 0x00, 0x50, 0x50, 0, 0]),
            is_extended=False,
        )

        # Vehicle speed message
        msg2 = CANMessage(
            arbitration_id=512,
            timestamp=1.0,
            data=bytes([0x50, 0x00, 0x03, 0, 0, 0, 0, 0]),
            is_extended=False,
        )

        decoded1 = parser.decode_message(msg1)
        decoded2 = parser.decode_message(msg2)

        assert "Engine_RPM" in decoded1
        assert "Speed" in decoded2

        # Clean up
        sample_dbc_file.unlink()


class TestLoadDBC:
    """Tests for load_dbc convenience function."""

    def test_load_dbc_basic(self, sample_dbc_file):
        """Test loading DBC file with convenience function."""
        parser = load_dbc(sample_dbc_file)

        assert parser is not None
        assert isinstance(parser, DBCParser)

        # Clean up
        sample_dbc_file.unlink()

    def test_load_dbc_path_object(self, sample_dbc_file):
        """Test loading with Path object."""
        parser = load_dbc(sample_dbc_file)

        assert parser is not None

        # Clean up
        sample_dbc_file.unlink()

    def test_load_dbc_string_path(self, sample_dbc_file):
        """Test loading with string path."""
        parser = load_dbc(str(sample_dbc_file))

        assert parser is not None

        # Clean up
        sample_dbc_file.unlink()


class TestDBCIntegration:
    """Integration tests for DBC parser with real scenarios."""

    def test_decode_sequence_of_messages(self, sample_dbc_file):
        """Test decoding a sequence of messages."""
        parser = DBCParser(sample_dbc_file)

        messages = []
        # Create 10 messages with increasing RPM
        for i in range(10):
            rpm = 1000 + i * 100
            raw_rpm = int(rpm / 0.25)

            data = bytearray(8)
            data[2] = (raw_rpm >> 8) & 0xFF
            data[3] = raw_rpm & 0xFF

            msg = CANMessage(
                arbitration_id=256,
                timestamp=i * 0.1,
                data=bytes(data),
                is_extended=False,
            )
            messages.append(msg)

        # Decode all messages
        decoded_values = []
        for msg in messages:
            decoded = parser.decode_message(msg)
            decoded_values.append(decoded["Engine_RPM"].value)

        # Values should increase
        for i in range(len(decoded_values) - 1):
            assert decoded_values[i + 1] > decoded_values[i]

        # Clean up
        sample_dbc_file.unlink()
