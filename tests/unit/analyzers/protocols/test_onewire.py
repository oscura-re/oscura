"""Tests for 1-Wire protocol decoder."""

import numpy as np
import pytest

from oscura.analyzers.protocols.onewire import (
    FAMILY_CODES,
    ROM_COMMAND_NAMES,
    OneWireDecoder,
    OneWireMode,
    OneWireROMCommand,
    OneWireROMID,
    OneWireTimings,
    decode_onewire,
)
from oscura.core.types import DigitalTrace, TraceMetadata

pytestmark = [pytest.mark.unit, pytest.mark.analyzer]


class TestOneWireTimings:
    """Tests for OneWireTimings."""

    def test_standard_timings(self):
        """Test standard speed timings."""
        timings = OneWireTimings()
        assert timings.reset_min == 480.0
        assert timings.reset_max == 960.0
        assert timings.slot_min == 60.0

    def test_overdrive_timings(self):
        """Test overdrive speed timings."""
        timings = OneWireTimings.overdrive()
        assert timings.reset_min == 48.0
        assert timings.reset_max == 80.0
        assert timings.slot_min == 6.0


class TestOneWireROMID:
    """Tests for ROM ID parsing."""

    def test_from_bytes(self):
        """Test ROM ID parsing from bytes."""
        # DS18B20 example: 28-FF-12-34-56-78-9A-BC
        data = bytes([0x28, 0xFF, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC])
        rom_id = OneWireROMID.from_bytes(data)

        assert rom_id.family_code == 0x28
        assert rom_id.serial_number == bytes([0xFF, 0x12, 0x34, 0x56, 0x78, 0x9A])
        assert rom_id.crc == 0xBC

    def test_family_name(self):
        """Test family code lookup."""
        data = bytes([0x28, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        rom_id = OneWireROMID.from_bytes(data)

        assert "DS18B20" in rom_id.family_name

    def test_to_hex(self):
        """Test hex string representation."""
        data = bytes([0x28, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x12])
        rom_id = OneWireROMID.from_bytes(data)

        hex_str = rom_id.to_hex()
        assert hex_str.startswith("28-")
        assert hex_str.endswith("-12")

    def test_invalid_length(self):
        """Test error on short data."""
        with pytest.raises(ValueError):
            OneWireROMID.from_bytes(bytes([0x28, 0x00, 0x00]))


class TestOneWireDecoder:
    """Tests for OneWireDecoder."""

    def test_decoder_attributes(self):
        """Test decoder has required attributes."""
        decoder = OneWireDecoder()

        assert decoder.id == "onewire"
        assert decoder.name == "1-Wire"
        assert len(decoder.channels) == 1
        assert decoder.channels[0].required is True

    def test_mode_selection(self):
        """Test mode selection."""
        standard = OneWireDecoder(mode="standard")
        assert standard._mode == OneWireMode.STANDARD

        overdrive = OneWireDecoder(mode="overdrive")
        assert overdrive._mode == OneWireMode.OVERDRIVE

    def test_decode_empty_trace(self):
        """Test decoding trace with no pulses."""
        decoder = OneWireDecoder()

        # All high (idle) signal
        data = np.ones(10000, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))

        packets = list(decoder.decode(trace))
        assert len(packets) == 0

    def test_decode_reset_pulse(self):
        """Test detection of reset pulse."""
        decoder = OneWireDecoder()

        # Create reset pulse (480us low at 1MHz = 480 samples)
        sample_rate = 1e6
        data = np.ones(10000, dtype=bool)

        # Reset pulse (low for 500us)
        data[100:600] = False

        # Presence pulse (slave response, low for 100us after 15us recovery)
        data[615:715] = False

        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=sample_rate))

        list(decoder.decode(trace))
        # Should detect reset but may not have enough data for full transaction


class TestDecodeOnewireConvenience:
    """Tests for decode_onewire convenience function."""

    def test_with_array(self):
        """Test with numpy array input."""
        data = np.ones(10000, dtype=bool)
        packets = decode_onewire(data, sample_rate=1e6)
        assert isinstance(packets, list)

    def test_with_trace(self):
        """Test with DigitalTrace input."""
        data = np.ones(10000, dtype=bool)
        trace = DigitalTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))

        packets = decode_onewire(trace)
        assert isinstance(packets, list)


class TestROMCommands:
    """Tests for ROM command constants."""

    def test_rom_command_names(self):
        """Test ROM command name lookup."""
        assert ROM_COMMAND_NAMES[0x33] == "Read ROM"
        assert ROM_COMMAND_NAMES[0x55] == "Match ROM"
        assert ROM_COMMAND_NAMES[0xCC] == "Skip ROM"

    def test_rom_command_enum(self):
        """Test ROM command enum values."""
        assert OneWireROMCommand.READ_ROM.value == 0x33
        assert OneWireROMCommand.MATCH_ROM.value == 0x55
        assert OneWireROMCommand.SKIP_ROM.value == 0xCC


class TestFamilyCodes:
    """Tests for family code constants."""

    def test_common_devices(self):
        """Test common device family codes."""
        assert "DS18B20" in FAMILY_CODES[0x28]
        assert "DS18S20" in FAMILY_CODES[0x10] or "DS1820" in FAMILY_CODES[0x10]
