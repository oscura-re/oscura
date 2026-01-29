"""Tests for configurable binary packet loader.

Tests coverage for:
"""

import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from oscura.core.exceptions import ConfigurationError
from oscura.loaders.configurable import (
    BitfieldExtractor,
    ConfigurablePacketLoader,
    DeviceConfig,
    DeviceMapper,
    HeaderFieldDef,
    PacketFormatConfig,
    SampleFormatDef,
    detect_source_type,
    extract_channels,
    load_binary_packets,
)
from oscura.loaders.preprocessing import (
    detect_idle_regions,
    get_idle_statistics,
    trim_idle,
)
from oscura.loaders.validation import PacketValidator

pytestmark = [pytest.mark.unit, pytest.mark.loader]


class TestBitfieldExtractor:
    """Test bitfield extraction functionality.

    Tests: BDL-001
    """

    def test_extract_single_bit(self) -> None:
        """Test extracting a single bit."""
        extractor = BitfieldExtractor()

        # Test value: 0b10101100 = 0xAC
        value = 0xAC

        assert extractor.extract_bit(value, 0) == 0  # LSB
        assert extractor.extract_bit(value, 1) == 0
        assert extractor.extract_bit(value, 2) == 1
        assert extractor.extract_bit(value, 3) == 1
        assert extractor.extract_bit(value, 4) == 0
        assert extractor.extract_bit(value, 5) == 1
        assert extractor.extract_bit(value, 6) == 0
        assert extractor.extract_bit(value, 7) == 1  # MSB

    def test_extract_bit_range(self) -> None:
        """Test extracting a range of bits."""
        extractor = BitfieldExtractor()

        # Test value: 0b10101100 = 0xAC
        value = 0xAC

        # Lower nibble (bits 0-3)
        assert extractor.extract_bits(value, 0, 3) == 0b1100  # 12

        # Upper nibble (bits 4-7)
        assert extractor.extract_bits(value, 4, 7) == 0b1010  # 10

        # Middle 4 bits (bits 2-5)
        assert extractor.extract_bits(value, 2, 5) == 0b1011  # 11

    def test_extract_full_byte(self) -> None:
        """Test extracting all 8 bits."""
        extractor = BitfieldExtractor()

        value = 0xAC
        assert extractor.extract_bits(value, 0, 7) == 0xAC


class TestHeaderFieldDef:
    """Test header field definition.

    Tests: BDL-001
    """

    def test_valid_field_definition(self) -> None:
        """Test creating a valid field definition."""
        field = HeaderFieldDef(
            name="sequence",
            offset=0,
            size=4,
            type="uint32",
            endian="big",
        )

        assert field.name == "sequence"
        assert field.offset == 0
        assert field.size == 4
        assert field.type == "uint32"
        assert field.endian == "big"

    def test_negative_offset_raises_error(self) -> None:
        """Test that negative offset raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            HeaderFieldDef(
                name="bad_field",
                offset=-1,
                size=4,
                type="uint32",
            )

        assert "offset must be non-negative" in str(exc_info.value)

    def test_zero_size_raises_error(self) -> None:
        """Test that zero size raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            HeaderFieldDef(
                name="bad_field",
                offset=0,
                size=0,
                type="uint32",
            )

        assert "size must be positive" in str(exc_info.value)

    def test_invalid_endian_raises_error(self) -> None:
        """Test that invalid endianness raises error."""
        with pytest.raises(ConfigurationError) as exc_info:
            HeaderFieldDef(
                name="bad_field",
                offset=0,
                size=4,
                type="uint32",
                endian="invalid",
            )

        assert "Invalid endianness" in str(exc_info.value)


class TestPacketFormatConfig:
    """Test packet format configuration.

    Tests: BDL-001
    """

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Test loading configuration from YAML file."""
        config_file = tmp_path / "test_format.yaml"

        config_data = {
            "name": "test_format",
            "version": "1.0",
            "description": "Test packet format",
            "packet": {"size": 64, "byte_order": "big"},
            "header": {
                "size": 8,
                "fields": [
                    {
                        "name": "sequence",
                        "offset": 0,
                        "size": 4,
                        "type": "uint32",
                        "endian": "big",
                    },
                    {
                        "name": "device_id",
                        "offset": 4,
                        "size": 1,
                        "type": "uint8",
                    },
                ],
            },
            "samples": {
                "offset": 8,
                "count": 7,
                "format": {"size": 8, "type": "uint64", "endian": "little"},
            },
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = PacketFormatConfig.from_yaml(config_file)

        assert config.name == "test_format"
        assert config.version == "1.0"
        assert config.packet_size == 64
        assert config.byte_order == "big"
        assert config.header_size == 8
        assert len(config.header_fields) == 2
        assert config.sample_offset == 8
        assert config.sample_count == 7

    def test_missing_required_fields_raises_error(self, tmp_path: Path) -> None:
        """Test that missing required fields raises error."""
        config_file = tmp_path / "bad_config.yaml"

        # Missing 'samples' section
        config_data = {
            "name": "incomplete",
            "version": "1.0",
            "packet": {"size": 64},
            "header": {"size": 8, "fields": []},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigurationError) as exc_info:
            PacketFormatConfig.from_yaml(config_file)

        assert "Missing required configuration keys" in str(exc_info.value)


class TestDeviceConfig:
    """Test device configuration.

    Tests: BDL-002
    """

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Test loading device configuration from YAML."""
        config_file = tmp_path / "device_mapping.yaml"

        config_data = {
            "devices": {
                "0x01": {
                    "name": "Test Device 1",
                    "description": "First test device",
                    "sample_rate": 100e6,
                },
                "0x02": {
                    "name": "Test Device 2",
                    "description": "Second test device",
                    "sample_rate": 100e6,
                },
            },
            "unknown_device": {"policy": "warn"},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = DeviceConfig.from_yaml(config_file)

        assert len(config.devices) == 2
        assert 0x01 in config.devices
        assert 0x02 in config.devices
        assert config.devices[0x01]["name"] == "Test Device 1"
        assert config.unknown_policy == "warn"


class TestDeviceMapper:
    """Test device ID to name mapping.

    Tests: BDL-002
    """

    def test_get_device_name_known(self) -> None:
        """Test getting name for known device."""
        config = DeviceConfig(
            devices={
                0x01: {"name": "Controller A"},
                0x02: {"name": "Controller B"},
            }
        )

        mapper = DeviceMapper(config)

        assert mapper.get_device_name(0x01) == "Controller A"
        assert mapper.get_device_name(0x02) == "Controller B"

    def test_get_device_name_unknown_warn(self) -> None:
        """Test getting name for unknown device with warn policy."""
        config = DeviceConfig(devices={}, unknown_policy="warn")

        mapper = DeviceMapper(config)

        name = mapper.get_device_name(0xFF)
        assert "Unknown Device" in name
        assert "0xFF" in name

    def test_get_device_name_unknown_error(self) -> None:
        """Test getting name for unknown device with error policy."""
        config = DeviceConfig(devices={}, unknown_policy="error")

        mapper = DeviceMapper(config)

        with pytest.raises(ConfigurationError) as exc_info:
            mapper.get_device_name(0xFF)

        assert "Unknown device ID" in str(exc_info.value)

    def test_get_device_info(self) -> None:
        """Test getting full device information."""
        config = DeviceConfig(
            devices={
                0x01: {
                    "name": "Controller A",
                    "description": "Primary controller",
                    "sample_rate": 100e6,
                }
            }
        )

        mapper = DeviceMapper(config)

        info = mapper.get_device_info(0x01)
        assert info["name"] == "Controller A"
        assert info["description"] == "Primary controller"
        assert info["sample_rate"] == 100e6


class TestConfigurablePacketLoader:
    """Test configurable packet loader.

    Tests: BDL-001, BDL-003
    """

    def create_test_packet(self, sequence: int, device_id: int, samples: list[int]) -> bytes:
        """Create a test packet with known structure.

        Packet format:
        - Bytes 0-3: Sequence number (uint32 big-endian)
        - Byte 4: Device ID (uint8)
        - Byte 5: Sync marker (0xFA)
        - Bytes 6-7: Reserved (zeros)
        - Bytes 8+: Sample data (uint64 little-endian each)
        """
        packet = bytearray()

        # Header
        packet.extend(struct.pack(">I", sequence))  # Sequence (big-endian)
        packet.append(device_id)  # Device ID
        packet.append(0xFA)  # Sync marker
        packet.extend(b"\x00\x00")  # Reserved

        # Samples
        for sample in samples:
            packet.extend(struct.pack("<Q", sample))  # uint64 little-endian

        return bytes(packet)

    def test_parse_packet(self, tmp_path: Path) -> None:
        """Test parsing a single packet."""
        # Create configuration
        format_config = PacketFormatConfig(
            name="test",
            version="1.0",
            packet_size=64,
            byte_order="big",
            header_size=8,
            header_fields=[
                HeaderFieldDef("sequence", 0, 4, "uint32", "big"),
                HeaderFieldDef("device_id", 4, 1, "uint8"),
                HeaderFieldDef("sync_marker", 5, 1, "uint8"),
            ],
            sample_offset=8,
            sample_count=7,
            sample_format=SampleFormatDef(8, "uint64", "little"),
        )

        # Create binary file with one packet
        bin_file = tmp_path / "test.bin"
        packet_data = self.create_test_packet(
            sequence=0, device_id=0x01, samples=[100, 200, 300, 400, 500, 600, 700]
        )

        with open(bin_file, "wb") as f:
            f.write(packet_data)

        # Load and parse
        loader = ConfigurablePacketLoader(format_config)
        packets = loader.load_packets(bin_file)

        assert len(packets) == 1
        packet = packets[0]

        assert packet["header"]["sequence"] == 0
        assert packet["header"]["device_id"] == 0x01
        assert packet["header"]["sync_marker"] == 0xFA
        assert len(packet["samples"]) == 7
        assert packet["samples"][0] == 100
        assert packet["samples"][-1] == 700

    def test_streaming_load(self, tmp_path: Path) -> None:
        """Test streaming packet loading."""
        format_config = PacketFormatConfig(
            name="test",
            version="1.0",
            packet_size=64,
            byte_order="big",
            header_size=8,
            header_fields=[
                HeaderFieldDef("sequence", 0, 4, "uint32", "big"),
            ],
            sample_offset=8,
            sample_count=7,
            sample_format=SampleFormatDef(8, "uint64", "little"),
        )

        # Create binary file with multiple packets
        bin_file = tmp_path / "multi.bin"
        with open(bin_file, "wb") as f:
            for seq in range(10):
                packet = self.create_test_packet(sequence=seq, device_id=0x01, samples=[seq] * 7)
                f.write(packet)

        # Stream packets
        loader = ConfigurablePacketLoader(format_config)
        packets = list(loader.load_packets_streaming(bin_file, chunk_size=3))

        assert len(packets) == 10
        for i, packet in enumerate(packets):
            assert packet["header"]["sequence"] == i

    def test_bitfield_extraction(self, tmp_path: Path) -> None:
        """Test extracting bitfields from header."""
        format_config = PacketFormatConfig(
            name="test",
            version="1.0",
            packet_size=16,
            byte_order="big",
            header_size=8,
            header_fields=[
                HeaderFieldDef(
                    "status",
                    0,
                    2,
                    "bitfield",
                    "big",
                    fields={
                        "overflow": {"bit": 15},
                        "error": {"bit": 14},
                        "channel": {"bits": [0, 7]},
                    },
                )
            ],
            sample_offset=8,
            sample_count=1,
            sample_format=SampleFormatDef(8, "uint64", "little"),
        )

        # Create packet with bitfield status = 0x8042 (overflow=1, error=0, channel=0x42)
        bin_file = tmp_path / "bitfield.bin"
        packet = struct.pack(">H", 0x8042)  # Status bitfield
        packet += b"\x00" * 6  # Padding
        packet += struct.pack("<Q", 12345)  # One sample

        with open(bin_file, "wb") as f:
            f.write(packet)

        loader = ConfigurablePacketLoader(format_config)
        packets = loader.load_packets(bin_file)

        assert len(packets) == 1
        status = packets[0]["header"]["status"]

        assert status["overflow"] == 1
        assert status["error"] == 0
        assert status["channel"] == 0x42


class TestPacketValidator:
    """Test packet validation.

    Tests: BDL-004
    """

    def test_sync_validation_success(self) -> None:
        """Test successful sync marker validation."""
        validator = PacketValidator(sync_marker=0xFA)

        packet = {"index": 0, "header": {"sync_marker": 0xFA}}
        result = validator.validate_packet(packet)

        assert result.is_valid
        assert result.sync_valid
        assert len(result.errors) == 0

    def test_sync_validation_failure(self) -> None:
        """Test sync marker validation failure."""
        validator = PacketValidator(sync_marker=0xFA, strictness="strict")

        packet = {"index": 0, "header": {"sync_marker": 0xAB}}
        result = validator.validate_packet(packet)

        assert not result.is_valid
        assert not result.sync_valid
        assert len(result.errors) > 0
        assert "Sync marker mismatch" in result.errors[0]

    def test_sequence_validation_gap_detection(self) -> None:
        """Test sequence gap detection."""
        validator = PacketValidator(strictness="normal")

        # First packet
        packet1 = {"index": 0, "header": {"sequence": 0}}
        result1 = validator.validate_packet(packet1)
        assert result1.is_valid

        # Gap - skip to sequence 5
        packet2 = {"index": 1, "header": {"sequence": 5}}
        result2 = validator.validate_packet(packet2)

        assert len(result2.warnings) > 0
        assert "Sequence gap" in result2.warnings[0]

    def test_sequence_validation_duplicate_detection(self) -> None:
        """Test duplicate sequence detection."""
        validator = PacketValidator()

        packet1 = {"index": 0, "header": {"sequence": 10}}
        validator.validate_packet(packet1)

        # Duplicate sequence
        packet2 = {"index": 1, "header": {"sequence": 10}}
        result2 = validator.validate_packet(packet2)

        assert len(result2.warnings) > 0
        assert "Duplicate sequence" in result2.warnings[0]

    def test_checksum_crc16(self) -> None:
        """Test CRC-16 checksum validation."""
        validator = PacketValidator(checksum_type="crc16", checksum_field="checksum")

        # Create test data
        test_data = b"Hello, World!"
        expected_crc = validator._crc16(test_data)

        packet = {"index": 0, "header": {"checksum": expected_crc}}
        result = validator.validate_packet(packet, packet_data=test_data)

        assert result.checksum_valid

    def test_validation_statistics(self) -> None:
        """Test validation statistics accumulation."""
        validator = PacketValidator(sync_marker=0xFA, strictness="normal")

        # Validate multiple packets with varying results
        packets = [
            {"index": 0, "header": {"sync_marker": 0xFA, "sequence": 0}},  # Valid
            {"index": 1, "header": {"sync_marker": 0xAB, "sequence": 1}},  # Bad sync
            {"index": 2, "header": {"sync_marker": 0xFA, "sequence": 2}},  # Valid
            {"index": 3, "header": {"sync_marker": 0xFA, "sequence": 10}},  # Gap
        ]

        for packet in packets:
            validator.validate_packet(packet)

        stats = validator.get_statistics()

        assert stats.total_packets == 4
        assert stats.sync_failures == 1
        assert stats.sequence_gaps == 1


class TestSourceTypeDetection:
    """Test source type detection.

    Tests: BDL-003
    """

    def test_detect_raw_binary(self, tmp_path: Path) -> None:
        """Test detecting raw binary files."""
        bin_file = tmp_path / "capture.bin"
        bin_file.write_bytes(b"\x00" * 100)

        assert detect_source_type(bin_file) == "raw"

    def test_detect_pcap(self, tmp_path: Path) -> None:
        """Test detecting PCAP files."""
        pcap_file = tmp_path / "capture.pcap"
        # PCAP magic bytes
        pcap_file.write_bytes(b"\xa1\xb2\xc3\xd4" + b"\x00" * 96)

        assert detect_source_type(pcap_file) == "pcap"

    def test_detect_by_extension(self, tmp_path: Path) -> None:
        """Test detection by file extension."""
        vcd_file = tmp_path / "signals.vcd"
        vcd_file.write_bytes(b"$date\n")

        assert detect_source_type(vcd_file) == "vcd"


class TestChannelExtraction:
    """Test channel extraction from packets.

    Tests: BDL-003
    """

    def test_extract_channels_from_samples(self) -> None:
        """Test extracting individual channels from multi-bit samples."""
        # Create packets with 16-bit samples
        packets = [
            {"index": 0, "samples": [0xABCD, 0x1234, 0x5678]},
            {"index": 1, "samples": [0x9ABC, 0xDEF0, 0x1111]},
        ]

        # Extract two 8-bit channels
        channel_map = {
            "ch0": {"bits": [0, 7]},  # Lower byte
            "ch1": {"bits": [8, 15]},  # Upper byte
        }

        traces = extract_channels(packets, channel_map)

        assert "ch0" in traces
        assert "ch1" in traces

        # Check first packet samples
        assert len(traces["ch0"].data) == 6  # 3 samples per packet * 2 packets
        assert traces["ch0"].data[0] == 1  # 0xCD -> 205 -> True (as bool)
        assert traces["ch1"].data[0] == 1  # 0xAB -> 171 -> True (as bool)


class TestIdleDetection:
    """Test idle region detection and trimming.

    Tests: BDL-005
    """

    def create_trace_with_idle(self, start_idle: int, end_idle: int, total: int) -> Any:
        """Create a digital trace with idle regions."""
        from oscura.core.types import DigitalTrace, TraceMetadata

        data = np.zeros(total, dtype=bool)
        # Active region in the middle
        data[start_idle:end_idle] = True

        metadata = TraceMetadata(sample_rate=100e6, channel_name="test")
        return DigitalTrace(data=data, metadata=metadata)

    def test_detect_idle_zeros(self) -> None:
        """Test detecting idle regions with zeros pattern."""
        trace = self.create_trace_with_idle(start_idle=100, end_idle=900, total=1000)

        regions = detect_idle_regions(trace, pattern="zeros", min_duration=50)

        # Should find idle at start and end
        assert len(regions) >= 2
        assert regions[0].start == 0
        assert regions[-1].end == 1000

    def test_trim_idle_start_and_end(self) -> None:
        """Test trimming idle from start and end."""
        trace = self.create_trace_with_idle(start_idle=100, end_idle=900, total=1000)

        trimmed = trim_idle(trace, trim_start=True, trim_end=True, pattern="zeros", min_duration=50)

        # Should have only active region
        assert len(trimmed.data) < len(trace.data)
        assert len(trimmed.data) <= 800  # Active region size

    def test_idle_statistics(self) -> None:
        """Test idle statistics calculation."""
        trace = self.create_trace_with_idle(start_idle=100, end_idle=900, total=1000)

        stats = get_idle_statistics(trace, pattern="zeros", min_duration=50)

        assert stats.total_samples == 1000
        assert stats.idle_samples > 0
        assert stats.active_samples > 0
        assert stats.idle_fraction > 0
        assert stats.active_fraction > 0
        assert stats.idle_fraction + stats.active_fraction == pytest.approx(1.0)


class TestEndToEndIntegration:
    """End-to-end integration tests.

    Tests: BDL-001 through BDL-005
    """

    def test_complete_workflow(self, tmp_path: Path) -> None:
        """Test complete workflow from config to channel extraction."""
        # Create format configuration
        format_file = tmp_path / "format.yaml"
        format_data = {
            "name": "test_daq",
            "version": "1.0",
            "packet": {"size": 32, "byte_order": "big"},
            "header": {
                "size": 8,
                "fields": [
                    {"name": "sequence", "offset": 0, "size": 4, "type": "uint32"},
                    {"name": "sync_marker", "offset": 4, "size": 1, "type": "uint8", "value": 0xFA},
                ],
            },
            "samples": {
                "offset": 8,
                "count": 3,
                "format": {"size": 8, "type": "uint64", "endian": "little"},
            },
        }

        with open(format_file, "w") as f:
            yaml.dump(format_data, f)

        # Create binary data file
        bin_file = tmp_path / "capture.bin"
        with open(bin_file, "wb") as f:
            for seq in range(5):
                # Header
                f.write(struct.pack(">I", seq))  # Sequence
                f.write(b"\xfa")  # Sync marker
                f.write(b"\x00\x00\x00")  # Padding

                # Samples (each is 8 bytes)
                for sample_idx in range(3):
                    sample = (seq << 8) | sample_idx
                    f.write(struct.pack("<Q", sample))

        # Load packets
        packets = load_binary_packets(bin_file, format_file)

        assert len(packets) == 5
        assert all(p["header"]["sync_marker"] == 0xFA for p in packets)

        # Extract channels
        channel_map = {"ch0": {"bits": [0, 7]}}
        traces = extract_channels(packets, channel_map)

        assert "ch0" in traces
        assert len(traces["ch0"].data) == 15  # 5 packets * 3 samples
