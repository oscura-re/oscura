"""Edge case tests for packet loaders to improve coverage.

This module tests boundary conditions and edge cases in the configurable
packet loader and binary data loaders.

(Edge Case Tests)

- Boundary conditions testing (+1% coverage)
- Empty/single-element array handling
- Maximum/minimum value validation
- Off-by-one conditions
- Buffer boundary testing
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
import yaml

from oscura.core.exceptions import ConfigurationError, LoaderError
from oscura.loaders.configurable import (
    BitfieldExtractor,
    ConfigurablePacketLoader,
    DeviceMapper,
    HeaderFieldDef,
    PacketFormatConfig,
    SampleFormatDef,
    extract_channels,
    load_binary_packets,
)

pytestmark = [pytest.mark.unit, pytest.mark.loader]


@pytest.mark.unit
@pytest.mark.loader
class TestBitfieldExtractorEdgeCases:
    """Test edge cases in BitfieldExtractor."""

    def test_extract_bit_zero(self) -> None:
        """Test extracting bit 0 (LSB)."""
        extractor = BitfieldExtractor()
        assert extractor.extract_bit(0b1010_1100, 0) == 0
        assert extractor.extract_bit(0b1010_1101, 0) == 1

    def test_extract_bit_max(self) -> None:
        """Test extracting bit 7 (MSB of byte)."""
        extractor = BitfieldExtractor()
        assert extractor.extract_bit(0b1010_1100, 7) == 1
        assert extractor.extract_bit(0b0010_1100, 7) == 0

    def test_extract_bit_63(self) -> None:
        """Test extracting bit 63 (max for 64-bit value)."""
        extractor = BitfieldExtractor()
        value = (1 << 63) | 0xFF  # MSB set
        assert extractor.extract_bit(value, 63) == 1
        assert extractor.extract_bit(0xFF, 63) == 0

    def test_extract_bits_single_bit_range(self) -> None:
        """Test extracting single-bit range (start == end)."""
        extractor = BitfieldExtractor()
        value = 0b1010_1100
        # Extract bit 2 as a range
        assert extractor.extract_bits(value, 2, 2) == 1
        # Extract bit 3 as a range
        assert extractor.extract_bits(value, 3, 3) == 1

    def test_extract_bits_full_byte(self) -> None:
        """Test extracting all 8 bits."""
        extractor = BitfieldExtractor()
        value = 0b1010_1100
        assert extractor.extract_bits(value, 0, 7) == 0b1010_1100

    def test_extract_bits_zero_value(self) -> None:
        """Test extracting from zero value."""
        extractor = BitfieldExtractor()
        assert extractor.extract_bits(0, 0, 7) == 0
        assert extractor.extract_bit(0, 0) == 0

    def test_extract_bits_max_value(self) -> None:
        """Test extracting from maximum uint64 value."""
        extractor = BitfieldExtractor()
        max_uint64 = (1 << 64) - 1
        assert extractor.extract_bits(max_uint64, 0, 63) == max_uint64
        assert extractor.extract_bit(max_uint64, 63) == 1


@pytest.mark.unit
@pytest.mark.loader
class TestHeaderFieldEdgeCases:
    """Test edge cases in HeaderFieldDef validation."""

    def test_field_offset_zero(self) -> None:
        """Test field at offset 0 (valid)."""
        field = HeaderFieldDef(name="header", offset=0, size=4, type="uint32")
        assert field.offset == 0

    def test_field_offset_negative_raises(self) -> None:
        """Test that negative offset raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="non-negative"):
            HeaderFieldDef(name="header", offset=-1, size=4, type="uint32")

    def test_field_size_one(self) -> None:
        """Test minimum field size of 1."""
        field = HeaderFieldDef(name="byte", offset=0, size=1, type="uint8")
        assert field.size == 1

    def test_field_size_zero_raises(self) -> None:
        """Test that zero size raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="positive"):
            HeaderFieldDef(name="invalid", offset=0, size=0, type="uint8")

    def test_field_size_negative_raises(self) -> None:
        """Test that negative size raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="positive"):
            HeaderFieldDef(name="invalid", offset=0, size=-1, type="uint8")

    def test_field_endian_native(self) -> None:
        """Test native endianness is valid."""
        field = HeaderFieldDef(name="data", offset=0, size=4, type="uint32", endian="native")
        assert field.endian == "native"

    def test_field_invalid_endian_raises(self) -> None:
        """Test invalid endianness raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="endianness"):
            HeaderFieldDef(name="data", offset=0, size=4, type="uint32", endian="middle")


@pytest.mark.unit
@pytest.mark.loader
class TestSampleFormatEdgeCases:
    """Test edge cases in SampleFormatDef validation."""

    def test_sample_size_one(self) -> None:
        """Test minimum sample size of 1."""
        fmt = SampleFormatDef(size=1, type="uint8")
        assert fmt.size == 1

    def test_sample_size_zero_raises(self) -> None:
        """Test that zero size raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="positive"):
            SampleFormatDef(size=0, type="uint8")

    def test_sample_size_negative_raises(self) -> None:
        """Test that negative size raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="positive"):
            SampleFormatDef(size=-1, type="uint8")


@pytest.mark.unit
@pytest.mark.loader
class TestPacketLoaderBoundaryConditions:
    """Test boundary conditions in packet loading."""

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Test loading from empty binary file."""
        # Create minimal config
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 16, "byte_order": "big"},
            "header": {"size": 8, "fields": []},
            "samples": {"offset": 8, "count": 1, "format": {"size": 8, "type": "uint64"}},
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Create empty binary file
        bin_path = tmp_path / "empty.bin"
        bin_path.write_bytes(b"")

        # Should return empty list, not raise error
        packets = load_binary_packets(bin_path, config_path)
        assert packets == []

    def test_load_single_packet(self, tmp_path: Path) -> None:
        """Test loading exactly one packet (minimum)."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 16, "byte_order": "little"},  # Use little endian for samples
            "header": {
                "size": 8,
                "fields": [
                    {"name": "id", "offset": 0, "size": 4, "type": "uint32", "endian": "big"}
                ],
            },
            "samples": {
                "offset": 8,
                "count": 1,
                "format": {"size": 8, "type": "uint64", "endian": "little"},
            },
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Create one packet with explicit little-endian for sample
        bin_path = tmp_path / "single.bin"
        packet_data = struct.pack(">I", 42) + b"\x00" * 4 + struct.pack("<Q", 100)
        bin_path.write_bytes(packet_data)

        packets = load_binary_packets(bin_path, config_path)
        assert len(packets) == 1
        assert packets[0]["header"]["id"] == 42
        assert packets[0]["samples"] == [100]

    def test_load_truncated_packet(self, tmp_path: Path) -> None:
        """Test loading file with incomplete final packet."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 16, "byte_order": "big"},
            "header": {"size": 8, "fields": []},
            "samples": {"offset": 8, "count": 1, "format": {"size": 8, "type": "uint64"}},
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Create 1.5 packets (second packet incomplete)
        bin_path = tmp_path / "truncated.bin"
        full_packet = b"\x00" * 16
        partial_packet = b"\x00" * 8  # Only half a packet
        bin_path.write_bytes(full_packet + partial_packet)

        packets = load_binary_packets(bin_path, config_path)
        # Should only return complete packets
        assert len(packets) == 1

    def test_packet_exactly_at_boundary(self, tmp_path: Path) -> None:
        """Test packet size exactly matches file size."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 16, "byte_order": "big"},
            "header": {"size": 8, "fields": []},
            "samples": {"offset": 8, "count": 1, "format": {"size": 8, "type": "uint64"}},
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Create file with exactly 16 bytes
        bin_path = tmp_path / "exact.bin"
        bin_path.write_bytes(b"\x00" * 16)

        packets = load_binary_packets(bin_path, config_path)
        assert len(packets) == 1

    def test_sample_count_zero(self, tmp_path: Path) -> None:
        """Test packet with zero samples."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 8, "byte_order": "big"},
            "header": {"size": 8, "fields": []},
            "samples": {
                "offset": 8,
                "count": 0,  # No samples
                "format": {"size": 8, "type": "uint64"},
            },
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        bin_path = tmp_path / "no_samples.bin"
        bin_path.write_bytes(b"\x00" * 8)

        packets = load_binary_packets(bin_path, config_path)
        assert len(packets) == 1
        assert packets[0]["samples"] == []

    def test_sample_exceeds_packet_bounds(self, tmp_path: Path) -> None:
        """Test sample that would exceed packet boundaries."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 16, "byte_order": "big"},
            "header": {"size": 8, "fields": []},
            "samples": {
                "offset": 8,
                "count": 2,  # Would need 16 bytes but only 8 available
                "format": {"size": 8, "type": "uint64"},
            },
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        bin_path = tmp_path / "overflow.bin"
        bin_path.write_bytes(b"\x00" * 16)

        packets = load_binary_packets(bin_path, config_path)
        # Should only load samples that fit
        assert len(packets) == 1
        assert len(packets[0]["samples"]) == 1  # Only first sample fits


@pytest.mark.unit
@pytest.mark.loader
class TestVariableLengthPacketEdgeCases:
    """Test edge cases for variable-length packets."""

    def test_variable_length_minimum_packet(self, tmp_path: Path) -> None:
        """Test minimum-sized variable-length packet."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {
                "size": "variable",
                "byte_order": "big",
                "length_field": "length",
                "length_includes_header": True,
            },
            "header": {
                "size": 4,
                "fields": [{"name": "length", "offset": 0, "size": 4, "type": "uint32"}],
            },
            "samples": {"offset": 4, "count": 0, "format": {"size": 1, "type": "uint8"}},
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Packet with only header (length = 4)
        bin_path = tmp_path / "min_var.bin"
        bin_path.write_bytes(struct.pack(">I", 4))

        packets = load_binary_packets(bin_path, config_path)
        assert len(packets) == 1
        assert packets[0]["header"]["length"] == 4

    def test_variable_length_without_length_field_raises(self, tmp_path: Path) -> None:
        """Test that variable-length without length_field raises error."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {
                "size": "variable",
                "byte_order": "big",
                # Missing length_field
            },
            "header": {"size": 4, "fields": []},
            "samples": {"offset": 4, "count": 1, "format": {"size": 1, "type": "uint8"}},
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        bin_path = tmp_path / "data.bin"
        bin_path.write_bytes(b"\x00" * 10)

        with pytest.raises(ConfigurationError, match="length_field"):
            load_binary_packets(bin_path, config_path)

    def test_variable_length_truncated_payload(self, tmp_path: Path) -> None:
        """Test variable-length packet with truncated payload."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {
                "size": "variable",
                "byte_order": "big",
                "length_field": "length",
                "length_includes_header": True,
            },
            "header": {
                "size": 4,
                "fields": [{"name": "length", "offset": 0, "size": 4, "type": "uint32"}],
            },
            "samples": {"offset": 4, "count": 1, "format": {"size": 1, "type": "uint8"}},
        }

        config_path = tmp_path / "format.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Header says length=10, but only provide 6 bytes total
        bin_path = tmp_path / "truncated.bin"
        bin_path.write_bytes(struct.pack(">I", 10) + b"\x00" * 2)

        packets = load_binary_packets(bin_path, config_path)
        # Should not load incomplete packet
        assert len(packets) == 0


@pytest.mark.unit
@pytest.mark.loader
class TestChannelExtractionEdgeCases:
    """Test edge cases in channel extraction."""

    def test_extract_channels_empty_packets(self) -> None:
        """Test extracting channels from empty packet list."""
        channel_map = {"ch0": {"bits": [0, 7]}}

        with pytest.raises(ConfigurationError, match="No packets"):
            extract_channels([], channel_map)

    def test_extract_channels_packets_with_no_samples(self) -> None:
        """Test extracting channels from packets with no samples."""
        packets = [
            {"index": 0, "header": {}, "samples": []},
            {"index": 1, "header": {}, "samples": []},
        ]
        channel_map = {"ch0": {"bits": [0, 7]}}

        traces = extract_channels(packets, channel_map)
        assert "ch0" in traces
        assert len(traces["ch0"].data) == 0

    def test_extract_single_bit_channel(self) -> None:
        """Test extracting single-bit channel."""
        packets = [
            {"index": 0, "header": {}, "samples": [0b10101010, 0b11001100]},
        ]
        channel_map = {"bit0": {"bit": 0}}

        traces = extract_channels(packets, channel_map)
        assert "bit0" in traces
        # Bit 0 of 0b10101010 = 0, bit 0 of 0b11001100 = 0
        assert list(traces["bit0"].data) == [False, False]

    def test_extract_msb_channel(self) -> None:
        """Test extracting MSB (bit 7) channel."""
        packets = [
            {"index": 0, "header": {}, "samples": [0b10000000, 0b01111111]},
        ]
        channel_map = {"msb": {"bit": 7}}

        traces = extract_channels(packets, channel_map)
        assert list(traces["msb"].data) == [True, False]

    def test_extract_full_width_channel(self) -> None:
        """Test extracting full 8-bit channel."""
        packets = [
            {"index": 0, "header": {}, "samples": [0xFF, 0x00, 0xAA]},
        ]
        channel_map = {"full": {"bits": [0, 7]}}

        traces = extract_channels(packets, channel_map)
        assert list(traces["full"].data) == [True, False, True]  # Boolean array


@pytest.mark.unit
@pytest.mark.loader
class TestDeviceMapperEdgeCases:
    """Test edge cases in DeviceMapper."""

    def test_device_mapper_unknown_device_error_policy(self, tmp_path: Path) -> None:
        """Test DeviceMapper with 'error' policy for unknown devices."""
        config_data = {
            "devices": {"0x10": {"name": "Known Device"}},
            "unknown_device": {"policy": "error"},
        }

        config_path = tmp_path / "devices.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        mapper = DeviceMapper.from_file(config_path)

        # Should raise error for unknown device
        with pytest.raises(ConfigurationError, match="Unknown device"):
            mapper.get_device(0xFF)

    def test_device_mapper_unknown_device_warn_policy(self, tmp_path: Path) -> None:
        """Test DeviceMapper with 'warn' policy for unknown devices."""
        config_data = {
            "devices": {"0x10": {"name": "Known Device"}},
            "unknown_device": {"policy": "warn"},
        }

        config_path = tmp_path / "devices.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        mapper = DeviceMapper.from_file(config_path)

        # Should return None, not raise
        device = mapper.get_device(0xFF)
        assert device is None

    def test_device_mapper_unknown_device_ignore_policy(self, tmp_path: Path) -> None:
        """Test DeviceMapper with 'ignore' policy for unknown devices."""
        config_data = {
            "devices": {"0x10": {"name": "Known Device"}},
            "unknown_device": {"policy": "ignore"},
        }

        config_path = tmp_path / "devices.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        mapper = DeviceMapper.from_file(config_path)

        # Should return None silently
        device = mapper.get_device(0xFF)
        assert device is None

    def test_device_mapper_hex_device_id(self, tmp_path: Path) -> None:
        """Test DeviceMapper with hexadecimal device IDs."""
        config_data = {"devices": {"0xFF": {"name": "Max Device"}, "0x00": {"name": "Min Device"}}}

        config_path = tmp_path / "devices.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        mapper = DeviceMapper.from_file(config_path)

        assert mapper.resolve_name(0xFF) == "Max Device"
        assert mapper.resolve_name(0x00) == "Min Device"


@pytest.mark.unit
@pytest.mark.loader
class TestBytesToIntEdgeCases:
    """Test edge cases in byte-to-integer conversion."""

    def test_bytes_to_int_single_byte(self, tmp_path: Path) -> None:
        """Test converting single byte to integer."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 1, "byte_order": "big"},
            "header": {
                "size": 1,
                "fields": [{"name": "byte", "offset": 0, "size": 1, "type": "uint8"}],
            },
            "samples": {"offset": 1, "count": 0, "format": {"size": 1, "type": "uint8"}},
        }

        config = PacketFormatConfig.from_dict(config_data)
        loader = ConfigurablePacketLoader(config)

        # Test min and max byte values
        assert loader._bytes_to_int(b"\x00", "big", False) == 0
        assert loader._bytes_to_int(b"\xff", "big", False) == 255

    def test_bytes_to_int_signed_negative(self, tmp_path: Path) -> None:
        """Test converting bytes to signed negative integer."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 1, "byte_order": "big"},
            "header": {
                "size": 1,
                "fields": [{"name": "byte", "offset": 0, "size": 1, "type": "int8"}],
            },
            "samples": {"offset": 1, "count": 0, "format": {"size": 1, "type": "uint8"}},
        }

        config = PacketFormatConfig.from_dict(config_data)
        loader = ConfigurablePacketLoader(config)

        # 0xFF as signed int8 = -1
        assert loader._bytes_to_int(b"\xff", "big", True) == -1
        # 0x80 as signed int8 = -128
        assert loader._bytes_to_int(b"\x80", "big", True) == -128

    def test_bytes_to_int_multibyte_little_endian(self, tmp_path: Path) -> None:
        """Test multi-byte conversion with little endian."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 1, "byte_order": "little"},
            "header": {
                "size": 1,
                "fields": [{"name": "val", "offset": 0, "size": 4, "type": "uint32"}],
            },
            "samples": {"offset": 1, "count": 0, "format": {"size": 1, "type": "uint8"}},
        }

        config = PacketFormatConfig.from_dict(config_data)
        loader = ConfigurablePacketLoader(config)

        # 0x01020304 in little endian = [04, 03, 02, 01]
        assert loader._bytes_to_int(b"\x04\x03\x02\x01", "little", False) == 0x01020304

    def test_bytes_to_int_unusual_sizes(self, tmp_path: Path) -> None:
        """Test conversion of unusual byte sizes (uint40, uint48)."""
        config_data = {
            "name": "test",
            "version": "1.0",
            "packet": {"size": 1, "byte_order": "big"},
            "header": {
                "size": 1,
                "fields": [{"name": "val", "offset": 0, "size": 5, "type": "uint40"}],
            },
            "samples": {"offset": 1, "count": 0, "format": {"size": 1, "type": "uint8"}},
        }

        config = PacketFormatConfig.from_dict(config_data)
        loader = ConfigurablePacketLoader(config)

        # 5 bytes: max value = 2^40 - 1
        max_40bit = (1 << 40) - 1
        bytes_data = max_40bit.to_bytes(5, byteorder="big")
        assert loader._bytes_to_int(bytes_data, "big", False) == max_40bit


@pytest.mark.unit
@pytest.mark.loader
class TestConfigurationValidationEdgeCases:
    """Test edge cases in configuration validation."""

    def test_config_missing_required_section_name(self) -> None:
        """Test config missing 'name' field."""
        config_data = {
            # Missing "name"
            "version": "1.0",
            "packet": {"size": 16, "byte_order": "big"},
            "header": {"size": 8, "fields": []},
            "samples": {"offset": 8, "count": 1, "format": {"size": 8, "type": "uint64"}},
        }

        with pytest.raises(ConfigurationError, match="Missing required"):
            PacketFormatConfig.from_dict(config_data)

    def test_config_missing_required_section_version(self) -> None:
        """Test config missing 'version' field."""
        config_data = {
            "name": "test",
            # Missing "version"
            "packet": {"size": 16, "byte_order": "big"},
            "header": {"size": 8, "fields": []},
            "samples": {"offset": 8, "count": 1, "format": {"size": 8, "type": "uint64"}},
        }

        with pytest.raises(ConfigurationError, match="Missing required"):
            PacketFormatConfig.from_dict(config_data)

    def test_config_file_not_found(self, tmp_path: Path) -> None:
        """Test loading config from non-existent file."""
        non_existent = tmp_path / "does_not_exist.yaml"

        with pytest.raises(LoaderError, match="not found"):
            PacketFormatConfig.from_file(non_existent)
