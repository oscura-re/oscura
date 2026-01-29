"""Unit tests for parallel bus decoding.

This module provides comprehensive unit tests for the bus analyzer module,
covering all public functions and classes with edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.analyzers.digital.bus import (
    BusConfig,
    BusDecoder,
    BusTransaction,
    ParallelBusConfig,
    decode_bus,
    sample_at_clock,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.digital]


# =============================================================================
# Test BusConfig
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusConfig:
    """Test BusConfig class and its methods."""

    def test_basic_creation(self) -> None:
        """Test basic BusConfig creation."""
        config = BusConfig(name="data_bus", width=8)
        assert config.name == "data_bus"
        assert config.width == 8
        assert config.bit_order == "lsb_first"
        assert config.active_low is False
        assert config.bits == []

    def test_creation_with_all_params(self) -> None:
        """Test BusConfig creation with all parameters."""
        bits = [{"channel": i, "bit": i, "name": f"D{i}"} for i in range(8)]
        config = BusConfig(
            name="address_bus",
            width=8,
            bit_order="msb_first",
            active_low=True,
            bits=bits,
        )
        assert config.name == "address_bus"
        assert config.width == 8
        assert config.bit_order == "msb_first"
        assert config.active_low is True
        assert len(config.bits) == 8

    def test_invalid_width(self) -> None:
        """Test that invalid width raises ValueError."""
        with pytest.raises(ValueError, match="Bus width must be positive"):
            BusConfig(name="test", width=0)

        with pytest.raises(ValueError, match="Bus width must be positive"):
            BusConfig(name="test", width=-1)

    def test_invalid_bit_order(self) -> None:
        """Test that invalid bit_order raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bit_order"):
            BusConfig(name="test", width=8, bit_order="invalid")  # type: ignore

    def test_from_dict_minimal(self) -> None:
        """Test creating BusConfig from minimal dictionary."""
        config_dict = {"width": 4}
        config = BusConfig.from_dict(config_dict)
        assert config.name == "bus"
        assert config.width == 4
        assert config.bit_order == "lsb_first"
        assert config.active_low is False

    def test_from_dict_complete(self) -> None:
        """Test creating BusConfig from complete dictionary."""
        config_dict = {
            "name": "control_bus",
            "width": 16,
            "bit_order": "msb_first",
            "active_low": True,
            "bits": [{"channel": i, "bit": i} for i in range(16)],
        }
        config = BusConfig.from_dict(config_dict)
        assert config.name == "control_bus"
        assert config.width == 16
        assert config.bit_order == "msb_first"
        assert config.active_low is True
        assert len(config.bits) == 16

    def test_from_dict_missing_width(self) -> None:
        """Test that missing width raises KeyError."""
        config_dict = {"name": "test"}
        with pytest.raises(KeyError):
            BusConfig.from_dict(config_dict)

    def test_from_yaml_missing_file(self, tmp_path: Path) -> None:
        """Test that missing YAML file raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            BusConfig.from_yaml(nonexistent)

    def test_from_yaml_missing_pyyaml(self, tmp_path: Path, monkeypatch) -> None:
        """Test that missing PyYAML raises ImportError."""
        # Create a valid YAML file
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("width: 8\n")

        # Mock import failure
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="PyYAML is required"):
            BusConfig.from_yaml(yaml_file)

    def test_from_yaml_valid(self, tmp_path: Path) -> None:
        """Test loading BusConfig from valid YAML file."""
        yaml_content = """
name: test_bus
width: 8
bit_order: lsb_first
active_low: false
bits:
  - channel: 0
    bit: 0
  - channel: 1
    bit: 1
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        config = BusConfig.from_yaml(yaml_file)
        assert config.name == "test_bus"
        assert config.width == 8
        assert len(config.bits) == 2


# =============================================================================
# Test ParallelBusConfig
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestParallelBusConfig:
    """Test ParallelBusConfig class."""

    def test_basic_creation(self) -> None:
        """Test basic ParallelBusConfig creation."""
        config = ParallelBusConfig(data_width=8)
        assert config.data_width == 8
        assert config.bit_order == "lsb_first"
        assert config.has_clock is False
        assert config.address_width is None
        assert config.active_low is False

    def test_creation_with_all_params(self) -> None:
        """Test ParallelBusConfig with all parameters."""
        config = ParallelBusConfig(
            data_width=16,
            bit_order="msb_first",
            has_clock=True,
            address_width=12,
            active_low=True,
        )
        assert config.data_width == 16
        assert config.bit_order == "msb_first"
        assert config.has_clock is True
        assert config.address_width == 12
        assert config.active_low is True

    def test_invalid_data_width(self) -> None:
        """Test that invalid data_width raises ValueError."""
        with pytest.raises(ValueError, match="data_width must be positive"):
            ParallelBusConfig(data_width=0)

        with pytest.raises(ValueError, match="data_width must be positive"):
            ParallelBusConfig(data_width=-1)

    def test_invalid_address_width(self) -> None:
        """Test that invalid address_width raises ValueError."""
        with pytest.raises(ValueError, match="address_width must be positive"):
            ParallelBusConfig(data_width=8, address_width=0)

        with pytest.raises(ValueError, match="address_width must be positive"):
            ParallelBusConfig(data_width=8, address_width=-1)

    def test_to_bus_config_default_name(self) -> None:
        """Test converting to BusConfig with default name."""
        parallel_config = ParallelBusConfig(data_width=8)
        bus_config = parallel_config.to_bus_config()

        assert bus_config.name == "parallel_bus"
        assert bus_config.width == 8
        assert bus_config.bit_order == "lsb_first"
        assert bus_config.active_low is False
        assert len(bus_config.bits) == 8

    def test_to_bus_config_custom_name(self) -> None:
        """Test converting to BusConfig with custom name."""
        parallel_config = ParallelBusConfig(data_width=4, bit_order="msb_first", active_low=True)
        bus_config = parallel_config.to_bus_config(name="custom_bus")

        assert bus_config.name == "custom_bus"
        assert bus_config.width == 4
        assert bus_config.bit_order == "msb_first"
        assert bus_config.active_low is True


# =============================================================================
# Test BusTransaction
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusTransaction:
    """Test BusTransaction class."""

    def test_basic_creation(self) -> None:
        """Test basic BusTransaction creation."""
        transaction = BusTransaction(
            timestamp=1.5e-6, sample_index=150, value=0xAB, raw_bits=[1, 1, 0, 1, 0, 1, 0, 1]
        )
        assert transaction.timestamp == 1.5e-6
        assert transaction.sample_index == 150
        assert transaction.value == 0xAB
        assert len(transaction.raw_bits) == 8
        assert transaction.transaction_type == ""
        assert transaction.address is None
        assert transaction.data is None

    def test_creation_with_optional_fields(self) -> None:
        """Test BusTransaction with optional fields."""
        transaction = BusTransaction(
            timestamp=2.0e-6,
            sample_index=200,
            value=0xFF,
            raw_bits=[1] * 8,
            transaction_type="write",
            address=0x1000,
            data=0xDEAD,
        )
        assert transaction.transaction_type == "write"
        assert transaction.address == 0x1000
        assert transaction.data == 0xDEAD


# =============================================================================
# Test BusDecoder Initialization
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderInit:
    """Test BusDecoder initialization."""

    def test_init_with_bus_config(self) -> None:
        """Test initializing decoder with BusConfig."""
        config = BusConfig(name="test", width=8)
        decoder = BusDecoder(config, sample_rate=100e6)

        assert decoder.config == config
        assert decoder.sample_rate == 100e6
        assert decoder._time_base == 1.0 / 100e6

    def test_init_with_parallel_bus_config(self) -> None:
        """Test initializing decoder with ParallelBusConfig."""
        parallel_config = ParallelBusConfig(data_width=8)
        decoder = BusDecoder(parallel_config, sample_rate=1e6)

        assert decoder.sample_rate == 1e6
        assert decoder.config.width == 8
        assert decoder._parallel_config == parallel_config

    def test_init_invalid_sample_rate(self) -> None:
        """Test that invalid sample rate raises ValueError."""
        config = BusConfig(name="test", width=8)

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            BusDecoder(config, sample_rate=0)

        with pytest.raises(ValueError, match="Sample rate must be positive"):
            BusDecoder(config, sample_rate=-1)

    def test_init_default_sample_rate(self) -> None:
        """Test initializing with default sample rate."""
        config = BusConfig(name="test", width=8)
        decoder = BusDecoder(config)

        assert decoder.sample_rate == 1.0


# =============================================================================
# Test BusDecoder Core Decoding Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderDecoding:
    """Test BusDecoder decoding methods."""

    def test_decode_bus_empty_traces(self) -> None:
        """Test that empty bit_traces raises ValueError."""
        config = BusConfig(name="test", width=8)
        config.bits = [{"channel": i, "bit": i} for i in range(8)]
        decoder = BusDecoder(config, sample_rate=1e6)

        with pytest.raises(ValueError, match="bit_traces cannot be empty"):
            decoder.decode_bus({})

    def test_decode_bus_without_clock_lsb_first(self) -> None:
        """Test decoding without clock using LSB first."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Binary: 1101 (LSB first) = 0b1011 = 11
        bit_traces = {
            0: np.array([1, 0, 1], dtype=np.uint8),  # LSB
            1: np.array([1, 1, 0], dtype=np.uint8),
            2: np.array([0, 0, 1], dtype=np.uint8),
            3: np.array([1, 1, 0], dtype=np.uint8),  # MSB
        }

        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) == 3
        assert transactions[0].value == 0b1011  # 11
        assert transactions[1].value == 0b1010  # 10
        assert transactions[2].value == 0b0101  # 5

    def test_decode_bus_without_clock_msb_first(self) -> None:
        """Test decoding without clock using MSB first."""
        config = BusConfig(name="test", width=4, bit_order="msb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Binary: 1101 (MSB first) = 0b1101 = 13
        bit_traces = {
            0: np.array([1, 0, 1], dtype=np.uint8),  # MSB
            1: np.array([1, 1, 0], dtype=np.uint8),
            2: np.array([0, 0, 1], dtype=np.uint8),
            3: np.array([1, 1, 0], dtype=np.uint8),  # LSB
        }

        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) == 3
        assert transactions[0].value == 0b1101  # 13
        assert transactions[1].value == 0b0101  # 5
        assert transactions[2].value == 0b1010  # 10

    def test_decode_bus_with_active_low(self) -> None:
        """Test decoding with active-low signals."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first", active_low=True)
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Physical: 0000 -> Active-low inverted: 1111 = 15
        bit_traces = {
            0: np.array([0, 1], dtype=np.uint8),
            1: np.array([0, 1], dtype=np.uint8),
            2: np.array([0, 1], dtype=np.uint8),
            3: np.array([0, 1], dtype=np.uint8),
        }

        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) == 2
        assert transactions[0].value == 0b1111  # All inverted
        assert transactions[1].value == 0b0000  # All inverted

    def test_decode_bus_with_clock_rising_edge(self) -> None:
        """Test decoding with clock using rising edge."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {
            0: np.array([1, 1, 0, 0, 1], dtype=np.uint8),
            1: np.array([0, 1, 1, 1, 0], dtype=np.uint8),
            2: np.array([1, 0, 0, 1, 1], dtype=np.uint8),
            3: np.array([0, 0, 1, 1, 0], dtype=np.uint8),
        }
        clock = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

        transactions = decoder.decode_bus(bit_traces, clock_trace=clock, clock_edge="rising")

        # Rising edges at indices 1 and 3
        assert len(transactions) == 2
        assert transactions[0].sample_index == 1
        assert transactions[1].sample_index == 3

    def test_decode_bus_with_clock_falling_edge(self) -> None:
        """Test decoding with clock using falling edge."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {
            0: np.array([1, 1, 0, 0, 1], dtype=np.uint8),
            1: np.array([0, 1, 1, 1, 0], dtype=np.uint8),
            2: np.array([1, 0, 0, 1, 1], dtype=np.uint8),
            3: np.array([0, 0, 1, 1, 0], dtype=np.uint8),
        }
        clock = np.array([1, 0, 1, 0, 1], dtype=np.uint8)

        transactions = decoder.decode_bus(bit_traces, clock_trace=clock, clock_edge="falling")

        # Falling edges at indices 1 and 3
        assert len(transactions) == 2
        assert transactions[0].sample_index == 1
        assert transactions[1].sample_index == 3


# =============================================================================
# Test BusDecoder Parallel Decode Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderParallel:
    """Test BusDecoder parallel decoding methods."""

    def test_decode_parallel_empty_channels(self) -> None:
        """Test decode_parallel with empty channel list."""
        config = BusConfig(name="test", width=8)
        decoder = BusDecoder(config, sample_rate=1e6)

        values = decoder.decode_parallel([])
        assert values == []

    def test_decode_parallel_lsb_first(self) -> None:
        """Test decode_parallel with LSB first ordering."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        channels = [
            np.array([1, 0, 1], dtype=np.uint8),  # Bit 0 (LSB)
            np.array([0, 1, 1], dtype=np.uint8),  # Bit 1
            np.array([1, 1, 0], dtype=np.uint8),  # Bit 2
            np.array([0, 0, 1], dtype=np.uint8),  # Bit 3 (MSB)
        ]

        values = decoder.decode_parallel(channels)
        assert len(values) == 3
        assert values[0] == 0b0101  # 5
        assert values[1] == 0b0110  # 6
        assert values[2] == 0b1011  # 11

    def test_decode_parallel_msb_first(self) -> None:
        """Test decode_parallel with MSB first ordering."""
        config = BusConfig(name="test", width=4, bit_order="msb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        channels = [
            np.array([1, 0, 1], dtype=np.uint8),  # Bit 0 (MSB)
            np.array([0, 1, 1], dtype=np.uint8),  # Bit 1
            np.array([1, 1, 0], dtype=np.uint8),  # Bit 2
            np.array([0, 0, 1], dtype=np.uint8),  # Bit 3 (LSB)
        ]

        values = decoder.decode_parallel(channels)
        assert len(values) == 3
        assert values[0] == 0b1010  # 10
        assert values[1] == 0b0110  # 6
        assert values[2] == 0b1101  # 13

    def test_decode_parallel_with_active_low(self) -> None:
        """Test decode_parallel with active-low signals."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first", active_low=True)
        decoder = BusDecoder(config, sample_rate=1e6)

        channels = [
            np.array([0, 1], dtype=np.uint8),  # Inverted 1, 0
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
        ]

        values = decoder.decode_parallel(channels)
        assert len(values) == 2
        assert values[0] == 0b1111  # All inverted to 1
        assert values[1] == 0b0000  # All inverted to 0

    def test_decode_parallel_8bit_values(self) -> None:
        """Test decode_parallel with 8-bit values."""
        config = BusConfig(name="test", width=8, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        # Create channels for value 0xA5 = 0b10100101
        channels = [
            np.array([1], dtype=np.uint8),  # Bit 0
            np.array([0], dtype=np.uint8),  # Bit 1
            np.array([1], dtype=np.uint8),  # Bit 2
            np.array([0], dtype=np.uint8),  # Bit 3
            np.array([0], dtype=np.uint8),  # Bit 4
            np.array([1], dtype=np.uint8),  # Bit 5
            np.array([0], dtype=np.uint8),  # Bit 6
            np.array([1], dtype=np.uint8),  # Bit 7
        ]

        values = decoder.decode_parallel(channels)
        assert len(values) == 1
        assert values[0] == 0xA5


# =============================================================================
# Test BusDecoder Clock-Based Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderClock:
    """Test BusDecoder clock-based decoding."""

    def test_decode_with_clock_empty_channels(self) -> None:
        """Test decode_with_clock with empty channels."""
        config = BusConfig(name="test", width=8)
        decoder = BusDecoder(config, sample_rate=1e6)

        clock = np.array([0, 1, 0, 1], dtype=np.uint8)
        values = decoder.decode_with_clock([], clock, "rising")
        assert values == []

    def test_decode_with_clock_rising_edge(self) -> None:
        """Test decode_with_clock on rising edges."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        channels = [
            np.array([0, 1, 1, 0, 1], dtype=np.uint8),
            np.array([1, 0, 1, 1, 0], dtype=np.uint8),
            np.array([0, 1, 0, 1, 1], dtype=np.uint8),
            np.array([1, 1, 0, 0, 1], dtype=np.uint8),
        ]
        clock = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

        values = decoder.decode_with_clock(channels, clock, "rising")

        # Rising edges at indices 1 and 3
        assert len(values) == 2

    def test_decode_with_clock_falling_edge(self) -> None:
        """Test decode_with_clock on falling edges."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        channels = [
            np.array([0, 1, 1, 0, 1], dtype=np.uint8),
            np.array([1, 0, 1, 1, 0], dtype=np.uint8),
            np.array([0, 1, 0, 1, 1], dtype=np.uint8),
            np.array([1, 1, 0, 0, 1], dtype=np.uint8),
        ]
        clock = np.array([1, 0, 1, 0, 1], dtype=np.uint8)

        values = decoder.decode_with_clock(channels, clock, "falling")

        # Falling edges at indices 1 and 3
        assert len(values) == 2

    def test_decode_with_clock_no_edges(self) -> None:
        """Test decode_with_clock when clock has no edges."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        channels = [np.array([1, 1, 1], dtype=np.uint8) for _ in range(4)]
        clock = np.array([0, 0, 0], dtype=np.uint8)  # No edges

        values = decoder.decode_with_clock(channels, clock, "rising")
        assert len(values) == 0

    def test_decode_with_clock_msb_first(self) -> None:
        """Test decode_with_clock with MSB first ordering."""
        config = BusConfig(name="test", width=4, bit_order="msb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        channels = [
            np.array([1, 0, 1], dtype=np.uint8),  # MSB
            np.array([0, 1, 0], dtype=np.uint8),
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([0, 1, 1], dtype=np.uint8),  # LSB
        ]
        clock = np.array([0, 1, 0], dtype=np.uint8)

        values = decoder.decode_with_clock(channels, clock, "rising")

        # Rising edge at index 1
        assert len(values) == 1
        assert values[0] == 0b0101  # 5


# =============================================================================
# Test BusDecoder Transaction Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderTransactions:
    """Test BusDecoder transaction decoding."""

    def test_decode_transactions_basic(self) -> None:
        """Test basic transaction decoding with address and data."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        # 4-bit address and 4-bit data
        addr_channels = [
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([0, 1, 0], dtype=np.uint8),
            np.array([1, 1, 1], dtype=np.uint8),
            np.array([0, 0, 1], dtype=np.uint8),
        ]
        data_channels = [
            np.array([1, 1, 0], dtype=np.uint8),
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([0, 1, 1], dtype=np.uint8),
            np.array([1, 1, 0], dtype=np.uint8),
        ]
        clock = np.array([0, 1, 0], dtype=np.uint8)

        transactions = decoder.decode_transactions(addr_channels, data_channels, clock, "rising")

        # Rising edge at index 1
        assert len(transactions) == 1
        assert "address" in transactions[0]
        assert "data" in transactions[0]
        assert "sample_index" in transactions[0]
        assert transactions[0]["sample_index"] == 1

    def test_decode_transactions_multiple_clocks(self) -> None:
        """Test transaction decoding with multiple clock edges."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        addr_channels = [
            np.array([1, 0, 1, 1, 0], dtype=np.uint8),
            np.array([0, 1, 0, 1, 1], dtype=np.uint8),
            np.array([1, 1, 1, 0, 0], dtype=np.uint8),
            np.array([0, 0, 1, 0, 1], dtype=np.uint8),
        ]
        data_channels = [
            np.array([1, 1, 0, 0, 1], dtype=np.uint8),
            np.array([1, 0, 1, 1, 0], dtype=np.uint8),
            np.array([0, 1, 1, 0, 1], dtype=np.uint8),
            np.array([1, 1, 0, 1, 0], dtype=np.uint8),
        ]
        clock = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

        transactions = decoder.decode_transactions(addr_channels, data_channels, clock, "rising")

        # Rising edges at indices 1 and 3
        assert len(transactions) == 2
        assert transactions[0]["sample_index"] == 1
        assert transactions[1]["sample_index"] == 3

    def test_decode_transactions_falling_edge(self) -> None:
        """Test transaction decoding on falling edges."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        addr_channels = [np.array([1, 0, 1, 0], dtype=np.uint8) for _ in range(4)]
        data_channels = [np.array([0, 1, 0, 1], dtype=np.uint8) for _ in range(4)]
        clock = np.array([1, 0, 1, 0], dtype=np.uint8)

        transactions = decoder.decode_transactions(addr_channels, data_channels, clock, "falling")

        # Falling edges at indices 1 and 3
        assert len(transactions) == 2

    def test_decode_transactions_msb_first(self) -> None:
        """Test transaction decoding with MSB first ordering."""
        config = BusConfig(name="test", width=4, bit_order="msb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        # Known values: address=0b1010 (10), data=0b0101 (5)
        addr_channels = [
            np.array([0, 1], dtype=np.uint8),  # MSB
            np.array([0, 0], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 0], dtype=np.uint8),  # LSB
        ]
        data_channels = [
            np.array([0, 0], dtype=np.uint8),  # MSB
            np.array([0, 1], dtype=np.uint8),
            np.array([0, 0], dtype=np.uint8),
            np.array([0, 1], dtype=np.uint8),  # LSB
        ]
        clock = np.array([0, 1], dtype=np.uint8)

        transactions = decoder.decode_transactions(addr_channels, data_channels, clock, "rising")

        assert len(transactions) == 1
        # At index 1: address should be 0, data should be 0
        # (all channels have value 0 at index 1 except those marked with 1)


# =============================================================================
# Test BusDecoder Sampling Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderSampling:
    """Test BusDecoder sampling methods."""

    def test_sample_at_clock_rising(self) -> None:
        """Test sample_at_clock with rising edge."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {
            0: np.array([1, 0, 1, 0, 1], dtype=np.uint8),
            1: np.array([0, 1, 0, 1, 0], dtype=np.uint8),
            2: np.array([1, 1, 0, 0, 1], dtype=np.uint8),
            3: np.array([0, 0, 1, 1, 0], dtype=np.uint8),
        }
        clock = np.array([0, 1, 0, 1, 0], dtype=np.uint8)

        transactions = decoder.sample_at_clock(bit_traces, clock, "rising")

        # Rising edges at indices 1 and 3
        assert len(transactions) == 2
        assert all(isinstance(t, BusTransaction) for t in transactions)
        assert transactions[0].sample_index == 1
        assert transactions[1].sample_index == 3

    def test_sample_at_clock_falling(self) -> None:
        """Test sample_at_clock with falling edge."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([1, 0, 1, 0], dtype=np.uint8) for i in range(4)}
        clock = np.array([1, 0, 1, 0], dtype=np.uint8)

        transactions = decoder.sample_at_clock(bit_traces, clock, "falling")

        # Falling edges at indices 1 and 3
        assert len(transactions) == 2
        assert transactions[0].sample_index == 1
        assert transactions[1].sample_index == 3

    def test_sample_at_clock_missing_channel(self) -> None:
        """Test sample_at_clock when some channels are missing."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Only provide 2 channels
        bit_traces = {
            0: np.array([1, 0, 1], dtype=np.uint8),
            1: np.array([0, 1, 0], dtype=np.uint8),
        }
        clock = np.array([0, 1, 0], dtype=np.uint8)

        transactions = decoder.sample_at_clock(bit_traces, clock, "rising")

        # Should still work, missing channels treated as 0
        assert len(transactions) == 1
        assert len(transactions[0].raw_bits) == 4

    def test_sample_at_intervals_basic(self) -> None:
        """Test sample_at_intervals with basic interval."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8) for i in range(4)}

        transactions = decoder.sample_at_intervals(bit_traces, interval_samples=2)

        # Should sample at indices 0, 2, 4
        assert len(transactions) == 3
        assert transactions[0].sample_index == 0
        assert transactions[1].sample_index == 2
        assert transactions[2].sample_index == 4

    def test_sample_at_intervals_single_sample(self) -> None:
        """Test sample_at_intervals with interval=1 (all samples)."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([1, 0, 1], dtype=np.uint8) for i in range(4)}

        transactions = decoder.sample_at_intervals(bit_traces, interval_samples=1)

        # Should sample all indices
        assert len(transactions) == 3

    def test_sample_at_intervals_invalid_interval(self) -> None:
        """Test that invalid interval raises ValueError."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([1, 0, 1], dtype=np.uint8) for i in range(4)}

        with pytest.raises(ValueError, match="interval_samples must be positive"):
            decoder.sample_at_intervals(bit_traces, interval_samples=0)

        with pytest.raises(ValueError, match="interval_samples must be positive"):
            decoder.sample_at_intervals(bit_traces, interval_samples=-1)

    def test_sample_at_intervals_timestamp_calculation(self) -> None:
        """Test that timestamps are calculated correctly."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)  # 1 MHz = 1 µs per sample

        bit_traces = {i: np.array([1, 0, 1, 0], dtype=np.uint8) for i in range(4)}

        transactions = decoder.sample_at_intervals(bit_traces, interval_samples=1)

        assert len(transactions) == 4
        assert transactions[0].timestamp == 0.0
        assert transactions[1].timestamp == 1e-6
        assert transactions[2].timestamp == 2e-6
        assert transactions[3].timestamp == 3e-6


# =============================================================================
# Test BusDecoder Private Methods
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderPrivateMethods:
    """Test BusDecoder private/internal methods."""

    def test_reconstruct_value_lsb_first(self) -> None:
        """Test _reconstruct_value with LSB first."""
        config = BusConfig(name="test", width=8, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        # Binary: 10100101 (LSB first) = 0xA5
        bit_values = [1, 0, 1, 0, 0, 1, 0, 1]
        value = decoder._reconstruct_value(bit_values)
        assert value == 0xA5

    def test_reconstruct_value_msb_first(self) -> None:
        """Test _reconstruct_value with MSB first."""
        config = BusConfig(name="test", width=8, bit_order="msb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        # Binary: 10100101 (MSB first) = 0xA5
        bit_values = [1, 0, 1, 0, 0, 1, 0, 1]
        value = decoder._reconstruct_value(bit_values)
        assert value == 0xA5

    def test_reconstruct_value_empty(self) -> None:
        """Test _reconstruct_value with empty bit list."""
        config = BusConfig(name="test", width=8, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        value = decoder._reconstruct_value([])
        assert value == 0

    def test_reconstruct_value_all_zeros(self) -> None:
        """Test _reconstruct_value with all zeros."""
        config = BusConfig(name="test", width=8, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        value = decoder._reconstruct_value([0, 0, 0, 0, 0, 0, 0, 0])
        assert value == 0

    def test_reconstruct_value_all_ones(self) -> None:
        """Test _reconstruct_value with all ones."""
        config = BusConfig(name="test", width=8, bit_order="lsb_first")
        decoder = BusDecoder(config, sample_rate=1e6)

        value = decoder._reconstruct_value([1, 1, 1, 1, 1, 1, 1, 1])
        assert value == 0xFF

    def test_apply_active_low_enabled(self) -> None:
        """Test _apply_active_low when active_low is True."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first", active_low=True)
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_values = [0, 1, 0, 1]
        inverted = decoder._apply_active_low(bit_values)
        assert inverted == [1, 0, 1, 0]

    def test_apply_active_low_disabled(self) -> None:
        """Test _apply_active_low when active_low is False."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first", active_low=False)
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_values = [0, 1, 0, 1]
        result = decoder._apply_active_low(bit_values)
        assert result == [0, 1, 0, 1]


# =============================================================================
# Test Convenience Functions
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_decode_bus_function_basic(self) -> None:
        """Test decode_bus convenience function."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]

        bit_traces = {
            0: np.array([1, 0, 1], dtype=np.uint8),
            1: np.array([0, 1, 0], dtype=np.uint8),
            2: np.array([1, 1, 0], dtype=np.uint8),
            3: np.array([0, 0, 1], dtype=np.uint8),
        }

        transactions = decode_bus(bit_traces, config, sample_rate=1e6)

        assert len(transactions) == 3
        assert all(isinstance(t, BusTransaction) for t in transactions)

    def test_decode_bus_function_with_clock(self) -> None:
        """Test decode_bus convenience function with clock."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]

        bit_traces = {i: np.array([1, 0, 1, 0], dtype=np.uint8) for i in range(4)}
        clock = np.array([0, 1, 0, 1], dtype=np.uint8)

        transactions = decode_bus(
            bit_traces, config, sample_rate=1e6, clock_trace=clock, clock_edge="rising"
        )

        # Rising edges at indices 1 and 3
        assert len(transactions) == 2

    def test_decode_bus_function_from_yaml(self, tmp_path: Path) -> None:
        """Test decode_bus convenience function with YAML config path."""
        yaml_content = """
name: test_bus
width: 4
bit_order: lsb_first
bits:
  - channel: 0
    bit: 0
  - channel: 1
    bit: 1
  - channel: 2
    bit: 2
  - channel: 3
    bit: 3
"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(yaml_content)

        bit_traces = {i: np.array([1, 0], dtype=np.uint8) for i in range(4)}

        transactions = decode_bus(bit_traces, yaml_file, sample_rate=1e6)

        assert len(transactions) == 2

    def test_sample_at_clock_function(self) -> None:
        """Test sample_at_clock convenience function."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]

        bit_traces = {i: np.array([1, 0, 1, 0], dtype=np.uint8) for i in range(4)}
        clock = np.array([0, 1, 0, 1], dtype=np.uint8)

        transactions = sample_at_clock(bit_traces, clock, config, sample_rate=1e6, edge="rising")

        # Rising edges at indices 1 and 3
        assert len(transactions) == 2
        assert all(isinstance(t, BusTransaction) for t in transactions)

    def test_sample_at_clock_function_falling_edge(self) -> None:
        """Test sample_at_clock convenience function with falling edge."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]

        bit_traces = {i: np.array([1, 0, 1, 0], dtype=np.uint8) for i in range(4)}
        clock = np.array([1, 0, 1, 0], dtype=np.uint8)

        transactions = sample_at_clock(bit_traces, clock, config, sample_rate=1e6, edge="falling")

        # Falling edges at indices 1 and 3
        assert len(transactions) == 2


# =============================================================================
# Test Edge Cases
# =============================================================================


@pytest.mark.unit
@pytest.mark.digital
@pytest.mark.requirement("DSP-003")
class TestBusDecoderEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_bit_bus(self) -> None:
        """Test bus with width=1."""
        config = BusConfig(name="test", width=1, bit_order="lsb_first")
        config.bits = [{"channel": 0, "bit": 0}]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {0: np.array([0, 1, 0, 1], dtype=np.uint8)}

        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) == 4
        assert transactions[0].value == 0
        assert transactions[1].value == 1
        assert transactions[2].value == 0
        assert transactions[3].value == 1

    def test_wide_bus_16bit(self) -> None:
        """Test wide 16-bit bus."""
        config = BusConfig(name="test", width=16, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(16)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Create value 0xABCD
        bit_traces = {}
        value = 0xABCD
        for i in range(16):
            bit = (value >> i) & 1
            bit_traces[i] = np.array([bit], dtype=np.uint8)

        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) == 1
        assert transactions[0].value == 0xABCD

    def test_clock_all_high(self) -> None:
        """Test clock that stays high (no edges)."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([1, 1, 1], dtype=np.uint8) for i in range(4)}
        clock = np.array([1, 1, 1], dtype=np.uint8)

        transactions = decoder.decode_bus(bit_traces, clock_trace=clock)
        assert len(transactions) == 0

    def test_clock_all_low(self) -> None:
        """Test clock that stays low (no edges)."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([0, 0, 0], dtype=np.uint8) for i in range(4)}
        clock = np.array([0, 0, 0], dtype=np.uint8)

        transactions = decoder.decode_bus(bit_traces, clock_trace=clock)
        assert len(transactions) == 0

    def test_mismatched_trace_lengths(self) -> None:
        """Test that mismatched trace lengths are handled."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Different length traces
        bit_traces = {
            0: np.array([1, 0, 1], dtype=np.uint8),
            1: np.array([0, 1], dtype=np.uint8),  # Shorter
            2: np.array([1, 1, 0, 1], dtype=np.uint8),  # Longer
            3: np.array([0, 0, 1], dtype=np.uint8),
        }

        # Should use first trace's length
        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) > 0

    def test_boolean_traces(self) -> None:
        """Test that boolean traces work correctly."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Use boolean arrays instead of uint8
        bit_traces = {
            0: np.array([True, False, True], dtype=bool),
            1: np.array([False, True, False], dtype=bool),
            2: np.array([True, True, False], dtype=bool),
            3: np.array([False, False, True], dtype=bool),
        }

        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) == 3

    def test_large_values(self) -> None:
        """Test decoding large bus values."""
        config = BusConfig(name="test", width=32, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(32)]
        decoder = BusDecoder(config, sample_rate=1e6)

        # Create value 0xDEADBEEF
        bit_traces = {}
        value = 0xDEADBEEF
        for i in range(32):
            bit = (value >> i) & 1
            bit_traces[i] = np.array([bit], dtype=np.uint8)

        transactions = decoder.decode_bus(bit_traces)
        assert len(transactions) == 1
        assert transactions[0].value == 0xDEADBEEF

    def test_clock_edge_at_boundary(self) -> None:
        """Test clock edge at trace boundary."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([1, 0], dtype=np.uint8) for i in range(4)}
        clock = np.array([0, 1], dtype=np.uint8)  # Edge at last sample

        transactions = decoder.decode_bus(bit_traces, clock_trace=clock, clock_edge="rising")

        # Should handle edge at boundary
        assert len(transactions) == 1
        assert transactions[0].sample_index == 1

    def test_alternating_clock(self) -> None:
        """Test perfectly alternating clock signal."""
        config = BusConfig(name="test", width=4, bit_order="lsb_first")
        config.bits = [{"channel": i, "bit": i} for i in range(4)]
        decoder = BusDecoder(config, sample_rate=1e6)

        bit_traces = {i: np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8) for i in range(4)}
        clock = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)

        transactions = decoder.decode_bus(bit_traces, clock_trace=clock, clock_edge="rising")

        # Rising edges at indices 1, 3, 5
        assert len(transactions) == 3
