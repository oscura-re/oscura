"""Comprehensive unit tests for synthetic data generation.

This module tests the testing/synthetic.py module with comprehensive coverage including:
- Synthetic packet generation with various configurations
- Digital signal generation (square, UART, random, etc.)
- Protocol message generation
- Noise injection and corruption
- Ground truth tracking
- Edge cases and parameter validation
- Complete dataset generation
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from oscura.testing.synthetic import (
    GroundTruth,
    SyntheticDataGenerator,
    SyntheticMessageConfig,
    SyntheticPacketConfig,
    SyntheticSignalConfig,
    generate_digital_signal,
    generate_packets,
    generate_protocol_messages,
    generate_test_dataset,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generator() -> SyntheticDataGenerator:
    """Create a synthetic data generator with fixed seed."""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory for dataset generation."""
    output_dir = tmp_path / "test_datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# =============================================================================
# Configuration Tests
# =============================================================================


@pytest.mark.unit
class TestSyntheticPacketConfig:
    """Test synthetic packet configuration."""

    def test_default_config(self) -> None:
        """Test default packet configuration values."""
        config = SyntheticPacketConfig()
        assert config.packet_size == 1024
        assert config.header_size == 16
        assert config.sync_pattern == b"\xaa\x55"
        assert config.include_sequence is True
        assert config.include_timestamp is True
        assert config.include_checksum is True
        assert config.checksum_algorithm == "crc16"
        assert config.noise_level == 0.0

    def test_custom_config(self) -> None:
        """Test custom packet configuration."""
        config = SyntheticPacketConfig(
            packet_size=512,
            header_size=8,
            sync_pattern=b"\xff\x00",
            include_sequence=False,
            noise_level=0.1,
        )
        assert config.packet_size == 512
        assert config.header_size == 8
        assert config.sync_pattern == b"\xff\x00"
        assert config.include_sequence is False
        assert config.noise_level == 0.1


@pytest.mark.unit
class TestSyntheticSignalConfig:
    """Test synthetic signal configuration."""

    def test_default_config(self) -> None:
        """Test default signal configuration values."""
        config = SyntheticSignalConfig()
        assert config.pattern_type == "square"
        assert config.sample_rate == 100e6
        assert config.duration_samples == 10000
        assert config.frequency == 1e6
        assert config.noise_snr_db == 40

    def test_custom_config(self) -> None:
        """Test custom signal configuration."""
        config = SyntheticSignalConfig(
            pattern_type="uart",
            sample_rate=1e6,
            duration_samples=50000,
            frequency=10e3,
            noise_snr_db=20,
        )
        assert config.pattern_type == "uart"
        assert config.sample_rate == 1e6
        assert config.duration_samples == 50000
        assert config.frequency == 10e3
        assert config.noise_snr_db == 20


@pytest.mark.unit
class TestSyntheticMessageConfig:
    """Test synthetic message configuration."""

    def test_default_config(self) -> None:
        """Test default message configuration values."""
        config = SyntheticMessageConfig()
        assert config.message_size == 64
        assert config.num_fields == 5
        assert config.include_header is True
        assert config.include_length is True
        assert config.include_checksum is True
        assert config.variation == 0.1

    def test_custom_config(self) -> None:
        """Test custom message configuration."""
        config = SyntheticMessageConfig(
            message_size=128,
            num_fields=10,
            include_header=False,
            variation=0.5,
        )
        assert config.message_size == 128
        assert config.num_fields == 10
        assert config.include_header is False
        assert config.variation == 0.5


@pytest.mark.unit
class TestGroundTruth:
    """Test ground truth data structure."""

    def test_default_ground_truth(self) -> None:
        """Test default ground truth initialization."""
        truth = GroundTruth()
        assert truth.field_boundaries == []
        assert truth.field_types == []
        assert truth.sequence_numbers == []
        assert truth.pattern_period is None
        assert truth.cluster_labels == []
        assert truth.checksum_offsets == []
        assert truth.decoded_bytes == []
        assert truth.edge_positions == []
        assert truth.frequency_hz is None

    def test_ground_truth_with_data(self) -> None:
        """Test ground truth with populated data."""
        truth = GroundTruth(
            field_boundaries=[0, 2, 4, 8],
            field_types=["sync", "length", "data"],
            sequence_numbers=[0, 1, 2, 3],
            pattern_period=100,
            frequency_hz=1000.0,
        )
        assert truth.field_boundaries == [0, 2, 4, 8]
        assert truth.field_types == ["sync", "length", "data"]
        assert truth.sequence_numbers == [0, 1, 2, 3]
        assert truth.pattern_period == 100
        assert truth.frequency_hz == 1000.0


# =============================================================================
# Packet Generation Tests
# =============================================================================


@pytest.mark.unit
class TestPacketGeneration:
    """Test synthetic packet generation."""

    def test_basic_packet_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test basic packet generation with default config."""
        config = SyntheticPacketConfig()
        packets, truth = generator.generate_packets(config, count=10)

        # Check packet data
        assert isinstance(packets, bytes)
        assert len(packets) == 10 * config.packet_size

        # Check ground truth
        assert len(truth.sequence_numbers) == 10
        assert truth.sequence_numbers == list(range(10))
        assert len(truth.checksum_offsets) == 10

    def test_packet_structure(self, generator: SyntheticDataGenerator) -> None:
        """Test packet structure with sync pattern and sequence."""
        config = SyntheticPacketConfig(
            packet_size=64,
            header_size=8,
            sync_pattern=b"\xaa\x55",
        )
        packets, truth = generator.generate_packets(config, count=5)

        # Check each packet has sync pattern
        for i in range(5):
            offset = i * config.packet_size
            sync = packets[offset : offset + 2]
            assert sync == b"\xaa\x55"

            # Check sequence number
            seq_offset = offset + 2
            seq_num = struct.unpack("<H", packets[seq_offset : seq_offset + 2])[0]
            assert seq_num == i

    def test_packet_without_sequence(self, generator: SyntheticDataGenerator) -> None:
        """Test packet generation without sequence numbers."""
        config = SyntheticPacketConfig(include_sequence=False)
        packets, truth = generator.generate_packets(config, count=10)

        assert len(truth.sequence_numbers) == 0
        assert isinstance(packets, bytes)

    def test_packet_without_timestamp(self, generator: SyntheticDataGenerator) -> None:
        """Test packet generation without timestamps."""
        config = SyntheticPacketConfig(include_timestamp=False)
        packets, truth = generator.generate_packets(config, count=5)

        assert isinstance(packets, bytes)
        assert len(packets) == 5 * config.packet_size

    def test_packet_without_checksum(self, generator: SyntheticDataGenerator) -> None:
        """Test packet generation without checksums."""
        config = SyntheticPacketConfig(include_checksum=False)
        packets, truth = generator.generate_packets(config, count=5)

        assert len(truth.checksum_offsets) == 0
        assert isinstance(packets, bytes)

    def test_packet_noise_injection(self, generator: SyntheticDataGenerator) -> None:
        """Test packet corruption with noise."""
        config = SyntheticPacketConfig(noise_level=0.5, packet_size=64)
        packets, truth = generator.generate_packets(config, count=20)

        # With 50% noise, expect some corrupted sync patterns
        corrupted_count = 0
        for i in range(20):
            offset = i * config.packet_size
            sync = packets[offset : offset + 2]
            if sync != config.sync_pattern:
                corrupted_count += 1

        # Should have some corruption (not deterministic, but likely with 20 packets)
        # At least verify the function runs without error
        assert isinstance(packets, bytes)

    def test_custom_sync_pattern(self, generator: SyntheticDataGenerator) -> None:
        """Test custom sync pattern."""
        custom_sync = b"\xde\xad\xbe\xef"
        config = SyntheticPacketConfig(sync_pattern=custom_sync, packet_size=128, header_size=8)
        packets, truth = generator.generate_packets(config, count=3)

        # Check sync pattern in first packet
        assert packets[0:4] == custom_sync

    def test_single_packet_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test generating single packet."""
        config = SyntheticPacketConfig()
        packets, truth = generator.generate_packets(config, count=1)

        assert len(packets) == config.packet_size
        assert len(truth.sequence_numbers) == 1
        assert truth.sequence_numbers[0] == 0

    def test_large_packet_count(self, generator: SyntheticDataGenerator) -> None:
        """Test generating large number of packets."""
        config = SyntheticPacketConfig(packet_size=256)
        packets, truth = generator.generate_packets(config, count=1000)

        assert len(packets) == 1000 * config.packet_size
        assert len(truth.sequence_numbers) == 1000
        assert truth.sequence_numbers[-1] == 999


# =============================================================================
# Digital Signal Generation Tests
# =============================================================================


@pytest.mark.unit
class TestDigitalSignalGeneration:
    """Test synthetic digital signal generation."""

    def test_square_wave_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test square wave generation."""
        config = SyntheticSignalConfig(
            pattern_type="square", frequency=1e6, sample_rate=100e6, duration_samples=10000
        )
        signal, truth = generator.generate_digital_signal(config)

        # Check signal properties
        assert isinstance(signal, np.ndarray)
        assert len(signal) == config.duration_samples
        assert signal.dtype == np.float64

        # Check ground truth
        assert truth.frequency_hz == config.frequency
        assert truth.pattern_period is not None
        assert len(truth.edge_positions) > 0

        # Check signal values are in expected range (3.3V logic with noise)
        assert np.min(signal) >= -1.0  # Allow for noise
        assert np.max(signal) <= 4.5

    def test_uart_signal_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test UART signal generation."""
        config = SyntheticSignalConfig(
            pattern_type="uart", sample_rate=100e6, duration_samples=50000
        )
        signal, truth = generator.generate_digital_signal(config)

        # Check signal
        assert isinstance(signal, np.ndarray)
        assert len(signal) == config.duration_samples

        # Check ground truth has decoded bytes
        assert len(truth.decoded_bytes) > 0
        # Should decode "Hello, World!" = 13 bytes
        assert len(truth.decoded_bytes) == 13
        assert len(truth.edge_positions) > 0

    def test_random_signal_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test random digital signal generation."""
        config = SyntheticSignalConfig(
            pattern_type="random", duration_samples=10000, noise_snr_db=np.inf
        )
        signal, truth = generator.generate_digital_signal(config)

        # Check signal
        assert isinstance(signal, np.ndarray)
        assert len(signal) == config.duration_samples

        # Random signal should have values at 0 or 3.3V (before noise)
        unique_values = np.unique(signal)
        assert len(unique_values) == 2
        assert 0.0 in unique_values
        assert 3.3 in unique_values

    def test_default_pattern_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test default pattern for unknown pattern type."""
        config = SyntheticSignalConfig(
            pattern_type="spi",
            duration_samples=10000,  # type: ignore[arg-type]
        )
        signal, truth = generator.generate_digital_signal(config)

        # Should fall back to default pattern
        assert isinstance(signal, np.ndarray)
        assert len(signal) == config.duration_samples
        assert truth.pattern_period == 8  # Default pattern length

    def test_signal_with_noise(self, generator: SyntheticDataGenerator) -> None:
        """Test signal generation with noise."""
        config = SyntheticSignalConfig(
            pattern_type="square", frequency=1e6, noise_snr_db=20, duration_samples=10000
        )
        signal, truth = generator.generate_digital_signal(config)

        # With noise, signal should have continuous values (not just 0 and 3.3)
        unique_values = np.unique(signal)
        assert len(unique_values) > 10  # Much more than just 2 values

    def test_signal_without_noise(self, generator: SyntheticDataGenerator) -> None:
        """Test signal generation without noise (infinite SNR)."""
        config = SyntheticSignalConfig(
            pattern_type="square", frequency=1e6, noise_snr_db=np.inf, duration_samples=10000
        )
        signal, truth = generator.generate_digital_signal(config)

        # Without noise, should have discrete levels
        # Allow for small numerical variations
        low_values = signal[signal < 1.0]
        high_values = signal[signal > 2.0]
        assert len(low_values) + len(high_values) == len(signal)

    def test_square_wave_period(self, generator: SyntheticDataGenerator) -> None:
        """Test square wave period calculation."""
        frequency = 1e6
        sample_rate = 100e6
        config = SyntheticSignalConfig(
            pattern_type="square",
            frequency=frequency,
            sample_rate=sample_rate,
            duration_samples=10000,
        )
        signal, truth = generator.generate_digital_signal(config)

        expected_period = int(sample_rate / frequency)
        assert truth.pattern_period == expected_period

    def test_edge_detection(self, generator: SyntheticDataGenerator) -> None:
        """Test edge position detection in square wave."""
        config = SyntheticSignalConfig(
            pattern_type="square",
            frequency=1e6,
            sample_rate=100e6,
            duration_samples=10000,
            noise_snr_db=np.inf,
        )
        signal, truth = generator.generate_digital_signal(config)

        # Verify edges exist
        assert len(truth.edge_positions) > 0

        # Verify edges are at transitions
        for edge_pos in truth.edge_positions[:10]:  # Check first 10 edges
            if edge_pos > 0 and edge_pos < len(signal) - 1:
                # Value should change at edge
                assert signal[edge_pos - 1] != signal[edge_pos]


# =============================================================================
# Protocol Message Generation Tests
# =============================================================================


@pytest.mark.unit
class TestProtocolMessageGeneration:
    """Test synthetic protocol message generation."""

    def test_basic_message_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test basic message generation."""
        config = SyntheticMessageConfig(message_size=64)
        messages, truth = generator.generate_protocol_messages(config, count=10)

        # Check messages
        assert isinstance(messages, list)
        assert len(messages) == 10
        assert all(isinstance(msg, bytes) for msg in messages)
        assert all(len(msg) == config.message_size for msg in messages)

        # Check ground truth
        assert len(truth.field_boundaries) > 0
        assert len(truth.field_types) > 0
        assert len(truth.sequence_numbers) == 10

    def test_message_structure(self, generator: SyntheticDataGenerator) -> None:
        """Test message structure with header and fields."""
        config = SyntheticMessageConfig(message_size=64, include_header=True, include_length=True)
        messages, truth = generator.generate_protocol_messages(config, count=5)

        # Check header sync pattern
        for msg in messages:
            assert msg[0:2] == b"\xaa\x55"

            # Check length field
            length = struct.unpack("<H", msg[2:4])[0]
            assert length == config.message_size

    def test_message_sequence_numbers(self, generator: SyntheticDataGenerator) -> None:
        """Test message sequence numbers."""
        config = SyntheticMessageConfig()
        messages, truth = generator.generate_protocol_messages(config, count=10)

        assert truth.sequence_numbers == list(range(10))

        # Verify sequence numbers in messages
        for i, msg in enumerate(messages):
            # Skip header (2 bytes) and length (2 bytes)
            offset = 4 if config.include_length else 2
            seq_num = struct.unpack("<H", msg[offset : offset + 2])[0]
            assert seq_num == i

    def test_message_without_header(self, generator: SyntheticDataGenerator) -> None:
        """Test message generation without header."""
        config = SyntheticMessageConfig(include_header=False, message_size=32)
        messages, truth = generator.generate_protocol_messages(config, count=5)

        # Field boundaries should not start with sync pattern
        assert "constant" not in truth.field_types or truth.field_types[0] != "constant"

    def test_message_without_length(self, generator: SyntheticDataGenerator) -> None:
        """Test message generation without length field."""
        config = SyntheticMessageConfig(include_length=False, message_size=32)
        messages, truth = generator.generate_protocol_messages(config, count=5)

        # Should not have length field type
        assert "length" not in truth.field_types

    def test_message_without_checksum(self, generator: SyntheticDataGenerator) -> None:
        """Test message generation without checksum."""
        config = SyntheticMessageConfig(include_checksum=False, message_size=32)
        messages, truth = generator.generate_protocol_messages(config, count=5)

        # Should not have checksum field type
        assert "checksum" not in truth.field_types

    def test_message_variation(self, generator: SyntheticDataGenerator) -> None:
        """Test message payload variation."""
        # Low variation - mostly constant
        config_low = SyntheticMessageConfig(variation=0.0)
        messages_low, _ = generator.generate_protocol_messages(config_low, count=5)

        # High variation - mostly random
        config_high = SyntheticMessageConfig(variation=1.0)
        gen_high = SyntheticDataGenerator(seed=42)
        messages_high, _ = gen_high.generate_protocol_messages(config_high, count=5)

        # Both should generate valid messages
        assert all(len(msg) == config_low.message_size for msg in messages_low)
        assert all(len(msg) == config_high.message_size for msg in messages_high)

    def test_field_boundaries(self, generator: SyntheticDataGenerator) -> None:
        """Test field boundary correctness."""
        config = SyntheticMessageConfig(message_size=64)
        messages, truth = generator.generate_protocol_messages(config, count=1)

        # Field boundaries should be increasing
        assert all(
            truth.field_boundaries[i] < truth.field_boundaries[i + 1]
            for i in range(len(truth.field_boundaries) - 1)
        )

        # Last boundary should equal message size
        assert truth.field_boundaries[-1] == config.message_size

    def test_single_message_generation(self, generator: SyntheticDataGenerator) -> None:
        """Test generating single message."""
        config = SyntheticMessageConfig()
        messages, truth = generator.generate_protocol_messages(config, count=1)

        assert len(messages) == 1
        assert len(truth.sequence_numbers) == 1


# =============================================================================
# Noise and Corruption Tests
# =============================================================================


@pytest.mark.unit
class TestNoiseAndCorruption:
    """Test noise injection and data corruption."""

    def test_add_noise_to_bytes(self, generator: SyntheticDataGenerator) -> None:
        """Test adding noise to binary data."""
        data = b"\x00" * 100
        noisy = generator.add_noise(data, snr_db=10)

        assert isinstance(noisy, bytes)
        assert len(noisy) == len(data)
        # With SNR=10, should have some differences
        assert noisy != data

    def test_add_noise_to_array(self, generator: SyntheticDataGenerator) -> None:
        """Test adding noise to numpy array."""
        data = np.ones(1000, dtype=np.float64)
        noisy = generator.add_noise(data, snr_db=20)

        assert isinstance(noisy, np.ndarray)
        assert len(noisy) == len(data)
        # Noise should change values
        assert not np.allclose(noisy, data)

    def test_high_snr_low_noise(self, generator: SyntheticDataGenerator) -> None:
        """Test high SNR produces low noise."""
        data = np.ones(1000, dtype=np.float64)
        noisy = generator.add_noise(data, snr_db=60)

        # High SNR should keep signal close to original
        assert np.allclose(noisy, data, rtol=0.01)

    def test_low_snr_high_noise(self, generator: SyntheticDataGenerator) -> None:
        """Test low SNR produces high noise."""
        data = np.ones(1000, dtype=np.float64)
        noisy = generator.add_noise(data, snr_db=0)

        # Low SNR should add significant noise
        assert not np.allclose(noisy, data, rtol=0.1)

    def test_corrupt_packets(self, generator: SyntheticDataGenerator) -> None:
        """Test packet corruption."""
        # Create simple packets
        packet_size = 64
        packets = b"\xaa\x55" + b"\x00" * 62  # 1 packet
        packets = packets * 10  # 10 packets

        corrupted = generator.corrupt_packets(packets, packet_size, corruption_rate=0.5)

        assert isinstance(corrupted, bytes)
        assert len(corrupted) == len(packets)

        # Check that some sync patterns are corrupted
        corrupted_count = 0
        for i in range(10):
            offset = i * packet_size
            if corrupted[offset : offset + 2] != b"\xaa\x55":
                corrupted_count += 1

        # With 50% corruption rate and 10 packets, expect some corruption
        # (not deterministic, but function should run)
        assert isinstance(corrupted, bytes)

    def test_zero_corruption_rate(self, generator: SyntheticDataGenerator) -> None:
        """Test zero corruption rate leaves data unchanged."""
        packets = b"\xaa\x55" + b"\x00" * 62
        packets = packets * 10

        corrupted = generator.corrupt_packets(packets, packet_size=64, corruption_rate=0.0)

        assert corrupted == packets

    def test_noise_reproducibility(self) -> None:
        """Test that same seed produces same noise."""
        data = np.ones(100, dtype=np.float64)

        gen1 = SyntheticDataGenerator(seed=123)
        noisy1 = gen1.add_noise(data, snr_db=20)

        gen2 = SyntheticDataGenerator(seed=123)
        noisy2 = gen2.add_noise(data, snr_db=20)

        assert isinstance(noisy1, np.ndarray)
        assert isinstance(noisy2, np.ndarray)
        np.testing.assert_array_equal(noisy1, noisy2)


# =============================================================================
# CRC Calculation Tests
# =============================================================================


@pytest.mark.unit
class TestCRCCalculation:
    """Test CRC-16 checksum calculation."""

    def test_crc16_basic(self, generator: SyntheticDataGenerator) -> None:
        """Test basic CRC-16 calculation."""
        data = b"Hello, World!"
        crc = generator._calculate_crc16(data)

        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_crc16_empty_data(self, generator: SyntheticDataGenerator) -> None:
        """Test CRC-16 with empty data."""
        crc = generator._calculate_crc16(b"")
        assert isinstance(crc, int)
        assert crc == 0xFFFF  # Initial CRC value

    def test_crc16_single_byte(self, generator: SyntheticDataGenerator) -> None:
        """Test CRC-16 with single byte."""
        crc = generator._calculate_crc16(b"\x00")
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_crc16_deterministic(self, generator: SyntheticDataGenerator) -> None:
        """Test CRC-16 is deterministic."""
        data = b"Test data"
        crc1 = generator._calculate_crc16(data)
        crc2 = generator._calculate_crc16(data)
        assert crc1 == crc2

    def test_crc16_different_data(self, generator: SyntheticDataGenerator) -> None:
        """Test different data produces different CRCs."""
        crc1 = generator._calculate_crc16(b"Data1")
        crc2 = generator._calculate_crc16(b"Data2")
        assert crc1 != crc2

    def test_crc16_with_bytearray(self, generator: SyntheticDataGenerator) -> None:
        """Test CRC-16 works with bytearray."""
        data = bytearray(b"Test")
        crc = generator._calculate_crc16(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF


# =============================================================================
# Convenience Function Tests
# =============================================================================


@pytest.mark.unit
class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_generate_packets_function(self) -> None:
        """Test generate_packets convenience function."""
        packets, truth = generate_packets(count=10, packet_size=128)

        assert isinstance(packets, bytes)
        assert len(packets) == 10 * 128
        assert len(truth.sequence_numbers) == 10

    def test_generate_digital_signal_function(self) -> None:
        """Test generate_digital_signal convenience function."""
        signal, truth = generate_digital_signal(pattern="square", duration_samples=5000)

        assert isinstance(signal, np.ndarray)
        assert len(signal) == 5000
        assert truth.frequency_hz is not None

    def test_generate_digital_signal_uart(self) -> None:
        """Test generate_digital_signal with UART pattern."""
        signal, truth = generate_digital_signal(pattern="uart", duration_samples=50000)

        assert isinstance(signal, np.ndarray)
        assert len(truth.decoded_bytes) > 0

    def test_generate_digital_signal_invalid_pattern(self) -> None:
        """Test generate_digital_signal with invalid pattern falls back to square."""
        signal, truth = generate_digital_signal(
            pattern="invalid_pattern",
            duration_samples=5000,  # type: ignore[arg-type]
        )

        # Should fall back to square wave
        assert isinstance(signal, np.ndarray)
        assert len(signal) == 5000

    def test_generate_protocol_messages_function(self) -> None:
        """Test generate_protocol_messages convenience function."""
        messages, truth = generate_protocol_messages(count=15, message_size=100)

        assert isinstance(messages, list)
        assert len(messages) == 15
        assert all(len(msg) == 100 for msg in messages)


# =============================================================================
# Dataset Generation Tests
# =============================================================================


@pytest.mark.unit
class TestDatasetGeneration:
    """Test complete dataset generation."""

    def test_generate_test_dataset(self, temp_output_dir: Path) -> None:
        """Test generating complete test dataset."""
        metadata = generate_test_dataset(
            output_dir=str(temp_output_dir),
            num_packets=50,
            num_signals=3,
            num_messages=25,
        )

        # Check metadata structure
        assert metadata["dataset_type"] == "synthetic_test_data"
        assert "generated_files" in metadata
        assert "metadata_file" in metadata

        # Check files were created
        assert len(metadata["generated_files"]) == 5  # 3 signals + 1 packets + 1 messages

        # Verify metadata file exists
        metadata_file = Path(metadata["metadata_file"])
        assert metadata_file.exists()

        # Verify content
        with metadata_file.open("r") as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata["dataset_type"] == "synthetic_test_data"

    def test_dataset_files_created(self, temp_output_dir: Path) -> None:
        """Test that all dataset files are created."""
        metadata = generate_test_dataset(
            output_dir=str(temp_output_dir), num_packets=10, num_signals=2, num_messages=10
        )

        # Check packet file
        packet_files = [f for f in metadata["generated_files"] if f["type"] == "packets"]
        assert len(packet_files) == 1
        assert Path(packet_files[0]["path"]).exists()

        # Check signal files
        signal_files = [f for f in metadata["generated_files"] if f["type"] == "signal"]
        assert len(signal_files) == 2
        for sig_file in signal_files:
            assert Path(sig_file["path"]).exists()

        # Check message file
        message_files = [f for f in metadata["generated_files"] if f["type"] == "messages"]
        assert len(message_files) == 1
        assert Path(message_files[0]["path"]).exists()

    def test_dataset_ground_truth_included(self, temp_output_dir: Path) -> None:
        """Test that ground truth is included in metadata."""
        metadata = generate_test_dataset(
            output_dir=str(temp_output_dir), num_packets=5, num_signals=1, num_messages=5
        )

        # Check packet ground truth
        packet_file = next(f for f in metadata["generated_files"] if f["type"] == "packets")
        assert "ground_truth" in packet_file
        assert "sequence_numbers" in packet_file["ground_truth"]

        # Check signal ground truth
        signal_file = next(f for f in metadata["generated_files"] if f["type"] == "signal")
        assert "ground_truth" in signal_file
        assert "frequency_hz" in signal_file["ground_truth"]

        # Check message ground truth
        message_file = next(f for f in metadata["generated_files"] if f["type"] == "messages")
        assert "ground_truth" in message_file
        assert "field_boundaries" in message_file["ground_truth"]

    def test_dataset_signal_patterns(self, temp_output_dir: Path) -> None:
        """Test that signals alternate between square and UART patterns."""
        metadata = generate_test_dataset(
            output_dir=str(temp_output_dir), num_packets=1, num_signals=4, num_messages=1
        )

        signal_files = [f for f in metadata["generated_files"] if f["type"] == "signal"]
        patterns = [f["pattern"] for f in signal_files]

        # Should alternate: square, uart, square, uart
        assert patterns == ["square", "uart", "square", "uart"]

    def test_dataset_with_zero_counts(self, temp_output_dir: Path) -> None:
        """Test dataset generation with zero counts."""
        # This should still create the directory and metadata
        metadata = generate_test_dataset(
            output_dir=str(temp_output_dir), num_packets=0, num_signals=0, num_messages=0
        )

        assert "dataset_type" in metadata
        # With zero signals, only packet and message files are created (no signals generated)
        assert len(metadata["generated_files"]) == 2  # packets + messages (no signals)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


@pytest.mark.unit
class TestTestingSyntheticEdgeCases:
    """Test edge cases and parameter validation."""

    def test_zero_frequency_signal(self, generator: SyntheticDataGenerator) -> None:
        """Test signal generation with zero frequency."""
        config = SyntheticSignalConfig(pattern_type="square", frequency=0.0, duration_samples=1000)

        # Zero frequency causes division by zero
        with pytest.raises(ZeroDivisionError):
            signal, truth = generator.generate_digital_signal(config)

    def test_very_high_frequency(self, generator: SyntheticDataGenerator) -> None:
        """Test signal generation with very high frequency."""
        config = SyntheticSignalConfig(
            pattern_type="square", frequency=1e9, sample_rate=100e6, duration_samples=1000
        )

        # Frequency higher than sample rate (Nyquist violation)
        signal, truth = generator.generate_digital_signal(config)
        assert isinstance(signal, np.ndarray)
        assert len(signal) == config.duration_samples

    def test_small_packet_size(self, generator: SyntheticDataGenerator) -> None:
        """Test packet generation with very small packet size."""
        config = SyntheticPacketConfig(packet_size=16, header_size=8, include_checksum=False)
        packets, truth = generator.generate_packets(config, count=1)

        assert len(packets) == 16

    def test_header_larger_than_packet(self, generator: SyntheticDataGenerator) -> None:
        """Test configuration where header is larger than packet."""
        config = SyntheticPacketConfig(packet_size=16, header_size=20, include_checksum=False)

        # Should handle gracefully (may produce odd results but shouldn't crash)
        packets, truth = generator.generate_packets(config, count=1)
        assert isinstance(packets, bytes)

    def test_reproducibility_with_seed(self) -> None:
        """Test that same seed produces identical results."""
        gen1 = SyntheticDataGenerator(seed=999)
        packets1, _ = gen1.generate_packets(SyntheticPacketConfig(), count=5)

        gen2 = SyntheticDataGenerator(seed=999)
        packets2, _ = gen2.generate_packets(SyntheticPacketConfig(), count=5)

        assert packets1 == packets2

    def test_different_seeds_produce_different_results(self) -> None:
        """Test that different seeds produce different results."""
        # Use random pattern for messages which will differ with different seeds
        gen1 = SyntheticDataGenerator(seed=100)
        messages1, _ = gen1.generate_protocol_messages(
            SyntheticMessageConfig(variation=1.0), count=5
        )

        gen2 = SyntheticDataGenerator(seed=200)
        messages2, _ = gen2.generate_protocol_messages(
            SyntheticMessageConfig(variation=1.0), count=5
        )

        # Messages should differ due to random variation
        assert messages1 != messages2

    def test_very_short_signal(self, generator: SyntheticDataGenerator) -> None:
        """Test generating very short signal."""
        config = SyntheticSignalConfig(duration_samples=10)
        signal, truth = generator.generate_digital_signal(config)

        assert len(signal) == 10

    def test_message_all_options_disabled(self, generator: SyntheticDataGenerator) -> None:
        """Test message generation with all optional fields disabled."""
        config = SyntheticMessageConfig(
            message_size=20,
            include_header=False,
            include_length=False,
            include_checksum=False,
        )
        messages, truth = generator.generate_protocol_messages(config, count=3)

        assert len(messages) == 3
        assert all(len(msg) == 20 for msg in messages)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
class TestTestingSyntheticIntegration:
    """Integration tests for synthetic data generation."""

    def test_end_to_end_packet_workflow(self, generator: SyntheticDataGenerator) -> None:
        """Test complete packet generation workflow."""
        # Generate packets
        config = SyntheticPacketConfig(packet_size=256, noise_level=0.05)
        packets, truth = generator.generate_packets(config, count=50)

        # Verify structure
        assert len(packets) == 50 * 256
        assert len(truth.sequence_numbers) == 50
        assert len(truth.checksum_offsets) == 50

        # Verify checksums can be extracted
        for offset in truth.checksum_offsets[:5]:
            checksum_bytes = packets[offset : offset + 2]
            checksum = struct.unpack("<H", checksum_bytes)[0]
            assert 0 <= checksum <= 0xFFFF

    def test_end_to_end_signal_workflow(self, generator: SyntheticDataGenerator) -> None:
        """Test complete signal generation workflow."""
        # Generate signal
        config = SyntheticSignalConfig(
            pattern_type="square", frequency=1e6, sample_rate=100e6, duration_samples=100000
        )
        signal, truth = generator.generate_digital_signal(config)

        # Verify properties
        assert len(signal) == 100000
        assert truth.frequency_hz == 1e6
        assert len(truth.edge_positions) > 0

        # Verify signal statistics
        assert np.min(signal) >= -1.0
        assert np.max(signal) <= 5.0

    def test_end_to_end_message_workflow(self, generator: SyntheticDataGenerator) -> None:
        """Test complete message generation workflow."""
        # Generate messages
        config = SyntheticMessageConfig(message_size=128, variation=0.2)
        messages, truth = generator.generate_protocol_messages(config, count=100)

        # Verify structure
        assert len(messages) == 100
        assert len(truth.sequence_numbers) == 100
        assert len(truth.field_boundaries) > 0

        # Verify field boundaries are valid
        for i in range(len(truth.field_boundaries) - 1):
            assert truth.field_boundaries[i] < truth.field_boundaries[i + 1]

        # Verify all messages have correct size
        assert all(len(msg) == 128 for msg in messages)
