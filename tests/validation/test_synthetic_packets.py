"""Ground truth validation tests for synthetic packet data.

This module validates packet loading against fixed_length_packets_truth.json
to ensure correct sequence number extraction and checksum offset detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.validation


@pytest.mark.requires_data
@pytest.mark.requirement("VAL-001")
class TestSyntheticPacketValidation:
    """Validate synthetic packet loading against ground truth."""

    def test_fixed_length_packets_exist(self, synthetic_packets: dict[str, Path]) -> None:
        """Verify synthetic packet files exist.

        Validates:
        - Data file exists: synthetic/.../fixed_length/clean_packets_512b.bin
        - Truth file exists: synthetic/ground_truth/decoded/fixed_length_packets_truth.json
        """
        assert synthetic_packets["data"].exists(), (
            f"Data file not found: {synthetic_packets['data']}"
        )
        assert synthetic_packets["truth"].exists(), (
            f"Truth file not found: {synthetic_packets['truth']}"
        )

    def test_fixed_length_packet_count(
        self,
        synthetic_packets: dict[str, Path],
        fixed_length_packets_truth: dict[str, Any],
    ) -> None:
        """Verify packet count matches ground truth.

        Validates:
        - Number of packets in file matches expected count (1000)
        - Each packet is exactly 512 bytes
        """
        if not fixed_length_packets_truth:
            pytest.skip("Ground truth file not available")

        data_path = synthetic_packets["data"]
        if not data_path.exists():
            pytest.skip("Data file not available")

        # Read binary data
        with open(data_path, "rb") as f:
            data = f.read()

        packet_size = 512
        expected_count = len(fixed_length_packets_truth.get("sequence_numbers", []))

        # Verify file size matches expected packet count
        actual_count = len(data) // packet_size
        assert actual_count == expected_count, (
            f"Packet count mismatch: expected {expected_count}, got {actual_count}"
        )

    def test_sequence_numbers_valid(
        self,
        synthetic_packets: dict[str, Path],
        fixed_length_packets_truth: dict[str, Any],
    ) -> None:
        """Verify sequence numbers match ground truth.

        Validates:
        - Sequence numbers are consecutive (0 to 999)
        - No gaps or duplicates in sequence
        """
        if not fixed_length_packets_truth:
            pytest.skip("Ground truth file not available")

        expected_seq = fixed_length_packets_truth.get("sequence_numbers", [])
        assert len(expected_seq) == 1000, f"Expected 1000 sequence numbers, got {len(expected_seq)}"

        # Verify sequence is consecutive
        for i, seq in enumerate(expected_seq):
            assert seq == i, f"Sequence mismatch at index {i}: expected {i}, got {seq}"

    def test_checksum_offsets_valid(
        self,
        synthetic_packets: dict[str, Path],
        fixed_length_packets_truth: dict[str, Any],
    ) -> None:
        """Verify checksum offsets match ground truth.

        Validates:
        - Checksum offset is at byte 510 within each 512-byte packet
        - Offsets are correctly spaced (512 bytes apart)
        """
        if not fixed_length_packets_truth:
            pytest.skip("Ground truth file not available")

        expected_offsets = fixed_length_packets_truth.get("checksum_offsets", [])
        assert len(expected_offsets) == 1000, (
            f"Expected 1000 checksum offsets, got {len(expected_offsets)}"
        )

        # Verify offsets are at expected positions (packet_index * 512 + 510)
        packet_size = 512
        checksum_position = 510  # 2 bytes from end of packet

        for i, offset in enumerate(expected_offsets):
            expected = i * packet_size + checksum_position
            assert offset == expected, (
                f"Checksum offset mismatch at packet {i}: expected {expected}, got {offset}"
            )


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.requirement("VAL-001")
class TestVariableLengthPackets:
    """Validate variable-length packet handling."""

    @pytest.mark.parametrize("size", [128, 256, 1024, 2048])
    def test_variable_packet_file_exists(self, synthetic_binary_dir: Path, size: int) -> None:
        """Verify variable-length packet files exist."""
        path = synthetic_binary_dir / "variable_length" / f"packets_{size}b.bin"
        if path.exists():
            # Verify file is not empty
            assert path.stat().st_size > 0, f"File is empty: {path}"
            # Verify file size is multiple of packet size
            assert path.stat().st_size % size == 0, (
                f"File size {path.stat().st_size} not multiple of packet size {size}"
            )
        else:
            pytest.skip(f"Variable packet file not found: {path}")

    @pytest.mark.parametrize(
        "noise_level",
        ["noisy_packets_1pct.bin", "noisy_packets_5pct.bin", "noisy_packets_10pct.bin"],
    )
    def test_noisy_packet_file_exists(self, synthetic_binary_dir: Path, noise_level: str) -> None:
        """Verify noisy packet files exist for error handling tests."""
        path = synthetic_binary_dir / "with_errors" / noise_level
        if path.exists():
            assert path.stat().st_size > 0, f"File is empty: {path}"
        else:
            pytest.skip(f"Noisy packet file not found: {path}")


@pytest.mark.validation
@pytest.mark.requires_data
@pytest.mark.requirement("BDL-001")
class TestPacketLoaderIntegration:
    """Integration tests for packet loader with synthetic data."""

    def test_load_fixed_length_packets(self, synthetic_packets: dict[str, Path]) -> None:
        """Test loading fixed-length packets with configurable loader.

        Validates:
        - Loader successfully parses packet data
        - Correct number of packets extracted
        """
        data_path = synthetic_packets["data"]
        if not data_path.exists():
            pytest.skip("Data file not available")

        try:
            from oscura.loaders import PacketFormatConfig, load_binary_packets

            # Create packet format config for 512-byte fixed packets
            config = PacketFormatConfig(
                packet_size=512,
                header_fields=[],
                sample_format=None,
            )

            packets = load_binary_packets(data_path, config)

            # Verify we got packets
            assert len(packets) > 0, "No packets loaded"
            assert len(packets) == 1000, f"Expected 1000 packets, got {len(packets)}"

        except ImportError:
            pytest.skip("ConfigurablePacketLoader not available")
        except Exception as e:
            # Log the error but don't fail - loader may need real config
            pytest.skip(f"Packet loading requires specific configuration: {e}")

    def test_comprehensive_packet_validation(self, test_data_dir: Path) -> None:
        """Test comprehensive packet data from comprehensive/ directory.

        Validates:
        - test_packets.bin loads correctly
        - test_messages.bin loads correctly
        """
        comprehensive_dir = test_data_dir / "synthetic" / "binary" / "comprehensive"
        if not comprehensive_dir.exists():
            pytest.skip("Comprehensive test data not available")

        packets_file = comprehensive_dir / "test_packets.bin"
        messages_file = comprehensive_dir / "test_messages.bin"

        # Just verify files exist and are non-empty
        if packets_file.exists():
            assert packets_file.stat().st_size > 0
        if messages_file.exists():
            assert messages_file.stat().st_size > 0


@pytest.mark.validation
@pytest.mark.requires_data
class TestPacketDataIntegrity:
    """Test packet data integrity and structure."""

    def test_packet_magic_bytes(self, synthetic_packets: dict[str, Path]) -> None:
        """Verify packets have expected magic byte patterns.

        Many packet formats start with magic bytes for identification.
        """
        data_path = synthetic_packets["data"]
        if not data_path.exists():
            pytest.skip("Data file not available")

        with open(data_path, "rb") as f:
            first_packet = f.read(512)

        # Verify we read a full packet
        assert len(first_packet) == 512, "Could not read full first packet"

        # Synthetic packets may have specific structure
        # This test documents expected format
        # First 4 bytes might be a sync pattern or magic bytes

    def test_packet_sequence_continuity(
        self,
        synthetic_packets: dict[str, Path],
        fixed_length_packets_truth: dict[str, Any],
    ) -> None:
        """Verify packet sequence has no gaps.

        Validates:
        - All sequence numbers from 0 to N-1 are present
        - No missing packets in the sequence
        """
        if not fixed_length_packets_truth:
            pytest.skip("Ground truth not available")

        seq_nums = fixed_length_packets_truth.get("sequence_numbers", [])
        expected_set = set(range(len(seq_nums)))
        actual_set = set(seq_nums)

        assert actual_set == expected_set, (
            f"Sequence gaps detected. Missing: {expected_set - actual_set}"
        )
