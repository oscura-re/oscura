"""Unit tests for fuzzy synchronization pattern search.

This module tests the fuzzy sync pattern matching functionality for finding
sync words and markers in noisy or corrupted logic analyzer captures.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.exploratory.sync import (
    PacketParseResult,
    RecoveryStrategy,
    SyncMatch,
    fuzzy_sync_search,
    hamming_distance,
    parse_variable_length_packets,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Hamming Distance Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-001")
class TestHammingDistance:
    """Test Hamming distance calculation."""

    def test_identical_values(self) -> None:
        """Test Hamming distance between identical values is 0."""
        assert hamming_distance(0b10101010, 0b10101010, 8) == 0
        assert hamming_distance(0xAA55, 0xAA55, 16) == 0
        assert hamming_distance(0xDEADBEEF, 0xDEADBEEF, 32) == 0

    def test_single_bit_difference(self) -> None:
        """Test Hamming distance with single bit difference."""
        assert hamming_distance(0b10101010, 0b10101011, 8) == 1
        assert hamming_distance(0xAA55, 0xAA54, 16) == 1
        assert hamming_distance(0b00000000, 0b00000001, 8) == 1

    def test_two_bit_difference(self) -> None:
        """Test Hamming distance with two bit differences."""
        assert hamming_distance(0b10101010, 0b10101001, 8) == 2
        assert hamming_distance(0b00000000, 0b00000011, 8) == 2

    def test_all_bits_different(self) -> None:
        """Test Hamming distance when all bits differ."""
        assert hamming_distance(0b00000000, 0b11111111, 8) == 8
        assert hamming_distance(0x0000, 0xFFFF, 16) == 16
        assert hamming_distance(0x00000000, 0xFFFFFFFF, 32) == 32

    def test_pattern_bits_masking(self) -> None:
        """Test that only specified pattern bits are compared."""
        # Difference in upper bits should be ignored for 8-bit comparison
        assert hamming_distance(0x00, 0x100, 8) == 0
        assert hamming_distance(0xAA, 0x1AA, 8) == 0

    def test_various_pattern_lengths(self) -> None:
        """Test Hamming distance with different pattern lengths."""
        # 8-bit
        assert hamming_distance(0xFF, 0x00, 8) == 8
        # 16-bit
        assert hamming_distance(0xFFFF, 0x0000, 16) == 16
        # 32-bit
        assert hamming_distance(0xFFFFFFFF, 0x00000000, 32) == 32
        # 64-bit
        assert hamming_distance(0xFFFFFFFFFFFFFFFF, 0x0000000000000000, 64) == 64


# =============================================================================
# Fuzzy Sync Search Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-001")
class TestFuzzySyncSearch:
    """Test fuzzy sync pattern search functionality."""

    def test_exact_match_8bit(self) -> None:
        """Test finding exact 8-bit sync pattern."""
        data = np.array([0x00, 0x00, 0xAA, 0x00, 0x00], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=0)

        assert len(matches) == 1
        assert matches[0].index == 2
        assert matches[0].matched_value == 0xAA
        assert matches[0].hamming_distance == 0
        assert matches[0].confidence == 1.0
        assert matches[0].pattern_length == 8

    def test_exact_match_16bit(self) -> None:
        """Test finding exact 16-bit sync pattern."""
        data = np.array([0x00, 0xAA, 0x55, 0x00], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        assert len(matches) == 1
        assert matches[0].index == 1
        assert matches[0].matched_value == 0xAA55
        assert matches[0].hamming_distance == 0
        assert matches[0].confidence == 1.0

    def test_exact_match_32bit(self) -> None:
        """Test finding exact 32-bit sync pattern."""
        data = np.array([0x00, 0xAA, 0x55, 0xF0, 0xF0, 0x00], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA55F0F0, pattern_bits=32, max_errors=0)

        assert len(matches) == 1
        assert matches[0].index == 1
        assert matches[0].matched_value == 0xAA55F0F0
        assert matches[0].hamming_distance == 0

    def test_fuzzy_match_one_bit_error(self) -> None:
        """Test finding pattern with 1 bit error."""
        # 0xAA54 instead of 0xAA55 (1 bit different)
        data = np.array([0xAA, 0x54, 0x00], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=2)

        assert len(matches) == 1
        assert matches[0].index == 0
        assert matches[0].matched_value == 0xAA54
        assert matches[0].hamming_distance == 1
        assert matches[0].confidence == 1.0 - (1 / 16)

    def test_fuzzy_match_two_bit_errors(self) -> None:
        """Test finding pattern with 2 bit errors."""
        # Create pattern with 2 bit flips
        # 0xAA55 = 0b1010101001010101
        # 0xAA50 = 0b1010101001010000 (bits 0 and 2 flipped, hamming distance = 2)
        data = np.array([0xAA, 0x50, 0x00], dtype=np.uint8)  # 0x50 vs 0x55
        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=2)

        assert len(matches) == 1
        assert matches[0].hamming_distance == 2
        assert matches[0].confidence == 1.0 - (2 / 16)

    def test_multiple_matches(self) -> None:
        """Test finding multiple sync patterns."""
        data = np.array([0xAA, 0x55, 0x00, 0xAA, 0x55, 0xFF], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        assert len(matches) == 2
        assert matches[0].index == 0
        assert matches[1].index == 3

    def test_overlapping_matches(self) -> None:
        """Test that overlapping windows are searched."""
        # Pattern 0xAA appears at multiple positions
        data = np.array([0xAA, 0xAA, 0xAA], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=0)

        assert len(matches) == 3

    def test_no_match_too_many_errors(self) -> None:
        """Test that patterns with too many errors are rejected."""
        # 0x00 vs 0xFF = 8 bit errors
        data = np.array([0x00, 0x00], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xFF, pattern_bits=8, max_errors=2)

        assert len(matches) == 0

    def test_confidence_threshold(self) -> None:
        """Test min_confidence threshold filtering."""
        # 2 bit errors in 8 bits = confidence 0.75
        data = np.array([0b11111100], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xFF, pattern_bits=8, max_errors=2, min_confidence=0.8)

        # Should be rejected due to low confidence
        assert len(matches) == 0

        # Lower threshold should accept it
        matches = fuzzy_sync_search(data, 0xFF, pattern_bits=8, max_errors=2, min_confidence=0.7)
        assert len(matches) == 1

    def test_empty_data(self) -> None:
        """Test search on empty data array."""
        data = np.array([], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=0)

        assert len(matches) == 0

    def test_data_too_short(self) -> None:
        """Test search when data is shorter than pattern."""
        data = np.array([0xAA], dtype=np.uint8)
        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        assert len(matches) == 0

    def test_64bit_pattern(self) -> None:
        """Test searching for 64-bit pattern."""
        data = np.array(
            [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE, 0x00],
            dtype=np.uint8,
        )
        pattern = 0xDEADBEEFCAFEBABE
        matches = fuzzy_sync_search(data, pattern, pattern_bits=64, max_errors=0)

        assert len(matches) == 1
        assert matches[0].index == 0
        assert matches[0].matched_value == pattern


# =============================================================================
# Fuzzy Sync Search - Validation Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-001")
class TestFuzzySyncSearchValidation:
    """Test input validation for fuzzy sync search."""

    def test_invalid_pattern_bits(self) -> None:
        """Test that invalid pattern_bits raises ValueError."""
        data = np.array([0xAA, 0x55], dtype=np.uint8)

        with pytest.raises(ValueError, match="pattern_bits must be"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=12, max_errors=0)

        with pytest.raises(ValueError, match="pattern_bits must be"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=7, max_errors=0)

    def test_invalid_max_errors_negative(self) -> None:
        """Test that negative max_errors raises ValueError."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="max_errors must be"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=-1)

    def test_invalid_max_errors_too_large(self) -> None:
        """Test that max_errors > 8 raises ValueError."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="max_errors must be"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=9)

    def test_invalid_min_confidence_below_zero(self) -> None:
        """Test that min_confidence < 0 raises ValueError."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="min_confidence must be"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=0, min_confidence=-0.1)

    def test_invalid_min_confidence_above_one(self) -> None:
        """Test that min_confidence > 1 raises ValueError."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="min_confidence must be"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=0, min_confidence=1.1)

    def test_valid_boundary_values(self) -> None:
        """Test that boundary values are accepted."""
        data = np.array([0xAA, 0x55], dtype=np.uint8)

        # Should not raise
        fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=0, min_confidence=0.0)
        fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=8, min_confidence=1.0)


# =============================================================================
# Variable Length Packet Parsing Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-002")
class TestVariableLengthPacketParsing:
    """Test variable-length packet parsing with error recovery."""

    def test_simple_valid_packet(self) -> None:
        """Test parsing single valid packet."""
        # Sync (0xAA55) + Length (0x0004) + Data (0x01, 0x02)
        data = np.array([0xAA, 0x55, 0x00, 0x06, 0x01, 0x02], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
        )

        assert len(result.packets) == 1
        assert result.packets[0] == bytes([0xAA, 0x55, 0x00, 0x06, 0x01, 0x02])
        assert result.valid[0] is True
        assert result.errors[0] is None
        assert result.recovery_count == 0

    def test_multiple_valid_packets(self) -> None:
        """Test parsing multiple consecutive valid packets."""
        # Two packets: each is sync + length + data
        data = np.array(
            [
                0xAA,
                0x55,
                0x00,
                0x05,
                0x01,  # Packet 1
                0xAA,
                0x55,
                0x00,
                0x06,
                0x02,
                0x03,  # Packet 2
            ],
            dtype=np.uint8,
        )

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
        )

        assert len(result.packets) == 2
        assert result.recovery_count == 0

    def test_length_field_1byte(self) -> None:
        """Test parsing with 1-byte length field."""
        # Sync (0xAA) + Length (0x04) + Data (0x01, 0x02)
        data = np.array([0xAA, 0x04, 0x01, 0x02], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA,
            sync_bits=8,
            length_offset=1,
            length_size=1,
        )

        assert len(result.packets) == 1
        assert result.packets[0] == bytes([0xAA, 0x04, 0x01, 0x02])

    def test_truncated_packet(self) -> None:
        """Test detection of truncated packet at end of data."""
        # Packet says length=10 but only 6 bytes available
        data = np.array([0xAA, 0x55, 0x00, 0x0A, 0x01, 0x02], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
        )

        assert len(result.packets) == 0
        assert "truncated" in result.errors


# =============================================================================
# Error Recovery Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-002")
class TestErrorRecovery:
    """Test error recovery strategies."""

    def test_next_sync_recovery(self) -> None:
        """Test NEXT_SYNC recovery strategy."""
        # Good packet, corrupted length, good packet
        data = np.array(
            [
                0xAA,
                0x55,
                0x00,
                0x05,
                0x01,  # Good packet
                0xAA,
                0x55,
                0xFF,
                0xFF,
                0xFF,  # Corrupted length (0xFFFF)
                0xAA,
                0x55,
                0x00,
                0x05,
                0x02,  # Good packet
            ],
            dtype=np.uint8,
        )

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.NEXT_SYNC,
        )

        # Should recover and find the third packet
        assert len(result.packets) >= 1
        assert result.recovery_count >= 1

    def test_skip_bytes_recovery(self) -> None:
        """Test SKIP_BYTES recovery strategy."""
        # Good packet, corrupted byte, good packet
        data = np.array(
            [
                0xAA,
                0x55,
                0x00,
                0x05,
                0x01,  # Good packet
                0xFF,  # Corrupted byte (skip over this)
                0xAA,
                0x55,
                0x00,
                0x05,
                0x02,  # Good packet
            ],
            dtype=np.uint8,
        )

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
            skip_bytes=1,
        )

        # Should find both good packets
        assert len(result.packets) >= 1
        assert result.recovery_count >= 1

    def test_heuristic_recovery(self) -> None:
        """Test HEURISTIC recovery strategy using median length."""
        # Several good packets to establish median, then corrupted, then good
        data = np.array(
            [
                0xAA,
                0x55,
                0x00,
                0x05,
                0x01,  # Length 5
                0xAA,
                0x55,
                0x00,
                0x05,
                0x02,  # Length 5
                0xAA,
                0x55,
                0x00,
                0x05,
                0x03,  # Length 5
                0xAA,
                0x55,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                0xFF,  # Corrupted
                0xAA,
                0x55,
                0x00,
                0x05,
                0x04,  # Good packet
            ],
            dtype=np.uint8,
        )

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.HEURISTIC,
        )

        # Should recover using median length (5)
        assert len(result.packets) >= 3
        assert result.recovery_count >= 1


# =============================================================================
# Error Detection Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-002")
class TestErrorDetection:
    """Test error detection in packet parsing."""

    def test_zero_length_detection(self) -> None:
        """Test detection of zero-length field."""
        data = np.array([0xAA, 0x55, 0x00, 0x00, 0xFF], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
            skip_bytes=1,
        )

        assert result.recovery_count >= 1
        assert "length_corruption" in result.errors

    def test_excessive_length_detection(self) -> None:
        """Test detection of length exceeding max_packet_size."""
        data = np.array([0xAA, 0x55, 0x10, 0x00, 0xFF], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            max_packet_size=1024,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
            skip_bytes=1,
        )

        # 0x1000 = 4096 > 1024
        assert result.recovery_count >= 1

    def test_ff_pattern_detection(self) -> None:
        """Test detection of 0xFF00 pattern indicating corruption."""
        data = np.array([0xAA, 0x55, 0xFF, 0x00, 0xFF], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
            skip_bytes=1,
        )

        # 0xFF00 pattern should be detected as corruption
        assert result.recovery_count >= 1

    def test_length_below_minimum(self) -> None:
        """Test detection of length below min_packet_size."""
        data = np.array([0xAA, 0x55, 0x00, 0x02, 0xFF], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            min_packet_size=4,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
            skip_bytes=1,
        )

        # Length 2 < min 4
        assert result.recovery_count >= 1

    def test_suspicious_length_heuristic(self) -> None:
        """Test detection of suspiciously large lengths using statistics."""
        # Build up packet length history
        data = np.array(
            [
                # 10 packets of length 5
                *([0xAA, 0x55, 0x00, 0x05, 0x01] * 10),
                # Then one suspicious packet claiming length 50
                0xAA,
                0x55,
                0x00,
                0x32,
                0xFF,
            ],
            dtype=np.uint8,
        )

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            max_packet_size=1000,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
            skip_bytes=1,
        )

        # Should detect length 50 as suspicious (>2x 90th percentile)
        assert result.recovery_count >= 1


# =============================================================================
# Packet Parsing Validation Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-002")
class TestPacketParsingValidation:
    """Test input validation for packet parsing."""

    def test_invalid_length_size(self) -> None:
        """Test that invalid length_size raises ValueError."""
        data = np.array([0xAA, 0x55], dtype=np.uint8)

        with pytest.raises(ValueError, match="length_size must be"):
            parse_variable_length_packets(
                data,
                sync_pattern=0xAA55,
                sync_bits=16,
                length_size=3,
            )

    def test_next_sync_without_pattern(self) -> None:
        """Test that NEXT_SYNC without sync_pattern raises ValueError."""
        data = np.array([0xAA, 0x55], dtype=np.uint8)

        with pytest.raises(ValueError, match="NEXT_SYNC strategy requires"):
            parse_variable_length_packets(
                data,
                sync_pattern=None,
                recovery_strategy=RecoveryStrategy.NEXT_SYNC,
            )

    def test_valid_without_sync_pattern(self) -> None:
        """Test parsing without sync pattern (pure length-based)."""
        # Valid use case: no sync pattern, just length fields
        data = np.array([0x00, 0x04, 0x01, 0x02], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=None,
            length_offset=0,
            length_size=2,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
        )

        # Should work without sync pattern
        assert isinstance(result, PacketParseResult)


# =============================================================================
# Edge Cases and Stress Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-002")
class TestExploratorySyncEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_data_array(self) -> None:
        """Test parsing empty data array."""
        data = np.array([], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
        )

        assert len(result.packets) == 0
        assert result.recovery_count == 0

    def test_single_byte_data(self) -> None:
        """Test parsing single byte (insufficient for any packet)."""
        data = np.array([0xAA], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA,
            sync_bits=8,
            length_offset=1,
            length_size=1,
        )

        assert len(result.packets) == 0

    def test_header_only_no_data(self) -> None:
        """Test packet with header but no payload."""
        data = np.array([0xAA, 0x55, 0x00, 0x04], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
        )

        # Packet claims length 4 and we have exactly 4 bytes
        assert len(result.packets) == 1

    def test_all_corrupted_data(self) -> None:
        """Test data that is entirely corrupted."""
        data = np.array([0xFF] * 20, dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.NEXT_SYNC,
        )

        # Should find no valid packets
        assert len(result.packets) == 0

    def test_sync_at_end_of_data(self) -> None:
        """Test sync pattern appearing at very end of data."""
        data = np.array([0x00, 0x00, 0xAA, 0x55], dtype=np.uint8)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
        )

        # Not enough data for length field
        assert len(result.packets) == 0

    def test_large_packet_count(self) -> None:
        """Test parsing many small packets."""
        # 100 packets of 5 bytes each
        packet = np.array([0xAA, 0x55, 0x00, 0x05, 0x01], dtype=np.uint8)
        data = np.tile(packet, 100)

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
        )

        assert len(result.packets) == 100
        assert result.recovery_count == 0


# =============================================================================
# Data Structure Tests
# =============================================================================


@pytest.mark.unit
class TestDataStructures:
    """Test data classes and enums."""

    def test_sync_match_dataclass(self) -> None:
        """Test SyncMatch dataclass instantiation."""
        match = SyncMatch(
            index=10,
            matched_value=0xAA55,
            hamming_distance=1,
            confidence=0.9375,
            pattern_length=16,
        )

        assert match.index == 10
        assert match.matched_value == 0xAA55
        assert match.hamming_distance == 1
        assert match.confidence == 0.9375
        assert match.pattern_length == 16

    def test_packet_parse_result_dataclass(self) -> None:
        """Test PacketParseResult dataclass instantiation."""
        result = PacketParseResult(
            packets=[b"\xaa\x55", b"\xff\x00"],
            valid=[True, False],
            errors=[None, "length_corruption"],
            error_positions=[10],
            recovery_count=1,
        )

        assert len(result.packets) == 2
        assert len(result.valid) == 2
        assert len(result.errors) == 2
        assert len(result.error_positions) == 1
        assert result.recovery_count == 1

    def test_recovery_strategy_enum(self) -> None:
        """Test RecoveryStrategy enum values."""
        assert RecoveryStrategy.NEXT_SYNC.value == "next_sync"
        assert RecoveryStrategy.SKIP_BYTES.value == "skip_bytes"
        assert RecoveryStrategy.HEURISTIC.value == "heuristic"

        # Test enum membership
        assert RecoveryStrategy.NEXT_SYNC in RecoveryStrategy
        assert RecoveryStrategy.SKIP_BYTES in RecoveryStrategy
        assert RecoveryStrategy.HEURISTIC in RecoveryStrategy


# =============================================================================
# Performance Characterization Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-001")
class TestPerformance:
    """Test performance characteristics (not strict benchmarks)."""

    def test_search_completes_quickly(self) -> None:
        """Test that search completes in reasonable time on large data."""
        # 10 KB of random data
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=10_000, dtype=np.uint8)

        # Inject a few sync patterns
        data[1000:1002] = [0xAA, 0x55]
        data[5000:5002] = [0xAA, 0x55]

        # Should complete without hanging
        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=2)

        # Should find at least the injected patterns
        assert len(matches) >= 2

    def test_packet_parsing_completes_quickly(self) -> None:
        """Test that packet parsing completes in reasonable time."""
        # Create 1000 packets
        packet = np.array([0xAA, 0x55, 0x00, 0x05, 0x01], dtype=np.uint8)
        data = np.tile(packet, 1000)

        # Should complete without hanging
        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
        )

        assert len(result.packets) == 1000


# =============================================================================
# Integration-Style Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.requirement("DAQ-001")
@pytest.mark.requirement("DAQ-002")
class TestExploratorySyncIntegration:
    """Test integration between fuzzy search and packet parsing."""

    def test_fuzzy_search_then_parse(self) -> None:
        """Test using fuzzy search to find sync, then parse packets."""
        # Data with sync patterns and packets
        data = np.array(
            [
                0x00,
                0x00,  # Noise
                0xAA,
                0x55,
                0x00,
                0x05,
                0x01,  # Good packet
                0xFF,
                0xFF,  # Noise
                0xAA,
                0x54,
                0x00,
                0x05,
                0x02,  # Packet with 1-bit error in sync
            ],
            dtype=np.uint8,
        )

        # First, find sync patterns (including fuzzy matches)
        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=2)

        # Should find both syncs (one exact, one fuzzy)
        assert len(matches) >= 2

        # Then parse packets
        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.NEXT_SYNC,
        )

        # Should parse at least one good packet
        assert len(result.packets) >= 1

    def test_real_world_corrupted_capture(self) -> None:
        """Test realistic scenario with mixed good/bad data."""
        # Simulate logic analyzer capture with corruption
        data = np.array(
            [
                # Good packet 1
                0xAA,
                0x55,
                0x00,
                0x06,
                0x12,
                0x34,
                # Corrupted region (bit errors)
                0xAA,
                0x54,
                0xFF,
                0xFF,
                0xFF,
                0xFF,
                # Good packet 2
                0xAA,
                0x55,
                0x00,
                0x05,
                0x56,
                # Another corruption
                0x00,
                0x00,
                0x00,
                # Good packet 3
                0xAA,
                0x55,
                0x00,
                0x07,
                0x78,
                0x9A,
                0xBC,
            ],
            dtype=np.uint8,
        )

        result = parse_variable_length_packets(
            data,
            sync_pattern=0xAA55,
            sync_bits=16,
            length_offset=2,
            length_size=2,
            recovery_strategy=RecoveryStrategy.NEXT_SYNC,
        )

        # Should recover and find multiple packets
        assert len(result.packets) >= 2
        assert result.recovery_count >= 1

    def test_no_sync_pattern_length_only_parsing(self) -> None:
        """Test parsing based on length field only, no sync pattern."""
        # Create TLV-style data without sync markers
        data = np.array(
            [
                0x00,
                0x04,
                0x01,
                0x02,  # Length 4
                0x00,
                0x05,
                0x03,
                0x04,
                0x05,  # Length 5
            ],
            dtype=np.uint8,
        )

        result = parse_variable_length_packets(
            data,
            sync_pattern=None,
            length_offset=0,
            length_size=2,
            recovery_strategy=RecoveryStrategy.SKIP_BYTES,
        )

        assert len(result.packets) == 2
