"""Comprehensive tests for fuzzy synchronization module.

Tests requirements:
  - SyncMatch dataclass
  - hamming_distance calculation
  - fuzzy_sync_search with various patterns
  - Error tolerance settings
  - Confidence scoring
  - Pattern matching with bit errors
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.jupyter.exploratory.sync import (
    SyncMatch,
    fuzzy_sync_search,
    hamming_distance,
)

pytestmark = pytest.mark.unit


class TestHammingDistance:
    """Test Hamming distance calculation."""

    def test_identical_values(self) -> None:
        """Test Hamming distance between identical values."""
        assert hamming_distance(0b10101010, 0b10101010, 8) == 0
        assert hamming_distance(0xAA55, 0xAA55, 16) == 0
        assert hamming_distance(0xDEADBEEF, 0xDEADBEEF, 32) == 0

    def test_single_bit_difference(self) -> None:
        """Test Hamming distance with single bit flip."""
        assert hamming_distance(0b10101010, 0b10101011, 8) == 1
        assert hamming_distance(0b10101010, 0b10101110, 8) == 1
        assert hamming_distance(0xAA55, 0xAA54, 16) == 1

    def test_multiple_bit_differences(self) -> None:
        """Test Hamming distance with multiple bit flips."""
        assert hamming_distance(0b10101010, 0b01010101, 8) == 8
        assert hamming_distance(0b11110000, 0b00001111, 8) == 8
        assert hamming_distance(0xAAAA, 0x5555, 16) == 16

    def test_8_bit_patterns(self) -> None:
        """Test Hamming distance for 8-bit patterns."""
        assert hamming_distance(0xFF, 0x00, 8) == 8
        assert hamming_distance(0xF0, 0x0F, 8) == 8
        assert hamming_distance(0xAA, 0xAB, 8) == 1

    def test_16_bit_patterns(self) -> None:
        """Test Hamming distance for 16-bit patterns."""
        assert hamming_distance(0xAA55, 0xAA54, 16) == 1
        assert hamming_distance(0xFFFF, 0x0000, 16) == 16
        assert hamming_distance(0x1234, 0x1235, 16) == 1

    def test_32_bit_patterns(self) -> None:
        """Test Hamming distance for 32-bit patterns."""
        assert hamming_distance(0xDEADBEEF, 0xDEADBEEE, 32) == 1
        assert hamming_distance(0xFFFFFFFF, 0x00000000, 32) == 32

    def test_64_bit_patterns(self) -> None:
        """Test Hamming distance for 64-bit patterns."""
        assert hamming_distance(0xDEADBEEFCAFEBABE, 0xDEADBEEFCAFEBABF, 64) == 1

    def test_masking_to_pattern_length(self) -> None:
        """Test that Hamming distance masks to pattern length."""
        # Only compare lower 8 bits
        assert hamming_distance(0x1FF, 0x100, 8) == 8


class TestSyncMatchDataclass:
    """Test SyncMatch dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test SyncMatch initialization."""
        match = SyncMatch(
            index=100,
            matched_value=0xAA54,
            hamming_distance=1,
            confidence=0.9375,
            pattern_length=16,
        )

        assert match.index == 100
        assert match.matched_value == 0xAA54
        assert match.hamming_distance == 1
        assert match.confidence == 0.9375
        assert match.pattern_length == 16


class TestFuzzySyncSearchBasic:
    """Test fuzzy_sync_search basic functionality."""

    def test_exact_match_8bit(self) -> None:
        """Test exact match for 8-bit pattern."""
        data = np.array([0x00, 0xAA, 0x55, 0xFF], dtype=np.uint8)
        pattern = 0xAA

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=0)

        assert len(matches) == 1
        assert matches[0].index == 1
        assert matches[0].matched_value == 0xAA
        assert matches[0].hamming_distance == 0
        assert matches[0].confidence == 1.0

    def test_exact_match_16bit(self) -> None:
        """Test exact match for 16-bit pattern."""
        data = np.array([0xAA, 0x55, 0xF0, 0xF0], dtype=np.uint8)
        pattern = 0xAA55

        matches = fuzzy_sync_search(data, pattern, pattern_bits=16, max_errors=0)

        assert len(matches) == 1
        assert matches[0].index == 0
        assert matches[0].matched_value == 0xAA55
        assert matches[0].hamming_distance == 0
        assert matches[0].confidence == 1.0

    def test_exact_match_32bit(self) -> None:
        """Test exact match for 32-bit pattern."""
        data = np.array([0xAA, 0x55, 0xF0, 0xF0, 0xFF], dtype=np.uint8)
        pattern = 0xAA55F0F0

        matches = fuzzy_sync_search(data, pattern, pattern_bits=32, max_errors=0)

        assert len(matches) == 1
        assert matches[0].index == 0
        assert matches[0].confidence == 1.0

    def test_no_match(self) -> None:
        """Test when pattern not found."""
        data = np.array([0x00, 0x11, 0x22, 0x33], dtype=np.uint8)
        pattern = 0xAA55

        matches = fuzzy_sync_search(data, pattern, pattern_bits=16, max_errors=0)

        assert len(matches) == 0

    def test_multiple_matches(self) -> None:
        """Test finding multiple occurrences."""
        data = np.array([0xAA, 0x55, 0x00, 0xAA, 0x55], dtype=np.uint8)
        pattern = 0xAA55

        matches = fuzzy_sync_search(data, pattern, pattern_bits=16, max_errors=0)

        assert len(matches) == 2
        assert matches[0].index == 0
        assert matches[1].index == 3


class TestFuzzySyncSearchWithErrors:
    """Test fuzzy_sync_search with bit errors."""

    def test_single_bit_error_8bit(self) -> None:
        """Test finding pattern with single bit error in 8-bit pattern."""
        # Pattern 0xAA = 10101010, data has 0xAB = 10101011 (1 bit diff)
        data = np.array([0xAB, 0x00], dtype=np.uint8)
        pattern = 0xAA

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=2)

        assert len(matches) == 1
        assert matches[0].index == 0
        assert matches[0].matched_value == 0xAB
        assert matches[0].hamming_distance == 1
        assert matches[0].confidence > 0.8  # 1 - 1/8 = 0.875

    def test_single_bit_error_16bit(self) -> None:
        """Test finding pattern with single bit error in 16-bit pattern."""
        # 0xAA55 with 1 bit flipped to 0xAA54
        data = np.array([0xAA, 0x54, 0x00], dtype=np.uint8)
        pattern = 0xAA55

        matches = fuzzy_sync_search(data, pattern, pattern_bits=16, max_errors=2)

        assert len(matches) == 1
        assert matches[0].index == 0
        assert matches[0].hamming_distance == 1
        assert matches[0].confidence > 0.9  # 1 - 1/16 = 0.9375

    def test_two_bit_errors(self) -> None:
        """Test finding pattern with two bit errors."""
        # 0xAA = 10101010, 0xAE = 10101110 (2 bits different)
        data = np.array([0xAE, 0x00], dtype=np.uint8)
        pattern = 0xAA

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=2)

        assert len(matches) == 1
        assert matches[0].hamming_distance == 2
        assert 0.7 < matches[0].confidence < 0.8  # 1 - 2/8 = 0.75

    def test_too_many_errors_rejected(self) -> None:
        """Test that patterns with too many errors are rejected."""
        # 0xAA = 10101010, 0x55 = 01010101 (8 bits different)
        data = np.array([0x55, 0x00], dtype=np.uint8)
        pattern = 0xAA

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=2)

        assert len(matches) == 0

    def test_max_errors_zero(self) -> None:
        """Test that max_errors=0 requires exact match."""
        data = np.array([0xAB, 0x00], dtype=np.uint8)
        pattern = 0xAA

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=0)

        assert len(matches) == 0

    def test_max_errors_boundary(self) -> None:
        """Test pattern at exact max_errors boundary."""
        # 2 bit errors
        data = np.array([0xAE, 0x00], dtype=np.uint8)
        pattern = 0xAA

        # Should match with max_errors=2
        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=2)
        assert len(matches) == 1

        # Should not match with max_errors=1
        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=1)
        assert len(matches) == 0


class TestConfidenceThreshold:
    """Test confidence threshold filtering."""

    def test_low_confidence_rejected(self) -> None:
        """Test that low confidence matches are rejected."""
        # 3 bit errors in 8-bit pattern = 62.5% confidence
        data = np.array([0xA8, 0x00], dtype=np.uint8)  # 10101000 vs 10101010
        pattern = 0xAA

        # With min_confidence=0.85, should reject
        matches = fuzzy_sync_search(
            data, pattern, pattern_bits=8, max_errors=8, min_confidence=0.85
        )
        assert len(matches) == 0

        # With min_confidence=0.60, should accept
        matches = fuzzy_sync_search(
            data, pattern, pattern_bits=8, max_errors=8, min_confidence=0.60
        )
        assert len(matches) == 1

    def test_high_confidence_accepted(self) -> None:
        """Test that high confidence matches are accepted."""
        # 1 bit error in 16-bit pattern = 93.75% confidence
        data = np.array([0xAA, 0x54], dtype=np.uint8)
        pattern = 0xAA55

        matches = fuzzy_sync_search(
            data, pattern, pattern_bits=16, max_errors=2, min_confidence=0.90
        )

        assert len(matches) == 1
        assert matches[0].confidence > 0.90


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data(self) -> None:
        """Test with empty data array."""
        data = np.array([], dtype=np.uint8)
        pattern = 0xAA

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8)

        assert len(matches) == 0

    def test_data_shorter_than_pattern(self) -> None:
        """Test with data shorter than pattern."""
        data = np.array([0xAA], dtype=np.uint8)
        pattern = 0xAA55

        matches = fuzzy_sync_search(data, pattern, pattern_bits=16)

        assert len(matches) == 0

    def test_data_exactly_pattern_size(self) -> None:
        """Test with data exactly matching pattern size."""
        data = np.array([0xAA, 0x55], dtype=np.uint8)
        pattern = 0xAA55

        matches = fuzzy_sync_search(data, pattern, pattern_bits=16)

        assert len(matches) == 1
        assert matches[0].index == 0

    def test_invalid_pattern_bits(self) -> None:
        """Test validation of pattern_bits parameter."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="pattern_bits"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=12)

    def test_invalid_max_errors_negative(self) -> None:
        """Test validation of negative max_errors."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="max_errors"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=-1)

    def test_invalid_max_errors_too_large(self) -> None:
        """Test validation of max_errors > 8."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="max_errors"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, max_errors=9)

    def test_invalid_min_confidence_negative(self) -> None:
        """Test validation of negative min_confidence."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="min_confidence"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, min_confidence=-0.1)

    def test_invalid_min_confidence_too_large(self) -> None:
        """Test validation of min_confidence > 1.0."""
        data = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="min_confidence"):
            fuzzy_sync_search(data, 0xAA, pattern_bits=8, min_confidence=1.5)


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    def test_corrupted_can_sync(self) -> None:
        """Test finding corrupted CAN bus sync pattern."""
        # CAN SOF is dominant bit (0x00 extended)
        # Simulate corrupted capture with 1-2 bit errors
        data = np.array([0xFF, 0x00, 0x01, 0x80, 0x00], dtype=np.uint8)
        pattern = 0x00

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=2)

        # Should find sync at indices 1 and 4 (exact), possibly 2 (1 error)
        assert len(matches) >= 2

    def test_noisy_uart_start_bit(self) -> None:
        """Test finding UART start bit in noisy data."""
        # UART typically has 0x00 start followed by data
        # Simulate noisy capture
        data = np.array([0xFF, 0x01, 0x48, 0xFF], dtype=np.uint8)  # 'H' with noise
        pattern = 0x00

        matches = fuzzy_sync_search(data, pattern, pattern_bits=8, max_errors=3)

        # Should detect the mostly-zero byte
        assert len(matches) >= 1

    def test_64bit_sync_word(self) -> None:
        """Test finding 64-bit sync word with errors."""
        # Large sync word with 2 bit errors
        sync = 0xDEADBEEFCAFEBABE
        corrupted_bytes = [
            0xDE,
            0xAD,
            0xBE,
            0xEE,
            0xCA,
            0xFE,
            0xBA,
            0xBF,
        ]  # 2 bits flipped
        data = np.array([0x00] + corrupted_bytes + [0x00], dtype=np.uint8)

        matches = fuzzy_sync_search(data, sync, pattern_bits=64, max_errors=3)

        assert len(matches) == 1
        assert matches[0].index == 1
        assert matches[0].hamming_distance <= 2


class TestPerformance:
    """Test performance characteristics."""

    def test_large_data_array(self) -> None:
        """Test search on large data array."""
        # 1 MB of random data with embedded patterns
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=1_000_000, dtype=np.uint8)

        # Insert known patterns
        data[10000:10002] = [0xAA, 0x55]
        data[50000:50002] = [0xAA, 0x55]

        matches = fuzzy_sync_search(data, 0xAA55, pattern_bits=16, max_errors=0)

        # Should find at least the two we inserted
        assert len(matches) >= 2
        indices = [m.index for m in matches]
        assert 10000 in indices
        assert 50000 in indices

    def test_overlapping_matches(self) -> None:
        """Test that overlapping matches are all found."""
        # Pattern that can overlap: 0xAAAA
        data = np.array([0xAA, 0xAA, 0xAA], dtype=np.uint8)
        pattern = 0xAAAA

        matches = fuzzy_sync_search(data, pattern, pattern_bits=16, max_errors=0)

        # Should find matches at indices 0 and 1 (overlapping)
        assert len(matches) == 2
