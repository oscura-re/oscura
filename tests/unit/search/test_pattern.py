"""Unit tests for pattern search functionality.

Tests SRCH-001: Pattern Search
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.search.pattern import find_pattern

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternBasic:
    """Test basic pattern finding functionality."""

    def test_find_simple_pattern_in_digital_trace(self) -> None:
        """Test finding a simple byte pattern in digital data."""
        # Create digital trace with known pattern
        digital = np.array([0xAA, 0x55, 0xAA, 0x00, 0xFF], dtype=np.uint8)

        # Search for 0xAA pattern
        matches = find_pattern(digital, 0xAA)

        assert len(matches) == 2
        assert matches[0][0] == 0  # First match at index 0
        assert matches[0][1][0] == 0xAA
        assert matches[1][0] == 2  # Second match at index 2
        assert matches[1][1][0] == 0xAA

    def test_find_pattern_in_analog_trace(self) -> None:
        """Test finding pattern in analog trace with threshold."""
        # Create analog trace representing: 1010 1010 (0xAA)
        analog = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        # Search for 0xAA pattern (10101010 in binary)
        matches = find_pattern(analog, 0xAA, threshold=0.5)

        assert len(matches) == 1
        assert matches[0][0] == 0
        assert matches[0][1][0] == 0xAA

    def test_find_multi_byte_pattern(self) -> None:
        """Test finding multi-byte pattern."""
        digital = np.array([0x12, 0x34, 0x56, 0x12, 0x34, 0x78], dtype=np.uint8)

        # Search for 2-byte pattern 0x1234
        pattern = np.array([0x12, 0x34], dtype=np.uint8)
        matches = find_pattern(digital, pattern)

        assert len(matches) == 2
        assert matches[0][0] == 0
        assert np.array_equal(matches[0][1], [0x12, 0x34])
        assert matches[1][0] == 3
        assert np.array_equal(matches[1][1], [0x12, 0x34])

    def test_find_pattern_no_matches(self) -> None:
        """Test when pattern is not found."""
        digital = np.array([0x00, 0x11, 0x22, 0x33], dtype=np.uint8)

        matches = find_pattern(digital, 0xFF)

        assert len(matches) == 0
        assert isinstance(matches, list)

    def test_find_pattern_at_end(self) -> None:
        """Test finding pattern at the end of trace."""
        digital = np.array([0x00, 0x11, 0x22, 0xAA], dtype=np.uint8)

        matches = find_pattern(digital, 0xAA)

        assert len(matches) == 1
        assert matches[0][0] == 3

    def test_find_pattern_entire_trace(self) -> None:
        """Test when pattern matches entire trace."""
        digital = np.array([0xAA, 0xBB], dtype=np.uint8)
        pattern = np.array([0xAA, 0xBB], dtype=np.uint8)

        matches = find_pattern(digital, pattern)

        assert len(matches) == 1
        assert matches[0][0] == 0
        assert np.array_equal(matches[0][1], [0xAA, 0xBB])


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternWithMask:
    """Test wildcard pattern matching with mask."""

    def test_find_pattern_with_full_mask(self) -> None:
        """Test pattern with full mask (all bits matter)."""
        digital = np.array([0xAA, 0x55, 0xAA], dtype=np.uint8)

        # Mask of 0xFF means all bits must match
        matches = find_pattern(digital, 0xAA, mask=0xFF)

        assert len(matches) == 2

    def test_find_pattern_with_partial_mask_upper_nibble(self) -> None:
        """Test wildcard search with upper nibble mask."""
        digital = np.array([0xA0, 0xA5, 0xAF, 0xB0, 0x5A], dtype=np.uint8)

        # Pattern 0xA0, mask 0xF0 matches 0xA0-0xAF
        matches = find_pattern(digital, 0xA0, mask=0xF0)

        assert len(matches) == 3  # 0xA0, 0xA5, 0xAF
        assert matches[0][1][0] == 0xA0
        assert matches[1][1][0] == 0xA5
        assert matches[2][1][0] == 0xAF

    def test_find_pattern_with_partial_mask_lower_nibble(self) -> None:
        """Test wildcard search with lower nibble mask."""
        digital = np.array([0x05, 0x15, 0x25, 0x50], dtype=np.uint8)

        # Pattern 0x05, mask 0x0F matches any byte ending in 0x5
        matches = find_pattern(digital, 0x05, mask=0x0F)

        assert len(matches) == 3  # 0x05, 0x15, 0x25
        assert matches[0][1][0] == 0x05
        assert matches[1][1][0] == 0x15
        assert matches[2][1][0] == 0x25

    def test_find_pattern_with_single_bit_mask(self) -> None:
        """Test wildcard search with single bit mask."""
        digital = np.array([0x80, 0x81, 0x00, 0xFF], dtype=np.uint8)

        # Pattern 0x80, mask 0x80 matches any byte with MSB set
        matches = find_pattern(digital, 0x80, mask=0x80)

        assert len(matches) == 3  # 0x80, 0x81, 0xFF

    def test_find_pattern_with_zero_mask(self) -> None:
        """Test wildcard search with zero mask (all don't care)."""
        digital = np.array([0x00, 0xFF, 0xAA], dtype=np.uint8)

        # Mask 0x00 means all positions are "don't care"
        matches = find_pattern(digital, 0x00, mask=0x00)

        # Should match every position
        assert len(matches) == 3

    def test_find_multi_byte_pattern_with_mask(self) -> None:
        """Test multi-byte pattern with per-byte mask."""
        digital = np.array([0x12, 0x34, 0x12, 0x00, 0x13, 0x35], dtype=np.uint8)

        pattern = np.array([0x12, 0x34], dtype=np.uint8)
        # First byte exact, second byte upper nibble only
        mask = np.array([0xFF, 0xF0], dtype=np.uint8)

        matches = find_pattern(digital, pattern, mask=mask)

        # Should match [0x12, 0x34] and [0x12, 0x00] (upper nibble 0x0)
        # but not [0x13, 0x35]
        assert len(matches) == 1  # Only [0x12, 0x34] matches both conditions

    def test_find_pattern_mask_array(self) -> None:
        """Test using mask as array instead of integer."""
        digital = np.array([0xA1, 0xA2, 0xB1, 0xA3], dtype=np.uint8)

        mask_arr = np.array([0xF0], dtype=np.uint8)  # Upper nibble only
        matches = find_pattern(digital, 0xA0, mask=mask_arr)

        assert len(matches) == 3  # 0xA1, 0xA2, 0xA3


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternMinSpacing:
    """Test minimum spacing between matches."""

    def test_overlapping_patterns_default_spacing(self) -> None:
        """Test that adjacent patterns are found with default spacing."""
        # Pattern that could overlap
        digital = np.array([0xAA, 0xAA, 0xAA], dtype=np.uint8)

        matches = find_pattern(digital, 0xAA)

        # Default min_spacing=1, all positions match
        assert len(matches) == 3

    def test_min_spacing_prevents_overlaps(self) -> None:
        """Test min_spacing parameter prevents overlapping matches."""
        digital = np.array([0xAA, 0xAA, 0xAA, 0xAA], dtype=np.uint8)

        matches = find_pattern(digital, 0xAA, min_spacing=2)

        # With spacing=2, should skip every other match
        assert len(matches) == 2
        assert matches[0][0] == 0
        assert matches[1][0] == 2

    def test_min_spacing_multi_byte_pattern(self) -> None:
        """Test min_spacing with multi-byte pattern."""
        digital = np.array([0x12, 0x34, 0x12, 0x34, 0x12, 0x34], dtype=np.uint8)
        pattern = np.array([0x12, 0x34], dtype=np.uint8)

        matches = find_pattern(digital, pattern, min_spacing=2)

        # Should find pattern at 0, then skip to 2 (spacing=2), then 4
        assert len(matches) == 3
        assert matches[0][0] == 0
        assert matches[1][0] == 2
        assert matches[2][0] == 4

    def test_min_spacing_larger_than_pattern(self) -> None:
        """Test min_spacing larger than pattern length."""
        digital = np.array([0xAA] * 10, dtype=np.uint8)

        matches = find_pattern(digital, 0xAA, min_spacing=5)

        # Should get matches at 0, 5
        assert len(matches) == 2
        assert matches[0][0] == 0
        assert matches[1][0] == 5


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_trace(self) -> None:
        """Test with empty trace."""
        empty = np.array([], dtype=np.uint8)

        matches = find_pattern(empty, 0xAA)

        assert len(matches) == 0

    def test_trace_shorter_than_pattern(self) -> None:
        """Test when trace is shorter than pattern."""
        digital = np.array([0x12], dtype=np.uint8)
        pattern = np.array([0x12, 0x34, 0x56], dtype=np.uint8)

        matches = find_pattern(digital, pattern)

        assert len(matches) == 0

    def test_empty_pattern_raises_error(self) -> None:
        """Test that empty pattern raises ValueError."""
        digital = np.array([0xAA], dtype=np.uint8)
        pattern = np.array([], dtype=np.uint8)

        with pytest.raises(ValueError, match="Pattern cannot be empty"):
            find_pattern(digital, pattern)

    def test_negative_pattern_raises_error(self) -> None:
        """Test that negative pattern value raises ValueError."""
        digital = np.array([0xAA], dtype=np.uint8)

        with pytest.raises(ValueError, match="Pattern must be non-negative"):
            find_pattern(digital, -1)

    def test_analog_without_threshold_raises_error(self) -> None:
        """Test that analog trace without threshold raises ValueError."""
        analog = np.array([1.0, 0.0, 1.0], dtype=np.float64)

        with pytest.raises(ValueError, match="Threshold required"):
            find_pattern(analog, 0xAA)

    def test_mask_pattern_length_mismatch(self) -> None:
        """Test error when mask and pattern have different lengths."""
        digital = np.array([0xAA], dtype=np.uint8)
        pattern = np.array([0x12, 0x34], dtype=np.uint8)
        mask = np.array([0xFF], dtype=np.uint8)  # Wrong length

        with pytest.raises(ValueError, match="Mask and pattern must have same length"):
            find_pattern(digital, pattern, mask=mask)

    def test_zero_pattern(self) -> None:
        """Test searching for zero pattern."""
        digital = np.array([0x00, 0x11, 0x00], dtype=np.uint8)

        matches = find_pattern(digital, 0)

        assert len(matches) == 2
        assert matches[0][0] == 0
        assert matches[1][0] == 2

    def test_large_pattern_value(self) -> None:
        """Test with large multi-byte pattern value."""
        digital = np.array([0x12, 0x34, 0x56, 0x78], dtype=np.uint8)

        # Pattern 0x12345678 is 4 bytes
        matches = find_pattern(digital, 0x12345678)

        assert len(matches) == 1
        assert matches[0][0] == 0
        assert np.array_equal(matches[0][1], [0x12, 0x34, 0x56, 0x78])


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternAnalogConversion:
    """Test analog to digital conversion for pattern matching."""

    def test_analog_threshold_conversion_simple(self) -> None:
        """Test simple threshold conversion of analog signal."""
        # 8 samples = 1 byte when packed
        analog = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)

        # Pattern 10101010 = 0xAA
        matches = find_pattern(analog, 0xAA, threshold=0.5)

        assert len(matches) == 1
        assert matches[0][1][0] == 0xAA

    def test_analog_threshold_at_boundary(self) -> None:
        """Test threshold conversion at exact boundary."""
        # Values exactly at threshold
        analog = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)

        # >= threshold is 1, so all 1s = 0xFF
        matches = find_pattern(analog, 0xFF, threshold=0.5)

        assert len(matches) == 1

    def test_analog_different_thresholds(self) -> None:
        """Test different threshold values affect conversion."""
        # Create 8 samples for one byte
        analog = np.array([2.0, 1.5, 1.0, 0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)

        # With threshold 1.0: values >= 1.0 are 1, < 1.0 are 0
        # [1, 1, 1, 0, 0, 0, 1, 1] = 11100011 = 0xE3
        matches = find_pattern(analog, 0xE3, threshold=1.0)

        assert len(matches) == 1
        assert matches[0][1][0] == 0xE3

    def test_analog_padding_to_byte_boundary(self) -> None:
        """Test that analog signal is padded to byte boundary."""
        # 10 samples should be padded to 16 (2 bytes)
        analog = np.ones(10, dtype=np.float64)

        # All 1s for 10 samples, padded with 6 zeros
        # [11111111, 11000000] = [0xFF, 0xC0]
        matches = find_pattern(analog, 0xFF, threshold=0.5)

        assert len(matches) == 1

    def test_analog_multi_byte_pattern(self) -> None:
        """Test finding multi-byte pattern in analog signal."""
        # 16 samples = 2 bytes
        # Pattern: 11110000 00001111 = 0xF00F
        analog = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)

        pattern = np.array([0xF0, 0x0F], dtype=np.uint8)
        matches = find_pattern(analog, pattern, threshold=0.5)

        assert len(matches) == 1
        assert np.array_equal(matches[0][1], [0xF0, 0x0F])


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternReturnFormat:
    """Test the format and content of returned matches."""

    def test_match_tuple_format(self) -> None:
        """Test that matches are returned as (index, data) tuples."""
        digital = np.array([0xAA, 0x55], dtype=np.uint8)

        matches = find_pattern(digital, 0xAA)

        assert len(matches) == 1
        assert isinstance(matches[0], tuple)
        assert len(matches[0]) == 2
        assert isinstance(matches[0][0], int)  # index
        assert isinstance(matches[0][1], np.ndarray)  # matched data

    def test_match_data_is_copy(self) -> None:
        """Test that returned match data is a copy, not a view."""
        digital = np.array([0xAA, 0x55], dtype=np.uint8)

        matches = find_pattern(digital, 0xAA)

        # Modify the match data
        matches[0][1][0] = 0xFF

        # Original should be unchanged
        assert digital[0] == 0xAA

    def test_match_preserves_actual_bytes(self) -> None:
        """Test that match returns actual bytes, not masked pattern."""
        digital = np.array([0xA5, 0xA7, 0xB5], dtype=np.uint8)

        # Search with mask - matches should return actual data
        # Pattern 0xA0, mask 0xF0 matches 0xA5 and 0xA7 (upper nibble = 0xA)
        # but not 0xB5 (upper nibble = 0xB)
        matches = find_pattern(digital, 0xA0, mask=0xF0)

        assert len(matches) == 2
        assert matches[0][1][0] == 0xA5  # Actual byte, not 0xA0
        assert matches[1][1][0] == 0xA7

    def test_multi_byte_match_length(self) -> None:
        """Test that multi-byte matches return correct length."""
        digital = np.array([0x12, 0x34, 0x56], dtype=np.uint8)
        pattern = np.array([0x12, 0x34], dtype=np.uint8)

        matches = find_pattern(digital, pattern)

        assert len(matches) == 1
        assert len(matches[0][1]) == 2
        assert matches[0][1].dtype == np.uint8


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternIntegerPatterns:
    """Test pattern specified as integers with various bit widths."""

    def test_8bit_pattern(self) -> None:
        """Test 8-bit integer pattern."""
        digital = np.array([0xFF, 0x00, 0xFF], dtype=np.uint8)

        matches = find_pattern(digital, 0xFF)

        assert len(matches) == 2

    def test_16bit_pattern(self) -> None:
        """Test 16-bit integer pattern (2 bytes)."""
        digital = np.array([0x12, 0x34, 0x56, 0x12, 0x34], dtype=np.uint8)

        # 0x1234 = [0x12, 0x34]
        matches = find_pattern(digital, 0x1234)

        assert len(matches) == 2
        assert matches[0][0] == 0
        assert matches[1][0] == 3

    def test_24bit_pattern(self) -> None:
        """Test 24-bit integer pattern (3 bytes)."""
        digital = np.array([0xAB, 0xCD, 0xEF, 0x00], dtype=np.uint8)

        # 0xABCDEF = [0xAB, 0xCD, 0xEF]
        matches = find_pattern(digital, 0xABCDEF)

        assert len(matches) == 1
        assert np.array_equal(matches[0][1], [0xAB, 0xCD, 0xEF])

    def test_32bit_pattern(self) -> None:
        """Test 32-bit integer pattern (4 bytes)."""
        digital = np.array([0x12, 0x34, 0x56, 0x78, 0x9A], dtype=np.uint8)

        # 0x12345678 = [0x12, 0x34, 0x56, 0x78]
        matches = find_pattern(digital, 0x12345678)

        assert len(matches) == 1
        assert np.array_equal(matches[0][1], [0x12, 0x34, 0x56, 0x78])

    def test_pattern_integer_mask_conversion(self) -> None:
        """Test that integer mask is converted to match pattern length."""
        digital = np.array([0x12, 0x34, 0x12, 0x00], dtype=np.uint8)

        # 16-bit pattern with 16-bit mask
        matches = find_pattern(digital, 0x1234, mask=0xFF00)

        # Mask 0xFF00 = [0xFF, 0x00], so second byte is don't care
        assert len(matches) == 2  # Both [0x12, 0x34] and [0x12, 0x00]


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternPerformance:
    """Test pattern search with realistic data sizes."""

    def test_large_trace_search(self) -> None:
        """Test pattern search in large trace."""
        # Create 10KB trace with few patterns
        digital = np.zeros(10000, dtype=np.uint8)
        digital[100] = 0xAA
        digital[5000] = 0xAA
        digital[9999] = 0xAA

        matches = find_pattern(digital, 0xAA)

        assert len(matches) == 3
        assert matches[0][0] == 100
        assert matches[1][0] == 5000
        assert matches[2][0] == 9999

    def test_many_matches(self) -> None:
        """Test when pattern appears many times."""
        # Repeating pattern
        digital = np.tile(np.array([0xAA, 0x55], dtype=np.uint8), 1000)

        matches = find_pattern(digital, 0xAA)

        # Should find 1000 matches (every other byte)
        assert len(matches) == 1000

    def test_worst_case_no_match(self) -> None:
        """Test performance when no matches exist."""
        # Large trace with no matches
        digital = np.zeros(10000, dtype=np.uint8)

        matches = find_pattern(digital, 0xFF)

        assert len(matches) == 0

    def test_complex_mask_large_trace(self) -> None:
        """Test masked search on large trace."""
        digital = np.random.randint(0, 256, size=5000, dtype=np.uint8)

        # Search for any byte with MSB set
        matches = find_pattern(digital, 0x80, mask=0x80)

        # Should find approximately half (those with MSB=1)
        # Allow wide tolerance due to randomness
        assert 2000 <= len(matches) <= 3000


@pytest.mark.unit
@pytest.mark.requirement("SRCH-001")
class TestFindPatternBitPatterns:
    """Test specific bit patterns and common protocols."""

    def test_alternating_bits_pattern(self) -> None:
        """Test finding alternating bit pattern (0x55 and 0xAA)."""
        digital = np.array([0x55, 0xAA, 0x55, 0xAA], dtype=np.uint8)

        matches_55 = find_pattern(digital, 0x55)
        matches_aa = find_pattern(digital, 0xAA)

        assert len(matches_55) == 2
        assert len(matches_aa) == 2

    def test_sync_pattern(self) -> None:
        """Test finding common sync patterns."""
        # UART-like frame with sync byte 0x7E
        digital = np.array([0x7E, 0x01, 0x02, 0x7E, 0x03, 0x04], dtype=np.uint8)

        matches = find_pattern(digital, 0x7E)

        assert len(matches) == 2
        assert matches[0][0] == 0
        assert matches[1][0] == 3

    def test_magic_number_pattern(self) -> None:
        """Test finding magic number / file signature."""
        # PNG file signature
        digital = np.array([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A], dtype=np.uint8)
        pattern = np.array([0x89, 0x50, 0x4E, 0x47], dtype=np.uint8)

        matches = find_pattern(digital, pattern)

        assert len(matches) == 1
        assert matches[0][0] == 0

    def test_all_zeros_pattern(self) -> None:
        """Test finding all zeros pattern."""
        digital = np.array([0x00, 0x00, 0x00, 0xFF], dtype=np.uint8)
        pattern = np.array([0x00, 0x00], dtype=np.uint8)

        matches = find_pattern(digital, pattern)

        # Should find at index 0 and 1 (overlapping)
        assert len(matches) == 2

    def test_all_ones_pattern(self) -> None:
        """Test finding all ones pattern."""
        digital = np.array([0xFF, 0xFF, 0x00], dtype=np.uint8)

        matches = find_pattern(digital, 0xFF)

        assert len(matches) == 2
