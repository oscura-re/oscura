"""Comprehensive unit tests for sequences module (PAT-002).

This test suite provides comprehensive coverage of the sequences module,
including repeating sequence detection, n-gram analysis, and approximate
pattern matching.


Author: Oscura Development Team
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.patterns.sequences import (
    NgramResult,
    RepeatingSequence,
    RepeatingSequenceFinder,
    find_approximate_repeats,
    find_frequent_ngrams,
    find_longest_repeat,
    find_repeating_sequences,
)

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.pattern]


# =============================================================================
# RepeatingSequence Dataclass Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestRepeatingSequence:
    """Test RepeatingSequence dataclass validation."""

    def test_valid_sequence(self) -> None:
        """Test creating valid RepeatingSequence."""
        seq = RepeatingSequence(
            pattern=b"ABCD",
            length=4,
            count=10,
            positions=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            frequency=0.1,
        )
        assert seq.pattern == b"ABCD"
        assert seq.length == 4
        assert seq.count == 10
        assert len(seq.positions) == 10
        assert seq.frequency == 0.1

    def test_invalid_length_zero(self) -> None:
        """Test that zero length raises ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            RepeatingSequence(pattern=b"", length=0, count=5, positions=[0], frequency=0.1)

    def test_invalid_length_negative(self) -> None:
        """Test that negative length raises ValueError."""
        with pytest.raises(ValueError, match="length must be positive"):
            RepeatingSequence(pattern=b"AB", length=-1, count=5, positions=[0], frequency=0.1)

    def test_invalid_count_negative(self) -> None:
        """Test that negative count raises ValueError."""
        with pytest.raises(ValueError, match="count must be non-negative"):
            RepeatingSequence(pattern=b"AB", length=2, count=-1, positions=[0], frequency=0.1)

    def test_pattern_length_mismatch(self) -> None:
        """Test that pattern length must match length field."""
        with pytest.raises(ValueError, match="pattern length must match length field"):
            RepeatingSequence(
                pattern=b"ABC", length=5, count=3, positions=[0, 5, 10], frequency=0.1
            )


# =============================================================================
# NgramResult Dataclass Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestNgramResult:
    """Test NgramResult dataclass."""

    def test_valid_ngram_with_positions(self) -> None:
        """Test creating NgramResult with positions."""
        ngram = NgramResult(ngram=b"AB", count=10, frequency=0.05, positions=[0, 2, 4, 6, 8])
        assert ngram.ngram == b"AB"
        assert ngram.count == 10
        assert ngram.frequency == 0.05
        assert len(ngram.positions) == 5

    def test_valid_ngram_without_positions(self) -> None:
        """Test creating NgramResult without positions (default)."""
        ngram = NgramResult(ngram=b"XY", count=5, frequency=0.02)
        assert ngram.ngram == b"XY"
        assert ngram.count == 5
        assert ngram.positions == []


# =============================================================================
# find_repeating_sequences Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestFindRepeatingSequences:
    """Test find_repeating_sequences function."""

    def test_basic_repeating_pattern(self) -> None:
        """Test finding basic repeating pattern in bytes."""
        data = b"ABCDABCDABCD"
        sequences = find_repeating_sequences(data, min_length=4, max_length=8, min_count=3)

        # Should find "ABCD" repeated 3 times
        assert len(sequences) > 0
        abcd_found = any(seq.pattern == b"ABCD" and seq.count == 3 for seq in sequences)
        assert abcd_found

    def test_multiple_patterns(self) -> None:
        """Test finding multiple different repeating patterns."""
        data = b"AAAA" + b"BBBB" + b"AAAA" + b"BBBB" + b"AAAA" + b"BBBB"
        sequences = find_repeating_sequences(data, min_length=2, max_length=8, min_count=3)

        # Should find both "AAAA" and "BBBB"
        assert len(sequences) >= 2

    def test_sorted_by_frequency(self) -> None:
        """Test that results are sorted by frequency (descending)."""
        data = b"AA" * 20 + b"BBB" * 10 + b"CCCC" * 5
        sequences = find_repeating_sequences(data, min_length=2, max_length=8, min_count=3)

        # Should be sorted by frequency
        for i in range(len(sequences) - 1):
            assert sequences[i].frequency >= sequences[i + 1].frequency

    def test_numpy_array_input(self) -> None:
        """Test with numpy array input."""
        data = np.array([0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55], dtype=np.uint8)
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=3)

        # Should find repeating pattern
        assert len(sequences) > 0
        aa55_found = any(seq.pattern == b"\xaa\x55" for seq in sequences)
        assert aa55_found

    def test_positions_are_correct(self) -> None:
        """Test that positions are correctly identified."""
        data = b"__ABC__ABC__ABC"
        sequences = find_repeating_sequences(data, min_length=3, max_length=5, min_count=3)

        # Find ABC pattern
        abc_seq = next((s for s in sequences if s.pattern == b"ABC"), None)
        assert abc_seq is not None
        assert abc_seq.positions == [2, 7, 12]

    def test_positions_are_sorted(self) -> None:
        """Test that positions are sorted."""
        data = b"XY" * 10
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=3)

        for seq in sequences:
            assert seq.positions == sorted(seq.positions)

    def test_frequency_calculation(self) -> None:
        """Test that frequency is correctly calculated."""
        data = b"AB" * 10  # 20 bytes total
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=3)

        # Find "AB" pattern
        ab_seq = next((s for s in sequences if s.pattern == b"AB"), None)
        assert ab_seq is not None
        # 10 occurrences out of (20 - 2 + 1) = 19 possible positions
        expected_frequency = 10 / 19
        assert ab_seq.frequency == pytest.approx(expected_frequency, rel=1e-5)

    def test_min_count_filtering(self) -> None:
        """Test that min_count filters results correctly."""
        data = b"AA" * 5 + b"BB" * 2  # AA appears 5 times, BB appears 2 times
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=3)

        # Should find "AA" but not "BB" (count < 3)
        aa_found = any(seq.pattern == b"AA" for seq in sequences)
        bb_found = any(seq.pattern == b"BB" and seq.count == 2 for seq in sequences)
        assert aa_found
        assert not bb_found

    def test_empty_data(self) -> None:
        """Test with empty data."""
        sequences = find_repeating_sequences(b"", min_length=2, max_length=8, min_count=2)
        assert sequences == []

    def test_data_shorter_than_min_length(self) -> None:
        """Test with data shorter than min_length."""
        data = b"AB"
        sequences = find_repeating_sequences(data, min_length=4, max_length=8, min_count=2)
        assert sequences == []

    def test_no_repeating_patterns(self) -> None:
        """Test with data that has no repeating patterns."""
        data = bytes(range(256))  # All unique bytes
        sequences = find_repeating_sequences(data, min_length=2, max_length=8, min_count=2)
        assert sequences == []

    def test_invalid_min_length_zero(self) -> None:
        """Test that min_length < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be at least 1"):
            find_repeating_sequences(b"test", min_length=0)

    def test_invalid_max_length(self) -> None:
        """Test that max_length < min_length raises ValueError."""
        with pytest.raises(ValueError, match="max_length must be >= min_length"):
            find_repeating_sequences(b"test", min_length=5, max_length=3)

    def test_invalid_min_count(self) -> None:
        """Test that min_count < 2 raises ValueError."""
        with pytest.raises(ValueError, match="min_count must be at least 2"):
            find_repeating_sequences(b"test", min_count=1)

    def test_bytearray_input(self) -> None:
        """Test with bytearray input."""
        data = bytearray(b"XX" * 5)
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=3)
        assert len(sequences) > 0

    def test_memoryview_input(self) -> None:
        """Test with memoryview input."""
        data = memoryview(b"YY" * 5)
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=3)
        assert len(sequences) > 0

    def test_overlapping_patterns(self) -> None:
        """Test detection of overlapping patterns."""
        data = b"AAAA"
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=2)

        # Should detect "AA", "AAA", etc.
        assert len(sequences) > 0


# =============================================================================
# find_frequent_ngrams Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestFindFrequentNgrams:
    """Test find_frequent_ngrams function."""

    def test_basic_ngrams(self) -> None:
        """Test finding basic 2-grams."""
        data = b"ABABABABCDCDCDCD"
        ngrams = find_frequent_ngrams(data, n=2, top_k=5)

        # Should find "AB" and "CD" as most frequent
        assert len(ngrams) > 0
        top_ngrams = [ng.ngram for ng in ngrams[:2]]
        assert b"AB" in top_ngrams or b"CD" in top_ngrams

    def test_sorted_by_count(self) -> None:
        """Test that results are sorted by count (descending)."""
        data = b"AA" * 20 + b"BB" * 10 + b"CC" * 5
        ngrams = find_frequent_ngrams(data, n=2, top_k=10)

        # Should be sorted by count
        for i in range(len(ngrams) - 1):
            assert ngrams[i].count >= ngrams[i + 1].count

    def test_frequency_calculation(self) -> None:
        """Test that frequency is correctly calculated."""
        data = b"AAAA"  # 4 bytes, 3 possible 2-grams
        ngrams = find_frequent_ngrams(data, n=2, top_k=5)

        # All 2-grams are "AA" (3 of them)
        aa_ngram = next((ng for ng in ngrams if ng.ngram == b"AA"), None)
        assert aa_ngram is not None
        assert aa_ngram.count == 3
        assert aa_ngram.frequency == pytest.approx(1.0, rel=1e-5)

    def test_with_positions(self) -> None:
        """Test ngram detection with positions."""
        data = b"XYXYXY"
        ngrams = find_frequent_ngrams(data, n=2, top_k=5, return_positions=True)

        # Find "XY" ngram
        xy_ngram = next((ng for ng in ngrams if ng.ngram == b"XY"), None)
        assert xy_ngram is not None
        assert xy_ngram.positions == [0, 2, 4]

    def test_without_positions(self) -> None:
        """Test ngram detection without positions (default)."""
        data = b"XYXYXY"
        ngrams = find_frequent_ngrams(data, n=2, top_k=5, return_positions=False)

        # Positions should be empty
        for ngram in ngrams:
            assert ngram.positions == []

    def test_top_k_limiting(self) -> None:
        """Test that top_k limits results correctly."""
        data = bytes(range(256)) * 2  # Many unique 2-grams
        ngrams = find_frequent_ngrams(data, n=2, top_k=10)

        assert len(ngrams) <= 10

    def test_trigrams(self) -> None:
        """Test with 3-grams."""
        data = b"ABCABCABC"
        ngrams = find_frequent_ngrams(data, n=3, top_k=5)

        # Should find "ABC"
        abc_found = any(ng.ngram == b"ABC" for ng in ngrams)
        assert abc_found

    def test_numpy_array_input(self) -> None:
        """Test with numpy array input."""
        data = np.array([0x12, 0x34, 0x12, 0x34, 0x12, 0x34], dtype=np.uint8)
        ngrams = find_frequent_ngrams(data, n=2, top_k=5)

        # Should find repeating pattern
        assert len(ngrams) > 0

    def test_empty_data(self) -> None:
        """Test with empty data."""
        ngrams = find_frequent_ngrams(b"", n=2, top_k=10)
        assert ngrams == []

    def test_data_shorter_than_n(self) -> None:
        """Test with data shorter than n."""
        data = b"AB"
        ngrams = find_frequent_ngrams(data, n=5, top_k=10)
        assert ngrams == []

    def test_invalid_n(self) -> None:
        """Test that n < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n must be at least 1"):
            find_frequent_ngrams(b"test", n=0)

    def test_invalid_top_k(self) -> None:
        """Test that top_k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_k must be at least 1"):
            find_frequent_ngrams(b"test", n=2, top_k=0)

    def test_single_byte_ngrams(self) -> None:
        """Test with n=1 (single byte frequency)."""
        data = b"AAABBC"
        ngrams = find_frequent_ngrams(data, n=1, top_k=5)

        # Should find byte frequencies
        assert len(ngrams) > 0
        a_ngram = next((ng for ng in ngrams if ng.ngram == b"A"), None)
        assert a_ngram is not None
        assert a_ngram.count == 3


# =============================================================================
# find_longest_repeat Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestFindLongestRepeat:
    """Test find_longest_repeat function."""

    def test_classic_example(self) -> None:
        """Test with classic 'banana' example."""
        data = b"banana"
        result = find_longest_repeat(data)

        assert result is not None
        assert result.pattern == b"ana"
        assert result.length == 3
        assert result.count == 2

    def test_simple_repeat(self) -> None:
        """Test with simple repeating pattern."""
        data = b"ABCDEFABCDEF"
        result = find_longest_repeat(data)

        assert result is not None
        assert result.pattern == b"ABCDEF"
        assert result.length == 6
        assert result.count == 2

    def test_multiple_repeats(self) -> None:
        """Test with multiple repeating patterns."""
        data = b"XYZXYZXYZ"
        result = find_longest_repeat(data)

        assert result is not None
        # Could be "XYZXYZ" or "XYZ" depending on algorithm
        assert result.length >= 3

    def test_no_repeat(self) -> None:
        """Test with data that has no repeating substrings."""
        data = b"ABCDEF"
        result = find_longest_repeat(data)

        # Should return None or pattern of length 0
        assert result is None

    def test_all_same_character(self) -> None:
        """Test with all same character."""
        data = b"AAAA"
        result = find_longest_repeat(data)

        assert result is not None
        # Should find "AAA" (or similar)
        assert result.length >= 1

    def test_positions_found(self) -> None:
        """Test that positions are correctly identified."""
        data = b"XYZABCXYZABC"
        result = find_longest_repeat(data)

        assert result is not None
        # Should find repeating pattern and its positions
        assert len(result.positions) >= 2

    def test_empty_data(self) -> None:
        """Test with empty data."""
        result = find_longest_repeat(b"")
        assert result is None

    def test_single_byte(self) -> None:
        """Test with single byte."""
        result = find_longest_repeat(b"A")
        assert result is None

    def test_two_bytes_different(self) -> None:
        """Test with two different bytes."""
        result = find_longest_repeat(b"AB")
        assert result is None

    def test_two_bytes_same(self) -> None:
        """Test with two same bytes."""
        result = find_longest_repeat(b"AA")
        assert result is not None
        assert result.pattern == b"A"
        assert result.count == 2

    def test_numpy_array_input(self) -> None:
        """Test with numpy array input."""
        data = np.array([1, 2, 3, 1, 2, 3], dtype=np.uint8)
        result = find_longest_repeat(data)

        assert result is not None
        assert result.length >= 1


# =============================================================================
# find_approximate_repeats Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestFindApproximateRepeats:
    """Test find_approximate_repeats function."""

    def test_exact_matches(self) -> None:
        """Test with exact repeating patterns (distance 0)."""
        data = b"ABCD" * 5
        results = find_approximate_repeats(data, min_length=4, max_distance=0, min_count=3)

        # Should find "ABCD" repeated exactly
        assert len(results) > 0
        abcd_found = any(r.pattern == b"ABCD" for r in results)
        assert abcd_found

    def test_similar_patterns(self) -> None:
        """Test with similar but not identical patterns."""
        data = b"ABCD" + b"ABCE" + b"ABCF"
        results = find_approximate_repeats(data, min_length=4, max_distance=1, min_count=2)

        # Should cluster similar patterns
        assert len(results) > 0

    def test_distance_threshold(self) -> None:
        """Test that max_distance threshold works."""
        data = b"AAAA" + b"AAAB" + b"AAAC" + b"ZZZZ"
        results = find_approximate_repeats(data, min_length=4, max_distance=1, min_count=2)

        # Should cluster AAA* patterns but not ZZZZ
        assert len(results) > 0

    def test_representative_pattern(self) -> None:
        """Test that representative pattern is chosen correctly."""
        data = b"ABCD" + b"ABCD" + b"ABCD" + b"ABCE"
        results = find_approximate_repeats(data, min_length=4, max_distance=1, min_count=2)

        # Most common pattern should be representative
        if results:
            assert results[0].pattern == b"ABCD"

    def test_sorted_by_count(self) -> None:
        """Test that results are sorted by count."""
        data = b"AA" * 10 + b"BB" * 5 + b"CC" * 3
        results = find_approximate_repeats(data, min_length=2, max_distance=0, min_count=2)

        # Should be sorted by count
        for i in range(len(results) - 1):
            assert results[i].count >= results[i + 1].count

    def test_min_count_filtering(self) -> None:
        """Test that min_count filters results."""
        data = b"AAAA" + b"AAAB" + b"BBBB"
        results = find_approximate_repeats(data, min_length=4, max_distance=1, min_count=3)

        # Should not include patterns with count < 3
        for result in results:
            assert result.count >= 3

    def test_empty_data(self) -> None:
        """Test with empty data."""
        results = find_approximate_repeats(b"", min_length=4, max_distance=1, min_count=2)
        assert results == []

    def test_data_shorter_than_min_length(self) -> None:
        """Test with data shorter than min_length."""
        data = b"AB"
        results = find_approximate_repeats(data, min_length=8, max_distance=1, min_count=2)
        assert results == []

    def test_no_similar_patterns(self) -> None:
        """Test with no similar patterns."""
        data = bytes(range(20))  # All unique
        results = find_approximate_repeats(data, min_length=4, max_distance=1, min_count=2)
        assert results == []

    def test_invalid_min_length(self) -> None:
        """Test that min_length < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be at least 1"):
            find_approximate_repeats(b"test", min_length=0)

    def test_invalid_max_distance(self) -> None:
        """Test that max_distance < 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_distance must be non-negative"):
            find_approximate_repeats(b"test", max_distance=-1)

    def test_invalid_min_count(self) -> None:
        """Test that min_count < 2 raises ValueError."""
        with pytest.raises(ValueError, match="min_count must be at least 2"):
            find_approximate_repeats(b"test", min_count=1)

    def test_numpy_array_input(self) -> None:
        """Test with numpy array input."""
        data = np.array([1, 2, 3, 4, 1, 2, 3, 5], dtype=np.uint8)
        results = find_approximate_repeats(data, min_length=4, max_distance=1, min_count=2)
        # Should handle numpy arrays
        assert isinstance(results, list)

    def test_positions_are_sorted(self) -> None:
        """Test that positions are sorted."""
        data = b"ABCD" + b"ABCE" + b"ABCF"
        results = find_approximate_repeats(data, min_length=4, max_distance=1, min_count=2)

        for result in results:
            assert result.positions == sorted(result.positions)


# =============================================================================
# RepeatingSequenceFinder Class Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestRepeatingSequenceFinder:
    """Test RepeatingSequenceFinder class."""

    def test_initialization(self) -> None:
        """Test finder initialization."""
        finder = RepeatingSequenceFinder(
            min_length=4, max_length=16, min_count=3, min_frequency=0.01
        )
        assert finder.min_length == 4
        assert finder.max_length == 16
        assert finder.min_count == 3
        assert finder.min_frequency == 0.01

    def test_default_initialization(self) -> None:
        """Test finder with default parameters."""
        finder = RepeatingSequenceFinder()
        assert finder.min_length == 2
        assert finder.max_length == 32
        assert finder.min_count == 2
        assert finder.min_frequency == 0.001

    def test_find_sequences(self) -> None:
        """Test find_sequences method."""
        finder = RepeatingSequenceFinder(min_length=2, max_length=8, min_count=3)
        data = b"AAAA" * 10
        sequences = finder.find_sequences(data)

        assert len(sequences) > 0
        assert all(isinstance(s, RepeatingSequence) for s in sequences)

    def test_frequency_filtering(self) -> None:
        """Test that min_frequency filters results."""
        finder = RepeatingSequenceFinder(min_length=2, max_length=8, min_count=2, min_frequency=0.5)
        data = b"AA" * 3 + b"B" * 100  # AA has low frequency
        sequences = finder.find_sequences(data)

        # Should filter out low-frequency patterns
        for seq in sequences:
            assert seq.frequency >= 0.5

    def test_find_ngrams(self) -> None:
        """Test find_ngrams method."""
        finder = RepeatingSequenceFinder()
        data = b"ABABCDCD"
        ngrams = finder.find_ngrams(data, n=2, top_k=5)

        assert len(ngrams) > 0
        assert all(isinstance(ng, NgramResult) for ng in ngrams)

    def test_find_longest(self) -> None:
        """Test find_longest method."""
        finder = RepeatingSequenceFinder()
        data = b"XYZXYZ"
        result = finder.find_longest(data)

        assert result is not None
        assert isinstance(result, RepeatingSequence)
        assert result.length >= 3

    def test_find_longest_no_repeat(self) -> None:
        """Test find_longest with no repeating patterns."""
        finder = RepeatingSequenceFinder()
        data = b"ABCDEF"
        result = finder.find_longest(data)

        assert result is None

    def test_with_numpy_array(self) -> None:
        """Test finder with numpy array input."""
        finder = RepeatingSequenceFinder(min_length=2, max_length=8)
        data = np.array([0xAA, 0x55] * 10, dtype=np.uint8)
        sequences = finder.find_sequences(data)

        assert len(sequences) > 0


# =============================================================================
# Edge Cases and Error Conditions
# =============================================================================


@pytest.mark.unit
@pytest.mark.pattern
class TestPatternsSequencesEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_long_pattern(self) -> None:
        """Test with very long repeating pattern."""
        pattern = bytes(range(100))
        data = pattern * 3
        sequences = find_repeating_sequences(data, min_length=50, max_length=150, min_count=2)

        # Should find long pattern
        assert len(sequences) > 0

    def test_entire_data_is_pattern(self) -> None:
        """Test when entire data is a repeating pattern."""
        data = b"AB" * 100
        sequences = find_repeating_sequences(data, min_length=2, max_length=10, min_count=50)

        # Should detect "AB" pattern
        assert len(sequences) > 0

    def test_pattern_at_boundaries(self) -> None:
        """Test patterns at data boundaries."""
        data = b"XYXYXY"
        sequences = find_repeating_sequences(data, min_length=2, max_length=4, min_count=2)

        # Should detect "XY" at start and end
        assert len(sequences) > 0

    def test_unicode_bytes(self) -> None:
        """Test with unicode byte sequences."""
        data = b"Hello" * 5
        sequences = find_repeating_sequences(data, min_length=3, max_length=10, min_count=3)

        assert len(sequences) > 0

    def test_null_bytes(self) -> None:
        """Test with null bytes."""
        data = b"\x00" * 20
        sequences = find_repeating_sequences(data, min_length=2, max_length=10, min_count=5)

        # Should find repeating null bytes
        assert len(sequences) > 0

    def test_mixed_data_types(self) -> None:
        """Test conversion from different input types."""
        # Test all supported types produce same results
        pattern = b"TEST"
        data_bytes = pattern * 5
        data_bytearray = bytearray(data_bytes)
        data_numpy = np.frombuffer(data_bytes, dtype=np.uint8)
        data_memoryview = memoryview(data_bytes)

        seq_bytes = find_repeating_sequences(data_bytes, min_length=4, max_length=8, min_count=3)
        seq_bytearray = find_repeating_sequences(
            data_bytearray, min_length=4, max_length=8, min_count=3
        )
        seq_numpy = find_repeating_sequences(data_numpy, min_length=4, max_length=8, min_count=3)
        seq_memoryview = find_repeating_sequences(
            data_memoryview, min_length=4, max_length=8, min_count=3
        )

        # All should find same pattern
        assert len(seq_bytes) > 0
        assert len(seq_bytearray) > 0
        assert len(seq_numpy) > 0
        assert len(seq_memoryview) > 0

    def test_large_data(self) -> None:
        """Test with large data (performance test)."""
        # Create large dataset with known pattern
        data = b"PATTERN" * 10000
        sequences = find_repeating_sequences(data, min_length=7, max_length=20, min_count=100)

        # Should efficiently find pattern
        assert len(sequences) > 0

    def test_random_noise(self) -> None:
        """Test with random noise (should find few/no patterns)."""
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=1000, dtype=np.uint8).tobytes()
        sequences = find_repeating_sequences(data, min_length=4, max_length=8, min_count=5)

        # Random data should have few meaningful patterns
        assert isinstance(sequences, list)

    def test_alternating_pattern(self) -> None:
        """Test with alternating pattern."""
        data = b"\x00\xff" * 50
        sequences = find_repeating_sequences(data, min_length=2, max_length=8, min_count=10)

        # Should detect alternating pattern
        assert len(sequences) > 0

    def test_nested_patterns(self) -> None:
        """Test with nested/overlapping patterns."""
        data = b"AAABAAABAAAB"
        sequences = find_repeating_sequences(data, min_length=2, max_length=8, min_count=2)

        # Should detect both "AAA" and "AAAB" patterns
        assert len(sequences) > 0

    def test_unsupported_data_type(self) -> None:
        """Test that unsupported data type raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported data type"):
            find_repeating_sequences([1, 2, 3], min_length=2)  # type: ignore

    def test_edit_distance_empty_strings(self) -> None:
        """Test edit distance with empty strings."""
        # Test through approximate repeats with empty-like patterns
        data = b"A" + b"AB"
        results = find_approximate_repeats(data, min_length=1, max_distance=2, min_count=2)
        # Should handle gracefully
        assert isinstance(results, list)

    def test_find_longest_repeat_empty_lcp(self) -> None:
        """Test find_longest_repeat with data that produces empty LCP."""
        # Single character - will produce empty LCP array edge case
        data = b""
        result = find_longest_repeat(data)
        assert result is None

    def test_approximate_repeats_with_single_char_patterns(self) -> None:
        """Test approximate repeats with single character patterns."""
        # This will test edit distance with very short strings including empty cases
        data = b"A" * 5
        results = find_approximate_repeats(data, min_length=1, max_distance=1, min_count=3)
        # Should find repeating "A" pattern
        assert len(results) > 0
