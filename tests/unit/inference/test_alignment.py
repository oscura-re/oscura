"""Comprehensive unit tests for sequence alignment.

Requirements addressed: PSI-003

This module tests all public functions and classes in the alignment module,
including Needleman-Wunsch global alignment, Smith-Waterman local alignment,
multiple sequence alignment, and conserved/variable region detection.
"""

import numpy as np
import pytest

from oscura.inference.alignment import (
    AlignmentResult,
    align_global,
    align_local,
    align_multiple,
    compute_similarity,
    find_conserved_regions,
    find_variable_regions,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


# =============================================================================
# Test Data Classes
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestAlignmentResult:
    """Test AlignmentResult dataclass."""

    def test_create_alignment_result(self) -> None:
        """Test creating an AlignmentResult instance."""
        result = AlignmentResult(
            aligned_a=[1, 2, 3, -1, 4],
            aligned_b=[1, 2, -1, 3, 4],
            score=5.0,
            similarity=0.75,
            identity=0.6,
            gaps=2,
            conserved_regions=[(0, 2)],
            variable_regions=[(2, 4)],
        )

        assert result.aligned_a == [1, 2, 3, -1, 4]
        assert result.aligned_b == [1, 2, -1, 3, 4]
        assert result.score == 5.0
        assert result.similarity == 0.75
        assert result.identity == 0.6
        assert result.gaps == 2
        assert result.conserved_regions == [(0, 2)]
        assert result.variable_regions == [(2, 4)]

    def test_alignment_result_with_bytes(self) -> None:
        """Test AlignmentResult can store bytes."""
        result = AlignmentResult(
            aligned_a=b"ABC",
            aligned_b=b"ABC",
            score=10.0,
            similarity=1.0,
            identity=1.0,
            gaps=0,
            conserved_regions=[(0, 3)],
            variable_regions=[],
        )

        assert result.aligned_a == b"ABC"
        assert result.aligned_b == b"ABC"

    def test_alignment_result_empty_regions(self) -> None:
        """Test AlignmentResult with empty regions."""
        result = AlignmentResult(
            aligned_a=[1, 2, 3],
            aligned_b=[1, 2, 3],
            score=3.0,
            similarity=1.0,
            identity=1.0,
            gaps=0,
            conserved_regions=[],
            variable_regions=[],
        )

        assert result.conserved_regions == []
        assert result.variable_regions == []


# =============================================================================
# Test Global Alignment (Needleman-Wunsch)
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestAlignGlobal:
    """Test align_global function."""

    def test_identical_sequences_bytes(self) -> None:
        """Test global alignment of identical byte sequences."""
        seq_a = b"ABCDEF"
        seq_b = b"ABCDEF"

        result = align_global(seq_a, seq_b)

        # Should have perfect alignment
        assert len(result.aligned_a) == len(result.aligned_b)
        assert result.similarity == 1.0
        assert result.identity == 1.0
        assert result.gaps == 0
        assert result.score > 0

    def test_identical_sequences_arrays(self) -> None:
        """Test global alignment of identical array sequences."""
        seq_a = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        seq_b = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

        result = align_global(seq_a, seq_b)

        assert result.similarity == 1.0
        assert result.identity == 1.0
        assert result.gaps == 0

    def test_completely_different_sequences(self) -> None:
        """Test global alignment of completely different sequences."""
        seq_a = b"AAAA"
        seq_b = b"BBBB"

        result = align_global(seq_a, seq_b)

        # Should have same length (no gaps needed)
        assert len(result.aligned_a) == len(result.aligned_b)
        # Low similarity (only mismatches)
        assert result.similarity == 0.0
        assert result.identity == 0.0
        # No gaps needed if same length
        assert result.gaps == 0

    def test_sequences_with_insertion(self) -> None:
        """Test global alignment with insertion."""
        seq_a = b"ABCDEF"
        seq_b = b"ABXCDEF"  # X inserted

        result = align_global(seq_a, seq_b)

        # Should have gaps in seq_a
        assert -1 in result.aligned_a
        assert len(result.aligned_a) == len(result.aligned_b)
        assert result.gaps > 0

    def test_sequences_with_deletion(self) -> None:
        """Test global alignment with deletion."""
        seq_a = b"ABCDEF"
        seq_b = b"ABDEF"  # C deleted

        result = align_global(seq_a, seq_b)

        # Should have gaps in seq_b
        assert -1 in result.aligned_b
        assert len(result.aligned_a) == len(result.aligned_b)
        assert result.gaps > 0

    def test_different_length_sequences(self) -> None:
        """Test global alignment of different length sequences."""
        seq_a = b"ABC"
        seq_b = b"ABCDEFGH"

        result = align_global(seq_a, seq_b)

        # Final alignment should be same length
        assert len(result.aligned_a) == len(result.aligned_b)
        # Should have gaps to account for length difference
        assert result.gaps > 0
        assert -1 in result.aligned_a

    def test_empty_sequences(self) -> None:
        """Test global alignment of empty sequences."""
        seq_a = b""
        seq_b = b""

        # Empty sequences are now handled gracefully
        result = align_global(seq_a, seq_b)
        assert result.identity == 0.0
        assert result.gaps == 0
        assert len(result.aligned_a) == 0
        assert len(result.aligned_b) == 0

    def test_one_empty_sequence(self) -> None:
        """Test global alignment with one empty sequence."""
        seq_a = b"ABC"
        seq_b = b""

        result = align_global(seq_a, seq_b)

        # All gaps in seq_b
        assert len(result.aligned_a) == len(result.aligned_b)
        assert all(v == -1 for v in result.aligned_b)
        assert result.gaps == len(seq_a)

    def test_custom_scoring_parameters(self) -> None:
        """Test global alignment with custom scoring parameters."""
        seq_a = b"ABCD"
        seq_b = b"ABCD"

        result = align_global(
            seq_a, seq_b, gap_penalty=-2.0, match_score=2.0, mismatch_penalty=-2.0
        )

        # Should still align perfectly
        assert result.similarity == 1.0
        assert result.identity == 1.0
        # Score should reflect custom match_score
        assert result.score == 8.0  # 4 matches * 2.0

    def test_single_byte_sequences(self) -> None:
        """Test global alignment of single byte sequences."""
        seq_a = b"A"
        seq_b = b"B"

        result = align_global(seq_a, seq_b)

        assert len(result.aligned_a) == 1
        assert len(result.aligned_b) == 1
        assert result.gaps == 0

    def test_alignment_preserves_order(self) -> None:
        """Test that alignment preserves sequence order."""
        seq_a = b"ABCDEF"
        seq_b = b"ABCDEF"

        result = align_global(seq_a, seq_b)

        # Remove gaps and check order
        a_no_gaps = [v for v in result.aligned_a if v != -1]
        b_no_gaps = [v for v in result.aligned_b if v != -1]

        assert a_no_gaps == list(seq_a)
        assert b_no_gaps == list(seq_b)

    def test_conserved_regions_detection(self) -> None:
        """Test that conserved regions are detected in global alignment."""
        # Sequences with matching prefix and suffix
        seq_a = b"AAAABBBBCCCC"
        seq_b = b"AAAACCCCDDDD"

        result = align_global(seq_a, seq_b)

        # Should have at least one conserved region (the AAAA prefix)
        assert len(result.conserved_regions) > 0

    def test_variable_regions_detection(self) -> None:
        """Test that variable regions are detected in global alignment."""
        seq_a = b"AAAABBBB"
        seq_b = b"AAAACCCC"

        result = align_global(seq_a, seq_b)

        # Should have variable region in the second half
        assert len(result.variable_regions) > 0


# =============================================================================
# Test Local Alignment (Smith-Waterman)
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestAlignLocal:
    """Test align_local function."""

    def test_identical_sequences_bytes(self) -> None:
        """Test local alignment of identical byte sequences."""
        seq_a = b"ABCDEF"
        seq_b = b"ABCDEF"

        result = align_local(seq_a, seq_b)

        # Should find complete alignment
        assert result.similarity == 1.0
        assert result.identity == 1.0
        assert result.gaps == 0
        assert result.score > 0

    def test_partial_match(self) -> None:
        """Test local alignment finds best local match."""
        seq_a = b"XXXABCDEFYYY"
        seq_b = b"ZZZZABCDEZZZ"

        result = align_local(seq_a, seq_b)

        # Should find ABCDE as best local alignment
        assert result.score > 0
        assert result.similarity > 0.5
        # Aligned region should be shorter than full sequences
        assert len(result.aligned_a) < len(seq_a)
        assert len(result.aligned_b) < len(seq_b)

    def test_no_match(self) -> None:
        """Test local alignment with no significant match."""
        seq_a = b"AAAA"
        seq_b = b"BBBB"

        result = align_local(seq_a, seq_b)

        # Should have low or zero score
        assert result.score <= 0
        # Empty alignment or very short
        if len(result.aligned_a) > 0:
            assert result.similarity < 0.5

    def test_small_subsequence_match(self) -> None:
        """Test local alignment finds small matching subsequence."""
        seq_a = b"XXXXXXXXXABCYYYYYYYY"
        seq_b = b"ZZZZZZZZABCZZZZZZZZ"

        result = align_local(seq_a, seq_b)

        # Should find ABC as matching region
        assert result.score > 0
        assert len(result.aligned_a) >= 3

    def test_empty_sequences(self) -> None:
        """Test local alignment of empty sequences."""
        seq_a = b""
        seq_b = b""

        result = align_local(seq_a, seq_b)

        assert len(result.aligned_a) == 0
        assert len(result.aligned_b) == 0
        assert result.score == 0.0

    def test_one_empty_sequence(self) -> None:
        """Test local alignment with one empty sequence."""
        seq_a = b"ABC"
        seq_b = b""

        result = align_local(seq_a, seq_b)

        # Should have no alignment
        assert result.score == 0.0
        assert len(result.aligned_a) == 0

    def test_custom_scoring_parameters(self) -> None:
        """Test local alignment with custom scoring parameters."""
        seq_a = b"ABCD"
        seq_b = b"ABCD"

        result = align_local(seq_a, seq_b, gap_penalty=-2.0, match_score=3.0, mismatch_penalty=-2.0)

        # Should find full match with higher score
        assert result.similarity == 1.0
        assert result.score == 12.0  # 4 matches * 3.0

    def test_alignment_arrays(self) -> None:
        """Test local alignment with numpy arrays."""
        seq_a = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        seq_b = np.array([10, 20, 3, 4, 5, 30], dtype=np.uint8)

        result = align_local(seq_a, seq_b)

        # Should find [3, 4, 5] as best local match
        assert result.score > 0
        assert len(result.aligned_a) >= 3

    def test_multiple_matching_regions(self) -> None:
        """Test local alignment with multiple matching regions."""
        seq_a = b"ABCXXXABC"
        seq_b = b"YYYABCZZZ"

        result = align_local(seq_a, seq_b)

        # Should find one of the ABC regions
        assert result.score > 0
        assert len(result.aligned_a) >= 3


# =============================================================================
# Test Multiple Sequence Alignment
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestAlignMultiple:
    """Test align_multiple function."""

    def test_empty_sequences_list(self) -> None:
        """Test multiple alignment with empty list."""
        sequences: list[bytes] = []

        result = align_multiple(sequences)

        assert result == []

    def test_single_sequence(self) -> None:
        """Test multiple alignment with single sequence."""
        sequences = [b"ABCD"]

        result = align_multiple(sequences)

        assert len(result) == 1
        assert result[0] == [65, 66, 67, 68]  # ASCII values

    def test_two_identical_sequences(self) -> None:
        """Test multiple alignment of two identical sequences."""
        sequences = [b"ABCD", b"ABCD"]

        result = align_multiple(sequences)

        assert len(result) == 2
        # Should have same length
        assert len(result[0]) == len(result[1])
        # Should be identical
        assert result[0] == result[1]

    def test_three_sequences_progressive(self) -> None:
        """Test progressive multiple alignment of three sequences."""
        sequences = [b"ABCD", b"ABCD", b"ABCD"]

        result = align_multiple(sequences, method="progressive")

        assert len(result) == 3
        # All should have same length
        lengths = [len(seq) for seq in result]
        assert len(set(lengths)) == 1

    def test_sequences_with_variations(self) -> None:
        """Test multiple alignment with variations."""
        sequences = [b"ABCD", b"AXCD", b"ABXD"]

        result = align_multiple(sequences, method="progressive")

        assert len(result) == 3
        # All should have same length
        lengths = [len(seq) for seq in result]
        assert len(set(lengths)) == 1

    def test_sequences_different_lengths(self) -> None:
        """Test multiple alignment of different length sequences."""
        sequences = [b"AB", b"ABCD", b"ABCDEF"]

        result = align_multiple(sequences, method="progressive")

        assert len(result) == 3
        # All should have same length (with gaps)
        lengths = [len(seq) for seq in result]
        assert len(set(lengths)) == 1
        # Shorter sequences should have gaps
        assert -1 in result[0]

    def test_iterative_method_fallback(self) -> None:
        """Test that iterative method falls back to progressive."""
        sequences = [b"ABCD", b"ABCD"]

        result = align_multiple(sequences, method="iterative")

        # Should still work (falls back to progressive)
        assert len(result) == 2

    def test_array_sequences(self) -> None:
        """Test multiple alignment with numpy arrays."""
        sequences = [
            np.array([1, 2, 3], dtype=np.uint8),
            np.array([1, 2, 3], dtype=np.uint8),
        ]

        result = align_multiple(sequences)

        assert len(result) == 2
        assert result[0] == result[1]

    def test_mixed_sequences(self) -> None:
        """Test multiple alignment with mixed bytes and arrays."""
        sequences = [b"ABC", np.array([65, 66, 67], dtype=np.uint8)]

        result = align_multiple(sequences)

        assert len(result) == 2
        # Should produce similar results (both are ABC)
        assert len(result[0]) == len(result[1])

    def test_many_sequences(self) -> None:
        """Test multiple alignment with many sequences."""
        sequences = [b"ABCD"] * 10

        result = align_multiple(sequences, method="progressive")

        assert len(result) == 10
        # All should be identical
        for seq in result:
            assert seq == result[0]


# =============================================================================
# Test Similarity Computation
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestComputeSimilarity:
    """Test compute_similarity function."""

    def test_identical_sequences(self) -> None:
        """Test similarity of identical sequences."""
        aligned_a = [1, 2, 3, 4, 5]
        aligned_b = [1, 2, 3, 4, 5]

        similarity = compute_similarity(aligned_a, aligned_b)

        assert similarity == 1.0

    def test_completely_different_sequences(self) -> None:
        """Test similarity of completely different sequences."""
        aligned_a = [1, 2, 3, 4, 5]
        aligned_b = [6, 7, 8, 9, 10]

        similarity = compute_similarity(aligned_a, aligned_b)

        assert similarity == 0.0

    def test_partially_similar_sequences(self) -> None:
        """Test similarity of partially similar sequences."""
        aligned_a = [1, 2, 3, 4, 5]
        aligned_b = [1, 2, 8, 9, 5]

        similarity = compute_similarity(aligned_a, aligned_b)

        assert 0.0 < similarity < 1.0
        assert similarity == 0.6  # 3/5 matches

    def test_sequences_with_gaps(self) -> None:
        """Test similarity computation with gaps."""
        aligned_a = [1, 2, -1, 4, 5]
        aligned_b = [1, 2, 3, 4, 5]

        similarity = compute_similarity(aligned_a, aligned_b)

        # Gap is not a match, so similarity < 1.0
        assert similarity < 1.0
        # 4 matches (1,2,4,5) out of 5 non-double-gap positions = 0.8
        assert similarity == 0.8

    def test_sequences_with_double_gaps(self) -> None:
        """Test that double gaps are skipped in similarity."""
        aligned_a = [1, 2, -1, 4, 5]
        aligned_b = [1, 2, -1, 4, 5]

        similarity = compute_similarity(aligned_a, aligned_b)

        # Double gaps should be ignored
        assert similarity == 1.0  # 4/4 non-double-gap positions match

    def test_empty_sequences(self) -> None:
        """Test similarity of empty sequences."""
        aligned_a: list[int] = []
        aligned_b: list[int] = []

        similarity = compute_similarity(aligned_a, aligned_b)

        assert similarity == 0.0

    def test_all_gaps(self) -> None:
        """Test similarity when all positions are double gaps."""
        aligned_a = [-1, -1, -1]
        aligned_b = [-1, -1, -1]

        similarity = compute_similarity(aligned_a, aligned_b)

        # All positions are double gaps, so similarity is 0
        assert similarity == 0.0

    def test_bytes_sequences(self) -> None:
        """Test similarity with bytes sequences."""
        aligned_a = b"ABC"
        aligned_b = b"ABC"

        similarity = compute_similarity(aligned_a, aligned_b)

        assert similarity == 1.0

    def test_different_length_sequences_error(self) -> None:
        """Test that different length sequences raise error."""
        aligned_a = [1, 2, 3]
        aligned_b = [1, 2, 3, 4, 5]

        with pytest.raises(ValueError, match="must have same length"):
            compute_similarity(aligned_a, aligned_b)

    def test_single_element_match(self) -> None:
        """Test similarity with single element."""
        aligned_a = [1]
        aligned_b = [1]

        similarity = compute_similarity(aligned_a, aligned_b)

        assert similarity == 1.0

    def test_single_element_mismatch(self) -> None:
        """Test similarity with single element mismatch."""
        aligned_a = [1]
        aligned_b = [2]

        similarity = compute_similarity(aligned_a, aligned_b)

        assert similarity == 0.0


# =============================================================================
# Test Conserved Region Detection
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestFindConservedRegions:
    """Test find_conserved_regions function."""

    def test_empty_sequences(self) -> None:
        """Test conserved regions with empty sequences."""
        sequences: list[list[int]] = []

        regions = find_conserved_regions(sequences)

        assert regions == []

    def test_single_sequence(self) -> None:
        """Test conserved regions with single sequence."""
        sequences = [[1, 2, 3, 4, 5, 6, 7, 8]]

        regions = find_conserved_regions(sequences)

        # All positions are "conserved" (100% conservation)
        assert len(regions) > 0

    def test_identical_sequences(self) -> None:
        """Test conserved regions with identical sequences."""
        sequences = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)

        # Entire sequence should be conserved
        assert len(regions) > 0
        assert (0, 8) in regions

    def test_partially_conserved_sequences(self) -> None:
        """Test conserved regions with partial conservation."""
        sequences = [
            [1, 1, 1, 1, 5, 6, 7, 8],  # First 4 conserved
            [1, 1, 1, 1, 9, 10, 11, 12],
            [1, 1, 1, 1, 13, 14, 15, 16],
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)

        # First 4 positions should be conserved
        assert (0, 4) in regions

    def test_multiple_conserved_regions(self) -> None:
        """Test detection of multiple conserved regions."""
        sequences = [
            [1, 1, 1, 1, 99, 99, 2, 2, 2, 2],  # Two conserved regions
            [1, 1, 1, 1, 88, 88, 2, 2, 2, 2],
            [1, 1, 1, 1, 77, 77, 2, 2, 2, 2],
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)

        # Should find two regions: [0:4] and [6:10]
        assert len(regions) >= 2
        assert (0, 4) in regions
        assert (6, 10) in regions

    def test_min_length_filter(self) -> None:
        """Test that min_length parameter filters short regions."""
        sequences = [
            [1, 1, 99, 2, 2, 2, 2, 2],  # 2 conserved, then 5 conserved
            [1, 1, 88, 2, 2, 2, 2, 2],
            [1, 1, 77, 2, 2, 2, 2, 2],
        ]

        # With min_length=4, should only find second region
        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)

        # First region [0:2] is too short (length 2 < 4)
        assert (0, 2) not in regions
        # Second region [3:8] should be found (length 5 >= 4)
        assert (3, 8) in regions

    def test_conservation_threshold(self) -> None:
        """Test conservation threshold parameter."""
        sequences = [
            [1, 1, 1, 1, 1, 1],  # 100% conservation
            [1, 1, 1, 1, 2, 2],  # 66% conservation at positions 4-5
            [1, 1, 1, 1, 3, 3],
        ]

        # With 90% threshold, only first 4 positions
        regions_90 = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)
        assert (0, 4) in regions_90

        # With 60% threshold, might include more
        regions_60 = find_conserved_regions(sequences, min_conservation=0.6, min_length=4)
        assert len(regions_60) >= len(regions_90)

    def test_sequences_with_gaps(self) -> None:
        """Test conserved regions with gaps."""
        sequences = [
            [1, 1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1],
            [1, 1, 1, 1, -1, -1],
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)

        # Non-gap region should be conserved
        assert (0, 4) in regions

    def test_short_sequences(self) -> None:
        """Test conserved regions with sequences shorter than min_length."""
        sequences = [
            [1, 1],  # Length 2, but min_length=4
            [1, 1],
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)

        # No regions should be found (too short)
        assert regions == []

    def test_all_variable_sequences(self) -> None:
        """Test conserved regions with all variable sequences."""
        sequences = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=4)

        # No conserved regions
        assert regions == []


# =============================================================================
# Test Variable Region Detection
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestFindVariableRegions:
    """Test find_variable_regions function."""

    def test_empty_sequences(self) -> None:
        """Test variable regions with empty sequences."""
        sequences: list[list[int]] = []

        regions = find_variable_regions(sequences)

        assert regions == []

    def test_identical_sequences(self) -> None:
        """Test variable regions with identical sequences."""
        sequences = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        # No variable regions (all conserved)
        assert regions == []

    def test_completely_variable_sequences(self) -> None:
        """Test variable regions with completely variable sequences."""
        sequences = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        # Entire sequence should be variable
        assert len(regions) > 0
        assert (0, 5) in regions

    def test_partially_variable_sequences(self) -> None:
        """Test variable regions with partial variability."""
        sequences = [
            [1, 1, 1, 99, 99, 99],  # Last 3 variable
            [1, 1, 1, 88, 88, 88],
            [1, 1, 1, 77, 77, 77],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        # Last 3 positions should be variable
        assert (3, 6) in regions

    def test_multiple_variable_regions(self) -> None:
        """Test detection of multiple variable regions."""
        sequences = [
            [99, 99, 1, 1, 88, 88, 1, 1],  # Two variable regions
            [77, 77, 1, 1, 66, 66, 1, 1],
            [55, 55, 1, 1, 44, 44, 1, 1],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        # Should find two regions: [0:2] and [4:6]
        assert len(regions) >= 2
        assert (0, 2) in regions
        assert (4, 6) in regions

    def test_min_length_filter(self) -> None:
        """Test that min_length parameter filters short regions."""
        sequences = [
            [99, 1, 1, 1, 88, 88, 88],  # 1 variable, then 3 variable
            [77, 1, 1, 1, 66, 66, 66],
        ]

        # With min_length=2, should only find second region
        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        # First region [0:1] is too short (length 1 < 2)
        assert (0, 1) not in regions
        # Second region [4:7] should be found (length 3 >= 2)
        assert (4, 7) in regions

    def test_conservation_threshold(self) -> None:
        """Test conservation threshold parameter."""
        sequences = [
            [1, 2, 3, 4, 5],  # All different
            [1, 2, 6, 7, 8],  # Some same
        ]

        # With 10% threshold, even slight conservation excludes
        regions_10 = find_variable_regions(sequences, max_conservation=0.1, min_length=2)

        # With 60% threshold, allows more variation
        regions_60 = find_variable_regions(sequences, max_conservation=0.6, min_length=2)

        assert len(regions_60) >= len(regions_10)

    def test_sequences_with_gaps(self) -> None:
        """Test variable regions with gaps."""
        sequences = [
            [99, 99, -1, -1],
            [88, 88, -1, -1],
            [77, 77, -1, -1],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        # First two positions should be variable
        assert (0, 2) in regions

    def test_single_sequence(self) -> None:
        """Test variable regions with single sequence."""
        sequences = [[1, 2, 3, 4, 5]]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        # Single sequence has 100% conservation, so no variable regions
        assert regions == []

    def test_alternating_conservation(self) -> None:
        """Test variable regions with alternating conservation."""
        sequences = [
            [1, 99, 1, 99, 1, 99],  # Alternating conserved/variable
            [1, 88, 1, 88, 1, 88],
            [1, 77, 1, 77, 1, 77],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=1)

        # With min_length=1, should find the three single-position variable regions
        # at positions 1, 3, and 5
        assert len(regions) == 3
        assert (1, 2) in regions
        assert (3, 4) in regions
        assert (5, 6) in regions


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.inference
class TestAlignmentIntegration:
    """Integration tests for alignment module."""

    def test_global_then_similarity(self) -> None:
        """Test global alignment followed by similarity computation."""
        seq_a = b"ABCDEFGH"
        seq_b = b"ABXDEFGH"

        result = align_global(seq_a, seq_b)

        # Manually compute similarity
        manual_similarity = compute_similarity(result.aligned_a, result.aligned_b)

        # Should match the result's similarity
        assert result.similarity == manual_similarity

    def test_multiple_alignment_conservation(self) -> None:
        """Test multiple alignment followed by conservation detection."""
        sequences = [
            b"AAAABBBBCCCC",
            b"AAAABBBBCCCC",
            b"AAAABBBBCCCC",
        ]

        aligned = align_multiple(sequences)
        conserved = find_conserved_regions(aligned, min_conservation=0.9, min_length=4)

        # Entire sequence should be conserved
        assert len(conserved) > 0

    def test_multiple_alignment_variation(self) -> None:
        """Test multiple alignment followed by variation detection."""
        sequences = [
            b"AAAABBBB",
            b"AAAACCCC",
            b"AAAADDDD",
        ]

        aligned = align_multiple(sequences)
        variable = find_variable_regions(aligned, max_conservation=0.5, min_length=2)

        # Second half should be variable
        assert len(variable) > 0

    def test_local_alignment_of_long_sequences(self) -> None:
        """Test local alignment finds best match in long sequences."""
        # Create sequences with a small matching region
        prefix = b"X" * 50
        matching = b"ABCDEFGH"
        suffix = b"Y" * 50

        seq_a = prefix + matching + suffix
        seq_b = b"Z" * 50 + matching + b"W" * 50

        result = align_local(seq_a, seq_b)

        # Should find the matching region
        assert result.score > 0
        assert len(result.aligned_a) >= len(matching)

    def test_edge_case_all_same_byte(self) -> None:
        """Test alignment of sequences with all same byte."""
        seq_a = b"AAAAAAA"
        seq_b = b"AAAAAAA"

        result = align_global(seq_a, seq_b)

        assert result.similarity == 1.0
        assert result.identity == 1.0
        assert result.gaps == 0

    def test_edge_case_binary_pattern(self) -> None:
        """Test alignment with binary pattern."""
        seq_a = bytes([0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF])
        seq_b = bytes([0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF])

        result = align_global(seq_a, seq_b)

        assert result.similarity == 1.0
        assert result.identity == 1.0

    def test_realistic_protocol_messages(self) -> None:
        """Test alignment of realistic protocol-like messages."""
        # Messages with header, length, data, checksum structure
        msg1 = bytes([0xAA, 0x55, 0x08, 0x01, 0x02, 0x03, 0x04, 0xCC])
        msg2 = bytes([0xAA, 0x55, 0x08, 0x05, 0x06, 0x07, 0x08, 0xDD])

        result = align_global(msg1, msg2)

        # Header should match
        assert result.aligned_a[0] == result.aligned_b[0]  # 0xAA
        assert result.aligned_a[1] == result.aligned_b[1]  # 0x55
        # Data portion should differ
        assert result.similarity < 1.0
