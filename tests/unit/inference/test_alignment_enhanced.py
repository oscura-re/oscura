"""Enhanced tests for alignment module to improve coverage.

Requirements addressed: PSI-003

This module adds additional edge case tests to improve coverage beyond
the existing comprehensive test suite.
"""

import numpy as np
import pytest

from oscura.inference.alignment import (
    _compute_consensus,
    _insert_gaps_from_alignment,
    align_global,
    align_local,
    align_multiple,
    compute_similarity,
    find_conserved_regions,
    find_variable_regions,
)

pytestmark = [pytest.mark.unit, pytest.mark.inference]


@pytest.mark.unit
@pytest.mark.inference
class TestAlignmentEdgeCases:
    """Additional edge case tests for alignment functions."""

    def test_align_global_single_element(self) -> None:
        """Test global alignment with single element sequences."""
        seq_a = b"A"
        seq_b = b"A"

        result = align_global(seq_a, seq_b)

        assert result.identity == 1.0
        assert len(result.aligned_a) == 1
        assert len(result.aligned_b) == 1

    def test_align_global_very_different_lengths(self) -> None:
        """Test global alignment with very different length sequences."""
        seq_a = b"A"
        seq_b = b"BCDEFGHIJKLMNOP"

        result = align_global(seq_a, seq_b)

        # Final alignment should be same length
        assert len(result.aligned_a) == len(result.aligned_b)
        assert result.gaps > 10

    def test_align_global_custom_penalties_high_gap(self) -> None:
        """Test global alignment with very high gap penalty."""
        seq_a = b"ABCD"
        seq_b = b"ABXCD"

        result = align_global(seq_a, seq_b, gap_penalty=-10.0)

        # High gap penalty should discourage gaps
        assert result.score < 0  # Negative due to mismatches

    def test_align_global_zero_penalties(self) -> None:
        """Test global alignment with zero penalties."""
        seq_a = b"ABC"
        seq_b = b"XYZ"

        result = align_global(seq_a, seq_b, gap_penalty=0.0, mismatch_penalty=0.0)

        # All options have same score (0), so any alignment is valid
        assert len(result.aligned_a) == len(result.aligned_b)

    def test_align_local_single_match_in_noise(self) -> None:
        """Test local alignment finds single matching byte in noise."""
        seq_a = bytes([0xFF] * 100 + [0xAA] + [0xFF] * 100)
        seq_b = bytes([0x00] * 100 + [0xAA] + [0x00] * 100)

        result = align_local(seq_a, seq_b)

        # Should find the matching 0xAA
        assert result.score > 0
        assert 0xAA in result.aligned_a

    def test_align_local_zero_score_empty_result(self) -> None:
        """Test local alignment returns empty when max score is 0."""
        seq_a = b"AAAA"
        seq_b = b"BBBB"

        result = align_local(seq_a, seq_b, match_score=1.0, mismatch_penalty=-1.0)

        # No positive scoring alignment possible
        assert result.score == 0
        # Empty or very short alignment
        assert len(result.aligned_a) == 0 or result.similarity < 0.5

    def test_align_multiple_with_gaps(self) -> None:
        """Test multiple alignment handles gaps correctly."""
        sequences = [
            b"ABCD",
            b"AXCD",
            b"ABCD",
        ]

        result = align_multiple(sequences)

        # All sequences should have same length with gaps if needed
        lengths = [len(seq) for seq in result]
        assert len(set(lengths)) == 1

    def test_align_multiple_very_different_sequences(self) -> None:
        """Test multiple alignment with very different sequences."""
        sequences = [
            b"AAA",
            b"BBBBBBB",
            b"C",
        ]

        result = align_multiple(sequences)

        # All should be padded to same length
        lengths = [len(seq) for seq in result]
        assert len(set(lengths)) == 1
        # Shortest should have most gaps
        assert result[2].count(-1) > result[0].count(-1)

    def test_compute_similarity_single_gap(self) -> None:
        """Test similarity computation with single gap."""
        aligned_a = [1, 2, 3, -1]
        aligned_b = [1, 2, 3, 4]

        similarity = compute_similarity(aligned_a, aligned_b)

        assert 0.0 < similarity < 1.0
        assert similarity == 0.75  # 3/4 positions match (ignoring gaps)

    def test_compute_similarity_alternating_gaps(self) -> None:
        """Test similarity with alternating gaps."""
        aligned_a = [1, -1, 3, -1, 5]
        aligned_b = [-1, 2, -1, 4, -1]

        similarity = compute_similarity(aligned_a, aligned_b)

        # No actual matches (all gaps or different values)
        assert similarity == 0.0

    def test_find_conserved_regions_single_long_region(self) -> None:
        """Test finding single long conserved region."""
        sequences = [
            [1] * 100,
            [1] * 100,
            [1] * 100,
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=50)

        assert len(regions) == 1
        assert regions[0] == (0, 100)

    def test_find_conserved_regions_low_threshold(self) -> None:
        """Test conserved regions with low conservation threshold."""
        sequences = [
            [1, 1, 2, 2],
            [1, 1, 3, 3],
            [1, 1, 4, 4],
        ]

        # With 0.5 threshold, first two positions (66% conservation) should qualify
        regions = find_conserved_regions(sequences, min_conservation=0.5, min_length=2)

        # First two positions have 100% conservation, last two have 0%
        assert len(regions) >= 1
        assert (0, 2) in regions

    def test_find_variable_regions_high_threshold(self) -> None:
        """Test variable regions with high conservation threshold."""
        sequences = [
            [1, 1, 2, 3],
            [1, 1, 4, 5],
            [1, 1, 6, 7],
        ]

        # First two are conserved, last two are variable
        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=2)

        assert len(regions) >= 1
        assert (2, 4) in regions

    def test_find_variable_regions_single_position_variable(self) -> None:
        """Test variable region with min_length=1."""
        sequences = [
            [1, 2, 1, 1],
            [1, 3, 1, 1],
            [1, 4, 1, 1],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=1)

        # Position 1 is variable
        assert (1, 2) in regions

    def test_compute_consensus_empty(self) -> None:
        """Test consensus computation with empty sequences."""
        aligned_sequences: list[list[int]] = []

        consensus = _compute_consensus(aligned_sequences)

        assert consensus == []

    def test_compute_consensus_single_sequence(self) -> None:
        """Test consensus with single sequence."""
        aligned_sequences = [[1, 2, 3, 4, 5]]

        consensus = _compute_consensus(aligned_sequences)

        assert consensus == [1, 2, 3, 4, 5]

    def test_compute_consensus_with_gaps(self) -> None:
        """Test consensus ignores gaps when possible."""
        aligned_sequences = [
            [1, -1, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]

        consensus = _compute_consensus(aligned_sequences)

        # Should pick most common non-gap value
        assert consensus[0] == 1
        assert consensus[1] == 2  # Most common (2 occurrences)
        assert consensus[2] == 3

    def test_compute_consensus_all_gaps_position(self) -> None:
        """Test consensus when position has all gaps."""
        aligned_sequences = [
            [1, -1, 3],
            [1, -1, 3],
            [1, -1, 3],
        ]

        consensus = _compute_consensus(aligned_sequences)

        # All gaps at position 1
        assert consensus[1] == -1

    def test_compute_consensus_varying_lengths(self) -> None:
        """Test consensus with sequences of varying lengths."""
        aligned_sequences = [
            [1, 2, 3],
            [1, 2],
            [1, 2, 3, 4],
        ]

        consensus = _compute_consensus(aligned_sequences)

        # Should handle varying lengths
        assert len(consensus) == 4
        assert consensus[0] == 1
        assert consensus[1] == 2

    def test_insert_gaps_equal_length(self) -> None:
        """Test gap insertion with equal length template."""
        sequence = [1, 2, 3]
        template = [1, 2, 3]

        result = _insert_gaps_from_alignment(sequence, template)

        assert result == [1, 2, 3]

    def test_insert_gaps_template_has_gaps(self) -> None:
        """Test gap insertion when template has gaps."""
        sequence = [1, 2, 3]
        template = [-1, 1, -1, 2, 3]

        result = _insert_gaps_from_alignment(sequence, template)

        assert result == [-1, 1, -1, 2, 3]
        assert result.count(-1) == 2

    def test_insert_gaps_sequence_too_short(self) -> None:
        """Test gap insertion when sequence is shorter than template."""
        sequence = [1, 2]
        template = [1, 1, 1, 1, 1]

        result = _insert_gaps_from_alignment(sequence, template)

        # Should pad with gaps when sequence runs out
        assert len(result) == 5
        assert -1 in result

    def test_insert_gaps_empty_sequence(self) -> None:
        """Test gap insertion with empty sequence."""
        sequence: list[int] = []
        template = [1, 2, 3]

        result = _insert_gaps_from_alignment(sequence, template)

        # All gaps in result
        assert result == [-1, -1, -1]

    def test_insert_gaps_empty_template(self) -> None:
        """Test gap insertion with empty template."""
        sequence = [1, 2, 3]
        template: list[int] = []

        result = _insert_gaps_from_alignment(sequence, template)

        assert result == []

    def test_align_global_large_sequences(self) -> None:
        """Test global alignment with large sequences."""
        seq_a = bytes(range(256)) * 2
        seq_b = bytes(range(256)) * 2

        result = align_global(seq_a, seq_b)

        assert result.similarity == 1.0
        assert result.gaps == 0

    def test_align_local_repeated_pattern(self) -> None:
        """Test local alignment finds best match in repeated pattern."""
        seq_a = b"ABCABCABC"
        seq_b = b"XYZABCXYZ"

        result = align_local(seq_a, seq_b)

        # Should find one of the ABC patterns
        assert result.score > 0
        assert len(result.aligned_a) >= 3

    def test_find_conserved_regions_with_varying_positions(self) -> None:
        """Test conserved regions with non-uniform conservation."""
        sequences = [
            [1, 1, 1, 2, 2, 2, 3, 3, 3],
            [1, 1, 1, 2, 2, 2, 4, 4, 4],
            [1, 1, 1, 2, 2, 2, 5, 5, 5],
        ]

        regions = find_conserved_regions(sequences, min_conservation=0.9, min_length=3)

        # First 6 positions (two groups of 3) should be conserved
        assert len(regions) >= 1
        # Should find contiguous region from 0-6
        assert (0, 6) in regions or ((0, 3) in regions and (3, 6) in regions)

    def test_find_variable_regions_boundary_cases(self) -> None:
        """Test variable regions at sequence boundaries."""
        sequences = [
            [99, 1, 1, 1, 88],
            [77, 1, 1, 1, 66],
            [55, 1, 1, 1, 44],
        ]

        regions = find_variable_regions(sequences, max_conservation=0.5, min_length=1)

        # First and last positions are variable
        assert (0, 1) in regions
        assert (4, 5) in regions


@pytest.mark.unit
@pytest.mark.inference
class TestAlignmentArrayInputs:
    """Test alignment functions with numpy array inputs."""

    def test_align_global_numpy_int_arrays(self) -> None:
        """Test global alignment with integer numpy arrays."""
        seq_a = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        seq_b = np.array([1, 2, 9, 4, 5], dtype=np.uint8)

        result = align_global(seq_a, seq_b)

        assert result.similarity < 1.0  # Due to mismatch at position 2
        assert result.identity < 1.0

    def test_align_local_numpy_arrays(self) -> None:
        """Test local alignment with numpy arrays."""
        seq_a = np.array([10, 20, 30, 40, 50], dtype=np.uint8)
        seq_b = np.array([5, 15, 30, 40, 50, 60], dtype=np.uint8)

        result = align_local(seq_a, seq_b)

        # Should find matching region [30, 40, 50]
        assert result.score > 0

    def test_align_multiple_mixed_types(self) -> None:
        """Test multiple alignment with mixed bytes and arrays."""
        sequences = [
            b"ABC",
            np.array([65, 66, 67], dtype=np.uint8),
            b"ABC",
        ]

        result = align_multiple(sequences)

        # All should produce same result (ABC = [65, 66, 67])
        assert len(result) == 3
        # All sequences should be aligned to same length
        assert len({len(seq) for seq in result}) == 1
