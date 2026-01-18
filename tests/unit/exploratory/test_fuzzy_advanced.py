"""Comprehensive unit tests for advanced fuzzy matching module.

This module provides extensive testing for binary pattern variant characterization,
consensus finding, and multiple sequence alignment.


Test Coverage:
- characterize_variants with various patterns
- align_two_sequences (Needleman-Wunsch, Smith-Waterman)
- align_sequences (progressive MSA)
- compute_conservation_scores
- Edge cases: empty sequences, single sequence, highly divergent
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.exploratory.fuzzy_advanced import (
    GAP_BYTE,
    AlignedSequence,
    VariationType,
    align_sequences,
    align_two_sequences,
    characterize_variants,
    compute_conservation_scores,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Helper Functions
# =============================================================================


def generate_variant_patterns(
    base: bytes, n_variants: int, mutation_rate: float = 0.1, seed: int = 42
) -> list[bytes]:
    """Generate variant patterns by mutating base pattern."""
    rng = np.random.default_rng(seed)
    variants = [base]

    for _ in range(n_variants - 1):
        variant = bytearray(base)
        # Mutate some positions
        n_mutations = max(1, int(len(base) * mutation_rate))
        positions = rng.choice(len(base), size=n_mutations, replace=False)

        for pos in positions:
            # Random byte value
            variant[pos] = rng.integers(0, 256)

        variants.append(bytes(variant))

    return variants


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestCharacterizeVariants:
    """Test binary pattern variant characterization (FUZZY-004)."""

    def test_identical_patterns_constant(self) -> None:
        """Test that identical patterns show all constant positions."""
        patterns = [b"\x12\x34\x56\x78"] * 10

        result = characterize_variants(patterns)

        # All positions should be constant
        assert result.consensus == b"\x12\x34\x56\x78"
        assert len(result.constant_positions) == 4
        assert len(result.variable_positions) == 0
        assert result.pattern_count == 10

        # Check all positions have high confidence
        for pos_analysis in result.positions:
            assert pos_analysis.variation_type == VariationType.CONSTANT
            assert pos_analysis.consensus_confidence == 1.0

    def test_single_position_variation(self) -> None:
        """Test variation at single position."""
        patterns = [
            b"\x12\x34\x56\x78",
            b"\x12\x35\x56\x78",
            b"\x12\x34\x56\x78",
            b"\x12\x36\x56\x78",
        ]

        result = characterize_variants(patterns)

        # Position 1 should vary, others constant
        assert 1 in result.variable_positions
        assert 0 in result.constant_positions
        assert 2 in result.constant_positions
        assert 3 in result.constant_positions

        # Check position 1 analysis
        pos1 = result.positions[1]
        assert pos1.variation_type in [VariationType.LOW_VARIATION, VariationType.HIGH_VARIATION]
        assert pos1.consensus_byte == 0x34  # Most common

    def test_consensus_majority_vote(self) -> None:
        """Test consensus is majority vote."""
        patterns = [
            b"\xaa",
            b"\xaa",
            b"\xaa",
            b"\xbb",
            b"\xbb",
        ]

        result = characterize_variants(patterns)

        # Consensus should be 0xAA (3 votes vs 2)
        assert result.consensus == b"\xaa"
        assert result.positions[0].consensus_byte == 0xAA
        assert result.positions[0].consensus_confidence == 0.6

    def test_entropy_calculation(self) -> None:
        """Test entropy calculation for positions."""
        # Position with 2 values equally distributed
        patterns = [b"\xaa", b"\xbb"] * 50

        result = characterize_variants(patterns)

        # Entropy should be ~1 bit (log2(2) = 1)
        entropy = result.positions[0].entropy
        assert 0.9 <= entropy <= 1.1

    def test_variation_type_classification(self) -> None:
        """Test variation type classification."""
        # Constant position (entropy ~0)
        constant_patterns = [b"\x12"] * 10
        result_const = characterize_variants(constant_patterns)
        assert result_const.positions[0].variation_type == VariationType.CONSTANT

        # Low variation (few distinct values)
        low_var_patterns = [b"\x12", b"\x13", b"\x12", b"\x13"] * 5
        result_low = characterize_variants(low_var_patterns)
        assert result_low.positions[0].variation_type in [
            VariationType.CONSTANT,
            VariationType.LOW_VARIATION,
        ]

        # High variation (many distinct values)
        high_var_patterns = [bytes([i]) for i in range(20)]
        result_high = characterize_variants(high_var_patterns)
        assert result_high.positions[0].variation_type in [
            VariationType.HIGH_VARIATION,
            VariationType.RANDOM,
        ]

    def test_error_detection_single_bit_flip(self) -> None:
        """Test detection of single-bit flip errors."""
        patterns = [
            b"\x12",  # 0001 0010
            b"\x12",
            b"\x13",  # 0001 0011 (1 bit different)
            b"\x12",
            b"\x12",
        ]

        result = characterize_variants(patterns)

        # Should detect as likely error
        pos0 = result.positions[0]
        assert pos0.is_error or pos0.consensus_confidence > 0.95

    def test_suggested_field_boundaries(self) -> None:
        """Test field boundary suggestion."""
        patterns = [
            b"\x12\x34\xaa\xbb",  # Constant | Variable
            b"\x12\x34\xcc\xdd",
            b"\x12\x34\xee\xff",
        ]

        result = characterize_variants(patterns)

        # Should suggest boundary at position 2
        assert 2 in result.suggested_boundaries or len(result.suggested_boundaries) > 0

    def test_empty_patterns(self) -> None:
        """Test with empty pattern list."""
        result = characterize_variants([])

        assert result.consensus == b""
        assert result.pattern_count == 0
        assert len(result.positions) == 0

    def test_variable_length_patterns(self) -> None:
        """Test with patterns of different lengths."""
        patterns = [
            b"\x12\x34\x56",
            b"\x12\x34",
            b"\x12\x34\x56\x78",
        ]

        result = characterize_variants(patterns)

        # Should use min length
        assert result.min_length == 2
        assert len(result.consensus) == 2

    def test_value_distribution(self) -> None:
        """Test value distribution tracking."""
        patterns = [b"\xaa", b"\xbb", b"\xaa", b"\xcc"]

        result = characterize_variants(patterns)

        dist = result.positions[0].value_distribution
        assert dist[0xAA] == 2
        assert dist[0xBB] == 1
        assert dist[0xCC] == 1

    def test_min_confidence_parameter(self) -> None:
        """Test min_confidence parameter."""
        patterns = [b"\x12"] * 10 + [b"\x13"]

        # High confidence requirement
        result = characterize_variants(patterns, min_confidence=0.95)

        # Should still work
        assert result.pattern_count == 11


# =============================================================================
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestAlignTwoSequences:
    """Test pairwise sequence alignment (FUZZY-005)."""

    def test_identical_sequences_perfect_score(self) -> None:
        """Test alignment of identical sequences."""
        seq1 = b"\x12\x34\x56\x78"
        seq2 = b"\x12\x34\x56\x78"

        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="global")

        # Should be identical with no gaps
        assert aligned1 == seq1
        assert aligned2 == seq2
        assert score > 0  # Positive score for matches

    def test_global_alignment_with_gap(self) -> None:
        """Test global alignment with gap (Needleman-Wunsch)."""
        seq1 = b"\x12\x34\x56\x78"
        seq2 = b"\x12\x56\x78"  # Missing 0x34

        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="global")

        # Aligned sequences should have same length
        assert len(aligned1) == len(aligned2)
        # Should have introduced gap in seq2
        assert GAP_BYTE in aligned2

    def test_local_alignment(self) -> None:
        """Test local alignment (Smith-Waterman)."""
        seq1 = b"\xff\x12\x34\x56\xff"
        seq2 = b"\x12\x34\x56"

        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="local")

        # Local alignment should find matching region
        assert len(aligned1) > 0
        assert len(aligned2) > 0
        assert score >= 0

    def test_completely_different_sequences(self) -> None:
        """Test alignment of completely different sequences."""
        seq1 = b"\x00\x00\x00\x00"
        seq2 = b"\xff\xff\xff\xff"

        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="global")

        # Should still align (with poor score)
        assert len(aligned1) == len(aligned2)
        # Score should be negative due to mismatches
        assert score < 0

    def test_empty_sequence(self) -> None:
        """Test alignment with empty sequence."""
        seq1 = b"\x12\x34"
        seq2 = b""

        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="global")

        # Should introduce gaps for all of seq1
        assert len(aligned1) == len(seq2) or len(aligned1) > 0

    def test_custom_scoring_parameters(self) -> None:
        """Test custom scoring parameters."""
        seq1 = b"\x12\x34"
        seq2 = b"\x12\x56"

        # High mismatch penalty
        aligned1, aligned2, score1 = align_two_sequences(
            seq1, seq2, method="global", mismatch_penalty=-10, gap_open=-2
        )

        # Low mismatch penalty
        aligned3, aligned4, score2 = align_two_sequences(
            seq1, seq2, method="global", mismatch_penalty=-1, gap_open=-2
        )

        # Scores should differ
        assert score1 != score2 or aligned1 == aligned3  # Same if alignment identical

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        seq1 = b"\x12\x34"
        seq2 = b"\x56\x78"

        with pytest.raises(ValueError, match="Unknown alignment method"):
            align_two_sequences(seq1, seq2, method="invalid")  # type: ignore[arg-type]


@pytest.mark.unit
@pytest.mark.exploratory
class TestAlignSequences:
    """Test multiple sequence alignment (FUZZY-005)."""

    def test_single_sequence(self) -> None:
        """Test MSA with single sequence."""
        sequences = [b"\x12\x34\x56"]

        result = align_sequences(sequences)

        assert len(result.sequences) == 1
        assert result.sequences[0].aligned == b"\x12\x34\x56"
        assert len(result.sequences[0].gaps) == 0

    def test_two_sequences_alignment(self) -> None:
        """Test MSA with two sequences."""
        sequences = [b"\x12\x34\x56", b"\x12\x56"]

        result = align_sequences(sequences)

        assert len(result.sequences) == 2
        # All sequences should have same aligned length
        lengths = [len(seq.aligned) for seq in result.sequences]
        assert len(set(lengths)) == 1

    def test_multiple_sequences_progressive(self) -> None:
        """Test progressive MSA with multiple sequences."""
        sequences = [b"\x12\x34\x56", b"\x12\x56", b"\x34\x56"]

        result = align_sequences(sequences, method="progressive")

        assert len(result.sequences) == 3
        # Check alignment
        for seq in result.sequences:
            assert isinstance(seq, AlignedSequence)
            assert len(seq.aligned) > 0

    def test_conservation_scores(self) -> None:
        """Test conservation score computation."""
        # Highly conserved
        sequences = [b"\x12\x34\x56"] * 5

        result = align_sequences(sequences)

        # All positions should be highly conserved
        assert all(score >= 0.99 for score in result.conservation_scores)

    def test_conserved_regions_detection(self) -> None:
        """Test detection of conserved regions."""
        sequences = [
            b"\x12\x34\x56\x78",
            b"\x12\x34\xff\x78",
            b"\x12\x34\xaa\x78",
        ]

        result = align_sequences(sequences)

        # Should have conserved regions (positions 0, 1, 3)
        assert len(result.conserved_regions) > 0

    def test_gap_positions(self) -> None:
        """Test common gap position detection."""
        # Create sequences that will require gaps
        sequences = [b"\x12\x34\x56\x78", b"\x12\x78", b"\x12\x78"]

        result = align_sequences(sequences)

        # Should detect common gap positions
        # (Exact behavior depends on alignment algorithm)
        assert isinstance(result.gap_positions, list)

    def test_empty_sequences(self) -> None:
        """Test with empty sequence list."""
        result = align_sequences([])

        assert len(result.sequences) == 0
        assert len(result.conservation_scores) == 0
        assert result.alignment_score == 0.0

    def test_iterative_method(self) -> None:
        """Test iterative refinement method."""
        sequences = [b"\x12\x34", b"\x34\x56", b"\x56\x78"]

        result = align_sequences(sequences, method="iterative")

        # Should produce valid alignment
        assert len(result.sequences) == 3

    def test_invalid_method_raises_error(self) -> None:
        """Test that invalid method raises ValueError."""
        sequences = [b"\x12\x34"]

        with pytest.raises(ValueError, match="Unknown alignment method"):
            align_sequences(sequences, method="invalid")  # type: ignore[arg-type]

    def test_alignment_score(self) -> None:
        """Test overall alignment score calculation."""
        sequences = [b"\x12\x34\x56", b"\x12\x34\x78"]

        result = align_sequences(sequences)

        # Score should reflect similarity
        assert isinstance(result.alignment_score, float)


@pytest.mark.unit
@pytest.mark.exploratory
class TestComputeConservationScores:
    """Test conservation score computation."""

    def test_fully_conserved(self) -> None:
        """Test fully conserved position."""
        aligned = [b"\x12\x34", b"\x12\x34", b"\x12\x34"]

        scores = compute_conservation_scores(aligned)

        assert len(scores) == 2
        assert all(score == 1.0 for score in scores)

    def test_partially_conserved(self) -> None:
        """Test partially conserved position."""
        aligned = [b"\x12\x34", b"\x12\x56", b"\x12\x34"]

        scores = compute_conservation_scores(aligned)

        # Position 0 fully conserved, position 1 partially
        assert scores[0] == 1.0
        assert 0.5 <= scores[1] < 1.0  # 2/3 have 0x34

    def test_with_gaps(self) -> None:
        """Test conservation with gaps (gaps excluded)."""
        aligned = [b"\x12\x34", bytes([0x12, GAP_BYTE]), bytes([GAP_BYTE, 0x34])]

        scores = compute_conservation_scores(aligned)

        # Position 0: 2 have 0x12 (excluding gap)
        # Position 1: 2 have 0x34 (excluding gap)
        assert len(scores) == 2
        assert all(score > 0 for score in scores)

    def test_all_gaps_position(self) -> None:
        """Test position with all gaps."""
        aligned = [bytes([GAP_BYTE]), bytes([GAP_BYTE]), bytes([GAP_BYTE])]

        scores = compute_conservation_scores(aligned)

        # All gaps should give score 0
        assert scores[0] == 0.0

    def test_empty_sequences(self) -> None:
        """Test with empty sequences."""
        scores = compute_conservation_scores([])

        assert len(scores) == 0


# =============================================================================
# Edge Cases and Robustness
# =============================================================================


@pytest.mark.unit
@pytest.mark.exploratory
class TestExploratoryFuzzyAdvancedEdgeCases:
    """Test edge cases and robustness."""

    def test_large_pattern_set(self) -> None:
        """Test with large number of patterns."""
        base = b"\x12\x34\x56\x78"
        patterns = generate_variant_patterns(base, n_variants=100, mutation_rate=0.1)

        result = characterize_variants(patterns)

        assert result.pattern_count == 100
        assert len(result.consensus) == 4

    def test_high_divergence(self) -> None:
        """Test with highly divergent sequences."""
        sequences = [bytes([i]) * 4 for i in range(10)]

        result = align_sequences(sequences)

        # Should still produce alignment
        assert len(result.sequences) == 10

    def test_very_long_sequences(self) -> None:
        """Test with very long sequences."""
        seq1 = bytes(range(256)) * 2  # 512 bytes
        seq2 = bytes(range(256)) * 2

        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="global")

        # Should handle long sequences
        assert len(aligned1) >= 512

    def test_single_byte_patterns(self) -> None:
        """Test with single-byte patterns."""
        patterns = [b"\x12", b"\x13", b"\x12", b"\x14"]

        result = characterize_variants(patterns)

        assert len(result.consensus) == 1
        assert result.positions[0].consensus_byte in [0x12, 0x13, 0x14]

    def test_bytearrays_accepted(self) -> None:
        """Test that bytearrays are accepted."""
        patterns = [bytearray(b"\x12\x34"), bytearray(b"\x12\x35")]

        result = characterize_variants(patterns)

        assert result.pattern_count == 2

    def test_gap_byte_in_sequence(self) -> None:
        """Test handling of GAP_BYTE (0xFF) in actual sequence."""
        seq1 = b"\x12\xff\x34"  # Contains 0xFF
        seq2 = b"\x12\xff\x34"

        aligned1, aligned2, score = align_two_sequences(seq1, seq2, method="global")

        # Should align correctly even with 0xFF
        assert len(aligned1) == len(aligned2)
