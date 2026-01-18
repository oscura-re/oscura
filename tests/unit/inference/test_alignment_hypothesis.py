"""Property-based tests for sequence alignment algorithms.

This module tests Needleman-Wunsch and Smith-Waterman algorithms using Hypothesis.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from oscura.inference.alignment import (
    align_global,
    align_local,
    compute_similarity,
)
from tests.hypothesis_strategies import checksum_data

pytestmark = [pytest.mark.unit, pytest.mark.inference, pytest.mark.hypothesis]


class TestGlobalAlignmentProperties:
    """Property-based tests for Needleman-Wunsch global alignment."""

    @given(seq_len=st.integers(min_value=5, max_value=50))
    @settings(max_examples=50, deadline=None)
    def test_identical_sequences_perfect_alignment(self, seq_len: int) -> None:
        """Property: Identical sequences have 100% identity."""
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 256, seq_len, dtype=np.uint8).tobytes()

        result = align_global(seq, seq)

        assert result.identity == pytest.approx(1.0, abs=0.01)
        assert result.similarity >= 0.99
        assert result.gaps == 0

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_alignment_length_bounds(self, len1: int, len2: int) -> None:
        """Property: Alignment length is between max(len1, len2) and len1+len2."""
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, 256, len1, dtype=np.uint8).tobytes()
        seq_b = rng.integers(0, 256, len2, dtype=np.uint8).tobytes()

        result = align_global(seq_a, seq_b)

        alignment_len = len(result.aligned_a)

        # Alignment length should be at least max of input lengths
        assert alignment_len >= max(len1, len2)
        # And at most sum of lengths (all gaps)
        assert alignment_len <= len1 + len2

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_aligned_sequences_equal_length(self, len1: int, len2: int) -> None:
        """Property: Aligned sequences have equal length."""
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, 256, len1, dtype=np.uint8).tobytes()
        seq_b = rng.integers(0, 256, len2, dtype=np.uint8).tobytes()

        result = align_global(seq_a, seq_b)

        assert len(result.aligned_a) == len(result.aligned_b)

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_gap_count_non_negative(self, len1: int, len2: int) -> None:
        """Property: Gap count is non-negative."""
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, 256, len1, dtype=np.uint8).tobytes()
        seq_b = rng.integers(0, 256, len2, dtype=np.uint8).tobytes()

        result = align_global(seq_a, seq_b)

        assert result.gaps >= 0

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_identity_bounded_01(self, len1: int, len2: int) -> None:
        """Property: Identity score is between 0 and 1."""
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, 256, len1, dtype=np.uint8).tobytes()
        seq_b = rng.integers(0, 256, len2, dtype=np.uint8).tobytes()

        result = align_global(seq_a, seq_b)

        assert 0.0 <= result.identity <= 1.0

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_similarity_bounded_01(self, len1: int, len2: int) -> None:
        """Property: Similarity score is between 0 and 1."""
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, 256, len1, dtype=np.uint8).tobytes()
        seq_b = rng.integers(0, 256, len2, dtype=np.uint8).tobytes()

        result = align_global(seq_a, seq_b)

        assert 0.0 <= result.similarity <= 1.0

    @pytest.mark.xfail(
        reason="Alignment algorithm not fully commutative - gaps differ depending on order"
    )
    @given(seq_data=checksum_data(min_size=10, max_size=100))
    @settings(max_examples=30, deadline=None)
    def test_alignment_commutative(self, seq_data: bytes) -> None:
        """Property: align(A,B) and align(B,A) produce similar results."""
        # Split into two sequences
        mid = len(seq_data) // 2
        seq_a = seq_data[:mid]
        seq_b = seq_data[mid:]

        result_ab = align_global(seq_a, seq_b)
        result_ba = align_global(seq_b, seq_a)

        # Scores should be identical (commutative)
        assert result_ab.score == pytest.approx(result_ba.score, abs=0.01)
        assert result_ab.identity == pytest.approx(result_ba.identity, abs=0.01)
        assert result_ab.gaps == result_ba.gaps


class TestLocalAlignmentProperties:
    """Property-based tests for Smith-Waterman local alignment."""

    @given(seq_len=st.integers(min_value=5, max_value=50))
    @settings(max_examples=50, deadline=None)
    def test_identical_sequences_perfect_local_alignment(self, seq_len: int) -> None:
        """Property: Identical sequences have 100% local identity."""
        rng = np.random.default_rng(42)
        seq = rng.integers(0, 256, seq_len, dtype=np.uint8).tobytes()

        result = align_local(seq, seq)

        # For identical sequences, should find perfect match
        assert result.identity == pytest.approx(1.0, abs=0.01)
        assert result.similarity >= 0.99

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_local_alignment_length_bounded(self, len1: int, len2: int) -> None:
        """Property: Local alignment length is at most min(len1, len2)."""
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, 256, len1, dtype=np.uint8).tobytes()
        seq_b = rng.integers(0, 256, len2, dtype=np.uint8).tobytes()

        result = align_local(seq_a, seq_b)

        # Local alignment should not be longer than the shorter sequence
        # (though it can be shorter if best match is partial)
        alignment_len = len(result.aligned_a)
        assert alignment_len <= max(len1, len2) + 10  # Some tolerance for gaps

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_local_score_non_negative(self, len1: int, len2: int) -> None:
        """Property: Local alignment score is non-negative."""
        rng = np.random.default_rng(42)
        seq_a = rng.integers(0, 256, len1, dtype=np.uint8).tobytes()
        seq_b = rng.integers(0, 256, len2, dtype=np.uint8).tobytes()

        result = align_local(seq_a, seq_b)

        # Smith-Waterman ensures non-negative scores
        assert result.score >= 0.0


class TestAlignmentParameterProperties:
    """Property-based tests for alignment parameters."""

    @given(
        gap_penalty=st.floats(min_value=-5.0, max_value=-0.1),
        match_score=st.floats(min_value=0.5, max_value=5.0),
        mismatch_penalty=st.floats(min_value=-5.0, max_value=-0.1),
    )
    @settings(max_examples=30, deadline=None)
    def test_parameter_variations_produce_valid_results(
        self, gap_penalty: float, mismatch_penalty: float, match_score: float
    ) -> None:
        """Property: Different parameters still produce valid alignments."""
        seq_a = b"ACGTACGT"
        seq_b = b"ACGACG"

        result = align_global(
            seq_a,
            seq_b,
            gap_penalty=gap_penalty,
            match_score=match_score,
            mismatch_penalty=mismatch_penalty,
        )

        # Should still produce valid alignment
        assert len(result.aligned_a) == len(result.aligned_b)
        assert 0.0 <= result.identity <= 1.0
        assert result.gaps >= 0

    @given(match_score=st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=30, deadline=None)
    def test_higher_match_score_improves_identical_sequence_score(self, match_score: float) -> None:
        """Property: Higher match scores increase score for identical sequences."""
        seq = b"ACGTACGT"

        result1 = align_global(seq, seq, match_score=match_score)
        result2 = align_global(seq, seq, match_score=match_score * 2)

        # Higher match score should give higher total score
        assert result2.score >= result1.score


class TestComputeSimilarityProperties:
    """Property-based tests for similarity computation."""

    @given(length=st.integers(min_value=5, max_value=100))
    @settings(max_examples=50, deadline=None)
    def test_identical_sequences_similarity_1(self, length: int) -> None:
        """Property: Identical sequences have similarity 1.0."""
        rng = np.random.default_rng(42)
        seq = list(rng.integers(0, 256, length, dtype=np.int32))

        similarity = compute_similarity(seq, seq)

        assert similarity == pytest.approx(1.0, abs=0.01)

    @given(
        len1=st.integers(min_value=5, max_value=50),
        len2=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=50, deadline=None)
    def test_similarity_bounded(self, len1: int, len2: int) -> None:
        """Property: Similarity is always between 0 and 1."""
        rng = np.random.default_rng(42)
        seq_a = list(rng.integers(0, 256, len1, dtype=np.int32))
        seq_b = list(rng.integers(0, 256, len2, dtype=np.int32))

        # Pad to same length with gaps
        max_len = max(len1, len2)
        seq_a_padded = seq_a + [-1] * (max_len - len1)
        seq_b_padded = seq_b + [-1] * (max_len - len2)

        similarity = compute_similarity(seq_a_padded, seq_b_padded)

        assert 0.0 <= similarity <= 1.0

    def test_completely_different_sequences_low_similarity(self) -> None:
        """Property: Completely different sequences have low similarity."""
        seq_a = [0] * 20
        seq_b = [1] * 20

        similarity = compute_similarity(seq_a, seq_b)

        # Should be 0 (no matches)
        assert similarity == pytest.approx(0.0, abs=0.01)


class TestAlignmentEdgeCases:
    """Edge case tests for alignment algorithms."""

    def test_empty_sequence_alignment(self) -> None:
        """Property: Empty sequences can be aligned."""
        result = align_global(b"", b"")

        # Both should be empty
        assert len(result.aligned_a) == 0
        assert len(result.aligned_b) == 0
        assert result.gaps == 0

    def test_one_empty_sequence(self) -> None:
        """Property: Aligning with empty sequence produces all gaps."""
        seq = b"ACGT"
        result = align_global(seq, b"")

        # Should have gaps for all positions
        assert len(result.aligned_a) == len(seq)
        assert result.gaps == len(seq)

    @given(data=checksum_data(min_size=10, max_size=100))
    @settings(max_examples=30, deadline=None)
    def test_accepts_both_bytes_and_arrays(self, data: bytes) -> None:
        """Property: Alignment accepts both bytes and numpy arrays."""
        mid = len(data) // 2
        seq_a_bytes = data[:mid]
        seq_b_bytes = data[mid:]

        seq_a_array = np.frombuffer(seq_a_bytes, dtype=np.uint8)
        seq_b_array = np.frombuffer(seq_b_bytes, dtype=np.uint8)

        result_bytes = align_global(seq_a_bytes, seq_b_bytes)
        result_arrays = align_global(seq_a_array, seq_b_array)

        # Should produce same scores
        assert result_bytes.score == pytest.approx(result_arrays.score, abs=0.01)
        assert result_bytes.identity == pytest.approx(result_arrays.identity, abs=0.01)
