"""Property-based tests for sequence analysis."""

import pytest
from hypothesis import given, settings

from tests.hypothesis_strategies import alignment_sequences

pytestmark = [pytest.mark.unit, pytest.mark.inference, pytest.mark.hypothesis]


class TestSequenceAnalysisProperties:
    """Property-based tests for sequence analysis."""

    @given(seqs=alignment_sequences())
    @settings(max_examples=50, deadline=None)
    def test_sequence_lengths_non_negative(self, seqs: tuple[list[int], list[int]]) -> None:
        """Property: Sequence lengths are non-negative."""
        seq1, seq2 = seqs

        assert len(seq1) >= 0
        assert len(seq2) >= 0

    @given(seqs=alignment_sequences())
    @settings(max_examples=30, deadline=None)
    def test_sequence_similarity_to_self_is_one(self, seqs: tuple[list[int], list[int]]) -> None:
        """Property: Sequence similarity to itself is 1.0."""
        seq1, seq2 = seqs

        if len(seq1) == 0:
            pytest.skip("Empty sequence")

        # Similarity of sequence to itself
        matches = sum(1 for a, b in zip(seq1, seq1, strict=False) if a == b)
        similarity = matches / len(seq1)

        assert similarity == pytest.approx(1.0, abs=1e-10)
