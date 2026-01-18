"""Property-based tests for n-gram analysis.

Tests n-gram extraction, frequency analysis, and pattern detection.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import ngram_sequences

pytestmark = [pytest.mark.unit, pytest.mark.statistical, pytest.mark.hypothesis]


class TestNGramExtractionProperties:
    """Property-based tests for n-gram extraction."""

    @given(
        sequence_length=st.integers(min_value=20, max_value=200),
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_ngram_count_bounded(self, sequence_length: int, n: int) -> None:
        """Property: Number of n-grams is sequence_length - n + 1."""
        assume(sequence_length >= n)

        try:
            from oscura.analyzers.statistical.ngrams import extract_ngrams
        except ImportError:
            pytest.skip("ngrams module not available")

        # Create simple sequence
        sequence = bytes(range(sequence_length % 256))

        ngrams = extract_ngrams(sequence, n=n)

        expected_count = sequence_length - n + 1
        assert len(ngrams) == expected_count

    @given(
        data=ngram_sequences(),
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50, deadline=None)
    def test_all_ngrams_have_length_n(self, data: bytes, n: int) -> None:
        """Property: All extracted n-grams have length n."""
        assume(len(data) >= n)

        try:
            from oscura.analyzers.statistical.ngrams import extract_ngrams
        except ImportError:
            pytest.skip("ngrams module not available")

        ngrams = extract_ngrams(data, n=n)

        for ngram in ngrams:
            assert len(ngram) == n

    @given(
        sequence_length=st.integers(min_value=10, max_value=100),
        n=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=30, deadline=None)
    def test_ngram_extraction_deterministic(self, sequence_length: int, n: int) -> None:
        """Property: N-gram extraction is deterministic."""
        assume(sequence_length >= n)

        try:
            from oscura.analyzers.statistical.ngrams import extract_ngrams
        except ImportError:
            pytest.skip("ngrams module not available")

        sequence = bytes(range(sequence_length % 256))

        ngrams1 = extract_ngrams(sequence, n=n)
        ngrams2 = extract_ngrams(sequence, n=n)

        assert ngrams1 == ngrams2


class TestNGramFrequencyProperties:
    """Property-based tests for n-gram frequency analysis."""

    @given(
        data=ngram_sequences(),
        n=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_frequency_sum_equals_total_ngrams(self, data: bytes, n: int) -> None:
        """Property: Sum of n-gram frequencies equals total count."""
        assume(len(data) >= n)

        try:
            from oscura.analyzers.statistical.ngrams import (
                ngram_frequencies,
            )
        except ImportError:
            pytest.skip("ngrams module not available")

        frequencies = ngram_frequencies(data, n=n)

        total_count = sum(frequencies.values())
        expected_count = len(data) - n + 1

        assert total_count == expected_count

    @given(
        data=ngram_sequences(),
        n=st.integers(min_value=2, max_value=4),
    )
    @settings(max_examples=30, deadline=None)
    def test_frequency_values_non_negative(self, data: bytes, n: int) -> None:
        """Property: All n-gram frequencies are non-negative."""
        assume(len(data) >= n)

        try:
            from oscura.analyzers.statistical.ngrams import (
                ngram_frequencies,
            )
        except ImportError:
            pytest.skip("ngrams module not available")

        frequencies = ngram_frequencies(data, n=n)

        for count in frequencies.values():
            assert count >= 0


class TestNGramPatternDetectionProperties:
    """Property-based tests for pattern detection using n-grams."""

    @given(
        pattern_length=st.integers(min_value=2, max_value=10),
        num_repetitions=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=30, deadline=None)
    def test_repeated_pattern_detected(self, pattern_length: int, num_repetitions: int) -> None:
        """Property: Repeated patterns are detected in n-gram analysis."""
        try:
            from oscura.analyzers.statistical.ngrams import (
                find_common_ngrams,  # noqa: F401
                ngram_frequencies,
            )
        except ImportError:
            pytest.skip("ngrams module not available")

        # Create data with repeated pattern
        pattern = bytes(range(pattern_length))
        data = pattern * num_repetitions

        # Use pattern length as n
        frequencies = ngram_frequencies(data, n=pattern_length)

        # The pattern should be the most common n-gram
        if frequencies:
            most_common = max(frequencies.items(), key=lambda x: x[1])
            assert most_common[1] >= num_repetitions - pattern_length + 1
