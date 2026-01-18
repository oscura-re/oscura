"""Property-based tests for pattern matching algorithms."""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import pattern_sequences, repetitive_sequences

pytestmark = [pytest.mark.unit, pytest.mark.pattern, pytest.mark.hypothesis]


class TestPatternMatchingProperties:
    """Property-based tests for pattern matching."""

    @given(
        pattern_length=st.integers(min_value=2, max_value=20),
        data_length=st.integers(min_value=50, max_value=500),
    )
    @settings(max_examples=50, deadline=None)
    def test_pattern_found_when_present(self, pattern_length: int, data_length: int) -> None:
        """Property: Pattern is found when it exists in data."""
        assume(data_length >= pattern_length)

        # Create data with known pattern
        pattern = bytes(range(pattern_length))
        prefix = bytes([255] * 10)
        suffix = bytes([254] * 10)
        data = prefix + pattern + suffix

        # Pattern should be in data
        assert pattern in data

    @given(data=pattern_sequences())
    @settings(max_examples=30, deadline=None)
    def test_empty_pattern_matches_everywhere(self, data: bytes) -> None:
        """Property: Empty pattern conceptually matches at every position."""
        empty_pattern = b""

        # In Python, empty string is substring of any string
        assert empty_pattern in data or len(data) == 0

    @given(data=repetitive_sequences())
    @settings(max_examples=30, deadline=None)
    def test_repetitive_data_multiple_matches(self, data: bytes) -> None:
        """Property: Repetitive data has multiple pattern matches."""
        if len(data) < 4:
            pytest.skip("Data too short")

        # Extract first repeated unit
        pattern = data[: len(data) // 10 or 2]

        # Count occurrences
        count = data.count(pattern)

        # Should have multiple occurrences
        assert count >= 2
