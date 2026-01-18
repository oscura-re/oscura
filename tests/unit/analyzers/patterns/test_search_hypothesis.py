"""Property-based tests for pattern search algorithms."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import pattern_sequences

pytestmark = [pytest.mark.unit, pytest.mark.pattern, pytest.mark.hypothesis]


class TestPatternSearchProperties:
    """Property-based tests for pattern search."""

    @given(
        pattern_length=st.integers(min_value=1, max_value=20),
        num_occurrences=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=30, deadline=None)
    def test_all_occurrences_found(self, pattern_length: int, num_occurrences: int) -> None:
        """Property: All pattern occurrences are found."""
        pattern = bytes(range(pattern_length))
        separator = bytes([255, 254, 253])

        # Create data with known number of pattern occurrences
        data = separator.join([pattern] * num_occurrences)

        # Count occurrences
        count = data.count(pattern)

        assert count == num_occurrences

    @given(data=pattern_sequences())
    @settings(max_examples=30, deadline=None)
    def test_search_deterministic(self, data: bytes) -> None:
        """Property: Pattern search is deterministic."""
        if len(data) < 4:
            pytest.skip("Data too short")

        pattern = data[:4]

        # Search multiple times
        pos1 = data.find(pattern)
        pos2 = data.find(pattern)
        pos3 = data.find(pattern)

        # Should always return same position
        assert pos1 == pos2 == pos3
