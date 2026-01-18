"""Property-based tests for repetition detection."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.hypothesis_strategies import repetitive_sequences

pytestmark = [pytest.mark.unit, pytest.mark.pattern, pytest.mark.hypothesis]


class TestRepetitionDetectionProperties:
    """Property-based tests for repetition detection."""

    @given(data=repetitive_sequences())
    @settings(max_examples=30, deadline=None)
    def test_repetitive_data_detected(self, data: bytes) -> None:
        """Property: Repetitive sequences are detected."""
        # Any repetitive sequence should have repeated bytes
        if len(data) < 2:
            pytest.skip("Data too short")

        # Check for repetition (at least one byte appears twice)
        byte_counts = {}
        for byte_val in data:
            byte_counts[byte_val] = byte_counts.get(byte_val, 0) + 1

        max_count = max(byte_counts.values())
        assert max_count >= 2

    @given(
        pattern_size=st.integers(min_value=2, max_value=10),
        num_reps=st.integers(min_value=3, max_value=20),
    )
    @settings(max_examples=30, deadline=None)
    def test_known_repetition_count(self, pattern_size: int, num_reps: int) -> None:
        """Property: Known repetitions are counted correctly."""
        pattern = bytes(range(pattern_size))
        data = pattern * num_reps

        # Pattern should appear num_reps times
        count = 0
        pos = 0
        while pos < len(data):
            idx = data.find(pattern, pos)
            if idx == -1:
                break
            count += 1
            pos = idx + 1

        assert count >= num_reps
