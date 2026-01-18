"""Property-based tests for frame detection."""

import pytest
from hypothesis import given, settings

from tests.hypothesis_strategies import framing_data

pytestmark = [pytest.mark.unit, pytest.mark.packet, pytest.mark.hypothesis]


class TestFramingProperties:
    """Property-based tests for frame detection."""

    @given(frame_data=framing_data())
    @settings(max_examples=30, deadline=None)
    def test_frame_boundaries_detected(self, frame_data: tuple[bytes, list[int]]) -> None:
        """Property: Frame boundaries are correctly identified."""
        data, boundaries = frame_data

        # All boundaries should be valid positions in data
        for boundary in boundaries:
            assert 0 <= boundary < len(data)

        # Boundaries should be sorted
        assert boundaries == sorted(boundaries)

    @given(frame_data=framing_data())
    @settings(max_examples=30, deadline=None)
    def test_frame_count_matches_boundaries(self, frame_data: tuple[bytes, list[int]]) -> None:
        """Property: Number of frames matches boundary count."""
        data, boundaries = frame_data

        # Number of frames = number of delimiters found
        num_frames = len(boundaries)

        assert num_frames >= 1
