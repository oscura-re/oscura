"""Tests for utils/geometry.py - Geometric utility functions.

Tests:
- generate_leader_line: Create L-shaped orthogonal leader lines
- Various anchor and label positions
- Edge cases
"""

from oscura.utils import geometry


class TestGenerateLeaderLine:
    """Test generate_leader_line function."""

    def test_basic_leader_line(self) -> None:
        """Test basic leader line generation."""
        anchor = (0.0, 0.0)
        label = (10.0, 5.0)

        result = geometry.generate_leader_line(anchor, label)

        # Should create L-shape: anchor -> (label_x, anchor_y) -> label
        assert len(result) == 3
        assert result[0] == (0.0, 0.0)  # Start at anchor
        assert result[1] == (10.0, 0.0)  # Horizontal to label x
        assert result[2] == (10.0, 5.0)  # Vertical to label

    def test_right_upward(self) -> None:
        """Test leader line going right and up."""
        anchor = (0.0, 0.0)
        label = (5.0, 3.0)

        result = geometry.generate_leader_line(anchor, label)

        assert result == [(0.0, 0.0), (5.0, 0.0), (5.0, 3.0)]

    def test_same_point(self) -> None:
        """Test when anchor and label are the same point."""
        anchor = (5.0, 5.0)
        label = (5.0, 5.0)

        result = geometry.generate_leader_line(anchor, label)

        # All points are the same
        assert len(result) == 3
        assert result[0] == (5.0, 5.0)
        assert result[1] == (5.0, 5.0)
        assert result[2] == (5.0, 5.0)

    def test_negative_coordinates(self) -> None:
        """Test with negative coordinates."""
        anchor = (-5.0, -3.0)
        label = (2.0, 4.0)

        result = geometry.generate_leader_line(anchor, label)

        assert result == [(-5.0, -3.0), (2.0, -3.0), (2.0, 4.0)]
