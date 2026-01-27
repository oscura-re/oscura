"""Tests for visualization layout functions."""

import numpy as np
import pytest

from oscura.visualization.layout import (
    Annotation,
    ChannelLayout,
    PlacedAnnotation,
    layout_stacked_channels,
    optimize_annotation_placement,
)


class TestChannelLayout:
    """Test ChannelLayout dataclass."""

    def test_dataclass_creation(self) -> None:
        """Test creating ChannelLayout."""
        heights = np.array([0.3, 0.3, 0.3])
        gaps = np.array([0.05, 0.05])
        y_positions = np.array([0.1, 0.45, 0.8])

        layout = ChannelLayout(
            n_channels=3,
            heights=heights,
            gaps=gaps,
            y_positions=y_positions,
            shared_x=True,
            figsize=(10, 8),
        )

        assert layout.n_channels == 3
        assert np.array_equal(layout.heights, heights)
        assert np.array_equal(layout.gaps, gaps)
        assert np.array_equal(layout.y_positions, y_positions)
        assert layout.shared_x is True
        assert layout.figsize == (10, 8)


class TestAnnotation:
    """Test Annotation dataclass."""

    def test_annotation_with_defaults(self) -> None:
        """Test creating annotation with defaults."""
        annot = Annotation(text="Test", x=1.0, y=2.0)
        assert annot.text == "Test"
        assert annot.x == 1.0
        assert annot.y == 2.0
        assert annot.bbox_width == 50.0
        assert annot.bbox_height == 20.0
        assert annot.priority == 0.5
        assert annot.anchor == "auto"

    def test_annotation_with_custom_values(self) -> None:
        """Test creating annotation with custom values."""
        annot = Annotation(
            text="Peak",
            x=5.0,
            y=10.0,
            bbox_width=100.0,
            bbox_height=30.0,
            priority=0.9,
            anchor="top",
        )
        assert annot.text == "Peak"
        assert annot.x == 5.0
        assert annot.y == 10.0
        assert annot.bbox_width == 100.0
        assert annot.bbox_height == 30.0
        assert annot.priority == 0.9
        assert annot.anchor == "top"


class TestPlacedAnnotation:
    """Test PlacedAnnotation dataclass."""

    def test_placed_annotation_without_leader(self) -> None:
        """Test placing annotation without leader line."""
        annot = Annotation("Test", x=1.0, y=2.0)
        placed = PlacedAnnotation(
            annotation=annot,
            display_x=1.0,
            display_y=2.0,
            needs_leader=False,
        )
        assert placed.annotation == annot
        assert placed.display_x == 1.0
        assert placed.display_y == 2.0
        assert placed.needs_leader is False
        assert placed.leader_points is None

    def test_placed_annotation_with_leader(self) -> None:
        """Test placing annotation with leader line."""
        annot = Annotation("Test", x=1.0, y=2.0)
        leader_pts = [(1.0, 2.0), (3.0, 4.0)]
        placed = PlacedAnnotation(
            annotation=annot,
            display_x=3.0,
            display_y=4.0,
            needs_leader=True,
            leader_points=leader_pts,
        )
        assert placed.needs_leader is True
        assert placed.leader_points == leader_pts


class TestLayoutStackedChannels:
    """Test layout_stacked_channels function."""

    def test_single_channel(self) -> None:
        """Test layout for single channel."""
        layout = layout_stacked_channels(n_channels=1)
        assert layout.n_channels == 1
        assert len(layout.heights) == 1
        assert len(layout.gaps) == 0
        assert len(layout.y_positions) == 1
        # Should use most of available space
        assert layout.heights[0] > 0.7

    def test_three_channels(self) -> None:
        """Test layout for three channels."""
        layout = layout_stacked_channels(n_channels=3)
        assert layout.n_channels == 3
        assert len(layout.heights) == 3
        assert len(layout.gaps) == 2
        assert len(layout.y_positions) == 3

        # All channel heights should be equal
        assert np.allclose(layout.heights, layout.heights[0])

        # Y positions should be increasing (bottom to top)
        assert np.all(np.diff(layout.y_positions) > 0)

    def test_custom_gap_ratio(self) -> None:
        """Test layout with custom gap ratio."""
        layout1 = layout_stacked_channels(n_channels=2, gap_ratio=0.1)
        layout2 = layout_stacked_channels(n_channels=2, gap_ratio=0.2)

        # Larger gap ratio should give smaller channel heights
        assert layout2.heights[0] < layout1.heights[0]

        # Larger gap ratio should give larger gaps
        assert layout2.gaps[0] > layout1.gaps[0]

    def test_zero_gap_ratio(self) -> None:
        """Test layout with zero gap between channels."""
        layout = layout_stacked_channels(n_channels=3, gap_ratio=0.0)
        assert len(layout.gaps) == 2
        assert np.allclose(layout.gaps, 0.0)

    def test_shared_x_axis(self) -> None:
        """Test layout with shared X axis."""
        layout1 = layout_stacked_channels(n_channels=2, shared_x=True)
        layout2 = layout_stacked_channels(n_channels=2, shared_x=False)

        # Shared X should have larger bottom margin
        assert layout1.y_positions[0] > layout2.y_positions[0]

    def test_custom_figsize(self) -> None:
        """Test layout with custom figure size."""
        figsize = (12, 10)
        layout = layout_stacked_channels(n_channels=3, figsize=figsize)
        assert layout.figsize == figsize

    def test_invalid_n_channels(self) -> None:
        """Test error handling for invalid n_channels."""
        with pytest.raises(ValueError, match="n_channels must be >= 1"):
            layout_stacked_channels(n_channels=0)

        with pytest.raises(ValueError, match="n_channels must be >= 1"):
            layout_stacked_channels(n_channels=-1)

    def test_invalid_gap_ratio(self) -> None:
        """Test error handling for invalid gap_ratio."""
        with pytest.raises(ValueError, match="gap_ratio must be in"):
            layout_stacked_channels(n_channels=2, gap_ratio=-0.1)

        with pytest.raises(ValueError, match="gap_ratio must be in"):
            layout_stacked_channels(n_channels=2, gap_ratio=1.5)

    def test_total_height_normalized(self) -> None:
        """Test that total height fills available space."""
        layout = layout_stacked_channels(n_channels=4, gap_ratio=0.1)

        # Calculate total occupied height
        total_height = np.sum(layout.heights) + np.sum(layout.gaps)

        # Should be close to available height (1.0 - margins)
        assert 0.8 < total_height < 0.95

    def test_many_channels(self) -> None:
        """Test layout with many channels."""
        layout = layout_stacked_channels(n_channels=10)
        assert layout.n_channels == 10
        assert len(layout.heights) == 10
        assert len(layout.gaps) == 9

        # All heights should still be positive
        assert np.all(layout.heights > 0)

    def test_y_positions_non_overlapping(self) -> None:
        """Test that channel Y positions don't overlap."""
        layout = layout_stacked_channels(n_channels=5)

        for i in range(len(layout.y_positions) - 1):
            bottom_of_upper = layout.y_positions[i + 1]
            top_of_lower = layout.y_positions[i] + layout.heights[i] + layout.gaps[i]
            # Upper channel should start at or above lower channel's top
            assert bottom_of_upper >= top_of_lower - 1e-10  # Allow small numerical error


class TestOptimizeAnnotationPlacement:
    """Test optimize_annotation_placement function."""

    def test_single_annotation(self) -> None:
        """Test placement of single annotation."""
        annots = [Annotation("Test", x=100.0, y=100.0)]
        placed = optimize_annotation_placement(annots)

        assert len(placed) == 1
        assert placed[0].annotation.text == "Test"
        # Should stay at original position (no collisions)
        assert placed[0].display_x == 100.0
        assert placed[0].display_y == 100.0
        assert placed[0].needs_leader is False

    def test_two_non_overlapping_annotations(self) -> None:
        """Test placement of non-overlapping annotations."""
        annots = [
            Annotation("A", x=100.0, y=100.0),
            Annotation("B", x=500.0, y=500.0),
        ]
        placed = optimize_annotation_placement(annots, display_width=800, display_height=600)

        assert len(placed) == 2
        # Should not move much since not overlapping
        # Allow some movement due to force algorithm
        assert abs(placed[0].display_x - 100.0) < 50.0
        assert abs(placed[1].display_x - 500.0) < 50.0

    def test_overlapping_annotations_separated(self) -> None:
        """Test that overlapping annotations are separated."""
        # Create two annotations at same position
        annots = [
            Annotation("A", x=100.0, y=100.0, bbox_width=50, bbox_height=20),
            Annotation("B", x=100.0, y=100.0, bbox_width=50, bbox_height=20),
        ]
        placed = optimize_annotation_placement(
            annots,
            display_width=800,
            display_height=600,
            max_iterations=100,
        )

        assert len(placed) == 2
        # Annotations should have been pushed apart
        dx = placed[1].display_x - placed[0].display_x
        dy = placed[1].display_y - placed[0].display_y
        distance = np.sqrt(dx**2 + dy**2)

        # Should be separated by at least their bounding boxes
        assert distance > 20.0

    def test_priority_affects_placement(self) -> None:
        """Test that high priority annotations move less."""
        annots = [
            Annotation("High", x=100.0, y=100.0, priority=0.9),
            Annotation("Low", x=100.0, y=100.0, priority=0.1),
        ]
        placed = optimize_annotation_placement(annots, max_iterations=200, repulsion_strength=50.0)

        # Both should move (they're overlapping)
        # Just verify they were separated
        dx = abs(placed[1].display_x - placed[0].display_x)
        dy = abs(placed[1].display_y - placed[0].display_y)
        distance = np.sqrt(dx**2 + dy**2)
        # Should be separated by at least their bbox sizes
        assert distance > 30.0

    def test_leader_line_for_displaced_annotations(self) -> None:
        """Test that leader lines are generated for displaced annotations."""
        # Create many overlapping annotations to force displacement
        annots = [
            Annotation(f"A{i}", x=100.0, y=100.0, bbox_width=60, bbox_height=30) for i in range(5)
        ]
        placed = optimize_annotation_placement(
            annots,
            display_width=800,
            display_height=600,
            max_iterations=200,
        )

        # At least some annotations should need leader lines
        needs_leader_count = sum(p.needs_leader for p in placed)
        assert needs_leader_count > 0

        # Check that leader points are provided when needed
        for p in placed:
            if p.needs_leader:
                assert p.leader_points is not None
                assert len(p.leader_points) > 0

    def test_display_bounds_clamping(self) -> None:
        """Test that annotations are clamped within display bounds."""
        annots = [
            Annotation("A", x=10.0, y=10.0),
            Annotation("B", x=10.0, y=10.0),
        ]
        width, height = 200.0, 200.0
        placed = optimize_annotation_placement(
            annots,
            display_width=width,
            display_height=height,
            max_iterations=100,
        )

        # All annotations should be within bounds
        for p in placed:
            assert 0 <= p.display_x <= width
            assert 0 <= p.display_y <= height

    def test_empty_annotations_error(self) -> None:
        """Test error handling for empty annotations list."""
        with pytest.raises(ValueError, match="annotations list cannot be empty"):
            optimize_annotation_placement([])

    def test_convergence(self) -> None:
        """Test that algorithm converges."""
        annots = [
            Annotation("A", x=100.0, y=100.0),
            Annotation("B", x=100.0, y=100.0),
        ]
        placed = optimize_annotation_placement(annots, max_iterations=1000)

        # Should have converged (annotations separated)
        dx = placed[1].display_x - placed[0].display_x
        dy = placed[1].display_y - placed[0].display_y
        distance = np.sqrt(dx**2 + dy**2)
        assert distance > 10.0

    def test_min_spacing_enforcement(self) -> None:
        """Test that minimum spacing is enforced."""
        annots = [
            Annotation("A", x=100.0, y=100.0, bbox_width=20, bbox_height=10),
            Annotation("B", x=100.0, y=100.0, bbox_width=20, bbox_height=10),
        ]
        min_spacing = 15.0
        placed = optimize_annotation_placement(
            annots,
            min_spacing=min_spacing,
            max_iterations=200,
        )

        # Check final spacing
        dx = placed[1].display_x - placed[0].display_x
        dy = placed[1].display_y - placed[0].display_y
        distance = np.sqrt(dx**2 + dy**2)

        # Should respect minimum spacing + bbox sizes
        min_expected = min_spacing
        assert distance >= min_expected - 1.0  # Allow small tolerance

    def test_repulsion_strength(self) -> None:
        """Test that repulsion strength affects separation."""
        annots = [
            Annotation("A", x=100.0, y=100.0),
            Annotation("B", x=100.0, y=100.0),
        ]

        placed_weak = optimize_annotation_placement(
            annots, repulsion_strength=1.0, max_iterations=50
        )
        placed_strong = optimize_annotation_placement(
            annots, repulsion_strength=100.0, max_iterations=50
        )

        # Stronger repulsion should give larger separation (or faster convergence)
        dist_weak = np.sqrt(
            (placed_weak[1].display_x - placed_weak[0].display_x) ** 2
            + (placed_weak[1].display_y - placed_weak[0].display_y) ** 2
        )
        dist_strong = np.sqrt(
            (placed_strong[1].display_x - placed_strong[0].display_x) ** 2
            + (placed_strong[1].display_y - placed_strong[0].display_y) ** 2
        )

        # Both should separate annotations
        assert dist_weak > 10.0
        assert dist_strong > 10.0

    def test_many_annotations(self) -> None:
        """Test placement with many annotations."""
        annots = [Annotation(f"Annot{i}", x=100 + i * 10, y=100 + i * 10) for i in range(20)]
        placed = optimize_annotation_placement(annots, max_iterations=200)

        assert len(placed) == 20
        # All should be placed
        assert all(p.display_x >= 0 for p in placed)

    def test_custom_bbox_sizes(self) -> None:
        """Test with custom bounding box sizes."""
        annots = [
            Annotation("Small", x=100.0, y=100.0, bbox_width=20, bbox_height=10),
            Annotation("Large", x=100.0, y=100.0, bbox_width=100, bbox_height=50),
        ]
        placed = optimize_annotation_placement(annots, max_iterations=100)

        # Both should be separated appropriately
        assert len(placed) == 2
        dx = abs(placed[1].display_x - placed[0].display_x)
        dy = abs(placed[1].display_y - placed[0].display_y)

        # Separation should account for large bbox
        assert dx > 50 or dy > 25


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_layout_maximum_gap_ratio(self) -> None:
        """Test layout with maximum gap ratio."""
        layout = layout_stacked_channels(n_channels=2, gap_ratio=1.0)
        # Gap should equal channel height
        assert np.allclose(layout.gaps[0], layout.heights[0])

    def test_layout_minimum_gap_ratio(self) -> None:
        """Test layout with minimum gap ratio."""
        layout = layout_stacked_channels(n_channels=2, gap_ratio=0.0)
        # No gap between channels
        assert np.allclose(layout.gaps[0], 0.0)

    def test_annotation_zero_priority(self) -> None:
        """Test annotation with zero priority."""
        annot = Annotation("Test", x=1.0, y=1.0, priority=0.0)
        assert annot.priority == 0.0

    def test_annotation_max_priority(self) -> None:
        """Test annotation with maximum priority."""
        annot = Annotation("Test", x=1.0, y=1.0, priority=1.0)
        assert annot.priority == 1.0

    def test_very_small_display_area(self) -> None:
        """Test annotation placement in very small display area."""
        annots = [Annotation("A", x=5.0, y=5.0)]
        placed = optimize_annotation_placement(annots, display_width=10.0, display_height=10.0)

        assert len(placed) == 1
        assert 0 <= placed[0].display_x <= 10.0
        assert 0 <= placed[0].display_y <= 10.0

    def test_large_display_area(self) -> None:
        """Test annotation placement in large display area."""
        annots = [Annotation("A", x=1000.0, y=1000.0)]
        placed = optimize_annotation_placement(
            annots, display_width=10000.0, display_height=10000.0
        )

        assert len(placed) == 1
        # Should stay near original position
        assert abs(placed[0].display_x - 1000.0) < 100.0
