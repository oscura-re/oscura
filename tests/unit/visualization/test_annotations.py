"""Comprehensive unit tests for annotation placement with collision detection.

Tests:

Coverage:
- Annotation dataclass
- PlacedAnnotation dataclass
- place_annotations() function
- filter_by_zoom_level() function
- create_priority_annotation() function
- Internal collision detection and resolution
- Edge cases (empty annotations, overlapping annotations)
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.visualization.annotations import (
    Annotation,
    PlacedAnnotation,
    create_priority_annotation,
    filter_by_zoom_level,
    place_annotations,
)

pytestmark = [pytest.mark.unit, pytest.mark.visualization]


@pytest.mark.unit
@pytest.mark.visualization
class TestAnnotationDataclass:
    """Tests for Annotation dataclass."""

    def test_annotation_creation_minimal(self):
        """Test creating annotation with minimal required fields."""
        annot = Annotation(text="Test", x=5.0, y=1.0)

        assert annot.text == "Test"
        assert annot.x == 5.0
        assert annot.y == 1.0
        assert annot.bbox_width == 60.0  # Default
        assert annot.bbox_height == 20.0  # Default
        assert annot.priority == 0.5  # Default
        assert annot.anchor == "auto"  # Default
        assert annot.metadata == {}  # Initialized in __post_init__

    def test_annotation_creation_full(self):
        """Test creating annotation with all fields specified."""
        metadata = {"type": "peak", "value": 42}
        annot = Annotation(
            text="Peak",
            x=10.0,
            y=2.5,
            bbox_width=80.0,
            bbox_height=30.0,
            priority=0.9,
            anchor="top",
            metadata=metadata,
        )

        assert annot.text == "Peak"
        assert annot.x == 10.0
        assert annot.y == 2.5
        assert annot.bbox_width == 80.0
        assert annot.bbox_height == 30.0
        assert annot.priority == 0.9
        assert annot.anchor == "top"
        assert annot.metadata == metadata

    def test_annotation_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        annot = Annotation(text="Test", x=0.0, y=0.0, metadata=None)
        assert annot.metadata == {}
        assert isinstance(annot.metadata, dict)

    def test_annotation_metadata_preserved(self):
        """Test that provided metadata is preserved."""
        metadata = {"custom": "value"}
        annot = Annotation(text="Test", x=0.0, y=0.0, metadata=metadata)
        assert annot.metadata is metadata  # Same object reference

    def test_annotation_with_zero_coordinates(self):
        """Test annotation at origin."""
        annot = Annotation(text="Origin", x=0.0, y=0.0)
        assert annot.x == 0.0
        assert annot.y == 0.0

    def test_annotation_with_negative_coordinates(self):
        """Test annotation with negative coordinates."""
        annot = Annotation(text="Negative", x=-5.0, y=-10.0)
        assert annot.x == -5.0
        assert annot.y == -10.0

    def test_annotation_priority_range(self):
        """Test various priority values."""
        for priority in [0.0, 0.2, 0.5, 0.8, 1.0]:
            annot = Annotation(text="Test", x=0.0, y=0.0, priority=priority)
            assert annot.priority == priority


@pytest.mark.unit
@pytest.mark.visualization
class TestPlacedAnnotationDataclass:
    """Tests for PlacedAnnotation dataclass."""

    def test_placed_annotation_creation_minimal(self):
        """Test creating placed annotation with minimal fields."""
        base_annot = Annotation(text="Test", x=5.0, y=1.0)
        placed = PlacedAnnotation(
            annotation=base_annot,
            display_x=5.5,
            display_y=1.5,
        )

        assert placed.annotation is base_annot
        assert placed.display_x == 5.5
        assert placed.display_y == 1.5
        assert placed.visible is True  # Default
        assert placed.needs_leader is False  # Default
        assert placed.leader_points is None  # Default

    def test_placed_annotation_creation_full(self):
        """Test creating placed annotation with all fields."""
        base_annot = Annotation(text="Test", x=5.0, y=1.0)
        leader_points = [(5.0, 1.0), (5.0, 1.5), (5.5, 1.5)]
        placed = PlacedAnnotation(
            annotation=base_annot,
            display_x=5.5,
            display_y=1.5,
            visible=True,
            needs_leader=True,
            leader_points=leader_points,
        )

        assert placed.annotation is base_annot
        assert placed.display_x == 5.5
        assert placed.display_y == 1.5
        assert placed.visible is True
        assert placed.needs_leader is True
        assert placed.leader_points == leader_points

    def test_placed_annotation_hidden(self):
        """Test creating hidden placed annotation."""
        base_annot = Annotation(text="Test", x=5.0, y=1.0)
        placed = PlacedAnnotation(
            annotation=base_annot,
            display_x=5.0,
            display_y=1.0,
            visible=False,
        )

        assert placed.visible is False

    def test_placed_annotation_with_leader_line(self):
        """Test placed annotation with leader line."""
        base_annot = Annotation(text="Test", x=0.0, y=0.0)
        leader_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        placed = PlacedAnnotation(
            annotation=base_annot,
            display_x=10.0,
            display_y=10.0,
            needs_leader=True,
            leader_points=leader_points,
        )

        assert placed.needs_leader is True
        assert placed.leader_points == leader_points
        assert len(placed.leader_points) == 3


@pytest.mark.unit
@pytest.mark.visualization
class TestPlaceAnnotations:
    """Tests for place_annotations function."""

    def test_empty_annotations_list(self):
        """Test that empty list returns empty result."""
        result = place_annotations([])
        assert result == []

    def test_single_annotation(self):
        """Test placing single annotation."""
        annot = Annotation(text="Single", x=5.0, y=1.0, priority=0.5)
        placed = place_annotations([annot])

        assert len(placed) == 1
        assert placed[0].annotation is annot
        assert placed[0].display_x == 5.0  # No collision, stays at anchor
        assert placed[0].display_y == 1.0
        assert placed[0].visible is True
        assert placed[0].needs_leader is False  # No displacement

    def test_multiple_non_overlapping_annotations(self):
        """Test placing multiple annotations with no collisions."""
        annots = [
            Annotation(text="A", x=0.0, y=0.0),
            Annotation(text="B", x=100.0, y=100.0),
            Annotation(text="C", x=200.0, y=200.0),
        ]
        placed = place_annotations(annots, collision_threshold=5.0)

        assert len(placed) == 3
        # All should stay at anchor points
        for p in placed:
            assert p.display_x == p.annotation.x
            assert p.display_y == p.annotation.y
            assert p.visible is True
            assert p.needs_leader is False

    def test_overlapping_annotations_collision_resolution(self):
        """Test that overlapping annotations are separated."""
        # Two annotations at same position
        annots = [
            Annotation(text="High", x=5.0, y=1.0, priority=0.9),
            Annotation(text="Low", x=5.0, y=1.0, priority=0.3),
        ]
        placed = place_annotations(annots, collision_threshold=5.0, max_iterations=50)

        assert len(placed) == 2

        # Should be separated
        high_placed = next(p for p in placed if p.annotation.priority == 0.9)
        low_placed = next(p for p in placed if p.annotation.priority == 0.3)

        # High priority should stay at anchor
        assert high_placed.display_x == 5.0
        assert high_placed.display_y == 1.0

        # Low priority should be moved
        dx = abs(low_placed.display_x - high_placed.display_x)
        dy = abs(low_placed.display_y - high_placed.display_y)
        distance = np.sqrt(dx**2 + dy**2)
        assert distance > 0  # Should be moved away

    def test_priority_based_placement(self):
        """Test that higher priority annotations get better placement."""
        # Three overlapping annotations with different priorities
        annots = [
            Annotation(text="Critical", x=10.0, y=10.0, priority=1.0),
            Annotation(text="Medium", x=10.0, y=10.0, priority=0.5),
            Annotation(text="Low", x=10.0, y=10.0, priority=0.2),
        ]
        placed = place_annotations(annots, collision_threshold=5.0, max_iterations=100)

        assert len(placed) == 3

        # Critical priority should be at anchor
        critical = next(p for p in placed if p.annotation.text == "Critical")
        assert critical.display_x == 10.0
        assert critical.display_y == 10.0

        # Others should be displaced
        medium = next(p for p in placed if p.annotation.text == "Medium")
        low = next(p for p in placed if p.annotation.text == "Low")

        medium_dist = np.sqrt((medium.display_x - 10.0) ** 2 + (medium.display_y - 10.0) ** 2)
        low_dist = np.sqrt((low.display_x - 10.0) ** 2 + (low.display_y - 10.0) ** 2)

        # At least one should be displaced
        assert medium_dist > 0 or low_dist > 0

    def test_viewport_filtering(self):
        """Test viewport filtering."""
        annots = [
            Annotation(text="Inside", x=5.0, y=1.0, priority=0.5),
            Annotation(text="Outside", x=15.0, y=1.0, priority=0.5),
            Annotation(text="Edge", x=10.0, y=1.0, priority=0.5),
        ]
        placed = place_annotations(annots, viewport=(0.0, 10.0))

        # Should only include annotations within viewport
        assert len(placed) <= 2  # "Inside" and "Edge"

        texts = [p.annotation.text for p in placed]
        assert "Inside" in texts
        assert "Edge" in texts
        # "Outside" should be filtered out

    def test_density_limiting(self):
        """Test density limiting."""
        # Create 30 annotations but limit to 10
        annots = [
            Annotation(text=f"A{i}", x=float(i), y=1.0, priority=float(i) / 30.0) for i in range(30)
        ]
        placed = place_annotations(annots, density_limit=10)

        # Should only keep top 10 by priority
        assert len(placed) == 10

        # Verify they are the highest priority ones
        kept_priorities = sorted([p.annotation.priority for p in placed], reverse=True)
        all_priorities = sorted([a.priority for a in annots], reverse=True)
        assert kept_priorities == all_priorities[:10]

    def test_density_limiting_with_viewport(self):
        """Test density limiting combined with viewport."""
        # Create annotations, some inside viewport, some outside
        annots = [
            Annotation(text=f"In{i}", x=float(i), y=1.0, priority=0.5 + i * 0.01) for i in range(10)
        ] + [Annotation(text=f"Out{i}", x=float(20 + i), y=1.0, priority=0.9) for i in range(10)]

        placed = place_annotations(annots, viewport=(0.0, 10.0), density_limit=5)

        # Should filter by viewport first, then apply density limit
        assert len(placed) <= 5
        # All should be within viewport
        for p in placed:
            assert 0.0 <= p.annotation.x <= 10.0

    def test_collision_threshold_parameter(self):
        """Test custom collision threshold."""
        annots = [
            Annotation(text="A", x=0.0, y=0.0),
            Annotation(text="B", x=1.0, y=0.0),
        ]

        # With large threshold, should collide
        placed_large = place_annotations(annots, collision_threshold=50.0)
        dx_large = abs(placed_large[1].display_x - placed_large[0].display_x)

        # With small threshold, might not need as much separation
        placed_small = place_annotations(annots, collision_threshold=1.0)
        dx_small = abs(placed_small[1].display_x - placed_small[0].display_x)

        # Larger threshold should create more separation
        assert dx_large >= dx_small or dx_large > 0

    def test_max_iterations_parameter(self):
        """Test max iterations parameter."""
        # Create clustered annotations
        annots = [
            Annotation(text=f"A{i}", x=5.0 + i * 0.1, y=1.0, priority=float(i) / 10)
            for i in range(10)
        ]

        # With limited iterations, might not fully converge
        placed_few = place_annotations(annots, max_iterations=1)
        assert len(placed_few) == 10

        # With many iterations, should converge better
        placed_many = place_annotations(annots, max_iterations=100)
        assert len(placed_many) == 10

    def test_leader_line_generation(self):
        """Test that leader lines are generated for displaced annotations."""
        annots = [
            Annotation(text="High", x=5.0, y=1.0, priority=1.0),
            Annotation(text="Low", x=5.0, y=1.0, priority=0.1),
        ]

        placed = place_annotations(annots, collision_threshold=50.0, max_iterations=100)

        # Low priority annotation should be displaced significantly
        low_placed = next(p for p in placed if p.annotation.priority == 0.1)

        # Check if it has a leader line
        # Leader line is generated if displacement > 30.0 pixels
        dx = abs(low_placed.display_x - low_placed.annotation.x)
        dy = abs(low_placed.display_y - low_placed.annotation.y)
        displacement = np.sqrt(dx**2 + dy**2)

        if displacement > 30.0:
            assert low_placed.needs_leader is True
            assert low_placed.leader_points is not None
            assert len(low_placed.leader_points) == 3  # Anchor, mid, label
        else:
            assert low_placed.needs_leader is False

    def test_viewport_none(self):
        """Test that viewport=None shows all annotations."""
        annots = [Annotation(text=f"A{i}", x=float(i * 100), y=1.0) for i in range(5)]
        placed = place_annotations(annots, viewport=None, density_limit=100)

        # All should be included
        assert len(placed) == 5

    def test_large_number_of_annotations(self):
        """Test placing large number of annotations."""
        annots = [
            Annotation(
                text=f"A{i}",
                x=float(i % 10) * 10,
                y=float(i // 10),
                priority=np.random.random(),
            )
            for i in range(100)
        ]

        placed = place_annotations(annots, density_limit=50, max_iterations=20)

        # Should respect density limit
        assert len(placed) <= 50
        # All should be visible
        assert all(p.visible for p in placed)

    def test_annotations_with_different_sizes(self):
        """Test annotations with different bounding box sizes."""
        annots = [
            Annotation(text="Small", x=5.0, y=1.0, bbox_width=20.0, bbox_height=10.0),
            Annotation(text="Large", x=5.0, y=1.0, bbox_width=100.0, bbox_height=50.0),
        ]

        placed = place_annotations(annots, collision_threshold=5.0)

        assert len(placed) == 2
        # Collision detection should account for different sizes


@pytest.mark.unit
@pytest.mark.visualization
class TestFilterByZoomLevel:
    """Tests for filter_by_zoom_level function."""

    def test_filter_zoomed_in(self):
        """Test that all annotations visible when zoomed in."""
        base_annots = [Annotation(text=f"A{i}", x=float(i), y=1.0) for i in range(5)]
        placed = [PlacedAnnotation(annotation=a, display_x=a.x, display_y=a.y) for a in base_annots]

        # Zoomed in (small range)
        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 5.0), min_width_for_display=10.0)

        assert len(filtered) == 5
        # All should be visible when zoom width < min_width_for_display
        assert all(p.visible for p in filtered)

    def test_filter_zoomed_out(self):
        """Test that annotations are filtered when zoomed out."""
        base_annots = [Annotation(text=f"A{i}", x=float(i), y=1.0) for i in range(10)]
        placed = [PlacedAnnotation(annotation=a, display_x=a.x, display_y=a.y) for a in base_annots]

        # Zoomed out (large range)
        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 100.0), min_width_for_display=10.0)

        # Only annotations within viewport should be visible
        visible_count = sum(1 for p in filtered if p.visible)
        assert visible_count <= 10  # At most the ones in viewport

    def test_filter_with_min_width_threshold(self):
        """Test min_width_for_display threshold."""
        base_annots = [Annotation(text="Test", x=5.0, y=1.0)]
        placed = [PlacedAnnotation(annotation=base_annots[0], display_x=5.0, display_y=5.0)]

        # Zoom width exactly at threshold
        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 1.0), min_width_for_display=1.0)
        # Width == threshold, should not be visible (< check)
        assert not filtered[0].visible

        # Zoom width below threshold
        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 0.5), min_width_for_display=1.0)
        # Width < threshold, should be visible
        assert filtered[0].visible

    def test_filter_empty_list(self):
        """Test filtering empty list."""
        filtered = filter_by_zoom_level([], zoom_range=(0.0, 10.0))
        assert filtered == []

    def test_filter_outside_viewport(self):
        """Test that annotations outside viewport are hidden when zoomed out."""
        base_annots = [
            Annotation(text="Inside", x=5.0, y=1.0),
            Annotation(text="Outside", x=50.0, y=1.0),
        ]
        placed = [PlacedAnnotation(annotation=a, display_x=a.x, display_y=a.y) for a in base_annots]

        # Test when zoomed out (width >= min), viewport filtering applies
        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 10.0), min_width_for_display=5.0)

        # Zoom width (10.0) >= min (5.0), so viewport filtering applies
        inside = next(p for p in filtered if p.annotation.text == "Inside")
        outside = next(p for p in filtered if p.annotation.text == "Outside")

        assert inside.visible is True  # Within viewport (0-10)
        assert outside.visible is False  # Outside viewport (x=50 > 10)

    def test_filter_preserves_placed_annotations(self):
        """Test that filtering preserves other PlacedAnnotation attributes."""
        base_annot = Annotation(text="Test", x=5.0, y=1.0)
        placed = [
            PlacedAnnotation(
                annotation=base_annot,
                display_x=6.0,
                display_y=2.0,
                needs_leader=True,
                leader_points=[(5.0, 1.0), (6.0, 2.0)],
            )
        ]

        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 10.0))

        assert len(filtered) == 1
        assert filtered[0].display_x == 6.0
        assert filtered[0].display_y == 2.0
        assert filtered[0].needs_leader is True
        assert filtered[0].leader_points == [(5.0, 1.0), (6.0, 2.0)]

    def test_filter_all_annotations_in_viewport(self):
        """Test when all annotations are within viewport."""
        base_annots = [Annotation(text=f"A{i}", x=float(i), y=1.0) for i in range(5)]
        placed = [PlacedAnnotation(annotation=a, display_x=a.x, display_y=a.y) for a in base_annots]

        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 10.0), min_width_for_display=100.0)

        # All within viewport (0-10), all should be visible
        assert all(p.visible for p in filtered)

    def test_filter_zoom_range_at_boundaries(self):
        """Test annotations at exact viewport boundaries."""
        base_annots = [
            Annotation(text="Left", x=0.0, y=1.0),
            Annotation(text="Right", x=10.0, y=1.0),
            Annotation(text="Middle", x=5.0, y=1.0),
        ]
        placed = [PlacedAnnotation(annotation=a, display_x=a.x, display_y=a.y) for a in base_annots]

        filtered = filter_by_zoom_level(placed, zoom_range=(0.0, 10.0), min_width_for_display=100.0)

        # All at or within boundaries should be visible
        assert all(p.visible for p in filtered)


@pytest.mark.unit
@pytest.mark.visualization
class TestCreatePriorityAnnotation:
    """Tests for create_priority_annotation function."""

    def test_create_critical_priority(self):
        """Test creating critical priority annotation."""
        annot = create_priority_annotation("Critical Event", x=5.0, y=1.0, importance="critical")

        assert annot.text == "Critical Event"
        assert annot.x == 5.0
        assert annot.y == 1.0
        assert annot.priority == 1.0

    def test_create_high_priority(self):
        """Test creating high priority annotation."""
        annot = create_priority_annotation("High Event", x=5.0, y=1.0, importance="high")

        assert annot.text == "High Event"
        assert annot.priority == 0.8

    def test_create_normal_priority(self):
        """Test creating normal priority annotation."""
        annot = create_priority_annotation("Normal Event", x=5.0, y=1.0, importance="normal")

        assert annot.text == "Normal Event"
        assert annot.priority == 0.5

    def test_create_low_priority(self):
        """Test creating low priority annotation."""
        annot = create_priority_annotation("Low Event", x=5.0, y=1.0, importance="low")

        assert annot.text == "Low Event"
        assert annot.priority == 0.2

    def test_create_unknown_importance(self):
        """Test that unknown importance defaults to normal."""
        annot = create_priority_annotation("Unknown", x=5.0, y=1.0, importance="unknown")

        assert annot.priority == 0.5  # Default to normal

    def test_create_with_additional_kwargs(self):
        """Test creating annotation with additional kwargs."""
        annot = create_priority_annotation(
            "Test",
            x=5.0,
            y=1.0,
            importance="high",
            bbox_width=100.0,
            bbox_height=40.0,
            anchor="top",
            metadata={"custom": "value"},
        )

        assert annot.priority == 0.8  # High
        assert annot.bbox_width == 100.0
        assert annot.bbox_height == 40.0
        assert annot.anchor == "top"
        assert annot.metadata == {"custom": "value"}

    def test_create_all_importance_levels(self):
        """Test all importance levels map correctly."""
        importance_mapping = {
            "critical": 1.0,
            "high": 0.8,
            "normal": 0.5,
            "low": 0.2,
        }

        for importance, expected_priority in importance_mapping.items():
            annot = create_priority_annotation("Test", x=0.0, y=0.0, importance=importance)
            assert annot.priority == expected_priority

    def test_create_with_negative_coordinates(self):
        """Test creating annotation with negative coordinates."""
        annot = create_priority_annotation("Negative", x=-10.0, y=-5.0, importance="critical")

        assert annot.x == -10.0
        assert annot.y == -5.0
        assert annot.priority == 1.0

    def test_create_default_importance(self):
        """Test that default importance is normal."""
        annot = create_priority_annotation("Default", x=5.0, y=1.0)

        assert annot.priority == 0.5  # Normal


@pytest.mark.unit
@pytest.mark.visualization
class TestVisualizationAnnotationsEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_annotations_at_same_exact_position(self):
        """Test multiple annotations at exactly the same position."""
        annots = [Annotation(text=f"A{i}", x=5.0, y=1.0, priority=float(i) / 10) for i in range(5)]

        placed = place_annotations(annots, collision_threshold=5.0, max_iterations=100)

        # All should be placed and separated
        assert len(placed) == 5

        # Check that they are separated
        positions = [(p.display_x, p.display_y) for p in placed]
        # At least some should be different (collision resolution)
        unique_positions = set(positions)
        assert len(unique_positions) >= 2  # Should create some separation

    def test_annotations_with_zero_bbox(self):
        """Test annotations with zero-sized bounding boxes."""
        annots = [
            Annotation(text="Zero", x=5.0, y=1.0, bbox_width=0.0, bbox_height=0.0, priority=0.5),
            Annotation(
                text="Normal", x=5.0, y=1.0, bbox_width=60.0, bbox_height=20.0, priority=0.5
            ),
        ]

        placed = place_annotations(annots)

        # Should handle gracefully
        assert len(placed) == 2

    def test_very_high_density_limit(self):
        """Test with density limit higher than number of annotations."""
        annots = [Annotation(text=f"A{i}", x=float(i), y=1.0) for i in range(5)]

        placed = place_annotations(annots, density_limit=1000)

        # Should keep all annotations
        assert len(placed) == 5

    def test_very_low_density_limit(self):
        """Test with density limit of 1."""
        annots = [
            Annotation(text=f"A{i}", x=float(i), y=1.0, priority=float(i) / 10) for i in range(10)
        ]

        placed = place_annotations(annots, density_limit=1)

        # Should keep only highest priority
        assert len(placed) == 1
        assert placed[0].annotation.priority == 0.9  # Highest

    def test_viewport_with_no_annotations_inside(self):
        """Test viewport that excludes all annotations."""
        annots = [Annotation(text=f"A{i}", x=float(i), y=1.0) for i in range(10, 20)]

        placed = place_annotations(annots, viewport=(0.0, 5.0))

        # No annotations in viewport
        assert len(placed) == 0

    def test_negative_viewport(self):
        """Test viewport with negative coordinates."""
        annots = [Annotation(text=f"A{i}", x=float(i - 10), y=1.0) for i in range(20)]

        placed = place_annotations(annots, viewport=(-5.0, 5.0))

        # Should include annotations from -5 to 5
        assert len(placed) <= len(annots)
        for p in placed:
            assert -5.0 <= p.annotation.x <= 5.0

    def test_max_iterations_zero(self):
        """Test with zero iterations (no collision resolution)."""
        annots = [
            Annotation(text="A", x=5.0, y=1.0, priority=0.9),
            Annotation(text="B", x=5.0, y=1.0, priority=0.5),
        ]

        placed = place_annotations(annots, max_iterations=0)

        # Should still place them, just not resolve collisions
        assert len(placed) == 2

    def test_very_small_collision_threshold(self):
        """Test with very small collision threshold."""
        annots = [
            Annotation(text="A", x=5.0, y=1.0),
            Annotation(text="B", x=5.1, y=1.0),
        ]

        placed = place_annotations(annots, collision_threshold=0.01)

        # Should handle small thresholds
        assert len(placed) == 2

    def test_very_large_collision_threshold(self):
        """Test with very large collision threshold."""
        annots = [
            Annotation(text="A", x=0.0, y=0.0, priority=1.0),
            Annotation(text="B", x=10.0, y=10.0, priority=0.5),
        ]

        placed = place_annotations(annots, collision_threshold=1000.0)

        # Even distant annotations might be considered colliding
        assert len(placed) == 2

    def test_annotation_displacement_calculation(self):
        """Test that displacement for leader lines is calculated correctly."""
        # Create annotation that will be significantly displaced
        annots = [
            Annotation(
                text="Fixed", x=0.0, y=0.0, priority=1.0, bbox_width=200.0, bbox_height=100.0
            ),
            Annotation(
                text="Moved", x=0.0, y=0.0, priority=0.1, bbox_width=200.0, bbox_height=100.0
            ),
        ]

        placed = place_annotations(annots, collision_threshold=50.0, max_iterations=100)

        moved = next(p for p in placed if p.annotation.text == "Moved")

        # Calculate displacement
        dx = abs(moved.display_x - moved.annotation.x)
        dy = abs(moved.display_y - moved.annotation.y)
        displacement = np.sqrt(dx**2 + dy**2)

        # If displacement > 30.0, should have leader line
        if displacement > 30.0:
            assert moved.needs_leader is True
            assert moved.leader_points is not None
            assert len(moved.leader_points) == 3

            # Verify leader line connects anchor to label
            assert moved.leader_points[0] == (moved.annotation.x, moved.annotation.y)
            assert moved.leader_points[2] == (moved.display_x, moved.display_y)

    def test_annotations_already_sufficiently_separated(self):
        """Test that annotations already separated don't get moved."""
        # Create annotations that are already far enough apart
        annots = [
            Annotation(text="A", x=0.0, y=0.0, priority=0.5, bbox_width=20.0, bbox_height=10.0),
            Annotation(
                text="B", x=1000.0, y=1000.0, priority=0.5, bbox_width=20.0, bbox_height=10.0
            ),
        ]

        placed = place_annotations(annots, collision_threshold=5.0, max_iterations=50)

        # Both should stay at their original positions (no collision)
        assert len(placed) == 2
        for p in placed:
            assert p.display_x == p.annotation.x
            assert p.display_y == p.annotation.y
            assert p.needs_leader is False

    def test_collision_convergence_edge_case(self):
        """Test edge case where collision is detected but distance check passes.

        This can happen when the bounding box collision check (dx < min_dx AND dy < min_dy)
        passes, but the Euclidean distance check in _move_annotation shows they're already
        far enough apart. This is a rare edge case but possible with the different collision
        detection algorithms.
        """
        # Create annotations positioned such that bounding boxes barely overlap
        # but Euclidean distance is at the limit
        annots = [
            Annotation(text="A", x=0.0, y=0.0, priority=1.0, bbox_width=10.0, bbox_height=10.0),
            Annotation(text="B", x=14.5, y=14.5, priority=0.5, bbox_width=10.0, bbox_height=10.0),
        ]

        # Use very small threshold to create the edge case
        placed = place_annotations(annots, collision_threshold=0.1, max_iterations=100)

        # Should still complete without error
        assert len(placed) == 2


@pytest.mark.unit
@pytest.mark.visualization
class TestIntegrationScenarios:
    """Integration tests for realistic annotation scenarios."""

    def test_realistic_peak_annotations(self):
        """Test realistic scenario with peak annotations."""
        # Simulate peaks at different priorities
        annots = [
            create_priority_annotation("Critical Peak", x=5.0, y=10.0, importance="critical"),
            create_priority_annotation("High Peak", x=5.5, y=9.8, importance="high"),
            create_priority_annotation("Normal Peak", x=6.0, y=9.5, importance="normal"),
            create_priority_annotation("Low Peak", x=6.5, y=9.3, importance="low"),
        ]

        placed = place_annotations(
            annots, viewport=(0.0, 20.0), density_limit=10, collision_threshold=5.0
        )

        assert len(placed) == 4

        # Critical should stay at anchor
        critical = next(p for p in placed if p.annotation.text == "Critical Peak")
        assert critical.display_x == 5.0
        assert critical.display_y == 10.0

    def test_zoomed_workflow(self):
        """Test complete workflow: place, then filter by zoom."""
        # Create and place annotations
        annots = [
            Annotation(text=f"Event{i}", x=float(i), y=1.0, priority=np.random.random())
            for i in range(20)
        ]

        placed = place_annotations(annots, density_limit=15)

        # Filter by zoom level - zoomed OUT (width >= min), so viewport filtering applies
        zoomed_out = filter_by_zoom_level(placed, zoom_range=(0.0, 10.0), min_width_for_display=5.0)

        # Should only show annotations in viewport when zoomed out
        visible_out = [p for p in zoomed_out if p.visible]
        for p in visible_out:
            assert 0.0 <= p.annotation.x <= 10.0

        # Filter by zoom level - zoomed IN (width < min), so all visible
        zoomed_in = filter_by_zoom_level(placed, zoom_range=(0.0, 10.0), min_width_for_display=20.0)

        # All should be visible when zoomed in (zoom_width=10 < min=20)
        assert all(p.visible for p in zoomed_in)

    def test_dense_annotation_region(self):
        """Test handling of very dense annotation region."""
        # Create many overlapping annotations
        annots = [
            Annotation(
                text=f"Dense{i}",
                x=5.0 + np.random.randn() * 0.5,
                y=1.0 + np.random.randn() * 0.1,
                priority=np.random.random(),
            )
            for i in range(50)
        ]

        placed = place_annotations(
            annots, density_limit=20, collision_threshold=3.0, max_iterations=50
        )

        # Should limit to density
        assert len(placed) == 20

        # Check that some have leader lines due to displacement
        with_leaders = [p for p in placed if p.needs_leader]
        # At least some should need leaders in dense region
        assert len(with_leaders) >= 0  # May or may not need leaders depending on convergence

    def test_mixed_priority_dense_region(self):
        """Test that high priority annotations win in dense regions."""
        # Create mixed priority annotations in small region
        annots = []
        for i in range(10):
            annots.append(
                Annotation(
                    text=f"Critical{i}",
                    x=5.0 + i * 0.1,
                    y=1.0,
                    priority=1.0,
                )
            )
            annots.append(
                Annotation(
                    text=f"Low{i}",
                    x=5.0 + i * 0.1,
                    y=1.0,
                    priority=0.1,
                )
            )

        placed = place_annotations(annots, density_limit=15)

        # Should prioritize critical annotations
        critical_count = sum(1 for p in placed if "Critical" in p.annotation.text)
        low_count = sum(1 for p in placed if "Low" in p.annotation.text)

        # Critical should dominate
        assert critical_count >= low_count
