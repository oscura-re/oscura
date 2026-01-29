"""Enhanced annotation placement with collision detection.

This module provides intelligent annotation placement with collision avoidance,
priority-based positioning, and dynamic hiding at different zoom levels.


Example:
    >>> from oscura.visualization.annotations import place_annotations
    >>> placed = place_annotations(annotations, viewport=(0, 10), density_limit=20)

References:
    - Force-directed graph layout (Fruchterman-Reingold)
    - Greedy placement with priority
    - Leader line routing algorithms
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oscura.utils.geometry import generate_leader_line


@dataclass
class Annotation:
    """Annotation specification with position and metadata.

    Attributes:
        text: Annotation text
        x: X coordinate in data units
        y: Y coordinate in data units
        bbox_width: Bounding box width in pixels
        bbox_height: Bounding box height in pixels
        priority: Priority for placement (0-1, higher is more important)
        anchor: Preferred anchor position
        metadata: Additional metadata
    """

    text: str
    x: float
    y: float
    bbox_width: float = 60.0
    bbox_height: float = 20.0
    priority: float = 0.5
    anchor: str = "auto"
    metadata: dict | None = None  # type: ignore[type-arg]

    def __post_init__(self):  # type: ignore[no-untyped-def]
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PlacedAnnotation:
    """Annotation with optimized placement and leader line.

    Attributes:
        annotation: Original annotation
        display_x: Optimized X position in data units
        display_y: Optimized Y position in data units
        visible: Whether annotation is visible at current zoom
        needs_leader: Whether a leader line is needed
        leader_points: Points for leader line (if needed)
    """

    annotation: Annotation
    display_x: float
    display_y: float
    visible: bool = True
    needs_leader: bool = False
    leader_points: list[tuple[float, float]] | None = None


def place_annotations(
    annotations: list[Annotation],
    *,
    viewport: tuple[float, float] | None = None,
    density_limit: int = 20,
    collision_threshold: float = 5.0,
    max_iterations: int = 50,
) -> list[PlacedAnnotation]:
    """Place annotations with collision detection and density limiting.

    Enhanced version with viewport-aware density limiting and dynamic hiding.

    Args:
        annotations: List of annotations to place.
        viewport: Viewport range (x_min, x_max) for density calculation (None = all visible).
        density_limit: Maximum annotations per viewport.
        collision_threshold: Minimum spacing in pixels.
        max_iterations: Maximum iterations for collision resolution.

    Returns:
        List of PlacedAnnotation with optimized positions.

    Example:
        >>> annots = [
        ...     Annotation("Peak", 5.0, 1.0, priority=0.9),
        ...     Annotation("Min", 3.0, -0.5, priority=0.7),
        ... ]
        >>> placed = place_annotations(annots, density_limit=10)

    References:
        VIS-016: Annotation Placement Intelligence (enhanced)
    """
    if len(annotations) == 0:
        return []

    # Filter and limit annotations
    visible_annots = _filter_by_viewport(annotations, viewport)
    visible_annots = _apply_density_limit(visible_annots, density_limit)

    # Initialize placement
    placed = _initialize_placements(visible_annots)

    # Resolve collisions
    _resolve_collisions(placed, collision_threshold, max_iterations)

    # Add leader lines where needed
    _add_leader_lines(placed, leader_threshold=30.0)

    return placed


def _filter_by_viewport(
    annotations: list[Annotation],
    viewport: tuple[float, float] | None,
) -> list[Annotation]:
    """Filter annotations by viewport range."""
    if viewport is None:
        return annotations

    x_min, x_max = viewport
    return [a for a in annotations if x_min <= a.x <= x_max]


def _apply_density_limit(
    annotations: list[Annotation],
    density_limit: int,
) -> list[Annotation]:
    """Apply density limiting by keeping top priority annotations."""
    if len(annotations) <= density_limit:
        return annotations

    sorted_annots = sorted(annotations, key=lambda a: a.priority, reverse=True)
    return sorted_annots[:density_limit]


def _initialize_placements(annotations: list[Annotation]) -> list[PlacedAnnotation]:
    """Initialize placed annotations at anchor points."""
    return [
        PlacedAnnotation(
            annotation=annot,
            display_x=annot.x,
            display_y=annot.y,
            visible=True,
            needs_leader=False,
        )
        for annot in annotations
    ]


def _resolve_collisions(
    placed: list[PlacedAnnotation],
    collision_threshold: float,
    max_iterations: int,
) -> None:
    """Resolve collisions using iterative adjustment."""
    for _iteration in range(max_iterations):
        moved = False

        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                if _check_collision(placed[i], placed[j], collision_threshold):
                    # Move lower-priority annotation
                    if placed[i].annotation.priority >= placed[j].annotation.priority:
                        moved = _move_annotation(placed[j], placed[i], collision_threshold) or moved
                    else:
                        moved = _move_annotation(placed[i], placed[j], collision_threshold) or moved

        if not moved:
            break


def _add_leader_lines(placed: list[PlacedAnnotation], leader_threshold: float) -> None:
    """Add leader lines for displaced annotations."""
    for p in placed:
        dx = abs(p.display_x - p.annotation.x)
        dy = abs(p.display_y - p.annotation.y)
        displacement = np.sqrt(dx**2 + dy**2)

        if displacement > leader_threshold:
            p.needs_leader = True
            p.leader_points = generate_leader_line(
                (p.annotation.x, p.annotation.y),
                (p.display_x, p.display_y),
            )


def _check_collision(
    p1: PlacedAnnotation,
    p2: PlacedAnnotation,
    threshold: float,
) -> bool:
    """Check if two annotations collide.

    Args:
        p1: First annotation
        p2: Second annotation
        threshold: Minimum spacing threshold

    Returns:
        True if annotations collide
    """
    # Bounding box collision detection
    dx = abs(p2.display_x - p1.display_x)
    dy = abs(p2.display_y - p1.display_y)

    # Minimum separation (sum of half-widths + threshold)
    min_dx = (p1.annotation.bbox_width + p2.annotation.bbox_width) / 2 + threshold
    min_dy = (p1.annotation.bbox_height + p2.annotation.bbox_height) / 2 + threshold

    return dx < min_dx and dy < min_dy


def _move_annotation(
    to_move: PlacedAnnotation,
    fixed: PlacedAnnotation,
    threshold: float,
) -> bool:
    """Move annotation away from collision.

    Args:
        to_move: Annotation to move
        fixed: Fixed annotation to move away from
        threshold: Minimum spacing

    Returns:
        True if annotation was moved
    """
    dx = to_move.display_x - fixed.display_x
    dy = to_move.display_y - fixed.display_y

    distance = np.sqrt(dx**2 + dy**2)

    if distance < 1e-6:
        # Randomize if overlapping exactly
        dx = np.random.randn() * 10
        dy = np.random.randn() * 10
        distance = np.sqrt(dx**2 + dy**2)

    # Required separation
    min_dx = (to_move.annotation.bbox_width + fixed.annotation.bbox_width) / 2 + threshold
    min_dy = (to_move.annotation.bbox_height + fixed.annotation.bbox_height) / 2 + threshold
    min_dist = np.sqrt(min_dx**2 + min_dy**2)

    # Move away if too close
    if distance < min_dist:
        # Move proportionally to required distance
        scale = min_dist / distance
        new_x = fixed.display_x + dx * scale
        new_y = fixed.display_y + dy * scale

        # Apply with damping to avoid oscillation
        damping = 0.5
        to_move.display_x += (new_x - to_move.display_x) * damping
        to_move.display_y += (new_y - to_move.display_y) * damping

        return True

    return False


def filter_by_zoom_level(
    placed: list[PlacedAnnotation],
    zoom_range: tuple[float, float],
    *,
    min_width_for_display: float = 0.1,
) -> list[PlacedAnnotation]:
    """Filter annotations based on zoom level.

    Hide annotations when zoom range is too large for readability.

    Args:
        placed: List of placed annotations.
        zoom_range: Current zoom range (x_min, x_max).
        min_width_for_display: Minimum zoom width to display annotations.

    Returns:
        Filtered list with visibility updated.

    Example:
        >>> # Hide annotations when zoomed out too far
        >>> filtered = filter_by_zoom_level(placed, (0, 1000), min_width_for_display=1.0)

    References:
        VIS-016: Annotation Placement Intelligence (dynamic hiding)
    """
    x_min, x_max = zoom_range
    zoom_width = x_max - x_min

    result = []
    for p in placed:
        # Update visibility based on zoom level
        if zoom_width < min_width_for_display:
            p.visible = True
        else:
            # Hide if outside viewport or too zoomed out
            in_viewport = x_min <= p.annotation.x <= x_max
            p.visible = in_viewport

        result.append(p)

    return result


def create_priority_annotation(  # type: ignore[no-untyped-def]
    text: str,
    x: float,
    y: float,
    *,
    importance: str = "normal",
    **kwargs,
) -> Annotation:
    """Create annotation with priority based on importance level.

    Args:
        text: Annotation text.
        x: X position in data units.
        y: Y position in data units.
        importance: Importance level ("critical", "high", "normal", "low").
        **kwargs: Additional Annotation parameters.

    Returns:
        Annotation with appropriate priority.

    Example:
        >>> peak_annot = create_priority_annotation(
        ...     "Critical Peak", 5.0, 1.0, importance="critical"
        ... )

    References:
        VIS-016: Annotation Placement Intelligence (priority-based positioning)
    """
    priority_map = {
        "critical": 1.0,
        "high": 0.8,
        "normal": 0.5,
        "low": 0.2,
    }

    priority = priority_map.get(importance, 0.5)

    return Annotation(
        text=text,
        x=x,
        y=y,
        priority=priority,
        **kwargs,
    )


__all__ = [
    "Annotation",
    "PlacedAnnotation",
    "create_priority_annotation",
    "filter_by_zoom_level",
    "place_annotations",
]
