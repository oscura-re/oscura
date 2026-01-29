"""Visualization layout functions for multi-channel plots and annotation placement.

This module provides intelligent layout algorithms for stacking multiple
channels and optimizing annotation placement with collision avoidance.


Example:
    >>> from oscura.visualization.layout import layout_stacked_channels
    >>> layout = layout_stacked_channels(n_channels=4, figsize=(10, 8))
    >>> print(f"Channel heights: {layout['heights']}")

References:
    - Force-directed graph layout (Fruchterman-Reingold)
    - Constrained layout solver for equal spacing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from oscura.utils.geometry import generate_leader_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ChannelLayout:
    """Layout specification for stacked channels.

    Attributes:
        n_channels: Number of channels to stack
        heights: Array of subplot heights (normalized 0-1)
        gaps: Array of gap sizes between channels (normalized 0-1)
        y_positions: Array of Y positions for each channel (normalized 0-1)
        shared_x: Whether channels share X-axis
        figsize: Figure size (width, height) in inches
    """

    n_channels: int
    heights: NDArray[np.float64]
    gaps: NDArray[np.float64]
    y_positions: NDArray[np.float64]
    shared_x: bool
    figsize: tuple[float, float]


@dataclass
class Annotation:
    """Annotation specification with position and bounding box.

    Attributes:
        text: Annotation text
        x: X coordinate in data units
        y: Y coordinate in data units
        bbox_width: Bounding box width in display units
        bbox_height: Bounding box height in display units
        priority: Priority for placement (0-1, higher is more important)
        anchor: Preferred anchor position ("top", "bottom", "left", "right", "auto")
    """

    text: str
    x: float
    y: float
    bbox_width: float = 50.0
    bbox_height: float = 20.0
    priority: float = 0.5
    anchor: str = "auto"


@dataclass
class PlacedAnnotation:
    """Annotation with optimized placement.

    Attributes:
        annotation: Original annotation
        display_x: Optimized X position in display units
        display_y: Optimized Y position in display units
        needs_leader: Whether a leader line is needed
        leader_points: Points for leader line (if needed)
    """

    annotation: Annotation
    display_x: float
    display_y: float
    needs_leader: bool
    leader_points: list[tuple[float, float]] | None = None


def layout_stacked_channels(
    n_channels: int,
    *,
    figsize: tuple[float, float] = (10, 8),
    gap_ratio: float = 0.1,
    shared_x: bool = True,
) -> ChannelLayout:
    """Calculate equal vertical spacing for stacked multi-channel plots.

    Implements constrained layout solver for equal spacing with configurable
    gaps between channels, ensuring proper vertical alignment.

    Args:
        n_channels: Number of channels to stack.
        figsize: Figure size (width, height) in inches.
        gap_ratio: Ratio of gap to channel height (default 0.1 = 10%).
        shared_x: Whether channels share X-axis (affects bottom margin).

    Returns:
        ChannelLayout with heights, gaps, and positions.

    Raises:
        ValueError: If n_channels < 1 or gap_ratio invalid.

    Example:
        >>> layout = layout_stacked_channels(n_channels=3, gap_ratio=0.1)
        >>> print(f"Channel 0 position: {layout.y_positions[0]:.3f}")

    References:
        VIS-015: Multi-Channel Stack Optimization
    """
    if n_channels < 1:
        raise ValueError("n_channels must be >= 1")

    if gap_ratio < 0 or gap_ratio > 1:
        raise ValueError(f"gap_ratio must be in [0, 1], got {gap_ratio}")

    # Total available height (normalized to 1.0)
    # Reserve space for margins
    top_margin = 0.05
    bottom_margin = 0.1 if shared_x else 0.05
    available_height = 1.0 - top_margin - bottom_margin

    # Calculate channel height with gaps
    # Total height = n_channels * h + (n_channels - 1) * gap
    # where gap = gap_ratio * h
    # Solving: available_height = n_channels * h + (n_channels - 1) * gap_ratio * h
    #         = h * (n_channels + (n_channels - 1) * gap_ratio)
    denominator = n_channels + (n_channels - 1) * gap_ratio
    channel_height = available_height / denominator
    gap_height = channel_height * gap_ratio

    # Calculate heights and gaps arrays
    heights = np.full(n_channels, channel_height, dtype=np.float64)
    gaps = np.full(n_channels - 1, gap_height, dtype=np.float64) if n_channels > 1 else np.array([])

    # Calculate Y positions (from bottom)
    y_positions = np.zeros(n_channels, dtype=np.float64)
    current_y = bottom_margin

    for i in range(n_channels):
        # Channels are indexed from bottom to top
        y_positions[i] = current_y
        current_y += channel_height
        if i < n_channels - 1:
            current_y += gap_height

    return ChannelLayout(
        n_channels=n_channels,
        heights=heights,
        gaps=gaps,
        y_positions=y_positions,
        shared_x=shared_x,
        figsize=figsize,
    )


def _initialize_placed_annotations(
    annotations: list[Annotation],
) -> list[PlacedAnnotation]:
    """Initialize placed annotations at anchor points.

    Args:
        annotations: List of annotations to place.

    Returns:
        List of PlacedAnnotation initially at anchor points.
    """
    placed = []
    for annot in annotations:
        placed.append(
            PlacedAnnotation(
                annotation=annot,
                display_x=annot.x,
                display_y=annot.y,
                needs_leader=False,
                leader_points=None,
            )
        )
    return placed


def _calculate_repulsive_force(
    placed_i: PlacedAnnotation,
    placed_j: PlacedAnnotation,
    min_spacing: float,
    repulsion_strength: float,
) -> tuple[float, float]:
    """Calculate repulsive force between two annotations.

    Args:
        placed_i: First annotation.
        placed_j: Second annotation.
        min_spacing: Minimum spacing in pixels.
        repulsion_strength: Repulsive force strength.

    Returns:
        Tuple of (fx, fy) force components.
    """
    # Check for bounding box overlap
    dx = placed_j.display_x - placed_i.display_x
    dy = placed_j.display_y - placed_i.display_y

    # Bounding box sizes
    w1 = placed_i.annotation.bbox_width
    h1 = placed_i.annotation.bbox_height
    w2 = placed_j.annotation.bbox_width
    h2 = placed_j.annotation.bbox_height

    # Minimum separation (sum of half-widths + spacing)
    min_dx = (w1 + w2) / 2 + min_spacing
    min_dy = (h1 + h2) / 2 + min_spacing

    # Check if overlapping
    if abs(dx) < min_dx and abs(dy) < min_dy:
        # Calculate repulsive force
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 1e-6:
            # Avoid division by zero
            distance = 1e-6
            dx = np.random.randn() * 0.1
            dy = np.random.randn() * 0.1

        # Repulsion inversely proportional to distance
        force = repulsion_strength / distance

        # Return force in direction away from overlap
        return -force * dx / distance, -force * dy / distance

    return 0.0, 0.0


def _apply_force_iteration(
    placed: list[PlacedAnnotation],
    display_width: float,
    display_height: float,
    min_spacing: float,
    repulsion_strength: float,
) -> bool:
    """Apply one iteration of force-directed layout.

    Args:
        placed: List of placed annotations to update.
        display_width: Display width for clamping.
        display_height: Display height for clamping.
        min_spacing: Minimum spacing in pixels.
        repulsion_strength: Repulsive force strength.

    Returns:
        True if any annotation moved significantly.
    """
    moved = False

    for i in range(len(placed)):
        fx = 0.0
        fy = 0.0

        # Calculate forces from all other annotations
        for j in range(len(placed)):
            if i != j:
                force_x, force_y = _calculate_repulsive_force(
                    placed[i], placed[j], min_spacing, repulsion_strength
                )
                fx += force_x
                fy += force_y

        # Apply forces with damping (priority affects inertia)
        damping = 0.5
        priority_factor = 1.0 - placed[i].annotation.priority
        step_size = damping * priority_factor

        new_x = placed[i].display_x + fx * step_size
        new_y = placed[i].display_y + fy * step_size

        # Clamp to display bounds
        new_x = np.clip(new_x, 0, display_width)
        new_y = np.clip(new_y, 0, display_height)

        # Update if moved significantly
        if abs(new_x - placed[i].display_x) > 0.1 or abs(new_y - placed[i].display_y) > 0.1:
            placed[i] = PlacedAnnotation(
                annotation=placed[i].annotation,
                display_x=new_x,
                display_y=new_y,
                needs_leader=False,
                leader_points=None,
            )
            moved = True

    return moved


def _add_leader_lines(placed: list[PlacedAnnotation], leader_threshold: float = 20.0) -> None:
    """Add leader lines to annotations displaced from anchor points.

    Args:
        placed: List of placed annotations to update in-place.
        leader_threshold: Displacement threshold for leader line in pixels.
    """
    for i, p in enumerate(placed):
        anchor_x = p.annotation.x
        anchor_y = p.annotation.y

        displacement = np.sqrt((p.display_x - anchor_x) ** 2 + (p.display_y - anchor_y) ** 2)

        if displacement > leader_threshold:
            # Generate simple orthogonal leader line
            leader_points = generate_leader_line(
                (anchor_x, anchor_y),
                (p.display_x, p.display_y),
            )

            placed[i] = PlacedAnnotation(
                annotation=p.annotation,
                display_x=p.display_x,
                display_y=p.display_y,
                needs_leader=True,
                leader_points=leader_points,
            )


def optimize_annotation_placement(
    annotations: list[Annotation],
    *,
    display_width: float = 800.0,
    display_height: float = 600.0,
    max_iterations: int = 100,
    repulsion_strength: float = 10.0,
    min_spacing: float = 5.0,
) -> list[PlacedAnnotation]:
    """Optimize annotation placement with collision avoidance.

    Uses force-directed layout algorithm to separate overlapping labels
    with repulsive forces. Generates leader lines when labels must be
    displaced from anchor points.

    Args:
        annotations: List of annotations to place.
        display_width: Display area width in pixels.
        display_height: Display area height in pixels.
        max_iterations: Maximum iterations for force-directed layout.
        repulsion_strength: Strength of repulsive force between overlapping labels.
        min_spacing: Minimum spacing between annotations in pixels.

    Returns:
        List of PlacedAnnotation with optimized positions.

    Raises:
        ValueError: If annotations list is empty.

    Example:
        >>> annots = [Annotation("Peak", 0.5, 1.0, priority=0.9)]
        >>> placed = optimize_annotation_placement(annots)
        >>> print(f"Needs leader: {placed[0].needs_leader}")

    References:
        VIS-016: Annotation Placement Intelligence
        Force-directed graph layout (Fruchterman-Reingold)
    """
    if len(annotations) == 0:
        raise ValueError("annotations list cannot be empty")

    # Data preparation - initialize at anchor points
    placed = _initialize_placed_annotations(annotations)

    # Force-directed layout iterations
    for _iteration in range(max_iterations):
        moved = _apply_force_iteration(
            placed, display_width, display_height, min_spacing, repulsion_strength
        )

        # Converged if nothing moved
        if not moved:
            break

    # Annotation - add leader lines for displaced annotations
    _add_leader_lines(placed, leader_threshold=20.0)

    return placed


__all__ = [
    "Annotation",
    "ChannelLayout",
    "PlacedAnnotation",
    "layout_stacked_channels",
    "optimize_annotation_placement",
]
