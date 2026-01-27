"""Geometric utility functions for visualization.

This module provides geometric calculation utilities used across visualization modules.
"""


def generate_leader_line(
    anchor: tuple[float, float],
    label: tuple[float, float],
) -> list[tuple[float, float]]:
    """Generate orthogonal leader line from anchor to label.

    Creates an L-shaped leader line connecting an anchor point to a label position.
    The line is orthogonal (horizontal then vertical or vice versa).

    Args:
        anchor: Anchor point (x, y)
        label: Label position (x, y)

    Returns:
        List of points for leader line [(x1, y1), (x2, y2), ...]

    Example:
        >>> generate_leader_line((0, 0), (10, 5))
        [(0, 0), (10, 0), (10, 5)]
    """
    ax, ay = anchor
    lx, ly = label

    # Create L-shaped line: horizontal then vertical
    return [(ax, ay), (lx, ay), (lx, ly)]
