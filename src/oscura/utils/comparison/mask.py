"""Mask testing for Oscura.

This module provides mask-based pass/fail testing for waveforms,
including eye diagram masks and custom polygon masks.


Example:
    >>> from oscura.utils.comparison import mask_test, eye_mask
    >>> mask = eye_mask(0.5, 0.4, 0.3)
    >>> result = mask_test(trace, mask)

References:
    IEEE 802.3: Ethernet eye diagram mask specifications
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from oscura.core.types import WaveformTrace


@dataclass
class MaskRegion:
    """A region in a mask definition.

    Represents a polygon region that waveform data must avoid
    (violation region) or must stay within (boundary region).

    Attributes:
        vertices: List of (x, y) vertices defining the polygon.
        region_type: "violation" (must avoid) or "boundary" (must stay within).
        name: Optional name for the region.
    """

    vertices: list[tuple[float, float]]
    region_type: Literal["violation", "boundary"] = "violation"
    name: str = ""

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the polygon.

        Uses ray casting algorithm for point-in-polygon test.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if point is inside the polygon.
        """
        n = len(self.vertices)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside


@dataclass
class Mask:
    """Mask definition for waveform testing.

    A mask consists of one or more regions that define pass/fail criteria
    for waveform data.

    Attributes:
        regions: List of MaskRegion polygons.
        name: Name of the mask.
        x_unit: Unit for X axis (e.g., "UI", "ns", "samples").
        y_unit: Unit for Y axis (e.g., "V", "mV", "normalized").
        description: Optional description.
    """

    regions: list[MaskRegion] = field(default_factory=list)
    name: str = "mask"
    x_unit: str = "UI"
    y_unit: str = "V"
    description: str = ""

    def add_region(
        self,
        vertices: list[tuple[float, float]],
        region_type: Literal["violation", "boundary"] = "violation",
        name: str = "",
    ) -> None:
        """Add a region to the mask.

        Args:
            vertices: List of (x, y) vertices.
            region_type: "violation" or "boundary".
            name: Optional region name.
        """
        self.regions.append(MaskRegion(vertices, region_type, name))


@dataclass
class MaskTestResult:
    """Result of a mask test.

    Attributes:
        passed: True if all samples pass the mask test.
        num_violations: Number of samples violating the mask.
        violation_rate: Fraction of samples violating the mask.
        violation_points: List of (x, y) coordinates that violated.
        violations_by_region: Count of violations per region.
        margin: Estimated margin to mask boundary.
    """

    passed: bool
    num_violations: int
    violation_rate: float
    violation_points: list[tuple[float, float]] = field(default_factory=list)
    violations_by_region: dict[str, int] = field(default_factory=dict)
    margin: float | None = None


def create_mask(
    regions: list[dict],  # type: ignore[type-arg]
    *,
    name: str = "custom_mask",
    x_unit: str = "samples",
    y_unit: str = "V",
) -> Mask:
    """Create a mask from region definitions.

    Args:
        regions: List of region dicts with 'vertices' and optional
            'type' and 'name' keys.
        name: Mask name.
        x_unit: X axis unit.
        y_unit: Y axis unit.

    Returns:
        Mask instance.

    Example:
        >>> mask = create_mask([
        ...     {"vertices": [(0, 0.5), (0.5, 0.5), (0.5, -0.5), (0, -0.5)],
        ...      "type": "violation", "name": "center"}
        ... ])
    """
    mask = Mask(name=name, x_unit=x_unit, y_unit=y_unit)

    for region in regions:
        vertices = region["vertices"]
        region_type = region.get("type", "violation")
        region_name = region.get("name", "")
        mask.add_region(vertices, region_type, region_name)

    return mask


def eye_mask(
    eye_width: float = 0.5,
    eye_height: float = 0.4,
    center_height: float = 0.3,
    *,
    x_margin: float = 0.0,
    y_margin: float = 0.1,
    unit_interval: float = 1.0,
    amplitude: float = 1.0,
) -> Mask:
    """Create a standard eye diagram mask.

    Creates a hexagonal eye mask with center violation region and
    optional boundary regions based on eye opening parameters.

    Args:
        eye_width: Width of eye opening (fraction of UI).
        eye_height: Height of eye opening (fraction of amplitude).
        center_height: Height of center violation region.
        x_margin: X margin for boundary (fraction of UI). Reserved for future use.
        y_margin: Y margin for boundary (fraction of amplitude).
        unit_interval: Duration of unit interval.
        amplitude: Signal amplitude.

    Returns:
        Mask for eye diagram testing.

    Example:
        >>> mask = eye_mask(0.5, 0.4)  # Standard 50% width, 40% height
        >>> # Creates violation region in center of eye
    """
    mask = Mask(
        name="eye_mask",
        x_unit="UI",
        y_unit="normalized",
        description=f"Eye mask: {eye_width * 100:.0f}% width, {eye_height * 100:.0f}% height",
    )

    # Scale parameters
    ui = unit_interval
    amp = amplitude

    # Center violation region (hexagonal)
    # Points arranged clockwise from left
    center_width = eye_width * ui
    center_top = eye_height * amp / 2
    center_bottom = -eye_height * amp / 2
    mid_width = center_width * 0.7  # Narrower at top/bottom

    center_vertices = [
        (-center_width / 2, 0),  # Left
        (-mid_width / 2, center_top),  # Upper left
        (mid_width / 2, center_top),  # Upper right
        (center_width / 2, 0),  # Right
        (mid_width / 2, center_bottom),  # Lower right
        (-mid_width / 2, center_bottom),  # Lower left
    ]
    mask.add_region(center_vertices, "violation", "eye_center")

    # Top violation region (above eye)
    top_y = amp / 2 + y_margin * amp
    top_vertices = [
        (-ui / 2, center_top + center_height * amp),
        (ui / 2, center_top + center_height * amp),
        (ui / 2, top_y),
        (-ui / 2, top_y),
    ]
    mask.add_region(top_vertices, "violation", "top")

    # Bottom violation region (below eye)
    bottom_y = -amp / 2 - y_margin * amp
    bottom_vertices = [
        (-ui / 2, bottom_y),
        (ui / 2, bottom_y),
        (ui / 2, center_bottom - center_height * amp),
        (-ui / 2, center_bottom - center_height * amp),
    ]
    mask.add_region(bottom_vertices, "violation", "bottom")

    return mask


def mask_test(
    trace: WaveformTrace,
    mask: Mask,
    *,
    x_data: NDArray[np.floating[Any]] | None = None,
    normalize: bool = True,
    sample_rate: float | None = None,
) -> MaskTestResult:
    """Test waveform against a mask.

    Checks if any samples of the waveform violate the mask regions.

    Args:
        trace: Input waveform trace.
        mask: Mask to test against.
        x_data: X coordinates for each sample (if different from time).
        normalize: Normalize Y data to [-1, 1] range.
        sample_rate: Sample rate override.

    Returns:
        MaskTestResult with pass/fail status and violation details.

    Example:
        >>> result = mask_test(eye_trace, mask)
        >>> print(f"Violations: {result.num_violations}")
    """
    # Prepare test data
    x_arr, y_arr = _prepare_mask_test_data(trace, x_data, normalize)

    # Find all violations
    violations, violations_by_region = _find_mask_violations(x_arr, y_arr, mask)

    # Calculate statistics
    unique_violations = list(set(violations))
    num_violations = len(unique_violations)
    violation_rate = num_violations / len(y_arr) if len(y_arr) > 0 else 0.0

    # Calculate margin to mask boundary
    margin = _calculate_mask_margin(x_arr, y_arr, mask, num_violations)

    return MaskTestResult(
        passed=num_violations == 0,
        num_violations=num_violations,
        violation_rate=violation_rate,
        violation_points=unique_violations,
        violations_by_region=violations_by_region,
        margin=margin,
    )


def _prepare_mask_test_data(
    trace: WaveformTrace,
    x_data: NDArray[np.floating[Any]] | None,
    normalize: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Prepare X and Y data arrays for mask testing.

    Args:
        trace: Waveform trace.
        x_data: Optional X coordinates (defaults to sample indices).
        normalize: Whether to normalize Y data to [-1, 1].

    Returns:
        Tuple of (x_array, y_array) as float64.
    """
    y_data = trace.data.astype(np.float64)

    # Generate or use provided X data
    if x_data is None:
        x_arr = np.arange(len(y_data), dtype=np.float64)
    else:
        x_arr = x_data

    # Normalize Y data if requested
    if normalize:
        y_min, y_max = np.min(y_data), np.max(y_data)
        if y_max - y_min > 0:
            y_data = 2 * (y_data - y_min) / (y_max - y_min) - 1

    return (x_arr, y_data)


def _find_mask_violations(
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    mask: Mask,
) -> tuple[list[tuple[float, float]], dict[str, int]]:
    """Find all mask violations in waveform data.

    Args:
        x_data: X coordinates.
        y_data: Y coordinates.
        mask: Mask with regions to test.

    Returns:
        Tuple of (violation_points, violations_by_region_name).
    """
    violations: list[tuple[float, float]] = []
    violations_by_region: dict[str, int] = {}

    for region in mask.regions:
        region_name = region.name or "unnamed"
        violations_by_region[region_name] = 0

        if region.region_type == "violation":
            # Points inside violation region are violations
            region_violations = _check_violation_region(x_data, y_data, region)
            violations.extend(region_violations)
            violations_by_region[region_name] = len(region_violations)
        elif region.region_type == "boundary":
            # Points outside boundary region are violations
            region_violations = _check_boundary_region(x_data, y_data, region)
            violations.extend(region_violations)
            violations_by_region[region_name] = len(region_violations)

    return (violations, violations_by_region)


def _check_violation_region(
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    region: MaskRegion,
) -> list[tuple[float, float]]:
    """Check for points inside violation region.

    Args:
        x_data: X coordinates.
        y_data: Y coordinates.
        region: Violation region.

    Returns:
        List of (x, y) points violating this region.
    """
    violations = []
    for x, y in zip(x_data, y_data, strict=False):
        if region.contains_point(float(x), float(y)):
            violations.append((float(x), float(y)))
    return violations


def _check_boundary_region(
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    region: MaskRegion,
) -> list[tuple[float, float]]:
    """Check for points outside boundary region.

    Args:
        x_data: X coordinates.
        y_data: Y coordinates.
        region: Boundary region.

    Returns:
        List of (x, y) points violating this region.
    """
    violations = []
    for x, y in zip(x_data, y_data, strict=False):
        if not region.contains_point(float(x), float(y)):
            violations.append((float(x), float(y)))
    return violations


def _calculate_mask_margin(
    x_data: NDArray[np.float64],
    y_data: NDArray[np.float64],
    mask: Mask,
    num_violations: int,
) -> float | None:
    """Calculate margin to nearest mask edge.

    Only calculated when there are no violations.

    Args:
        x_data: X coordinates.
        y_data: Y coordinates.
        mask: Mask with regions.
        num_violations: Number of violations found.

    Returns:
        Minimum distance to mask boundary, or None if violations exist or no regions.
    """
    if num_violations > 0 or not mask.regions:
        return None

    # Find minimum distance to any violation region edge
    min_dist = float("inf")
    for region in mask.regions:
        if region.region_type == "violation":
            for x, y in zip(x_data, y_data, strict=False):
                for i in range(len(region.vertices)):
                    x1, y1 = region.vertices[i]
                    x2, y2 = region.vertices[(i + 1) % len(region.vertices)]
                    dist = _point_to_segment_distance(x, y, x1, y1, x2, y2)
                    min_dist = min(min_dist, dist)

    return min_dist if min_dist != float("inf") else None


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    """Calculate distance from point to line segment."""
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq == 0:
        # Segment is a point
        return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)  # type: ignore[no-any-return]

    # Project point onto line
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return float(np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2))


def eye_diagram_mask_test(
    eye_data: NDArray[np.floating[Any]],
    *,
    eye_width: float = 0.5,
    eye_height: float = 0.4,
    unit_interval: float = 1.0,
) -> MaskTestResult:
    """Specialized eye diagram mask test.

    Tests 2D eye diagram data against a standard eye mask.

    Args:
        eye_data: 2D array of shape (num_traces, samples_per_ui).
        eye_width: Eye opening width (fraction of UI).
        eye_height: Eye opening height (fraction of amplitude).
        unit_interval: Duration of unit interval in samples.

    Returns:
        MaskTestResult for the eye diagram.

    Raises:
        AnalysisError: If eye data is not a 2D array.
    """
    if eye_data.ndim != 2:
        raise AnalysisError("Eye data must be 2D array (num_traces x samples_per_ui)")

    num_traces, samples_per_ui = eye_data.shape

    # Create mask
    mask = eye_mask(
        eye_width=eye_width,
        eye_height=eye_height,
        unit_interval=unit_interval,
        amplitude=1.0,
    )

    # Normalize data
    flat_data = eye_data.flatten()
    y_min, y_max = np.min(flat_data), np.max(flat_data)
    normalized = 2 * (eye_data - y_min) / (y_max - y_min) - 1 if y_max - y_min > 0 else eye_data

    # Create X coordinates (relative to UI center)
    x_coords = np.linspace(-0.5, 0.5, samples_per_ui) * unit_interval

    # Test all traces
    violations: list[tuple[float, float]] = []
    violations_by_region: dict[str, int] = {r.name or "unnamed": 0 for r in mask.regions}

    for trace_idx in range(num_traces):
        for sample_idx in range(samples_per_ui):
            x = float(x_coords[sample_idx])
            y = float(normalized[trace_idx, sample_idx])

            for region in mask.regions:
                if region.region_type == "violation":
                    if region.contains_point(x, y):
                        violations.append((x, y))
                        region_name = region.name or "unnamed"
                        violations_by_region[region_name] += 1

    unique_violations = list(set(violations))
    num_violations = len(unique_violations)
    total_points = num_traces * samples_per_ui

    return MaskTestResult(
        passed=num_violations == 0,
        num_violations=num_violations,
        violation_rate=num_violations / total_points if total_points > 0 else 0,
        violation_points=unique_violations,
        violations_by_region=violations_by_region,
        margin=None,
    )
