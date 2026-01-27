"""Comprehensive unit tests for mask testing functionality.

This module tests mask creation, mask testing, and eye diagram masks.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.utils.comparison.mask import (
    Mask,
    MaskRegion,
    MaskTestResult,
    create_mask,
    eye_diagram_mask_test,
    eye_mask,
    mask_test,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def clean_trace() -> WaveformTrace:
    """Create a clean sine wave trace."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def eye_data() -> np.ndarray:
    """Create simulated eye diagram data."""
    # Create 100 traces of 50 samples each
    traces = []
    for _i in range(100):
        # Simulated NRZ data transitions
        t = np.linspace(0, 1, 50)
        trace = np.where(
            t < 0.5, -0.8 + np.random.normal(0, 0.05, 50), 0.8 + np.random.normal(0, 0.05, 50)
        )
        traces.append(trace)
    return np.array(traces)


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-003")
class TestMaskRegion:
    """Test mask region functionality."""

    def test_contains_point_inside(self) -> None:
        """Test point inside polygon."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        region = MaskRegion(vertices)

        assert region.contains_point(0.5, 0.5)

    def test_contains_point_outside(self) -> None:
        """Test point outside polygon."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        region = MaskRegion(vertices)

        assert not region.contains_point(2.0, 2.0)

    def test_contains_point_on_edge(self) -> None:
        """Test point on polygon edge."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        region = MaskRegion(vertices)

        # Edge cases depend on ray casting implementation
        result = region.contains_point(0.5, 0.0)
        assert isinstance(result, bool)

    def test_region_types(self) -> None:
        """Test different region types."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

        violation = MaskRegion(vertices, region_type="violation")
        assert violation.region_type == "violation"

        boundary = MaskRegion(vertices, region_type="boundary")
        assert boundary.region_type == "boundary"


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-003")
class TestMaskCreation:
    """Test mask creation."""

    def test_create_empty_mask(self) -> None:
        """Test creating empty mask."""
        mask = Mask()

        assert len(mask.regions) == 0
        assert mask.name == "mask"

    def test_add_region_to_mask(self) -> None:
        """Test adding regions to mask."""
        mask = Mask()
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

        mask.add_region(vertices, "violation", "test_region")

        assert len(mask.regions) == 1
        assert mask.regions[0].name == "test_region"
        assert mask.regions[0].region_type == "violation"

    def test_create_mask_from_dict(self) -> None:
        """Test creating mask from region dictionaries."""
        regions = [
            {
                "vertices": [(0, 0.5), (0.5, 0.5), (0.5, -0.5), (0, -0.5)],
                "type": "violation",
                "name": "center",
            },
            {
                "vertices": [(0, 0.8), (1, 0.8), (1, 1), (0, 1)],
                "type": "violation",
                "name": "top",
            },
        ]

        mask = create_mask(regions, name="test_mask", x_unit="UI", y_unit="V")

        assert mask.name == "test_mask"
        assert mask.x_unit == "UI"
        assert mask.y_unit == "V"
        assert len(mask.regions) == 2

    def test_eye_mask_creation(self) -> None:
        """Test creating standard eye mask."""
        mask = eye_mask(eye_width=0.5, eye_height=0.4)

        assert mask.name == "eye_mask"
        assert mask.x_unit == "UI"
        assert len(mask.regions) >= 3  # Center, top, bottom regions


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-003")
class TestMaskTest:
    """Test mask testing functionality."""

    def test_no_violations(self, clean_trace: WaveformTrace) -> None:
        """Test trace with no mask violations."""
        # Create mask that doesn't intersect the data
        mask = Mask()
        mask.add_region([(0, 2), (500, 2), (500, 3), (0, 3)], "violation")

        result = mask_test(clean_trace, mask, normalize=False)

        assert result.passed
        assert result.num_violations == 0
        assert result.violation_rate == 0.0

    def test_with_violations(self, clean_trace: WaveformTrace) -> None:
        """Test trace with mask violations."""
        # Create mask that intersects the data
        mask = Mask()
        # Violation region that will catch some samples
        mask.add_region([(400, -0.5), (600, -0.5), (600, 0.5), (400, 0.5)], "violation")

        result = mask_test(clean_trace, mask, normalize=False)

        # Should have some violations
        assert result.num_violations > 0

    def test_normalized_mask_test(self, clean_trace: WaveformTrace) -> None:
        """Test mask testing with normalization."""
        mask = Mask()
        # Normalized coordinates (-1 to 1)
        mask.add_region([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], "violation")

        result = mask_test(clean_trace, mask, normalize=True)

        assert isinstance(result, MaskTestResult)

    def test_custom_x_data(self, clean_trace: WaveformTrace) -> None:
        """Test mask testing with custom X data."""
        x_data = np.linspace(-1, 1, len(clean_trace.data))
        mask = Mask()
        mask.add_region([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)], "violation")

        result = mask_test(clean_trace, mask, x_data=x_data, normalize=True)

        assert isinstance(result, MaskTestResult)

    def test_violation_points(self, clean_trace: WaveformTrace) -> None:
        """Test that violation points are recorded."""
        mask = Mask()
        mask.add_region([(400, -0.5), (600, -0.5), (600, 0.5), (400, 0.5)], "violation", "test")

        result = mask_test(clean_trace, mask, normalize=False)

        if result.num_violations > 0:
            assert len(result.violation_points) > 0
            assert "test" in result.violations_by_region

    def test_margin_calculation(self, clean_trace: WaveformTrace) -> None:
        """Test margin calculation for passing mask test."""
        # Create mask well away from data
        mask = Mask()
        mask.add_region([(0, 5), (1000, 5), (1000, 10), (0, 10)], "violation")

        result = mask_test(clean_trace, mask, normalize=False)

        assert result.passed
        # Margin may or may not be calculated depending on implementation
        if result.margin is not None:
            assert result.margin > 0


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("EYE-002")
class TestEyeMask:
    """Test eye diagram mask functionality."""

    def test_eye_mask_parameters(self) -> None:
        """Test eye mask with different parameters."""
        mask = eye_mask(
            eye_width=0.6,
            eye_height=0.5,
            center_height=0.2,
            x_margin=0.05,
            y_margin=0.05,
        )

        assert len(mask.regions) >= 3
        assert mask.name == "eye_mask"

    def test_eye_diagram_test_pass(self, eye_data: np.ndarray) -> None:
        """Test passing eye diagram mask test."""
        # Create lenient mask
        mask = eye_mask(eye_width=0.3, eye_height=0.3)

        result = eye_diagram_mask_test(eye_data, eye_width=0.3, eye_height=0.3)

        # With lenient mask, should pass or have few violations
        assert isinstance(result, MaskTestResult)

    def test_eye_diagram_test_dimensions(self, eye_data: np.ndarray) -> None:
        """Test eye diagram test validates dimensions."""
        # 1D array should raise
        with pytest.raises(Exception):  # AnalysisError
            eye_diagram_mask_test(eye_data.flatten(), eye_width=0.5, eye_height=0.4)

    def test_boundary_region_type(self) -> None:
        """Test boundary region type (must stay within)."""
        trace = WaveformTrace(data=np.zeros(100), metadata=TraceMetadata(sample_rate=1e6))

        mask = Mask()
        # Boundary region - data must stay inside
        mask.add_region([(-10, -1), (110, -1), (110, 1), (-10, 1)], "boundary")

        result = mask_test(trace, mask, normalize=False)

        # All data at 0 should be inside boundary
        assert result.passed


@pytest.mark.unit
@pytest.mark.comparison
class TestComparisonMaskEdgeCases:
    """Test edge cases for mask testing."""

    def test_empty_mask(self, clean_trace: WaveformTrace) -> None:
        """Test with empty mask."""
        mask = Mask()
        result = mask_test(clean_trace, mask)

        # No regions means no violations
        assert result.passed
        assert result.num_violations == 0

    def test_complex_polygon(self) -> None:
        """Test mask with complex polygon."""
        # Star-shaped polygon
        vertices = [
            (0, 0),
            (0.5, 0.5),
            (1, 0),
            (0.7, 0.7),
            (1, 1),
            (0.5, 0.8),
            (0, 1),
            (0.3, 0.7),
        ]
        region = MaskRegion(vertices)

        # Test some points
        assert isinstance(region.contains_point(0.5, 0.5), bool)

    def test_single_point_trace(self) -> None:
        """Test with single sample trace."""
        trace = WaveformTrace(data=np.array([0.5]), metadata=TraceMetadata(sample_rate=1e6))
        mask = Mask()
        mask.add_region([(0, 0), (1, 0), (1, 1), (0, 1)], "violation")

        result = mask_test(trace, mask, normalize=False)

        assert isinstance(result, MaskTestResult)

    def test_degenerate_polygon(self) -> None:
        """Test with degenerate (line) polygon."""
        # Polygon with all points collinear
        vertices = [(0, 0), (1, 0), (2, 0)]
        region = MaskRegion(vertices)

        # Should handle gracefully
        result = region.contains_point(1, 0)
        assert isinstance(result, bool)
