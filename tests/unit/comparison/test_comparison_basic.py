"""Tests for comparison and limit testing module.

Tests requirements:
"""

import numpy as np
import pytest

from oscura.comparison.compare import (
    compare_traces,
    correlation,
    difference,
    similarity_score,
)
from oscura.comparison.golden import (
    compare_to_golden,
    create_golden,
)
from oscura.comparison.limits import (
    LimitSpec,
    check_limits,
    create_limit_spec,
    margin_analysis,
)
from oscura.comparison.mask import (
    Mask,
    MaskRegion,
    eye_mask,
    mask_test,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_trace():
    """Create a sample trace for testing."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def constant_trace():
    """Create a constant trace for testing."""
    data = np.ones(1000) * 0.5
    metadata = TraceMetadata(sample_rate=1e6)
    return WaveformTrace(data=data, metadata=metadata)


class TestCompareTraces:
    """Tests for trace comparison (CMP-001)."""

    def test_identical_traces(self, sample_trace):
        """Test comparing identical traces."""
        result = compare_traces(sample_trace, sample_trace)
        assert result.match
        assert result.max_difference < 1e-10
        assert result.similarity > 0.99

    def test_different_traces(self, sample_trace, constant_trace):
        """Test comparing different traces."""
        result = compare_traces(sample_trace, constant_trace)
        assert result.max_difference > 0

    def test_tolerance(self, sample_trace):
        """Test tolerance-based comparison."""
        # Create slightly different trace
        data = sample_trace.data + 0.001
        trace2 = WaveformTrace(data=data, metadata=sample_trace.metadata)

        result = compare_traces(sample_trace, trace2, tolerance=0.01)
        assert result.match


class TestDifference:
    """Tests for trace difference calculation."""

    def test_difference_identical(self, sample_trace):
        """Test difference of identical traces."""
        result = difference(sample_trace, sample_trace)
        np.testing.assert_allclose(result.data, 0, atol=1e-10)

    def test_difference_offset(self, sample_trace):
        """Test difference with offset."""
        data = sample_trace.data + 1.0
        trace2 = WaveformTrace(data=data, metadata=sample_trace.metadata)
        result = difference(sample_trace, trace2)
        np.testing.assert_allclose(result.data, -1.0)


class TestCorrelation:
    """Tests for trace correlation."""

    def test_self_correlation(self, sample_trace):
        """Test self-correlation peak at zero lag."""
        lags, corr = correlation(sample_trace, sample_trace)
        peak_idx = np.argmax(corr)
        # Peak should be near zero lag
        assert abs(lags[peak_idx]) < len(sample_trace.data) // 10


class TestSimilarityScore:
    """Tests for similarity scoring."""

    def test_identical_similarity(self, sample_trace):
        """Test similarity of identical traces."""
        score = similarity_score(sample_trace, sample_trace)
        assert score > 0.99

    def test_inverted_similarity(self, sample_trace):
        """Test similarity of inverted traces."""
        data = -sample_trace.data
        trace2 = WaveformTrace(data=data, metadata=sample_trace.metadata)
        score = similarity_score(sample_trace, trace2)
        # Inverted should have low similarity after normalization
        assert score < 0.1


class TestLimitSpec:
    """Tests for limit specification (CMP-002)."""

    def test_create_limit_spec(self):
        """Test creating limit specification."""
        spec = LimitSpec(upper=1.0, lower=-1.0)
        assert spec.upper == 1.0
        assert spec.lower == -1.0

    def test_create_from_center_tolerance(self):
        """Test creating spec from center/tolerance."""
        spec = create_limit_spec(center=1.0, tolerance=0.1)
        assert spec.upper == 1.1
        assert spec.lower == 0.9

    def test_create_from_percentage(self):
        """Test creating spec from percentage."""
        spec = create_limit_spec(center=1.0, tolerance_pct=10)
        assert spec.upper == 1.1
        assert spec.lower == 0.9


class TestCheckLimits:
    """Tests for limit checking."""

    def test_within_limits(self, constant_trace):
        """Test data within limits."""
        result = check_limits(constant_trace, upper=1.0, lower=0.0)
        assert result.passed
        assert result.num_violations == 0

    def test_exceeds_upper(self, constant_trace):
        """Test data exceeding upper limit."""
        result = check_limits(constant_trace, upper=0.4, lower=0.0)
        assert not result.passed
        assert result.num_violations > 0

    def test_below_lower(self, constant_trace):
        """Test data below lower limit."""
        result = check_limits(constant_trace, upper=1.0, lower=0.6)
        assert not result.passed


class TestMarginAnalysis:
    """Tests for margin analysis."""

    def test_margin_calculation(self, constant_trace):
        """Test margin calculation."""
        spec = LimitSpec(upper=1.0, lower=0.0)
        result = margin_analysis(constant_trace, spec)
        assert result.upper_margin == 0.5
        assert result.lower_margin == 0.5

    def test_margin_status(self, constant_trace):
        """Test margin status determination."""
        spec = LimitSpec(upper=1.0, lower=0.0)
        result = margin_analysis(constant_trace, spec)
        assert result.margin_status == "pass"


class TestMask:
    """Tests for mask testing (CMP-003)."""

    def test_mask_region_contains(self):
        """Test point containment in mask region."""
        vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        region = MaskRegion(vertices)
        assert region.contains_point(0.5, 0.5)
        assert not region.contains_point(2.0, 2.0)

    def test_eye_mask_creation(self):
        """Test eye mask creation."""
        mask = eye_mask(0.5, 0.4)
        assert len(mask.regions) > 0
        assert mask.name == "eye_mask"


class TestMaskTest:
    """Tests for mask-based testing."""

    def test_no_violations(self):
        """Test data with no mask violations."""
        data = np.zeros(100)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        mask = Mask()
        mask.add_region([(10, 0.5), (20, 0.5), (20, 1.0), (10, 1.0)], "violation")

        result = mask_test(trace, mask)
        assert result.passed


class TestGoldenReference:
    """Tests for golden waveform comparison."""

    def test_create_golden(self, sample_trace):
        """Test creating golden reference."""
        golden = create_golden(sample_trace, tolerance_pct=5)
        assert golden.num_samples == len(sample_trace.data)
        assert golden.sample_rate == sample_trace.metadata.sample_rate

    def test_compare_to_golden_pass(self, sample_trace):
        """Test passing comparison to golden."""
        golden = create_golden(sample_trace, tolerance_pct=5)
        result = compare_to_golden(sample_trace, golden)
        assert result.passed

    def test_compare_to_golden_fail(self, sample_trace):
        """Test failing comparison to golden."""
        golden = create_golden(sample_trace, tolerance=0.001)

        # Create trace that exceeds tolerance
        data = sample_trace.data + 0.1
        trace2 = WaveformTrace(data=data, metadata=sample_trace.metadata)

        result = compare_to_golden(trace2, golden)
        assert not result.passed
