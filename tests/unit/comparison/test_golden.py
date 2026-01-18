"""Comprehensive unit tests for golden waveform comparison.

This module tests golden reference creation, comparison, and persistence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oscura.comparison.golden import (
    GoldenComparisonResult,
    GoldenReference,
    batch_compare_to_golden,
    compare_to_golden,
    create_golden,
    golden_from_average,
    tolerance_envelope,
)
from oscura.core.exceptions import AnalysisError, LoaderError
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


@pytest.fixture
def reference_trace() -> WaveformTrace:
    """Create a reference sine wave trace."""
    data = np.sin(2 * np.pi * 1e6 * np.linspace(0, 1e-3, 1000))
    metadata = TraceMetadata(sample_rate=1e6, channel_name="CH1")
    return WaveformTrace(data=data, metadata=metadata)


@pytest.fixture
def golden_ref(reference_trace: WaveformTrace) -> GoldenReference:
    """Create a golden reference from the reference trace."""
    return create_golden(reference_trace, tolerance_pct=5.0, name="test_golden")


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestCreateGolden:
    """Test golden reference creation."""

    def test_create_with_absolute_tolerance(self, reference_trace: WaveformTrace) -> None:
        """Test creating golden with absolute tolerance."""
        golden = create_golden(reference_trace, tolerance=0.1, name="abs_golden")

        assert golden.name == "abs_golden"
        assert golden.num_samples == len(reference_trace.data)
        assert golden.sample_rate == reference_trace.metadata.sample_rate
        assert golden.tolerance == 0.1
        assert golden.tolerance_type == "absolute"
        np.testing.assert_array_equal(golden.data, reference_trace.data)
        np.testing.assert_array_equal(golden.upper_bound, reference_trace.data + 0.1)
        np.testing.assert_array_equal(golden.lower_bound, reference_trace.data - 0.1)

    def test_create_with_percentage_tolerance(self, reference_trace: WaveformTrace) -> None:
        """Test creating golden with percentage tolerance."""
        golden = create_golden(reference_trace, tolerance_pct=10.0)

        assert golden.tolerance_type == "percentage"
        data_range = np.ptp(reference_trace.data)
        expected_tol = data_range * 0.1
        assert abs(golden.tolerance - expected_tol) < 1e-10

    def test_create_with_sigma_tolerance(self, reference_trace: WaveformTrace) -> None:
        """Test creating golden with sigma-based tolerance."""
        golden = create_golden(reference_trace, tolerance_sigma=3.0)

        assert golden.tolerance_type == "sigma"
        expected_tol = np.std(reference_trace.data) * 3.0
        assert abs(golden.tolerance - expected_tol) < 1e-10

    def test_create_default_tolerance(self, reference_trace: WaveformTrace) -> None:
        """Test creating golden with default tolerance."""
        golden = create_golden(reference_trace)

        # Default is 1% of range
        assert golden.tolerance_type == "percentage"
        data_range = np.ptp(reference_trace.data)
        expected_tol = data_range * 0.01
        assert abs(golden.tolerance - expected_tol) < 1e-10

    def test_create_with_description(self, reference_trace: WaveformTrace) -> None:
        """Test creating golden with description."""
        desc = "Test golden reference for unit tests"
        golden = create_golden(reference_trace, tolerance=0.1, description=desc)

        assert golden.description == desc

    def test_golden_properties(self, golden_ref: GoldenReference) -> None:
        """Test golden reference properties."""
        assert golden_ref.num_samples == 1000
        assert golden_ref.duration == 1000 / 1e6


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestToleranceEnvelope:
    """Test tolerance envelope creation."""

    def test_absolute_envelope(self, reference_trace: WaveformTrace) -> None:
        """Test absolute tolerance envelope."""
        upper, lower = tolerance_envelope(reference_trace, absolute=0.1)

        np.testing.assert_array_equal(upper, reference_trace.data + 0.1)
        np.testing.assert_array_equal(lower, reference_trace.data - 0.1)

    def test_percentage_envelope(self, reference_trace: WaveformTrace) -> None:
        """Test percentage tolerance envelope."""
        upper, lower = tolerance_envelope(reference_trace, percentage=5.0)

        data_range = np.ptp(reference_trace.data)
        expected_tol = data_range * 0.05
        np.testing.assert_allclose(upper, reference_trace.data + expected_tol)
        np.testing.assert_allclose(lower, reference_trace.data - expected_tol)

    def test_sigma_envelope(self, reference_trace: WaveformTrace) -> None:
        """Test sigma-based tolerance envelope."""
        upper, lower = tolerance_envelope(reference_trace, sigma=2.0)

        expected_tol = np.std(reference_trace.data) * 2.0
        np.testing.assert_allclose(upper, reference_trace.data + expected_tol)
        np.testing.assert_allclose(lower, reference_trace.data - expected_tol)

    def test_no_tolerance_raises(self, reference_trace: WaveformTrace) -> None:
        """Test that missing tolerance raises error."""
        with pytest.raises(ValueError, match="Must specify"):
            tolerance_envelope(reference_trace)


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestCompareToGolden:
    """Test comparison to golden reference."""

    def test_compare_identical_pass(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test passing comparison with identical trace."""
        result = compare_to_golden(reference_trace, golden_ref)

        assert result.passed
        assert result.num_violations == 0
        assert result.violation_rate == 0.0
        assert result.max_deviation < 1e-10
        assert result.rms_deviation < 1e-10

    def test_compare_within_tolerance(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test passing comparison within tolerance."""
        # Add small noise within tolerance
        noisy_data = reference_trace.data + np.random.normal(0, 0.01, 1000)
        trace = WaveformTrace(data=noisy_data, metadata=reference_trace.metadata)

        result = compare_to_golden(trace, golden_ref)
        assert result.passed

    def test_compare_outside_tolerance_fail(self, reference_trace: WaveformTrace) -> None:
        """Test failing comparison outside tolerance."""
        # Create tight tolerance golden
        golden = create_golden(reference_trace, tolerance=0.01)

        # Add large deviation
        violated_data = reference_trace.data + 0.1
        trace = WaveformTrace(data=violated_data, metadata=reference_trace.metadata)

        result = compare_to_golden(trace, golden)
        assert not result.passed
        assert result.num_violations > 0
        assert result.violation_rate > 0

    def test_compare_upper_violations(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test detection of upper bound violations."""
        violated_data = reference_trace.data.copy()
        violated_data[100:110] += 1.0  # Exceed upper bound

        trace = WaveformTrace(data=violated_data, metadata=reference_trace.metadata)
        result = compare_to_golden(trace, golden_ref)

        assert not result.passed
        assert result.upper_violations is not None
        assert len(result.upper_violations) >= 10

    def test_compare_lower_violations(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test detection of lower bound violations."""
        violated_data = reference_trace.data.copy()
        violated_data[200:210] -= 1.0  # Below lower bound

        trace = WaveformTrace(data=violated_data, metadata=reference_trace.metadata)
        result = compare_to_golden(trace, golden_ref)

        assert not result.passed
        assert result.lower_violations is not None
        assert len(result.lower_violations) >= 10

    def test_compare_statistics(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test statistics in comparison result."""
        result = compare_to_golden(reference_trace, golden_ref)

        assert result.statistics is not None
        assert "mean_deviation" in result.statistics
        assert "std_deviation" in result.statistics
        assert "max_positive_deviation" in result.statistics
        assert "max_negative_deviation" in result.statistics
        assert "correlation" in result.statistics

    def test_compare_margin(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test margin calculation."""
        result = compare_to_golden(reference_trace, golden_ref)

        assert result.margin is not None
        assert result.margin_percentage is not None
        # Margin should be positive for passing comparison
        assert result.margin > 0

    def test_compare_with_alignment(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test comparison with alignment enabled."""
        # Create slightly shifted version
        shifted_data = np.roll(reference_trace.data, 5)
        trace = WaveformTrace(data=shifted_data, metadata=reference_trace.metadata)

        result = compare_to_golden(trace, golden_ref, align=True)
        # Alignment should improve comparison
        assert isinstance(result, GoldenComparisonResult)

    def test_compare_without_alignment(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test comparison without alignment."""
        result = compare_to_golden(reference_trace, golden_ref, align=False)
        assert result.passed

    def test_compare_length_mismatch_interpolate(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test comparison with length mismatch using interpolation."""
        # Create shorter trace
        short_data = reference_trace.data[:500]
        trace = WaveformTrace(data=short_data, metadata=reference_trace.metadata)

        result = compare_to_golden(trace, golden_ref, interpolate=True)
        assert isinstance(result, GoldenComparisonResult)

    def test_compare_length_mismatch_truncate(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test comparison with length mismatch using truncation."""
        # Create longer trace
        long_data = np.concatenate([reference_trace.data, reference_trace.data])
        trace = WaveformTrace(data=long_data, metadata=reference_trace.metadata)

        result = compare_to_golden(trace, golden_ref, interpolate=False)
        assert isinstance(result, GoldenComparisonResult)


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestBatchCompare:
    """Test batch comparison to golden reference."""

    def test_batch_compare_all_pass(
        self, reference_trace: WaveformTrace, golden_ref: GoldenReference
    ) -> None:
        """Test batch comparison with all passing traces."""
        # Create multiple similar traces
        traces = []
        for _i in range(5):
            noise = np.random.normal(0, 0.01, 1000)
            data = reference_trace.data + noise
            trace = WaveformTrace(data=data, metadata=reference_trace.metadata)
            traces.append(trace)

        results = batch_compare_to_golden(traces, golden_ref)

        assert len(results) == 5
        assert all(r.passed for r in results)

    def test_batch_compare_mixed_results(self, reference_trace: WaveformTrace) -> None:
        """Test batch comparison with mixed pass/fail."""
        golden = create_golden(reference_trace, tolerance=0.05)

        traces = []
        # Good traces
        for _i in range(3):
            noise = np.random.normal(0, 0.01, 1000)
            data = reference_trace.data + noise
            traces.append(WaveformTrace(data=data, metadata=reference_trace.metadata))

        # Bad traces
        for _i in range(2):
            data = reference_trace.data + 0.2  # Exceeds tolerance
            traces.append(WaveformTrace(data=data, metadata=reference_trace.metadata))

        results = batch_compare_to_golden(traces, golden)

        assert len(results) == 5
        passed = sum(r.passed for r in results)
        assert passed == 3


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestGoldenFromAverage:
    """Test creating golden from averaged traces."""

    def test_golden_from_average_basic(self, reference_trace: WaveformTrace) -> None:
        """Test creating golden from average of traces."""
        # Create multiple similar traces
        traces = []
        for _i in range(10):
            noise = np.random.normal(0, 0.01, 1000)
            data = reference_trace.data + noise
            trace = WaveformTrace(data=data, metadata=reference_trace.metadata)
            traces.append(trace)

        golden = golden_from_average(traces, tolerance_sigma=3.0)

        assert golden.name == "averaged_golden"
        assert golden.tolerance_type == "sigma"
        assert golden.num_samples == 1000
        # Averaged data should be close to original
        np.testing.assert_allclose(golden.data, reference_trace.data, atol=0.05)

    def test_golden_from_average_custom_name(self, reference_trace: WaveformTrace) -> None:
        """Test custom name for averaged golden."""
        traces = [reference_trace, reference_trace]
        golden = golden_from_average(traces, name="my_avg_golden")

        assert golden.name == "my_avg_golden"

    def test_golden_from_average_empty_raises(self) -> None:
        """Test that empty trace list raises error."""
        with pytest.raises(AnalysisError, match="No traces provided"):
            golden_from_average([])

    def test_golden_from_average_different_lengths(self, reference_trace: WaveformTrace) -> None:
        """Test averaging traces of different lengths."""
        trace1 = reference_trace
        short_data = reference_trace.data[:500]
        trace2 = WaveformTrace(data=short_data, metadata=reference_trace.metadata)

        golden = golden_from_average([trace1, trace2])
        # Should use minimum length
        assert golden.num_samples == 500


@pytest.mark.unit
@pytest.mark.comparison
@pytest.mark.requirement("CMP-001")
class TestGoldenPersistence:
    """Test saving and loading golden references."""

    def test_save_and_load(self, golden_ref: GoldenReference, tmp_path: Path) -> None:
        """Test saving and loading golden reference."""
        filepath = tmp_path / "golden.json"

        # Save
        golden_ref.save(filepath)
        assert filepath.exists()

        # Load
        loaded = GoldenReference.load(filepath)

        assert loaded.name == golden_ref.name
        assert loaded.tolerance == golden_ref.tolerance
        assert loaded.tolerance_type == golden_ref.tolerance_type
        np.testing.assert_array_equal(loaded.data, golden_ref.data)
        np.testing.assert_array_equal(loaded.upper_bound, golden_ref.upper_bound)
        np.testing.assert_array_equal(loaded.lower_bound, golden_ref.lower_bound)

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        """Test loading nonexistent file raises error."""
        filepath = tmp_path / "nonexistent.json"

        with pytest.raises(LoaderError, match="not found"):
            GoldenReference.load(filepath)

    def test_to_dict(self, golden_ref: GoldenReference) -> None:
        """Test converting golden to dictionary."""
        data_dict = golden_ref.to_dict()

        assert "data" in data_dict
        assert "sample_rate" in data_dict
        assert "upper_bound" in data_dict
        assert "lower_bound" in data_dict
        assert "tolerance" in data_dict
        assert "tolerance_type" in data_dict
        assert "name" in data_dict
        assert "created" in data_dict

    def test_from_dict(self, golden_ref: GoldenReference) -> None:
        """Test creating golden from dictionary."""
        data_dict = golden_ref.to_dict()
        loaded = GoldenReference.from_dict(data_dict)

        assert loaded.name == golden_ref.name
        assert loaded.tolerance == golden_ref.tolerance
        np.testing.assert_array_equal(loaded.data, golden_ref.data)

    def test_roundtrip_serialization(self, golden_ref: GoldenReference, tmp_path: Path) -> None:
        """Test complete roundtrip serialization."""
        filepath = tmp_path / "roundtrip.json"

        # Save
        golden_ref.save(filepath)

        # Load
        loaded = GoldenReference.load(filepath)

        # Compare to golden
        result1 = compare_to_golden(
            WaveformTrace(data=golden_ref.data, metadata=TraceMetadata(sample_rate=1e6)),
            golden_ref,
        )
        result2 = compare_to_golden(
            WaveformTrace(data=loaded.data, metadata=TraceMetadata(sample_rate=1e6)),
            loaded,
        )

        assert result1.passed
        assert result2.passed


@pytest.mark.unit
@pytest.mark.comparison
class TestComparisonGoldenEdgeCases:
    """Test edge cases for golden reference functionality."""

    def test_constant_trace_golden(self) -> None:
        """Test creating golden from constant trace."""
        data = np.ones(1000) * 0.5
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        golden = create_golden(trace, tolerance=0.1)
        result = compare_to_golden(trace, golden)

        assert result.passed

    def test_zero_range_trace(self) -> None:
        """Test golden with zero range trace."""
        data = np.zeros(1000)
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        golden = create_golden(trace, tolerance=0.1)
        # Should handle zero range gracefully
        assert golden.tolerance == 0.1

    def test_single_sample_golden(self) -> None:
        """Test golden with single sample."""
        data = np.array([1.0])
        metadata = TraceMetadata(sample_rate=1e6)
        trace = WaveformTrace(data=data, metadata=metadata)

        golden = create_golden(trace, tolerance=0.1)
        result = compare_to_golden(trace, golden)

        assert result.passed
        assert golden.num_samples == 1
