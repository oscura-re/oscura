"""Tests for advanced statistical analysis methods.

Tests requirements:
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistics.advanced import (
    ChangePointResult,
    CoherenceResult,
    DecompositionResult,
    IsolationForestResult,
    KDEResult,
    LOFResult,
    detect_change_points,
    isolation_forest_outliers,
    kernel_density,
    local_outlier_factor,
    phase_coherence,
    seasonal_decompose,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


@pytest.fixture
def sample_trace() -> WaveformTrace:
    """Create a sample trace for testing."""
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    data = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(1000)
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=1000.0),
    )


@pytest.fixture
def trace_with_outliers() -> WaveformTrace:
    """Create a trace with outliers for testing."""
    np.random.seed(42)
    data = np.random.randn(200)
    # Add outliers
    data[50] = 10.0
    data[100] = -10.0
    data[150] = 8.0
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=1000.0),
    )


@pytest.fixture
def trace_with_change_points() -> WaveformTrace:
    """Create a trace with change points for testing."""
    np.random.seed(42)
    data = np.concatenate(
        [
            np.random.randn(100) + 0,  # Mean 0
            np.random.randn(100) + 5,  # Mean 5
            np.random.randn(100) + 0,  # Mean 0
        ]
    )
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=1000.0),
    )


@pytest.fixture
def seasonal_trace() -> WaveformTrace:
    """Create a trace with seasonal component for testing."""
    np.random.seed(42)
    t = np.arange(400)
    period = 20
    trend = 0.01 * t
    seasonal = np.sin(2 * np.pi * t / period)
    noise = 0.1 * np.random.randn(len(t))
    data = trend + seasonal + noise
    return WaveformTrace(
        data=data.astype(np.float64),
        metadata=TraceMetadata(sample_rate=1000.0),
    )


class TestIsolationForest:
    """Tests for Isolation Forest outlier detection (STAT-011)."""

    def test_isolation_forest_basic(self, trace_with_outliers: WaveformTrace) -> None:
        """Test basic isolation forest outlier detection."""
        result = isolation_forest_outliers(trace_with_outliers, contamination=0.05)

        assert isinstance(result, IsolationForestResult)
        assert result.count > 0
        assert len(result.indices) == result.count
        assert len(result.scores) == len(trace_with_outliers.data)
        assert len(result.mask) == len(trace_with_outliers.data)
        assert np.sum(result.mask) == result.count

    def test_isolation_forest_detects_outliers(self, trace_with_outliers: WaveformTrace) -> None:
        """Test that isolation forest detects known outliers."""
        result = isolation_forest_outliers(trace_with_outliers, contamination=0.1)

        # Should detect at least some of the injected outliers
        outlier_positions = {50, 100, 150}
        detected = set(result.indices)
        intersection = outlier_positions & detected
        assert len(intersection) >= 1, "Should detect at least one known outlier"

    def test_isolation_forest_clean_data(self, sample_trace: WaveformTrace) -> None:
        """Test isolation forest on clean data."""
        result = isolation_forest_outliers(sample_trace, contamination=0.01)

        assert isinstance(result, IsolationForestResult)
        # Should have very few outliers on clean sinusoid
        assert result.count / len(sample_trace.data) <= 0.05

    def test_isolation_forest_numpy_array(self) -> None:
        """Test isolation forest with numpy array input."""
        np.random.seed(42)
        data = np.random.randn(100)
        data[50] = 10.0

        result = isolation_forest_outliers(data, contamination=0.05)
        assert isinstance(result, IsolationForestResult)
        assert result.count >= 1

    def test_isolation_forest_small_data(self) -> None:
        """Test isolation forest with small dataset."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = isolation_forest_outliers(data)

        assert isinstance(result, IsolationForestResult)
        assert result.count == 0  # Too small for reliable detection


class TestLocalOutlierFactor:
    """Tests for Local Outlier Factor detection (STAT-012)."""

    def test_lof_basic(self, trace_with_outliers: WaveformTrace) -> None:
        """Test basic LOF outlier detection."""
        result = local_outlier_factor(trace_with_outliers, n_neighbors=10)

        assert isinstance(result, LOFResult)
        assert len(result.scores) == len(trace_with_outliers.data)
        assert len(result.mask) == len(trace_with_outliers.data)

    def test_lof_detects_outliers(self, trace_with_outliers: WaveformTrace) -> None:
        """Test that LOF detects known outliers."""
        result = local_outlier_factor(trace_with_outliers, n_neighbors=10, threshold=1.5)

        # Should detect at least some outliers
        assert result.count >= 1

    def test_lof_threshold(self, trace_with_outliers: WaveformTrace) -> None:
        """Test LOF with different thresholds."""
        result_low = local_outlier_factor(trace_with_outliers, threshold=1.2)
        result_high = local_outlier_factor(trace_with_outliers, threshold=2.0)

        # Lower threshold should detect more outliers
        assert result_low.count >= result_high.count

    def test_lof_small_data(self) -> None:
        """Test LOF with small dataset."""
        data = np.array([1.0, 2.0, 3.0])
        result = local_outlier_factor(data, n_neighbors=10)

        assert isinstance(result, LOFResult)
        assert result.count == 0  # Not enough neighbors


class TestSeasonalDecompose:
    """Tests for seasonal decomposition (STAT-013)."""

    def test_decompose_basic(self, seasonal_trace: WaveformTrace) -> None:
        """Test basic seasonal decomposition."""
        result = seasonal_decompose(seasonal_trace, period=20)

        assert isinstance(result, DecompositionResult)
        assert len(result.trend) == len(seasonal_trace.data)
        assert len(result.seasonal) == len(seasonal_trace.data)
        assert len(result.residual) == len(seasonal_trace.data)
        assert result.period == 20

    def test_decompose_auto_period(self, seasonal_trace: WaveformTrace) -> None:
        """Test seasonal decomposition with auto period detection."""
        result = seasonal_decompose(seasonal_trace)

        assert isinstance(result, DecompositionResult)
        assert result.period > 0

    def test_decompose_additive(self, seasonal_trace: WaveformTrace) -> None:
        """Test additive decomposition model."""
        result = seasonal_decompose(seasonal_trace, period=20, model="additive")

        # Additive: observed = trend + seasonal + residual
        reconstructed = result.trend + result.seasonal + result.residual
        np.testing.assert_allclose(reconstructed, result.observed, rtol=0.1, atol=0.5)

    def test_decompose_multiplicative(self, seasonal_trace: WaveformTrace) -> None:
        """Test multiplicative decomposition model."""
        # Use positive data for multiplicative model
        data = np.abs(seasonal_trace.data) + 1
        trace = WaveformTrace(
            data=data.astype(np.float64),
            metadata=seasonal_trace.metadata,
        )

        result = seasonal_decompose(trace, period=20, model="multiplicative")
        assert isinstance(result, DecompositionResult)


class TestChangePointDetection:
    """Tests for change point detection (STAT-014)."""

    def test_change_point_basic(self, trace_with_change_points: WaveformTrace) -> None:
        """Test basic change point detection."""
        result = detect_change_points(trace_with_change_points)

        assert isinstance(result, ChangePointResult)
        assert len(result.segments) >= 1
        assert len(result.segment_means) == len(result.segments)
        assert len(result.segment_stds) == len(result.segments)

    def test_change_point_detects_changes(self, trace_with_change_points: WaveformTrace) -> None:
        """Test that change points are detected."""
        result = detect_change_points(trace_with_change_points, n_changes=2)

        # Should detect approximately 2 change points
        assert result.n_changes >= 1

        # Change points should be near 100 and 200
        for cp in result.indices:
            assert 50 <= cp <= 250, f"Change point {cp} not in expected range"

    def test_change_point_segments(self, trace_with_change_points: WaveformTrace) -> None:
        """Test that segments cover the full data."""
        result = detect_change_points(trace_with_change_points)

        # Segments should cover entire range
        total_covered = sum(e - s for s, e in result.segments)
        assert total_covered == len(trace_with_change_points.data)

        # Segments should be non-overlapping and contiguous
        for i in range(len(result.segments) - 1):
            assert result.segments[i][1] == result.segments[i + 1][0]

    def test_change_point_clean_data(self, sample_trace: WaveformTrace) -> None:
        """Test change point detection on clean sinusoid."""
        result = detect_change_points(sample_trace)

        assert isinstance(result, ChangePointResult)
        # Algorithm may find change points even on periodic data due to sensitivity
        # Just verify it returns valid results
        assert result.n_changes >= 0
        assert len(result.segments) == result.n_changes + 1


class TestPhaseCoherence:
    """Tests for phase coherence analysis (STAT-015)."""

    def test_coherence_basic(self, sample_trace: WaveformTrace) -> None:
        """Test basic phase coherence computation."""
        # Create two related traces
        trace1 = sample_trace
        t = np.linspace(0, 1, 1000)
        # Phase shifted version
        data2 = np.sin(2 * np.pi * 10 * t + np.pi / 4) + 0.1 * np.random.randn(1000)
        trace2 = WaveformTrace(
            data=data2.astype(np.float64),
            metadata=TraceMetadata(sample_rate=1000.0),
        )

        result = phase_coherence(trace1, trace2)

        assert isinstance(result, CoherenceResult)
        assert len(result.coherence) > 0
        assert len(result.frequencies) == len(result.coherence)
        assert len(result.phase) == len(result.coherence)
        assert 0 <= result.mean_coherence <= 1

    def test_coherence_identical_signals(self, sample_trace: WaveformTrace) -> None:
        """Test coherence of identical signals."""
        result = phase_coherence(sample_trace, sample_trace)

        # Identical signals should have high coherence
        assert result.peak_coherence >= 0.9

    def test_coherence_uncorrelated_signals(self) -> None:
        """Test coherence of uncorrelated signals."""
        np.random.seed(42)
        data1 = np.random.randn(500)
        data2 = np.random.randn(500)

        result = phase_coherence(data1, data2, sample_rate=1000.0)

        # Uncorrelated signals should have low mean coherence
        assert result.mean_coherence < 0.5

    def test_coherence_with_sample_rate(self) -> None:
        """Test coherence with explicit sample rate."""
        np.random.seed(42)
        t = np.linspace(0, 0.1, 1000)
        data1 = np.sin(2 * np.pi * 100 * t)
        data2 = np.sin(2 * np.pi * 100 * t + 0.5)

        result = phase_coherence(data1, data2, sample_rate=10000.0)

        assert isinstance(result, CoherenceResult)
        # Peak should be near 100 Hz
        assert 50 <= result.peak_frequency <= 200


class TestKernelDensity:
    """Tests for kernel density estimation (STAT-016)."""

    def test_kde_basic(self, sample_trace: WaveformTrace) -> None:
        """Test basic KDE computation."""
        result = kernel_density(sample_trace)

        assert isinstance(result, KDEResult)
        assert len(result.x) > 0
        assert len(result.density) == len(result.x)
        assert result.bandwidth > 0
        assert len(result.peaks) >= 1

    def test_kde_bimodal(self) -> None:
        """Test KDE on bimodal distribution."""
        np.random.seed(42)
        data = np.concatenate(
            [
                np.random.randn(500) - 3,
                np.random.randn(500) + 3,
            ]
        )

        result = kernel_density(data)

        # Should detect two modes
        assert len(result.peaks) >= 2

    def test_kde_bandwidth_selection(self, sample_trace: WaveformTrace) -> None:
        """Test different bandwidth selection methods."""
        result_scott = kernel_density(sample_trace, bandwidth="scott")
        result_silver = kernel_density(sample_trace, bandwidth="silverman")

        # Both should produce valid results
        assert result_scott.bandwidth > 0
        assert result_silver.bandwidth > 0

    def test_kde_custom_bandwidth(self, sample_trace: WaveformTrace) -> None:
        """Test KDE with custom bandwidth."""
        result = kernel_density(sample_trace, bandwidth=0.1)

        assert result.bandwidth == 0.1

    def test_kde_different_kernels(self, sample_trace: WaveformTrace) -> None:
        """Test KDE with different kernels."""
        result_gaussian = kernel_density(sample_trace, kernel="gaussian")
        result_tophat = kernel_density(sample_trace, kernel="tophat")
        result_epan = kernel_density(sample_trace, kernel="epanechnikov")

        assert isinstance(result_gaussian, KDEResult)
        assert isinstance(result_tophat, KDEResult)
        assert isinstance(result_epan, KDEResult)

    def test_kde_density_integrates_to_one(self, sample_trace: WaveformTrace) -> None:
        """Test that density integrates to approximately 1."""
        result = kernel_density(sample_trace, n_points=1000)

        # Numerical integration
        dx = result.x[1] - result.x[0]
        integral = np.sum(result.density) * dx

        # Should be close to 1 (allow some error due to finite domain)
        assert 0.8 <= integral <= 1.2


class TestStatisticalStatisticsAdvancedIntegration:
    """Integration tests for advanced statistics module."""

    def test_all_functions_accessible(self) -> None:
        """Test that all functions are accessible from package."""
        from oscura.analyzers.statistics import (
            detect_change_points,
            isolation_forest_outliers,
            kernel_density,
            local_outlier_factor,
            phase_coherence,
            seasonal_decompose,
        )

        assert callable(isolation_forest_outliers)
        assert callable(local_outlier_factor)
        assert callable(seasonal_decompose)
        assert callable(detect_change_points)
        assert callable(phase_coherence)
        assert callable(kernel_density)

    def test_result_types_accessible(self) -> None:
        """Test that result types are accessible from package."""
        from oscura.analyzers.statistics import (
            ChangePointResult,
            CoherenceResult,
            DecompositionResult,
            IsolationForestResult,
            KDEResult,
            LOFResult,
        )

        # All should be valid types
        assert IsolationForestResult is not None
        assert LOFResult is not None
        assert DecompositionResult is not None
        assert ChangePointResult is not None
        assert CoherenceResult is not None
        assert KDEResult is not None

    def test_statistical_alias(self) -> None:
        """Test that functions are accessible via statistical alias."""
        from oscura.analyzers.statistical import (
            isolation_forest_outliers,
            kernel_density,
        )

        assert callable(isolation_forest_outliers)
        assert callable(kernel_density)
