"""Unit tests for statistical analysis functions."""

import numpy as np
import pytest

from oscura.analyzers.statistics.basic import (
    basic_stats,
    percentiles,
    quartiles,
    running_stats,
)
from oscura.analyzers.statistics.distribution import (
    bimodality_coefficient,
    distribution_metrics,
    histogram,
    moment,
    normality_test,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


@pytest.fixture
def normal_data() -> np.ndarray:
    """Generate normally distributed data."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.0, scale=2.0, size=1000)


@pytest.fixture
def uniform_data() -> np.ndarray:
    """Generate uniformly distributed data."""
    rng = np.random.default_rng(42)
    return rng.uniform(low=0.0, high=10.0, size=1000)


@pytest.fixture
def bimodal_data() -> np.ndarray:
    """Generate bimodal data."""
    rng = np.random.default_rng(42)
    mode1 = rng.normal(loc=0.0, scale=1.0, size=500)
    mode2 = rng.normal(loc=5.0, scale=1.0, size=500)
    return np.concatenate([mode1, mode2])


@pytest.fixture
def waveform_trace(normal_data: np.ndarray) -> WaveformTrace:
    """Create WaveformTrace from normal data."""
    return WaveformTrace(
        data=normal_data,
        metadata=TraceMetadata(sample_rate=1e6),
    )


class TestBasicStats:
    """Tests for basic_stats function."""

    def test_basic_stats_returns_dict(self, normal_data: np.ndarray) -> None:
        """basic_stats should return dictionary with required keys."""
        stats = basic_stats(normal_data)

        assert "mean" in stats
        assert "variance" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "range" in stats
        assert "count" in stats

    def test_basic_stats_values(self, normal_data: np.ndarray) -> None:
        """basic_stats values should be correct."""
        stats = basic_stats(normal_data)

        assert abs(stats["mean"] - np.mean(normal_data)) < 1e-10
        assert abs(stats["std"] - np.std(normal_data)) < 1e-10
        assert stats["min"] == np.min(normal_data)
        assert stats["max"] == np.max(normal_data)
        assert stats["count"] == len(normal_data)

    def test_basic_stats_with_trace(self, waveform_trace: WaveformTrace) -> None:
        """basic_stats should work with WaveformTrace."""
        stats = basic_stats(waveform_trace)

        assert "mean" in stats
        assert stats["count"] == len(waveform_trace.data)


class TestPercentiles:
    """Tests for percentiles function."""

    def test_percentiles_default(self, normal_data: np.ndarray) -> None:
        """Default percentiles should be quartiles."""
        pct = percentiles(normal_data)

        assert "p0" in pct
        assert "p25" in pct
        assert "p50" in pct
        assert "p75" in pct
        assert "p100" in pct

    def test_percentiles_custom(self, normal_data: np.ndarray) -> None:
        """Custom percentiles should be returned."""
        pct = percentiles(normal_data, [10, 90])

        assert "p10" in pct
        assert "p90" in pct
        assert pct["p10"] < pct["p90"]

    def test_percentiles_median_is_p50(self, normal_data: np.ndarray) -> None:
        """p50 should equal numpy median."""
        pct = percentiles(normal_data)

        assert abs(pct["p50"] - np.median(normal_data)) < 1e-10


class TestQuartiles:
    """Tests for quartiles function."""

    def test_quartiles_returns_iqr(self, normal_data: np.ndarray) -> None:
        """quartiles should return IQR."""
        q = quartiles(normal_data)

        assert "q1" in q
        assert "median" in q
        assert "q3" in q
        assert "iqr" in q
        assert abs(q["iqr"] - (q["q3"] - q["q1"])) < 1e-10

    def test_quartiles_fences(self, normal_data: np.ndarray) -> None:
        """quartiles should return fences for outlier detection."""
        q = quartiles(normal_data)

        assert "lower_fence" in q
        assert "upper_fence" in q
        assert q["lower_fence"] < q["q1"]
        assert q["upper_fence"] > q["q3"]


class TestRunningStats:
    """Tests for running_stats function."""

    def test_running_stats_shapes(self, normal_data: np.ndarray) -> None:
        """Running stats should return correct shapes."""
        window_size = 100
        running = running_stats(normal_data, window_size)

        expected_len = len(normal_data) - window_size + 1

        assert len(running["mean"]) == expected_len
        assert len(running["std"]) == expected_len
        assert len(running["min"]) == expected_len
        assert len(running["max"]) == expected_len


class TestHistogram:
    """Tests for histogram function."""

    def test_histogram_returns_counts_and_edges(self, normal_data: np.ndarray) -> None:
        """histogram should return counts and bin edges."""
        counts, edges = histogram(normal_data, bins=50)

        assert len(counts) == 50
        assert len(edges) == 51  # n+1 edges for n bins

    def test_histogram_density(self, normal_data: np.ndarray) -> None:
        """histogram with density should integrate to 1."""
        counts, edges = histogram(normal_data, bins=50, density=True)

        bin_widths = np.diff(edges)
        integral = np.sum(counts * bin_widths)

        assert abs(integral - 1.0) < 0.01


class TestDistributionMetrics:
    """Tests for distribution_metrics function."""

    def test_distribution_metrics_keys(self, normal_data: np.ndarray) -> None:
        """distribution_metrics should return required keys."""
        metrics = distribution_metrics(normal_data)

        assert "skewness" in metrics
        assert "kurtosis" in metrics
        assert "excess_kurtosis" in metrics
        assert "crest_factor" in metrics

    def test_normal_distribution_metrics(self, normal_data: np.ndarray) -> None:
        """Normal distribution should have ~0 skewness and ~3 kurtosis."""
        metrics = distribution_metrics(normal_data)

        # Skewness should be close to 0
        assert abs(metrics["skewness"]) < 0.3

        # Kurtosis should be close to 3 (Pearson's)
        assert abs(metrics["kurtosis"] - 3) < 0.5

    def test_crest_factor(self, normal_data: np.ndarray) -> None:
        """Crest factor should be positive."""
        metrics = distribution_metrics(normal_data)

        assert metrics["crest_factor"] > 0
        assert metrics["crest_factor_db"] > 0


class TestMoment:
    """Tests for moment function."""

    def test_second_moment_is_variance(self, normal_data: np.ndarray) -> None:
        """Second central moment should be variance."""
        m2 = moment(normal_data, 2)
        var = np.var(normal_data)

        assert abs(m2 - var) < 1e-10

    def test_raw_moment(self, normal_data: np.ndarray) -> None:
        """Raw (non-central) moment should differ from central."""
        raw = moment(normal_data, 2, central=False)
        central = moment(normal_data, 2, central=True)

        # Raw second moment = variance + mean^2
        expected_raw = central + np.mean(normal_data) ** 2
        assert abs(raw - expected_raw) < 1e-10


class TestNormalityTest:
    """Tests for normality_test function."""

    def test_normal_data_passes(self, normal_data: np.ndarray) -> None:
        """Normal data should pass normality test."""
        result = normality_test(normal_data)

        assert "statistic" in result
        assert "p_value" in result
        assert "is_normal" in result

        # Should be normal (p > 0.05)
        assert result["p_value"] > 0.05

    def test_uniform_data_fails(self, uniform_data: np.ndarray) -> None:
        """Uniform data should fail normality test."""
        result = normality_test(uniform_data)

        # Should not be normal (p < 0.05)
        assert result["p_value"] < 0.05
        assert not result["is_normal"]


class TestBimodalityCoefficient:
    """Tests for bimodality_coefficient function."""

    def test_unimodal_coefficient(self, normal_data: np.ndarray) -> None:
        """Unimodal data should have low bimodality coefficient."""
        bc = bimodality_coefficient(normal_data)

        # Unimodal should be < 0.555
        assert bc < 0.7  # Some margin for sampling variation

    def test_bimodal_coefficient(self, bimodal_data: np.ndarray) -> None:
        """Bimodal data should have higher coefficient."""
        bc_bimodal = bimodality_coefficient(bimodal_data)

        # Bimodal should be > 0.555
        assert bc_bimodal > 0.5
