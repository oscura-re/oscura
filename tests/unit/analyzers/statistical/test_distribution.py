"""Comprehensive tests for distribution analysis functions.

This module tests histogram generation, distribution metrics, normality tests,
and distribution fitting functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.analyzers.statistics.distribution import (
    bimodality_coefficient,
    distribution_metrics,
    fit_distribution,
    histogram,
    moment,
    normality_test,
)
from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = [pytest.mark.unit, pytest.mark.analyzer, pytest.mark.statistical]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def normal_data() -> np.ndarray:
    """Generate normally distributed data."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.0, scale=2.0, size=1000)


@pytest.fixture
def standard_normal_data() -> np.ndarray:
    """Generate standard normal data (mean=0, std=1)."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.0, scale=1.0, size=1000)


@pytest.fixture
def uniform_data() -> np.ndarray:
    """Generate uniformly distributed data."""
    rng = np.random.default_rng(42)
    return rng.uniform(low=0.0, high=10.0, size=1000)


@pytest.fixture
def exponential_data() -> np.ndarray:
    """Generate exponentially distributed data."""
    rng = np.random.default_rng(42)
    return rng.exponential(scale=2.0, size=1000)


@pytest.fixture
def lognormal_data() -> np.ndarray:
    """Generate log-normally distributed data."""
    rng = np.random.default_rng(42)
    return rng.lognormal(mean=0.0, sigma=1.0, size=1000)


@pytest.fixture
def bimodal_data() -> np.ndarray:
    """Generate bimodal data (mixture of two normals)."""
    rng = np.random.default_rng(42)
    mode1 = rng.normal(loc=0.0, scale=1.0, size=500)
    mode2 = rng.normal(loc=6.0, scale=1.0, size=500)
    return np.concatenate([mode1, mode2])


@pytest.fixture
def skewed_data() -> np.ndarray:
    """Generate right-skewed data."""
    rng = np.random.default_rng(42)
    # Chi-squared with low df is right-skewed
    return rng.chisquare(df=3, size=1000)


@pytest.fixture
def waveform_trace(normal_data: np.ndarray) -> WaveformTrace:
    """Create WaveformTrace from normal data."""
    return WaveformTrace(
        data=normal_data,
        metadata=TraceMetadata(sample_rate=1e6),
    )


# =============================================================================
# Histogram Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestHistogram:
    """Tests for histogram function."""

    def test_histogram_returns_counts_and_edges(self, normal_data: np.ndarray) -> None:
        """Test histogram returns counts and bin edges."""
        counts, edges = histogram(normal_data, bins=50)

        assert len(counts) == 50
        assert len(edges) == 51  # n+1 edges for n bins

    def test_histogram_auto_bins(self, normal_data: np.ndarray) -> None:
        """Test histogram with automatic bin selection."""
        counts, edges = histogram(normal_data, bins="auto")

        assert len(counts) > 0
        assert len(edges) == len(counts) + 1

    def test_histogram_sturges_bins(self, normal_data: np.ndarray) -> None:
        """Test histogram with Sturges' formula for bins."""
        counts, edges = histogram(normal_data, bins="sturges")

        # Sturges' formula: k = ceil(log2(n) + 1)
        expected_bins = int(np.ceil(np.log2(len(normal_data)) + 1))
        assert len(counts) == expected_bins

    def test_histogram_fd_bins(self, normal_data: np.ndarray) -> None:
        """Test histogram with Freedman-Diaconis rule."""
        counts, edges = histogram(normal_data, bins="fd")

        assert len(counts) > 0
        assert len(edges) == len(counts) + 1

    def test_histogram_explicit_edges(self, normal_data: np.ndarray) -> None:
        """Test histogram with explicit bin edges."""
        explicit_edges = np.linspace(0, 10, 11)  # 10 bins
        counts, edges = histogram(normal_data, bins=explicit_edges)

        assert len(counts) == 10
        assert len(edges) == 11
        np.testing.assert_array_equal(edges, explicit_edges)

    def test_histogram_density_integrates_to_one(self, normal_data: np.ndarray) -> None:
        """Test histogram with density=True integrates to 1."""
        counts, edges = histogram(normal_data, bins=50, density=True)

        bin_widths = np.diff(edges)
        integral = np.sum(counts * bin_widths)

        assert integral == pytest.approx(1.0, rel=0.01)

    def test_histogram_with_range(self, normal_data: np.ndarray) -> None:
        """Test histogram with specified range."""
        counts, edges = histogram(normal_data, bins=20, range=(0, 10))

        assert edges[0] == pytest.approx(0.0)
        assert edges[-1] == pytest.approx(10.0)

    def test_histogram_waveform_trace_input(self, waveform_trace: WaveformTrace) -> None:
        """Test histogram with WaveformTrace input."""
        counts, edges = histogram(waveform_trace, bins=50)

        assert len(counts) == 50
        assert len(edges) == 51

    def test_histogram_dtype(self, normal_data: np.ndarray) -> None:
        """Test histogram returns float64 arrays."""
        counts, edges = histogram(normal_data, bins=20)

        assert counts.dtype == np.float64
        assert edges.dtype == np.float64

    def test_histogram_total_counts(self, normal_data: np.ndarray) -> None:
        """Test histogram total counts equals data length."""
        counts, _edges = histogram(normal_data, bins=50, density=False)

        assert np.sum(counts) == len(normal_data)

    def test_histogram_uniform_distribution(self, uniform_data: np.ndarray) -> None:
        """Test histogram of uniform data has roughly equal bin counts."""
        counts, _edges = histogram(uniform_data, bins=10, range=(0, 10))

        # Each bin should have approximately n/10 = 100 samples
        expected_per_bin = len(uniform_data) / 10
        # Allow 50% variation due to randomness
        assert all(counts > expected_per_bin * 0.5)
        assert all(counts < expected_per_bin * 1.5)


# =============================================================================
# Distribution Metrics Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestDistributionMetrics:
    """Tests for distribution_metrics function."""

    def test_distribution_metrics_keys(self, normal_data: np.ndarray) -> None:
        """Test distribution_metrics returns required keys."""
        metrics = distribution_metrics(normal_data)

        assert "skewness" in metrics
        assert "kurtosis" in metrics
        assert "excess_kurtosis" in metrics
        assert "crest_factor" in metrics
        assert "crest_factor_db" in metrics

    def test_normal_distribution_skewness(self, normal_data: np.ndarray) -> None:
        """Test normal distribution has skewness near 0."""
        metrics = distribution_metrics(normal_data)

        # Skewness should be close to 0 for symmetric distributions
        assert abs(metrics["skewness"]) < 0.3

    def test_normal_distribution_kurtosis(self, normal_data: np.ndarray) -> None:
        """Test normal distribution has kurtosis near 3."""
        metrics = distribution_metrics(normal_data)

        # Pearson's kurtosis is 3 for normal
        assert metrics["kurtosis"] == pytest.approx(3.0, rel=0.3)
        # Excess kurtosis is 0 for normal
        assert abs(metrics["excess_kurtosis"]) < 0.5

    def test_skewed_distribution_positive_skewness(self, skewed_data: np.ndarray) -> None:
        """Test right-skewed distribution has positive skewness."""
        metrics = distribution_metrics(skewed_data)

        # Right-skewed data should have positive skewness
        assert metrics["skewness"] > 0.5

    def test_crest_factor_positive(self, normal_data: np.ndarray) -> None:
        """Test crest factor is positive."""
        metrics = distribution_metrics(normal_data)

        assert metrics["crest_factor"] > 0
        assert metrics["crest_factor_db"] > 0

    def test_crest_factor_calculation(self) -> None:
        """Test crest factor calculation is correct."""
        # Simple signal with known crest factor
        data = np.array([1.0, 1.0, 1.0, 1.0, 5.0])  # Peak=5, RMS=sqrt(6.6)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=100))

        metrics = distribution_metrics(trace)

        rms = np.sqrt(np.mean(data**2))
        expected_crest = 5.0 / rms
        assert metrics["crest_factor"] == pytest.approx(expected_crest, rel=1e-6)

    def test_crest_factor_db_conversion(self, normal_data: np.ndarray) -> None:
        """Test crest factor dB conversion."""
        metrics = distribution_metrics(normal_data)

        expected_db = 20 * np.log10(metrics["crest_factor"])
        assert metrics["crest_factor_db"] == pytest.approx(expected_db, rel=1e-6)

    def test_distribution_metrics_waveform_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test distribution_metrics with WaveformTrace input."""
        metrics = distribution_metrics(waveform_trace)

        assert "skewness" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_uniform_distribution_kurtosis(self, uniform_data: np.ndarray) -> None:
        """Test uniform distribution has lower kurtosis than normal."""
        metrics = distribution_metrics(uniform_data)

        # Uniform distribution has lower kurtosis (lighter tails)
        # Pearson's kurtosis for uniform is 1.8
        assert metrics["kurtosis"] < 2.5

    def test_zero_rms_crest_factor(self) -> None:
        """Test crest factor with zero RMS data."""
        data = np.zeros(100)
        metrics = distribution_metrics(data)

        # Zero RMS should give inf crest factor
        assert metrics["crest_factor"] == float("inf")
        assert metrics["crest_factor_db"] == float("inf")


# =============================================================================
# Moment Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestMoment:
    """Tests for moment function."""

    def test_second_central_moment_is_variance(self, normal_data: np.ndarray) -> None:
        """Test second central moment equals variance."""
        m2 = moment(normal_data, 2, central=True)
        var = np.var(normal_data)

        assert m2 == pytest.approx(var, rel=1e-10)

    def test_first_central_moment_is_zero(self, normal_data: np.ndarray) -> None:
        """Test first central moment is zero."""
        m1 = moment(normal_data, 1, central=True)

        assert m1 == pytest.approx(0.0, abs=1e-10)

    def test_raw_moment(self, normal_data: np.ndarray) -> None:
        """Test raw (non-central) moment."""
        raw_m2 = moment(normal_data, 2, central=False)
        central_m2 = moment(normal_data, 2, central=True)
        mean = np.mean(normal_data)

        # Raw second moment = central second moment + mean^2
        expected_raw = central_m2 + mean**2
        assert raw_m2 == pytest.approx(expected_raw, rel=1e-10)

    def test_higher_order_moments(self, standard_normal_data: np.ndarray) -> None:
        """Test higher order moments for standard normal."""
        # Third central moment (related to skewness)
        m3 = moment(standard_normal_data, 3, central=True)
        # Fourth central moment (related to kurtosis)
        m4 = moment(standard_normal_data, 4, central=True)

        # For standard normal: m3 should be near 0, m4 should be near 3*var^2 = 3
        assert abs(m3) < 0.5
        assert m4 == pytest.approx(3.0, rel=0.3)

    def test_moment_waveform_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test moment with WaveformTrace input."""
        m2 = moment(waveform_trace, 2, central=True)

        assert isinstance(m2, float)
        assert m2 > 0  # Variance is always positive


# =============================================================================
# Fit Distribution Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestFitDistribution:
    """Tests for fit_distribution function."""

    def test_fit_normal_distribution(self, normal_data: np.ndarray) -> None:
        """Test fitting normal distribution."""
        result = fit_distribution(normal_data, distribution="normal")

        assert "loc" in result  # mean
        assert "scale" in result  # std
        assert "ks_statistic" in result
        assert "p_value" in result

        # Parameters should be close to true values (mean=5, std=2)
        assert result["loc"] == pytest.approx(5.0, rel=0.1)
        assert result["scale"] == pytest.approx(2.0, rel=0.1)

    def test_fit_normal_high_p_value(self, normal_data: np.ndarray) -> None:
        """Test that normal data passes KS test for normal distribution."""
        result = fit_distribution(normal_data, distribution="normal")

        # P-value should be reasonably high (not rejecting null hypothesis)
        assert result["p_value"] > 0.01

    def test_fit_exponential_distribution(self, exponential_data: np.ndarray) -> None:
        """Test fitting exponential distribution."""
        result = fit_distribution(exponential_data, distribution="exponential")

        assert "loc" in result
        assert "scale" in result
        assert "p_value" in result

    def test_fit_lognormal_distribution(self, lognormal_data: np.ndarray) -> None:
        """Test fitting lognormal distribution."""
        result = fit_distribution(lognormal_data, distribution="lognormal")

        assert "shape" in result
        assert "loc" in result
        assert "scale" in result
        assert "p_value" in result

    def test_fit_uniform_distribution(self, uniform_data: np.ndarray) -> None:
        """Test fitting uniform distribution."""
        result = fit_distribution(uniform_data, distribution="uniform")

        assert "loc" in result  # lower bound
        assert "scale" in result  # width
        assert "p_value" in result

        # loc should be near 0, scale should be near 10
        assert result["loc"] == pytest.approx(0.0, abs=0.5)
        assert result["scale"] == pytest.approx(10.0, rel=0.1)

    def test_fit_mismatched_distribution(self, exponential_data: np.ndarray) -> None:
        """Test fitting wrong distribution gives low p-value."""
        # Fit normal to exponential data
        result = fit_distribution(exponential_data, distribution="normal")

        # P-value should be low (rejecting null hypothesis)
        # Note: With large samples, KS test is very sensitive
        assert result["p_value"] < 0.5

    def test_fit_unknown_distribution_raises(self, normal_data: np.ndarray) -> None:
        """Test fitting unknown distribution raises error."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            fit_distribution(normal_data, distribution="unknown")

    def test_fit_distribution_waveform_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test fit_distribution with WaveformTrace input."""
        result = fit_distribution(waveform_trace, distribution="normal")

        assert "loc" in result
        assert "scale" in result


# =============================================================================
# Normality Test Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestNormalityTest:
    """Tests for normality_test function."""

    def test_normality_test_result_structure(self, normal_data: np.ndarray) -> None:
        """Test normality_test returns required keys."""
        result = normality_test(normal_data)

        assert "statistic" in result
        assert "p_value" in result
        assert "is_normal" in result

    def test_normal_data_passes_shapiro(self, normal_data: np.ndarray) -> None:
        """Test normal data passes Shapiro-Wilk test."""
        result = normality_test(normal_data, method="shapiro")

        # Should be normal (p > 0.05)
        assert result["p_value"] > 0.05
        assert result["is_normal"]  # True-ish value

    def test_uniform_data_fails_shapiro(self, uniform_data: np.ndarray) -> None:
        """Test uniform data fails Shapiro-Wilk test."""
        result = normality_test(uniform_data, method="shapiro")

        # Should not be normal (p < 0.05)
        assert result["p_value"] < 0.05
        assert not result["is_normal"]  # False-ish value

    def test_dagostino_method(self, normal_data: np.ndarray) -> None:
        """Test D'Agostino-Pearson test."""
        result = normality_test(normal_data, method="dagostino")

        assert "statistic" in result
        assert "p_value" in result
        # Normal data should pass
        assert result["p_value"] > 0.01

    def test_anderson_method(self, normal_data: np.ndarray) -> None:
        """Test Anderson-Darling test."""
        result = normality_test(normal_data, method="anderson")

        assert "statistic" in result
        assert "p_value" in result

    def test_shapiro_subsamples_large_data(self) -> None:
        """Test that Shapiro-Wilk subsamples data > 5000 samples."""
        rng = np.random.default_rng(42)
        large_data = rng.normal(0, 1, 10000)

        # Should not raise error
        result = normality_test(large_data, method="shapiro")
        assert "statistic" in result

    def test_unknown_method_raises(self, normal_data: np.ndarray) -> None:
        """Test unknown method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            normality_test(normal_data, method="unknown")

    def test_normality_test_waveform_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test normality_test with WaveformTrace input."""
        result = normality_test(waveform_trace)

        assert "statistic" in result
        assert "p_value" in result

    def test_exponential_fails_normality(self, exponential_data: np.ndarray) -> None:
        """Test exponential data fails normality tests."""
        result = normality_test(exponential_data, method="shapiro")

        assert result["p_value"] < 0.05
        assert not result["is_normal"]  # False-ish value


# =============================================================================
# Bimodality Coefficient Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestBimodalityCoefficient:
    """Tests for bimodality_coefficient function."""

    def test_unimodal_low_coefficient(self, normal_data: np.ndarray) -> None:
        """Test unimodal data has low bimodality coefficient."""
        bc = bimodality_coefficient(normal_data)

        # Unimodal should be < 0.555 (with some margin)
        assert bc < 0.7

    def test_bimodal_high_coefficient(self, bimodal_data: np.ndarray) -> None:
        """Test bimodal data has high bimodality coefficient."""
        bc = bimodality_coefficient(bimodal_data)

        # Bimodal should be > 0.555
        assert bc > 0.5

    def test_bimodality_coefficient_range(self, normal_data: np.ndarray) -> None:
        """Test bimodality coefficient is in valid range."""
        bc = bimodality_coefficient(normal_data)

        # BC should be between 0 and ~1
        assert 0 <= bc <= 1.5

    def test_bimodality_coefficient_waveform_trace(self, waveform_trace: WaveformTrace) -> None:
        """Test bimodality_coefficient with WaveformTrace input."""
        bc = bimodality_coefficient(waveform_trace)

        assert isinstance(bc, float)
        assert bc >= 0

    def test_uniform_bimodality(self, uniform_data: np.ndarray) -> None:
        """Test bimodality coefficient for uniform data."""
        bc = bimodality_coefficient(uniform_data)

        # Uniform has low kurtosis, which can affect BC
        assert isinstance(bc, float)

    def test_strongly_bimodal_data(self) -> None:
        """Test bimodality coefficient for clearly separated modes."""
        rng = np.random.default_rng(42)
        # Two well-separated modes
        mode1 = rng.normal(-5, 0.5, 500)
        mode2 = rng.normal(5, 0.5, 500)
        strongly_bimodal = np.concatenate([mode1, mode2])

        bc = bimodality_coefficient(strongly_bimodal)

        # Strongly bimodal should have BC > 0.555
        assert bc > 0.555


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


@pytest.mark.unit
@pytest.mark.analyzer
class TestStatisticalDistributionEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_array_histogram(self) -> None:
        """Test histogram with empty array."""
        data = np.array([])

        # May raise error or return empty
        try:
            counts, edges = histogram(data, bins=10)
            assert len(counts) == 10
        except (ValueError, IndexError):
            pass  # Empty array handling varies

    def test_single_value_histogram(self) -> None:
        """Test histogram with single value."""
        data = np.array([5.0])
        counts, edges = histogram(data, bins=10)

        # Should work, with most bins empty
        assert np.sum(counts) == 1

    def test_constant_array_distribution_metrics(self) -> None:
        """Test distribution_metrics with constant array."""
        import warnings

        data = np.ones(100)

        # Constant array has undefined skewness/kurtosis (NaN or 0)
        # May raise warning for precision loss
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                metrics = distribution_metrics(data)
                # Just verify no crash
                assert "skewness" in metrics
            except (RuntimeWarning, ValueError):
                pass  # Acceptable to raise warning for constant data

    def test_very_small_dataset_normality(self) -> None:
        """Test normality test with very small dataset."""
        data = np.array([1.0, 2.0, 3.0])

        # Should work or raise meaningful error
        try:
            result = normality_test(data, method="shapiro")
            assert "p_value" in result
        except ValueError:
            pass  # Too small for test

    def test_nan_in_data(self) -> None:
        """Test handling of NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        # Behavior may vary - just check no crash
        try:
            metrics = distribution_metrics(data)
            # May have NaN in results
            assert "skewness" in metrics
        except (ValueError, RuntimeWarning):
            pass

    def test_inf_in_data(self) -> None:
        """Test handling of infinity values."""
        data = np.array([1.0, 2.0, np.inf, 4.0, 5.0])

        # Should handle gracefully
        try:
            metrics = distribution_metrics(data)
            assert "skewness" in metrics
        except (ValueError, RuntimeWarning):
            pass

    def test_large_dataset_performance(self) -> None:
        """Test functions work with large datasets."""
        rng = np.random.default_rng(42)
        large_data = rng.normal(0, 1, 100000)

        # Should complete without error
        counts, _edges = histogram(large_data, bins=100)
        assert len(counts) == 100

        metrics = distribution_metrics(large_data)
        assert "skewness" in metrics

    def test_negative_bins_raises(self, normal_data: np.ndarray) -> None:
        """Test negative bin count raises error."""
        with pytest.raises((ValueError, TypeError)):
            histogram(normal_data, bins=-10)

    def test_zero_bins_raises(self, normal_data: np.ndarray) -> None:
        """Test zero bins raises error."""
        with pytest.raises((ValueError, TypeError)):
            histogram(normal_data, bins=0)
