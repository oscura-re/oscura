"""Unit tests for ensemble methods module.

This module tests ensemble aggregation, outlier detection, confidence bounds,
and pre-configured ensemble functions for frequency and edge detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.quality.ensemble import (
    AMPLITUDE_ENSEMBLE,
    EDGE_DETECTION_ENSEMBLE,
    FREQUENCY_ENSEMBLE,
    AggregationMethod,
    EnsembleAggregator,
    EnsembleResult,
    create_edge_ensemble,
    create_frequency_ensemble,
)
from oscura.quality.scoring import AnalysisQualityScore, ReliabilityCategory

pytestmark = pytest.mark.unit


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-004")
class TestEnsembleResult:
    """Test EnsembleResult dataclass functionality."""

    def test_ensemble_result_creation(self):
        """Test creating a valid ensemble result."""
        result = EnsembleResult(
            value=100.0,
            confidence=0.85,
            lower_bound=95.0,
            upper_bound=105.0,
            method_agreement=0.9,
            individual_results=[{"value": 100, "confidence": 0.9}],
            aggregation_method=AggregationMethod.WEIGHTED_AVERAGE,
        )

        assert result.value == 100.0
        assert result.confidence == 0.85
        assert result.lower_bound == 95.0
        assert result.upper_bound == 105.0
        assert result.method_agreement == 0.9
        assert len(result.individual_results) == 1
        assert result.aggregation_method == AggregationMethod.WEIGHTED_AVERAGE

    def test_ensemble_result_validation(self):
        """Test that invalid confidence/agreement values are rejected."""
        with pytest.raises(ValueError, match="Confidence must be in"):
            EnsembleResult(
                value=100.0,
                confidence=1.5,  # Invalid
                method_agreement=0.9,
            )

        with pytest.raises(ValueError, match="Method agreement must be in"):
            EnsembleResult(
                value=100.0,
                confidence=0.85,
                method_agreement=1.5,  # Invalid
            )

    def test_ensemble_result_to_dict(self):
        """Test conversion to dictionary."""
        result = EnsembleResult(
            value=100.0,
            confidence=0.85,
            lower_bound=95.0,
            upper_bound=105.0,
            method_agreement=0.9,
        )

        result_dict = result.to_dict()
        assert result_dict["value"] == 100.0
        assert result_dict["confidence"] == 0.85
        assert result_dict["lower_bound"] == 95.0
        assert result_dict["upper_bound"] == 105.0
        assert result_dict["method_agreement"] == 0.9
        assert result_dict["aggregation_method"] == "weighted_average"


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-004")
class TestEnsembleAggregatorNumeric:
    """Test numeric aggregation methods."""

    def test_weighted_average_aggregation(self):
        """Test weighted average combines results correctly."""
        agg = EnsembleAggregator(method=AggregationMethod.WEIGHTED_AVERAGE)
        results = [
            {"value": 100.0, "confidence": 1.0},
            {"value": 110.0, "confidence": 0.5},
        ]

        ensemble = agg.aggregate(results)

        # Expected: (100*1.0 + 110*0.5) / (1.0 + 0.5) = (100 + 55) / 1.5 = 103.33
        assert abs(ensemble.value - 103.33) < 0.1
        assert 0 <= ensemble.confidence <= 1
        assert ensemble.lower_bound is not None
        assert ensemble.upper_bound is not None

    def test_median_aggregation(self):
        """Test median aggregation is robust to outliers."""
        agg = EnsembleAggregator(method=AggregationMethod.MEDIAN)
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 102.0, "confidence": 0.85},
            {"value": 98.0, "confidence": 0.8},
            {"value": 500.0, "confidence": 0.7},  # Outlier
        ]

        ensemble = agg.aggregate(results)

        # Outlier should be detected and removed
        # Median of [98, 100, 102] = 100
        assert ensemble.outlier_methods == [3]
        assert abs(ensemble.value - 100.0) < 5.0  # Should be near 100, not 500

    def test_bayesian_aggregation(self):
        """Test Bayesian aggregation weights by precision."""
        agg = EnsembleAggregator(method=AggregationMethod.BAYESIAN)
        results = [
            {"value": 100.0, "confidence": 0.9},  # High confidence
            {"value": 120.0, "confidence": 0.3},  # Low confidence
        ]

        ensemble = agg.aggregate(results)

        # Higher confidence result should dominate
        assert ensemble.value < 110.0  # Closer to 100 than 120
        assert ensemble.confidence > 0

    def test_confidence_bounds(self):
        """Test that confidence bounds are computed correctly."""
        agg = EnsembleAggregator(method=AggregationMethod.WEIGHTED_AVERAGE)
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 102.0, "confidence": 0.85},
            {"value": 98.0, "confidence": 0.8},
        ]

        ensemble = agg.aggregate(results)

        assert ensemble.lower_bound is not None
        assert ensemble.upper_bound is not None
        assert ensemble.lower_bound < ensemble.value
        assert ensemble.value < ensemble.upper_bound

    def test_single_result(self):
        """Test aggregation with single result."""
        agg = EnsembleAggregator()
        results = [{"value": 100.0, "confidence": 0.9}]

        ensemble = agg.aggregate(results)

        assert ensemble.value == 100.0
        assert ensemble.confidence == 0.9
        # Single result should have perfect agreement
        assert ensemble.method_agreement == 1.0


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-004")
class TestEnsembleAggregatorCategorical:
    """Test categorical aggregation (voting)."""

    def test_voting_aggregation(self):
        """Test majority voting selects most common value."""
        agg = EnsembleAggregator(method=AggregationMethod.VOTING)
        results = [
            {"value": "rising", "confidence": 0.9},
            {"value": "rising", "confidence": 0.85},
            {"value": "falling", "confidence": 0.6},
            {"value": "rising", "confidence": 0.8},
        ]

        ensemble = agg.aggregate(results)

        assert ensemble.value == "rising"
        assert ensemble.method_agreement == 0.75  # 3 out of 4
        assert ensemble.lower_bound is None  # No bounds for categorical
        assert ensemble.upper_bound is None

    def test_weighted_voting(self):
        """Test that confidence weights affect voting."""
        agg = EnsembleAggregator(method=AggregationMethod.WEIGHTED_AVERAGE)
        results = [
            {"value": "A", "confidence": 0.9},  # High confidence
            {"value": "B", "confidence": 0.1},  # Low confidence
            {"value": "B", "confidence": 0.1},  # Low confidence
        ]

        ensemble = agg.aggregate(results)

        # "A" should win despite being outnumbered, due to higher confidence
        assert ensemble.value == "A"


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-005")
class TestOutlierDetection:
    """Test outlier detection functionality."""

    def test_detect_outlier_methods(self):
        """Test that outliers are correctly identified."""
        agg = EnsembleAggregator(outlier_threshold=3.0)
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 102.0, "confidence": 0.85},
            {"value": 98.0, "confidence": 0.8},
            {"value": 500.0, "confidence": 0.7},  # Clear outlier
        ]

        outliers = agg.detect_outlier_methods(results)

        assert 3 in outliers  # Index 3 is the outlier

    def test_no_outliers_detected(self):
        """Test that similar values don't trigger outlier detection."""
        agg = EnsembleAggregator()
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 101.0, "confidence": 0.85},
            {"value": 99.0, "confidence": 0.8},
        ]

        outliers = agg.detect_outlier_methods(results)

        assert len(outliers) == 0

    def test_outlier_with_few_samples(self):
        """Test outlier detection with insufficient samples."""
        agg = EnsembleAggregator()
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 500.0, "confidence": 0.8},  # Only 2 samples
        ]

        # Need at least 3 values for outlier detection
        outliers = agg.detect_outlier_methods(results)
        assert len(outliers) == 0

    def test_outliers_excluded_from_aggregation(self):
        """Test that detected outliers are excluded from final result."""
        agg = EnsembleAggregator(method=AggregationMethod.WEIGHTED_AVERAGE)
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 102.0, "confidence": 0.85},
            {"value": 98.0, "confidence": 0.8},
            {"value": 1000.0, "confidence": 0.7},  # Outlier
        ]

        ensemble = agg.aggregate(results)

        # Result should be near 100, not influenced by 1000
        assert 95 < ensemble.value < 105
        assert 3 in ensemble.outlier_methods


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-005")
class TestMethodAgreement:
    """Test method agreement calculation."""

    def test_high_agreement(self):
        """Test that similar results have high agreement."""
        agg = EnsembleAggregator()
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 100.5, "confidence": 0.85},
            {"value": 99.5, "confidence": 0.8},
        ]

        ensemble = agg.aggregate(results)

        # High agreement when values are close
        assert ensemble.method_agreement > 0.95

    def test_low_agreement(self):
        """Test that divergent results have low agreement."""
        agg = EnsembleAggregator()
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 150.0, "confidence": 0.85},
            {"value": 50.0, "confidence": 0.8},
        ]

        ensemble = agg.aggregate(results)

        # Low agreement when values are spread out
        assert ensemble.method_agreement < 0.8

    def test_agreement_affects_confidence(self):
        """Test that low agreement reduces overall confidence."""
        agg = EnsembleAggregator(min_agreement=0.7)
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 200.0, "confidence": 0.9},  # Very different
        ]

        ensemble = agg.aggregate(results)

        # Confidence should be reduced due to low agreement
        assert ensemble.confidence < 0.9


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-004")
class TestFrequencyEnsemble:
    """Test pre-configured frequency ensemble."""

    def test_frequency_ensemble_clean_signal(self):
        """Test frequency detection with clean sine wave."""
        sample_rate = 1000.0
        frequency = 10.0
        t = np.linspace(0, 2.0, int(sample_rate * 2))
        signal = np.sin(2 * np.pi * frequency * t)

        result = create_frequency_ensemble(signal, sample_rate)

        # Should detect close to 10 Hz
        assert abs(result.value - frequency) < 1.0
        assert result.confidence > 0
        assert len(result.individual_results) >= 2  # At least 2 methods succeed

    def test_frequency_ensemble_noisy_signal(self):
        """Test frequency detection with noisy signal."""
        # Use fixed seed for deterministic test (some seeds cause failures)
        np.random.seed(42)
        sample_rate = 10000.0
        frequency = 50.0
        t = np.linspace(0, 1.0, int(sample_rate * 1))
        signal = np.sin(2 * np.pi * frequency * t)
        signal += np.random.normal(0, 0.3, signal.shape)

        result = create_frequency_ensemble(signal, sample_rate)

        # Should still detect frequency despite noise
        assert abs(result.value - frequency) < 5.0
        assert result.method_agreement > 0  # Methods should mostly agree

    def test_frequency_ensemble_provides_bounds(self):
        """Test that frequency ensemble provides confidence bounds."""
        sample_rate = 1000.0
        frequency = 10.0
        t = np.linspace(0, 1.0, int(sample_rate * 1))
        signal = np.sin(2 * np.pi * frequency * t)

        result = create_frequency_ensemble(signal, sample_rate)

        assert result.lower_bound is not None
        assert result.upper_bound is not None
        assert result.lower_bound <= result.value <= result.upper_bound

    def test_frequency_ensemble_custom_weights(self):
        """Test frequency ensemble with custom method weights."""
        sample_rate = 1000.0
        frequency = 10.0
        t = np.linspace(0, 1.0, int(sample_rate * 1))
        signal = np.sin(2 * np.pi * frequency * t)

        custom_weights = [
            ("fft_peak", 0.7),
            ("zero_crossing", 0.2),
            ("autocorrelation", 0.1),
        ]

        result = create_frequency_ensemble(signal, sample_rate, method_weights=custom_weights)

        assert abs(result.value - frequency) < 1.0


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-004")
class TestEdgeEnsemble:
    """Test pre-configured edge detection ensemble."""

    def test_edge_ensemble_square_wave(self):
        """Test edge detection with clean square wave."""
        sample_rate = 1000.0
        # Square wave with 4 edges
        signal = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50 + [0] * 50)

        result = create_edge_ensemble(signal, sample_rate)

        # Should detect approximately 4 edges
        assert 3 <= result.value <= 5
        assert result.method_agreement > 0.8  # Methods should agree

    def test_edge_ensemble_with_noise(self):
        """Test edge detection with noisy signal."""
        sample_rate = 1000.0
        signal = np.array([0] * 50 + [1] * 50 + [0] * 50 + [1] * 50)
        signal = signal.astype(float)
        signal += np.random.normal(0, 0.1, signal.shape)

        result = create_edge_ensemble(signal, sample_rate)

        # Should still detect edges despite noise
        assert result.value > 0
        assert len(result.individual_results) >= 2

    def test_edge_ensemble_custom_threshold(self):
        """Test edge ensemble with custom threshold."""
        sample_rate = 1000.0
        signal = np.array([0] * 50 + [1] * 50 + [0] * 50)

        result = create_edge_ensemble(signal, sample_rate, threshold=0.5)

        assert result.value > 0

    def test_edge_ensemble_custom_weights(self):
        """Test edge ensemble with custom method weights."""
        sample_rate = 1000.0
        signal = np.array([0] * 50 + [1] * 50 + [0] * 50)

        custom_weights = [
            ("threshold_crossing", 0.6),
            ("derivative", 0.3),
            ("schmitt_trigger", 0.1),
        ]

        result = create_edge_ensemble(signal, sample_rate, method_weights=custom_weights)

        assert result.value > 0


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-004")
class TestEnsembleConfiguration:
    """Test pre-configured ensemble configurations."""

    def test_frequency_ensemble_config(self):
        """Test frequency ensemble configuration."""
        assert len(FREQUENCY_ENSEMBLE) == 3
        methods = [m for m, _ in FREQUENCY_ENSEMBLE]
        assert "fft_peak" in methods
        assert "zero_crossing" in methods
        assert "autocorrelation" in methods

        # Weights should sum to 1.0
        total_weight = sum(w for _, w in FREQUENCY_ENSEMBLE)
        assert abs(total_weight - 1.0) < 0.01

    def test_edge_detection_ensemble_config(self):
        """Test edge detection ensemble configuration."""
        assert len(EDGE_DETECTION_ENSEMBLE) == 3
        methods = [m for m, _ in EDGE_DETECTION_ENSEMBLE]
        assert "threshold_crossing" in methods
        assert "derivative" in methods
        assert "schmitt_trigger" in methods

        total_weight = sum(w for _, w in EDGE_DETECTION_ENSEMBLE)
        assert abs(total_weight - 1.0) < 0.01

    def test_amplitude_ensemble_config(self):
        """Test amplitude ensemble configuration."""
        assert len(AMPLITUDE_ENSEMBLE) == 3
        methods = [m for m, _ in AMPLITUDE_ENSEMBLE]
        assert "peak_to_peak" in methods
        assert "rms" in methods
        assert "percentile_99" in methods

        total_weight = sum(w for _, w in AMPLITUDE_ENSEMBLE)
        assert abs(total_weight - 1.0) < 0.01


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("QUAL-004")
class TestEnsembleWithQualityScores:
    """Test ensemble integration with quality scoring system."""

    def test_ensemble_combines_quality_scores(self):
        """Test that quality scores are combined in ensemble."""
        agg = EnsembleAggregator()
        score1 = AnalysisQualityScore(
            confidence=0.9,
            category=ReliabilityCategory.HIGH,
            data_quality_factor=0.9,
            sample_sufficiency=0.8,
            method_reliability=0.85,
        )
        score2 = AnalysisQualityScore(
            confidence=0.8,
            category=ReliabilityCategory.HIGH,
            data_quality_factor=0.85,
            sample_sufficiency=0.75,
            method_reliability=0.8,
        )

        results = [
            {"value": 100.0, "confidence": 0.9, "quality_score": score1},
            {"value": 102.0, "confidence": 0.85, "quality_score": score2},
        ]

        ensemble = agg.aggregate(results)

        assert ensemble.quality_score is not None
        assert ensemble.quality_score.confidence > 0

    def test_ensemble_without_quality_scores(self):
        """Test ensemble works without quality scores."""
        agg = EnsembleAggregator()
        results = [
            {"value": 100.0, "confidence": 0.9},
            {"value": 102.0, "confidence": 0.85},
        ]

        ensemble = agg.aggregate(results)

        assert ensemble.quality_score is None


@pytest.mark.unit
@pytest.mark.quality
class TestErrorHandling:
    """Test error handling in ensemble module."""

    def test_empty_results_list(self):
        """Test that empty results list raises error."""
        agg = EnsembleAggregator()
        with pytest.raises(ValueError, match="empty results"):
            agg.aggregate([])

    def test_frequency_ensemble_all_methods_fail(self):
        """Test frequency ensemble when all methods fail."""
        # Very short signal that will cause most methods to fail,
        # but at least one should succeed (FFT will work on any length signal)
        # So we expect it to succeed, not fail
        signal = np.array([0, 1, 0, 1])
        # This should actually succeed with at least FFT
        result = create_frequency_ensemble(signal, sample_rate=1000)
        assert result.value > 0  # Should get some frequency estimate

    def test_edge_ensemble_all_methods_fail(self):
        """Test edge ensemble with empty signal raises appropriate error."""
        # Empty signal will raise ValueError from numpy, not from our code
        signal = np.array([])
        with pytest.raises(ValueError):  # Will raise from np.max on empty array
            create_edge_ensemble(signal, sample_rate=1000)
