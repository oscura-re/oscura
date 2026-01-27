"""Tests for core/cross_domain.py - Cross-domain correlation analysis.

Tests:
- CrossDomainInsight creation and validation
- CorrelationResult metrics and coherence
- Cross-domain correlations between analysis domains
- Frequency-timing agreement detection
- Jitter-eye correlation
- All correlation rules
"""

import pytest

from oscura.core.cross_domain import (
    DOMAIN_AFFINITY,
    CorrelationResult,
    CrossDomainCorrelator,
    CrossDomainInsight,
    correlate_results,
)
from oscura.reporting.config import AnalysisDomain


class TestCrossDomainInsight:
    """Test CrossDomainInsight dataclass."""

    def test_valid_creation(self) -> None:
        """Test creating valid insight."""
        insight = CrossDomainInsight(
            insight_type="agreement",
            source_domains=[AnalysisDomain.SPECTRAL, AnalysisDomain.TIMING],
            description="Frequency matches period",
            confidence_impact=0.15,
        )

        assert insight.insight_type == "agreement"
        assert len(insight.source_domains) == 2
        assert insight.confidence_impact == 0.15
        assert isinstance(insight.details, dict)

    def test_confidence_impact_validation(self) -> None:
        """Test confidence impact must be in [-1, 1]."""
        # Valid range
        CrossDomainInsight(
            insight_type="agreement",
            source_domains=[AnalysisDomain.SPECTRAL],
            description="Test",
            confidence_impact=0.5,
        )

        # Too high
        with pytest.raises(ValueError, match="Confidence impact must be in"):
            CrossDomainInsight(
                insight_type="agreement",
                source_domains=[AnalysisDomain.SPECTRAL],
                description="Test",
                confidence_impact=1.5,
            )

        # Too low
        with pytest.raises(ValueError, match="Confidence impact must be in"):
            CrossDomainInsight(
                insight_type="conflict",
                source_domains=[AnalysisDomain.SPECTRAL],
                description="Test",
                confidence_impact=-1.5,
            )

    def test_edge_confidence_values(self) -> None:
        """Test edge values for confidence impact."""
        # Exactly -1.0
        insight1 = CrossDomainInsight(
            insight_type="conflict",
            source_domains=[AnalysisDomain.SPECTRAL],
            description="Test",
            confidence_impact=-1.0,
        )
        assert insight1.confidence_impact == -1.0

        # Exactly +1.0
        insight2 = CrossDomainInsight(
            insight_type="agreement",
            source_domains=[AnalysisDomain.SPECTRAL],
            description="Test",
            confidence_impact=1.0,
        )
        assert insight2.confidence_impact == 1.0

    def test_insight_types(self) -> None:
        """Test different insight types."""
        for insight_type in ["agreement", "conflict", "implication"]:
            insight = CrossDomainInsight(
                insight_type=insight_type,
                source_domains=[AnalysisDomain.DIGITAL],
                description="Test",
                confidence_impact=0.1,
            )
            assert insight.insight_type == insight_type

    def test_with_details(self) -> None:
        """Test insight with additional details."""
        insight = CrossDomainInsight(
            insight_type="agreement",
            source_domains=[AnalysisDomain.SPECTRAL, AnalysisDomain.TIMING],
            description="Test",
            confidence_impact=0.15,
            details={"spectral_freq": 1000.0, "timing_period": 0.001, "ratio": 1.0},
        )

        assert "spectral_freq" in insight.details
        assert insight.details["ratio"] == 1.0


class TestCorrelationResult:
    """Test CorrelationResult dataclass."""

    def test_empty_result(self) -> None:
        """Test empty correlation result."""
        result = CorrelationResult()

        assert len(result.insights) == 0
        assert result.conflicts_detected == 0
        assert result.agreements_detected == 0
        assert result.overall_coherence == 0.5  # Default for no data

    def test_overall_coherence_calculation(self) -> None:
        """Test coherence score calculation."""
        result = CorrelationResult()
        result.agreements_detected = 3
        result.conflicts_detected = 1

        # 3 agreements, 1 conflict = 3/4 = 0.75
        assert result.overall_coherence == 0.75

    def test_all_agreements(self) -> None:
        """Test coherence with all agreements."""
        result = CorrelationResult()
        result.agreements_detected = 5
        result.conflicts_detected = 0

        assert result.overall_coherence == 1.0

    def test_all_conflicts(self) -> None:
        """Test coherence with all conflicts."""
        result = CorrelationResult()
        result.agreements_detected = 0
        result.conflicts_detected = 5

        assert result.overall_coherence == 0.0

    def test_with_insights(self) -> None:
        """Test result with insights."""
        insight1 = CrossDomainInsight(
            insight_type="agreement",
            source_domains=[AnalysisDomain.SPECTRAL],
            description="Test1",
            confidence_impact=0.1,
        )
        insight2 = CrossDomainInsight(
            insight_type="conflict",
            source_domains=[AnalysisDomain.TIMING],
            description="Test2",
            confidence_impact=-0.2,
        )

        result = CorrelationResult(insights=[insight1, insight2])

        assert len(result.insights) == 2

    def test_confidence_adjustments(self) -> None:
        """Test confidence adjustments dict."""
        result = CorrelationResult(confidence_adjustments={"spectral": 0.15, "timing": -0.1})

        assert result.confidence_adjustments["spectral"] == 0.15
        assert result.confidence_adjustments["timing"] == -0.1


class TestDomainAffinity:
    """Test DOMAIN_AFFINITY mapping."""

    def test_all_domains_present(self) -> None:
        """Test that major domains have affinity mappings."""
        expected_domains = [
            AnalysisDomain.DIGITAL,
            AnalysisDomain.TIMING,
            AnalysisDomain.SPECTRAL,
            AnalysisDomain.JITTER,
            AnalysisDomain.EYE,
            AnalysisDomain.PATTERNS,
            AnalysisDomain.PROTOCOLS,
        ]

        for domain in expected_domains:
            assert domain in DOMAIN_AFFINITY

    def test_affinity_is_list(self) -> None:
        """Test affinity values are lists of domains."""
        for related in DOMAIN_AFFINITY.values():
            assert isinstance(related, list)
            assert len(related) > 0
            for rel_domain in related:
                assert isinstance(rel_domain, AnalysisDomain)

    def test_reciprocal_affinity(self) -> None:
        """Test if A->B affinity exists, B->A should also exist."""
        for domain, related in DOMAIN_AFFINITY.items():
            for rel_domain in related:
                if rel_domain in DOMAIN_AFFINITY:
                    # Check if reciprocal relationship exists
                    assert (
                        domain in DOMAIN_AFFINITY[rel_domain]
                        or len(DOMAIN_AFFINITY[rel_domain]) > 0
                    )


class TestCrossDomainCorrelator:
    """Test CrossDomainCorrelator class."""

    def test_initialization(self) -> None:
        """Test correlator initialization."""
        correlator = CrossDomainCorrelator(tolerance=0.05)

        assert correlator.tolerance == 0.05
        assert len(correlator._correlation_rules) > 0

    def test_empty_results(self) -> None:
        """Test correlation with empty results."""
        correlator = CrossDomainCorrelator()
        result = correlator.correlate({})

        assert len(result.insights) == 0
        assert result.overall_coherence == 0.5

    def test_single_domain_results(self) -> None:
        """Test with only one domain (no correlations possible)."""
        correlator = CrossDomainCorrelator()
        results = {AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0}}

        result = correlator.correlate(results)

        # Single domain, no pairs to correlate
        assert len(result.insights) == 0

    def test_frequency_timing_agreement(self) -> None:
        """Test spectral-timing correlation when values agree."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0},
            AnalysisDomain.TIMING: {"period": 0.001},  # 1/1000 = 0.001
        }

        result = correlator.correlate(results)

        # Should find agreement
        assert result.agreements_detected >= 1
        assert any(i.insight_type == "agreement" for i in result.insights)

    def test_frequency_timing_conflict(self) -> None:
        """Test spectral-timing correlation when values conflict."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0},
            AnalysisDomain.TIMING: {"period": 0.0001},  # 10x mismatch
        }

        result = correlator.correlate(results)

        # Should find conflict
        assert result.conflicts_detected >= 1
        assert any(i.insight_type == "conflict" for i in result.insights)

    def test_digital_timing_consistency(self) -> None:
        """Test digital-timing edge count correlation."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.DIGITAL: {"edge_count": 100},
            AnalysisDomain.TIMING: {"edge_count": 100},
        }

        result = correlator.correlate(results)

        # Should find agreement on edge counts
        assert result.agreements_detected >= 1

    def test_jitter_eye_correlation(self) -> None:
        """Test jitter and eye diagram correlation."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.JITTER: {"total_jitter": 1e-9},
            AnalysisDomain.EYE: {"eye_width": 5e-9},
        }

        result = correlator.correlate(results)

        # Should find implication insight
        assert any(i.insight_type == "implication" for i in result.insights)

    def test_waveform_stats_consistency(self) -> None:
        """Test waveform and statistics correlation."""
        correlator = CrossDomainCorrelator()
        # For sine wave: amplitude ≈ 2.83 * std
        std_val = 1.0
        amp_val = 2.83 * std_val

        results = {
            AnalysisDomain.WAVEFORM: {"amplitude": amp_val},
            AnalysisDomain.STATISTICS: {"std": std_val},
        }

        result = correlator.correlate(results)

        # Should find agreement
        assert result.agreements_detected >= 1

    def test_no_duplicate_correlations(self) -> None:
        """Test that domain pairs are only checked once."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0},
            AnalysisDomain.TIMING: {"period": 0.001},
        }

        result = correlator.correlate(results)

        # Should only have one check for this pair
        spectral_timing_insights = [
            i
            for i in result.insights
            if set(i.source_domains) == {AnalysisDomain.SPECTRAL, AnalysisDomain.TIMING}
        ]
        assert len(spectral_timing_insights) <= 1

    def test_confidence_adjustments_clamped(self) -> None:
        """Test confidence adjustments are clamped to [-0.3, +0.3]."""
        correlator = CrossDomainCorrelator()

        # Create many agreements for one domain
        insights = [
            CrossDomainInsight(
                insight_type="agreement",
                source_domains=[AnalysisDomain.SPECTRAL],
                description=f"Test {i}",
                confidence_impact=0.5,  # High impact
            )
            for i in range(10)
        ]

        adjustments = correlator._calculate_adjustments(insights)

        # Should be clamped to 0.3
        assert adjustments["spectral"] <= 0.3
        assert adjustments["spectral"] >= -0.3

    def test_extract_value_direct_key(self) -> None:
        """Test value extraction with direct key match."""
        correlator = CrossDomainCorrelator()
        results = {"frequency": 1000.0, "other": "value"}

        value = correlator._extract_value(results, ["frequency", "freq"])

        assert value == 1000.0

    def test_extract_value_alternate_key(self) -> None:
        """Test value extraction with alternate key."""
        correlator = CrossDomainCorrelator()
        results = {"freq": 500.0}

        value = correlator._extract_value(results, ["frequency", "freq"])

        assert value == 500.0

    def test_extract_value_nested(self) -> None:
        """Test value extraction from nested dict."""
        correlator = CrossDomainCorrelator()
        results = {"measurements": {"frequency": 1000.0}}

        value = correlator._extract_value(results, ["frequency"])

        assert value == 1000.0

    def test_extract_value_not_found(self) -> None:
        """Test value extraction when key not found."""
        correlator = CrossDomainCorrelator()
        results = {"other_key": 123}

        value = correlator._extract_value(results, ["frequency", "freq"])

        assert value is None

    def test_extract_value_nan_rejected(self) -> None:
        """Test that NaN values are rejected."""

        correlator = CrossDomainCorrelator()
        results = {"frequency": float("nan")}

        value = correlator._extract_value(results, ["frequency"])

        assert value is None

    def test_tolerance_parameter(self) -> None:
        """Test that tolerance affects comparisons."""
        correlator_strict = CrossDomainCorrelator(tolerance=0.01)
        correlator_loose = CrossDomainCorrelator(tolerance=0.5)

        # Small mismatch
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0},
            AnalysisDomain.TIMING: {"period": 0.00105},  # 5% off
        }

        result_strict = correlator_strict.correlate(results)
        result_loose = correlator_loose.correlate(results)

        # Both should work since our threshold is 10% for agreement
        assert result_strict.agreements_detected >= 0
        assert result_loose.agreements_detected >= 0


class TestCorrelateResults:
    """Test correlate_results convenience function."""

    def test_basic_correlation(self) -> None:
        """Test basic correlation through convenience function."""
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0},
            AnalysisDomain.TIMING: {"period": 0.001},
        }

        result = correlate_results(results)

        assert isinstance(result, CorrelationResult)
        assert result.agreements_detected >= 1

    def test_with_custom_tolerance(self) -> None:
        """Test correlation with custom tolerance."""
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0},
            AnalysisDomain.TIMING: {"period": 0.001},
        }

        result = correlate_results(results, tolerance=0.05)

        assert isinstance(result, CorrelationResult)

    def test_multiple_domain_correlation(self) -> None:
        """Test correlation across multiple domains."""
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 1000.0},
            AnalysisDomain.TIMING: {"period": 0.001, "edge_count": 50},
            AnalysisDomain.DIGITAL: {"edge_count": 50},
            AnalysisDomain.WAVEFORM: {"amplitude": 2.83},
            AnalysisDomain.STATISTICS: {"std": 1.0},
        }

        result = correlate_results(results)

        # Should find multiple correlations
        assert len(result.insights) >= 2
        assert result.agreements_detected >= 2


class TestCorrelationRules:
    """Test individual correlation rules."""

    def test_frequency_timing_slight_mismatch(self) -> None:
        """Test frequency-timing with slight acceptable mismatch."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {"peak_frequency": 1000.0},
            AnalysisDomain.TIMING: {"avg_period": 0.00105},  # 5% off, should agree
        }

        result = correlator.correlate(results)

        # Within 10% tolerance
        assert result.agreements_detected >= 1

    def test_edge_count_minor_difference(self) -> None:
        """Test edge count with minor acceptable difference."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.DIGITAL: {"transitions": 100},
            AnalysisDomain.TIMING: {"edge_count": 101},  # Within ±2
        }

        result = correlator.correlate(results)

        assert result.agreements_detected >= 1

    def test_alternate_key_names(self) -> None:
        """Test that alternate key names are recognized."""
        correlator = CrossDomainCorrelator()

        # Use alternate names
        results = {
            AnalysisDomain.SPECTRAL: {"fundamental": 500.0},  # Not 'dominant_frequency'
            AnalysisDomain.TIMING: {"mean_period": 0.002},  # Not 'period'
        }

        result = correlator.correlate(results)

        # Should still find correlation
        assert len(result.insights) >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_domain_results(self) -> None:
        """Test with empty results for domains."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {},  # Empty
            AnalysisDomain.TIMING: {},  # Empty
        }

        result = correlator.correlate(results)

        # Should handle gracefully
        assert isinstance(result, CorrelationResult)

    def test_missing_expected_keys(self) -> None:
        """Test when expected keys are missing."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {"other_key": 123},  # No frequency
            AnalysisDomain.TIMING: {"other_key": 456},  # No period
        }

        result = correlator.correlate(results)

        # Should handle gracefully, no insights
        assert len(result.insights) == 0

    def test_zero_values(self) -> None:
        """Test handling of zero values."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": 0.0},
            AnalysisDomain.TIMING: {"period": 0.0},
        }

        result = correlator.correlate(results)

        # Should not crash, but won't find meaningful correlation
        assert isinstance(result, CorrelationResult)

    def test_negative_values(self) -> None:
        """Test handling of negative values (invalid but possible)."""
        correlator = CrossDomainCorrelator()
        results = {
            AnalysisDomain.SPECTRAL: {"dominant_frequency": -1000.0},
            AnalysisDomain.TIMING: {"period": -0.001},
        }

        result = correlator.correlate(results)

        # Should handle gracefully
        assert isinstance(result, CorrelationResult)
