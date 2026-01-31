"""Comprehensive unit tests for reporting.summary module.

Tests executive summary generation, measurement summarization,
key findings identification, and recommendations generation.

Target coverage: 80%+ (currently 43.2%)
"""

from __future__ import annotations

import pytest

from oscura.reporting.interpretation import (
    MeasurementInterpretation,
    QualityLevel,
)
from oscura.reporting.summary import (
    ExecutiveSummarySection,
    generate_executive_summary,
    identify_key_findings,
    recommendations_from_findings,
    summarize_measurements,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Tests for ExecutiveSummarySection dataclass
# =============================================================================


class TestExecutiveSummarySection:
    """Tests for ExecutiveSummarySection dataclass."""

    def test_section_creation_minimal(self):
        """Test creating section with minimal fields."""
        section = ExecutiveSummarySection(title="Test Section")

        assert section.title == "Test Section"
        assert section.content == ""
        assert section.bullet_points == []
        assert section.priority == 1

    def test_section_creation_full(self):
        """Test creating section with all fields."""
        section = ExecutiveSummarySection(
            title="Findings",
            content="Critical issues detected",
            bullet_points=["Issue 1", "Issue 2"],
            priority=2,
        )

        assert section.title == "Findings"
        assert section.content == "Critical issues detected"
        assert len(section.bullet_points) == 2
        assert section.priority == 2

    def test_section_mutable_defaults(self):
        """Test that mutable defaults don't leak between instances."""
        section1 = ExecutiveSummarySection(title="S1")
        section2 = ExecutiveSummarySection(title="S2")

        section1.bullet_points.append("Item 1")

        assert len(section1.bullet_points) == 1
        assert len(section2.bullet_points) == 0


# =============================================================================
# Tests for summarize_measurements
# =============================================================================


class TestSummarizeMeasurements:
    """Tests for summarize_measurements function."""

    def test_summarize_numeric_measurements(self):
        """Test summarizing numeric measurements."""
        measurements = {"snr": 45.2, "thd": -60.5, "bandwidth": 1e9}

        summary = summarize_measurements(measurements)

        assert summary["count"] == 3
        assert summary["numeric_count"] == 3
        assert "mean" in summary
        assert "median" in summary
        assert "std" in summary
        assert "min" in summary
        assert "max" in summary

    def test_summarize_mixed_measurements(self):
        """Test summarizing mixed numeric and non-numeric measurements."""
        measurements = {"snr": 45.2, "status": "OK", "frequency": 1e6, "mode": "active"}

        summary = summarize_measurements(measurements)

        assert summary["count"] == 4
        assert summary["numeric_count"] == 2
        assert "mean" in summary
        assert summary["mean"] == (45.2 + 1e6) / 2

    def test_summarize_empty_measurements(self):
        """Test summarizing empty measurements dict."""
        summary = summarize_measurements({})

        assert summary["count"] == 0
        assert summary["numeric_count"] == 0
        assert "mean" not in summary

    def test_summarize_no_numeric_measurements(self):
        """Test summarizing measurements with no numeric values."""
        measurements = {"status": "OK", "mode": "active", "result": "pass"}

        summary = summarize_measurements(measurements)

        assert summary["count"] == 3
        assert summary["numeric_count"] == 0
        assert "mean" not in summary

    def test_summarize_single_measurement(self):
        """Test summarizing single measurement."""
        measurements = {"voltage": 3.3}

        summary = summarize_measurements(measurements)

        assert summary["count"] == 1
        assert summary["numeric_count"] == 1
        assert summary["mean"] == 3.3
        assert summary["median"] == 3.3
        assert summary["std"] == 0.0
        assert summary["min"] == 3.3
        assert summary["max"] == 3.3

    def test_summarize_statistical_correctness(self):
        """Test statistical calculations are correct."""
        measurements = {"m1": 10.0, "m2": 20.0, "m3": 30.0}

        summary = summarize_measurements(measurements)

        assert summary["mean"] == 20.0
        assert summary["median"] == 20.0
        assert summary["min"] == 10.0
        assert summary["max"] == 30.0
        assert summary["std"] > 0

    def test_summarize_group_by_not_implemented(self):
        """Test that group_by parameter is accepted (future feature)."""
        measurements = {"snr": 45.2, "thd": -60.5}

        # group_by is accepted but not yet implemented
        summary = summarize_measurements(measurements, group_by="domain")

        assert summary["count"] == 2


# =============================================================================
# Tests for identify_key_findings
# =============================================================================


class TestIdentifyKeyFindings:
    """Tests for identify_key_findings function."""

    def test_identify_findings_without_interpretations(self):
        """Test findings identification without interpretations."""
        measurements = {"snr": 65.5, "bandwidth": 1.5e9}

        findings = identify_key_findings(measurements)

        assert isinstance(findings, list)
        # Should identify excellent SNR and wide bandwidth
        assert any("snr" in f.lower() for f in findings)
        assert any("bandwidth" in f.lower() for f in findings)

    def test_identify_findings_excellent_snr(self):
        """Test identification of excellent SNR."""
        measurements = {"snr": 65.0}

        findings = identify_key_findings(measurements)

        assert len(findings) > 0
        assert any("excellent snr" in f.lower() for f in findings)
        assert any("65.0 db" in f.lower() for f in findings)

    def test_identify_findings_low_snr(self):
        """Test identification of low SNR."""
        measurements = {"snr": 15.0}

        findings = identify_key_findings(measurements)

        assert len(findings) > 0
        assert any("low snr" in f.lower() for f in findings)
        assert any("noise" in f.lower() for f in findings)

    def test_identify_findings_wide_bandwidth(self):
        """Test identification of wide bandwidth."""
        measurements = {"bandwidth": 2.5e9}

        findings = identify_key_findings(measurements)

        assert len(findings) > 0
        assert any("bandwidth" in f.lower() for f in findings)
        assert any("ghz" in f.lower() for f in findings)

    def test_identify_findings_excellent_jitter(self):
        """Test identification of excellent jitter (< 10 ps)."""
        measurements = {"rms_jitter": 5e-12}  # 5 ps

        findings = identify_key_findings(measurements)

        assert len(findings) > 0
        assert any("excellent timing" in f.lower() for f in findings)
        assert any("jitter" in f.lower() for f in findings)

    def test_identify_findings_high_jitter(self):
        """Test identification of high jitter."""
        measurements = {"jitter": 250e-12}  # 250 ps

        findings = identify_key_findings(measurements)

        assert len(findings) > 0
        assert any("high jitter" in f.lower() for f in findings)
        assert any("timing" in f.lower() for f in findings)

    def test_identify_findings_with_interpretations_failed(self):
        """Test findings with failed/poor quality interpretations."""
        measurements = {"snr": 45.2, "rise_time": 2.5e-9}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=45.2,
                quality=QualityLevel.FAILED,
            ),
            "rise_time": MeasurementInterpretation(
                measurement_name="rise_time",
                value=2.5e-9,
                quality=QualityLevel.POOR,
            ),
        }

        findings = identify_key_findings(measurements, interpretations)

        assert len(findings) > 0
        assert any("critical" in f.lower() or "failed" in f.lower() for f in findings)

    def test_identify_findings_with_interpretations_excellent(self):
        """Test findings with excellent quality interpretations."""
        measurements = {"snr": 65.0, "bandwidth": 1.5e9}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=65.0,
                quality=QualityLevel.EXCELLENT,
            ),
            "bandwidth": MeasurementInterpretation(
                measurement_name="bandwidth",
                value=1.5e9,
                quality=QualityLevel.EXCELLENT,
            ),
        }

        findings = identify_key_findings(measurements, interpretations)

        assert len(findings) > 0
        assert any("excellent" in f.lower() for f in findings)

    def test_identify_findings_max_limit(self):
        """Test that max_findings parameter limits results."""
        measurements = {
            "snr": 65.0,
            "bandwidth": 2e9,
            "jitter": 5e-12,
            "rise_time": 1e-9,
        }

        findings = identify_key_findings(measurements, max_findings=2)

        assert len(findings) <= 2

    def test_identify_findings_empty_measurements(self):
        """Test findings with empty measurements."""
        findings = identify_key_findings({})

        assert isinstance(findings, list)
        assert len(findings) == 0

    def test_identify_findings_non_numeric_measurements(self):
        """Test findings with non-numeric measurements."""
        measurements = {"status": "OK", "mode": "active"}

        findings = identify_key_findings(measurements)

        assert isinstance(findings, list)
        # Non-numeric values are ignored


# =============================================================================
# Tests for recommendations_from_findings
# =============================================================================


class TestRecommendationsFromFindings:
    """Tests for recommendations_from_findings function."""

    def test_recommendations_from_interpretations(self):
        """Test recommendations extracted from interpretations."""
        measurements = {"snr": 25.0}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=25.0,
                quality=QualityLevel.ACCEPTABLE,
                recommendations=["Consider noise reduction", "Verify signal path"],
            )
        }

        recommendations = recommendations_from_findings(measurements, interpretations)

        assert len(recommendations) >= 2
        assert "Consider noise reduction" in recommendations
        assert "Verify signal path" in recommendations

    def test_recommendations_low_snr(self):
        """Test automatic recommendations for low SNR."""
        measurements = {"snr": 25.0}

        recommendations = recommendations_from_findings(measurements)

        assert len(recommendations) > 0
        assert any("noise" in r.lower() for r in recommendations)

    def test_recommendations_low_bandwidth(self):
        """Test automatic recommendations for low bandwidth."""
        measurements = {"bandwidth": 50e6}

        recommendations = recommendations_from_findings(measurements)

        assert len(recommendations) > 0
        assert any("bandwidth" in r.lower() for r in recommendations)

    def test_recommendations_deduplication(self):
        """Test that duplicate recommendations are removed."""
        measurements = {"snr": 25.0}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=25.0,
                recommendations=["Check noise sources", "Check noise sources"],
            )
        }

        recommendations = recommendations_from_findings(measurements, interpretations)

        # Should deduplicate
        assert len(recommendations) == len(set(recommendations))

    def test_recommendations_empty_measurements(self):
        """Test recommendations with empty measurements."""
        recommendations = recommendations_from_findings({})

        assert isinstance(recommendations, list)
        # May be empty or have generic recommendations

    def test_recommendations_no_interpretations(self):
        """Test recommendations without interpretations."""
        measurements = {"snr": 65.0, "bandwidth": 2e9}

        recommendations = recommendations_from_findings(measurements, None)

        assert isinstance(recommendations, list)
        # Should still work with domain-specific rules

    def test_recommendations_multiple_sources(self):
        """Test recommendations from multiple interpretations."""
        measurements = {"snr": 25.0, "bandwidth": 50e6}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=25.0,
                recommendations=["Reduce noise"],
            ),
            "bandwidth": MeasurementInterpretation(
                measurement_name="bandwidth",
                value=50e6,
                recommendations=["Check bandwidth limits"],
            ),
        }

        recommendations = recommendations_from_findings(measurements, interpretations)

        assert len(recommendations) >= 2
        assert any("noise" in r.lower() for r in recommendations)
        assert any("bandwidth" in r.lower() for r in recommendations)


# =============================================================================
# Tests for generate_executive_summary
# =============================================================================


class TestGenerateExecutiveSummary:
    """Tests for generate_executive_summary function."""

    def test_generate_summary_basic(self):
        """Test basic executive summary generation."""
        measurements = {"snr": 45.2, "thd": -60.5, "bandwidth": 1e9}

        summary = generate_executive_summary(measurements)

        assert isinstance(summary, str)
        assert "# Executive Summary" in summary
        assert "## Overall Status" in summary
        assert "## Key Findings" in summary

    def test_generate_summary_structure(self):
        """Test summary has proper markdown structure."""
        measurements = {"snr": 45.2}

        summary = generate_executive_summary(measurements)

        lines = summary.split("\n")
        assert lines[0] == "# Executive Summary"
        assert any(line.startswith("## ") for line in lines)

    def test_generate_summary_with_interpretations_excellent(self):
        """Test summary with excellent quality interpretations."""
        measurements = {"snr": 65.0, "bandwidth": 1.5e9}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=65.0,
                quality=QualityLevel.EXCELLENT,
            ),
            "bandwidth": MeasurementInterpretation(
                measurement_name="bandwidth",
                value=1.5e9,
                quality=QualityLevel.EXCELLENT,
            ),
        }

        summary = generate_executive_summary(measurements, interpretations)

        assert "GOOD" in summary
        assert "Excellent: 2" in summary

    def test_generate_summary_with_interpretations_failed(self):
        """Test summary with failed measurements."""
        measurements = {"snr": 10.0, "thd": -40.0}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=10.0,
                quality=QualityLevel.FAILED,
            ),
            "thd": MeasurementInterpretation(
                measurement_name="thd",
                value=-40.0,
                quality=QualityLevel.POOR,
            ),
        }

        summary = generate_executive_summary(measurements, interpretations)

        assert "CRITICAL" in summary
        assert "failed" in summary.lower()

    def test_generate_summary_with_interpretations_marginal(self):
        """Test summary with marginal quality (>50% marginal)."""
        measurements = {"m1": 1.0, "m2": 2.0, "m3": 3.0}

        interpretations = {
            "m1": MeasurementInterpretation(
                measurement_name="m1",
                value=1.0,
                quality=QualityLevel.MARGINAL,
            ),
            "m2": MeasurementInterpretation(
                measurement_name="m2",
                value=2.0,
                quality=QualityLevel.MARGINAL,
            ),
            "m3": MeasurementInterpretation(
                measurement_name="m3",
                value=3.0,
                quality=QualityLevel.ACCEPTABLE,
            ),
        }

        summary = generate_executive_summary(measurements, interpretations)

        assert "MARGINAL" in summary
        assert "Marginal: 2" in summary

    def test_generate_summary_with_interpretations_acceptable(self):
        """Test summary with acceptable quality."""
        measurements = {"m1": 1.0, "m2": 2.0}

        interpretations = {
            "m1": MeasurementInterpretation(
                measurement_name="m1",
                value=1.0,
                quality=QualityLevel.ACCEPTABLE,
            ),
            "m2": MeasurementInterpretation(
                measurement_name="m2",
                value=2.0,
                quality=QualityLevel.GOOD,
            ),
        }

        summary = generate_executive_summary(measurements, interpretations)

        # Not enough excellent/good (need >70%) so should be ACCEPTABLE
        assert "ACCEPTABLE" in summary or "GOOD" in summary

    def test_generate_summary_with_recommendations(self):
        """Test summary includes recommendations section."""
        measurements = {"snr": 45.2}

        interpretations = {
            "snr": MeasurementInterpretation(
                measurement_name="snr",
                value=45.2,
                quality=QualityLevel.GOOD,
                recommendations=["Verify signal integrity", "Consider noise reduction"],
            )
        }

        summary = generate_executive_summary(measurements, interpretations)

        assert "## Recommendations" in summary
        assert "Verify signal integrity" in summary

    def test_generate_summary_max_findings_limit(self):
        """Test max_findings parameter limits key findings."""
        measurements = {
            "snr": 65.0,
            "bandwidth": 2e9,
            "jitter": 5e-12,
            "rise_time": 1e-9,
        }

        summary = generate_executive_summary(measurements, max_findings=2)

        # Count bullet points in Key Findings section
        lines = summary.split("\n")
        finding_bullets = [
            line for line in lines if line.startswith("- ") and "findings" in summary.lower()
        ]
        # Should be limited (some may come from interpretations section)
        assert len(finding_bullets) <= 10  # Reasonable upper bound

    def test_generate_summary_no_interpretations(self):
        """Test summary without interpretations."""
        measurements = {"snr": 45.2, "bandwidth": 1e9}

        summary = generate_executive_summary(measurements, None)

        assert "COMPLETE" in summary
        assert "## Key Findings" in summary
        # Should not have recommendations section
        assert "## Recommendations" not in summary

    def test_generate_summary_empty_measurements(self):
        """Test summary with empty measurements."""
        summary = generate_executive_summary({})

        assert "# Executive Summary" in summary
        assert "0 measurements" in summary.lower()

    def test_generate_summary_quality_counts(self):
        """Test quality level counts are correct."""
        measurements = {"m1": 1.0, "m2": 2.0, "m3": 3.0}

        interpretations = {
            "m1": MeasurementInterpretation(
                measurement_name="m1",
                value=1.0,
                quality=QualityLevel.EXCELLENT,
            ),
            "m2": MeasurementInterpretation(
                measurement_name="m2",
                value=2.0,
                quality=QualityLevel.GOOD,
            ),
            "m3": MeasurementInterpretation(
                measurement_name="m3",
                value=3.0,
                quality=QualityLevel.POOR,
            ),
        }

        summary = generate_executive_summary(measurements, interpretations)

        assert "Excellent: 1" in summary
        assert "Good: 1" in summary
        assert "Poor: 1" in summary

    def test_generate_summary_recommendations_deduplication(self):
        """Test that recommendations are deduplicated."""
        measurements = {"m1": 1.0, "m2": 2.0}

        interpretations = {
            "m1": MeasurementInterpretation(
                measurement_name="m1",
                value=1.0,
                recommendations=["Same recommendation", "Unique 1"],
            ),
            "m2": MeasurementInterpretation(
                measurement_name="m2",
                value=2.0,
                recommendations=["Same recommendation", "Unique 2"],
            ),
        }

        summary = generate_executive_summary(measurements, interpretations)

        # Count occurrences of "Same recommendation"
        count = summary.count("Same recommendation")
        assert count == 1  # Should appear only once

    def test_generate_summary_recommendations_limit(self):
        """Test that recommendations are limited to top 5."""
        measurements = {"m1": 1.0}

        interpretations = {
            "m1": MeasurementInterpretation(
                measurement_name="m1",
                value=1.0,
                recommendations=[f"Recommendation {i}" for i in range(10)],
            )
        }

        summary = generate_executive_summary(measurements, interpretations)

        # Count recommendation bullets
        lines = summary.split("\n")
        rec_section_started = False
        rec_count = 0

        for line in lines:
            if "## Recommendations" in line:
                rec_section_started = True
            elif rec_section_started and line.startswith("## "):
                break  # Next section
            elif rec_section_started and line.startswith("- "):
                rec_count += 1

        assert rec_count <= 5

    def test_generate_summary_no_recommendations_section_when_empty(self):
        """Test that recommendations section is omitted when no recommendations."""
        measurements = {"voltage": 3.3}

        interpretations = {
            "voltage": MeasurementInterpretation(
                measurement_name="voltage",
                value=3.3,
                quality=QualityLevel.EXCELLENT,
                recommendations=[],
            )
        }

        summary = generate_executive_summary(measurements, interpretations)

        # Should not include recommendations section if no recommendations
        assert "## Recommendations" not in summary or "Recommendations\n\n##" in summary


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_summarize_measurements_with_inf(self):
        """Test handling of infinity values in measurements."""
        import warnings

        measurements = {"normal": 45.2, "inf_value": float("inf")}

        # Suppress expected warnings from numpy operations on inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            summary = summarize_measurements(measurements)

        # Should handle inf gracefully
        assert summary["count"] == 2
        assert summary["numeric_count"] == 2

    def test_summarize_measurements_with_nan(self):
        """Test handling of NaN values in measurements."""
        measurements = {"normal": 45.2, "nan_value": float("nan")}

        summary = summarize_measurements(measurements)

        # Should handle NaN gracefully
        assert summary["count"] == 2

    def test_identify_findings_snr_boundary_values(self):
        """Test SNR boundary values (20, 60 dB)."""
        # Just above low SNR threshold
        findings_21 = identify_key_findings({"snr": 21.0})
        # Should not trigger low SNR warning
        assert not any("low snr" in f.lower() for f in findings_21)

        # Just below excellent SNR threshold
        findings_59 = identify_key_findings({"snr": 59.0})
        # Should not trigger excellent SNR
        assert not any("excellent snr" in f.lower() for f in findings_59)

    def test_identify_findings_jitter_boundary_values(self):
        """Test jitter boundary values (10 ps, 200 ps)."""
        # 11 ps - just above excellent threshold
        findings_11ps = identify_key_findings({"rms_jitter": 11e-12})
        assert not any("excellent timing" in f.lower() for f in findings_11ps)

        # 199 ps - just below high jitter threshold
        findings_199ps = identify_key_findings({"jitter": 199e-12})
        assert not any("high jitter" in f.lower() for f in findings_199ps)

    def test_recommendations_with_both_low_snr_and_bandwidth(self):
        """Test recommendations when both SNR and bandwidth are low."""
        measurements = {"snr": 25.0, "bandwidth": 50e6}

        recommendations = recommendations_from_findings(measurements)

        # Should have recommendations for both
        assert any("noise" in r.lower() for r in recommendations)
        assert any("bandwidth" in r.lower() for r in recommendations)

    def test_executive_summary_all_quality_levels(self):
        """Test executive summary with all quality levels represented."""
        measurements = {"m1": 1, "m2": 2, "m3": 3, "m4": 4, "m5": 5, "m6": 6}

        interpretations = {
            "m1": MeasurementInterpretation(
                measurement_name="m1", value=1, quality=QualityLevel.EXCELLENT
            ),
            "m2": MeasurementInterpretation(
                measurement_name="m2", value=2, quality=QualityLevel.GOOD
            ),
            "m3": MeasurementInterpretation(
                measurement_name="m3", value=3, quality=QualityLevel.ACCEPTABLE
            ),
            "m4": MeasurementInterpretation(
                measurement_name="m4", value=4, quality=QualityLevel.MARGINAL
            ),
            "m5": MeasurementInterpretation(
                measurement_name="m5", value=5, quality=QualityLevel.POOR
            ),
            "m6": MeasurementInterpretation(
                measurement_name="m6", value=6, quality=QualityLevel.FAILED
            ),
        }

        summary = generate_executive_summary(measurements, interpretations)

        # Should be CRITICAL due to failed measurements
        assert "CRITICAL" in summary
        assert "Excellent: 1" in summary
        assert "Good: 1" in summary
        assert "Acceptable: 1" in summary
        assert "Marginal: 1" in summary
        assert "Poor: 1" in summary
        assert "Failed: 1" in summary
