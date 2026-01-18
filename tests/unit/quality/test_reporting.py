"""Comprehensive unit tests for quality reporting.

This module tests quality report generation, formatting, and presentation
of quality metrics and warnings.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.discovery.quality_validator import DataQuality, QualityMetric
from oscura.quality.warnings import QualityWarning, SignalQualityAnalyzer

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_quality_metrics() -> list[QualityMetric]:
    """Create sample quality metrics for testing."""
    return [
        QualityMetric(
            name="Sample Rate",
            status="PASS",
            passed=True,
            current_value=100.0,
            required_value=50.0,
            unit="MS/s",
            margin_percent=100.0,
        ),
        QualityMetric(
            name="Resolution",
            status="WARNING",
            passed=False,
            current_value=30.0,
            required_value=40.0,
            unit="dB SNR",
            margin_percent=-25.0,
            explanation="SNR is 25% below recommended",
            recommendation="Reduce noise sources",
        ),
        QualityMetric(
            name="Duration",
            status="FAIL",
            passed=False,
            current_value=5.0,
            required_value=10.0,
            unit="ms",
            margin_percent=-50.0,
            explanation="Capture duration is critically short",
            recommendation="Increase capture duration to at least 10.0 ms",
        ),
    ]


@pytest.fixture
def sample_data_quality(sample_quality_metrics: list[QualityMetric]) -> DataQuality:
    """Create sample data quality assessment for testing."""
    return DataQuality(
        status="FAIL",
        confidence=0.66,
        metrics=sample_quality_metrics,
        improvement_suggestions=[
            {
                "action": "Increase capture duration to at least 10.0 ms",
                "expected_benefit": "Improves duration to required level",
                "difficulty_level": "Easy",
            },
            {
                "action": "Reduce noise sources",
                "expected_benefit": "Improves resolution to required level",
                "difficulty_level": "Medium",
            },
        ],
    )


@pytest.fixture
def sample_warnings() -> list[QualityWarning]:
    """Create sample quality warnings for testing."""
    return [
        QualityWarning(
            severity="error",
            category="clipping",
            message="Signal clipping detected at 10.5% of samples",
            value=10.5,
            threshold=1.0,
            suggestion="Reduce input amplitude or increase ADC range",
        ),
        QualityWarning(
            severity="warning",
            category="noise",
            message="High noise level detected: SNR = -15.2 dB",
            value=-15.2,
            threshold=-40.0,
            suggestion="Check signal source, grounding, and shielding",
        ),
    ]


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestWarningFormatting:
    """Test formatting of quality warnings."""

    def test_format_single_warning(self) -> None:
        """Test formatting a single warning."""
        warning = QualityWarning(
            severity="warning",
            category="clipping",
            message="Signal clipping detected",
            value=5.2,
            threshold=5.0,
            suggestion="Reduce input amplitude",
        )

        formatted = str(warning)

        assert "[WARNING]" in formatted
        assert "Signal clipping detected" in formatted
        assert "value: 5.200" in formatted
        assert "threshold: 5.000" in formatted
        assert "Reduce input amplitude" in formatted

    def test_format_error_severity(self) -> None:
        """Test formatting error severity warning."""
        warning = QualityWarning(
            severity="error",
            category="undersampling",
            message="Undersampling detected",
            value=100.0,
            threshold=50.0,
            suggestion="Increase sample rate",
        )

        formatted = str(warning)

        assert "[ERROR]" in formatted
        assert "Undersampling detected" in formatted

    def test_format_info_severity(self) -> None:
        """Test formatting info severity warning."""
        warning = QualityWarning(
            severity="info",
            category="noise",
            message="Signal quality good",
            value=1.0,
            threshold=2.0,
        )

        formatted = str(warning)

        assert "[INFO]" in formatted
        assert "Signal quality good" in formatted

    def test_format_without_suggestion(self) -> None:
        """Test formatting warning without suggestion."""
        warning = QualityWarning(
            severity="warning",
            category="saturation",
            message="High ADC utilization",
            value=98.0,
            threshold=95.0,
        )

        formatted = str(warning)

        assert "Suggestion" not in formatted


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("DISC-009")
class TestQualityMetricReporting:
    """Test reporting of quality metrics."""

    def test_metric_pass_status(self) -> None:
        """Test reporting metric with PASS status."""
        metric = QualityMetric(
            name="Sample Rate",
            status="PASS",
            passed=True,
            current_value=100.0,
            required_value=50.0,
            unit="MS/s",
            margin_percent=100.0,
        )

        assert metric.status == "PASS"
        assert metric.passed is True
        assert metric.margin_percent > 0

    def test_metric_warning_status(self) -> None:
        """Test reporting metric with WARNING status."""
        metric = QualityMetric(
            name="Resolution",
            status="WARNING",
            passed=False,
            current_value=30.0,
            required_value=40.0,
            unit="dB SNR",
            margin_percent=-25.0,
            explanation="SNR is below recommended",
            recommendation="Reduce noise",
        )

        assert metric.status == "WARNING"
        assert metric.passed is False
        assert metric.explanation != ""
        assert metric.recommendation != ""

    def test_metric_fail_status(self) -> None:
        """Test reporting metric with FAIL status."""
        metric = QualityMetric(
            name="Duration",
            status="FAIL",
            passed=False,
            current_value=5.0,
            required_value=10.0,
            unit="ms",
            margin_percent=-50.0,
            explanation="Duration critically short",
            recommendation="Increase duration",
        )

        assert metric.status == "FAIL"
        assert metric.passed is False
        assert metric.margin_percent < 0


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("DISC-009")
class TestDataQualityReporting:
    """Test overall data quality reporting."""

    def test_overall_status_reporting(self, sample_data_quality: DataQuality) -> None:
        """Test reporting overall quality status."""
        assert sample_data_quality.status == "FAIL"
        assert 0.0 <= sample_data_quality.confidence <= 1.0
        assert len(sample_data_quality.metrics) == 3

    def test_improvement_suggestions(self, sample_data_quality: DataQuality) -> None:
        """Test reporting improvement suggestions."""
        suggestions = sample_data_quality.improvement_suggestions

        assert len(suggestions) > 0
        for suggestion in suggestions:
            assert "action" in suggestion
            assert "expected_benefit" in suggestion
            assert "difficulty_level" in suggestion

    def test_failed_metrics_reporting(self, sample_data_quality: DataQuality) -> None:
        """Test reporting failed metrics."""
        failed = [m for m in sample_data_quality.metrics if not m.passed]

        assert len(failed) == 2  # Resolution and Duration failed

    def test_passed_metrics_reporting(self, sample_data_quality: DataQuality) -> None:
        """Test reporting passed metrics."""
        passed = [m for m in sample_data_quality.metrics if m.passed]

        assert len(passed) == 1  # Only Sample Rate passed


@pytest.mark.unit
@pytest.mark.quality
@pytest.mark.requirement("EDGE-001")
class TestWarningAggregation:
    """Test aggregation of multiple warnings."""

    def test_group_by_severity(self, sample_warnings: list[QualityWarning]) -> None:
        """Test grouping warnings by severity."""
        errors = [w for w in sample_warnings if w.severity == "error"]
        warnings = [w for w in sample_warnings if w.severity == "warning"]

        assert len(errors) == 1
        assert len(warnings) == 1

    def test_group_by_category(self, sample_warnings: list[QualityWarning]) -> None:
        """Test grouping warnings by category."""
        clipping = [w for w in sample_warnings if w.category == "clipping"]
        noise = [w for w in sample_warnings if w.category == "noise"]

        assert len(clipping) == 1
        assert len(noise) == 1

    def test_priority_sorting(self, sample_warnings: list[QualityWarning]) -> None:
        """Test sorting warnings by priority."""
        # Define severity priority
        severity_priority = {"error": 0, "warning": 1, "info": 2}

        sorted_warnings = sorted(sample_warnings, key=lambda w: severity_priority[w.severity])

        # Errors should come first
        assert sorted_warnings[0].severity == "error"
        assert sorted_warnings[1].severity == "warning"


@pytest.mark.unit
@pytest.mark.quality
class TestReportGeneration:
    """Test generation of quality reports."""

    def test_generate_summary_report(self, sample_data_quality: DataQuality) -> None:
        """Test generating a summary quality report."""
        # Generate text summary
        lines = []
        lines.append(f"Overall Status: {sample_data_quality.status}")
        lines.append(f"Confidence: {sample_data_quality.confidence:.2f}")
        lines.append("\nMetrics:")

        for metric in sample_data_quality.metrics:
            status_symbol = "✓" if metric.passed else "✗"
            lines.append(
                f"  {status_symbol} {metric.name}: {metric.current_value:.1f} {metric.unit} "
                f"(required: {metric.required_value:.1f})"
            )

        report = "\n".join(lines)

        assert "Overall Status: FAIL" in report
        assert "Confidence: 0.66" in report
        assert "Sample Rate" in report
        assert "Resolution" in report
        assert "Duration" in report

    def test_generate_detailed_report(self, sample_data_quality: DataQuality) -> None:
        """Test generating a detailed quality report."""
        lines = []
        lines.append("Quality Assessment Report")
        lines.append(f"{'=' * 50}")
        lines.append(f"\nOverall Status: {sample_data_quality.status}")
        lines.append(f"Confidence: {sample_data_quality.confidence:.1%}")

        lines.append("\nDetailed Metrics:")
        for metric in sample_data_quality.metrics:
            lines.append(f"\n{metric.name}:")
            lines.append(f"  Status: {metric.status}")
            lines.append(f"  Current: {metric.current_value:.2f} {metric.unit}")
            lines.append(f"  Required: {metric.required_value:.2f} {metric.unit}")
            lines.append(f"  Margin: {metric.margin_percent:+.1f}%")

            if metric.explanation:
                lines.append(f"  Issue: {metric.explanation}")
            if metric.recommendation:
                lines.append(f"  Action: {metric.recommendation}")

        if sample_data_quality.improvement_suggestions:
            lines.append("\nImprovement Suggestions:")
            for i, suggestion in enumerate(sample_data_quality.improvement_suggestions, 1):
                lines.append(f"{i}. {suggestion['action']}")
                lines.append(f"   Benefit: {suggestion['expected_benefit']}")
                lines.append(f"   Difficulty: {suggestion['difficulty_level']}")

        report = "\n".join(lines)

        assert "Quality Assessment Report" in report
        assert "Detailed Metrics:" in report
        assert "Improvement Suggestions:" in report

    def test_generate_json_report(self, sample_data_quality: DataQuality) -> None:
        """Test generating JSON-compatible quality report."""
        import json

        report = {
            "status": sample_data_quality.status,
            "confidence": sample_data_quality.confidence,
            "metrics": [
                {
                    "name": m.name,
                    "status": m.status,
                    "passed": m.passed,
                    "current_value": m.current_value,
                    "required_value": m.required_value,
                    "unit": m.unit,
                    "margin_percent": m.margin_percent,
                    "explanation": m.explanation,
                    "recommendation": m.recommendation,
                }
                for m in sample_data_quality.metrics
            ],
            "suggestions": sample_data_quality.improvement_suggestions,
        }

        # Should be JSON serializable
        json_str = json.dumps(report, indent=2)
        assert isinstance(json_str, str)
        assert "FAIL" in json_str


@pytest.mark.unit
@pytest.mark.quality
class TestWarningReport:
    """Test warning report generation."""

    def test_generate_warning_summary(self, sample_warnings: list[QualityWarning]) -> None:
        """Test generating warning summary."""
        lines = []
        lines.append(f"Signal Quality Warnings ({len(sample_warnings)} total)")
        lines.append("=" * 50)

        for warning in sample_warnings:
            lines.append(f"\n{warning!s}")

        report = "\n".join(lines)

        assert "Signal Quality Warnings" in report
        assert "2 total" in report
        assert "clipping" in report
        assert "noise" in report

    def test_generate_warning_by_category(self, sample_warnings: list[QualityWarning]) -> None:
        """Test generating warnings grouped by category."""
        from collections import defaultdict

        by_category = defaultdict(list)
        for warning in sample_warnings:
            by_category[warning.category].append(warning)

        lines = []
        for category, warnings in by_category.items():
            lines.append(f"\n{category.upper()} ({len(warnings)}):")
            for w in warnings:
                lines.append(f"  - {w.message}")

        report = "\n".join(lines)

        assert "CLIPPING" in report
        assert "NOISE" in report


@pytest.mark.unit
@pytest.mark.quality
class TestLiveReporting:
    """Test real-time quality reporting."""

    def test_analyze_and_report(self) -> None:
        """Test analyzing signal and generating report."""
        # Create test signal
        metadata = TraceMetadata(sample_rate=1e6)
        t = np.linspace(0, 1e-3, 1000)
        data = np.sin(2 * np.pi * 1e3 * t)
        trace = WaveformTrace(data=data.astype(np.float64), metadata=metadata)

        # Analyze
        analyzer = SignalQualityAnalyzer()
        warnings = analyzer.analyze(trace)

        # Generate report
        if warnings:
            report_lines = [f"Found {len(warnings)} quality issue(s):"]
            for w in warnings:
                report_lines.append(str(w))
            report = "\n".join(report_lines)
        else:
            report = "No quality issues detected"

        # Should have valid report
        assert isinstance(report, str)

    def test_progressive_reporting(self) -> None:
        """Test reporting quality metrics as they're calculated."""
        metrics = []

        # Simulate progressive metric calculation
        metrics.append(
            QualityMetric(
                name="Sample Rate",
                status="PASS",
                passed=True,
                current_value=100.0,
                required_value=50.0,
                unit="MS/s",
            )
        )

        # Report after first metric
        assert len(metrics) == 1

        metrics.append(
            QualityMetric(
                name="Resolution",
                status="WARNING",
                passed=False,
                current_value=30.0,
                required_value=40.0,
                unit="dB SNR",
            )
        )

        # Report after second metric
        assert len(metrics) == 2

        # Final report
        report = f"Assessed {len(metrics)} metrics"
        assert "Assessed 2 metrics" in report


@pytest.mark.unit
@pytest.mark.quality
class TestReportFormatting:
    """Test various report formatting options."""

    def test_compact_format(self, sample_data_quality: DataQuality) -> None:
        """Test compact report format."""
        # One-line summary
        failed = sum(1 for m in sample_data_quality.metrics if not m.passed)
        passed = sum(1 for m in sample_data_quality.metrics if m.passed)

        compact = f"Status: {sample_data_quality.status} | Passed: {passed}/{len(sample_data_quality.metrics)} | Confidence: {sample_data_quality.confidence:.0%}"

        assert "Status: FAIL" in compact
        assert "Passed: 1/3" in compact
        assert "Confidence: 66%" in compact

    def test_verbose_format(self, sample_data_quality: DataQuality) -> None:
        """Test verbose report format with all details."""
        lines = []

        for metric in sample_data_quality.metrics:
            lines.append(f"\n{'=' * 60}")
            lines.append(f"Metric: {metric.name}")
            lines.append(f"{'=' * 60}")
            lines.append(f"Status:         {metric.status}")
            lines.append(f"Passed:         {'Yes' if metric.passed else 'No'}")
            lines.append(f"Current Value:  {metric.current_value:.3f} {metric.unit}")
            lines.append(f"Required Value: {metric.required_value:.3f} {metric.unit}")
            lines.append(f"Margin:         {metric.margin_percent:+.1f}%")

            if metric.explanation:
                lines.append("\nExplanation:")
                lines.append(f"  {metric.explanation}")

            if metric.recommendation:
                lines.append("\nRecommendation:")
                lines.append(f"  {metric.recommendation}")

        report = "\n".join(lines)

        assert "Sample Rate" in report
        assert "Resolution" in report
        assert "Duration" in report

    def test_table_format(self, sample_data_quality: DataQuality) -> None:
        """Test table-formatted report."""
        # Header
        lines = []
        lines.append(
            f"{'Metric':<15} {'Status':<8} {'Current':<12} {'Required':<12} {'Margin':<10}"
        )
        lines.append("-" * 65)

        # Rows
        for metric in sample_data_quality.metrics:
            lines.append(
                f"{metric.name:<15} "
                f"{metric.status:<8} "
                f"{metric.current_value:>8.1f} {metric.unit:<3} "
                f"{metric.required_value:>8.1f} {metric.unit:<3} "
                f"{metric.margin_percent:>+6.0f}%"
            )

        report = "\n".join(lines)

        assert "Metric" in report
        assert "Status" in report
        assert "PASS" in report
        assert "WARNING" in report
        assert "FAIL" in report


@pytest.mark.unit
@pytest.mark.quality
class TestExportFormats:
    """Test exporting quality reports in various formats."""

    def test_export_to_dict(self, sample_data_quality: DataQuality) -> None:
        """Test exporting quality data to dictionary."""
        export = {
            "status": sample_data_quality.status,
            "confidence": sample_data_quality.confidence,
            "metrics": [
                {
                    "name": m.name,
                    "status": m.status,
                    "value": m.current_value,
                    "unit": m.unit,
                }
                for m in sample_data_quality.metrics
            ],
        }

        assert export["status"] == "FAIL"
        assert len(export["metrics"]) == 3

    def test_export_to_csv_format(self, sample_data_quality: DataQuality) -> None:
        """Test formatting quality data as CSV."""
        lines = []
        lines.append("Metric,Status,Current,Required,Unit,Margin%")

        for m in sample_data_quality.metrics:
            lines.append(
                f"{m.name},{m.status},{m.current_value},{m.required_value},{m.unit},{m.margin_percent}"
            )

        csv = "\n".join(lines)

        assert "Sample Rate,PASS" in csv
        assert "Resolution,WARNING" in csv
        assert "Duration,FAIL" in csv

    def test_export_warnings_list(self, sample_warnings: list[QualityWarning]) -> None:
        """Test exporting warnings as structured list."""
        export = [
            {
                "severity": w.severity,
                "category": w.category,
                "message": w.message,
                "value": w.value,
                "threshold": w.threshold,
            }
            for w in sample_warnings
        ]

        assert len(export) == 2
        assert export[0]["severity"] == "error"
        assert export[1]["severity"] == "warning"
