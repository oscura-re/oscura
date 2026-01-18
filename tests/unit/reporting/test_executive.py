"""Comprehensive tests for executive summary auto-generation.

Tests for:

Test coverage:
- ExecutiveSummary dataclass initialization and fields
- generate_executive_summary with various result scenarios
- Edge cases and boundary conditions
- Summary text generation for different modes
- Key findings extraction
- Critical violation handling
"""

from __future__ import annotations

import pytest

from oscura.reporting.content.executive import (
    ExecutiveSummary,
    generate_executive_summary,
)

pytestmark = pytest.mark.unit


@pytest.mark.unit
class TestExecutiveSummary:
    """Test ExecutiveSummary dataclass."""

    def test_default_initialization(self):
        """Test creating ExecutiveSummary with minimal required fields."""
        summary = ExecutiveSummary(
            overall_status=True,
            pass_count=10,
            total_count=10,
        )

        assert summary.overall_status is True
        assert summary.pass_count == 10
        assert summary.total_count == 10
        assert summary.key_findings == []
        assert summary.critical_violations == []
        assert summary.min_margin_pct is None
        assert summary.summary_text == ""

    def test_full_initialization(self):
        """Test creating ExecutiveSummary with all fields."""
        summary = ExecutiveSummary(
            overall_status=False,
            pass_count=8,
            total_count=10,
            key_findings=["Finding 1", "Finding 2"],
            critical_violations=["Violation 1"],
            min_margin_pct=15.5,
            summary_text="Test summary text",
        )

        assert summary.overall_status is False
        assert summary.pass_count == 8
        assert summary.total_count == 10
        assert summary.key_findings == ["Finding 1", "Finding 2"]
        assert summary.critical_violations == ["Violation 1"]
        assert summary.min_margin_pct == 15.5
        assert summary.summary_text == "Test summary text"

    def test_mutable_defaults(self):
        """Test that mutable defaults work correctly (no shared state)."""
        summary1 = ExecutiveSummary(overall_status=True, pass_count=5, total_count=5)
        summary2 = ExecutiveSummary(overall_status=True, pass_count=5, total_count=5)

        summary1.key_findings.append("Finding 1")
        summary1.critical_violations.append("Violation 1")

        # summary2 should not be affected
        assert summary2.key_findings == []
        assert summary2.critical_violations == []


@pytest.mark.unit
class TestGenerateExecutiveSummary:
    """Test generate_executive_summary function."""

    def test_all_tests_passed_no_margin(self):
        """Test summary when all tests pass with no margin info."""
        results = {
            "pass_count": 10,
            "total_count": 10,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True
        assert summary.pass_count == 10
        assert summary.total_count == 10
        assert summary.key_findings == []
        assert summary.critical_violations == []
        assert summary.min_margin_pct is None
        assert "All 10 tests passed" in summary.summary_text

    def test_all_tests_passed_with_good_margin(self):
        """Test summary when all tests pass with good margin."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 30.5,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True
        assert summary.pass_count == 10
        assert summary.total_count == 10
        assert summary.min_margin_pct == 30.5
        assert "All 10 tests passed" in summary.summary_text
        assert "Minimum margin: 30.5%" in summary.summary_text

    def test_some_tests_failed(self):
        """Test summary when some tests fail."""
        results = {
            "pass_count": 8,
            "total_count": 10,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is False
        assert summary.pass_count == 8
        assert summary.total_count == 10
        assert "2 of 10 tests failed" in summary.summary_text
        assert "20% failure rate" in summary.summary_text

    def test_all_tests_failed(self):
        """Test summary when all tests fail."""
        results = {
            "pass_count": 0,
            "total_count": 5,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is False
        assert summary.pass_count == 0
        assert summary.total_count == 5
        assert "5 of 5 tests failed" in summary.summary_text
        assert "100% failure rate" in summary.summary_text

    def test_no_tests_run(self):
        """Test summary when no tests were run."""
        results = {
            "pass_count": 0,
            "total_count": 0,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True  # No failures
        assert summary.pass_count == 0
        assert summary.total_count == 0
        assert "Analysis completed successfully" in summary.summary_text

    def test_empty_results(self):
        """Test summary with empty results dictionary."""
        results: dict[str, int] = {}

        summary = generate_executive_summary(results)

        assert summary.overall_status is True  # No failures
        assert summary.pass_count == 0
        assert summary.total_count == 0
        assert "Analysis completed successfully" in summary.summary_text

    def test_critical_violations_detected(self):
        """Test summary with critical violations."""
        results = {
            "pass_count": 8,
            "total_count": 10,
            "violations": [
                {"severity": "critical", "message": "Critical issue 1"},
                {"severity": "critical", "message": "Critical issue 2"},
                {"severity": "warning", "message": "Warning issue"},
            ],
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is False
        assert len(summary.critical_violations) == 2
        assert len(summary.key_findings) >= 1
        assert "2 critical violation(s) require immediate attention" in summary.key_findings
        assert "Critical: 2 violation(s) require immediate action" in summary.summary_text

    def test_non_critical_violations_only(self):
        """Test summary with only non-critical violations."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"severity": "warning", "message": "Warning issue 1"},
                {"severity": "info", "message": "Info issue 1"},
            ],
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True
        assert len(summary.critical_violations) == 0
        assert "2 violation(s) detected" in summary.key_findings

    def test_marginal_margin(self):
        """Test summary with marginal margin (10-20%)."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 15.0,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True
        assert summary.min_margin_pct == 15.0
        assert any("15.0% (marginal)" in finding for finding in summary.key_findings)

    def test_critical_margin(self):
        """Test summary with critical margin (<10%)."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 5.5,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True
        assert summary.min_margin_pct == 5.5
        assert any("5.5% (critical)" in finding for finding in summary.key_findings)

    def test_good_margin_not_in_findings(self):
        """Test that good margin (>=20%) is not in key findings."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 25.0,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True
        assert summary.min_margin_pct == 25.0
        # Should not be in key findings, only in summary text
        assert not any("25.0%" in finding for finding in summary.key_findings)

    def test_max_findings_limit(self):
        """Test that max_findings parameter limits key findings."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [{"severity": "critical", "message": f"Issue {i}"} for i in range(10)],
            "min_margin": 5.0,
        }

        summary = generate_executive_summary(results, max_findings=3)

        assert len(summary.key_findings) <= 3

    def test_max_findings_custom_value(self):
        """Test custom max_findings value."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [{"severity": "warning", "message": f"Issue {i}"} for i in range(10)],
            "min_margin": 8.0,
        }

        summary = generate_executive_summary(results, max_findings=2)

        assert len(summary.key_findings) <= 2

    def test_short_length_mode(self):
        """Test short length mode (default)."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"severity": "warning", "message": "Issue 1"},
            ],
        }

        summary = generate_executive_summary(results, length="short")

        # Short mode should not include detailed findings
        assert "Key Findings:" not in summary.summary_text

    def test_detailed_length_mode(self):
        """Test detailed length mode."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"severity": "warning", "message": "Issue 1"},
            ],
        }

        summary = generate_executive_summary(results, length="detailed")

        # Detailed mode should include key findings section
        assert "Key Findings:" in summary.summary_text
        assert "1 violation(s) detected" in summary.summary_text

    def test_detailed_mode_with_no_findings(self):
        """Test detailed mode when there are no key findings."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 25.0,  # Good margin, not in findings
        }

        summary = generate_executive_summary(results, length="detailed")

        # Should not add Key Findings section if list is empty
        assert "Key Findings:" not in summary.summary_text

    def test_detailed_mode_respects_max_findings(self):
        """Test that detailed mode respects max_findings limit."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [{"severity": "warning", "message": f"Issue {i}"} for i in range(10)],
            "min_margin": 5.0,
        }

        summary = generate_executive_summary(results, length="detailed", max_findings=3)

        # Count bullet points in detailed findings
        findings_section = summary.summary_text.split("Key Findings:")[1]
        bullet_count = findings_section.count("  - ")
        assert bullet_count <= 3

    def test_violations_with_missing_severity(self):
        """Test handling violations without severity field."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"message": "Issue without severity"},
            ],
        }

        summary = generate_executive_summary(results)

        # Should not crash, should treat as non-critical
        assert len(summary.critical_violations) == 0

    def test_violations_case_insensitive_severity(self):
        """Test that severity check is case-insensitive."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"severity": "CRITICAL", "message": "Upper case"},
                {"severity": "Critical", "message": "Title case"},
                {"severity": "critical", "message": "Lower case"},
            ],
        }

        summary = generate_executive_summary(results)

        assert len(summary.critical_violations) == 3

    def test_complex_scenario_all_features(self):
        """Test complex scenario with multiple issues."""
        results = {
            "pass_count": 15,
            "total_count": 20,
            "min_margin": 8.5,
            "violations": [
                {"severity": "critical", "message": "Critical issue 1"},
                {"severity": "critical", "message": "Critical issue 2"},
                {"severity": "warning", "message": "Warning 1"},
                {"severity": "warning", "message": "Warning 2"},
            ],
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is False
        assert summary.pass_count == 15
        assert summary.total_count == 20
        assert len(summary.critical_violations) == 2
        assert summary.min_margin_pct == 8.5

        # Check key findings include both violations and margin
        assert len(summary.key_findings) >= 2
        assert any("critical violation" in f.lower() for f in summary.key_findings)
        assert any("8.5% (critical)" in f for f in summary.key_findings)

        # Check summary text
        assert "5 of 20 tests failed" in summary.summary_text
        assert "25% failure rate" in summary.summary_text
        assert "Critical: 2 violation(s)" in summary.summary_text

    def test_zero_margin(self):
        """Test handling of zero margin."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 0.0,
        }

        summary = generate_executive_summary(results)

        assert summary.min_margin_pct == 0.0
        assert any("0.0% (critical)" in finding for finding in summary.key_findings)

    def test_negative_margin(self):
        """Test handling of negative margin."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": -5.0,
        }

        summary = generate_executive_summary(results)

        assert summary.min_margin_pct == -5.0
        # Should still be reported as critical
        assert any("-5.0% (critical)" in finding for finding in summary.key_findings)

    def test_exactly_20_percent_margin(self):
        """Test boundary condition: exactly 20% margin."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 20.0,
        }

        summary = generate_executive_summary(results)

        assert summary.min_margin_pct == 20.0
        # At exactly 20%, should not be in findings (< 20 is the condition)
        assert not any("20.0%" in finding for finding in summary.key_findings)

    def test_exactly_10_percent_margin(self):
        """Test boundary condition: exactly 10% margin."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 10.0,
        }

        summary = generate_executive_summary(results)

        assert summary.min_margin_pct == 10.0
        # At exactly 10%, should be marginal (< 10 is critical)
        assert any("10.0% (marginal)" in finding for finding in summary.key_findings)

    def test_violation_string_conversion(self):
        """Test that violations are converted to strings."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"severity": "critical", "message": "Issue", "code": 123},
            ],
        }

        summary = generate_executive_summary(results)

        assert len(summary.critical_violations) == 1
        assert isinstance(summary.critical_violations[0], str)

    def test_failure_rate_rounding(self):
        """Test that failure rate percentage is rounded appropriately."""
        results = {
            "pass_count": 7,
            "total_count": 9,  # 2/9 = 22.222...%
        }

        summary = generate_executive_summary(results)

        assert "22% failure rate" in summary.summary_text

    def test_single_test_passed(self):
        """Test with single passing test."""
        results = {
            "pass_count": 1,
            "total_count": 1,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is True
        assert "All 1 tests passed" in summary.summary_text

    def test_single_test_failed(self):
        """Test with single failing test."""
        results = {
            "pass_count": 0,
            "total_count": 1,
        }

        summary = generate_executive_summary(results)

        assert summary.overall_status is False
        assert "1 of 1 tests failed" in summary.summary_text
        assert "100% failure rate" in summary.summary_text

    def test_violations_priority_in_findings(self):
        """Test that critical violations appear first in key findings."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "min_margin": 5.0,
            "violations": [
                {"severity": "critical", "message": "Critical issue"},
            ],
        }

        summary = generate_executive_summary(results)

        # Critical violations should be first finding
        assert summary.key_findings[0].startswith("1 critical violation")

    def test_max_findings_zero(self):
        """Test with max_findings set to 0."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"severity": "critical", "message": "Issue"},
            ],
            "min_margin": 5.0,
        }

        summary = generate_executive_summary(results, max_findings=0)

        assert summary.key_findings == []

    def test_pass_count_greater_than_total(self):
        """Test edge case where pass_count > total_count (data inconsistency)."""
        # This shouldn't happen in practice, but we should handle it gracefully
        results = {
            "pass_count": 12,
            "total_count": 10,
        }

        summary = generate_executive_summary(results)

        # With pass_count > total_count, fail_count will be negative
        # This results in overall_status being False
        assert summary.pass_count == 12
        assert summary.total_count == 10

    def test_detailed_mode_formatting(self):
        """Test that detailed mode has proper formatting."""
        results = {
            "pass_count": 10,
            "total_count": 10,
            "violations": [
                {"severity": "warning", "message": "Issue 1"},
                {"severity": "warning", "message": "Issue 2"},
            ],
            "min_margin": 8.0,
        }

        summary = generate_executive_summary(results, length="detailed")

        # Check formatting
        assert "Key Findings:\n" in summary.summary_text
        assert summary.summary_text.count("  - ") >= 1  # At least one bullet point
        lines = summary.summary_text.split("\n")
        # Key Findings header should be on its own line
        assert any(line.strip() == "Key Findings:" for line in lines)


@pytest.mark.unit
class TestExecutiveSummaryModuleExports:
    """Test module exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from oscura.reporting.content import executive

        assert hasattr(executive, "__all__")
        assert "ExecutiveSummary" in executive.__all__
        assert "generate_executive_summary" in executive.__all__
        assert len(executive.__all__) == 2


@pytest.mark.unit
class TestExecutiveSummaryIntegration:
    """Integration tests for executive summary generation."""

    def test_realistic_passing_analysis(self):
        """Test realistic scenario: analysis mostly passes."""
        results = {
            "pass_count": 47,
            "total_count": 50,
            "min_margin": 28.5,
            "violations": [
                {"severity": "warning", "message": "Signal noise elevated"},
                {"severity": "info", "message": "Timing variance detected"},
            ],
        }

        summary = generate_executive_summary(results, length="detailed")

        assert summary.overall_status is False  # Some tests failed
        assert summary.pass_count == 47
        assert summary.total_count == 50
        assert "3 of 50 tests failed" in summary.summary_text
        assert "6% failure rate" in summary.summary_text
        assert "2 violation(s) detected" in summary.key_findings

    def test_realistic_failing_analysis(self):
        """Test realistic scenario: analysis has critical failures."""
        results = {
            "pass_count": 8,
            "total_count": 25,
            "min_margin": 3.2,
            "violations": [
                {"severity": "critical", "message": "Protocol violation detected"},
                {"severity": "critical", "message": "Timing constraint exceeded"},
                {"severity": "critical", "message": "Data integrity check failed"},
                {"severity": "warning", "message": "Unexpected state transition"},
            ],
        }

        summary = generate_executive_summary(results, length="detailed", max_findings=5)

        assert summary.overall_status is False
        assert summary.pass_count == 8
        assert summary.total_count == 25
        assert len(summary.critical_violations) == 3
        assert "17 of 25 tests failed" in summary.summary_text
        assert "68% failure rate" in summary.summary_text
        assert "Critical: 3 violation(s)" in summary.summary_text
        assert "3 critical violation(s) require immediate attention" in summary.key_findings
        assert "3.2% (critical)" in str(summary.key_findings)
