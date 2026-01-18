"""Tests for smart content filtering."""

import pytest

from oscura.reporting.content.filtering import (
    AudienceType,
    ContentFilter,
    calculate_relevance_score,
    filter_by_audience,
    filter_by_severity,
)
from oscura.reporting.formatting.standards import Severity

pytestmark = pytest.mark.unit


class TestAudienceType:
    """Tests for AudienceType enum."""

    def test_audience_type_values(self):
        """Test that all expected audience types are defined."""
        expected_values = ["executive", "engineering", "debug", "regulatory", "production"]

        for value in expected_values:
            assert any(at.value == value for at in AudienceType)

    def test_audience_type_from_string(self):
        """Test creating AudienceType from string."""
        assert AudienceType("executive") == AudienceType.EXECUTIVE
        assert AudienceType("engineering") == AudienceType.ENGINEERING
        assert AudienceType("debug") == AudienceType.DEBUG
        assert AudienceType("regulatory") == AudienceType.REGULATORY
        assert AudienceType("production") == AudienceType.PRODUCTION

    def test_audience_type_invalid_raises(self):
        """Test that invalid audience type raises ValueError."""
        with pytest.raises(ValueError):
            AudienceType("invalid_audience")


class TestContentFilter:
    """Tests for ContentFilter dataclass."""

    def test_content_filter_defaults(self):
        """Test default values for ContentFilter."""
        filter_config = ContentFilter()

        assert filter_config.severity_threshold == Severity.INFO
        assert filter_config.audience == AudienceType.ENGINEERING
        assert filter_config.show_only == "all"
        assert filter_config.hide_empty_sections is True
        assert filter_config.relevance_threshold == 0.5

    def test_content_filter_custom_values(self):
        """Test creating ContentFilter with custom values."""
        filter_config = ContentFilter(
            severity_threshold=Severity.CRITICAL,
            audience=AudienceType.EXECUTIVE,
            show_only="violations",
            hide_empty_sections=False,
            relevance_threshold=0.8,
        )

        assert filter_config.severity_threshold == Severity.CRITICAL
        assert filter_config.audience == AudienceType.EXECUTIVE
        assert filter_config.show_only == "violations"
        assert filter_config.hide_empty_sections is False
        assert filter_config.relevance_threshold == 0.8

    def test_content_filter_show_only_options(self):
        """Test valid show_only options."""
        # Test valid options
        for option in ["all", "violations", "changes"]:
            filter_config = ContentFilter(show_only=option)  # type: ignore
            assert filter_config.show_only == option

    def test_content_filter_relevance_threshold_bounds(self):
        """Test relevance threshold can be set to boundary values."""
        filter_min = ContentFilter(relevance_threshold=0.0)
        filter_max = ContentFilter(relevance_threshold=1.0)

        assert filter_min.relevance_threshold == 0.0
        assert filter_max.relevance_threshold == 1.0


class TestFilterBySeverity:
    """Tests for filter_by_severity function (REPORT-005)."""

    def test_filter_by_severity_with_severity_enum(self):
        """Test filtering with Severity enum."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2", "severity": "warning"},
            {"name": "test3", "severity": "info"},
        ]

        filtered = filter_by_severity(items, Severity.WARNING)

        assert len(filtered) == 2
        assert filtered[0]["name"] == "test1"
        assert filtered[1]["name"] == "test2"

    def test_filter_by_severity_with_string(self):
        """Test filtering with severity string."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2", "severity": "warning"},
            {"name": "test3", "severity": "info"},
        ]

        filtered = filter_by_severity(items, "warning")

        assert len(filtered) == 2
        assert filtered[0]["name"] == "test1"
        assert filtered[1]["name"] == "test2"

    def test_filter_by_severity_critical_only(self):
        """Test filtering for critical severity only."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2", "severity": "error"},
            {"name": "test3", "severity": "warning"},
            {"name": "test4", "severity": "info"},
        ]

        filtered = filter_by_severity(items, Severity.CRITICAL)

        assert len(filtered) == 1
        assert filtered[0]["name"] == "test1"

    def test_filter_by_severity_info_includes_all(self):
        """Test that INFO severity includes all items."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2", "severity": "error"},
            {"name": "test3", "severity": "warning"},
            {"name": "test4", "severity": "info"},
        ]

        filtered = filter_by_severity(items, Severity.INFO)

        assert len(filtered) == 4

    def test_filter_by_severity_case_insensitive(self):
        """Test that severity matching is case-insensitive."""
        items = [
            {"name": "test1", "severity": "CRITICAL"},
            {"name": "test2", "severity": "Warning"},
            {"name": "test3", "severity": "INFO"},
        ]

        filtered = filter_by_severity(items, "WARNING")

        assert len(filtered) == 2
        assert filtered[0]["name"] == "test1"
        assert filtered[1]["name"] == "test2"

    def test_filter_by_severity_missing_severity_field(self):
        """Test handling of items without severity field."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2"},  # Missing severity - should default to info
            {"name": "test3", "severity": "warning"},
        ]

        filtered = filter_by_severity(items, Severity.WARNING)

        # test2 defaults to info, so should not be included
        assert len(filtered) == 2
        assert filtered[0]["name"] == "test1"
        assert filtered[1]["name"] == "test3"

    def test_filter_by_severity_invalid_severity_value(self):
        """Test handling of invalid severity values."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2", "severity": "invalid_severity"},
            {"name": "test3", "severity": "warning"},
        ]

        filtered = filter_by_severity(items, Severity.WARNING)

        # Invalid severity item should be skipped
        assert len(filtered) == 2
        assert filtered[0]["name"] == "test1"
        assert filtered[1]["name"] == "test3"

    def test_filter_by_severity_empty_list(self):
        """Test filtering empty list."""
        filtered = filter_by_severity([], Severity.WARNING)
        assert filtered == []

    def test_filter_by_severity_error_level(self):
        """Test filtering at ERROR level."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2", "severity": "error"},
            {"name": "test3", "severity": "warning"},
            {"name": "test4", "severity": "info"},
        ]

        filtered = filter_by_severity(items, Severity.ERROR)

        assert len(filtered) == 2
        assert filtered[0]["name"] == "test1"
        assert filtered[1]["name"] == "test2"

    def test_filter_by_severity_preserves_order(self):
        """Test that filtering preserves item order."""
        items = [
            {"name": "item1", "severity": "critical"},
            {"name": "item2", "severity": "info"},
            {"name": "item3", "severity": "warning"},
            {"name": "item4", "severity": "critical"},
        ]

        filtered = filter_by_severity(items, Severity.WARNING)

        assert len(filtered) == 3
        assert filtered[0]["name"] == "item1"
        assert filtered[1]["name"] == "item3"
        assert filtered[2]["name"] == "item4"


class TestFilterByAudience:
    """Tests for filter_by_audience function (REPORT-005)."""

    def test_filter_by_audience_executive(self):
        """Test filtering for executive audience."""
        content = {
            "executive_summary": "Summary for execs",
            "key_findings": "Important findings",
            "recommendations": "What to do",
            "raw_data": "Detailed data",
            "logs": "Debug logs",
        }

        filtered = filter_by_audience(content, AudienceType.EXECUTIVE)

        assert "executive_summary" in filtered
        assert "key_findings" in filtered
        assert "recommendations" in filtered
        assert "raw_data" not in filtered
        assert "logs" not in filtered

    def test_filter_by_audience_engineering(self):
        """Test filtering for engineering audience."""
        content = {
            "summary": "Summary",
            "results": "Results",
            "methodology": "Methods",
            "plots": "Charts",
            "raw_data": "Raw data",
            "logs": "Logs",
        }

        filtered = filter_by_audience(content, AudienceType.ENGINEERING)

        assert "summary" in filtered
        assert "results" in filtered
        assert "methodology" in filtered
        assert "plots" in filtered
        assert "raw_data" not in filtered
        assert "logs" not in filtered

    def test_filter_by_audience_debug(self):
        """Test filtering for debug audience includes all technical data."""
        content = {
            "summary": "Summary",
            "results": "Results",
            "methodology": "Methods",
            "plots": "Charts",
            "raw_data": "Raw data",
            "logs": "Debug logs",
        }

        filtered = filter_by_audience(content, AudienceType.DEBUG)

        # Debug audience should see everything
        assert len(filtered) == 6
        assert "raw_data" in filtered
        assert "logs" in filtered

    def test_filter_by_audience_regulatory(self):
        """Test filtering for regulatory audience."""
        content = {
            "summary": "Summary",
            "compliance": "Compliance info",
            "test_procedures": "Test procedures",
            "standards": "Standards ref",
            "raw_data": "Raw data",
            "methodology": "Methods",
        }

        filtered = filter_by_audience(content, AudienceType.REGULATORY)

        assert "summary" in filtered
        assert "compliance" in filtered
        assert "test_procedures" in filtered
        assert "standards" in filtered
        assert "raw_data" not in filtered
        assert "methodology" not in filtered

    def test_filter_by_audience_production(self):
        """Test filtering for production audience."""
        content = {
            "summary": "Summary",
            "pass_fail": "Pass/Fail status",
            "margin": "Margins",
            "yield": "Yield data",
            "raw_data": "Raw data",
            "logs": "Logs",
        }

        filtered = filter_by_audience(content, AudienceType.PRODUCTION)

        assert "summary" in filtered
        assert "pass_fail" in filtered
        assert "margin" in filtered
        assert "yield" in filtered
        assert "raw_data" not in filtered
        assert "logs" not in filtered

    def test_filter_by_audience_with_string(self):
        """Test filtering with audience as string."""
        content = {
            "executive_summary": "Summary for execs",
            "key_findings": "Findings",
            "raw_data": "Data",
        }

        filtered = filter_by_audience(content, "executive")

        assert "executive_summary" in filtered
        assert "key_findings" in filtered
        assert "raw_data" not in filtered

    def test_filter_by_audience_empty_content(self):
        """Test filtering empty content."""
        filtered = filter_by_audience({}, AudienceType.EXECUTIVE)
        assert filtered == {}

    def test_filter_by_audience_preserves_values(self):
        """Test that filtering preserves content values."""
        content = {
            "executive_summary": "Important summary",
            "key_findings": ["Finding 1", "Finding 2"],
            "recommendations": {"action": "Do this"},
        }

        filtered = filter_by_audience(content, AudienceType.EXECUTIVE)

        assert filtered["executive_summary"] == "Important summary"
        assert filtered["key_findings"] == ["Finding 1", "Finding 2"]
        assert filtered["recommendations"] == {"action": "Do this"}

    def test_filter_by_audience_unknown_sections_excluded(self):
        """Test that sections not in allowed list are excluded."""
        content = {
            "executive_summary": "Summary",
            "unknown_section": "Should be filtered",
            "another_unknown": "Also filtered",
        }

        filtered = filter_by_audience(content, AudienceType.EXECUTIVE)

        assert "executive_summary" in filtered
        assert "unknown_section" not in filtered
        assert "another_unknown" not in filtered


class TestCalculateRelevanceScore:
    """Tests for calculate_relevance_score function (REPORT-005)."""

    def test_calculate_relevance_score_base_score(self):
        """Test base relevance score without any factors."""
        item = {"name": "test"}
        score = calculate_relevance_score(item)
        assert score == 0.5

    def test_calculate_relevance_score_with_failure(self):
        """Test relevance score increases for failed items."""
        item = {"name": "test", "status": "fail"}
        score = calculate_relevance_score(item)
        assert score == 0.8  # 0.5 base + 0.3 fail

    def test_calculate_relevance_score_with_critical_severity(self):
        """Test relevance score increases for critical severity."""
        item = {"name": "test", "severity": "critical"}
        score = calculate_relevance_score(item)
        assert score == 0.8  # 0.5 base + 0.3 critical

    def test_calculate_relevance_score_with_warning_severity(self):
        """Test relevance score increases slightly for warning severity."""
        item = {"name": "test", "severity": "warning"}
        score = calculate_relevance_score(item)
        assert score == 0.6  # 0.5 base + 0.1 warning

    def test_calculate_relevance_score_with_outlier(self):
        """Test relevance score increases for outliers."""
        item = {"name": "test", "is_outlier": True}
        score = calculate_relevance_score(item)
        assert score == 0.7  # 0.5 base + 0.2 outlier

    def test_calculate_relevance_score_with_low_margin(self):
        """Test relevance score increases for low margin."""
        item = {"name": "test", "margin_pct": 15}
        score = calculate_relevance_score(item)
        assert score == 0.7  # 0.5 base + 0.2 low margin

    def test_calculate_relevance_score_with_high_margin(self):
        """Test relevance score doesn't increase for high margin."""
        item = {"name": "test", "margin_pct": 50}
        score = calculate_relevance_score(item)
        assert score == 0.5  # 0.5 base, no increase

    def test_calculate_relevance_score_margin_boundary(self):
        """Test relevance score at margin boundary (20%)."""
        item_below = {"name": "test", "margin_pct": 19}
        item_at = {"name": "test", "margin_pct": 20}
        item_above = {"name": "test", "margin_pct": 21}

        assert calculate_relevance_score(item_below) == 0.7  # Increases
        assert calculate_relevance_score(item_at) == 0.5  # No increase
        assert calculate_relevance_score(item_above) == 0.5  # No increase

    def test_calculate_relevance_score_multiple_factors(self):
        """Test relevance score with multiple contributing factors."""
        item = {
            "name": "test",
            "status": "fail",
            "severity": "critical",
            "is_outlier": True,
            "margin_pct": 10,
        }
        score = calculate_relevance_score(item)
        # 0.5 base + 0.3 fail + 0.3 critical + 0.2 outlier + 0.2 margin = 1.5
        # Should be capped at 1.0
        assert score == 1.0

    def test_calculate_relevance_score_capped_at_one(self):
        """Test that relevance score is capped at 1.0."""
        item = {
            "status": "fail",
            "severity": "critical",
        }
        score = calculate_relevance_score(item)
        assert score == 1.0  # 0.5 + 0.3 + 0.3 = 1.1, capped at 1.0

    def test_calculate_relevance_score_with_context(self):
        """Test relevance score calculation with context (currently unused)."""
        item = {"name": "test"}
        context = {"focus": "performance"}
        score = calculate_relevance_score(item, context)
        # Context currently not used, should return base score
        assert score == 0.5

    def test_calculate_relevance_score_case_insensitive_severity(self):
        """Test that severity matching is case-insensitive."""
        item1 = {"severity": "critical"}
        item2 = {"severity": "CRITICAL"}
        item3 = {"severity": "Critical"}

        assert calculate_relevance_score(item1) == 0.8
        assert calculate_relevance_score(item2) == 0.8
        assert calculate_relevance_score(item3) == 0.8

    def test_calculate_relevance_score_info_severity(self):
        """Test that info severity doesn't increase score."""
        item = {"severity": "info"}
        score = calculate_relevance_score(item)
        assert score == 0.5  # Base score only

    def test_calculate_relevance_score_pass_status(self):
        """Test that pass status doesn't increase score."""
        item = {"status": "pass"}
        score = calculate_relevance_score(item)
        assert score == 0.5  # Base score only

    def test_calculate_relevance_score_outlier_false(self):
        """Test that is_outlier=False doesn't increase score."""
        item = {"is_outlier": False}
        score = calculate_relevance_score(item)
        assert score == 0.5  # Base score only

    def test_calculate_relevance_score_zero_margin(self):
        """Test relevance score with zero margin."""
        item = {"margin_pct": 0}
        score = calculate_relevance_score(item)
        assert score == 0.7  # 0.5 base + 0.2 low margin

    def test_calculate_relevance_score_negative_margin(self):
        """Test relevance score with negative margin."""
        item = {"margin_pct": -10}
        score = calculate_relevance_score(item)
        assert score == 0.7  # 0.5 base + 0.2 low margin


class TestReportingFilteringIntegration:
    """Integration tests combining multiple filtering functions."""

    def test_filter_pipeline_severity_then_audience(self):
        """Test applying severity filter then audience filter."""
        items = [
            {"name": "test1", "severity": "critical", "category": "summary"},
            {"name": "test2", "severity": "info", "category": "summary"},
            {"name": "test3", "severity": "warning", "category": "raw_data"},
        ]

        # First filter by severity
        severity_filtered = filter_by_severity(items, Severity.WARNING)
        assert len(severity_filtered) == 2

        # Create content structure
        content = {
            "summary": severity_filtered,
            "raw_data": [{"name": "test4", "severity": "critical"}],
        }

        # Then filter by audience
        audience_filtered = filter_by_audience(content, AudienceType.EXECUTIVE)
        assert "summary" not in audience_filtered  # Executive doesn't see summary
        assert "raw_data" not in audience_filtered

    def test_relevance_scoring_for_filtered_items(self):
        """Test calculating relevance scores for filtered items."""
        items = [
            {"name": "test1", "severity": "critical", "status": "fail"},
            {"name": "test2", "severity": "info", "status": "pass"},
            {"name": "test3", "severity": "warning", "is_outlier": True},
        ]

        # Filter by severity
        filtered = filter_by_severity(items, Severity.WARNING)

        # Calculate relevance scores
        scores = [calculate_relevance_score(item) for item in filtered]

        assert len(scores) == 2
        assert scores[0] == 1.0  # critical + fail (0.5 + 0.3 + 0.3 = 1.1, capped at 1.0)
        assert scores[1] == 0.8  # warning + outlier (0.5 + 0.1 + 0.2 = 0.8)

    def test_content_filter_configuration(self):
        """Test using ContentFilter as configuration object."""
        config = ContentFilter(
            severity_threshold=Severity.WARNING,
            audience=AudienceType.ENGINEERING,
            show_only="violations",
            hide_empty_sections=True,
            relevance_threshold=0.7,
        )

        items = [
            {"name": "test1", "severity": "critical", "status": "fail", "is_outlier": True},
            {"name": "test2", "severity": "info", "status": "pass"},
        ]

        # Apply severity filter from config
        severity_filtered = filter_by_severity(items, config.severity_threshold)
        assert len(severity_filtered) == 1

        # Filter by relevance threshold
        relevant_items = [
            item
            for item in severity_filtered
            if calculate_relevance_score(item) >= config.relevance_threshold
        ]
        assert len(relevant_items) == 1
        assert relevant_items[0]["name"] == "test1"


class TestReportingFilteringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_filter_by_severity_all_items_filtered_out(self):
        """Test when all items are filtered out by severity."""
        items = [
            {"name": "test1", "severity": "info"},
            {"name": "test2", "severity": "warning"},
        ]

        filtered = filter_by_severity(items, Severity.CRITICAL)
        assert filtered == []

    def test_filter_by_audience_no_matching_sections(self):
        """Test when no content sections match audience."""
        content = {
            "raw_data": "Data",
            "logs": "Logs",
            "debug_info": "Debug",
        }

        filtered = filter_by_audience(content, AudienceType.EXECUTIVE)
        assert filtered == {}

    def test_calculate_relevance_score_empty_item(self):
        """Test relevance score for empty item."""
        score = calculate_relevance_score({})
        assert score == 0.5  # Base score

    def test_filter_by_severity_with_explicit_none_values(self):
        """Test handling of explicit None values in severity field.

        Note: Current implementation doesn't handle None gracefully and will raise AttributeError.
        This test documents the current behavior. Items with None severity should omit the field
        entirely to use the default, or be handled explicitly.
        """
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2", "severity": None},
        ]

        # Current implementation raises AttributeError on None values
        with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'lower'"):
            filter_by_severity(items, Severity.WARNING)

    def test_filter_by_severity_missing_field_uses_default(self):
        """Test that missing severity field defaults to 'info'."""
        items = [
            {"name": "test1", "severity": "critical"},
            {"name": "test2"},  # Missing severity field - should default to 'info'
            {"name": "test3", "severity": "warning"},
        ]

        filtered = filter_by_severity(items, Severity.WARNING)

        # test2 defaults to 'info', so should not be included
        assert len(filtered) == 2
        assert filtered[0]["name"] == "test1"
        assert filtered[1]["name"] == "test3"

    def test_filter_by_audience_with_none_values(self):
        """Test handling of None values in content."""
        content = {
            "summary": None,
            "results": "Some results",
        }

        filtered = filter_by_audience(content, AudienceType.ENGINEERING)

        # Should preserve None values for included sections
        assert "summary" in filtered
        assert filtered["summary"] is None
        assert filtered["results"] == "Some results"

    def test_calculate_relevance_score_with_none_margin(self):
        """Test relevance score when margin is explicitly None."""
        item = {"margin_pct": None}
        score = calculate_relevance_score(item)
        assert score == 0.5  # Should not crash, just ignore margin
