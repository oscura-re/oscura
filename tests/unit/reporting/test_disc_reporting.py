import pytest

"""Unit tests for DISC reporting and guidance modules.

Tests for:
"""

import numpy as np

from oscura.core.types import TraceMetadata, WaveformTrace
from oscura.discovery.comparison import TraceDiff, compare_traces
from oscura.guidance.recommender import Recommendation, suggest_next_steps
from oscura.jupyter.ui.progressive_display import ProgressiveDisplay
from oscura.reporting.auto_report import generate_report as generate_auto_report
from oscura.reporting.summary_generator import Finding, Summary, generate_summary

pytestmark = pytest.mark.unit


def create_test_trace(
    length: int = 1000,
    sample_rate: float = 1e6,
    signal_type: str = "digital",
) -> WaveformTrace:
    """Create a test waveform trace.

    Args:
        length: Number of samples.
        sample_rate: Sample rate in Hz.
        signal_type: "digital", "analog", or "dc"

    Returns:
        WaveformTrace for testing.
    """
    if signal_type == "digital":
        # Digital signal with two levels
        data = np.where(np.random.rand(length) > 0.5, 3.3, 0.0)
    elif signal_type == "analog":
        # Sinusoidal signal
        t = np.arange(length) / sample_rate
        data = 1.65 + 1.0 * np.sin(2 * np.pi * 1000 * t)
    else:  # dc
        # Constant DC level
        data = np.full(length, 1.5)

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel="test_channel",
    )

    return WaveformTrace(data=data, metadata=metadata)


class TestSummaryGenerator:
    """Tests for DISC-003: Natural Language Summaries."""

    def test_generate_summary_basic(self):
        """Test basic summary generation."""
        trace = create_test_trace()
        summary = generate_summary(trace)

        assert isinstance(summary, Summary)
        assert len(summary.text) > 0
        assert summary.word_count > 0
        assert summary.grade_level >= 0
        assert len(summary.findings) >= 3  # Minimum 3 findings
        assert len(summary.recommendations) >= 1  # At least 1 recommendation

    def test_summary_word_limit(self):
        """Test summary respects word limit."""
        trace = create_test_trace()
        summary = generate_summary(trace, max_words=50)

        assert summary.word_count <= 50

    def test_summary_findings(self):
        """Test summary includes proper findings."""
        trace = create_test_trace()
        summary = generate_summary(trace)

        assert all(isinstance(f, Finding) for f in summary.findings)
        assert all(hasattr(f, "title") for f in summary.findings)
        assert all(hasattr(f, "description") for f in summary.findings)
        assert all(0.0 <= f.confidence <= 1.0 for f in summary.findings)

    def test_summary_readability(self):
        """Test summary readability estimation."""
        trace = create_test_trace()
        summary = generate_summary(trace)

        # Grade level should be reasonable (not negative, not impossibly high)
        assert 0 <= summary.grade_level <= 20


class TestTraceComparison:
    """Tests for DISC-004: Intelligent Trace Comparison."""

    def test_compare_identical_traces(self):
        """Test comparison of identical traces."""
        trace = create_test_trace()
        diff = compare_traces(trace, trace)

        assert isinstance(diff, TraceDiff)
        assert diff.similarity_score > 0.95  # Should be very similar
        assert len(diff.differences) == 0  # No differences expected

    def test_compare_different_traces(self):
        """Test comparison of different traces."""
        trace1 = create_test_trace(signal_type="digital")
        trace2 = create_test_trace(signal_type="analog")

        diff = compare_traces(trace1, trace2)

        assert diff.similarity_score < 0.95  # Should be different
        assert len(diff.differences) > 0  # Some differences expected

    def test_alignment_methods(self):
        """Test different alignment methods."""
        trace1 = create_test_trace()
        trace2 = create_test_trace()

        for method in ["time", "trigger", "pattern", "auto"]:
            diff = compare_traces(trace1, trace2, alignment=method)
            assert isinstance(diff, TraceDiff)
            assert method in diff.alignment_method or diff.alignment_method.endswith("-based")

    def test_difference_severity(self):
        """Test difference severity classification."""
        trace1 = create_test_trace()
        # Create trace with amplitude difference
        trace2_data = trace1.data * 1.3  # 30% amplitude increase
        trace2 = WaveformTrace(data=trace2_data, metadata=trace1.metadata)

        diff = compare_traces(trace1, trace2)

        if len(diff.differences) > 0:
            assert all(d.severity in ["CRITICAL", "WARNING", "INFO"] for d in diff.differences)

    def test_difference_categories(self):
        """Test difference categorization."""
        trace1 = create_test_trace()
        trace2 = create_test_trace(signal_type="analog")

        diff = compare_traces(trace1, trace2)

        if len(diff.differences) > 0:
            categories = {d.category for d in diff.differences}
            assert categories.issubset({"timing", "amplitude", "pattern", "transitions"})


class TestAutoReport:
    """Tests for DISC-005: Automatic Executive Report."""

    def test_generate_report_basic(self):
        """Test basic report generation."""
        trace = create_test_trace()
        report = generate_auto_report(trace)

        assert len(report.sections) > 0
        assert report.page_count > 0
        assert isinstance(report.metadata.title, str)

    def test_report_sections(self):
        """Test report includes required sections."""
        trace = create_test_trace()
        report = generate_auto_report(trace)

        # Should include at least some of the default sections
        expected_sections = [
            "executive_summary",
            "key_findings",
            "methodology",
            "detailed_results",
        ]
        assert any(section in report.sections for section in expected_sections)

    def test_report_save_html(self, tmp_path):
        """Test HTML report export."""
        trace = create_test_trace()
        report = generate_auto_report(trace)

        output_path = tmp_path / "test_report.html"
        report.save_html(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_report_save_markdown(self, tmp_path):
        """Test Markdown report export."""
        trace = create_test_trace()
        report = generate_auto_report(trace)

        output_path = tmp_path / "test_report.md"
        report.save_markdown(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_report_custom_sections(self):
        """Test custom section options."""
        trace = create_test_trace()
        report = generate_auto_report(trace, options={"select_sections": ["summary", "findings"]})

        assert "summary" in report.sections or "executive_summary" in report.sections


class TestRecommender:
    """Tests for DISC-008: Recommendation Engine."""

    def test_suggest_next_steps_basic(self):
        """Test basic recommendation generation."""
        trace = create_test_trace()
        recommendations = suggest_next_steps(trace)

        assert isinstance(recommendations, list)
        assert 2 <= len(recommendations) <= 5  # Should return 2-5 recommendations
        assert all(isinstance(r, Recommendation) for r in recommendations)

    def test_recommendation_priority_sorting(self):
        """Test recommendations are sorted by priority."""
        trace = create_test_trace()
        recommendations = suggest_next_steps(trace)

        priorities = [r.priority for r in recommendations]
        assert priorities == sorted(priorities, reverse=True)  # Descending order

    def test_recommendation_content(self):
        """Test recommendation has required content."""
        trace = create_test_trace()
        recommendations = suggest_next_steps(trace)

        for rec in recommendations:
            assert len(rec.title) > 0
            assert len(rec.explanation) > 0
            assert 0.0 <= rec.priority <= 1.0
            assert 0.0 <= rec.urgency <= 1.0
            assert 0.0 <= rec.ease <= 1.0
            assert 0.0 <= rec.impact <= 1.0

    def test_recommendation_deduplication(self):
        """Test recommendations change based on state."""
        trace = create_test_trace()

        # First call - no state
        recs1 = suggest_next_steps(trace)
        assert len(recs1) > 0

        # Add characterization to state - should get different recommendations
        state = {"characterization": type("obj", (), {"signal_type": "UART", "confidence": 0.95})}
        recs2 = suggest_next_steps(trace, current_state=state)

        # With characterization, should recommend protocol decode
        rec_ids = {r.id for r in recs2}
        # Should not recommend characterization again (it's in state)
        assert "characterization" not in rec_ids

    def test_max_suggestions_limit(self):
        """Test max_suggestions parameter."""
        trace = create_test_trace()

        for max_count in [2, 3, 5]:
            recommendations = suggest_next_steps(trace, max_suggestions=max_count)
            assert len(recommendations) <= max_count


class TestProgressiveDisplay:
    """Tests for DISC-011: Progressive Disclosure."""

    def test_progressive_display_basic(self):
        """Test basic progressive display."""
        display = ProgressiveDisplay()

        # Create mock result
        class MockResult:
            signal_type = "digital"
            confidence = 0.95
            quality = "good"
            status = "ready"

        result = MockResult()
        output = display.render(result)

        assert output.summary()  # Should have summary
        assert output.details()  # Should have details

    def test_detail_levels(self):
        """Test three detail levels."""
        display = ProgressiveDisplay()

        class MockResult:
            signal_type = "test"
            confidence = 0.9
            parameters = {"rate": 115200}

        result = MockResult()
        output = display.render(result)

        level1 = output.summary()
        level2 = output.details()

        assert len(level1) < len(level2)  # L2 should have more content than L1

    def test_section_expansion(self):
        """Test section expand/collapse."""
        display = ProgressiveDisplay(enable_collapsible_sections=True)

        class MockResult:
            signal_type = "test"
            parameters = {"rate": 115200}
            findings = [{"title": "Test", "description": "Test finding"}]

        result = MockResult()
        output = display.render(result)

        # Find a section to expand
        if output.level2_sections:
            section_title = output.level2_sections[0].title
            output.expand_section(section_title)
            assert not output.level2_sections[0].is_collapsed

            output.collapse_section(section_title)
            assert output.level2_sections[0].is_collapsed

    def test_export_detail_levels(self, tmp_path):
        """Test export at different detail levels."""
        display = ProgressiveDisplay()

        class MockResult:
            signal_type = "test"
            confidence = 0.9

        result = MockResult()
        output = display.render(result)

        for level in ["summary", "intermediate"]:
            output_path = tmp_path / f"test_{level}.txt"
            output.export(str(output_path), detail_level=level)

            assert output_path.exists()
            assert output_path.stat().st_size > 0

    def test_max_summary_items(self):
        """Test max_summary_items configuration."""
        display = ProgressiveDisplay(max_summary_items=3)

        class MockResult:
            signal_type = "test"
            confidence = 0.9
            quality = "good"
            status = "ready"
            rate = 115200

        result = MockResult()
        output = display.render(result)

        # Summary should be limited
        summary_lines = output.summary().split("\n")
        # Filter out empty lines
        summary_lines = [line for line in summary_lines if line.strip()]
        assert len(summary_lines) <= 3
