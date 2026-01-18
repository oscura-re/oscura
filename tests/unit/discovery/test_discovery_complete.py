"""Comprehensive tests for all Discovery requirements.

This test module verifies that all Discovery requirements (DISC-003, DISC-004,
DISC-005, DISC-006, DISC-008, DISC-011, DISC-012) are properly implemented
and functional.
"""

import numpy as np
import pytest

from oscura.core.types import TraceMetadata, WaveformTrace

pytestmark = pytest.mark.unit


@pytest.fixture
def sample_trace():
    """Create a sample waveform trace for testing."""
    # Generate a 1 MHz square wave
    sample_rate = 100e6  # 100 MS/s
    duration = 1e-3  # 1 ms
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)

    # Square wave between 0V and 3.3V
    signal = 3.3 * (np.sin(2 * np.pi * 1e6 * t) > 0).astype(np.float64)

    metadata = TraceMetadata(
        sample_rate=sample_rate,
        channel_name="CH1",
    )

    return WaveformTrace(data=signal, metadata=metadata)


@pytest.fixture
def noisy_trace():
    """Create a noisy waveform for testing quality assessment."""
    sample_rate = 50e6
    duration = 1e-3
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)

    # Sine wave with noise
    signal = 2.0 * np.sin(2 * np.pi * 100e3 * t)
    noise = 0.3 * np.random.randn(samples)
    noisy_signal = signal + noise

    metadata = TraceMetadata(sample_rate=sample_rate)

    return WaveformTrace(data=noisy_signal, metadata=metadata)


def test_disc003_natural_language_summaries(sample_trace):
    """Test DISC-003: Natural Language Summaries.

    Verifies that generate_summary produces plain-English descriptions
    with appropriate grade level and structure.
    """
    from oscura.reporting import generate_summary

    # Generate summary
    summary = generate_summary(sample_trace)

    # Verify structure
    assert hasattr(summary, "text"), "Summary should have text attribute"
    assert hasattr(summary, "overview"), "Summary should have overview"
    assert hasattr(summary, "findings"), "Summary should have findings list"
    assert hasattr(summary, "recommendations"), "Summary should have recommendations"

    # Verify content requirements
    assert len(summary.text) > 0, "Summary text should not be empty"
    assert summary.word_count <= 200, f"Summary should be ≤200 words, got {summary.word_count}"
    assert summary.grade_level <= 12, f"Grade level should be ≤12, got {summary.grade_level:.1f}"

    # Verify findings (minimum 3 key findings)
    assert len(summary.findings) >= 3, f"Should have ≥3 findings, got {len(summary.findings)}"

    # Verify finding structure
    for finding in summary.findings:
        assert hasattr(finding, "title"), "Finding should have title"
        assert hasattr(finding, "description"), "Finding should have description"
        assert hasattr(finding, "confidence"), "Finding should have confidence"
        assert 0.0 <= finding.confidence <= 1.0, "Confidence should be 0.0-1.0"

    print(
        f"✓ DISC-003: Summary generated with {summary.word_count} words, grade {summary.grade_level:.1f}"
    )
    print(f"  Overview: {summary.overview}")
    print(f"  Findings: {len(summary.findings)}")


def test_disc004_intelligent_trace_comparison(sample_trace, noisy_trace):
    """Test DISC-004: Intelligent Trace Comparison.

    Verifies automatic trace alignment and difference detection with
    plain-language explanations.
    """
    from oscura.discovery import compare_traces

    # Compare traces
    diff = compare_traces(sample_trace, noisy_trace)

    # Verify structure
    assert hasattr(diff, "summary"), "Should have summary"
    assert hasattr(diff, "alignment_method"), "Should have alignment method"
    assert hasattr(diff, "similarity_score"), "Should have similarity score"
    assert hasattr(diff, "differences"), "Should have differences list"

    # Verify alignment method
    assert diff.alignment_method in ["time-based", "trigger-based", "pattern-based"], (
        f"Invalid alignment method: {diff.alignment_method}"
    )

    # Verify similarity score
    assert 0.0 <= diff.similarity_score <= 1.0, (
        f"Similarity score should be 0.0-1.0, got {diff.similarity_score}"
    )

    # Verify differences
    for d in diff.differences:
        assert hasattr(d, "category"), "Difference should have category"
        assert d.category in ["timing", "amplitude", "pattern", "transitions"], (
            f"Invalid category: {d.category}"
        )
        assert hasattr(d, "description"), "Difference should have description"
        assert hasattr(d, "severity"), "Difference should have severity"
        assert d.severity in ["INFO", "WARNING", "CRITICAL"], f"Invalid severity: {d.severity}"
        assert hasattr(d, "impact_score"), "Difference should have impact score"

    # Verify statistics
    assert diff.stats is not None, "Should have statistics"
    assert "correlation" in diff.stats, "Stats should include correlation"

    print(f"✓ DISC-004: Comparison completed ({diff.alignment_method})")
    print(f"  Similarity: {diff.similarity_score:.1%}")
    print(f"  Differences: {len(diff.differences)}")


def test_disc005_executive_report_generation(sample_trace):
    """Test DISC-005: Automatic Executive Report.

    Verifies one-click report generation with multiple formats.
    """
    import os
    import tempfile

    from oscura.reporting.auto_report import generate_report

    # Generate report
    report = generate_report(sample_trace)

    # Verify structure
    assert hasattr(report, "sections"), "Should have sections"
    assert hasattr(report, "plots"), "Should have plots list"
    assert hasattr(report, "page_count"), "Should have page count"
    assert hasattr(report, "metadata"), "Should have metadata"

    # Verify mandatory sections
    mandatory_sections = [
        "executive_summary",
        "key_findings",
        "methodology",
        "detailed_results",
    ]
    for section in mandatory_sections:
        assert section in report.sections, f"Missing mandatory section: {section}"

    # Test export to different formats
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test PDF export
        pdf_path = os.path.join(tmpdir, "test_report.pdf")  # noqa: PTH118
        report.save_pdf(pdf_path)
        assert os.path.exists(pdf_path), "PDF should be created"

        # Test HTML export
        html_path = os.path.join(tmpdir, "test_report.html")  # noqa: PTH118
        report.save_html(html_path)
        assert os.path.exists(html_path), "HTML should be created"

        # Test Markdown export
        md_path = os.path.join(tmpdir, "test_report.md")  # noqa: PTH118
        report.save_markdown(md_path)
        assert os.path.exists(md_path), "Markdown should be created"

    print(f"✓ DISC-005: Report generated with {report.page_count} pages")
    print(f"  Sections: {', '.join(report.sections)}")
    print(f"  Plots: {len(report.plots)}")


def test_disc006_interactive_wizard(sample_trace):
    """Test DISC-006: Interactive Analysis Wizard.

    Verifies step-by-step guided analysis workflow.
    """
    from oscura.onboarding.wizard import AnalysisWizard

    # Create wizard (non-interactive mode for testing)
    wizard = AnalysisWizard(sample_trace)

    # Verify structure
    assert hasattr(wizard, "trace"), "Wizard should have trace"
    assert hasattr(wizard, "steps"), "Wizard should have steps"
    assert hasattr(wizard, "result"), "Wizard should have result"

    # Verify steps
    assert len(wizard.steps) > 0, "Should have at least one step"
    assert len(wizard.steps) <= 5, "Should have ≤5 steps per spec"

    # Verify step structure
    for step in wizard.steps:
        assert hasattr(step, "title"), "Step should have title"
        assert hasattr(step, "question"), "Step should have question"
        assert hasattr(step, "options"), "Step should have options"
        assert hasattr(step, "help_text"), "Step should have help text"

    # Test non-interactive run
    result = wizard.run(interactive=False)

    # Verify result
    assert hasattr(result, "steps_completed"), "Result should track steps"
    assert hasattr(result, "measurements"), "Result should have measurements"
    assert hasattr(result, "recommendations"), "Result should have recommendations"
    assert hasattr(result, "summary"), "Result should have summary"

    print(f"✓ DISC-006: Wizard completed {result.steps_completed} steps")
    print(f"  Measurements: {len(result.measurements)}")
    print(f"  Recommendations: {len(result.recommendations)}")


def test_disc008_recommendation_engine(sample_trace):
    """Test DISC-008: Recommendation Engine.

    Verifies contextual next-step recommendations.
    """
    from oscura.guidance import suggest_next_steps

    # Get recommendations with empty state
    recs = suggest_next_steps(sample_trace)

    # Verify structure
    assert isinstance(recs, list), "Should return list of recommendations"
    assert 2 <= len(recs) <= 5, f"Should return 2-5 recommendations, got {len(recs)}"

    # Verify recommendation structure
    for rec in recs:
        assert hasattr(rec, "id"), "Recommendation should have ID"
        assert hasattr(rec, "title"), "Recommendation should have title"
        assert hasattr(rec, "explanation"), "Recommendation should have explanation"
        assert hasattr(rec, "priority"), "Recommendation should have priority"
        assert 0.0 <= rec.priority <= 1.0, "Priority should be 0.0-1.0"

        # Verify weighted scoring components
        assert hasattr(rec, "urgency"), "Should have urgency score"
        assert hasattr(rec, "ease"), "Should have ease score"
        assert hasattr(rec, "impact"), "Should have impact score"

        # Verify explanation length (≤50 words)
        word_count = len(rec.explanation.split())
        assert word_count <= 50, f"Explanation should be ≤50 words, got {word_count}"

    # Verify ranking (should be sorted by priority)
    priorities = [rec.priority for rec in recs]
    assert priorities == sorted(priorities, reverse=True), "Should be sorted by priority"

    # Test with analysis state
    state = {"characterization": type("obj", (), {"signal_type": "digital", "confidence": 0.95})()}
    recs_with_state = suggest_next_steps(sample_trace, current_state=state)

    assert isinstance(recs_with_state, list), "Should work with state"

    print(f"✓ DISC-008: Generated {len(recs)} recommendations")
    print(f"  Top priority: {recs[0].title} ({recs[0].priority:.2f})")


def test_disc011_progressive_disclosure(sample_trace):
    """Test DISC-011: Progressive Disclosure.

    Verifies hierarchical information display with multiple detail levels.
    """
    from oscura.discovery import characterize_signal
    from oscura.ui import ProgressiveDisplay

    # Create display manager
    display = ProgressiveDisplay(
        default_level="summary", max_summary_items=5, enable_collapsible_sections=True
    )

    # Create mock result with required attributes
    result = characterize_signal(sample_trace)

    # Render with progressive disclosure
    output = display.render(result)

    # Verify structure
    assert hasattr(output, "level1_content"), "Should have L1 content"
    assert hasattr(output, "level2_sections"), "Should have L2 sections"
    assert hasattr(output, "level3_data"), "Should have L3 data"

    # Test Level 1: Summary
    summary = output.summary()
    assert len(summary) > 0, "Summary should not be empty"

    # Verify summary length (≤100 words typical)
    summary_words = len(summary.split())
    assert summary_words <= 150, f"Summary should be concise, got {summary_words} words"

    # Test Level 2: Intermediate
    details = output.details(level="intermediate")
    assert len(details) > len(summary), "Details should be longer than summary"

    # Test Level 3: Expert
    if output.has_level3():
        expert = output.expert()
        assert len(expert) > len(details), "Expert view should be most detailed"

    # Test section expansion/collapse
    if output.level2_sections:
        first_section = output.level2_sections[0]
        output.expand_section(first_section.title)
        assert not first_section.is_collapsed, "Section should be expanded"

        output.collapse_section(first_section.title)
        assert first_section.is_collapsed, "Section should be collapsed"

    print(f"✓ DISC-011: Progressive disclosure with {len(output.level2_sections)} sections")
    print(f"  Summary: {summary_words} words")


def test_disc012_plain_english_help():
    """Test DISC-012: Plain English Help.

    Verifies accessible documentation and help text.
    """
    from oscura.onboarding import explain_result, get_help, suggest_commands

    # Test help database
    help_topics = [
        "rise_time",
        "fall_time",
        "frequency",
        "thd",
        "snr",
        "fft",
        "load",
        "measure",
    ]

    for topic in help_topics:
        help_text = get_help(topic)
        assert help_text is not None, f"Should have help for {topic}"
        assert len(help_text) > 100, f"Help for {topic} should be detailed"

        # Verify contains plain English explanations (check for 'help' keyword at minimum)
        assert (
            "plain" in help_text.lower()
            or "summary" in help_text.lower()
            or "help" in help_text.lower()
        ), f"Help for {topic} should have explanatory sections"

    # Test command suggestions
    suggestions = suggest_commands()
    assert len(suggestions) > 0, "Should provide suggestions"

    for suggestion in suggestions:
        assert "command" in suggestion, "Suggestion should have command"
        assert "description" in suggestion, "Suggestion should have description"
        assert "reason" in suggestion, "Suggestion should have reason"

    # Test result explanations
    explanations = [
        (2.5e-9, "rise_time"),
        (1e6, "frequency"),
        (-45.0, "thd"),
        (50.0, "snr"),
    ]

    for value, measurement in explanations:
        explanation = explain_result(value, measurement)
        assert len(explanation) > 0, f"Should explain {measurement}"
        assert isinstance(explanation, str), "Explanation should be string"

    print(f"✓ DISC-012: Plain English help available for {len(help_topics)} topics")
    print(f"  Suggestions: {len(suggestions)}")


def test_all_requirements_integrated(sample_trace):
    """Integration test for all Discovery requirements together.

    This test verifies that all requirements work together in a complete
    analysis workflow.
    """
    from oscura.discovery import characterize_signal
    from oscura.guidance import suggest_next_steps
    from oscura.onboarding import get_help
    from oscura.reporting import generate_summary
    from oscura.reporting.auto_report import generate_report
    from oscura.ui import ProgressiveDisplay

    # Step 1: Characterize signal (used by other requirements)
    char_result = characterize_signal(sample_trace)
    assert char_result is not None

    # Step 2: Get recommendations based on characterization
    state = {"characterization": char_result}
    recommendations = suggest_next_steps(sample_trace, current_state=state)
    assert len(recommendations) > 0

    # Step 3: Generate natural language summary
    summary = generate_summary(sample_trace, context=state)
    assert summary.word_count > 0

    # Step 4: Progressive disclosure of results
    display = ProgressiveDisplay()
    output = display.render(char_result)
    assert len(output.summary()) > 0

    # Step 5: Generate executive report
    report = generate_report(sample_trace, context=state)
    assert report.page_count > 0

    # Step 6: Get help for next steps
    help_text = get_help("measure")
    assert help_text is not None

    print("✓ All Discovery requirements integrated successfully")
    print(f"  Signal: {char_result.signal_type} ({char_result.confidence:.0%} confidence)")
    print(f"  Recommendations: {len(recommendations)}")
    print(f"  Summary: {summary.word_count} words")
    print(f"  Report: {report.page_count} pages")


if __name__ == "__main__":
    # Run tests directly
    import sys

    print("\n" + "=" * 70)
    print("Testing Discovery Requirements Implementation")
    print("=" * 70 + "\n")

    # Create fixtures
    sample_rate = 100e6
    duration = 1e-3
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples)
    signal = 3.3 * (np.sin(2 * np.pi * 1e6 * t) > 0).astype(np.float64)
    metadata = TraceMetadata(sample_rate=sample_rate, channel_name="CH1")
    trace = WaveformTrace(data=signal, metadata=metadata)

    # Noisy trace
    noisy_signal = 2.0 * np.sin(2 * np.pi * 100e3 * t) + 0.3 * np.random.randn(samples)
    noisy = WaveformTrace(data=noisy_signal, metadata=TraceMetadata(sample_rate=sample_rate))

    try:
        test_disc003_natural_language_summaries(trace)
        print()
        test_disc004_intelligent_trace_comparison(trace, noisy)
        print()
        test_disc005_executive_report_generation(trace)
        print()
        test_disc006_interactive_wizard(trace)
        print()
        test_disc008_recommendation_engine(trace)
        print()
        test_disc011_progressive_disclosure(trace)
        print()
        test_disc012_plain_english_help()
        print()
        test_all_requirements_integrated(trace)

        print("\n" + "=" * 70)
        print("All Discovery Requirements Tests Passed!")
        print("=" * 70)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
