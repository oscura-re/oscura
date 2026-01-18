"""Tests for the onboarding and help system.

Tests requirements:
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestTutorials:
    """Test the tutorial system."""

    def test_list_tutorials(self) -> None:
        """Test listing available tutorials."""
        from oscura.onboarding import list_tutorials

        tutorials = list_tutorials()
        assert isinstance(tutorials, list)
        assert len(tutorials) >= 1

        # Check structure
        for t in tutorials:
            assert "id" in t
            assert "title" in t
            assert "difficulty" in t
            assert "steps" in t

    def test_get_tutorial(self) -> None:
        """Test getting a specific tutorial."""
        from oscura.onboarding import Tutorial, get_tutorial

        tutorial = get_tutorial("getting_started")
        assert tutorial is not None
        assert isinstance(tutorial, Tutorial)
        assert tutorial.id == "getting_started"
        assert len(tutorial.steps) >= 3

    def test_get_tutorial_not_found(self) -> None:
        """Test getting a non-existent tutorial returns None."""
        from oscura.onboarding import get_tutorial

        result = get_tutorial("nonexistent_tutorial")
        assert result is None

    def test_tutorial_step_structure(self) -> None:
        """Test tutorial step has all required fields."""
        from oscura.onboarding import get_tutorial

        tutorial = get_tutorial("getting_started")
        assert tutorial is not None

        for step in tutorial.steps:
            assert step.title
            assert step.description
            assert step.code


class TestHelp:
    """Test the help system."""

    def test_get_help_known_topic(self) -> None:
        """Test getting help for a known topic."""
        from oscura.onboarding import get_help

        help_text = get_help("rise_time")
        assert help_text is not None
        assert "rise time" in help_text.lower() or "Rise time" in help_text
        assert len(help_text) > 100  # Should have substantial content

    def test_get_help_unknown_topic(self) -> None:
        """Test getting help for an unknown topic."""
        from oscura.onboarding import get_help

        get_help("unknown_function_xyz")
        # Should return None or fallback to docstring
        # (depends on whether function exists in oscura)

    def test_get_help_topics(self) -> None:
        """Test help is available for common topics."""
        from oscura.onboarding import get_help

        topics = ["rise_time", "frequency", "thd", "snr", "fft", "load", "measure"]

        for topic in topics:
            help_text = get_help(topic)
            assert help_text is not None, f"No help for {topic}"
            assert len(help_text) > 50, f"Help too short for {topic}"


class TestCommandSuggestions:
    """Test command suggestions."""

    def test_suggestions_no_trace(self) -> None:
        """Test suggestions when no trace is loaded."""
        from oscura.onboarding import suggest_commands

        suggestions = suggest_commands(trace=None)
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1

        # Should suggest loading a trace
        commands = [s["command"] for s in suggestions]
        assert any("load" in c for c in commands)

    def test_suggestions_with_trace(self) -> None:
        """Test suggestions when trace is loaded."""
        from oscura.core.types import TraceMetadata, WaveformTrace
        from oscura.onboarding import suggest_commands

        # Create a mock trace
        data = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        trace = WaveformTrace(data=data.astype(np.float64), metadata=TraceMetadata(sample_rate=1e6))

        suggestions = suggest_commands(trace=trace)
        assert isinstance(suggestions, list)
        assert len(suggestions) >= 1

        # Should suggest measurements
        commands = [s["command"] for s in suggestions]
        assert any("measure" in c for c in commands)

    def test_suggestions_with_context(self) -> None:
        """Test context-aware suggestions."""
        from oscura.core.types import TraceMetadata, WaveformTrace
        from oscura.onboarding import suggest_commands

        data = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)
        trace = WaveformTrace(data=data, metadata=TraceMetadata(sample_rate=1e6))

        suggestions = suggest_commands(trace=trace, context="I2C bus")
        commands = [s["command"] for s in suggestions]
        assert any("i2c" in c.lower() for c in commands)


class TestExplainResult:
    """Test result explanation."""

    def test_explain_rise_time(self) -> None:
        """Test explaining rise time results."""
        from oscura.onboarding import explain_result

        explanation = explain_result(2.5e-9, "rise_time")
        assert "ns" in explanation or "nanosecond" in explanation.lower()

    def test_explain_frequency(self) -> None:
        """Test explaining frequency results."""
        from oscura.onboarding import explain_result

        explanation = explain_result(10e6, "frequency")
        assert "MHz" in explanation or "megahertz" in explanation.lower()

    def test_explain_thd(self) -> None:
        """Test explaining THD results."""
        from oscura.onboarding import explain_result

        # Good THD
        explanation = explain_result(-50, "thd")
        assert "dB" in explanation
        assert "good" in explanation.lower() or "excellent" in explanation.lower()

        # Bad THD
        explanation = explain_result(-15, "thd")
        assert "poor" in explanation.lower() or "significant" in explanation.lower()


class TestGetExample:
    """Test getting code examples."""

    def test_get_example_load(self) -> None:
        """Test getting example for load function."""
        from oscura.onboarding import get_example

        example = get_example("load")
        assert example is not None
        assert "load" in example
        assert "import" in example

    def test_get_example_unknown(self) -> None:
        """Test getting example for unknown function."""
        from oscura.onboarding import get_example

        example = get_example("unknown_function_xyz")
        assert example is None


class TestWizard:
    """Test the analysis wizard."""

    def test_wizard_creation(self) -> None:
        """Test wizard can be created."""
        from oscura.core.types import TraceMetadata, WaveformTrace
        from oscura.onboarding import AnalysisWizard

        data = np.sin(2 * np.pi * np.linspace(0, 1, 1000))
        trace = WaveformTrace(data=data.astype(np.float64), metadata=TraceMetadata(sample_rate=1e6))

        wizard = AnalysisWizard(trace)
        assert wizard.trace is trace
        assert len(wizard.steps) >= 3

    def test_wizard_result_structure(self) -> None:
        """Test WizardResult has expected structure."""
        from oscura.onboarding.wizard import WizardResult

        result = WizardResult()
        assert result.steps_completed == 0
        assert isinstance(result.measurements, dict)
        assert isinstance(result.recommendations, list)
        assert result.summary == ""
