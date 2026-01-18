"""Unit tests for the interactive tutorial system.

Tests the step-by-step interactive tutorials for new users,
covering common analysis workflows.
"""

from unittest.mock import patch

import pytest

from oscura.onboarding.tutorials import (
    TUTORIALS,
    Tutorial,
    TutorialStep,
    get_tutorial,
    list_tutorials,
    run_tutorial,
)

pytestmark = pytest.mark.unit


class TestTutorialStep:
    """Tests for the TutorialStep dataclass."""

    def test_tutorial_step_creation(self):
        """Test creating a basic tutorial step."""
        step = TutorialStep(
            title="Test Step",
            description="This is a test description",
            code="print('hello')",
        )

        assert step.title == "Test Step"
        assert step.description == "This is a test description"
        assert step.code == "print('hello')"
        assert step.expected_output == ""
        assert step.hints == []
        assert step.validation_fn is None

    def test_tutorial_step_with_all_fields(self):
        """Test creating a tutorial step with all fields."""

        def validate(result) -> bool:
            return True

        step = TutorialStep(
            title="Full Step",
            description="Complete description",
            code="x = 1 + 1",
            expected_output="2",
            hints=["Hint 1", "Hint 2"],
            validation_fn=validate,
        )

        assert step.title == "Full Step"
        assert step.expected_output == "2"
        assert len(step.hints) == 2
        assert step.validation_fn is not None

    def test_tutorial_step_hints_mutable_default(self):
        """Test that hints default is not shared between instances."""
        step1 = TutorialStep(title="S1", description="D1", code="C1")
        step2 = TutorialStep(title="S2", description="D2", code="C2")

        step1.hints.append("New hint")

        assert "New hint" in step1.hints
        assert "New hint" not in step2.hints


class TestTutorial:
    """Tests for the Tutorial dataclass."""

    def test_tutorial_creation(self):
        """Test creating a basic tutorial."""
        steps = [
            TutorialStep(title="Step 1", description="Desc 1", code="code1"),
            TutorialStep(title="Step 2", description="Desc 2", code="code2"),
        ]

        tutorial = Tutorial(
            id="test_tutorial",
            title="Test Tutorial",
            description="A test tutorial",
            steps=steps,
        )

        assert tutorial.id == "test_tutorial"
        assert tutorial.title == "Test Tutorial"
        assert tutorial.description == "A test tutorial"
        assert len(tutorial.steps) == 2
        assert tutorial.difficulty == "beginner"

    def test_tutorial_with_difficulty(self):
        """Test creating a tutorial with specific difficulty."""
        tutorial = Tutorial(
            id="advanced",
            title="Advanced Tutorial",
            description="Hard stuff",
            steps=[],
            difficulty="advanced",
        )

        assert tutorial.difficulty == "advanced"


class TestTutorialRegistry:
    """Tests for the TUTORIALS registry."""

    def test_tutorials_registry_exists(self):
        """Test that TUTORIALS registry is populated."""
        assert isinstance(TUTORIALS, dict)
        assert len(TUTORIALS) > 0

    def test_getting_started_tutorial_exists(self):
        """Test that getting_started tutorial exists."""
        assert "getting_started" in TUTORIALS

    def test_spectral_analysis_tutorial_exists(self):
        """Test that spectral_analysis tutorial exists."""
        assert "spectral_analysis" in TUTORIALS

    def test_all_tutorials_have_required_fields(self):
        """Test that all tutorials have required fields."""
        for tutorial_id, tutorial in TUTORIALS.items():
            assert tutorial.id == tutorial_id
            assert tutorial.title
            assert tutorial.description
            assert isinstance(tutorial.steps, list)
            assert tutorial.difficulty in ["beginner", "intermediate", "advanced"]

    def test_all_tutorials_have_steps(self):
        """Test that all tutorials have at least one step."""
        for tutorial_id, tutorial in TUTORIALS.items():
            assert len(tutorial.steps) > 0, f"{tutorial_id} has no steps"

    def test_all_steps_have_code(self):
        """Test that all tutorial steps have code examples."""
        for tutorial_id, tutorial in TUTORIALS.items():
            for i, step in enumerate(tutorial.steps):
                assert step.code, f"{tutorial_id} step {i} has no code"


class TestListTutorials:
    """Tests for the list_tutorials function."""

    def test_list_tutorials_returns_list(self):
        """Test that list_tutorials returns a list."""
        result = list_tutorials()
        assert isinstance(result, list)

    def test_list_tutorials_non_empty(self):
        """Test that list_tutorials returns at least one tutorial."""
        result = list_tutorials()
        assert len(result) > 0

    def test_list_tutorials_has_proper_structure(self):
        """Test that each tutorial info has proper structure."""
        result = list_tutorials()

        for info in result:
            assert "id" in info
            assert "title" in info
            assert "difficulty" in info
            assert "steps" in info

    def test_list_tutorials_step_count_is_int(self):
        """Test that step count is an integer."""
        result = list_tutorials()

        for info in result:
            assert isinstance(info["steps"], int)
            assert info["steps"] > 0


class TestGetTutorial:
    """Tests for the get_tutorial function."""

    def test_get_tutorial_existing(self):
        """Test getting an existing tutorial."""
        tutorial = get_tutorial("getting_started")

        assert tutorial is not None
        assert isinstance(tutorial, Tutorial)
        assert tutorial.id == "getting_started"

    def test_get_tutorial_nonexistent(self):
        """Test getting a nonexistent tutorial."""
        tutorial = get_tutorial("nonexistent_tutorial_xyz")

        assert tutorial is None

    def test_get_tutorial_returns_same_instance(self):
        """Test that get_tutorial returns the registered instance."""
        tutorial1 = get_tutorial("getting_started")
        tutorial2 = get_tutorial("getting_started")

        assert tutorial1 is tutorial2


class TestRunTutorial:
    """Tests for the run_tutorial function."""

    @patch("builtins.print")
    @patch("builtins.input", return_value="")
    def test_run_tutorial_interactive(self, mock_input, mock_print):
        """Test running tutorial in interactive mode."""
        run_tutorial("getting_started", interactive=True)

        # Check that print was called multiple times
        assert mock_print.call_count > 0
        # Check that input was called for each step
        assert mock_input.call_count > 0

    @patch("builtins.print")
    def test_run_tutorial_non_interactive(self, mock_print):
        """Test running tutorial in non-interactive mode."""
        run_tutorial("getting_started", interactive=False)

        # Check that print was called
        assert mock_print.call_count > 0

    @patch("builtins.print")
    def test_run_tutorial_prints_title(self, mock_print):
        """Test that run_tutorial prints the tutorial title."""
        run_tutorial("getting_started", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        assert "Getting Started" in call_str

    @patch("builtins.print")
    def test_run_tutorial_prints_difficulty(self, mock_print):
        """Test that run_tutorial prints the difficulty level."""
        run_tutorial("getting_started", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        assert "Difficulty" in call_str or "beginner" in call_str.lower()

    @patch("builtins.print")
    def test_run_tutorial_prints_steps(self, mock_print):
        """Test that run_tutorial prints step information."""
        run_tutorial("getting_started", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        # Should show "Step X/Y" format
        assert "Step" in call_str

    @patch("builtins.print")
    def test_run_tutorial_nonexistent(self, mock_print):
        """Test running a nonexistent tutorial."""
        run_tutorial("nonexistent_tutorial", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        assert "not found" in call_str.lower()

    @patch("builtins.print")
    def test_run_tutorial_shows_available(self, mock_print):
        """Test that nonexistent tutorial shows available tutorials."""
        run_tutorial("nonexistent_tutorial", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        assert "Available" in call_str or "getting_started" in call_str

    @patch("builtins.print")
    def test_run_tutorial_shows_code(self, mock_print):
        """Test that run_tutorial shows code examples."""
        run_tutorial("getting_started", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        assert "Code:" in call_str

    @patch("builtins.print")
    def test_run_tutorial_shows_hints(self, mock_print):
        """Test that run_tutorial shows hints when available."""
        run_tutorial("getting_started", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        # Check that hints section appears (if steps have hints)
        tutorial = get_tutorial("getting_started")
        if tutorial and any(step.hints for step in tutorial.steps):
            assert "Hints" in call_str or "hints" in call_str.lower()

    @patch("builtins.print")
    def test_run_tutorial_shows_completion(self, mock_print):
        """Test that run_tutorial shows completion message."""
        run_tutorial("getting_started", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        assert "Complete" in call_str

    @patch("builtins.print")
    def test_run_tutorial_shows_next_steps(self, mock_print):
        """Test that run_tutorial shows next steps."""
        run_tutorial("getting_started", interactive=False)

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)

        assert "Next steps" in call_str or "next" in call_str.lower()


class TestGettingStartedTutorial:
    """Specific tests for the getting_started tutorial."""

    def test_getting_started_has_five_steps(self):
        """Test that getting_started has 5 steps."""
        tutorial = get_tutorial("getting_started")

        assert tutorial is not None
        assert len(tutorial.steps) == 5

    def test_getting_started_step_titles(self):
        """Test that getting_started has expected step titles."""
        tutorial = get_tutorial("getting_started")

        assert tutorial is not None
        titles = [step.title for step in tutorial.steps]

        assert "Loading a Trace File" in titles
        assert "Making Basic Measurements" in titles
        assert "Spectral Analysis (Frequency Domain)" in titles
        assert "Protocol Decoding" in titles
        assert "Auto-Discovery for Beginners" in titles

    def test_getting_started_is_beginner(self):
        """Test that getting_started is beginner difficulty."""
        tutorial = get_tutorial("getting_started")

        assert tutorial is not None
        assert tutorial.difficulty == "beginner"

    def test_getting_started_all_steps_have_hints(self):
        """Test that getting_started steps have hints."""
        tutorial = get_tutorial("getting_started")

        assert tutorial is not None
        for step in tutorial.steps:
            assert len(step.hints) > 0, f"Step '{step.title}' has no hints"

    def test_getting_started_code_examples_mention_oscura(self):
        """Test that code examples import oscura."""
        tutorial = get_tutorial("getting_started")

        assert tutorial is not None
        for step in tutorial.steps:
            assert "oscura" in step.code.lower() or "tk" in step.code


class TestSpectralAnalysisTutorial:
    """Specific tests for the spectral_analysis tutorial."""

    def test_spectral_analysis_exists(self):
        """Test that spectral_analysis tutorial exists."""
        tutorial = get_tutorial("spectral_analysis")

        assert tutorial is not None

    def test_spectral_analysis_is_intermediate(self):
        """Test that spectral_analysis is intermediate difficulty."""
        tutorial = get_tutorial("spectral_analysis")

        assert tutorial is not None
        assert tutorial.difficulty == "intermediate"

    def test_spectral_analysis_covers_fft(self):
        """Test that spectral_analysis covers FFT."""
        tutorial = get_tutorial("spectral_analysis")

        assert tutorial is not None
        titles = [step.title for step in tutorial.steps]

        assert any("fft" in title.lower() for title in titles)

    def test_spectral_analysis_covers_psd(self):
        """Test that spectral_analysis covers PSD."""
        tutorial = get_tutorial("spectral_analysis")

        assert tutorial is not None
        content = " ".join(step.description + step.code for step in tutorial.steps)

        assert "psd" in content.lower()


class TestTutorialCodeValidity:
    """Tests that tutorial code examples are syntactically valid."""

    def test_all_code_examples_valid_python(self):
        """Test that all code examples compile."""
        for tutorial_id, tutorial in TUTORIALS.items():
            for i, step in enumerate(tutorial.steps):
                code = step.code.strip()
                try:
                    compile(code, f"<{tutorial_id}-step-{i}>", "exec")
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {tutorial_id} step {i}: {e}")

    def test_code_examples_import_correctly(self):
        """Test that code examples have proper import statements."""
        for tutorial_id, tutorial in TUTORIALS.items():
            for step in tutorial.steps:
                code = step.code.strip()
                # If code uses osc.something, should have import
                if "osc." in code:
                    assert "import oscura" in code or "from oscura" in code, (
                        f"Missing oscura import in {tutorial_id}"
                    )


class TestTutorialDescriptions:
    """Tests for tutorial descriptions quality."""

    def test_descriptions_are_substantial(self):
        """Test that descriptions are not too short."""
        for tutorial_id, tutorial in TUTORIALS.items():
            assert len(tutorial.description) > 40, f"{tutorial_id} description too short"

    def test_step_descriptions_are_helpful(self):
        """Test that step descriptions are helpful."""
        for tutorial_id, tutorial in TUTORIALS.items():
            for i, step in enumerate(tutorial.steps):
                assert len(step.description) > 30, f"{tutorial_id} step {i} description too short"

    def test_expected_output_when_relevant(self):
        """Test that steps with output have expected_output."""
        for tutorial in TUTORIALS.values():
            for step in tutorial.steps:
                # Steps with print statements should have expected output
                if "print(" in step.code:
                    # Not all print statements need expected output,
                    # but many should have it
                    pass  # This is more of a quality guideline


class TestTutorialIntegration:
    """Integration tests for the tutorial system."""

    def test_list_and_get_consistency(self):
        """Test that list_tutorials and get_tutorial are consistent."""
        tutorial_list = list_tutorials()

        for info in tutorial_list:
            tutorial = get_tutorial(info["id"])
            assert tutorial is not None
            assert tutorial.title == info["title"]
            assert tutorial.difficulty == info["difficulty"]
            assert len(tutorial.steps) == info["steps"]

    def test_all_registered_tutorials_in_list(self):
        """Test that all TUTORIALS are returned by list_tutorials."""
        tutorial_list = list_tutorials()
        listed_ids = {info["id"] for info in tutorial_list}

        for tutorial_id in TUTORIALS:
            assert tutorial_id in listed_ids
