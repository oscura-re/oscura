import pytest

"""Unit tests for the analysis wizard module.

Tests the interactive analysis wizard that guides users through
signal analysis with intelligent recommendations.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np

from oscura.onboarding.wizard import (
    AnalysisWizard,
    WizardAction,
    WizardResult,
    WizardStep,
    run_wizard,
)

pytestmark = pytest.mark.unit


@dataclass
class MockTraceMetadata:
    """Mock metadata for test traces."""

    sample_rate: float = 1_000_000.0
    channel_name: str = "CH1"


@dataclass
class MockTrace:
    """Mock trace for testing the wizard."""

    data: np.ndarray
    metadata: MockTraceMetadata

    def __init__(
        self,
        samples: int = 1000,
        sample_rate: float = 1_000_000.0,
        channel_name: str = "CH1",
    ):
        self.data = np.sin(2 * np.pi * 1000 * np.arange(samples) / sample_rate)
        self.metadata = MockTraceMetadata(sample_rate=sample_rate, channel_name=channel_name)


class TestWizardAction:
    """Tests for the WizardAction enum."""

    def test_wizard_actions_exist(self):
        """Test that all expected wizard actions are defined."""
        assert WizardAction.MEASURE.value == "measure"
        assert WizardAction.CHARACTERIZE.value == "characterize"
        assert WizardAction.DECODE.value == "decode"
        assert WizardAction.FILTER.value == "filter"
        assert WizardAction.SPECTRAL.value == "spectral"
        assert WizardAction.COMPARE.value == "compare"

    def test_wizard_action_values(self):
        """Test that all actions have string values."""
        for action in WizardAction:
            assert isinstance(action.value, str)


class TestWizardStep:
    """Tests for the WizardStep dataclass."""

    def test_wizard_step_creation(self):
        """Test creating a basic wizard step."""
        step = WizardStep(
            title="Test Step",
            question="What would you like to do?",
            options=["Option A", "Option B"],
        )

        assert step.title == "Test Step"
        assert step.question == "What would you like to do?"
        assert len(step.options) == 2
        assert step.action is None
        assert step.help_text == ""
        assert step.skip_condition is None

    def test_wizard_step_with_action(self):
        """Test creating a wizard step with an action callback."""

        def mock_action(choice: int) -> None:
            pass

        step = WizardStep(
            title="Action Step",
            question="Select an option",
            options=["Option 1"],
            action=mock_action,
            help_text="This is a helpful hint",
        )

        assert step.action is not None
        assert step.help_text == "This is a helpful hint"

    def test_wizard_step_with_skip_condition(self):
        """Test wizard step with a skip condition."""

        def should_skip(result: WizardResult) -> bool:
            return result.steps_completed > 2

        step = WizardStep(
            title="Optional Step",
            question="Do you want to continue?",
            options=["Yes", "No"],
            skip_condition=should_skip,
        )

        result = WizardResult(steps_completed=3)
        assert step.skip_condition is not None
        assert step.skip_condition(result) is True

        result2 = WizardResult(steps_completed=1)
        assert step.skip_condition(result2) is False


class TestWizardResult:
    """Tests for the WizardResult dataclass."""

    def test_wizard_result_defaults(self):
        """Test WizardResult default values."""
        result = WizardResult()

        assert result.steps_completed == 0
        assert result.measurements == {}
        assert result.recommendations == []
        assert result.summary == ""

    def test_wizard_result_with_data(self):
        """Test WizardResult with populated data."""
        result = WizardResult(
            steps_completed=3,
            measurements={"frequency": 1000.0, "amplitude": 3.3},
            recommendations=["Consider filtering the signal"],
            summary="Analysis complete",
        )

        assert result.steps_completed == 3
        assert result.measurements["frequency"] == 1000.0
        assert len(result.recommendations) == 1
        assert result.summary == "Analysis complete"

    def test_wizard_result_mutable_defaults(self):
        """Test that mutable defaults are handled correctly."""
        result1 = WizardResult()
        result2 = WizardResult()

        result1.measurements["test"] = 123
        assert "test" not in result2.measurements


class TestAnalysisWizard:
    """Tests for the AnalysisWizard class."""

    def test_wizard_initialization(self):
        """Test wizard initialization with a trace."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        assert wizard.trace is trace
        assert isinstance(wizard.result, WizardResult)
        assert wizard.current_step == 0
        assert len(wizard.steps) > 0

    def test_wizard_builds_steps(self):
        """Test that wizard builds expected steps."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        # Check that steps have expected structure
        for step in wizard.steps:
            assert isinstance(step, WizardStep)
            assert step.title
            assert step.question
            assert len(step.options) > 0

    def test_wizard_step_titles(self):
        """Test that wizard steps have meaningful titles."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        titles = [step.title for step in wizard.steps]
        assert "Signal Type Detection" in titles
        assert "Basic Measurements" in titles
        assert "Spectral Analysis" in titles
        assert "Signal Quality" in titles

    @patch("builtins.print")
    def test_show_trace_summary(self, mock_print):
        """Test trace summary display."""
        trace = MockTrace(samples=10000, sample_rate=1e6, channel_name="CH1")
        wizard = AnalysisWizard(trace)

        wizard._show_trace_summary()

        # Verify print was called with summary info
        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)
        assert "Loaded trace summary" in call_str

    @patch("builtins.print")
    @patch("builtins.input", return_value="1")
    def test_wizard_run_non_interactive(self, mock_input, mock_print):
        """Test running wizard in non-interactive mode."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        result = wizard.run(interactive=False)

        assert isinstance(result, WizardResult)
        assert result.steps_completed == len(wizard.steps)

    def test_get_user_choice_validation(self):
        """Test user choice validation logic."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        # Test with mock input
        with patch("builtins.input", side_effect=["5", "0", "abc", "2"]):
            choice = wizard._get_user_choice(4)
            assert choice == 2

    @patch("builtins.print")
    def test_handle_signal_type_auto(self, mock_print):
        """Test auto signal type detection."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        # Mock the discovery module
        with patch("oscura.discovery.characterize_signal") as mock_characterize:
            mock_result = MagicMock()
            mock_result.signal_type = "digital"
            mock_result.confidence = 0.95
            mock_characterize.return_value = mock_result

            wizard._handle_signal_type(1)

            assert wizard.result.measurements.get("signal_type") == "digital"
            assert wizard.result.measurements.get("signal_confidence") == 0.95

    @patch("builtins.print")
    def test_handle_signal_type_digital(self, mock_print):
        """Test manual digital signal type selection."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._handle_signal_type(2)
        assert wizard.result.measurements.get("signal_type") == "digital"

    @patch("builtins.print")
    def test_handle_signal_type_analog(self, mock_print):
        """Test manual analog signal type selection."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._handle_signal_type(3)
        assert wizard.result.measurements.get("signal_type") == "analog"

    @patch("builtins.print")
    def test_handle_signal_type_protocol(self, mock_print):
        """Test protocol signal type selection."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._handle_signal_type(4)
        assert wizard.result.measurements.get("signal_type") == "protocol"
        assert len(wizard.result.recommendations) > 0

    @patch("builtins.print")
    def test_handle_signal_type_power(self, mock_print):
        """Test power analysis signal type selection."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._handle_signal_type(5)
        assert wizard.result.measurements.get("signal_type") == "power"

    @patch("builtins.print")
    def test_handle_measurements_skip(self, mock_print):
        """Test skipping measurements."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._handle_measurements(4)  # Skip option
        # No measurements should be added beyond defaults
        assert "rise_time" not in wizard.result.measurements

    @patch("builtins.print")
    def test_handle_spectral_skip(self, mock_print):
        """Test skipping spectral analysis."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._handle_spectral(4)  # Skip option
        assert "fft_peak_freq" not in wizard.result.measurements

    @patch("builtins.print")
    def test_handle_quality_skip(self, mock_print):
        """Test skipping quality assessment."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._handle_quality(4)  # Skip option
        assert "thd" not in wizard.result.measurements

    @patch("builtins.print")
    def test_generate_summary_empty(self, mock_print):
        """Test summary generation with no measurements."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard._generate_summary()
        assert wizard.result.summary.startswith("Analysis Summary:")

    @patch("builtins.print")
    def test_generate_summary_with_measurements(self, mock_print):
        """Test summary generation with measurements."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        wizard.result.measurements = {
            "signal_type": "digital",
            "frequency": 10e6,  # 10 MHz
            "rise_time": 2.5e-9,  # 2.5 ns
            "thd": -45.0,
            "snr": 52.0,
        }

        wizard._generate_summary()

        summary = wizard.result.summary
        assert "digital" in summary
        assert "MHz" in summary  # Frequency formatted with MHz
        assert "ns" in summary  # Rise time formatted with ns
        assert "THD" in summary
        assert "SNR" in summary

    @patch("builtins.print")
    def test_generate_summary_dict_format(self, mock_print):
        """Test summary generation with dict-format measurements."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        # Some measurement functions return dicts with 'value' key
        wizard.result.measurements = {
            "frequency": {"value": 1e6},
            "rise_time": {"value": 5e-9},
        }

        wizard._generate_summary()

        summary = wizard.result.summary
        assert "Analysis Summary:" in summary


class TestRunWizard:
    """Tests for the run_wizard convenience function."""

    @patch("builtins.print")
    @patch("builtins.input", return_value="1")
    def test_run_wizard_non_interactive(self, mock_input, mock_print):
        """Test run_wizard in non-interactive mode."""
        trace = MockTrace()
        result = run_wizard(trace, interactive=False)

        assert isinstance(result, WizardResult)
        assert result.steps_completed > 0

    @patch("builtins.print")
    @patch("builtins.input", return_value="1")
    def test_run_wizard_returns_result(self, mock_input, mock_print):
        """Test that run_wizard returns a WizardResult."""
        trace = MockTrace()
        result = run_wizard(trace, interactive=False)

        assert hasattr(result, "steps_completed")
        assert hasattr(result, "measurements")
        assert hasattr(result, "recommendations")
        assert hasattr(result, "summary")


class TestWizardEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("builtins.print")
    def test_wizard_with_minimal_trace(self, mock_print):
        """Test wizard with minimal trace data."""

        @dataclass
        class MinimalTrace:
            data: np.ndarray

        trace = MinimalTrace(data=np.array([1.0, 2.0, 3.0]))
        wizard = AnalysisWizard(trace)

        # Should not raise
        wizard._show_trace_summary()

    @patch("builtins.print")
    def test_wizard_trace_without_metadata(self, mock_print):
        """Test wizard with trace lacking metadata."""

        @dataclass
        class NoMetadataTrace:
            data: np.ndarray

        trace = NoMetadataTrace(data=np.zeros(100))
        wizard = AnalysisWizard(trace)

        # Should handle missing metadata gracefully
        wizard._show_trace_summary()

    @patch("builtins.print")
    def test_handle_signal_type_auto_failure(self, mock_print):
        """Test auto signal type detection failure handling."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        with patch(
            "oscura.discovery.characterize_signal",
            side_effect=Exception("Detection failed"),
        ):
            wizard._handle_signal_type(1)
            assert wizard.result.measurements.get("signal_type") == "unknown"

    @patch("builtins.print")
    def test_handle_measurements_error(self, mock_print):
        """Test measurement error handling."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        with patch("oscura.measure", side_effect=Exception("Measurement failed")):
            # Should not raise, but should print error
            wizard._handle_measurements(1)

    @patch("builtins.print")
    def test_handle_spectral_error(self, mock_print):
        """Test spectral analysis error handling."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        with patch("oscura.fft", side_effect=Exception("FFT failed")):
            # Should not raise, but should print error
            wizard._handle_spectral(1)

    @patch("builtins.print")
    def test_handle_quality_error(self, mock_print):
        """Test quality assessment error handling."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        with patch("oscura.thd", side_effect=Exception("THD failed")):
            # Should not raise, but should print error
            wizard._handle_quality(1)


class TestWizardRecommendations:
    """Tests for wizard recommendation generation."""

    @patch("builtins.print")
    def test_low_confidence_recommendation(self, mock_print):
        """Test recommendation for low confidence detection."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        with patch("oscura.discovery.characterize_signal") as mock_characterize:
            mock_result = MagicMock()
            mock_result.signal_type = "unknown"
            mock_result.confidence = 0.5  # Low confidence
            mock_result.alternatives = [
                MagicMock(signal_type="digital"),
                MagicMock(signal_type="analog"),
            ]
            mock_characterize.return_value = mock_result

            wizard._handle_signal_type(1)

            # Should add recommendation about low confidence
            assert any("confidence" in rec.lower() for rec in wizard.result.recommendations)

    @patch("builtins.print")
    def test_high_thd_recommendation(self, mock_print):
        """Test recommendation for high THD."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        with patch("oscura.thd", return_value=-30.0):  # High THD (bad)
            with patch("oscura.snr", return_value=50.0):
                wizard._handle_quality(1)

                # Should recommend filtering
                assert any("filter" in rec.lower() for rec in wizard.result.recommendations)

    @patch("builtins.print")
    def test_low_snr_recommendation(self, mock_print):
        """Test recommendation for low SNR."""
        trace = MockTrace()
        wizard = AnalysisWizard(trace)

        with patch("oscura.thd", return_value=-50.0):
            with patch("oscura.snr", return_value=30.0):  # Low SNR (noisy)
                wizard._handle_quality(1)

                # Should recommend noise reduction
                assert any("noisy" in rec.lower() for rec in wizard.result.recommendations)


class TestWizardSampleRateDisplay:
    """Tests for sample rate display formatting."""

    @patch("builtins.print")
    def test_gsa_display(self, mock_print):
        """Test GSa/s display for high sample rates."""
        trace = MockTrace(sample_rate=2e9)  # 2 GSa/s
        wizard = AnalysisWizard(trace)

        wizard._show_trace_summary()

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)
        assert "GSa/s" in call_str

    @patch("builtins.print")
    def test_msa_display(self, mock_print):
        """Test MSa/s display for medium sample rates."""
        trace = MockTrace(sample_rate=100e6)  # 100 MSa/s
        wizard = AnalysisWizard(trace)

        wizard._show_trace_summary()

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)
        assert "MSa/s" in call_str

    @patch("builtins.print")
    def test_ksa_display(self, mock_print):
        """Test kSa/s display for low sample rates."""
        trace = MockTrace(sample_rate=50e3)  # 50 kSa/s
        wizard = AnalysisWizard(trace)

        wizard._show_trace_summary()

        calls = [str(call) for call in mock_print.call_args_list]
        call_str = " ".join(calls)
        assert "kSa/s" in call_str
