"""Comprehensive unit tests for wizard module.

Tests all public functions and classes in oscura.guidance.wizard.

Requirements tested:
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from oscura.guidance.wizard import (
    AnalysisWizard,
    WizardResult,
    WizardStep,
    _format_params,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_trace():
    """Create a mock waveform trace."""
    trace = MagicMock()
    trace.path = "/fake/path/test.wfm"
    trace.sample_rate = 1000000  # 1 MHz
    trace.duration = 0.01  # 10ms
    trace.num_samples = 10000
    return trace


@pytest.fixture
def mock_characterization():
    """Create a mock signal characterization result."""
    result = MagicMock()
    result.signal_type = "UART"
    result.confidence = 0.85
    result.parameters = {
        "baud_rate": 9600,
        "clock_freq_hz": 1000000,
        "bits": 8,
    }
    return result


@pytest.fixture
def mock_quality():
    """Create a mock data quality result."""
    quality = MagicMock()
    quality.status = "PASS"
    quality.confidence = 0.9
    return quality


@pytest.fixture
def mock_decode():
    """Create a mock decode result."""
    decode = MagicMock()
    decode.data = b"Hello World"
    decode.overall_confidence = 0.88
    return decode


# =============================================================================
# Test WizardStep
# =============================================================================


class TestWizardStep:
    """Tests for WizardStep dataclass."""

    def test_wizard_step_creation(self):
        """Test creating a wizard step with minimal fields."""
        step = WizardStep(
            number=1,
            id="test_step",
            question="What is your favorite color?",
        )
        assert step.number == 1
        assert step.id == "test_step"
        assert step.question == "What is your favorite color?"
        assert step.options == []
        assert step.default is None
        assert step.skip_if_confident is False
        assert step.user_response is None
        assert step.confidence_before == 0.0
        assert step.confidence_after == 0.0
        assert step.preview is None

    def test_wizard_step_with_all_fields(self):
        """Test creating a wizard step with all fields."""
        step = WizardStep(
            number=2,
            id="signal_type",
            question="What signal type?",
            options=["UART", "SPI", "I2C"],
            default="UART",
            skip_if_confident=True,
            user_response="SPI",
            confidence_before=0.5,
            confidence_after=0.85,
            preview={"type": "SPI"},
        )
        assert step.number == 2
        assert step.id == "signal_type"
        assert step.options == ["UART", "SPI", "I2C"]
        assert step.default == "UART"
        assert step.skip_if_confident is True
        assert step.user_response == "SPI"
        assert step.confidence_before == 0.5
        assert step.confidence_after == 0.85
        assert step.preview == {"type": "SPI"}


# =============================================================================
# Test WizardResult
# =============================================================================


class TestWizardResult:
    """Tests for WizardResult dataclass."""

    def test_wizard_result_minimal(self):
        """Test creating wizard result with minimal fields."""
        result = WizardResult(summary="Analysis complete")
        assert result.summary == "Analysis complete"
        assert result.signal_type is None
        assert result.parameters is None
        assert result.quality is None
        assert result.decode is None
        assert result.recommendations == []
        assert result.confidence == 0.0

    def test_wizard_result_complete(self):
        """Test creating wizard result with all fields."""
        quality = MagicMock()
        decode = MagicMock()

        result = WizardResult(
            summary="UART signal detected",
            signal_type="UART",
            parameters={"baud_rate": 9600},
            quality=quality,
            decode=decode,
            recommendations=["Check timing", "Verify protocol"],
            confidence=0.92,
        )
        assert result.summary == "UART signal detected"
        assert result.signal_type == "UART"
        assert result.parameters == {"baud_rate": 9600}
        assert result.quality is quality
        assert result.decode is decode
        assert len(result.recommendations) == 2
        assert result.confidence == 0.92


# =============================================================================
# Test AnalysisWizard Initialization
# =============================================================================


class TestAnalysisWizardInit:
    """Tests for AnalysisWizard initialization."""

    def test_wizard_init_defaults(self, mock_trace):
        """Test wizard initialization with default parameters."""
        wizard = AnalysisWizard(mock_trace)
        assert wizard.trace is mock_trace
        assert wizard.max_questions == 5
        assert wizard.auto_detect_threshold == 0.8
        assert wizard.enable_preview is True
        assert wizard.allow_backtrack is True
        assert wizard.interactive is True
        assert wizard.step_history == []
        assert wizard.steps_completed == 0
        assert wizard.questions_asked == 0
        assert wizard.questions_skipped == 0
        assert wizard.session_duration_seconds == 0.0

    def test_wizard_init_custom_params(self, mock_trace):
        """Test wizard initialization with custom parameters."""
        wizard = AnalysisWizard(
            mock_trace,
            max_questions=3,
            auto_detect_threshold=0.7,
            enable_preview=False,
            allow_backtrack=False,
            interactive=False,
        )
        assert wizard.max_questions == 3
        assert wizard.auto_detect_threshold == 0.7
        assert wizard.enable_preview is False
        assert wizard.allow_backtrack is False
        assert wizard.interactive is False

    def test_wizard_max_questions_clamping(self, mock_trace):
        """Test that max_questions is clamped to valid range."""
        # Test lower bound
        wizard = AnalysisWizard(mock_trace, max_questions=1)
        assert wizard.max_questions == 3

        # Test upper bound
        wizard = AnalysisWizard(mock_trace, max_questions=10)
        assert wizard.max_questions == 7

        # Test normal value
        wizard = AnalysisWizard(mock_trace, max_questions=5)
        assert wizard.max_questions == 5


# =============================================================================
# Test AnalysisWizard Methods
# =============================================================================


class TestAnalysisWizardMethods:
    """Tests for AnalysisWizard methods."""

    def test_add_custom_step(self, mock_trace):
        """Test adding custom steps to wizard."""
        wizard = AnalysisWizard(mock_trace)

        wizard.add_custom_step(
            "custom1",
            question="Custom question?",
            options=["Yes", "No"],
            default="Yes",
            skip_if_confident=True,
        )

        assert hasattr(wizard, "_custom_steps")
        assert len(wizard._custom_steps) == 1
        assert wizard._custom_steps[0]["id"] == "custom1"
        assert wizard._custom_steps[0]["question"] == "Custom question?"
        assert wizard._custom_steps[0]["options"] == ["Yes", "No"]
        assert wizard._custom_steps[0]["default"] == "Yes"
        assert wizard._custom_steps[0]["skip_if_confident"] is True

    def test_add_multiple_custom_steps(self, mock_trace):
        """Test adding multiple custom steps."""
        wizard = AnalysisWizard(mock_trace)

        wizard.add_custom_step(
            "step1",
            question="Question 1?",
            options=["A", "B"],
        )
        wizard.add_custom_step(
            "step2",
            question="Question 2?",
            options=["C", "D"],
        )

        assert len(wizard._custom_steps) == 2
        assert wizard._custom_steps[0]["id"] == "step1"
        assert wizard._custom_steps[1]["id"] == "step2"

    def test_set_answers(self, mock_trace):
        """Test setting predefined answers."""
        wizard = AnalysisWizard(mock_trace)

        answers = {
            "signal_type": "UART",
            "check_quality": "Yes",
            "decode_data": "Yes",
        }
        wizard.set_answers(answers)

        assert wizard._predefined_answers == answers
        assert wizard._predefined_answers["signal_type"] == "UART"

    def test_set_answers_empty(self, mock_trace):
        """Test setting empty answers dictionary."""
        wizard = AnalysisWizard(mock_trace)
        wizard.set_answers({})
        assert wizard._predefined_answers == {}


# =============================================================================
# Test AnalysisWizard.run()
# =============================================================================


class TestAnalysisWizardRun:
    """Tests for AnalysisWizard.run() method."""

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_high_confidence_auto_detect(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
        mock_characterization,
    ):
        """Test wizard run with high confidence auto-detection."""
        # Setup mocks
        mock_char.return_value = mock_characterization
        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = ["recommendation1"]

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        # Verify characterization was called
        mock_char.assert_called_once_with(mock_trace)

        # Verify result
        assert isinstance(result, WizardResult)
        assert result.signal_type == "UART"
        assert result.confidence == 0.85
        assert wizard.steps_completed >= 2
        assert wizard.session_duration_seconds > 0

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_with_decode(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
        mock_characterization,
    ):
        """Test wizard run that includes protocol decode."""
        # Setup mocks
        mock_characterization.signal_type = "UART"
        mock_characterization.confidence = 0.9
        mock_char.return_value = mock_characterization

        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result

        decode_result = MagicMock()
        decode_result.data = b"test data"
        decode_result.overall_confidence = 0.88
        mock_decode.return_value = decode_result

        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        wizard.set_answers({"decode_data": "Yes"})
        result = wizard.run()

        # Verify decode was called
        mock_decode.assert_called()
        assert result.decode is not None

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_with_quality_warning_triggers_anomaly_check(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
        mock_characterization,
    ):
        """Test that quality warnings trigger anomaly detection."""
        # Setup mocks
        mock_char.return_value = mock_characterization

        quality_result = MagicMock(status="WARNING", confidence=0.6)
        mock_quality.return_value = quality_result

        anomaly_result = [MagicMock(severity="CRITICAL")]
        mock_anomalies.return_value = anomaly_result

        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        # Verify anomaly detection was called
        mock_anomalies.assert_called()
        assert wizard.steps_completed >= 3

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_with_quality_fail_triggers_anomaly_check(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
        mock_characterization,
    ):
        """Test that quality failures trigger anomaly detection."""
        # Setup mocks
        mock_char.return_value = mock_characterization

        quality_result = MagicMock(status="FAIL", confidence=0.4)
        mock_quality.return_value = quality_result

        anomaly_result = []
        mock_anomalies.return_value = anomaly_result

        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        # Verify anomaly detection was called
        mock_anomalies.assert_called()

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_with_preview_callback(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
        mock_characterization,
    ):
        """Test wizard run with preview callback."""
        # Setup mocks
        mock_char.return_value = mock_characterization
        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = []

        preview_callback = Mock()

        wizard = AnalysisWizard(mock_trace, enable_preview=True, interactive=False)
        result = wizard.run(preview_callback=preview_callback)

        # Verify callback was called
        assert preview_callback.call_count >= 2  # At least char and quality

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_without_preview_callback(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
        mock_characterization,
    ):
        """Test wizard run without preview callback."""
        # Setup mocks
        mock_char.return_value = mock_characterization
        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, enable_preview=False, interactive=False)
        result = wizard.run()

        # Should complete without errors
        assert isinstance(result, WizardResult)

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_step_history_tracking(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
        mock_characterization,
    ):
        """Test that step history is properly tracked."""
        # Setup mocks
        mock_char.return_value = mock_characterization
        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        # Check step history
        assert len(wizard.step_history) >= 2
        assert all(isinstance(step, WizardStep) for step in wizard.step_history)
        assert wizard.step_history[0].id == "characterization"
        assert wizard.step_history[1].id == "quality"

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_run_interactive_mode_low_confidence(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
    ):
        """Test interactive mode with low confidence auto-detection."""
        # Setup mocks with low confidence
        char_result = MagicMock()
        char_result.signal_type = "Unknown"
        char_result.confidence = 0.5  # Below threshold
        char_result.parameters = {}
        mock_char.return_value = char_result

        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=True)
        wizard.set_answers({"signal_type": "Not sure - auto-detect"})
        result = wizard.run()

        # Should ask questions due to low confidence
        assert wizard.questions_asked > 0


# =============================================================================
# Test AnalysisWizard Session Persistence
# =============================================================================


class TestAnalysisWizardSession:
    """Tests for session save/load functionality."""

    def test_save_session(self, mock_trace, tmp_path):
        """Test saving wizard session to file."""
        wizard = AnalysisWizard(
            mock_trace,
            max_questions=4,
            auto_detect_threshold=0.75,
            enable_preview=False,
        )
        wizard.set_answers({"signal_type": "UART"})

        # Add some history
        wizard.steps_completed = 2
        wizard.questions_asked = 1
        wizard.questions_skipped = 1
        wizard.session_duration_seconds = 5.5

        step = WizardStep(
            number=1,
            id="test",
            question="Test?",
            user_response="Yes",
            confidence_before=0.5,
            confidence_after=0.8,
        )
        wizard.step_history.append(step)

        session_file = tmp_path / "session.json"
        wizard.save_session(str(session_file))

        # Verify file exists and has correct content
        assert session_file.exists()

        with session_file.open() as f:
            data = json.load(f)

        assert data["max_questions"] == 4
        assert data["auto_detect_threshold"] == 0.75
        assert data["enable_preview"] is False
        assert data["steps_completed"] == 2
        assert data["questions_asked"] == 1
        assert data["questions_skipped"] == 1
        assert data["session_duration_seconds"] == 5.5
        assert data["answers"] == {"signal_type": "UART"}
        assert len(data["step_history"]) == 1
        assert data["step_history"][0]["id"] == "test"

    @patch("oscura.load")
    def test_from_session_success(self, mock_load, mock_trace, tmp_path):
        """Test loading wizard from session file."""
        mock_load.return_value = mock_trace

        # Create session file
        session_data = {
            "trace_path": "/fake/path/test.wfm",
            "max_questions": 4,
            "auto_detect_threshold": 0.75,
            "enable_preview": False,
            "allow_backtrack": False,
            "interactive": False,
            "answers": {"signal_type": "UART"},
        }

        session_file = tmp_path / "session.json"
        with session_file.open("w") as f:
            json.dump(session_data, f)

        # Load wizard
        wizard = AnalysisWizard.from_session(str(session_file))

        # Verify settings
        assert wizard.max_questions == 4
        assert wizard.auto_detect_threshold == 0.75
        assert wizard.enable_preview is False
        assert wizard.allow_backtrack is False
        assert wizard.interactive is False
        assert wizard._predefined_answers == {"signal_type": "UART"}
        mock_load.assert_called_once_with("/fake/path/test.wfm")

    def test_from_session_file_not_found(self):
        """Test loading from non-existent session file."""
        with pytest.raises(FileNotFoundError, match="Session file not found"):
            AnalysisWizard.from_session("/nonexistent/session.json")

    def test_from_session_missing_trace_path(self, tmp_path):
        """Test loading from session file missing trace_path."""
        session_file = tmp_path / "bad_session.json"
        with session_file.open("w") as f:
            json.dump({"max_questions": 5}, f)

        with pytest.raises(ValueError, match="Session file missing trace_path"):
            AnalysisWizard.from_session(str(session_file))

    @patch("oscura.load")
    def test_from_session_default_values(self, mock_load, mock_trace, tmp_path):
        """Test loading from session file with missing optional fields."""
        mock_load.return_value = mock_trace

        # Minimal session file
        session_data = {
            "trace_path": "/fake/path/test.wfm",
        }

        session_file = tmp_path / "minimal_session.json"
        with session_file.open("w") as f:
            json.dump(session_data, f)

        wizard = AnalysisWizard.from_session(str(session_file))

        # Should use defaults
        assert wizard.max_questions == 5
        assert wizard.auto_detect_threshold == 0.8
        assert wizard.enable_preview is True
        assert wizard.allow_backtrack is True
        assert wizard.interactive is True


# =============================================================================
# Test _format_params Helper
# =============================================================================


class TestFormatParams:
    """Tests for _format_params helper function."""

    def test_format_params_empty(self):
        """Test formatting empty parameters."""
        assert _format_params({}) == ""
        assert _format_params(None) == ""

    def test_format_params_frequency(self):
        """Test formatting frequency parameters."""
        params = {"clock_freq_hz": 1000000}
        result = _format_params(params)
        assert "clock_freq_hz=1000.0kHz" in result

    def test_format_params_baud_rate(self):
        """Test formatting baud rate parameters."""
        params = {"baud_rate": 9600}
        result = _format_params(params)
        assert "baud_rate=9600" in result

    def test_format_params_mixed(self):
        """Test formatting mixed parameters."""
        params = {
            "baud_rate": 115200,
            "clock_freq_hz": 8000000,
            "bits": 8,
            "parity": "none",
        }
        result = _format_params(params)
        # Should limit to 3 params
        parts = result.split(", ")
        assert len(parts) == 3

    def test_format_params_generic_numbers(self):
        """Test formatting generic numeric parameters."""
        params = {
            "value1": 42,
            "value2": 3.14,
        }
        result = _format_params(params)
        assert "value1=42" in result
        assert "value2=3.14" in result

    def test_format_params_string_values(self):
        """Test formatting string parameters."""
        params = {
            "protocol": "UART",
            "mode": "async",
        }
        result = _format_params(params)
        assert "protocol=UART" in result
        assert "mode=async" in result

    def test_format_params_limit_three(self):
        """Test that parameters are limited to first three."""
        params = {
            "param1": 1,
            "param2": 2,
            "param3": 3,
            "param4": 4,
            "param5": 5,
        }
        result = _format_params(params)
        parts = result.split(", ")
        assert len(parts) == 3


# =============================================================================
# Test Summary Generation
# =============================================================================


class TestSummaryGeneration:
    """Tests for wizard summary generation."""

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_summary_includes_signal_type(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
    ):
        """Test that summary includes signal type."""
        char_result = MagicMock()
        char_result.signal_type = "UART"
        char_result.confidence = 0.9
        char_result.parameters = {"baud_rate": 9600}
        mock_char.return_value = char_result

        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        assert "Signal type: UART" in result.summary

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_summary_includes_quality_status(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
    ):
        """Test that summary includes quality status."""
        char_result = MagicMock()
        char_result.signal_type = "UART"
        char_result.confidence = 0.9
        mock_char.return_value = char_result

        # Test each quality status
        quality_statuses = [
            ("PASS", "Quality: Good"),
            ("WARNING", "Quality: Fair (some concerns)"),
            ("FAIL", "Quality: Poor (issues detected)"),
        ]

        for status, expected in quality_statuses:
            quality_result = MagicMock(status=status, confidence=0.9)
            mock_quality.return_value = quality_result
            mock_suggest.return_value = []

            wizard = AnalysisWizard(mock_trace, interactive=False)
            result = wizard.run()

            assert expected in result.summary

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_summary_includes_decode_info(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
    ):
        """Test that summary includes decode information."""
        char_result = MagicMock()
        char_result.signal_type = "UART"
        char_result.confidence = 0.9
        mock_char.return_value = char_result

        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result

        decode_result = MagicMock()
        decode_result.data = b"Hello World"
        decode_result.overall_confidence = 0.88
        mock_decode.return_value = decode_result

        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        assert "Decoded: 11 bytes" in result.summary

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.discovery.decode_protocol")
    @patch("oscura.discovery.find_anomalies")
    @patch("oscura.guidance.suggest_next_steps")
    def test_summary_includes_critical_anomalies(
        self,
        mock_suggest,
        mock_anomalies,
        mock_decode,
        mock_quality,
        mock_char,
        mock_trace,
    ):
        """Test that summary includes critical anomalies."""
        char_result = MagicMock()
        char_result.signal_type = "UART"
        char_result.confidence = 0.9
        mock_char.return_value = char_result

        quality_result = MagicMock(status="WARNING", confidence=0.6)
        mock_quality.return_value = quality_result

        # Create anomalies with critical severity
        anomaly1 = MagicMock(severity="CRITICAL")
        anomaly2 = MagicMock(severity="WARNING")
        anomaly3 = MagicMock(severity="CRITICAL")
        mock_anomalies.return_value = [anomaly1, anomaly2, anomaly3]

        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        assert "Anomalies: 2 critical issues" in result.summary


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestGuidanceWizardEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.guidance.suggest_next_steps")
    def test_characterization_without_signal_type(
        self,
        mock_suggest,
        mock_quality,
        mock_char,
        mock_trace,
    ):
        """Test handling characterization without signal_type attribute."""
        char_result = MagicMock(spec=[])  # No attributes
        char_result.confidence = 0.5
        mock_char.return_value = char_result

        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, interactive=False)
        result = wizard.run()

        # Should not crash
        assert result.signal_type is None
        assert result.parameters is None

    @patch("oscura.discovery.characterize_signal")
    @patch("oscura.discovery.assess_data_quality")
    @patch("oscura.guidance.suggest_next_steps")
    def test_max_questions_limit_enforced(
        self,
        mock_suggest,
        mock_quality,
        mock_char,
        mock_trace,
    ):
        """Test that max_questions limit is enforced."""
        char_result = MagicMock()
        char_result.signal_type = "UART"
        char_result.confidence = 0.5  # Low confidence
        char_result.parameters = {}
        mock_char.return_value = char_result

        quality_result = MagicMock(status="PASS", confidence=0.9)
        mock_quality.return_value = quality_result
        mock_suggest.return_value = []

        wizard = AnalysisWizard(mock_trace, max_questions=3, interactive=True)
        result = wizard.run()

        # Should not ask more than max_questions
        assert wizard.questions_asked <= 3

    def test_session_save_creates_parent_directories(self, mock_trace, tmp_path):
        """Test that save_session creates parent directories."""
        wizard = AnalysisWizard(mock_trace)

        nested_path = tmp_path / "nested" / "dir" / "session.json"
        wizard.save_session(str(nested_path))

        assert nested_path.exists()
        assert nested_path.parent.exists()
