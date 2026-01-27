"""Interactive analysis wizard for guided workflows.

This module provides step-by-step guided workflows that walk non-experts
through signal analysis, asking simple questions and adapting based on
responses.


Example:
    >>> from oscura.guidance import AnalysisWizard
    >>> wizard = AnalysisWizard(trace)
    >>> result = wizard.run()

References:
    Oscura Auto-Discovery Specification
    Phase 34 Task-247
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from oscura.core.types import WaveformTrace


@dataclass
class WizardStep:
    """Single step in the analysis wizard.

    Attributes:
        number: Step number (1-based).
        id: Step identifier.
        question: Question text in plain English.
        options: List of answer options.
        default: Default/recommended answer.
        skip_if_confident: Skip if auto-detection confidence >= threshold.
        user_response: User's answer.
        confidence_before: Confidence before step.
        confidence_after: Confidence after step.
        preview: Preview of result.
    """

    number: int
    id: str
    question: str
    options: list[str] = field(default_factory=list)
    default: str | None = None
    skip_if_confident: bool = False
    user_response: str | None = None
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    preview: Any = None


@dataclass
class WizardResult:
    """Result from wizard analysis.

    Attributes:
        summary: Summary of analysis.
        signal_type: Detected signal type.
        parameters: Signal parameters.
        quality: Quality assessment.
        decode: Decoded data (if applicable).
        recommendations: Next step recommendations.
        confidence: Overall confidence.
    """

    summary: str
    signal_type: str | None = None
    parameters: dict | None = None  # type: ignore[type-arg]
    quality: Any = None
    decode: Any = None
    recommendations: list = field(default_factory=list)  # type: ignore[type-arg]
    confidence: float = 0.0


class AnalysisWizard:
    """Interactive analysis wizard for guided workflows.

    Provides step-by-step guided analysis with smart defaults,
    auto-skip based on confidence, progress tracking, and live previews.

    Attributes:
        trace: Waveform being analyzed.
        max_questions: Maximum questions to ask (default 5).
        auto_detect_threshold: Confidence threshold for auto-skip (default 0.8).
        enable_preview: Enable live result preview.
        allow_backtrack: Allow back/forward navigation.
        interactive: Enable interactive mode.
        step_history: History of completed steps.
        steps_completed: Number of steps completed.
        questions_asked: Number of questions asked.
        questions_skipped: Number of questions skipped.
        session_duration_seconds: Total session duration.
        _start_time: Session start time.
        _current_state: Current analysis state.
    """

    def __init__(
        self,
        trace: WaveformTrace,
        *,
        max_questions: int = 5,
        auto_detect_threshold: float = 0.8,
        enable_preview: bool = True,
        allow_backtrack: bool = True,
        interactive: bool = True,
    ) -> None:
        """Initialize analysis wizard.

        Args:
            trace: Waveform to analyze.
            max_questions: Maximum questions (default 5, range 3-7).
            auto_detect_threshold: Skip question if confidence >= this.
            enable_preview: Enable live result preview after each step.
            allow_backtrack: Allow back/forward navigation.
            interactive: Enable interactive mode (vs programmatic).

        References:
            DISC-006: Interactive Analysis Wizard
        """
        self.trace = trace
        self.max_questions = max(3, min(7, max_questions))
        self.auto_detect_threshold = auto_detect_threshold
        self.enable_preview = enable_preview
        self.allow_backtrack = allow_backtrack
        self.interactive = interactive

        self.step_history: list[WizardStep] = []
        self.steps_completed = 0
        self.questions_asked = 0
        self.questions_skipped = 0
        self.session_duration_seconds = 0.0

        self._start_time = datetime.now()
        self._current_state: dict[str, Any] = {}
        self._predefined_answers: dict[str, str] = {}

    def add_custom_step(
        self,
        step_id: str,
        *,
        question: str,
        options: list[str],
        default: str | None = None,
        skip_if_confident: bool = True,
    ) -> None:
        """Add a custom step to the wizard.

        Args:
            step_id: Unique step identifier.
            question: Question text in plain English.
            options: List of answer options.
            default: Default/recommended answer.
            skip_if_confident: Skip if auto-detection confident.
        """
        # Store for later use during run()
        if not hasattr(self, "_custom_steps"):
            self._custom_steps = []

        self._custom_steps.append(
            {
                "id": step_id,
                "question": question,
                "options": options,
                "default": default,
                "skip_if_confident": skip_if_confident,
            }
        )

    def set_answers(self, answers: dict[str, str]) -> None:
        """Set predefined answers for programmatic mode.

        Args:
            answers: Dictionary mapping step IDs to answers.
        """
        self._predefined_answers = answers

    def _execute_characterization_step(self, preview_callback: Callable[[Any], None] | None) -> Any:
        """Execute signal characterization step.

        Args:
            preview_callback: Optional callback for step preview.

        Returns:
            Characterization result object with signal_type and confidence.

        Example:
            >>> result = wizard._execute_characterization_step(None)
            >>> print(result.signal_type)
        """
        from oscura.discovery import characterize_signal

        step = WizardStep(
            number=1,
            id="characterization",
            question="What type of signal are you analyzing?",
            options=[
                "Serial data (UART, SPI, I2C)",
                "PWM / Motor control",
                "Analog sensor output",
                "Not sure - auto-detect",
            ],
            default="Not sure - auto-detect",
            skip_if_confident=False,
        )

        char_result = characterize_signal(self.trace)
        step.confidence_before = 0.0
        step.confidence_after = char_result.confidence
        self._current_state["characterization"] = char_result

        if self.interactive and char_result.confidence < self.auto_detect_threshold:
            step.user_response = self._predefined_answers.get("signal_type", step.default)
            self.questions_asked += 1
        else:
            signal_type = getattr(char_result, "signal_type", "Unknown")
            step.user_response = f"Auto-detected: {signal_type}"
            self.questions_skipped += 1

        self.step_history.append(step)
        self.steps_completed += 1

        if preview_callback and self.enable_preview:
            preview_callback(char_result)

        return char_result

    def _execute_quality_assessment_step(
        self, char_result: Any, preview_callback: Callable[[Any], None] | None
    ) -> Any:
        """Execute data quality assessment step.

        Args:
            char_result: Result from characterization step.
            preview_callback: Optional callback for step preview.

        Returns:
            Quality assessment result with status and confidence.

        Example:
            >>> quality = wizard._execute_quality_assessment_step(char_result, None)
            >>> print(quality.status)
        """
        from oscura.discovery import assess_data_quality

        step = WizardStep(
            number=2,
            id="quality",
            question="Check data quality?",
            options=["Yes", "No"],
            default="Yes",
            skip_if_confident=True,
        )

        quality = assess_data_quality(self.trace)
        step.confidence_before = char_result.confidence
        step.confidence_after = quality.confidence
        self._current_state["quality"] = quality

        if self.interactive and self.questions_asked < self.max_questions:
            step.user_response = self._predefined_answers.get("check_quality", "Yes")
            if step.user_response == "Yes":
                self.questions_asked += 1
        else:
            step.user_response = "Skipped (auto-assessed)"
            self.questions_skipped += 1

        self.step_history.append(step)
        self.steps_completed += 1

        if preview_callback and self.enable_preview:
            preview_callback(quality)

        return quality

    def _execute_protocol_decode_step(
        self, char_result: Any, preview_callback: Callable[[Any], None] | None
    ) -> Any | None:
        """Execute protocol decoding step if applicable.

        Args:
            char_result: Result from characterization step.
            preview_callback: Optional callback for step preview.

        Returns:
            Decode result if protocol detected, None otherwise.

        Example:
            >>> decode = wizard._execute_protocol_decode_step(char_result, None)
            >>> if decode:
            ...     print(len(decode.data))
        """
        from oscura.discovery import decode_protocol

        decode_result = None

        if not (hasattr(char_result, "signal_type") and char_result.confidence >= 0.7):
            return decode_result

        signal_type = char_result.signal_type.lower()
        if not any(proto in signal_type for proto in ["uart", "spi", "i2c", "can"]):
            return decode_result

        step = WizardStep(
            number=3,
            id="decode",
            question=f"Auto-detected {char_result.signal_type}. Decode data?",
            options=["Yes", "No"],
            default="Yes",
            skip_if_confident=True,
        )

        if self.interactive and self.questions_asked < self.max_questions:
            step.user_response = self._predefined_answers.get("decode_data", "Yes")
            if step.user_response == "Yes":
                decode_result = decode_protocol(self.trace)
                self._current_state["decode"] = decode_result
                self.questions_asked += 1
        else:
            decode_result = decode_protocol(self.trace)
            self._current_state["decode"] = decode_result
            step.user_response = "Auto-decoded"
            self.questions_skipped += 1

        step.confidence_before = char_result.confidence
        step.confidence_after = decode_result.overall_confidence if decode_result else 0.0

        self.step_history.append(step)
        self.steps_completed += 1

        if preview_callback and self.enable_preview and decode_result:
            preview_callback(decode_result)

        return decode_result

    def _execute_anomaly_detection_step(
        self, quality: Any, preview_callback: Callable[[Any], None] | None
    ) -> Any | None:
        """Execute anomaly detection step if quality issues exist.

        Args:
            quality: Result from quality assessment step.
            preview_callback: Optional callback for step preview.

        Returns:
            Anomaly detection results if quality issues found, None otherwise.

        Example:
            >>> anomalies = wizard._execute_anomaly_detection_step(quality, None)
            >>> if anomalies:
            ...     print(len(anomalies))
        """
        from oscura.discovery import find_anomalies

        anomalies = None

        if quality.status not in ["WARNING", "FAIL"]:
            return anomalies

        step = WizardStep(
            number=self.steps_completed + 1,
            id="anomalies",
            question="Quality concerns detected. Check for anomalies?",
            options=["Yes", "No"],
            default="Yes",
        )

        if self.interactive and self.questions_asked < self.max_questions:
            step.user_response = self._predefined_answers.get("check_anomalies", "Yes")
            if step.user_response == "Yes":
                anomalies = find_anomalies(self.trace)
                self._current_state["anomalies"] = anomalies
                self.questions_asked += 1
        else:
            anomalies = find_anomalies(self.trace)
            self._current_state["anomalies"] = anomalies
            step.user_response = "Auto-checked"
            self.questions_skipped += 1

        self.step_history.append(step)
        self.steps_completed += 1

        if preview_callback and self.enable_preview and anomalies:
            preview_callback(anomalies)

        return anomalies

    def _build_summary(
        self, char_result: Any, quality: Any, decode_result: Any | None, anomalies: Any | None
    ) -> str:
        """Build wizard result summary from analysis steps.

        Args:
            char_result: Characterization result.
            quality: Quality assessment result.
            decode_result: Protocol decode result (optional).
            anomalies: Anomaly detection results (optional).

        Returns:
            Multi-line summary string.

        Example:
            >>> summary = wizard._build_summary(char, quality, decode, anomalies)
            >>> print(summary)
            Signal type: UART
            Quality: Good
            Decoded: 1024 bytes
        """
        summary_parts = []

        if hasattr(char_result, "signal_type") and char_result.signal_type:
            summary_parts.append(f"Signal type: {char_result.signal_type}")
            if hasattr(char_result, "parameters"):
                summary_parts.append(f"Parameters: {_format_params(char_result.parameters)}")

        if quality.status == "PASS":
            summary_parts.append("Quality: Good")
        elif quality.status == "WARNING":
            summary_parts.append("Quality: Fair (some concerns)")
        else:
            summary_parts.append("Quality: Poor (issues detected)")

        if decode_result:
            byte_count = len(decode_result.data) if hasattr(decode_result, "data") else 0
            summary_parts.append(f"Decoded: {byte_count} bytes")

        if anomalies and len(anomalies) > 0:
            critical = sum(1 for a in anomalies if a.severity == "CRITICAL")
            if critical > 0:
                summary_parts.append(f"Anomalies: {critical} critical issues")

        return "\n".join(summary_parts)

    def run(
        self,
        *,
        preview_callback: Callable[[Any], None] | None = None,
    ) -> WizardResult:
        """Run the analysis wizard.

        Guides user through analysis steps, auto-skipping where confident,
        showing progress, and providing live previews.

        Args:
            preview_callback: Optional callback for step previews.

        Returns:
            WizardResult with analysis summary and findings.

        Example:
            >>> wizard = AnalysisWizard(trace)
            >>> result = wizard.run()
            >>> print(result.summary)

        References:
            DISC-006: Interactive Analysis Wizard
        """
        from oscura.guidance import suggest_next_steps

        # Execute analysis steps
        char_result = self._execute_characterization_step(preview_callback)
        quality = self._execute_quality_assessment_step(char_result, preview_callback)
        decode_result = self._execute_protocol_decode_step(char_result, preview_callback)
        anomalies = self._execute_anomaly_detection_step(quality, preview_callback)

        # Generate recommendations and results
        recommendations = suggest_next_steps(
            self.trace,
            current_state=self._current_state,
        )

        summary = self._build_summary(char_result, quality, decode_result, anomalies)
        self.session_duration_seconds = (datetime.now() - self._start_time).total_seconds()

        return WizardResult(
            summary=summary,
            signal_type=char_result.signal_type if hasattr(char_result, "signal_type") else None,
            parameters=char_result.parameters if hasattr(char_result, "parameters") else None,
            quality=quality,
            decode=decode_result,
            recommendations=recommendations,
            confidence=char_result.confidence,
        )

    @classmethod
    def from_session(cls, session_file: str) -> AnalysisWizard:
        """Load wizard from saved session file.

        Args:
            session_file: Path to session JSON file.

        Returns:
            AnalysisWizard instance configured from session.

        Raises:
            FileNotFoundError: If session file doesn't exist.
            ValueError: If session file is invalid.
        """
        import json
        from pathlib import Path

        path = Path(session_file)
        if not path.exists():
            msg = f"Session file not found: {session_file}"
            raise FileNotFoundError(msg)

        with path.open() as f:
            session_data = json.load(f)

        # Extract trace path and load
        from oscura import load

        trace_path = session_data.get("trace_path")
        if not trace_path:
            msg = "Session file missing trace_path"
            raise ValueError(msg)

        trace = load(trace_path)

        # Create wizard with saved settings
        wizard = cls(
            trace,  # type: ignore[arg-type]
            max_questions=session_data.get("max_questions", 5),
            auto_detect_threshold=session_data.get("auto_detect_threshold", 0.8),
            enable_preview=session_data.get("enable_preview", True),
            allow_backtrack=session_data.get("allow_backtrack", True),
            interactive=session_data.get("interactive", True),
        )

        # Set predefined answers
        if "answers" in session_data:
            wizard.set_answers(session_data["answers"])

        return wizard

    def save_session(self, output_path: str) -> None:
        """Save wizard session to JSON file.

        Args:
            output_path: Path for output JSON file.
        """
        import json
        from pathlib import Path

        session_data = {
            "trace_path": str(getattr(self.trace, "path", "")),
            "max_questions": self.max_questions,
            "auto_detect_threshold": self.auto_detect_threshold,
            "enable_preview": self.enable_preview,
            "allow_backtrack": self.allow_backtrack,
            "interactive": self.interactive,
            "steps_completed": self.steps_completed,
            "questions_asked": self.questions_asked,
            "questions_skipped": self.questions_skipped,
            "session_duration_seconds": self.session_duration_seconds,
            "answers": self._predefined_answers,
            "step_history": [
                {
                    "number": step.number,
                    "id": step.id,
                    "question": step.question,
                    "user_response": step.user_response,
                    "confidence_before": step.confidence_before,
                    "confidence_after": step.confidence_after,
                }
                for step in self.step_history
            ],
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(session_data, f, indent=2)


def _format_params(params: dict) -> str:  # type: ignore[type-arg]
    """Format parameters dictionary for display.

    Args:
        params: Parameters dictionary.

    Returns:
        Formatted string.
    """
    if not params:
        return ""

    parts = []
    for key, value in params.items():
        if isinstance(value, int | float):
            if key.endswith("_hz") or key.endswith("_freq"):
                parts.append(f"{key}={value / 1e3:.1f}kHz")
            elif key.endswith("_baud") or key.endswith("baud_rate"):
                parts.append(f"{key}={value:.0f}")
            else:
                parts.append(f"{key}={value}")
        else:
            parts.append(f"{key}={value}")

    return ", ".join(parts[:3])  # Limit to 3 params


__all__ = [
    "AnalysisWizard",
    "WizardResult",
    "WizardStep",
]
