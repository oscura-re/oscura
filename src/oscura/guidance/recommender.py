"""Recommendation engine for guided analysis workflow.

This module provides contextual "What should I look at next?" recommendations
based on current analysis state.


Example:
    >>> from oscura.guidance import suggest_next_steps
    >>> recommendations = suggest_next_steps(trace, current_state)
    >>> for rec in recommendations:
    ...     print(f"{rec.title}: {rec.explanation}")

References:
    Oscura Auto-Discovery Specification
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from oscura.core.types import WaveformTrace


@dataclass
class Recommendation:
    """Analysis step recommendation.

    Attributes:
        id: Unique recommendation ID.
        title: Short title (≤50 chars).
        explanation: Why this step is relevant (≤50 words).
        rationale: Detailed reasoning.
        priority: Priority score (0.0-1.0).
        urgency: Urgency score (0.0-1.0).
        ease: Ease of execution (0.0-1.0, higher = easier).
        impact: Expected impact (0.0-1.0, higher = more valuable).
        result_key: Key for storing result in state.
        execute: Optional callable to execute this step.
        impact_description: Description of expected impact.
    """

    id: str
    title: str
    explanation: str
    priority: float
    urgency: float = 0.5
    ease: float = 0.5
    impact: float = 0.5
    rationale: str = ""
    result_key: str = ""
    execute: Callable | None = None  # type: ignore[type-arg]
    impact_description: str = ""


@dataclass
class AnalysisHistory:
    """Track completed analysis steps.

    Attributes:
        steps_completed: List of completed step IDs.
        step_timestamps: Timestamps for each step.
        results: Stored results from each step.
    """

    steps_completed: list[str] = field(default_factory=list)
    step_timestamps: dict[str, datetime] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step_id: str, result: Any = None) -> None:
        """Record a completed step.

        Args:
            step_id: Step identifier.
            result: Optional result from step.
        """
        if step_id not in self.steps_completed:
            self.steps_completed.append(step_id)

        self.step_timestamps[step_id] = datetime.now()

        if result is not None:
            self.results[step_id] = result

    def was_recent(self, step_id: str, seconds: float = 60.0) -> bool:
        """Check if step was completed recently.

        Args:
            step_id: Step identifier.
            seconds: Time window in seconds.

        Returns:
            True if step was done within time window.
        """
        if step_id not in self.step_timestamps:
            return False

        elapsed = datetime.now() - self.step_timestamps[step_id]
        return elapsed.total_seconds() < seconds


def _calculate_priority(
    urgency: float,
    ease: float,
    impact: float,
) -> float:
    """Calculate recommendation priority.

    Uses weighted scoring: urgency (40%), ease (30%), impact (30%).

    Args:
        urgency: Urgency score (0.0-1.0).
        ease: Ease score (0.0-1.0).
        impact: Impact score (0.0-1.0).

    Returns:
        Priority score (0.0-1.0).
    """
    priority = 0.4 * urgency + 0.3 * ease + 0.3 * impact
    return round(priority, 2)


def _recommend_characterization(
    trace: WaveformTrace,
    state: dict,  # type: ignore[type-arg]
    history: AnalysisHistory,
) -> Recommendation | None:
    """Recommend signal characterization if not done.

    Args:
        trace: Waveform being analyzed.
        state: Current analysis state.
        history: Analysis history.

    Returns:
        Recommendation or None if already done.
    """
    if "characterization" in state or history.was_recent("characterization"):
        return None

    return Recommendation(
        id="characterization",
        title="Characterize signal",
        explanation="Identify what type of signal this is (UART, SPI, analog, etc.) to guide further analysis.",
        urgency=0.95,
        ease=0.90,
        impact=0.95,
        priority=_calculate_priority(0.95, 0.90, 0.95),
        rationale="Signal characterization is the foundation for all other analysis",
        result_key="characterization",
        impact_description="Enables protocol-specific analysis and targeted measurements",
    )


def _recommend_anomaly_check(
    trace: WaveformTrace,
    state: dict,  # type: ignore[type-arg]
    history: AnalysisHistory,
) -> Recommendation | None:
    """Recommend anomaly detection.

    Args:
        trace: Waveform being analyzed.
        state: Current analysis state.
        history: Analysis history.

    Returns:
        Recommendation or None if not applicable.
    """
    if "anomalies" in state or history.was_recent("anomalies"):
        return None

    # Higher priority if characterization shows issues
    urgency = 0.70

    if "characterization" in state:
        char = state["characterization"]
        if hasattr(char, "confidence") and char.confidence < 0.8:
            urgency = 0.85

    if "quality" in state:
        quality = state["quality"]
        if hasattr(quality, "status") and quality.status in ["WARNING", "FAIL"]:
            urgency = 0.90

    return Recommendation(
        id="anomaly_detection",
        title="Check for anomalies",
        explanation="Scan the signal for glitches, dropouts, noise spikes, and other problems that could affect data integrity.",
        urgency=urgency,
        ease=0.85,
        impact=0.80,
        priority=_calculate_priority(urgency, 0.85, 0.80),
        rationale="Quality concerns detected - anomaly scan recommended",
        result_key="anomalies",
        impact_description="Identifies specific problem areas and their severity",
    )


def _recommend_quality_assessment(
    trace: WaveformTrace,
    state: dict,  # type: ignore[type-arg]
    history: AnalysisHistory,
) -> Recommendation | None:
    """Recommend data quality assessment.

    Args:
        trace: Waveform being analyzed.
        state: Current analysis state.
        history: Analysis history.

    Returns:
        Recommendation or None if not applicable.
    """
    if "quality" in state or history.was_recent("quality"):
        return None

    # Higher priority early in analysis
    urgency = 0.75 if len(state) < 2 else 0.65

    return Recommendation(
        id="quality_assessment",
        title="Assess data quality",
        explanation="Verify that sample rate and resolution are adequate for reliable analysis of this signal.",
        urgency=urgency,
        ease=0.90,
        impact=0.75,
        priority=_calculate_priority(urgency, 0.90, 0.75),
        rationale="Ensures captured data is suitable for intended analysis",
        result_key="quality",
        impact_description="Confirms data is good enough or identifies capture improvements needed",
    )


def _recommend_protocol_decode(
    trace: WaveformTrace,
    state: dict,  # type: ignore[type-arg]
    history: AnalysisHistory,
) -> Recommendation | None:
    """Recommend protocol decoding.

    Args:
        trace: Waveform being analyzed.
        state: Current analysis state.
        history: Analysis history.

    Returns:
        Recommendation or None if not applicable.
    """
    if "decode" in state or history.was_recent("decode"):
        return None

    # Only recommend if signal type is identified
    if "characterization" not in state:
        return None

    char = state["characterization"]

    # Check if it's a protocol signal
    if hasattr(char, "signal_type"):
        signal_type = char.signal_type.lower()

        if any(proto in signal_type for proto in ["uart", "spi", "i2c", "can"]):
            confidence = getattr(char, "confidence", 0.0)

            if confidence >= 0.7:
                urgency = 0.85
                explanation = f"Signal identified as {char.signal_type} with high confidence. Decode to extract transmitted data."
            else:
                urgency = 0.60
                explanation = f"Signal possibly {char.signal_type} but confidence is low. Decode may help verify."

            return Recommendation(
                id="protocol_decode",
                title="Decode protocol data",
                explanation=explanation,
                urgency=urgency,
                ease=0.80,
                impact=0.90,
                priority=_calculate_priority(urgency, 0.80, 0.90),
                rationale=f"{char.signal_type} protocol detected",
                result_key="decode",
                impact_description="Extracts meaningful data from signal",
            )

    return None


def _recommend_spectral_analysis(
    trace: WaveformTrace,
    state: dict,  # type: ignore[type-arg]
    history: AnalysisHistory,
) -> Recommendation | None:
    """Recommend spectral analysis.

    Args:
        trace: Waveform being analyzed.
        state: Current analysis state.
        history: Analysis history.

    Returns:
        Recommendation or None if not applicable.
    """
    if "spectral" in state or history.was_recent("spectral"):
        return None

    # Recommend for analog/periodic signals
    if "characterization" in state:
        char = state["characterization"]

        if hasattr(char, "signal_type"):
            signal_type = char.signal_type.lower()

            if "analog" in signal_type or "periodic" in signal_type:
                return Recommendation(
                    id="spectral_analysis",
                    title="Perform spectral analysis",
                    explanation="Analyze frequency content to identify dominant frequencies and harmonics in this analog signal.",
                    urgency=0.65,
                    ease=0.75,
                    impact=0.80,
                    priority=_calculate_priority(0.65, 0.75, 0.80),
                    rationale="Periodic/analog signal detected",
                    result_key="spectral",
                    impact_description="Reveals frequency components and signal purity",
                )

    return None


def suggest_next_steps(
    trace: WaveformTrace,
    *,
    current_state: dict[str, Any] | None = None,
    max_suggestions: int = 3,
    include_rationale: bool = False,
) -> list[Recommendation]:
    """Suggest next analysis steps based on current state.

    Provides contextual recommendations guiding users through the investigation
    process without requiring expertise.

    Args:
        trace: Waveform being analyzed.
        current_state: Current analysis state with completed steps and results.
        max_suggestions: Maximum number of suggestions (default 3, range 2-5).
        include_rationale: Include detailed rationale in recommendations.

    Returns:
        List of 2-5 recommended next steps, ranked by priority.

    Example:
        >>> trace = load("capture.wfm")
        >>> state = {"characterization": char_result}
        >>> recommendations = suggest_next_steps(trace, current_state=state)
        >>> for rec in recommendations:
        ...     print(f"{rec.priority:.2f}: {rec.title}")

    References:
        DISC-008: Recommendation Engine
    """
    current_state = current_state or {}

    # Extract or create analysis history
    if "steps_completed" in current_state:
        history = AnalysisHistory(
            steps_completed=current_state["steps_completed"],
        )
    else:
        history = AnalysisHistory()

    # Generate candidate recommendations
    candidates = []

    # Try each recommendation generator
    generators = [
        _recommend_characterization,
        _recommend_quality_assessment,
        _recommend_anomaly_check,
        _recommend_protocol_decode,
        _recommend_spectral_analysis,
    ]

    for generator in generators:
        rec = generator(trace, current_state, history)
        if rec is not None:
            candidates.append(rec)

    # If no specific recommendations, provide escape hatch
    if not candidates:
        candidates.append(
            Recommendation(
                id="basic_characterization",
                title="Start with basic signal characterization",
                explanation="Not sure where to start? Begin with automatic signal characterization to identify the signal type.",
                urgency=0.50,
                ease=0.95,
                impact=0.85,
                priority=_calculate_priority(0.50, 0.95, 0.85),
                rationale="Default starting point when no analysis has been done",
                result_key="characterization",
                impact_description="Provides foundation for further analysis",
            )
        )

    # Sort by priority (descending)
    candidates.sort(key=lambda r: r.priority, reverse=True)

    # Limit to max_suggestions
    max_suggestions = max(2, min(5, max_suggestions))
    recommendations = candidates[:max_suggestions]

    # Remove rationale if not requested
    if not include_rationale:
        for rec in recommendations:
            rec.rationale = ""

    return recommendations


__all__ = [
    "AnalysisHistory",
    "Recommendation",
    "suggest_next_steps",
]
