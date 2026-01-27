"""Base classes and utilities for Oscura triggering module.

Provides abstract base class for triggers and common trigger event
data structure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

from oscura.core.exceptions import AnalysisError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from oscura.core.types import DigitalTrace, WaveformTrace


class TriggerType(Enum):
    """Types of trigger events."""

    RISING_EDGE = "rising_edge"
    FALLING_EDGE = "falling_edge"
    PATTERN_MATCH = "pattern_match"
    PULSE_WIDTH = "pulse_width"
    GLITCH = "glitch"
    RUNT = "runt"
    WINDOW_ENTRY = "window_entry"
    WINDOW_EXIT = "window_exit"
    ZONE_VIOLATION = "zone_violation"


@dataclass
class TriggerEvent:
    """Represents a detected trigger event.

    Attributes:
        timestamp: Time of the trigger event in seconds.
        sample_index: Sample index where trigger occurred.
        event_type: Type of trigger event.
        level: Voltage/signal level at trigger point.
        data: Additional event-specific data.
    """

    timestamp: float
    sample_index: int
    event_type: TriggerType
    level: float | None = None
    duration: float | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TriggerEvent({self.event_type.value} at t={self.timestamp:.6e}s, "
            f"sample={self.sample_index})"
        )


class Trigger(ABC):
    """Abstract base class for all trigger types.

    Defines the common interface for finding trigger events in traces.
    """

    @abstractmethod
    def find_events(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> list[TriggerEvent]:
        """Find all trigger events in a trace.

        Args:
            trace: Input trace to search.

        Returns:
            List of trigger events found.
        """
        ...

    def find_first(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> TriggerEvent | None:
        """Find the first trigger event.

        Args:
            trace: Input trace to search.

        Returns:
            First trigger event, or None if no triggers found.
        """
        events = self.find_events(trace)
        return events[0] if events else None

    def count_events(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> int:
        """Count trigger events.

        Args:
            trace: Input trace to search.

        Returns:
            Number of trigger events found.
        """
        return len(self.find_events(trace))


def find_triggers(
    trace: WaveformTrace | DigitalTrace,
    trigger_type: Literal["edge", "pattern", "pulse_width", "glitch", "runt", "window"],
    **kwargs: Any,
) -> list[TriggerEvent]:
    """Unified function to find trigger events.

    Args:
        trace: Input trace to search.
        trigger_type: Type of trigger to use.
        **kwargs: Trigger-specific parameters.

    Returns:
        List of trigger events.

    Raises:
        AnalysisError: If unknown trigger type.

    Example:
        >>> events = find_triggers(trace, "edge", level=1.5, edge="rising")
        >>> events = find_triggers(trace, "pulse_width", min_width=1e-6, max_width=2e-6)
        >>> events = find_triggers(trace, "glitch", max_width=50e-9)
    """
    from oscura.utils.triggering.edge import EdgeTrigger
    from oscura.utils.triggering.pulse import PulseWidthTrigger
    from oscura.utils.triggering.window import WindowTrigger

    if trigger_type == "edge":
        trigger = EdgeTrigger(
            level=kwargs.get("level", 0.0),
            edge=kwargs.get("edge", "rising"),
            hysteresis=kwargs.get("hysteresis", 0.0),
        )
    elif trigger_type == "pattern":
        from oscura.utils.triggering.pattern import PatternTrigger

        trigger = PatternTrigger(  # type: ignore[assignment]
            pattern=kwargs.get("pattern", []),
            levels=kwargs.get("levels"),
        )
    elif trigger_type == "pulse_width":
        trigger = PulseWidthTrigger(  # type: ignore[assignment]
            level=kwargs.get("level", 0.0),
            polarity=kwargs.get("polarity", "positive"),
            min_width=kwargs.get("min_width"),
            max_width=kwargs.get("max_width"),
        )
    elif trigger_type == "glitch":
        from oscura.utils.triggering.pulse import GlitchTrigger

        trigger = GlitchTrigger(  # type: ignore[assignment]
            level=kwargs.get("level", 0.0),
            max_width=kwargs.get("max_width", 100e-9),
            polarity=kwargs.get("polarity", "either"),
        )
    elif trigger_type == "runt":
        from oscura.utils.triggering.pulse import RuntTrigger

        trigger = RuntTrigger(  # type: ignore[assignment]
            low_threshold=kwargs.get("low_threshold", 0.0),
            high_threshold=kwargs.get("high_threshold", 1.0),
            polarity=kwargs.get("polarity", "either"),
        )
    elif trigger_type == "window":
        trigger = WindowTrigger(  # type: ignore[assignment]
            low_threshold=kwargs.get("low_threshold", 0.0),
            high_threshold=kwargs.get("high_threshold", 1.0),
            trigger_on=kwargs.get("trigger_on", "exit"),
        )
    else:
        raise AnalysisError(f"Unknown trigger type: {trigger_type}")

    return trigger.find_events(trace)


def interpolate_crossing(
    data: NDArray[np.floating[Any]],
    idx: int,
    threshold: float,
    sample_period: float,
    rising: bool = True,
) -> float:
    """Interpolate exact threshold crossing time.

    Args:
        data: Waveform data array.
        idx: Sample index near crossing.
        threshold: Threshold level.
        sample_period: Time between samples.
        rising: True for rising edge, False for falling.

    Returns:
        Interpolated crossing time in seconds.
    """
    if idx < 0 or idx >= len(data) - 1:
        return idx * sample_period

    v1, v2 = data[idx], data[idx + 1]

    if abs(v2 - v1) < 1e-12:
        return (idx + 0.5) * sample_period

    # Linear interpolation
    t_offset = (threshold - v1) / (v2 - v1) * sample_period
    t_offset = max(0, min(sample_period, t_offset))

    return idx * sample_period + t_offset  # type: ignore[no-any-return]


__all__ = [
    "Trigger",
    "TriggerEvent",
    "TriggerType",
    "find_triggers",
    "interpolate_crossing",
]
