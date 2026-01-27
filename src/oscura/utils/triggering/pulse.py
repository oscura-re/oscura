"""Pulse width and glitch triggering for Oscura.

Provides pulse width triggering, glitch detection, and runt pulse
detection for signal integrity analysis.

Example:
    >>> from oscura.utils.triggering.pulse import PulseWidthTrigger, find_glitches
    >>> # Find pulses between 100ns and 200ns
    >>> trigger = PulseWidthTrigger(level=1.5, min_width=100e-9, max_width=200e-9)
    >>> events = trigger.find_events(trace)
    >>> # Find glitches shorter than 50ns
    >>> glitches = find_glitches(trace, max_width=50e-9)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from oscura.core.exceptions import AnalysisError
from oscura.core.types import DigitalTrace, WaveformTrace
from oscura.utils.triggering.base import (
    Trigger,
    TriggerEvent,
    TriggerType,
    interpolate_crossing,
)


@dataclass
class PulseInfo:
    """Information about a detected pulse.

    Attributes:
        start_time: Start time of pulse in seconds.
        end_time: End time of pulse in seconds.
        width: Pulse width in seconds.
        polarity: "positive" or "negative".
        start_index: Sample index at pulse start.
        end_index: Sample index at pulse end.
        amplitude: Peak amplitude during pulse.
    """

    start_time: float
    end_time: float
    width: float
    polarity: Literal["positive", "negative"]
    start_index: int
    end_index: int
    amplitude: float


class PulseWidthTrigger(Trigger):
    """Pulse width trigger for detecting pulses in a width range.

    Triggers on pulses that fall within the specified width range.

    Attributes:
        level: Threshold level for pulse detection.
        polarity: Pulse polarity - "positive", "negative", or "either".
        min_width: Minimum pulse width (None for no minimum).
        max_width: Maximum pulse width (None for no maximum).
    """

    def __init__(
        self,
        level: float,
        polarity: Literal["positive", "negative", "either"] = "positive",
        min_width: float | None = None,
        max_width: float | None = None,
    ) -> None:
        """Initialize pulse width trigger.

        Args:
            level: Threshold level for pulse detection.
            polarity: Pulse polarity to detect.
            min_width: Minimum pulse width in seconds.
            max_width: Maximum pulse width in seconds.

        Raises:
            AnalysisError: If min_width is greater than max_width.
        """
        self.level = level
        self.polarity = polarity
        self.min_width = min_width
        self.max_width = max_width

        if min_width is not None and max_width is not None and min_width > max_width:
            raise AnalysisError("min_width cannot be greater than max_width")

    def find_events(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> list[TriggerEvent]:
        """Find pulses matching the width criteria.

        Args:
            trace: Input trace.

        Returns:
            List of trigger events for matching pulses.
        """
        pulses = self._find_all_pulses(trace)

        # Filter by width
        events: list[TriggerEvent] = []
        for pulse in pulses:
            if self.min_width is not None and pulse.width < self.min_width:
                continue
            if self.max_width is not None and pulse.width > self.max_width:
                continue

            events.append(
                TriggerEvent(
                    timestamp=pulse.start_time,
                    sample_index=pulse.start_index,
                    event_type=TriggerType.PULSE_WIDTH,
                    level=pulse.amplitude,
                    duration=pulse.width,
                    data={
                        "polarity": pulse.polarity,
                        "end_time": pulse.end_time,
                        "end_index": pulse.end_index,
                    },
                )
            )

        return events

    def _find_all_pulses(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> list[PulseInfo]:
        """Find all pulses in the trace."""
        if isinstance(trace, DigitalTrace):
            data = trace.data.astype(np.float64)
            level = 0.5
        else:
            data = trace.data
            level = self.level

        sample_period = trace.metadata.time_base
        pulses: list[PulseInfo] = []

        # Find all threshold crossings
        above = data >= level
        below = data < level

        # Rising edges: transition from below to above
        rising = np.where(below[:-1] & above[1:])[0]
        # Falling edges: transition from above to below
        falling = np.where(above[:-1] & below[1:])[0]

        if self.polarity in ("positive", "either"):
            # Positive pulses: rising -> falling
            for r_idx in rising:
                # Find next falling edge
                next_falling = falling[falling > r_idx]
                if len(next_falling) == 0:
                    continue
                f_idx = next_falling[0]

                start_time = interpolate_crossing(data, r_idx, level, sample_period, True)
                end_time = interpolate_crossing(data, f_idx, level, sample_period, False)
                width = end_time - start_time

                # Get peak amplitude
                pulse_data = data[r_idx : f_idx + 1]
                amplitude = float(np.max(pulse_data)) if len(pulse_data) > 0 else level

                pulses.append(
                    PulseInfo(
                        start_time=start_time,
                        end_time=end_time,
                        width=width,
                        polarity="positive",
                        start_index=int(r_idx),
                        end_index=int(f_idx),
                        amplitude=amplitude,
                    )
                )

        if self.polarity in ("negative", "either"):
            # Negative pulses: falling -> rising
            for f_idx in falling:
                # Find next rising edge
                next_rising = rising[rising > f_idx]
                if len(next_rising) == 0:
                    continue
                r_idx = next_rising[0]

                start_time = interpolate_crossing(data, f_idx, level, sample_period, False)
                end_time = interpolate_crossing(data, r_idx, level, sample_period, True)
                width = end_time - start_time

                # Get peak (minimum) amplitude
                pulse_data = data[f_idx : r_idx + 1]
                amplitude = float(np.min(pulse_data)) if len(pulse_data) > 0 else level

                pulses.append(
                    PulseInfo(
                        start_time=start_time,
                        end_time=end_time,
                        width=width,
                        polarity="negative",
                        start_index=int(f_idx),
                        end_index=int(r_idx),
                        amplitude=amplitude,
                    )
                )

        # Sort by start time
        pulses.sort(key=lambda p: p.start_time)
        return pulses


class GlitchTrigger(Trigger):
    """Glitch trigger for detecting narrow pulses.

    Glitches are pulses shorter than a maximum width threshold.

    Attributes:
        level: Threshold level.
        max_width: Maximum pulse width to be considered a glitch.
        polarity: Glitch polarity - "positive", "negative", or "either".
    """

    def __init__(
        self,
        level: float,
        max_width: float = 100e-9,
        polarity: Literal["positive", "negative", "either"] = "either",
    ) -> None:
        """Initialize glitch trigger.

        Args:
            level: Threshold level.
            max_width: Maximum pulse width to trigger (in seconds).
            polarity: Glitch polarity to detect.
        """
        self.level = level
        self.max_width = max_width
        self.polarity = polarity

    def find_events(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> list[TriggerEvent]:
        """Find all glitches in the trace.

        Args:
            trace: Input trace.

        Returns:
            List of trigger events for each glitch.
        """
        pulse_trigger = PulseWidthTrigger(
            level=self.level,
            polarity=self.polarity,
            min_width=None,
            max_width=self.max_width,
        )

        events = pulse_trigger.find_events(trace)

        # Reclassify as glitch events
        for event in events:
            event.event_type = TriggerType.GLITCH

        return events


class RuntTrigger(Trigger):
    """Runt pulse trigger for detecting incomplete transitions.

    Runt pulses cross one threshold but not the other, indicating
    incomplete signal transitions.

    Attributes:
        low_threshold: Lower threshold level.
        high_threshold: Upper threshold level.
        polarity: Runt polarity - "positive", "negative", or "either".
    """

    def __init__(
        self,
        low_threshold: float,
        high_threshold: float,
        polarity: Literal["positive", "negative", "either"] = "either",
    ) -> None:
        """Initialize runt trigger.

        Args:
            low_threshold: Lower threshold (e.g., logic low).
            high_threshold: Upper threshold (e.g., logic high).
            polarity: "positive" for rising runts, "negative" for falling.

        Raises:
            AnalysisError: If low_threshold is not less than high_threshold.
        """
        if low_threshold >= high_threshold:
            raise AnalysisError("low_threshold must be less than high_threshold")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.polarity = polarity

    def _get_zone(self, value: float) -> int:
        """Get zone for a signal value.

        Args:
            value: Signal value

        Returns:
            Zone index (0=low, 1=middle, 2=high)
        """
        if value < self.low_threshold:
            return 0
        if value > self.high_threshold:
            return 2
        return 1

    def _find_positive_runt(
        self,
        zones: NDArray[np.int_],
        data: NDArray[np.float64],
        start_idx: int,
        sample_period: float,
    ) -> tuple[TriggerEvent | None, int]:
        """Find positive runt pulse starting at index.

        Args:
            zones: Zone classification array
            data: Signal data
            start_idx: Starting index
            sample_period: Sample period

        Returns:
            Tuple of (event or None, next index)
        """
        j = start_idx + 1
        while j < len(zones) and zones[j] == 1:
            j += 1

        if j < len(zones) and zones[j] == 0:
            peak = float(np.max(data[start_idx : j + 1]))
            event = TriggerEvent(
                timestamp=start_idx * sample_period,
                sample_index=start_idx,
                event_type=TriggerType.RUNT,
                level=peak,
                duration=(j - start_idx) * sample_period,
                data={
                    "polarity": "positive",
                    "expected_high": self.high_threshold,
                    "actual_peak": peak,
                },
            )
            return event, j

        return None, j

    def _find_negative_runt(
        self,
        zones: NDArray[np.int_],
        data: NDArray[np.float64],
        start_idx: int,
        sample_period: float,
    ) -> tuple[TriggerEvent | None, int]:
        """Find negative runt pulse starting at index.

        Args:
            zones: Zone classification array
            data: Signal data
            start_idx: Starting index
            sample_period: Sample period

        Returns:
            Tuple of (event or None, next index)
        """
        j = start_idx + 1
        while j < len(zones) and zones[j] == 1:
            j += 1

        if j < len(zones) and zones[j] == 2:
            trough = float(np.min(data[start_idx : j + 1]))
            event = TriggerEvent(
                timestamp=start_idx * sample_period,
                sample_index=start_idx,
                event_type=TriggerType.RUNT,
                level=trough,
                duration=(j - start_idx) * sample_period,
                data={
                    "polarity": "negative",
                    "expected_low": self.low_threshold,
                    "actual_trough": trough,
                },
            )
            return event, j

        return None, j

    def find_events(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> list[TriggerEvent]:
        """Find all runt pulses in the trace.

        Args:
            trace: Input trace.

        Returns:
            List of trigger events for each runt pulse.
        """
        if isinstance(trace, DigitalTrace):
            return []

        data = trace.data
        sample_period = trace.metadata.time_base
        events: list[TriggerEvent] = []

        zones = np.array([self._get_zone(v) for v in data])

        i = 0
        while i < len(zones) - 1:
            curr_zone = zones[i]

            # Check for positive runt (starting from zone 0)
            if curr_zone == 0 and self.polarity in ("positive", "either") and zones[i + 1] == 1:
                event, next_i = self._find_positive_runt(zones, data, i, sample_period)
                if event:
                    events.append(event)
                i = next_i
                continue

            # Check for negative runt (starting from zone 2)
            if curr_zone == 2 and self.polarity in ("negative", "either") and zones[i + 1] == 1:
                event, next_i = self._find_negative_runt(zones, data, i, sample_period)
                if event:
                    events.append(event)
                i = next_i
                continue

            i += 1

        return events


def find_pulses(
    trace: WaveformTrace,
    *,
    level: float | None = None,
    polarity: Literal["positive", "negative", "either"] = "positive",
    min_width: float | None = None,
    max_width: float | None = None,
) -> list[TriggerEvent]:
    """Find pulses matching width criteria.

    Args:
        trace: Input waveform trace.
        level: Threshold level. If None, uses 50% of amplitude.
        polarity: Pulse polarity to find.
        min_width: Minimum pulse width in seconds.
        max_width: Maximum pulse width in seconds.

    Returns:
        List of trigger events for matching pulses.

    Example:
        >>> # Find all positive pulses between 1us and 10us
        >>> pulses = find_pulses(trace, min_width=1e-6, max_width=10e-6)
    """
    if level is None:
        level = (np.min(trace.data) + np.max(trace.data)) / 2

    trigger = PulseWidthTrigger(
        level=level,
        polarity=polarity,
        min_width=min_width,
        max_width=max_width,
    )
    return trigger.find_events(trace)


def find_glitches(
    trace: WaveformTrace,
    max_width: float = 100e-9,
    *,
    level: float | None = None,
    polarity: Literal["positive", "negative", "either"] = "either",
) -> list[TriggerEvent]:
    """Find glitches (narrow pulses) in a trace.

    Args:
        trace: Input waveform trace.
        max_width: Maximum width to be considered a glitch (default 100ns).
        level: Threshold level. If None, uses 50% of amplitude.
        polarity: Glitch polarity to find.

    Returns:
        List of trigger events for each glitch.

    Example:
        >>> # Find all glitches shorter than 50ns
        >>> glitches = find_glitches(trace, max_width=50e-9)
        >>> print(f"Found {len(glitches)} glitches")
    """
    if level is None:
        level = (np.min(trace.data) + np.max(trace.data)) / 2

    trigger = GlitchTrigger(
        level=level,
        max_width=max_width,
        polarity=polarity,
    )
    return trigger.find_events(trace)


def find_runt_pulses(
    trace: WaveformTrace,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
    *,
    polarity: Literal["positive", "negative", "either"] = "either",
) -> list[TriggerEvent]:
    """Find runt pulses (incomplete transitions) in a trace.

    Args:
        trace: Input waveform trace.
        low_threshold: Lower threshold. If None, uses 20% of amplitude.
        high_threshold: Upper threshold. If None, uses 80% of amplitude.
        polarity: Runt polarity to find.

    Returns:
        List of trigger events for each runt pulse.

    Example:
        >>> # Find runts using standard 20%/80% thresholds
        >>> runts = find_runt_pulses(trace)
        >>> for runt in runts:
        ...     print(f"Runt at {runt.timestamp*1e6:.2f} us")
    """
    if low_threshold is None:
        amplitude = np.max(trace.data) - np.min(trace.data)
        low_threshold = np.min(trace.data) + 0.2 * amplitude

    if high_threshold is None:
        amplitude = np.max(trace.data) - np.min(trace.data)
        high_threshold = np.min(trace.data) + 0.8 * amplitude

    trigger = RuntTrigger(
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        polarity=polarity,
    )
    return trigger.find_events(trace)


def pulse_statistics(
    trace: WaveformTrace,
    *,
    level: float | None = None,
    polarity: Literal["positive", "negative"] = "positive",
) -> dict[str, float]:
    """Calculate pulse width statistics.

    Args:
        trace: Input waveform trace.
        level: Threshold level.
        polarity: Pulse polarity to analyze.

    Returns:
        Dictionary with pulse statistics:
        - count: Number of pulses
        - min_width: Minimum pulse width
        - max_width: Maximum pulse width
        - mean_width: Mean pulse width
        - std_width: Standard deviation of pulse widths

    Example:
        >>> stats = pulse_statistics(trace)
        >>> print(f"Mean pulse width: {stats['mean_width']*1e6:.2f} us")
    """
    if level is None:
        level = (np.min(trace.data) + np.max(trace.data)) / 2

    trigger = PulseWidthTrigger(level=level, polarity=polarity)
    events = trigger.find_events(trace)

    if len(events) == 0:
        return {
            "count": 0,
            "min_width": np.nan,
            "max_width": np.nan,
            "mean_width": np.nan,
            "std_width": np.nan,
        }

    widths = np.array([e.duration for e in events if e.duration is not None])

    return {
        "count": len(widths),
        "min_width": float(np.min(widths)),
        "max_width": float(np.max(widths)),
        "mean_width": float(np.mean(widths)),
        "std_width": float(np.std(widths)),
    }


__all__ = [
    "GlitchTrigger",
    "PulseInfo",
    "PulseWidthTrigger",
    "RuntTrigger",
    "find_glitches",
    "find_pulses",
    "find_runt_pulses",
    "pulse_statistics",
]
