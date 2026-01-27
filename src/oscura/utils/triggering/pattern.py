"""Pattern triggering for Oscura.

Provides digital pattern matching for multi-channel logic signals.
Supports exact matches, wildcards, and edge conditions.

Example:
    >>> from oscura.utils.triggering.pattern import PatternTrigger, find_pattern
    >>> # Find pattern 1010 on 4 channels
    >>> trigger = PatternTrigger(pattern=[1, 0, 1, 0])
    >>> events = trigger.find_events(trace)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from oscura.core.exceptions import AnalysisError
from oscura.core.types import DigitalTrace, WaveformTrace
from oscura.utils.triggering.base import (
    Trigger,
    TriggerEvent,
    TriggerType,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PatternTrigger(Trigger):
    """Pattern trigger for multi-bit digital pattern matching.

    Detects when a digital signal or set of signals matches a
    specified pattern.

    For single-channel waveforms, the pattern specifies a sequence
    of high/low states that must occur consecutively.

    Attributes:
        pattern: Pattern to match (list of 0, 1, or None for don't care).
        levels: Threshold levels for converting analog to digital.
        match_type: Type of match - "exact", "any", or "sequence".
    """

    def __init__(
        self,
        pattern: list[int | None],
        levels: float | list[float] | None = None,
        match_type: Literal["exact", "sequence"] = "sequence",
    ) -> None:
        """Initialize pattern trigger.

        Args:
            pattern: Pattern to match. Values are 0, 1, or None (don't care).
                    For multi-channel, this is the pattern across channels.
                    For single-channel sequence, this is the bit sequence.
            levels: Threshold level(s) for analog-to-digital conversion.
                   If None, uses 50% of signal amplitude.
            match_type: "exact" matches pattern at each sample,
                       "sequence" finds the pattern as a sequence in time.

        Raises:
            AnalysisError: If pattern contains invalid values.
        """
        self.pattern = pattern
        self.levels = levels
        self.match_type = match_type

        # Validate pattern
        for val in pattern:
            if val is not None and val not in (0, 1):
                raise AnalysisError(f"Pattern values must be 0, 1, or None, got {val}")

    def find_events(
        self,
        trace: WaveformTrace | DigitalTrace,
    ) -> list[TriggerEvent]:
        """Find pattern matches in the trace.

        Args:
            trace: Input trace (single channel for sequence matching).

        Returns:
            List of trigger events for each pattern match.
        """
        # Convert to digital if needed
        if isinstance(trace, DigitalTrace):
            digital = trace.data
        else:
            level = self._get_level(trace)
            digital = trace.data >= level

        sample_period = trace.metadata.time_base
        events: list[TriggerEvent] = []

        if self.match_type == "sequence":
            events = self._find_sequence_matches(digital, sample_period)
        else:
            events = self._find_exact_matches(digital, sample_period)

        return events

    def _get_level(self, trace: WaveformTrace) -> float:
        """Get threshold level for analog-to-digital conversion."""
        if isinstance(self.levels, int | float):
            return float(self.levels)
        elif self.levels is None:
            return (np.min(trace.data) + np.max(trace.data)) / 2  # type: ignore[no-any-return]
        else:
            # Multi-channel case - use first level for single trace
            return float(self.levels[0])

    def _find_sequence_matches(
        self,
        digital: NDArray[np.bool_],
        sample_period: float,
    ) -> list[TriggerEvent]:
        """Find pattern as a sequence in the data."""
        events: list[TriggerEvent] = []

        # Convert pattern to expected transitions
        pattern_len = len(self.pattern)
        pattern_arr = np.array([p if p is not None else -1 for p in self.pattern])

        # Slide pattern across data
        for i in range(len(digital) - pattern_len + 1):
            segment = digital[i : i + pattern_len].astype(np.int8)

            # Check match (ignoring don't care values)
            match = True
            for j, p in enumerate(pattern_arr):
                if p >= 0 and segment[j] != p:
                    match = False
                    break

            if match:
                events.append(
                    TriggerEvent(
                        timestamp=i * sample_period,
                        sample_index=i,
                        event_type=TriggerType.PATTERN_MATCH,
                        duration=pattern_len * sample_period,
                        data={"pattern": self.pattern},
                    )
                )

        return events

    def _find_exact_matches(
        self,
        digital: NDArray[np.bool_],
        sample_period: float,
    ) -> list[TriggerEvent]:
        """Find exact pattern matches at each sample."""
        events: list[TriggerEvent] = []

        # For single channel, check if current value matches first pattern bit
        pattern_val = self.pattern[0]
        if pattern_val is None:
            # Don't care - matches everything
            return events

        expected = bool(pattern_val)
        prev_match = digital[0] == expected

        for i in range(1, len(digital)):
            curr_match = digital[i] == expected
            if curr_match and not prev_match:
                # Transition to matching state
                events.append(
                    TriggerEvent(
                        timestamp=i * sample_period,
                        sample_index=i,
                        event_type=TriggerType.PATTERN_MATCH,
                        data={"pattern": self.pattern},
                    )
                )
            prev_match = curr_match

        return events


class MultiChannelPatternTrigger(Trigger):
    """Pattern trigger for multiple parallel channels.

    Triggers when all channels simultaneously match the specified pattern.

    Example:
        >>> trigger = MultiChannelPatternTrigger(
        ...     pattern=[1, 0, 1, None],  # Ch0=1, Ch1=0, Ch2=1, Ch3=don't care
        ...     levels=[1.5, 1.5, 1.5, 1.5]
        ... )
    """

    def __init__(
        self,
        pattern: list[int | None],
        levels: list[float] | None = None,
    ) -> None:
        """Initialize multi-channel pattern trigger.

        Args:
            pattern: Pattern for each channel (0, 1, or None for don't care).
            levels: Threshold level for each channel.
        """
        self.pattern = pattern
        self.levels = levels

    def find_events(
        self,
        traces: list[WaveformTrace | DigitalTrace],  # type: ignore[override]
    ) -> list[TriggerEvent]:
        """Find pattern matches across multiple channels.

        Args:
            traces: List of traces (one per channel).

        Returns:
            List of trigger events where pattern matches.

        Raises:
            AnalysisError: If number of traces doesn't match pattern length.
        """
        if len(traces) != len(self.pattern):
            raise AnalysisError(
                f"Number of traces ({len(traces)}) must match pattern length ({len(self.pattern)})"
            )

        # Convert all traces to digital
        digitals: list[NDArray[np.bool_]] = []
        for i, trace in enumerate(traces):
            if isinstance(trace, DigitalTrace):
                digitals.append(trace.data)
            else:
                if self.levels is not None:
                    level = self.levels[i]
                else:
                    level = (np.min(trace.data) + np.max(trace.data)) / 2
                digitals.append(trace.data >= level)

        # Find samples where all channels match pattern
        sample_period = traces[0].metadata.time_base
        n_samples = min(len(d) for d in digitals)
        events: list[TriggerEvent] = []

        prev_match = False
        for i in range(n_samples):
            curr_match = True
            for _j, (digital, pattern_val) in enumerate(zip(digitals, self.pattern, strict=False)):
                if pattern_val is not None and digital[i] != bool(pattern_val):
                    curr_match = False
                    break

            if curr_match and not prev_match:
                # Transition to matching state
                events.append(
                    TriggerEvent(
                        timestamp=i * sample_period,
                        sample_index=i,
                        event_type=TriggerType.PATTERN_MATCH,
                        data={"pattern": self.pattern},
                    )
                )
            prev_match = curr_match

        return events


def find_pattern(
    trace: WaveformTrace | DigitalTrace,
    pattern: list[int | None],
    *,
    level: float | None = None,
    return_indices: bool = False,
) -> NDArray[np.float64] | NDArray[np.int64]:
    """Find all occurrences of a bit pattern in a trace.

    Args:
        trace: Input trace.
        pattern: Bit pattern to find (0, 1, None for don't care).
        level: Threshold level. If None, uses 50% of amplitude.
        return_indices: If True, return sample indices instead of timestamps.

    Returns:
        Array of timestamps or indices where pattern was found.

    Example:
        >>> # Find start bits (0 followed by data)
        >>> starts = find_pattern(trace, [0, 1, 1])
    """
    trigger = PatternTrigger(pattern=pattern, levels=level)
    events = trigger.find_events(trace)

    if return_indices:
        return np.array([e.sample_index for e in events], dtype=np.int64)
    return np.array([e.timestamp for e in events], dtype=np.float64)


def find_bit_sequence(
    trace: WaveformTrace,
    bits: str,
    *,
    level: float | None = None,
) -> list[TriggerEvent]:
    """Find a specific bit sequence in a trace.

    Args:
        trace: Input waveform trace.
        bits: Bit string (e.g., "10101010", "1X0X" where X is don't care).
        level: Threshold level for digitization.

    Returns:
        List of trigger events for each match.

    Raises:
        AnalysisError: If invalid bit character in bits string.

    Example:
        >>> events = find_bit_sequence(trace, "10110")
        >>> events = find_bit_sequence(trace, "1XX0")  # X = don't care
    """
    # Convert string to pattern list
    pattern: list[int | None] = []
    for char in bits:
        if char == "0":
            pattern.append(0)
        elif char == "1":
            pattern.append(1)
        elif char.upper() == "X":
            pattern.append(None)
        else:
            raise AnalysisError(f"Invalid bit character: {char}")

    trigger = PatternTrigger(pattern=pattern, levels=level)
    return trigger.find_events(trace)


__all__ = [
    "MultiChannelPatternTrigger",
    "PatternTrigger",
    "find_bit_sequence",
    "find_pattern",
]
