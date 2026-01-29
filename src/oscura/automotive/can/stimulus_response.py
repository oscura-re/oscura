"""Stimulus-response mapping for CAN bus reverse engineering.

This module helps identify which CAN messages and signals change in response
to user actions, enabling rapid identification of relevant data during reverse
engineering work.

The primary use case is comparing:
1. Baseline capture (no actions)
2. Stimulus capture (button press, pedal movement, etc.)

This allows answering questions like:
- "What messages change when I press the brake?"
- "Which signals react to throttle position?"
- "What initializes when I turn the key?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

if TYPE_CHECKING:
    from oscura.automotive.can.models import CANMessage
    from oscura.automotive.can.session import CANSession

__all__ = [
    "ByteChange",
    "FrequencyChange",
    "StimulusResponseAnalyzer",
    "StimulusResponseReport",
]


@dataclass
class ByteChange:
    """Detected change in a specific byte position.

    Attributes:
        byte_position: Byte position (0-7 for CAN 2.0).
        baseline_values: Set of values observed in baseline.
        stimulus_values: Set of values observed in stimulus.
        change_magnitude: Normalized change magnitude (0.0-1.0).
        value_range_change: Change in value range (max - min).
        mean_change: Change in mean value.
        new_values: Values that appear only in stimulus.
        disappeared_values: Values that appear only in baseline.
    """

    byte_position: int
    baseline_values: set[int]
    stimulus_values: set[int]
    change_magnitude: float
    value_range_change: float
    mean_change: float
    new_values: set[int] = field(default_factory=set)
    disappeared_values: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Calculate derived fields."""
        self.new_values = self.stimulus_values - self.baseline_values
        self.disappeared_values = self.baseline_values - self.stimulus_values


@dataclass
class FrequencyChange:
    """Detected frequency change for a message ID.

    Attributes:
        message_id: CAN arbitration ID.
        baseline_hz: Frequency in baseline (Hz).
        stimulus_hz: Frequency in stimulus (Hz).
        change_ratio: Ratio of stimulus to baseline frequency.
        change_type: Type of change ('increased', 'decreased', 'appeared', 'disappeared').
        significance: Statistical significance (0.0-1.0).
    """

    message_id: int
    baseline_hz: float
    stimulus_hz: float
    change_ratio: float
    change_type: str
    significance: float

    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"FrequencyChange(0x{self.message_id:03X}: "
            f"{self.baseline_hz:.1f}Hz -> {self.stimulus_hz:.1f}Hz, "
            f"ratio={self.change_ratio:.2f}, {self.change_type})"
        )


@dataclass
class StimulusResponseReport:
    """Complete stimulus-response analysis report.

    Attributes:
        changed_messages: Message IDs with detected changes.
        new_messages: Message IDs only in stimulus.
        disappeared_messages: Message IDs only in baseline.
        frequency_changes: Frequency changes by message ID.
        byte_changes: Byte-level changes by message ID.
        duration_baseline: Duration of baseline session (seconds).
        duration_stimulus: Duration of stimulus session (seconds).
        confidence_threshold: Minimum confidence used for detection.
    """

    changed_messages: list[int]
    new_messages: list[int]
    disappeared_messages: list[int]
    frequency_changes: dict[int, FrequencyChange]
    byte_changes: dict[int, list[ByteChange]]
    duration_baseline: float
    duration_stimulus: float
    confidence_threshold: float

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns:
            Multi-line summary string.
        """
        lines = [
            "=== Stimulus-Response Analysis ===",
            f"Baseline duration: {self.duration_baseline:.2f}s",
            f"Stimulus duration: {self.duration_stimulus:.2f}s",
            f"Confidence threshold: {self.confidence_threshold:.2f}",
            "",
        ]

        if self.new_messages:
            lines.append(f"New Messages ({len(self.new_messages)}):")
            for msg_id in sorted(self.new_messages):
                lines.append(f"  0x{msg_id:03X}")
            lines.append("")

        if self.disappeared_messages:
            lines.append(f"Disappeared Messages ({len(self.disappeared_messages)}):")
            for msg_id in sorted(self.disappeared_messages):
                lines.append(f"  0x{msg_id:03X}")
            lines.append("")

        if self.frequency_changes:
            lines.append(f"Frequency Changes ({len(self.frequency_changes)}):")
            for msg_id in sorted(self.frequency_changes.keys()):
                fc = self.frequency_changes[msg_id]
                lines.append(
                    f"  0x{msg_id:03X}: {fc.baseline_hz:.1f}Hz -> {fc.stimulus_hz:.1f}Hz "
                    f"({fc.change_type}, sig={fc.significance:.2f})"
                )
            lines.append("")

        if self.byte_changes:
            lines.append(f"Byte-Level Changes ({len(self.byte_changes)} messages):")
            for msg_id in sorted(self.byte_changes.keys()):
                changes = self.byte_changes[msg_id]
                lines.append(f"  0x{msg_id:03X}: {len(changes)} bytes changed")
                for bc in changes:
                    lines.append(
                        f"    Byte {bc.byte_position}: "
                        f"magnitude={bc.change_magnitude:.2f}, "
                        f"mean_change={bc.mean_change:.1f}"
                    )
            lines.append("")

        if not any([self.new_messages, self.disappeared_messages, self.changed_messages]):
            lines.append("No significant changes detected.")

        return "\n".join(lines)


class StimulusResponseAnalyzer:
    """Analyzer for detecting CAN message changes between sessions.

    This class compares a baseline session (no user action) against a stimulus
    session (with user action) to identify which messages and signals respond
    to the stimulus.
    """

    def detect_responses(
        self,
        baseline_session: CANSession,
        stimulus_session: CANSession,
        time_window_ms: float = 100,
        change_threshold: float = 0.1,
    ) -> StimulusResponseReport:
        """Detect which messages changed between sessions.

        Args:
            baseline_session: Baseline capture (no action).
            stimulus_session: Stimulus capture (with action).
            time_window_ms: Time window for aligning messages (milliseconds).
            change_threshold: Minimum normalized change to report (0.0-1.0).

        Returns:
            StimulusResponseReport with detected changes.
        """
        # Get message IDs from both sessions
        baseline_ids = baseline_session.unique_ids()
        stimulus_ids = stimulus_session.unique_ids()

        # Detect new and disappeared messages
        new_messages = sorted(stimulus_ids - baseline_ids)
        disappeared_messages = sorted(baseline_ids - stimulus_ids)

        # Messages present in both sessions
        common_ids = baseline_ids & stimulus_ids

        # Detect frequency changes
        frequency_changes = {}
        for msg_id in common_ids:
            freq_change = self._detect_frequency_change(baseline_session, stimulus_session, msg_id)
            if freq_change and freq_change.significance >= change_threshold:
                frequency_changes[msg_id] = freq_change

        # Detect byte-level changes
        byte_changes = {}
        changed_messages = []
        for msg_id in common_ids:
            changes = self.analyze_signal_changes(
                baseline_session, stimulus_session, msg_id, byte_threshold=1
            )
            # Filter by change threshold
            significant_changes = [c for c in changes if c.change_magnitude >= change_threshold]
            if significant_changes:
                byte_changes[msg_id] = significant_changes
                changed_messages.append(msg_id)

        # Get durations
        baseline_start, baseline_end = baseline_session.time_range()
        stimulus_start, stimulus_end = stimulus_session.time_range()

        return StimulusResponseReport(
            changed_messages=sorted(set(changed_messages) | set(frequency_changes.keys())),
            new_messages=new_messages,
            disappeared_messages=disappeared_messages,
            frequency_changes=frequency_changes,
            byte_changes=byte_changes,
            duration_baseline=baseline_end - baseline_start,
            duration_stimulus=stimulus_end - stimulus_start,
            confidence_threshold=change_threshold,
        )

    def analyze_signal_changes(
        self,
        baseline_session: CANSession,
        stimulus_session: CANSession,
        message_id: int,
        byte_threshold: int = 1,
    ) -> list[ByteChange]:
        """Analyze byte-level changes in a specific message.

        Args:
            baseline_session: Baseline capture.
            stimulus_session: Stimulus capture.
            message_id: CAN arbitration ID to analyze.
            byte_threshold: Minimum number of unique values to consider changing.

        Returns:
            List of ByteChange objects for changed bytes.
        """
        baseline_msgs = baseline_session._messages.filter_by_id(message_id)
        stimulus_msgs = stimulus_session._messages.filter_by_id(message_id)

        if not baseline_msgs.messages or not stimulus_msgs.messages:
            return []

        max_dlc = max(
            max(msg.dlc for msg in baseline_msgs.messages),
            max(msg.dlc for msg in stimulus_msgs.messages),
        )

        changes = []
        for byte_pos in range(max_dlc):
            change = self._analyze_byte_change(
                baseline_msgs, stimulus_msgs, byte_pos, byte_threshold
            )
            if change:
                changes.append(change)

        return changes

    def _analyze_byte_change(
        self, baseline_msgs: Any, stimulus_msgs: Any, byte_pos: int, byte_threshold: int
    ) -> ByteChange | None:
        """Analyze change for a single byte position."""
        # Extract byte values
        baseline_values = [
            msg.data[byte_pos] for msg in baseline_msgs.messages if len(msg.data) > byte_pos
        ]
        stimulus_values = [
            msg.data[byte_pos] for msg in stimulus_msgs.messages if len(msg.data) > byte_pos
        ]

        if not baseline_values or not stimulus_values:
            return None

        baseline_set, stimulus_set = set(baseline_values), set(stimulus_values)

        # Skip if not enough unique values
        if len(baseline_set) < byte_threshold and len(stimulus_set) < byte_threshold:
            return None

        # Calculate statistics and change magnitude
        baseline_arr, stimulus_arr = np.array(baseline_values), np.array(stimulus_values)
        mean_change, value_range_change = self._compute_byte_stats(baseline_arr, stimulus_arr)
        change_magnitude = self._compute_change_magnitude(
            baseline_arr, stimulus_arr, baseline_set, stimulus_set, mean_change, value_range_change
        )

        if change_magnitude <= 0.0:
            return None

        return ByteChange(
            byte_position=byte_pos,
            baseline_values=baseline_set,
            stimulus_values=stimulus_set,
            change_magnitude=change_magnitude,
            value_range_change=value_range_change,
            mean_change=mean_change,
        )

    def _compute_byte_stats(
        self, baseline_arr: NDArray[np.int_], stimulus_arr: NDArray[np.int_]
    ) -> tuple[float, float]:
        """Compute mean and range changes for byte values."""
        baseline_mean, stimulus_mean = float(np.mean(baseline_arr)), float(np.mean(stimulus_arr))
        mean_change = stimulus_mean - baseline_mean

        baseline_range = float(np.max(baseline_arr) - np.min(baseline_arr))
        stimulus_range = float(np.max(stimulus_arr) - np.min(stimulus_arr))
        value_range_change = stimulus_range - baseline_range

        return mean_change, value_range_change

    def _compute_change_magnitude(
        self,
        baseline_arr: NDArray[np.int_],
        stimulus_arr: NDArray[np.int_],
        baseline_set: set[int],
        stimulus_set: set[int],
        mean_change: float,
        value_range_change: float,
    ) -> float:
        """Compute normalized change magnitude using multiple factors."""
        # 1. Mean change (normalized by full byte range)
        mean_change_norm = abs(mean_change) / 255.0

        # 2. Range change (normalized by full byte range)
        range_change_norm = abs(value_range_change) / 255.0

        # 3. Set difference (Jaccard distance)
        union_size, intersection_size = (
            len(baseline_set | stimulus_set),
            len(baseline_set & stimulus_set),
        )
        jaccard_dist = 1.0 - (intersection_size / union_size) if union_size > 0 else 0.0

        # 4. Distribution change (Kolmogorov-Smirnov test)
        try:
            ks_stat, _ = stats.ks_2samp(baseline_arr, stimulus_arr)
            ks_change_norm = float(ks_stat)
        except Exception:
            ks_change_norm = 0.0

        # Combine factors (weighted average)
        return (
            0.3 * mean_change_norm
            + 0.2 * range_change_norm
            + 0.3 * jaccard_dist
            + 0.2 * ks_change_norm
        )

    def find_responsive_messages(
        self,
        baseline_session: CANSession,
        stimulus_session: CANSession,
    ) -> list[int]:
        """Find message IDs that changed.

        This is a convenience method that returns just the list of message IDs
        that showed any type of change.

        Args:
            baseline_session: Baseline capture.
            stimulus_session: Stimulus capture.

        Returns:
            Sorted list of message IDs that changed.
        """
        report = self.detect_responses(baseline_session, stimulus_session)
        return report.changed_messages + report.new_messages

    def _detect_frequency_change(
        self,
        baseline_session: CANSession,
        stimulus_session: CANSession,
        message_id: int,
    ) -> FrequencyChange | None:
        """Detect frequency change for a specific message ID.

        Args:
            baseline_session: Baseline capture.
            stimulus_session: Stimulus capture.
            message_id: CAN arbitration ID.

        Returns:
            FrequencyChange if detected, None otherwise.
        """
        # Get messages
        baseline_msgs = baseline_session._messages.filter_by_id(message_id)
        stimulus_msgs = stimulus_session._messages.filter_by_id(message_id)

        # Calculate frequencies
        baseline_hz = self._calculate_frequency(
            baseline_msgs.messages, baseline_session.time_range()
        )
        stimulus_hz = self._calculate_frequency(
            stimulus_msgs.messages, stimulus_session.time_range()
        )

        # Determine change type and ratio
        if baseline_hz == 0.0 and stimulus_hz > 0.0:
            change_type = "appeared"
            change_ratio = float("inf")
            significance = 1.0
        elif stimulus_hz == 0.0 and baseline_hz > 0.0:
            change_type = "disappeared"
            change_ratio = 0.0
            significance = 1.0
        elif baseline_hz > 0.0:
            change_ratio = stimulus_hz / baseline_hz

            # Determine if change is significant
            # Use a threshold of 20% change
            if change_ratio > 1.2:
                change_type = "increased"
                significance = min(1.0, (change_ratio - 1.0) / 1.0)
            elif change_ratio < 0.8:
                change_type = "decreased"
                significance = min(1.0, (1.0 - change_ratio) / 1.0)
            else:
                # No significant change
                return None
        else:
            # Both frequencies are zero
            return None

        return FrequencyChange(
            message_id=message_id,
            baseline_hz=baseline_hz,
            stimulus_hz=stimulus_hz,
            change_ratio=change_ratio,
            change_type=change_type,
            significance=significance,
        )

    @staticmethod
    def _calculate_frequency(messages: list[CANMessage], time_range: tuple[float, float]) -> float:
        """Calculate message frequency.

        Args:
            messages: List of CAN messages.
            time_range: Tuple of (start_time, end_time).

        Returns:
            Frequency in Hz.
        """
        if not messages or len(messages) < 2:
            return 0.0

        start_time, end_time = time_range
        duration = end_time - start_time

        if duration <= 0.0:
            return 0.0

        # Use message count divided by duration
        return (len(messages) - 1) / duration
